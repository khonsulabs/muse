use kempt::Map;
use serde::{Deserialize, Serialize};

use crate::symbol::Symbol;
use crate::syntax::{
    self, AssignTarget, BinaryExpression, BreakExpression, Chain, ContinueExpression, Expression,
    FunctionCall, Index, LogicalKind, LoopExpression, LoopKind, Ranged, SourceRange,
    UnaryExpression, Variable,
};
use crate::vm::bitcode::{
    BinaryKind, BitcodeBlock, BitcodeFunction, Label, Op, OpDestination, ValueOrSource,
};
use crate::vm::{Code, Register, Stack};

#[derive(Default, Debug)]
pub struct Compiler {
    function_name: Option<Symbol>,
    parsed: Vec<Result<Ranged<Expression>, Ranged<syntax::Error>>>,
    errors: Vec<Ranged<Error>>,
    code: BitcodeBlock,
    declarations: Map<Symbol, BlockDeclaration>,
    scopes: Vec<ScopeInfo>,
}

impl Compiler {
    pub fn push(&mut self, source: &str) {
        self.parsed.push(syntax::parse(source));
    }

    #[must_use]
    pub fn with(mut self, source: &str) -> Self {
        self.push(source);
        self
    }

    pub fn compile(source: &str) -> Result<Code, Vec<Ranged<Error>>> {
        Self::default().with(source).build()
    }

    pub fn build(&mut self) -> Result<Code, Vec<Ranged<Error>>> {
        self.code.clear();
        let mut expressions = Vec::with_capacity(self.parsed.len());

        for result in self.parsed.drain(..) {
            match result {
                Ok(expr) => expressions.push(expr),
                Err(err) => self.errors.push(err.into()),
            }
        }

        let expression = Chain::from_expressions(expressions);

        Scope::module_root(self)
            .compile_expression(&expression, OpDestination::Register(Register(0)));

        if self.errors.is_empty() {
            Ok(Code::from(&self.code))
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn new_variable(&mut self) -> Stack {
        let id = self.code.stack_requirement;
        self.code.stack_requirement += 1;
        Stack(id)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BlockDeclaration {
    pub stack: Stack,
    pub mutable: bool,
}

struct LocalDeclaration {
    name: Symbol,
    previous_declaration: Option<BlockDeclaration>,
}

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
enum ScopeKind {
    Module,
    Function,
    Block,
    Loop,
}

#[derive(Debug)]
struct ScopeInfo {
    kind: ScopeKind,
    name: Option<Symbol>,
    break_info: Option<(Label, OpDestination)>,
    continue_label: Option<Label>,
}

struct Scope<'a> {
    compiler: &'a mut Compiler,
    module: bool,
    depth: usize,
    locals_count: usize,
    local_declarations: Vec<LocalDeclaration>,
}

impl<'a> Scope<'a> {
    fn module_root(compiler: &'a mut Compiler) -> Self {
        compiler.scopes.push(ScopeInfo {
            kind: ScopeKind::Module,
            name: None,
            continue_label: None,
            break_info: None,
        });
        Self {
            compiler,
            module: true,
            depth: 0,
            locals_count: 0,
            local_declarations: Vec::new(),
        }
    }

    fn function_root(compiler: &'a mut Compiler) -> Self {
        compiler.scopes.push(ScopeInfo {
            kind: ScopeKind::Function,
            name: None,
            continue_label: None,
            break_info: None,
        });
        Self {
            compiler,
            module: false,
            depth: 0,
            locals_count: 0,
            local_declarations: Vec::new(),
        }
    }

    fn enter_block(&mut self, name: Option<(Symbol, Label, OpDestination)>) -> Scope<'_> {
        let (name, break_info) = name
            .map(|(name, break_label, dest)| (Some(name), Some((break_label, dest))))
            .unwrap_or_default();
        self.compiler.scopes.push(ScopeInfo {
            kind: ScopeKind::Block,
            name,
            break_info,
            continue_label: None,
        });
        Scope {
            compiler: self.compiler,
            module: self.module,
            depth: self.depth + 1,
            locals_count: 0,
            local_declarations: Vec::new(),
        }
    }

    fn enter_loop(
        &mut self,
        name: Option<Symbol>,
        continue_label: Label,
        break_label: Label,
        dest: OpDestination,
    ) -> Scope<'_> {
        self.compiler.scopes.push(ScopeInfo {
            kind: ScopeKind::Loop,
            break_info: Some((break_label, dest)),
            name,
            continue_label: Some(continue_label),
        });
        Scope {
            compiler: self.compiler,
            module: self.module,
            depth: self.depth + 1,
            locals_count: 0,
            local_declarations: Vec::new(),
        }
    }

    fn new_temporary(&mut self) -> Stack {
        self.locals_count += 1;
        self.compiler.new_variable()
    }

    #[allow(clippy::too_many_lines)]
    fn compile_expression(&mut self, expr: &Ranged<Expression>, dest: OpDestination) {
        match &**expr {
            Expression::Nil => self.compiler.code.copy((), dest),
            Expression::Bool(bool) => self.compiler.code.copy(*bool, dest),
            Expression::Int(int) => self.compiler.code.copy(*int, dest),
            Expression::Float(float) => self.compiler.code.copy(*float, dest),
            Expression::String(string) => self
                .compiler
                .code
                .copy(ValueOrSource::String(string.clone()), dest),
            Expression::Symbol(symbol) => self.compiler.code.copy(symbol.clone(), dest),
            Expression::RegEx(regex) => self.compiler.code.copy(regex.clone(), dest),
            Expression::Lookup(lookup) => {
                if let Some(base) = &lookup.base {
                    let target = self.compile_source(base);
                    self.compiler.code.copy(lookup.name.clone(), Register(0));
                    self.compiler
                        .code
                        .invoke(target, Symbol::get_symbol().clone(), 1);
                    self.compiler.code.copy(Register(0), dest);
                } else if let Some(var) = self.compiler.declarations.get(&lookup.name) {
                    self.compiler.code.copy(var.stack, dest);
                } else {
                    self.compiler.code.resolve(lookup.name.clone(), dest);
                }
            }
            Expression::Map(map) => {
                let mut elements = Vec::with_capacity(map.values.len());
                for field in &map.values {
                    let key = self.compile_expression_into_temporary(&field.key);
                    let value = self.compile_expression_into_temporary(&field.value);
                    elements.push((key, value));
                }

                let mut num_elements = u8::try_from(elements.len()).unwrap_or(u8::MAX);
                if num_elements >= 128 {
                    num_elements = 127;
                    eprintln!("TODO Ignoring more than 127 elements in map");
                }

                for (index, (key, value)) in (0..=num_elements).zip(elements) {
                    self.compiler.code.copy(key, Register(index * 2));
                    self.compiler.code.copy(value, Register(index * 2 + 1));
                }
                self.compiler.code.new_map(num_elements, dest);
            }
            Expression::List(list) | Expression::Tuple(list) => {
                let mut elements = Vec::with_capacity(list.len());
                for field in list {
                    let value = self.compile_expression_into_temporary(field);
                    elements.push(value);
                }

                let num_elements = if let Ok(elements) = u8::try_from(elements.len()) {
                    elements
                } else {
                    eprintln!("TODO Ignoring more than 127 elements in map");
                    255
                };

                for (index, value) in (0..=num_elements).zip(elements) {
                    self.compiler.code.copy(value, Register(index));
                }
                self.compiler.code.new_list(num_elements, dest);
            }
            Expression::If(if_expr) => {
                let condition = self.compile_source(&if_expr.condition);
                let if_true = self.compiler.code.new_label();
                self.compiler.code.jump_if(if_true, condition, ());
                let after_true = self.compiler.code.new_label();
                if let Some(when_false) = &if_expr.when_false {
                    self.compile_expression(when_false, dest);
                }
                self.compiler.code.jump(after_true, ());
                self.compiler.code.label(if_true);
                self.compile_expression(&if_expr.when_true, dest);
                self.compiler.code.label(after_true);
            }
            Expression::Loop(loop_expr) => {
                self.compile_loop(loop_expr, dest);
            }
            Expression::Break(break_expr) => {
                self.compile_break(break_expr, expr.1);
            }
            Expression::Continue(continue_expr) => {
                self.compile_continue(continue_expr, expr.1);
            }
            Expression::Return(result) => {
                self.compile_expression(result, OpDestination::Register(Register(0)));
                self.compiler.code.return_early();
            }
            Expression::Assign(assign) => match &assign.target.0 {
                AssignTarget::Lookup(target) => {
                    if let Some(base) = &target.base {
                        let target_source = self.compile_source(base);
                        self.compile_expression(
                            &assign.value,
                            OpDestination::Register(Register(1)),
                        );
                        self.compiler.code.copy(target.name.clone(), Register(0));
                        self.compiler
                            .code
                            .invoke(target_source, Symbol::set_symbol().clone(), 2);
                    } else if let Some(var) = self.compiler.declarations.get(&target.name) {
                        if var.mutable {
                            let var = var.stack;
                            self.compile_expression(&assign.value, OpDestination::Stack(var));
                            self.compiler.code.copy(var, dest);
                        } else {
                            self.compiler
                                .errors
                                .push(Ranged::new(expr.range(), Error::VariableNotMutable));
                        }
                    } else {
                        let value = self.compile_source(&assign.value);
                        self.compiler.code.assign(target.name.clone(), value, dest);
                    }
                    self.compiler.code.copy(Register(0), dest);
                }
                AssignTarget::Index(index) => {
                    let target_source = self.compile_source(&index.target);
                    let value = self.compile_source(&assign.value);
                    let arity = if let Some(arity) = index
                        .parameters
                        .len()
                        .checked_add(1)
                        .and_then(|arity| u8::try_from(arity).ok())
                    {
                        arity
                    } else {
                        self.compiler
                            .errors
                            .push(Ranged::new(expr.range(), Error::TooManyArguments));
                        u8::MAX
                    };
                    self.compile_function_args(&index.parameters, arity - 1);
                    self.compiler
                        .code
                        .copy(value, OpDestination::Register(Register(arity - 1)));
                    self.compiler
                        .code
                        .invoke(target_source, Symbol::set_symbol().clone(), arity);
                }
            },
            Expression::Unary(unary) => self.compile_unary(unary, dest),
            Expression::Binary(binop) => {
                self.compile_binop(binop, dest);
            }
            Expression::Chain(chain) => {
                self.compile_expression(&chain.0, OpDestination::Void);
                self.compile_expression(&chain.1, dest);
            }
            Expression::Block(block) => {
                let break_label = self.compiler.code.new_label();
                let mut scope = self.enter_block(
                    block
                        .name
                        .as_ref()
                        .map(|name| (name.clone(), break_label, dest)),
                );
                scope.compile_expression(&block.body, dest);
                scope.compiler.code.label(break_label);
            }
            Expression::Call(call) => self.compile_function_call(call, expr.range(), dest),
            Expression::Index(index) => self.compile_index(index, expr.range(), dest),
            Expression::Variable(decl) => {
                self.declare_variable(decl, dest);
            }
            Expression::Function(decl) => {
                let arity = if let Ok(arity) = u8::try_from(decl.parameters.len()) {
                    arity
                } else {
                    self.compiler
                        .errors
                        .push(Ranged::new(expr.range(), Error::TooManyArguments));
                    255
                };
                let mut fn_compiler = Compiler {
                    function_name: decl.name.as_ref().map(|name| name.0.clone()),
                    ..Compiler::default()
                };
                let mut fn_scope = Scope::function_root(&mut fn_compiler);
                for (var, index) in decl.parameters.iter().zip(0..arity) {
                    let stack = fn_scope.new_temporary();
                    fn_scope.compiler.code.copy(Register(index), stack);
                    fn_scope.compiler.declarations.insert(
                        var.0.clone(),
                        BlockDeclaration {
                            stack,
                            mutable: false,
                        },
                    );
                }
                fn_scope.compile_expression(&decl.body, OpDestination::Register(Register(0)));
                drop(fn_scope);

                self.compiler.errors.append(&mut fn_compiler.errors);
                let fun = BitcodeFunction::new(decl.name.as_ref().map(|name| name.0.clone()))
                    .when(arity, fn_compiler.code);
                match (&decl.name, decl.publish) {
                    (Some(name), true) => {
                        self.ensure_in_module(name.1);
                        self.compiler.code.declare(name.0.clone(), false, fun, dest);
                    }
                    (Some(name), false) => {
                        let stack = self.new_temporary();
                        self.compiler.code.copy(fun, stack);
                        self.declare_local(name.0.clone(), false, stack, dest);
                    }
                    (None, true) => {
                        self.compiler
                            .errors
                            .push(Ranged::new(expr.range(), Error::InvalidDeclaration));
                    }
                    (None, false) => {
                        self.compiler.code.copy(fun, dest);
                    }
                }
            }
            Expression::Module(module) => {
                let Expression::Block(block) = &module.contents.0 else {
                    self.compiler
                        .errors
                        .push(Ranged::new(module.contents.1, Error::InvalidDeclaration));
                    return;
                };
                let mut mod_compiler = Compiler::default();
                let mut mod_scope = Scope::module_root(&mut mod_compiler);

                mod_scope.compile_expression(&block.body, OpDestination::Void);
                drop(mod_scope);
                let name = &module.name.0;
                let instance = BitcodeModule {
                    name: name.clone(),
                    initializer: mod_compiler.code,
                };
                let stack = self.new_temporary();
                self.compiler
                    .code
                    .load_module(instance, OpDestination::Stack(stack));
                if module.publish {
                    self.ensure_in_module(module.name.1);

                    self.compiler.code.declare(name.clone(), false, stack, dest);
                } else {
                    self.declare_local(name.clone(), false, stack, dest);
                }

                self.compiler.errors.append(&mut mod_compiler.errors);
            }
        }
    }

    fn compile_break(&mut self, break_expr: &BreakExpression, from: SourceRange) {
        let mut break_info = None;
        for scope in self.compiler.scopes.iter().rev() {
            match (break_expr.name.as_ref(), &scope.name) {
                (Some(query), Some(scope_name)) if &query.0 == scope_name => {
                    if !matches!(scope.kind, ScopeKind::Block | ScopeKind::Loop) {
                        self.compiler
                            .errors
                            .push(Ranged::new(query.1, Error::InvalidLabel));
                    }
                    break_info = scope.break_info;
                    break;
                }
                (None, _) if scope.kind == ScopeKind::Loop => {
                    break_info = scope.break_info;
                    break;
                }
                _ => {}
            }
        }

        if let Some((label, dest)) = break_info {
            self.compile_expression(&break_expr.value, dest);
            self.compiler.code.jump(label, ());
        } else {
            self.compiler
                .errors
                .push(Ranged::new(from, Error::InvalidLabel));
        }
    }

    fn compile_continue(&mut self, continue_expr: &ContinueExpression, from: SourceRange) {
        let mut label = None;
        for scope in self.compiler.scopes.iter().rev() {
            match (continue_expr.name.as_ref(), &scope.name) {
                (Some(query), Some(scope_name)) if &query.0 == scope_name => {
                    if !matches!(scope.kind, ScopeKind::Loop) {
                        self.compiler
                            .errors
                            .push(Ranged::new(query.1, Error::InvalidLabel));
                    }
                    label = scope.continue_label;
                    break;
                }
                (None, _) if scope.kind == ScopeKind::Loop => {
                    label = scope.continue_label;
                    break;
                }
                _ => {}
            }
        }

        if let Some(label) = label {
            self.compiler.code.jump(label, ());
        } else {
            self.compiler
                .errors
                .push(Ranged::new(from, Error::InvalidLabel));
        }
    }

    #[allow(clippy::too_many_lines)]
    fn compile_loop(&mut self, expr: &LoopExpression, dest: OpDestination) {
        let continue_label = self.compiler.code.new_label();
        let break_label = self.compiler.code.new_label();

        let mut outer_scope = self.enter_block(None);
        let (loop_start, previous_handler) = match &expr.kind {
            LoopKind::Infinite => {
                outer_scope.compiler.code.label(continue_label);
                (ValueOrSource::Label(continue_label), None)
            }
            LoopKind::While(condition) => {
                outer_scope.compiler.code.label(continue_label);
                let condition = outer_scope.compile_source(condition);
                outer_scope
                    .compiler
                    .code
                    .jump_if_not(break_label, condition, ());
                (ValueOrSource::Label(continue_label), None)
            }
            LoopKind::TailWhile(_) => {
                let tail_condition = outer_scope.compiler.code.new_label();
                outer_scope.compiler.code.label(tail_condition);
                (ValueOrSource::Label(tail_condition), None)
            }
            LoopKind::For { bindings, source } => {
                let mut variables = Vec::new();
                let mut splat = false;
                match &bindings.0 {
                    Expression::Lookup(lookup) if lookup.base.is_none() => {
                        let stack = outer_scope.new_temporary();
                        outer_scope.declare_local(lookup.name.clone(), true, stack, OpDestination::Void);
                        variables.push(stack);
                    }
                    Expression::Tuple(names) if names.iter().all(|name| matches!(&name.0, Expression::Lookup(lookup) if lookup.base.is_none())) => {
                        splat = true;
                        for name in names {
                            let Expression::Lookup(lookup) = &name.0 else { unreachable!("just matched")};
                            let stack = outer_scope.new_temporary();
                            outer_scope.declare_local(lookup.name.clone(), true, stack, OpDestination::Void);
                            variables.push(stack);
                        }
                    }
                    _ => todo!("invalid binding"),
                }

                let source = outer_scope.compile_expression_into_temporary(source);

                // We try to call :iterate on source. If this throws an
                // exception, we switch to trying to iterate by calling `nth()`
                // and counting.
                let iter_using_nth = outer_scope.compiler.code.new_label();
                let loop_body = outer_scope.compiler.code.new_label();
                let next_iteration = outer_scope.new_temporary();
                let previous_handler = outer_scope.new_temporary();
                let i = outer_scope.new_temporary();
                // The continue_label is going to be used for iterating over the
                // result of `source.iterate()`. We load this into the
                // `next_iteration` stack location, as we are going to use
                // `next_iteration` as the jump target. This allows us to jump
                // directly to the correct iteration code when continuing the
                // loop.
                let iterate_start = continue_label;
                outer_scope
                    .compiler
                    .code
                    .copy(iterate_start, next_iteration);
                // Install our exception handling, storing the previous handler.
                outer_scope
                    .compiler
                    .code
                    .set_exception_handler(iter_using_nth, previous_handler);
                // Invoke iterate(). If this isn't supported, we jump to `iter_using_nth`
                outer_scope
                    .compiler
                    .code
                    .invoke(source, Symbol::iterate_symbol(), 0);

                // Store the iterator in `i`.
                outer_scope.compiler.code.copy(Register(0), i);
                // Update exception handling to jump to break. This ultimately
                // should point to a separate set of instructions that checks
                // the error.
                let exception_handler = break_label;
                outer_scope
                    .compiler
                    .code
                    .set_exception_handler(exception_handler, ());
                // Begin the iteration.
                outer_scope.compiler.code.label(iterate_start);
                // Invoke next(), store the result in the first variable slot.
                outer_scope
                    .compiler
                    .code
                    .invoke(i, Symbol::next_symbol(), 0);
                outer_scope
                    .compiler
                    .code
                    .copy(Register(0), variables[variables.len() - 1]);

                // Jump to the common loop body code.
                outer_scope.compiler.code.jump(loop_body, ());

                // This code is for when source.iterate() faults.
                outer_scope.compiler.code.label(iter_using_nth);
                // Set the next_iteration to the correct address for nth-based
                // iteration, and update the exception handler to the break
                // label.
                outer_scope
                    .compiler
                    .code
                    .set_exception_handler(exception_handler, ());
                let nth_next_iteration = outer_scope.compiler.code.new_label();

                // Initialize the counter to 0. We then jump past the increment
                // code that will normally happen each loop.
                outer_scope
                    .compiler
                    .code
                    .copy(nth_next_iteration, next_iteration);
                outer_scope.compiler.code.copy(0, i);
                let after_first_increment = outer_scope.compiler.code.new_label();
                outer_scope.compiler.code.jump(after_first_increment, ());

                // Each iteration of the nth-based iteration will return here.
                // The first step is to increment i.
                outer_scope.compiler.code.label(nth_next_iteration);
                outer_scope.compiler.code.add(i, 1, i);
                outer_scope.compiler.code.label(after_first_increment);

                // Next, invoke nth(i)
                outer_scope.compiler.code.copy(i, Register(0));
                outer_scope
                    .compiler
                    .code
                    .invoke(source, Symbol::nth_symbol(), 1);
                // Copy the result into the first variable -- just like the
                // iterator() code.
                outer_scope
                    .compiler
                    .code
                    .copy(Register(0), variables[variables.len() - 1]);

                // Finally, the shared loop body.
                outer_scope.compiler.code.label(loop_body);
                if splat {
                    for (var, index) in variables.iter().zip(0_u8..) {
                        outer_scope.compiler.code.copy(index, Register(0));
                        outer_scope.compiler.code.invoke(
                            variables[variables.len() - 1],
                            Symbol::nth_symbol(),
                            1,
                        );
                        outer_scope.compiler.code.copy(Register(0), *var);
                    }
                }

                (ValueOrSource::Stack(next_iteration), Some(previous_handler))
            }
        };

        let mut loop_scope =
            outer_scope.enter_loop(expr.block.name.clone(), continue_label, break_label, dest);

        loop_scope.compile_expression(&expr.block.body, dest);
        drop(loop_scope);

        match &expr.kind {
            LoopKind::Infinite | LoopKind::While(_) | LoopKind::For { .. } => {}
            LoopKind::TailWhile(condition) => {
                outer_scope.compiler.code.label(continue_label);
                let condition = outer_scope.compile_source(condition);
                outer_scope
                    .compiler
                    .code
                    .jump_if_not(break_label, condition, ());
            }
        }

        outer_scope.compiler.code.jump(loop_start, ());
        outer_scope.compiler.code.label(break_label);

        if let Some(previous_handler) = previous_handler {
            outer_scope
                .compiler
                .code
                .set_exception_handler(previous_handler, ());
        }
        drop(outer_scope);
    }

    fn declare_variable(&mut self, decl: &Variable, dest: OpDestination) {
        let stack = self.new_temporary();
        self.compile_expression(&decl.value, OpDestination::Stack(stack));
        if decl.publish {
            self.compiler
                .code
                .declare(decl.name.clone(), decl.mutable, stack, dest);
        } else {
            self.declare_local(decl.name.clone(), decl.mutable, stack, dest);
        }
    }

    fn declare_local(&mut self, name: Symbol, mutable: bool, value: Stack, dest: OpDestination) {
        self.compiler.code.copy(value, dest);
        let previous_declaration = self
            .compiler
            .declarations
            .insert(
                name.clone(),
                BlockDeclaration {
                    stack: value,
                    mutable,
                },
            )
            .map(|field| field.value);
        self.local_declarations.push(LocalDeclaration {
            name,
            previous_declaration,
        });
    }

    fn ensure_in_module(&mut self, range: SourceRange) {
        if !self.module {
            self.compiler
                .errors
                .push(Ranged::new(range, Error::PubOnlyInModules));
        }
    }

    fn compile_unary(&mut self, unary: &UnaryExpression, dest: OpDestination) {
        let op = self.compile_source(&unary.operand);
        let kind = match unary.kind {
            syntax::UnaryKind::LogicalNot => UnaryKind::LogicalNot,
            syntax::UnaryKind::BitwiseNot => UnaryKind::BitwiseNot,
            syntax::UnaryKind::Negate => UnaryKind::Negate,
            syntax::UnaryKind::Copy => UnaryKind::Copy,
        };
        self.compiler.code.push(Op::Unary { op, dest, kind });
    }

    fn compile_binop(&mut self, binop: &BinaryExpression, dest: OpDestination) {
        let left = self.compile_source(&binop.left);
        let kind = match binop.kind {
            syntax::BinaryKind::Add => BinaryKind::Add,
            syntax::BinaryKind::Subtract => BinaryKind::Subtract,
            syntax::BinaryKind::Multiply => BinaryKind::Multiply,
            syntax::BinaryKind::Divide => BinaryKind::Divide,
            syntax::BinaryKind::IntegerDivide => BinaryKind::IntegerDivide,
            syntax::BinaryKind::Remainder => BinaryKind::Remainder,
            syntax::BinaryKind::Power => BinaryKind::Power,
            syntax::BinaryKind::Bitwise(kind) => BinaryKind::Bitwise(kind),
            syntax::BinaryKind::Compare(kind) => BinaryKind::Compare(kind),
            syntax::BinaryKind::Logical(LogicalKind::And) => {
                let after = self.compiler.code.new_label();
                self.compile_expression(&binop.left, dest);
                self.compiler.code.jump_if_not(after, dest, ());

                self.compile_expression(&binop.right, dest);
                self.compiler.code.truthy(dest, dest);

                self.compiler.code.label(after);

                return;
            }
            syntax::BinaryKind::Logical(LogicalKind::Or) => {
                let after = self.compiler.code.new_label();
                self.compile_expression(&binop.left, dest);
                self.compiler.code.jump_if(after, dest, ());

                self.compile_expression(&binop.right, dest);
                self.compiler.code.truthy(dest, dest);

                self.compiler.code.label(after);

                return;
            }
            syntax::BinaryKind::Logical(LogicalKind::Xor) => BinaryKind::LogicalXor,
        };
        let right = self.compile_source(&binop.right);
        self.compiler.code.push(Op::BinOp {
            op1: left,
            op2: right,
            dest,
            kind,
        });
    }

    fn compile_source(&mut self, source: &Ranged<Expression>) -> ValueOrSource {
        match &source.0 {
            Expression::Nil => ValueOrSource::Nil,
            Expression::Bool(bool) => ValueOrSource::Bool(*bool),
            Expression::Int(int) => ValueOrSource::Int(*int),
            Expression::Float(float) => ValueOrSource::Float(*float),
            Expression::RegEx(regex) => ValueOrSource::RegEx(regex.clone()),
            Expression::String(string) => ValueOrSource::String(string.clone()),
            Expression::Symbol(symbol) => ValueOrSource::Symbol(symbol.clone()),
            Expression::Lookup(lookup) => {
                if let Some(decl) = self.compiler.declarations.get(&lookup.name) {
                    ValueOrSource::Stack(decl.stack)
                } else {
                    ValueOrSource::Stack(self.compile_expression_into_temporary(source))
                }
            }
            Expression::Map(_)
            | Expression::List(_)
            | Expression::Tuple(_)
            | Expression::If(_)
            | Expression::Function(_)
            | Expression::Module(_)
            | Expression::Call(_)
            | Expression::Index(_)
            | Expression::Variable(_)
            | Expression::Assign(_)
            | Expression::Unary(_)
            | Expression::Binary(_)
            | Expression::Block(_)
            | Expression::Loop(_)
            | Expression::Chain(_)
            | Expression::Break(_)
            | Expression::Continue(_)
            | Expression::Return(_) => {
                ValueOrSource::Stack(self.compile_expression_into_temporary(source))
            }
        }
    }

    fn compile_expression_into_temporary(&mut self, source: &Ranged<Expression>) -> Stack {
        let var = self.new_temporary();
        self.compile_expression(source, OpDestination::Stack(var));
        var
    }

    fn compile_function_call(
        &mut self,
        call: &FunctionCall,
        range: SourceRange,
        dest: OpDestination,
    ) {
        let arity = if let Ok(arity) = u8::try_from(call.parameters.len()) {
            arity
        } else {
            self.compiler
                .errors
                .push(Ranged::new(range, Error::TooManyArguments));
            255
        };
        let function = match &call.function.0 {
            Expression::Lookup(lookup)
                if self
                    .compiler
                    .function_name
                    .as_ref()
                    .map_or(false, |f| f == &lookup.name) =>
            {
                ValueOrSource::Nil
            }
            Expression::Lookup(lookup) if lookup.base.is_some() => {
                let base = lookup.base.as_ref().expect("just matched");
                let base = self.compile_source(base);

                self.compile_function_args(&call.parameters, arity);
                self.compiler.code.invoke(base, lookup.name.clone(), arity);
                self.compiler.code.copy(Register(0), dest);
                return;
            }
            _ => self.compile_source(&call.function),
        };

        self.compile_function_args(&call.parameters, arity);
        self.compiler.code.call(function, arity);
        self.compiler.code.copy(Register(0), dest);
    }

    fn compile_index(&mut self, index: &Index, range: SourceRange, dest: OpDestination) {
        let arity = if let Ok(arity) = u8::try_from(index.parameters.len()) {
            arity
        } else {
            self.compiler
                .errors
                .push(Ranged::new(range, Error::TooManyArguments));
            255
        };

        let target = self.compile_source(&index.target);
        self.compile_function_args(&index.parameters, arity);
        self.compiler
            .code
            .invoke(target, Symbol::get_symbol().clone(), arity);
        self.compiler.code.copy(Register(0), dest);
    }

    fn compile_function_args(&mut self, args: &[Ranged<Expression>], arity: u8) {
        let mut parameters = Vec::with_capacity(args.len());
        for param in &args[..usize::from(arity)] {
            parameters.push(self.compile_source(param));
        }
        for (parameter, index) in parameters.into_iter().zip(0..arity) {
            self.compiler.code.copy(parameter, Register(index));
        }
    }
}

impl Drop for Scope<'_> {
    fn drop(&mut self) {
        self.compiler.scopes.pop();
        // The root scope preserves its declarations.
        if self.depth > 0 {
            while let Some(LocalDeclaration {
                name,
                previous_declaration,
            }) = self.local_declarations.pop()
            {
                match previous_declaration {
                    Some(previous) => {
                        self.compiler.declarations.insert(name, previous);
                    }
                    None => {
                        self.compiler.declarations.remove(&name);
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Error {
    UnknownVariable,
    VariableNotMutable,
    TooManyArguments,
    UsizeTooLarge,
    InvalidDeclaration,
    InvalidLabel,
    PubOnlyInModules,
    ExpectedBlock,
    Syntax(syntax::Error),
}

impl From<Ranged<syntax::Error>> for Ranged<Error> {
    fn from(err: Ranged<syntax::Error>) -> Self {
        err.map(Error::Syntax)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum UnaryKind {
    Truthy,
    LogicalNot,
    BitwiseNot,
    Negate,
    Copy,
    Resolve,
    Jump,
    NewMap,
    NewList,
    SetExceptionHandler,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BitcodeModule {
    pub name: Symbol,
    pub initializer: BitcodeBlock,
}
