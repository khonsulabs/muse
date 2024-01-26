use std::mem;

use kempt::Map;
use serde::{Deserialize, Serialize};

use crate::symbol::Symbol;
use crate::syntax::{
    self, BinaryExpression, BinaryKind, Expression, LogicalKind, Ranged, UnaryExpression, Variable,
};
use crate::vm::bitcode::{BitcodeBlock, BitcodeFunction, Op, OpDestination, ValueOrSource};
use crate::vm::ops::Stack;
use crate::vm::{Code, Register};

#[derive(Default, Debug)]
pub struct Compiler {
    function_name: Option<Symbol>,
    parsed: Vec<Result<Ranged<Expression>, Ranged<syntax::Error>>>,
    errors: Vec<Ranged<Error>>,
    code: BitcodeBlock,
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

    pub fn build(mut self) -> Result<Code, Vec<Ranged<Error>>> {
        let mut expressions = Vec::with_capacity(self.parsed.len());

        for result in self.parsed.drain(..) {
            match result {
                Ok(expr) => expressions.push(expr),
                Err(err) => self.errors.push(err.into()),
            }
        }

        let expression = match expressions.len() {
            0 => Ranged::new(0..0, Expression::Nil),
            1 => mem::take(&mut expressions[0]),
            len => {
                let range = expressions[0].range().start..expressions[len - 1].range().end();
                Ranged::new(
                    range,
                    Expression::Block {
                        name: None,
                        expressions,
                    },
                )
            }
        };

        Scope::root(&mut self, &mut Map::new())
            .compile_expression(&expression, OpDestination::Register(Register(0)));

        if self.errors.is_empty() {
            Ok(Code::from(&self.code))
        } else {
            Err(self.errors)
        }
    }

    fn new_variable(&mut self) -> Stack {
        let id = self.code.stack_requirement;
        self.code.stack_requirement += 1;
        Stack(id)
    }
}

struct BlockDeclaration {
    stack: Stack,
    mutable: bool,
}

struct LocalDeclaration {
    name: Symbol,
    previous_declaration: Option<BlockDeclaration>,
}

struct Scope<'a> {
    compiler: &'a mut Compiler,
    declarations: &'a mut Map<Symbol, BlockDeclaration>,
    locals_count: usize,
    local_declarations: Vec<LocalDeclaration>,
}

impl<'a> Scope<'a> {
    fn root(compiler: &'a mut Compiler, variables: &'a mut Map<Symbol, BlockDeclaration>) -> Self {
        Self {
            compiler,
            locals_count: 0,
            declarations: variables,
            local_declarations: Vec::new(),
        }
    }

    fn enter_block(&mut self) -> Scope<'_> {
        Scope {
            compiler: self.compiler,
            declarations: self.declarations,
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
            Expression::Nil => self.compiler.code.push(Op::Unary {
                op: ValueOrSource::Nil,
                dest,
                kind: UnaryKind::Copy,
            }),
            Expression::Bool(bool) => self.compiler.code.push(Op::Unary {
                op: ValueOrSource::Bool(*bool),
                dest,
                kind: UnaryKind::Copy,
            }),
            Expression::Int(int) => self.compiler.code.push(Op::Unary {
                op: ValueOrSource::Int(*int),
                dest,
                kind: UnaryKind::Copy,
            }),
            Expression::Float(float) => self.compiler.code.push(Op::Unary {
                op: ValueOrSource::Float(*float),
                dest,
                kind: UnaryKind::Copy,
            }),
            Expression::Lookup(lookup) => {
                if let Some(base) = &lookup.base {
                    let target = self.compile_source(base);
                    self.compiler.code.copy(lookup.name.clone(), Register(0));
                    self.compiler.code.push(Op::Invoke {
                        target,
                        arity: ValueOrSource::Int(1),
                        name: Symbol::get_symbol(),
                        dest,
                    });
                } else if let Some(var) = self.declarations.get(&lookup.name) {
                    self.compiler.code.push(Op::Unary {
                        op: ValueOrSource::Stack(var.stack),
                        dest,
                        kind: UnaryKind::Copy,
                    });
                } else {
                    self.compiler.code.push(Op::Unary {
                        op: ValueOrSource::Symbol(lookup.name.clone()),
                        dest,
                        kind: UnaryKind::Resolve,
                    });
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
            Expression::Assign(assign) => {
                if let Some(base) = &assign.target.base {
                    self.compile_expression(&assign.value, OpDestination::Register(Register(1)));
                    self.compiler
                        .code
                        .copy(assign.target.name.clone(), Register(0));
                    let target = self.compile_source(base);
                    self.compiler.code.push(Op::Invoke {
                        target,
                        name: Symbol::set_symbol(),
                        arity: ValueOrSource::Int(2),
                        dest,
                    });
                } else if let Some(var) = self.declarations.get(&assign.target.name) {
                    if var.mutable {
                        let var = var.stack;
                        self.compile_expression(&assign.value, OpDestination::Stack(var));
                        self.compiler.code.push(Op::Unary {
                            op: ValueOrSource::Stack(var),
                            dest,
                            kind: UnaryKind::Copy,
                        });
                    } else {
                        self.compiler
                            .errors
                            .push(Ranged::new(expr.range(), Error::VariableNotMutable));
                    }
                } else {
                    self.compiler
                        .errors
                        .push(Ranged::new(expr.range(), Error::UnknownVariable));
                }
            }
            Expression::Unary(unary) => self.compile_unary(unary, dest),
            Expression::Binary(binop) => {
                self.compile_binop(binop, dest);
            }
            Expression::Block { expressions, .. } => {
                let mut block = self.enter_block();
                for (index, exp) in expressions.iter().enumerate() {
                    let dest = if index == expressions.len() - 1 {
                        dest
                    } else {
                        OpDestination::Void
                    };
                    block.compile_expression(exp, dest);
                }
            }
            Expression::Call(call) => {
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
                    _ => self.compile_source(&call.function),
                };
                let arity = if let Ok(arity) = u8::try_from(call.parameters.len()) {
                    arity
                } else {
                    self.compiler
                        .errors
                        .push(Ranged::new(expr.range(), Error::TooManyArguments));
                    255
                };
                let mut parameters = Vec::with_capacity(call.parameters.len());
                for param in &call.parameters[..usize::from(arity)] {
                    parameters.push(self.compile_source(param));
                }
                for (parameter, index) in parameters.into_iter().zip(0..arity) {
                    self.compiler.code.copy(parameter, Register(index));
                }
                self.compiler.code.call(function, arity, dest);
            }
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
                    function_name: Some(decl.name.0.clone()),
                    ..Compiler::default()
                };
                let mut fn_variables = Map::new();
                let mut fn_scope = Scope::root(&mut fn_compiler, &mut fn_variables);
                for (var, index) in decl.parameters.iter().zip(0..arity) {
                    let stack = fn_scope.new_temporary();
                    fn_scope.compiler.code.copy(Register(index), stack);
                    fn_scope.declarations.insert(
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
                let fun = BitcodeFunction::new(&decl.name.0).when(arity, fn_compiler.code);
                self.compiler.code.declare_function(fun);
            }
        }
    }

    fn declare_variable(&mut self, decl: &Variable, dest: OpDestination) {
        let stack = self.new_temporary();
        self.compile_expression(&decl.value, OpDestination::Stack(stack));
        self.compiler.code.push(Op::Unary {
            op: ValueOrSource::Stack(stack),
            dest,
            kind: UnaryKind::Copy,
        });
        let previous_declaration = self
            .declarations
            .insert(
                decl.name.clone(),
                BlockDeclaration {
                    stack,
                    mutable: decl.mutable,
                },
            )
            .map(|field| field.value);
        self.local_declarations.push(LocalDeclaration {
            name: decl.name.clone(),
            previous_declaration,
        });
    }

    fn compile_unary(&mut self, unary: &UnaryExpression, dest: OpDestination) {
        let op = self.compile_source(&unary.operand);
        self.compiler.code.push(Op::Unary {
            op,
            dest,
            kind: match unary.kind {
                syntax::UnaryKind::LogicalNot => UnaryKind::LogicalNot,
                syntax::UnaryKind::BitwiseNot => UnaryKind::BitwiseNot,
                syntax::UnaryKind::Negate => UnaryKind::Negate,
                syntax::UnaryKind::Copy => UnaryKind::Copy,
            },
        });
    }

    fn compile_binop(&mut self, binop: &BinaryExpression, dest: OpDestination) {
        if matches!(
            binop.kind,
            BinaryKind::Logical(LogicalKind::And | LogicalKind::Or)
        ) {
            todo!("implement short circuiting logic operators")
        }
        self.compile_basic_binop(&binop.left, &binop.right, dest, binop.kind);
    }

    fn compile_basic_binop(
        &mut self,
        lhs: &Ranged<Expression>,
        rhs: &Ranged<Expression>,
        dest: OpDestination,
        kind: BinaryKind,
    ) {
        let left = self.compile_source(lhs);
        let right = self.compile_source(rhs);
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
            Expression::Lookup(lookup) => {
                if let Some(decl) = self.declarations.get(&lookup.name) {
                    ValueOrSource::Stack(decl.stack)
                } else {
                    self.compile_expression_into_temporary(source)
                }
            }
            Expression::Map(_)
            | Expression::If(_)
            | Expression::Function(_)
            | Expression::Call(_)
            | Expression::Variable(_)
            | Expression::Assign(_)
            | Expression::Unary(_)
            | Expression::Binary(_)
            | Expression::Block { .. } => self.compile_expression_into_temporary(source),
        }
    }

    fn compile_expression_into_temporary(&mut self, source: &Ranged<Expression>) -> ValueOrSource {
        let var = self.new_temporary();
        self.compile_expression(source, OpDestination::Stack(var));
        ValueOrSource::Stack(var)
    }
}

impl Drop for Scope<'_> {
    fn drop(&mut self) {
        while let Some(LocalDeclaration {
            name,
            previous_declaration,
        }) = self.local_declarations.pop()
        {
            match previous_declaration {
                Some(previous) => {
                    self.declarations.insert(name, previous);
                }
                None => {
                    self.declarations.remove(&name);
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
    Syntax(syntax::Error),
}

impl From<Ranged<syntax::Error>> for Ranged<Error> {
    fn from(err: Ranged<syntax::Error>) -> Self {
        err.map(Error::Syntax)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum UnaryKind {
    LogicalNot,
    BitwiseNot,
    Negate,
    Copy,
    Resolve,
    Jump,
    NewMap,
}
