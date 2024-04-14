//! The Muse compiler.

use std::collections::VecDeque;
use std::fmt::{Debug, Display};
use std::ops::{BitOr, BitOrAssign, Range};
use std::slice;
use std::sync::Arc;

use kempt::{Map, Set};
use refuse::CollectionGuard;
use regex::Regex;
use serde::{Deserialize, Serialize};

pub mod syntax;

use syntax::token::Token;
use syntax::{
    AssignTarget, BinaryExpression, BreakExpression, CompareKind, ContinueExpression, Delimited,
    DelimitedIter, EntryKeyPattern, Expression, FunctionCall, FunctionDefinition, Index, Literal,
    LogicalKind, Lookup, LoopExpression, LoopKind, MatchExpression, MatchPattern, Matches,
    PatternKind, Ranged, SingleMatch, SourceCode, SourceRange, TryExpression, UnaryExpression,
};

use crate::runtime::symbol::Symbol;
use crate::vm::bitcode::{
    Access, BinaryKind, BitcodeBlock, BitcodeFunction, FaultKind, Label, Op, OpDestination,
    ValueOrSource,
};
use crate::vm::{Code, Register, Stack};

/// A Muse compiler instance.
#[derive(Debug)]
pub struct Compiler {
    function_name: Option<Symbol>,
    parsed: Vec<Result<Ranged<Expression>, Ranged<syntax::ParseError>>>,
    errors: Vec<Ranged<Error>>,
    code: BitcodeBlock,
    declarations: Map<Symbol, BlockDeclaration>,
    scopes: Vec<ScopeInfo>,
    macros: Map<Symbol, Macro>,
    infix_macros: Map<Symbol, InfixMacro>,
}

impl Compiler {
    /// Pushes `source` into the list of sources to build.
    pub fn push<'a>(&mut self, source: impl Into<SourceCode<'a>>) {
        let mut parsed = syntax::parse(source);
        if let Ok(parsed) = &mut parsed {
            self.expand_macros(parsed);
        }
        self.parsed.push(parsed);
    }

    /// Adds `source` to this compiler, and returns self.
    #[must_use]
    pub fn with<'a>(mut self, source: impl Into<SourceCode<'a>>) -> Self {
        self.push(source.into());
        self
    }

    /// Adds a macro to this compiler instance.
    ///
    /// Macros are functions that accept a `VecDeque<Ranged<Token>>` and return
    /// a `VecDeque<Ranged<Token>>`.
    pub fn push_macro<M>(&mut self, name: impl Into<Symbol>, func: M)
    where
        M: MacroFn + 'static,
    {
        self.macros.insert(name.into(), Macro(Box::new(func)));
    }

    /// Adds a macro to this compiler instance and returns self.
    ///
    /// Macros are functions that accept a `VecDeque<Ranged<Token>>` and return
    /// a `VecDeque<Ranged<Token>>`.
    #[must_use]
    pub fn with_macro<M>(mut self, name: impl Into<Symbol>, func: M) -> Self
    where
        M: MacroFn + 'static,
    {
        self.push_macro(name, func);
        self
    }

    /// Adds an infix macro to this compiler instance.
    ///
    /// Infix macros are functions that accept a `Ranged<Expression>` and a
    /// `VecDeque<Ranged<Token>>` and returns a `VecDeque<Ranged<Token>>`.
    pub fn push_infix_macro<M>(&mut self, name: impl Into<Symbol>, func: M)
    where
        M: InfixMacroFn + 'static,
    {
        self.infix_macros
            .insert(name.into(), InfixMacro(Box::new(func)));
    }

    /// Adds an infix macro to this compiler instance, and returns self.
    ///
    /// Infix macros are functions that accept a `Ranged<Expression>` and a
    /// `VecDeque<Ranged<Token>>` and returns a `VecDeque<Ranged<Token>>`.
    #[must_use]
    pub fn with_infix_macro<M>(mut self, name: impl Into<Symbol>, func: M) -> Self
    where
        M: InfixMacroFn + 'static,
    {
        self.push_infix_macro(name, func);
        self
    }

    /// Compile's `source`.
    pub fn compile<'a>(
        source: impl Into<SourceCode<'a>>,
        guard: &CollectionGuard,
    ) -> Result<Code, Vec<Ranged<Error>>> {
        Self::default().with(source).build(guard)
    }

    /// Builds all of the sources added to this compiler into a single code
    /// block.
    pub fn build(&mut self, guard: &CollectionGuard) -> Result<Code, Vec<Ranged<Error>>> {
        self.code.clear();
        let mut expressions = Vec::with_capacity(self.parsed.len());

        for result in self.parsed.drain(..) {
            match result {
                Ok(expr) => expressions.push(expr),
                Err(err) => self.errors.push(err.into()),
            }
        }

        let expression = Expression::chain(expressions, Vec::new());

        Scope::module_root(self)
            .compile_expression(&expression, OpDestination::Register(Register(0)));

        if self.errors.is_empty() {
            Ok(self.code.to_code(guard))
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn new_variable(&mut self) -> Stack {
        let id = self.code.stack_requirement;
        self.code.stack_requirement += 1;
        Stack(id)
    }

    fn parse_macro_expansion(&mut self, tokens: VecDeque<Ranged<Token>>) -> Ranged<Expression> {
        match syntax::parse_tokens(tokens) {
            Ok(mut expanded) => {
                self.expand_macros(&mut expanded);
                expanded
            }
            Err(err) => {
                let range = err.range();
                self.errors.push(err.map(Error::SigilSyntax));
                Ranged::new(range, Expression::Literal(Literal::Nil))
            }
        }
    }

    /// Perform macro expansion on `expr`.
    ///
    /// After this function executions `expr` and all subexpressions will not
    /// contain any [`Expression::Macro`]s.
    #[allow(clippy::too_many_lines)]
    pub fn expand_macros(&mut self, expr: &mut Ranged<Expression>) {
        match &mut expr.0 {
            Expression::Macro(e) => {
                if let Some(m) = self.macros.get_mut(&e.name.0) {
                    let tokens = std::mem::take(&mut e.tokens);
                    let tokens = m.0.transform(tokens);
                    *expr = self.parse_macro_expansion(tokens);
                } else {
                    self.errors
                        .push(Ranged::new(e.name.range(), Error::UnknownSigil));
                }
            }
            Expression::InfixMacro(e) => {
                if let Some(m) = self.infix_macros.get_mut(&e.invocation.name.0) {
                    let tokens = std::mem::take(&mut e.invocation.tokens);
                    let tokens = m.0.transform(&e.subject, tokens);
                    *expr = self.parse_macro_expansion(tokens);
                } else {
                    self.errors
                        .push(Ranged::new(e.invocation.name.range(), Error::UnknownSigil));
                }
            }
            Expression::RootModule | Expression::Literal(_) | Expression::Continue(_) => {}
            Expression::Lookup(e) => {
                self.expand_macros_in_lookup(e);
            }
            Expression::If(e) => {
                self.expand_macros(&mut e.condition);
                self.expand_macros(&mut e.when_true);
                if let Some(when_false) = &mut e.when_false {
                    self.expand_macros(&mut when_false.expression);
                }
            }
            Expression::Match(e) => {
                self.expand_macros(&mut e.condition);

                self.expand_macros_in_matches(&mut e.matches);
            }
            Expression::Try(e) => {
                self.expand_macros(&mut e.body);
                if let Some(catch) = &mut e.catch {
                    self.expand_macros_in_matches(&mut catch.matches);
                }
            }
            Expression::TryOrNil(e) => {
                self.expand_macros(&mut e.body);
            }
            Expression::Throw(e) => {
                self.expand_macros(&mut e.value);
            }
            Expression::Map(e) => {
                for field in &mut e.fields.enclosed {
                    self.expand_macros(&mut field.key);
                    self.expand_macros(&mut field.value);
                }
            }
            Expression::List(list) => {
                for value in &mut list.values.enclosed {
                    self.expand_macros(value);
                }
            }
            Expression::Call(e) => {
                for value in &mut e.parameters.enclosed {
                    self.expand_macros(value);
                }
            }
            Expression::Index(e) => {
                self.expand_macros_in_index(e);
            }
            Expression::Assign(e) => {
                match &mut e.target.0 {
                    AssignTarget::Index(index) => self.expand_macros_in_index(index),
                    AssignTarget::Lookup(lookup) => self.expand_macros_in_lookup(lookup),
                }
                self.expand_macros(&mut e.value);
            }
            Expression::Unary(e) => self.expand_macros(&mut e.operand),
            Expression::Binary(e) => {
                self.expand_macros(&mut e.left);
                self.expand_macros(&mut e.right);
            }
            Expression::Block(e) => {
                self.expand_macros(&mut e.body.enclosed);
            }
            Expression::Loop(e) => {
                match &mut e.kind {
                    LoopKind::Infinite => {}
                    LoopKind::While(expression) | LoopKind::TailWhile { expression, .. } => {
                        self.expand_macros(expression);
                    }
                    LoopKind::For {
                        pattern, source, ..
                    } => {
                        if let Some(guard) = &mut pattern.guard {
                            self.expand_macros(guard);
                        }
                        self.expand_macros(source);
                    }
                }
                self.expand_macros(&mut e.block.body.enclosed);
            }
            Expression::Break(e) => self.expand_macros(&mut e.value),
            Expression::Return(e) => {
                self.expand_macros(&mut e.value);
            }
            Expression::Module(e) => {
                self.expand_macros(&mut e.contents.body.enclosed);
            }
            Expression::Function(e) => {
                self.expand_macros_in_matches(&mut e.body);
            }
            Expression::SingleMatch(e) => {
                self.expand_macros(&mut e.value);
                if let Some(guard) = &mut e.pattern.guard {
                    self.expand_macros(guard);
                }
                if let Some(else_expr) = &mut e.r#else {
                    self.expand_macros(&mut else_expr.expression);
                }
            }
            Expression::Group(e) => {
                self.expand_macros(&mut e.enclosed);
            }
            Expression::FormatString(parts) => {
                for (joiner, _) in &mut parts.remaining {
                    self.expand_macros(joiner);
                }
            }
        }
    }

    fn expand_macros_in_lookup(&mut self, e: &mut Lookup) {
        if let Some(base) = &mut e.base {
            self.expand_macros(&mut base.expression);
        }
    }

    fn expand_macros_in_index(&mut self, e: &mut Index) {
        self.expand_macros(&mut e.target);
        for value in &mut e.parameters.enclosed {
            self.expand_macros(value);
        }
    }

    fn expand_macros_in_matches(&mut self, matches: &mut Matches) {
        for pattern in &mut matches.patterns {
            if let Some(guard) = &mut pattern.pattern.guard {
                self.expand_macros(guard);
            }
            self.expand_macros(&mut pattern.body);
        }
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self {
            function_name: None,
            parsed: Vec::new(),
            errors: Vec::new(),
            code: BitcodeBlock::default(),
            declarations: Map::new(),
            scopes: Vec::new(),
            macros: Map::new(),
            infix_macros: Map::new(),
        }
        .with_macro("$assert", |mut tokens: VecDeque<Ranged<Token>>| {
            let start_range = tokens.front().map(Ranged::range).unwrap_or_default();
            let end_range = tokens.back().map(Ranged::range).unwrap_or_default();
            tokens.push_back(Ranged::new(
                end_range,
                Token::Identifier(Symbol::else_symbol().clone()),
            ));
            tokens.push_back(Ranged::new(
                end_range,
                Token::Identifier(Symbol::throw_symbol().clone()),
            ));
            tokens.push_back(Ranged::new(
                end_range,
                Token::Symbol(Symbol::from("assertion_failed")),
            ));
            // tokens.push_back(Ranged::default_for(Token::Close(Paired::Paren)));
            // tokens.push_front(Ranged::default_for(Token::Open(Paired::Paren)));
            tokens.push_front(Ranged::new(start_range, Token::Char('!')));
            tokens.push_front(Ranged::new(start_range, Token::Char('=')));
            tokens.push_front(Ranged::new(
                start_range,
                Token::Identifier(Symbol::false_symbol().clone()),
            ));
            tokens.push_front(Ranged::new(
                start_range,
                Token::Identifier(Symbol::let_symbol().clone()),
            ));
            tokens
        })
    }
}

/// A function that can be used as a macro in a [`Compiler`].
pub trait MacroFn: Send + Sync {
    /// Returns a series of tokens from the given tokens.
    fn transform(&mut self, tokens: VecDeque<Ranged<Token>>) -> VecDeque<Ranged<Token>>;
}

impl<F> MacroFn for F
where
    F: FnMut(VecDeque<Ranged<Token>>) -> VecDeque<Ranged<Token>> + Send + Sync,
{
    fn transform(&mut self, tokens: VecDeque<Ranged<Token>>) -> VecDeque<Ranged<Token>> {
        self(tokens)
    }
}

struct Macro(Box<dyn MacroFn>);

impl Debug for Macro {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Macro").finish_non_exhaustive()
    }
}

/// A function that can be used as an infix macro in a [`Compiler`].
pub trait InfixMacroFn: Send + Sync {
    /// Returns a series of tokens from the given expression and tokens.
    fn transform(
        &mut self,
        expression: &Ranged<Expression>,
        tokens: VecDeque<Ranged<Token>>,
    ) -> VecDeque<Ranged<Token>>;
}

impl<F> InfixMacroFn for F
where
    F: FnMut(&Ranged<Expression>, VecDeque<Ranged<Token>>) -> VecDeque<Ranged<Token>> + Send + Sync,
{
    fn transform(
        &mut self,
        expression: &Ranged<Expression>,
        tokens: VecDeque<Ranged<Token>>,
    ) -> VecDeque<Ranged<Token>> {
        self(expression, tokens)
    }
}

struct InfixMacro(Box<dyn InfixMacroFn>);

impl Debug for InfixMacro {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InfixMacro").finish_non_exhaustive()
    }
}

/// A declaration made within a block.
#[derive(Debug, Clone, Copy)]
pub struct BlockDeclaration {
    /// The stack location of the value.
    pub stack: Stack,
    /// If true, this declaration can be reassigned to.
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

    fn is_module_root(&self) -> bool {
        self.module && self.depth == 0
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
        self.compiler.code.set_current_source_range(expr.range());
        match &**expr {
            Expression::RootModule => {
                self.compiler
                    .code
                    .resolve(Symbol::sigil_symbol().clone(), dest);
            }
            Expression::Literal(literal) => self.compile_literal(literal, dest),
            Expression::Lookup(lookup) => {
                if let Some(base) = &lookup.base {
                    let (target, after_label) =
                        if let Expression::TryOrNil(try_op) = &base.expression.0 {
                            let exception_error = self.compiler.code.new_label();
                            let after_expression = self.compiler.code.new_label();
                            let after_handler = self.compiler.code.new_label();
                            let previous_handler = self.new_temporary();
                            self.compiler
                                .code
                                .set_exception_handler(exception_error, previous_handler);

                            let base = self.compile_source(&try_op.body);
                            self.compiler.code.set_current_source_range(expr.range());
                            self.compiler.code.copy((), dest);
                            self.compiler
                                .code
                                .set_exception_handler(previous_handler, ());
                            self.compiler.code.jump(after_handler, ());

                            self.compiler.code.label(exception_error);
                            self.compiler.code.copy((), dest);
                            self.compiler
                                .code
                                .set_exception_handler(previous_handler, ());
                            self.compiler.code.jump(after_expression, ());

                            self.compiler.code.label(after_handler);
                            (base, Some(after_expression))
                        } else {
                            (self.compile_source(&base.expression), None)
                        };
                    self.compiler.code.set_current_source_range(expr.range());
                    self.compiler.code.copy(lookup.name.0.clone(), Register(0));
                    self.compiler
                        .code
                        .invoke(target, Symbol::get_symbol().clone(), 1);
                    self.compiler.code.copy(Register(0), dest);

                    if let Some(label) = after_label {
                        self.compiler.code.label(label);
                    }
                } else if let Some(var) = self.compiler.declarations.get(&lookup.name.0) {
                    self.compiler.code.copy(var.stack, dest);
                } else {
                    self.compiler.code.resolve(lookup.name.0.clone(), dest);
                }
            }
            Expression::Map(map) => {
                let mut elements = Vec::with_capacity(map.fields.enclosed.len());
                for field in &map.fields.enclosed {
                    let key = self.compile_expression_into_temporary(&field.key);
                    let value = self.compile_expression_into_temporary(&field.value);
                    elements.push((key, value));
                }

                let mut num_elements = u8::try_from(elements.len()).unwrap_or(u8::MAX);
                if num_elements >= 128 {
                    num_elements = 127;
                    eprintln!("TODO Ignoring more than 127 elements in map");
                }

                self.compiler.code.set_current_source_range(expr.range());
                for (index, (key, value)) in (0..=num_elements).zip(elements) {
                    self.compiler.code.copy(key, Register(index * 2));
                    self.compiler.code.copy(value, Register(index * 2 + 1));
                }
                self.compiler.code.new_map(num_elements, dest);
            }
            Expression::List(list) => {
                let mut elements = Vec::with_capacity(list.values.enclosed.len());
                for field in &list.values.enclosed {
                    let value = self.compile_expression_into_temporary(field);
                    elements.push(value);
                }

                let num_elements = if let Ok(elements) = u8::try_from(elements.len()) {
                    elements
                } else {
                    eprintln!("TODO Ignoring more than 127 elements in map");
                    255
                };

                self.compiler.code.set_current_source_range(expr.range());
                for (index, value) in (0..=num_elements).zip(elements) {
                    self.compiler.code.copy(value, Register(index));
                }
                self.compiler.code.new_list(num_elements, dest);
            }
            Expression::If(if_expr) => {
                let condition = self.compile_source(&if_expr.condition);
                let if_true = self.compiler.code.new_label();
                self.compiler.code.set_current_source_range(expr.range());
                self.compiler.code.jump_if(if_true, condition, ());
                let after_true = self.compiler.code.new_label();
                if let Some(when_false) = &if_expr.when_false {
                    self.compile_expression(&when_false.expression, dest);
                    self.compiler.code.set_current_source_range(expr.range());
                }
                self.compiler.code.jump(after_true, ());
                self.compiler.code.label(if_true);
                self.compile_expression(&if_expr.when_true, dest);
                self.compiler.code.label(after_true);
            }
            Expression::Match(match_expr) => self.compile_match_expression(match_expr, dest),
            Expression::Loop(loop_expr) => {
                self.compile_loop(loop_expr, expr.range(), dest);
            }
            Expression::Break(break_expr) => {
                self.compile_break(break_expr, expr.1);
            }
            Expression::Continue(continue_expr) => {
                self.compile_continue(continue_expr, expr.1);
            }
            Expression::Return(result) => {
                self.compile_expression(&result.value, OpDestination::Register(Register(0)));
                self.compiler.code.set_current_source_range(expr.range());
                self.compiler.code.return_early();
            }
            Expression::Assign(assign) => match &assign.target.0 {
                AssignTarget::Lookup(target) => {
                    if let Some(base) = &target.base {
                        let target_source = self.compile_source(&base.expression);
                        self.compile_expression(
                            &assign.value,
                            OpDestination::Register(Register(1)),
                        );
                        self.compiler.code.set_current_source_range(expr.range());
                        self.compiler.code.copy(target.name.0.clone(), Register(0));
                        self.compiler
                            .code
                            .invoke(target_source, Symbol::set_symbol().clone(), 2);
                        self.compiler.code.copy(Register(0), dest);
                    } else if let Some(var) = self.compiler.declarations.get(&target.name.0) {
                        if var.mutable {
                            let var = var.stack;
                            self.compile_expression(&assign.value, OpDestination::Stack(var));
                            self.compiler.code.set_current_source_range(expr.range());
                            self.compiler.code.copy(var, dest);
                        } else {
                            self.compiler
                                .errors
                                .push(Ranged::new(expr.range(), Error::VariableNotMutable));
                        }
                    } else {
                        let value = self.compile_source(&assign.value);
                        self.compiler.code.set_current_source_range(expr.range());
                        self.compiler
                            .code
                            .assign(target.name.0.clone(), value, dest);
                    }
                }
                AssignTarget::Index(index) => {
                    let target_source = self.compile_source(&index.target);
                    let value = self.compile_source(&assign.value);
                    let arity = if let Some(arity) = index
                        .parameters
                        .enclosed
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
                    self.compile_function_args(&index.parameters.enclosed, arity - 1);
                    self.compiler.code.set_current_source_range(expr.range());
                    self.compiler
                        .code
                        .copy(value, OpDestination::Register(Register(arity - 1)));
                    self.compiler
                        .code
                        .invoke(target_source, Symbol::set_symbol().clone(), arity);
                }
            },
            Expression::Unary(unary) => self.compile_unary(unary, expr.range(), dest),
            Expression::Binary(binop) => {
                self.compile_binop(binop, expr.range(), dest);
            }
            Expression::Block(block) => {
                let break_label = self.compiler.code.new_label();
                let mut scope = self.enter_block(
                    block
                        .label
                        .as_ref()
                        .map(|label| (label.name.0.clone(), break_label, dest)),
                );
                scope.compile_expression(&block.body.enclosed, dest);
                scope.compiler.code.set_current_source_range(expr.range());
                scope.compiler.code.label(break_label);
            }
            Expression::Try(try_expr) => self.compile_try(try_expr, expr.range(), dest),
            Expression::TryOrNil(try_expr) => self.compile_try_catch(
                expr.range(),
                |this| this.compile_expression(&try_expr.body, dest),
                |this| this.compiler.code.copy((), dest),
            ),
            Expression::Throw(throw) => self.compile_throw(&throw.value, expr.range()),
            Expression::Call(call) => self.compile_function_call(call, expr.range(), dest),
            Expression::Index(index) => self.compile_index(index, expr.range(), dest),
            Expression::SingleMatch(decl) => {
                self.compile_single_match(decl, expr.range(), dest);
            }
            Expression::Function(decl) => {
                self.compile_function(decl, expr.range(), dest);
            }
            Expression::Module(module) => {
                let block = &module.contents.0;
                let mut mod_compiler = Compiler::default();
                let mut mod_scope = Scope::module_root(&mut mod_compiler);

                mod_scope.compile_expression(&block.body.enclosed, OpDestination::Void);
                drop(mod_scope);
                self.compiler.code.set_current_source_range(expr.range());
                let name = &module.name.0;
                let instance = BitcodeModule {
                    name: name.clone(),
                    initializer: mod_compiler.code,
                };
                let stack = self.new_temporary();
                self.compiler
                    .code
                    .load_module(instance, OpDestination::Stack(stack));
                if self.is_module_root() || module.publish.is_some() {
                    self.ensure_in_module(module.name.1);

                    let access = if module.publish.is_some() {
                        Access::Public
                    } else {
                        Access::Private
                    };
                    self.compiler
                        .code
                        .declare(name.clone(), false, access, stack, dest);
                } else {
                    self.declare_local(name.clone(), false, stack, dest);
                }

                self.compiler.errors.append(&mut mod_compiler.errors);
            }
            Expression::Group(e) => {
                self.compile_expression(&e.enclosed, dest);
            }
            Expression::FormatString(parts) => self.compile_format_string(parts, dest),
            Expression::Macro(_) | Expression::InfixMacro(_) => {
                unreachable!("macros should be expanded already")
            }
        }
    }

    fn compile_format_string(
        &mut self,
        parts: &Delimited<Ranged<Symbol>, Ranged<Expression>>,
        dest: OpDestination,
    ) {
        let mut initial =
            ValueOrSource::Symbol(parts.first.clone().expect("missing first format part").0);
        if parts.remaining.len() < 255 {
            self.compile_format_string_chunk(initial, &parts.remaining, dest);
        } else {
            let joined = self.new_temporary();
            for chunk in parts.remaining.chunks(254) {
                self.compile_format_string_chunk(initial, chunk, OpDestination::Stack(joined));
                initial = ValueOrSource::Stack(joined);
            }
            self.compiler.code.copy(joined, dest);
        }
    }

    fn compile_format_string_chunk(
        &mut self,
        initial: ValueOrSource,
        parts: &[(Ranged<Expression>, Ranged<Symbol>)],
        dest: OpDestination,
    ) {
        let mut sources = Vec::with_capacity(256);
        sources.push(initial);
        for (joiner, part) in parts {
            sources.push(self.compile_source(joiner));
            sources.push(ValueOrSource::Symbol(part.0.clone()));
        }
        let arity = u8::try_from(sources.len()).expect("chunk size matches u8");
        for (index, source) in sources.into_iter().enumerate() {
            self.compiler.code.copy(
                source,
                OpDestination::Register(Register(index.try_into().expect("chunk size matches u8"))),
            );
        }
        self.compiler
            .code
            .call(Symbol::from("$.core.String"), arity);
        self.compiler.code.copy(Register(0), dest);
    }

    fn compile_literal(&mut self, literal: &Literal, dest: OpDestination) {
        match literal {
            Literal::Nil => self.compiler.code.copy((), dest),
            Literal::Bool(bool) => self.compiler.code.copy(*bool, dest),
            Literal::Int(int) => self.compiler.code.copy(*int, dest),
            Literal::UInt(int) => self.compiler.code.copy(*int, dest),
            Literal::Float(float) => self.compiler.code.copy(*float, dest),
            Literal::String(string) => {
                self.compiler.code.copy(string.clone(), Register(0));
                self.compiler.code.call(Symbol::from("$.core.String"), 1);
                self.compiler.code.copy(Register(0), dest);
            }
            Literal::Symbol(symbol) => self.compiler.code.copy(symbol.clone(), dest),
            Literal::Regex(regex) => self.compiler.code.copy(regex.clone(), dest),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn compile_function(
        &mut self,
        decl: &FunctionDefinition,
        range: SourceRange,
        dest: OpDestination,
    ) {
        let mut bodies_by_arity = kempt::Map::<(u8, bool), Vec<&MatchPattern>>::new();
        for def in &decl.body.patterns {
            let arity = if let Some(arity) = def.arity() {
                arity
            } else {
                self.compiler
                    .errors
                    .push(Ranged::new(range, Error::TooManyArguments));
                (255, false)
            };
            bodies_by_arity.entry(arity).or_default().push(def);
        }

        let mut fun = BitcodeFunction::new(decl.name.as_ref().map(|name| name.0.clone()));
        for ((arity, var_arg), bodies) in bodies_by_arity
            .into_iter()
            .map(kempt::map::Field::into_parts)
        {
            let mut fn_compiler = Compiler {
                function_name: decl.name.as_ref().map(|name| name.0.clone()),
                ..Compiler::default()
            };
            let mut fn_scope = Scope::function_root(&mut fn_compiler);

            let mut refutability = Refutability::Irrefutable;
            let previous_handler = self.new_temporary();
            let mut stored_previous_handler = false;
            for body in bodies {
                let next_body = fn_scope.compiler.code.new_label();
                let mut body_block = fn_scope.enter_block(None);

                let previous_handler = if stored_previous_handler {
                    OpDestination::Void
                } else {
                    stored_previous_handler = true;
                    OpDestination::Stack(previous_handler)
                };
                body_block
                    .compiler
                    .code
                    .set_exception_handler(next_body, previous_handler);

                let parameters = match &body.pattern.kind.0 {
                    PatternKind::Any(None) | PatternKind::AnyRemaining => None,
                    PatternKind::Any(_)
                    | PatternKind::Literal(_)
                    | PatternKind::Or(_, _, _)
                    | PatternKind::DestructureMap(_) => Some(PatternKinds::Slice(
                        std::slice::from_ref(&body.pattern.kind),
                    )),
                    PatternKind::DestructureTuple(patterns) => {
                        Some(PatternKinds::Delimited(&patterns.enclosed))
                    }
                };
                if let Some(parameters) = parameters {
                    refutability |= body_block.compile_function_parameters_pattern(
                        arity,
                        &parameters,
                        next_body,
                    );
                }

                if let Some(guard) = &body.pattern.guard {
                    let guard = body_block.compile_source(guard);
                    body_block.compiler.code.set_current_source_range(range);
                    body_block.compiler.code.jump_if_not(next_body, guard, ());
                }
                body_block
                    .compiler
                    .code
                    .set_exception_handler(previous_handler, ());

                body_block.compile_expression(&body.body, OpDestination::Register(Register(0)));
                body_block.compiler.code.set_current_source_range(range);
                body_block.compiler.code.return_early();
                drop(body_block);
                fn_scope.compiler.code.label(next_body);
            }

            if stored_previous_handler {
                self.compiler
                    .code
                    .set_exception_handler(previous_handler, ());
            }

            if refutability == Refutability::Refutable {
                self.compiler.code.copy((), Register(0));
            }

            drop(fn_scope);

            self.compiler.errors.append(&mut fn_compiler.errors);

            if var_arg {
                fun.insert_variable_arity(arity, fn_compiler.code);
            } else {
                fun.insert_arity(arity, fn_compiler.code);
            }
        }

        match (&decl.name, decl.publish.is_some(), self.is_module_root()) {
            (Some(name), true, _) | (Some(name), _, true) => {
                let access = if decl.publish.is_some() {
                    Access::Public
                } else {
                    Access::Private
                };
                self.compiler
                    .code
                    .declare(name.0.clone(), false, access, fun, dest);
            }
            (Some(name), false, _) => {
                let stack = self.new_temporary();
                self.compiler.code.copy(fun, stack);
                self.declare_local(name.0.clone(), false, stack, dest);
            }
            (None, true, _) => {
                self.compiler
                    .errors
                    .push(Ranged::new(range, Error::PublicFunctionRequiresName));
            }
            (None, false, _) => {
                self.compiler.code.copy(fun, dest);
            }
        }
    }

    #[must_use]
    fn compile_function_parameters_pattern(
        &mut self,
        arity: u8,
        parameters: &PatternKinds<'_>,
        doesnt_match: Label,
    ) -> Refutability {
        let mut refutable = Refutability::Irrefutable;

        for (parameter, register) in parameters.iter().zip(0..arity) {
            refutable |= self.compile_pattern_binding(
                parameter,
                ValueOrSource::Register(Register(register)),
                doesnt_match,
                &mut PatternBindings::default(),
            );
        }

        refutable
    }

    fn compile_literal_binding(
        &mut self,
        literal: &Literal,
        range: SourceRange,
        source: ValueOrSource,
        doesnt_match: Label,
        bindings: &mut PatternBindings,
    ) -> Refutability {
        let matches = self.new_temporary();

        if let Literal::Regex(regex_literal) = literal {
            match Regex::new(&regex_literal.pattern) {
                Ok(regex) if regex.capture_names().any(|s| s.is_some()) => {
                    self.compiler.code.copy(source, Register(0));

                    self.compiler.code.invoke(
                        ValueOrSource::Regex(regex_literal.clone()),
                        Symbol::captures_symbol(),
                        1,
                    );
                    self.compiler
                        .code
                        .jump_if_not(doesnt_match, Register(0), ());
                    self.compiler.code.copy(Register(0), matches);

                    for capture_name in regex.capture_names().flatten() {
                        let name = Symbol::from(capture_name);
                        self.check_bound_name(Ranged::new(range, name.clone()), bindings);

                        let variable = self.new_temporary();
                        self.compiler.code.copy(name.clone(), Register(0));
                        self.compiler.code.invoke(matches, Symbol::get_symbol(), 1);
                        self.compiler.code.copy(Register(0), variable);

                        if self.is_module_root() || bindings.publish {
                            self.ensure_in_module(range);
                            let access = if bindings.publish {
                                Access::Public
                            } else {
                                Access::Private
                            };

                            self.compiler.code.declare(
                                name.clone(),
                                bindings.mutable,
                                access,
                                variable,
                                (),
                            );
                        } else {
                            self.declare_local(
                                name.clone(),
                                bindings.mutable,
                                variable,
                                OpDestination::Void,
                            );
                        }
                    }

                    return Refutability::Refutable;
                }
                _ => {}
            }
        }

        match ValueOrSource::try_from(literal.clone()) {
            Ok(literal) => self.compiler.code.matches(literal.clone(), source, matches),
            Err(string) => self.compiler.code.matches(string.clone(), source, matches),
        }
        self.compiler.code.jump_if_not(doesnt_match, matches, ());

        Refutability::Refutable
    }

    fn check_bound_name(&mut self, name: Ranged<Symbol>, bindings: &mut PatternBindings) {
        if !bindings.bound_names.insert(name.0) {
            self.compiler
                .errors
                .push(Ranged::new(name.1, Error::NameAlreadyBound));
        }
    }

    #[allow(clippy::too_many_lines)]
    fn compile_pattern_binding(
        &mut self,
        pattern: &Ranged<PatternKind>,
        source: ValueOrSource,
        doesnt_match: Label,
        bindings: &mut PatternBindings,
    ) -> Refutability {
        self.compiler.code.set_current_source_range(pattern.range());
        match &pattern.0 {
            PatternKind::Any(name) => {
                if let Some(name) = name {
                    self.check_bound_name(Ranged::new(pattern.range(), name.clone()), bindings);

                    if self.is_module_root() || bindings.publish {
                        let access = if bindings.publish {
                            Access::Public
                        } else {
                            Access::Private
                        };
                        self.compiler.code.declare(
                            name.clone(),
                            bindings.mutable,
                            access,
                            source,
                            (),
                        );
                    } else {
                        let stack = self.new_temporary();
                        self.compiler.code.copy(source, stack);
                        self.declare_local(
                            name.clone(),
                            bindings.mutable,
                            stack,
                            OpDestination::Void,
                        );
                    }
                }
                Refutability::Irrefutable
            }
            PatternKind::AnyRemaining => Refutability::Irrefutable,
            PatternKind::Literal(literal) => self.compile_literal_binding(
                literal,
                pattern.range(),
                source,
                doesnt_match,
                bindings,
            ),
            PatternKind::Or(a, _, b) => {
                let b_label = self.compiler.code.new_label();
                let body_label = self.compiler.code.new_label();

                let mut b_bindings = PatternBindings::default();

                let mut refutable =
                    self.compile_pattern_binding(a, source.clone(), b_label, bindings);
                self.compiler.code.set_current_source_range(pattern.range());
                self.compiler.code.jump(body_label, ());

                self.compiler.code.label(b_label);
                refutable |=
                    self.compile_pattern_binding(b, source.clone(), doesnt_match, &mut b_bindings);

                if bindings.bound_names != b_bindings.bound_names {
                    self.compiler.errors.push(Ranged::new(
                        pattern.range(),
                        Error::OrPatternBindingsMismatch,
                    ));
                }

                self.compiler.code.label(body_label);
                refutable
            }
            PatternKind::DestructureTuple(patterns) => {
                self.compile_tuple_destructure(
                    &patterns.enclosed,
                    &source,
                    doesnt_match,
                    pattern.range(),
                    bindings,
                );
                Refutability::Refutable
            }
            PatternKind::DestructureMap(entries) => {
                let expected_count = u64::try_from(entries.enclosed.len()).unwrap_or_else(|_| {
                    self.compiler
                        .errors
                        .push(Ranged::new(pattern.range(), Error::UsizeTooLarge));
                    u64::MAX
                });

                self.compiler
                    .code
                    .invoke(source.clone(), Symbol::len_symbol(), 0);
                self.compiler.code.compare(
                    CompareKind::Equal,
                    Register(0),
                    expected_count,
                    Register(0),
                );
                self.compiler
                    .code
                    .jump_if_not(doesnt_match, Register(0), ());

                let element = self.new_temporary();
                for entry in &entries.enclosed {
                    self.compiler.code.set_current_source_range(pattern.range());
                    let key = match &entry.key.0 {
                        EntryKeyPattern::Nil => ValueOrSource::Nil,
                        EntryKeyPattern::Bool(value) => ValueOrSource::Bool(*value),
                        EntryKeyPattern::Int(value) => ValueOrSource::Int(*value),
                        EntryKeyPattern::UInt(value) => ValueOrSource::UInt(*value),
                        EntryKeyPattern::Float(value) => ValueOrSource::Float(*value),
                        EntryKeyPattern::String(value) => {
                            self.compiler.code.copy(value.clone(), Register(0));
                            self.compiler.code.call(Symbol::from("$.core.String"), 1);
                            ValueOrSource::Register(Register(0))
                        }
                        EntryKeyPattern::Identifier(value) => ValueOrSource::Symbol(value.clone()),
                    };
                    self.compiler.code.copy(key, Register(0));
                    self.compiler
                        .code
                        .invoke(source.clone(), Symbol::get_symbol(), 1);
                    self.compiler.code.copy(Register(0), element);
                    self.compile_pattern_binding(
                        &entry.value,
                        ValueOrSource::Stack(element),
                        doesnt_match,
                        bindings,
                    );
                }

                Refutability::Refutable
            }
        }
    }

    fn compile_tuple_destructure(
        &mut self,
        patterns: &Delimited<Ranged<PatternKind>>,
        source: &ValueOrSource,
        doesnt_match: Label,
        pattern_range: SourceRange,
        bindings: &mut PatternBindings,
    ) {
        let expected_count = u64::try_from(patterns.len()).unwrap_or_else(|_| {
            self.compiler
                .errors
                .push(Ranged::new(pattern_range, Error::UsizeTooLarge));
            u64::MAX
        });

        self.compiler
            .code
            .invoke(source.clone(), Symbol::len_symbol(), 0);
        self.compiler
            .code
            .compare(CompareKind::Equal, Register(0), expected_count, Register(0));
        self.compiler
            .code
            .jump_if_not(doesnt_match, Register(0), ());

        let element = self.new_temporary();
        for (i, index) in (0..expected_count).zip(0_usize..) {
            if i == expected_count - 1 && matches!(&patterns[index].0, PatternKind::AnyRemaining) {
                break;
            }
            self.compiler
                .code
                .set_current_source_range(patterns[index].range());
            self.compiler.code.copy(i, Register(0));
            self.compiler
                .code
                .invoke(source.clone(), Symbol::nth_symbol(), 1);
            self.compiler.code.copy(Register(0), element);
            self.compile_pattern_binding(
                &patterns[index],
                ValueOrSource::Stack(element),
                doesnt_match,
                bindings,
            );
        }
    }

    fn compile_match_expression(&mut self, match_expr: &MatchExpression, dest: OpDestination) {
        let conditions = if let Expression::List(list) = &match_expr.condition.0 {
            MatchExpressions::List(
                list.values
                    .enclosed
                    .iter()
                    .map(|expr| self.compile_source(expr))
                    .collect(),
            )
        } else {
            MatchExpressions::Single(self.compile_source(&match_expr.condition))
        };
        let after_expression = self.compiler.code.new_label();
        self.compile_match(&conditions, &match_expr.matches, dest);
        self.compiler.code.label(after_expression);
    }

    #[allow(clippy::too_many_lines)]
    fn compile_match(
        &mut self,
        conditions: &MatchExpressions,
        matches: &Ranged<Matches>,
        dest: OpDestination,
    ) {
        self.compiler.code.set_current_source_range(matches.range());
        let mut refutable = Refutability::Irrefutable;
        let previous_handler = self.new_temporary();
        let mut stored_previous_handler = false;
        let after_expression = self.compiler.code.new_label();

        for matches in &matches.patterns {
            let mut pattern_block = self.enter_block(None);
            let next_pattern = pattern_block.compiler.code.new_label();

            let previous_handler = if stored_previous_handler {
                OpDestination::Void
            } else {
                stored_previous_handler = true;
                OpDestination::Stack(previous_handler)
            };
            pattern_block
                .compiler
                .code
                .set_exception_handler(next_pattern, previous_handler);

            match &conditions {
                MatchExpressions::Single(condition) => {
                    match &matches.pattern.kind.0 {
                        PatternKind::Any(None) | PatternKind::AnyRemaining => {}
                        PatternKind::Any(_)
                        | PatternKind::Literal(_)
                        | PatternKind::Or(_, _, _)
                        | PatternKind::DestructureMap(_) => {
                            refutable |= pattern_block.compile_pattern_binding(
                                &matches.pattern.kind,
                                condition.clone(),
                                next_pattern,
                                &mut PatternBindings::default(),
                            );
                        }
                        PatternKind::DestructureTuple(patterns) => {
                            pattern_block.compile_tuple_destructure(
                                &patterns.enclosed,
                                condition,
                                next_pattern,
                                matches.range(),
                                &mut PatternBindings::default(),
                            );
                            refutable = Refutability::Refutable;
                        }
                    }

                    pattern_block
                        .compiler
                        .code
                        .set_current_source_range(matches.range());
                }
                MatchExpressions::List(conditions) => {
                    let parameters = match &matches.pattern.kind.0 {
                        PatternKind::Any(None) | PatternKind::AnyRemaining => None,
                        PatternKind::Any(_)
                        | PatternKind::Literal(_)
                        | PatternKind::Or(_, _, _)
                        | PatternKind::DestructureMap(_) => Some(PatternKinds::Slice(
                            std::slice::from_ref(&matches.pattern.kind),
                        )),
                        PatternKind::DestructureTuple(patterns) => {
                            Some(PatternKinds::Delimited(&patterns.enclosed))
                        }
                    };

                    if let Some(parameters) = parameters {
                        if parameters.len() == conditions.len()
                            || parameters.last().map_or(false, |param| {
                                matches!(&param.0, PatternKind::AnyRemaining)
                            })
                        {
                            for (parameter, condition) in parameters.iter().zip(conditions) {
                                refutable |= pattern_block.compile_pattern_binding(
                                    parameter,
                                    condition.clone(),
                                    next_pattern,
                                    &mut PatternBindings::default(),
                                );
                            }
                            pattern_block
                                .compiler
                                .code
                                .set_current_source_range(matches.range());
                        } else {
                            refutable = Refutability::Refutable;
                            pattern_block.compiler.code.jump(next_pattern, ());
                        }
                    }
                }
            }

            if let Some(guard) = &matches.pattern.guard {
                let guard = pattern_block.compile_source(guard);
                pattern_block
                    .compiler
                    .code
                    .set_current_source_range(matches.range());
                pattern_block
                    .compiler
                    .code
                    .jump_if_not(next_pattern, guard, ());
            }

            pattern_block
                .compiler
                .code
                .set_exception_handler(previous_handler, ());

            pattern_block.compile_expression(&matches.body, dest);

            pattern_block
                .compiler
                .code
                .set_current_source_range(matches.range());
            pattern_block.compiler.code.jump(after_expression, ());
            drop(pattern_block);

            self.compiler.code.label(next_pattern);
        }

        if stored_previous_handler {
            self.compiler
                .code
                .set_exception_handler(previous_handler, ());
        }

        if refutable == Refutability::Refutable {
            self.compiler.code.throw(FaultKind::PatternMismatch);
        }
        self.compiler.code.label(after_expression);
    }

    fn compile_throw(&mut self, value: &Ranged<Expression>, range: SourceRange) {
        let value = self.compile_source(value);

        self.compiler.code.set_current_source_range(range);
        self.compiler.code.copy(value, Register(0));
        self.compiler.code.throw(FaultKind::Exception);
    }

    fn compile_try(&mut self, try_expr: &TryExpression, range: SourceRange, dest: OpDestination) {
        self.compile_try_catch(
            range,
            |this| this.compile_expression(&try_expr.body, dest),
            |this| {
                if let Some(catch) = &try_expr.catch {
                    this.compile_match(
                        &MatchExpressions::Single(ValueOrSource::Register(Register(0))),
                        &catch.matches,
                        dest,
                    );
                } else {
                    // A catch-less try converts to nil. It's just a different
                    // form of the ? operator.
                    this.compiler.code.copy((), dest);
                }
            },
        );
    }

    fn compile_try_catch(
        &mut self,
        range: SourceRange,
        tried: impl FnOnce(&mut Self),
        catch: impl FnOnce(&mut Self),
    ) {
        let exception_error = self.compiler.code.new_label();
        let after_handler = self.compiler.code.new_label();
        let previous_handler = self.new_temporary();

        self.compiler
            .code
            .set_exception_handler(exception_error, previous_handler);
        tried(self);
        self.compiler.code.set_current_source_range(range);
        self.compiler
            .code
            .set_exception_handler(previous_handler, ());
        self.compiler.code.jump(after_handler, ());

        self.compiler.code.label(exception_error);
        self.compiler
            .code
            .set_exception_handler(previous_handler, ());
        catch(self);
        self.compiler.code.set_current_source_range(range);
        self.compiler.code.label(after_handler);
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
            self.compiler.code.set_current_source_range(from);
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
            self.compiler.code.set_current_source_range(from);
            self.compiler.code.jump(label, ());
        } else {
            self.compiler
                .errors
                .push(Ranged::new(from, Error::InvalidLabel));
        }
    }

    #[allow(clippy::too_many_lines)]
    fn compile_loop(&mut self, expr: &LoopExpression, range: SourceRange, dest: OpDestination) {
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
                outer_scope.compiler.code.set_current_source_range(range);
                outer_scope
                    .compiler
                    .code
                    .jump_if_not(break_label, condition, ());
                (ValueOrSource::Label(continue_label), None)
            }
            LoopKind::TailWhile { .. } => {
                let tail_condition = outer_scope.compiler.code.new_label();
                outer_scope.compiler.code.label(tail_condition);
                (ValueOrSource::Label(tail_condition), None)
            }
            LoopKind::For {
                pattern, source, ..
            } => {
                let source = outer_scope.compile_expression_into_temporary(source);
                outer_scope.compiler.code.set_current_source_range(range);

                // We try to call :iterate on source. If this throws an
                // exception, we switch to trying to iterate by calling `nth()`
                // and counting.
                let iter_using_nth = outer_scope.compiler.code.new_label();
                let loop_body = outer_scope.compiler.code.new_label();
                let next_iteration = outer_scope.new_temporary();
                let previous_handler = outer_scope.new_temporary();
                let i = outer_scope.new_temporary();
                let step = outer_scope.new_temporary();
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

                // Store the iterator in `step`.
                outer_scope.compiler.code.copy(Register(0), step);
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
                // Invoke next(), store the result in the `i`.
                outer_scope
                    .compiler
                    .code
                    .invoke(step, Symbol::next_symbol(), 0);
                outer_scope.compiler.code.copy(Register(0), i);

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
                outer_scope.compiler.code.copy(0, step);
                let after_first_increment = outer_scope.compiler.code.new_label();
                outer_scope.compiler.code.jump(after_first_increment, ());

                // Each iteration of the nth-based iteration will return here.
                // The first step is to increment i.
                outer_scope.compiler.code.label(nth_next_iteration);
                outer_scope.compiler.code.add(step, 1, step);
                outer_scope.compiler.code.label(after_first_increment);

                // Next, invoke nth(i)
                outer_scope.compiler.code.copy(step, Register(0));
                outer_scope
                    .compiler
                    .code
                    .invoke(source, Symbol::nth_symbol(), 1);
                // Copy the result into `i` -- just like the iterator() code.
                outer_scope.compiler.code.copy(Register(0), i);

                // Finally, the shared loop body.
                outer_scope.compiler.code.label(loop_body);

                // Try to apply the pattern. If we have an error applying the
                // pattern, continue to the next iteration.
                let loop_body_with_bindings = outer_scope.compiler.code.new_label();
                let pattern_mismatch = outer_scope.compiler.code.new_label();
                outer_scope
                    .compiler
                    .code
                    .set_exception_handler(pattern_mismatch, ());

                outer_scope.compile_pattern_binding(
                    &pattern.kind,
                    ValueOrSource::Stack(i),
                    pattern_mismatch,
                    &mut PatternBindings::default(),
                );
                outer_scope.compiler.code.set_current_source_range(range);

                if let Some(guard) = &pattern.guard {
                    let guard = outer_scope.compile_source(guard);
                    outer_scope.compiler.code.set_current_source_range(range);
                    outer_scope
                        .compiler
                        .code
                        .jump_if_not(pattern_mismatch, guard, ());
                }

                outer_scope
                    .compiler
                    .code
                    .set_exception_handler(exception_handler, ());
                outer_scope.compiler.code.jump(loop_body_with_bindings, ());

                // When the pattern mismatches, we need to restore the correct
                // exception handler before jumping back to the next iteration.
                outer_scope.compiler.code.label(pattern_mismatch);
                outer_scope
                    .compiler
                    .code
                    .set_exception_handler(exception_handler, ());
                outer_scope.compiler.code.jump(next_iteration, ());

                // And truly finally, the actual loop body.
                outer_scope.compiler.code.label(loop_body_with_bindings);

                (ValueOrSource::Stack(next_iteration), Some(previous_handler))
            }
        };

        let mut loop_scope = outer_scope.enter_loop(
            expr.block.label.as_ref().map(|label| label.name.0.clone()),
            continue_label,
            break_label,
            dest,
        );

        loop_scope.compile_expression(&expr.block.body.enclosed, dest);
        drop(loop_scope);

        outer_scope.compiler.code.set_current_source_range(range);

        match &expr.kind {
            LoopKind::Infinite | LoopKind::While(_) | LoopKind::For { .. } => {}
            LoopKind::TailWhile { expression, .. } => {
                outer_scope.compiler.code.label(continue_label);
                let condition = outer_scope.compile_source(expression);
                outer_scope.compiler.code.set_current_source_range(range);
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

    fn compile_single_match(
        &mut self,
        decl: &SingleMatch,
        range: SourceRange,
        dest: OpDestination,
    ) {
        let value = self.compile_source(&decl.value);
        self.compiler.code.set_current_source_range(range);

        let pattern_mismatch = self.compiler.code.new_label();
        let after_declare = self.compiler.code.new_label();
        let previous_handler = self.new_temporary();
        self.compiler
            .code
            .set_exception_handler(pattern_mismatch, previous_handler);

        let mut refutable = self.compile_pattern_binding(
            &decl.pattern.kind,
            value,
            pattern_mismatch,
            &mut PatternBindings {
                publish: decl.publish.is_some(),
                mutable: &decl.kind.0 == Symbol::var_symbol(),
                bound_names: Set::new(),
            },
        );

        self.compiler.code.set_current_source_range(range);

        if let Some(guard) = &decl.pattern.guard {
            refutable = Refutability::Refutable;
            let guard = self.compile_source(guard);
            self.compiler.code.set_current_source_range(range);
            self.compiler.code.jump_if_not(pattern_mismatch, guard, ());
        }

        self.compiler
            .code
            .set_exception_handler(previous_handler, ());
        self.compiler.code.jump(after_declare, ());

        self.compiler.code.label(pattern_mismatch);
        self.compiler
            .code
            .set_exception_handler(previous_handler, ());

        if let Some(r#else) = &decl.r#else {
            self.compile_expression(&r#else.expression, OpDestination::Void);
            self.compiler.code.set_current_source_range(range);
            match refutable {
                Refutability::Refutable => {
                    if let Err(range) = Self::check_all_branches_diverge(&r#else.expression) {
                        self.compiler
                            .errors
                            .push(Ranged::new(range, Error::LetElseMustDiverge));
                    }
                }
                Refutability::Irrefutable => self.compiler.errors.push(Ranged::new(
                    r#else.expression.1,
                    Error::ElseOnIrrefutablePattern,
                )),
            }
        } else {
            self.compiler.code.throw(FaultKind::PatternMismatch);
        }

        self.compiler.code.label(after_declare);
        self.compiler.code.copy(true, dest);
    }

    fn check_all_branches_diverge(expr: &Ranged<Expression>) -> Result<(), SourceRange> {
        match &expr.0 {
            Expression::Break(_)
            | Expression::Return(_)
            | Expression::Continue(_)
            | Expression::Throw(_) => Ok(()),
            Expression::If(if_expr) => {
                let Some(when_false) = &if_expr.when_false else {
                    return Err(expr.1);
                };
                Self::check_all_branches_diverge(&if_expr.when_true)?;

                Self::check_all_branches_diverge(&when_false.expression)
            }
            Expression::Block(block) => Self::check_all_branches_diverge(&block.body.enclosed),
            Expression::Binary(binary) if binary.kind == syntax::BinaryKind::Chain => {
                Self::check_all_branches_diverge(&binary.left)
                    .or_else(|_| Self::check_all_branches_diverge(&binary.right))
            }
            Expression::Try(try_expr) => {
                Self::check_all_branches_diverge(&try_expr.body)?;
                if let Some(catch) = &try_expr.catch {
                    Self::check_matches_diverge(&catch.matches)
                } else {
                    Err(expr.range())
                }
            }
            Expression::Loop(loop_expr) => {
                let conditional = match &loop_expr.kind {
                    LoopKind::Infinite => false,
                    LoopKind::While(_) | LoopKind::TailWhile { .. } | LoopKind::For { .. } => true,
                };

                if !conditional {
                    return Err(expr.1);
                }

                Self::check_all_branches_diverge(&loop_expr.block.body.enclosed)
            }

            Expression::Match(match_expr) => Self::check_matches_diverge(&match_expr.matches),
            Expression::Group(e) => Self::check_all_branches_diverge(&e.enclosed),
            Expression::Literal(_)
            | Expression::Lookup(_)
            | Expression::TryOrNil(_)
            | Expression::Map(_)
            | Expression::List(_)
            | Expression::Call(_)
            | Expression::Index(_)
            | Expression::Assign(_)
            | Expression::Unary(_)
            | Expression::Binary(_)
            | Expression::Module(_)
            | Expression::Function(_)
            | Expression::SingleMatch(_)
            | Expression::RootModule
            | Expression::FormatString(_) => Err(expr.1),
            Expression::Macro(_) | Expression::InfixMacro(_) => {
                unreachable!("macros should be expanded already")
            }
        }
    }

    fn check_matches_diverge(matches: &Ranged<Matches>) -> Result<(), SourceRange> {
        for pattern in &matches.patterns {
            if matches!(&pattern.0.pattern.kind, Ranged(PatternKind::Any(None), _))
                && pattern.0.pattern.guard.is_none()
            {
                return Ok(());
            }
        }

        Err(matches.range())
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

    fn compile_unary(&mut self, unary: &UnaryExpression, range: SourceRange, dest: OpDestination) {
        let op = self.compile_source(&unary.operand);
        let kind = match unary.kind {
            syntax::UnaryKind::LogicalNot => UnaryKind::LogicalNot,
            syntax::UnaryKind::BitwiseNot => UnaryKind::BitwiseNot,
            syntax::UnaryKind::Negate => UnaryKind::Negate,
        };
        self.compiler.code.set_current_source_range(range);
        self.compiler.code.push(Op::Unary { op, dest, kind });
    }

    fn try_compile_chain_compare(
        &mut self,
        binop: &BinaryExpression,
        range: SourceRange,
        kind: CompareKind,
        dest: OpDestination,
    ) -> bool {
        if let Expression::Binary(left_binop) = &binop.left.0 {
            let mut left_binop: &BinaryExpression = left_binop;
            if let syntax::BinaryKind::Compare(left_kind) = left_binop.kind {
                let after_compare = self.compiler.code.new_label();
                let mut stack = Vec::new();
                stack.push((kind, &binop.right));
                stack.push((left_kind, &left_binop.right));
                loop {
                    if let Expression::Binary(next_binop) = &left_binop.left.0 {
                        if let syntax::BinaryKind::Compare(left_kind) = next_binop.kind {
                            stack.push((left_kind, &next_binop.right));
                            left_binop = next_binop;

                            continue;
                        }
                    }
                    break;
                }

                let mut left = self.compile_source(&left_binop.left);
                while let Some((kind, right)) = stack.pop() {
                    let right = self.compile_source(right);
                    self.compiler.code.set_current_source_range(range);

                    if stack.is_empty() {
                        self.compiler.code.push(Op::BinOp {
                            op1: left,
                            op2: right,
                            dest,
                            kind: BinaryKind::Compare(kind),
                        });

                        break;
                    }

                    let continue_compare = self.compiler.code.new_label();
                    self.compiler.code.push(Op::BinOp {
                        op1: left,
                        op2: right.clone(),
                        dest: OpDestination::Label(continue_compare),
                        kind: BinaryKind::Compare(kind),
                    });
                    self.compiler.code.copy(false, dest);
                    self.compiler.code.jump(after_compare, ());

                    self.compiler.code.label(continue_compare);

                    left = right;
                }

                self.compiler.code.label(after_compare);
                return true;
            }
        }

        false
    }

    fn compile_binop(&mut self, binop: &BinaryExpression, range: SourceRange, dest: OpDestination) {
        let kind = match binop.kind {
            syntax::BinaryKind::Add => BinaryKind::Add,
            syntax::BinaryKind::Subtract => BinaryKind::Subtract,
            syntax::BinaryKind::Multiply => BinaryKind::Multiply,
            syntax::BinaryKind::Divide => BinaryKind::Divide,
            syntax::BinaryKind::IntegerDivide => BinaryKind::IntegerDivide,
            syntax::BinaryKind::Remainder => BinaryKind::Remainder,
            syntax::BinaryKind::Power => BinaryKind::Power,
            syntax::BinaryKind::Bitwise(kind) => BinaryKind::Bitwise(kind),
            syntax::BinaryKind::Compare(kind) => {
                if self.try_compile_chain_compare(binop, range, kind, dest) {
                    return;
                }

                BinaryKind::Compare(kind)
            }
            syntax::BinaryKind::Chain => {
                self.compile_expression(&binop.left, OpDestination::Void);
                self.compile_expression(&binop.right, dest);
                return;
            }
            syntax::BinaryKind::NilCoalesce => {
                let source = self.compile_source(&binop.left);
                self.compiler.code.set_current_source_range(range);
                let when_nil = self.compiler.code.new_label();
                let after_expression = self.compiler.code.new_label();

                // Jump when source is nil.
                self.compiler
                    .code
                    .compare(CompareKind::Equal, source.clone(), (), when_nil);

                // When not nil, copy source to dest, then skip the right hand
                // code.
                self.compiler.code.copy(source, dest);
                self.compiler.code.jump(after_expression, ());

                // When nil, evaluate the right hand side.
                self.compiler.code.label(when_nil);
                self.compile_expression(&binop.right, dest);

                self.compiler.code.label(after_expression);

                return;
            }
            syntax::BinaryKind::Logical(LogicalKind::And) => {
                let after = self.compiler.code.new_label();
                self.compile_expression(&binop.left, dest);
                self.compiler.code.set_current_source_range(range);
                self.compiler.code.jump_if_not(after, dest, ());

                self.compile_expression(&binop.right, dest);
                self.compiler.code.set_current_source_range(range);
                self.compiler.code.truthy(dest, dest);

                self.compiler.code.label(after);

                return;
            }
            syntax::BinaryKind::Logical(LogicalKind::Or) => {
                let after = self.compiler.code.new_label();
                self.compile_expression(&binop.left, dest);
                self.compiler.code.set_current_source_range(range);
                self.compiler.code.jump_if(after, dest, ());

                self.compile_expression(&binop.right, dest);
                self.compiler.code.set_current_source_range(range);
                self.compiler.code.truthy(dest, dest);

                self.compiler.code.label(after);

                return;
            }
            syntax::BinaryKind::Logical(LogicalKind::Xor) => BinaryKind::LogicalXor,
        };
        let left = self.compile_source(&binop.left);
        let right = self.compile_source(&binop.right);
        self.compiler.code.set_current_source_range(range);
        self.compiler.code.push(Op::BinOp {
            op1: left,
            op2: right,
            dest,
            kind,
        });
    }

    fn compile_source(&mut self, source: &Ranged<Expression>) -> ValueOrSource {
        match &source.0 {
            Expression::Literal(literal) => match literal {
                Literal::Nil => ValueOrSource::Nil,
                Literal::Bool(bool) => ValueOrSource::Bool(*bool),
                Literal::Int(int) => ValueOrSource::Int(*int),
                Literal::UInt(int) => ValueOrSource::UInt(*int),
                Literal::Float(float) => ValueOrSource::Float(*float),
                Literal::Regex(regex) => ValueOrSource::Regex(regex.clone()),
                Literal::String(_) => {
                    ValueOrSource::Stack(self.compile_expression_into_temporary(source))
                }
                Literal::Symbol(symbol) => ValueOrSource::Symbol(symbol.clone()),
            },
            Expression::Lookup(lookup) => {
                if let Some(decl) = self.compiler.declarations.get(&lookup.name.0) {
                    ValueOrSource::Stack(decl.stack)
                } else {
                    ValueOrSource::Stack(self.compile_expression_into_temporary(source))
                }
            }
            Expression::Group(e) => self.compile_source(&e.enclosed),
            Expression::Map(_)
            | Expression::List(_)
            | Expression::If(_)
            | Expression::Match(_)
            | Expression::Function(_)
            | Expression::Module(_)
            | Expression::Call(_)
            | Expression::Index(_)
            | Expression::SingleMatch(_)
            | Expression::Assign(_)
            | Expression::Unary(_)
            | Expression::Binary(_)
            | Expression::Try(_)
            | Expression::TryOrNil(_)
            | Expression::Throw(_)
            | Expression::Block(_)
            | Expression::Loop(_)
            | Expression::Break(_)
            | Expression::Continue(_)
            | Expression::RootModule
            | Expression::Return(_)
            | Expression::FormatString(_) => {
                ValueOrSource::Stack(self.compile_expression_into_temporary(source))
            }
            Expression::Macro(_) | Expression::InfixMacro(_) => {
                unreachable!("macros should be expanded already")
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
        let arity = if let Ok(arity) = u8::try_from(call.parameters.enclosed.len()) {
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
                    .map_or(false, |f| f == &lookup.name.0) =>
            {
                ValueOrSource::Nil
            }
            Expression::Lookup(lookup) if lookup.base.is_some() => {
                let base = lookup.base.as_ref().expect("just matched");
                let base = self.compile_source(&base.expression);

                self.compile_function_args(&call.parameters.enclosed, arity);
                self.compiler.code.set_current_source_range(range);
                self.compiler
                    .code
                    .invoke(base, lookup.name.0.clone(), arity);
                self.compiler.code.copy(Register(0), dest);
                return;
            }
            _ => self.compile_source(&call.function),
        };

        self.compile_function_args(&call.parameters.enclosed, arity);
        self.compiler.code.set_current_source_range(range);
        self.compiler.code.call(function, arity);
        self.compiler.code.copy(Register(0), dest);
    }

    fn compile_index(&mut self, index: &Index, range: SourceRange, dest: OpDestination) {
        let arity = if let Ok(arity) = u8::try_from(index.parameters.enclosed.len()) {
            arity
        } else {
            self.compiler
                .errors
                .push(Ranged::new(range, Error::TooManyArguments));
            255
        };

        let target = self.compile_source(&index.target);
        self.compile_function_args(&index.parameters.enclosed, arity);
        self.compiler.code.set_current_source_range(range);
        self.compiler
            .code
            .invoke(target, Symbol::get_symbol().clone(), arity);
        self.compiler.code.copy(Register(0), dest);
    }

    fn compile_function_args(&mut self, args: &Delimited<Ranged<Expression>>, arity: u8) {
        let mut parameters = Vec::with_capacity(args.len());
        for param in args.iter().take(usize::from(arity)) {
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

enum PatternKinds<'a> {
    Slice(&'a [Ranged<PatternKind>]),
    Delimited(&'a Delimited<Ranged<PatternKind>>),
}

impl PatternKinds<'_> {
    fn iter(&self) -> PatternKindsIter<'_> {
        match self {
            PatternKinds::Slice(contents) => PatternKindsIter::Slice(contents.iter()),
            PatternKinds::Delimited(contents) => PatternKindsIter::Delimited(contents.iter()),
        }
    }

    fn len(&self) -> usize {
        match self {
            PatternKinds::Slice(kinds) => kinds.len(),
            PatternKinds::Delimited(kinds) => kinds.len(),
        }
    }

    fn last(&self) -> Option<&'_ Ranged<PatternKind>> {
        match self {
            PatternKinds::Slice(kinds) => kinds.last(),
            PatternKinds::Delimited(kinds) => kinds.last(),
        }
    }
}

enum PatternKindsIter<'a> {
    Slice(slice::Iter<'a, Ranged<PatternKind>>),
    Delimited(DelimitedIter<'a, Ranged<PatternKind>>),
}

impl<'a> Iterator for PatternKindsIter<'a> {
    type Item = &'a Ranged<PatternKind>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            PatternKindsIter::Slice(iter) => iter.next(),
            PatternKindsIter::Delimited(iter) => iter.next(),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
enum Refutability {
    Refutable,
    Irrefutable,
}

impl BitOr for Refutability {
    type Output = Self;

    fn bitor(mut self, rhs: Self) -> Self::Output {
        self |= rhs;
        self
    }
}

impl BitOrAssign for Refutability {
    fn bitor_assign(&mut self, rhs: Self) {
        if *self != Refutability::Refutable {
            *self = rhs;
        }
    }
}

/// A compilation error.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Error {
    /// A variable cannot be assigned to if it is not mutable.
    VariableNotMutable,
    /// Too many arguments were provided to a function call. The maximum number
    /// of arguments to a function is 255.
    TooManyArguments,
    /// A `usize` was too large to convert into a smaller integer type. In
    /// general, this should not ever be returned given that other limits will
    /// likely be reached before this could be.
    UsizeTooLarge,
    /// A public function requires a name.
    PublicFunctionRequiresName,
    /// A label is not valid.
    InvalidLabel,
    /// Public declarations can only exist within a module's scope.
    PubOnlyInModules,
    /// Expected a block/
    ExpectedBlock,
    /// This name is already bound in the current pattern match.
    NameAlreadyBound,
    /// This sigil is not known to the compiler.
    UnknownSigil,
    /// All pattern options must contain the same name bindings.
    OrPatternBindingsMismatch,
    /// Else was provided on an irrefutable pattern.
    ElseOnIrrefutablePattern,
    /// The else expression in a `let..else` expression must diverge.
    LetElseMustDiverge,
    /// A syntax error occurred.
    Syntax(syntax::ParseError),
    /// A syntax error occurred while parsing an expanded macro.
    SigilSyntax(syntax::ParseError),
}

impl crate::ErrorKind for Error {
    fn kind(&self) -> &'static str {
        match self {
            Error::VariableNotMutable => "variable not mutable",
            Error::TooManyArguments => "too many arguments",
            Error::UsizeTooLarge => "usize too large",
            Error::PublicFunctionRequiresName => "public function must have a name",
            Error::InvalidLabel => "invalid label",
            Error::PubOnlyInModules => "pub only in modules",
            Error::ExpectedBlock => "expected block",
            Error::NameAlreadyBound => "name already bound",
            Error::OrPatternBindingsMismatch => "or pattern bindings mismatch",
            Error::ElseOnIrrefutablePattern => "else on irrefutable pattern",
            Error::LetElseMustDiverge => "let else must diverge",
            Error::UnknownSigil => "unknown sigil",
            Error::Syntax(err) | Error::SigilSyntax(err) => err.kind(),
        }
    }
}

impl std::error::Error for Error {}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::VariableNotMutable => f.write_str("declaration cannot be reassigned"),
            Error::TooManyArguments => f.write_str("no more than 256 arguments may be passed"),
            Error::UsizeTooLarge => {
                f.write_str("integer too large for the architecture's index type")
            }
            Error::PublicFunctionRequiresName => f.write_str("public function must have a name"),
            Error::InvalidLabel => f.write_str("invalid label"),
            Error::PubOnlyInModules => {
                f.write_str("declarations may only be published within modules")
            }
            Error::ExpectedBlock => f.write_str("expected a block"),
            Error::NameAlreadyBound => {
                f.write_str("this name is already bound within the same pattern match")
            }
            Error::OrPatternBindingsMismatch => {
                f.write_str("all options must contain the same named bindings")
            }
            Error::ElseOnIrrefutablePattern => {
                f.write_str("this else block is unreachable, because the pattern is irrefutable")
            }
            Error::LetElseMustDiverge => f.write_str("all code paths must diverge"),
            Error::UnknownSigil => f.write_str("unknown sigil"),
            Error::Syntax(err) => Display::fmt(err, f),
            Error::SigilSyntax(err) => write!(f, "syntax error in sigil expansion: {err}"),
        }
    }
}

impl From<Ranged<syntax::ParseError>> for Ranged<Error> {
    fn from(err: Ranged<syntax::ParseError>) -> Self {
        err.map(Error::Syntax)
    }
}

/// An operation that is performed on a single argument.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum UnaryKind {
    /// Evaluates the argument's truthyness
    Truthy,
    /// `not op`
    LogicalNot,
    /// `!op`
    BitwiseNot,
    /// `-op`
    Negate,
    /// Copies the argument.
    Copy,
    /// Resolves a name in the current scope.
    Resolve,
    /// Jumps to the instruction address. The destination will contain the
    /// current address before jumping.
    Jump,
    /// Sets the exception handler to the label provided. The destination will
    /// contain the overwritten exception handler.
    SetExceptionHandler,
}

/// An IR [`Module`](crate::vm::Module).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BitcodeModule {
    /// The name of the module.
    pub name: Symbol,
    /// The initializer that loads the module.
    pub initializer: BitcodeBlock,
}

#[derive(Default)]
struct PatternBindings {
    publish: bool,
    mutable: bool,
    bound_names: Set<Symbol>,
}

/// A mapping of [`SourceRange`]s to instruction addresses.
#[derive(Default, Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct SourceMap(Arc<SourceMapData>);

#[derive(Default, Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
struct SourceMapData {
    instructions: Vec<InstructionRange>,
}

impl SourceMap {
    /// Record a new instruction with `range` as its source.
    pub(crate) fn push(&mut self, range: SourceRange) {
        let map = Arc::make_mut(&mut self.0);
        if let Some(inst) = map
            .instructions
            .last_mut()
            .filter(|inst| inst.range == range)
        {
            inst.instructions.end += 1;
        } else {
            let instruction = map
                .instructions
                .last()
                .map_or(0, |inst| inst.instructions.end);

            #[allow(clippy::range_plus_one)]
            map.instructions.push(InstructionRange {
                range,
                instructions: instruction..instruction + 1,
            });
        }
    }

    /// Returns the range of a given instruction, if found.
    #[must_use]
    pub fn get(&self, instruction: usize) -> Option<SourceRange> {
        self.0.instructions.iter().find_map(|probe| {
            probe
                .instructions
                .contains(&instruction)
                .then_some(probe.range)
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
struct InstructionRange {
    range: SourceRange,
    instructions: Range<usize>,
}

enum MatchExpressions {
    Single(ValueOrSource),
    List(Vec<ValueOrSource>),
}
