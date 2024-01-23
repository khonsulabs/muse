use std::mem;

use crate::syntax::{
    self, BinaryExpression, BinaryKind, Expression, LogicalKind, Ranged, UnaryKind,
};
use crate::vm::bitcode::{BitcodeBlock, Op, OpDestination, ValueOrSource};
use crate::vm::ops::Stack;
use crate::vm::{Code, Register};

#[derive(Default, Debug)]
pub struct Compiler {
    parsed: Vec<Result<Ranged<Expression>, Ranged<syntax::Error>>>,
    errors: Vec<Ranged<Error>>,
    code: BitcodeBlock,
    variables: usize,
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

        Scope::root(&mut self)
            .compile_expression(&expression, OpDestination::Register(Register::R1));
        self.code.push(Op::Unary {
            source: ValueOrSource::Int(1),
            dest: OpDestination::Register(Register::R0),
            kind: UnaryKind::Copy,
        });

        if self.errors.is_empty() {
            if self.variables > 0 {
                let variable_count = u16::try_from(self.variables).unwrap();
                match self.code.first_mut() {
                    Some(Op::Allocate(amount)) => {
                        if let Some(new_count) = amount.checked_add(variable_count) {
                            *amount += new_count;
                        } else {
                            todo!("too many vars")
                        }
                    }
                    _ => self
                        .code
                        .insert(0, Op::Allocate(u16::try_from(self.variables).unwrap())),
                }
            }
            Ok(Code::from(&self.code))
        } else {
            Err(self.errors)
        }
    }

    fn new_variable(&mut self) -> Stack {
        let id = self.variables;
        self.variables += 1;
        Stack(id)
    }
}

struct Scope<'a> {
    compiler: &'a mut Compiler,
    local_start: usize,
    local_count: usize,
    variables: usize,
}

impl<'a> Scope<'a> {
    fn root(compiler: &'a mut Compiler) -> Self {
        Self {
            local_start: compiler.variables,
            compiler,
            local_count: 0,
            variables: 0,
        }
    }

    fn enter_block(&mut self) -> Scope<'_> {
        Scope {
            local_start: self.compiler.variables,
            compiler: self.compiler,
            local_count: 0,
            variables: 0,
        }
    }

    fn new_variable(&mut self) -> Stack {
        if self.variables < self.local_count {
            let id = self.variables + self.local_start;
            self.variables += 1;
            Stack(id)
        } else {
            self.local_count += 1;
            self.variables += 1;
            self.compiler.new_variable()
        }
    }

    fn reuse_locals(&mut self) {
        self.variables = 0;
    }

    fn compile_expression(&mut self, expr: &Ranged<Expression>, dest: OpDestination) {
        match &**expr {
            Expression::Nil => self.compiler.code.push(Op::Unary {
                source: ValueOrSource::Nil,
                dest,
                kind: UnaryKind::Copy,
            }),
            Expression::Int(_) => todo!(),
            Expression::Float(_) => todo!(),
            Expression::Lookup(_) => todo!(),
            Expression::Unary(_) => todo!(),
            Expression::Binary(binop) => {
                self.compile_binop(binop, dest);
            }
            Expression::Block { expressions, .. } => {
                let mut block = self.enter_block();
                for exp in expressions {
                    block.compile_expression(exp, dest);
                    block.reuse_locals();
                }
            }
        }
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
            left,
            right,
            dest,
            kind,
        });
    }

    fn compile_source(&mut self, source: &Ranged<Expression>) -> ValueOrSource {
        match &source.0 {
            Expression::Nil => ValueOrSource::Nil,
            Expression::Int(int) => ValueOrSource::Int(*int),
            Expression::Float(float) => ValueOrSource::Float(*float),
            _ => {
                let var = self.new_variable();
                self.compile_expression(source, OpDestination::Stack(var));
                ValueOrSource::Stack(var)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    Syntax(syntax::Error),
}

impl From<Ranged<syntax::Error>> for Ranged<Error> {
    fn from(err: Ranged<syntax::Error>) -> Self {
        err.map(Error::Syntax)
    }
}
