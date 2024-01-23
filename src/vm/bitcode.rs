use std::ops::{Deref, DerefMut};
use std::str;

use serde::{Deserialize, Serialize};

use super::ops::{
    Add, Allocate, Destination, Divide, IntegerDivide, Load, Multiply, Power, Remainder, Source,
    Subtract,
};
use super::{Code, Register};
use crate::symbol::Symbol;
use crate::syntax::{BinaryKind, UnaryKind};
use crate::vm::ops::Stack;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValueOrSource {
    Nil,
    Int(i64),
    Float(f64),
    Symbol(Symbol),
    Register(Register),
    Stack(Stack),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum OpDestination {
    Register(Register),
    Stack(Stack),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    Allocate(u16),
    Unary {
        source: ValueOrSource,
        dest: OpDestination,
        kind: UnaryKind,
    },
    BinOp {
        left: ValueOrSource,
        right: ValueOrSource,
        dest: OpDestination,
        kind: BinaryKind,
    },
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct BitcodeBlock(Vec<Op>);

impl BitcodeBlock {
    pub fn push(&mut self, op: Op) {
        self.0.push(op);
    }
}

impl From<&'_ BitcodeBlock> for Code {
    fn from(bitcode: &'_ BitcodeBlock) -> Self {
        let mut code = Code::default();
        for op in &bitcode.0 {
            match op {
                Op::Allocate(amount) => code.push(Allocate(*amount)),
                Op::Unary { source, dest, kind } => match kind {
                    UnaryKind::Copy => match_copy(source, dest, &mut code),
                    UnaryKind::LogicalNot => todo!(),
                    UnaryKind::BitwiseNot => todo!(),
                    UnaryKind::Negate => todo!(),
                },
                Op::BinOp {
                    left,
                    right,
                    dest,
                    kind,
                } => match kind {
                    BinaryKind::Add => match_add(left, right, dest, &mut code),
                    BinaryKind::Subtract => match_subtract(left, right, dest, &mut code),
                    BinaryKind::Multiply => match_multiply(left, right, dest, &mut code),
                    BinaryKind::Divide => match_divide(left, right, dest, &mut code),
                    BinaryKind::IntegerDivide => match_integer_divide(left, right, dest, &mut code),
                    BinaryKind::Remainder => match_remainder(left, right, dest, &mut code),
                    BinaryKind::Power => match_power(left, right, dest, &mut code),
                    BinaryKind::Bitwise(_) => todo!(),
                    BinaryKind::Logical(_) => todo!(),
                    BinaryKind::Compare(_) => todo!(),
                },
            }
        }
        code
    }
}

impl From<&'_ Code> for BitcodeBlock {
    fn from(code: &'_ Code) -> Self {
        let mut ops = Vec::with_capacity(code.instructions.len());

        for instruction in &*code.instructions {
            ops.push(instruction.as_op());
        }

        BitcodeBlock(ops)
    }
}

impl Deref for BitcodeBlock {
    type Target = Vec<Op>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BitcodeBlock {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

macro_rules! decode_source {
    ($decode:expr, $source:expr, $code:ident, $next_fn:ident $(, $($arg:tt)*)?) => {{
        match $decode {
            ValueOrSource::Nil => $next_fn($source, $code $(, $($arg)*)?, ()),
            ValueOrSource::Int(source) => $next_fn($source, $code $(, $($arg)*)?, *source),
            ValueOrSource::Float(source) => $next_fn($source, $code $(, $($arg)*)?, *source),
            ValueOrSource::Symbol(source) => $next_fn($source, $code $(, $($arg)*)?, source.clone()),
            ValueOrSource::Register(source) => $next_fn($source, $code $(, $($arg)*)?, *source),
            ValueOrSource::Stack(source) => $next_fn($source, $code $(, $($arg)*)?, *source),
        }
    }};
}

macro_rules! decode_dest {
    ($decode:expr, $source:expr, $code:ident, $next_fn:ident $(, $($arg:tt)*)?) => {{
        match $decode {
            OpDestination::Stack(stack) => $next_fn($source, $code $(, $($arg)*)?, *stack),
            OpDestination::Register(register) => $next_fn($source, $code $(, $($arg)*)?, *register),
        }
    }};
}

macro_rules! decode_sd {
    ($decode_name:ident, $compile_name:ident) => {
        fn $decode_name(s: &ValueOrSource, d: &OpDestination, code: &mut Code) {
            fn source<Lhs>(dest: &OpDestination, code: &mut Code, source1: Lhs)
            where
                Lhs: Source,
            {
                decode_dest!(dest, dest, code, $compile_name, source1)
            }

            decode_source!(s, d, code, source)
        }
    };
}

decode_sd!(match_copy, compile_copy);

fn compile_copy<From, Dest>(_dest: &OpDestination, code: &mut Code, source: From, dest: Dest)
where
    From: Source,
    Dest: Destination,
{
    code.push(Load { source, dest });
}

macro_rules! decode_ssd {
    ($decode_name:ident, $compile_name:ident) => {
        fn $decode_name(
            s1: &ValueOrSource,
            s2: &ValueOrSource,
            d: &OpDestination,
            code: &mut Code,
        ) {
            fn source<Lhs>(source: (&ValueOrSource, &OpDestination), code: &mut Code, source1: Lhs)
            where
                Lhs: Source,
            {
                decode_source!(source.0, source, code, source_source, source1)
            }

            fn source_source<Lhs, Rhs>(
                source: (&ValueOrSource, &OpDestination),
                code: &mut Code,
                source1: Lhs,
                source2: Rhs,
            ) where
                Lhs: Source,
                Rhs: Source,
            {
                decode_dest!(source.1, source, code, $compile_name, source1, source2)
            }

            decode_source!(s1, (s2, d), code, source)
        }
    };
}

macro_rules! define_match_binop {
    ($match:ident, $compile:ident, $instruction:ident) => {
        decode_ssd!($match, $compile);

        fn $compile<Lhs, Rhs, Dest>(
            _source: (&ValueOrSource, &OpDestination),
            code: &mut Code,
            lhs: Lhs,
            rhs: Rhs,
            dest: Dest,
        ) where
            Lhs: Source,
            Rhs: Source,
            Dest: Destination,
        {
            code.push($instruction { lhs, rhs, dest });
        }
    };
}

define_match_binop!(match_add, compile_add, Add);
define_match_binop!(match_subtract, compile_subtract, Subtract);
define_match_binop!(match_multiply, compile_multiply, Multiply);
define_match_binop!(match_divide, compile_divide, Divide);
define_match_binop!(match_integer_divide, compile_integer_divide, IntegerDivide);
define_match_binop!(match_remainder, compile_remainder, Remainder);
define_match_binop!(match_power, compile_power, Power);
