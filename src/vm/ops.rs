use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use super::bitcode::{Op, OpDestination, ValueOrSource};
use crate::symbol::Symbol;
use crate::syntax::{BinaryKind, UnaryKind};
use crate::value::Value;
use crate::vm::{Fault, Vm};

pub trait Instruction: Debug + 'static {
    fn execute(&self, vm: &mut Vm) -> Result<(), Fault>;
    fn as_op(&self) -> Op;
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub struct Allocate(pub u16);

impl Instruction for Allocate {
    fn execute(&self, vm: &mut Vm) -> Result<(), Fault> {
        vm.allocate(self.0)?;
        Ok(())
    }

    fn as_op(&self) -> Op {
        Op::Allocate(self.0)
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct Load<From, Dest> {
    pub source: From,
    pub dest: Dest,
}

impl<From, Dest> Instruction for Load<From, Dest>
where
    From: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<(), Fault> {
        let source = self.source.load(vm)?;

        self.dest.store(vm, source)
    }

    fn as_op(&self) -> Op {
        Op::Unary {
            source: self.source.as_source(),
            dest: self.dest.as_dest(),
            kind: UnaryKind::Copy,
        }
    }
}

macro_rules! declare_binop_instruction {
    ($name:ident, $function:ident, $kind:expr) => {
        #[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
        pub struct $name<Lhs, Rhs, Dest> {
            pub lhs: Lhs,
            pub rhs: Rhs,
            pub dest: Dest,
        }

        impl<Lhs, Rhs, Dest> Instruction for $name<Lhs, Rhs, Dest>
        where
            Lhs: Source,
            Rhs: Source,
            Dest: Destination,
        {
            fn execute(&self, vm: &mut Vm) -> Result<(), Fault> {
                let lhs = self.lhs.load(vm)?;
                let rhs = self.rhs.load(vm)?;
                let result = lhs.$function(vm, &rhs)?;
                self.dest.store(vm, result)
            }

            fn as_op(&self) -> Op {
                Op::BinOp {
                    left: self.lhs.as_source(),
                    right: self.rhs.as_source(),
                    dest: self.dest.as_dest(),
                    kind: $kind,
                }
            }
        }
    };
}

declare_binop_instruction!(Add, add, BinaryKind::Add);
declare_binop_instruction!(Subtract, sub, BinaryKind::Subtract);
declare_binop_instruction!(Multiply, mul, BinaryKind::Multiply);
declare_binop_instruction!(Divide, div, BinaryKind::Divide);
declare_binop_instruction!(IntegerDivide, div_i, BinaryKind::IntegerDivide);
declare_binop_instruction!(Remainder, rem, BinaryKind::Remainder);
declare_binop_instruction!(Power, pow, BinaryKind::Power);

pub trait Source: Debug + 'static {
    fn load(&self, vm: &mut Vm) -> Result<Value, Fault>;
    fn as_source(&self) -> ValueOrSource;
}

impl Source for () {
    fn load(&self, _vm: &mut Vm) -> Result<Value, Fault> {
        Ok(Value::Nil)
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::Nil
    }
}

impl Source for i64 {
    fn load(&self, _vm: &mut Vm) -> Result<Value, Fault> {
        Ok(Value::Int(*self))
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::Int(*self)
    }
}

impl Source for f64 {
    fn load(&self, _vm: &mut Vm) -> Result<Value, Fault> {
        Ok(Value::Float(*self))
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::Float(*self)
    }
}

impl Source for Symbol {
    fn load(&self, _vm: &mut Vm) -> Result<Value, Fault> {
        Ok(Value::Symbol(self.clone()))
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::Symbol(self.clone())
    }
}

pub trait Destination: Debug + 'static {
    fn store(&self, vm: &mut Vm, value: Value) -> Result<(), Fault>;
    fn as_dest(&self) -> OpDestination;
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Stack(pub usize);

impl Source for Stack {
    fn load(&self, vm: &mut Vm) -> Result<Value, Fault> {
        vm.current_frame()
            .get(self.0)
            .cloned()
            .ok_or(Fault::OutOfBounds)
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::Stack(*self)
    }
}

impl Destination for Stack {
    fn store(&self, vm: &mut Vm, value: Value) -> Result<(), Fault> {
        *vm.current_frame_mut()
            .get_mut(self.0)
            .ok_or(Fault::OutOfBounds)? = value;
        Ok(())
    }

    fn as_dest(&self) -> OpDestination {
        OpDestination::Stack(*self)
    }
}
