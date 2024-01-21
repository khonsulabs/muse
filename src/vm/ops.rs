use std::fmt::Debug;

use super::bitcode::{Arg, AsArg, BitcodeEncoder, Opcode};
use crate::symbol::Symbol;
use crate::value::Value;
use crate::vm::{Fault, Vm};

pub trait Instruction: Debug + 'static {
    fn execute(&self, vm: &mut Vm) -> Result<(), Fault>;
    fn encode_into(&self, encoder: &mut BitcodeEncoder);
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub struct Allocate(pub u16);

impl Instruction for Allocate {
    fn execute(&self, vm: &mut Vm) -> Result<(), Fault> {
        vm.allocate(self.0)?;
        Ok(())
    }

    fn encode_into(&self, encoder: &mut BitcodeEncoder) {
        encoder.encode(Opcode::Allocate, &[Arg::Int(i64::from(self.0))]);
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct Add<Lhs, Rhs, Dest> {
    pub lhs: Lhs,
    pub rhs: Rhs,
    pub dest: Dest,
}

impl<Lhs, Rhs, Dest> Instruction for Add<Lhs, Rhs, Dest>
where
    Lhs: Source,
    Rhs: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<(), Fault> {
        let lhs = self.lhs.load(vm)?;
        let rhs = self.rhs.load(vm)?;
        let result = lhs.add(vm, rhs)?;
        self.dest.store(vm, result)
    }

    fn encode_into(&self, encoder: &mut BitcodeEncoder) {
        encoder.encode(
            Opcode::Add,
            &[self.lhs.as_arg(), self.rhs.as_arg(), self.dest.as_arg()],
        )
    }
}

pub trait Source: AsArg + Debug + 'static {
    fn load(&self, vm: &mut Vm) -> Result<Value, Fault>;
}

pub trait Destination: AsArg + Debug + 'static {
    fn store(&self, vm: &mut Vm, value: Value) -> Result<(), Fault>;
}

impl Source for i64 {
    fn load(&self, _vm: &mut Vm) -> Result<Value, Fault> {
        Ok(Value::Int(*self))
    }
}

impl Source for f64 {
    fn load(&self, _vm: &mut Vm) -> Result<Value, Fault> {
        Ok(Value::Float(*self))
    }
}

impl Source for Symbol {
    fn load(&self, _vm: &mut Vm) -> Result<Value, Fault> {
        Ok(Value::Symbol(self.clone()))
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct Stack(pub usize);

impl Source for Stack {
    fn load(&self, vm: &mut Vm) -> Result<Value, Fault> {
        vm.current_frame()
            .get(self.0)
            .cloned()
            .ok_or(Fault::OutOfBounds)
    }
}

impl Destination for Stack {
    fn store(&self, vm: &mut Vm, value: Value) -> Result<(), Fault> {
        *vm.current_frame_mut()
            .get_mut(self.0)
            .ok_or(Fault::OutOfBounds)? = value;
        Ok(())
    }
}

impl AsArg for Stack {
    fn as_arg(&self) -> Arg {
        Arg::Variable { index: self.0 }
    }
}
