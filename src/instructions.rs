use std::fmt::Debug;

use crate::symbol::Symbol;
use crate::value::Value;
use crate::{Fault, Vm};

pub trait Instruction: Debug + 'static {
    fn execute(&self, vm: &mut Vm) -> Result<(), Fault>;
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub struct Allocate(pub usize);

impl Instruction for Allocate {
    fn execute(&self, vm: &mut Vm) -> Result<(), Fault> {
        vm.allocate(self.0)?;
        Ok(())
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
}

pub trait Source: Debug + 'static {
    fn load(&self, vm: &mut Vm) -> Result<Value, Fault>;
}

pub trait Destination: Debug + 'static {
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
