use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::ControlFlow;

use serde::{Deserialize, Serialize};

use super::bitcode::{Op, OpDestination, ValueOrSource};
use super::Register;
use crate::compiler::UnaryKind;
use crate::map::Map;
use crate::symbol::Symbol;
use crate::syntax::{BinaryKind, CompareKind};
use crate::value::Value;
use crate::vm::{Fault, Vm};

pub trait Instruction: Send + Sync + Debug + 'static {
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault>;
    fn as_op(&self) -> Op;
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
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let source = self.source.load(vm)?;

        self.dest.store(vm, source)?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::Unary {
            op: self.source.as_source(),
            dest: self.dest.as_dest(),
            kind: UnaryKind::Copy,
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct Return;

impl Instruction for Return {
    fn execute(&self, _vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        Ok(ControlFlow::Break(()))
    }

    fn as_op(&self) -> Op {
        Op::Return
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct Jump<From, Dest> {
    pub target: From,
    pub previous_address: Dest,
}

impl<From, Dest> Instruction for Jump<From, Dest>
where
    From: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let current_instruction = u64::try_from(vm.current_instruction())
            .map_err(|_| Fault::InvalidInstructionAddress)?;

        let target = self.target.load(vm)?;
        if let Some(target) = target.as_u64() {
            vm.jump_to(usize::try_from(target).map_err(|_| Fault::InvalidInstructionAddress)?);
            self.previous_address
                .store(vm, Value::UInt(current_instruction))?;

            Ok(ControlFlow::Continue(()))
        } else {
            Err(Fault::InvalidInstructionAddress)
        }
    }

    fn as_op(&self) -> Op {
        Op::Unary {
            op: self.target.as_source(),
            dest: self.previous_address.as_dest(),
            kind: UnaryKind::Copy,
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct JumpIf<Target, Condition, PreviousAddr> {
    pub target: Target,
    pub condition: Condition,
    pub previous_address: PreviousAddr,
}

impl<Target, Condition, PreviousAddr> Instruction for JumpIf<Target, Condition, PreviousAddr>
where
    Target: Source,
    Condition: Source,
    PreviousAddr: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let condition = self.condition.load(vm)?;

        if condition.truthy(vm) {
            let current_instruction = u64::try_from(vm.current_instruction())
                .map_err(|_| Fault::InvalidInstructionAddress)?;

            let target = self.target.load(vm)?;
            if let Some(target) = target.as_u64() {
                vm.jump_to(usize::try_from(target).map_err(|_| Fault::InvalidInstructionAddress)?);
                self.previous_address
                    .store(vm, Value::UInt(current_instruction))?;

                Ok(ControlFlow::Continue(()))
            } else {
                Err(Fault::InvalidInstructionAddress)
            }
        } else {
            Ok(ControlFlow::Continue(()))
        }
    }

    fn as_op(&self) -> Op {
        Op::Unary {
            op: self.target.as_source(),
            dest: self.previous_address.as_dest(),
            kind: UnaryKind::Copy,
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct NewMap<Count, Dest> {
    pub element_count: Count,
    pub dest: Dest,
}

impl<From, Dest> Instruction for NewMap<From, Dest>
where
    From: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let element_count = self.element_count.load(vm)?;
        if let Some(element_count) = element_count
            .as_u64()
            .and_then(|c| u8::try_from(c).ok())
            .filter(|count| count < &128)
        {
            let map = Map::new();
            for reg_index in (0..element_count * 2).step_by(2) {
                let key = vm[Register(reg_index)].take();
                let value = vm[Register(reg_index)].take();
                map.insert(vm, key, value)?;
            }
            self.dest.store(vm, Value::dynamic(map))?;

            Ok(ControlFlow::Continue(()))
        } else {
            Err(Fault::InvalidArity)
        }
    }

    fn as_op(&self) -> Op {
        Op::Unary {
            op: self.element_count.as_source(),
            dest: self.dest.as_dest(),
            kind: UnaryKind::Copy,
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct Resolve<From, Dest> {
    pub source: From,
    pub dest: Dest,
}

impl<From, Dest> Instruction for Resolve<From, Dest>
where
    From: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let source = self.source.load(vm)?;
        let Some(source) = source.as_symbol() else {
            return Err(Fault::ExpectedSymbol);
        };

        let resolved = vm.resolve(source)?;
        self.dest.store(vm, resolved)?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::Unary {
            op: self.source.as_source(),
            dest: self.dest.as_dest(),
            kind: UnaryKind::Resolve,
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct Call<Func, NumArgs, Dest> {
    pub function: Func,
    pub arity: NumArgs,
    pub dest: Dest,
}

impl<Func, NumArgs, Dest> Instruction for Call<Func, NumArgs, Dest>
where
    Func: Source,
    NumArgs: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let function = self.function.load(vm)?;
        let arity = self.arity.load(vm)?;
        let arity = match arity.as_u64() {
            Some(int) => u8::try_from(int).map_err(|_| Fault::InvalidArity)?,
            _ => return Err(Fault::InvalidArity),
        };
        let result = function.call(vm, arity)?;

        self.dest.store(vm, result)?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::BinOp {
            op1: self.function.as_source(),
            op2: self.arity.as_source(),
            dest: self.dest.as_dest(),
            kind: BinaryKind::Call,
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct Invoke<Func, NumArgs, Dest> {
    pub name: Symbol,
    pub target: Func,
    pub arity: NumArgs,
    pub dest: Dest,
}

impl<Func, NumArgs, Dest> Instruction for Invoke<Func, NumArgs, Dest>
where
    Func: Source,
    NumArgs: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let target = self.target.load(vm)?;
        let arity = self.arity.load(vm)?;
        let arity = match arity.as_u64() {
            Some(int) => u8::try_from(int).map_err(|_| Fault::InvalidArity)?,
            _ => return Err(Fault::InvalidArity),
        };
        let result = target.invoke(vm, &self.name, arity)?;

        self.dest.store(vm, result)?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::Invoke {
            target: self.target.as_source(),
            arity: self.arity.as_source(),
            name: self.name.clone(),
            dest: self.dest.as_dest(),
        }
    }
}

macro_rules! declare_comparison_instruction {
    ($kind:ident, $pattern:pat) => {
        #[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
        pub struct $kind<Lhs, Rhs, Dest> {
            pub lhs: Lhs,
            pub rhs: Rhs,
            pub dest: Dest,
        }

        impl<Func, NumArgs, Dest> Instruction for $kind<Func, NumArgs, Dest>
        where
            Func: Source,
            NumArgs: Source,
            Dest: Destination,
        {
            fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
                let lhs = self.lhs.load(vm)?;
                let rhs = self.rhs.load(vm)?;
                let ordering = lhs.total_cmp(vm, &rhs)?;
                self.dest
                    .store(vm, Value::Bool(matches!(ordering, $pattern)))?;

                Ok(ControlFlow::Continue(()))
            }

            fn as_op(&self) -> Op {
                Op::BinOp {
                    op1: self.lhs.as_source(),
                    op2: self.rhs.as_source(),
                    dest: self.dest.as_dest(),
                    kind: BinaryKind::Compare(CompareKind::$kind),
                }
            }
        }
    };
}

declare_comparison_instruction!(LessThanOrEqual, Ordering::Less | Ordering::Equal);
declare_comparison_instruction!(LessThan, Ordering::Less);
declare_comparison_instruction!(Equal, Ordering::Equal);
declare_comparison_instruction!(GreaterThan, Ordering::Greater);
declare_comparison_instruction!(GreaterThanOrEqual, Ordering::Greater | Ordering::Equal);

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
            fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
                let lhs = self.lhs.load(vm)?;
                let rhs = self.rhs.load(vm)?;
                let result = lhs.$function(vm, &rhs)?;
                self.dest.store(vm, result)?;

                Ok(ControlFlow::Continue(()))
            }

            fn as_op(&self) -> Op {
                Op::BinOp {
                    op1: self.lhs.as_source(),
                    op2: self.rhs.as_source(),
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

pub trait Source: Send + Sync + Debug + 'static {
    fn load(&self, vm: &Vm) -> Result<Value, Fault>;
    fn as_source(&self) -> ValueOrSource;
}

impl Source for () {
    fn load(&self, _vm: &Vm) -> Result<Value, Fault> {
        Ok(Value::Nil)
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::Nil
    }
}

impl Source for bool {
    fn load(&self, _vm: &Vm) -> Result<Value, Fault> {
        Ok(Value::Bool(*self))
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::Bool(*self)
    }
}

impl Source for i64 {
    fn load(&self, _vm: &Vm) -> Result<Value, Fault> {
        Ok(Value::Int(*self))
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::Int(*self)
    }
}

impl Source for u64 {
    fn load(&self, _vm: &Vm) -> Result<Value, Fault> {
        Ok(Value::UInt(*self))
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::UInt(*self)
    }
}

impl Source for f64 {
    fn load(&self, _vm: &Vm) -> Result<Value, Fault> {
        Ok(Value::Float(*self))
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::Float(*self)
    }
}

impl Source for Symbol {
    fn load(&self, _vm: &Vm) -> Result<Value, Fault> {
        Ok(Value::Symbol(self.clone()))
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::Symbol(self.clone())
    }
}

pub trait Destination: Send + Sync + Debug + 'static {
    fn store(&self, vm: &mut Vm, value: Value) -> Result<(), Fault>;
    fn as_dest(&self) -> OpDestination;
}

impl Destination for () {
    fn store(&self, _vm: &mut Vm, _value: Value) -> Result<(), Fault> {
        Ok(())
    }

    fn as_dest(&self) -> OpDestination {
        OpDestination::Void
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Stack(pub usize);

impl Source for Stack {
    fn load(&self, vm: &Vm) -> Result<Value, Fault> {
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
