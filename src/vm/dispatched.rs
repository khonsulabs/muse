use std::cmp::Ordering;
use std::fmt::Debug;
use std::num::NonZeroUsize;
use std::ops::ControlFlow;
use std::sync::Arc;

use super::bitcode::{
    trusted_loaded_source_to_value, BinaryKind, BitcodeFunction, FaultKind, Label, Op,
    OpDestination, ValueOrSource,
};
use super::{
    precompiled_regex, Code, CodeData, Function, LoadedOp, Module, PrecompiledRegex, Register,
    Stack,
};
use crate::compiler::{BitcodeModule, UnaryKind};
use crate::list::List;
use crate::map::Map;
use crate::string::MuseString;
use crate::symbol::Symbol;
use crate::syntax::{BitwiseKind, CompareKind};
use crate::value::{Dynamic, Value};
use crate::vm::{Fault, SourceRange, Vm};

impl CodeData {
    #[allow(clippy::too_many_lines)]
    pub(super) fn push_loaded(&mut self, loaded: LoadedOp, range: SourceRange) {
        self.map.push(range);
        match loaded {
            LoadedOp::Return => {
                self.push_dispatched(Return);
            }
            LoadedOp::Declare {
                name,
                mutable,
                value,
                dest,
            } => {
                let name = self.symbols[name].clone();
                match_declare_function(
                    &trusted_loaded_source_to_value(&value, self),
                    &dest,
                    self,
                    &name,
                    mutable,
                );
            }
            LoadedOp::Truthy(loaded) => match_truthy(
                &trusted_loaded_source_to_value(&loaded.op, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::LogicalNot(loaded) => match_logical_not(
                &trusted_loaded_source_to_value(&loaded.op, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::BitwiseNot(loaded) => match_bitwise_not(
                &trusted_loaded_source_to_value(&loaded.op, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::Negate(loaded) => match_negate(
                &trusted_loaded_source_to_value(&loaded.op, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::Copy(loaded) => match_copy(
                &trusted_loaded_source_to_value(&loaded.op, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::Resolve(loaded) => match_resolve(
                &trusted_loaded_source_to_value(&loaded.op, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::Jump(loaded) => match_jump(
                &trusted_loaded_source_to_value(&loaded.op, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::NewMap(loaded) => match_new_map(
                &trusted_loaded_source_to_value(&loaded.op, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::NewList(loaded) => match_new_list(
                &trusted_loaded_source_to_value(&loaded.op, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::SetExceptionHandler(loaded) => match_set_exception_handler(
                &trusted_loaded_source_to_value(&loaded.op, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::LogicalXor(loaded) => match_logical_xor(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::Assign(loaded) => match_assign(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::Add(loaded) => match_add(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::Subtract(loaded) => match_subtract(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::Multiply(loaded) => match_multiply(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::Divide(loaded) => match_divide(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::IntegerDivide(loaded) => match_integer_divide(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::Remainder(loaded) => match_remainder(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::Power(loaded) => match_power(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::JumpIf(loaded) => match_jump_if(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::JumpIfNot(loaded) => match_jump_if_not(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::LessThanOrEqual(loaded) => match_lte(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::LessThan(loaded) => match_lt(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::Equal(loaded) => match_equal(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::NotEqual(loaded) => match_not_equal(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::GreaterThan(loaded) => match_gt(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::GreaterThanOrEqual(loaded) => match_gte(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::Matches(loaded) => match_matches(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::Call { name, arity } => match_call(
                &trusted_loaded_source_to_value(&name, self),
                &trusted_loaded_source_to_value(&arity, self),
                self,
            ),
            LoadedOp::Invoke {
                target,
                name,
                arity,
            } => {
                let name = self.symbols[name].clone();
                match_invoke(
                    &trusted_loaded_source_to_value(&target, self),
                    &trusted_loaded_source_to_value(&arity, self),
                    self,
                    &name,
                );
            }
            LoadedOp::BitwiseAnd(loaded) => match_bitwise_and(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::BitwiseOr(loaded) => match_bitwise_or(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::BitwiseXor(loaded) => match_bitwise_xor(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::BitwiseShiftLeft(loaded) => match_bitwise_shl(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::BitwiseShiftRight(loaded) => match_bitwise_shr(
                &trusted_loaded_source_to_value(&loaded.op1, self),
                &trusted_loaded_source_to_value(&loaded.op2, self),
                &loaded.dest,
                self,
            ),
            LoadedOp::LoadModule { module, dest } => {
                match_load_module(&self.modules[module].clone(), &dest, self);
            }
            LoadedOp::Throw(kind) => {
                self.push_dispatched(Throw(kind));
            }
        }
    }

    fn push_dispatched<T>(&mut self, dispatched: T)
    where
        T: Instruction,
    {
        self.instructions.push(Arc::new(dispatched));
    }
}

pub trait Instruction: Send + Sync + Debug + 'static {
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault>;
    fn as_op(&self) -> Op;
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
pub struct Throw(pub FaultKind);

impl Instruction for Throw {
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        Err(Fault::from_kind(self.0, vm))
    }

    fn as_op(&self) -> Op {
        Op::Throw(self.0)
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
pub struct Truthy<From, Dest> {
    pub source: From,
    pub dest: Dest,
}

impl<From, Dest> Instruction for Truthy<From, Dest>
where
    From: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let source = self.source.load(vm)?.truthy(vm);

        self.dest.store(vm, Value::Bool(source))?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::Unary {
            op: self.source.as_source(),
            dest: self.dest.as_dest(),
            kind: UnaryKind::Truthy,
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct LogicalNot<From, Dest> {
    pub source: From,
    pub dest: Dest,
}

impl<From, Dest> Instruction for LogicalNot<From, Dest>
where
    From: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let source = self.source.load(vm)?.not(vm)?;

        self.dest.store(vm, source)?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::Unary {
            op: self.source.as_source(),
            dest: self.dest.as_dest(),
            kind: UnaryKind::LogicalNot,
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct BitwiseNot<From, Dest> {
    pub source: From,
    pub dest: Dest,
}

impl<From, Dest> Instruction for BitwiseNot<From, Dest>
where
    From: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let source = self.source.load(vm)?.bitwise_not(vm)?;

        self.dest.store(vm, source)?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::Unary {
            op: self.source.as_source(),
            dest: self.dest.as_dest(),
            kind: UnaryKind::BitwiseNot,
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct Negate<From, Dest> {
    pub source: From,
    pub dest: Dest,
}

impl<From, Dest> Instruction for Negate<From, Dest>
where
    From: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let source = self.source.load(vm)?.negate(vm)?;

        self.dest.store(vm, source)?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::Unary {
            op: self.source.as_source(),
            dest: self.dest.as_dest(),
            kind: UnaryKind::Negate,
        }
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
pub struct JumpIf<Target, Condition, PreviousAddr, const NOT: bool> {
    pub target: Target,
    pub condition: Condition,
    pub previous_address: PreviousAddr,
}

impl<Target, Condition, PreviousAddr, const NOT: bool> Instruction
    for JumpIf<Target, Condition, PreviousAddr, NOT>
where
    Target: Source,
    Condition: Source,
    PreviousAddr: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let mut condition = self.condition.load(vm)?.truthy(vm);

        if NOT {
            condition = !condition;
        }

        if condition {
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
        Op::BinOp {
            op1: self.target.as_source(),
            op2: self.condition.as_source(),
            dest: self.previous_address.as_dest(),
            kind: if NOT {
                BinaryKind::JumpIfNot
            } else {
                BinaryKind::JumpIf
            },
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
                let value = vm[Register(reg_index + 1)].take();
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
pub struct NewList<Count, Dest> {
    pub element_count: Count,
    pub dest: Dest,
}

impl<From, Dest> Instruction for NewList<From, Dest>
where
    From: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let element_count = self.element_count.load(vm)?;
        if let Some(element_count) = element_count.as_u64().and_then(|c| u8::try_from(c).ok()) {
            let list = List::new();
            for reg_index in 0..element_count {
                let value = vm[Register(reg_index)].take();
                list.push(value)?;
            }
            self.dest.store(vm, Value::dynamic(list))?;

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
pub struct SetExceptionHandler<Handler, Dest> {
    pub handler: Handler,
    pub previous_handler: Dest,
}

impl<Handler, Dest> Instruction for SetExceptionHandler<Handler, Dest>
where
    Handler: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let handler = self.handler.load(vm)?;
        let handler = handler.as_usize().and_then(NonZeroUsize::new);

        let previous_handler_address =
            std::mem::replace(&mut vm.frames[vm.current_frame].exception_handler, handler);
        self.previous_handler
            .store(
                vm,
                previous_handler_address
                    .and_then(|addr| Value::try_from(addr.get()).ok())
                    .unwrap_or_default(),
            )
            .map(|()| ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::Unary {
            op: self.handler.as_source(),
            dest: self.previous_handler.as_dest(),
            kind: UnaryKind::SetExceptionHandler,
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
pub struct Call<Func, NumArgs> {
    pub function: Func,
    pub arity: NumArgs,
}

impl<Func, NumArgs> Instruction for Call<Func, NumArgs>
where
    Func: Source,
    NumArgs: Source,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let (function, arity) = try_all!(self.function.load(vm), self.arity.load(vm));
        let arity = match arity.as_u64() {
            Some(int) => u8::try_from(int).map_err(|_| Fault::InvalidArity)?,
            _ => return Err(Fault::InvalidArity),
        };
        vm[Register(0)] = function.call(vm, arity)?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::Call {
            name: self.function.as_source(),
            arity: self.arity.as_source(),
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
pub struct Invoke<Func, NumArgs> {
    pub name: Symbol,
    pub target: Func,
    pub arity: NumArgs,
}

impl<Func, NumArgs> Instruction for Invoke<Func, NumArgs>
where
    Func: Source,
    NumArgs: Source,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let (target, arity) = try_all!(self.target.load(vm), self.arity.load(vm));
        let arity = match arity.as_u64() {
            Some(int) => u8::try_from(int).map_err(|_| Fault::InvalidArity)?,
            _ => return Err(Fault::InvalidArity),
        };
        vm[Register(0)] = target.invoke(vm, &self.name, arity)?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::Invoke {
            target: self.target.as_source(),
            arity: self.arity.as_source(),
            name: self.name.clone(),
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
                let (lhs, rhs) = try_all!(self.lhs.load(vm), self.rhs.load(vm));
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
                let (lhs, rhs) = try_all!(self.lhs.load(vm), self.rhs.load(vm));
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

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct LogicalXor<Lhs, Rhs, Dest> {
    pub lhs: Lhs,
    pub rhs: Rhs,
    pub dest: Dest,
}

impl<Lhs, Rhs, Dest> Instruction for LogicalXor<Lhs, Rhs, Dest>
where
    Lhs: Source,
    Rhs: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let (lhs, rhs) = try_all!(self.lhs.load(vm), self.rhs.load(vm));
        let (lhs, rhs) = (lhs.truthy(vm), rhs.truthy(vm));
        self.dest.store(vm, Value::Bool(lhs ^ rhs))?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::BinOp {
            op1: self.lhs.as_source(),
            op2: self.rhs.as_source(),
            dest: self.dest.as_dest(),
            kind: BinaryKind::LogicalXor,
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct Eq<Lhs, Rhs, Dest, const NOT: bool = false> {
    pub lhs: Lhs,
    pub rhs: Rhs,
    pub dest: Dest,
}

impl<Lhs, Rhs, Dest, const NOT: bool> Instruction for Eq<Lhs, Rhs, Dest, NOT>
where
    Lhs: Source,
    Rhs: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let (lhs, rhs) = try_all!(self.lhs.load(vm), self.rhs.load(vm));

        let equals = lhs.equals(Some(vm), &rhs)?;

        if NOT {
            self.dest.store(vm, Value::Bool(!equals))?;
        } else {
            self.dest.store(vm, Value::Bool(equals))?;
        }

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::BinOp {
            op1: self.lhs.as_source(),
            op2: self.rhs.as_source(),
            dest: self.dest.as_dest(),
            kind: BinaryKind::LogicalXor,
        }
    }
}

pub type Equal<Lhs, Rhs, Dest> = Eq<Lhs, Rhs, Dest, false>;
pub type NotEqual<Lhs, Rhs, Dest> = Eq<Lhs, Rhs, Dest, true>;

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct Matches<Lhs, Rhs, Dest> {
    pub lhs: Lhs,
    pub rhs: Rhs,
    pub dest: Dest,
}

impl<Lhs, Rhs, Dest> Instruction for Matches<Lhs, Rhs, Dest>
where
    Lhs: Source,
    Rhs: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let lhs = self.lhs.load(vm)?;
        let rhs = self.rhs.load(vm)?;
        let matches = lhs.matches(vm, &rhs)?;
        self.dest.store(vm, Value::Bool(matches))?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::BinOp {
            op1: self.lhs.as_source(),
            op2: self.rhs.as_source(),
            dest: self.dest.as_dest(),
            kind: BinaryKind::LogicalXor,
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub struct Assign<Lhs, Rhs, Dest> {
    pub lhs: Lhs,
    pub rhs: Rhs,
    pub dest: Dest,
}

impl<Lhs, Rhs, Dest> Instruction for Assign<Lhs, Rhs, Dest>
where
    Lhs: Source,
    Rhs: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let name = self.lhs.load(vm)?;
        let name = name.as_symbol().ok_or(Fault::ExpectedSymbol)?;
        let value = self.rhs.load(vm)?;
        vm.assign(name, value.clone())?;

        self.dest.store(vm, value)?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::BinOp {
            op1: self.lhs.as_source(),
            op2: self.rhs.as_source(),
            dest: self.dest.as_dest(),
            kind: BinaryKind::LogicalXor,
        }
    }
}

declare_binop_instruction!(Add, add, BinaryKind::Add);
declare_binop_instruction!(Subtract, sub, BinaryKind::Subtract);
declare_binop_instruction!(Multiply, mul, BinaryKind::Multiply);
declare_binop_instruction!(Divide, div, BinaryKind::Divide);
declare_binop_instruction!(IntegerDivide, idiv, BinaryKind::IntegerDivide);
declare_binop_instruction!(Remainder, rem, BinaryKind::Remainder);
declare_binop_instruction!(Power, pow, BinaryKind::Power);
declare_binop_instruction!(
    BitwiseAnd,
    bitwise_and,
    BinaryKind::Bitwise(BitwiseKind::And)
);
declare_binop_instruction!(BitwiseOr, bitwise_or, BinaryKind::Bitwise(BitwiseKind::Or));
declare_binop_instruction!(
    BitwiseXor,
    bitwise_xor,
    BinaryKind::Bitwise(BitwiseKind::Xor)
);
declare_binop_instruction!(
    ShiftLeft,
    shift_left,
    BinaryKind::Bitwise(BitwiseKind::ShiftLeft)
);
declare_binop_instruction!(
    ShiftRight,
    shift_right,
    BinaryKind::Bitwise(BitwiseKind::ShiftRight)
);

trait Source: Send + Sync + Debug + 'static {
    fn load(&self, vm: &Vm) -> Result<Value, Fault>;
    fn as_source(&self) -> ValueOrSource;
}

impl Source for Value {
    fn load(&self, _vm: &Vm) -> Result<Value, Fault> {
        Ok(self.clone())
    }

    fn as_source(&self) -> ValueOrSource {
        match self {
            Value::Nil => ValueOrSource::Nil,
            Value::Bool(value) => ValueOrSource::Bool(*value),
            Value::Int(value) => ValueOrSource::Int(*value),
            Value::UInt(value) => ValueOrSource::UInt(*value),
            Value::Float(value) => ValueOrSource::Float(*value),
            Value::Symbol(value) => ValueOrSource::Symbol(value.clone()),
            Value::Dynamic(value) => {
                if let Some(str) = value.downcast_ref::<MuseString>() {
                    ValueOrSource::String(str.to_string())
                } else if let Some(func) = value.downcast_ref::<Function>() {
                    ValueOrSource::Function(BitcodeFunction::from(func))
                } else {
                    todo!("Error handling")
                }
            }
        }
    }
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

impl Source for Function {
    fn load(&self, vm: &Vm) -> Result<Value, Fault> {
        Ok(Value::dynamic(
            self.clone().in_module(vm.frames[vm.current_frame].module),
        ))
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::Function(BitcodeFunction::from(self))
    }
}

trait Destination: Send + Sync + Debug + 'static {
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

impl Source for Register {
    fn load(&self, vm: &Vm) -> Result<Value, Fault> {
        Ok(vm[*self].clone())
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::Register(*self)
    }
}

impl Destination for Register {
    fn store(&self, vm: &mut Vm, value: Value) -> Result<(), Fault> {
        vm[*self] = value;
        Ok(())
    }

    fn as_dest(&self) -> OpDestination {
        OpDestination::Register(*self)
    }
}

impl Source for PrecompiledRegex {
    fn load(&self, _vm: &super::Vm) -> Result<Value, Fault> {
        self.result.clone()
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::Regex(self.literal.clone())
    }
}
impl Source for Label {
    fn load(&self, vm: &super::Vm) -> Result<Value, Fault> {
        let instruction = vm
            .current_code()
            .expect("missing frame code")
            .data
            .labels
            .get(self.0)
            .ok_or(Fault::InvalidLabel)?;
        let instruction = u64::try_from(*instruction).map_err(|_| Fault::InvalidLabel)?;
        Ok(Value::UInt(instruction))
    }

    fn as_source(&self) -> ValueOrSource {
        ValueOrSource::Label(*self)
    }
}

impl Destination for Label {
    fn store(&self, vm: &mut super::Vm, value: Value) -> Result<(), Fault> {
        if value.truthy(vm) {
            let instruction = vm
                .current_code()
                .expect("missing frame code")
                .data
                .labels
                .get(self.0)
                .ok_or(Fault::InvalidLabel)?;
            vm.jump_to(*instruction);
        }
        Ok(())
    }

    fn as_dest(&self) -> OpDestination {
        OpDestination::Label(*self)
    }
}

macro_rules! decode_source {
    ($decode:expr, $source:expr, $code:ident, $next_fn:ident $(, $($arg:tt)*)?) => {{
        match $decode {
            ValueOrSource::Nil => $next_fn($source, $code $(, $($arg)*)?, ()),
            ValueOrSource::Bool(source) => $next_fn($source, $code $(, $($arg)*)?, *source),
            ValueOrSource::Int(source) => $next_fn($source, $code $(, $($arg)*)?, *source),
            ValueOrSource::UInt(source) => $next_fn($source, $code $(, $($arg)*)?, *source),
            ValueOrSource::Float(source) => $next_fn($source, $code $(, $($arg)*)?, *source),
            ValueOrSource::Regex(source) => $next_fn($source, $code $(, $($arg)*)?, precompiled_regex(source)),
            ValueOrSource::String(source) => $next_fn($source, $code $(, $($arg)*)?, Value::dynamic(MuseString::from(source.clone()))),
            ValueOrSource::Symbol(source) => $next_fn($source, $code $(, $($arg)*)?, source.clone()),
            ValueOrSource::Function(source) => $next_fn($source, $code $(, $($arg)*)?, Function::from(source)),
            ValueOrSource::Register(source) => $next_fn($source, $code $(, $($arg)*)?, *source),
            ValueOrSource::Stack(source) => $next_fn($source, $code $(, $($arg)*)?, *source),
            ValueOrSource::Label(source) => $next_fn($source, $code $(, $($arg)*)?, *source),
        }
    }};
}

macro_rules! decode_dest {
    ($decode:expr, $source:expr, $code:ident, $next_fn:ident $(, $($arg:tt)*)?) => {{
        match $decode {
            OpDestination::Void => $next_fn($source, $code $(, $($arg)*)?, ()),
            OpDestination::Stack(stack) => $next_fn($source, $code $(, $($arg)*)?, *stack),
            OpDestination::Register(register) => $next_fn($source, $code $(, $($arg)*)?, *register),
            OpDestination::Label(label) => $next_fn($source, $code $(, $($arg)*)?, *label),
        }
    }};
}

macro_rules! decode_sd {
    ($decode_name:ident, $compile_name:ident $(, $($name:ident: $type:ty),+)?) => {
        fn $decode_name(s: &ValueOrSource, d: &OpDestination, code: &mut CodeData, $($($name: $type),+,)?) {
            fn source<Lhs>(dest: &OpDestination, code: &mut CodeData, $($($name: $type),+,)? source1: Lhs)
            where
                Lhs: Source,
            {
                decode_dest!(dest, dest, code, $compile_name, source1 $(, $($name),+)?)
            }

            decode_source!(s, d, code, source $(, $($name),+)?)
        }
    };
}

macro_rules! decode_sd_simple {
    ($decode_name:ident, $compile_name:ident, $inst:ident) => {
        decode_sd!($decode_name, $compile_name);

        fn $compile_name<From, Dest>(
            _dest: &OpDestination,
            code: &mut CodeData,
            source: From,
            dest: Dest,
        ) where
            From: Source,
            Dest: Destination,
        {
            code.push_dispatched($inst { source, dest });
        }
    };
}

decode_sd_simple!(match_copy, compile_copy, Load);
decode_sd_simple!(match_truthy, compile_truthy, Truthy);
decode_sd_simple!(match_logical_not, compile_logical_not, LogicalNot);
decode_sd_simple!(match_bitwise_not, compile_bitwise_not, BitwiseNot);
decode_sd_simple!(match_negate, compile_negate, Negate);

decode_sd!(match_declare_function, compile_declare_function, name: &Symbol, mutable: bool);

fn compile_declare_function<Value, Dest>(
    _dest: &OpDestination,
    code: &mut CodeData,
    f: Value,
    name: &Symbol,
    mutable: bool,
    dest: Dest,
) where
    Value: Source,
    Dest: Destination,
{
    code.push_dispatched(Declare {
        name: name.clone(),
        mutable,
        declaration: f,
        dest,
    });
}

decode_sd!(match_resolve, compile_resolve);

fn compile_resolve<From, Dest>(_dest: &OpDestination, code: &mut CodeData, source: From, dest: Dest)
where
    From: Source,
    Dest: Destination,
{
    code.push_dispatched(Resolve { source, dest });
}

decode_sd!(match_jump, compile_jump);

fn compile_jump<From, Dest>(_dest: &OpDestination, code: &mut CodeData, source: From, dest: Dest)
where
    From: Source,
    Dest: Destination,
{
    code.push_dispatched(Jump {
        target: source,
        previous_address: dest,
    });
}

decode_sd!(match_new_map, compile_new_map);

fn compile_new_map<Arity, Dest>(
    _dest: &OpDestination,
    code: &mut CodeData,
    element_count: Arity,
    dest: Dest,
) where
    Arity: Source,
    Dest: Destination,
{
    code.push_dispatched(NewMap {
        element_count,
        dest,
    });
}

decode_sd!(match_new_list, compile_new_list);

fn compile_new_list<Arity, Dest>(
    _dest: &OpDestination,
    code: &mut CodeData,
    element_count: Arity,
    dest: Dest,
) where
    Arity: Source,
    Dest: Destination,
{
    code.push_dispatched(NewList {
        element_count,
        dest,
    });
}

decode_sd!(match_set_exception_handler, compile_set_exception_handler);

fn compile_set_exception_handler<Handler, Dest>(
    _dest: &OpDestination,
    code: &mut CodeData,
    handler: Handler,
    previous_handler: Dest,
) where
    Handler: Source,
    Dest: Destination,
{
    code.push_dispatched(SetExceptionHandler {
        handler,
        previous_handler,
    });
}

macro_rules! decode_ssd {
    ($decode_name:ident, $compile_name:ident $(, $($name:ident: $type:ty),+)?) => {
        fn $decode_name(
            s1: &ValueOrSource,
            s2: &ValueOrSource,
            d: &OpDestination,
            code: &mut CodeData,
            $($($name: $type),+,)?
        ) {
            fn source<Lhs>(source: (&ValueOrSource, &OpDestination), code: &mut CodeData,
            $($($name: $type),+,)? source1: Lhs)
            where
                Lhs: Source,
            {
                decode_source!(source.0, source, code, source_source, source1 $(, $($name),+)?)
            }

            fn source_source<Lhs, Rhs>(
                source: (&ValueOrSource, &OpDestination),
                code: &mut CodeData,
                source1: Lhs,
                $($($name: $type),+,)?
                source2: Rhs,
            ) where
                Lhs: Source,
                Rhs: Source,
            {
                decode_dest!(source.1, source, code, $compile_name, source1, source2 $(, $($name),+)?)
            }

            decode_source!(s1, (s2, d), code, source $(, $($name),+)?)
        }
    };
}

macro_rules! decode_ss {
    ($decode_name:ident, $compile_name:ident $(, $($name:ident: $type:ty),+)?) => {
        fn $decode_name(
            s1: &ValueOrSource,
            s2: &ValueOrSource,
            code: &mut CodeData,
            $($($name: $type),+,)?
        ) {
            fn source<Lhs>(source: &ValueOrSource, code: &mut CodeData,
            $($($name: $type),+,)? source1: Lhs)
            where
                Lhs: Source,
            {
                decode_source!(source, source, code, $compile_name, source1 $(, $($name),+)?)
            }

            decode_source!(s1, s2, code, source $(, $($name),+)?)
        }
    };
}

decode_ss!(match_call, compile_call);

fn compile_call<Func, NumArgs>(
    _source: &ValueOrSource,
    code: &mut CodeData,
    function: Func,
    arity: NumArgs,
) where
    Func: Source,
    NumArgs: Source,
{
    code.push_dispatched(Call { function, arity });
}

decode_ss!(match_invoke, compile_invoke, name: &Symbol);

fn compile_invoke<Func, NumArgs>(
    _source: &ValueOrSource,
    code: &mut CodeData,
    target: Func,
    name: &Symbol,
    arity: NumArgs,
) where
    Func: Source,
    NumArgs: Source,
{
    code.push_dispatched(Invoke {
        name: name.clone(),
        target,
        arity,
    });
}

decode_ssd!(match_jump_if, compile_jump_if);

fn compile_jump_if<Func, NumArgs, Dest>(
    _source: (&ValueOrSource, &OpDestination),
    code: &mut CodeData,
    target: Func,
    condition: NumArgs,
    previous_address: Dest,
) where
    Func: Source,
    NumArgs: Source,
    Dest: Destination,
{
    code.push_dispatched(JumpIf::<_, _, _, false> {
        target,
        condition,
        previous_address,
    });
}

decode_ssd!(match_jump_if_not, compile_jump_if_not);

fn compile_jump_if_not<Func, NumArgs, Dest>(
    _source: (&ValueOrSource, &OpDestination),
    code: &mut CodeData,
    target: Func,
    condition: NumArgs,
    previous_address: Dest,
) where
    Func: Source,
    NumArgs: Source,
    Dest: Destination,
{
    code.push_dispatched(JumpIf::<_, _, _, true> {
        target,
        condition,
        previous_address,
    });
}

macro_rules! define_match_binop {
    ($match:ident, $compile:ident, $instruction:ident) => {
        decode_ssd!($match, $compile);

        fn $compile<Lhs, Rhs, Dest>(
            _source: (&ValueOrSource, &OpDestination),
            code: &mut CodeData,
            lhs: Lhs,
            rhs: Rhs,
            dest: Dest,
        ) where
            Lhs: Source,
            Rhs: Source,
            Dest: Destination,
        {
            code.push_dispatched($instruction { lhs, rhs, dest });
        }
    };
}

define_match_binop!(match_logical_xor, compile_logical_xor, LogicalXor);
define_match_binop!(match_assign, compile_assign, Assign);
define_match_binop!(match_add, compile_add, Add);
define_match_binop!(match_subtract, compile_subtract, Subtract);
define_match_binop!(match_multiply, compile_multiply, Multiply);
define_match_binop!(match_divide, compile_divide, Divide);
define_match_binop!(match_integer_divide, compile_integer_divide, IntegerDivide);
define_match_binop!(match_remainder, compile_remainder, Remainder);
define_match_binop!(match_power, compile_power, Power);
define_match_binop!(match_lte, compile_lte, LessThanOrEqual);
define_match_binop!(match_lt, compile_lt, LessThan);
define_match_binop!(match_equal, compile_equal, Equal);
define_match_binop!(match_not_equal, compile_not_equal, NotEqual);
define_match_binop!(match_gt, compile_gt, GreaterThan);
define_match_binop!(match_gte, compile_gte, GreaterThanOrEqual);
define_match_binop!(match_matches, compile_matches, Matches);
define_match_binop!(match_bitwise_and, compile_bitwise_and, BitwiseAnd);
define_match_binop!(match_bitwise_or, compile_bitwise_or, BitwiseOr);
define_match_binop!(match_bitwise_xor, compile_bitwise_xor, BitwiseXor);
define_match_binop!(match_bitwise_shl, compile_bitwise_shl, ShiftLeft);
define_match_binop!(match_bitwise_shr, compile_bitwise_shr, ShiftRight);

fn match_load_module(module: &BitcodeModule, dest: &OpDestination, code: &mut CodeData) {
    decode_dest!(dest, module, code, compile_load_module);
}

fn compile_load_module<Dest>(module: &BitcodeModule, code: &mut CodeData, dest: Dest)
where
    Dest: Destination,
{
    code.push_dispatched(LoadModule {
        bitcode: module.clone(),
        dest,
    });
}

#[derive(Debug)]
struct LoadModule<Dest> {
    bitcode: BitcodeModule,
    dest: Dest,
}

impl<Dest> Instruction for LoadModule<Dest>
where
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let loading_module = if let Some(index) = vm.frames[vm.current_frame].loading_module.take()
        {
            index
        } else {
            // Replace the current module and stage the initializer
            let executing_frame = vm.current_frame;
            let initializer = Code::from(&self.bitcode.initializer);
            let code = vm.push_code(&initializer, None);
            vm.enter_frame(Some(code))?;
            vm.allocate(initializer.data.stack_requirement)?;
            let module_index = NonZeroUsize::new(vm.modules.len()).expect("always at least one");
            vm.modules.push(Dynamic::new(Module {
                parent: Some(vm.modules[vm.frames[executing_frame].module].downgrade()),
                ..Module::default()
            }));
            vm.frames[vm.current_frame].module = module_index.get();
            vm.frames[executing_frame].loading_module = Some(module_index);
            let _init_result = vm.resume_async_inner(vm.current_frame)?;
            module_index
        };

        self.dest
            .store(vm, Value::Dynamic(vm.modules[loading_module.get()].clone()))?;
        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::LoadModule {
            module: self.bitcode.clone(),
            dest: self.dest.as_dest(),
        }
    }
}

#[derive(Debug)]
struct Declare<Value, Dest> {
    name: Symbol,
    mutable: bool,
    declaration: Value,
    dest: Dest,
}

impl<Value, Dest> Instruction for Declare<Value, Dest>
where
    Value: Source,
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let value = self.declaration.load(vm)?;
        vm.declare_inner(self.name.clone(), value.clone(), self.mutable)?;

        self.dest.store(vm, value)?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> Op {
        Op::Declare {
            name: self.name.clone(),
            mutable: self.mutable,
            value: self.declaration.as_source(),
            dest: self.dest.as_dest(),
        }
    }
}
