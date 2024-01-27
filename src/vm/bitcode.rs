use std::ops::{Deref, DerefMut};
use std::str;
use std::sync::Arc;

use kempt::Map;
use serde::{Deserialize, Serialize};

use super::ops::{
    Add, Call, Destination, Divide, Equal, GreaterThan, GreaterThanOrEqual, IntegerDivide, Invoke,
    Jump, JumpIf, LessThan, LessThanOrEqual, Load, Multiply, NewMap, Power, Remainder, Resolve,
    Return, Source, Subtract,
};
use super::{Arity, Code, Fault, Function, Register};
use crate::compiler::UnaryKind;
use crate::symbol::Symbol;
use crate::syntax::{BinaryKind, CompareKind};
use crate::value::Value;
use crate::vm::ops::Stack;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValueOrSource {
    Nil,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Symbol(Symbol),
    Register(Register),
    Stack(Stack),
    Label(Label),
}

impl From<()> for ValueOrSource {
    fn from(_value: ()) -> Self {
        Self::Nil
    }
}

impl_from!(ValueOrSource, i8, Int);
impl_from!(ValueOrSource, i16, Int);
impl_from!(ValueOrSource, i32, Int);
impl_from!(ValueOrSource, i64, Int);
impl_from!(ValueOrSource, u8, UInt);
impl_from!(ValueOrSource, u16, UInt);
impl_from!(ValueOrSource, u32, UInt);
impl_from!(ValueOrSource, u64, UInt);
impl_from!(ValueOrSource, f64, Float);
impl_from!(ValueOrSource, Symbol, Symbol);
impl_from!(ValueOrSource, &'_ str, Symbol);
impl_from!(ValueOrSource, Register, Register);
impl_from!(ValueOrSource, Stack, Stack);
impl_from!(ValueOrSource, Label, Label);
impl_from!(ValueOrSource, bool, Bool);

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum OpDestination {
    Void,
    Register(Register),
    Stack(Stack),
    Label(Label),
}

impl From<()> for OpDestination {
    fn from(_value: ()) -> Self {
        Self::Void
    }
}

impl_from!(OpDestination, Register, Register);
impl_from!(OpDestination, Stack, Stack);
impl_from!(OpDestination, Label, Label);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    Return,
    Label(Label),
    DeclareFunction(BitcodeFunction),
    Unary {
        op: ValueOrSource,
        dest: OpDestination,
        kind: UnaryKind,
    },
    BinOp {
        op1: ValueOrSource,
        op2: ValueOrSource,
        dest: OpDestination,
        kind: BinaryKind,
    },
    Invoke {
        target: ValueOrSource,
        arity: ValueOrSource,
        name: Symbol,
        dest: OpDestination,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Label(usize);

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

#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct BitcodeBlock {
    ops: Vec<Op>,
    pub stack_requirement: usize,
    labels: usize,
}

impl BitcodeBlock {
    pub fn new_label(&mut self) -> Label {
        let label = Label(self.labels);
        self.labels += 1;
        label
    }

    pub fn new_variable(&mut self) -> Stack {
        let var = Stack(self.stack_requirement);
        self.stack_requirement += 1;
        var
    }

    pub fn push(&mut self, op: Op) {
        self.ops.push(op);
    }

    pub fn declare_function(&mut self, function: impl Into<BitcodeFunction>) {
        self.push(Op::DeclareFunction(function.into()));
    }

    pub fn clear(&mut self) {
        self.ops.clear();
        self.labels = 0;
    }

    pub fn compare(
        &mut self,
        comparison: CompareKind,
        lhs: impl Into<ValueOrSource>,
        rhs: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.push(Op::BinOp {
            op1: lhs.into(),
            op2: rhs.into(),
            dest: dest.into(),
            kind: BinaryKind::Compare(comparison),
        });
    }

    pub fn call(
        &mut self,
        function: impl Into<ValueOrSource>,
        arity: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.push(Op::BinOp {
            op1: function.into(),
            op2: arity.into(),
            dest: dest.into(),
            kind: BinaryKind::Call,
        });
    }

    pub fn add(
        &mut self,
        lhs: impl Into<ValueOrSource>,
        rhs: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.push(Op::BinOp {
            op1: lhs.into(),
            op2: rhs.into(),
            dest: dest.into(),
            kind: BinaryKind::Add,
        });
    }

    pub fn sub(
        &mut self,
        lhs: impl Into<ValueOrSource>,
        rhs: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.push(Op::BinOp {
            op1: lhs.into(),
            op2: rhs.into(),
            dest: dest.into(),
            kind: BinaryKind::Subtract,
        });
    }

    pub fn mul(
        &mut self,
        lhs: impl Into<ValueOrSource>,
        rhs: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.push(Op::BinOp {
            op1: lhs.into(),
            op2: rhs.into(),
            dest: dest.into(),
            kind: BinaryKind::Multiply,
        });
    }

    pub fn div(
        &mut self,
        lhs: impl Into<ValueOrSource>,
        rhs: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.push(Op::BinOp {
            op1: lhs.into(),
            op2: rhs.into(),
            dest: dest.into(),
            kind: BinaryKind::Divide,
        });
    }

    pub fn idiv(
        &mut self,
        lhs: impl Into<ValueOrSource>,
        rhs: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.push(Op::BinOp {
            op1: lhs.into(),
            op2: rhs.into(),
            dest: dest.into(),
            kind: BinaryKind::IntegerDivide,
        });
    }

    pub fn rem(
        &mut self,
        lhs: impl Into<ValueOrSource>,
        rhs: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.push(Op::BinOp {
            op1: lhs.into(),
            op2: rhs.into(),
            dest: dest.into(),
            kind: BinaryKind::Remainder,
        });
    }

    pub fn pow(
        &mut self,
        lhs: impl Into<ValueOrSource>,
        rhs: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.push(Op::BinOp {
            op1: lhs.into(),
            op2: rhs.into(),
            dest: dest.into(),
            kind: BinaryKind::Power,
        });
    }

    pub fn copy(&mut self, source: impl Into<ValueOrSource>, dest: impl Into<OpDestination>) {
        self.push(Op::Unary {
            op: source.into(),
            dest: dest.into(),
            kind: UnaryKind::Copy,
        });
    }

    pub fn new_map(
        &mut self,
        element_count: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.push(Op::Unary {
            op: element_count.into(),
            dest: dest.into(),
            kind: UnaryKind::NewMap,
        });
    }

    pub fn resolve(&mut self, source: impl Into<ValueOrSource>, dest: impl Into<OpDestination>) {
        self.push(Op::Unary {
            op: source.into(),
            dest: dest.into(),
            kind: UnaryKind::Resolve,
        });
    }

    pub fn return_early(&mut self) {
        self.push(Op::Return);
    }

    pub fn jump(
        &mut self,
        target: impl Into<ValueOrSource>,
        instruction_before_jump: impl Into<OpDestination>,
    ) {
        self.push(Op::Unary {
            op: target.into(),
            dest: instruction_before_jump.into(),
            kind: UnaryKind::Jump,
        });
    }

    pub fn label(&mut self, label: Label) {
        self.push(Op::Label(label));
    }

    pub fn jump_if(
        &mut self,
        target: impl Into<ValueOrSource>,
        condition: impl Into<ValueOrSource>,
        instruction_before_jump: impl Into<OpDestination>,
    ) {
        self.push(Op::BinOp {
            op1: target.into(),
            op2: condition.into(),
            dest: instruction_before_jump.into(),
            kind: BinaryKind::JumpIf,
        });
    }
}

impl From<&'_ BitcodeBlock> for Code {
    fn from(bitcode: &'_ BitcodeBlock) -> Self {
        let mut labels = Vec::new();
        labels.resize(bitcode.labels, usize::MAX);

        let mut found_labels = 0;
        for (index, op) in bitcode.ops.iter().enumerate() {
            if let Op::Label(label) = op {
                let instruction_index = index - found_labels;
                found_labels += 1;
                if let Some(label_instruction) = labels.get_mut(label.0) {
                    *label_instruction = instruction_index;
                }
            }
        }

        let mut code = Code {
            data: Arc::new(super::CodeData {
                instructions: Vec::with_capacity(bitcode.len()),
                stack_requirement: bitcode.stack_requirement,
                labels,
            }),
        };
        for op in &bitcode.ops {
            match op {
                Op::Return => {
                    code.push(Return);
                }
                Op::Label(_) => {}
                Op::DeclareFunction(func) => code.push(Function::from(func)),
                Op::Unary {
                    op: source,
                    dest,
                    kind,
                } => match kind {
                    UnaryKind::Copy => match_copy(source, dest, &mut code),
                    UnaryKind::Resolve => match_resolve(source, dest, &mut code),
                    UnaryKind::Jump => match_jump(source, dest, &mut code),
                    UnaryKind::NewMap => match_new_map(source, dest, &mut code),
                    UnaryKind::LogicalNot => todo!(),
                    UnaryKind::BitwiseNot => todo!(),
                    UnaryKind::Negate => todo!(),
                },
                Op::BinOp {
                    op1: left,
                    op2: right,
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
                    BinaryKind::Call => match_call(left, right, dest, &mut code),
                    BinaryKind::JumpIf => match_jump_if(left, right, dest, &mut code),
                    BinaryKind::Bitwise(_) => todo!(),
                    BinaryKind::Logical(_) => todo!(),
                    BinaryKind::Compare(kind) => match kind {
                        CompareKind::LessThanOrEqual => match_lte(left, right, dest, &mut code),
                        CompareKind::LessThan => match_lt(left, right, dest, &mut code),
                        CompareKind::Equal => match_equal(left, right, dest, &mut code),
                        CompareKind::GreaterThan => match_gt(left, right, dest, &mut code),
                        CompareKind::GreaterThanOrEqual => match_gte(left, right, dest, &mut code),
                    },
                },
                Op::Invoke {
                    target,
                    arity,
                    name,
                    dest,
                } => match_invoke(target, arity, dest, &mut code, name),
            }
        }
        code
    }
}

impl From<&'_ Code> for BitcodeBlock {
    fn from(code: &'_ Code) -> Self {
        let mut ops = Vec::with_capacity(code.data.instructions.len());
        let mut labels = 0;
        for instruction in &*code.data.instructions {
            let op = instruction.as_op();
            if let Op::Label(label) = &op {
                labels = labels.max(label.0 + 1);
            }
            ops.push(op);
        }

        BitcodeBlock {
            ops,
            stack_requirement: code.data.stack_requirement,
            labels,
        }
    }
}

impl Deref for BitcodeBlock {
    type Target = Vec<Op>;

    fn deref(&self) -> &Self::Target {
        &self.ops
    }
}

impl DerefMut for BitcodeBlock {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ops
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
            ValueOrSource::Symbol(source) => $next_fn($source, $code $(, $($arg)*)?, source.clone()),
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

decode_sd!(match_resolve, compile_resolve);

fn compile_resolve<From, Dest>(_dest: &OpDestination, code: &mut Code, source: From, dest: Dest)
where
    From: Source,
    Dest: Destination,
{
    code.push(Resolve { source, dest });
}

decode_sd!(match_jump, compile_jump);

fn compile_jump<From, Dest>(_dest: &OpDestination, code: &mut Code, source: From, dest: Dest)
where
    From: Source,
    Dest: Destination,
{
    code.push(Jump {
        target: source,
        previous_address: dest,
    });
}

decode_sd!(match_new_map, compile_new_map);

fn compile_new_map<Arity, Dest>(
    _dest: &OpDestination,
    code: &mut Code,
    element_count: Arity,
    dest: Dest,
) where
    Arity: Source,
    Dest: Destination,
{
    code.push(NewMap {
        element_count,
        dest,
    });
}

macro_rules! decode_ssd {
    ($decode_name:ident, $compile_name:ident $(, $($name:ident: $type:ty),+)?) => {
        fn $decode_name(
            s1: &ValueOrSource,
            s2: &ValueOrSource,
            d: &OpDestination,
            code: &mut Code,
            $($($name: $type),+,)?
        ) {
            fn source<Lhs>(source: (&ValueOrSource, &OpDestination), code: &mut Code,
            $($($name: $type),+,)? source1: Lhs)
            where
                Lhs: Source,
            {
                decode_source!(source.0, source, code, source_source, source1 $(, $($name),+)?)
            }

            fn source_source<Lhs, Rhs>(
                source: (&ValueOrSource, &OpDestination),
                code: &mut Code,
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

decode_ssd!(match_call, compile_call);

fn compile_call<Func, NumArgs, Dest>(
    _source: (&ValueOrSource, &OpDestination),
    code: &mut Code,
    function: Func,
    arity: NumArgs,
    dest: Dest,
) where
    Func: Source,
    NumArgs: Source,
    Dest: Destination,
{
    code.push(Call {
        function,
        arity,
        dest,
    });
}

decode_ssd!(match_invoke, compile_invoke, name: &Symbol);

fn compile_invoke<Func, NumArgs, Dest>(
    _source: (&ValueOrSource, &OpDestination),
    code: &mut Code,
    function: Func,
    arity: NumArgs,
    name: &Symbol,
    dest: Dest,
) where
    Func: Source,
    NumArgs: Source,
    Dest: Destination,
{
    code.push(Invoke {
        name: name.clone(),
        target: function,
        arity,
        dest,
    });
}

decode_ssd!(match_jump_if, compile_jump_if);

fn compile_jump_if<Func, NumArgs, Dest>(
    _source: (&ValueOrSource, &OpDestination),
    code: &mut Code,
    target: Func,
    condition: NumArgs,
    previous_address: Dest,
) where
    Func: Source,
    NumArgs: Source,
    Dest: Destination,
{
    code.push(JumpIf {
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
define_match_binop!(match_lte, compile_lte, LessThanOrEqual);
define_match_binop!(match_lt, compile_lt, LessThan);
define_match_binop!(match_equal, compile_equal, Equal);
define_match_binop!(match_gt, compile_gt, GreaterThan);
define_match_binop!(match_gte, compile_gte, GreaterThanOrEqual);

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BitcodeFunction {
    name: Symbol,
    bodies: Map<Arity, BitcodeBlock>,
    varg_bodies: Map<Arity, BitcodeBlock>,
}

impl BitcodeFunction {
    pub fn new(name: impl Into<Symbol>) -> Self {
        Self {
            name: name.into(),
            bodies: Map::new(),
            varg_bodies: Map::new(),
        }
    }

    pub fn insert_arity(&mut self, arity: impl Into<Arity>, body: impl Into<BitcodeBlock>) {
        self.bodies.insert(arity.into(), body.into());
    }

    #[must_use]
    pub fn when(mut self, arity: impl Into<Arity>, body: impl Into<BitcodeBlock>) -> Self {
        self.insert_arity(arity, body);
        self
    }

    #[must_use]
    pub const fn name(&self) -> &Symbol {
        &self.name
    }
}

impl From<&'_ BitcodeFunction> for Function {
    fn from(bit: &'_ BitcodeFunction) -> Self {
        Self {
            name: bit.name.clone(),
            bodies: bit
                .bodies
                .iter()
                .map(|f| (*f.key(), Code::from(&f.value)))
                .collect(),
            varg_bodies: bit
                .varg_bodies
                .iter()
                .map(|f| (*f.key(), Code::from(&f.value)))
                .collect(),
        }
    }
}

impl From<&'_ Function> for BitcodeFunction {
    fn from(bit: &'_ Function) -> Self {
        Self {
            name: bit.name.clone(),
            bodies: bit
                .bodies
                .iter()
                .map(|f| (*f.key(), BitcodeBlock::from(&f.value)))
                .collect(),
            varg_bodies: bit
                .bodies
                .iter()
                .map(|f| (*f.key(), BitcodeBlock::from(&f.value)))
                .collect(),
        }
    }
}
