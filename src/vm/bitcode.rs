use std::ops::{Deref, DerefMut};
use std::str;

use kempt::Map;
use refuse::CollectionGuard;
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "dispatched"))]
use super::LoadedOp;
use super::{Arity, Code, CodeData, Function, LoadedSource, Register};
use crate::compiler::{BitcodeModule, SourceMap, UnaryKind};
use crate::symbol::Symbol;
use crate::syntax::token::RegexLiteral;
use crate::syntax::{BitwiseKind, CompareKind, Literal, SourceRange};
use crate::vm::Stack;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValueOrSource {
    Nil,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Symbol(Symbol),
    Regex(RegexLiteral),
    Register(Register),
    Function(BitcodeFunction),
    Stack(Stack),
    Label(Label),
}

impl From<()> for ValueOrSource {
    fn from(_value: ()) -> Self {
        Self::Nil
    }
}

impl TryFrom<Literal> for ValueOrSource {
    type Error = Symbol;

    fn try_from(value: Literal) -> Result<Self, Self::Error> {
        match value {
            Literal::Nil => Ok(ValueOrSource::Nil),
            Literal::Bool(value) => Ok(ValueOrSource::Bool(value)),
            Literal::Int(value) => Ok(ValueOrSource::Int(value)),
            Literal::UInt(value) => Ok(ValueOrSource::UInt(value)),
            Literal::Float(value) => Ok(ValueOrSource::Float(value)),
            Literal::String(value) => Err(value),
            Literal::Symbol(value) => Ok(ValueOrSource::Symbol(value)),
            Literal::Regex(value) => Ok(ValueOrSource::Regex(value)),
        }
    }
}

impl From<OpDestination> for ValueOrSource {
    fn from(value: OpDestination) -> Self {
        match value {
            OpDestination::Void => Self::Nil,
            OpDestination::Register(dest) => Self::Register(dest),
            OpDestination::Stack(dest) => Self::Stack(dest),
            OpDestination::Label(dest) => Self::Label(dest),
        }
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
// impl_from!(ValueOrSource, &'_ str, Symbol);
impl_from!(ValueOrSource, Register, Register);
impl_from!(ValueOrSource, Stack, Stack);
impl_from!(ValueOrSource, Label, Label);
impl_from!(ValueOrSource, bool, Bool);
impl_from!(ValueOrSource, BitcodeFunction, Function);
impl_from!(ValueOrSource, RegexLiteral, Regex);

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
    LoadModule {
        module: BitcodeModule,
        dest: OpDestination,
    },
    Declare {
        name: Symbol,
        mutable: bool,
        value: ValueOrSource,
        dest: OpDestination,
    },
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
    Call {
        name: ValueOrSource,
        arity: ValueOrSource,
    },
    Invoke {
        target: ValueOrSource,
        name: Symbol,
        arity: ValueOrSource,
    },
    Throw(FaultKind),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, Hash)]
pub enum FaultKind {
    Exception,
    PatternMismatch,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Label(pub(crate) usize);

#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct BitcodeBlock {
    ops: Vec<Op>,
    pub stack_requirement: usize,
    labels: usize,
    current_location: SourceRange,
    map: SourceMap,
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

    pub fn set_current_source_range(&mut self, range: SourceRange) {
        self.current_location = range;
    }

    pub fn push(&mut self, op: Op) {
        self.ops.push(op);
        self.map.push(self.current_location);
    }

    pub fn declare(
        &mut self,
        name: Symbol,
        mutable: bool,
        value: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        let value = value.into();
        self.push(Op::Declare {
            name,
            mutable,
            value,
            dest: dest.into(),
        });
    }

    pub fn clear(&mut self) {
        self.ops.clear();
        self.map = SourceMap::default();
        self.current_location = SourceRange::default();
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

    pub fn matches(
        &mut self,
        lhs: impl Into<ValueOrSource>,
        rhs: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.push(Op::BinOp {
            op1: lhs.into(),
            op2: rhs.into(),
            dest: dest.into(),
            kind: BinaryKind::Matches,
        });
    }

    pub fn call(&mut self, function: impl Into<ValueOrSource>, arity: impl Into<ValueOrSource>) {
        self.push(Op::Call {
            name: function.into(),
            arity: arity.into(),
        });
    }

    pub fn invoke(
        &mut self,
        target: impl Into<ValueOrSource>,
        name: impl Into<Symbol>,
        arity: impl Into<ValueOrSource>,
    ) {
        self.push(Op::Invoke {
            target: target.into(),
            name: name.into(),
            arity: arity.into(),
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

    pub fn load_module(&mut self, module: BitcodeModule, dest: impl Into<OpDestination>) {
        self.push(Op::LoadModule {
            module,
            dest: dest.into(),
        });
    }

    pub fn copy(&mut self, source: impl Into<ValueOrSource>, dest: impl Into<OpDestination>) {
        self.push(Op::Unary {
            op: source.into(),
            dest: dest.into(),
            kind: UnaryKind::Copy,
        });
    }

    pub fn throw(&mut self, kind: FaultKind) {
        self.push(Op::Throw(kind));
    }

    pub fn set_exception_handler(
        &mut self,
        target: impl Into<ValueOrSource>,
        previous_handler: impl Into<OpDestination>,
    ) {
        self.push(Op::Unary {
            op: target.into(),
            dest: previous_handler.into(),
            kind: UnaryKind::SetExceptionHandler,
        });
    }

    pub fn new_map(
        &mut self,
        element_count: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.call(Symbol::from("$.core.Map"), element_count);
        self.copy(Register(0), dest);
    }

    pub fn new_list(
        &mut self,
        element_count: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.call(Symbol::from("$.core.List"), element_count);
        self.copy(Register(0), dest);
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

    pub fn assign(
        &mut self,
        target: impl Into<ValueOrSource>,
        value: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.push(Op::BinOp {
            op1: target.into(),
            op2: value.into(),
            dest: dest.into(),
            kind: BinaryKind::Assign,
        });
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

    pub fn jump_if_not(
        &mut self,
        target: impl Into<ValueOrSource>,
        condition: impl Into<ValueOrSource>,
        instruction_before_jump: impl Into<OpDestination>,
    ) {
        self.push(Op::BinOp {
            op1: target.into(),
            op2: condition.into(),
            dest: instruction_before_jump.into(),
            kind: BinaryKind::JumpIfNot,
        });
    }

    pub fn not(&mut self, value: impl Into<ValueOrSource>, dest: impl Into<OpDestination>) {
        self.push(Op::Unary {
            op: value.into(),
            dest: dest.into(),
            kind: UnaryKind::LogicalNot,
        });
    }

    pub fn truthy(&mut self, value: impl Into<ValueOrSource>, dest: impl Into<OpDestination>) {
        self.push(Op::Unary {
            op: value.into(),
            dest: dest.into(),
            kind: UnaryKind::Truthy,
        });
    }

    pub fn bitwise_not(&mut self, value: impl Into<ValueOrSource>, dest: impl Into<OpDestination>) {
        self.push(Op::Unary {
            op: value.into(),
            dest: dest.into(),
            kind: UnaryKind::BitwiseNot,
        });
    }

    #[must_use]
    pub fn to_code(&self, guard: &CollectionGuard) -> Code {
        let mut code = Code::default();
        let mut previous_location = SourceRange::default();
        for (index, op) in self.ops.iter().enumerate() {
            let location = if let Some(new_location) = self.map.get(index) {
                previous_location = new_location;
                new_location
            } else {
                previous_location
            };
            code.push(op, location, guard);
        }
        code
    }
}

pub(super) fn trusted_loaded_source_to_value(
    loaded: &LoadedSource,
    code: &CodeData,
) -> ValueOrSource {
    match loaded {
        LoadedSource::Nil => ValueOrSource::Nil,
        LoadedSource::Bool(loaded) => ValueOrSource::Bool(*loaded),
        LoadedSource::Int(loaded) => ValueOrSource::Int(*loaded),
        LoadedSource::UInt(loaded) => ValueOrSource::UInt(*loaded),
        LoadedSource::Float(loaded) => ValueOrSource::Float(*loaded),
        LoadedSource::Symbol(loaded) => ValueOrSource::Symbol(code.symbols[*loaded].clone()),
        LoadedSource::Register(loaded) => ValueOrSource::Register(*loaded),
        LoadedSource::Stack(loaded) => ValueOrSource::Stack(*loaded),
        LoadedSource::Label(loaded) => ValueOrSource::Label(*loaded),
        LoadedSource::Regex(loaded) => ValueOrSource::Regex(code.regexes[*loaded].literal.clone()),
        LoadedSource::Function(loaded) => ValueOrSource::Function(code.functions[*loaded].clone()),
    }
}

impl From<&'_ Code> for BitcodeBlock {
    #[allow(clippy::too_many_lines)]
    fn from(code: &'_ Code) -> Self {
        let mut ops = Vec::with_capacity(code.data.instructions.len() + code.data.labels.len());
        let mut label_addrs = code.data.labels.iter().copied().peekable();
        let mut labels = 0;
        let mut map = SourceMap::default();
        let mut current_location = map.get(0).unwrap_or_default();

        for (index, instruction) in code.data.instructions.iter().enumerate() {
            if label_addrs.peek().map_or(false, |label| label == &index) {
                label_addrs.next();
                let label = Label(labels);
                labels += 1;
                map.push(current_location);
                ops.push(Op::Label(label));
            }

            current_location = code.data.map.get(index).unwrap_or_default();
            map.push(current_location);

            #[cfg(feature = "dispatched")]
            ops.push(instruction.as_op());

            #[cfg(not(feature = "dispatched"))]
            ops.push(match instruction {
                LoadedOp::Return => Op::Return,
                LoadedOp::Declare {
                    name,
                    mutable,
                    value,
                    dest,
                } => Op::Declare {
                    name: code.data.symbols[*name].clone(),
                    mutable: *mutable,
                    value: trusted_loaded_source_to_value(value, &code.data),
                    dest: *dest,
                },
                LoadedOp::Truthy(loaded) => loaded.as_op(UnaryKind::Truthy, code),
                LoadedOp::LogicalNot(loaded) => loaded.as_op(UnaryKind::LogicalNot, code),
                LoadedOp::BitwiseNot(loaded) => loaded.as_op(UnaryKind::BitwiseNot, code),
                LoadedOp::Negate(loaded) => loaded.as_op(UnaryKind::Negate, code),
                LoadedOp::Copy(loaded) => loaded.as_op(UnaryKind::Copy, code),
                LoadedOp::Resolve(loaded) => loaded.as_op(UnaryKind::Resolve, code),
                LoadedOp::Jump(loaded) => loaded.as_op(UnaryKind::Jump, code),
                LoadedOp::SetExceptionHandler(loaded) => {
                    loaded.as_op(UnaryKind::SetExceptionHandler, code)
                }
                LoadedOp::LogicalXor(loaded) => loaded.as_op(BinaryKind::LogicalXor, code),
                LoadedOp::Assign(loaded) => loaded.as_op(BinaryKind::Assign, code),
                LoadedOp::Add(loaded) => loaded.as_op(BinaryKind::Add, code),
                LoadedOp::Subtract(loaded) => loaded.as_op(BinaryKind::Subtract, code),
                LoadedOp::Multiply(loaded) => loaded.as_op(BinaryKind::Multiply, code),
                LoadedOp::Divide(loaded) => loaded.as_op(BinaryKind::Divide, code),
                LoadedOp::IntegerDivide(loaded) => loaded.as_op(BinaryKind::IntegerDivide, code),
                LoadedOp::Remainder(loaded) => loaded.as_op(BinaryKind::Remainder, code),
                LoadedOp::Power(loaded) => loaded.as_op(BinaryKind::Power, code),
                LoadedOp::JumpIf(loaded) => loaded.as_op(BinaryKind::JumpIf, code),
                LoadedOp::JumpIfNot(loaded) => loaded.as_op(BinaryKind::JumpIfNot, code),
                LoadedOp::LessThanOrEqual(loaded) => {
                    loaded.as_op(BinaryKind::Compare(CompareKind::LessThanOrEqual), code)
                }
                LoadedOp::LessThan(loaded) => {
                    loaded.as_op(BinaryKind::Compare(CompareKind::LessThan), code)
                }
                LoadedOp::Equal(loaded) => {
                    loaded.as_op(BinaryKind::Compare(CompareKind::Equal), code)
                }
                LoadedOp::NotEqual(loaded) => {
                    loaded.as_op(BinaryKind::Compare(CompareKind::NotEqual), code)
                }
                LoadedOp::GreaterThan(loaded) => {
                    loaded.as_op(BinaryKind::Compare(CompareKind::GreaterThan), code)
                }
                LoadedOp::GreaterThanOrEqual(loaded) => {
                    loaded.as_op(BinaryKind::Compare(CompareKind::GreaterThanOrEqual), code)
                }
                LoadedOp::Matches(loaded) => loaded.as_op(BinaryKind::Matches, code),
                LoadedOp::Call { name, arity } => Op::Call {
                    name: trusted_loaded_source_to_value(name, &code.data),
                    arity: trusted_loaded_source_to_value(arity, &code.data),
                },
                LoadedOp::Invoke {
                    target,
                    name,
                    arity,
                } => Op::Invoke {
                    target: trusted_loaded_source_to_value(target, &code.data),
                    name: code.data.symbols[*name].clone(),
                    arity: trusted_loaded_source_to_value(arity, &code.data),
                },
                LoadedOp::BitwiseAnd(loaded) => {
                    loaded.as_op(BinaryKind::Bitwise(BitwiseKind::And), code)
                }
                LoadedOp::BitwiseOr(loaded) => {
                    loaded.as_op(BinaryKind::Bitwise(BitwiseKind::Or), code)
                }
                LoadedOp::BitwiseXor(loaded) => {
                    loaded.as_op(BinaryKind::Bitwise(BitwiseKind::Xor), code)
                }
                LoadedOp::BitwiseShiftLeft(loaded) => {
                    loaded.as_op(BinaryKind::Bitwise(BitwiseKind::ShiftLeft), code)
                }
                LoadedOp::BitwiseShiftRight(loaded) => {
                    loaded.as_op(BinaryKind::Bitwise(BitwiseKind::ShiftRight), code)
                }
                LoadedOp::LoadModule {
                    module: initializer,
                    dest,
                } => Op::LoadModule {
                    module: code.data.modules[*initializer].clone(),
                    dest: *dest,
                },
                LoadedOp::Throw(kind) => Op::Throw(*kind),
            });
        }

        BitcodeBlock {
            ops,
            stack_requirement: code.data.stack_requirement,
            labels,
            current_location: SourceRange::default(),
            map,
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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BitcodeFunction {
    name: Option<Symbol>,
    bodies: Map<Arity, BitcodeBlock>,
    varg_bodies: Map<Arity, BitcodeBlock>,
}

impl BitcodeFunction {
    pub fn new(name: impl Into<Option<Symbol>>) -> Self {
        Self {
            name: name.into(),
            bodies: Map::new(),
            varg_bodies: Map::new(),
        }
    }

    pub fn insert_arity(&mut self, arity: impl Into<Arity>, body: impl Into<BitcodeBlock>) {
        self.bodies.insert(arity.into(), body.into());
    }

    pub fn insert_variable_arity(
        &mut self,
        arity: impl Into<Arity>,
        body: impl Into<BitcodeBlock>,
    ) {
        self.varg_bodies.insert(arity.into(), body.into());
    }

    #[must_use]
    pub fn when(mut self, arity: impl Into<Arity>, body: impl Into<BitcodeBlock>) -> Self {
        self.insert_arity(arity, body);
        self
    }

    #[must_use]
    pub const fn name(&self) -> &Option<Symbol> {
        &self.name
    }

    #[must_use]
    pub fn to_function(&self, guard: &CollectionGuard<'_>) -> Function {
        Function {
            module: None,
            name: self.name.clone(),
            bodies: self
                .bodies
                .iter()
                .map(|f| (*f.key(), f.value.to_code(guard)))
                .collect(),
            varg_bodies: self
                .varg_bodies
                .iter()
                .map(|f| (*f.key(), f.value.to_code(guard)))
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BinaryKind {
    Add,
    Subtract,
    Multiply,
    Divide,
    IntegerDivide,
    Remainder,
    Power,
    JumpIf,
    JumpIfNot,
    LogicalXor,
    Assign,
    Matches,
    Bitwise(BitwiseKind),
    Compare(CompareKind),
}
