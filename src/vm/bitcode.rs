//! The Muse intermediate representation (IR).

use std::ops::{Deref, DerefMut};
use std::str;

use kempt::Map;
use refuse::CollectionGuard;
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "dispatched"))]
use super::LoadedOp;
use super::{Arity, Code, CodeData, Function, LoadedSource, Register};
use crate::compiler::syntax::token::RegexLiteral;
use crate::compiler::syntax::{BitwiseKind, CompareKind, Literal, SourceRange};
use crate::compiler::{BitcodeModule, SourceMap, UnaryKind};
use crate::runtime::symbol::{IntoOptionSymbol, Symbol};
use crate::vm::Stack;

/// A value or a source of a value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValueOrSource {
    /// [`Value::Nil`](crate::runtime::value::Value::Nil)
    Nil,
    /// [`Value::Nil`](crate::runtime::value::Value::Bool)
    Bool(bool),
    /// [`Value::Nil`](crate::runtime::value::Value::Int)
    Int(i64),
    /// [`Value::Nil`](crate::runtime::value::Value::UInt)
    UInt(u64),
    /// [`Value::Nil`](crate::runtime::value::Value::Float)
    Float(f64),
    /// [`Value::Nil`](crate::runtime::value::Value::Symbol)
    Symbol(Symbol),
    /// A regular expression literal. When loaded, it is compiled into a
    /// [`MuseRegex`](crate::runtime::regex::MuseRegex).
    Regex(RegexLiteral),
    /// A function declaration. When loaded, it becomes a [`Function`].
    Function(BitcodeFunction),
    /// A virtual machine register.
    Register(Register),
    /// A location on the stack.
    Stack(Stack),
    /// A label representing an instruction offset.
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
impl_from!(ValueOrSource, Register, Register);
impl_from!(ValueOrSource, Stack, Stack);
impl_from!(ValueOrSource, Label, Label);
impl_from!(ValueOrSource, bool, Bool);
impl_from!(ValueOrSource, BitcodeFunction, Function);
impl_from!(ValueOrSource, RegexLiteral, Regex);

/// A destination for an operation.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum OpDestination {
    /// Ignore the result of the operation.
    Void,
    /// Store the result in the provided register.
    Register(Register),
    /// Store the result at a given stack offset.
    Stack(Stack),
    /// Jump to the label if the value is truthy.
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

/// The level of access of a member.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Ord, PartialOrd)]
pub enum Access {
    /// The member is accessible by any code.
    Private,
    /// The member is only accessible to code in the same module.
    Public,
}

/// A virtual machine operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Op {
    /// Return from the current function.
    Return,
    /// Assign this label to the next instruction.
    Label(Label),
    /// Load `module`, storing a reference into `dest`.
    LoadModule {
        /// The module to load.
        module: BitcodeModule,
        /// The location to store the loaded module.
        dest: OpDestination,
    },
    /// Declare `name` with `value`, storing a copy of the value in `dest`.
    Declare {
        /// The name of the declaration.
        name: Symbol,
        /// If true, the value will be able to be updated with an assignment.
        mutable: bool,
        /// The access level to allow for this declaration.
        access: Access,
        /// The initial value of the declaration.
        value: ValueOrSource,
        /// The destination to store a copy of `value`.
        dest: OpDestination,
    },
    /// An operation with one argument that stores its result in `dest`.
    Unary {
        /// The value to operate on.
        op: ValueOrSource,
        /// The destination for the result.
        dest: OpDestination,
        /// The operation kind.
        kind: UnaryKind,
    },
    /// An operation with two arguments that stores its result in `dest`.
    BinOp {
        /// The first value to operate on.
        op1: ValueOrSource,
        /// The second value to operate on.
        op2: ValueOrSource,
        /// The destination for the result.
        dest: OpDestination,
        /// The operation kind.
        kind: BinaryKind,
    },
    /// Invoke `name` as a function with `arity` number of arguments provided.
    Call {
        /// The name of the function.
        name: ValueOrSource,
        /// The number of arguments provided to this invocation.
        arity: ValueOrSource,
    },
    /// Invoke `name` on `target` with `arity` number of arguments provided.
    Invoke {
        /// The target to invoke the function on.
        target: ValueOrSource,
        /// The name of the function.
        name: Symbol,
        /// The number of arguments provided to this invocation.
        arity: ValueOrSource,
    },
    /// Throw an exception.
    Throw(FaultKind),
}

/// An IR [`Fault`](crate::vm::Fault).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize, Hash)]
pub enum FaultKind {
    /// Create an exception from the contents of register 0.
    Exception,
    /// Return a [`Fault::PatternMismatch`](crate::vm::Fault::PatternMismatch).
    PatternMismatch,
}

/// A tag that can be converted to the instruction address of an [`Op::Label`]
/// instruction.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Label(pub(crate) usize);

/// A collection of [`Op`]s that represent a block of code that can be executed.
#[derive(Default, Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct BitcodeBlock {
    ops: Vec<Op>,
    /// The amount of stack space this code needs to have allocated to execute.
    pub stack_requirement: usize,
    labels: usize,
    current_location: SourceRange,
    map: SourceMap,
}

impl BitcodeBlock {
    /// Returns an IR code block from a runtime code block.
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn from_code(code: &'_ Code, guard: &CollectionGuard<'_>) -> Self {
        #[cfg(not(feature = "dispatched"))]
        let _ = guard;

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
            ops.push(instruction.as_op(guard));

            #[cfg(not(feature = "dispatched"))]
            ops.push(match instruction {
                LoadedOp::Return => Op::Return,
                LoadedOp::Declare {
                    name,
                    mutable,
                    access,
                    value,
                    dest,
                } => Op::Declare {
                    name: code.data.symbols[*name].clone(),
                    mutable: *mutable,
                    access: *access,
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

    /// Returns a newly allocated [`Label`].
    ///
    /// This value must be used in an [`Op::Label`], otherwise an error will
    /// occur when used.
    pub fn new_label(&mut self) -> Label {
        let label = Label(self.labels);
        self.labels += 1;
        label
    }

    /// Returns a newly allocated stack address.
    pub fn new_variable(&mut self) -> Stack {
        let var = Stack(self.stack_requirement);
        self.stack_requirement += 1;
        var
    }

    /// Applies `range` to all operations after this call.
    ///
    /// The range information is compiled into a [`SourceMap`], enabling
    /// exception backtraces to be traced to the original source code locations.
    pub fn set_current_source_range(&mut self, range: SourceRange) {
        self.current_location = range;
    }

    /// Pushes a new operation to this block.
    ///
    /// Most operations have helpers that make it easier to push instructions
    /// with various automatic datatype conversions.
    pub fn push(&mut self, op: Op) {
        self.ops.push(op);
        self.map.push(self.current_location);
    }

    /// Empties this code block.
    pub fn clear(&mut self) {
        self.ops.clear();
        self.map = SourceMap::default();
        self.current_location = SourceRange::default();
        self.labels = 0;
    }

    /// Pushes an [`Op::Declare`] operation.
    pub fn declare(
        &mut self,
        name: Symbol,
        mutable: bool,
        access: Access,
        value: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        let value = value.into();
        self.push(Op::Declare {
            name,
            mutable,
            access,
            value,
            dest: dest.into(),
        });
    }

    /// Pushes an [`Op::BinOp`] operation with kind [`BinaryKind::Compare`].
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

    /// Pushes an [`Op::BinOp`] operation with kind [`BinaryKind::Matches`].
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

    /// Pushes an [`Op::Call`] operation.
    pub fn call(&mut self, function: impl Into<ValueOrSource>, arity: impl Into<ValueOrSource>) {
        self.push(Op::Call {
            name: function.into(),
            arity: arity.into(),
        });
    }

    /// Pushes an [`Op::Invoke`] operation.
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

    /// Pushes an [`Op::BinOp`] operation with kind [`BinaryKind::Add`].
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

    /// Pushes an [`Op::BinOp`] operation with kind [`BinaryKind::Subtract`].
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

    /// Pushes an [`Op::BinOp`] operation with kind [`BinaryKind::Multiply`].
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

    /// Pushes an [`Op::BinOp`] operation with kind [`BinaryKind::Divide`].
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

    /// Pushes an [`Op::BinOp`] operation with kind [`BinaryKind::IntegerDivide`].
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

    /// Pushes an [`Op::BinOp`] operation with kind [`BinaryKind::Remainder`].
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

    /// Pushes an [`Op::BinOp`] operation with kind [`BinaryKind::Power`].
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

    /// Pushes an [`Op::LoadModule`] operation.
    pub fn load_module(&mut self, module: BitcodeModule, dest: impl Into<OpDestination>) {
        self.push(Op::LoadModule {
            module,
            dest: dest.into(),
        });
    }

    /// Pushes an [`Op::Unary`] operation with kind [`UnaryKind::Copy`].
    pub fn copy(&mut self, source: impl Into<ValueOrSource>, dest: impl Into<OpDestination>) {
        self.push(Op::Unary {
            op: source.into(),
            dest: dest.into(),
            kind: UnaryKind::Copy,
        });
    }

    /// Pushes an [`Op::Throw`] operation.
    pub fn throw(&mut self, kind: FaultKind) {
        self.push(Op::Throw(kind));
    }

    /// Pushes an [`Op::Unary`] operation with kind [`UnaryKind::SetExceptionHandler`].
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

    /// Initializes a new map with `element_count` elements.
    ///
    /// This pushes two instructions:
    ///
    /// - [`Op::Call`] with name `$.core.Map`
    /// - A copy from register 0 to `dest`.
    pub fn new_map(
        &mut self,
        element_count: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.call(Symbol::from("$.core.Map"), element_count);
        self.copy(Register(0), dest);
    }

    /// Initializes a new list with `element_count` elements.
    ///
    /// This pushes two instructions:
    ///
    /// - [`Op::Call`] with name `$.core.List`
    /// - A copy from register 0 to `dest`.
    pub fn new_list(
        &mut self,
        element_count: impl Into<ValueOrSource>,
        dest: impl Into<OpDestination>,
    ) {
        self.call(Symbol::from("$.core.List"), element_count);
        self.copy(Register(0), dest);
    }

    /// Pushes an [`Op::Unary`] operation with kind [`UnaryKind::Resolve`].
    pub fn resolve(&mut self, source: impl Into<ValueOrSource>, dest: impl Into<OpDestination>) {
        self.push(Op::Unary {
            op: source.into(),
            dest: dest.into(),
            kind: UnaryKind::Resolve,
        });
    }

    /// Pushes an [`Op::Return`] operation.
    pub fn return_early(&mut self) {
        self.push(Op::Return);
    }

    /// Pushes an [`Op::Unary`] operation with kind [`UnaryKind::Jump`].
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

    /// Pushes an [`Op::Label`] operation.
    pub fn label(&mut self, label: Label) {
        self.push(Op::Label(label));
    }

    /// Pushes an [`Op::BinOp`] operation with kind [`BinaryKind::Assign`].
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

    /// Pushes an [`Op::BinOp`] operation with kind [`BinaryKind::JumpIf`].
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

    /// Pushes an [`Op::BinOp`] operation with kind [`BinaryKind::JumpIfNot`].
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

    /// Pushes an [`Op::Unary`] operation with kind [`UnaryKind::LogicalNot`].
    pub fn not(&mut self, value: impl Into<ValueOrSource>, dest: impl Into<OpDestination>) {
        self.push(Op::Unary {
            op: value.into(),
            dest: dest.into(),
            kind: UnaryKind::LogicalNot,
        });
    }

    /// Pushes an [`Op::Unary`] operation with kind [`UnaryKind::Truthy`].
    pub fn truthy(&mut self, value: impl Into<ValueOrSource>, dest: impl Into<OpDestination>) {
        self.push(Op::Unary {
            op: value.into(),
            dest: dest.into(),
            kind: UnaryKind::Truthy,
        });
    }

    /// Pushes an [`Op::Unary`] operation with kind [`UnaryKind::BitwiseNot`].
    pub fn bitwise_not(&mut self, value: impl Into<ValueOrSource>, dest: impl Into<OpDestination>) {
        self.push(Op::Unary {
            op: value.into(),
            dest: dest.into(),
            kind: UnaryKind::BitwiseNot,
        });
    }

    /// Prepare this code block for executing in the virtual machine.
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

/// An IR [`Function`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BitcodeFunction {
    name: Option<Symbol>,
    bodies: Map<Arity, BitcodeBlock>,
    varg_bodies: Map<Arity, BitcodeBlock>,
}

impl BitcodeFunction {
    /// Returns a new function with `name`.
    pub fn new(name: impl IntoOptionSymbol) -> Self {
        Self {
            name: name.into_symbol(),
            bodies: Map::new(),
            varg_bodies: Map::new(),
        }
    }

    /// Returns a function from the contents of `bit`.
    #[must_use]
    pub fn from_function(bit: &'_ Function, guard: &CollectionGuard<'_>) -> Self {
        Self {
            name: bit.name.clone(),
            bodies: bit
                .bodies
                .iter()
                .map(|f| (*f.key(), BitcodeBlock::from_code(&f.value, guard)))
                .collect(),
            varg_bodies: bit
                .bodies
                .iter()
                .map(|f| (*f.key(), BitcodeBlock::from_code(&f.value, guard)))
                .collect(),
        }
    }

    /// Inserts a new function body to be executed when `arity` number of
    /// arguments are provided.
    pub fn insert_arity(&mut self, arity: impl Into<Arity>, body: impl Into<BitcodeBlock>) {
        self.bodies.insert(arity.into(), body.into());
    }

    /// Inserts a new function body to be executed when `arity` or more number
    /// of arguments are provided.
    pub fn insert_variable_arity(
        &mut self,
        arity: impl Into<Arity>,
        body: impl Into<BitcodeBlock>,
    ) {
        self.varg_bodies.insert(arity.into(), body.into());
    }

    /// Adds a new function body to be executed when `arity` number of arguments
    /// are provided, and returns self.
    #[must_use]
    pub fn when(mut self, arity: impl Into<Arity>, body: impl Into<BitcodeBlock>) -> Self {
        self.insert_arity(arity, body);
        self
    }

    /// Returns the name of this function.
    #[must_use]
    pub const fn name(&self) -> &Option<Symbol> {
        &self.name
    }

    /// Loads this function for execution in the virtual machine.
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

/// An IR binary (two-argument) operation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BinaryKind {
    /// `op1 + op2`
    Add,
    /// `op1 - op2`
    Subtract,
    /// `op1 * op2`
    Multiply,
    /// `op1 / op2`
    Divide,
    /// `op1 // op2`
    IntegerDivide,
    /// `op1 % op2`
    Remainder,
    /// `op1 ** op2`
    Power,
    /// If `op1` is truthy, jump to `op2`.
    JumpIf,
    /// If `op1` is falsey, jump to `op2`.
    JumpIfNot,
    /// `op1 xor op2`
    LogicalXor,
    /// `op1 = op2`
    Assign,
    /// True if `op1` matches `op2`
    Matches,
    /// Perform a binary bitwise operation.
    Bitwise(BitwiseKind),
    /// The result of comparing the two arguments.
    Compare(CompareKind),
}
