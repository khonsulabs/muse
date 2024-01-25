use std::array;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{ControlFlow, Index, IndexMut};
use std::sync::{Arc, OnceLock};

use kempt::Map;
use ops::{Destination, Instruction, Source};
use serde::{Deserialize, Serialize};

use self::bitcode::{BitcodeFunction, OpDestination, ValueOrSource};
use self::ops::Stack;
use crate::symbol::Symbol;
use crate::value::{CustomType, Dynamic, Value};

pub mod bitcode;
pub mod ops;

pub struct Vm {
    registers: [Value; 16],
    stack: Vec<Value>,
    max_stack: usize,
    frames: Vec<Frame>,
    current_frame: usize,
    max_depth: usize,
    module: Module,
}

impl Default for Vm {
    fn default() -> Self {
        Self {
            registers: array::from_fn(|_| Value::Nil),
            frames: vec![Frame::default()],
            stack: Vec::new(),
            current_frame: 0,
            max_stack: usize::MAX,
            max_depth: usize::MAX,
            module: Module::default(),
        }
    }
}

impl Vm {
    pub fn execute(&mut self, code: &Code, owner: Option<Value>) -> Result<Value, Fault> {
        self.frames[self.current_frame].code = Some(code.clone());
        self.frames[self.current_frame].code_owner = owner.unwrap_or_default();
        self.frames[self.current_frame].instruction = 0;
        self.execute_current_frame()
    }

    fn execute_current_frame(&mut self) -> Result<Value, Fault> {
        let mut code = self.frames[self.current_frame]
            .code
            .clone()
            .expect("missing frame code");
        self.allocate(code.data.stack_requirement)?;
        let base_frame = self.current_frame;
        let mut code_frame = self.current_frame;
        loop {
            match code.step(self, self.frames[code_frame].instruction)? {
                StepResult::Complete if code_frame >= base_frame && base_frame > 0 => {
                    self.exit_frame()?;
                    if self.current_frame < base_frame {
                        break;
                    }
                }
                StepResult::Complete => break,
                StepResult::NextAddress(addr) => {
                    self.frames[code_frame].instruction = addr;
                }
            }

            if code_frame != self.current_frame {
                code_frame = self.current_frame;
                code = self.frames[self.current_frame]
                    .code
                    .clone()
                    .expect("missing frame code");
            }
        }

        Ok(self.registers[0].take())
    }

    pub fn execute_function(&mut self, body: &Code, function: Value) -> Result<Value, Fault> {
        self.enter_frame(body.clone(), function)?;
        self.execute_current_frame()
    }

    pub fn recurse_current_function(&mut self, arity: Arity) -> Result<Value, Fault> {
        let current_function = self.frames[self.current_frame].code_owner.clone();
        current_function.call(self, arity)
    }

    pub fn declare_variable(&mut self, name: Symbol) -> Result<Stack, Fault> {
        let current_frame = &mut self.frames[self.current_frame];
        if current_frame.end < self.max_stack {
            Ok(*current_frame.variables.entry(name).or_insert_with(|| {
                let index = Stack(current_frame.end);
                current_frame.end += 1;
                index
            }))
        } else {
            Err(Fault::StackOverflow)
        }
    }

    pub fn declare(&mut self, name: Symbol, value: Value) -> Option<Value> {
        self.module
            .declarations
            .insert(name, value)
            .map(|f| f.value)
    }

    pub fn declare_function(&mut self, function: Function) -> Option<Value> {
        self.module
            .declarations
            .insert(function.name().clone(), Value::dynamic(function))
            .map(|f| f.value)
    }

    pub fn resolve(&self, name: &Symbol) -> Result<Value, Fault> {
        let current_frame = &self.frames[self.current_frame];
        if let Some(stack) = current_frame.variables.get(name) {
            stack.load(self)
        } else if let Some(value) = self.module.declarations.get(name) {
            Ok(value.clone())
        } else {
            Err(Fault::UnknownSymbol(name.clone()))
        }
    }

    fn enter_frame(&mut self, code: Code, owner: Value) -> Result<(), Fault> {
        if self.current_frame < self.max_depth {
            let current_frame_end = self.frames[self.current_frame].end;

            self.current_frame += 1;
            if self.current_frame < self.frames.len() {
                self.frames[self.current_frame].clear();
                self.frames[self.current_frame].start = current_frame_end;
                self.frames[self.current_frame].end = current_frame_end;
                self.frames[self.current_frame].code = Some(code);
                self.frames[self.current_frame].instruction = 0;
                self.frames[self.current_frame].code_owner = owner;
            } else {
                self.frames.push(Frame {
                    start: current_frame_end,
                    end: current_frame_end,
                    code: Some(code),
                    code_owner: owner,
                    instruction: 0,
                    variables: Map::new(),
                });
            }
            Ok(())
        } else {
            Err(Fault::StackOverflow)
        }
    }

    pub fn exit_frame(&mut self) -> Result<(), Fault> {
        if self.current_frame >= 1 {
            let current_frame = &self.frames[self.current_frame];
            self.stack[current_frame.start..current_frame.end].fill_with(|| Value::Nil);
            self.current_frame -= 1;
            Ok(())
        } else {
            Err(Fault::StackUnderflow)
        }
    }

    pub fn allocate(&mut self, count: usize) -> Result<Stack, Fault> {
        let current_frame = &mut self.frames[self.current_frame];
        let index = Stack(current_frame.end);
        match current_frame.end.checked_add(count) {
            Some(end) if end < self.max_stack => {
                current_frame.end += count;
                if self.stack.len() < current_frame.end {
                    self.stack.resize_with(current_frame.end, || Value::Nil);
                }
                Ok(index)
            }
            _ => Err(Fault::StackOverflow),
        }
    }

    #[must_use]
    pub fn current_instruction(&self) -> usize {
        self.frames[self.current_frame].instruction
    }

    #[must_use]
    pub fn current_code(&self) -> &Code {
        self.frames[self.current_frame]
            .code
            .as_ref()
            .expect("missing frame code")
    }

    pub fn jump_to(&mut self, instruction: usize) {
        self.frames[self.current_frame].instruction = instruction;
    }

    #[must_use]
    pub fn current_frame(&self) -> &[Value] {
        &self.stack[self.frames[self.current_frame].start..self.frames[self.current_frame].end]
    }

    #[must_use]
    pub fn current_frame_mut(&mut self) -> &mut [Value] {
        &mut self.stack[self.frames[self.current_frame].start..self.frames[self.current_frame].end]
    }

    #[must_use]
    pub fn current_frame_start(&self) -> Option<Stack> {
        (self.frames[self.current_frame].end > 0)
            .then_some(Stack(self.frames[self.current_frame].start))
    }

    #[must_use]
    pub fn current_frame_size(&self) -> usize {
        self.frames[self.current_frame].end - self.frames[self.current_frame].start
    }

    pub fn reset(&mut self) {
        self.current_frame = 0;
        self.frames[0].clear();
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Register {
    R0 = 0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

impl TryFrom<u8> for Register {
    type Error = InvalidRegister;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::R0),
            1 => Ok(Self::R1),
            2 => Ok(Self::R2),
            3 => Ok(Self::R3),
            4 => Ok(Self::R4),
            5 => Ok(Self::R5),
            6 => Ok(Self::R6),
            7 => Ok(Self::R7),
            8 => Ok(Self::R8),
            9 => Ok(Self::R9),
            10 => Ok(Self::R10),
            11 => Ok(Self::R11),
            12 => Ok(Self::R12),
            13 => Ok(Self::R13),
            14 => Ok(Self::R14),
            15 => Ok(Self::R15),
            _ => Err(InvalidRegister),
        }
    }
}

impl TryFrom<usize> for Register {
    type Error = InvalidRegister;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        u8::try_from(value)
            .map_err(|_| InvalidRegister)
            .and_then(Self::try_from)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct InvalidRegister;

impl Index<Register> for Vm {
    type Output = Value;

    fn index(&self, index: Register) -> &Self::Output {
        &self.registers[usize::from(index as u8)]
    }
}

impl IndexMut<Register> for Vm {
    fn index_mut(&mut self, index: Register) -> &mut Self::Output {
        &mut self.registers[usize::from(index as u8)]
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

impl Index<usize> for Vm {
    type Output = Value;

    fn index(&self, index: usize) -> &Self::Output {
        &self.stack[index]
    }
}

impl IndexMut<usize> for Vm {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.stack[index]
    }
}

#[derive(Default, Debug, Clone)]
struct Frame {
    start: usize,
    end: usize,
    instruction: usize,
    code: Option<Code>,
    code_owner: Value,
    variables: Map<Symbol, Stack>,
}

impl Frame {
    fn clear(&mut self) {
        self.variables.clear();
        self.instruction = usize::MAX;
        self.code_owner = Value::default();
        self.code = None;
    }
}

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Fault {
    UnknownSymbol(Symbol),
    IncorrectNumberOfArguments,
    OperationOnNil,
    NotAFunction,
    StackOverflow,
    StackUnderflow,
    UnsupportedOperation,
    OutOfMemory,
    OutOfBounds,
    DivideByZero,
    InvalidInstructionAddress,
    ExpectedSymbol,
    InvalidArity,
    InvalidLabel,
}

#[derive(Debug, Clone)]
pub struct Function {
    name: Symbol,
    bodies: Map<Arity, Code>,
    varg_bodies: Map<Arity, Code>,
}

impl Function {
    #[must_use]
    pub fn new(name: impl Into<Symbol>) -> Self {
        Self {
            name: name.into(),
            bodies: Map::new(),
            varg_bodies: Map::new(),
        }
    }

    pub fn insert_arity(&mut self, arity: impl Into<Arity>, body: impl Into<Code>) {
        self.bodies.insert(arity.into(), body.into());
    }

    #[must_use]
    pub fn when(mut self, arity: impl Into<Arity>, body: impl Into<Code>) -> Self {
        self.insert_arity(arity, body);
        self
    }

    #[must_use]
    pub const fn name(&self) -> &Symbol {
        &self.name
    }
}

impl CustomType for Function {
    fn call(&self, vm: &mut Vm, this: &Dynamic, arity: Arity) -> Result<Value, Fault> {
        if let Some(body) = self.bodies.get(&arity).or_else(|| {
            self.varg_bodies
                .iter()
                .rev()
                .find_map(|va| (va.key() <= &arity).then_some(&va.value))
        }) {
            vm.execute_function(body, Value::Dynamic(this.clone()))
        } else {
            Err(Fault::IncorrectNumberOfArguments)
        }
    }
}

impl Instruction for Function {
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        vm.declare_function(self.clone());
        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> bitcode::Op {
        bitcode::Op::DeclareFunction(BitcodeFunction::from(self))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub struct Arity(pub u8);

impl From<u8> for Arity {
    fn from(arity: u8) -> Self {
        Self(arity)
    }
}

impl From<Arity> for u8 {
    fn from(arity: Arity) -> Self {
        arity.0
    }
}

impl PartialEq<u8> for Arity {
    fn eq(&self, other: &u8) -> bool {
        &self.0 == other
    }
}

#[derive(Debug, Clone)]
pub struct Code {
    data: Arc<CodeData>,
}

#[derive(Default, Debug, Clone)]
struct CodeData {
    instructions: Vec<Arc<dyn Instruction>>,
    labels: Vec<usize>,
    stack_requirement: usize,
}

impl Code {
    pub fn push_boxed(&mut self, instruction: Arc<dyn Instruction>) {
        Arc::make_mut(&mut self.data).instructions.push(instruction);
    }

    pub fn push<I>(&mut self, instruction: I)
    where
        I: Instruction,
    {
        self.push_boxed(Arc::new(instruction));
    }

    #[must_use]
    pub fn with<I>(mut self, instruction: I) -> Self
    where
        I: Instruction,
    {
        self.push(instruction);
        self
    }

    fn step(&self, vm: &mut Vm, address: usize) -> Result<StepResult, Fault> {
        let instruction = self
            .data
            .instructions
            .get(address)
            .ok_or(Fault::InvalidInstructionAddress)?;
        // println!("Executing {instruction:?}");
        if instruction.execute(vm)?.is_break() {
            Ok(StepResult::Complete)
        } else if vm.current_instruction() == address {
            match address.checked_add(1) {
                Some(next) if next < self.data.instructions.len() => {
                    Ok(StepResult::NextAddress(next))
                }
                _ => Ok(StepResult::Complete),
            }
        } else if vm.current_instruction() < self.data.instructions.len() {
            // Execution caused a jump
            Ok(StepResult::NextAddress(vm.current_instruction()))
        } else {
            Ok(StepResult::Complete)
        }
    }

    // pub fn decode_from(encoded: &[u8]) -> Result<Self, DecodeError> {
    //     bitcode::decode(encoded)
    // }
}

impl Default for Code {
    fn default() -> Self {
        static EMPTY: OnceLock<Code> = OnceLock::new();
        EMPTY
            .get_or_init(|| Code {
                data: Arc::default(),
            })
            .clone()
    }
}

enum StepResult {
    Complete,
    NextAddress(usize),
}

#[derive(Default, Debug)]
pub struct Module {
    declarations: Map<Symbol, Value>,
}
