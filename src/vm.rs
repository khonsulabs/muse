use std::array;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Add, Index, IndexMut};
use std::sync::Arc;

use kempt::Map;
use ops::{Destination, Instruction, Source};
use serde::{Deserialize, Serialize};

use self::bitcode::{OpDestination, ValueOrSource};
use crate::symbol::Symbol;
use crate::value::Value;

pub mod bitcode;
pub mod ops;

pub struct Vm {
    registers: [Value; 16],
    stack: Vec<Value>,
    frames: Vec<Frame>,
    current_frame: usize,
    max_stack: usize,
    max_depth: usize,
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
        }
    }
}

impl Vm {
    pub fn declare_variable(&mut self, name: Symbol) -> Result<Variable, Fault> {
        let current_frame = &mut self.frames[self.current_frame];
        if current_frame.end < self.max_stack {
            Ok(*current_frame.variables.entry(name).or_insert_with(|| {
                let index = Variable(current_frame.end);
                current_frame.end += 1;
                index
            }))
        } else {
            Err(Fault::StackOverflow)
        }
    }

    pub fn resolve(&self, name: &Symbol) -> Result<Variable, Fault> {
        let current_frame = &self.frames[self.current_frame];
        current_frame
            .variables
            .get(name)
            .copied()
            .ok_or_else(|| Fault::UnknownSymbol(name.clone()))
    }

    pub fn enter_frame(&mut self, code: Code) -> Result<(), Fault> {
        if self.current_frame < self.max_depth {
            let current_frame_end = self.frames[self.current_frame].end;

            self.current_frame += 1;
            if self.current_frame < self.frames.len() {
                self.frames[self.current_frame].start = current_frame_end;
                self.frames[self.current_frame].end = current_frame_end;
                self.frames[self.current_frame].variables.clear();
            } else {
                self.frames.push(Frame {
                    start: current_frame_end,
                    end: current_frame_end,
                    code,
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
        if self.current_frame > 1 {
            let current_frame = &self.frames[self.current_frame];
            self.stack[current_frame.start..current_frame.end].fill_with(|| Value::Nil);
            self.current_frame -= 1;
            Ok(())
        } else {
            Err(Fault::StackUnderflow)
        }
    }

    pub fn allocate(&mut self, count: u16) -> Result<Variable, Fault> {
        let count = usize::from(count);
        let current_frame = &mut self.frames[self.current_frame];
        let index = Variable(current_frame.end);
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
    pub fn current_frame(&self) -> &[Value] {
        &self.stack[self.frames[self.current_frame].start..self.frames[self.current_frame].end]
    }

    #[must_use]
    pub fn current_frame_mut(&mut self) -> &mut [Value] {
        &mut self.stack[self.frames[self.current_frame].start..self.frames[self.current_frame].end]
    }

    #[must_use]
    pub fn current_frame_start(&self) -> Option<Variable> {
        (self.frames[self.current_frame].end > 0)
            .then_some(Variable(self.frames[self.current_frame].start))
    }

    #[must_use]
    pub fn current_frame_size(&self) -> usize {
        self.frames[self.current_frame].end - self.frames[self.current_frame].start
    }

    pub fn execute(&mut self, code: &Code) -> Result<Value, Fault> {
        let mut code = code.clone();
        self.frames[self.current_frame].code = code.clone();
        self.frames[self.current_frame].instruction = 0;

        let mut code_frame = self.current_frame;
        loop {
            match code.step(self, self.frames[self.current_frame].instruction)? {
                StepResult::Complete if self.current_frame > 0 => self.exit_frame()?,
                StepResult::Complete => break,
                StepResult::NextAddress(addr) => {
                    self.frames[self.current_frame].instruction = addr;
                }
            }

            if code_frame != self.current_frame {
                code_frame = self.current_frame;
                code = self.frames[self.current_frame].code.clone();
            }
        }

        Ok(self.registers[1].take())
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
    fn load(&self, vm: &mut Vm) -> Result<Value, Fault> {
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

impl Index<Variable> for Vm {
    type Output = Value;

    fn index(&self, index: Variable) -> &Self::Output {
        &self[index.0]
    }
}

impl IndexMut<Variable> for Vm {
    fn index_mut(&mut self, index: Variable) -> &mut Self::Output {
        &mut self[index.0]
    }
}

#[derive(Default, Debug, Clone)]
struct Frame {
    start: usize,
    end: usize,
    instruction: usize,
    code: Code,
    variables: Map<Symbol, Variable>,
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct Variable(usize);

impl Add<usize> for Variable {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs)
    }
}

#[derive(Debug, Eq, PartialEq)]
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
}

#[derive(Default, Debug, Clone)]
pub struct Code {
    instructions: Arc<Vec<Arc<dyn Instruction>>>,
}

impl Code {
    pub fn push_boxed(&mut self, instruction: Arc<dyn Instruction>) {
        Arc::make_mut(&mut self.instructions).push(instruction);
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
            .instructions
            .get(address)
            .ok_or(Fault::InvalidInstructionAddress)?;
        instruction.execute(vm)?;

        match address.checked_add(1) {
            Some(next) if next < self.instructions.len() => Ok(StepResult::NextAddress(next)),
            _ => Ok(StepResult::Complete),
        }
    }

    // pub fn decode_from(encoded: &[u8]) -> Result<Self, DecodeError> {
    //     bitcode::decode(encoded)
    // }
}

enum StepResult {
    Complete,
    NextAddress(usize),
}
