use std::array;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::{Add, Index, IndexMut};
use std::sync::Arc;

use instructions::Instruction;
use kempt::Map;
use symbol::Symbol;
use value::Value;

pub mod instructions;
pub mod symbol;
pub mod value;

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
    pub fn declare_variable(&mut self, name: Symbol) -> Result<VariableIndex, Fault> {
        let current_frame = &mut self.frames[self.current_frame];
        if current_frame.end < self.max_stack {
            Ok(*current_frame.variables.entry(name).or_insert_with(|| {
                let index = VariableIndex(current_frame.end);
                current_frame.end += 1;
                index
            }))
        } else {
            Err(Fault::StackOverflow)
        }
    }

    pub fn resolve(&self, name: &Symbol) -> Result<VariableIndex, Fault> {
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

    pub fn allocate(&mut self, count: usize) -> Result<VariableIndex, Fault> {
        let current_frame = &mut self.frames[self.current_frame];
        let index = VariableIndex(current_frame.end);
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

    pub fn current_frame(&self) -> &[Value] {
        &self.stack[self.frames[self.current_frame].start..self.frames[self.current_frame].end]
    }

    pub fn current_frame_mut(&mut self) -> &mut [Value] {
        &mut self.stack[self.frames[self.current_frame].start..self.frames[self.current_frame].end]
    }

    pub fn current_frame_start(&self) -> Option<VariableIndex> {
        (self.frames[self.current_frame].end > 0)
            .then_some(VariableIndex(self.frames[self.current_frame].start))
    }

    pub fn current_frame_size(&self) -> usize {
        self.frames[self.current_frame].end - self.frames[self.current_frame].start
    }

    pub fn execute(&mut self, mut code: Code) -> Result<(), Fault> {
        self.frames[self.current_frame].code = code.clone();
        self.frames[self.current_frame].instruction = 0;

        let mut code_frame = self.current_frame;
        loop {
            match code.step(self, self.frames[self.current_frame].instruction)? {
                StepResult::Complete if self.current_frame > 0 => self.exit_frame()?,
                StepResult::Complete => return Ok(()),
                StepResult::NextAddress(addr) => {
                    self.frames[self.current_frame].instruction = addr;
                }
            }

            if code_frame != self.current_frame {
                code_frame = self.current_frame;
                code = self.frames[self.current_frame].code.clone();
            }
        }
    }
}

macro_rules! impl_registers {
    ($($name:ident => $index:literal),+) => {
        $(impl_registers!(impl $name => $index);)+
    };
    (impl $name:ident => $index:literal) => {
        #[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
        pub struct $name;

        impl Index<$name> for Vm {
            type Output = Value;

            fn index(&self, _index: $name) -> &Self::Output {
                &self.registers[$index]
            }
        }

        impl IndexMut<$name> for Vm {
            fn index_mut(&mut self, _index: $name) -> &mut Self::Output {
                &mut self.stack[$index]
            }
        }

        impl crate::instructions::Source for $name {
            fn load(&self, vm: &mut Vm) -> Result<Value, Fault> {
                Ok(vm[*self].clone())
            }
        }

        impl crate::instructions::Destination for $name {
            fn store(&self, vm: &mut Vm, value: Value) -> Result<(), Fault> {
                vm[*self] = value;
                Ok(())
            }
        }
    };
}

impl_registers!(
    R0 => 0,
    R1=> 1,
    R2=> 2,
    R3=> 3,
    R4=> 4,
    R5=> 5,
    R6=> 6,
    R7=> 7,
    R8=> 8,
    R9=> 9,
    R10=> 10,
    R11=> 11,
    R12=> 12,
    R13=> 13,
    R14=> 14,
    R15=> 15);

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

impl Index<VariableIndex> for Vm {
    type Output = Value;

    fn index(&self, index: VariableIndex) -> &Self::Output {
        &self[index.0]
    }
}

impl IndexMut<VariableIndex> for Vm {
    fn index_mut(&mut self, index: VariableIndex) -> &mut Self::Output {
        &mut self[index.0]
    }
}

#[derive(Default, Debug, Clone)]
struct Frame {
    start: usize,
    end: usize,
    instruction: usize,
    code: Code,
    variables: Map<Symbol, VariableIndex>,
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct VariableIndex(usize);

impl Add<usize> for VariableIndex {
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
    pub fn with<I>(mut self, instruction: I) -> Self
    where
        I: Instruction,
    {
        Arc::make_mut(&mut self.instructions).push(Arc::new(instruction));
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
}

enum StepResult {
    Complete,
    NextAddress(usize),
}

// #[derive(Default, Debug, Clone)]
// pub struct Code {
//     code: Vec<u8>,
//     statics: Vec<Value>,
// }

// impl Code {
//     pub fn next_instruction(&self, offset: usize) -> Result<DecodedInstruction, CodeError> {
//         let code = self
//             .code
//             .get(offset)
//             .ok_or(CodeError::InvalidInstructionAddress)?;
//         let opcode = code & 0xff;
//         let code = code >> 8;
//         let arg1 = code >> 8;

//         match opcode {}
//     }
// }

// #[derive(Debug, Clone, Copy, Eq, PartialEq)]
// pub enum CodeError {
//     UnknownOpcode,
//     InvalidInstructionAddress,
// }

// pub struct DecodedInstruction {
//     pub next: usize,
//     pub decoded: Instruction,
// }

// pub struct Instruction {
//     pub opcode: Opcode,
//     pub args: [IArg; 3],
// }

// pub enum IArg {
//     None,
//     Static(u32),
//     Stack(u32),
// }

// #[repr(u32)]
// pub enum Opcode {
//     Add,
//     Sub,
//     Mul,
//     Div,
//     Rem,
// }
