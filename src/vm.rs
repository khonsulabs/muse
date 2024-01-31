use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::future::Future;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::ops::{ControlFlow, Index, IndexMut};
use std::pin::{pin, Pin};
use std::sync::{Arc, OnceLock};
use std::task::{Poll, Wake, Waker};
use std::time::{Duration, Instant};
use std::{array, task};

use crossbeam_utils::sync::{Parker, Unparker};
use kempt::Map;
use ops::{Destination, Instruction, Source};
use serde::{Deserialize, Serialize};

use self::bitcode::{BitcodeFunction, OpDestination, ValueOrSource};
use self::ops::Stack;
use crate::symbol::{IntoOptionSymbol, Symbol};
use crate::value::{CustomType, Dynamic, Value};

pub mod bitcode;
pub mod ops;

pub struct Vm {
    registers: [Value; 256],
    stack: Vec<Value>,
    max_stack: usize,
    frames: Vec<Frame>,
    current_frame: usize,
    has_anonymous_frame: bool,
    max_depth: usize,
    budget: Budget,
    execute_until: Option<Instant>,
    module: Module,
    waker: Waker,
    parker: Parker,
    code: Vec<RegisteredCode>,
    code_map: Map<usize, CodeIndex>,
}

impl Default for Vm {
    fn default() -> Self {
        let parker = Parker::new();
        let unparker = parker.unparker().clone();
        Self {
            registers: array::from_fn(|_| Value::Nil),
            frames: vec![Frame::default()],
            stack: Vec::new(),
            current_frame: 0,
            has_anonymous_frame: false,
            max_stack: usize::MAX,
            max_depth: usize::MAX,
            budget: Budget::default(),
            execute_until: None,
            module: Module::default(),
            waker: Waker::from(Arc::new(VmWaker(unparker))),
            parker,
            code: Vec::new(),
            code_map: Map::new(),
        }
    }
}

struct RegisteredCode {
    code: Code,
    owner: Option<Dynamic>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct CodeIndex(usize);

impl Vm {
    fn push_code(&mut self, code: &Code, owner: Option<&Dynamic>) -> CodeIndex {
        *self
            .code_map
            .entry(Arc::as_ptr(&code.data) as usize)
            .or_insert_with(|| {
                let index = CodeIndex(self.code.len());
                self.code.push(RegisteredCode {
                    code: code.clone(),
                    owner: owner.cloned(),
                });
                index
            })
    }

    pub fn execute(&mut self, code: &Code) -> Result<Value, Fault> {
        let code = self.push_code(code, None);
        self.execute_owned(code)
    }

    pub fn execute_for(&mut self, code: &Code, duration: Duration) -> Result<Value, Fault> {
        self.execute_until(
            code,
            Instant::now().checked_add(duration).ok_or(Fault::Timeout)?,
        )
    }

    pub fn execute_until(&mut self, code: &Code, instant: Instant) -> Result<Value, Fault> {
        let code = self.push_code(code, None);
        self.execute_until = Some(instant);
        self.execute_owned(code)
    }

    fn prepare_owned(&mut self, code: CodeIndex) -> Result<(), Fault> {
        self.frames[self.current_frame].code = Some(code);
        self.frames[self.current_frame].instruction = 0;

        self.allocate(self.code[code.0].code.data.stack_requirement)?;
        Ok(())
    }

    fn execute_owned(&mut self, code: CodeIndex) -> Result<Value, Fault> {
        self.prepare_owned(code)?;

        self.resume()
    }

    pub fn execute_async(&mut self, code: &Code) -> Result<ExecuteAsync<'_>, Fault> {
        let code = self.push_code(code, None);
        self.prepare_owned(code)?;

        Ok(ExecuteAsync(self))
    }

    pub fn resume_async(&mut self) -> Result<ExecuteAsync<'_>, Fault> {
        Ok(ExecuteAsync(self))
    }

    pub fn block_on<R>(&self, mut future: impl Future<Output = R> + Unpin) -> R {
        let mut context = task::Context::from_waker(&self.waker);

        loop {
            match Future::poll(pin!(&mut future), &mut context) {
                Poll::Ready(result) => return result,
                Poll::Pending => self.parker.park(),
            }
        }
    }

    #[must_use]
    pub const fn waker(&self) -> &Waker {
        &self.waker
    }

    pub fn increase_budget(&mut self, amount: usize) {
        self.budget.allocate(amount);
    }

    pub fn resume(&mut self) -> Result<Value, Fault> {
        loop {
            match self.resume_async_inner() {
                Err(Fault::Waiting) => {
                    self.check_timeout()?;

                    self.parker.park();
                }
                other => return other,
            }
        }
    }

    fn resume_async_inner(&mut self) -> Result<Value, Fault> {
        if self.has_anonymous_frame {
            self.current_frame -= 1;
        }
        let base_frame = self.current_frame;
        let mut code_frame = base_frame;
        let mut code = self.code[self.frames[base_frame].code.expect("missing frame code").0]
            .code
            .clone();
        loop {
            self.budget.charge()?;
            match code.step(self, self.frames[code_frame].instruction)? {
                StepResult::Complete if code_frame >= base_frame && code_frame > 0 => {
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
                self.check_timeout()?;
                code_frame = self.current_frame;
                code = self.code[self.frames[self.current_frame]
                    .code
                    .expect("missing frame code")
                    .0]
                    .code
                    .clone();
            }
        }

        if self.current_frame == 0 {
            self.execute_until = None;
        }
        Ok(self.registers[0].take())
    }

    fn check_timeout(&mut self) -> Result<(), Fault> {
        if matches!(self.execute_until, Some(execute_until) if execute_until < Instant::now()) {
            self.execute_until = None;
            Err(Fault::Timeout)
        } else {
            Ok(())
        }
    }

    pub fn enter_anonymous_frame(&mut self) -> Result<(), Fault> {
        if self.has_anonymous_frame {
            self.current_frame += 1;
            Ok(())
        } else {
            self.has_anonymous_frame = true;
            self.enter_frame(None)
        }
    }

    fn execute_function(&mut self, body: &Code, function: &Dynamic) -> Result<Value, Fault> {
        let body_index = self.push_code(body, Some(function));
        self.enter_frame(Some(body_index))?;

        self.allocate(body.data.stack_requirement)?;

        Err(Fault::FrameChanged)
    }

    pub fn recurse_current_function(&mut self, arity: Arity) -> Result<Value, Fault> {
        let current_function = self.code[self.frames[self.current_frame]
            .code
            .expect("missing function")
            .0]
            .owner
            .clone()
            .ok_or(Fault::NotAFunction)?;
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

    pub fn declare(&mut self, name: impl Into<Symbol>, value: Value) -> Option<Value> {
        self.module
            .declarations
            .insert(name.into(), value)
            .map(|f| f.value)
    }

    pub fn declare_function(&mut self, function: Function) -> Option<Value> {
        let name = function.name().as_ref()?.clone();
        self.module
            .declarations
            .insert(name, Value::dynamic(function))
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

    fn enter_frame(&mut self, code: Option<CodeIndex>) -> Result<(), Fault> {
        if self.current_frame < self.max_depth {
            let current_frame_end = self.frames[self.current_frame].end;

            self.current_frame += 1;
            if self.current_frame < self.frames.len() {
                self.frames[self.current_frame].clear();
                self.frames[self.current_frame].start = current_frame_end;
                self.frames[self.current_frame].end = current_frame_end;
                self.frames[self.current_frame].code = code;
                self.frames[self.current_frame].instruction = 0;
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
        if self.current_frame >= 1 {
            self.has_anonymous_frame = false;
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
    pub fn current_code(&self) -> Option<&Code> {
        self.frames[self.current_frame]
            .code
            .map(|index| &self.code[index.0].code)
    }

    #[must_use]
    pub fn current_instruction(&self) -> usize {
        self.frames[self.current_frame].instruction
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
        for frame in &mut self.frames {
            frame.clear();
        }
        self.registers.fill_with(|| Value::Nil);
        self.stack.fill_with(|| Value::Nil);
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]

pub struct Register(pub u8);

impl From<u8> for Register {
    fn from(value: u8) -> Self {
        Self(value)
    }
}

impl TryFrom<usize> for Register {
    type Error = InvalidRegister;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        u8::try_from(value).map(Self).map_err(|_| InvalidRegister)
    }
}

impl From<Register> for usize {
    fn from(value: Register) -> Self {
        usize::from(value.0)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct InvalidRegister;

impl Index<Register> for Vm {
    type Output = Value;

    fn index(&self, index: Register) -> &Self::Output {
        &self.registers[usize::from(index)]
    }
}

impl IndexMut<Register> for Vm {
    fn index_mut(&mut self, index: Register) -> &mut Self::Output {
        &mut self.registers[usize::from(index)]
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
    code: Option<CodeIndex>,
    variables: Map<Symbol, Stack>,
}

impl Frame {
    fn clear(&mut self) {
        self.variables.clear();
        self.instruction = usize::MAX;
        self.code = None;
    }
}

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize, Clone)]
pub enum Fault {
    UnknownSymbol(Symbol),
    IncorrectNumberOfArguments,
    OperationOnNil,
    NotAFunction,
    StackOverflow,
    StackUnderflow,
    InvalidIndex,
    UnsupportedOperation,
    OutOfMemory,
    OutOfBounds,
    DivideByZero,
    InvalidInstructionAddress,
    ExpectedSymbol,
    ExpectedInteger,
    ExpectedString,
    InvalidArity,
    InvalidLabel,
    NoBudget,
    Timeout,
    Waiting,
    FrameChanged,
    Custom {
        // TODO add an optional Type as a source.
        code: u32,
        message: Cow<'static, String>,
    },
}

#[derive(Debug, Clone)]
pub struct Function {
    name: Option<Symbol>,
    bodies: Map<Arity, Code>,
    varg_bodies: Map<Arity, Code>,
}

impl Function {
    #[must_use]
    pub fn new(name: impl IntoOptionSymbol) -> Self {
        Self {
            name: name.into_symbol(),
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
    pub const fn name(&self) -> &Option<Symbol> {
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
            vm.execute_function(body, this)
        } else {
            Err(Fault::IncorrectNumberOfArguments)
        }
    }
}

#[derive(Debug)]
struct DeclareFunction<Dest> {
    function: Function,
    dest: Dest,
}

impl<Dest> Instruction for DeclareFunction<Dest>
where
    Dest: Destination,
{
    fn execute(&self, vm: &mut Vm) -> Result<ControlFlow<()>, Fault> {
        let function = Value::dynamic(self.function.clone());
        if let Some(name) = &self.function.name {
            vm.declare(name.clone(), function.clone());
        }
        self.dest.store(vm, function)?;

        Ok(ControlFlow::Continue(()))
    }

    fn as_op(&self) -> bitcode::Op {
        bitcode::Op::DeclareFunction {
            function: BitcodeFunction::from(&self.function),
            dest: self.dest.as_dest(),
        }
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
        let instruction = match address.cmp(&self.data.instructions.len()) {
            Ordering::Less => &self.data.instructions[address],
            Ordering::Equal => return Ok(StepResult::Complete),
            Ordering::Greater => return Err(Fault::InvalidInstructionAddress),
        };
        // println!("Executing {instruction:?}");
        let next_address_step = match address.checked_add(1) {
            Some(next) if next <= self.data.instructions.len() => StepResult::NextAddress(next),
            _ => StepResult::Complete,
        };
        match instruction.execute(vm) {
            Ok(ControlFlow::Continue(())) => {
                if vm.current_instruction() == address {
                    Ok(next_address_step)
                } else if vm.current_instruction() < self.data.instructions.len() {
                    // Execution caused a jump
                    Ok(StepResult::NextAddress(vm.current_instruction()))
                } else {
                    Ok(StepResult::Complete)
                }
            }
            Ok(ControlFlow::Break(())) => Ok(StepResult::Complete),
            Err(Fault::FrameChanged) => Ok(next_address_step),
            Err(other) => Err(other),
        }
    }
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

#[derive(Clone, Copy, Debug, Default)]
struct Budget(Option<NonZeroUsize>);

impl Budget {
    fn allocate(&mut self, amount: usize) {
        self.0 = match self.0 {
            Some(budget) => Some(budget.saturating_add(amount)),
            None => NonZeroUsize::new(amount.saturating_add(1)),
        };
    }

    fn charge(&mut self) -> Result<(), Fault> {
        if let Some(amount) = &mut self.0 {
            *amount = NonZeroUsize::new(amount.get().saturating_sub(1)).ok_or(Fault::NoBudget)?;
        }
        Ok(())
    }
}

pub struct ExecuteAsync<'a>(&'a mut Vm);

impl Future for ExecuteAsync<'_> {
    type Output = Result<Value, Fault>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
        // Temporarily replace the VM's waker with this context's waker.
        let previous_waker = std::mem::replace(&mut self.0.waker, cx.waker().clone());
        let result = match self.0.resume_async_inner() {
            Err(Fault::Waiting) => Poll::Pending,
            other => Poll::Ready(other),
        };
        // Restore the VM's waker.
        self.0.waker = previous_waker;
        result
    }
}

struct VmWaker(Unparker);

impl Wake for VmWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
