use std::cmp::Ordering;
use std::fmt::{Debug, Write};
use std::future::Future;
use std::hash::Hash;
use std::num::{NonZeroUsize, TryFromIntError};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Poll, Wake, Waker};
use std::time::{Duration, Instant};
use std::{array, task};

use ahash::AHashMap;
use crossbeam_utils::sync::{Parker, Unparker};
use kempt::map::Entry;
use kempt::Map;
use parking_lot::{Mutex, MutexGuard};
use refuse::{CollectionGuard, ContainsNoRefs, NoMapping, Root, Trace};
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "dispatched"))]
use self::bitcode::trusted_loaded_source_to_value;
use self::bitcode::{
    BinaryKind, BitcodeFunction, FaultKind, Label, Op, OpDestination, ValueOrSource,
};
use crate::compiler::{BitcodeModule, BlockDeclaration, SourceMap, UnaryKind};
use crate::exception::Exception;
use crate::regex::MuseRegex;
use crate::string::MuseString;
use crate::symbol::{IntoOptionSymbol, Symbol, SymbolRef};
use crate::syntax::token::RegexLiteral;
use crate::syntax::{BitwiseKind, CompareKind, SourceRange};
#[cfg(not(feature = "dispatched"))]
use crate::value::ContextOrGuard;
use crate::value::{CustomType, Dynamic, Rooted, RustType, StaticRustFunctionTable, Value};

pub mod bitcode;

macro_rules! try_all {
    ($a:expr, $b:expr) => {
        match ($a, $b) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(err), _) | (_, Err(err)) => return Err(err),
        }
    };
    ($a:expr, $b:expr, $c:expr) => {
        match ($a, $b, $c) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            (Err(err), _, _) | (_, Err(err), _) | (_, _, Err(err)) => return Err(err),
        }
    };
}

#[cfg(feature = "dispatched")]
mod dispatched;

#[derive(Clone)]
pub struct Vm {
    memory: Root<VmMemory>,
}

impl Vm {
    #[must_use]
    pub fn new(guard: &CollectionGuard) -> Self {
        let parker = Parker::new();
        let unparker = parker.unparker().clone();
        Self {
            memory: Root::new(
                VmMemory(Mutex::new(VmState {
                    registers: array::from_fn(|_| Value::Nil),
                    frames: vec![Frame::default()],
                    stack: Vec::new(),
                    current_frame: 0,
                    has_anonymous_frame: false,
                    max_stack: usize::MAX,
                    max_depth: usize::MAX,
                    counter: 16,
                    steps_per_charge: 16,
                    budget: Budget::default(),
                    execute_until: None,
                    modules: vec![Module::with_core(guard)],
                    waker: Waker::from(Arc::new(VmWaker(unparker))),
                    parker,
                    code: Vec::new(),
                    code_map: Map::new(),
                })),
                guard,
            ),
        }
    }

    pub fn execute(
        &self,
        code: &Code,
        guard: &mut CollectionGuard,
    ) -> Result<Value, ExecutionError> {
        self.context(guard).execute(code)
    }

    pub fn execute_for(
        &self,
        code: &Code,
        duration: Duration,
        guard: &mut CollectionGuard,
    ) -> Result<Value, ExecutionError> {
        self.context(guard).execute_for(code, duration)
    }

    pub fn execute_until(
        &self,
        code: &Code,
        instant: Instant,
        guard: &mut CollectionGuard,
    ) -> Result<Value, ExecutionError> {
        self.context(guard).execute_until(code, instant)
    }

    pub fn execute_async<'context, 'guard>(
        &'context self,
        code: &Code,
        guard: &'context mut CollectionGuard<'guard>,
    ) -> Result<ExecuteAsync<'context, 'guard>, ExecutionError> {
        self.context(guard).execute_async(code)
    }

    pub fn resume_async<'context, 'guard>(
        &'context self,
        guard: &'context mut CollectionGuard<'guard>,
    ) -> Result<ExecuteAsync<'context, 'guard>, ExecutionError> {
        self.context(guard).resume_async()
    }

    pub fn increase_budget(&self, amount: usize) {
        self.memory.0.lock().budget.allocate(amount);
    }

    pub fn resume(&self, guard: &mut CollectionGuard) -> Result<Value, ExecutionError> {
        self.context(guard).resume()
    }

    pub fn invoke(
        &self,
        name: &SymbolRef,
        params: impl InvokeArgs,
        guard: &mut CollectionGuard,
    ) -> Result<Value, ExecutionError> {
        self.context(guard).invoke(name, params)
    }

    fn context<'context, 'guard>(
        &'context self,
        guard: &'context mut CollectionGuard<'guard>,
    ) -> VmContext<'context, 'guard> {
        VmContext {
            guard,
            vm: self.memory.0.lock(),
        }
    }

    pub fn set_steps_per_charge(&self, steps: u16) {
        self.memory.0.lock().set_steps_per_charge(steps);
    }

    #[must_use]
    pub fn register(&self, register: Register) -> Value {
        self.memory.0.lock()[register]
    }

    #[allow(clippy::must_use_candidate)]
    pub fn set_register(&self, register: Register, value: Value) -> Value {
        std::mem::replace(&mut self.memory.0.lock()[register], value)
    }

    #[must_use]
    pub fn stack(&self, index: usize) -> Value {
        self.memory.0.lock()[index]
    }

    #[must_use]
    pub fn set_stack(&self, index: usize, value: Value) -> Value {
        std::mem::replace(&mut self.memory.0.lock()[index], value)
    }

    pub fn declare_variable(
        &self,
        name: SymbolRef,
        mutable: bool,
        guard: &mut CollectionGuard<'_>,
    ) -> Result<Stack, Fault> {
        VmContext::new(self, guard).declare_variable(name, mutable)
    }

    pub fn declare(
        &self,
        name: impl Into<SymbolRef>,
        value: Value,
        guard: &mut CollectionGuard<'_>,
    ) -> Result<Option<Value>, Fault> {
        VmContext::new(self, guard).declare_inner(name, value, false)
    }

    pub fn declare_mut(
        &self,
        name: impl Into<SymbolRef>,
        value: Value,
        guard: &mut CollectionGuard<'_>,
    ) -> Result<Option<Value>, Fault> {
        VmContext::new(self, guard).declare_inner(name, value, true)
    }

    pub fn declare_function(
        &self,
        function: Function,
        guard: &mut CollectionGuard<'_>,
    ) -> Result<Option<Value>, Fault> {
        VmContext::new(self, guard).declare_function(function)
    }
}

#[derive(Clone)]
struct RegisteredCode {
    code: Code,
    owner: Option<Rooted<Function>>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct CodeIndex(usize);

struct VmMemory(Mutex<VmState>);

impl refuse::Trace for VmMemory {
    const MAY_CONTAIN_REFERENCES: bool = true;

    fn trace(&self, tracer: &mut refuse::Tracer) {
        let state = self.0.lock();
        for register in state
            .registers
            .iter()
            .chain(state.stack[0..state.frames[state.current_frame].end].iter())
            .filter_map(Value::as_any_dynamic)
        {
            tracer.mark(register.0);
        }

        for frame in &state.frames {
            for key in frame.variables.keys() {
                key.trace(tracer);
            }
        }

        for module in &state.modules {
            tracer.mark(*module);
        }
    }
}

impl NoMapping for VmMemory {}

pub struct VmState {
    registers: [Value; 256],
    stack: Vec<Value>,
    max_stack: usize,
    frames: Vec<Frame>,
    current_frame: usize,
    has_anonymous_frame: bool,
    max_depth: usize,
    counter: u16,
    steps_per_charge: u16,
    budget: Budget,
    execute_until: Option<Instant>,
    modules: Vec<Dynamic<Module>>,
    waker: Waker,
    parker: Parker,
    code: Vec<RegisteredCode>,
    code_map: Map<usize, CodeIndex>,
}

impl VmState {
    pub fn set_steps_per_charge(&mut self, steps: u16) {
        self.steps_per_charge = steps;
        self.counter = self.counter.min(steps);
    }
}

impl<'context, 'guard> VmContext<'context, 'guard> {
    fn push_code(&mut self, code: &Code, owner: Option<&Rooted<Function>>) -> CodeIndex {
        let vm = self.vm();
        *vm.code_map
            .entry(Arc::as_ptr(&code.data) as usize)
            .or_insert_with(|| {
                let index = CodeIndex(vm.code.len());
                vm.code.push(RegisteredCode {
                    code: code.clone(),
                    owner: owner.cloned(),
                });
                index
            })
    }

    pub fn execute(&mut self, code: &Code) -> Result<Value, ExecutionError> {
        let code = self.push_code(code, None);
        self.execute_owned(code)
    }

    pub fn execute_for(
        &mut self,
        code: &Code,
        duration: Duration,
    ) -> Result<Value, ExecutionError> {
        self.execute_until(
            code,
            Instant::now()
                .checked_add(duration)
                .ok_or(ExecutionError::Timeout)?,
        )
    }

    pub fn execute_until(
        &mut self,
        code: &Code,
        instant: Instant,
    ) -> Result<Value, ExecutionError> {
        let code = self.push_code(code, None);
        self.execute_until = Some(instant);
        self.execute_owned(code)
    }

    fn prepare_owned(&mut self, code: CodeIndex) -> Result<(), ExecutionError> {
        let vm = self.vm();
        vm.frames[vm.current_frame].code = Some(code);
        vm.frames[vm.current_frame].instruction = 0;

        self.allocate(self.code[code.0].code.data.stack_requirement)
            .map_err(|err| ExecutionError::new(err, self))?;
        Ok(())
    }

    fn execute_owned(&mut self, code: CodeIndex) -> Result<Value, ExecutionError> {
        self.prepare_owned(code)?;

        self.resume()
    }

    pub fn execute_async(
        mut self,
        code: &Code,
    ) -> Result<ExecuteAsync<'context, 'guard>, ExecutionError> {
        let code = self.push_code(code, None);
        self.prepare_owned(code)?;

        Ok(ExecuteAsync(self))
    }

    pub fn resume_async(self) -> Result<ExecuteAsync<'context, 'guard>, ExecutionError> {
        Ok(ExecuteAsync(self))
    }

    pub fn increase_budget(&mut self, amount: usize) {
        self.budget.allocate(amount);
    }

    pub fn resume(&mut self) -> Result<Value, ExecutionError> {
        loop {
            match self.resume_async_inner(0) {
                Err(Fault::Waiting) => {
                    self.check_timeout()
                        .map_err(|err| ExecutionError::new(err, self))?;

                    self.parker.park();
                }
                Err(other) => return Err(ExecutionError::new(other, self)),
                Ok(value) => return Ok(value),
            }
        }
    }

    pub fn invoke(
        &mut self,
        name: &SymbolRef,
        params: impl InvokeArgs,
    ) -> Result<Value, ExecutionError> {
        let arity = params.load(self)?;

        let module_dynamic = self.modules[0]
            .as_rooted(self.guard)
            .expect("module missing");
        let module_declarations = module_dynamic.declarations();
        let function = module_declarations
            .get(name)
            .ok_or_else(|| ExecutionError::new(Fault::UnknownSymbol, self))?
            .value;
        drop(module_declarations);

        let Some(function) = function.as_dynamic::<Function>() else {
            return Err(ExecutionError::new(Fault::NotAFunction, self));
        };

        match function.as_any_dynamic().call(self, arity) {
            Ok(value) => Ok(value),
            Err(Fault::FrameChanged) => self.resume(),
            Err(other) => Err(ExecutionError::new(other, self)),
        }
    }

    #[must_use]
    pub fn waker(&self) -> &Waker {
        &self.waker
    }

    #[cfg(feature = "dispatched")]
    fn resume_async_inner(&mut self, base_frame: usize) -> Result<Value, Fault> {
        if self.has_anonymous_frame {
            self.current_frame -= 1;
        }
        let mut code_frame = self.current_frame;
        let mut code = self.code[self.frames[code_frame].code.expect("missing frame code").0]
            .code
            .clone();
        loop {
            self.budget_and_yield()?;
            let instruction = self.frames[code_frame].instruction;
            match self.step(&code, instruction) {
                Ok(StepResult::Complete) if code_frame >= base_frame && code_frame > 0 => {
                    self.exit_frame()?;
                    if self.current_frame < base_frame {
                        break;
                    }
                }
                Ok(StepResult::Complete) => break,
                Ok(StepResult::NextAddress(addr)) => {
                    // Only step to the next address if the frame's instruction
                    // wasn't altered by a jump.
                    if self.frames[code_frame].instruction == instruction {
                        self.frames[code_frame].instruction = addr;
                    }
                }
                Err(err) => {
                    self.handle_fault(err, base_frame)?;
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

        if base_frame == 0 {
            self.execute_until = None;
        }
        Ok(self.registers[0].take())
    }

    #[allow(clippy::too_many_lines)]
    #[cfg(feature = "dispatched")]
    fn step(&mut self, code: &Code, address: usize) -> Result<StepResult, Fault> {
        use std::ops::ControlFlow;

        let instructions = &code.data.instructions;
        let instruction = match address.cmp(&instructions.len()) {
            Ordering::Less => &instructions[address],
            Ordering::Equal => return Ok(StepResult::Complete),
            Ordering::Greater => return Err(Fault::InvalidInstructionAddress),
        };
        let next_instruction = StepResult::from(address.checked_add(1));
        match instruction.execute(self) {
            Ok(ControlFlow::Continue(())) | Err(Fault::FrameChanged) => Ok(next_instruction),
            Ok(ControlFlow::Break(())) => Ok(StepResult::Complete),
            Err(err) => Err(err),
        }
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

    fn execute_function(
        &mut self,
        body: &Code,
        function: &Rooted<Function>,
        module: usize,
    ) -> Result<Value, Fault> {
        let body_index = self.push_code(body, Some(function));
        self.enter_frame(Some(body_index))?;
        let vm = self.vm();
        vm.frames[vm.current_frame].module = module;

        self.allocate(body.data.stack_requirement)?;

        Err(Fault::FrameChanged)
    }

    pub fn recurse_current_function(&mut self, arity: Arity) -> Result<Value, Fault> {
        let current_function = self.code[self.frames[self.current_frame]
            .code
            .expect("missing function")
            .0]
            .owner
            .as_ref()
            .map(Rooted::as_any_dynamic)
            .ok_or(Fault::NotAFunction)?;
        current_function.call(self, arity)
    }

    pub fn declare_variable(&mut self, name: SymbolRef, mutable: bool) -> Result<Stack, Fault> {
        let vm = self.vm();
        let current_frame = &mut vm.frames[vm.current_frame];
        if current_frame.end < vm.max_stack {
            Ok(current_frame
                .variables
                .entry(name)
                .or_insert_with(|| {
                    let index = Stack(current_frame.end - current_frame.start);
                    current_frame.end += 1;
                    BlockDeclaration {
                        stack: index,
                        mutable,
                    }
                })
                .stack)
        } else {
            Err(Fault::StackOverflow)
        }
    }

    pub fn declare(
        &mut self,
        name: impl Into<SymbolRef>,
        value: Value,
    ) -> Result<Option<Value>, Fault> {
        self.declare_inner(name, value, false)
    }

    pub fn declare_mut(
        &mut self,
        name: impl Into<SymbolRef>,
        value: Value,
    ) -> Result<Option<Value>, Fault> {
        self.declare_inner(name, value, true)
    }

    fn declare_inner(
        &mut self,
        name: impl Into<SymbolRef>,
        value: Value,
        mutable: bool,
    ) -> Result<Option<Value>, Fault> {
        match self.modules[self.frames[self.current_frame].module]
            .load(self.guard)
            .ok_or(Fault::ValueFreed)?
            .declarations()
            .entry(name.into())
        {
            Entry::Occupied(mut field) if field.mutable => {
                Ok(Some(std::mem::replace(&mut field.value, value)))
            }
            Entry::Occupied(_) => Err(Fault::NotMutable),
            Entry::Vacant(entry) => {
                entry.insert(ModuleDeclaration { mutable, value });
                Ok(None)
            }
        }
    }

    pub fn declare_function(&mut self, mut function: Function) -> Result<Option<Value>, Fault> {
        let Some(name) = function.name().clone() else {
            return Ok(None);
        };

        function.module = Some(0);
        self.declare_inner(name, Value::dynamic(function, &self), true)
    }

    pub fn resolve(&mut self, name: &Symbol) -> Result<Value, Fault> {
        if let Some(path) = name.strip_prefix('$') {
            let mut module_dynamic = self.modules[0];
            let mut module = module_dynamic.try_load(self.guard)?;
            let mut path = path.split('.').peekable();
            path.next();
            return if path.peek().is_none() {
                Ok(module_dynamic.to_value())
            } else {
                let name = loop {
                    let Some(name) = path.next() else {
                        return Err(Fault::UnknownSymbol);
                    };
                    let name = Symbol::from(name);
                    if path.peek().is_some() {
                        let declarations = module.declarations();
                        let value = &declarations
                            .get(&name.downgrade())
                            .ok_or(Fault::UnknownSymbol)?
                            .value;
                        let Some(inner) = value.as_dynamic::<Module>() else {
                            return Err(Fault::MissingModule);
                        };
                        drop(declarations);
                        module_dynamic = inner;
                        module = module_dynamic.try_load(self.guard)?;
                    } else {
                        // Final path component
                        break name;
                    }
                };

                Ok(module_dynamic
                    .try_load(self.guard)?
                    .declarations()
                    .get(&name.downgrade())
                    .ok_or(Fault::UnknownSymbol)?
                    .value)
            };
        }

        let current_frame = &self.frames[self.current_frame];
        if let Some(decl) = current_frame.variables.get(&name.downgrade()) {
            self.current_frame()
                .get(decl.stack.0)
                .copied()
                .ok_or(Fault::OutOfBounds)
        } else {
            let module =
                self.modules[self.frames[self.current_frame].module].try_load(self.guard)?;
            if let Some(value) = module
                .declarations()
                .get(&name.downgrade())
                .map(|decl| decl.value)
            {
                Ok(value)
            } else if name == Symbol::super_symbol() {
                Ok(module
                    .parent
                    .as_ref()
                    .map(Dynamic::to_value)
                    .unwrap_or_default())
            } else {
                Err(Fault::UnknownSymbol)
            }
        }
    }

    pub fn assign(&mut self, name: &SymbolRef, value: Value) -> Result<(), Fault> {
        let vm = &mut *self.vm;
        let current_frame = &mut vm.frames[vm.current_frame];
        if let Some(decl) = current_frame.variables.get_mut(name) {
            if decl.mutable {
                let stack = decl.stack;
                *self
                    .current_frame_mut()
                    .get_mut(stack.0)
                    .ok_or(Fault::OutOfBounds)? = value;
                Ok(())
            } else {
                Err(Fault::NotMutable)
            }
        } else {
            let module = &vm.modules[vm.frames[vm.current_frame].module];
            if let Some(decl) = module.try_load(self.guard)?.declarations().get_mut(name) {
                if decl.mutable {
                    decl.value = value;
                    Ok(())
                } else {
                    Err(Fault::NotMutable)
                }
            } else {
                Err(Fault::UnknownSymbol)
            }
        }
    }

    fn enter_frame(&mut self, code: Option<CodeIndex>) -> Result<(), Fault> {
        let vm = self.vm();
        if vm.current_frame < vm.max_depth {
            let current_frame_end = vm.frames[vm.current_frame].end;
            let current_frame_module = vm.frames[vm.current_frame].module;

            vm.current_frame += 1;
            if vm.current_frame < vm.frames.len() {
                vm.frames[vm.current_frame].clear();
                vm.frames[vm.current_frame].start = current_frame_end;
                vm.frames[vm.current_frame].end = current_frame_end;
                vm.frames[vm.current_frame].module = current_frame_module;
                vm.frames[vm.current_frame].code = code;
                vm.frames[vm.current_frame].instruction = 0;
            } else {
                vm.frames.push(Frame {
                    start: current_frame_end,
                    end: current_frame_end,
                    code,
                    instruction: 0,
                    variables: Map::new(),
                    module: current_frame_module,
                    loading_module: None,
                    exception_handler: None,
                });
            }
            Ok(())
        } else {
            Err(Fault::StackOverflow)
        }
    }

    pub fn exit_frame(&mut self) -> Result<(), Fault> {
        let vm = self.vm();
        if vm.current_frame >= 1 {
            vm.has_anonymous_frame = false;
            let current_frame = &vm.frames[vm.current_frame];
            vm.stack[current_frame.start..current_frame.end].fill_with(|| Value::Nil);
            vm.current_frame -= 1;
            Ok(())
        } else {
            Err(Fault::StackUnderflow)
        }
    }

    pub fn allocate(&mut self, count: usize) -> Result<Stack, Fault> {
        let vm = self.vm();
        let current_frame = &mut vm.frames[vm.current_frame];
        let index = Stack(current_frame.end - current_frame.start);
        match current_frame.end.checked_add(count) {
            Some(end) if end < vm.max_stack => {
                current_frame.end += count;
                if vm.stack.len() < current_frame.end {
                    vm.stack.resize_with(current_frame.end, || Value::Nil);
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
        let vm = self.vm();
        vm.frames[vm.current_frame].instruction = instruction;
    }

    #[must_use]
    pub fn current_frame(&self) -> &[Value] {
        &self.stack[self.frames[self.current_frame].start..self.frames[self.current_frame].end]
    }

    #[must_use]
    pub fn current_frame_mut(&mut self) -> &mut [Value] {
        let vm = self.vm();
        &mut vm.stack[vm.frames[vm.current_frame].start..vm.frames[vm.current_frame].end]
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

    #[must_use]
    pub fn stack_trace(&self) -> Vec<StackFrame> {
        self.frames[..=self.current_frame]
            .iter()
            .filter_map(|f| {
                f.code
                    .map(|index| StackFrame::new(self.code[index.0].code.clone(), f.instruction))
            })
            .collect()
    }

    fn handle_fault(&mut self, err: Fault, base_frame: usize) -> Result<(), Fault> {
        if err.is_execution_error() {
            return Err(err);
        }
        let exception = err.as_exception(self);
        let mut handled = false;
        while self.current_frame >= base_frame {
            if let Some(exception_target) = self.frames[self.current_frame].exception_handler {
                self[Register(0)] = err.as_exception(self);
                let vm = self.vm();
                vm.frames[vm.current_frame].instruction = exception_target.get();
                handled = true;
                break;
            }

            if self.current_frame == 0 {
                break;
            }

            self.current_frame -= 1;
        }

        if handled {
            Ok(())
        } else {
            Err(Fault::Exception(exception))
        }
    }
}

impl Index<Register> for VmState {
    type Output = Value;

    fn index(&self, index: Register) -> &Self::Output {
        &self.registers[usize::from(index)]
    }
}

impl IndexMut<Register> for VmState {
    fn index_mut(&mut self, index: Register) -> &mut Self::Output {
        &mut self.registers[usize::from(index)]
    }
}

impl Index<usize> for VmState {
    type Output = Value;

    fn index(&self, index: usize) -> &Self::Output {
        &self.stack[index]
    }
}

impl IndexMut<usize> for VmState {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.stack[index]
    }
}

pub struct VmContext<'a, 'guard> {
    guard: &'a mut CollectionGuard<'guard>,
    vm: MutexGuard<'a, VmState>,
}

impl<'a, 'guard> VmContext<'a, 'guard> {
    pub fn new(vm: &'a Vm, guard: &'a mut CollectionGuard<'guard>) -> Self {
        Self {
            guard,
            vm: vm.memory.0.lock(),
        }
    }

    #[must_use]
    pub fn guard_mut(&mut self) -> &mut CollectionGuard<'guard> {
        self.guard
    }

    #[must_use]
    pub fn guard(&self) -> &CollectionGuard<'guard> {
        self.guard
    }

    pub fn vm(&mut self) -> &mut VmState {
        &mut self.vm
    }

    fn budget_and_yield(&mut self) -> Result<(), Fault> {
        let next_count = self.counter - 1;
        if next_count > 0 {
            self.counter = next_count;
            Ok(())
        } else {
            self.counter = self.steps_per_charge;
            self.budget.charge()?;
            self.guard
                .coordinated_yield(|yielder| MutexGuard::unlocked(&mut self.vm, || yielder.wait()));
            self.check_timeout()
        }
    }
}

impl<'guard> AsRef<CollectionGuard<'guard>> for VmContext<'_, 'guard> {
    fn as_ref(&self) -> &CollectionGuard<'guard> {
        self.guard()
    }
}

impl Deref for VmContext<'_, '_> {
    type Target = VmState;

    fn deref(&self) -> &Self::Target {
        &self.vm
    }
}

impl DerefMut for VmContext<'_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.vm()
    }
}

#[cfg(not(feature = "dispatched"))]
impl VmContext<'_, '_> {
    fn resume_async_inner(&mut self, base_frame: usize) -> Result<Value, Fault> {
        if self.has_anonymous_frame {
            self.current_frame -= 1;
        }
        let mut code_frame = self.current_frame;
        let mut code = self.frames[code_frame].code.expect("missing frame code");
        loop {
            self.budget_and_yield()?;
            let instruction = self.frames[code_frame].instruction;
            match self.step(code.0, instruction) {
                Ok(StepResult::Complete) if code_frame >= base_frame && code_frame > 0 => {
                    self.exit_frame()?;
                    if self.current_frame < base_frame {
                        break;
                    }
                }
                Ok(StepResult::Complete) => break,
                Ok(StepResult::NextAddress(addr)) => {
                    // Only step to the next address if the frame's instruction
                    // wasn't altered by a jump.
                    if self.frames[code_frame].instruction == instruction {
                        self.frames[code_frame].instruction = addr;
                    }
                }
                Err(err) => {
                    self.handle_fault(err, base_frame)?;
                }
            }

            if code_frame != self.current_frame {
                code_frame = self.current_frame;
                code = self.frames[self.current_frame]
                    .code
                    .expect("missing frame code");
            }
        }

        if self.current_frame == 0 {
            self.execute_until = None;
        }
        Ok(self.registers[0].take())
    }

    #[allow(clippy::too_many_lines)]
    fn step(&mut self, code_index: usize, address: usize) -> Result<StepResult, Fault> {
        let code = &self.code[code_index].code;
        let instructions = &code.data.instructions;
        let instruction = match address.cmp(&instructions.len()) {
            Ordering::Less => &instructions[address],
            Ordering::Equal => return Ok(StepResult::Complete),
            Ordering::Greater => return Err(Fault::InvalidInstructionAddress),
        };
        // println!("Executing {instruction:?}");
        let next_instruction = StepResult::from(address.checked_add(1));
        let result = match instruction {
            LoadedOp::Return => return Ok(StepResult::Complete),
            LoadedOp::Declare {
                name,
                mutable,
                value,
                dest,
            } => self.op_declare(code_index, *name, *mutable, *value, *dest),
            LoadedOp::Call { name, arity } => self.op_call(code_index, *name, *arity),
            LoadedOp::Invoke {
                target,
                name,
                arity,
            } => self.op_invoke(code_index, *target, *name, *arity),
            LoadedOp::Truthy(loaded) => self.op_truthy(code_index, loaded.op, loaded.dest),
            LoadedOp::LogicalNot(loaded) => self.op_not(code_index, loaded.op, loaded.dest),
            LoadedOp::BitwiseNot(loaded) => self.op_bitwise_not(code_index, loaded.op, loaded.dest),
            LoadedOp::Negate(loaded) => self.op_negate(code_index, loaded.op, loaded.dest),
            LoadedOp::Copy(loaded) => self.op_copy(code_index, loaded.op, loaded.dest),
            LoadedOp::Resolve(loaded) => self.op_resolve(code_index, loaded.op, loaded.dest),
            LoadedOp::Jump(loaded) => self.op_jump(code_index, loaded.op, loaded.dest),
            LoadedOp::SetExceptionHandler(loaded) => {
                self.op_set_exception_handler(code_index, loaded.op, loaded.dest)
            }
            LoadedOp::LogicalXor(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| Ok(Value::Bool(lhs.truthy(vm) ^ rhs.truthy(vm))),
            ),
            LoadedOp::Assign(loaded) => {
                self.op_assign(code_index, loaded.op1, loaded.op2, loaded.dest)
            }
            LoadedOp::Add(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| lhs.add(vm, &rhs),
            ),
            LoadedOp::Subtract(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| lhs.sub(vm, &rhs),
            ),
            LoadedOp::Multiply(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| lhs.mul(vm, &rhs),
            ),
            LoadedOp::Divide(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| lhs.div(vm, &rhs),
            ),
            LoadedOp::IntegerDivide(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| lhs.idiv(vm, &rhs),
            ),
            LoadedOp::Remainder(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| lhs.rem(vm, &rhs),
            ),
            LoadedOp::Power(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| lhs.pow(vm, &rhs),
            ),
            LoadedOp::JumpIf(loaded) => {
                self.op_jump_if(code_index, loaded.op1, loaded.op2, loaded.dest, false)
            }
            LoadedOp::JumpIfNot(loaded) => {
                self.op_jump_if(code_index, loaded.op1, loaded.op2, loaded.dest, true)
            }
            LoadedOp::LessThanOrEqual(loaded) => {
                self.op_compare(code_index, loaded.op1, loaded.op2, loaded.dest, |ord| {
                    matches!(ord, Ordering::Less | Ordering::Equal)
                })
            }
            LoadedOp::LessThan(loaded) => {
                self.op_compare(code_index, loaded.op1, loaded.op2, loaded.dest, |ord| {
                    matches!(ord, Ordering::Less)
                })
            }
            LoadedOp::Equal(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| {
                    lhs.equals(ContextOrGuard::Context(vm), &rhs)
                        .map(Value::Bool)
                },
            ),
            LoadedOp::NotEqual(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| {
                    lhs.equals(ContextOrGuard::Context(vm), &rhs)
                        .map(|result| Value::Bool(!result))
                },
            ),
            LoadedOp::GreaterThan(loaded) => {
                self.op_compare(code_index, loaded.op1, loaded.op2, loaded.dest, |ord| {
                    matches!(ord, Ordering::Greater)
                })
            }
            LoadedOp::GreaterThanOrEqual(loaded) => {
                self.op_compare(code_index, loaded.op1, loaded.op2, loaded.dest, |ord| {
                    matches!(ord, Ordering::Greater | Ordering::Equal)
                })
            }
            LoadedOp::Matches(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| lhs.matches(vm, &rhs).map(Value::Bool),
            ),
            LoadedOp::BitwiseAnd(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| lhs.bitwise_and(vm, &rhs),
            ),
            LoadedOp::BitwiseOr(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| lhs.bitwise_or(vm, &rhs),
            ),
            LoadedOp::BitwiseXor(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| lhs.bitwise_xor(vm, &rhs),
            ),
            LoadedOp::BitwiseShiftLeft(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| lhs.shift_left(vm, &rhs),
            ),
            LoadedOp::BitwiseShiftRight(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| lhs.shift_right(vm, &rhs),
            ),
            LoadedOp::LoadModule { module, dest } => {
                self.op_load_module(code_index, *module, *dest)
            }
            LoadedOp::Throw(kind) => Err(Fault::from_kind(*kind, self)),
        };

        match result {
            Ok(()) | Err(Fault::FrameChanged) => Ok(next_instruction),
            Err(err) => Err(err),
        }
    }

    fn op_load_module(
        &mut self,
        code_index: usize,
        module: usize,
        dest: OpDestination,
    ) -> Result<(), Fault> {
        let vm = &mut *self.vm;
        let loading_module = if let Some(index) = vm.frames[vm.current_frame].loading_module.take()
        {
            index
        } else {
            // Replace the current module and stage the initializer
            let executing_frame = vm.current_frame;
            let initializer = vm.code[code_index].code.data.modules[module]
                .initializer
                .to_code(self.guard);
            let code = self.push_code(&initializer, None);
            self.enter_frame(Some(code))?;
            self.allocate(initializer.data.stack_requirement)?;
            let module_index = NonZeroUsize::new(self.modules.len()).expect("always at least one");
            let vm = &mut *self.vm;
            vm.modules.push(Dynamic::new(
                Module {
                    parent: Some(vm.modules[vm.frames[executing_frame].module]),
                    ..Module::default()
                },
                &*self.guard,
            ));
            vm.frames[vm.current_frame].module = module_index.get();
            vm.frames[executing_frame].loading_module = Some(module_index);
            let _init_result = self.resume_async_inner(self.current_frame)?;
            module_index
        };

        self.op_store(
            code_index,
            self.modules[loading_module.get()].to_value(),
            dest,
        )?;
        Ok(())
    }

    fn op_declare(
        &mut self,
        code_index: usize,
        name: usize,
        mutable: bool,
        value: LoadedSource,
        dest: OpDestination,
    ) -> Result<(), Fault> {
        let value = self.op_load(code_index, value)?;
        let name = self.code[code_index]
            .code
            .data
            .symbols
            .get(name)
            .cloned()
            .ok_or(Fault::InvalidOpcode)?;

        self.op_store(code_index, value, dest)?;
        self.declare_inner(name, value, mutable)?;
        Ok(())
    }

    fn op_bitwise_not(
        &mut self,
        code_index: usize,
        value: LoadedSource,
        dest: OpDestination,
    ) -> Result<(), Fault> {
        let value = self.op_load(code_index, value)?;
        let value = value.bitwise_not(self)?;
        self.op_store(code_index, value, dest)
    }

    fn op_truthy(
        &mut self,
        code_index: usize,
        value: LoadedSource,
        dest: OpDestination,
    ) -> Result<(), Fault> {
        let value = self.op_load(code_index, value)?;
        let value = Value::Bool(value.truthy(self));
        self.op_store(code_index, value, dest)
    }

    fn op_not(
        &mut self,
        code_index: usize,
        value: LoadedSource,
        dest: OpDestination,
    ) -> Result<(), Fault> {
        let value = self.op_load(code_index, value)?;
        let value = value.not(self)?;
        self.op_store(code_index, value, dest)
    }

    fn op_negate(
        &mut self,
        code_index: usize,
        value: LoadedSource,
        dest: OpDestination,
    ) -> Result<(), Fault> {
        let value = self.op_load(code_index, value)?;
        let value = value.negate(self)?;
        self.op_store(code_index, value, dest)
    }

    fn op_copy(
        &mut self,
        code_index: usize,
        value: LoadedSource,
        dest: OpDestination,
    ) -> Result<(), Fault> {
        let value = self.op_load(code_index, value)?;
        self.op_store(code_index, value, dest)
    }

    fn op_resolve(
        &mut self,
        code_index: usize,
        value: LoadedSource,
        dest: OpDestination,
    ) -> Result<(), Fault> {
        let Some(name) = self.op_load(code_index, value)?.as_symbol(self.guard) else {
            return Err(Fault::ExpectedSymbol);
        };

        let resolved = self.resolve(&name)?;
        self.op_store(code_index, resolved, dest)
    }

    fn op_call(
        &mut self,
        code_index: usize,
        function: LoadedSource,
        arity: LoadedSource,
    ) -> Result<(), Fault> {
        let (function, arity) = try_all!(
            self.op_load(code_index, function),
            self.op_load(code_index, arity)
        );

        let Some(arity) = arity.as_u64().and_then(|arity| u8::try_from(arity).ok()) else {
            return Err(Fault::InvalidArity);
        };

        self[Register(0)] = function.call(self, arity)?;

        Ok(())
    }

    fn op_invoke(
        &mut self,
        code_index: usize,
        target: LoadedSource,
        name: usize,
        arity: LoadedSource,
    ) -> Result<(), Fault> {
        let (target, arity, name) = try_all!(
            self.op_load(code_index, target),
            self.op_load(code_index, arity),
            self.op_load_symbol(code_index, name)
        );

        let Some(arity) = arity.as_u64().and_then(|arity| u8::try_from(arity).ok()) else {
            return Err(Fault::InvalidArity);
        };

        self[Register(0)] = target.invoke(self, &name.downgrade(), arity)?;

        Ok(())
    }

    fn op_compare(
        &mut self,
        code_index: usize,
        lhs: LoadedSource,
        rhs: LoadedSource,
        dest: OpDestination,
        op: impl FnOnce(Ordering) -> bool,
    ) -> Result<(), Fault> {
        self.op_binop(code_index, lhs, rhs, dest, |vm, lhs, rhs| {
            lhs.total_cmp(vm, &rhs).map(|ord| Value::Bool(op(ord)))
        })
    }

    fn op_binop(
        &mut self,
        code_index: usize,
        lhs: LoadedSource,
        rhs: LoadedSource,
        dest: OpDestination,
        op: impl FnOnce(&mut Self, Value, Value) -> Result<Value, Fault>,
    ) -> Result<(), Fault> {
        let (s1, s2) = try_all!(self.op_load(code_index, lhs), self.op_load(code_index, rhs));

        let result = op(self, s1, s2)?;

        self.op_store(code_index, result, dest)
    }

    fn op_assign(
        &mut self,
        code_index: usize,
        target: LoadedSource,
        value: LoadedSource,
        dest: OpDestination,
    ) -> Result<(), Fault> {
        let (target, value) = try_all!(
            self.op_load(code_index, target),
            self.op_load(code_index, value)
        );
        let Some(target) = target.as_symbol_ref() else {
            return Err(Fault::ExpectedSymbol);
        };

        self.op_store(code_index, value, dest)?;
        self.assign(target, value)
    }

    fn op_jump(
        &mut self,
        code_index: usize,
        target: LoadedSource,
        previous_instruction: OpDestination,
    ) -> Result<(), Fault> {
        let target = self.op_load(code_index, target)?;

        let current_instruction = self.current_instruction();
        self.jump_to(target.as_usize().ok_or(Fault::InvalidLabel)?);
        self.op_store(
            code_index,
            Value::try_from(current_instruction)?,
            previous_instruction,
        )
    }

    fn op_jump_if(
        &mut self,
        code_index: usize,
        target: LoadedSource,
        condition: LoadedSource,
        previous_instruction: OpDestination,
        not: bool,
    ) -> Result<(), Fault> {
        let (target, condition) = try_all!(
            self.op_load(code_index, target),
            self.op_load(code_index, condition)
        );

        let mut condition = condition.truthy(self);
        if not {
            condition = !condition;
        }

        if condition {
            let current_instruction = self.current_instruction();
            self.jump_to(target.as_usize().ok_or(Fault::InvalidLabel)?);
            self.op_store(
                code_index,
                Value::try_from(current_instruction)?,
                previous_instruction,
            )
        } else {
            Ok(())
        }
    }

    fn op_set_exception_handler(
        &mut self,
        code_index: usize,
        handler: LoadedSource,
        previous_handler: OpDestination,
    ) -> Result<(), Fault> {
        let handler = self
            .op_load(code_index, handler)?
            .as_usize()
            .and_then(NonZeroUsize::new);

        let vm = self.vm();
        let previous_handler_address =
            std::mem::replace(&mut vm.frames[vm.current_frame].exception_handler, handler);
        self.op_store(
            code_index,
            previous_handler_address
                .and_then(|addr| Value::try_from(addr.get()).ok())
                .unwrap_or_default(),
            previous_handler,
        )
    }

    fn op_load_symbol(&mut self, code_index: usize, symbol: usize) -> Result<Symbol, Fault> {
        self.code[code_index]
            .code
            .data
            .symbols
            .get(symbol)
            .cloned()
            .ok_or(Fault::InvalidOpcode)
    }

    fn op_load(&mut self, code_index: usize, value: LoadedSource) -> Result<Value, Fault> {
        match value {
            LoadedSource::Nil => Ok(Value::Nil),
            LoadedSource::Bool(v) => Ok(Value::Bool(v)),
            LoadedSource::Int(v) => Ok(Value::Int(v)),
            LoadedSource::UInt(v) => Ok(Value::UInt(v)),
            LoadedSource::Float(v) => Ok(Value::Float(v)),
            LoadedSource::Symbol(v) => self
                .op_load_symbol(code_index, v)
                .map(|s| Value::Symbol(s.downgrade())),
            LoadedSource::Register(v) => Ok(self[v]),
            LoadedSource::Stack(v) => self
                .current_frame()
                .get(v.0)
                .copied()
                .ok_or(Fault::InvalidOpcode),
            LoadedSource::Label(v) => self.code[code_index]
                .code
                .data
                .labels
                .get(v.0)
                .and_then(|label| u64::try_from(*label).ok())
                .map(Value::UInt)
                .ok_or(Fault::InvalidOpcode),
            LoadedSource::Regex(v) => self.code[code_index]
                .code
                .data
                .regexes
                .get(v)
                .ok_or(Fault::InvalidOpcode)
                .and_then(|regex| regex.result.clone()),
            LoadedSource::Function(v) => self.code[code_index]
                .code
                .data
                .functions
                .get(v)
                .map(|function| {
                    Value::dynamic(
                        function
                            .to_function(self.guard)
                            .in_module(self.frames[self.current_frame].module),
                        &self,
                    )
                })
                .ok_or(Fault::InvalidOpcode),
        }
    }

    fn op_store<'a>(
        &mut self,
        code_index: usize,
        value: impl Into<MaybeOwnedValue<'a>>,
        dest: OpDestination,
    ) -> Result<(), Fault> {
        let value = value.into();
        match dest {
            OpDestination::Void => Ok(()),
            OpDestination::Register(register) => {
                self[register] = value.into_owned();
                Ok(())
            }
            OpDestination::Stack(stack) => {
                if let Some(stack) = self.current_frame_mut().get_mut(stack.0) {
                    *stack = value.into_owned();
                    Ok(())
                } else {
                    Err(Fault::InvalidOpcode)
                }
            }
            OpDestination::Label(label) => {
                if value.truthy(self) {
                    let instruction = self.code[code_index]
                        .code
                        .data
                        .labels
                        .get(label.0)
                        .ok_or(Fault::InvalidLabel)?;
                    self.jump_to(*instruction);
                }
                Ok(())
            }
        }
    }
}

enum MaybeOwnedValue<'a> {
    Ref(&'a Value),
    Owned(Value),
}

#[cfg(not(feature = "dispatched"))]
impl MaybeOwnedValue<'_> {
    fn into_owned(self) -> Value {
        match self {
            MaybeOwnedValue::Ref(value) => *value,
            MaybeOwnedValue::Owned(value) => value,
        }
    }
}

impl Deref for MaybeOwnedValue<'_> {
    type Target = Value;

    fn deref(&self) -> &Self::Target {
        match self {
            MaybeOwnedValue::Ref(value) => value,
            MaybeOwnedValue::Owned(value) => value,
        }
    }
}

impl<'a> From<&'a Value> for MaybeOwnedValue<'a> {
    fn from(value: &'a Value) -> Self {
        Self::Ref(value)
    }
}

impl From<Value> for MaybeOwnedValue<'_> {
    fn from(value: Value) -> Self {
        Self::Owned(value)
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

#[derive(Default, Debug, Clone)]
struct Frame {
    start: usize,
    end: usize,
    instruction: usize,
    code: Option<CodeIndex>,
    variables: Map<SymbolRef, BlockDeclaration>,
    module: usize,
    loading_module: Option<NonZeroUsize>,
    exception_handler: Option<NonZeroUsize>,
}

impl Frame {
    fn clear(&mut self) {
        self.variables.clear();
        self.instruction = usize::MAX;
        self.code = None;
        self.module = 0;
        self.loading_module = None;
        self.exception_handler = None;
    }
}

#[derive(Debug, PartialEq, Clone, Trace)]
pub enum ExecutionError {
    NoBudget,
    Waiting,
    Timeout,
    Exception(Value),
}

impl ExecutionError {
    fn new(fault: Fault, vm: &mut VmContext<'_, '_>) -> Self {
        match fault {
            Fault::NoBudget => Self::NoBudget,
            Fault::Waiting => Self::Waiting,
            Fault::Timeout => Self::Timeout,
            Fault::Exception(exc) => Self::Exception(exc),
            other => Self::Exception(other.as_exception(vm)),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Fault {
    UnknownSymbol,
    IncorrectNumberOfArguments,
    PatternMismatch,
    OperationOnNil,
    ValueFreed,
    MissingModule,
    NotAFunction,
    StackOverflow,
    StackUnderflow,
    UnsupportedOperation,
    OutOfMemory,
    OutOfBounds,
    NotMutable,
    DivideByZero,
    InvalidInstructionAddress,
    ExpectedSymbol,
    ExpectedInteger,
    ExpectedString,
    InvalidArity,
    InvalidLabel,
    InvalidOpcode,
    NoBudget,
    Timeout,
    Waiting,
    FrameChanged,
    Exception(Value),
}

impl Fault {
    fn is_execution_error(&self) -> bool {
        matches!(self, Fault::NoBudget | Fault::Waiting | Fault::Timeout)
    }

    #[must_use]
    pub fn as_exception(&self, vm: &mut VmContext<'_, '_>) -> Value {
        let exception = match self {
            Fault::UnknownSymbol => Symbol::from("undefined").into(),
            Fault::IncorrectNumberOfArguments => Symbol::from("args").into(),
            Fault::OperationOnNil => Symbol::from("nil").into(),
            Fault::ValueFreed => Symbol::from("out-of-scope").into(),
            Fault::NotAFunction => Symbol::from("not_invokable").into(),
            Fault::MissingModule => Symbol::from("internal").into(),
            Fault::StackOverflow => Symbol::from("overflow").into(),
            Fault::StackUnderflow => Symbol::from("underflow").into(),
            Fault::UnsupportedOperation => Symbol::from("unsupported").into(),
            Fault::OutOfMemory => Symbol::from("out_of_memory").into(),
            Fault::OutOfBounds => Symbol::from("out_of_bounds").into(),
            Fault::NotMutable => Symbol::from("immutable").into(),
            Fault::DivideByZero => Symbol::from("divided_by_zero").into(),
            Fault::InvalidInstructionAddress => Symbol::from("invalid_instruction").into(),
            Fault::ExpectedSymbol => Symbol::from("expected_symbol").into(),
            Fault::ExpectedInteger => Symbol::from("expected_integer").into(),
            Fault::ExpectedString => Symbol::from("expected_string").into(),
            Fault::InvalidArity => Symbol::from("invalid_arity").into(),
            Fault::InvalidLabel => Symbol::from("invalid_label").into(),
            Fault::InvalidOpcode => Symbol::from("invalid_opcode").into(),
            Fault::NoBudget => Symbol::from("no_budget").into(),
            Fault::Timeout => Symbol::from("timeout").into(),
            Fault::Waiting => Symbol::from("waiting").into(),
            Fault::FrameChanged => Symbol::from("frame_changed").into(),
            Fault::Exception(value) => return *value,
            Fault::PatternMismatch => Symbol::from("mismatch").into(),
        };
        Value::dynamic(Exception::new(exception, vm), vm)
    }

    fn from_kind(kind: FaultKind, vm: &mut VmContext<'_, '_>) -> Self {
        let exception = match kind {
            FaultKind::Exception => Value::dynamic(Exception::new(vm[Register(0)].take(), vm), vm),
            FaultKind::PatternMismatch => Self::PatternMismatch.as_exception(vm),
        };
        Self::Exception(exception)
    }
}

#[derive(Debug, Clone)]
pub struct Function {
    module: Option<usize>,
    name: Option<Symbol>,
    bodies: Map<Arity, Code>,
    varg_bodies: Map<Arity, Code>,
}

impl Function {
    #[must_use]
    pub fn new(name: impl IntoOptionSymbol) -> Self {
        Self {
            module: None,
            name: name.into_symbol(),
            bodies: Map::new(),
            varg_bodies: Map::new(),
        }
    }

    fn in_module(mut self, module: usize) -> Self {
        self.module = Some(module);
        self
    }

    pub fn insert_arity(&mut self, arity: impl Into<Arity>, body: Code) {
        self.bodies.insert(arity.into(), body);
    }

    #[must_use]
    pub fn when(mut self, arity: impl Into<Arity>, body: Code) -> Self {
        self.insert_arity(arity, body);
        self
    }

    #[must_use]
    pub const fn name(&self) -> &Option<Symbol> {
        &self.name
    }
}

impl CustomType for Function {
    fn muse_type(&self) -> &crate::value::TypeRef {
        static TYPE: RustType<Function> = RustType::new("Function", |t| {
            t.with_call(|_| {
                |this, vm, arity| {
                    if let Some(body) = this.bodies.get(&arity).or_else(|| {
                        this.varg_bodies
                            .iter()
                            .rev()
                            .find_map(|va| (va.key() <= &arity).then_some(&va.value))
                    }) {
                        let module = this.module.ok_or(Fault::MissingModule)?;
                        vm.execute_function(body, &this, module)
                    } else {
                        Err(Fault::IncorrectNumberOfArguments)
                    }
                }
            })
            .with_clone()
        });
        &TYPE
    }
}

impl ContainsNoRefs for Function {}

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

impl TryFrom<usize> for Arity {
    type Error = TryFromIntError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        u8::try_from(value).map(Self)
    }
}

#[derive(Debug, Clone)]
pub struct Code {
    data: Arc<CodeData>,
}

impl Code {
    pub fn push(&mut self, op: &Op, range: SourceRange, guard: &CollectionGuard) {
        Arc::make_mut(&mut self.data).push(op, range, guard);
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

impl PartialEq for Code {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }
}

#[cfg(feature = "dispatched")]
type Inst = Arc<dyn dispatched::Instruction>;

#[cfg(not(feature = "dispatched"))]
type Inst = LoadedOp;

#[derive(Default, Debug, Clone)]
struct CodeData {
    instructions: Vec<Inst>,
    labels: Vec<usize>,
    regexes: Vec<PrecompiledRegex>,
    stack_requirement: usize,
    symbols: Vec<Symbol>,
    known_symbols: AHashMap<Symbol, usize>,
    functions: Vec<BitcodeFunction>,
    modules: Vec<BitcodeModule>,
    map: SourceMap,
}

impl CodeData {
    #[allow(clippy::too_many_lines)]
    pub fn push(&mut self, op: &Op, range: SourceRange, guard: &CollectionGuard) {
        match op {
            Op::Return => self.push_loaded(LoadedOp::Return, range, guard),
            Op::Label(label) => {
                if self.labels.len() <= label.0 {
                    self.labels.resize(label.0 + 1, usize::MAX);
                }
                self.labels[label.0] = self.instructions.len();
            }
            Op::Declare {
                name,
                mutable,
                value,
                dest,
            } => {
                let name = self.push_symbol(name.clone());
                let value = self.load_source(value, guard);
                let dest = self.load_dest(dest);
                self.push_loaded(
                    LoadedOp::Declare {
                        name,
                        mutable: *mutable,
                        value,
                        dest,
                    },
                    range,
                    guard,
                );
            }
            Op::Unary {
                dest: OpDestination::Void,
                kind: UnaryKind::Copy,
                ..
            } => {}
            Op::Unary { op, dest, kind } => {
                let op = self.load_source(op, guard);
                let dest = self.load_dest(dest);
                let unary = LoadedUnary { op, dest };
                self.push_loaded(
                    match kind {
                        UnaryKind::Truthy => LoadedOp::Truthy(unary),
                        UnaryKind::LogicalNot => LoadedOp::LogicalNot(unary),
                        UnaryKind::BitwiseNot => LoadedOp::BitwiseNot(unary),
                        UnaryKind::Negate => LoadedOp::Negate(unary),
                        UnaryKind::Copy => LoadedOp::Copy(unary),
                        UnaryKind::Resolve => LoadedOp::Resolve(unary),
                        UnaryKind::Jump => LoadedOp::Jump(unary),
                        UnaryKind::SetExceptionHandler => LoadedOp::SetExceptionHandler(unary),
                    },
                    range,
                    guard,
                );
            }
            Op::BinOp {
                op1,
                op2,
                dest,
                kind,
            } => {
                let op1 = self.load_source(op1, guard);
                let op2 = self.load_source(op2, guard);
                let dest = self.load_dest(dest);
                let binary = LoadedBinary { op1, op2, dest };
                self.push_loaded(
                    match kind {
                        BinaryKind::Add => LoadedOp::Add(binary),
                        BinaryKind::Subtract => LoadedOp::Subtract(binary),
                        BinaryKind::Multiply => LoadedOp::Multiply(binary),
                        BinaryKind::Divide => LoadedOp::Divide(binary),
                        BinaryKind::IntegerDivide => LoadedOp::IntegerDivide(binary),
                        BinaryKind::Remainder => LoadedOp::Remainder(binary),
                        BinaryKind::Power => LoadedOp::Power(binary),
                        BinaryKind::JumpIf => LoadedOp::JumpIf(binary),
                        BinaryKind::JumpIfNot => LoadedOp::JumpIfNot(binary),
                        BinaryKind::LogicalXor => LoadedOp::LogicalXor(binary),
                        BinaryKind::Assign => LoadedOp::Assign(binary),
                        BinaryKind::Matches => LoadedOp::Matches(binary),
                        BinaryKind::Bitwise(kind) => match kind {
                            BitwiseKind::And => LoadedOp::BitwiseAnd(binary),
                            BitwiseKind::Or => LoadedOp::BitwiseOr(binary),
                            BitwiseKind::Xor => LoadedOp::BitwiseXor(binary),
                            BitwiseKind::ShiftLeft => LoadedOp::BitwiseShiftLeft(binary),
                            BitwiseKind::ShiftRight => LoadedOp::BitwiseShiftRight(binary),
                        },
                        BinaryKind::Compare(kind) => match kind {
                            CompareKind::LessThanOrEqual => LoadedOp::LessThanOrEqual(binary),
                            CompareKind::LessThan => LoadedOp::LessThan(binary),
                            CompareKind::Equal => LoadedOp::Equal(binary),
                            CompareKind::NotEqual => LoadedOp::NotEqual(binary),
                            CompareKind::GreaterThan => LoadedOp::GreaterThan(binary),
                            CompareKind::GreaterThanOrEqual => LoadedOp::GreaterThanOrEqual(binary),
                        },
                    },
                    range,
                    guard,
                );
            }
            Op::Call { name, arity } => {
                let name = self.load_source(name, guard);
                let arity = self.load_source(arity, guard);
                self.push_loaded(LoadedOp::Call { name, arity }, range, guard);
            }
            Op::Invoke {
                target,
                name,
                arity,
            } => {
                let target = self.load_source(target, guard);
                let name = self.push_symbol(name.clone());
                let arity = self.load_source(arity, guard);
                self.push_loaded(
                    LoadedOp::Invoke {
                        target,
                        name,
                        arity,
                    },
                    range,
                    guard,
                );
            }
            Op::LoadModule { module, dest } => {
                let module = self.push_module(module);
                let dest = self.load_dest(dest);
                self.push_loaded(LoadedOp::LoadModule { module, dest }, range, guard);
            }
            Op::Throw(kind) => self.push_loaded(LoadedOp::Throw(*kind), range, guard),
        }
    }

    #[cfg(not(feature = "dispatched"))]
    fn push_loaded(&mut self, loaded: LoadedOp, range: SourceRange, _guard: &CollectionGuard<'_>) {
        self.instructions.push(loaded);
        self.map.push(range);
    }

    fn push_function(&mut self, function: BitcodeFunction) -> usize {
        let index = self.functions.len();
        self.functions.push(function);
        index
    }

    fn push_symbol(&mut self, symbol: Symbol) -> usize {
        *self.known_symbols.entry(symbol.clone()).or_insert_with(|| {
            let index = self.symbols.len();
            self.symbols.push(symbol);
            index
        })
    }

    fn push_module(&mut self, module: &BitcodeModule) -> usize {
        let index = self.modules.len();
        self.modules.push(module.clone());
        index
    }

    fn push_regex(&mut self, regex: &RegexLiteral, guard: &CollectionGuard) -> usize {
        let index = self.regexes.len();
        self.regexes.push(precompiled_regex(regex, guard));
        index
    }

    fn load_source(&mut self, source: &ValueOrSource, guard: &CollectionGuard) -> LoadedSource {
        match source {
            ValueOrSource::Nil => LoadedSource::Nil,
            ValueOrSource::Bool(bool) => LoadedSource::Bool(*bool),

            ValueOrSource::Int(int) => LoadedSource::Int(*int),
            ValueOrSource::UInt(uint) => LoadedSource::UInt(*uint),
            ValueOrSource::Float(float) => LoadedSource::Float(*float),
            ValueOrSource::Symbol(sym) => LoadedSource::Symbol(self.push_symbol(sym.clone())),
            ValueOrSource::Regex(regex) => LoadedSource::Regex(self.push_regex(regex, guard)),
            ValueOrSource::Register(reg) => LoadedSource::Register(*reg),
            ValueOrSource::Stack(stack) => {
                self.stack_requirement = self.stack_requirement.max(stack.0 + 1);
                LoadedSource::Stack(*stack)
            }
            ValueOrSource::Label(label) => LoadedSource::Label(*label),
            ValueOrSource::Function(function) => {
                let function = self.push_function(function.clone());
                LoadedSource::Function(function)
            }
        }
    }

    fn load_dest(&mut self, dest: &OpDestination) -> OpDestination {
        match *dest {
            OpDestination::Stack(stack) => {
                self.stack_requirement = self.stack_requirement.max(stack.0 + 1);
                OpDestination::Stack(stack)
            }
            other => other,
        }
    }
}

struct InstructionFormatter<'a>(&'a [u64]);

impl Debug for InstructionFormatter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut i = f.debug_list();
        let mut hex = String::new();
        for op in self.0 {
            hex.clear();
            write!(&mut hex, "{op:x}")?;
            i.entry(&hex);
        }
        i.finish()
    }
}

#[derive(Clone, Copy, Debug)]
enum StepResult {
    Complete,
    NextAddress(usize),
}

impl From<Option<usize>> for StepResult {
    fn from(value: Option<usize>) -> Self {
        value.map_or(StepResult::Complete, StepResult::NextAddress)
    }
}

#[derive(Default, Debug)]
pub struct Module {
    parent: Option<Dynamic<Module>>,
    declarations: Mutex<Map<SymbolRef, ModuleDeclaration>>,
}

impl Module {
    #[must_use]
    pub fn with_core(guard: &CollectionGuard) -> Dynamic<Self> {
        let module = Dynamic::new(Self::default(), guard);

        let core = Self {
            parent: Some(module),
            ..Self::core()
        };

        module
            .load(guard)
            .expect("guard held")
            .declarations()
            .insert(
                SymbolRef::from("core"),
                ModuleDeclaration {
                    mutable: false,
                    value: Value::dynamic(core, guard),
                },
            );

        module
    }

    pub fn core() -> Self {
        let core = Self::default();

        let mut declarations = core.declarations();
        declarations.insert(
            SymbolRef::from("Map"),
            ModuleDeclaration {
                mutable: false,
                value: Value::Dynamic(crate::map::MAP_TYPE.as_any_dynamic()),
            },
        );
        declarations.insert(
            SymbolRef::from("List"),
            ModuleDeclaration {
                mutable: false,
                value: Value::Dynamic(crate::list::LIST_TYPE.as_any_dynamic()),
            },
        );
        declarations.insert(
            SymbolRef::from("String"),
            ModuleDeclaration {
                mutable: false,
                value: Value::Dynamic(crate::string::STRING_TYPE.as_any_dynamic()),
            },
        );
        drop(declarations);

        core
    }

    fn declarations(&self) -> MutexGuard<'_, Map<SymbolRef, ModuleDeclaration>> {
        self.declarations.lock()
    }
}

impl CustomType for Module {
    fn muse_type(&self) -> &crate::value::TypeRef {
        static TYPE: RustType<Module> = RustType::new("Module", |t| {
            t.with_invoke(|_| {
                |this, vm, name, arity| {
                    static FUNCTIONS: StaticRustFunctionTable<Module> =
                        StaticRustFunctionTable::new(|table| {
                            table
                                .with_fn(Symbol::set_symbol(), 2, |vm, this| {
                                    let field = vm[Register(0)].take();
                                    let sym = field.as_symbol_ref().ok_or(Fault::ExpectedSymbol)?;
                                    let value = vm[Register(1)].take();

                                    match this.declarations().get_mut(sym) {
                                        Some(decl) if decl.mutable => {
                                            Ok(std::mem::replace(&mut decl.value, value))
                                        }
                                        Some(_) => Err(Fault::NotMutable),
                                        None => Err(Fault::UnknownSymbol),
                                    }
                                })
                                .with_fn(Symbol::get_symbol(), 1, |vm, this| {
                                    let field = vm[Register(0)].take();
                                    let sym = field.as_symbol_ref().ok_or(Fault::ExpectedSymbol)?;

                                    this.declarations()
                                        .get(sym)
                                        .map(|decl| decl.value)
                                        .ok_or(Fault::UnknownSymbol)
                                })
                        });
                    let declarations = this.declarations();
                    if let Some(decl) = declarations.get(name) {
                        let possible_invoke = decl.value;
                        drop(declarations);
                        possible_invoke.call(vm, arity)
                    } else {
                        drop(declarations);
                        FUNCTIONS.invoke(vm, name, arity, &this)
                    }
                }
            })
        });
        &TYPE
    }
}

impl Trace for Module {
    const MAY_CONTAIN_REFERENCES: bool = true;

    fn trace(&self, tracer: &mut refuse::Tracer) {
        if let Some(parent) = self.parent {
            tracer.mark(parent);
        }

        for decl in &*self.declarations() {
            if let Some(dynamic) = decl.value.value.as_any_dynamic() {
                tracer.mark(dynamic);
            }
        }
    }
}

#[derive(Debug)]
struct ModuleDeclaration {
    mutable: bool,
    value: Value,
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

pub struct ExecuteAsync<'context, 'guard>(VmContext<'context, 'guard>);

impl Future for ExecuteAsync<'_, '_> {
    type Output = Result<Value, ExecutionError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> Poll<Self::Output> {
        // Temporarily replace the VM's waker with this context's waker.
        let previous_waker = std::mem::replace(&mut self.0.waker, cx.waker().clone());
        let result = match self.0.resume_async_inner(0) {
            Err(Fault::Waiting) => Poll::Pending,
            Err(other) => Poll::Ready(Err(ExecutionError::new(other, &mut self.0))),
            Ok(value) => Poll::Ready(Ok(value)),
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

#[derive(Debug, Clone, Copy, PartialEq)]
enum LoadedOp {
    Return,
    Declare {
        name: usize,
        mutable: bool,
        value: LoadedSource,
        dest: OpDestination,
    },
    Truthy(LoadedUnary),
    LogicalNot(LoadedUnary),
    BitwiseNot(LoadedUnary),
    Negate(LoadedUnary),
    Copy(LoadedUnary),
    Resolve(LoadedUnary),
    Jump(LoadedUnary),
    SetExceptionHandler(LoadedUnary),
    LogicalXor(LoadedBinary),
    Assign(LoadedBinary),
    Add(LoadedBinary),
    Subtract(LoadedBinary),
    Multiply(LoadedBinary),
    Divide(LoadedBinary),
    IntegerDivide(LoadedBinary),
    Remainder(LoadedBinary),
    Power(LoadedBinary),
    JumpIf(LoadedBinary),
    JumpIfNot(LoadedBinary),
    LessThanOrEqual(LoadedBinary),
    LessThan(LoadedBinary),
    Equal(LoadedBinary),
    NotEqual(LoadedBinary),
    GreaterThan(LoadedBinary),
    GreaterThanOrEqual(LoadedBinary),
    Matches(LoadedBinary),
    Call {
        name: LoadedSource,
        arity: LoadedSource,
    },
    Invoke {
        target: LoadedSource,
        name: usize,
        arity: LoadedSource,
    },
    BitwiseAnd(LoadedBinary),
    BitwiseOr(LoadedBinary),
    BitwiseXor(LoadedBinary),
    BitwiseShiftLeft(LoadedBinary),
    BitwiseShiftRight(LoadedBinary),
    LoadModule {
        module: usize,
        dest: OpDestination,
    },
    Throw(FaultKind),
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LoadedUnary {
    op: LoadedSource,
    dest: OpDestination,
}

impl LoadedUnary {
    #[cfg(not(feature = "dispatched"))]
    fn as_op(&self, kind: UnaryKind, code: &Code) -> Op {
        Op::Unary {
            op: trusted_loaded_source_to_value(&self.op, &code.data),
            dest: self.dest,
            kind,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LoadedBinary {
    op1: LoadedSource,
    op2: LoadedSource,
    dest: OpDestination,
}

impl LoadedBinary {
    #[cfg(not(feature = "dispatched"))]
    fn as_op(&self, kind: BinaryKind, code: &Code) -> Op {
        Op::BinOp {
            op1: trusted_loaded_source_to_value(&self.op1, &code.data),
            op2: trusted_loaded_source_to_value(&self.op2, &code.data),
            dest: self.dest,
            kind,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum LoadedSource {
    Nil,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Symbol(usize),
    Register(Register),
    Function(usize),
    Stack(Stack),
    Label(Label),
    Regex(usize),
}

#[derive(Debug, Clone)]
struct PrecompiledRegex {
    literal: RegexLiteral,
    result: Result<Value, Fault>,
}
fn precompiled_regex(regex: &RegexLiteral, guard: &CollectionGuard) -> PrecompiledRegex {
    PrecompiledRegex {
        literal: regex.clone(),
        result: MuseRegex::new(regex)
            .map(|r| Value::dynamic(r, guard))
            .map_err(|err| {
                Fault::Exception(Value::dynamic(MuseString::from(err.to_string()), guard))
            }),
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Stack(pub usize);

#[derive(PartialEq, Clone)]
pub struct StackFrame {
    code: Code,
    instruction: usize,
}

impl StackFrame {
    #[must_use]
    pub fn new(code: Code, instruction: usize) -> Self {
        Self { code, instruction }
    }

    #[must_use]
    pub fn source_range(&self) -> Option<SourceRange> {
        self.code.data.map.get(self.instruction)
    }
}

impl Debug for StackFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StackFrame")
            .field("code", &Arc::as_ptr(&self.code.data))
            .field("instruction", &self.instruction)
            .finish()
    }
}

pub trait InvokeArgs {
    fn load(self, vm: &mut VmContext<'_, '_>) -> Result<Arity, ExecutionError>;
}

impl<const N: usize> InvokeArgs for [Value; N] {
    fn load(self, vm: &mut VmContext<'_, '_>) -> Result<Arity, ExecutionError> {
        let arity = Arity::try_from(N)
            .map_err(|_| ExecutionError::Exception(Fault::InvalidArity.as_exception(vm)))?;

        for (arg, register) in self.iter().zip(0..arity.0) {
            vm[Register(register)] = *arg;
        }

        Ok(arity)
    }
}
