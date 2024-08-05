//! Virtual Machine types for executing Muse code.
//!
//! Muse contains a compiler that generates an Intermediate Representation (IR),
//! and a virtual machine that load and execute the intermediate representation.
//!
//! These are the types that Muse uses that have both IR and loaded
//! representations:
//!
//! | IR | Loaded | Purpose |
//! |----|--------|--|
//! | [`BitcodeBlock`](bitcode::BitcodeBlock) | [`Code`] | An isolated block of code that can be executed. |
//! | [`BitcodeFunction`] | [`Function`] | A function definition. |
//! | [`BitcodeModule`] | [`Module`] | A module definition. |

use std::cmp::Ordering;
use std::fmt::Debug;
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
use kempt::Map;
use parking_lot::{Mutex, MutexGuard};
use refuse::{CollectionGuard, ContainsNoRefs, NoMapping, Root, Trace};
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "dispatched"))]
use self::bitcode::trusted_loaded_source_to_value;
use self::bitcode::{
    Access, BinaryKind, BitcodeFunction, FaultKind, Label, Op, OpDestination, ValueOrSource,
};
use crate::compiler::syntax::token::RegexLiteral;
use crate::compiler::syntax::{BitwiseKind, CompareKind, SourceCode, SourceRange};
use crate::compiler::{BitcodeModule, BlockDeclaration, Compiler, SourceMap, UnaryKind};
use crate::runtime::exception::Exception;
use crate::runtime::regex::MuseRegex;
use crate::runtime::symbol::{IntoOptionSymbol, Symbol, SymbolRef};
use crate::runtime::types::{BitcodeEnum, BitcodeStruct};
#[cfg(not(feature = "dispatched"))]
use crate::runtime::value::{ContextOrGuard, Primitive};
use crate::runtime::value::{
    CustomType, Dynamic, Rooted, RustType, StaticRustFunctionTable, Value,
};

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

/// The ID of a module loaded in a virtual machine.
///
/// Module IDs are not compatible between different virtual machine instances.
#[derive(Default, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct ModuleId(usize);

/// A virtual machine that executes compiled Muse [`Code`].
#[derive(Clone)]
pub struct Vm {
    memory: Root<VmMemory>,
}

impl Vm {
    /// Returns a new virtual machine.
    ///
    /// Virtual machines are allocated within the [`refuse`] garbage collector.
    /// Each virtual machine acts as a "root" reference to all of the values
    /// contained within its stack and registers.
    #[must_use]
    pub fn new(guard: &CollectionGuard) -> Self {
        let parker = Parker::new();
        let unparker = parker.unparker().clone();
        Self {
            memory: Root::new(
                VmMemory(Mutex::new(VmState {
                    registers: array::from_fn(|_| Value::nil()),
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

    /// Compiles `source`, executes it, and returns the result.
    ///
    /// When building an interactive environment like a REPL, reusing a
    /// [`Compiler`] and calling [`execute`](Self::execute) enables the compiler
    /// to remember declarations from previous compilations.
    pub fn compile_and_execute<'a>(
        &self,
        source: impl Into<SourceCode<'a>>,
        guard: &mut CollectionGuard,
    ) -> Result<Value, crate::Error> {
        let mut compiler = Compiler::default();
        compiler.push(source);
        let code = compiler.build(guard)?;
        Ok(self.execute(&code, guard)?)
    }

    /// Prepares to execute `code`.
    ///
    /// This function does not actually execute any code. A call to
    /// [`resume`](Self::resume)/[`resume_async`](Self::resume_async) is needed
    /// to begin executing the function call.
    pub fn prepare(&self, code: &Code, guard: &mut CollectionGuard) -> Result<(), ExecutionError> {
        self.context(guard).prepare(code)
    }

    /// Prepares to execute the function body with `arity` arguments.
    ///
    /// This function does not actually execute any code. A call to
    /// [`resume`](Self::resume)/[`resume_async`](Self::resume_async) is needed
    /// to begin executing the function call.
    pub fn prepare_call(
        &self,
        function: &Rooted<Function>,
        arity: Arity,
        guard: &mut CollectionGuard,
    ) -> Result<(), ExecutionError> {
        self.context(guard).prepare_call(function, arity)
    }

    /// Executes `code` and returns the result.
    pub fn execute(
        &self,
        code: &Code,
        guard: &mut CollectionGuard,
    ) -> Result<Value, ExecutionError> {
        self.context(guard).execute(code)
    }

    /// Executes `code` for at most `duration` before returning a timout.
    pub fn execute_for(
        &self,
        code: &Code,
        duration: Duration,
        guard: &mut CollectionGuard,
    ) -> Result<Value, ExecutionError> {
        self.context(guard).execute_for(code, duration)
    }

    /// Executes `code` for until `instant` before returning a timout.
    pub fn execute_until(
        &self,
        code: &Code,
        instant: Instant,
        guard: &mut CollectionGuard,
    ) -> Result<Value, ExecutionError> {
        self.context(guard).execute_until(code, instant)
    }

    /// Resumes executing the current code.
    ///
    /// This should only be called if an [`ExecutionError::Waiting`],
    /// [`ExecutionError::NoBudget`], or [`ExecutionError::Timeout`] was
    /// returned when executing code.
    pub fn resume(&self, guard: &mut CollectionGuard) -> Result<Value, ExecutionError> {
        self.context(guard).resume()
    }

    /// Resumes executing the currently executing code until `instant`.
    ///
    /// This should only be called if an [`ExecutionError::Waiting`],
    /// [`ExecutionError::NoBudget`], or [`ExecutionError::Timeout`] was
    /// returned when executing code.
    pub fn resume_until(
        &mut self,
        instant: Instant,
        guard: &mut CollectionGuard,
    ) -> Result<Value, ExecutionError> {
        self.context(guard).resume_until(instant)
    }

    /// Resumes executing the currently executing code until `duration` as
    /// elapsed.
    ///
    /// This should only be called if an [`ExecutionError::Waiting`],
    /// [`ExecutionError::NoBudget`], or [`ExecutionError::Timeout`] was
    /// returned when executing code.
    pub fn resume_for(
        &mut self,
        duration: Duration,
        guard: &mut CollectionGuard,
    ) -> Result<Value, ExecutionError> {
        self.context(guard).resume_for(duration)
    }

    /// Returns a future that executes `code` asynchronously.
    pub fn execute_async<'context, 'guard>(
        &'context self,
        code: &Code,
        guard: &'context mut CollectionGuard<'guard>,
    ) -> Result<ExecuteAsync<'static, 'context, 'guard>, ExecutionError> {
        MaybeOwnedContext::Owned(self.context(guard)).execute_async(code)
    }

    /// Resumes executing the current code asynchronously.
    ///
    /// This should only be called if an [`ExecutionError::Waiting`],
    /// [`ExecutionError::NoBudget`], or [`ExecutionError::Timeout`] was
    /// returned when executing code.
    pub fn resume_async<'context, 'guard>(
        &'context self,
        guard: &'context mut CollectionGuard<'guard>,
    ) -> ExecuteAsync<'static, 'context, 'guard> {
        MaybeOwnedContext::Owned(self.context(guard)).resume_async()
    }

    /// Resumes executing the currently executing code until `duration` as
    /// elapsed.
    ///
    /// This should only be called if an [`ExecutionError::Waiting`],
    /// [`ExecutionError::NoBudget`], or [`ExecutionError::Timeout`] was
    /// returned when executing code.
    pub fn resume_for_async<'context, 'guard>(
        &'context self,
        duration: Duration,
        guard: &'context mut CollectionGuard<'guard>,
    ) -> ExecuteAsync<'static, 'context, 'guard> {
        MaybeOwnedContext::Owned(self.context(guard)).resume_for_async(duration)
    }

    /// Resumes executing the currently executing code until `instant`.
    ///
    /// This should only be called if an [`ExecutionError::Waiting`],
    /// [`ExecutionError::NoBudget`], or [`ExecutionError::Timeout`] was
    /// returned when executing code.
    pub fn resume_until_async<'context, 'guard>(
        &'context self,
        instant: Instant,
        guard: &'context mut CollectionGuard<'guard>,
    ) -> ExecuteAsync<'static, 'context, 'guard> {
        MaybeOwnedContext::Owned(self.context(guard)).resume_until_async(instant)
    }

    /// Increases the current budget by `amount`.
    ///
    /// If the virtual machine currently is unbudgeted, calling this function
    /// enables budgeting.
    pub fn increase_budget(&self, amount: usize) {
        self.memory.0.lock().budget.allocate(amount);
    }

    /// Invokes a public function at path `name` with the given parameters.
    pub fn invoke(
        &self,
        name: impl Into<SymbolRef>,
        params: impl InvokeArgs,
        guard: &mut CollectionGuard,
    ) -> Result<Value, ExecutionError> {
        self.context(guard).invoke(name, params)
    }

    /// Returns an execution context that synchronizes with the garbage
    /// collector.
    pub fn context<'context, 'guard>(
        &'context self,
        guard: &'context mut CollectionGuard<'guard>,
    ) -> VmContext<'context, 'guard> {
        VmContext {
            guard,
            vm: self.memory.0.lock(),
        }
    }

    /// Sets the number of virtual machine steps to take per budget being
    /// charged.
    ///
    /// This also affects how often the virtual machine checks if it should
    /// yield to the garbage collector.
    pub fn set_steps_per_charge(&self, steps: u16) {
        self.memory.0.lock().set_steps_per_charge(steps);
    }

    /// Returns the value contained in `register`.
    #[must_use]
    pub fn register(&self, register: Register) -> Value {
        self.memory.0.lock()[register]
    }

    /// Replaces the current value in `register` with `value`.
    #[allow(clippy::must_use_candidate)]
    pub fn set_register(&self, register: Register, value: Value) -> Value {
        std::mem::replace(&mut self.memory.0.lock()[register], value)
    }

    /// Returns the value contained at `index` on the stack.
    ///
    /// # Panics
    ///
    /// This function panics if `index` is out of bounds of the stack.
    #[must_use]
    pub fn stack(&self, index: Stack) -> Value {
        self.memory.0.lock()[index.0]
    }

    /// Replaces the current value at `index` on the stack with `value`.
    ///
    /// # Panics
    ///
    /// This function panics if `index` is out of bounds of the stack.
    #[must_use]
    pub fn set_stack(&self, index: Stack, value: Value) -> Value {
        std::mem::replace(&mut self.memory.0.lock()[index.0], value)
    }

    /// Allocates a variable declaration.
    ///
    /// Returns a stack index that has been allocated. The Muse virtual machine
    /// ensures the stack is Nil-initialized.
    pub fn declare_variable(
        &self,
        name: SymbolRef,
        mutable: bool,
        guard: &mut CollectionGuard<'_>,
    ) -> Result<Stack, Fault> {
        VmContext::new(self, guard).declare_variable(name, mutable)
    }

    /// Declares an immutable variable with `name` containing `value`.
    pub fn declare(
        &self,
        name: impl Into<SymbolRef>,
        value: Value,
        guard: &mut CollectionGuard<'_>,
    ) -> Result<Option<Value>, Fault> {
        VmContext::new(self, guard).declare(name, value)
    }

    /// Declares an mutable variable with `name` containing `value`.
    pub fn declare_mut(
        &self,
        name: impl Into<SymbolRef>,
        value: Value,
        guard: &mut CollectionGuard<'_>,
    ) -> Result<Option<Value>, Fault> {
        VmContext::new(self, guard).declare_mut(name, value)
    }

    /// Declares a compiled function.
    ///
    /// Returns a reference to the function, or `None` if the function could not
    /// be declared because it has no name.
    pub fn declare_function(
        &self,
        function: Function,
        guard: &mut CollectionGuard<'_>,
    ) -> Result<Option<Value>, Fault> {
        VmContext::new(self, guard).declare_function(function)
    }
}

impl Debug for Vm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("Vm");
        if let Some(memory) = self.memory.0.try_lock() {
            s.field("stack_frames", &memory.frames.len())
                .field("budget", &memory.budget);
        }

        s.finish_non_exhaustive()
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

/// The state of a [`Vm`].
///
/// Virtual machines are garbage collected types, which requires interior
/// mutability so that the garbage collector can trace what every allocated
/// virtual machine is currently referencing.
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
    /// Sets the number of virtual machine steps to take per budget being
    /// charged.
    ///
    /// This also affects how often the virtual machine checks if it should
    /// yield to the garbage collector.
    pub fn set_steps_per_charge(&mut self, steps: u16) {
        self.steps_per_charge = steps;
        self.counter = self.counter.min(steps);
    }

    /// Returns a slice of the registers.
    #[must_use]
    pub const fn registers(&self) -> &[Value; 256] {
        &self.registers
    }

    /// Returns exclusive access to the registers.
    #[must_use]
    pub fn registers_mut(&mut self) -> &mut [Value; 256] {
        &mut self.registers
    }

    /// Returns the id of the module that owns the code currently executing.
    #[must_use]
    pub fn current_module(&self) -> ModuleId {
        self.frames[self.current_frame].module
    }

    /// Returns the root module for this virtual machine.
    pub fn root_module(&self) -> Dynamic<Module> {
        self.modules[0]
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

/// A virtual machine execution context.
///
/// While a [`VmContext`] is held and not executing code, the garbage collector
/// cannot run and the virtual machine is exclusivly accessible by the current
/// thread.
pub struct VmContext<'a, 'guard> {
    guard: &'a mut CollectionGuard<'guard>,
    vm: MutexGuard<'a, VmState>,
}

impl<'context, 'guard> VmContext<'context, 'guard> {
    /// Returns a new execution context for `vm` using `guard`.
    pub fn new(vm: &'context Vm, guard: &'context mut CollectionGuard<'guard>) -> Self {
        Self {
            guard,
            vm: vm.memory.0.lock(),
        }
    }

    /// Returns an exclusive reference to the collection guard.
    #[must_use]
    pub fn guard_mut(&mut self) -> &mut CollectionGuard<'guard> {
        self.guard
    }

    /// Returns a reference to the collection guard.
    #[must_use]
    pub fn guard(&self) -> &CollectionGuard<'guard> {
        self.guard
    }

    /// Returns a reference to the virtual machine.
    pub fn vm(&mut self) -> &mut VmState {
        &mut self.vm
    }

    /// Returns a new virtual machine with the same modules and registered code.
    ///
    /// All registers, stack values, and stack frames will be empty on the new
    /// Vm.
    pub fn cloned_vm(&self) -> Vm {
        let parker = Parker::new();
        let unparker = parker.unparker().clone();
        Vm {
            memory: Root::new(
                VmMemory(Mutex::new(VmState {
                    registers: array::from_fn(|_| Value::nil()),
                    stack: Vec::new(),
                    max_stack: self.vm.max_stack,
                    frames: vec![Frame::default()],
                    current_frame: 0,
                    has_anonymous_frame: false,
                    max_depth: self.vm.max_depth,
                    counter: self.vm.steps_per_charge,
                    steps_per_charge: self.vm.steps_per_charge,
                    budget: Budget::default(),
                    execute_until: None,
                    modules: self.vm.modules.clone(),
                    waker: Waker::from(Arc::new(VmWaker(unparker))),
                    parker,
                    code: self.vm.code.clone(),
                    code_map: self.vm.code_map.clone(),
                })),
                self.guard(),
            ),
        }
    }

    /// Returns the access to allow the caller of the current function.
    #[must_use]
    pub fn caller_access_level(&self, module: &Dynamic<Module>) -> Access {
        let current_module = &self.modules[self.frames[self.current_frame].module.0];
        if current_module == module {
            Access::Private
        } else {
            Access::Public
        }
    }

    /// Returns the access to allow the caller of the current function.
    pub(crate) fn caller_access_level_by_index(&self, module: ModuleId) -> Access {
        let module = self.modules[module.0];
        self.caller_access_level(&module)
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

    /// Executes `func` while the virtual machine is unlocked.
    ///
    /// This can be used in combination with [`CollectionGuard::while_unlocked`]
    /// to perform code while the garbage collector is free to execute.
    pub fn while_unlocked<R>(&mut self, func: impl FnOnce(&mut CollectionGuard<'_>) -> R) -> R {
        MutexGuard::unlocked(&mut self.vm, || func(self.guard))
    }

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

    /// Prepares to execute `code`.
    ///
    /// This function does not actually execute any code. A call to
    /// [`resume`](Self::resume)/[`resume_async`](Self::resume_async) is needed
    /// to begin executing the function call.
    pub fn prepare(&mut self, code: &Code) -> Result<(), ExecutionError> {
        let code = self.push_code(code, None);
        self.prepare_owned(code)
    }

    /// Executes `code` and returns the result.
    pub fn execute(&mut self, code: &Code) -> Result<Value, ExecutionError> {
        let code = self.push_code(code, None);
        self.execute_owned(code)
    }

    /// Executes `code` for at most `duration` before returning a timout.
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

    /// Executes `code` for until `instant` before returning a timout.
    pub fn execute_until(
        &mut self,
        code: &Code,
        instant: Instant,
    ) -> Result<Value, ExecutionError> {
        let code = self.push_code(code, None);
        self.execute_until = Some(instant);
        self.execute_owned(code)
    }

    /// Resumes executing the current code.
    ///
    /// This should only be called if an [`ExecutionError::Waiting`],
    /// [`ExecutionError::NoBudget`], or [`ExecutionError::Timeout`] was
    /// returned when executing code.
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

    /// Resumes executing the currently executing code until `instant`.
    ///
    /// This should only be called if an [`ExecutionError::Waiting`],
    /// [`ExecutionError::NoBudget`], or [`ExecutionError::Timeout`] was
    /// returned when executing code.
    pub fn resume_until(&mut self, instant: Instant) -> Result<Value, ExecutionError> {
        self.execute_until = Some(instant);
        self.resume()
    }

    /// Resumes executing the currently executing code until `duration` as
    /// elapsed.
    ///
    /// This should only be called if an [`ExecutionError::Waiting`],
    /// [`ExecutionError::NoBudget`], or [`ExecutionError::Timeout`] was
    /// returned when executing code.
    pub fn resume_for(&mut self, duration: Duration) -> Result<Value, ExecutionError> {
        self.resume_until(Instant::now() + duration)
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

    /// Returns a future that executes `code` asynchronously.
    pub fn execute_async<'vm>(
        &'vm mut self,
        code: &Code,
    ) -> Result<ExecuteAsync<'vm, 'context, 'guard>, ExecutionError> {
        let code = self.push_code(code, None);
        self.prepare_owned(code)?;

        Ok(ExecuteAsync(MaybeOwnedContext::Borrowed(self)))
    }

    /// Resumes executing the current code asynchronously.
    ///
    /// This should only be called if an [`ExecutionError::Waiting`],
    /// [`ExecutionError::NoBudget`], or [`ExecutionError::Timeout`] was
    /// returned when executing code.
    pub fn resume_async<'vm>(&'vm mut self) -> ExecuteAsync<'vm, 'context, 'guard> {
        ExecuteAsync(MaybeOwnedContext::Borrowed(self))
    }

    /// Resumes executing the currently executing code until `duration` as
    /// elapsed.
    ///
    /// This should only be called if an [`ExecutionError::Waiting`],
    /// [`ExecutionError::NoBudget`], or [`ExecutionError::Timeout`] was
    /// returned when executing code.
    pub fn resume_for_async<'vm>(
        &'vm mut self,
        duration: Duration,
    ) -> ExecuteAsync<'vm, 'context, 'guard> {
        self.resume_until_async(Instant::now() + duration)
    }

    /// Resumes executing the currently executing code until `instant`.
    ///
    /// This should only be called if an [`ExecutionError::Waiting`],
    /// [`ExecutionError::NoBudget`], or [`ExecutionError::Timeout`] was
    /// returned when executing code.
    pub fn resume_until_async<'vm>(
        &'vm mut self,
        instant: Instant,
    ) -> ExecuteAsync<'vm, 'context, 'guard> {
        self.execute_until = Some(instant);
        self.resume_async()
    }

    /// Increases the current budget by `amount`.
    ///
    /// If the virtual machine currently is unbudgeted, calling this function
    /// enables budgeting.
    pub fn increase_budget(&mut self, amount: usize) {
        self.budget.allocate(amount);
    }

    /// Prepares to execute the function body with `arity` arguments.
    ///
    /// This function does not actually execute any code. A call to
    /// [`resume`](Self::resume)/[`resume_async`](Self::resume_async) is needed
    /// to begin executing the function call.
    pub fn prepare_call(
        &mut self,
        function: &Rooted<Function>,
        arity: Arity,
    ) -> Result<(), ExecutionError> {
        let Some(body) = function.body(arity) else {
            return Err(ExecutionError::Exception(
                Fault::IncorrectNumberOfArguments.as_exception(self),
            ));
        };
        let code = self.push_code(body, Some(function));
        self.prepare_owned(code)
    }

    /// Invokes a public function at path `name` with the given parameters.
    pub fn invoke(
        &mut self,
        name: impl Into<SymbolRef>,
        params: impl InvokeArgs,
    ) -> Result<Value, ExecutionError> {
        self.invoke_inner(&name.into(), params)
    }

    fn invoke_inner(
        &mut self,
        name: &SymbolRef,
        params: impl InvokeArgs,
    ) -> Result<Value, ExecutionError> {
        let arity = params.load(self)?;

        let mut module_dynamic = self.modules[self.frames[self.current_frame].module.0]
            .as_rooted(self.guard)
            .expect("module missing");
        let mut module_declarations = module_dynamic.declarations();
        let function = if let Some(decl) = module_declarations.get(name) {
            if decl.access == Access::Public {
                decl.value
            } else {
                return Err(ExecutionError::new(Fault::UnknownSymbol, self));
            }
        } else {
            let name = name.try_load(self.guard)?;
            let mut parts = name.split('.').peekable();
            while let Some(part) = parts.next() {
                let part = SymbolRef::from(part);
                let Some(decl) = module_declarations
                    .get(&part)
                    .and_then(|decl| (decl.access == Access::Public).then_some(decl.value))
                else {
                    break;
                };
                if parts.peek().is_some() {
                    let Some(contained_module) = decl.as_rooted::<Module>(self.guard) else {
                        return Err(ExecutionError::new(Fault::UnknownSymbol, self));
                    };
                    drop(module_declarations);
                    module_dynamic = contained_module;
                    module_declarations = module_dynamic.declarations();
                } else {
                    drop(module_declarations);
                    return match decl.call(self, arity) {
                        Ok(result) => Ok(result),
                        Err(Fault::FrameChanged | Fault::Waiting) => self.resume(),
                        Err(other) => Err(ExecutionError::new(other, self)),
                    };
                }
            }
            return Err(ExecutionError::new(Fault::UnknownSymbol, self));
        };
        drop(module_declarations);

        let Some(function) = function.as_any_dynamic() else {
            return Err(ExecutionError::new(Fault::NotAFunction, self));
        };

        match function.call(self, arity) {
            Ok(value) => Ok(value),
            Err(Fault::FrameChanged) => self.resume(),
            Err(other) => Err(ExecutionError::new(other, self)),
        }
    }

    /// Returns a waker that will wake this virtual machine context when waiting
    /// on async tasks.
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

    pub(crate) fn enter_anonymous_frame(&mut self) -> Result<(), Fault> {
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
        module: ModuleId,
    ) -> Result<Value, Fault> {
        let body_index = self.push_code(body, Some(function));
        self.enter_frame(Some(body_index))?;
        let vm = self.vm();
        vm.frames[vm.current_frame].module = module;

        self.allocate(body.data.stack_requirement)?;

        Err(Fault::FrameChanged)
    }

    pub(crate) fn recurse_current_function(&mut self, arity: Arity) -> Result<Value, Fault> {
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

    /// Allocates a variable declaration.
    ///
    /// Returns a stack index that has been allocated. The Muse virtual machine
    /// ensures the stack is Nil-initialized.
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

    /// Declares an immutable variable with `name` containing `value`.
    pub fn declare(
        &mut self,
        name: impl Into<SymbolRef>,
        value: Value,
    ) -> Result<Option<Value>, Fault> {
        self.declare_inner(name, value, false, Access::Public)
    }

    /// Declares an mutable variable with `name` containing `value`.
    pub fn declare_mut(
        &mut self,
        name: impl Into<SymbolRef>,
        value: Value,
    ) -> Result<Option<Value>, Fault> {
        self.declare_inner(name, value, true, Access::Public)
    }

    fn declare_inner(
        &mut self,
        name: impl Into<SymbolRef>,
        value: Value,
        mutable: bool,
        access: Access,
    ) -> Result<Option<Value>, Fault> {
        Ok(self.modules[self.frames[self.current_frame].module.0]
            .load(self.guard)
            .ok_or(Fault::ValueFreed)?
            .declarations()
            .insert(
                name.into(),
                ModuleDeclaration {
                    access,
                    mutable,
                    value,
                },
            )
            .map(|d| d.value.value))
    }

    /// Declares a compiled function.
    ///
    /// Returns a reference to the function, or `None` if the function could not
    /// be declared because it has no name.
    pub fn declare_function(&mut self, mut function: Function) -> Result<Option<Value>, Fault> {
        let Some(name) = function.name().clone() else {
            return Ok(None);
        };

        function.module = Some(ModuleId(0));
        self.declare_inner(name, Value::dynamic(function, &self), true, Access::Public)
    }

    /// Resolves the value at `path`.
    // TODO write better documentation, but I'm not sure this function is "done"
    // with it's implementation yet.
    pub fn resolve(&mut self, name: &Symbol) -> Result<Value, Fault> {
        if let Some(path) = name.strip_prefix('$') {
            let mut module_dynamic = self.modules[0];
            let mut module = module_dynamic.try_load(self.guard)?;
            let mut path = path.split('.').peekable();
            path.next();
            return if path.peek().is_none() {
                Ok(module_dynamic.into_value())
            } else {
                let name = loop {
                    let Some(name) = path.next() else {
                        return Err(Fault::UnknownSymbol);
                    };
                    let name = Symbol::from(name);
                    if path.peek().is_some() {
                        let declarations = module.declarations();
                        let decl = &declarations
                            .get(&name.downgrade())
                            .ok_or(Fault::UnknownSymbol)?;
                        let value = if decl.access >= self.caller_access_level(&module_dynamic) {
                            decl.value
                        } else {
                            return Err(Fault::Forbidden);
                        };

                        let Some(inner) = value.as_dynamic::<Module>() else {
                            return Err(Fault::NotAModule);
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
                self.modules[self.frames[self.current_frame].module.0].try_load(self.guard)?;
            if let Some(value) = module
                .declarations()
                .get(&name.downgrade())
                .map(|decl| decl.value)
            {
                Ok(value)
            } else if name == Symbol::super_symbol() {
                Ok(module.parent.map(Dynamic::into_value).unwrap_or_default())
            } else {
                Err(Fault::UnknownSymbol)
            }
        }
    }

    pub(crate) fn assign(&mut self, name: &SymbolRef, value: Value) -> Result<(), Fault> {
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
            let module = &vm.modules[vm.frames[vm.current_frame].module.0];
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

    pub(crate) fn exit_frame(&mut self) -> Result<(), Fault> {
        let vm = self.vm();
        if vm.current_frame >= 1 {
            vm.has_anonymous_frame = false;
            let current_frame = &vm.frames[vm.current_frame];
            vm.stack[current_frame.start..current_frame.end].fill_with(Value::nil);
            vm.current_frame -= 1;
            Ok(())
        } else {
            Err(Fault::StackUnderflow)
        }
    }

    /// Allocates `count` entries on the stack. Returns the first allocated
    /// index.
    pub fn allocate(&mut self, count: usize) -> Result<Stack, Fault> {
        let vm = self.vm();
        let current_frame = &mut vm.frames[vm.current_frame];
        let index = Stack(current_frame.end - current_frame.start);
        match current_frame.end.checked_add(count) {
            Some(end) if end < vm.max_stack => {
                current_frame.end += count;
                if vm.stack.len() < current_frame.end {
                    vm.stack.resize_with(current_frame.end, Value::nil);
                }
                Ok(index)
            }
            _ => Err(Fault::StackOverflow),
        }
    }

    /// Returns a reference to the currently executing code.
    #[must_use]
    pub fn current_code(&self) -> Option<&Code> {
        self.frames[self.current_frame]
            .code
            .map(|index| &self.code[index.0].code)
    }

    /// Returns the instruction offset in the current frame.
    #[must_use]
    pub fn current_instruction(&self) -> usize {
        self.frames[self.current_frame].instruction
    }

    /// Jumps execution to `instruction`.
    pub(crate) fn jump_to(&mut self, instruction: usize) {
        let vm = self.vm();
        vm.frames[vm.current_frame].instruction = instruction;
    }

    /// Returns a slice of the current stack frame.
    #[must_use]
    pub fn current_frame(&self) -> &[Value] {
        &self.stack[self.frames[self.current_frame].start..self.frames[self.current_frame].end]
    }

    /// Returns an exclusive reference to the slice of the current stack frame.
    #[must_use]
    pub fn current_frame_mut(&mut self) -> &mut [Value] {
        let vm = self.vm();
        &mut vm.stack[vm.frames[vm.current_frame].start..vm.frames[vm.current_frame].end]
    }

    /// Returns the size of the current stack frame.
    #[must_use]
    pub fn current_frame_size(&self) -> usize {
        self.frames[self.current_frame].end - self.frames[self.current_frame].start
    }

    /// Resets the stack and registers to their initial state and clears any
    /// executing code.
    ///
    /// All declarations and modules will still be loaded in the virtual
    /// machine.
    pub fn reset(&mut self) {
        self.current_frame = 0;
        for frame in &mut self.frames {
            frame.clear();
        }
        self.registers.fill_with(Value::nil);
        self.stack.fill_with(Value::nil);
    }

    /// Generates a backtrace for the current virtual machine state.
    #[must_use]
    pub fn backtrace(&self) -> Vec<StackFrame> {
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
        println!("Executing {instruction:?}");
        let next_instruction = StepResult::from(address.checked_add(1));
        let result = match instruction {
            LoadedOp::Return => return Ok(StepResult::Complete),
            LoadedOp::Declare {
                name,
                mutable,
                access,
                value,
                dest,
            } => self.op_declare(code_index, *name, *mutable, *access, *value, *dest),
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
                |vm, lhs, rhs| {
                    Ok(Value::Primitive(Primitive::Bool(
                        lhs.truthy(vm) ^ rhs.truthy(vm),
                    )))
                },
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
                        .map(|b| Value::Primitive(Primitive::Bool(b)))
                },
            ),
            LoadedOp::NotEqual(loaded) => self.op_binop(
                code_index,
                loaded.op1,
                loaded.op2,
                loaded.dest,
                |vm, lhs, rhs| {
                    lhs.equals(ContextOrGuard::Context(vm), &rhs)
                        .map(|result| Value::Primitive(Primitive::Bool(!result)))
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
                |vm, lhs, rhs| {
                    lhs.matches(vm, &rhs)
                        .map(|b| Value::Primitive(Primitive::Bool(b)))
                },
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
                    parent: Some(vm.modules[vm.frames[executing_frame].module.0]),
                    ..Module::default()
                },
                &*self.guard,
            ));
            vm.frames[vm.current_frame].module = ModuleId(module_index.get());
            vm.frames[executing_frame].loading_module = Some(module_index);
            let _init_result = self.resume_async_inner(self.current_frame)?;
            self.frames[executing_frame].loading_module = None;
            module_index
        };

        self.op_store(
            code_index,
            self.modules[loading_module.get()].into_value(),
            dest,
        )?;
        Ok(())
    }

    fn op_declare(
        &mut self,
        code_index: usize,
        name: usize,
        mutable: bool,
        access: Access,
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
        self.declare_inner(name, value, mutable, access)?;
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
        let value = Value::Primitive(Primitive::Bool(value.truthy(self)));
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

        self[Register(0)] = target.invoke(self, &name, arity)?;

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
            lhs.total_cmp(vm, &rhs)
                .map(|ord| Value::Primitive(Primitive::Bool(op(ord))))
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

    fn op_load_symbol(&mut self, code_index: usize, symbol: usize) -> Result<SymbolRef, Fault> {
        self.code[code_index]
            .code
            .data
            .symbols
            .get(symbol)
            .map(Symbol::downgrade)
            .ok_or(Fault::InvalidOpcode)
    }

    fn op_load(&mut self, code_index: usize, value: LoadedSource) -> Result<Value, Fault> {
        match value {
            LoadedSource::Nil => Ok(Value::nil()),
            LoadedSource::Bool(v) => Ok(Value::Primitive(Primitive::Bool(v))),
            LoadedSource::Int(v) => Ok(Value::Primitive(Primitive::Int(v))),
            LoadedSource::UInt(v) => Ok(Value::Primitive(Primitive::UInt(v))),
            LoadedSource::Float(v) => Ok(Value::Primitive(Primitive::Float(v))),
            LoadedSource::Symbol(v) => self.op_load_symbol(code_index, v).map(Value::Symbol),
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
                .map(|i| Value::Primitive(Primitive::UInt(i)))
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
            LoadedSource::Struct(v) => self.code[code_index]
                .code
                .data
                .structs
                .get(v)
                .map(|ty| {
                    Value::dynamic(
                        ty.load(self.guard, self.frames[self.current_frame].module),
                        &self,
                    )
                })
                .ok_or(Fault::InvalidOpcode),
            LoadedSource::Enum(v) => {
                let ty = self.code[code_index]
                    .code
                    .data
                    .enums
                    .get(v)
                    .ok_or(Fault::InvalidOpcode)?;
                let ty = ty.load(self)?;
                Ok(Value::dynamic(ty, &self))
            }
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

/// A virtual machine register index.
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

/// A value was given that is not a valid register.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct InvalidRegister;

#[derive(Default, Debug, Clone)]
struct Frame {
    start: usize,
    end: usize,
    instruction: usize,
    code: Option<CodeIndex>,
    variables: Map<SymbolRef, BlockDeclaration>,
    module: ModuleId,
    loading_module: Option<NonZeroUsize>,
    exception_handler: Option<NonZeroUsize>,
}

impl Frame {
    fn clear(&mut self) {
        self.variables.clear();
        self.instruction = usize::MAX;
        self.code = None;
        self.module = ModuleId(0);
        self.loading_module = None;
        self.exception_handler = None;
    }
}

/// An error that arises from executing Muse code.
#[derive(Debug, PartialEq, Clone, Trace)]
pub enum ExecutionError {
    /// The budget for the virtual machine has been exhausted.
    NoBudget,
    /// The virtual machine is waiting for an async task.
    Waiting,
    /// Execution did not complete before a timeout occurred.
    Timeout,
    /// A thrown exception was not handled.
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

    /// Returns this error as a [`Value`].
    pub fn as_value(&self) -> Value {
        match self {
            ExecutionError::NoBudget => Value::Symbol(SymbolRef::from("no_budget")),
            ExecutionError::Waiting => Value::Symbol(SymbolRef::from("waiting")),
            ExecutionError::Timeout => Value::Symbol(SymbolRef::from("timeout")),
            ExecutionError::Exception(v) => *v,
        }
    }
}

/// A virtual machine error.
#[derive(Debug, PartialEq, Clone)]
#[non_exhaustive]
pub enum Fault {
    /// The budget for the virtual machine has been exhausted.
    NoBudget,
    /// Execution did not complete before a timeout occurred.
    Timeout,
    /// The virtual machine is waiting for an async task.
    Waiting,
    /// A thrown exception was not handled.
    Exception(Value),
    /// The execution frame has changed.
    FrameChanged,
    /// A declaration was found, but it is not accessible by the currently executing code.
    Forbidden,
    /// A symbol could not be resolved to a declaration or function.
    UnknownSymbol,
    /// A function was invoked with an unsupported number of arguments.
    IncorrectNumberOfArguments,
    /// A pattern could not be matched.
    PatternMismatch,
    /// An unsupported operation was performed on [`Value::Nil`].
    OperationOnNil,
    /// A value was freed.
    ///
    /// This, in general, should not happen, but it can happen through
    /// intentional usage or if a type does not implement [`Trace`] correctly.
    ValueFreed,
    /// A value was expected to be a module, but was not.
    NotAModule,
    /// A value was invoked but is not invokable.
    NotAFunction,
    /// An allocation failed because the stack is full.
    StackOverflow,
    /// An attempt to return from the execution root.
    StackUnderflow,
    /// A general error indicating an unsupported operation.
    UnsupportedOperation,
    /// An allocation could not be performed due to memory constraints.
    OutOfMemory,
    /// An operation attempted to access something outside of its bounds.
    OutOfBounds,
    /// An assignment was attempted to an immutable declaration.
    NotMutable,
    /// A value was divided by zero.
    DivideByZero,
    /// Execution jumped to an invalid instruction address.
    InvalidInstructionAddress,
    /// An operation expected a symbol.
    ExpectedSymbol,
    /// An operation expected an integer.
    ExpectedInteger,
    /// An operation expected a string.
    ExpectedString,
    /// An operation expected a list.
    ExpectedList,
    /// An invalid value was provided for the [`Arity`] of a function.
    InvalidArity,
    /// An invalid label was encountered.
    InvalidLabel,
    /// An instruction referenced an invalid index.
    ///
    /// This is differen than a [`Fault::OutOfBounds`] because this fault
    /// indicates invalid code rather than a logic error.
    InvalidOpcode,
}

impl Fault {
    fn is_execution_error(&self) -> bool {
        matches!(self, Fault::NoBudget | Fault::Waiting | Fault::Timeout)
    }

    /// Returns this error as a [`Symbol`].
    ///
    /// # Errors
    ///
    /// Returns the contained exception as `Err` if this fault is an exception.
    pub fn as_symbol(&self) -> Result<Symbol, Value> {
        match self {
            Fault::UnknownSymbol => Ok(Symbol::from("undefined")),
            Fault::Forbidden => Ok(Symbol::from("forbidden")),
            Fault::IncorrectNumberOfArguments => Ok(Symbol::from("args")),
            Fault::OperationOnNil => Ok(Symbol::from("nil")),
            Fault::ValueFreed => Ok(Symbol::from("Some(out-of-scope")),
            Fault::NotAFunction => Ok(Symbol::from("not_invokable")),
            Fault::NotAModule => Ok(Symbol::from("not_a_module")),
            Fault::StackOverflow => Ok(Symbol::from("overflow")),
            Fault::StackUnderflow => Ok(Symbol::from("underflow")),
            Fault::UnsupportedOperation => Ok(Symbol::from("unsupported")),
            Fault::OutOfMemory => Ok(Symbol::from("out_of_memory")),
            Fault::OutOfBounds => Ok(Symbol::from("out_of_bounds")),
            Fault::NotMutable => Ok(Symbol::from("immutable")),
            Fault::DivideByZero => Ok(Symbol::from("divided_by_zero")),
            Fault::InvalidInstructionAddress => Ok(Symbol::from("invalid_instruction")),
            Fault::ExpectedSymbol => Ok(Symbol::from("expected_symbol")),
            Fault::ExpectedInteger => Ok(Symbol::from("expected_integer")),
            Fault::ExpectedString => Ok(Symbol::from("expected_string")),
            Fault::ExpectedList => Ok(Symbol::from("expected_list")),
            Fault::InvalidArity => Ok(Symbol::from("invalid_arity")),
            Fault::InvalidLabel => Ok(Symbol::from("invalid_label")),
            Fault::InvalidOpcode => Ok(Symbol::from("invalid_opcode")),
            Fault::NoBudget => Ok(Symbol::from("no_budget")),
            Fault::Timeout => Ok(Symbol::from("timeout")),
            Fault::Waiting => Ok(Symbol::from("waiting")),
            Fault::FrameChanged => Ok(Symbol::from("frame_changed")),
            Fault::PatternMismatch => Ok(Symbol::from("mismatch")),
            Fault::Exception(exc) => Err(*exc),
        }
    }

    /// Converts this fault into an exception.
    #[must_use]
    pub fn as_exception(&self, vm: &mut VmContext<'_, '_>) -> Value {
        match self.as_symbol() {
            Ok(sym) => Value::dynamic(Exception::new(sym.into(), vm), vm),
            Err(exc) => exc,
        }
    }

    fn from_kind(kind: FaultKind, vm: &mut VmContext<'_, '_>) -> Self {
        let exception = match kind {
            FaultKind::Exception => Value::dynamic(Exception::new(vm[Register(0)].take(), vm), vm),
            FaultKind::PatternMismatch => Self::PatternMismatch.as_exception(vm),
        };
        Self::Exception(exception)
    }
}

/// A Muse function ready for execution.
#[derive(Debug, Clone)]
pub struct Function {
    module: Option<ModuleId>,
    name: Option<Symbol>,
    bodies: Map<Arity, Code>,
    varg_bodies: Map<Arity, Code>,
}

impl Function {
    /// Returns a function with the given name and no bodies.
    #[must_use]
    pub fn new(name: impl IntoOptionSymbol) -> Self {
        Self {
            module: None,
            name: name.into_symbol(),
            bodies: Map::new(),
            varg_bodies: Map::new(),
        }
    }

    pub(crate) fn in_module(mut self, module: ModuleId) -> Self {
        self.module = Some(module);
        self
    }

    /// Inserts `body` to be executed when this function is invoked with `arity`
    /// number of arguments.
    pub fn insert_arity(&mut self, arity: impl Into<Arity>, body: Code) {
        self.bodies.insert(arity.into(), body);
    }

    /// Adds `body` to be executed when this function is invoked with `arity`
    /// number of arguments, and returns the updated function.
    #[must_use]
    pub fn when(mut self, arity: impl Into<Arity>, body: Code) -> Self {
        self.insert_arity(arity, body);
        self
    }

    /// Returns the name of this function.
    #[must_use]
    pub const fn name(&self) -> &Option<Symbol> {
        &self.name
    }

    /// Returns the code of the body for a given number of arguments, if
    /// present.
    pub fn body(&self, arity: Arity) -> Option<&Code> {
        self.bodies.get(&arity)
    }
}

impl CustomType for Function {
    fn muse_type(&self) -> &crate::runtime::value::TypeRef {
        static TYPE: RustType<Function> = RustType::new("Function", |t| {
            t.with_call(|_| {
                |this, vm, arity| {
                    if let Some(body) = this.bodies.get(&arity).or_else(|| {
                        this.varg_bodies
                            .iter()
                            .rev()
                            .find_map(|va| (va.key() <= &arity).then_some(&va.value))
                    }) {
                        let module = this.module.ok_or(Fault::NotAModule)?;
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

/// The number of arguments provided to a function.
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

/// A series of instructions that are ready to execute.
#[derive(Debug, Clone)]
pub struct Code {
    data: Arc<CodeData>,
}

impl Code {
    /// Pushes another operation to this code block.
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
    structs: Vec<BitcodeStruct>,
    enums: Vec<BitcodeEnum>,
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
                access,
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
                        access: *access,
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

    fn push_struct(&mut self, ty: BitcodeStruct) -> usize {
        let index = self.structs.len();
        self.structs.push(ty);
        index
    }

    fn push_enum(&mut self, ty: BitcodeEnum) -> usize {
        let index = self.structs.len();
        self.enums.push(ty);
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
            ValueOrSource::Struct(ty) => {
                let ty = self.push_struct(ty.clone());
                LoadedSource::Struct(ty)
            }
            ValueOrSource::Enum(ty) => {
                let ty = self.push_enum(ty.clone());
                LoadedSource::Enum(ty)
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

/// A Muse module.
///
/// A module enables encapsulating and namespacing code and declarations.
#[derive(Default, Debug)]
pub struct Module {
    parent: Option<Dynamic<Module>>,
    declarations: Mutex<Map<SymbolRef, ModuleDeclaration>>,
}

impl Module {
    /// Returns a new module whose `super` is `parent`.
    pub const fn new(parent: Option<Dynamic<Module>>) -> Self {
        Self {
            parent,
            declarations: Mutex::new(Map::new()),
        }
    }

    /// Returns a new module with the built-in `core` module loaded.
    #[must_use]
    pub fn with_core(guard: &CollectionGuard) -> Dynamic<Self> {
        let module = Dynamic::new(Self::default(), guard);

        let core = Self::core(Some(module));

        module.load(guard).expect("guard held").declare(
            "core",
            Access::Public,
            Value::dynamic(core, guard),
        );

        module
    }

    /// Returns the default `core` module.
    pub fn core(parent: Option<Dynamic<Module>>) -> Self {
        let core = Self::new(parent);

        let mut declarations = core.declarations();
        declarations.insert(
            SymbolRef::from("Map"),
            ModuleDeclaration {
                mutable: false,
                value: Value::Dynamic(crate::runtime::map::MAP_TYPE.as_any_dynamic()),
                access: Access::Public,
            },
        );
        declarations.insert(
            SymbolRef::from("List"),
            ModuleDeclaration {
                mutable: false,
                value: Value::Dynamic(crate::runtime::list::LIST_TYPE.as_any_dynamic()),
                access: Access::Public,
            },
        );
        declarations.insert(
            SymbolRef::from("String"),
            ModuleDeclaration {
                mutable: false,
                value: Value::Dynamic(crate::runtime::string::STRING_TYPE.as_any_dynamic()),
                access: Access::Public,
            },
        );
        drop(declarations);

        core
    }

    fn declarations(&self) -> MutexGuard<'_, Map<SymbolRef, ModuleDeclaration>> {
        self.declarations.lock()
    }

    /// Declares a new read-only variable in this module.
    pub fn declare(&self, name: impl Into<SymbolRef>, access: Access, value: Value) {
        self.declarations().insert(
            name.into(),
            ModuleDeclaration {
                access,
                mutable: false,
                value,
            },
        );
    }

    /// Declares a new mutable variable in this module.
    pub fn declare_mut(&self, name: impl Into<SymbolRef>, access: Access, value: Value) {
        self.declarations().insert(
            name.into(),
            ModuleDeclaration {
                access,
                mutable: true,
                value,
            },
        );
    }
}

impl CustomType for Module {
    fn muse_type(&self) -> &crate::runtime::value::TypeRef {
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
                                        Some(decl)
                                            if decl.mutable
                                                && decl.access
                                                    >= vm
                                                        .caller_access_level(&this.downgrade()) =>
                                        {
                                            Ok(std::mem::replace(&mut decl.value, value))
                                        }
                                        Some(decl) if !decl.mutable => Err(Fault::NotMutable),
                                        Some(_) => Err(Fault::Forbidden),
                                        None => Err(Fault::UnknownSymbol),
                                    }
                                })
                                .with_fn(Symbol::get_symbol(), 1, |vm, this| {
                                    let field = vm[Register(0)].take();
                                    let sym = field.as_symbol_ref().ok_or(Fault::ExpectedSymbol)?;

                                    let declarations = this.declarations();
                                    let decl = declarations.get(sym).ok_or(Fault::UnknownSymbol)?;
                                    if decl.access >= vm.caller_access_level(&this.downgrade()) {
                                        Ok(decl.value)
                                    } else {
                                        Err(Fault::Forbidden)
                                    }
                                })
                        });
                    let declarations = this.declarations();
                    if let Some(decl) = declarations.get(name) {
                        if decl.access >= vm.caller_access_level(&this.downgrade()) {
                            let possible_invoke = decl.value;
                            drop(declarations);
                            possible_invoke.call(vm, arity)
                        } else {
                            Err(Fault::Forbidden)
                        }
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
            decl.key().trace(tracer);
            decl.value.value.trace(tracer);
        }
    }
}

#[derive(Debug)]
struct ModuleDeclaration {
    access: Access,
    mutable: bool,
    value: Value,
}

#[derive(Clone, Copy, Debug)]
struct Budget(usize);

impl Budget {
    const DISABLED: usize = usize::MAX;

    fn allocate(&mut self, amount: usize) {
        if self.0 == Self::DISABLED {
            self.0 = amount;
        } else {
            self.0 = self.0.saturating_add(amount).min(Self::DISABLED - 1);
        }
    }

    fn charge(&mut self) -> Result<(), Fault> {
        if self.0 != Self::DISABLED {
            self.0 = self.0.checked_sub(1).ok_or(Fault::NoBudget)?;
        }
        Ok(())
    }
}

impl Default for Budget {
    fn default() -> Self {
        Self(Self::DISABLED)
    }
}

enum MaybeOwnedContext<'vm, 'context, 'guard> {
    Owned(VmContext<'context, 'guard>),
    Borrowed(&'vm mut VmContext<'context, 'guard>),
}

impl<'vm, 'context, 'guard> MaybeOwnedContext<'vm, 'context, 'guard> {
    pub fn execute_async(
        mut self,
        code: &Code,
    ) -> Result<ExecuteAsync<'vm, 'context, 'guard>, ExecutionError> {
        let code = self.push_code(code, None);
        self.prepare_owned(code)?;

        Ok(ExecuteAsync(self))
    }

    fn resume_async(self) -> ExecuteAsync<'vm, 'context, 'guard> {
        ExecuteAsync(self)
    }

    fn resume_for_async(self, duration: Duration) -> ExecuteAsync<'vm, 'context, 'guard> {
        self.resume_until_async(Instant::now() + duration)
    }

    fn resume_until_async(mut self, instant: Instant) -> ExecuteAsync<'vm, 'context, 'guard> {
        self.execute_until = Some(instant);
        self.resume_async()
    }
}

impl<'context, 'guard> Deref for MaybeOwnedContext<'_, 'context, 'guard> {
    type Target = VmContext<'context, 'guard>;

    fn deref(&self) -> &Self::Target {
        match self {
            MaybeOwnedContext::Owned(v) => v,
            MaybeOwnedContext::Borrowed(v) => v,
        }
    }
}

impl<'context, 'guard> DerefMut for MaybeOwnedContext<'_, 'context, 'guard> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            MaybeOwnedContext::Owned(v) => v,
            MaybeOwnedContext::Borrowed(v) => v,
        }
    }
}

/// An asynchronous code execution.
#[must_use = "futures must be awaited to be exected"]
pub struct ExecuteAsync<'vm, 'context, 'guard>(MaybeOwnedContext<'vm, 'context, 'guard>);

impl Future for ExecuteAsync<'_, '_, '_> {
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
        access: Access,
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
    Struct(usize),
    Enum(usize),
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
        result: MuseRegex::load(regex, guard),
    }
}

/// An offset into a virtual machine stack.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Stack(pub usize);

/// Information about an executing stack frame.
#[derive(PartialEq, Clone)]
pub struct StackFrame {
    code: Code,
    instruction: usize,
}

impl StackFrame {
    #[must_use]
    fn new(code: Code, instruction: usize) -> Self {
        Self { code, instruction }
    }

    /// Returns the code executing in this frame.
    #[must_use]
    pub const fn code(&self) -> &Code {
        &self.code
    }

    /// Returns the instruction offset of the frame.
    #[must_use]
    pub const fn instruction(&self) -> usize {
        self.instruction
    }

    /// Returns the source range for this instruction, if available.
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

/// A set of arguments that can be loaded into a virtual machine when invoking a
/// function.
pub trait InvokeArgs {
    /// Loads the arguments into `vm`.
    fn load(self, vm: &mut VmContext<'_, '_>) -> Result<Arity, ExecutionError>;
}

impl<T, const N: usize> InvokeArgs for [T; N]
where
    T: Into<Value>,
{
    fn load(self, vm: &mut VmContext<'_, '_>) -> Result<Arity, ExecutionError> {
        let arity = Arity::try_from(N)
            .map_err(|_| ExecutionError::Exception(Fault::InvalidArity.as_exception(vm)))?;

        for (arg, register) in self.into_iter().zip(0..arity.0) {
            vm[Register(register)] = arg.into();
        }

        Ok(arity)
    }
}
