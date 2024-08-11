//! # muse-reactor
//!
//! `muse-reactor` provides a multi-threaded runtime that executes Muse code.
//! It's primary features include:
//!
//! - Spawning tasks across a threadpool
//! - Waiting on a task's completion
//! - Cancelling a task
//! - Budgeting tasks in pools
//!
//! # Basic Usage
//!
//! ```rust
//! use muse_reactor::Reactor;
//! use muse_lang::runtime::value::{Primitive, RootedValue};
//!
//! // Create a new reactor for tasks to run in.
//! let reactor = Reactor::spawn();
//!
//! // Spawn a task that computes 1 + 2
//! let task = reactor.spawn_source("1 + 2").unwrap();
//!
//! // Wait for the result and verify it's 3.
//! assert_eq!(
//!     task.join().unwrap(),
//!     RootedValue::Primitive(Primitive::Int(3))
//! );
//! ```
//!
//! [`TaskHandle`] is also a future that can be awaited to wait for the task to
//! complete.
//!
//! # Budget Pools
//!
//! Budget pools enable efficiently restricting groups of tasks to execution
//! budgets. Each time a task assigned to a budget pool exhausts its budget, it
//! requests additional budget from the pool. If no budget is available, the
//! task is put to sleep and will automatically be resumed when the budget has
//! been replenished.
//!
//! ```rust
//! use muse_reactor::{BudgetPoolId, BudgetPoolConfig, Reactor};
//! use muse_lang::runtime::value::{Primitive, RootedValue};
//! use std::time::Duration;
//!
//! // Create a new reactor for tasks to run in.
//! let reactor = Reactor::spawn();
//!
//! // Create a budget pool that we can spawn tasks within.
//! let pool = reactor.create_budget_pool(BudgetPoolConfig::default()).unwrap();
//!
//! // Spawn a task within the budget pool
//! let task = pool.spawn_source("var i = 0; while i < 100 { i = i + 1; }; i").unwrap();
//!
//! // Verify the task isn't able to complete.
//! assert!(task.join_for(Duration::from_secs(1)).is_none());
//!
//! // Allocate enough budget.
//! pool.increase_budget(1_000);
//!
//! // Wait for the task to complete
//! assert_eq!(
//!     task.join().unwrap(),
//!     RootedValue::Primitive(Primitive::Int(100))
//! );
//! ```
use std::any::Any;
use std::backtrace::Backtrace;
use std::cell::Cell;
use std::collections::VecDeque;
use std::fmt::{Debug, Display, Write};
use std::future::Future;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::ControlFlow;
use std::panic::{self, AssertUnwindSafe, PanicInfo};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use alot::{LotId, Lots};
use crossbeam_utils::sync::{Parker, Unparker};
use flume::{Receiver, SendError, Sender, TryRecvError};
use kempt::{Map, Set};
use muse_lang::compiler::syntax::Ranged;
use muse_lang::compiler::{self, Compiler};
use muse_lang::runtime::list::List;
use muse_lang::runtime::symbol::{Symbol, SymbolRef};
use muse_lang::runtime::value::{
    CustomType, Dynamic, Rooted, RootedValue, RustFunction, RustType, Value,
};
use muse_lang::vm::bitcode::Access;
use muse_lang::vm::{
    Arity, Code, ExecutionError, Fault, Function, Module, Register, Vm, VmContext,
};
use parking_lot::{Condvar, Mutex, MutexGuard};
use refuse::{CollectionGuard, ContainsNoRefs, Trace};

#[cfg(feature = "tracing")]
#[macro_use]
extern crate tracing;
#[cfg(not(feature = "tracing"))]
#[macro_use]
mod mock_tracing;

static PANIC_HOOK_INSTALL: OnceLock<()> = OnceLock::new();
thread_local! {
    static PANIC_INFO: Cell<Option<(String, Option<Backtrace>)>> = const { Cell::new(None) };
}

/// A builder for a [`Reactor`].
#[must_use]
pub struct Builder<Work> {
    vm_source: Option<Arc<dyn NewVm<Work>>>,
    threads: usize,
    thread_name: Option<String>,
    work_queue_limit: Option<usize>,
    _work: PhantomData<Work>,
}

impl Default for Builder<NoWork> {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder<NoWork> {
    /// Returns a new builder for the default reactor settings.
    pub fn new() -> Self {
        Self {
            vm_source: None,
            threads: std::thread::available_parallelism().map_or(0, NonZeroUsize::get),
            thread_name: None,
            work_queue_limit: None,
            _work: PhantomData,
        }
    }

    /// Customizes the process for which virtual machines are initialized.
    ///
    /// For convenience, [`NewVm`] is implemented for `Fn(&mut
    /// CollectionGuard<'_>, &ReactorHandle<Work>)` where `Work` implements
    /// [`WorkUnit`].
    pub fn new_vm<F, Work>(self, new_vm: F) -> Builder<Work>
    where
        F: NewVm<Work>,
    {
        Builder {
            vm_source: Some(Arc::new(new_vm)),
            threads: self.threads,
            thread_name: self.thread_name,
            work_queue_limit: self.work_queue_limit,
            _work: PhantomData,
        }
    }
}

impl<Work> Builder<Work>
where
    Work: WorkUnit,
{
    /// Sets the number of threads to execute tasks across.
    pub fn threads(mut self, thread_count: usize) -> Self {
        self.threads = thread_count;
        self
    }

    /// Sets the maximum number of work items in queue.
    ///
    /// By default, work queues are not limited.
    pub fn work_queue_limit(mut self, limit: usize) -> Self {
        self.work_queue_limit = Some(limit);
        self
    }

    /// Spawns a reactor with the given settings and returns a handle to it.
    #[must_use]
    pub fn finish(self) -> ReactorHandle<Work> {
        PANIC_HOOK_INSTALL.get_or_init(|| {
            let default_hook = panic::take_hook();
            panic::set_hook(Box::new(move |info: &PanicInfo| {
                PANIC_INFO.set(Some((info.to_string(), Some(Backtrace::capture()))));
                default_hook(info);
            }));
        });

        let (sender, receiver) = if let Some(limit) = self.work_queue_limit {
            flume::bounded(limit)
        } else {
            flume::unbounded()
        };

        let thread_name = self
            .thread_name
            .unwrap_or_else(|| String::from("muse-reactor"));

        let mut threads = Vec::with_capacity(self.threads);
        let shared = Arc::new(SharedReactorData {
            shutdown: AtomicBool::new(false),
        });

        let vm_source = self.vm_source.unwrap_or_else(|| Arc::new(()));

        let handle = ReactorHandle {
            data: Arc::new(HandleData {
                sender,
                shared,
                threads: Arc::default(),
                next_task_id: AtomicUsize::new(0),
                next_budget_pool_id: AtomicUsize::new(1),
            }),
        };

        for id in 0..self.threads {
            let (spawn_send, spawn_recv) = flume::unbounded();
            let parker = Parker::new();
            let data = Arc::new(PerThreadData::new(id, parker.unparker().clone()));
            let reactor = Reactor {
                id,
                receiver: spawn_recv,
                vm_source: vm_source.clone(),
                handle: handle.clone(),
                budgets: Map::new(),
            };
            threads.push(PerThread {
                data: data.clone(),
                spawner: spawn_send.clone(),
                handle: thread::Builder::new()
                    .name(thread_name.clone())
                    .spawn(move || reactor.run(spawn_send, data, &parker))
                    .expect("error spawning thread"),
            });
        }

        *handle.data.threads.lock() = threads;

        thread::Builder::new()
            .name(String::from("dispatcher"))
            .spawn({
                let handle = handle.clone();
                move || Dispatcher::new(receiver, handle).run()
            })
            .expect("error spawning dispatcher");

        handle
    }
}

enum ThreadCommand<Work> {
    Spawn(Spawn<Work>),
    Cancel(usize),
    NewBudgetPool(ReactorBudgetPool),
    DestroyBudgetPool(BudgetPoolId),
}

struct DispatcherThread<Work> {
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    num: usize,
    spawner: Sender<ThreadCommand<Work>>,
    load: usize,
    unparker: Unparker,
}

struct RechargingBudget {
    pool: ReactorBudgetPool,
    recharge_at: Instant,
}

struct Dispatcher<Work> {
    commands: Receiver<Command<Work>>,
    handle: ReactorHandle<Work>,
    threads: VecDeque<DispatcherThread<Work>>,
    next_rebalance: Instant,
    budget_ids: Map<BudgetPoolId, LotId>,
    recharging_budgets: Lots<RechargingBudget>,
    recharge_queue: VecDeque<LotId>,
}

impl<Work> Dispatcher<Work> {
    const REBALANCE_DELAY: Duration = Duration::from_millis(30);

    fn new(commands: Receiver<Command<Work>>, handle: ReactorHandle<Work>) -> Self {
        let mut this = Self {
            commands,
            handle,
            threads: VecDeque::new(),
            next_rebalance: Instant::now(),
            budget_ids: Map::new(),
            recharging_budgets: Lots::new(),
            recharge_queue: VecDeque::new(),
        };
        this.cache_thread_loads(this.next_rebalance);
        this
    }

    fn cache_thread_loads(&mut self, now: Instant) {
        let threads = self.handle.data.threads.lock();
        self.threads.clear();
        for t in &*threads {
            self.threads.push_back(DispatcherThread {
                num: t.data.num,
                spawner: t.spawner.clone(),
                load: t.spawner.len() * 2
                    + t.data.executing.load(Ordering::Relaxed)
                    + t.data.total.load(Ordering::Relaxed),
                unparker: t.data.unparker.clone(),
            });
        }
        for i in 0..self.threads.len() {
            for j in i + 1..self.threads.len() {
                if self.threads[j].load < self.threads[i].load {
                    self.threads.swap(i, j);
                }
            }
        }
        self.next_rebalance = now + Self::REBALANCE_DELAY;
    }

    fn run(mut self) {
        while !self.commands.is_disconnected() {
            while let Some(command) = if let Some(budget) = self
                .recharge_queue
                .iter()
                .find_map(|id| self.recharging_budgets.get(*id))
            {
                self.commands.recv_deadline(budget.recharge_at).ok()
            } else {
                self.commands.recv().ok()
            } {
                self.process_command(command);
            }

            self.recharge_budgets();
        }
    }

    fn process_command(&mut self, command: Command<Work>) {
        match command {
            Command::NewBudgetPool(pool) => {
                self.threads.retain_mut(|th| {
                    let sent = th
                        .spawner
                        .send(ThreadCommand::NewBudgetPool(pool.clone()))
                        .is_ok();
                    th.unparker.unpark();
                    sent
                });
                if pool.0.config.recharges() {
                    let recharge_at = Instant::now() + pool.0.config.recharge_every;
                    let pool_id = pool.0.id;
                    let id = self
                        .recharging_budgets
                        .push(RechargingBudget { pool, recharge_at });
                    self.budget_ids.insert(pool_id, id);
                    self.recharge_queue
                        .insert(self.find_recharge_queue_position(recharge_at, 0), id);
                }
            }
            Command::DestroyBudgetPool(pool) => {
                self.threads.retain_mut(|th| {
                    let sent = th
                        .spawner
                        .send(ThreadCommand::DestroyBudgetPool(pool))
                        .is_ok();
                    th.unparker.unpark();
                    sent
                });
                if let Some(id) = self.budget_ids.remove(&pool) {
                    self.recharging_budgets.remove(id.value);
                }
            }
            Command::Spawn(mut spawn) => {
                // We loop in case a thread dies.
                loop {
                    let Some(thread) = self.threads.front_mut() else {
                        return;
                    };

                    trace!(task = spawn.id, thread = thread.num, "spawn");
                    match thread.spawner.send(ThreadCommand::Spawn(spawn)) {
                        Ok(()) => {
                            thread.unparker.unpark();
                            thread.load += 1;
                            let new_load = thread.load;
                            if let Some(next_load) = self
                                .threads
                                .get(1)
                                .and_then(|next| (next.load < new_load).then_some(next.load))
                            {
                                // The next thread has a load lower than the current
                                // thread after this spawn. We need to find a new
                                // place for this thread. To avoid moving the buffer
                                // too much we are going to scan to find the right
                                // location.
                                //
                                // First, find the next highest load after the new
                                // "front" load `next_load`.
                                match self.threads.iter().enumerate().skip(2).find_map(
                                    |(index, thread)| (next_load < thread.load).then_some(index),
                                ) {
                                    Some(insert_at) => {
                                        let current =
                                            self.threads.pop_front().expect("already checked");
                                        self.threads.insert(insert_at, current);
                                    }
                                    None => {
                                        // Move to the end.
                                        self.threads.rotate_left(1);
                                    }
                                }
                            }
                            break;
                        }
                        Err(SendError(returned_work)) => {
                            let ThreadCommand::Spawn(returned_work) = returned_work else {
                                unreachable!("just sent")
                            };
                            spawn = returned_work;
                            self.threads.remove(0);
                        }
                    }

                    let now = Instant::now();
                    if now > self.next_rebalance {
                        self.cache_thread_loads(now);
                    }
                }
            }
        }
    }

    fn recharge_budgets(&mut self) {
        let now = Instant::now();
        while let Some(id) = self.recharge_queue.front().copied() {
            let Some(pool) = self.recharging_budgets.get_mut(id) else {
                self.recharge_queue.pop_front();
                continue;
            };
            if pool.recharge_at > now {
                break;
            }

            pool.pool
                .increase_budget(pool.pool.0.config.recharge_amount);
            // Step the recharge by the duration, but make sure that we always
            // use a timestamp that is at least now. This ensures that if for
            // some reason we can't keep up, we don't get in a situation where
            // we can never recharge enough.
            let new_deadline = (pool.recharge_at + pool.pool.0.config.recharge_every).max(now);
            pool.recharge_at = new_deadline;
            let new_index = self.find_recharge_queue_position(new_deadline, 1);
            match new_index {
                0 => {}
                n if n == self.recharge_queue.len() => self.recharge_queue.rotate_left(1),
                _ => {
                    self.recharge_queue.pop_front();
                    self.recharge_queue.insert(new_index, id);
                }
            }
        }
    }

    fn find_recharge_queue_position(&self, recharge_at: Instant, start_at: usize) -> usize {
        let mut insert_at = self.recharge_queue.len();
        for (index, budget) in self
            .recharge_queue
            .iter()
            .skip(start_at)
            .enumerate()
            .filter_map(|(index, id)| self.recharging_budgets.get(*id).map(|b| (index, b)))
        {
            if budget.recharge_at > recharge_at {
                insert_at = index;
                break;
            }
        }
        insert_at
    }
}

/// A multi-threaded executor for Muse workloads.
pub struct Reactor<Work = NoWork> {
    id: usize,
    receiver: Receiver<ThreadCommand<Work>>,
    vm_source: Arc<dyn NewVm<Work>>,
    handle: ReactorHandle<Work>,
    budgets: Map<BudgetPoolId, ThreadBudget>,
}

impl Reactor<NoWork> {
    /// Spawns the default executor and returns a handle to it.
    #[must_use]
    pub fn spawn() -> ReactorHandle<NoWork> {
        Self::build().finish()
    }

    /// Returns a builder for a new reactor.
    pub fn build() -> Builder<NoWork> {
        Builder::new()
    }
}

fn root_result(
    result: Result<Value, Value>,
    context: &mut VmContext<'_, '_>,
) -> Result<RootedValue, RootedValue> {
    match result {
        Ok(v) => v.upgrade(context.guard()).ok_or_else(|| {
            Fault::ValueFreed
                .as_exception(context)
                .upgrade(context.guard())
                .expect("just allocated")
        }),
        Err(v) => {
            if let Some(v) = v.upgrade(context.guard()) {
                Err(v)
            } else {
                Err(Fault::ValueFreed
                    .as_exception(context)
                    .upgrade(context.guard())
                    .expect("just allocated"))
            }
        }
    }
}

impl<Work> Reactor<Work>
where
    Work: WorkUnit,
{
    fn run(
        mut self,
        sender: Sender<ThreadCommand<Work>>,
        data: Arc<PerThreadData>,
        parker: &Parker,
    ) {
        let canceller = TaskCanceller {
            canceller: Arc::new(sender),
            unparker: parker.unparker().clone(),
        };
        let mut tasks = ReactorTasks::new(data);
        while !self.handle.data.shared.shutdown.load(Ordering::Relaxed) {
            tasks.wake_woken();
            self.wake_exhausted(&mut tasks);
            let mut guard = CollectionGuard::acquire();
            self.execute_executing(&mut tasks, &mut guard, parker);

            if self
                .process_commands(&canceller, &mut tasks, &mut guard)
                .is_break()
            {
                break;
            }

            drop(guard);

            if !tasks.has_work() {
                parker.park();
            }
        }
    }

    fn process_command(
        &mut self,
        command: ThreadCommand<Work>,
        canceller: &TaskCanceller,
        tasks: &mut ReactorTasks,
        guard: &mut CollectionGuard<'_>,
    ) {
        match command {
            ThreadCommand::Spawn(command) => {
                let spawn = match command.what {
                    Spawnable::Spawn(vm) => Ok(vm),
                    Spawnable::SpawnSource(source) => {
                        self.vm_source
                            .compile_and_prepare(&source, guard, &self.handle)
                    }
                    Spawnable::SpawnCall(code, args) => {
                        self.vm_source.prepare_call(code, args, guard, &self.handle)
                    }
                    Spawnable::SpawnWork(work) => {
                        work.initialize(self.vm_source.as_ref(), guard, &self.handle)
                    }
                };
                match spawn {
                    Ok(vm) => {
                        let mut locked = command.result.0.locked.lock();
                        if locked.cancelled {
                            return;
                        }
                        locked.cancellation = Some(canceller.clone());
                        drop(locked);

                        let budget = if let Some(pool) = command.pool {
                            if let Some(budget) = self.budgets.get_mut(&pool) {
                                vm.increase_budget(budget.allocate());
                                Some(budget)
                            } else {
                                command.result.send(Err(TaskError::Exception(
                                    RootedValue::Symbol(Symbol::from("no_budget")),
                                )));
                                return;
                            }
                        } else {
                            None
                        };
                        tasks.push(command.id, command.pool, vm, command.result, budget);
                    }
                    Err(err) => {
                        let err = match err {
                            PrepareError::Compilation(errors) => TaskError::Compilation(errors),
                            PrepareError::Execution(err) => TaskError::Exception(
                                err.as_value().upgrade(guard).expect("just allocated"),
                            ),
                        };
                        command.result.send(Err(err));
                    }
                }
            }
            ThreadCommand::NewBudgetPool(pool) => {
                self.budgets.insert(
                    pool.0.id,
                    ThreadBudget {
                        pool,
                        exhausted_at: Cell::new(0),
                        exhausted: VecDeque::new(),
                        tasks: Set::new(),
                    },
                );
            }
            ThreadCommand::DestroyBudgetPool(pool) => {
                let Some(mut budget) = self.budgets.remove(&pool) else {
                    return;
                };
                for paused in std::mem::take(&mut budget.value.tasks).drain() {
                    tasks.complete_task(
                        paused,
                        Err(TaskError::Exception(RootedValue::Symbol(Symbol::from(
                            "no_budget",
                        )))),
                        &mut self.budgets,
                    );
                }
            }
            ThreadCommand::Cancel(global_id) => {
                let Some(task_id) = tasks.registered.get(&global_id).copied() else {
                    return;
                };
                let task =
                    tasks.complete_task(task_id, Err(TaskError::Cancelled), &mut self.budgets);

                if task.executing {
                    let (index, _) = tasks
                        .executing
                        .iter()
                        .enumerate()
                        .find(|(_, id)| task_id == **id)
                        .expect("task is executing");
                    tasks.executing.remove(index);
                }
            }
        }
    }

    fn process_commands(
        &mut self,
        canceller: &TaskCanceller,
        tasks: &mut ReactorTasks,
        guard: &mut CollectionGuard<'_>,
    ) -> ControlFlow<()> {
        loop {
            match self.receiver.try_recv() {
                Ok(command) => self.process_command(command, canceller, tasks, guard),
                Err(TryRecvError::Empty) => return ControlFlow::Continue(()),
                Err(TryRecvError::Disconnected) => return ControlFlow::Break(()),
            }
        }
    }

    fn wake_exhausted(&mut self, tasks: &mut ReactorTasks) {
        for (_, budget) in &mut self.budgets {
            let last_updated = budget.pool.0.last_updated.load(Ordering::Relaxed);
            if budget.exhausted_at.get() != last_updated {
                budget.exhausted_at.set(last_updated);
                for task_id in budget.exhausted.drain(..) {
                    let Some(task) = tasks.all.get_mut(task_id) else {
                        continue;
                    };
                    task.executing = true;
                    tasks.executing.push_back(task_id);
                }
            }
        }
    }

    fn execute_executing(
        &mut self,
        tasks: &mut ReactorTasks,
        guard: &mut CollectionGuard<'_>,
        parker: &Parker,
    ) {
        for _ in 0..tasks.executing.len() {
            let task_id = tasks.executing[0];
            let task = &mut tasks.all[task_id];
            let mut vm_context = task.vm.context(guard);
            let mut future = vm_context.resume_for_async(Duration::from_micros(100));
            let pinned_future = Pin::new(&mut future);

            let mut context = Context::from_waker(&task.waker);
            match panic::catch_unwind(AssertUnwindSafe(|| pinned_future.poll(&mut context))) {
                Ok(Poll::Ready(Ok(result))) => {
                    drop(future);
                    let result = root_result(Ok(result), &mut vm_context);
                    drop(vm_context);
                    tasks.complete_running_task(result, &mut self.budgets);
                }
                Ok(Poll::Ready(Err(ExecutionError::Exception(err)))) => {
                    drop(future);
                    let result = root_result(Err(err), &mut vm_context);
                    drop(vm_context);
                    tasks.complete_running_task(result, &mut self.budgets);
                }
                Ok(Poll::Ready(Err(ExecutionError::Waiting)) | Poll::Pending) => {
                    task.executing = false;
                    tasks.executing.pop_front();
                }
                Ok(Poll::Ready(Err(ExecutionError::Timeout))) => {
                    // Task is still executing, but took longer than its
                    // time slice. Keep it in queue for the next iteration
                    // of the loop.
                    tasks.executing.rotate_left(1);
                }
                Ok(Poll::Ready(Err(ExecutionError::NoBudget))) => {
                    if let Some(budget) = task
                        .budget_pool
                        .and_then(|pool_id| self.budgets.get_mut(&pool_id))
                    {
                        let allocated = budget.allocate();
                        if allocated > 0 {
                            // We gathered some budget, give it back to the
                            // task and keep it executing.
                            vm_context.increase_budget(allocated);
                            tasks.executing.rotate_left(1);
                        } else {
                            let mut parked_threads =
                                budget.pool.0.potentially_parked_threads.lock();
                            // In the time it took to lock the thread, we
                            // might have received budget.
                            let allocated = budget.allocate();
                            if allocated > 0 {
                                vm_context.increase_budget(allocated);
                                tasks.executing.rotate_left(1);
                            } else {
                                task.executing = false;
                                parked_threads
                                    .entry(self.id)
                                    .or_insert_with(|| parker.unparker().clone());
                                budget.exhausted.push_back(task_id);
                                tasks.executing.pop_front();
                            }
                        }
                    } else {
                        drop(future);
                        let result = root_result(
                            Err(Fault::NoBudget.as_exception(&mut vm_context)),
                            &mut vm_context,
                        );
                        drop(vm_context);
                        tasks.complete_running_task(result, &mut self.budgets);
                    }
                }
                Err(mut panic) => {
                    drop(future);
                    let (mut summary, backtrace) = PANIC_INFO.take().unwrap_or_default();
                    if let Some(backtrace) = backtrace {
                        let _result = write!(&mut summary, "\n{backtrace}");
                    }
                    let result = root_result(
                        Err(Value::dynamic(
                            List::from_iter([
                                Value::from(SymbolRef::from("panic")),
                                Value::from(SymbolRef::from(summary)),
                            ]),
                            vm_context.guard(),
                        )),
                        &mut vm_context,
                    );
                    drop(vm_context);
                    tasks.complete_running_task(result, &mut self.budgets);
                    while let Err(new_panic) =
                        panic::catch_unwind(AssertUnwindSafe(move || drop(panic)))
                    {
                        panic = new_panic;
                    }
                }
            }
        }
    }
}

struct ThreadBudget {
    pool: ReactorBudgetPool,
    exhausted_at: Cell<u64>,
    exhausted: VecDeque<LotId>,
    tasks: Set<LotId>,
}

impl ThreadBudget {
    fn allocate(&self) -> usize {
        let mut allocated = 0;
        let allocation_amount = self.pool.0.config.allocation_size;
        let generation = self.pool.0.last_updated.load(Ordering::Relaxed);
        self.pool
            .0
            .budget
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |available| {
                if let Some(remaining) = available.checked_sub(allocation_amount) {
                    allocated = allocation_amount;
                    Some(remaining)
                } else {
                    allocated = available;
                    Some(0)
                }
            })
            .expect("always returned a value");
        if allocated == 0 {
            self.exhausted_at.set(generation);
        }
        allocated
    }
}

struct WokenTasks {
    tasks: Mutex<Set<usize>>,
    data: Arc<PerThreadData>,
}

struct ReactorTaskWaker {
    task: usize,
    woken: Arc<WokenTasks>,
}

impl Wake for ReactorTaskWaker {
    fn wake(self: Arc<Self>) {
        if self.woken.tasks.lock().insert(self.task) {
            self.woken.data.unparker.unpark();
        }
    }
}

#[derive(Default, Clone)]
struct ResultHandle(Arc<ResultHandleData>);

impl ResultHandle {
    fn send(&self, result: Result<RootedValue, TaskError>) {
        let mut data = self.0.locked.lock();
        data.result = Some(result);
        for waker in data.wakers.drain(..) {
            waker.wake();
        }
        drop(data);
        self.0.sync.notify_all();
    }

    #[allow(clippy::needless_pass_by_value)]
    fn recv<Deadline>(&self, deadline: Deadline) -> Deadline::Result
    where
        Deadline: ResultDeadline,
    {
        let mut data = self.0.locked.lock();
        loop {
            if let Some(result) = &data.result {
                return deadline.result(result.clone());
            } else if !deadline.wait(&self.0.sync, &mut data) {
                return deadline.cancelled_result();
            }
        }
    }

    fn try_recv(&self) -> Option<Result<RootedValue, TaskError>> {
        self.0.locked.lock().result.clone()
    }
}

impl Debug for ResultHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_tuple("ResultHandle");

        if let Some(data) = self.0.locked.try_lock() {
            f.field(&data.result);
        }

        f.finish()
    }
}

impl ContainsNoRefs for ResultHandle {}

trait ResultDeadline {
    type Result;

    fn wait<T>(&self, sync: &Condvar, mutex_guard: &mut MutexGuard<'_, T>) -> bool;
    fn result(&self, result: Result<RootedValue, TaskError>) -> Self::Result;
    fn cancelled_result(&self) -> Self::Result;
}

impl ResultDeadline for () {
    type Result = Result<RootedValue, TaskError>;

    fn wait<T>(&self, sync: &Condvar, mutex_guard: &mut MutexGuard<'_, T>) -> bool {
        sync.wait(mutex_guard);
        true
    }

    fn result(&self, result: Result<RootedValue, TaskError>) -> Self::Result {
        result
    }

    fn cancelled_result(&self) -> Self::Result {
        unreachable!()
    }
}

impl ResultDeadline for Instant {
    type Result = Option<Result<RootedValue, TaskError>>;

    fn wait<T>(&self, sync: &Condvar, mutex_guard: &mut MutexGuard<'_, T>) -> bool {
        !sync.wait_until(mutex_guard, *self).timed_out()
    }

    fn result(&self, result: Result<RootedValue, TaskError>) -> Self::Result {
        Some(result)
    }

    fn cancelled_result(&self) -> Self::Result {
        None
    }
}

#[derive(Default)]
struct ResultHandleData {
    sync: Condvar,
    locked: Mutex<ResultHandleResult>,
}

/// An error waiting for a task to execute.
#[derive(Debug, Clone)]
pub enum TaskError {
    /// The task's execution was cancelled.
    Cancelled,
    /// The source failed to compile.
    Compilation(Vec<Ranged<compiler::Error>>),
    /// An uncaught exception was raised while executing the task.
    Exception(RootedValue),
}

#[derive(Default)]
struct ResultHandleResult {
    result: Option<Result<RootedValue, TaskError>>,
    cancelled: bool,
    cancellation: Option<TaskCanceller>,
    wakers: Vec<Waker>,
}

struct ReactorTask {
    vm: Vm,
    budget_pool: Option<BudgetPoolId>,
    waker: Waker,
    global_id: usize,
    executing: bool,
    result: ResultHandle,
}

struct ReactorTasks {
    all: Lots<ReactorTask>,
    executing: VecDeque<LotId>,
    registered: Map<usize, LotId>,
    woken: Arc<WokenTasks>,
}

impl ReactorTasks {
    fn new(data: Arc<PerThreadData>) -> Self {
        Self {
            all: Lots::default(),
            executing: VecDeque::default(),
            registered: Map::default(),
            woken: Arc::new(WokenTasks {
                tasks: Mutex::default(),
                data,
            }),
        }
    }

    fn push(
        &mut self,
        global_id: usize,
        budget_pool: Option<BudgetPoolId>,
        vm: Vm,
        result: ResultHandle,
        budget: Option<&mut ThreadBudget>,
    ) -> LotId {
        let id = self.all.push(ReactorTask {
            vm,
            budget_pool,
            waker: Waker::from(Arc::new(ReactorTaskWaker {
                task: global_id,
                woken: self.woken.clone(),
            })),
            executing: true,
            result,
            global_id,
        });
        self.executing.push_back(id);
        self.registered.insert(global_id, id);
        if let Some(budget) = budget {
            budget.tasks.insert(id);
        }

        id
    }

    fn has_work(&self) -> bool {
        !(self.executing.is_empty() && self.woken.tasks.lock().is_empty())
    }
    fn complete_running_task(
        &mut self,
        result: Result<RootedValue, RootedValue>,
        budgets: &mut Map<BudgetPoolId, ThreadBudget>,
    ) {
        let task_id = self.executing.pop_front().expect("no running task");
        self.complete_task(task_id, result.map_err(TaskError::Exception), budgets);
    }

    fn complete_task(
        &mut self,
        task_id: LotId,
        result: Result<RootedValue, TaskError>,
        budgets: &mut Map<BudgetPoolId, ThreadBudget>,
    ) -> ReactorTask {
        let task = self.all.remove(task_id).expect("task missing");
        task.result.send(result);
        self.registered.remove(&task.global_id);
        if let Some(budget) = task.budget_pool.and_then(|p| budgets.get_mut(&p)) {
            budget.tasks.remove(&task_id);
        }
        task
    }

    fn wake_woken(&mut self) {
        let mut woken = self.woken.tasks.lock();
        for woken in woken.drain() {
            let Some(id) = self.registered.get(&woken).copied() else {
                continue;
            };
            let Some(task) = self.all.get_mut(id) else {
                continue;
            };
            if !task.executing {
                task.executing = true;
                self.executing.push_back(id);
            }
        }
        drop(woken);
    }
}

#[derive(Clone)]
struct TaskCanceller {
    canceller: Arc<dyn Cancel>,
    unparker: Unparker,
}

impl TaskCanceller {
    fn cancel(&self, global_id: usize) {
        self.canceller.cancel(global_id);
        self.unparker.unpark();
    }
}

trait Cancel: Send + Sync + 'static {
    fn cancel(&self, id: usize);
}

impl<Work> Cancel for Sender<ThreadCommand<Work>>
where
    Work: WorkUnit,
{
    fn cancel(&self, id: usize) {
        let _result = self.send(ThreadCommand::Cancel(id));
    }
}

/// A handle to a task spawned in a reactor.
#[derive(Trace)]
pub struct TaskHandle {
    global_id: usize,
    result: ResultHandle,
}

impl TaskHandle {
    /// Blocks the current thread until the task is finished.
    ///
    /// This function is not safe to execute from async code. [`TaskHandle`]
    /// implements [`Future`] and can be awaited.
    pub fn join(&self) -> Result<RootedValue, TaskError> {
        self.result.recv(())
    }

    /// Checks if the task has executed, returning the result if it has.
    ///
    /// This function returns `None` if the task is pending execution or still
    /// executing.
    ///
    /// This function is safe to call from both async and non-async code.
    #[must_use]
    pub fn try_join(&self) -> Option<Result<RootedValue, TaskError>> {
        self.result.try_recv()
    }

    /// Blocks the current thread until the task is finished or `deadline` has
    /// passed.
    ///
    /// This function is not safe to execute from async code. [`TaskHandle`]
    /// implements [`Future`] and can be awaited, and the future can be
    /// cancelled using the async runtime's timeout functionality.
    #[must_use]
    pub fn join_until(&self, deadline: Instant) -> Option<Result<RootedValue, TaskError>> {
        self.result.recv(deadline)
    }

    /// Blocks the current thread until the task is finished or `duration` has
    /// elapsed.
    ///
    /// This function is not safe to execute from async code. [`TaskHandle`]
    /// implements [`Future`] and can be awaited, and the future can be
    /// cancelled using the async runtime's timeout functionality.
    #[must_use]
    pub fn join_for(&self, duration: Duration) -> Option<Result<RootedValue, TaskError>> {
        self.join_until(Instant::now() + duration)
    }

    /// Cancels the task if it is still running.
    ///
    /// Joining the task will return [`TaskError::Cancelled`] if the task was
    /// successfully cancelled.
    pub fn cancel(&self) {
        let mut locked = self.result.0.locked.lock();
        locked.cancelled = true;
        if let Some(cancellation) = &locked.cancellation {
            cancellation.cancel(self.global_id);
        }
    }
}

impl CustomType for TaskHandle {
    fn muse_type(&self) -> &muse_lang::runtime::value::TypeRef {
        static TYPE: RustType<TaskHandle> = RustType::new("TaskHandle", |t| {
            t.with_call(|_| {
                |this, vm, _arity| {
                    let waker = vm.waker().clone();
                    let mut context = Context::from_waker(&waker);
                    let mut future = &*this;
                    match Pin::new(&mut future).poll(&mut context) {
                        Poll::Ready(result) => result.map(|v| v.downgrade()).map_err(|e| match e {
                            TaskError::Cancelled => {
                                Fault::Exception(Value::Symbol(SymbolRef::from("cancelled")))
                            }
                            TaskError::Compilation(errors) => {
                                let err = errors.first().expect("at least one error");
                                Fault::Exception(Value::Symbol(SymbolRef::from(format!(
                                    "{}-{}: {}",
                                    err.range().start,
                                    err.range().end(),
                                    err.0
                                ))))
                            }
                            TaskError::Exception(e) => Fault::Exception(e.downgrade()),
                        }),
                        Poll::Pending => Err(Fault::Waiting),
                    }
                }
            })
        });
        &TYPE
    }
}

impl Debug for TaskHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskHandle")
            .field("global_id", &self.global_id)
            .field("result", &self.result)
            .finish()
    }
}

impl Future for &'_ TaskHandle {
    type Output = Result<RootedValue, TaskError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut data = self.result.0.locked.lock();
        if let Some(result) = &data.result {
            Poll::Ready(result.clone())
        } else {
            let will_wake = data.wakers.iter().any(|w| w.will_wake(cx.waker()));
            if !will_wake {
                data.wakers.push(cx.waker().clone());
            }
            Poll::Pending
        }
    }
}

/// An operation could not be completed because the reactor is not running.
#[derive(Debug)]
pub struct ReactorShutdown;

impl Display for ReactorShutdown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("reactor is not running")
    }
}

/// A handle to a spawned `Reactor<Work>`].
#[derive(Debug)]
pub struct ReactorHandle<Work = NoWork> {
    data: Arc<HandleData<Work>>,
}

impl<Work> ReactorHandle<Work>
where
    Work: WorkUnit,
{
    /// Returns a new module containing the reactor's functionality.
    ///
    /// This module needs to be declared within `parent` before it can be
    /// accessed.
    #[must_use]
    pub fn runtime_module_in(
        &self,
        parent: Dynamic<Module>,
        guard: &CollectionGuard<'_>,
    ) -> Rooted<Module> {
        let reactor = self.clone();
        let module = Rooted::new(Module::new(Some(parent)), guard);
        module.declare(
            "spawn_call",
            Access::Public,
            Value::dynamic(
                RustFunction::new(move |ctx: &mut VmContext<'_, '_>, arity: Arity| {
                    if arity == 2 {
                        let vm = ctx.cloned_vm();
                        vm.set_register(Register(0), ctx[Register(0)]);
                        let f = ctx[Register(0)]
                            .as_rooted::<Function>(ctx.guard())
                            .ok_or(Fault::NotAFunction)?;
                        let args = ctx[Register(1)]
                            .as_downcast_ref::<List>(ctx.guard())
                            .ok_or(Fault::ExpectedList)?
                            .to_vec()
                            .into_iter()
                            .map(|v| v.upgrade(ctx.guard()).ok_or(Fault::ValueFreed))
                            .collect::<Result<Vec<_>, _>>()?;
                        if let Some(code) =
                            f.body(Arity::try_from(args.len()).map_err(|_| Fault::InvalidArity)?)
                        {
                            let task = reactor.spawn_call(code.clone(), args).map_err(|_| {
                                Fault::Exception(Value::Symbol(SymbolRef::from(
                                    "reactor-is-shut-down",
                                )))
                            })?;
                            Ok(Value::dynamic(task, ctx.guard()))
                        } else {
                            Err(Fault::InvalidArity)
                        }
                    } else {
                        Err(Fault::InvalidArity)
                    }
                }),
                guard,
            ),
        );
        module
    }

    fn spawn_spawnable(
        &self,
        spawnable: Spawnable<Work>,
        pool: Option<BudgetPoolId>,
    ) -> Result<TaskHandle, ReactorShutdown> {
        let command = Spawn::new(&self.data.next_task_id, spawnable, pool);
        let handle = TaskHandle {
            result: command.result.clone(),
            global_id: command.id,
        };
        self.data
            .sender
            .send(Command::Spawn(command))
            .map_err(|_| ReactorShutdown)?;
        Ok(handle)
    }

    /// Spawns `vm` in the reactor, returning a handle to the spawned task.
    ///
    /// [`Builder::new_vm`] can be used to customize what functionality is
    /// available in every virtual machine.
    pub fn spawn(&self, vm: Vm) -> Result<TaskHandle, ReactorShutdown> {
        self.spawn_spawnable(Spawnable::Spawn(vm), None)
    }

    /// Spawns a task that compiles and executes `source`.
    ///
    /// [`Builder::new_vm`] can be used to customize how `source` is compiled
    /// and what functionality is available in every virtual machine.
    pub fn spawn_source(&self, source: impl Into<String>) -> Result<TaskHandle, ReactorShutdown> {
        self.spawn_spawnable(Spawnable::SpawnSource(source.into()), None)
    }

    /// Spawns a task that executes `code` after loading `args` to the virtual
    /// machine registers.
    ///
    /// [`Builder::new_vm`] can be used to customize what functionality is
    /// available in every virtual machine.
    pub fn spawn_call(
        &self,
        code: Code,
        args: Vec<RootedValue>,
    ) -> Result<TaskHandle, ReactorShutdown> {
        self.spawn_spawnable(Spawnable::SpawnCall(code, args), None)
    }

    /// Spawns a task that executes `work`.
    ///
    /// This function allows a user-specified type to be spawned and converted
    /// into a task using the [`WorkUnit`] trait.
    ///
    /// [`Builder::new_vm`] can be used to customize the `Work` generic and what
    /// functionality is available in every virtual machine.
    pub fn spawn_work(&self, work: Work) -> Result<TaskHandle, ReactorShutdown> {
        self.spawn_spawnable(Spawnable::SpawnWork(work), None)
    }

    /// Shuts the reactor down.
    ///
    /// Only the first call to this function will do anything. All subsequent
    /// calls will return `Ok(())` without blocking. The call that shuts the
    /// reactor down will block the current thread until the reactor has shut
    /// down or an error has occurred.
    pub fn shutdown(&self) -> Result<(), Box<dyn Any + Send + 'static>> {
        if self
            .data
            .shared
            .shutdown
            .compare_exchange(false, true, Ordering::Release, Ordering::Relaxed)
            .is_ok()
        {
            let mut threads = self.data.threads.lock();

            for thread in &*threads {
                thread.data.unparker.unpark();
            }
            while let Some(thread) = threads.pop() {
                thread.handle.join()?;
            }
        }

        Ok(())
    }

    /// Returns a new budget pool.
    pub fn create_budget_pool(
        &self,
        config: BudgetPoolConfig,
    ) -> Result<BudgetPoolHandle<Work>, ReactorShutdown> {
        loop {
            let Some(id) = NonZeroUsize::new(
                self.data
                    .next_budget_pool_id
                    .fetch_add(1, Ordering::Relaxed),
            ) else {
                continue;
            };
            let pool = ReactorBudgetPool(Arc::new(ReactorBudgetPoolData {
                id: BudgetPoolId(id),
                budget: AtomicUsize::new(config.start),
                last_updated: AtomicU64::new(0),
                config,
                potentially_parked_threads: Mutex::default(),
            }));

            if self
                .data
                .sender
                .send(Command::NewBudgetPool(pool.clone()))
                .is_err()
            {
                return Err(ReactorShutdown);
            }

            return Ok(BudgetPoolHandle(Arc::new(BudgetPoolHandleData {
                pool,
                reactor: self.clone(),
            })));
        }
    }
}

impl<Work> Clone for ReactorHandle<Work> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl<Work> ContainsNoRefs for ReactorHandle<Work> where Work: WorkUnit {}

#[derive(Debug)]
struct SharedReactorData {
    shutdown: AtomicBool,
}

#[derive(Debug)]
struct PerThread<Work> {
    spawner: Sender<ThreadCommand<Work>>,
    handle: JoinHandle<()>,
    data: Arc<PerThreadData>,
}

#[derive(Debug)]
struct PerThreadData {
    num: usize,
    unparker: Unparker,
    executing: AtomicUsize,
    total: AtomicUsize,
}

impl PerThreadData {
    fn new(num: usize, unparker: Unparker) -> Self {
        Self {
            num,
            unparker,
            executing: AtomicUsize::new(0),
            total: AtomicUsize::new(0),
        }
    }
}

#[derive(Debug)]
struct HandleData<Work> {
    sender: Sender<Command<Work>>,
    threads: Arc<Mutex<Vec<PerThread<Work>>>>,
    shared: Arc<SharedReactorData>,
    next_task_id: AtomicUsize,
    next_budget_pool_id: AtomicUsize,
}

enum Command<Work> {
    Spawn(Spawn<Work>),
    NewBudgetPool(ReactorBudgetPool),
    DestroyBudgetPool(BudgetPoolId),
}

#[derive(Clone)]
struct ReactorBudgetPool(Arc<ReactorBudgetPoolData>);

impl ReactorBudgetPool {
    fn increase_budget(&self, amount: usize) {
        let previous_budget = self
            .0
            .budget
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |available| {
                let max = if self.0.config.maximum == 0 {
                    usize::MAX
                } else {
                    self.0.config.maximum
                };
                Some(available.saturating_add(amount).min(max))
            })
            .expect("always returned a value");
        self.0.last_updated.fetch_add(1, Ordering::SeqCst);
        // Only look for parked threads if our budget was smaller than the
        // allocation amount.
        if previous_budget < self.0.config.allocation_size {
            let mut parked_threads = self.0.potentially_parked_threads.lock();
            for th in parked_threads.drain() {
                th.value.unpark();
            }
        }
    }
}

struct ReactorBudgetPoolData {
    id: BudgetPoolId,
    budget: AtomicUsize,
    last_updated: AtomicU64,
    config: BudgetPoolConfig,
    potentially_parked_threads: Mutex<Map<usize, Unparker>>,
}

struct Spawn<Work> {
    id: usize,
    pool: Option<BudgetPoolId>,
    what: Spawnable<Work>,
    result: ResultHandle,
}

impl<Work> Spawn<Work> {
    fn new(ids: &AtomicUsize, kind: Spawnable<Work>, pool: Option<BudgetPoolId>) -> Self {
        let result = ResultHandle::default();

        Self {
            id: ids.fetch_add(1, Ordering::Acquire),
            pool,
            what: kind,
            result,
        }
    }
}

enum Spawnable<Work> {
    Spawn(Vm),
    SpawnSource(String),
    SpawnCall(Code, Vec<RootedValue>),
    SpawnWork(Work),
}

/// An error while preparing and executing code.
#[derive(Debug)]
pub enum PrepareError {
    /// One or more compilation errors.
    Compilation(Vec<Ranged<compiler::Error>>),
    /// An error occurred while executing code.
    Execution(ExecutionError),
}

impl From<ExecutionError> for PrepareError {
    fn from(err: ExecutionError) -> Self {
        Self::Execution(err)
    }
}

impl From<Fault> for PrepareError {
    fn from(err: Fault) -> Self {
        Self::Execution(ExecutionError::Exception(match err.as_symbol() {
            Ok(sym) => Value::Symbol(sym.downgrade()),
            Err(exc) => exc,
        }))
    }
}

impl From<Vec<Ranged<compiler::Error>>> for PrepareError {
    fn from(err: Vec<Ranged<compiler::Error>>) -> Self {
        Self::Compilation(err)
    }
}

/// Creates new [`Vm`]s for a [`Reactor`].
pub trait NewVm<Work>: Send + Sync + 'static {
    /// Returns a newly initialized virtual machine.
    fn new_vm(
        &self,
        guard: &mut CollectionGuard<'_>,
        reactor: &ReactorHandle<Work>,
    ) -> Result<Vm, PrepareError>;

    /// Returns a virtual machine that is prepared to execute `source`.
    fn compile_and_prepare(
        &self,
        source: &str,
        guard: &mut CollectionGuard<'_>,
        reactor: &ReactorHandle<Work>,
    ) -> Result<Vm, PrepareError> {
        let code = Compiler::default().with(source).build(guard)?;
        let vm = self.new_vm(guard, reactor)?;
        vm.prepare(&code, guard)?;
        Ok(vm)
    }

    /// Returns a virtual machine that is prepared to execute `code` with
    /// `args`.
    fn prepare_call(
        &self,
        code: Code,
        args: Vec<RootedValue>,
        guard: &mut CollectionGuard<'_>,
        reactor: &ReactorHandle<Work>,
    ) -> Result<Vm, PrepareError> {
        let vm = self.new_vm(guard, reactor)?;
        let mut ctx = vm.context(guard);
        for (arg, reg) in args.into_iter().zip(0..=255) {
            ctx[Register(reg)] = arg.downgrade();
        }
        ctx.prepare(&code)?;
        drop(ctx);

        Ok(vm)
    }
}

impl<F, Work> NewVm<Work> for F
where
    F: Fn(&mut CollectionGuard<'_>, &ReactorHandle<Work>) -> Result<Vm, PrepareError>
        + Send
        + Sync
        + 'static,
    Work: WorkUnit,
{
    fn new_vm(
        &self,
        guard: &mut CollectionGuard<'_>,
        reactor: &ReactorHandle<Work>,
    ) -> Result<Vm, PrepareError> {
        self(guard, reactor)
    }
}

impl<Work: WorkUnit> NewVm<Work> for () {
    fn new_vm(
        &self,
        guard: &mut CollectionGuard<'_>,
        reactor: &ReactorHandle<Work>,
    ) -> Result<Vm, PrepareError> {
        let locked_vm = Vm::new(guard);
        let mut vm = locked_vm.context(guard);
        let reactor_module = reactor.runtime_module_in(vm.root_module(), vm.guard());
        vm.declare("task", Value::Dynamic(reactor_module.as_any_dynamic()))?;
        drop(vm);
        Ok(locked_vm)
    }
}

/// A custom unit of work for a [`Reactor`].
pub trait WorkUnit: Sized + Send + 'static {
    /// Initializes a new virtual machine for this unit of work.
    fn initialize(
        self,
        vms: &dyn NewVm<Self>,
        guard: &mut CollectionGuard<'_>,
        reactor: &ReactorHandle<Self>,
    ) -> Result<Vm, PrepareError>;
}

impl WorkUnit for Code {
    fn initialize(
        self,
        vms: &dyn NewVm<Self>,
        guard: &mut CollectionGuard<'_>,
        reactor: &ReactorHandle<Self>,
    ) -> Result<Vm, PrepareError> {
        let vm = vms.new_vm(guard, reactor)?;
        vm.prepare(&self, guard)?;
        Ok(vm)
    }
}

/// A [`WorkUnit`] that can not be instantiated.
pub enum NoWork {}

impl WorkUnit for NoWork {
    fn initialize(
        self,
        vms: &dyn NewVm<Self>,
        guard: &mut CollectionGuard<'_>,
        reactor: &ReactorHandle<Self>,
    ) -> Result<Vm, PrepareError> {
        vms.new_vm(guard, reactor)
    }
}

/// The unique identifier of a budget pool in a [`Reactor`].
///
/// IDs are only unique within the same reactor.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub struct BudgetPoolId(NonZeroUsize);

/// A handle to a budget pool.
///
/// Tasks spawned in a budget pool will only be able to execute while the budget
/// pool has remaining budget.
///
/// When all handles to a budget pool have been dropped, all outstanding tasks
/// belonging to the budget pool will be cancelled.
pub struct BudgetPoolHandle<Work = NoWork>(Arc<BudgetPoolHandleData<Work>>);

impl<Work> ContainsNoRefs for BudgetPoolHandle<Work> where Work: WorkUnit {}

struct BudgetPoolHandleData<Work> {
    pool: ReactorBudgetPool,
    reactor: ReactorHandle<Work>,
}

impl<Work> BudgetPoolHandle<Work>
where
    Work: WorkUnit,
{
    /// Returns the id of this pool.
    #[must_use]
    pub fn id(&self) -> BudgetPoolId {
        self.0.pool.0.id
    }

    /// Spawns `vm` in the reactor, returning a handle to the spawned task.
    ///
    /// [`Builder::new_vm`] can be used to customize what functionality is
    /// available in every virtual machine.
    ///
    /// This task will execute with a shared budget and will be paused when no
    /// budget is available in this pool.
    pub fn spawn(&self, vm: Vm) -> Result<TaskHandle, ReactorShutdown> {
        self.0
            .reactor
            .spawn_spawnable(Spawnable::Spawn(vm), Some(self.0.pool.0.id))
    }

    /// Spawns a task that compiles and executes `source`.
    ///
    /// [`Builder::new_vm`] can be used to customize how `source` is compiled
    /// and what functionality is available in every virtual machine.
    ///
    /// This task will execute with a shared budget and will be paused when no
    /// budget is available in this pool.
    pub fn spawn_source(&self, source: impl Into<String>) -> Result<TaskHandle, ReactorShutdown> {
        self.0.reactor.spawn_spawnable(
            Spawnable::SpawnSource(source.into()),
            Some(self.0.pool.0.id),
        )
    }

    /// Spawns a task that executes `code` after loading `args` to the virtual
    /// machine registers.
    ///
    /// [`Builder::new_vm`] can be used to customize what functionality is
    /// available in every virtual machine.
    ///
    /// This task will execute with a shared budget and will be paused when no
    /// budget is available in this pool.
    pub fn spawn_call(
        &self,
        code: Code,
        args: Vec<RootedValue>,
    ) -> Result<TaskHandle, ReactorShutdown> {
        self.0
            .reactor
            .spawn_spawnable(Spawnable::SpawnCall(code, args), Some(self.0.pool.0.id))
    }

    /// Spawns a task that executes `work`.
    ///
    /// This function allows a user-specified type to be spawned and converted
    /// into a task using the [`WorkUnit`] trait.
    ///
    /// [`Builder::new_vm`] can be used to customize the `Work` generic and what
    /// functionality is available in every virtual machine.
    ///
    /// This task will execute with a shared budget and will be paused when no
    /// budget is available in this pool.
    pub fn spawn_work(&self, work: Work) -> Result<TaskHandle, ReactorShutdown> {
        self.0
            .reactor
            .spawn_spawnable(Spawnable::SpawnWork(work), Some(self.0.pool.0.id))
    }

    /// Adds `amount` to the budget.
    ///
    /// This function cannot increase the budget above
    /// [`BudgetPoolConfig::maximum`].
    pub fn increase_budget(&self, amount: usize) {
        self.0.pool.increase_budget(amount);
    }

    /// Returns the currently remaining budget.
    #[must_use]
    pub fn remaining_budget(&self) -> usize {
        self.0.pool.0.budget.load(Ordering::Relaxed)
    }

    /// Returns a handle to the reactor this pool belongs to.
    #[must_use]
    pub fn reactor(&self) -> &ReactorHandle<Work> {
        &self.0.reactor
    }
}

impl<Work> Debug for BudgetPoolHandle<Work> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BudgetPoolHandle")
            .field("id", &self.0.pool.0.id.0.get())
            .field("budget", &self.0.pool.0.id.0.get())
            .finish()
    }
}

impl<Work> Clone for BudgetPoolHandle<Work> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<Work> Drop for BudgetPoolHandleData<Work> {
    fn drop(&mut self) {
        let _ = self
            .reactor
            .data
            .sender
            .send(Command::DestroyBudgetPool(self.pool.0.id));
    }
}

/// The settings for a budget pool in a [`Reactor`].
#[non_exhaustive]
#[must_use]
pub struct BudgetPoolConfig {
    /// The maximum budget this pool can contain.
    ///
    /// If this is 0, there is no maximum budget.
    pub maximum: usize,
    /// Each time a virtual machine runs out of budget, this is the amount of
    /// budget it should be allocated.
    pub allocation_size: usize,
    /// When the pool is initialized, this is the initial budget of the pool.
    pub start: usize,
    /// When `recharge_amount` and `recharge_every` are non-zero, the budget is
    /// replenished by `recharge_amount` each time `recharge_every` elapses.
    pub recharge_amount: usize,
    /// When `recharge_amount` and `recharge_every` are non-zero, the budget is
    /// replenished by `recharge_amount` each time `recharge_every` elapses.
    pub recharge_every: Duration,
}

impl Default for BudgetPoolConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl BudgetPoolConfig {
    /// Returns the default budget pool configuration.
    pub const fn new() -> Self {
        Self {
            maximum: 0,
            start: 0,
            allocation_size: 100,
            recharge_amount: 0,
            recharge_every: Duration::ZERO,
        }
    }

    /// Sets the starting budget for this pool.
    pub const fn starting_with(mut self, start: usize) -> Self {
        self.start = start;
        self
    }

    /// Sets the maximum budget for this pool.
    pub const fn with_maximum(mut self, maximum: usize) -> Self {
        self.maximum = maximum;
        self
    }

    /// Sets the amount to allocate for each budget request.
    pub const fn with_per_task_allocation(mut self, allocation_size: usize) -> Self {
        self.allocation_size = allocation_size;
        self
    }

    /// Sets the budget to automatically replenish by `amount` every time
    /// `recharge_every` elapses.
    pub const fn with_recharge(mut self, amount: usize, recharge_every: Duration) -> Self {
        self.recharge_amount = amount;
        self.recharge_every = recharge_every;
        self
    }

    /// Returns true if this configuration automatically recharges.
    #[must_use]
    pub fn recharges(&self) -> bool {
        self.recharge_every > Duration::ZERO && self.recharge_amount > 0
    }
}

#[cfg(test)]
mod tests;
