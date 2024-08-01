#![allow(missing_docs)]
use std::{
    any::Any,
    collections::VecDeque,
    fmt::Debug,
    future::Future,
    marker::PhantomData,
    num::NonZeroUsize,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    task::{Context, Poll, Wake, Waker},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use alot::{LotId, Lots};
use crossbeam_utils::sync::{Parker, Unparker};
use flume::{Receiver, SendError, Sender, TryRecvError};
use kempt::{Map, Set};
use parking_lot::{Condvar, Mutex};
use refuse::{CollectionGuard, Trace};

use crate::{
    compiler::{self, syntax::Ranged, Compiler},
    runtime::value::RustType,
    vm::{Code, ExecutionError, Fault, Vm},
};

use super::value::{CustomType, Value};

pub struct Builder<Work> {
    vm_source: Option<Arc<dyn NewVm<Work>>>,
    threads: usize,
    thread_name: Option<String>,
    work_queue_limit: Option<usize>,
    _work: PhantomData<Work>,
}

impl Builder<NoWork> {
    pub fn new() -> Self {
        Self {
            vm_source: None,
            threads: std::thread::available_parallelism().map_or(0, NonZeroUsize::get),
            thread_name: None,
            work_queue_limit: None,
            _work: PhantomData,
        }
    }

    pub fn unit<Work>(self) -> Builder<Work> {
        assert!(
            self.vm_source.is_none(),
            "new_vm must be invoked after unit() to ensure the correct ReactorHandle type"
        );
        Builder {
            vm_source: None,
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
    pub fn threads(mut self, thread_count: usize) -> Self {
        self.threads = thread_count;
        self
    }

    pub fn work_queue_limit(mut self, limit: usize) -> Self {
        self.work_queue_limit = Some(limit);
        self
    }

    pub fn new_vm<F>(mut self, new_vm: F) -> Self
    where
        F: NewVm<Work>,
    {
        self.vm_source = Some(Arc::new(new_vm));
        self
    }

    pub fn finish(self) -> ReactorHandle<Work> {
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
            }),
        };

        for num in 0..self.threads {
            let (spawn_send, spawn_recv) = flume::unbounded();
            let parker = Parker::new();
            let data = Arc::new(PerThreadData::new(parker.unparker().clone()));
            let reactor = Reactor {
                receiver: spawn_recv,
                vm_source: vm_source.clone(),
                handle: handle.clone(),
            };
            threads.push(PerThread {
                num,
                data: data.clone(),
                spawner: spawn_send,
                handle: thread::Builder::new()
                    .name(thread_name.clone())
                    .spawn(move || reactor.run(num, data, parker))
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

struct DispatcherThread<Work> {
    spawner: Sender<Command<Work>>,
    load: usize,
    unparker: Unparker,
    num: usize,
}

struct Dispatcher<Work> {
    spawns: Receiver<Command<Work>>,
    handle: ReactorHandle<Work>,
    threads: VecDeque<DispatcherThread<Work>>,
    next_rebalance: Instant,
}

impl<Work> Dispatcher<Work> {
    const REBALANCE_DELAY: Duration = Duration::from_millis(30);

    fn new(spawns: Receiver<Command<Work>>, handle: ReactorHandle<Work>) -> Self {
        let mut this = Self {
            spawns,
            handle,
            threads: VecDeque::new(),
            next_rebalance: Instant::now(),
        };
        this.cache_thread_loads(this.next_rebalance);
        this
    }

    fn cache_thread_loads(&mut self, now: Instant) {
        let threads = self.handle.data.threads.lock();
        self.threads.clear();
        for t in &*threads {
            self.threads.push_back(DispatcherThread {
                num: t.num,
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
        while let Ok(mut work) = self.spawns.recv() {
            // We loop in case a thread dies.
            loop {
                let Some(thread) = self.threads.front_mut() else {
                    return;
                };

                match thread.spawner.send(work) {
                    Ok(()) => {
                        thread.unparker.unpark();
                        thread.load += 1;
                        let new_load = thread.load;
                        println!("Spawned on {} - new load {new_load}", thread.num);
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
                        work = returned_work;
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

pub struct Reactor<Work = NoWork> {
    receiver: Receiver<Command<Work>>,
    vm_source: Arc<dyn NewVm<Work>>,
    handle: ReactorHandle<Work>,
}

impl Reactor<NoWork> {
    pub fn new() -> ReactorHandle<NoWork> {
        Builder::new().finish()
    }
}

impl<Work> Reactor<Work>
where
    Work: WorkUnit,
{
    fn run(self, thread_num: usize, data: Arc<PerThreadData>, parker: Parker) {
        let mut tasks = ReactorTasks::new(data);
        while !self.handle.data.shared.shutdown.load(Ordering::Relaxed) {
            tasks.wake_woken();
            let mut guard = CollectionGuard::acquire();
            for _ in 0..tasks.executing.len() {
                let task_id = tasks.executing[0];
                let task = &mut tasks.all[task_id];
                let mut future = task
                    .vm
                    .resume_for_async(Duration::from_micros(100), &mut guard);
                let pinned_future = Pin::new(&mut future);

                let mut context = Context::from_waker(&task.waker);
                match pinned_future.poll(&mut context) {
                    Poll::Ready(Ok(result)) => {
                        drop(future);
                        tasks.complete_running_task(Ok(result));
                    }
                    Poll::Ready(Err(ExecutionError::Waiting)) | Poll::Pending => {
                        task.executing = false;
                        tasks.executing.pop_front();
                    }
                    Poll::Ready(Err(ExecutionError::Timeout)) => {
                        // Task is still executing, but took longer than its
                        // time slice. Keep it in queue for the next iteration
                        // of the loop.
                        tasks.executing.rotate_left(1);
                    }
                    Poll::Ready(Err(ExecutionError::NoBudget)) => {
                        todo!("how do we handle budgeting")
                    }
                    Poll::Ready(Err(ExecutionError::Exception(err))) => {
                        drop(future);
                        tasks.complete_running_task(Err(err));
                    }
                }
            }

            match self.receiver.try_recv() {
                Ok(command) => {
                    let vm = match command.kind {
                        Spawnable::Spawn(vm) => Ok(vm),
                        Spawnable::SpawnSource(source) => {
                            self.vm_source
                                .compile_and_prepare(&source, &mut guard, &self.handle)
                        }
                        Spawnable::SpawnWork(work) => {
                            work.initialize(self.vm_source.as_ref(), &mut guard, &self.handle)
                        }
                    }
                    .unwrap(); // TODO handle error
                    tasks.push(command.id, vm, command.result);
                }
                Err(TryRecvError::Empty) => {}
                Err(_) => break,
            }

            if tasks.executing.is_empty() {
                let woken = tasks.woken.tasks.lock();
                if woken.is_empty() {
                    drop(woken);
                    println!("Parking {thread_num}");
                    guard.while_unlocked(|| parker.park());
                    println!("Unparked {thread_num}");
                }
            }
        }
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
    fn send(&self, result: Result<Value, Value>) {
        let mut data = self.0.locked.lock();
        data.result = Some(result);
        for waker in data.wakers.drain(..) {
            waker.wake();
        }
        drop(data);
        self.0.sync.notify_all();
    }

    fn recv(&self) -> Result<Value, Value> {
        let mut data = self.0.locked.lock();
        loop {
            if let Some(result) = &data.result {
                return result.clone();
            } else {
                self.0.sync.wait(&mut data);
            }
        }
    }

    fn recv_async(&self) -> ResultHandleFuture<'_> {
        ResultHandleFuture(self)
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

impl Trace for ResultHandle {
    const MAY_CONTAIN_REFERENCES: bool = true;

    fn trace(&self, tracer: &mut refuse::Tracer) {
        let data = self.0.locked.lock();
        match &data.result {
            Some(Ok(v) | Err(v)) => v.trace(tracer),
            _ => {}
        }
    }
}

#[derive(Debug)]
struct ResultHandleFuture<'a>(&'a ResultHandle);

impl Future for ResultHandleFuture<'_> {
    type Output = Result<Value, Value>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut data = self.0 .0.locked.lock();
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

#[derive(Default)]
struct ResultHandleData {
    sync: Condvar,
    locked: Mutex<ResultHandleResult>,
}

#[derive(Default)]
struct ResultHandleResult {
    result: Option<Result<Value, Value>>,
    wakers: Vec<Waker>,
}

struct ReactorTask {
    vm: Vm,
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
    fn push(&mut self, global_id: usize, vm: Vm, result: ResultHandle) -> LotId {
        let id = self.all.push(ReactorTask {
            vm,
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

        id
    }

    fn complete_running_task(&mut self, result: Result<Value, Value>) {
        let task_id = self.executing.pop_front().expect("no running task");
        let task = self.all.remove(task_id).expect("task missing");
        let _result = task.result.send(result);
        self.registered.remove(&task.global_id);
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
                self.executing.push_back(id)
            }
        }
        drop(woken);
    }
}

#[derive(Trace)]
pub struct TaskHandle {
    global_id: usize,
    result: ResultHandle,
}

impl TaskHandle {
    pub fn join(&self) -> Result<Value, Value> {
        self.result.recv()
    }

    pub async fn join_async(&self) -> Result<Value, Value> {
        self.result.recv_async().await
    }
}

impl CustomType for TaskHandle {
    fn muse_type(&self) -> &super::value::TypeRef {
        static TYPE: RustType<TaskHandle> = RustType::new("TaskHandle", |t| {
            t.with_call(|_| {
                |this, vm, _arity| {
                    let waker = vm.waker().clone();
                    let mut context = Context::from_waker(&waker);
                    let mut future = this.result.recv_async();
                    match Pin::new(&mut future).poll(&mut context) {
                        Poll::Ready(result) => result.map_err(Fault::Exception),
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

#[derive(Debug)]
pub struct ReactorShutdown;

#[derive(Debug)]
pub struct ReactorHandle<Work = NoWork> {
    data: Arc<HandleData<Work>>,
}

impl<Work> ReactorHandle<Work> {
    fn spawn_spawnable(&self, spawnable: Spawnable<Work>) -> Result<TaskHandle, ReactorShutdown> {
        let command = Command::new(&self.data.next_task_id, spawnable);
        let handle = TaskHandle {
            result: command.result.clone(),
            global_id: command.id,
        };
        self.data
            .sender
            .send(command)
            .map_err(|_| ReactorShutdown)?;
        Ok(handle)
    }

    pub fn spawn(&self, vm: Vm) -> Result<TaskHandle, ReactorShutdown> {
        self.spawn_spawnable(Spawnable::Spawn(vm))
    }

    pub fn spawn_source(&self, source: impl Into<String>) -> Result<TaskHandle, ReactorShutdown> {
        self.spawn_spawnable(Spawnable::SpawnSource(source.into()))
    }

    pub fn spawn_work(&self, work: Work) -> Result<TaskHandle, ReactorShutdown> {
        self.spawn_spawnable(Spawnable::SpawnWork(work))
    }

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
}

impl<Work> Clone for ReactorHandle<Work> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

#[derive(Debug)]
struct SharedReactorData {
    shutdown: AtomicBool,
}

#[derive(Debug)]
struct PerThread<Work> {
    num: usize,
    spawner: Sender<Command<Work>>,
    handle: JoinHandle<()>,
    data: Arc<PerThreadData>,
}

#[derive(Debug)]
struct PerThreadData {
    unparker: Unparker,
    executing: AtomicUsize,
    total: AtomicUsize,
}

impl PerThreadData {
    fn new(unparker: Unparker) -> Self {
        Self {
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
}

struct Command<Work> {
    id: usize,
    kind: Spawnable<Work>,
    result: ResultHandle,
}

impl<Work> Command<Work> {
    fn new(ids: &AtomicUsize, kind: Spawnable<Work>) -> Self {
        let result = ResultHandle::default();

        Self {
            id: ids.fetch_add(1, Ordering::Acquire),
            kind,
            result,
        }
    }
}

enum Spawnable<Work> {
    Spawn(Vm),
    SpawnSource(String),
    SpawnWork(Work),
}

#[derive(Debug)]
pub enum PrepareError {
    Compilation(Vec<Ranged<compiler::Error>>),
    Execution(ExecutionError),
}

impl From<ExecutionError> for PrepareError {
    fn from(err: ExecutionError) -> Self {
        Self::Execution(err)
    }
}

impl From<Vec<Ranged<compiler::Error>>> for PrepareError {
    fn from(err: Vec<Ranged<compiler::Error>>) -> Self {
        Self::Compilation(err)
    }
}

pub trait NewVm<Work>: Send + Sync + 'static {
    fn new_vm(&self, guard: &mut CollectionGuard<'_>, reactor: &ReactorHandle<Work>) -> Vm;
    fn compile_and_prepare(
        &self,
        source: &str,
        guard: &mut CollectionGuard<'_>,
        reactor: &ReactorHandle<Work>,
    ) -> Result<Vm, PrepareError> {
        let code = Compiler::default().with(source).build(guard)?;
        let vm = self.new_vm(guard, reactor);
        vm.prepare(&code, guard)?;
        Ok(vm)
    }
}

impl<F, Work> NewVm<Work> for F
where
    F: Fn(&mut CollectionGuard<'_>, &ReactorHandle<Work>) -> Vm + Send + Sync + 'static,
    Work: WorkUnit,
{
    fn new_vm(&self, guard: &mut CollectionGuard<'_>, reactor: &ReactorHandle<Work>) -> Vm {
        self(guard, reactor)
    }
}

impl<Work: WorkUnit> NewVm<Work> for () {
    fn new_vm(&self, guard: &mut CollectionGuard<'_>, _reactor: &ReactorHandle<Work>) -> Vm {
        Vm::new(guard)
    }
}

pub trait WorkUnit: Sized + Send + 'static {
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
        let vm = vms.new_vm(guard, reactor);
        vm.prepare(&self, guard)?;
        Ok(vm)
    }
}

pub enum NoWork {}

impl WorkUnit for NoWork {
    fn initialize(
        self,
        vms: &dyn NewVm<Self>,
        guard: &mut CollectionGuard<'_>,
        reactor: &ReactorHandle<Self>,
    ) -> Result<Vm, PrepareError> {
        Ok(vms.new_vm(guard, reactor))
    }
}

#[test]
fn works() {
    let reactor = Reactor::new();
    let task = reactor.spawn_source("1 + 2").unwrap();
    let result = task.join().unwrap();
    assert_eq!(result, Value::Int(3));
}

#[test]
fn spawning() {
    use crate::runtime::value::RustFunction;
    use crate::vm::{Arity, Function, Register, VmContext};

    fn add_spawn_function(vm: Vm, guard: &mut CollectionGuard<'_>, reactor: &ReactorHandle) -> Vm {
        let reactor = reactor.clone();
        vm.declare(
            "spawn",
            Value::dynamic(
                RustFunction::new(move |ctx: &mut VmContext<'_, '_>, arity: Arity| {
                    if arity == 2 {
                        println!("Spawn called with {:?}", ctx[Register(0)]);
                        let vm = ctx.cloned_vm();
                        vm.set_register(Register(0), ctx[Register(0)]);
                        let f = ctx[Register(1)].as_rooted::<Function>(ctx.guard()).unwrap();
                        vm.prepare_call(&f, Arity(1), &mut CollectionGuard::acquire())
                            .unwrap();
                        Ok(Value::dynamic(reactor.spawn(vm).unwrap(), ctx.guard()))
                    } else {
                        Err(Fault::InvalidArity)
                    }
                }),
                &guard,
            ),
            guard,
        )
        .unwrap();
        vm
    }

    let reactor = Builder::new()
        .new_vm(|guard: &mut CollectionGuard<'_>, reactor: &ReactorHandle| {
            let vm = Vm::new(guard);
            add_spawn_function(vm, guard, reactor)
        })
        .finish();
    let task = reactor
        .spawn_source(
            r"
                fn recurse_spawn {
                    0 => 0,
                    n => {
                        let task = spawn(n, fn(n) {
                            recurse_spawn(n - 1)
                        });
                        task() + n
                    },
                };

                recurse_spawn(100)
            ",
        )
        .unwrap();
    let result = task.join().unwrap();
    assert_eq!(result, Value::Int((0..=100).sum()));
}
