#![allow(missing_docs)]
use std::{
    any::Any,
    collections::VecDeque,
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
    time::Duration,
};

use alot::{LotId, Lots};
use flume::{Receiver, Sender, TryRecvError};
use kempt::{Map, Set};
use parking_lot::Mutex;
use refuse::{CollectionGuard, Trace};

use crate::{
    compiler::{self, syntax::Ranged, Compiler},
    runtime::{
        symbol::SymbolRef,
        value::{RustFunction, RustType},
    },
    vm::{Code, ExecutionError, Fault, Function, Register, Vm},
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
        let shutdown = Arc::new(AtomicBool::new(false));

        let reactor = Reactor {
            receiver,
            shutdown: shutdown.clone(),
            vm_source: self.vm_source.unwrap_or_else(|| Arc::new(())),
            handle: ReactorHandle {
                data: Arc::new(HandleData {
                    sender,
                    shutdown,
                    threads: Mutex::new(Vec::new()),
                    next_task_id: AtomicUsize::new(0),
                }),
            },
        };

        for _ in 0..self.threads {
            let reactor = reactor.clone();
            threads.push(
                thread::Builder::new()
                    .name(thread_name.clone())
                    .spawn(move || reactor.run())
                    .expect("error spawning thread"),
            );
        }

        reactor.handle
    }
}

pub struct Reactor<Work = NoWork> {
    receiver: Receiver<Command<Work>>,
    shutdown: Arc<AtomicBool>,
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
    fn run(self) {
        let mut tasks = ReactorTasks::default();
        while !self.shutdown.load(Ordering::Relaxed) {
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
        }
    }
}

impl<Work> Clone for Reactor<Work> {
    fn clone(&self) -> Self {
        Self {
            receiver: self.receiver.clone(),
            shutdown: self.shutdown.clone(),
            vm_source: self.vm_source.clone(),
            handle: self.handle.clone(),
        }
    }
}

struct ReactorTaskWaker {
    task: usize,
    woken: Arc<Mutex<Set<usize>>>,
}

impl Wake for ReactorTaskWaker {
    fn wake(self: Arc<Self>) {
        self.woken.lock().insert(self.task);
    }
}

struct ReactorTask {
    vm: Vm,
    waker: Waker,
    global_id: usize,
    executing: bool,
    result: Sender<Result<Value, Value>>,
}

#[derive(Default)]
struct ReactorTasks {
    all: Lots<ReactorTask>,
    executing: VecDeque<LotId>,
    registered: Map<usize, LotId>,
    woken: Arc<Mutex<Set<usize>>>,
}

impl ReactorTasks {
    fn push(&mut self, global_id: usize, vm: Vm, result: Sender<Result<Value, Value>>) -> LotId {
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
        println!("Finished task {} with result {result:?}", task.global_id);
        let _result = task.result.send(result);
    }

    fn wake_woken(&mut self) {
        let mut woken = self.woken.lock();
        for woken in woken.drain() {
            let Some(id) = self.registered.get(&woken).copied() else {
                continue;
            };
            let Some(task) = self.all.get_mut(id) else {
                continue;
            };
            if !task.executing {
                println!("Waking {woken}");
                task.executing = true;
                self.executing.push_back(id)
            }
        }
        drop(woken);
    }
}

#[derive(Debug)]
pub struct TaskHandle {
    result: Receiver<Result<Value, Value>>,
}

impl TaskHandle {
    pub fn join(&self) -> Option<Result<Value, Value>> {
        self.result.recv().ok()
    }

    pub async fn join_async(&self) -> Option<Result<Value, Value>> {
        self.result.recv_async().await.ok()
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
                        Poll::Ready(Ok(result)) => result.map_err(Fault::Exception),
                        Poll::Ready(Err(_)) => Err(Fault::Exception(Value::Symbol(
                            SymbolRef::from("result-already-read"),
                        ))),
                        Poll::Pending => Err(Fault::Waiting),
                    }
                    // if let Some(body) = this.bodies.get(&arity).or_else(|| {
                    //     this.varg_bodies
                    //         .iter()
                    //         .rev()
                    //         .find_map(|va| (va.key() <= &arity).then_some(&va.value))
                    // }) {
                    //     let module = this.module.ok_or(Fault::NotAModule)?;
                    //     vm.execute_function(body, &this, module)
                    // } else {
                    //     Err(Fault::IncorrectNumberOfArguments)
                    // }
                }
            })
        });
        &TYPE
    }
}

impl Trace for TaskHandle {
    const MAY_CONTAIN_REFERENCES: bool = true;

    fn trace(&self, tracer: &mut refuse::Tracer) {
        // todo!()
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
        let (command, result) = Command::new(&self.data.next_task_id, spawnable);
        let handle = TaskHandle { result };
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
            .shutdown
            .compare_exchange(false, true, Ordering::Release, Ordering::Relaxed)
            .is_ok()
        {
            let mut threads = self.data.threads.lock();

            while let Some(thread) = threads.pop() {
                thread.join()?;
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
struct HandleData<Work> {
    sender: Sender<Command<Work>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    shutdown: Arc<AtomicBool>,
    next_task_id: AtomicUsize,
}

struct Command<Work> {
    id: usize,
    kind: Spawnable<Work>,
    result: Sender<Result<Value, Value>>,
}

impl<Work> Command<Work> {
    fn new(ids: &AtomicUsize, kind: Spawnable<Work>) -> (Self, Receiver<Result<Value, Value>>) {
        let (sender, receiver) = flume::bounded(1);
        (
            Self {
                id: ids.fetch_add(1, Ordering::Acquire),
                kind,
                result: sender,
            },
            receiver,
        )
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
    let result = task.join().unwrap().unwrap();
    assert_eq!(result, Value::Int(3));
}

#[test]
fn spawning() {
    use crate::vm::{Arity, VmContext};

    fn add_spawn_function(vm: Vm, guard: &mut CollectionGuard<'_>, reactor: &ReactorHandle) -> Vm {
        let reactor = reactor.clone();
        vm.declare(
            "spawn",
            Value::dynamic(
                RustFunction::new(move |ctx: &mut VmContext<'_, '_>, arity: Arity| {
                    if arity == 2 {
                        println!("Spawn called with {:?}", ctx[Register(0)]);
                        let f = ctx[Register(1)].as_rooted::<Function>(ctx.guard()).unwrap();
                        let vm =
                            add_spawn_function(Vm::new(ctx.guard()), ctx.guard_mut(), &reactor);
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

                recurse_spawn(5)
            ",
        )
        .unwrap();
    let result = task.join().unwrap().unwrap();
    assert_eq!(result, Value::Int(5 + 4 + 3 + 2 + 1));
}
