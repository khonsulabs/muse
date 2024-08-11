//! A multi-producer, multi-consumer channel of [`RootedValue`]s.
//!
//! This channel supports sending and receiving from blocking and async code,
//! and the channel types [`ValueSender`] and [`ValueReceiver`] can be passed
//! into the Muse runtime.

use std::{
    collections::VecDeque,
    fmt::Debug,
    future::Future,
    ops::ControlFlow,
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    task::{Context, Poll, Waker},
};

use muse_lang::{
    refuse::{CollectionGuard, ContainsNoRefs},
    runtime::{
        list::List,
        symbol::{StaticSymbol, SymbolRef},
        value::{
            CustomType, Dynamic, Rooted, RootedValue, RustFunction, RustType,
            StaticRustFunctionTable, TypeRef, Value,
        },
    },
    vm::{bitcode::Access, Fault, Module, Register, Vm, VmContext},
};
use parking_lot::{Condvar, Mutex, MutexGuard};

fn new_channel(limit: Option<usize>) -> (ValueSender, ValueReceiver) {
    let data = Arc::new(ChannelData {
        limit,
        locked: Mutex::new(LockedData {
            messages: if let Some(limit) = limit {
                VecDeque::with_capacity(limit)
            } else {
                VecDeque::new()
            },
            recv_wakers: Vec::new(),
            send_wakers: Vec::new(),
        }),
        value_read: Condvar::new(),
        value_sent: Condvar::new(),
        receivers: AtomicUsize::new(1),
        senders: AtomicUsize::new(1),
    });

    (ValueSender { data: data.clone() }, ValueReceiver { data })
}

/// Returns a new channel with no capacity restrictions.
///
/// This channel will panic in out-of-memory situations.
#[must_use]
pub fn unbounded() -> (ValueSender, ValueReceiver) {
    new_channel(None)
}

/// Returns a new channel with a fixed capacity.
#[must_use]
pub fn bounded(limit: usize) -> (ValueSender, ValueReceiver) {
    new_channel(Some(limit))
}

/// Registers a `new_channel` Muse function in `self`.
pub trait WithNewChannel: Sized {
    /// Registers a `new_channel` Muse function in `self`.
    #[must_use]
    fn with_new_channel(self, guard: &mut CollectionGuard<'_>) -> Self {
        self.with_new_channel_named(&NEW_CHANNEL, guard)
    }
    /// Registers [`new_channel_function()`] in `self` as `name`.
    #[must_use]
    fn with_new_channel_named(
        self,
        name: impl Into<SymbolRef>,
        guard: &mut CollectionGuard<'_>,
    ) -> Self;
}

impl WithNewChannel for Vm {
    fn with_new_channel_named(
        self,
        name: impl Into<SymbolRef>,
        guard: &mut CollectionGuard<'_>,
    ) -> Self {
        let module = self.context(guard).root_module();
        let _same_module = module.with_new_channel_named(name, guard);
        self
    }
}

/// A symbol for the default name for the new channel function.
pub static NEW_CHANNEL: StaticSymbol = StaticSymbol::new("new_channel");

impl WithNewChannel for Dynamic<Module> {
    fn with_new_channel_named(
        self,
        name: impl Into<SymbolRef>,
        guard: &mut CollectionGuard<'_>,
    ) -> Self {
        if let Some(module) = self.as_rooted(guard) {
            declare_new_channel_in(name, &module, guard);
        }
        self
    }
}

impl WithNewChannel for Rooted<Module> {
    fn with_new_channel_named(
        self,
        name: impl Into<SymbolRef>,
        guard: &mut CollectionGuard<'_>,
    ) -> Self {
        declare_new_channel_in(name, &self, guard);
        self
    }
}

/// Declares [`new_channel_function()`] in `module` with `name`.
pub fn declare_new_channel_in(
    name: impl Into<SymbolRef>,
    module: &Module,
    guard: &CollectionGuard<'_>,
) {
    module.declare(
        name,
        Access::Public,
        Value::dynamic(new_channel_function(), guard),
    );
}

/// Returns a function that creates a new channel.
///
/// When called with no arguments, this function returns an unbounded channel.
///
/// When called with one argument, it is converted to an integer and used as the
/// bounds of a bounded channel.
#[must_use]
pub fn new_channel_function() -> RustFunction {
    RustFunction::new(|vm: &mut VmContext<'_, '_>, arity| {
        let (sender, receiver) = match arity.0 {
            0 => unbounded(),
            1 => {
                let Some(limit) = vm[Register(0)].as_usize() else {
                    return Err(Fault::ExpectedInteger);
                };
                bounded(limit)
            }
            _ => return Err(Fault::IncorrectNumberOfArguments),
        };
        let result = [
            Value::dynamic(sender, vm.guard()),
            Value::dynamic(receiver, vm.guard()),
        ]
        .into_iter()
        .collect::<List>();
        Ok(Value::dynamic(result, vm.guard()))
    })
}

/// A sender of [`RootedValue`]s to be received by one or more [`ValueReceiver`]s.
pub struct ValueSender {
    data: Arc<ChannelData>,
}

impl ValueSender {
    /// Pushes `value` to the queue for [`ValueReceiver`]s to read from.
    ///
    /// If this sender is for a bounded channel and the channel is full, this
    /// function will block the current thread until space is available or it is
    /// disconnected. Use [`send_async(value).await`](Self::send_async) in async
    /// code instead to ensure proper interaction with the async runtime.
    ///
    /// # Errors
    ///
    /// This function returns `value` if the channel is disconnected before
    /// `value` can be pushed.
    pub fn send(&self, value: RootedValue) -> Result<(), RootedValue> {
        if self.is_disconnected() {
            return Err(value);
        }

        if let Some(limit) = self.data.limit {
            if let Err(TrySendError::Disconnected(value) | TrySendError::Full(value)) = self
                .bounded_send(value, limit, |locked| {
                    self.data.value_read.wait(locked);
                    ControlFlow::Continue(())
                })
            {
                Err(value)
            } else {
                Ok(())
            }
        } else {
            self.unbounded_send(value);
            Ok(())
        }
    }

    /// Tries to push `value` to the queue for [`ValueReceiver`]s to read from.
    ///
    /// This function is safe to be invoked from both async and non-async code.
    ///
    /// # Errors
    ///
    /// - [`TrySendError::Disconnected`]: All associated [`ValueReceiver`]s have
    ///       been dropped.
    /// - [`TrySendError::Full`]: The value cannot be pushed due to the channel
    ///       being full.
    pub fn try_send(&self, value: RootedValue) -> Result<(), TrySendError> {
        if self.is_disconnected() {
            return Err(TrySendError::Disconnected(value));
        }

        if let Some(limit) = self.data.limit {
            self.bounded_send(value, limit, |_| ControlFlow::Break(()))
        } else {
            self.unbounded_send(value);
            Ok(())
        }
    }

    /// Pushes `value` to the queue for [`ValueReceiver`]s to read from.
    ///
    /// If this sender is for a bounded channel and the channel is full, this
    /// function will block the current task until space is available or it is
    /// disconnected.
    ///
    /// # Errors
    ///
    /// This function returns `value` if the channel is disconnected before
    /// `value` can be pushed.
    pub fn send_async(&self, value: RootedValue) -> SendAsync<'_> {
        SendAsync {
            value: Some(value),
            sender: self,
        }
    }

    fn unbounded_send(&self, value: RootedValue) {
        self.finish_send(value, self.data.locked.lock());
    }

    fn bounded_send(
        &self,
        value: RootedValue,
        limit: usize,
        mut wait: impl FnMut(&mut MutexGuard<'_, LockedData>) -> ControlFlow<()>,
    ) -> Result<(), TrySendError> {
        let mut locked = self.data.locked.lock();
        // Acquiring the lock could have given enough time for a disconnect to
        // happen.
        if self.is_disconnected() {
            return Err(TrySendError::Disconnected(value));
        }
        while locked.messages.len() >= limit {
            if wait(&mut locked).is_break() {
                return Err(TrySendError::Full(value));
            } else if self.is_disconnected() {
                return Err(TrySendError::Disconnected(value));
            }
        }

        self.finish_send(value, locked);
        Ok(())
    }

    fn finish_send(&self, value: RootedValue, mut locked: MutexGuard<'_, LockedData>) {
        locked.messages.push_back(value);
        for waker in locked.recv_wakers.drain(..) {
            waker.wake();
        }
        drop(locked);
        self.data.value_sent.notify_all();
    }

    /// Returns the number of associated [`ValueReceiver`]s that have not been
    /// dropped.
    #[must_use]
    pub fn receivers(&self) -> usize {
        self.data.receivers.load(Ordering::Relaxed)
    }

    /// Returns true if all associated [`ValueReceiver`]s have been dropped.
    #[must_use]
    pub fn is_disconnected(&self) -> bool {
        self.receivers() == 0
    }
}

/// Errors returned from [`ValueSender::try_send`].
#[derive(Debug, Clone, PartialEq)]
pub enum TrySendError {
    /// All associated [`ValueReceiver`]s have been dropped.
    Disconnected(RootedValue),
    /// The value cannot be pushed due to the channel being full.
    Full(RootedValue),
}

fn debug_channel(
    name: &str,
    channel: &ChannelData,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result {
    let locked = channel.locked.lock();
    f.debug_struct(name)
        .field("receivers", &channel.receivers.load(Ordering::Relaxed))
        .field("senders", &channel.senders.load(Ordering::Relaxed))
        .field("messages", &locked.messages.len())
        .finish()
}

impl Debug for ValueSender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        debug_channel("Sender", &self.data, f)
    }
}

impl ContainsNoRefs for ValueSender {}

impl CustomType for ValueSender {
    fn muse_type(&self) -> &TypeRef {
        static TYPE: RustType<ValueSender> = RustType::new("Sender", |t| {
            t.with_invoke(|_| {
                |this, vm, name, arity| {
                    static FUNCTIONS: StaticRustFunctionTable<ValueSender> =
                        StaticRustFunctionTable::new(|table| {
                            table
                                .with_fn("send", 1, |vm, this| {
                                    let weak_value = vm[Register(0)];
                                    let value =
                                        weak_value.upgrade(vm.guard()).ok_or(Fault::ValueFreed)?;
                                    let mut cx = Context::from_waker(vm.waker());
                                    let mut future = this.send_async(value);
                                    match Pin::new(&mut future).poll(&mut cx) {
                                        Poll::Ready(Ok(())) => Ok(Value::NIL),
                                        Poll::Ready(Err(_)) => Err(Fault::Exception(weak_value)),
                                        Poll::Pending => Err(Fault::Waiting),
                                    }
                                })
                                .with_fn("try_send", 1, |vm, this| {
                                    let weak_value = vm[Register(0)];
                                    let value =
                                        weak_value.upgrade(vm.guard()).ok_or(Fault::ValueFreed)?;
                                    let error = match this.try_send(value) {
                                        Ok(()) => return Ok(Value::TRUE),
                                        Err(TrySendError::Disconnected(_value)) => {
                                            SymbolRef::from("disconnected")
                                        }
                                        Err(TrySendError::Full(_value)) => SymbolRef::from("full"),
                                    };
                                    let list = [Value::Symbol(error), weak_value]
                                        .into_iter()
                                        .collect::<List>();
                                    Ok(Value::dynamic(list, vm.guard()))
                                })
                        });

                    FUNCTIONS.invoke(vm, name, arity, &this)
                }
            })
        });
        &TYPE
    }
}

impl Clone for ValueSender {
    fn clone(&self) -> Self {
        self.data.senders.fetch_add(1, Ordering::Relaxed);
        Self {
            data: self.data.clone(),
        }
    }
}

impl Drop for ValueSender {
    fn drop(&mut self) {
        if self.data.senders.fetch_sub(1, Ordering::Relaxed) == 1 {
            // We were the last sender. Wake any readers to notify that they
            // are disconnected.
            let mut locked = self.data.locked.lock();
            for waker in locked.recv_wakers.drain(..) {
                waker.wake();
            }
            drop(locked);
            self.data.value_sent.notify_all();
        }
    }
}

/// A receiver of [`RootedValue`]s that are sent by one or more
/// [`ValueSender`]s.
pub struct ValueReceiver {
    data: Arc<ChannelData>,
}

impl ValueReceiver {
    /// Reads the oldest [`RootedValue`] from the channel, blocking the current
    /// thread until a value is received.
    ///
    /// Returns `None` if all [`ValueSender`]s have been dropped before a value
    /// is received.
    #[must_use]
    pub fn recv(&self) -> Option<RootedValue> {
        let mut locked = self.data.locked.lock();
        loop {
            if let Some(value) = locked.messages.pop_front() {
                for waker in locked.send_wakers.drain(..) {
                    waker.wake();
                }
                drop(locked);
                self.data.value_read.notify_all();
                return Some(value);
            } else if self.is_disconnected() {
                return None;
            }

            self.data.value_sent.wait(&mut locked);
        }
    }

    /// Reads the oldest [`RootedValue`] from the channel, blocking the current
    /// task until a value is received.
    ///
    /// Returns `None` if all [`ValueSender`]s have been dropped before a value
    /// is received.
    pub fn recv_async(&self) -> RecvAsync<'_> {
        RecvAsync(self)
    }

    /// Returns the number of associated [`ValueSender`]s that have not been
    /// dropped.
    #[must_use]
    pub fn senders(&self) -> usize {
        self.data.senders.load(Ordering::Relaxed)
    }

    /// Returns true if all associated [`ValueSender`]s have been dropped.
    #[must_use]
    pub fn is_disconnected(&self) -> bool {
        self.senders() == 0
    }
}

impl Debug for ValueReceiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        debug_channel("Receiver", &self.data, f)
    }
}

impl Clone for ValueReceiver {
    fn clone(&self) -> Self {
        self.data.receivers.fetch_add(1, Ordering::Relaxed);
        Self {
            data: self.data.clone(),
        }
    }
}

impl ContainsNoRefs for ValueReceiver {}

impl CustomType for ValueReceiver {
    fn muse_type(&self) -> &TypeRef {
        static TYPE: RustType<ValueReceiver> = RustType::new("Receiver", |t| {
            t.with_invoke(|_| {
                |this, vm, name, arity| {
                    static FUNCTIONS: StaticRustFunctionTable<ValueReceiver> =
                        StaticRustFunctionTable::new(|table| {
                            table
                                .with_fn("next", 0, |vm, this| {
                                    let mut cx = Context::from_waker(vm.waker());
                                    let mut future = this.recv_async();
                                    match Pin::new(&mut future).poll(&mut cx) {
                                        Poll::Ready(Some(value)) => Ok(value.downgrade()),
                                        // TODO out of bounds isn't exactly the best descriptor.
                                        Poll::Ready(None) => Err(Fault::OutOfBounds),
                                        Poll::Pending => Err(Fault::Waiting),
                                    }
                                })
                                .with_fn("iterate", 0, |_vm, this| {
                                    Ok(Value::Dynamic(this.as_any_dynamic()))
                                })
                        });

                    FUNCTIONS.invoke(vm, name, arity, &this)
                }
            })
        });
        &TYPE
    }
}

impl Drop for ValueReceiver {
    fn drop(&mut self) {
        if self.data.receivers.fetch_sub(1, Ordering::Relaxed) == 1 {
            // We were the last receiver. Wake any senders to notify that they
            // are disconnected.
            let mut locked = self.data.locked.lock();
            for waker in locked.send_wakers.drain(..) {
                waker.wake();
            }
            drop(locked);
            self.data.value_read.notify_all();
        }
    }
}

/// A future that sends a [`RootedValue`] when awaited.
#[must_use = "Futures must be awaited to execute"]
pub struct SendAsync<'a> {
    value: Option<RootedValue>,
    sender: &'a ValueSender,
}

impl Future for SendAsync<'_> {
    type Output = Result<(), RootedValue>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let Some(value) = self.value.take() else {
            return Poll::Ready(Ok(()));
        };
        if self.sender.is_disconnected() {
            return Poll::Ready(Err(value));
        }

        if let Some(limit) = self.sender.data.limit {
            let send_result = self.sender.bounded_send(value, limit, |locked| {
                let will_wake = locked.send_wakers.iter().any(|w| w.will_wake(cx.waker()));
                if !will_wake {
                    locked.send_wakers.push(cx.waker().clone());
                }
                ControlFlow::Break(())
            });
            match send_result {
                Ok(()) => Poll::Ready(Ok(())),
                Err(TrySendError::Disconnected(value)) => Poll::Ready(Err(value)),
                Err(TrySendError::Full(value)) => {
                    self.value = Some(value);
                    Poll::Pending
                }
            }
        } else {
            self.sender.unbounded_send(value);
            Poll::Ready(Ok(()))
        }
    }
}

/// A future that receives a [`RootedValue`] when awaited.
#[must_use = "Futures must be awaited to execute"]
pub struct RecvAsync<'a>(&'a ValueReceiver);

impl Future for RecvAsync<'_> {
    type Output = Option<RootedValue>;

    fn poll(self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let mut locked = self.0.data.locked.lock();

        if let Some(value) = locked.messages.pop_front() {
            for waker in locked.send_wakers.drain(..) {
                waker.wake();
            }
            drop(locked);
            self.0.data.value_read.notify_all();
            return Poll::Ready(Some(value));
        } else if self.0.is_disconnected() {
            return Poll::Ready(None);
        }

        let will_wake = locked.recv_wakers.iter().any(|w| w.will_wake(cx.waker()));
        if !will_wake {
            locked.recv_wakers.push(cx.waker().clone());
        }
        Poll::Pending
    }
}

struct ChannelData {
    limit: Option<usize>,
    locked: Mutex<LockedData>,
    value_read: Condvar,
    value_sent: Condvar,
    receivers: AtomicUsize,
    senders: AtomicUsize,
}

struct LockedData {
    messages: VecDeque<RootedValue>,
    recv_wakers: Vec<Waker>,
    send_wakers: Vec<Waker>,
}

#[cfg(test)]
mod tests;
