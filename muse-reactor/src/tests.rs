use std::thread;
use std::time::{Duration, Instant};

use muse_lang::compiler::syntax::Ranged;
use muse_lang::compiler::{self};
use muse_lang::runtime::value::{Primitive, RootedValue};
use muse_lang::vm::Vm;
use refuse::CollectionGuard;

use crate::{BudgetPoolConfig, NoWork, Reactor, ReactorHandle, TaskError};

#[test]
fn works() {
    let reactor = Reactor::new();
    let task = reactor.spawn_source("1 + 2").unwrap();
    assert_eq!(
        task.join().unwrap(),
        RootedValue::Primitive(Primitive::Int(3))
    );
}

#[test]
fn spawning() {
    let reactor = Reactor::new();
    let task = reactor
        .spawn_source(
            r"
                let func = fn(n, func) {
                    match n {
                        0 => 0,
                        n => {
                            let task = task.spawn_call(func, [n - 1, func]);
                            task() + n
                        }
                    }
                };

                func(100, func)
            ",
        )
        .unwrap();
    assert_eq!(
        task.join().unwrap(),
        RootedValue::Primitive(Primitive::Int((0..=100).sum()))
    );
}

#[test]
fn budgeting_basic() {
    let reactor = Reactor::build()
        .new_vm(
            |guard: &mut CollectionGuard<'_>, _reactor: &ReactorHandle<NoWork>| {
                let vm = Vm::new(guard);
                vm.set_steps_per_charge(1);
                Ok(vm)
            },
        )
        .finish();
    let pool = reactor.create_budget_pool(BudgetPoolConfig::new()).unwrap();
    let task = pool.spawn_source("1 + 2").unwrap();

    // Make sure the task doesn't complete
    thread::sleep(Duration::from_secs(1));
    assert!(task.try_join().is_none());

    // Give it some budget
    pool.increase_budget(100);

    assert_eq!(
        task.join().unwrap(),
        RootedValue::Primitive(Primitive::Int(3))
    );
}

#[test]
fn automatic_recharge() {
    let reactor = Reactor::build()
        .new_vm(
            |guard: &mut CollectionGuard<'_>, _reactor: &ReactorHandle<NoWork>| {
                let vm = Vm::new(guard);
                vm.set_steps_per_charge(1);
                Ok(vm)
            },
        )
        .finish();
    let pool = reactor
        .create_budget_pool(BudgetPoolConfig::new().with_recharge(50, Duration::from_millis(500)))
        .unwrap();
    let now = Instant::now();
    let task = pool
        .spawn_source("var n = 0; while n < 60 { n = n + 1 }")
        .unwrap();

    assert_eq!(
        task.join().unwrap(),
        RootedValue::Primitive(Primitive::Int(60))
    );
    assert!(now.elapsed() > Duration::from_secs(1));
}

#[test]
fn spawn_err() {
    let reactor = Reactor::new();
    let task = reactor.spawn_source("invalid source code").unwrap();
    match task.join() {
        Err(TaskError::Compilation(errors)) => assert_eq!(
            errors,
            vec![Ranged(
                compiler::Error::Syntax(crate::compiler::syntax::ParseError::ExpectedEof),
                compiler::syntax::SourceRange {
                    source_id: compiler::syntax::SourceId::anonymous(),
                    start: 8,
                    length: 6
                }
            )]
        ),
        other => unreachable!("unexpected result: {other:?}"),
    };
}

#[test]
fn task_cancellation() {
    let reactor = Reactor::new();
    // Spawn a task with an infinite loop
    let task = reactor.spawn_source("loop {}").unwrap();
    // Wait a bit to make sure it's running.
    assert!(task.try_join_for(Duration::from_secs(1)).is_none());

    // Cancel the task.
    println!("Cancelling");
    task.cancel();

    // Ensure we get a cancellation error.
    match task.join() {
        Err(TaskError::Cancelled) => {}
        other => unreachable!("unexpected result: {other:?}"),
    }
}

#[test]
fn pool_cancellation() {
    let reactor = Reactor::new();
    // Spawn a task with an infinite loop
    let pool = reactor
        .create_budget_pool(BudgetPoolConfig::default())
        .unwrap();
    let task = pool.spawn_source("loop {}").unwrap();
    // Wait a bit to make sure it's running.
    assert!(task.try_join_for(Duration::from_secs(1)).is_none());

    // Drop the pool, which should cause the task to be cancelled for running
    // out of budget.
    println!("Dropping pool");
    drop(pool);

    // Ensure we get a no_budget error.
    match task.join() {
        Err(TaskError::Exception(RootedValue::Symbol(sym))) if sym == "no_budget" => {}
        other => unreachable!("unexpected result: {other:?}"),
    }
}