use std::{thread, time::Duration};

use muse_lang::{
    compiler::{self, syntax::Ranged},
    runtime::value::{Primitive, RootedValue},
    vm::Vm,
};
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
    pool.increase_budget(100).unwrap();

    assert_eq!(
        task.join().unwrap(),
        RootedValue::Primitive(Primitive::Int(3))
    );
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
