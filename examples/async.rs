//! An example showing how to use Muse in async.

use muse::compiler::Compiler;
use muse::refuse::CollectionGuard;
use muse::runtime::symbol::Symbol;
use muse::runtime::value::{AsyncFunction, Value};
use muse::vm::{Arity, Fault, Register, Vm, VmContext};

fn main() {
    let (input_sender, input_receiver) = flume::unbounded();
    let (output_sender, output_receiver) = flume::unbounded();
    let mut guard = CollectionGuard::acquire();

    std::thread::spawn(move || {
        while let Ok(input) = input_receiver.recv() {
            output_sender.send(input + 1).unwrap();
        }
    });

    let async_func = AsyncFunction::new(move |vm: &mut VmContext<'_, '_>, _arity: Arity| {
        input_sender
            .send(vm[Register(0)].as_i64().expect("invalid arg"))
            .unwrap();
        let output_receiver = output_receiver.clone();
        async move { Ok::<_, Fault>(Value::Int(output_receiver.recv_async().await.unwrap())) }
    });

    let code = Compiler::compile(
        r"
            var a = increment_async(0);
            a = increment_async(a);
            a = increment_async(a);
            a = increment_async(a);
            increment_async(a)
        ",
        &guard,
    )
    .unwrap();
    let vm = Vm::new(&guard);
    let mut context = VmContext::new(&vm, &mut guard);
    context
        .declare(
            Symbol::from("increment_async"),
            Value::dynamic(async_func, context.guard()),
        )
        .unwrap();

    assert_eq!(
        pollster::block_on(context.execute_async(&code).unwrap()).unwrap(),
        Value::Int(5)
    );
}
