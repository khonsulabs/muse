use muse::compiler::Compiler;
use muse::symbol::Symbol;
use muse::value::{AsyncFunction, Value};
use muse::vm::{Arity, Fault, Register, Vm};

fn main() {
    let (input_sender, input_receiver) = flume::unbounded();
    let (output_sender, output_receiver) = flume::unbounded();

    std::thread::spawn(move || {
        while let Ok(input) = input_receiver.recv() {
            output_sender.send(input + 1).unwrap();
        }
    });

    let async_func = AsyncFunction::new(move |vm: &mut Vm, _arity: Arity| {
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
    )
    .unwrap();
    let mut vm = Vm::default();
    vm.declare(Symbol::from("increment_async"), Value::dynamic(async_func))
        .unwrap();

    assert_eq!(
        pollster::block_on(vm.execute_async(&code).unwrap()).unwrap(),
        Value::Int(5)
    );
}
