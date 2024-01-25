use muse::compiler::Compiler;
use muse::symbol::Symbol;
use muse::value::{RustFunction, Value};
use muse::vm::{Fault, Register, Vm};

fn main() {
    let code = Compiler::compile("is_even(42)").unwrap();

    let mut vm = Vm::default();
    vm.declare(
        Symbol::from("is_even"),
        Value::dynamic(RustFunction::new(|vm: &mut Vm, arity| {
            assert_eq!(arity, 1);

            let arg = &vm[Register(0)];
            println!("Called with {arg:?}");

            if let Some(int) = arg.as_i64() {
                Ok(Value::Bool(int % 2 == 0))
            } else {
                Err(Fault::UnsupportedOperation)
            }
        })),
    );
    assert!(vm
        .execute(&code, None)
        .unwrap()
        .eq(&mut vm, &Value::Bool(true)));
}
