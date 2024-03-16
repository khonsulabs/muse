use muse::compiler::Compiler;
use muse::symbol::Symbol;
use muse::syntax::SourceCode;
use muse::value::{RustFunction, Value};
use muse::vm::{Fault, Register, Vm, VmContext};
use refuse::CollectionGuard;

fn main() {
    let mut guard = CollectionGuard::acquire();
    let code = Compiler::compile(&SourceCode::anonymous("is_even(42)"), &guard).unwrap();

    let vm = Vm::new(&guard);
    let mut context = VmContext::new(&vm, &mut guard);
    context
        .declare(
            Symbol::from("is_even"),
            Value::dynamic(
                RustFunction::new(|vm: &mut VmContext<'_, '_>, arity| {
                    assert_eq!(arity, 1);

                    let arg = &vm[Register(0)];
                    println!("Called with {arg:?}");

                    if let Some(int) = arg.as_i64() {
                        Ok(Value::Bool(int % 2 == 0))
                    } else {
                        Err(Fault::UnsupportedOperation)
                    }
                }),
                context.guard(),
            ),
        )
        .unwrap();
    assert_eq!(context.execute(&code).unwrap(), Value::Bool(true));
}
