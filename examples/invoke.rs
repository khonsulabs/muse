//! An example demonstrating how to invoke a Muse-written function from Rust.
//!
//! Rust code can only access public declarations. Attempting to invoke a
//! private function will result in an error.

use muse::compiler::syntax::Sources;
use muse::refuse::CollectionGuard;
use muse::vm::Vm;

fn main() {
    let mut guard = CollectionGuard::acquire();
    let vm = Vm::new(&guard);
    let mut sources = Sources::default();
    let source = sources.push(
        "definition",
        r"
            pub fn muse_function(n) {
                n * 2
            }
        ",
    );
    if let Err(err) = vm.compile_and_execute(source, &mut guard) {
        let mut formatted = String::new();
        sources
            .format_error(err, &mut vm.context(&mut guard), &mut formatted)
            .unwrap();
        eprintln!("{formatted}");
        return;
    }

    assert_eq!(
        vm.invoke("muse_function", [21], &mut guard)
            .unwrap()
            .as_i64(),
        Some(42)
    );
}
