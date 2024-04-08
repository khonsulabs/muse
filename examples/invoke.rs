use muse::syntax::Sources;
use muse::vm::Vm;
use refuse::CollectionGuard;

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
