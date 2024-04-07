use muse::vm::Vm;
use refuse::CollectionGuard;

fn main() {
    let mut guard = CollectionGuard::acquire();
    let vm = Vm::new(&guard);
    vm.compile_and_execute(
        r"
        pub fn muse_function(n) {
            n * 2
        }
    ",
        &mut guard,
    )
    .unwrap();
    assert_eq!(
        vm.invoke("muse_function", [21], &mut guard)
            .unwrap()
            .as_i64(),
        Some(42)
    );
}
