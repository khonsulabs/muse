use muse::vm::Vm;
use refuse::CollectionGuard;

fn main() {
    let mut guard = CollectionGuard::acquire();
    dbg!(Vm::new(&guard)
        .compile_and_execute(include_str!("fib.muse"), &mut guard)
        .unwrap());
}
