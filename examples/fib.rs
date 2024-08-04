//! An example using Muse that calculates fibonacci using a naive recursive
//! algorithm.
//!
//! This is not how someone should implement fibonacci, but it provides a way to
//! test calling *a lot* of functions while also doing some other operations
//! between function calls. This ends up being a pretty good way to profile the
//! virtual machine to see overall bottlenecks in code execution.

use muse::refuse::CollectionGuard;
use muse::vm::Vm;

fn main() {
    let mut guard = CollectionGuard::acquire();
    dbg!(Vm::new(&guard)
        .compile_and_execute(include_str!("fib.muse"), &mut guard)
        .unwrap());
}
