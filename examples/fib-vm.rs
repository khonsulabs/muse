use muse::symbol::Symbol;
use muse::syntax::CompareKind::LessThanOrEqual;
use muse::vm::bitcode::BitcodeBlock;
use muse::vm::{Function, Register as R, Vm, VmContext};
use refuse::CollectionGuard;

fn main() {
    let mut guard = CollectionGuard::acquire();
    let vm = Vm::new(&guard);
    let mut fib = BitcodeBlock::default();

    // Special case two or less
    let two_or_less = fib.new_label();
    fib.compare(LessThanOrEqual, R(0), 2, two_or_less);

    // Calculate n - 1, store in R0
    fib.sub(R(0), 1, R(0));
    // Calculate n - 2, store on the stack.
    let temporary = fib.new_variable();
    fib.sub(R(0), 1, temporary);
    // Recurse, calculating fib(n - 1), storing in R1.
    fib.call((), 1);
    fib.copy(R(0), R(1));
    // Move n - 2 into R0
    fib.copy(temporary, R(0));
    // Move fib(n - 1) into temporary.
    fib.copy(R(1), temporary);
    // Recurse, calculating fib(n - 2), storing into R0.
    fib.call((), 1);
    // Add fib(n - 2) + fib(n - 1).
    fib.add(R(0), temporary, R(0));
    fib.return_early();

    // Less than two
    fib.label(two_or_less);
    fib.copy(1, R(0));
    let fib = fib.to_code(&guard);

    let mut main = BitcodeBlock::default();
    main.copy(35, R(0));
    main.resolve(Symbol::from("fib"), R(1));
    main.call(R(1), 1);
    let code = main.to_code(&guard);

    let mut context = VmContext::new(&vm, &mut guard);
    context
        .declare_function(dbg!(Function::new("fib").when(1, fib)))
        .unwrap();
    dbg!(context.execute(&code).unwrap());
}
