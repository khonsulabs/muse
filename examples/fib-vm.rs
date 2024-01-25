use muse::syntax::CompareKind::LessThanOrEqual;
use muse::vm::bitcode::BitcodeBlock;
use muse::vm::{Code, Function, Register as R, Vm};

fn main() {
    let mut vm = Vm::default();
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
    fib.call((), 1, R(1));
    // Move n - 2 into R0
    fib.copy(temporary, R(0));
    // Move fib(n - 1) into temporary.
    fib.copy(R(1), temporary);
    // Recurse, calculating fib(n - 2), storing into R0.
    fib.call((), 1, R(0));
    // Add fib(n - 2) + fib(n - 1).
    fib.add(R(0), temporary, R(0));
    fib.return_early();

    // Less than two
    fib.label(two_or_less);
    fib.copy(1, R(0));

    vm.declare_function(dbg!(Function::new("fib").when(1, &fib)));

    let mut main = BitcodeBlock::default();
    main.copy(35, R(0));
    main.resolve("fib", R(1));
    main.call(R(1), 1, R(0));
    let code = Code::from(&main);
    dbg!(vm.execute(&code, None).unwrap());
}
