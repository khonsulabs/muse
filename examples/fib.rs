use muse::compiler::Compiler;
use muse::vm::Vm;

fn main() {
    let code = dbg!(Compiler::compile(include_str!("fib.muse")).unwrap());
    dbg!(Vm::default().execute(&code, None).unwrap());
}
