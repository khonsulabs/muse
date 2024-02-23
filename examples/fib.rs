use muse::compiler::Compiler;
use muse::syntax::SourceCode;
use muse::vm::Vm;

fn main() {
    let code = Compiler::compile(&SourceCode::anonymous(include_str!("fib.muse"))).unwrap();
    dbg!(Vm::default().execute(&code).unwrap());
}
