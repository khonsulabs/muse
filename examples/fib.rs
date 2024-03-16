use muse::compiler::Compiler;
use muse::syntax::SourceCode;
use muse::vm::Vm;
use refuse::CollectionGuard;

fn main() {
    let mut guard = CollectionGuard::acquire();
    let code = Compiler::compile(&SourceCode::anonymous(include_str!("fib.muse")), &guard).unwrap();
    dbg!(Vm::new(&guard).execute(&code, &mut guard).unwrap());
}
