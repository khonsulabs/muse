use muse::compiler::Compiler;
use muse::vm::Vm;

fn main() {
    let mut vm = Vm::default();
    let code = Compiler::compile(
        r"
        {
            var a = 0 + 1;
            a = 2;
            a + b
        }
        ",
    )
    .unwrap();
    let result = vm.execute(&code).unwrap();
    assert_eq!(result.as_i64(), Some(5));
}
