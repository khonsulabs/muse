use muse::compiler::Compiler;
use muse::vm::bitcode::BitcodeBlock;
use muse::vm::Vm;

fn main() {
    let mut vm = Vm::default();
    let code = Compiler::compile("{1 + 2 + 3; 1 + 2 - 3}").unwrap();
    let bitcode = BitcodeBlock::from(&code);
    println!("{}", rsn::to_string_pretty(&bitcode));
    let result = vm.execute(&code).unwrap();
    assert_eq!(result.as_i64(), Some(0));
    // let mut vm = Vm::default();
    // let code = Code::default().with(Allocate(1)).with(Add {
    //     lhs: 1,
    //     rhs: 2,
    //     dest: Stack(0),
    // });
    // let encoded = code.encode();
    // let decoded = Code::decode_from(&encoded).unwrap();
    // vm.execute(decoded).unwrap();
    // assert_eq!(vm[0].as_i64(), Some(3));
}
