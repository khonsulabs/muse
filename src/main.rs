use muse::vm::ops::{Add, Allocate, Stack};
use muse::vm::{Code, Vm};

fn main() {
    let mut vm = Vm::default();
    let code = Code::default().with(Allocate(1)).with(Add {
        lhs: 1,
        rhs: 2,
        dest: Stack(0),
    });
    let encoded = code.encode();
    let decoded = Code::decode_from(&encoded).unwrap();
    vm.execute(decoded).unwrap();
    assert_eq!(vm[0].as_i64(), Some(3));
}
