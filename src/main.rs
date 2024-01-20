use muse::instructions::{Add, Allocate, Stack};
use muse::{Code, Vm};

fn main() {
    let mut vm = Vm::default();
    vm.execute(Code::default().with(Allocate(1)).with(Add {
        lhs: 1,
        rhs: 2,
        dest: Stack(0),
    }))
    .unwrap();
    assert_eq!(vm[0].as_i64(), Some(3));
}
