use crate::vm::bitcode::BitcodeBlock;
use crate::vm::{Code, Fault, Register, Vm};

#[test]
fn budgeting() {
    const COUNT_TO: i64 = 42;
    let mut code = BitcodeBlock::default();
    for value in 0..=COUNT_TO {
        code.copy(value, Register(0));
    }
    let code = Code::from(&code);
    let mut vm = Vm::default();
    // Turn on budgeting, but don't give any budget.
    vm.increase_budget(0);
    assert_eq!(vm.execute(&code).unwrap_err(), Fault::NoBudget);
    for value in 0..=COUNT_TO {
        // Step through by allowing one op at a time.
        vm.increase_budget(1);
        assert_eq!(vm.resume().unwrap_err(), Fault::NoBudget);
        assert_eq!(vm[Register(0)].as_i64(), Some(value));
    }
    vm.increase_budget(1);
    assert_eq!(vm.resume().unwrap().as_i64(), Some(COUNT_TO));
}
