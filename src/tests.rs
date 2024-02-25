use crate::compiler::Compiler;
use crate::syntax::SourceCode;
use crate::vm::bitcode::BitcodeBlock;
use crate::vm::{Code, ExecutionError, Register, Vm};

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
    assert_eq!(vm.execute(&code).unwrap_err(), ExecutionError::NoBudget);
    for value in 0..=COUNT_TO {
        // Step through by allowing one op at a time.
        vm.increase_budget(1);
        assert_eq!(vm.resume().unwrap_err(), ExecutionError::NoBudget);
        assert_eq!(vm[Register(0)].as_i64(), Some(value));
    }
    vm.increase_budget(1);
    assert_eq!(vm.resume().unwrap().as_i64(), Some(COUNT_TO));
}

#[test]
fn module_budgeting() {
    const MAX_OPS: usize = 24;
    let code = Compiler::compile(&SourceCode::anonymous(
        r"
            mod foo {
                pub var a = 1;
                a = a + 1;
                a = a + 1;
                a = a + 1;
                a = a + 1;
            };

            foo.a
        ",
    ))
    .unwrap();
    let mut vm = Vm::default();
    // Turn on budgeting, but don't give any budget.
    vm.increase_budget(0);
    assert_eq!(vm.execute(&code).unwrap_err(), ExecutionError::NoBudget);
    let mut ops = 0;
    for _ in 0..MAX_OPS {
        ops += 1;
        // Step through by allowing one op at a time.
        vm.increase_budget(1);
        match vm.resume() {
            Ok(value) => {
                assert_eq!(value.as_i64(), Some(5));
                break;
            }
            Err(err) => assert_eq!(err, ExecutionError::NoBudget),
        }
    }
    println!("Executed in {ops} steps");
    assert!(ops > 6);
    assert!(ops < MAX_OPS);
}
