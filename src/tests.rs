use crate::compiler::Compiler;
use crate::exception::Exception;
use crate::symbol::Symbol;
use crate::syntax::token::{Paired, Token};
use crate::syntax::{Ranged, SourceCode, SourceRange};
use crate::value::Value;
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

#[test]
fn invoke() {
    let code = Compiler::compile(&SourceCode::anonymous(
        r"
            pub fn test(n) => n * 2;
            fn private(n) => n * 2;
        ",
    ))
    .unwrap();
    let mut vm = Vm::default();
    vm.execute(&code).unwrap();

    let Value::Int(result) = vm.invoke(&Symbol::from("test"), [Value::Int(3)]).unwrap() else {
        unreachable!()
    };
    assert_eq!(result, 6);
    let ExecutionError::Exception(exception) = vm
        .invoke(&Symbol::from("private"), [Value::Int(3)])
        .unwrap_err()
    else {
        unreachable!()
    };
    assert_eq!(
        exception
            .as_downcast_ref::<Exception>()
            .expect("exception")
            .value()
            .as_symbol()
            .expect("symbol"),
        &Symbol::from("undefined"),
    );
}

#[test]
fn macros() {
    let code = Compiler::default()
        .with_macro("$test", |mut tokens: Vec<Ranged<Token>>| {
            assert_eq!(tokens[0].0, Token::Open(Paired::Paren));
            tokens.insert(2, Ranged::new(SourceRange::default(), Token::Char('+')));
            assert_eq!(tokens[4].0, Token::Close(Paired::Paren));
            dbg!(tokens)
        })
        .with(&SourceCode::anonymous(
            r"
                let hello = 5;
                let world = 3;
                $test(hello world)
            ",
        ))
        .build()
        .unwrap();
    let mut vm = Vm::default();
    let result = vm.execute(&code).unwrap().as_u64();
    assert_eq!(result, Some(8));
}
