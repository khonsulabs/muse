use std::collections::VecDeque;

use refuse::CollectionGuard;

use crate::compiler::syntax::token::{Paired, Token};
use crate::compiler::syntax::{Expression, Ranged, SourceRange, TokenizeInto};
use crate::compiler::Compiler;
use crate::runtime::exception::Exception;
use crate::runtime::symbol::Symbol;
use crate::runtime::value::Value;
use crate::vm::bitcode::BitcodeBlock;
use crate::vm::{ExecutionError, Register, Vm};

#[test]
fn budgeting() {
    const COUNT_TO: i64 = 42;

    let mut guard = CollectionGuard::acquire();
    let mut code = BitcodeBlock::default();
    for value in 0..=COUNT_TO {
        code.copy(value, Register(0));
    }
    let code = code.to_code(&guard);
    let vm = Vm::new(&guard);
    // Turn on budgeting, but don't give any budget.
    vm.set_steps_per_charge(1);
    vm.increase_budget(0);
    assert_eq!(
        vm.execute(&code, &mut guard).unwrap_err(),
        ExecutionError::NoBudget
    );
    for value in 0..=COUNT_TO {
        // Step through by allowing one op at a time.
        vm.increase_budget(1);
        assert_eq!(vm.resume(&mut guard).unwrap_err(), ExecutionError::NoBudget);
        assert_eq!(vm.register(Register(0)).as_i64(), Some(value));
    }
    vm.increase_budget(1);
    assert_eq!(vm.resume(&mut guard).unwrap().as_i64(), Some(COUNT_TO));
}

#[test]
fn module_budgeting() {
    const MAX_OPS: usize = 24;
    let mut guard = CollectionGuard::acquire();
    let code = Compiler::compile(
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
        &guard,
    )
    .unwrap();
    let vm = Vm::new(&guard);
    // Turn on budgeting, but don't give any budget.
    vm.set_steps_per_charge(1);
    vm.increase_budget(0);
    assert_eq!(
        vm.execute(&code, &mut guard).unwrap_err(),
        ExecutionError::NoBudget
    );
    let mut ops = 0;
    for _ in 0..MAX_OPS {
        ops += 1;
        // Step through by allowing one op at a time.
        vm.increase_budget(1);
        match vm.resume(&mut guard) {
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
    let mut guard = CollectionGuard::acquire();
    let code = Compiler::compile(
        r"
            pub fn test(n) => n * 2;
            fn private(n) => n * 2;
        ",
        &guard,
    )
    .unwrap();
    let vm = Vm::new(&guard);
    vm.execute(&code, &mut guard).unwrap();

    let Value::Int(result) = vm.invoke("test", [Value::Int(3)], &mut guard).unwrap() else {
        unreachable!()
    };
    assert_eq!(result, 6);
    let ExecutionError::Exception(exception) = vm
        .invoke("private", [Value::Int(3)], &mut guard)
        .unwrap_err()
    else {
        unreachable!()
    };
    assert_eq!(
        exception
            .as_downcast_ref::<Exception>(&guard)
            .expect("exception")
            .value()
            .as_symbol(&guard)
            .expect("symbol"),
        Symbol::from("undefined"),
    );
}

#[test]
fn macros() {
    let mut guard = CollectionGuard::acquire();
    let code = Compiler::default()
        .with_macro("$test", |mut tokens: VecDeque<Ranged<Token>>| {
            assert_eq!(tokens[0].0, Token::Open(Paired::Paren));
            tokens.insert(2, Ranged::new(SourceRange::default(), Token::Char('+')));
            assert_eq!(tokens[4].0, Token::Close(Paired::Paren));
            dbg!(tokens)
        })
        .with(
            r"
                let hello = 5;
                let world = 3;
                $test(hello world)
            ",
        )
        .build(&guard)
        .unwrap();
    let vm = Vm::new(&guard);
    let result = vm.execute(&code, &mut guard).unwrap().as_u64();
    assert_eq!(result, Some(8));
}

#[test]
fn recursive_macros() {
    let mut guard = CollectionGuard::acquire();
    let code = Compiler::default()
        .with_macro("$inner", |mut tokens: VecDeque<Ranged<Token>>| {
            assert_eq!(tokens[0].0, Token::Open(Paired::Paren));
            tokens.insert(2, Ranged::new(SourceRange::default(), Token::Char('+')));
            assert_eq!(tokens[4].0, Token::Close(Paired::Paren));
            dbg!(tokens)
        })
        .with_macro("$test", |mut tokens: VecDeque<Ranged<Token>>| {
            tokens.insert(
                0,
                Ranged::new(SourceRange::default(), Token::Sigil(Symbol::from("$inner"))),
            );
            tokens
        })
        .with(
            r"
                let hello = 5;
                let world = 3;
                $test(hello world)
            ",
        )
        .build(&guard)
        .unwrap();
    let vm = Vm::new(&guard);
    let result = vm.execute(&code, &mut guard).unwrap().as_u64();
    assert_eq!(result, Some(8));
}

#[test]
fn infix_macros() {
    let mut guard = CollectionGuard::acquire();
    let code = Compiler::default()
        .with_infix_macro(
            "$test",
            |expr: &Ranged<Expression>, mut tokens: VecDeque<Ranged<Token>>| {
                let mut expr = expr.to_tokens();

                assert_eq!(tokens[0].0, Token::Open(Paired::Paren));
                assert_eq!(tokens[1].0, Token::Close(Paired::Paren));

                let close = tokens.pop_back().unwrap();

                tokens.append(&mut expr);

                tokens.push_back(Ranged::new(SourceRange::default(), Token::Char('+')));
                tokens.push_back(Ranged::new(SourceRange::default(), Token::Int(1)));
                tokens.push_back(close);
                dbg!(tokens)
            },
        )
        .with(
            r"
                (5)$test()
            ",
        )
        .build(&guard)
        .unwrap();
    let vm = Vm::new(&guard);
    let result = vm.execute(&code, &mut guard).unwrap().as_u64();
    assert_eq!(result, Some(6));
}
