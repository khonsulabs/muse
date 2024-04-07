use std::collections::VecDeque;
use std::path::Path;
use std::process::exit;

use muse::compiler::Compiler;
use muse::exception::Exception;
use muse::symbol::Symbol;
use muse::syntax::token::{Paired, Token};
use muse::syntax::Ranged;
use muse::value::{CustomType, RustFunction, RustType, Value};
use muse::vm::{ExecutionError, Fault, Register, Vm, VmContext};
use refuse::{CollectionGuard, Trace};

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    // let filter = String::from("raw_format");
    for entry in std::fs::read_dir("tests/cases").unwrap() {
        let entry = entry.unwrap().path();
        if entry.extension().map_or(false, |ext| ext == "muse") {
            run_test_cases(&entry, filter.trim());
        }
    }
}

#[allow(clippy::too_many_lines)]
fn run_test_cases(path: &Path, filter: &str) {
    let path = path.to_str().expect("invalid test path");
    let contents = std::fs::read_to_string(path).unwrap();
    let eval = Symbol::from("eval");
    let mut guard = CollectionGuard::acquire();

    let code = Compiler::default()
        .with_macro("$case", {
            let eval = eval.clone();
            let filter = filter.to_string();
            move |mut tokens: VecDeque<Ranged<Token>>| {
                let open_paren = tokens.pop_front().expect("missing (");
                assert_eq!(open_paren.0, Token::Open(Paired::Paren));
                let name = tokens.pop_front().expect("missing case name");
                let Token::Identifier(name_ident) = &name.0 else {
                    unreachable!("expected test case name to be an identifier")
                };
                if !filter.is_empty() && !name_ident.starts_with(&filter) {
                    return VecDeque::new();
                }
                let test_range = name.range();
                let start =
                    i64::try_from(open_paren.range().start).expect("test file way too large");
                let colon = tokens.pop_front().expect("missing colon");
                assert_eq!(colon.0, Token::Char(':'));
                let source = tokens.front().expect("missing source").clone();
                let Token::String(source_contents) = &source.0 else {
                    unreachable!("expected source to be a string")
                };

                tokens.push_front(Ranged::new(test_range, Token::Char(',')));
                tokens.push_front(Ranged::new(
                    source.range(),
                    Token::String(source_contents.clone()),
                ));
                tokens.push_front(Ranged::new(test_range, Token::Char(',')));
                tokens.push_front(Ranged::new(test_range, Token::Int(start)));
                tokens.push_front(Ranged::new(test_range, Token::Char(',')));
                tokens.push_front(Ranged::new(test_range, Token::Symbol(name_ident.clone())));
                tokens.push_front(open_paren);
                tokens.push_front(Ranged::new(test_range, Token::Identifier(eval.clone())));
                tokens.push_front(colon.map(|_| Token::Char('=')));
                tokens.push_front(name);
                tokens.push_front(Ranged::new(
                    test_range,
                    Token::Identifier(Symbol::let_symbol().clone()),
                ));
                tokens
            }
        })
        .with(&contents)
        .build(&guard)
        .unwrap();

    let vm = Vm::new(&guard);
    let path = path.to_string();

    let mut context = VmContext::new(&vm, &mut guard);
    context
        .declare(
            eval,
            Value::dynamic(
                RustFunction::new(move |vm, arity| {
                    let 4 = arity.0 else {
                        return Err(Fault::InvalidArity);
                    };
                    let name = vm[Register(0)].take();
                    let name_string = name
                        .to_string(vm)
                        .expect("invalid name")
                        .try_upgrade(vm.guard())?;
                    println!("running case {name_string}");
                    let offset = vm[Register(1)].take();
                    let source = vm[Register(2)].take();
                    let code = vm[Register(3)]
                        .take()
                        .to_string(vm)?
                        .try_upgrade(vm.guard())?;

                    vm.while_unlocked(|guard| {
                        let code = Compiler::compile(&*code, guard).unwrap();
                        let sandbox = Vm::new(guard);
                        match sandbox.execute(&code, guard) {
                            Ok(result) => Ok(result),
                            Err(err) => Err(Fault::Exception(Value::dynamic(
                                TestError {
                                    err,
                                    name,
                                    offset,
                                    source,
                                },
                                guard,
                            ))),
                        }
                    })
                }),
                context.guard(),
            ),
        )
        .unwrap();

    match context.execute(&code) {
        Ok(_) => {}
        Err(ExecutionError::Exception(exc)) => {
            let Some(exc) = exc.as_rooted::<TestError>(context.guard()) else {
                unreachable!("expected test error: {exc:?}")
            };
            let Some(offset) = exc.offset.as_usize() else {
                unreachable!("invalid offset")
            };
            let mut line_offset = 0;
            let line_number = contents
                .lines()
                .enumerate()
                .find_map(|(index, line)| {
                    let line_start = line_offset;
                    let line_end = line_offset + line.len();
                    line_offset = line_end;
                    (offset >= line_start && offset < line_end).then_some(index)
                })
                .expect("invalid source offset")
                + 1;
            let name = exc
                .name
                .as_symbol(context.guard())
                .expect("expected name to be a symbol");
            eprintln!("error in {name} @ {path}:{line_number}");

            let ExecutionError::Exception(inner_exception) = &exc.err else {
                unreachable!("expected inner error to be exception")
            };

            if let Some(inner_exception) = inner_exception.as_rooted::<Exception>(context.guard()) {
                eprintln!("exception: {exc:?}", exc = inner_exception.value());
                let Some(source) = exc
                    .source
                    .to_string(&mut context)
                    .ok()
                    .and_then(|source| source.upgrade(context.guard()))
                else {
                    unreachable!("source was not a string")
                };
                let mut line_offset = 0;
                let line_ends = source
                    .lines()
                    .map(|line| {
                        line_offset += line.len();
                        line_offset
                    })
                    .collect::<Vec<_>>();
                for frame in inner_exception.backtrace() {
                    let Some(range) = frame.source_range() else {
                        continue;
                    };
                    let frame_line = line_number
                        + line_ends
                            .iter()
                            .copied()
                            .enumerate()
                            .find_map(|(index, offset)| (range.start < offset).then_some(index))
                            .unwrap_or(line_ends.len());
                    eprintln!("in {path}:{frame_line}");
                }
            } else {
                eprintln!("exception: {inner_exception:?}");
            }

            exit(-1);
        }
        Err(other) => unreachable!("expected exception: {other:?}"),
    }
}

#[derive(Debug, Trace)]
struct TestError {
    err: ExecutionError,
    name: Value,
    offset: Value,
    source: Value,
}

impl CustomType for TestError {
    fn muse_type(&self) -> &muse::value::TypeRef {
        static TYPE: RustType<TestError> = RustType::new("TestError", |t| t);
        &TYPE
    }
}

// fn execute_test_cases(path: &str, filter: &str, contents: &str) -> Result<(), TestError> {}

// struct TestError {
//     path: String,
//     range: Option<
// }
