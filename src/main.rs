use muse::compiler::Compiler;
use muse::syntax::{parse, Ranged};
use muse::vm::Vm;
use rustyline::completion::Completer;
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Editor, Helper};

fn main() {
    let mut editor: Editor<Muse, DefaultHistory> = Editor::new().unwrap();
    editor.set_helper(Some(Muse));
    let mut vm = Vm::default();
    let mut compiler = Compiler::default();
    loop {
        match editor.readline("> ") {
            Ok(line) => {
                compiler.push(&line);
                match compiler.build() {
                    Ok(code) => match vm.execute(&code) {
                        Ok(value) => {
                            let displayed = value
                                .map_str(&mut vm, |_vm, value| {
                                    println!("{value}");
                                })
                                .is_ok();
                            if !displayed {
                                println!("{value:?}");
                            }
                        }
                        Err(err) => {
                            eprintln!("Execution error: {err:?}");
                        }
                    },
                    Err(err) => {
                        eprintln!("Compilation error: {err:?}");
                    }
                }
            }
            Err(ReadlineError::Eof) => break,
            Err(ReadlineError::Interrupted) => {}
            Err(other) => unreachable!("unexpected error: {other}"),
        }
    }
}

struct Muse;

impl Helper for Muse {}

impl Validator for Muse {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        match parse(ctx.input()) {
            Ok(_) => Ok(ValidationResult::Valid(None)),
            Err(Ranged(muse::syntax::Error::UnexpectedEof, _)) => Ok(ValidationResult::Incomplete),
            Err(err) => Ok(ValidationResult::Invalid(Some(format!(
                "@{}:{}: {:?}",
                err.1.start,
                err.1.end(),
                err.0
            )))),
        }
    }

    fn validate_while_typing(&self) -> bool {
        true
    }
}

impl Highlighter for Muse {}

impl Hinter for Muse {
    type Hint = String;
}

impl Completer for Muse {
    type Candidate = String;
}
