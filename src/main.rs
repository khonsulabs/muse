use std::collections::HashMap;
use std::num::NonZeroUsize;

use ariadne::{Cache, Label, Span};
use muse::compiler::Compiler;
use muse::syntax::{parse, Ranged, SourceId, SourceRange, Sources};
use muse::vm::Vm;
use muse::Error;
use rustyline::completion::Completer;
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Editor, ExternalPrinter, Helper};

fn main() {
    let mut editor: Editor<Muse, DefaultHistory> = Editor::new().unwrap();
    editor.set_helper(Some(Muse));
    let mut vm = Vm::default();
    let mut sources = Sources::default();
    let mut compiler = Compiler::default();
    loop {
        match editor.readline("> ") {
            Ok(line) => {
                let source = sources.push(line);
                compiler.push(&source);
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
                        let errors = print_errors(err, &sources);
                        let mut printer = editor
                            .create_external_printer()
                            .expect("can't create printer");
                        printer.print(errors).unwrap();
                    }
                }
            }
            Err(ReadlineError::Eof) => break,
            Err(ReadlineError::Interrupted) => {}
            Err(other) => unreachable!("unexpected error: {other}"),
        }
    }
}

fn print_errors(errs: Vec<Ranged<muse::compiler::Error>>, sources: &Sources) -> String {
    let mut text = Vec::new();
    for (index, err) in errs.into_iter().enumerate() {
        if index > 0 {
            text.push(b'\n');
        }
        ariadne::Report::<MuseSpan>::build(
            ariadne::ReportKind::Error,
            err.1.source_id,
            err.1.start,
        )
        .with_message(err.kind())
        .with_label(Label::new(MuseSpan(err.1)).with_message(err.0.to_string()))
        .finish()
        .write_for_stdout(SourceCache(sources, HashMap::default()), &mut text)
        .expect("error building report");
    }
    String::from_utf8(text).expect("invalid utf-8 in error report")
}

struct MuseSpan(SourceRange);

impl Span for MuseSpan {
    type SourceId = SourceId;

    fn source(&self) -> &Self::SourceId {
        &self.0.source_id
    }

    fn start(&self) -> usize {
        self.0.start
    }

    fn end(&self) -> usize {
        self.0.end()
    }
}

struct SourceCache<'a>(&'a Sources, HashMap<SourceId, ariadne::Source<String>>);

impl Cache<SourceId> for SourceCache<'_> {
    type Storage = String;

    fn fetch(
        &mut self,
        id: &SourceId,
    ) -> Result<&ariadne::Source<Self::Storage>, Box<dyn std::fmt::Debug + '_>> {
        Ok(self.1.entry(*id).or_insert_with(|| {
            ariadne::Source::from(self.0.get(*id).expect("missing source").clone())
        }))
    }

    fn display<'a>(&self, id: &'a SourceId) -> Option<Box<dyn std::fmt::Display + 'a>> {
        Some(Box::new(id.get().map_or(0, NonZeroUsize::get)))
    }
}

struct Muse;

impl Helper for Muse {}

impl Validator for Muse {
    fn validate(&self, ctx: &mut ValidationContext) -> rustyline::Result<ValidationResult> {
        let mut sources = Sources::default();
        let source = sources.push(ctx.input().to_string());
        match parse(&source) {
            Ok(_) => Ok(ValidationResult::Valid(None)),
            Err(Ranged(muse::syntax::Error::UnexpectedEof, _)) => Ok(ValidationResult::Incomplete),
            Err(err) => {
                let mut errors = print_errors(vec![err.into()], &sources);
                errors.insert(0, '\n');
                Ok(ValidationResult::Invalid(Some(errors)))
            }
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
