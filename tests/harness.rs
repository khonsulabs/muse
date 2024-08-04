//! A test harness that executes `.rsn` tests in the `cases` directory.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::exit;

use muse::compiler::syntax::token::RegexLiteral;
use muse::compiler::syntax::{Ranged, SourceRange};
use muse::compiler::{Compiler, Error};
use muse::refuse::CollectionGuard;
use muse::runtime::exception::Exception;
use muse::runtime::list::List;
use muse::runtime::map::Map;
use muse::runtime::regex::{MuseMatch, MuseRegex};
use muse::runtime::string::MuseString;
use muse::runtime::symbol::Symbol;
use muse::runtime::value::Value;
use muse::vm::{Code, ExecutionError, StackFrame, Vm};
use serde::de::Visitor;
use serde::Deserialize;

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    // let filter = String::from("mod_multi");
    for entry in std::fs::read_dir("tests/cases").unwrap() {
        let entry = entry.unwrap().path();
        if entry.extension().map_or(false, |ext| ext == "rsn") {
            run_test_cases(&entry, filter.trim());
        }
    }
}

#[derive(Debug, PartialEq, Deserialize)]
enum TestOutput {
    Nil,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Symbol(Symbol),
    String(String),
    Regex(RegexLiteral),
    RegexMatch {
        content: String,
        start: usize,
    },
    Map(TestMap),
    List(Vec<TestOutput>),
    Error(Vec<Ranged<Error>>),
    Exception {
        value: Box<TestOutput>,
        backtrace: Vec<SourceRange>,
    },
    Fault(VmFault),
}

#[derive(Debug, PartialEq, Deserialize)]
enum VmFault {
    UnknownSymbol(Symbol),
    IncorrectNumberOfArguments,
    OperationOnNil,
    NotAModule,
    NotAFunction,
    StackOverflow,
    StackUnderflow,
    InvalidIndex,
    UnsupportedOperation,
    OutOfMemory,
    OutOfBounds,
    NotMutable,
    DivideByZero,
    InvalidInstructionAddress,
    ExpectedSymbol,
    ExpectedInteger,
    ExpectedString,
    InvalidArity,
    InvalidLabel,
    InvalidOpcode,
    NoBudget,
    Timeout,
    Waiting,
    FrameChanged,
    PatternMismatch,
    Exception(Box<TestOutput>),
}

impl From<ExecutionError> for VmFault {
    fn from(value: ExecutionError) -> Self {
        match value {
            ExecutionError::NoBudget => VmFault::NoBudget,
            ExecutionError::Waiting => VmFault::Waiting,
            ExecutionError::Timeout => VmFault::Timeout,
            ExecutionError::Exception(value) => Self::Exception(Box::new(TestOutput::from(value))),
        }
    }
}

#[derive(Debug, Deserialize)]
struct Case {
    pub src: String,
    pub output: TestOutput,
    #[serde(default)]
    pub ignore: bool,
}

impl Case {
    fn run(&self) -> (Option<Code>, TestOutput) {
        let mut guard = CollectionGuard::acquire();
        match Compiler::compile(&self.src, &guard) {
            Ok(code) => match Vm::new(&guard).execute(&code, &mut guard) {
                Ok(value) => (Some(code), TestOutput::from(value)),
                Err(fault) => (Some(code), TestOutput::Fault(VmFault::from(fault))),
            },
            Err(err) => (None, TestOutput::Error(err)),
        }
    }
}

impl From<Value> for TestOutput {
    fn from(value: Value) -> Self {
        let guard = CollectionGuard::acquire();
        match value {
            Value::Nil => TestOutput::Nil,
            Value::Bool(bool) => TestOutput::Bool(bool),
            Value::Int(v) => TestOutput::Int(v),
            Value::UInt(v) => TestOutput::UInt(v),
            Value::Float(v) => TestOutput::Float(v),
            Value::Symbol(v) => TestOutput::Symbol(v.upgrade(&guard).expect("missing symbol")),
            Value::Dynamic(v) => {
                if let Some(str) = v.downcast_ref::<MuseString>(&guard) {
                    TestOutput::String(str.to_string())
                } else if let Some(regex) = v.downcast_ref::<MuseRegex>(&guard) {
                    TestOutput::Regex(regex.literal().clone())
                } else if let Some(map) = v.downcast_ref::<Map>(&guard) {
                    TestOutput::Map(TestMap(
                        map.to_vec()
                            .into_iter()
                            .map(|field| {
                                let (key, value) = field.into_parts();
                                (TestOutput::from(key), TestOutput::from(value))
                            })
                            .collect(),
                    ))
                } else if let Some(list) = v.downcast_ref::<List>(&guard) {
                    TestOutput::List(list.to_vec().into_iter().map(TestOutput::from).collect())
                } else if let Some(m) = v.downcast_ref::<MuseMatch>(&guard) {
                    TestOutput::RegexMatch {
                        content: m
                            .content
                            .downcast_ref::<MuseString>(&guard)
                            .expect("capture is string")
                            .to_string(),
                        start: m.start,
                    }
                } else if let Some(m) = v.downcast_ref::<Exception>(&guard) {
                    TestOutput::Exception {
                        value: Box::new(Self::from(m.value())),
                        backtrace: m
                            .backtrace()
                            .iter()
                            .filter_map(StackFrame::source_range)
                            .collect(),
                    }
                } else {
                    unreachable!("test returned dynamic {v:?}, but the harness doesn't support it")
                }
            }
        }
    }
}

fn run_test_cases(path: &Path, filter: &str) {
    let contents = std::fs::read_to_string(path).unwrap().replace("\r\n", "\n");

    let cases: BTreeMap<String, Case> = match rsn::parser::Config::default()
        .allow_implicit_map(true)
        .deserialize(&contents)
    {
        Ok(cases) => cases,
        Err(err) => unreachable!("error parsing {}: {err}", path.display()),
    };
    for (name, case) in cases {
        if (!filter.is_empty() && filter != name) || case.ignore {
            println!("Skipping {name}");
            continue;
        }
        println!("Running {name}");
        let (code, output) = case.run();
        if output != case.output {
            eprintln!("== Test Case Error ==");
            eprintln!("{code:#?}");
            eprintln!(
                "in {path} @ {name}: expected {expected:?}, got {output:?}",
                path = path.display(),
                expected = case.output,
            );
            exit(-1);
        }
    }
}

#[derive(Debug)]
struct TestMap(Vec<(TestOutput, TestOutput)>);

impl PartialEq for TestMap {
    fn eq(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }

        'outer: for (keya, valuea) in &self.0 {
            for (keyb, valueb) in &self.0 {
                if keya == keyb {
                    if valuea == valueb {
                        continue 'outer;
                    }
                    return false;
                }
            }
        }

        true
    }
}

impl<'de> Deserialize<'de> for TestMap {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(MapVisitor)
    }
}

struct MapVisitor;

impl<'de> Visitor<'de> for MapVisitor {
    type Value = TestMap;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "a map")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut out = TestMap(Vec::with_capacity(map.size_hint().unwrap_or_default()));
        while let Some((key, value)) = map.next_entry()? {
            out.0.push((key, value));
        }

        Ok(out)
    }
}
