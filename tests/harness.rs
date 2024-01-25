use std::collections::BTreeMap;
use std::path::Path;

use muse::compiler::{Compiler, Error};
use muse::symbol::Symbol;
use muse::syntax::Ranged;
use muse::value::Value;
use muse::vm::{Fault, Vm};
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub enum TestOutput {
    Nil,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Symbol(Symbol),
    Error(Vec<Ranged<Error>>),
    Fault(Fault),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Case {
    pub src: String,
    pub output: TestOutput,
}

impl Case {
    fn run(&self) -> TestOutput {
        match Compiler::compile(&self.src) {
            Ok(code) => match Vm::default().execute(&code) {
                Ok(Value::Nil) => TestOutput::Nil,
                Ok(Value::Bool(bool)) => TestOutput::Bool(bool),
                Ok(Value::Int(v)) => TestOutput::Int(v),
                Ok(Value::UInt(v)) => TestOutput::UInt(v),
                Ok(Value::Float(v)) => TestOutput::Float(v),
                Ok(Value::Symbol(v)) => TestOutput::Symbol(v),
                Ok(Value::Dynamic(v)) => {
                    unreachable!("test returned dynamic {v:?}, but the harness doesn't support it")
                }
                Err(fault) => TestOutput::Fault(fault),
            },
            Err(err) => TestOutput::Error(err),
        }
    }
}

fn main() {
    for entry in std::fs::read_dir("tests/cases").unwrap() {
        let entry = entry.unwrap().path();
        if entry.extension().map_or(false, |ext| ext == "rsn") {
            run_test_cases(&entry);
        }
    }
}

fn run_test_cases(path: &Path) {
    let contents = std::fs::read_to_string(path).unwrap();

    let cases: BTreeMap<String, Case> = match rsn::parser::Config::default()
        .allow_implicit_map(true)
        .deserialize(&contents)
    {
        Ok(cases) => cases,
        Err(err) => unreachable!("error parsing {}: {err}", path.display()),
    };
    for (name, case) in cases {
        println!("Running {name}");
        let output = case.run();
        assert_eq!(
            output,
            case.output,
            "in {path} @ {name}: expected {expected:?}, got {output:?}",
            path = path.display(),
            expected = case.output
        );
    }
}

// let bitcode = BitcodeBlock::from(&code);
// println!("{}", rsn::to_string_pretty(&bitcode));
