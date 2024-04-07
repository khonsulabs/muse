macro_rules! impl_from {
    ($on:ty, $from:ty, $variant:ident) => {
        impl From<$from> for $on {
            fn from(value: $from) -> Self {
                Self::$variant(value.into())
            }
        }
    };
}

pub mod compiler;
pub mod map;
pub mod string;
pub mod symbol;
pub mod syntax;
pub mod value;
pub mod vm;

pub mod exception;
pub mod list;
pub mod regex;
#[cfg(test)]
mod tests;

pub use refuse;
use syntax::Ranged;

pub trait ErrorKind: std::error::Error {
    fn kind(&self) -> &'static str;
}

#[derive(Debug, PartialEq)]
pub enum Error {
    Compilation(Vec<Ranged<compiler::Error>>),
    Execution(vm::ExecutionError),
}

impl From<Vec<Ranged<compiler::Error>>> for Error {
    fn from(value: Vec<Ranged<compiler::Error>>) -> Self {
        Self::Compilation(value)
    }
}

impl From<vm::ExecutionError> for Error {
    fn from(value: vm::ExecutionError) -> Self {
        Self::Execution(value)
    }
}
