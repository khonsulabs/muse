//! A Safe, familiar, embeddable programming language.

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
pub mod vm;

pub mod runtime;

#[cfg(test)]
mod tests;

use compiler::syntax::Ranged;
pub use refuse;

/// Summarizes an error's kind.
pub trait ErrorKind {
    /// Returns the summary of the error being raised.
    fn kind(&self) -> &'static str;
}

/// One or more errors raised during compilation or execution.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A list of compilation errors.
    Compilation(Vec<Ranged<compiler::Error>>),
    /// An execution error.
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
