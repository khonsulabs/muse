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

pub trait Error: std::error::Error {
    fn kind(&self) -> &'static str;
}
