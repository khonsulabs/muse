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
pub mod symbol;
pub mod syntax;
pub mod value;
pub mod vm;
