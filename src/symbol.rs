use std::fmt::{Debug, Display};
use std::ops::{Add, Deref};

use interner::global::{GlobalString, StaticPooledString, StringPool};

static SYMBOLS: StringPool = StringPool::new();

#[derive(Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct Symbol(GlobalString);

static EMPTY: StaticPooledString = SYMBOLS.get_static("");

impl Symbol {
    pub fn empty() -> Self {
        Self(EMPTY.clone())
    }
}

impl From<String> for Symbol {
    fn from(value: String) -> Self {
        Symbol(SYMBOLS.get(value))
    }
}

impl From<&'_ String> for Symbol {
    fn from(value: &'_ String) -> Self {
        Symbol(SYMBOLS.get(value))
    }
}

impl From<&'_ str> for Symbol {
    fn from(value: &'_ str) -> Self {
        Symbol(SYMBOLS.get(value))
    }
}

impl PartialEq<&'_ str> for Symbol {
    fn eq(&self, other: &&'_ str) -> bool {
        self.0 == *other
    }
}

impl PartialEq<str> for Symbol {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl Deref for Symbol {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}
impl<'a, 'b> Add<&'a Symbol> for &'b Symbol {
    type Output = Symbol;

    fn add(self, rhs: &'a Symbol) -> Self::Output {
        let mut out = String::with_capacity(self.len() + rhs.len());
        out.push_str(self);
        out.push_str(rhs);
        Symbol::from(out)
    }
}
