use std::fmt::{Debug, Display};
use std::ops::{Add, Deref};
use std::sync::OnceLock;

use ahash::RandomState;
use interner::global::{GlobalString, StaticPooledString, StringPool};
use serde::de::Visitor;
use serde::{Deserialize, Serialize};

static SYMBOLS: StringPool<RandomState> = StringPool::with_hasher_init(RandomState::new);

#[derive(Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct Symbol(GlobalString<RandomState>);

macro_rules! static_symbols {
    ($($name:ident => $string:literal),+ $(,)?) => {
        impl Symbol {
            $(pub fn $name() -> Self {
                static S: StaticPooledString<RandomState> = SYMBOLS.get_static($string);
                Self(S.clone())
            })+
        }
    };
}

static_symbols!(
    empty => "",
    not_symbol => "not",
    let_symbol => "let",
    var_symbol => "var",
    and_symbol => "and",
    or_symbol => "or",
    xor_symbol => "xor",
    true_symbol => "true",
    false_symbol => "false",
    if_symbol => "if",
    then_symbol => "then",
    else_symbol => "else",
    defun_symbol => "defun",
);

impl From<String> for Symbol {
    fn from(value: String) -> Self {
        Symbol(SYMBOLS.get(value))
    }
}
impl From<&'_ Symbol> for Symbol {
    fn from(value: &'_ Symbol) -> Self {
        value.clone()
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

impl From<bool> for Symbol {
    fn from(bool: bool) -> Self {
        if bool {
            Symbol::true_symbol()
        } else {
            Symbol::false_symbol()
        }
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

impl Serialize for Symbol {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self)
    }
}

impl<'de> Deserialize<'de> for Symbol {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(SymbolVisitor)
    }
}
struct SymbolVisitor;

impl<'de> Visitor<'de> for SymbolVisitor {
    type Value = Symbol;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "an Symbol")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(Symbol::from(v))
    }

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(Symbol::from(v))
    }
}

pub struct StaticSymbol(OnceLock<Symbol>, &'static str);

impl StaticSymbol {
    #[must_use]
    pub const fn new(symbol: &'static str) -> Self {
        Self(OnceLock::new(), symbol)
    }
}

impl Debug for StaticSymbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Display for StaticSymbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.1, f)
    }
}

impl Deref for StaticSymbol {
    type Target = Symbol;

    fn deref(&self) -> &Self::Target {
        self.0.get_or_init(|| Symbol::from(self.1))
    }
}
