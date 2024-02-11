use std::fmt::{Debug, Display};
use std::ops::{Add, Deref};
use std::sync::OnceLock;
use std::{array, iter};

use ahash::RandomState;
use interner::global::{GlobalString, StringPool};
use serde::de::Visitor;
use serde::{Deserialize, Serialize};

static SYMBOLS: StringPool<RandomState> = StringPool::with_hasher_init(RandomState::new);

#[derive(Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct Symbol(GlobalString<RandomState>);

macro_rules! static_symbols {
    ($($name:ident => $string:literal),+ $(,)?) => {
        impl Symbol {
            $(pub fn $name() -> &'static Self {
                static S: OnceLock<Symbol> = OnceLock::new();
                S.get_or_init(|| Symbol::from($string))
            })+
        }
    };
}

static_symbols!(
    empty => "",
    and_symbol => "and",
    break_symbol => "break",
    continue_symbol => "continue",
    else_symbol => "else",
    false_symbol => "false",
    for_symbol => "for",
    fn_symbol => "fn",
    get_symbol => "get",
    if_symbol => "if",
    in_symbol => "in",
    iterate_symbol => "iterate",
    len_symbol => "len",
    let_symbol => "let",
    loop_symbol => "loop",
    match_symbol => "match",
    mod_symbol => "mod",
    next_symbol => "next",
    not_symbol => "not",
    none_symbol => "none",
    nth_symbol => "nth",
    or_symbol => "or",
    pub_symbol => "pub",
    return_symbol => "return",
    set_symbol => "set",
    super_symbol => "super",
    then_symbol => "then",
    true_symbol => "true",
    var_symbol => "var",
    while_symbol => "while",
    xor_symbol => "xor",
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

pub trait IntoOptionSymbol {
    fn into_symbol(self) -> Option<Symbol>;
}

impl<T> IntoOptionSymbol for T
where
    T: Into<Symbol>,
{
    fn into_symbol(self) -> Option<Symbol> {
        Some(self.into())
    }
}

impl IntoOptionSymbol for Option<Symbol> {
    fn into_symbol(self) -> Option<Symbol> {
        self
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
            Symbol::true_symbol().clone()
        } else {
            Symbol::false_symbol().clone()
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

pub trait SymbolList {
    type Iterator: Iterator<Item = Symbol>;
    fn into_symbols(self) -> Self::Iterator;
}

impl<const N: usize> SymbolList for [Symbol; N] {
    type Iterator = array::IntoIter<Symbol, N>;

    fn into_symbols(self) -> Self::Iterator {
        self.into_iter()
    }
}

impl<'a, const N: usize> SymbolList for [&'a Symbol; N] {
    type Iterator = iter::Cloned<array::IntoIter<&'a Symbol, N>>;

    fn into_symbols(self) -> Self::Iterator {
        self.into_iter().cloned()
    }
}

impl SymbolList for Symbol {
    type Iterator = iter::Once<Symbol>;

    fn into_symbols(self) -> Self::Iterator {
        iter::once(self)
    }
}

impl SymbolList for &'_ Symbol {
    type Iterator = iter::Once<Symbol>;

    fn into_symbols(self) -> Self::Iterator {
        self.clone().into_symbols()
    }
}

impl SymbolList for &'_ str {
    type Iterator = iter::Once<Symbol>;

    fn into_symbols(self) -> Self::Iterator {
        Symbol::from(self).into_symbols()
    }
}
