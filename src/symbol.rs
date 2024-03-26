use std::fmt::{Debug, Display};
use std::ops::{Add, Deref};
use std::sync::OnceLock;
use std::{array, iter};

use refuse::{CollectionGuard, Trace};
use refuse_pool::{RefString, RootString};
use serde::de::Visitor;
use serde::{Deserialize, Serialize};

use crate::value::ValueFreed;

#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash, Trace)]
pub struct SymbolRef(RefString);
impl SymbolRef {
    #[must_use]
    pub fn load<'guard>(&self, guard: &'guard CollectionGuard<'_>) -> Option<&'guard str> {
        self.0.load(guard)
    }

    pub fn try_load<'guard>(
        &self,
        guard: &'guard CollectionGuard<'_>,
    ) -> Result<&'guard str, ValueFreed> {
        self.load(guard).ok_or(ValueFreed)
    }

    pub fn upgrade(&self, guard: &CollectionGuard<'_>) -> Option<Symbol> {
        self.0.as_root(guard).map(Symbol)
    }

    pub fn try_upgrade(&self, guard: &CollectionGuard<'_>) -> Result<Symbol, ValueFreed> {
        self.upgrade(guard).ok_or(ValueFreed)
    }
}

impl kempt::Sort<SymbolRef> for Symbol {
    fn compare(&self, other: &SymbolRef) -> std::cmp::Ordering {
        self.0.as_any().cmp(&other.0.as_any())
    }
}

#[derive(Clone, Trace)]
pub struct Symbol(RootString);

impl Symbol {
    #[must_use]
    pub const fn downgrade(&self) -> SymbolRef {
        SymbolRef(self.0.downgrade())
    }
}

impl Eq for Symbol {}

impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        self.0.downgrade() == other.0.downgrade()
    }
}

impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.downgrade().cmp(&other.0.downgrade())
    }
}

impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::hash::Hash for Symbol {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.downgrade().hash(state);
    }
}

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
    catch_symbol => "catch",
    continue_symbol => "continue",
    else_symbol => "else",
    false_symbol => "false",
    for_symbol => "for",
    fn_symbol => "fn",
    get_symbol => "get",
    if_symbol => "if",
    in_symbol => "in",
    it_symbol => "it",
    iterate_symbol => "iterate",
    len_symbol => "len",
    let_symbol => "let",
    loop_symbol => "loop",
    match_symbol => "match",
    captures_symbol => "captures",
    mod_symbol => "mod",
    next_symbol => "next",
    nil_symbol => "nil",
    not_symbol => "not",
    nth_symbol => "nth",
    or_symbol => "or",
    pub_symbol => "pub",
    return_symbol => "return",
    set_symbol => "set",
    sigil_symbol => "$",
    super_symbol => "super",
    then_symbol => "then",
    throw_symbol => "throw",
    true_symbol => "true",
    try_symbol => "try",
    var_symbol => "var",
    while_symbol => "while",
    xor_symbol => "xor",
);

macro_rules! impl_froms {
    ($type:ty, $inner:ty) => {
        impl From<String> for $type {
            fn from(value: String) -> Self {
                Self(<$inner>::from(value))
            }
        }
        impl From<&'_ $type> for $type {
            fn from(value: &'_ $type) -> Self {
                value.clone()
            }
        }

        impl From<&'_ String> for $type {
            fn from(value: &'_ String) -> Self {
                Self(<$inner>::from(value))
            }
        }

        impl From<&'_ str> for $type {
            fn from(value: &'_ str) -> Self {
                Self(<$inner>::from(value))
            }
        }
    };
}

impl_froms!(Symbol, RootString);
impl_froms!(SymbolRef, RefString);

impl From<&'_ Symbol> for SymbolRef {
    fn from(value: &'_ Symbol) -> Self {
        value.downgrade()
    }
}

impl From<Symbol> for SymbolRef {
    fn from(value: Symbol) -> Self {
        value.downgrade()
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

impl PartialEq<SymbolRef> for Symbol {
    fn eq(&self, other: &SymbolRef) -> bool {
        self.0 == other.0
    }
}

impl PartialEq<Symbol> for SymbolRef {
    fn eq(&self, other: &Symbol) -> bool {
        self.0 == other.0
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

impl From<bool> for SymbolRef {
    fn from(bool: bool) -> Self {
        if bool {
            Symbol::true_symbol().downgrade()
        } else {
            Symbol::false_symbol().downgrade()
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
        Symbol::from(self + &**rhs)
    }
}

impl<'a, 'b> Add<&'a str> for &'b Symbol {
    type Output = String;

    fn add(self, rhs: &'a str) -> Self::Output {
        let mut out = String::with_capacity(self.len() + rhs.len());
        out.push_str(self);
        out.push_str(rhs);
        out
    }
}

impl<'a, 'b> Add<&'a Symbol> for &'b str {
    type Output = String;

    fn add(self, rhs: &'a Symbol) -> Self::Output {
        let mut out = String::with_capacity(self.len() + rhs.len());
        out.push_str(self);
        out.push_str(rhs);
        out
    }
}

impl<'a, 'b> Add<&'a String> for &'b Symbol {
    type Output = String;

    fn add(self, rhs: &'a String) -> Self::Output {
        self + rhs.as_str()
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
