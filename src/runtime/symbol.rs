//! Types for "symbols", an optimized string-like type that ensures only one
//! underlying copy of each unique string exists.

use std::fmt::{Debug, Display};
use std::ops::{Add, Deref};
use std::sync::OnceLock;
use std::{array, iter};

use refuse::{CollectionGuard, Trace};
use refuse_pool::{RefString, RootString};
use serde::de::Visitor;
use serde::{Deserialize, Serialize};

use crate::runtime::value::ValueFreed;

/// A garbage-collected weak reference to a [`Symbol`].
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash, Trace)]
pub struct SymbolRef(RefString);

impl SymbolRef {
    /// Loads the underlying string value of this symbol.
    ///
    /// Returns `None` if the underlying symbol has been freed by the garbage
    /// collector.
    #[must_use]
    pub fn load<'guard>(&self, guard: &'guard CollectionGuard<'_>) -> Option<&'guard str> {
        self.0.load(guard)
    }

    /// Tries to loads the underlying string value of this symbol.
    ///
    /// # Errors
    ///
    /// Returns [`ValueFreed`] if the underlying symbol has been freed by the
    /// garbage collector.
    pub fn try_load<'guard>(
        &self,
        guard: &'guard CollectionGuard<'_>,
    ) -> Result<&'guard str, ValueFreed> {
        self.load(guard).ok_or(ValueFreed)
    }

    /// Upgrades this weak reference to a reference-counted [`Symbol`].
    ///
    ///
    /// Returns `None` if the underlying symbol has been freed by the garbage
    /// collector.
    #[must_use]
    pub fn upgrade(&self, guard: &CollectionGuard<'_>) -> Option<Symbol> {
        self.0.as_root(guard).map(Symbol)
    }

    /// Tries to upgrade this weak reference to a reference-counted [`Symbol`].
    ///
    /// # Errors
    ///
    /// Returns [`ValueFreed`] if the underlying symbol has been freed by the
    /// garbage collector.
    pub fn try_upgrade(&self, guard: &CollectionGuard<'_>) -> Result<Symbol, ValueFreed> {
        self.upgrade(guard).ok_or(ValueFreed)
    }
}

impl kempt::Sort<SymbolRef> for Symbol {
    fn compare(&self, other: &SymbolRef) -> std::cmp::Ordering {
        self.0.downgrade_any().cmp(&other.0.as_any())
    }
}

/// A reference-counted, cheap-to-compare String type.
///
/// Symbols are optimized to be able to cheaply compare and hash without needing
/// to analyze the underlying string contents. This is done by ensuring that all
/// instances of the same underlying string data point to the same [`Symbol`].
#[derive(Clone, Trace)]
pub struct Symbol(RootString);

impl Symbol {
    /// Returns a weak reference to this symbol.
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
            $(
                #[doc = concat!("Returns the symbol for \"", $string, "\".")]
                pub fn $name() -> &'static Self {
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
    enum_symbol => "enum",
    false_symbol => "false",
    for_symbol => "for",
    fn_symbol => "fn",
    get_symbol => "get",
    if_symbol => "if",
    impl_symbol => "impl",
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
    new_symbol => "new",
    nil_symbol => "nil",
    not_symbol => "not",
    nth_symbol => "nth",
    or_symbol => "or",
    pub_symbol => "pub",
    return_symbol => "return",
    self_symbol => "self",
    set_symbol => "set",
    sigil_symbol => "$",
    struct_symbol => "struct",
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

/// A type that can be optionally be converted to a [`Symbol`].
pub trait IntoOptionSymbol {
    /// Returns this type as an optional symbol.
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

/// A [`Symbol`] that is initialized once and retains a reference to the
/// [`Symbol`].
///
/// This type is designed to be used as a `static`, allowing usages of common
/// symbols to never be garbage collected.
pub struct StaticSymbol(OnceLock<Symbol>, &'static str);

impl StaticSymbol {
    /// Returns a new static symbol from a static string.
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

/// A type that contains a list of symbols.
pub trait SymbolList {
    /// The iterator used for [`into_symbols`](Self::into_symbols).
    type Iterator: Iterator<Item = Symbol>;

    /// Returns `self` as an iterator over its contained symbols.
    fn into_symbols(self) -> Self::Iterator;
}

/// An iterator over an array of types that implement [`Into<Symbol>`].
pub struct ArraySymbolsIntoIter<T: Into<Symbol>, const N: usize>(array::IntoIter<T, N>);

impl<T: Into<Symbol>, const N: usize> Iterator for ArraySymbolsIntoIter<T, N> {
    type Item = Symbol;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(T::into)
    }
}

impl<T: Into<Symbol>, const N: usize> SymbolList for [T; N] {
    type Iterator = ArraySymbolsIntoIter<T, N>;

    fn into_symbols(self) -> Self::Iterator {
        ArraySymbolsIntoIter(self.into_iter())
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
