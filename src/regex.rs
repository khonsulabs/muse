use std::fmt::Display;
use std::hash::Hash;
use std::ops::Deref;

use regex::Regex;

use crate::symbol::Symbol;
use crate::syntax::token::RegExLiteral;
use crate::value::{CustomType, Dynamic, Value, ValueHasher};
use crate::vm::{Arity, Fault, Vm};

#[derive(Debug, Clone)]
pub struct MuseRegEx {
    expr: Regex,
    literal: RegExLiteral,
}

impl MuseRegEx {
    pub fn new(literal: &RegExLiteral) -> Result<Self, regex::Error> {
        let expr = regex::RegexBuilder::new(&literal.pattern)
            .ignore_whitespace(literal.expanded)
            .crlf(true)
            .build()?;

        Ok(Self {
            expr,
            literal: literal.clone(),
        })
    }

    #[must_use]
    pub const fn literal(&self) -> &RegExLiteral {
        &self.literal
    }
}

impl Deref for MuseRegEx {
    type Target = Regex;

    fn deref(&self) -> &Self::Target {
        &self.expr
    }
}

impl CustomType for MuseRegEx {
    fn hash(&self, _vm: &mut Vm, hasher: &mut ValueHasher) {
        self.expr.as_str().hash(hasher);
    }

    fn eq(&self, _vm: Option<&mut Vm>, rhs: &Value) -> Result<bool, Fault> {
        if let Some(rhs) = rhs.as_downcast_ref::<Self>() {
            Ok(self.expr.as_str() == rhs.expr.as_str())
        } else {
            Ok(false)
        }
    }

    fn total_cmp(&self, _vm: &mut Vm, rhs: &Value) -> Result<std::cmp::Ordering, Fault> {
        if let Some(rhs) = rhs.as_downcast_ref::<Self>() {
            Ok(self.expr.as_str().cmp(rhs.expr.as_str()))
        } else if rhs.as_dynamic().is_none() {
            // Dynamics sort after primitive values
            Ok(std::cmp::Ordering::Greater)
        } else {
            Err(Fault::UnsupportedOperation)
        }
    }

    fn invoke(&self, vm: &mut Vm, name: &Symbol, arity: Arity) -> Result<Value, Fault> {
        Err(Fault::UnknownSymbol(name.clone()))
    }

    fn add(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    fn add_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    fn mul(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    fn mul_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    fn div(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    fn truthy(&self, _vm: &mut Vm) -> bool {
        !self.expr.as_str().is_empty()
    }

    fn to_string(&self, _vm: &mut Vm) -> Result<Symbol, Fault> {
        Ok(Symbol::from(self.expr.as_str().to_string()))
    }

    fn deep_clone(&self) -> Option<Dynamic> {
        Some(Dynamic::new(self.clone()))
    }
}

impl Display for MuseRegEx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.expr, f)
    }
}
