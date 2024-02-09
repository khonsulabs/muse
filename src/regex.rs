use std::fmt::Display;
use std::hash::Hash;
use std::ops::Deref;

use regex::Regex;

use crate::symbol::Symbol;
use crate::syntax::token::RegexLiteral;
use crate::value::{CustomType, Dynamic, StaticRustFunctionTable, Value, ValueHasher};
use crate::vm::{Arity, Fault, Vm};

#[derive(Debug, Clone)]
pub struct MuseRegex {
    expr: Regex,
    literal: RegexLiteral,
}

impl MuseRegex {
    pub fn new(literal: &RegexLiteral) -> Result<Self, regex::Error> {
        let expr = regex::RegexBuilder::new(&literal.pattern)
            .ignore_whitespace(literal.expanded)
            .crlf(true)
            .unicode(literal.unicode)
            .dot_matches_new_line(literal.dot_matches_all)
            .case_insensitive(literal.ignore_case)
            .multi_line(literal.multiline)
            .build()?;

        Ok(Self {
            expr,
            literal: literal.clone(),
        })
    }

    #[must_use]
    pub const fn literal(&self) -> &RegexLiteral {
        &self.literal
    }
}

impl Deref for MuseRegex {
    type Target = Regex;

    fn deref(&self) -> &Self::Target {
        &self.expr
    }
}

impl CustomType for MuseRegex {
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
        static FUNCTIONS: StaticRustFunctionTable<MuseRegex> =
            StaticRustFunctionTable::new(|table| {
                table
                    .with_fn("total_captures", 0, |_vm: &mut Vm, this: &MuseRegex| {
                        Value::try_from(this.captures_len())
                    })
                    .with_fn(
                        "total_static_captures",
                        0,
                        |_vm: &mut Vm, this: &MuseRegex| {
                            Ok(this
                                .static_captures_len()
                                .map(Value::try_from)
                                .transpose()?
                                .unwrap_or_default())
                        },
                    )
            });

        FUNCTIONS.invoke(vm, name, arity, self)
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

impl Display for MuseRegex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.expr, f)
    }
}
