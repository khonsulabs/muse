use std::fmt::Display;
use std::hash::Hash;
use std::ops::Deref;

use kempt::Map;
use regex::{Captures, Regex};

use crate::string::MuseString;
use crate::symbol::Symbol;
use crate::syntax::token::RegexLiteral;
use crate::value::{AnyDynamic, CustomType, RustType, StaticRustFunctionTable, TypeRef, Value};
use crate::vm::{Fault, Register, Vm};

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

    #[must_use]
    pub fn captures(&self, haystack: &str) -> Option<MuseCaptures> {
        self.expr
            .captures(haystack)
            .map(|captures| MuseCaptures::new(&captures, haystack, self))
    }
}

impl Deref for MuseRegex {
    type Target = Regex;

    fn deref(&self) -> &Self::Target {
        &self.expr
    }
}

impl CustomType for MuseRegex {
    fn muse_type(&self) -> &TypeRef {
        static TYPE: RustType<MuseRegex> = RustType::new("Regex", |t| {
            t.with_invoke(|_| {
                |this, vm, name, arity| {
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
                                .with_fn(
                                    Symbol::captures_symbol(),
                                    1,
                                    |vm: &mut Vm, this: &MuseRegex| {
                                        let haystack = vm[Register(0)].take();
                                        haystack.map_str(vm, |_vm, haystack| {
                                            this.captures(haystack)
                                                .map(Value::dynamic)
                                                .unwrap_or_default()
                                        })
                                    },
                                )
                        });
                    FUNCTIONS.invoke(vm, name, arity, &this)
                }
            })
            .with_hash(|_| {
                |this, _vm, hasher| {
                    this.expr.as_str().hash(hasher);
                }
            })
            .with_eq(|_| {
                |this, _vm, rhs| {
                    if let Some(rhs) = rhs.as_downcast_ref::<MuseRegex>() {
                        Ok(this.expr.as_str() == rhs.expr.as_str())
                    } else {
                        Ok(false)
                    }
                }
            })
            .with_total_cmp(|_| {
                |this, _vm, rhs| {
                    if let Some(rhs) = rhs.as_downcast_ref::<MuseRegex>() {
                        Ok(this.expr.as_str().cmp(rhs.expr.as_str()))
                    } else if rhs.as_any_dynamic().is_none() {
                        // Dynamics sort after primitive values
                        Ok(std::cmp::Ordering::Greater)
                    } else {
                        Err(Fault::UnsupportedOperation)
                    }
                }
            })
            .with_truthy(|_| |this, _vm| !this.expr.as_str().is_empty())
            .with_to_string(|_| |this, _vm| Ok(Symbol::from(this.expr.as_str().to_string())))
            .with_clone()
            .with_matches(|_| {
                |this, vm, rhs| {
                    if let Some(rhs) = rhs.as_downcast_ref::<MuseString>() {
                        Ok(this.is_match(&rhs.lock()))
                    } else {
                        rhs.map_str(vm, |_vm, rhs| this.is_match(rhs))
                    }
                }
            })
        });
        &TYPE
    }
}

impl Display for MuseRegex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.expr, f)
    }
}

#[derive(Debug, Clone)]
pub struct MuseCaptures {
    matches: Vec<Value>,
    by_name: Map<Symbol, usize>,
}

impl MuseCaptures {
    fn new(captures: &Captures<'_>, haystack: &str, regex: &Regex) -> Self {
        let named_captures = regex.capture_names().filter(Option::is_some).count();
        let by_name = if named_captures == 0 {
            Map::new()
        } else {
            regex
                .capture_names()
                .enumerate()
                .filter_map(|(index, name)| name.map(|name| (Symbol::from(name), index)))
                .collect()
        };

        let matches = captures
            .iter()
            .map(|capture| {
                capture
                    .map(|capture| {
                        Value::dynamic(MuseMatch {
                            content: AnyDynamic::new(MuseString::from(&haystack[capture.range()])),
                            start: capture.start(),
                        })
                    })
                    .unwrap_or_default()
            })
            .collect();

        Self { matches, by_name }
    }
}

impl CustomType for MuseCaptures {
    fn muse_type(&self) -> &TypeRef {
        static TYPE: RustType<MuseCaptures> = RustType::new("RegexCaptures", |t| {
            t.with_invoke(|_| {
                |this, vm, name, arity| {
                    static FUNCTIONS: StaticRustFunctionTable<MuseCaptures> =
                        StaticRustFunctionTable::new(|table| {
                            table
                                .with_fn(
                                    Symbol::get_symbol(),
                                    1,
                                    |vm: &mut Vm, this: &MuseCaptures| {
                                        let index = vm[Register(0)].take();
                                        let index = if let Some(index) = index.as_usize() {
                                            index
                                        } else {
                                            let name = index.to_string(vm)?;
                                            let Some(index) = this.by_name.get(&name).copied()
                                            else {
                                                return Ok(Value::Nil);
                                            };
                                            index
                                        };
                                        this.matches.get(index).cloned().ok_or(Fault::OutOfBounds)
                                    },
                                )
                                .with_fn(
                                    Symbol::nth_symbol(),
                                    1,
                                    |vm: &mut Vm, this: &MuseCaptures| {
                                        let index =
                                            vm[Register(0)].as_usize().ok_or(Fault::OutOfBounds)?;
                                        this.matches.get(index).cloned().ok_or(Fault::OutOfBounds)
                                    },
                                )
                        });
                    FUNCTIONS.invoke(vm, name, arity, &this)
                }
            })
        });
        &TYPE
    }
}

#[derive(Clone, Debug)]
pub struct MuseMatch {
    pub content: AnyDynamic,
    pub start: usize,
}

impl CustomType for MuseMatch {
    fn muse_type(&self) -> &TypeRef {
        static TYPE: RustType<MuseMatch> = RustType::new("RegexMatch", |t| {
            t.with_invoke(|_| {
                |this, vm, name, arity| {
                    static FUNCTIONS: StaticRustFunctionTable<MuseMatch> =
                        StaticRustFunctionTable::new(|table| {
                            table
                                .with_fn("content", 0, |_vm: &mut Vm, this: &MuseMatch| {
                                    Ok(Value::Dynamic(this.content.clone()))
                                })
                                .with_fn("start", 0, |_vm: &mut Vm, this: &MuseMatch| {
                                    Value::try_from(this.start)
                                })
                        });
                    FUNCTIONS.invoke(vm, name, arity, &this)
                }
            })
            .with_to_string(|_| |this, vm| this.content.to_string(vm))
        });
        &TYPE
    }
}
