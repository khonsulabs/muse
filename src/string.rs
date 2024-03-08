use std::fmt::Display;
use std::hash::Hash;
use std::sync::{Mutex, MutexGuard};

use crate::list::List;
use crate::regex::MuseRegex;
use crate::symbol::Symbol;
use crate::value::{
    AnyDynamic, CustomType, Dynamic, RustType, StaticRustFunctionTable, TypeRef, Value,
};
use crate::vm::{Fault, Register, Vm};

#[derive(Debug)]
pub struct MuseString(Mutex<String>);

impl MuseString {
    pub(crate) fn lock(&self) -> MutexGuard<'_, String> {
        self.0.lock().expect("poisoned")
    }
}

impl From<String> for MuseString {
    fn from(value: String) -> Self {
        Self(Mutex::new(value))
    }
}

impl From<&'_ str> for MuseString {
    fn from(value: &'_ str) -> Self {
        Self::from(value.to_string())
    }
}

pub static STRING_TYPE: RustType<MuseString> = RustType::new("String", |t| {
    t.with_construct(|_| {
        |vm, arity| {
            if arity == 1 {
                if let Some(dynamic) = vm[Register(0)].as_dynamic::<MuseString>() {
                    Ok(dynamic)
                } else {
                    let value = vm[Register(0)].take().to_string(vm)?;
                    Ok(Dynamic::new(MuseString::from(&*value)))
                }
            } else if let Ok(length) = (0..arity.0).try_fold(0_usize, |accum, r| {
                vm[Register(r)]
                    .as_symbol()
                    .map_or(Err(()), |sym| Ok(accum + sym.len()))
            }) {
                let mut joined = String::with_capacity(length);
                for r in 0..arity.0 {
                    let Some(sym) = vm[Register(r)].as_symbol() else {
                        unreachable!()
                    };
                    joined.push_str(sym);
                }

                Ok(Dynamic::new(MuseString::from(joined)))
            } else {
                let mut joined = String::new();
                for r in 0..arity.0 {
                    let sym = vm[Register(r)].take().to_string(vm)?;
                    joined.push_str(&sym);
                }

                Ok(Dynamic::new(MuseString::from(joined)))
            }
        }
    })
    .with_hash(|_| {
        |this, _vm, hasher| {
            this.0.lock().expect("poisoned").hash(hasher);
        }
    })
    .with_eq(|_| {
        |this, _vm, rhs| {
            if let Some(rhs) = rhs.as_downcast_ref::<MuseString>() {
                Ok(*this.0.lock().expect("poisoned") == *rhs.0.lock().expect("poisoned"))
            } else if let Some(rhs) = rhs.as_symbol() {
                Ok(*this.0.lock().expect("poisoned") == **rhs)
            } else {
                Ok(false)
            }
        }
    })
    .with_total_cmp(|_| {
        |this, _vm, rhs| {
            if let Some(rhs) = rhs.as_downcast_ref::<MuseString>() {
                Ok(this
                    .0
                    .lock()
                    .expect("poisoned")
                    .cmp(&rhs.0.lock().expect("poisoned")))
            } else if rhs.as_any_dynamic().is_none() {
                // Dynamics sort after primitive values
                Ok(std::cmp::Ordering::Greater)
            } else {
                Err(Fault::UnsupportedOperation)
            }
        }
    })
    .with_invoke(|_| {
        |this, vm, name, arity| {
            static FUNCTIONS: StaticRustFunctionTable<MuseString> =
                StaticRustFunctionTable::new(|table| {
                    table
                        .with_fn(
                            Symbol::len_symbol(),
                            0,
                            |_vm: &mut Vm, this: &MuseString| {
                                Value::try_from(this.0.lock().expect("poisoned").len())
                            },
                        )
                        .with_fn("split", 1, |vm: &mut Vm, this: &MuseString| {
                            let needle = vm[Register(0)].take();
                            if let Some(needle) = needle.as_downcast_ref::<MuseString>() {
                                if std::ptr::eq(&this.0, &needle.0) {
                                    Ok(Value::dynamic(List::from(vec![Value::dynamic(
                                        MuseString::from(String::default()),
                                    )])))
                                } else {
                                    let haystack = this.lock();
                                    let needle = needle.lock();
                                    Ok(Value::dynamic(
                                        haystack
                                            .split(&*needle)
                                            .map(|segment| {
                                                Value::dynamic(MuseString::from(segment))
                                            })
                                            .collect::<List>(),
                                    ))
                                }
                            } else if let Some(needle) = needle.as_downcast_ref::<MuseRegex>() {
                                let haystack = this.lock();
                                Ok(Value::dynamic(
                                    needle
                                        .split(&haystack)
                                        .map(|segment| Value::dynamic(MuseString::from(segment)))
                                        .collect::<List>(),
                                ))
                            } else {
                                Err(Fault::ExpectedString)
                            }
                        })
                });

            FUNCTIONS.invoke(vm, name, arity, &this)
        }
    })
    .with_add(|_| {
        |this, vm, rhs| {
            let lhs = this.0.lock().expect("poisoned");
            if let Some(rhs) = rhs.as_downcast_ref::<MuseString>() {
                if std::ptr::eq(&this.0, &rhs.0) {
                    let mut repeated =
                        String::with_capacity(lhs.len().checked_mul(2).ok_or(Fault::OutOfMemory)?);
                    repeated.push_str(&lhs);
                    repeated.push_str(&lhs);
                    Ok(Value::dynamic(MuseString::from(repeated)))
                } else {
                    let rhs = rhs.0.lock().expect("poisoned");
                    let mut combined = String::with_capacity(lhs.len() + rhs.len());
                    combined.push_str(&lhs);
                    combined.push_str(&rhs);
                    Ok(Value::dynamic(MuseString::from(combined)))
                }
            } else {
                let rhs = rhs.to_string(vm)?;
                let mut combined = String::with_capacity(lhs.len() + rhs.len());
                combined.push_str(&lhs);
                combined.push_str(&rhs);
                Ok(Value::dynamic(MuseString::from(combined)))
            }
        }
    })
    .with_add_right(|_| {
        |this, vm, lhs| {
            let lhs = lhs.to_string(vm)?;
            let rhs = this.0.lock().expect("poisoned");
            let mut combined = String::with_capacity(rhs.len() + lhs.len());
            combined.push_str(&lhs);
            combined.push_str(&rhs);
            Ok(Value::dynamic(MuseString::from(combined)))
        }
    })
    .with_mul(|_| {
        |this, _vm, rhs| {
            let Some(times) = rhs.as_usize() else {
                return Err(Fault::ExpectedInteger);
            };
            let this = this.0.lock().expect("poisoned");

            Ok(Value::dynamic(MuseString::from(this.repeat(times))))
        }
    })
    .with_mul_right(|_| {
        |this, _vm, lhs| {
            let Some(times) = lhs.as_usize() else {
                return Err(Fault::ExpectedInteger);
            };
            let this = this.0.lock().expect("poisoned");

            Ok(Value::dynamic(MuseString::from(this.repeat(times))))
        }
    })
    .with_truthy(|_| |this, _vm| !this.0.lock().expect("poisoned").is_empty())
    .with_to_string(|_| |this, _vm| Ok(Symbol::from(&*this.0.lock().expect("poisoned"))))
    .with_deep_clone(|_| {
        |this| {
            Some(AnyDynamic::new(MuseString(Mutex::new(
                this.0.lock().expect("poisoned").clone(),
            ))))
        }
    })
});

impl CustomType for MuseString {
    #[allow(clippy::too_many_lines)]
    fn muse_type(&self) -> &TypeRef {
        &STRING_TYPE
    }
}

impl Display for MuseString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.lock().expect("poisioned").fmt(f)
    }
}
