//! Types used for strings/text.

use std::fmt::Display;
use std::hash::Hash;

use parking_lot::{Mutex, MutexGuard};
use refuse::ContainsNoRefs;

use crate::runtime::list::List;
use crate::runtime::regex::MuseRegex;
use crate::runtime::symbol::{Symbol, SymbolRef};
use crate::runtime::value::{
    AnyDynamic, CustomType, Dynamic, Rooted, RustFunctionTable, RustType, TypeRef, Value,
};
use crate::vm::{Fault, Register, VmContext};

/// The [`String`] type for Muse.
#[derive(Debug)]
pub struct MuseString(Mutex<String>);

impl MuseString {
    pub(crate) fn lock(&self) -> MutexGuard<'_, String> {
        self.0.lock()
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

impl ContainsNoRefs for MuseString {}

/// The type definition that the [`MuseString`] type uses.
///
/// In general, developers will not need this. However, if you are building your
/// own `core` module, this type can be used to populate `$.core.String`.
pub static STRING_TYPE: RustType<MuseString> = RustType::new("String", |t| {
    t.with_construct(|_| {
        |vm, arity| {
            if arity == 1 {
                if let Some(dynamic) = vm[Register(0)].as_dynamic::<MuseString>() {
                    Ok(dynamic)
                } else {
                    let value = vm[Register(0)].take().to_string(vm)?.try_load(vm.guard())?;
                    Ok(Dynamic::new(MuseString::from(value), vm))
                }
            } else if let Ok(length) = (0..arity.0).try_fold(0_usize, |accum, r| {
                let value = vm[Register(r)];
                value.map_str(vm, |_vm, s| accum + s.len())
            }) {
                let mut joined = String::with_capacity(length);
                for r in 0..arity.0 {
                    let value = vm[Register(r)];
                    value
                        .map_str(vm, |_vm, s| joined.push_str(s))
                        .expect("just tested");
                }

                Ok(Dynamic::new(MuseString::from(joined), vm))
            } else {
                let mut joined = String::new();
                for r in 0..arity.0 {
                    let sym = vm[Register(r)]
                        .take()
                        .to_string(vm)?
                        .try_upgrade(vm.guard())?;
                    joined.push_str(&sym);
                }

                Ok(Dynamic::new(MuseString::from(joined), vm))
            }
        }
    })
    .with_hash(|_| {
        |this, _vm, hasher| {
            this.0.lock().hash(hasher);
        }
    })
    .with_eq(|_| {
        |this, vm, rhs| {
            if let Some(rhs) = rhs.as_downcast_ref::<MuseString>(vm.as_ref()) {
                Ok(*this.0.lock() == *rhs.0.lock())
            } else if let Some(rhs) = rhs.as_symbol(vm.as_ref()) {
                Ok(*this.0.lock() == *rhs)
            } else {
                Ok(false)
            }
        }
    })
    .with_total_cmp(|_| {
        |this, vm, rhs| {
            if let Some(rhs) = rhs.as_downcast_ref::<MuseString>(vm.as_ref()) {
                Ok(this.0.lock().cmp(&rhs.0.lock()))
            } else if rhs.as_any_dynamic().is_none() {
                // Dynamics sort after primitive values
                Ok(std::cmp::Ordering::Greater)
            } else {
                Err(Fault::UnsupportedOperation)
            }
        }
    })
    .with_function_table(
        RustFunctionTable::new()
            .with_fn(
                Symbol::len_symbol(),
                0,
                |_vm: &mut VmContext<'_, '_>, this: &Rooted<MuseString>| {
                    Value::try_from(this.0.lock().len())
                },
            )
            .with_fn(
                "split",
                1,
                |vm: &mut VmContext<'_, '_>, this: &Rooted<MuseString>| {
                    let needle = vm[Register(0)].take();
                    if let Some(needle) = needle.as_rooted::<MuseString>(vm.as_ref()) {
                        if std::ptr::eq(&this.0, &needle.0) {
                            Ok(Value::dynamic(
                                List::from(vec![Value::dynamic(
                                    MuseString::from(String::default()),
                                    &vm,
                                )]),
                                vm,
                            ))
                        } else {
                            let haystack = this.lock();
                            let needle = needle.lock();
                            Ok(Value::dynamic(
                                haystack
                                    .split(&*needle)
                                    .map(|segment| Value::dynamic(MuseString::from(segment), &vm))
                                    .collect::<List>(),
                                vm,
                            ))
                        }
                    } else if let Some(needle) = needle.as_downcast_ref::<MuseRegex>(vm.as_ref()) {
                        let haystack = this.lock();
                        Ok(Value::dynamic(
                            needle
                                .split(&haystack)
                                .map(|segment| Value::dynamic(MuseString::from(segment), &vm))
                                .collect::<List>(),
                            vm,
                        ))
                    } else {
                        Err(Fault::ExpectedString)
                    }
                },
            ),
    )
    .with_add(|_| {
        |this, vm, rhs| {
            let lhs = this.0.lock();
            if let Some(rhs) = rhs.as_downcast_ref::<MuseString>(vm.as_ref()) {
                if std::ptr::eq(&this.0, &rhs.0) {
                    let mut repeated =
                        String::with_capacity(lhs.len().checked_mul(2).ok_or(Fault::OutOfMemory)?);
                    repeated.push_str(&lhs);
                    repeated.push_str(&lhs);
                    Ok(Value::dynamic(MuseString::from(repeated), vm))
                } else {
                    let rhs = rhs.0.lock();
                    let mut combined = String::with_capacity(lhs.len() + rhs.len());
                    combined.push_str(&lhs);
                    combined.push_str(&rhs);
                    Ok(Value::dynamic(MuseString::from(combined), &vm))
                }
            } else {
                let rhs = rhs.to_string(vm)?.try_upgrade(vm.guard())?;
                let mut combined = String::with_capacity(lhs.len() + rhs.len());
                combined.push_str(&lhs);
                combined.push_str(&rhs);
                Ok(Value::dynamic(MuseString::from(combined), vm))
            }
        }
    })
    .with_add_right(|_| {
        |this, vm, lhs| {
            let lhs = lhs.to_string(vm)?.try_upgrade(vm.guard())?;
            let rhs = this.0.lock();
            let mut combined = String::with_capacity(rhs.len() + lhs.len());
            combined.push_str(&lhs);
            combined.push_str(&rhs);
            Ok(Value::dynamic(MuseString::from(combined), vm))
        }
    })
    .with_mul(|_| {
        |this, vm, rhs| {
            let Some(times) = rhs.as_usize() else {
                return Err(Fault::ExpectedInteger);
            };
            let this = this.0.lock();

            Ok(Value::dynamic(MuseString::from(this.repeat(times)), vm))
        }
    })
    .with_mul_right(|_| {
        |this, vm, lhs| {
            let Some(times) = lhs.as_usize() else {
                return Err(Fault::ExpectedInteger);
            };
            let this = this.0.lock();

            Ok(Value::dynamic(MuseString::from(this.repeat(times)), vm))
        }
    })
    .with_truthy(|_| |this, _vm| !this.0.lock().is_empty())
    .with_to_string(|_| |this, _vm| Ok(SymbolRef::from(&*this.0.lock())))
    .with_deep_clone(|_| {
        |this, guard| {
            Some(AnyDynamic::new(
                MuseString(Mutex::new(this.0.lock().clone())),
                guard,
            ))
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
        self.0.lock().fmt(f)
    }
}
