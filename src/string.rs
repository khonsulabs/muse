use std::fmt::Display;
use std::hash::Hash;
use std::sync::{Mutex, MutexGuard};

use crate::list::List;
use crate::regex::MuseRegex;
use crate::symbol::Symbol;
use crate::value::{AnyDynamic, CustomType, StaticRustFunctionTable, Value, ValueHasher};
use crate::vm::{Arity, Fault, Register, Vm};

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

impl CustomType for MuseString {
    fn hash(&self, _vm: &mut Vm, hasher: &mut ValueHasher) {
        self.0.lock().expect("poisoned").hash(hasher);
    }

    fn eq(&self, _vm: Option<&mut Vm>, rhs: &Value) -> Result<bool, Fault> {
        if let Some(rhs) = rhs.as_downcast_ref::<Self>() {
            Ok(*self.0.lock().expect("poisoned") == *rhs.0.lock().expect("poisoned"))
        } else {
            Ok(false)
        }
    }

    fn total_cmp(&self, _vm: &mut Vm, rhs: &Value) -> Result<std::cmp::Ordering, Fault> {
        if let Some(rhs) = rhs.as_downcast_ref::<Self>() {
            Ok(self
                .0
                .lock()
                .expect("poisoned")
                .cmp(&rhs.0.lock().expect("poisoned")))
        } else if rhs.as_dynamic().is_none() {
            // Dynamics sort after primitive values
            Ok(std::cmp::Ordering::Greater)
        } else {
            Err(Fault::UnsupportedOperation)
        }
    }

    fn invoke(&self, vm: &mut Vm, name: &Symbol, arity: Arity) -> Result<Value, Fault> {
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
                                        .map(|segment| Value::dynamic(MuseString::from(segment)))
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

        FUNCTIONS.invoke(vm, name, arity, self)
    }

    fn add(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        let this = self.0.lock().expect("poisoned");
        if let Some(rhs) = rhs.as_downcast_ref::<MuseString>() {
            if std::ptr::eq(&self.0, &rhs.0) {
                let mut repeated =
                    String::with_capacity(this.len().checked_mul(2).ok_or(Fault::OutOfMemory)?);
                repeated.push_str(&this);
                repeated.push_str(&this);
                Ok(Value::dynamic(MuseString::from(repeated)))
            } else {
                let rhs = rhs.0.lock().expect("poisoned");
                let mut combined = String::with_capacity(this.len() + rhs.len());
                combined.push_str(&this);
                combined.push_str(&rhs);
                Ok(Value::dynamic(MuseString::from(combined)))
            }
        } else {
            let rhs = rhs.to_string(vm)?;
            let mut combined = String::with_capacity(this.len() + rhs.len());
            combined.push_str(&this);
            combined.push_str(&rhs);
            Ok(Value::dynamic(MuseString::from(combined)))
        }
    }

    fn add_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        let this = self.0.lock().expect("poisoned");
        let lhs = lhs.to_string(vm)?;
        let mut combined = String::with_capacity(this.len() + lhs.len());
        combined.push_str(&lhs);
        combined.push_str(&this);
        Ok(Value::dynamic(MuseString::from(combined)))
    }

    fn mul(&self, _vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        let Some(times) = rhs.as_usize() else {
            return Err(Fault::ExpectedInteger);
        };
        let this = self.0.lock().expect("poisoned");

        Ok(Value::dynamic(MuseString::from(this.repeat(times))))
    }

    fn mul_right(&self, _vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        let Some(times) = lhs.as_usize() else {
            return Err(Fault::ExpectedInteger);
        };
        let this = self.0.lock().expect("poisoned");

        Ok(Value::dynamic(MuseString::from(this.repeat(times))))
    }

    fn truthy(&self, _vm: &mut Vm) -> bool {
        !self.0.lock().expect("poisoned").is_empty()
    }

    fn to_string(&self, _vm: &mut Vm) -> Result<Symbol, Fault> {
        Ok(Symbol::from(&*self.0.lock().expect("poisoned")))
    }

    fn deep_clone(&self) -> Option<AnyDynamic> {
        Some(AnyDynamic::new(Self(Mutex::new(
            self.0.lock().expect("poisoned").clone(),
        ))))
    }
}

impl Display for MuseString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.lock().expect("poisioned").fmt(f)
    }
}
