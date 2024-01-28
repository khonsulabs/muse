use std::fmt::Display;
use std::hash::Hash;
use std::sync::Mutex;

use crate::symbol::Symbol;
use crate::value::{CustomType, Dynamic, Value, ValueHasher};
use crate::vm::{Arity, Fault, Vm};

#[derive(Debug)]
pub struct MuseString(Mutex<String>);

impl From<String> for MuseString {
    fn from(value: String) -> Self {
        Self(Mutex::new(value))
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
        !self.0.lock().expect("poisoned").is_empty()
    }

    fn to_string(&self, _vm: &mut Vm) -> Result<Symbol, Fault> {
        Ok(Symbol::from(&*self.0.lock().expect("poisoned")))
    }

    fn deep_clone(&self) -> Option<Dynamic> {
        Some(Dynamic::new(Self(Mutex::new(
            self.0.lock().expect("poisoned").clone(),
        ))))
    }
}

impl Display for MuseString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.lock().expect("poisioned").fmt(f)
    }
}
