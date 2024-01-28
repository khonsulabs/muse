use std::sync::Mutex;

use crate::symbol::Symbol;
use crate::value::{CustomType, StaticRustFunctionTable, Value};
use crate::vm::{Arity, Fault, Register, Vm};

#[derive(Debug)]
pub struct List(Mutex<Vec<Value>>);

impl List {
    #[must_use]
    pub const fn new() -> Self {
        Self(Mutex::new(Vec::new()))
    }

    pub fn get(&self, _vm: &mut Vm, index: &Value) -> Result<Option<Value>, Fault> {
        let Some(index) = index.as_usize() else {
            return Err(Fault::InvalidIndex);
        };
        let contents = self.0.lock().expect("poisoned");

        Ok(contents.get(index).cloned())
    }

    pub fn insert(&self, vm: &mut Vm, index: &Value, value: Value) -> Result<(), Fault> {
        let Some(index) = index.as_usize() else {
            return Err(Fault::InvalidIndex);
        };
        let mut contents = self.0.lock().expect("poisoned");
        contents.insert(index, value);
        Ok(())
    }

    pub fn push(&self, vm: &mut Vm, value: Value) -> Result<(), Fault> {
        let mut contents = self.0.lock().expect("poisoned");
        contents.push(value);
        Ok(())
    }

    pub fn set(&self, vm: &mut Vm, index: &Value, value: Value) -> Result<Option<Value>, Fault> {
        let Some(index) = index.as_usize() else {
            return Err(Fault::InvalidIndex);
        };
        let mut contents = self.0.lock().expect("poisoned");

        if let Some(entry) = contents.get_mut(index) {
            Ok(Some(std::mem::replace(entry, value)))
        } else {
            Err(Fault::OutOfBounds)
        }
    }

    pub fn to_vec(&self) -> Vec<Value> {
        self.0.lock().expect("poisoned").clone()
    }
}

impl CustomType for List {
    fn invoke(&self, vm: &mut Vm, name: &Symbol, arity: Arity) -> Result<Value, Fault> {
        static FUNCTIONS: StaticRustFunctionTable<List> = StaticRustFunctionTable::new(|table| {
            table
                .with_fn(Symbol::set_symbol(), 2, |vm, this| {
                    let index = vm[Register(0)].take();
                    let value = vm[Register(1)].take();
                    this.set(vm, &index, value.clone())?;
                    Ok(value)
                })
                .with_fn(Symbol::get_symbol(), 1, |vm, this| {
                    let key = vm[Register(0)].take();
                    Ok(this.get(vm, &key)?.unwrap_or_default())
                })
        });
        FUNCTIONS.invoke(vm, name, arity, self)
    }
}
