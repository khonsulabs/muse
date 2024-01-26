use std::cmp::Ordering;
use std::sync::Mutex;

use crate::symbol::Symbol;
use crate::value::{CustomType, Value};
use crate::vm::{Arity, Fault, Register, Vm};

#[derive(Debug)]
pub struct Map(Mutex<Vec<Field>>);

impl Map {
    #[must_use]
    pub const fn new() -> Self {
        Self(Mutex::new(Vec::new()))
    }

    pub fn get(&self, vm: &mut Vm, key: &Value) -> Result<Option<Value>, Fault> {
        let hash = key.hash(vm)?;
        let contents = self.0.lock().expect("poisoned");
        for field in &*contents {
            match hash.cmp(&field.key.hash) {
                Ordering::Less => continue,
                Ordering::Equal => {
                    if key.eq(vm, &field.key.value)? {
                        return Ok(Some(field.value.clone()));
                    }
                }
                Ordering::Greater => break,
            }
        }

        Ok(None)
    }

    pub fn insert(&self, vm: &mut Vm, key: Value, value: Value) -> Result<Option<Value>, Fault> {
        let key = MapKey::new(vm, key)?;
        let mut contents = self.0.lock().expect("poisoned");
        let mut insert_at = contents.len();
        for (index, field) in contents.iter_mut().enumerate() {
            match key.hash.cmp(&field.key.hash) {
                Ordering::Less => continue,
                Ordering::Equal => {
                    if key.value.eq(vm, &field.key.value)? {
                        return Ok(Some(std::mem::replace(&mut field.value, value)));
                    }
                }
                Ordering::Greater => {
                    insert_at = index;
                    break;
                }
            }
        }

        contents.insert(insert_at, Field { key, value });

        Ok(None)
    }
}

impl CustomType for Map {
    fn invoke(&self, vm: &mut Vm, name: &Symbol, arity: Arity) -> Result<Value, Fault> {
        if name == &Symbol::set_symbol() && (arity == 1 || arity == 2) {
            let key = vm[Register(0)].take();
            let value = if arity == 2 {
                vm[Register(1)].take()
            } else {
                key.clone()
            };
            self.insert(vm, key, value.clone())?;
            Ok(value)
        } else if name == &Symbol::get_symbol() && arity == 1 {
            let key = vm[Register(0)].take();
            Ok(self.get(vm, &key)?.unwrap_or_default())
        } else {
            Err(Fault::UnknownSymbol(name.clone()))
        }
    }
}

#[derive(Debug, Clone)]
struct Field {
    key: MapKey,
    value: Value,
}

#[derive(Debug, Clone)]
struct MapKey {
    value: Value,
    hash: u64,
}

impl MapKey {
    pub fn new(vm: &mut Vm, key: Value) -> Result<Self, Fault> {
        Ok(Self {
            hash: key.hash(vm)?,
            value: key,
        })
    }
}
