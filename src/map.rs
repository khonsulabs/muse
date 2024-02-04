use std::cmp::Ordering;
use std::sync::Mutex;

use crate::symbol::Symbol;
use crate::value::{CustomType, StaticRustFunctionTable, Value};
use crate::vm::{Arity, Fault, Register, Vm};

#[derive(Debug)]
pub struct Map(Mutex<Vec<Field>>);

impl Map {
    #[must_use]
    pub const fn new() -> Self {
        Self(Mutex::new(Vec::new()))
    }

    pub fn from_iterator(vm: &mut Vm, iter: impl IntoIterator<Item = (Value, Value)>) -> Self {
        let mut fields: Vec<_> = iter
            .into_iter()
            .map(|(key, value)| Field {
                key: MapKey::new(vm, key),
                value,
            })
            .collect();
        fields.sort_unstable_by(|a, b| a.key.hash.cmp(&b.key.hash));
        Self(Mutex::new(fields))
    }

    pub fn get(&self, vm: &mut Vm, key: &Value) -> Result<Option<Value>, Fault> {
        let hash = key.hash(vm);
        let contents = self.0.lock().expect("poisoned");
        for field in &*contents {
            match hash.cmp(&field.key.hash) {
                Ordering::Less => continue,
                Ordering::Equal => {
                    if key.equals(Some(vm), &field.key.value)? {
                        return Ok(Some(field.value.clone()));
                    }
                }
                Ordering::Greater => break,
            }
        }

        Ok(None)
    }

    pub fn insert(&self, vm: &mut Vm, key: Value, value: Value) -> Result<Option<Value>, Fault> {
        let key = MapKey::new(vm, key);
        let mut contents = self.0.lock().expect("poisoned");
        let mut insert_at = contents.len();
        for (index, field) in contents.iter_mut().enumerate() {
            match key.hash.cmp(&field.key.hash) {
                Ordering::Less => continue,
                Ordering::Equal => {
                    if key.value.equals(Some(vm), &field.key.value)? {
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

    pub fn to_vec(&self) -> Vec<Field> {
        self.0.lock().expect("poisoned").clone()
    }
}

impl CustomType for Map {
    fn invoke(&self, vm: &mut Vm, name: &Symbol, arity: Arity) -> Result<Value, Fault> {
        static FUNCTIONS: StaticRustFunctionTable<Map> = StaticRustFunctionTable::new(|table| {
            table
                .with_fn(Symbol::set_symbol(), 1, |vm, this| {
                    let key = vm[Register(0)].take();
                    let value = key.clone();
                    this.insert(vm, key, value.clone())?;
                    Ok(value)
                })
                .with_fn(Symbol::set_symbol(), 2, |vm, this| {
                    let key = vm[Register(0)].take();
                    let value = vm[Register(1)].take();
                    this.insert(vm, key, value.clone())?;
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

#[derive(Debug, Clone)]
pub struct Field {
    key: MapKey,
    value: Value,
}

impl Field {
    #[must_use]
    pub fn into_parts(self) -> (Value, Value) {
        (self.key.value, self.value)
    }
}

#[derive(Debug, Clone)]
struct MapKey {
    value: Value,
    hash: u64,
}

impl MapKey {
    pub fn new(vm: &mut Vm, key: Value) -> Self {
        Self {
            hash: key.hash(vm),
            value: key,
        }
    }
}
