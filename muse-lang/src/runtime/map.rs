//! Types used for maps/dictionaries.

use std::cmp::Ordering;

use parking_lot::Mutex;
use refuse::{CollectionGuard, Trace};

use crate::runtime::list::List;
use crate::runtime::symbol::Symbol;
use crate::runtime::value::{
    ContextOrGuard, CustomType, Dynamic, RustFunctionTable, RustType, TypeRef, Value,
};
use crate::vm::{Fault, Register, VmContext};

/// A collection of key-value pairs.
#[derive(Debug)]
pub struct Map(Mutex<Vec<Field>>);

/// The type definition that the [`Map`] type uses.
///
/// In general, developers will not need this. However, if you are building your
/// own `core` module, this type can be used to populate `$.core.Map`.
pub static MAP_TYPE: RustType<Map> = RustType::new("Map", |t| {
    t.with_construct(|_| {
        |vm, arity| {
            let map = Map::new();
            for reg_index in (0..arity.0 * 2).step_by(2) {
                let key = vm[Register(reg_index)].take();
                let value = vm[Register(reg_index + 1)].take();
                map.insert(vm, key, value)?;
            }
            Ok(Dynamic::new(map, vm))
        }
    })
    .with_function_table(
        RustFunctionTable::<Map>::new()
            .with_fn(Symbol::set_symbol(), 1, |vm, this| {
                let key = vm[Register(0)].take();
                this.insert(vm, key, key)?;
                Ok(key)
            })
            .with_fn(Symbol::set_symbol(), 2, |vm, this| {
                let key = vm[Register(0)].take();
                let value = vm[Register(1)].take();
                this.insert(vm, key, value)?;
                Ok(value)
            })
            .with_fn(Symbol::get_symbol(), 1, |vm, this| {
                let key = vm[Register(0)].take();
                this.get(vm, &key)?.ok_or(Fault::OutOfBounds)
            })
            .with_fn(Symbol::nth_symbol(), 1, |vm, this| {
                let index = vm[Register(0)].take();
                this.nth(&index, vm.as_ref())
            })
            .with_fn(Symbol::len_symbol(), 0, |_vm, this| {
                let contents = this.0.lock();
                Value::try_from(contents.len())
            }),
    )
});

impl Trace for Map {
    const MAY_CONTAIN_REFERENCES: bool = true;

    fn trace(&self, tracer: &mut refuse::Tracer) {
        for field in &*self.0.lock() {
            field.key.value.trace(tracer);
            field.value.trace(tracer);
        }
    }
}

impl Default for Map {
    fn default() -> Self {
        Self::new()
    }
}

impl Map {
    /// Returns an empty map.
    #[must_use]
    pub const fn new() -> Self {
        Self(Mutex::new(Vec::new()))
    }

    /// Creates a map from an iterator of key-value pairs.
    pub fn from_iterator(
        vm: &mut VmContext<'_, '_>,
        iter: impl IntoIterator<Item = (Value, Value)>,
    ) -> Self {
        let mut fields: Vec<_> = iter
            .into_iter()
            .map(|(key, value)| Field {
                key: MapKey::new(vm, key),
                value,
            })
            .collect();
        fields.sort_unstable_by(|a, b| a.key.hash.cmp(&b.key.hash));
        // TODO need to de-dup keys. Maybe there's a sorting algorithm that
        // supports deduping we can use here?
        Self(Mutex::new(fields))
    }

    /// Returns the value contained for `key`, or `None` if this map does not
    /// contain key.
    ///
    /// # Errors
    ///
    /// This function does not directly return any errors, but comparing values
    /// can result in runtime errors.
    pub fn get(&self, vm: &mut VmContext<'_, '_>, key: &Value) -> Result<Option<Value>, Fault> {
        let hash = key.hash(vm);
        let contents = self.0.lock();
        for field in &*contents {
            match hash.cmp(&field.key.hash) {
                Ordering::Less => continue,
                Ordering::Equal => {
                    if key.equals(ContextOrGuard::Context(vm), &field.key.value)? {
                        return Ok(Some(field.value));
                    }
                }
                Ordering::Greater => break,
            }
        }

        Ok(None)
    }

    /// Returns the value contained at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`Fault::OutOfBounds`] if `index` is out of bounds of this
    /// collection.
    pub fn nth(&self, index: &Value, guard: &CollectionGuard) -> Result<Value, Fault> {
        let Some(index) = index.as_usize() else {
            return Err(Fault::OutOfBounds);
        };
        let contents = self.0.lock();
        contents
            .get(index)
            .map(|field| Value::dynamic(List::from_iter([field.key.value, field.value]), guard))
            .ok_or(Fault::OutOfBounds)
    }

    /// Inserts a key-value pair into this map.
    ///
    /// If an existing value is stored for `key`, it will be returned.
    pub fn insert(
        &self,
        vm: &mut VmContext<'_, '_>,
        key: Value,
        value: Value,
    ) -> Result<Option<Value>, Fault> {
        let key = MapKey::new(vm, key);
        let mut contents = self.0.lock();
        let mut insert_at = contents.len();
        for (index, field) in contents.iter_mut().enumerate() {
            match key.hash.cmp(&field.key.hash) {
                Ordering::Less => continue,
                Ordering::Equal => {
                    if key
                        .value
                        .equals(ContextOrGuard::Context(vm), &field.key.value)?
                    {
                        return Ok(Some(std::mem::replace(&mut field.value, value)));
                    }
                }
                Ordering::Greater => {
                    insert_at = index;
                    break;
                }
            }
        }

        contents.try_reserve(1).map_err(|_| Fault::OutOfMemory)?;
        contents.insert(insert_at, Field { key, value });

        Ok(None)
    }

    /// Returns this map as a vec of fields.
    pub fn to_vec(&self) -> Vec<Field> {
        self.0.lock().clone()
    }
}

impl CustomType for Map {
    fn muse_type(&self) -> &TypeRef {
        &MAP_TYPE
    }
}

/// A key-value pair.
#[derive(Debug, Clone)]
pub struct Field {
    key: MapKey,
    value: Value,
}

impl Field {
    /// Returns the value of this field.
    #[must_use]
    pub const fn value(&self) -> Value {
        self.value
    }

    #[must_use]
    /// Returns the key of this field.
    pub const fn key(&self) -> Value {
        self.key.value
    }

    /// Splits this value into its key and value.
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
    fn new(vm: &mut VmContext<'_, '_>, key: Value) -> Self {
        Self {
            hash: key.hash(vm),
            value: key,
        }
    }
}
