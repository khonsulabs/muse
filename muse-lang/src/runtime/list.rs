//! Types used for lists/tuples.

use parking_lot::Mutex;
use refuse::Trace;

use crate::runtime::symbol::Symbol;
use crate::runtime::value::{
    CustomType, Dynamic, Rooted, RustFunctionTable, RustType, TypeRef, Value,
};
use crate::vm::{Fault, Register, VmContext};

/// The type definition that the [`List`] type uses.
///
/// In general, developers will not need this. However, if you are building your
/// own `core` module, this type can be used to populate `$.core.List`.
pub static LIST_TYPE: RustType<List> = RustType::new("List", |t| {
    t.with_construct(|_| {
        |vm, arity| {
            let list = List::new();
            for reg_index in 0..arity.0 {
                let value = vm[Register(reg_index)].take();
                list.push(value)?;
            }
            Ok(Dynamic::new(list, vm))
        }
    })
    .with_function_table(
        RustFunctionTable::new()
            .with_fn(
                Symbol::len_symbol(),
                0,
                |_vm: &mut VmContext<'_, '_>, this: &Rooted<List>| {
                    Value::try_from(this.0.lock().len())
                },
            )
            .with_fn(Symbol::set_symbol(), 2, |vm, this| {
                let index = vm[Register(0)].take();
                let value = vm[Register(1)].take();
                this.set(&index, value)?;
                Ok(value)
            })
            .with_fn(
                [Symbol::nth_symbol(), Symbol::get_symbol()],
                1,
                |vm, this| {
                    let key = vm[Register(0)].take();
                    this.get_by_value(&key)
                },
            ),
    )
});

/// A list of [`Value`]s.
#[derive(Debug)]
pub struct List(Mutex<Vec<Value>>);

impl List {
    /// Returns an empty list.
    #[must_use]
    pub const fn new() -> Self {
        Self(Mutex::new(Vec::new()))
    }

    /// Returns the value at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`Fault::OutOfBounds`] if `index` cannot be converted to a
    /// `usize` or is out of bounds of this list.
    pub fn get_by_value(&self, index: &Value) -> Result<Value, Fault> {
        let Some(index) = index.as_usize() else {
            return Err(Fault::OutOfBounds);
        };
        self.get(index).ok_or(Fault::OutOfBounds)
    }

    pub fn get(&self, index: usize) -> Option<Value> {
        let contents = self.0.lock();

        contents.get(index).copied()
    }

    /// Inserts `value` at `index`.
    ///
    /// # Errors
    ///
    /// Returns [`Fault::OutOfBounds`] if `index` cannot be converted to a
    /// `usize` or is greater than the length of this list.
    pub fn insert(&self, index: &Value, value: Value) -> Result<(), Fault> {
        let mut contents = self.0.lock();
        contents.try_reserve(1).map_err(|_| Fault::OutOfMemory)?;
        match index.as_usize() {
            Some(index) if index <= contents.len() => {
                contents.insert(index, value);

                Ok(())
            }
            _ => Err(Fault::OutOfBounds),
        }
    }

    /// Pushes `value` to the end of the list.
    pub fn push(&self, value: Value) -> Result<(), Fault> {
        let mut contents = self.0.lock();
        contents.try_reserve(1).map_err(|_| Fault::OutOfMemory)?;
        contents.push(value);
        Ok(())
    }

    /// Replaces the value at `index` with `value`. Returns the previous value.
    ///
    /// # Errors
    ///
    /// Returns [`Fault::OutOfBounds`] if `index` cannot be converted to a
    /// `usize` or is out of bounds of this list.
    pub fn set(&self, index: &Value, value: Value) -> Result<Value, Fault> {
        let Some(index) = index.as_usize() else {
            return Err(Fault::OutOfBounds);
        };
        let mut contents = self.0.lock();

        if let Some(entry) = contents.get_mut(index) {
            Ok(std::mem::replace(entry, value))
        } else {
            Err(Fault::OutOfBounds)
        }
    }

    /// Converts this list into a Vec.
    pub fn to_vec(&self) -> Vec<Value> {
        self.0.lock().clone()
    }
}

impl Default for List {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Vec<Value>> for List {
    fn from(value: Vec<Value>) -> Self {
        Self(Mutex::new(value))
    }
}

impl FromIterator<Value> for List {
    fn from_iter<T: IntoIterator<Item = Value>>(iter: T) -> Self {
        Self::from(Vec::from_iter(iter))
    }
}

impl CustomType for List {
    fn muse_type(&self) -> &TypeRef {
        &LIST_TYPE
    }
}

impl Trace for List {
    const MAY_CONTAIN_REFERENCES: bool = true;

    fn trace(&self, tracer: &mut refuse::Tracer) {
        self.0.lock().trace(tracer);
    }
}
