use parking_lot::Mutex;
use refuse::Trace;

use crate::symbol::Symbol;
use crate::value::{
    CustomType, Dynamic, Rooted, RustType, StaticRustFunctionTable, TypeRef, Value,
};
use crate::vm::{Fault, Register, VmContext};

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
    .with_invoke(|_| {
        |this, vm, name, arity| {
            static FUNCTIONS: StaticRustFunctionTable<List> =
                StaticRustFunctionTable::new(|table| {
                    table
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
                                this.get(&key)
                            },
                        )
                });
            FUNCTIONS.invoke(vm, name, arity, &this)
        }
    })
});

#[derive(Debug)]
pub struct List(Mutex<Vec<Value>>);

impl List {
    #[must_use]
    pub const fn new() -> Self {
        Self(Mutex::new(Vec::new()))
    }

    pub fn get(&self, index: &Value) -> Result<Value, Fault> {
        let Some(index) = index.as_usize() else {
            return Err(Fault::OutOfBounds);
        };
        let contents = self.0.lock();

        contents.get(index).copied().ok_or(Fault::OutOfBounds)
    }

    pub fn insert(&self, index: &Value, value: Value) -> Result<(), Fault> {
        let Some(index) = index.as_usize() else {
            return Err(Fault::OutOfBounds);
        };
        let mut contents = self.0.lock();
        contents.insert(index, value);
        Ok(())
    }

    pub fn push(&self, value: Value) -> Result<(), Fault> {
        let mut contents = self.0.lock();
        contents.push(value);
        Ok(())
    }

    pub fn set(&self, index: &Value, value: Value) -> Result<Option<Value>, Fault> {
        let Some(index) = index.as_usize() else {
            return Err(Fault::OutOfBounds);
        };
        let mut contents = self.0.lock();

        if let Some(entry) = contents.get_mut(index) {
            Ok(Some(std::mem::replace(entry, value)))
        } else {
            Err(Fault::OutOfBounds)
        }
    }

    pub fn to_vec(&self) -> Vec<Value> {
        self.0.lock().clone()
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
