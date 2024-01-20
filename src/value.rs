use std::any::Any;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ahash::AHasher;

use crate::symbol::Symbol;
use crate::{Fault, Vm};

#[derive(Clone, Debug)]
pub enum Value {
    Nil,
    Int(i64),
    Float(f64),
    Symbol(Symbol),
    Dynamic(Dynamic),
}

impl Value {
    pub const fn is_nil(&self) -> bool {
        matches!(self, Self::Nil)
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(value) => Some(*value),
            Value::Float(value) => Some(*value as i64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(value) => Some(*value as f64),
            Value::Float(value) => Some(*value),
            _ => None,
        }
    }

    pub fn call(&self, vm: &mut Vm) -> Result<Value, Fault> {
        match self {
            Value::Dynamic(dynamic) => dynamic.call(vm),
            _ => Err(Fault::NotAFunction),
        }
    }

    pub fn invoke(&self, vm: &mut Vm, name: &Symbol) -> Result<Value, Fault> {
        match (self, &**name) {
            (_, "add") => {
                let rhs = vm
                    .current_frame()
                    .first()
                    .ok_or(Fault::IncorrectNumberOfArguments)?
                    .clone();
                self.add(vm, rhs)
            }
            (Value::Dynamic(dynamic), _) => dynamic.0.invoke(vm, name),
            _ => Err(Fault::UnknownSymbol(name.clone())),
        }
    }

    pub fn add(&self, vm: &mut Vm, rhs: Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),

            (Value::Symbol(lhs), rhs) => {
                let rhs = rhs.to_string(vm)?;
                Ok(Value::Symbol(lhs + &rhs))
            }
            (lhs, Value::Symbol(rhs)) => {
                let lhs = lhs.to_string(vm)?;
                Ok(Value::Symbol(&lhs + &rhs))
            }

            (Value::Int(lhs), Value::Int(rhs)) => Ok(Self::Int(lhs.saturating_add(rhs))),

            (Value::Int(lhs), Value::Float(rhs)) => Ok(Value::Float(*lhs as f64 + rhs)),
            (Value::Float(lhs), Value::Int(rhs)) => Ok(Value::Float(lhs + rhs as f64)),
            (Value::Float(lhs), Value::Float(rhs)) => Ok(Value::Float(lhs + rhs)),

            (Value::Dynamic(lhs), rhs) => lhs.add(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.add_right(vm, lhs),
        }
    }

    pub fn sub(&self, vm: &mut Vm, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),

            (Value::Int(lhs), Value::Int(rhs)) => Ok(Self::Int(lhs.saturating_sub(*rhs))),

            (Value::Int(lhs), Value::Float(rhs)) => Ok(Value::Float(*lhs as f64 - rhs)),
            (Value::Float(lhs), Value::Int(rhs)) => Ok(Value::Float(lhs - *rhs as f64)),
            (Value::Float(lhs), Value::Float(rhs)) => Ok(Value::Float(lhs - rhs)),

            (Value::Dynamic(lhs), rhs) => lhs.sub(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.sub_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    pub fn mul(&self, vm: &mut Vm, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),

            (Value::Int(count), Value::Symbol(string))
            | (Value::Symbol(string), Value::Int(count)) => Ok(Value::Symbol(Symbol::from(
                string.repeat(usize::try_from(*count).map_err(|_| Fault::OutOfMemory)?),
            ))),

            (Value::Int(lhs), Value::Int(rhs)) => Ok(Self::Int(lhs.saturating_mul(*rhs))),

            (Value::Int(lhs), Value::Float(rhs)) => Ok(Value::Float(*lhs as f64 * rhs)),
            (Value::Float(lhs), Value::Int(rhs)) => Ok(Value::Float(lhs * *rhs as f64)),
            (Value::Float(lhs), Value::Float(rhs)) => Ok(Value::Float(lhs * rhs)),

            (Value::Dynamic(lhs), rhs) => lhs.mul(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.mul_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    pub fn div(&self, vm: &mut Vm, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),

            (Value::Symbol(_string), Value::Symbol(_separator)) => {
                todo!("split string using division")
            }

            (Value::Int(lhs), Value::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Value::Float(*lhs as f64 / *rhs as f64))
                } else {
                    Err(Fault::DivideByZero)
                }
            }

            (Value::Int(lhs), Value::Float(rhs)) => Ok(Value::Float(*lhs as f64 / rhs)),
            (Value::Float(lhs), Value::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Value::Float(lhs / *rhs as f64))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            (Value::Float(lhs), Value::Float(rhs)) => Ok(Value::Float(lhs / rhs)),

            (Value::Dynamic(lhs), rhs) => lhs.div(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.div_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    pub fn div_i(&self, vm: &mut Vm, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),

            (Value::Int(lhs), Value::Int(rhs)) => Ok(Self::Int(lhs.saturating_div(*rhs))),

            (Value::Int(lhs), Value::Float(rhs)) => Ok(Value::Int(*lhs / *rhs as i64)),
            (Value::Float(lhs), Value::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Value::Int(*lhs as i64 / *rhs))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            (Value::Float(lhs), Value::Float(rhs)) => Ok(Value::Int(*lhs as i64 / *rhs as i64)),

            (Value::Dynamic(lhs), rhs) => lhs.divi(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.divi_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    pub fn rem(&self, vm: &mut Vm, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),

            (Value::Int(lhs), Value::Int(rhs)) => Ok(Self::Int(lhs % rhs)),

            (Value::Int(lhs), Value::Float(rhs)) => Ok(Value::Int(*lhs % *rhs as i64)),
            (Value::Float(lhs), Value::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Value::Int(*lhs as i64 % *rhs))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            (Value::Float(lhs), Value::Float(rhs)) => Ok(Value::Int(*lhs as i64 % *rhs as i64)),

            (Value::Dynamic(lhs), rhs) => lhs.rem(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.rem_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    pub fn to_string(&self, vm: &mut Vm) -> Result<Symbol, Fault> {
        match self {
            Value::Nil => Ok(Symbol::empty()),
            Value::Int(value) => Ok(Symbol::from(value.to_string())),
            Value::Float(value) => Ok(Symbol::from(value.to_string())),
            Value::Symbol(value) => Ok(value.clone()),
            Value::Dynamic(value) => value.to_string(vm),
        }
    }

    pub fn hash(&self, vm: &mut Vm) -> Result<u64, Fault> {
        let mut hasher = AHasher::default();

        core::mem::discriminant(self).hash(&mut hasher);
        match self {
            Value::Nil => {}
            Value::Int(i) => i.hash(&mut hasher),
            Value::Float(f) => f.to_bits().hash(&mut hasher),
            Value::Symbol(s) => s.hash(&mut hasher),
            Value::Dynamic(d) => d.hash(vm).hash(&mut hasher),
        }
        Ok(hasher.finish())
    }

    pub fn eq(&self, vm: &mut Vm, other: &Self) -> bool {
        match (self, other) {
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
            (Self::Float(l0), Self::Float(r0)) => l0 == r0,
            (Self::Symbol(l0), Self::Symbol(r0)) => l0 == r0,
            (Self::Dynamic(l0), _) => l0.eq(vm, other),
            (_, Self::Dynamic(r0)) => r0.eq(vm, self),
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

#[derive(Clone)]
pub struct Dynamic(Arc<Box<dyn DynamicValue>>);

impl Dynamic {
    pub fn new<T>(value: T) -> Self
    where
        T: DynamicValue + 'static,
    {
        Self(Arc::new(Box::new(value)))
    }

    pub fn downcast_ref<T>(&self) -> Option<&T>
    where
        T: 'static,
    {
        let dynamic = &**self.0;
        dynamic.as_any().downcast_ref()
    }

    pub fn downcast_mut<T>(&mut self) -> Option<&mut T>
    where
        T: 'static,
    {
        if Arc::get_mut(&mut self.0).is_none() {
            *self = (**self.0).deep_clone();
        }

        let dynamic = &mut *Arc::get_mut(&mut self.0).expect("always 1 ref");
        dynamic.as_any_mut().downcast_mut()
    }

    pub fn ptr_eq(a: &Dynamic, b: &Dynamic) -> bool {
        Arc::ptr_eq(&a.0, &b.0)
    }

    pub fn call(&self, vm: &mut Vm) -> Result<Value, Fault> {
        self.0.call(vm)
    }

    pub fn add(&self, vm: &mut Vm, rhs: Value) -> Result<Value, Fault> {
        self.0.add(vm, rhs)
    }

    pub fn add_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        self.0.add_right(vm, lhs)
    }

    pub fn sub(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        self.0.sub(vm, rhs)
    }

    pub fn sub_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        self.0.sub_right(vm, lhs)
    }

    pub fn mul(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        self.0.mul(vm, rhs)
    }

    pub fn mul_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        self.0.mul_right(vm, lhs)
    }

    pub fn div(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        self.0.div(vm, rhs)
    }

    pub fn div_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        self.0.div_right(vm, lhs)
    }

    pub fn rem(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        self.0.rem(vm, rhs)
    }

    pub fn rem_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        self.0.rem_right(vm, lhs)
    }

    pub fn divi(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        self.0.div(vm, rhs)
    }

    pub fn divi_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        self.0.divi_right(vm, lhs)
    }

    pub fn hash(&self, vm: &mut Vm) -> u64 {
        self.0.hash(vm)
    }

    pub fn to_string(&self, vm: &mut Vm) -> Result<Symbol, Fault> {
        self.0.to_string(vm)
    }

    pub fn eq_right(&self, vm: &mut Vm, lhs: &Value) -> bool {
        match lhs {
            Value::Dynamic(dynamic) if Arc::ptr_eq(&self.0, &dynamic.0) => true,
            _ => self.0.eq_right(vm, lhs),
        }
    }

    pub fn eq(&self, vm: &mut Vm, rhs: &Value) -> bool {
        match rhs {
            Value::Dynamic(dynamic) if Arc::ptr_eq(&self.0, &dynamic.0) => true,
            _ => self.0.eq(vm, rhs),
        }
    }
}

impl Debug for Dynamic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self.0, f)
    }
}

pub trait CustomType: Debug {
    #[allow(unused_variables)]
    fn call(&self, vm: &mut Vm) -> Result<Value, Fault> {
        Err(Fault::NotAFunction)
    }

    #[allow(unused_variables)]
    fn hash(&self, vm: &mut Vm) -> u64 {
        let mut ahash = AHasher::default();
        (self as *const Self).hash(&mut ahash);
        ahash.finish()
    }

    #[allow(unused_variables)]
    fn eq(&self, vm: &mut Vm, rhs: &Value) -> bool {
        false
    }

    #[allow(unused_variables)]
    fn eq_right(&self, vm: &mut Vm, lhs: &Value) -> bool {
        false
    }

    #[allow(unused_variables)]
    fn invoke(&self, vm: &mut Vm, name: &Symbol) -> Result<Value, Fault> {
        Err(Fault::UnknownSymbol(name.clone()))
    }

    #[allow(unused_variables)]
    fn add(&self, vm: &mut Vm, rhs: Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn add_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn sub(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn sub_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn mul(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn mul_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn div(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn div_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn divi(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn divi_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn rem(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn rem_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn to_string(&self, vm: &mut Vm) -> Result<Symbol, Fault> {
        Err(Fault::UnsupportedOperation)
    }
}

#[derive(Clone)]
pub struct RustFunction<F>(F, Symbol);

impl<F> RustFunction<F> {
    pub fn new(name: impl Into<Symbol>, function: F) -> Self {
        Self(function, name.into())
    }
}

impl<F> Debug for RustFunction<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RustFunction").field(&self.1).finish()
    }
}

impl<F> CustomType for RustFunction<F>
where
    F: Clone + Fn(&mut Vm) -> Result<Value, Fault>,
{
    fn call(&self, vm: &mut Vm) -> Result<Value, Fault> {
        self.0(vm)
    }
}

pub trait DynamicValue: CustomType {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn deep_clone(&self) -> Dynamic;
}

impl<T> DynamicValue for T
where
    T: Clone + CustomType + 'static,
{
    fn deep_clone(&self) -> Dynamic {
        Dynamic::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[test]
fn dynamic() {
    impl CustomType for usize {}
    let mut dynamic = Dynamic::new(1_usize);
    assert_eq!(dynamic.downcast_ref::<usize>(), Some(&1));
    let dynamic2 = dynamic.clone();
    assert!(Dynamic::ptr_eq(&dynamic, &dynamic2));
    *dynamic.downcast_mut::<usize>().unwrap() = 2;
    assert!(!Dynamic::ptr_eq(&dynamic, &dynamic2));
    assert_eq!(dynamic2.downcast_ref::<usize>(), Some(&1));
}

#[test]
fn functions() {
    let func = Value::Dynamic(Dynamic::new(RustFunction::new("hello", |_vm: &mut Vm| {
        Ok(Value::Int(1))
    })));
    let mut runtime = Vm::default();
    let Value::Int(i) = func.call(&mut runtime).unwrap() else {
        unreachable!()
    };
    assert_eq!(i, 1);
}
