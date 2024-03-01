use std::any::Any;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::task::{Context, Poll};

pub type ValueHasher = ahash::AHasher;

use kempt::Map;

use crate::string::MuseString;
use crate::symbol::{Symbol, SymbolList};
use crate::vm::{Arity, Fault, Vm};

#[derive(Default, Clone, Debug)]
pub enum Value {
    #[default]
    Nil,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Symbol(Symbol),
    Dynamic(AnyDynamic),
}

impl Value {
    pub fn dynamic<T>(value: T) -> Self
    where
        T: DynamicValue,
    {
        Self::Dynamic(AnyDynamic::new(value))
    }

    #[must_use]
    pub const fn is_nil(&self) -> bool {
        matches!(self, Self::Nil)
    }

    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(value) => Some(*value),
            Value::UInt(value) => i64::try_from(*value).ok(),
            #[allow(clippy::cast_possible_truncation)]
            Value::Float(value) => Some(*value as i64),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Int(value) => u64::try_from(*value).ok(),
            Value::UInt(value) => Some(*value),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Value::Float(value) => Some(*value as u64),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Value::Int(value) => u32::try_from(*value).ok(),
            Value::UInt(value) => u32::try_from(*value).ok(),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Value::Float(value) => Some(*value as u32),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Value::Int(value) => usize::try_from(*value).ok(),
            Value::UInt(value) => usize::try_from(*value).ok(),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Value::Float(value) => Some(*value as usize),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            #[allow(clippy::cast_precision_loss)]
            Value::Int(value) => Some(*value as f64),
            Value::Float(value) => Some(*value),
            _ => None,
        }
    }

    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        match self {
            Value::Int(value) => Some(*value),
            Value::UInt(value) => i64::try_from(*value).ok(),
            #[allow(clippy::cast_possible_truncation)]
            Value::Float(value) => Some(*value as i64),
            Value::Nil => Some(0),
            Value::Bool(bool) => Some(i64::from(*bool)),
            Value::Symbol(_) | Value::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        match self {
            Value::Int(value) => u64::try_from(*value).ok(),
            Value::UInt(value) => Some(*value),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Value::Float(value) => Some(*value as u64),
            Value::Nil => Some(0),
            Value::Bool(bool) => Some(u64::from(*bool)),
            Value::Symbol(_) | Value::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    #[must_use]
    pub fn to_u32(&self) -> Option<u32> {
        match self {
            Value::Int(value) => u32::try_from(*value).ok(),
            Value::UInt(value) => u32::try_from(*value).ok(),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Value::Float(value) => Some(*value as u32),
            Value::Nil => Some(0),
            Value::Bool(bool) => Some(u32::from(*bool)),
            Value::Symbol(_) | Value::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    #[must_use]
    pub fn to_usize(&self) -> Option<usize> {
        match self {
            Value::Int(value) => usize::try_from(*value).ok(),
            Value::UInt(value) => usize::try_from(*value).ok(),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Value::Float(value) => Some(*value as usize),
            Value::Nil => Some(0),
            Value::Bool(bool) => Some(usize::from(*bool)),
            Value::Symbol(_) | Value::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    #[must_use]
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            #[allow(clippy::cast_precision_loss)]
            Value::Int(value) => Some(*value as f64),
            #[allow(clippy::cast_precision_loss)]
            Value::UInt(value) => Some(*value as f64),
            Value::Float(value) => Some(*value),
            Value::Nil => Some(0.),
            Value::Bool(bool) => Some(f64::from(*bool)),
            Value::Symbol(_) | Value::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    #[must_use]
    pub fn as_symbol(&self) -> Option<&Symbol> {
        match self {
            Value::Symbol(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_any_dynamic(&self) -> Option<&AnyDynamic> {
        match self {
            Value::Dynamic(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_dynamic<T>(&self) -> Option<Dynamic<T>>
    where
        T: DynamicValue,
    {
        match self {
            Value::Dynamic(value) => value.as_type(),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_downcast_ref<T>(&self) -> Option<&T>
    where
        T: DynamicValue,
    {
        match self {
            Value::Dynamic(value) => value.downcast_ref(),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_downcast_mut<T>(&mut self) -> Option<&mut T>
    where
        T: DynamicValue,
    {
        match self {
            Value::Dynamic(value) => value.downcast_mut(),
            _ => None,
        }
    }

    pub fn truthy(&self, vm: &mut Vm) -> bool {
        match self {
            Value::Nil => false,
            Value::Bool(value) => *value,
            Value::Int(value) => value != &0,
            Value::UInt(value) => value != &0,
            Value::Float(value) => value.abs() >= f64::EPSILON,
            Value::Symbol(sym) => !sym.is_empty(),
            Value::Dynamic(value) => value.truthy(vm),
        }
    }

    pub fn call(&self, vm: &mut Vm, arity: impl Into<Arity>) -> Result<Value, Fault> {
        match self {
            Value::Dynamic(dynamic) => dynamic.call(vm, arity),
            Value::Symbol(name) => vm.resolve(name).and_then(|named| named.call(vm, arity)),
            Value::Nil => vm.recurse_current_function(arity.into()),
            _ => Err(Fault::NotAFunction),
        }
    }

    pub fn invoke(
        &self,
        vm: &mut Vm,
        name: &Symbol,
        arity: impl Into<Arity>,
    ) -> Result<Value, Fault> {
        match (self, &**name) {
            (_, "add") => {
                let rhs = vm
                    .current_frame()
                    .first()
                    .ok_or(Fault::IncorrectNumberOfArguments)?
                    .clone();
                self.add(vm, &rhs)
            }
            (Value::Dynamic(dynamic), _) => dynamic.invoke(vm, name, arity.into()),
            (Value::Nil, _) => Err(Fault::OperationOnNil),
            _ => Err(Fault::UnknownSymbol),
        }
    }

    pub fn add(&self, vm: &mut Vm, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),
            (Value::Bool(lhs), Value::Bool(rhs)) => Ok(Value::Bool(*lhs || *rhs)),
            (Value::Bool(_), _) | (_, Value::Bool(_)) => Err(Fault::UnsupportedOperation),

            (Value::Symbol(lhs), rhs) => {
                rhs.map_str(vm, |_vm, rhs| Value::Symbol(Symbol::from(lhs + rhs)))
            }
            (lhs, Value::Symbol(rhs)) => {
                lhs.map_str(vm, |_vm, lhs| Value::Symbol(Symbol::from(lhs + rhs)))
            }

            (Value::Int(lhs), Value::Int(rhs)) => Ok(Self::Int(lhs.saturating_add(*rhs))),
            (Value::Int(lhs), Value::UInt(rhs)) => Ok(Self::Int(lhs.saturating_add_unsigned(*rhs))),
            (Value::UInt(lhs), Value::UInt(rhs)) => Ok(Self::UInt(lhs.saturating_add(*rhs))),
            (Value::UInt(lhs), Value::Int(rhs)) => Ok(Self::UInt(lhs.saturating_add_signed(*rhs))),

            #[allow(clippy::cast_precision_loss)]
            (Value::Int(lhs), Value::Float(rhs)) => Ok(Value::Float(*lhs as f64 + rhs)),
            #[allow(clippy::cast_precision_loss)]
            (Value::Float(lhs), Value::Int(rhs)) => Ok(Value::Float(lhs + *rhs as f64)),
            #[allow(clippy::cast_precision_loss)]
            (Value::UInt(lhs), Value::Float(rhs)) => Ok(Value::Float(*lhs as f64 + rhs)),
            #[allow(clippy::cast_precision_loss)]
            (Value::Float(lhs), Value::UInt(rhs)) => Ok(Value::Float(lhs + *rhs as f64)),
            (Value::Float(lhs), Value::Float(rhs)) => Ok(Value::Float(lhs + rhs)),

            (Value::Dynamic(lhs), rhs) => lhs.add(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.add_right(vm, lhs),
        }
    }

    pub fn sub(&self, vm: &mut Vm, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),

            (Value::Int(lhs), Value::Int(rhs)) => Ok(Self::Int(lhs.saturating_sub(*rhs))),
            (Value::Int(lhs), Value::UInt(rhs)) => Ok(Self::Int(lhs.saturating_sub_unsigned(*rhs))),
            (Value::UInt(lhs), Value::UInt(rhs)) => Ok(Self::UInt(lhs.saturating_sub(*rhs))),
            (Value::UInt(lhs), Value::Int(rhs)) => {
                Ok(Self::UInt(lhs.saturating_add_signed(rhs.saturating_neg())))
            }

            #[allow(clippy::cast_precision_loss)]
            (Value::Int(lhs), Value::Float(rhs)) => Ok(Value::Float(*lhs as f64 - rhs)),
            #[allow(clippy::cast_precision_loss)]
            (Value::Float(lhs), Value::Int(rhs)) => Ok(Value::Float(lhs - *rhs as f64)),
            #[allow(clippy::cast_precision_loss)]
            (Value::UInt(lhs), Value::Float(rhs)) => Ok(Value::Float(*lhs as f64 - rhs)),
            #[allow(clippy::cast_precision_loss)]
            (Value::Float(lhs), Value::UInt(rhs)) => Ok(Value::Float(lhs - *rhs as f64)),
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
            (Value::Int(lhs), Value::UInt(rhs)) => Ok(Self::Int(
                lhs.saturating_mul(i64::try_from(*rhs).unwrap_or(i64::MAX)),
            )),
            (Value::UInt(lhs), Value::Int(rhs)) => {
                Ok(Self::UInt(if let Ok(rhs) = u64::try_from(*rhs) {
                    lhs.saturating_mul(rhs)
                } else {
                    0
                }))
            }
            (Value::UInt(lhs), Value::UInt(rhs)) => Ok(Self::UInt(lhs.saturating_mul(*rhs))),

            #[allow(clippy::cast_precision_loss)]
            (Value::Int(lhs), Value::Float(rhs)) => Ok(Value::Float(*lhs as f64 * rhs)),
            #[allow(clippy::cast_precision_loss)]
            (Value::Float(lhs), Value::Int(rhs)) => Ok(Value::Float(lhs * *rhs as f64)),
            (Value::Float(lhs), Value::Float(rhs)) => Ok(Value::Float(lhs * rhs)),

            (Value::Dynamic(lhs), rhs) => lhs.mul(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.mul_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    pub fn pow(&self, vm: &mut Vm, exp: &Self) -> Result<Value, Fault> {
        match (self, exp) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),

            (Value::Int(lhs), Value::Int(rhs)) => Ok(if rhs.is_negative() {
                #[allow(clippy::cast_precision_loss)]
                Self::Float(powf64_i64(*lhs as f64, *rhs))
            } else {
                Self::Int(lhs.saturating_pow(u32::try_from(*rhs).unwrap_or(u32::MAX)))
            }),
            (Value::Int(lhs), Value::UInt(rhs)) => Ok(Self::Int(
                lhs.saturating_pow(u32::try_from(*rhs).unwrap_or(u32::MAX)),
            )),
            (Value::UInt(lhs), Value::Int(rhs)) => Ok(if rhs.is_negative() {
                #[allow(clippy::cast_precision_loss)]
                Self::Float(powf64_i64(*lhs as f64, *rhs))
            } else {
                Self::UInt(lhs.saturating_pow(u32::try_from(*rhs).unwrap_or(u32::MAX)))
            }),
            (Value::UInt(lhs), Value::UInt(rhs)) => Ok(Self::UInt(
                lhs.saturating_pow(u32::try_from(*rhs).unwrap_or(u32::MAX)),
            )),

            #[allow(clippy::cast_precision_loss)]
            (Value::Int(lhs), Value::Float(rhs)) => Ok(Value::Float((*lhs as f64).powf(*rhs))),
            #[allow(clippy::cast_precision_loss)]
            (Value::Float(lhs), Value::Int(rhs)) => Ok(Value::Float(powf64_i64(*lhs, *rhs))),
            #[allow(clippy::cast_precision_loss)]
            (Value::UInt(lhs), Value::Float(rhs)) => Ok(Value::Float((*lhs as f64).powf(*rhs))),
            #[allow(clippy::cast_precision_loss)]
            (Value::Float(lhs), Value::UInt(rhs)) => Ok(Value::Float(powf64_u64(*lhs, *rhs))),
            (Value::Float(lhs), Value::Float(rhs)) => Ok(Value::Float(lhs * rhs)),

            (Value::Dynamic(lhs), rhs) => lhs.mul(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.mul_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    pub fn div(&self, vm: &mut Vm, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),

            #[allow(clippy::cast_precision_loss)]
            (Value::Int(lhs), Value::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Value::Float(*lhs as f64 / *rhs as f64))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_precision_loss)]
            (Value::UInt(lhs), Value::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Value::Float(*lhs as f64 / *rhs as f64))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_precision_loss)]
            (Value::Int(lhs), Value::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Value::Float(*lhs as f64 / *rhs as f64))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_precision_loss)]
            (Value::UInt(lhs), Value::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Value::Float(*lhs as f64 / *rhs as f64))
                } else {
                    Err(Fault::DivideByZero)
                }
            }

            #[allow(clippy::cast_precision_loss)]
            (Value::Int(lhs), Value::Float(rhs)) => Ok(Value::Float(*lhs as f64 / rhs)),
            #[allow(clippy::cast_precision_loss)]
            (Value::UInt(lhs), Value::Float(rhs)) => Ok(Value::Float(*lhs as f64 / rhs)),
            #[allow(clippy::cast_precision_loss)]
            (Value::Float(lhs), Value::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Value::Float(lhs / *rhs as f64))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_precision_loss)]
            (Value::Float(lhs), Value::UInt(rhs)) => {
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

    pub fn idiv(&self, vm: &mut Vm, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),

            (Value::Int(lhs), Value::Int(rhs)) => Ok(Self::Int(lhs.saturating_div(*rhs))),
            (Value::Int(lhs), Value::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::Int(
                        lhs.saturating_div(i64::try_from(*rhs).unwrap_or(i64::MAX)),
                    ))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            (Value::UInt(lhs), Value::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::UInt(if let Ok(rhs) = u64::try_from(*rhs) {
                        lhs.saturating_div(rhs)
                    } else {
                        0
                    }))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            (Value::UInt(lhs), Value::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::UInt(lhs.saturating_div(*rhs)))
                } else {
                    Err(Fault::DivideByZero)
                }
            }

            #[allow(clippy::cast_possible_truncation)]
            (Value::Int(lhs), Value::Float(rhs)) => Ok(Value::Int(*lhs / *rhs as i64)),
            #[allow(clippy::cast_possible_truncation)]
            (Value::Float(lhs), Value::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Value::Int(*lhs as i64 / *rhs))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            (Value::UInt(lhs), Value::Float(rhs)) => {
                let rhs = *rhs as u64;
                if rhs == 0 {
                    Err(Fault::DivideByZero)
                } else {
                    Ok(Value::UInt(*lhs / rhs))
                }
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            (Value::Float(lhs), Value::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Value::UInt(*lhs as u64 / *rhs))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_possible_truncation)]
            (Value::Float(lhs), Value::Float(rhs)) => Ok(Value::Int(*lhs as i64 / *rhs as i64)),

            (Value::Dynamic(lhs), rhs) => lhs.idiv(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.idiv_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    pub fn rem(&self, vm: &mut Vm, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),

            (Value::Int(lhs), Value::Int(rhs)) => Ok(Self::Int(lhs % rhs)),
            (Value::Int(lhs), Value::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::Int(lhs % i64::try_from(*rhs).unwrap_or(i64::MAX)))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            (Value::UInt(lhs), Value::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::UInt(if let Ok(rhs) = u64::try_from(*rhs) {
                        lhs % rhs
                    } else {
                        0
                    }))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            (Value::UInt(lhs), Value::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::UInt(lhs % *rhs))
                } else {
                    Err(Fault::DivideByZero)
                }
            }

            #[allow(clippy::cast_possible_truncation)]
            (Value::Int(lhs), Value::Float(rhs)) => Ok(Value::Int(*lhs % *rhs as i64)),
            #[allow(clippy::cast_possible_truncation)]
            (Value::Float(lhs), Value::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Value::Int(*lhs as i64 % *rhs))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            (Value::UInt(lhs), Value::Float(rhs)) => {
                let rhs = *rhs as u64;
                if rhs == 0 {
                    Err(Fault::DivideByZero)
                } else {
                    Ok(Value::UInt(*lhs % rhs))
                }
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            (Value::Float(lhs), Value::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Value::UInt(*lhs as u64 % *rhs))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_possible_truncation)]
            (Value::Float(lhs), Value::Float(rhs)) => Ok(Value::Int(*lhs as i64 % *rhs as i64)),

            (Value::Dynamic(lhs), rhs) => lhs.rem(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.rem_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    pub fn not(&self, vm: &mut Vm) -> Result<Self, Fault> {
        Ok(Value::Bool(!self.truthy(vm)))
    }

    pub fn negate(&self, vm: &mut Vm) -> Result<Self, Fault> {
        match self {
            Value::Nil => Ok(Value::Nil),
            Value::Bool(bool) => Ok(Value::Bool(!bool)),
            Value::Int(value) => Ok(Value::Int(-*value)),
            Value::UInt(value) => Ok(if let Ok(value) = i64::try_from(*value) {
                Value::Int(value.saturating_neg())
            } else {
                Value::Int(i64::MIN)
            }),
            Value::Float(value) => Ok(Value::Float(-*value)),
            Value::Dynamic(value) => value.negate(vm),
            Value::Symbol(_) => Err(Fault::UnsupportedOperation),
        }
    }

    pub fn bitwise_not(&self, vm: &mut Vm) -> Result<Self, Fault> {
        match self {
            Value::Nil => Ok(Value::Nil),
            Value::Bool(bool) => Ok(Value::Bool(!bool)),
            Value::Int(value) => Ok(Value::Int(!*value)),
            Value::UInt(value) => Ok(Value::UInt(!*value)),
            Value::Dynamic(value) => value.bitwise_not(vm),
            Value::Float(_) | Value::Symbol(_) => Err(Fault::UnsupportedOperation),
        }
    }

    pub fn bitwise_and(&self, vm: &mut Vm, rhs: &Value) -> Result<Self, Fault> {
        match (self, rhs) {
            (Value::Dynamic(dymamic), other) | (other, Value::Dynamic(dymamic)) => {
                dymamic.bitwise_and(vm, other)
            }

            (Value::Int(lhs), Value::Int(rhs)) => Ok(Value::Int(lhs & rhs)),
            #[allow(clippy::cast_possible_wrap)]
            (Value::Int(lhs), Value::UInt(rhs)) => Ok(Value::Int(lhs & *rhs as i64)),
            (Value::Int(lhs), _) => {
                if let Some(rhs) = rhs.as_i64() {
                    Ok(Value::Int(lhs & rhs))
                } else {
                    Err(Fault::UnsupportedOperation)
                }
            }

            (Value::UInt(lhs), Value::UInt(rhs)) => Ok(Value::UInt(lhs & rhs)),
            #[allow(clippy::cast_sign_loss)]
            (Value::UInt(lhs), Value::Int(rhs)) => Ok(Value::UInt(lhs & *rhs as u64)),
            (Value::UInt(lhs), _) => {
                if let Some(rhs) = rhs.to_u64() {
                    Ok(Value::UInt(lhs & rhs))
                } else {
                    Err(Fault::UnsupportedOperation)
                }
            }

            (Value::Float(lhs), _) if lhs.is_sign_negative() => {
                match (self.as_i64(), rhs.as_i64()) {
                    (Some(lhs), Some(rhs)) => Ok(Value::Int(lhs & rhs)),
                    // If either are none, we know the result is 0.
                    _ => Ok(Value::Int(0)),
                }
            }

            _ => match (self.to_u64(), rhs.to_u64()) {
                (Some(lhs), Some(rhs)) => Ok(Value::UInt(lhs & rhs)),
                // If either are none, we know the result is 0.
                _ => Ok(Value::UInt(0)),
            },
        }
    }

    pub fn bitwise_or(&self, vm: &mut Vm, rhs: &Value) -> Result<Self, Fault> {
        match (self, rhs) {
            (Value::Dynamic(dymamic), other) | (other, Value::Dynamic(dymamic)) => {
                dymamic.bitwise_or(vm, other)
            }

            (Value::Int(lhs), Value::Int(rhs)) => Ok(Value::Int(lhs | rhs)),
            #[allow(clippy::cast_possible_wrap)]
            (Value::Int(lhs), Value::UInt(rhs)) => Ok(Value::Int(lhs | *rhs as i64)),
            (Value::Int(lhs), _) => {
                if let Some(rhs) = rhs.to_i64() {
                    Ok(Value::Int(lhs | rhs))
                } else {
                    Err(Fault::UnsupportedOperation)
                }
            }

            (Value::UInt(lhs), Value::UInt(rhs)) => Ok(Value::UInt(lhs | rhs)),
            #[allow(clippy::cast_sign_loss)]
            (Value::UInt(lhs), Value::Int(rhs)) => Ok(Value::UInt(lhs | *rhs as u64)),
            (Value::UInt(lhs), _) => {
                if let Some(rhs) = rhs.to_u64() {
                    Ok(Value::UInt(lhs | rhs))
                } else {
                    Err(Fault::UnsupportedOperation)
                }
            }

            (Value::Float(lhs), _) if lhs.is_sign_negative() => {
                match (self.to_i64(), rhs.to_i64()) {
                    (Some(lhs), Some(rhs)) => Ok(Value::Int(lhs | rhs)),
                    (Some(result), None) | (None, Some(result)) => Ok(Value::Int(result)),
                    (None, None) => Ok(Value::Int(0)),
                }
            }

            _ => match (self.to_u64(), rhs.to_u64()) {
                (Some(lhs), Some(rhs)) => Ok(Value::UInt(lhs | rhs)),
                (Some(result), None) | (None, Some(result)) => Ok(Value::UInt(result)),
                (None, None) => Ok(Value::UInt(0)),
            },
        }
    }

    pub fn bitwise_xor(&self, vm: &mut Vm, rhs: &Value) -> Result<Self, Fault> {
        match (self, rhs) {
            (Value::Dynamic(dymamic), other) | (other, Value::Dynamic(dymamic)) => {
                dymamic.bitwise_xor(vm, other)
            }

            (Value::Int(lhs), Value::Int(rhs)) => Ok(Value::Int(lhs ^ rhs)),
            #[allow(clippy::cast_possible_wrap)]
            (Value::Int(lhs), Value::UInt(rhs)) => Ok(Value::Int(lhs ^ *rhs as i64)),
            (Value::Int(lhs), _) => {
                if let Some(rhs) = rhs.to_i64() {
                    Ok(Value::Int(lhs ^ rhs))
                } else {
                    Err(Fault::UnsupportedOperation)
                }
            }

            (Value::UInt(lhs), Value::UInt(rhs)) => Ok(Value::UInt(lhs ^ rhs)),
            #[allow(clippy::cast_sign_loss)]
            (Value::UInt(lhs), Value::Int(rhs)) => Ok(Value::UInt(lhs ^ *rhs as u64)),
            (Value::UInt(lhs), _) => {
                if let Some(rhs) = rhs.to_u64() {
                    Ok(Value::UInt(lhs ^ rhs))
                } else {
                    Err(Fault::UnsupportedOperation)
                }
            }

            (Value::Float(lhs), _) if lhs.is_sign_negative() => {
                match (self.to_i64(), rhs.to_i64()) {
                    (Some(lhs), Some(rhs)) => Ok(Value::Int(lhs ^ rhs)),
                    (Some(result), None) | (None, Some(result)) => Ok(Value::Int(result)),
                    (None, None) => Ok(Value::Int(0)),
                }
            }

            _ => match (self.to_u64(), rhs.to_u64()) {
                (Some(lhs), Some(rhs)) => Ok(Value::UInt(lhs ^ rhs)),
                (Some(result), None) | (None, Some(result)) => Ok(Value::UInt(result)),
                (None, None) => Ok(Value::UInt(0)),
            },
        }
    }

    pub fn shift_left(&self, vm: &mut Vm, rhs: &Value) -> Result<Self, Fault> {
        match self {
            Value::Dynamic(dymamic) => dymamic.shift_left(vm, rhs),

            Value::Int(lhs) => Ok(Value::Int(
                lhs.checked_shl(rhs.to_u32().ok_or(Fault::UnsupportedOperation)?)
                    .unwrap_or_default(),
            )),
            Value::UInt(lhs) => Ok(Value::UInt(
                lhs.checked_shl(rhs.to_u32().ok_or(Fault::UnsupportedOperation)?)
                    .unwrap_or_default(),
            )),

            #[allow(clippy::cast_possible_truncation)]
            Value::Float(lhs) if lhs.is_sign_negative() => Ok(Value::Int(
                (*lhs as i64)
                    .checked_shl(rhs.to_u32().ok_or(Fault::UnsupportedOperation)?)
                    .unwrap_or_default(),
            )),

            _ => match (self.to_u64(), rhs.to_u32()) {
                (Some(lhs), Some(rhs)) => Ok(Value::UInt(lhs.checked_shl(rhs).unwrap_or_default())),
                _ => Err(Fault::UnsupportedOperation),
            },
        }
    }

    pub fn shift_right(&self, vm: &mut Vm, rhs: &Value) -> Result<Self, Fault> {
        match self {
            Value::Dynamic(dymamic) => dymamic.shift_right(vm, rhs),

            Value::Int(lhs) => Ok(Value::Int(
                lhs.checked_shr(rhs.to_u32().ok_or(Fault::UnsupportedOperation)?)
                    .unwrap_or_default(),
            )),
            Value::UInt(lhs) => Ok(Value::UInt(
                lhs.checked_shr(rhs.to_u32().ok_or(Fault::UnsupportedOperation)?)
                    .unwrap_or_default(),
            )),

            #[allow(clippy::cast_possible_truncation)]
            Value::Float(lhs) if lhs.is_sign_negative() => Ok(Value::Int(
                (*lhs as i64)
                    .checked_shr(rhs.to_u32().ok_or(Fault::UnsupportedOperation)?)
                    .unwrap_or_default(),
            )),

            _ => match (self.to_u64(), rhs.to_u32()) {
                (Some(lhs), Some(rhs)) => Ok(Value::UInt(lhs.checked_shr(rhs).unwrap_or_default())),
                _ => Err(Fault::UnsupportedOperation),
            },
        }
    }

    pub fn to_string(&self, vm: &mut Vm) -> Result<Symbol, Fault> {
        match self {
            Value::Nil => Ok(Symbol::empty().clone()),
            Value::Bool(bool) => Ok(Symbol::from(*bool)),
            Value::Int(value) => Ok(Symbol::from(value.to_string())),
            Value::UInt(value) => Ok(Symbol::from(value.to_string())),
            Value::Float(value) => Ok(Symbol::from(value.to_string())),
            Value::Symbol(value) => Ok(value.clone()),
            Value::Dynamic(value) => value.to_string(vm),
        }
    }

    pub fn map_str<R>(
        &self,
        vm: &mut Vm,
        map: impl FnOnce(&mut Vm, &str) -> R,
    ) -> Result<R, Fault> {
        if let Value::Dynamic(dynamic) = self {
            if let Some(str) = dynamic.downcast_ref::<MuseString>() {
                return Ok(map(vm, &str.lock()));
            }
        }

        self.to_string(vm).map(|string| map(vm, &string))
    }

    pub fn hash_into(&self, vm: &mut Vm, hasher: &mut ValueHasher) {
        core::mem::discriminant(self).hash(hasher);
        match self {
            Value::Nil => {}
            Value::Bool(b) => b.hash(hasher),
            Value::Int(i) => i.hash(hasher),
            Value::UInt(i) => i.hash(hasher),
            Value::Float(f) => f.to_bits().hash(hasher),
            Value::Symbol(s) => s.hash(hasher),
            Value::Dynamic(d) => d.hash(vm, hasher),
        }
    }

    pub fn hash(&self, vm: &mut Vm) -> u64 {
        let mut hasher = ValueHasher::default();

        core::mem::discriminant(self).hash(&mut hasher);
        match self {
            Value::Nil => {}
            Value::Bool(b) => b.hash(&mut hasher),
            Value::Int(i) => i.hash(&mut hasher),
            Value::UInt(i) => i.hash(&mut hasher),
            Value::Float(f) => f.to_bits().hash(&mut hasher),
            Value::Symbol(s) => s.hash(&mut hasher),
            Value::Dynamic(d) => d.hash(vm, &mut hasher),
        }

        hasher.finish()
    }

    pub fn equals(&self, vm: Option<&mut Vm>, other: &Self) -> Result<bool, Fault> {
        match (self, other) {
            (Self::Nil, Self::Nil) => Ok(true),
            (Self::Nil, _) | (_, Self::Nil) => Ok(false),

            (Self::Bool(l0), Self::Bool(r0)) => Ok(l0 == r0),
            (Self::Bool(b), Self::Int(i)) | (Self::Int(i), Self::Bool(b)) => {
                Ok(&i64::from(*b) == i)
            }
            (Self::Bool(b), Self::Float(f)) | (Self::Float(f), Self::Bool(b)) => {
                Ok((f64::from(u8::from(*b)) - f).abs() < f64::EPSILON)
            }

            (Self::Int(l0), Self::Int(r0)) => Ok(l0 == r0),
            (Self::Int(signed), Self::UInt(unsigned))
            | (Self::UInt(unsigned), Self::Int(signed)) => {
                Ok(u64::try_from(*signed).map_or(false, |signed| &signed == unsigned))
            }
            (Self::UInt(l0), Self::UInt(r0)) => Ok(l0 == r0),
            (Self::Float(l0), Self::Float(r0)) => Ok((l0 - r0).abs() < f64::EPSILON),
            #[allow(clippy::cast_precision_loss)]
            (Self::Int(i), Self::Float(f)) | (Self::Float(f), Self::Int(i)) => {
                Ok((*i as f64 - f).abs() < f64::EPSILON)
            }
            #[allow(clippy::cast_precision_loss)]
            (Self::UInt(i), Self::Float(f)) | (Self::Float(f), Self::UInt(i)) => {
                Ok((*i as f64 - f).abs() < f64::EPSILON)
            }

            (Self::Symbol(l0), Self::Symbol(r0)) => Ok(l0 == r0),
            (Self::Symbol(s), Self::Bool(b)) | (Self::Bool(b), Self::Symbol(s)) => {
                Ok(s == &Symbol::from(*b))
            }

            (Self::Dynamic(l0), _) => l0.eq(vm, other),
            (_, Self::Dynamic(r0)) => r0.eq(vm, self),

            _ => Ok(false),
        }
    }

    pub fn matches(&self, vm: &mut Vm, other: &Self) -> Result<bool, Fault> {
        match (self, other) {
            (Self::Dynamic(l0), _) => l0.matches(vm, other),
            (_, Self::Dynamic(r0)) => r0.matches(vm, self),
            _ => self.equals(Some(vm), other),
        }
    }

    pub fn total_cmp(&self, vm: &mut Vm, other: &Self) -> Result<Ordering, Fault> {
        match (self, other) {
            (Value::Nil, Value::Nil) => Ok(Ordering::Equal),

            (Value::Bool(l), Value::Bool(r)) => Ok(l.cmp(r)),

            (Self::Bool(l), Self::Int(r)) => {
                let l = i64::from(*l);
                Ok(l.cmp(r))
            }
            (Self::Int(l), Self::Bool(r)) => {
                let r = i64::from(*r);
                Ok(l.cmp(&r))
            }
            (Self::Bool(b), Self::Float(f)) => Ok(f64::from(*b).total_cmp(f)),
            (Self::Float(f), Self::Bool(b)) => Ok(f.total_cmp(&f64::from(*b))),

            (Value::Int(l), Value::Int(r)) => Ok(l.cmp(r)),
            (Value::Int(l), Value::UInt(r)) => Ok(if let Ok(l) = u64::try_from(*l) {
                l.cmp(r)
            } else {
                Ordering::Less
            }),
            (Value::UInt(l), Value::UInt(r)) => Ok(l.cmp(r)),
            (Value::Float(l), Value::Float(r)) => Ok(l.total_cmp(r)),

            #[allow(clippy::cast_precision_loss)]
            (Value::Int(l_int), Value::Float(r_float)) => Ok((*l_int as f64).total_cmp(r_float)),
            #[allow(clippy::cast_precision_loss)]
            (Value::Float(l_float), Value::Int(r_int)) => Ok(l_float.total_cmp(&(*r_int as f64))),

            (Value::Symbol(l), Value::Symbol(r)) => Ok(l.cmp(r)),

            (Self::Dynamic(l0), _) => l0.cmp(vm, other),
            (_, Self::Dynamic(r0)) => r0.cmp(vm, self).map(Ordering::reverse),

            (Value::Nil, _) => Ok(Ordering::Less),
            (_, Value::Nil) => Ok(Ordering::Greater),

            (Value::Bool(_), _) => Ok(Ordering::Less),
            (_, Value::Bool(_)) => Ok(Ordering::Greater),
            (Value::Int(_), _) => Ok(Ordering::Less),
            (_, Value::Int(_)) => Ok(Ordering::Greater),
            (Value::UInt(_), _) => Ok(Ordering::Less),
            (_, Value::UInt(_)) => Ok(Ordering::Greater),
            (Value::Float(_), _) => Ok(Ordering::Less),
            (_, Value::Float(_)) => Ok(Ordering::Greater),
        }
    }

    #[must_use]
    pub fn take(&mut self) -> Value {
        std::mem::take(self)
    }

    pub fn deep_clone(&self) -> Option<Value> {
        match self {
            Value::Nil => Some(Value::Nil),
            Value::Bool(value) => Some(Value::Bool(*value)),
            Value::Int(value) => Some(Value::Int(*value)),
            Value::UInt(value) => Some(Value::UInt(*value)),
            Value::Float(value) => Some(Value::Float(*value)),
            Value::Symbol(value) => Some(Value::Symbol(value.clone())),
            Value::Dynamic(value) => value.deep_clone().map(Value::Dynamic),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.equals(None, other).unwrap_or(false)
    }
}

#[allow(clippy::cast_precision_loss)]
fn powf64_i64(base: f64, exp: i64) -> f64 {
    if let Ok(exp) = i32::try_from(exp) {
        base.powi(exp)
    } else {
        base.powf(exp as f64)
    }
}

#[allow(clippy::cast_precision_loss)]
fn powf64_u64(base: f64, exp: u64) -> f64 {
    if let Ok(exp) = i32::try_from(exp) {
        base.powi(exp)
    } else {
        base.powf(exp as f64)
    }
}

impl_from!(Value, f32, Float);
impl_from!(Value, f64, Float);
impl_from!(Value, i8, Int);
impl_from!(Value, i16, Int);
impl_from!(Value, i32, Int);
impl_from!(Value, i64, Int);
impl_from!(Value, u8, UInt);
impl_from!(Value, u16, UInt);
impl_from!(Value, u32, UInt);
impl_from!(Value, u64, UInt);
impl_from!(Value, bool, Bool);
impl_from!(Value, Symbol, Symbol);
impl_from!(Value, &'_ Symbol, Symbol);

macro_rules! impl_try_from {
    ($on:ty, $from:ty, $variant:ident) => {
        impl TryFrom<$from> for $on {
            type Error = Fault;

            fn try_from(value: $from) -> Result<Self, Self::Error> {
                Ok(Self::$variant(
                    value.try_into().map_err(|_| Fault::OutOfBounds)?,
                ))
            }
        }
    };
}

impl_try_from!(Value, u128, UInt);
impl_try_from!(Value, usize, UInt);
impl_try_from!(Value, isize, Int);
impl_try_from!(Value, i128, UInt);

#[derive(Clone)]
pub struct AnyDynamic(Arc<Box<dyn DynamicValue>>);

impl AnyDynamic {
    pub fn new<T>(value: T) -> Self
    where
        T: DynamicValue,
    {
        Self(Arc::new(Box::new(value)))
    }

    #[must_use]
    pub fn as_type<T>(&self) -> Option<Dynamic<T>>
    where
        T: DynamicValue,
    {
        self.downcast_ref::<T>().map(|_| Dynamic {
            dynamic: self.clone(),
            _t: PhantomData,
        })
    }

    pub fn try_into_type<T>(self) -> Result<Dynamic<T>, Self>
    where
        T: DynamicValue,
    {
        let Some(_) = self.downcast_ref::<T>() else {
            return Err(self);
        };

        Ok(Dynamic {
            dynamic: self.clone(),
            _t: PhantomData,
        })
    }

    #[must_use]
    pub fn downcast_ref<T>(&self) -> Option<&T>
    where
        T: DynamicValue,
    {
        let dynamic = &**self.0;
        dynamic.as_any().downcast_ref()
    }

    pub fn downcast_mut<T>(&mut self) -> Option<&mut T>
    where
        T: DynamicValue,
    {
        if Arc::get_mut(&mut self.0).is_none() {
            *self = self.deep_clone()?;
        }

        let dynamic = &mut *Arc::get_mut(&mut self.0).expect("always 1 ref");
        dynamic.as_any_mut().downcast_mut()
    }

    #[must_use]
    pub fn downgrade(&self) -> WeakAnyDynamic {
        WeakAnyDynamic(Arc::downgrade(&self.0))
    }

    #[must_use]
    pub fn ptr_eq(a: &AnyDynamic, b: &AnyDynamic) -> bool {
        Arc::ptr_eq(&a.0, &b.0)
    }

    #[must_use]
    pub fn deep_clone(&self) -> Option<AnyDynamic> {
        (self.0.muse_type().vtable.deep_clone)(self)
    }

    pub fn call(&self, vm: &mut Vm, arity: impl Into<Arity>) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.call)(self, vm, arity.into())
    }

    pub fn invoke(
        &self,
        vm: &mut Vm,
        symbol: &Symbol,
        arity: impl Into<Arity>,
    ) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.invoke)(self, vm, symbol, arity.into())
    }

    pub fn add(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.add)(self, vm, rhs)
    }

    pub fn add_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.add_right)(self, vm, lhs)
    }

    pub fn sub(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.sub)(self, vm, rhs)
    }

    pub fn sub_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.sub_right)(self, vm, lhs)
    }

    pub fn mul(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.mul)(self, vm, rhs)
    }

    pub fn mul_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.mul_right)(self, vm, lhs)
    }

    pub fn div(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.div)(self, vm, rhs)
    }

    pub fn div_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.div_right)(self, vm, lhs)
    }

    pub fn rem(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.rem)(self, vm, rhs)
    }

    pub fn rem_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.rem_right)(self, vm, lhs)
    }

    pub fn idiv(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.div)(self, vm, rhs)
    }

    pub fn idiv_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.idiv_right)(self, vm, lhs)
    }

    pub fn hash(&self, vm: &mut Vm, hasher: &mut ValueHasher) {
        (self.0.muse_type().vtable.hash)(self, vm, hasher);
    }

    pub fn bitwise_not(&self, vm: &mut Vm) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.bitwise_not)(self, vm)
    }

    pub fn bitwise_and(&self, vm: &mut Vm, other: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.bitwise_and)(self, vm, other)
    }

    pub fn bitwise_or(&self, vm: &mut Vm, other: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.bitwise_or)(self, vm, other)
    }

    pub fn bitwise_xor(&self, vm: &mut Vm, other: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.bitwise_xor)(self, vm, other)
    }

    pub fn shift_left(&self, vm: &mut Vm, amount: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.shift_left)(self, vm, amount)
    }

    pub fn shift_right(&self, vm: &mut Vm, amount: &Value) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.shift_right)(self, vm, amount)
    }

    pub fn negate(&self, vm: &mut Vm) -> Result<Value, Fault> {
        (self.0.muse_type().vtable.negate)(self, vm)
    }

    pub fn to_string(&self, vm: &mut Vm) -> Result<Symbol, Fault> {
        (self.0.muse_type().vtable.to_string)(self, vm)
    }

    pub fn truthy(&self, vm: &mut Vm) -> bool {
        (self.0.muse_type().vtable.truthy)(self, vm)
    }

    pub fn eq(&self, vm: Option<&mut Vm>, rhs: &Value) -> Result<bool, Fault> {
        match rhs {
            Value::Dynamic(dynamic) if Arc::ptr_eq(&self.0, &dynamic.0) => Ok(true),
            _ => (self.0.muse_type().vtable.eq)(self, vm, rhs),
        }
    }

    pub fn matches(&self, vm: &mut Vm, rhs: &Value) -> Result<bool, Fault> {
        match rhs {
            Value::Dynamic(dynamic) if Arc::ptr_eq(&self.0, &dynamic.0) => Ok(true),
            _ => (self.0.muse_type().vtable.matches)(self, vm, rhs),
        }
    }

    pub fn cmp(&self, vm: &mut Vm, rhs: &Value) -> Result<Ordering, Fault> {
        (self.0.muse_type().vtable.total_cmp)(self, vm, rhs)
    }
}

impl Debug for AnyDynamic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self.0, f)
    }
}

pub struct Dynamic<T>
where
    T: CustomType,
{
    dynamic: AnyDynamic,
    _t: PhantomData<T>,
}

impl<T> Dynamic<T>
where
    T: CustomType,
{
    #[must_use]
    pub fn new(value: T) -> Self {
        Self {
            dynamic: AnyDynamic::new(value),
            _t: PhantomData,
        }
    }

    #[must_use]
    pub fn as_any_dynamic(&self) -> &AnyDynamic {
        &self.dynamic
    }

    #[must_use]
    pub fn downgrade(&self) -> WeakDynamic<T> {
        WeakDynamic {
            weak: self.dynamic.downgrade(),
            _t: PhantomData,
        }
    }

    #[must_use]
    pub fn to_value(&self) -> Value {
        Value::Dynamic(self.dynamic.clone())
    }

    #[must_use]
    pub fn into_any_dynamic(self) -> AnyDynamic {
        self.dynamic
    }
}

impl<T> Debug for Dynamic<T>
where
    T: CustomType,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.dynamic, f)
    }
}

impl<T> Deref for Dynamic<T>
where
    T: CustomType,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.dynamic.downcast_ref().expect("type checked")
    }
}

impl<T> DerefMut for Dynamic<T>
where
    T: CustomType,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.dynamic.downcast_mut().expect("type checked")
    }
}

impl<T> Clone for Dynamic<T>
where
    T: CustomType,
{
    fn clone(&self) -> Self {
        Self {
            dynamic: self.dynamic.clone(),
            _t: PhantomData,
        }
    }
}

pub struct StaticType(OnceLock<TypeRef>, fn() -> Type);

impl StaticType {
    pub const fn new(init: fn() -> Type) -> Self {
        Self(OnceLock::new(), init)
    }
}

impl Deref for StaticType {
    type Target = TypeRef;

    fn deref(&self) -> &Self::Target {
        self.0.get_or_init(|| self.1().seal())
    }
}

pub struct RustType<T>(
    OnceLock<TypeRef>,
    &'static str,
    fn(TypedTypeBuilder<T>) -> TypedTypeBuilder<T>,
);

impl<T> RustType<T> {
    pub const fn new(
        name: &'static str,
        init: fn(TypedTypeBuilder<T>) -> TypedTypeBuilder<T>,
    ) -> Self {
        Self(OnceLock::new(), name, init)
    }
}

impl<T> Deref for RustType<T>
where
    T: CustomType,
{
    type Target = TypeRef;

    fn deref(&self) -> &Self::Target {
        self.0
            .get_or_init(|| self.2(TypedTypeBuilder::new(self.1)).seal())
    }
}

// #[derive(Debug)]
pub struct Type {
    pub name: Symbol,
    pub vtable: TypeVtable,
}

impl Type {
    pub fn new(name: impl Into<Symbol>) -> Self {
        Self {
            name: name.into(),
            vtable: TypeVtable::default(),
        }
    }

    #[must_use]
    pub fn with_construct<Func>(mut self, func: impl FnOnce(ConstructFn) -> Func) -> Self
    where
        Func: Fn(&mut Vm, Arity) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.construct = Box::new(func(self.vtable.construct));
        self
    }

    #[must_use]
    pub fn with_call<Func>(mut self, func: impl FnOnce(CallFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, Arity) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.call = Box::new(func(self.vtable.call));
        self
    }

    #[must_use]
    pub fn with_invoke<Func>(mut self, func: impl FnOnce(InvokeFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Symbol, Arity) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.invoke = Box::new(func(self.vtable.invoke));
        self
    }

    #[must_use]
    pub fn with_hash<Func>(mut self, func: impl FnOnce(HashFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &mut ValueHasher) + Send + Sync + 'static,
    {
        self.vtable.hash = Box::new(func(self.vtable.hash));
        self
    }

    #[must_use]
    pub fn with_bitwise_not<Func>(mut self, func: impl FnOnce(UnaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.bitwise_not = Box::new(func(self.vtable.bitwise_not));
        self
    }

    #[must_use]
    pub fn with_bitwise_and<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.bitwise_and = Box::new(func(self.vtable.bitwise_and));
        self
    }

    #[must_use]
    pub fn with_bitwise_or<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.bitwise_or = Box::new(func(self.vtable.bitwise_or));
        self
    }

    #[must_use]
    pub fn with_bitwise_xor<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.bitwise_xor = Box::new(func(self.vtable.bitwise_xor));
        self
    }

    #[must_use]
    pub fn with_shift_left<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.shift_left = Box::new(func(self.vtable.shift_left));
        self
    }

    #[must_use]
    pub fn with_shift_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.shift_right = Box::new(func(self.vtable.shift_right));
        self
    }

    #[must_use]
    pub fn with_negate<Func>(mut self, func: impl FnOnce(UnaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.negate = Box::new(func(self.vtable.negate));
        self
    }

    #[must_use]
    pub fn with_eq<Func>(mut self, func: impl FnOnce(EqFn) -> Func) -> Self
    where
        Func:
            Fn(&AnyDynamic, Option<&mut Vm>, &Value) -> Result<bool, Fault> + Send + Sync + 'static,
    {
        let func = func(self.vtable.eq);
        self.vtable.eq = Box::new(move |this, vm, rhs| func(this, vm, rhs));
        self
    }

    #[must_use]
    pub fn with_matches<Func>(mut self, func: impl FnOnce(MatchesFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<bool, Fault> + Send + Sync + 'static,
    {
        let func = func(self.vtable.matches);
        self.vtable.matches = Box::new(move |this, vm, rhs| func(this, vm, rhs));
        self
    }

    #[must_use]
    pub fn with_total_cmp<Func>(mut self, func: impl FnOnce(TotalCmpFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Ordering, Fault> + Send + Sync + 'static,
    {
        let func = func(self.vtable.total_cmp);
        self.vtable.total_cmp = Box::new(move |this, vm, rhs| func(this, vm, rhs));
        self
    }

    #[must_use]
    pub fn with_add<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.add = Box::new(func(self.vtable.add));
        self
    }

    #[must_use]
    pub fn with_add_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.add_right = Box::new(func(self.vtable.add_right));
        self
    }

    #[must_use]
    pub fn with_sub<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.sub = Box::new(func(self.vtable.sub));
        self
    }

    #[must_use]
    pub fn with_sub_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.sub_right = Box::new(func(self.vtable.sub_right));
        self
    }

    #[must_use]
    pub fn with_mul<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.mul = Box::new(func(self.vtable.mul));
        self
    }

    #[must_use]
    pub fn with_mul_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.mul_right = Box::new(func(self.vtable.mul_right));
        self
    }

    #[must_use]
    pub fn with_div<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.div = Box::new(func(self.vtable.div));
        self
    }

    #[must_use]
    pub fn with_div_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.div_right = Box::new(func(self.vtable.div_right));
        self
    }

    #[must_use]
    pub fn with_idiv<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.idiv = Box::new(func(self.vtable.idiv));
        self
    }

    #[must_use]
    pub fn with_idiv_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.idiv_right = Box::new(func(self.vtable.idiv_right));
        self
    }

    #[must_use]
    pub fn with_rem<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.rem = Box::new(func(self.vtable.rem));
        self
    }

    #[must_use]
    pub fn with_rem_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.rem_right = Box::new(func(self.vtable.rem_right));
        self
    }

    #[must_use]
    pub fn with_truthy<Func>(mut self, func: impl FnOnce(TruthyFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm) -> bool + Send + Sync + 'static,
    {
        self.vtable.truthy = Box::new(func(self.vtable.truthy));
        self
    }

    #[must_use]
    pub fn with_to_string<Func>(mut self, func: impl FnOnce(ToStringFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut Vm) -> Result<Symbol, Fault> + Send + Sync + 'static,
    {
        self.vtable.to_string = Box::new(func(self.vtable.to_string));
        self
    }

    #[must_use]
    pub fn with_deep_clone<Func>(mut self, func: impl FnOnce(DeepCloneFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic) -> Option<AnyDynamic> + Send + Sync + Send + Sync + 'static,
    {
        self.vtable.deep_clone = Box::new(func(self.vtable.deep_clone));
        self
    }

    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn with_fallback<Mapping>(mut self, mapping: Mapping) -> Self
    where
        Mapping: Fn(&AnyDynamic) -> Value + Send + Sync + Clone + 'static,
    {
        let mapping = Arc::new(mapping);
        // Entries that aren't mutable are not fallible or do not make sense to
        // have a fallback
        let TypeVtable {
            construct,
            mut call,
            hash,
            mut bitwise_not,
            mut bitwise_and,
            mut bitwise_or,
            mut bitwise_xor,
            mut shift_left,
            mut shift_right,
            mut negate,
            mut eq,
            mut matches,
            mut total_cmp,
            mut invoke,
            mut add,
            mut add_right,
            mut sub,
            mut sub_right,
            mut mul,
            mut mul_right,
            mut div,
            mut div_right,
            mut idiv,
            mut idiv_right,
            mut rem,
            mut rem_right,
            truthy,
            mut to_string,
            mut deep_clone,
        } = self.vtable;

        call = Box::new({
            let mapping = mapping.clone();
            move |this, vm, arity| match call(this, vm, arity) {
                Ok(value) => Ok(value),
                Err(Fault::NotAFunction | Fault::UnsupportedOperation) => {
                    mapping(this).call(vm, arity)
                }
                Err(other) => Err(other),
            }
        });

        bitwise_not = Box::new({
            let mapping = mapping.clone();
            move |this, vm| match bitwise_not(this, vm) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).bitwise_not(vm)
                }
                Err(other) => Err(other),
            }
        });

        bitwise_and = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match bitwise_and(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).bitwise_and(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        bitwise_or = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match bitwise_or(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).bitwise_or(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        bitwise_xor = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match bitwise_xor(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).bitwise_xor(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        shift_left = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match shift_left(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).shift_left(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        shift_right = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match shift_right(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).shift_right(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        negate = Box::new({
            let mapping = mapping.clone();
            move |this, vm| match negate(this, vm) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => mapping(this).negate(vm),
                Err(other) => Err(other),
            }
        });

        eq = Box::new({
            let mapping = mapping.clone();
            move |this, mut vm, rhs| match eq(this, vm.as_deref_mut(), rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).equals(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        matches = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match matches(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).matches(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        total_cmp = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match total_cmp(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).total_cmp(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        invoke = Box::new({
            let mapping = mapping.clone();
            move |this, vm, name, arity| match invoke(this, vm, name, arity) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).invoke(vm, name, arity)
                }
                Err(other) => Err(other),
            }
        });

        add = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match add(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).add(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        add_right = Box::new({
            let mapping = mapping.clone();
            move |this, vm, lhs| match add_right(this, vm, lhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    lhs.add(vm, &mapping(this))
                }
                Err(other) => Err(other),
            }
        });

        sub = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match sub(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).sub(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        sub_right = Box::new({
            let mapping = mapping.clone();
            move |this, vm, lhs| match sub_right(this, vm, lhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    lhs.sub(vm, &mapping(this))
                }
                Err(other) => Err(other),
            }
        });

        mul = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match mul(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).mul(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        mul_right = Box::new({
            let mapping = mapping.clone();
            move |this, vm, lhs| match mul_right(this, vm, lhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    lhs.mul(vm, &mapping(this))
                }
                Err(other) => Err(other),
            }
        });

        div = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match div(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).div(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        div_right = Box::new({
            let mapping = mapping.clone();
            move |this, vm, lhs| match div_right(this, vm, lhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    lhs.div(vm, &mapping(this))
                }
                Err(other) => Err(other),
            }
        });

        idiv = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match idiv(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).idiv(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        idiv_right = Box::new({
            let mapping = mapping.clone();
            move |this, vm, lhs| match idiv_right(this, vm, lhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    lhs.idiv(vm, &mapping(this))
                }
                Err(other) => Err(other),
            }
        });

        rem = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match rem(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).rem(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        rem_right = Box::new({
            let mapping = mapping.clone();
            move |this, vm, lhs| match rem_right(this, vm, lhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    lhs.rem(vm, &mapping(this))
                }
                Err(other) => Err(other),
            }
        });

        to_string = Box::new({
            let mapping = mapping.clone();
            move |this, vm| match to_string(this, vm) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this).to_string(vm)
                }
                Err(other) => Err(other),
            }
        });

        deep_clone = Box::new({
            let mapping = mapping.clone();
            move |this| match deep_clone(this) {
                Some(value) => Some(value),
                None => mapping(this)
                    .deep_clone()
                    .and_then(|value| value.as_any_dynamic().cloned()),
            }
        });

        self.vtable = TypeVtable {
            construct,
            call,
            invoke,
            hash,
            bitwise_not,
            bitwise_and,
            bitwise_or,
            bitwise_xor,
            shift_left,
            shift_right,
            negate,
            eq,
            matches,
            total_cmp,
            add,
            add_right,
            sub,
            sub_right,
            mul,
            mul_right,
            div,
            div_right,
            idiv,
            idiv_right,
            rem,
            rem_right,
            truthy,
            to_string,
            deep_clone,
        };
        self
    }

    #[must_use]
    pub fn seal(self) -> TypeRef {
        TypeRef::new(self)
    }
}

impl Debug for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Type")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

pub struct TypedTypeBuilder<T> {
    t: Type,
    _t: PhantomData<T>,
}

impl<T> TypedTypeBuilder<T>
where
    T: CustomType,
{
    fn new(name: &'static str) -> Self {
        Self {
            t: Type::new(name),
            _t: PhantomData,
        }
    }

    #[must_use]
    pub fn with_construct<Func>(mut self, func: impl FnOnce(ConstructFn) -> Func) -> Self
    where
        Func: Fn(&mut Vm, Arity) -> Result<T, Fault> + Send + Sync + 'static,
    {
        let func = func(self.t.vtable.construct);
        self.t.vtable.construct = Box::new(move |vm, arity| func(vm, arity).map(Value::dynamic));
        self
    }

    #[must_use]
    pub fn with_call<Func>(mut self, func: impl FnOnce(CallFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, Arity) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        let func = func(self.t.vtable.call);
        self.t.vtable.call = Box::new(move |this, vm, arity| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, arity)
        });
        self
    }

    #[must_use]
    pub fn with_invoke<Func>(mut self, func: impl FnOnce(InvokeFn) -> Func) -> Self
    where
        Func:
            Fn(Dynamic<T>, &mut Vm, &Symbol, Arity) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        let func = func(self.t.vtable.invoke);
        self.t.vtable.invoke = Box::new(move |this, vm, name, arity| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, name, arity)
        });
        self
    }

    #[must_use]
    pub fn with_hash<Func>(mut self, func: impl FnOnce(HashFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &mut ValueHasher) + Send + Sync + Send + Sync + 'static,
    {
        let func = func(self.t.vtable.hash);
        self.t.vtable.hash = Box::new(move |this, vm, hasher| {
            let Some(this) = this.as_type::<T>() else {
                return;
            };
            func(this, vm, hasher);
        });
        self
    }

    #[must_use]
    pub fn with_bitwise_not<Func>(mut self, func: impl FnOnce(UnaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm) -> Result<Value, Fault> + Send + Sync + Send + Sync + 'static,
    {
        let func = func(self.t.vtable.bitwise_not);
        self.t.vtable.bitwise_not = Box::new(move |this, vm| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm)
        });
        self
    }

    #[must_use]
    pub fn with_bitwise_and<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.bitwise_and);
        self.t.vtable.bitwise_and = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_bitwise_or<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.bitwise_or);
        self.t.vtable.bitwise_or = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_bitwise_xor<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.bitwise_xor);
        self.t.vtable.bitwise_xor = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_shift_left<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.shift_left);
        self.t.vtable.shift_left = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_shift_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.shift_right);
        self.t.vtable.shift_right = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_negate<Func>(mut self, func: impl FnOnce(UnaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm) -> Result<Value, Fault> + Send + Sync + Send + Sync + 'static,
    {
        let func = func(self.t.vtable.negate);
        self.t.vtable.negate = Box::new(move |this, vm| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm)
        });
        self
    }

    #[must_use]
    pub fn with_eq<Func>(mut self, func: impl FnOnce(EqFn) -> Func) -> Self
    where
        Func:
            Fn(Dynamic<T>, Option<&mut Vm>, &Value) -> Result<bool, Fault> + Send + Sync + 'static,
    {
        let func = func(self.t.vtable.eq);
        self.t.vtable.eq = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_matches<Func>(mut self, func: impl FnOnce(MatchesFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<bool, Fault> + Send + Sync + 'static,
    {
        let func = func(self.t.vtable.matches);
        self.t.vtable.matches = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_total_cmp<Func>(mut self, func: impl FnOnce(TotalCmpFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Ordering, Fault> + Send + Sync + 'static,
    {
        let func = func(self.t.vtable.total_cmp);
        self.t.vtable.total_cmp = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_add<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.add);
        self.t.vtable.add = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_add_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.add_right);
        self.t.vtable.add_right = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_sub<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.sub);
        self.t.vtable.sub = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_sub_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.sub_right);
        self.t.vtable.sub_right = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_mul<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.mul);
        self.t.vtable.mul = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_mul_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.mul_right);
        self.t.vtable.mul_right = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_div<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.div);
        self.t.vtable.div = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_div_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.div_right);
        self.t.vtable.div_right = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_idiv<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.idiv);
        self.t.vtable.idiv = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_idiv_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.idiv_right);
        self.t.vtable.idiv_right = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_rem<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.rem);
        self.t.vtable.rem = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_rem_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.rem_right);
        self.t.vtable.rem_right = Box::new(move |this, vm, rhs| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_truthy<Func>(mut self, func: impl FnOnce(TruthyFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm) -> bool + Send + Sync + 'static,
    {
        let func = func(self.t.vtable.truthy);
        self.t.vtable.truthy = Box::new(move |this, vm| {
            let Some(this) = this.as_type::<T>() else {
                return true;
            };
            func(this, vm)
        });
        self
    }

    #[must_use]
    pub fn with_to_string<Func>(mut self, func: impl FnOnce(ToStringFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>, &mut Vm) -> Result<Symbol, Fault> + Send + Sync + 'static,
    {
        let func = func(self.t.vtable.to_string);
        self.t.vtable.to_string = Box::new(move |this, vm| {
            let this = this.as_type::<T>().ok_or(Fault::UnsupportedOperation)?;
            func(this, vm)
        });
        self
    }

    #[must_use]
    pub fn with_deep_clone<Func>(mut self, func: impl FnOnce(DeepCloneFn) -> Func) -> Self
    where
        Func: Fn(Dynamic<T>) -> Option<AnyDynamic> + Send + Sync + Send + Sync + 'static,
    {
        let func = func(self.t.vtable.deep_clone);
        self.t.vtable.deep_clone = Box::new(move |this| {
            let this = this.as_type::<T>()?;
            func(this)
        });
        self
    }

    #[must_use]
    pub fn with_clone(self) -> Self
    where
        T: Clone,
    {
        self.with_deep_clone(|_| |this| Some(AnyDynamic::new((*this).clone())))
    }

    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn with_fallback<Mapping>(mut self, mapping: Mapping) -> Self
    where
        Mapping: Fn(Dynamic<T>) -> Value + Send + Sync + Clone + 'static,
    {
        self.t = self
            .t
            .with_fallback(move |dynamic| dynamic.as_type::<T>().map_or(Value::Nil, &mapping));
        self
    }

    fn seal(self) -> TypeRef {
        self.t.seal()
    }
}

impl CustomType for Type {
    fn muse_type(&self) -> &TypeRef {
        static TYPE: RustType<Type> = RustType::new("Type", |t| {
            t.with_call(|_| |this, vm, arity| (this.vtable.construct)(vm, arity))
        });
        &TYPE
    }
}

pub type TypeRef = Dynamic<Type>;

pub type ConstructFn = Box<dyn Fn(&mut Vm, Arity) -> Result<Value, Fault> + Send + Sync>;
pub type CallFn = Box<dyn Fn(&AnyDynamic, &mut Vm, Arity) -> Result<Value, Fault> + Send + Sync>;
pub type HashFn = Box<dyn Fn(&AnyDynamic, &mut Vm, &mut ValueHasher) + Send + Sync>;
pub type UnaryFn = Box<dyn Fn(&AnyDynamic, &mut Vm) -> Result<Value, Fault> + Send + Sync>;
pub type BinaryFn = Box<dyn Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Value, Fault> + Send + Sync>;
pub type MatchesFn = Box<dyn Fn(&AnyDynamic, &mut Vm, &Value) -> Result<bool, Fault> + Send + Sync>;
pub type EqFn =
    Box<dyn Fn(&AnyDynamic, Option<&mut Vm>, &Value) -> Result<bool, Fault> + Send + Sync>;
pub type TotalCmpFn =
    Box<dyn Fn(&AnyDynamic, &mut Vm, &Value) -> Result<Ordering, Fault> + Send + Sync>;
pub type InvokeFn =
    Box<dyn Fn(&AnyDynamic, &mut Vm, &Symbol, Arity) -> Result<Value, Fault> + Send + Sync>;
pub type DeepCloneFn = Box<dyn Fn(&AnyDynamic) -> Option<AnyDynamic> + Send + Sync>;
pub type TruthyFn = Box<dyn Fn(&AnyDynamic, &mut Vm) -> bool + Send + Sync>;
pub type ToStringFn = Box<dyn Fn(&AnyDynamic, &mut Vm) -> Result<Symbol, Fault> + Send + Sync>;

#[allow(clippy::type_complexity)]
pub struct TypeVtable {
    construct: ConstructFn,
    call: CallFn,
    invoke: InvokeFn,
    hash: HashFn,
    bitwise_not: UnaryFn,
    bitwise_and: BinaryFn,
    bitwise_or: BinaryFn,
    bitwise_xor: BinaryFn,
    shift_left: BinaryFn,
    shift_right: BinaryFn,
    negate: UnaryFn,
    eq: EqFn,
    matches: MatchesFn,
    total_cmp: TotalCmpFn,
    add: BinaryFn,
    add_right: BinaryFn,
    sub: BinaryFn,
    sub_right: BinaryFn,
    mul: BinaryFn,
    mul_right: BinaryFn,
    div: BinaryFn,
    div_right: BinaryFn,
    idiv: BinaryFn,
    idiv_right: BinaryFn,
    rem: BinaryFn,
    rem_right: BinaryFn,
    truthy: TruthyFn,
    to_string: ToStringFn,
    deep_clone: DeepCloneFn,
}

impl Default for TypeVtable {
    fn default() -> Self {
        Self {
            construct: Box::new(|_vm, _arity| Err(Fault::UnsupportedOperation)),
            call: Box::new(|_this, _vm, _arity| Err(Fault::NotAFunction)),
            invoke: Box::new(|_this, _vm, _name, _arity| Err(Fault::UnknownSymbol)),
            hash: Box::new(|this, _vm, hasher| Arc::as_ptr(&this.0).hash(hasher)),
            bitwise_not: Box::new(|_this, _vm| Err(Fault::UnsupportedOperation)),
            bitwise_and: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            bitwise_or: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            bitwise_xor: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            shift_left: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            shift_right: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            negate: Box::new(|_this, _vm| Err(Fault::UnsupportedOperation)),
            eq: Box::new(|_this, _vm, _rhs| Ok(false)),
            matches: Box::new(|this, vm, rhs| this.eq(Some(vm), rhs)),
            total_cmp: Box::new(|_this, _vm, rhs| {
                if rhs.as_any_dynamic().is_none() {
                    // Dynamics sort after primitive values
                    Ok(Ordering::Greater)
                } else {
                    Err(Fault::UnsupportedOperation)
                }
            }),
            add: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            add_right: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            sub: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            sub_right: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            mul: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            mul_right: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            div: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            div_right: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            idiv: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            idiv_right: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            rem: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            rem_right: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            truthy: Box::new(|_this, _vm| true),
            to_string: Box::new(|_this, _vm| Err(Fault::UnsupportedOperation)),
            deep_clone: Box::new(|_this| None),
        }
    }
}

pub trait CustomType: Send + Sync + Debug + 'static {
    fn muse_type(&self) -> &TypeRef;
}

pub struct RustFunctionTable<T> {
    functions: Map<Symbol, Map<Arity, Arc<dyn RustFn<T>>>>,
}

impl<T> RustFunctionTable<T>
where
    T: 'static,
{
    #[must_use]
    pub const fn new() -> Self {
        Self {
            functions: Map::new(),
        }
    }

    #[must_use]
    pub fn with_fn<F>(mut self, name: impl SymbolList, arity: impl Into<Arity>, func: F) -> Self
    where
        F: Fn(&mut Vm, &T) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        let func = Arc::new(func);
        let arity = arity.into();
        for symbol in name.into_symbols() {
            self.functions
                .entry(symbol)
                .or_default()
                .insert(arity, func.clone());
        }
        self
    }

    pub fn invoke(
        &self,
        vm: &mut Vm,
        name: &Symbol,
        arity: Arity,
        this: &T,
    ) -> Result<Value, Fault> {
        if let Some(by_arity) = self.functions.get(name) {
            if let Some(func) = by_arity.get(&arity) {
                func.invoke(vm, this)
            } else {
                Err(Fault::IncorrectNumberOfArguments)
            }
        } else {
            Err(Fault::UnknownSymbol)
        }
    }
}

pub struct StaticRustFunctionTable<T>(
    OnceLock<RustFunctionTable<T>>,
    fn(RustFunctionTable<T>) -> RustFunctionTable<T>,
);

impl<T> StaticRustFunctionTable<T> {
    pub const fn new(init: fn(RustFunctionTable<T>) -> RustFunctionTable<T>) -> Self {
        Self(OnceLock::new(), init)
    }
}

impl<T> Deref for StaticRustFunctionTable<T>
where
    T: 'static,
{
    type Target = RustFunctionTable<T>;

    fn deref(&self) -> &Self::Target {
        self.0.get_or_init(|| self.1(RustFunctionTable::new()))
    }
}

trait RustFn<T>: Send + Sync + 'static {
    fn invoke(&self, vm: &mut Vm, this: &T) -> Result<Value, Fault>;
}

impl<T, F> RustFn<T> for F
where
    F: Fn(&mut Vm, &T) -> Result<Value, Fault> + Send + Sync + 'static,
{
    fn invoke(&self, vm: &mut Vm, this: &T) -> Result<Value, Fault> {
        self(vm, this)
    }
}

type ArcRustFn = Arc<dyn Fn(&mut Vm, Arity) -> Result<Value, Fault> + Send + Sync>;

#[derive(Clone)]
pub struct RustFunction(ArcRustFn);

impl RustFunction {
    pub fn new<F>(function: F) -> Self
    where
        F: Fn(&mut Vm, Arity) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        Self(Arc::new(function))
    }
}

impl Debug for RustFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RustFunction")
            .field(&std::ptr::addr_of!(self.0).cast::<()>())
            .finish()
    }
}

impl CustomType for RustFunction {
    fn muse_type(&self) -> &TypeRef {
        static TYPE: RustType<RustFunction> = RustType::new("NativFunction", |t| {
            t.with_call(|_previous| {
                |this, vm, arity| {
                    vm.enter_anonymous_frame()?;
                    let result = this.0(vm, arity)?;
                    vm.exit_frame()?;
                    Ok(result)
                }
            })
        });
        &TYPE
    }
}

type ArcAsyncFunction = Arc<
    dyn Fn(&mut Vm, Arity) -> Pin<Box<dyn Future<Output = Result<Value, Fault>> + Send + Sync>>
        + Send
        + Sync,
>;

#[derive(Clone)]
pub struct AsyncFunction(ArcAsyncFunction);

impl AsyncFunction {
    pub fn new<F, Fut>(function: F) -> Self
    where
        F: Fn(&mut Vm, Arity) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value, Fault>> + Send + Sync + 'static,
    {
        Self(Arc::new(move |vm, arity| Box::pin(function(vm, arity))))
    }
}

impl Debug for AsyncFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("AsyncFunction")
            .field(&std::ptr::addr_of!(self.0).cast::<()>())
            .finish()
    }
}

impl CustomType for AsyncFunction {
    fn muse_type(&self) -> &TypeRef {
        static TYPE: RustType<AsyncFunction> = RustType::new("AsyncNativeFunction", |t| {
            t.with_call(|_previous| {
                |this, vm, arity| {
                    vm.enter_anonymous_frame()?;
                    if vm.current_instruction() == 0 {
                        let future = this.0(vm, arity);
                        vm.allocate(1)?;
                        vm.current_frame_mut()[0] =
                            Value::dynamic(ValueFuture(Arc::new(Mutex::new(Box::pin(future)))));
                        vm.jump_to(1);
                    }

                    let Some(future) = vm.current_frame()[0]
                        .as_any_dynamic()
                        .and_then(|d| d.downcast_ref::<ValueFuture>())
                    else {
                        unreachable!("missing future")
                    };

                    let mut future = future.0.lock().expect("poisoned");
                    let mut cx = Context::from_waker(vm.waker());
                    match Future::poll(std::pin::pin!(&mut *future), &mut cx) {
                        Poll::Ready(Ok(result)) => {
                            drop(future);
                            vm.exit_frame()?;
                            Ok(result)
                        }
                        Poll::Ready(Err(Fault::Waiting)) | Poll::Pending => Err(Fault::Waiting),
                        Poll::Ready(Err(err)) => Err(err),
                    }
                }
            })
        });

        &TYPE
    }
}

type ArcFuture = Arc<Mutex<Pin<Box<dyn Future<Output = Result<Value, Fault>> + Send + Sync>>>>;

struct ValueFuture(ArcFuture);

impl Clone for ValueFuture {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl CustomType for ValueFuture {
    fn muse_type(&self) -> &TypeRef {
        static TYPE: StaticType = StaticType::new(|| Type::new("AsyncValue"));
        &TYPE
    }
}

impl Debug for ValueFuture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValueFuture").finish_non_exhaustive()
    }
}

pub trait DynamicValue: CustomType {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T> DynamicValue for T
where
    T: CustomType,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[derive(Debug, Clone)]
pub struct WeakAnyDynamic(Weak<Box<dyn DynamicValue>>);

impl WeakAnyDynamic {
    pub fn upgrade(&self) -> Option<AnyDynamic> {
        self.0.upgrade().map(AnyDynamic)
    }
}

#[derive(Debug, Clone)]
pub struct WeakDynamic<T> {
    weak: WeakAnyDynamic,
    _t: PhantomData<T>,
}

impl<T> WeakDynamic<T>
where
    T: CustomType,
{
    #[must_use]
    pub fn upgrade(&self) -> Option<Dynamic<T>> {
        self.weak.upgrade().map(|dynamic| Dynamic {
            dynamic,
            _t: PhantomData,
        })
    }
}

#[test]
fn dynamic() {
    impl CustomType for usize {
        fn muse_type(&self) -> &TypeRef {
            static TYPE: RustType<usize> = RustType::new("usize", TypedTypeBuilder::with_clone);
            &TYPE
        }
    }
    let mut dynamic = AnyDynamic::new(1_usize);
    assert_eq!(dynamic.downcast_ref::<usize>(), Some(&1));
    let dynamic2 = dynamic.clone();
    assert!(AnyDynamic::ptr_eq(&dynamic, &dynamic2));
    *dynamic.downcast_mut::<usize>().unwrap() = 2;
    assert!(!AnyDynamic::ptr_eq(&dynamic, &dynamic2));
    assert_eq!(dynamic2.downcast_ref::<usize>(), Some(&1));
}

#[test]
fn functions() {
    let func = Value::Dynamic(AnyDynamic::new(RustFunction::new(
        |_vm: &mut Vm, _arity| Ok(Value::Int(1)),
    )));
    let mut runtime = Vm::default();
    let Value::Int(i) = func.call(&mut runtime, 0).unwrap() else {
        unreachable!()
    };
    assert_eq!(i, 1);
}
