use std::any::Any;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::task::{Context, Poll};

pub type ValueHasher = ahash::AHasher;

use kempt::Map;

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
    Dynamic(Dynamic),
}

impl Value {
    pub fn dynamic<T>(value: T) -> Self
    where
        T: DynamicValue,
    {
        Self::Dynamic(Dynamic::new(value))
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
    pub fn as_dynamic(&self) -> Option<&Dynamic> {
        match self {
            Value::Dynamic(value) => Some(value),
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
            (Value::Dynamic(dynamic), _) => dynamic.0.invoke(vm, name, arity.into()),
            (Value::Nil, _) => Err(Fault::OperationOnNil),
            _ => Err(Fault::UnknownSymbol(name.clone())),
        }
    }

    pub fn add(&self, vm: &mut Vm, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),
            (Value::Bool(lhs), Value::Bool(rhs)) => Ok(Value::Bool(*lhs || *rhs)),
            (Value::Bool(_), _) | (_, Value::Bool(_)) => Err(Fault::UnsupportedOperation),

            (Value::Symbol(lhs), rhs) => {
                let rhs = rhs.to_string(vm)?;
                Ok(Value::Symbol(lhs + &rhs))
            }
            (lhs, Value::Symbol(rhs)) => {
                let lhs = lhs.to_string(vm)?;
                Ok(Value::Symbol(&lhs + rhs))
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
        match self {
            Value::Dynamic(value) => value.not(vm),
            _ => Ok(Value::Bool(!self.truthy(vm))),
        }
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
pub struct Dynamic(Arc<Box<dyn DynamicValue>>);

impl Dynamic {
    pub fn new<T>(value: T) -> Self
    where
        T: DynamicValue,
    {
        Self(Arc::new(Box::new(value)))
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
            *self = (**self.0).deep_clone()?;
        }

        let dynamic = &mut *Arc::get_mut(&mut self.0).expect("always 1 ref");
        dynamic.as_any_mut().downcast_mut()
    }

    #[must_use]
    pub fn downgrade(&self) -> WeakDynamic {
        WeakDynamic(Arc::downgrade(&self.0))
    }

    #[must_use]
    pub fn ptr_eq(a: &Dynamic, b: &Dynamic) -> bool {
        Arc::ptr_eq(&a.0, &b.0)
    }

    #[must_use]
    pub fn deep_clone(&self) -> Option<Dynamic> {
        self.0.deep_clone()
    }

    pub fn call(&self, vm: &mut Vm, arity: impl Into<Arity>) -> Result<Value, Fault> {
        self.0.call(vm, self, arity.into())
    }

    pub fn add(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
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

    pub fn idiv(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        self.0.div(vm, rhs)
    }

    pub fn idiv_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
        self.0.idiv_right(vm, lhs)
    }

    pub fn hash(&self, vm: &mut Vm, hasher: &mut ValueHasher) {
        self.0.hash(vm, hasher);
    }

    pub fn not(&self, vm: &mut Vm) -> Result<Value, Fault> {
        self.0.not(vm)
    }

    pub fn bitwise_not(&self, vm: &mut Vm) -> Result<Value, Fault> {
        self.0.bitwise_not(vm)
    }

    pub fn bitwise_and(&self, vm: &mut Vm, other: &Value) -> Result<Value, Fault> {
        self.0.bitwise_and(vm, other)
    }

    pub fn bitwise_or(&self, vm: &mut Vm, other: &Value) -> Result<Value, Fault> {
        self.0.bitwise_or(vm, other)
    }

    pub fn bitwise_xor(&self, vm: &mut Vm, other: &Value) -> Result<Value, Fault> {
        self.0.bitwise_xor(vm, other)
    }

    pub fn shift_left(&self, vm: &mut Vm, amount: &Value) -> Result<Value, Fault> {
        self.0.shift_left(vm, amount)
    }

    pub fn shift_right(&self, vm: &mut Vm, amount: &Value) -> Result<Value, Fault> {
        self.0.shift_right(vm, amount)
    }

    pub fn negate(&self, vm: &mut Vm) -> Result<Value, Fault> {
        self.0.negate(vm)
    }

    pub fn to_string(&self, vm: &mut Vm) -> Result<Symbol, Fault> {
        self.0.to_string(vm)
    }

    pub fn truthy(&self, vm: &mut Vm) -> bool {
        self.0.truthy(vm)
    }

    pub fn eq(&self, vm: Option<&mut Vm>, rhs: &Value) -> Result<bool, Fault> {
        match rhs {
            Value::Dynamic(dynamic) if Arc::ptr_eq(&self.0, &dynamic.0) => Ok(true),
            _ => self.0.eq(vm, rhs),
        }
    }

    pub fn cmp(&self, vm: &mut Vm, rhs: &Value) -> Result<Ordering, Fault> {
        self.0.total_cmp(vm, rhs)
    }
}

impl Debug for Dynamic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self.0, f)
    }
}

pub trait CustomType: Send + Sync + Debug + 'static {
    #[allow(unused_variables)]
    fn call(&self, vm: &mut Vm, this: &Dynamic, arity: Arity) -> Result<Value, Fault> {
        Err(Fault::NotAFunction)
    }

    #[allow(unused_variables)]
    fn hash(&self, vm: &mut Vm, hasher: &mut ValueHasher) {
        (self as *const Self).hash(hasher);
    }

    #[allow(unused_variables)]
    fn not(&self, vm: &mut Vm) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn and(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn bitwise_not(&self, vm: &mut Vm) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn bitwise_and(&self, vm: &mut Vm, other: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn bitwise_or(&self, vm: &mut Vm, other: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn bitwise_xor(&self, vm: &mut Vm, other: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn shift_left(&self, vm: &mut Vm, amount: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn shift_right(&self, vm: &mut Vm, amount: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn negate(&self, vm: &mut Vm) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn eq(&self, vm: Option<&mut Vm>, rhs: &Value) -> Result<bool, Fault> {
        Ok(false)
    }

    #[allow(unused_variables)]
    fn total_cmp(&self, vm: &mut Vm, rhs: &Value) -> Result<Ordering, Fault> {
        if rhs.as_dynamic().is_none() {
            // Dynamics sort after primitive values
            Ok(Ordering::Greater)
        } else {
            Err(Fault::UnsupportedOperation)
        }
    }

    #[allow(unused_variables)]
    fn invoke(&self, vm: &mut Vm, name: &Symbol, arity: Arity) -> Result<Value, Fault> {
        Err(Fault::UnknownSymbol(name.clone()))
    }

    #[allow(unused_variables)]
    fn add(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
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
    fn idiv(&self, vm: &mut Vm, rhs: &Value) -> Result<Value, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    #[allow(unused_variables)]
    fn idiv_right(&self, vm: &mut Vm, lhs: &Value) -> Result<Value, Fault> {
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
    fn truthy(&self, vm: &mut Vm) -> bool {
        true
    }

    #[allow(unused_variables)]
    fn to_string(&self, vm: &mut Vm) -> Result<Symbol, Fault> {
        Err(Fault::UnsupportedOperation)
    }

    fn deep_clone(&self) -> Option<Dynamic> {
        None
    }
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
            Err(Fault::UnknownSymbol(name.clone()))
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

#[derive(Clone)]
pub struct RustFunction<F>(F);

impl<F> RustFunction<F> {
    pub fn new(function: F) -> Self {
        Self(function)
    }
}

impl<F> Debug for RustFunction<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("RustFunction")
            .field(&std::ptr::addr_of!(self.0).cast::<()>())
            .finish()
    }
}

impl<F> CustomType for RustFunction<F>
where
    F: Fn(&mut Vm, Arity) -> Result<Value, Fault> + Send + Sync + 'static,
{
    fn call(&self, vm: &mut Vm, _this: &Dynamic, arity: Arity) -> Result<Value, Fault> {
        vm.enter_anonymous_frame()?;
        let result = self.0(vm, arity)?;
        vm.exit_frame()?;
        Ok(result)
    }
}

#[derive(Clone)]
pub struct AsyncFunction<F>(F);

impl<F> AsyncFunction<F> {
    pub fn new(function: F) -> Self {
        Self(function)
    }
}

impl<F> Debug for AsyncFunction<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("AsyncFunction")
            .field(&std::ptr::addr_of!(self.0).cast::<()>())
            .finish()
    }
}

impl<F, Fut> CustomType for AsyncFunction<F>
where
    F: Fn(&mut Vm, Arity) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Value, Fault>> + Send + Sync + 'static,
{
    fn call(&self, vm: &mut Vm, _this: &Dynamic, arity: Arity) -> Result<Value, Fault> {
        vm.enter_anonymous_frame()?;
        if vm.current_instruction() == 0 {
            let future = self.0(vm, arity);
            vm.allocate(1)?;
            vm.current_frame_mut()[0] =
                Value::dynamic(ValueFuture(Arc::new(Mutex::new(Box::pin(future)))));
            vm.jump_to(1);
        }

        let Some(future) = vm.current_frame()[0]
            .as_dynamic()
            .and_then(|d| d.downcast_ref::<ValueFuture<Fut>>())
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
}

struct ValueFuture<F>(Arc<Mutex<Pin<Box<F>>>>);

impl<F> Clone for ValueFuture<F> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<F> CustomType for ValueFuture<F> where F: Send + Sync + 'static {}

impl<F> Debug for ValueFuture<F> {
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
pub struct WeakDynamic(Weak<Box<dyn DynamicValue>>);

impl WeakDynamic {
    pub fn upgrade(&self) -> Option<Dynamic> {
        self.0.upgrade().map(Dynamic)
    }
}

#[test]
fn dynamic() {
    impl CustomType for usize {
        fn deep_clone(&self) -> Option<Dynamic> {
            Some(Dynamic::new(*self))
        }
    }
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
    let func = Value::Dynamic(Dynamic::new(RustFunction::new(|_vm: &mut Vm, _arity| {
        Ok(Value::Int(1))
    })));
    let mut runtime = Vm::default();
    let Value::Int(i) = func.call(&mut runtime, 0).unwrap() else {
        unreachable!()
    };
    assert_eq!(i, 1);
}
