use std::any::Any;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};

pub type ValueHasher = ahash::AHasher;

use kempt::Map;
use parking_lot::Mutex;
use refuse::{AnyRef, AnyRoot, CollectionGuard, ContainsNoRefs, MapAs, Ref, Root, Trace};

use crate::string::MuseString;
use crate::symbol::{Symbol, SymbolList, SymbolRef};
use crate::vm::{Arity, ExecutionError, Fault, VmContext};

#[derive(Default, Clone, Copy, Debug)]
pub enum Value {
    #[default]
    Nil,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Symbol(SymbolRef),
    Dynamic(AnyDynamic),
}

impl Value {
    pub fn dynamic<'guard, T>(value: T, guard: impl AsRef<CollectionGuard<'guard>>) -> Self
    where
        T: DynamicValue + Trace,
    {
        Self::Dynamic(AnyDynamic::new(value, guard))
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
    pub fn as_symbol_ref(&self) -> Option<&SymbolRef> {
        match self {
            Value::Symbol(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_symbol(&self, guard: &CollectionGuard<'_>) -> Option<Symbol> {
        match self {
            Value::Symbol(value) => value.upgrade(guard),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_any_dynamic(&self) -> Option<AnyDynamic> {
        match self {
            Value::Dynamic(value) => Some(*value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_dynamic<T>(&self) -> Option<Dynamic<T>>
    where
        T: DynamicValue + Trace,
    {
        match self {
            Value::Dynamic(value) => Some(value.as_dynamic()),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_rooted<T>(&self, guard: &CollectionGuard<'_>) -> Option<Rooted<T>>
    where
        T: DynamicValue + Trace,
    {
        match self {
            Value::Dynamic(value) => value.as_rooted(guard),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_downcast_ref<'guard, T>(&self, guard: &'guard CollectionGuard) -> Option<&'guard T>
    where
        T: DynamicValue + Trace,
    {
        match self {
            Value::Dynamic(value) => value.downcast_ref(guard),
            _ => None,
        }
    }

    pub fn truthy(&self, vm: &mut VmContext<'_, '_>) -> bool {
        match self {
            Value::Nil => false,
            Value::Bool(value) => *value,
            Value::Int(value) => value != &0,
            Value::UInt(value) => value != &0,
            Value::Float(value) => value.abs() >= f64::EPSILON,
            Value::Symbol(sym) => sym.load(vm.as_ref()).map_or(false, |sym| !sym.is_empty()),
            Value::Dynamic(value) => value.truthy(vm),
        }
    }

    pub fn call(
        &self,
        vm: &mut VmContext<'_, '_>,
        arity: impl Into<Arity>,
    ) -> Result<Value, Fault> {
        match self {
            Value::Dynamic(dynamic) => dynamic.call(vm, arity),
            Value::Symbol(name) => {
                let name = name.try_upgrade(vm.guard())?;
                vm.resolve(&name).and_then(|named| named.call(vm, arity))
            }
            Value::Nil => vm.recurse_current_function(arity.into()),
            _ => Err(Fault::NotAFunction),
        }
    }

    pub fn invoke(
        &self,
        vm: &mut VmContext<'_, '_>,
        name: &SymbolRef,
        arity: impl Into<Arity>,
    ) -> Result<Value, Fault> {
        match self {
            Value::Dynamic(dynamic) => dynamic.invoke(vm, name, arity.into()),
            Value::Nil => Err(Fault::OperationOnNil),
            // TODO we should pass through to the appropriate Type
            _ => Err(Fault::UnknownSymbol),
        }
    }

    pub fn add(&self, vm: &mut VmContext<'_, '_>, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),
            (Value::Bool(lhs), Value::Bool(rhs)) => Ok(Value::Bool(*lhs || *rhs)),
            (Value::Bool(_), _) | (_, Value::Bool(_)) => Err(Fault::UnsupportedOperation),

            (Value::Symbol(lhs), rhs) => {
                let lhs = lhs.try_upgrade(vm.guard())?;
                rhs.map_str(vm, |_vm, rhs| Value::Symbol(SymbolRef::from(&lhs + rhs)))
            }
            (lhs, Value::Symbol(rhs)) => {
                let rhs = rhs.try_upgrade(vm.guard())?;
                lhs.map_str(vm, |_vm, lhs| Value::Symbol(SymbolRef::from(lhs + &rhs)))
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

    pub fn sub(&self, vm: &mut VmContext<'_, '_>, rhs: &Self) -> Result<Value, Fault> {
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

    pub fn mul(&self, vm: &mut VmContext<'_, '_>, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Nil, _) | (_, Value::Nil) => Err(Fault::OperationOnNil),

            (Value::Int(count), Value::Symbol(string))
            | (Value::Symbol(string), Value::Int(count)) => {
                let string = string.try_upgrade(vm.guard())?;
                Ok(Value::Symbol(SymbolRef::from(string.repeat(
                    usize::try_from(*count).map_err(|_| Fault::OutOfMemory)?,
                ))))
            }

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

    pub fn pow(&self, vm: &mut VmContext<'_, '_>, exp: &Self) -> Result<Value, Fault> {
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

    pub fn div(&self, vm: &mut VmContext<'_, '_>, rhs: &Self) -> Result<Value, Fault> {
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

    pub fn idiv(&self, vm: &mut VmContext<'_, '_>, rhs: &Self) -> Result<Value, Fault> {
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

    pub fn rem(&self, vm: &mut VmContext<'_, '_>, rhs: &Self) -> Result<Value, Fault> {
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

    pub fn not(&self, vm: &mut VmContext<'_, '_>) -> Result<Self, Fault> {
        Ok(Value::Bool(!self.truthy(vm)))
    }

    pub fn negate(&self, vm: &mut VmContext<'_, '_>) -> Result<Self, Fault> {
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

    pub fn bitwise_not(&self, vm: &mut VmContext<'_, '_>) -> Result<Self, Fault> {
        match self {
            Value::Nil => Ok(Value::Nil),
            Value::Bool(bool) => Ok(Value::Bool(!bool)),
            Value::Int(value) => Ok(Value::Int(!*value)),
            Value::UInt(value) => Ok(Value::UInt(!*value)),
            Value::Dynamic(value) => value.bitwise_not(vm),
            Value::Float(_) | Value::Symbol(_) => Err(Fault::UnsupportedOperation),
        }
    }

    pub fn bitwise_and(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Self, Fault> {
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

    pub fn bitwise_or(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Self, Fault> {
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

    pub fn bitwise_xor(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Self, Fault> {
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

    pub fn shift_left(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Self, Fault> {
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

    pub fn shift_right(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Self, Fault> {
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

    pub fn to_string(&self, vm: &mut VmContext<'_, '_>) -> Result<SymbolRef, Fault> {
        match self {
            Value::Nil => Ok(Symbol::empty().downgrade()),
            Value::Bool(bool) => Ok(SymbolRef::from(*bool)),
            Value::Int(value) => Ok(SymbolRef::from(value.to_string())),
            Value::UInt(value) => Ok(SymbolRef::from(value.to_string())),
            Value::Float(value) => Ok(SymbolRef::from(value.to_string())),
            Value::Symbol(value) => Ok(*value),
            Value::Dynamic(value) => value.to_string(vm),
        }
    }

    pub fn map_str<R>(
        &self,
        vm: &mut VmContext<'_, '_>,
        map: impl FnOnce(&mut VmContext<'_, '_>, &str) -> R,
    ) -> Result<R, Fault> {
        if let Some(str) = self.as_rooted::<MuseString>(vm.as_ref()) {
            return Ok(map(vm, &str.lock()));
        }

        let str = self.to_string(vm)?.try_upgrade(vm.guard())?;
        Ok(map(vm, &str))
    }

    pub fn hash_into(&self, vm: &mut VmContext<'_, '_>, hasher: &mut ValueHasher) {
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

    pub fn hash(&self, vm: &mut VmContext<'_, '_>) -> u64 {
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

    pub fn equals(&self, vm: ContextOrGuard<'_, '_, '_>, other: &Self) -> Result<bool, Fault> {
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
                Ok(s == &SymbolRef::from(*b))
            }

            (Self::Dynamic(l0), _) => l0.eq(vm, other),
            (_, Self::Dynamic(r0)) => r0.eq(vm, self),

            _ => Ok(false),
        }
    }

    pub fn matches(&self, vm: &mut VmContext<'_, '_>, other: &Self) -> Result<bool, Fault> {
        match (self, other) {
            (Self::Dynamic(l0), _) => l0.matches(vm, other),
            (_, Self::Dynamic(r0)) => r0.matches(vm, self),
            _ => self.equals(ContextOrGuard::Context(vm), other),
        }
    }

    pub fn total_cmp(&self, vm: &mut VmContext<'_, '_>, other: &Self) -> Result<Ordering, Fault> {
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

    pub fn deep_clone(&self, guard: &CollectionGuard) -> Option<Value> {
        match self {
            Value::Nil => Some(Value::Nil),
            Value::Bool(value) => Some(Value::Bool(*value)),
            Value::Int(value) => Some(Value::Int(*value)),
            Value::UInt(value) => Some(Value::UInt(*value)),
            Value::Float(value) => Some(Value::Float(*value)),
            Value::Symbol(value) => Some(Value::Symbol(*value)),
            Value::Dynamic(value) => value.deep_clone(guard).map(Value::Dynamic),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.equals(ContextOrGuard::Guard(&CollectionGuard::acquire()), other)
            .unwrap_or(false)
    }
}

impl Trace for Value {
    const MAY_CONTAIN_REFERENCES: bool = true;

    fn trace(&self, tracer: &mut refuse::Tracer) {
        let Some(dynamic) = self.as_any_dynamic() else {
            return;
        };
        dynamic.trace(tracer);
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

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct AnyDynamic(pub(crate) AnyRef);

impl AnyDynamic {
    pub fn new<'guard, T>(value: T, guard: impl AsRef<CollectionGuard<'guard>>) -> Self
    where
        T: DynamicValue + Trace,
    {
        Self(Ref::new(Custom(value), guard).as_any())
    }

    #[must_use]
    pub fn as_dynamic<T>(&self) -> Dynamic<T>
    where
        T: DynamicValue + Trace,
    {
        Dynamic(self.0.downcast_ref::<Custom<T>>())
    }

    #[must_use]
    pub fn as_rooted<T>(&self, guard: &CollectionGuard<'_>) -> Option<Rooted<T>>
    where
        T: DynamicValue + Trace,
    {
        self.0
            .downcast_root::<Custom<T>>(guard)
            .map(|cast| Rooted(cast))
    }

    #[must_use]
    pub fn downcast_ref<'guard, T>(&self, guard: &'guard CollectionGuard) -> Option<&'guard T>
    where
        T: DynamicValue + Trace,
    {
        self.0
            .load::<Custom<T>>(guard)
            .and_then(|d| d.0.as_any().downcast_ref())
    }

    #[must_use]
    pub fn deep_clone(&self, guard: &CollectionGuard) -> Option<AnyDynamic> {
        (self
            .0
            .load_mapped::<dyn CustomType>(guard)?
            .muse_type()
            .vtable
            .deep_clone)(self, guard)
    }

    pub fn call(
        &self,
        vm: &mut VmContext<'_, '_>,
        arity: impl Into<Arity>,
    ) -> Result<Value, Fault> {
        if let Some(kind) = self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .map(|v| v.muse_type().clone())
        {
            (kind.vtable.call)(self, vm, arity.into())
        } else {
            Err(Fault::ValueFreed)
        }
    }

    pub fn invoke(
        &self,
        vm: &mut VmContext<'_, '_>,
        symbol: &SymbolRef,
        arity: impl Into<Arity>,
    ) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .invoke)(self, vm, symbol, arity.into())
    }

    pub fn add(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .add)(self, vm, rhs)
    }

    pub fn add_right(&self, vm: &mut VmContext<'_, '_>, lhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .add_right)(self, vm, lhs)
    }

    pub fn sub(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .sub)(self, vm, rhs)
    }

    pub fn sub_right(&self, vm: &mut VmContext<'_, '_>, lhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .sub_right)(self, vm, lhs)
    }

    pub fn mul(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .mul)(self, vm, rhs)
    }

    pub fn mul_right(&self, vm: &mut VmContext<'_, '_>, lhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .mul_right)(self, vm, lhs)
    }

    pub fn div(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .div)(self, vm, rhs)
    }

    pub fn div_right(&self, vm: &mut VmContext<'_, '_>, lhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .div_right)(self, vm, lhs)
    }

    pub fn rem(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .rem)(self, vm, rhs)
    }

    pub fn rem_right(&self, vm: &mut VmContext<'_, '_>, lhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .rem_right)(self, vm, lhs)
    }

    pub fn idiv(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .div)(self, vm, rhs)
    }

    pub fn idiv_right(&self, vm: &mut VmContext<'_, '_>, lhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .idiv_right)(self, vm, lhs)
    }

    pub fn hash(&self, vm: &mut VmContext<'_, '_>, hasher: &mut ValueHasher) {
        let Some(value) = self.0.load_mapped::<dyn CustomType>(vm.as_ref()) else {
            return;
        };
        (value.muse_type().clone().vtable.hash)(self, vm, hasher);
    }

    pub fn bitwise_not(&self, vm: &mut VmContext<'_, '_>) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .bitwise_not)(self, vm)
    }

    pub fn bitwise_and(&self, vm: &mut VmContext<'_, '_>, other: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .bitwise_and)(self, vm, other)
    }

    pub fn bitwise_or(&self, vm: &mut VmContext<'_, '_>, other: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .bitwise_or)(self, vm, other)
    }

    pub fn bitwise_xor(&self, vm: &mut VmContext<'_, '_>, other: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .bitwise_xor)(self, vm, other)
    }

    pub fn shift_left(&self, vm: &mut VmContext<'_, '_>, amount: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .shift_left)(self, vm, amount)
    }

    pub fn shift_right(&self, vm: &mut VmContext<'_, '_>, amount: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .shift_right)(self, vm, amount)
    }

    pub fn negate(&self, vm: &mut VmContext<'_, '_>) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .negate)(self, vm)
    }

    pub fn to_string(&self, vm: &mut VmContext<'_, '_>) -> Result<SymbolRef, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .to_string)(self, vm)
    }

    pub fn truthy(&self, vm: &mut VmContext<'_, '_>) -> bool {
        let Some(value) = self.0.load_mapped::<dyn CustomType>(vm.as_ref()) else {
            return false;
        };
        (value.muse_type().clone().vtable.truthy)(self, vm)
    }

    pub fn eq(&self, vm: ContextOrGuard<'_, '_, '_>, rhs: &Value) -> Result<bool, Fault> {
        match rhs {
            Value::Dynamic(dynamic) if self.0 == dynamic.0 => Ok(true),
            _ => (self
                .0
                .load_mapped::<dyn CustomType>(vm.as_ref())
                .ok_or(Fault::OperationOnNil)?
                .muse_type()
                .clone()
                .vtable
                .eq)(self, vm, rhs),
        }
    }

    pub fn matches(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<bool, Fault> {
        match rhs {
            Value::Dynamic(dynamic) if self.0 == dynamic.0 => Ok(true),
            _ => (self
                .0
                .load_mapped::<dyn CustomType>(vm.as_ref())
                .ok_or(Fault::OperationOnNil)?
                .muse_type()
                .clone()
                .vtable
                .matches)(self, vm, rhs),
        }
    }

    pub fn cmp(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Ordering, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(Fault::OperationOnNil)?
            .muse_type()
            .clone()
            .vtable
            .total_cmp)(self, vm, rhs)
    }
}

impl Debug for AnyDynamic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = CollectionGuard::acquire();
        let Some(value) = self.0.load_mapped::<dyn CustomType>(&guard) else {
            return f.write_str("<Deallocated>");
        };
        Debug::fmt(value, f)
    }
}
impl Trace for AnyDynamic {
    const MAY_CONTAIN_REFERENCES: bool = true;

    fn trace(&self, tracer: &mut refuse::Tracer) {
        self.0.trace(tracer);
    }
}

pub struct Rooted<T: CustomType + Trace>(Root<Custom<T>>);
impl<T> Rooted<T>
where
    T: CustomType + Trace,
{
    #[must_use]
    pub fn new<'guard>(value: T, guard: impl AsRef<CollectionGuard<'guard>>) -> Self {
        Self(Root::new(Custom(value), guard))
    }

    #[must_use]
    pub fn as_any_dynamic(&self) -> AnyDynamic {
        AnyDynamic(self.0.downgrade_any())
    }

    #[must_use]
    pub fn into_any_root(self) -> AnyRoot {
        self.0.into_any_root()
    }
}

impl<T: CustomType + Trace> AsRef<T> for Rooted<T> {
    fn as_ref(&self) -> &T {
        &self.0 .0
    }
}

impl<T: CustomType + Trace> Deref for Rooted<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T: CustomType + Trace> Clone for Rooted<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl From<AnyDynamic> for AnyRef {
    fn from(value: AnyDynamic) -> Self {
        value.0
    }
}

pub struct Dynamic<T: CustomType>(Ref<Custom<T>>);

impl<T> Dynamic<T>
where
    T: CustomType + Trace,
{
    #[must_use]
    pub fn new<'guard>(value: T, guard: impl AsRef<CollectionGuard<'guard>>) -> Self {
        Self(Ref::new(Custom(value), guard))
    }

    #[must_use]
    pub fn load<'guard>(&self, guard: &'guard CollectionGuard) -> Option<&'guard T> {
        self.0.load(guard).map(|c| &c.0)
    }

    pub fn try_load<'guard>(
        &self,
        guard: &'guard CollectionGuard,
    ) -> Result<&'guard T, ValueFreed> {
        self.load(guard).ok_or(ValueFreed)
    }

    #[must_use]
    pub fn as_rooted(&self, guard: &CollectionGuard<'_>) -> Option<Rooted<T>> {
        self.0.as_root(guard).map(Rooted)
    }

    #[must_use]
    pub fn as_any_dynamic(&self) -> AnyDynamic {
        AnyDynamic(self.0.as_any())
    }

    #[must_use]
    pub fn to_value(&self) -> Value {
        Value::Dynamic(self.as_any_dynamic())
    }

    #[must_use]
    pub fn into_value(self) -> Value {
        Value::Dynamic(self.into_any_dynamic())
    }

    #[must_use]
    pub fn into_any_dynamic(self) -> AnyDynamic {
        self.as_any_dynamic()
    }
}

impl<T> Debug for Dynamic<T>
where
    T: CustomType + Trace,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = CollectionGuard::acquire();
        let Some(value) = self.0.load(&guard) else {
            return f.write_str("<Deallocated>");
        };
        Debug::fmt(&value.0, f)
    }
}

impl<T> From<Dynamic<T>> for AnyRef
where
    T: CustomType + Trace,
{
    fn from(value: Dynamic<T>) -> Self {
        value.0.as_any()
    }
}

// impl<T> Deref for Dynamic<T>
// where
//     T: CustomType,
// {
//     type Target = T;

//     fn deref(&self) -> &Self::Target {
//         self.dynamic.downcast_ref().expect("type checked")
//     }
// }

// impl<T> DerefMut for Dynamic<T>
// where
//     T: CustomType,
// {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         self.dynamic.downcast_mut().expect("type checked")
//     }
// }

impl<T> Clone for Dynamic<T>
where
    T: CustomType,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Dynamic<T> where T: CustomType {}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ValueFreed;

impl From<ValueFreed> for Fault {
    fn from(_value: ValueFreed) -> Self {
        Self::ValueFreed
    }
}

impl From<ValueFreed> for ExecutionError {
    fn from(_value: ValueFreed) -> Self {
        ExecutionError::Exception(Symbol::from("out-of-scope").into())
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
        self.0
            .get_or_init(|| self.1().seal(&CollectionGuard::acquire()))
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
    T: CustomType + Trace,
{
    type Target = TypeRef;

    fn deref(&self) -> &Self::Target {
        self.0
            .get_or_init(|| self.2(TypedTypeBuilder::new(self.1)).seal(&CollectionGuard::acquire()))
    }
}

#[derive(Trace)]
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
        Func: Fn(&mut VmContext<'_, '_>, Arity) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.construct = Box::new(func(self.vtable.construct));
        self
    }

    #[must_use]
    pub fn with_call<Func>(mut self, func: impl FnOnce(CallFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, Arity) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.call = Box::new(func(self.vtable.call));
        self
    }

    #[must_use]
    pub fn with_invoke<Func>(mut self, func: impl FnOnce(InvokeFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &SymbolRef, Arity) -> Result<Value, Fault>
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
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &mut ValueHasher) + Send + Sync + 'static,
    {
        self.vtable.hash = Box::new(func(self.vtable.hash));
        self
    }

    #[must_use]
    pub fn with_bitwise_not<Func>(mut self, func: impl FnOnce(UnaryFn) -> Func) -> Self
    where
        Func:
            Fn(&AnyDynamic, &mut VmContext<'_, '_>) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.bitwise_not = Box::new(func(self.vtable.bitwise_not));
        self
    }

    #[must_use]
    pub fn with_bitwise_and<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.bitwise_and = Box::new(func(self.vtable.bitwise_and));
        self
    }

    #[must_use]
    pub fn with_bitwise_or<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.bitwise_or = Box::new(func(self.vtable.bitwise_or));
        self
    }

    #[must_use]
    pub fn with_bitwise_xor<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.bitwise_xor = Box::new(func(self.vtable.bitwise_xor));
        self
    }

    #[must_use]
    pub fn with_shift_left<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.shift_left = Box::new(func(self.vtable.shift_left));
        self
    }

    #[must_use]
    pub fn with_shift_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.shift_right = Box::new(func(self.vtable.shift_right));
        self
    }

    #[must_use]
    pub fn with_negate<Func>(mut self, func: impl FnOnce(UnaryFn) -> Func) -> Self
    where
        Func:
            Fn(&AnyDynamic, &mut VmContext<'_, '_>) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.negate = Box::new(func(self.vtable.negate));
        self
    }

    #[must_use]
    pub fn with_eq<Func>(mut self, func: impl FnOnce(EqFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, ContextOrGuard<'_, '_, '_>, &Value) -> Result<bool, Fault>
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.vtable.eq);
        self.vtable.eq = Box::new(move |this, vm, rhs| func(this, vm, rhs));
        self
    }

    #[must_use]
    pub fn with_matches<Func>(mut self, func: impl FnOnce(MatchesFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<bool, Fault>
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.vtable.matches);
        self.vtable.matches = Box::new(move |this, vm, rhs| func(this, vm, rhs));
        self
    }

    #[must_use]
    pub fn with_total_cmp<Func>(mut self, func: impl FnOnce(TotalCmpFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Ordering, Fault>
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.vtable.total_cmp);
        self.vtable.total_cmp = Box::new(move |this, vm, rhs| func(this, vm, rhs));
        self
    }

    #[must_use]
    pub fn with_add<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.add = Box::new(func(self.vtable.add));
        self
    }

    #[must_use]
    pub fn with_add_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.add_right = Box::new(func(self.vtable.add_right));
        self
    }

    #[must_use]
    pub fn with_sub<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.sub = Box::new(func(self.vtable.sub));
        self
    }

    #[must_use]
    pub fn with_sub_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.sub_right = Box::new(func(self.vtable.sub_right));
        self
    }

    #[must_use]
    pub fn with_mul<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.mul = Box::new(func(self.vtable.mul));
        self
    }

    #[must_use]
    pub fn with_mul_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.mul_right = Box::new(func(self.vtable.mul_right));
        self
    }

    #[must_use]
    pub fn with_div<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.div = Box::new(func(self.vtable.div));
        self
    }

    #[must_use]
    pub fn with_div_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.div_right = Box::new(func(self.vtable.div_right));
        self
    }

    #[must_use]
    pub fn with_idiv<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.idiv = Box::new(func(self.vtable.idiv));
        self
    }

    #[must_use]
    pub fn with_idiv_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.idiv_right = Box::new(func(self.vtable.idiv_right));
        self
    }

    #[must_use]
    pub fn with_rem<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.rem = Box::new(func(self.vtable.rem));
        self
    }

    #[must_use]
    pub fn with_rem_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.rem_right = Box::new(func(self.vtable.rem_right));
        self
    }

    #[must_use]
    pub fn with_truthy<Func>(mut self, func: impl FnOnce(TruthyFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>) -> bool + Send + Sync + 'static,
    {
        self.vtable.truthy = Box::new(func(self.vtable.truthy));
        self
    }

    #[must_use]
    pub fn with_to_string<Func>(mut self, func: impl FnOnce(ToStringFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>) -> Result<SymbolRef, Fault>
            + Send
            + Sync
            + 'static,
    {
        self.vtable.to_string = Box::new(func(self.vtable.to_string));
        self
    }

    #[must_use]
    pub fn with_deep_clone<Func>(mut self, func: impl FnOnce(DeepCloneFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &CollectionGuard) -> Option<AnyDynamic>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        self.vtable.deep_clone = Box::new(func(self.vtable.deep_clone));
        self
    }

    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn with_fallback<Mapping>(mut self, mapping: Mapping) -> Self
    where
        Mapping: Fn(&AnyDynamic, &CollectionGuard) -> Value + Send + Sync + Clone + 'static,
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
                    mapping(this, vm.as_ref()).call(vm, arity)
                }
                Err(other) => Err(other),
            }
        });

        bitwise_not = Box::new({
            let mapping = mapping.clone();
            move |this, vm| match bitwise_not(this, vm) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).bitwise_not(vm)
                }
                Err(other) => Err(other),
            }
        });

        bitwise_and = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match bitwise_and(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).bitwise_and(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        bitwise_or = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match bitwise_or(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).bitwise_or(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        bitwise_xor = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match bitwise_xor(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).bitwise_xor(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        shift_left = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match shift_left(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).shift_left(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        shift_right = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match shift_right(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).shift_right(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        negate = Box::new({
            let mapping = mapping.clone();
            move |this, vm| match negate(this, vm) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).negate(vm)
                }
                Err(other) => Err(other),
            }
        });

        eq = Box::new({
            let mapping = mapping.clone();
            move |this, mut vm, rhs| match eq(this, vm.borrowed(), rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).equals(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        matches = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match matches(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).matches(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        total_cmp = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match total_cmp(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).total_cmp(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        invoke = Box::new({
            let mapping = mapping.clone();
            move |this, vm, name, arity| match invoke(this, vm, name, arity) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).invoke(vm, name, arity)
                }
                Err(other) => Err(other),
            }
        });

        add = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match add(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).add(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        add_right = Box::new({
            let mapping = mapping.clone();
            move |this, vm, lhs| match add_right(this, vm, lhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    lhs.add(vm, &mapping(this, vm.as_ref()))
                }
                Err(other) => Err(other),
            }
        });

        sub = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match sub(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).sub(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        sub_right = Box::new({
            let mapping = mapping.clone();
            move |this, vm, lhs| match sub_right(this, vm, lhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    lhs.sub(vm, &mapping(this, vm.as_ref()))
                }
                Err(other) => Err(other),
            }
        });

        mul = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match mul(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).mul(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        mul_right = Box::new({
            let mapping = mapping.clone();
            move |this, vm, lhs| match mul_right(this, vm, lhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    lhs.mul(vm, &mapping(this, vm.as_ref()))
                }
                Err(other) => Err(other),
            }
        });

        div = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match div(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).div(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        div_right = Box::new({
            let mapping = mapping.clone();
            move |this, vm, lhs| match div_right(this, vm, lhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    lhs.div(vm, &mapping(this, vm.as_ref()))
                }
                Err(other) => Err(other),
            }
        });

        idiv = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match idiv(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).idiv(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        idiv_right = Box::new({
            let mapping = mapping.clone();
            move |this, vm, lhs| match idiv_right(this, vm, lhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    lhs.idiv(vm, &mapping(this, vm.as_ref()))
                }
                Err(other) => Err(other),
            }
        });

        rem = Box::new({
            let mapping = mapping.clone();
            move |this, vm, rhs| match rem(this, vm, rhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).rem(vm, rhs)
                }
                Err(other) => Err(other),
            }
        });

        rem_right = Box::new({
            let mapping = mapping.clone();
            move |this, vm, lhs| match rem_right(this, vm, lhs) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    lhs.rem(vm, &mapping(this, vm.as_ref()))
                }
                Err(other) => Err(other),
            }
        });

        to_string = Box::new({
            let mapping = mapping.clone();
            move |this, vm| match to_string(this, vm) {
                Ok(value) => Ok(value),
                Err(Fault::UnknownSymbol | Fault::UnsupportedOperation) => {
                    mapping(this, vm.as_ref()).to_string(vm)
                }
                Err(other) => Err(other),
            }
        });

        deep_clone = Box::new({
            let mapping = mapping.clone();
            move |this, guard| match deep_clone(this, guard) {
                Some(value) => Some(value),
                None => mapping(this, guard)
                    .deep_clone(guard)
                    .and_then(|value| value.as_any_dynamic()),
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
    pub fn seal(self, guard: &CollectionGuard) -> TypeRef {
        TypeRef::new(self, guard)
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
    T: CustomType + Trace,
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
        Func:
            Fn(&mut VmContext<'_, '_>, Arity) -> Result<Dynamic<T>, Fault> + Send + Sync + 'static,
    {
        let func = func(self.t.vtable.construct);
        self.t.vtable.construct =
            Box::new(move |vm, arity| func(vm, arity).map(Dynamic::into_value));
        self
    }

    #[must_use]
    pub fn with_call<Func>(mut self, func: impl FnOnce(CallFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, Arity) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.call);
        self.t.vtable.call = Box::new(move |this, vm, arity| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, arity)
        });
        self
    }

    #[must_use]
    pub fn with_invoke<Func>(mut self, func: impl FnOnce(InvokeFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &SymbolRef, Arity) -> Result<Value, Fault>
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.invoke);
        self.t.vtable.invoke = Box::new(move |this, vm, name, arity| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, name, arity)
        });
        self
    }

    #[must_use]
    pub fn with_hash<Func>(mut self, func: impl FnOnce(HashFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &mut ValueHasher)
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.hash);
        self.t.vtable.hash = Box::new(move |this, vm, hasher| {
            let Some(this) = this.as_rooted::<T>(vm.as_ref()) else {
                return;
            };
            func(this, vm, hasher);
        });
        self
    }

    #[must_use]
    pub fn with_bitwise_not<Func>(mut self, func: impl FnOnce(UnaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.bitwise_not);
        self.t.vtable.bitwise_not = Box::new(move |this, vm| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm)
        });
        self
    }

    #[must_use]
    pub fn with_bitwise_and<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.bitwise_and);
        self.t.vtable.bitwise_and = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_bitwise_or<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.bitwise_or);
        self.t.vtable.bitwise_or = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_bitwise_xor<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.bitwise_xor);
        self.t.vtable.bitwise_xor = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_shift_left<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.shift_left);
        self.t.vtable.shift_left = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_shift_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.shift_right);
        self.t.vtable.shift_right = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_negate<Func>(mut self, func: impl FnOnce(UnaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.negate);
        self.t.vtable.negate = Box::new(move |this, vm| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm)
        });
        self
    }

    #[must_use]
    pub fn with_eq<Func>(mut self, func: impl FnOnce(EqFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, ContextOrGuard<'_, '_, '_>, &Value) -> Result<bool, Fault>
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.eq);
        self.t.vtable.eq = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_matches<Func>(mut self, func: impl FnOnce(MatchesFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<bool, Fault>
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.matches);
        self.t.vtable.matches = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_total_cmp<Func>(mut self, func: impl FnOnce(TotalCmpFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Ordering, Fault>
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.total_cmp);
        self.t.vtable.total_cmp = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_add<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.add);
        self.t.vtable.add = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_add_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.add_right);
        self.t.vtable.add_right = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_sub<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.sub);
        self.t.vtable.sub = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_sub_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.sub_right);
        self.t.vtable.sub_right = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_mul<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.mul);
        self.t.vtable.mul = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_mul_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.mul_right);
        self.t.vtable.mul_right = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_div<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.div);
        self.t.vtable.div = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_div_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.div_right);
        self.t.vtable.div_right = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_idiv<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.idiv);
        self.t.vtable.idiv = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_idiv_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.idiv_right);
        self.t.vtable.idiv_right = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_rem<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.rem);
        self.t.vtable.rem = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_rem_right<Func>(mut self, func: impl FnOnce(BinaryFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.rem_right);
        self.t.vtable.rem_right = Box::new(move |this, vm, rhs| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm, rhs)
        });
        self
    }

    #[must_use]
    pub fn with_truthy<Func>(mut self, func: impl FnOnce(TruthyFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>) -> bool + Send + Sync + 'static,
    {
        let func = func(self.t.vtable.truthy);
        self.t.vtable.truthy = Box::new(move |this, vm| {
            let Some(this) = this.as_rooted::<T>(vm.as_ref()) else {
                return true;
            };
            func(this, vm)
        });
        self
    }

    #[must_use]
    pub fn with_to_string<Func>(mut self, func: impl FnOnce(ToStringFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &mut VmContext<'_, '_>) -> Result<SymbolRef, Fault>
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.to_string);
        self.t.vtable.to_string = Box::new(move |this, vm| {
            let this = this
                .as_rooted::<T>(vm.as_ref())
                .ok_or(Fault::UnsupportedOperation)?;
            func(this, vm)
        });
        self
    }

    #[must_use]
    pub fn with_deep_clone<Func>(mut self, func: impl FnOnce(DeepCloneFn) -> Func) -> Self
    where
        Func: Fn(Rooted<T>, &CollectionGuard) -> Option<AnyDynamic>
            + Send
            + Sync
            + Send
            + Sync
            + 'static,
    {
        let func = func(self.t.vtable.deep_clone);
        self.t.vtable.deep_clone = Box::new(move |this, guard| {
            let this = this.as_rooted::<T>(guard)?;
            func(this, guard)
        });
        self
    }

    #[must_use]
    pub fn with_clone(self) -> Self
    where
        T: Clone,
    {
        self.with_deep_clone(|_| |this, guard| Some(AnyDynamic::new((*this).clone(), guard)))
    }

    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn with_fallback<Mapping>(mut self, mapping: Mapping) -> Self
    where
        Mapping: Fn(Rooted<T>, &CollectionGuard) -> Value + Send + Sync + Clone + 'static,
    {
        self.t = self.t.with_fallback(move |dynamic, guard| {
            dynamic
                .as_rooted::<T>(guard)
                .map_or(Value::Nil, |rooted| mapping(rooted, guard))
        });
        self
    }

    fn seal(self, guard: &CollectionGuard) -> TypeRef {
        self.t.seal(guard)
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

pub enum ContextOrGuard<'a, 'context, 'guard> {
    Guard(&'a CollectionGuard<'guard>),
    Context(&'a mut VmContext<'context, 'guard>),
}

impl<'context, 'guard> AsRef<CollectionGuard<'guard>> for ContextOrGuard<'_, 'context, 'guard> {
    fn as_ref(&self) -> &CollectionGuard<'guard> {
        match self {
            ContextOrGuard::Guard(guard) => guard,
            ContextOrGuard::Context(context) => context.as_ref(),
        }
    }
}

impl<'a, 'context, 'guard> ContextOrGuard<'a, 'context, 'guard> {
    pub fn vm(&mut self) -> Option<&mut VmContext<'context, 'guard>> {
        let Self::Context(context) = self else {
            return None;
        };
        Some(context)
    }

    pub fn borrowed(&mut self) -> ContextOrGuard<'_, 'context, 'guard> {
        match self {
            ContextOrGuard::Guard(guard) => ContextOrGuard::Guard(guard),
            ContextOrGuard::Context(vm) => ContextOrGuard::Context(vm),
        }
    }
}

pub type TypeRef = Rooted<Type>;

pub type ConstructFn =
    Box<dyn Fn(&mut VmContext<'_, '_>, Arity) -> Result<Value, Fault> + Send + Sync>;
pub type CallFn =
    Box<dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>, Arity) -> Result<Value, Fault> + Send + Sync>;
pub type HashFn = Box<dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>, &mut ValueHasher) + Send + Sync>;
pub type UnaryFn =
    Box<dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>) -> Result<Value, Fault> + Send + Sync>;
pub type BinaryFn =
    Box<dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault> + Send + Sync>;
pub type MatchesFn =
    Box<dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<bool, Fault> + Send + Sync>;
pub type EqFn = Box<
    dyn Fn(&AnyDynamic, ContextOrGuard<'_, '_, '_>, &Value) -> Result<bool, Fault> + Send + Sync,
>;
pub type TotalCmpFn = Box<
    dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Ordering, Fault> + Send + Sync,
>;
pub type InvokeFn = Box<
    dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>, &SymbolRef, Arity) -> Result<Value, Fault>
        + Send
        + Sync,
>;
pub type DeepCloneFn =
    Box<dyn Fn(&AnyDynamic, &CollectionGuard) -> Option<AnyDynamic> + Send + Sync>;
pub type TruthyFn = Box<dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>) -> bool + Send + Sync>;
pub type ToStringFn =
    Box<dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>) -> Result<SymbolRef, Fault> + Send + Sync>;

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
            hash: Box::new(|this, _vm, hasher| this.0.hash(hasher)),
            bitwise_not: Box::new(|_this, _vm| Err(Fault::UnsupportedOperation)),
            bitwise_and: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            bitwise_or: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            bitwise_xor: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            shift_left: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            shift_right: Box::new(|_this, _vm, _rhs| Err(Fault::UnsupportedOperation)),
            negate: Box::new(|_this, _vm| Err(Fault::UnsupportedOperation)),
            eq: Box::new(|_this, _vm, _rhs| Ok(false)),
            matches: Box::new(|this, vm, rhs| this.eq(ContextOrGuard::Context(vm), rhs)),
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
            deep_clone: Box::new(|_this, _guard| None),
        }
    }
}

impl ContainsNoRefs for TypeVtable {}

pub trait CustomType: Send + Sync + Debug + 'static {
    fn muse_type(&self) -> &TypeRef;
}

struct Custom<T>(T);

impl<T> Trace for Custom<T>
where
    T: Trace,
{
    const MAY_CONTAIN_REFERENCES: bool = T::MAY_CONTAIN_REFERENCES;

    fn trace(&self, tracer: &mut refuse::Tracer) {
        self.0.trace(tracer);
    }
}

impl<T> MapAs for Custom<T>
where
    T: CustomType,
{
    type Target = dyn CustomType;

    fn map_as(&self) -> &Self::Target {
        &self.0
    }
}

pub struct RustFunctionTable<T> {
    functions: Map<Symbol, Map<Arity, Arc<dyn RustFn<T>>>>,
}

impl<T> RustFunctionTable<T>
where
    T: CustomType + Trace,
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
        F: Fn(&mut VmContext<'_, '_>, &Rooted<T>) -> Result<Value, Fault> + Send + Sync + 'static,
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
        vm: &mut VmContext<'_, '_>,
        name: &SymbolRef,
        arity: Arity,
        this: &Rooted<T>,
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
    T: CustomType + Trace,
{
    type Target = RustFunctionTable<T>;

    fn deref(&self) -> &Self::Target {
        self.0.get_or_init(|| self.1(RustFunctionTable::new()))
    }
}

trait RustFn<T>: Send + Sync + 'static
where
    T: CustomType + Trace,
{
    fn invoke(&self, vm: &mut VmContext<'_, '_>, this: &Rooted<T>) -> Result<Value, Fault>;
}

impl<T, F> RustFn<T> for F
where
    F: Fn(&mut VmContext<'_, '_>, &Rooted<T>) -> Result<Value, Fault> + Send + Sync + 'static,
    T: CustomType + Trace,
{
    fn invoke(&self, vm: &mut VmContext<'_, '_>, this: &Rooted<T>) -> Result<Value, Fault> {
        self(vm, this)
    }
}

type ArcRustFn = Arc<dyn Fn(&mut VmContext<'_, '_>, Arity) -> Result<Value, Fault> + Send + Sync>;

#[derive(Clone)]
pub struct RustFunction(ArcRustFn);

impl RustFunction {
    pub fn new<F>(function: F) -> Self
    where
        F: Fn(&mut VmContext<'_, '_>, Arity) -> Result<Value, Fault> + Send + Sync + 'static,
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
                    let result = (*this).0(vm, arity)?;
                    vm.exit_frame()?;
                    Ok(result)
                }
            })
        });
        &TYPE
    }
}

impl ContainsNoRefs for RustFunction {}

type ArcAsyncFunction = Arc<
    dyn Fn(
            &mut VmContext<'_, '_>,
            Arity,
        ) -> Pin<Box<dyn Future<Output = Result<Value, Fault>> + Send + Sync>>
        + Send
        + Sync,
>;

#[derive(Clone)]
pub struct AsyncFunction(ArcAsyncFunction);

impl AsyncFunction {
    pub fn new<F, Fut>(function: F) -> Self
    where
        F: Fn(&mut VmContext<'_, '_>, Arity) -> Fut + Send + Sync + 'static,
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
                        let future = (*this).0(vm, arity);
                        vm.allocate(1)?;
                        vm.current_frame_mut()[0] = Value::dynamic(
                            ValueFuture(Arc::new(Mutex::new(Box::pin(future)))),
                            &vm,
                        );
                        vm.jump_to(1);
                    }

                    let Some(future) = vm.current_frame()[0]
                        .as_any_dynamic()
                        .and_then(|d| d.downcast_ref::<ValueFuture>(vm.as_ref()))
                    else {
                        unreachable!("missing future")
                    };

                    let mut future = future.0.lock();
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

impl ContainsNoRefs for AsyncFunction {}

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

impl ContainsNoRefs for ValueFuture {}

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

#[test]
fn dynamic() {
    impl CustomType for usize {
        fn muse_type(&self) -> &TypeRef {
            static TYPE: RustType<usize> = RustType::new("usize", TypedTypeBuilder::with_clone);
            &TYPE
        }
    }
    let guard = CollectionGuard::acquire();
    let dynamic = AnyDynamic::new(1_usize, &guard);
    assert_eq!(dynamic.downcast_ref::<usize>(&guard), Some(&1));
    let dynamic2 = dynamic;
    assert_eq!(dynamic, dynamic2);
}

#[test]
fn functions() {
    let mut guard = CollectionGuard::acquire();
    let func = Value::Dynamic(AnyDynamic::new(
        RustFunction::new(|_vm: &mut VmContext<'_, '_>, _arity| Ok(Value::Int(1))),
        &guard,
    ));
    let runtime = crate::vm::Vm::new(&guard);
    let Value::Int(i) = func
        .call(&mut VmContext::new(&runtime, &mut guard), 0)
        .unwrap()
    else {
        unreachable!()
    };
    assert_eq!(i, 1);
}
