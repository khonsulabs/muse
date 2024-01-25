use std::any::Any;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use ahash::AHasher;

use crate::symbol::Symbol;
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
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            #[allow(clippy::cast_precision_loss)]
            Value::Int(value) => Some(*value as f64),
            Value::Float(value) => Some(*value),
            _ => None,
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
            Value::Dynamic(dynamic) => {
                vm.enter_anonymous_frame()?;
                let result = dynamic.call(vm, arity)?;
                vm.exit_frame()?;
                Ok(result)
            }
            Value::Nil => vm.recurse_current_function(arity.into()),
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
                self.add(vm, &rhs)
            }
            (Value::Dynamic(dynamic), _) => dynamic.0.invoke(vm, name),
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

            (Value::Dynamic(lhs), rhs) => lhs.add(vm, rhs.clone()),
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

    pub fn div_i(&self, vm: &mut Vm, rhs: &Self) -> Result<Value, Fault> {
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

            (Value::Dynamic(lhs), rhs) => lhs.divi(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.divi_right(vm, lhs),
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

    pub fn to_string(&self, vm: &mut Vm) -> Result<Symbol, Fault> {
        match self {
            Value::Nil => Ok(Symbol::empty()),
            Value::Bool(bool) => Ok(Symbol::from(*bool)),
            Value::Int(value) => Ok(Symbol::from(value.to_string())),
            Value::UInt(value) => Ok(Symbol::from(value.to_string())),
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
            Value::Bool(b) => b.hash(&mut hasher),
            Value::Int(i) => i.hash(&mut hasher),
            Value::UInt(i) => i.hash(&mut hasher),
            Value::Float(f) => f.to_bits().hash(&mut hasher),
            Value::Symbol(s) => s.hash(&mut hasher),
            Value::Dynamic(d) => d.hash(vm).hash(&mut hasher),
        }
        Ok(hasher.finish())
    }

    pub fn eq(&self, vm: &mut Vm, other: &Self) -> bool {
        match (self, other) {
            (Self::Nil, Self::Nil) => true,

            (Self::Bool(l0), Self::Bool(r0)) => l0 == r0,
            (Self::Bool(b), Self::Int(i)) | (Self::Int(i), Self::Bool(b)) => &i64::from(*b) == i,
            (Self::Bool(b), Self::Float(f)) | (Self::Float(f), Self::Bool(b)) => {
                (f64::from(u8::from(*b)) - f).abs() < f64::EPSILON
            }

            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
            (Self::Int(signed), Self::UInt(unsigned))
            | (Self::UInt(unsigned), Self::Int(signed)) => {
                u64::try_from(*signed).map_or(false, |signed| &signed == unsigned)
            }
            (Self::UInt(l0), Self::UInt(r0)) => l0 == r0,
            (Self::Float(l0), Self::Float(r0)) => (l0 - r0).abs() < f64::EPSILON,
            #[allow(clippy::cast_precision_loss)]
            (Self::Int(i), Self::Float(f)) | (Self::Float(f), Self::Int(i)) => {
                (*i as f64 - f).abs() < f64::EPSILON
            }
            #[allow(clippy::cast_precision_loss)]
            (Self::UInt(i), Self::Float(f)) | (Self::Float(f), Self::UInt(i)) => {
                (*i as f64 - f).abs() < f64::EPSILON
            }

            (Self::Symbol(l0), Self::Symbol(r0)) => l0 == r0,
            (Self::Symbol(s), Self::Bool(b)) | (Self::Bool(b), Self::Symbol(s)) => {
                s == &Symbol::from(*b)
            }

            (Self::Dynamic(l0), _) => l0.eq(vm, other),
            (_, Self::Dynamic(r0)) => r0.eq(vm, self),

            _ => false,
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
            *self = (**self.0).deep_clone();
        }

        let dynamic = &mut *Arc::get_mut(&mut self.0).expect("always 1 ref");
        dynamic.as_any_mut().downcast_mut()
    }

    #[must_use]
    pub fn ptr_eq(a: &Dynamic, b: &Dynamic) -> bool {
        Arc::ptr_eq(&a.0, &b.0)
    }

    pub fn call(&self, vm: &mut Vm, arity: impl Into<Arity>) -> Result<Value, Fault> {
        self.0.call(vm, self, arity.into())
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

    pub fn truthy(&self, vm: &mut Vm) -> bool {
        self.0.truthy(vm)
    }

    pub fn eq(&self, vm: &mut Vm, rhs: &Value) -> bool {
        match rhs {
            Value::Dynamic(dynamic) if Arc::ptr_eq(&self.0, &dynamic.0) => true,
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

pub trait CustomType: Debug + 'static {
    #[allow(unused_variables)]
    fn call(&self, vm: &mut Vm, this: &Dynamic, arity: Arity) -> Result<Value, Fault> {
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
    fn total_cmp(&self, vm: &mut Vm, rhs: &Value) -> Result<Ordering, Fault> {
        if rhs.as_dynamic().is_none() {
            // Dynamics sort after primitive values
            Ok(Ordering::Greater)
        } else {
            Err(Fault::UnsupportedOperation)
        }
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
    fn truthy(&self, vm: &mut Vm) -> bool {
        true
    }

    #[allow(unused_variables)]
    fn to_string(&self, vm: &mut Vm) -> Result<Symbol, Fault> {
        Err(Fault::UnsupportedOperation)
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
    F: Clone + Fn(&mut Vm, Arity) -> Result<Value, Fault> + 'static,
{
    fn call(&self, vm: &mut Vm, _this: &Dynamic, arity: Arity) -> Result<Value, Fault> {
        self.0(vm, arity)
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
    F: Clone + Fn(&mut Vm, Arity) -> Fut + 'static,
    Fut: Future<Output = Result<Value, Fault>> + 'static,
{
    fn call(&self, vm: &mut Vm, _this: &Dynamic, arity: Arity) -> Result<Value, Fault> {
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
            Poll::Ready(result) => result,
            Poll::Pending => Err(Fault::Waiting),
        }
    }
}

struct ValueFuture<F>(Arc<Mutex<Pin<Box<F>>>>);

impl<F> Clone for ValueFuture<F> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<F> CustomType for ValueFuture<F> where F: 'static {}

impl<F> Debug for ValueFuture<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValueFuture").finish_non_exhaustive()
    }
}

pub trait DynamicValue: CustomType {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn deep_clone(&self) -> Dynamic;
}

impl<T> DynamicValue for T
where
    T: Clone + CustomType,
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
    let func = Value::Dynamic(Dynamic::new(RustFunction::new(|_vm: &mut Vm, _arity| {
        Ok(Value::Int(1))
    })));
    let mut runtime = Vm::default();
    let Value::Int(i) = func.call(&mut runtime, 0).unwrap() else {
        unreachable!()
    };
    assert_eq!(i, 1);
}
