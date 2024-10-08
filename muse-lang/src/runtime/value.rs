//! Types for representing data in the Muse runtime.

use std::any::Any;
use std::cmp::Ordering;
use std::fmt::{self, Debug};
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};

/// The [`Hash`] used by Muse when hashing values.
pub type ValueHasher = ahash::AHasher;

use kempt::Map;
use parking_lot::Mutex;
use refuse::{AnyRef, AnyRoot, CollectionGuard, ContainsNoRefs, MapAs, Ref, Root, Trace};
use serde::{Deserialize, Serialize};

use crate::runtime::string::MuseString;
use crate::runtime::symbol::{Symbol, SymbolList, SymbolRef};
#[cfg(feature = "dispatched")]
use crate::runtime::types::{RuntimeEnum, RuntimeStruct};
#[cfg(feature = "dispatched")]
use crate::vm::bitcode::{BitcodeFunction, ValueOrSource};
#[cfg(feature = "dispatched")]
use crate::vm::Function;
use crate::vm::{Arity, ExecutionError, Fault, VmContext};

/// A primitive virtual machine value.
#[derive(Default, Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Primitive {
    /// A value representing nothing.
    #[default]
    Nil,
    /// A boolean value.
    Bool(bool),
    /// A signed 64-bit integer.
    Int(i64),
    /// An unsigned 64-bit integer.
    UInt(u64),
    /// A double-preceision floating point number.
    Float(f64),
}

impl Primitive {
    /// Returns true if this value is nil.
    #[must_use]
    pub const fn is_nil(&self) -> bool {
        matches!(self, Self::Nil)
    }

    /// Returns this value as an i64, if possible.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(value) => Some(*value),
            Self::UInt(value) => i64::try_from(*value).ok(),
            #[allow(clippy::cast_possible_truncation)]
            Self::Float(value) => Some(*value as i64),
            _ => None,
        }
    }

    /// Returns this value as an u64, if possible.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Int(value) => u64::try_from(*value).ok(),
            Self::UInt(value) => Some(*value),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Self::Float(value) => Some(*value as u64),
            _ => None,
        }
    }

    /// Returns this value as an u32, if possible.
    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Int(value) => u32::try_from(*value).ok(),
            Self::UInt(value) => u32::try_from(*value).ok(),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Self::Float(value) => Some(*value as u32),
            _ => None,
        }
    }

    /// Returns this value as an u16, if possible.
    #[must_use]
    pub fn as_u16(&self) -> Option<u16> {
        match self {
            Self::Int(value) => u16::try_from(*value).ok(),
            Self::UInt(value) => u16::try_from(*value).ok(),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Self::Float(value) => Some(*value as u16),
            _ => None,
        }
    }

    /// Returns this value as an usize, if possible.
    #[must_use]
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Self::Int(value) => usize::try_from(*value).ok(),
            Self::UInt(value) => usize::try_from(*value).ok(),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Self::Float(value) => Some(*value as usize),
            _ => None,
        }
    }

    /// Returns this value as an f64, if possible.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            #[allow(clippy::cast_precision_loss)]
            Self::Int(value) => Some(*value as f64),
            Self::Float(value) => Some(*value),
            _ => None,
        }
    }

    /// Converts this value to an i64, if possible.
    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        match self {
            Self::Int(value) => Some(*value),
            Self::UInt(value) => i64::try_from(*value).ok(),
            #[allow(clippy::cast_possible_truncation)]
            Self::Float(value) => Some(*value as i64),
            Self::Nil => Some(0),
            Self::Bool(bool) => Some(i64::from(*bool)),
        }
    }

    /// Converts this value to an u64, if possible.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        match self {
            Self::Int(value) => u64::try_from(*value).ok(),
            Self::UInt(value) => Some(*value),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Self::Float(value) => Some(*value as u64),
            Self::Nil => Some(0),
            Self::Bool(bool) => Some(u64::from(*bool)),
        }
    }

    /// Converts this value to an u32, if possible.
    #[must_use]
    pub fn to_u32(&self) -> Option<u32> {
        match self {
            Self::Int(value) => u32::try_from(*value).ok(),
            Self::UInt(value) => u32::try_from(*value).ok(),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Self::Float(value) => Some(*value as u32),
            Self::Nil => Some(0),
            Self::Bool(bool) => Some(u32::from(*bool)),
        }
    }

    /// Converts this value to an usize, if possible.
    #[must_use]
    pub fn to_usize(&self) -> Option<usize> {
        match self {
            Self::Int(value) => usize::try_from(*value).ok(),
            Self::UInt(value) => usize::try_from(*value).ok(),
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Self::Float(value) => Some(*value as usize),
            Self::Nil => Some(0),
            Self::Bool(bool) => Some(usize::from(*bool)),
        }
    }

    /// Converts this value to an f64, if possible.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        match self {
            #[allow(clippy::cast_precision_loss)]
            Self::Int(value) => *value as f64,
            #[allow(clippy::cast_precision_loss)]
            Self::UInt(value) => *value as f64,
            Self::Float(value) => *value,
            Self::Nil => 0.,
            Self::Bool(bool) => f64::from(*bool),
        }
    }

    /// Returns true if this value should be considered `true` in a boolean
    /// expression.
    pub fn truthy(&self) -> bool {
        match self {
            Self::Nil => false,
            Self::Bool(value) => *value,
            Self::Int(value) => value != &0,
            Self::UInt(value) => value != &0,
            Self::Float(value) => value.abs() >= f64::EPSILON,
        }
    }

    /// Adds `self` to `rhs`.
    pub fn add(&self, rhs: &Self) -> Result<Self, Fault> {
        match (self, rhs) {
            (Self::Nil, _) | (_, Self::Nil) => Err(Fault::OperationOnNil),
            (Self::Bool(lhs), Self::Bool(rhs)) => Ok(Self::Bool(*lhs || *rhs)),
            (Self::Bool(_), _) | (_, Self::Bool(_)) => Err(Fault::UnsupportedOperation),

            (Self::Int(lhs), Self::Int(rhs)) => Ok(Self::Int(lhs.saturating_add(*rhs))),
            (Self::Int(lhs), Self::UInt(rhs)) => Ok(Self::Int(lhs.saturating_add_unsigned(*rhs))),
            (Self::UInt(lhs), Self::UInt(rhs)) => Ok(Self::UInt(lhs.saturating_add(*rhs))),
            (Self::UInt(lhs), Self::Int(rhs)) => Ok(Self::UInt(lhs.saturating_add_signed(*rhs))),

            #[allow(clippy::cast_precision_loss)]
            (Self::Int(lhs), Self::Float(rhs)) => Ok(Self::Float(*lhs as f64 + rhs)),
            #[allow(clippy::cast_precision_loss)]
            (Self::Float(lhs), Self::Int(rhs)) => Ok(Self::Float(lhs + *rhs as f64)),
            #[allow(clippy::cast_precision_loss)]
            (Self::UInt(lhs), Self::Float(rhs)) => Ok(Self::Float(*lhs as f64 + rhs)),
            #[allow(clippy::cast_precision_loss)]
            (Self::Float(lhs), Self::UInt(rhs)) => Ok(Self::Float(lhs + *rhs as f64)),
            (Self::Float(lhs), Self::Float(rhs)) => Ok(Self::Float(lhs + rhs)),
        }
    }

    /// Subtracts `rhs` from `self`.
    pub fn sub(&self, rhs: &Self) -> Result<Self, Fault> {
        match (self, rhs) {
            (Self::Nil, _) | (_, Self::Nil) => Err(Fault::OperationOnNil),

            (Self::Int(lhs), Self::Int(rhs)) => Ok(Self::Int(lhs.saturating_sub(*rhs))),
            (Self::Int(lhs), Self::UInt(rhs)) => Ok(Self::Int(lhs.saturating_sub_unsigned(*rhs))),
            (Self::UInt(lhs), Self::UInt(rhs)) => Ok(Self::UInt(lhs.saturating_sub(*rhs))),
            (Self::UInt(lhs), Self::Int(rhs)) => {
                Ok(Self::UInt(lhs.saturating_add_signed(rhs.saturating_neg())))
            }

            #[allow(clippy::cast_precision_loss)]
            (Self::Int(lhs), Self::Float(rhs)) => Ok(Self::Float(*lhs as f64 - rhs)),
            #[allow(clippy::cast_precision_loss)]
            (Self::Float(lhs), Self::Int(rhs)) => Ok(Self::Float(lhs - *rhs as f64)),
            #[allow(clippy::cast_precision_loss)]
            (Self::UInt(lhs), Self::Float(rhs)) => Ok(Self::Float(*lhs as f64 - rhs)),
            #[allow(clippy::cast_precision_loss)]
            (Self::Float(lhs), Self::UInt(rhs)) => Ok(Self::Float(lhs - *rhs as f64)),
            (Self::Float(lhs), Self::Float(rhs)) => Ok(Self::Float(lhs - rhs)),

            _ => Err(Fault::UnsupportedOperation),
        }
    }

    /// Multiplies `self` by `rhs`.
    pub fn mul(&self, rhs: &Self) -> Result<Self, Fault> {
        match (self, rhs) {
            (Self::Nil, _) | (_, Self::Nil) => Err(Fault::OperationOnNil),

            (Self::Int(lhs), Self::Int(rhs)) => Ok(Self::Int(lhs.saturating_mul(*rhs))),
            (Self::Int(lhs), Self::UInt(rhs)) => Ok(Self::Int(
                lhs.saturating_mul(i64::try_from(*rhs).unwrap_or(i64::MAX)),
            )),
            (Self::UInt(lhs), Self::Int(rhs)) => {
                Ok(Self::UInt(if let Ok(rhs) = u64::try_from(*rhs) {
                    lhs.saturating_mul(rhs)
                } else {
                    0
                }))
            }
            (Self::UInt(lhs), Self::UInt(rhs)) => Ok(Self::UInt(lhs.saturating_mul(*rhs))),

            #[allow(clippy::cast_precision_loss)]
            (Self::Int(lhs), Self::Float(rhs)) => Ok(Self::Float(*lhs as f64 * rhs)),
            #[allow(clippy::cast_precision_loss)]
            (Self::Float(lhs), Self::Int(rhs)) => Ok(Self::Float(lhs * *rhs as f64)),
            (Self::Float(lhs), Self::Float(rhs)) => Ok(Self::Float(lhs * rhs)),

            _ => Err(Fault::UnsupportedOperation),
        }
    }

    /// Raises `self` to the `rhs` power.
    pub fn pow(&self, exp: &Self) -> Result<Self, Fault> {
        match (self, exp) {
            (Self::Nil, _) | (_, Self::Nil) => Err(Fault::OperationOnNil),

            (Self::Int(lhs), Self::Int(rhs)) => Ok(if rhs.is_negative() {
                #[allow(clippy::cast_precision_loss)]
                Self::Float(powf64_i64(*lhs as f64, *rhs))
            } else {
                Self::Int(lhs.saturating_pow(u32::try_from(*rhs).unwrap_or(u32::MAX)))
            }),
            (Self::Int(lhs), Self::UInt(rhs)) => Ok(Self::Int(
                lhs.saturating_pow(u32::try_from(*rhs).unwrap_or(u32::MAX)),
            )),
            (Self::UInt(lhs), Self::Int(rhs)) => Ok(if rhs.is_negative() {
                #[allow(clippy::cast_precision_loss)]
                Self::Float(powf64_i64(*lhs as f64, *rhs))
            } else {
                Self::UInt(lhs.saturating_pow(u32::try_from(*rhs).unwrap_or(u32::MAX)))
            }),
            (Self::UInt(lhs), Self::UInt(rhs)) => Ok(Self::UInt(
                lhs.saturating_pow(u32::try_from(*rhs).unwrap_or(u32::MAX)),
            )),

            #[allow(clippy::cast_precision_loss)]
            (Self::Int(lhs), Self::Float(rhs)) => Ok(Self::Float((*lhs as f64).powf(*rhs))),
            #[allow(clippy::cast_precision_loss)]
            (Self::Float(lhs), Self::Int(rhs)) => Ok(Self::Float(powf64_i64(*lhs, *rhs))),
            #[allow(clippy::cast_precision_loss)]
            (Self::UInt(lhs), Self::Float(rhs)) => Ok(Self::Float((*lhs as f64).powf(*rhs))),
            #[allow(clippy::cast_precision_loss)]
            (Self::Float(lhs), Self::UInt(rhs)) => Ok(Self::Float(powf64_u64(*lhs, *rhs))),
            (Self::Float(lhs), Self::Float(rhs)) => Ok(Self::Float(lhs * rhs)),

            _ => Err(Fault::UnsupportedOperation),
        }
    }

    /// Divides `self` by `rhs`.
    pub fn div(&self, rhs: &Self) -> Result<Self, Fault> {
        match (self, rhs) {
            (Self::Nil, _) | (_, Self::Nil) => Err(Fault::OperationOnNil),

            #[allow(clippy::cast_precision_loss)]
            (Self::Int(lhs), Self::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::Float(*lhs as f64 / *rhs as f64))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_precision_loss)]
            (Self::UInt(lhs), Self::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::Float(*lhs as f64 / *rhs as f64))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_precision_loss)]
            (Self::Int(lhs), Self::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::Float(*lhs as f64 / *rhs as f64))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_precision_loss)]
            (Self::UInt(lhs), Self::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::Float(*lhs as f64 / *rhs as f64))
                } else {
                    Err(Fault::DivideByZero)
                }
            }

            #[allow(clippy::cast_precision_loss)]
            (Self::Int(lhs), Self::Float(rhs)) => Ok(Self::Float(*lhs as f64 / rhs)),
            #[allow(clippy::cast_precision_loss)]
            (Self::UInt(lhs), Self::Float(rhs)) => Ok(Self::Float(*lhs as f64 / rhs)),
            #[allow(clippy::cast_precision_loss)]
            (Self::Float(lhs), Self::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::Float(lhs / *rhs as f64))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_precision_loss)]
            (Self::Float(lhs), Self::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::Float(lhs / *rhs as f64))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            (Self::Float(lhs), Self::Float(rhs)) => Ok(Self::Float(lhs / rhs)),

            _ => Err(Fault::UnsupportedOperation),
        }
    }

    /// Divides `self` by `rhs`, using whole numbers.
    pub fn idiv(&self, rhs: &Self) -> Result<Self, Fault> {
        match (self, rhs) {
            (Self::Nil, _) | (_, Self::Nil) => Err(Fault::OperationOnNil),

            (Self::Int(lhs), Self::Int(rhs)) => Ok(Self::Int(lhs.saturating_div(*rhs))),
            (Self::Int(lhs), Self::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::Int(
                        lhs.saturating_div(i64::try_from(*rhs).unwrap_or(i64::MAX)),
                    ))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            (Self::UInt(lhs), Self::Int(rhs)) => {
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
            (Self::UInt(lhs), Self::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::UInt(lhs.saturating_div(*rhs)))
                } else {
                    Err(Fault::DivideByZero)
                }
            }

            #[allow(clippy::cast_possible_truncation)]
            (Self::Int(lhs), Self::Float(rhs)) => Ok(Self::Int(*lhs / *rhs as i64)),
            #[allow(clippy::cast_possible_truncation)]
            (Self::Float(lhs), Self::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::Int(*lhs as i64 / *rhs))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            (Self::UInt(lhs), Self::Float(rhs)) => {
                let rhs = *rhs as u64;
                if rhs == 0 {
                    Err(Fault::DivideByZero)
                } else {
                    Ok(Self::UInt(*lhs / rhs))
                }
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            (Self::Float(lhs), Self::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::UInt(*lhs as u64 / *rhs))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_possible_truncation)]
            (Self::Float(lhs), Self::Float(rhs)) => Ok(Self::Int(*lhs as i64 / *rhs as i64)),

            _ => Err(Fault::UnsupportedOperation),
        }
    }

    /// Calcualtes the remainder of dividing `self` by `rhs` using whole
    /// numbers.
    pub fn rem(&self, rhs: &Self) -> Result<Self, Fault> {
        match (self, rhs) {
            (Self::Nil, _) | (_, Self::Nil) => Err(Fault::OperationOnNil),

            (Self::Int(lhs), Self::Int(rhs)) => Ok(Self::Int(lhs % rhs)),
            (Self::Int(lhs), Self::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::Int(lhs % i64::try_from(*rhs).unwrap_or(i64::MAX)))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            (Self::UInt(lhs), Self::Int(rhs)) => {
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
            (Self::UInt(lhs), Self::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::UInt(lhs % *rhs))
                } else {
                    Err(Fault::DivideByZero)
                }
            }

            #[allow(clippy::cast_possible_truncation)]
            (Self::Int(lhs), Self::Float(rhs)) => Ok(Self::Int(*lhs % *rhs as i64)),
            #[allow(clippy::cast_possible_truncation)]
            (Self::Float(lhs), Self::Int(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::Int(*lhs as i64 % *rhs))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            (Self::UInt(lhs), Self::Float(rhs)) => {
                let rhs = *rhs as u64;
                if rhs == 0 {
                    Err(Fault::DivideByZero)
                } else {
                    Ok(Self::UInt(*lhs % rhs))
                }
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            (Self::Float(lhs), Self::UInt(rhs)) => {
                if *rhs != 0 {
                    Ok(Self::UInt(*lhs as u64 % *rhs))
                } else {
                    Err(Fault::DivideByZero)
                }
            }
            #[allow(clippy::cast_possible_truncation)]
            (Self::Float(lhs), Self::Float(rhs)) => Ok(Self::Int(*lhs as i64 % *rhs as i64)),

            _ => Err(Fault::UnsupportedOperation),
        }
    }

    /// Returns the inverse of [`Self::truthy`].
    pub fn not(&self) -> Result<Self, Fault> {
        Ok(Self::Bool(!self.truthy()))
    }

    /// Negates this value.
    pub fn negate(&self) -> Self {
        match self {
            Self::Nil => Self::Nil,
            Self::Bool(bool) => Self::Bool(!bool),
            Self::Int(value) => Self::Int(-*value),
            Self::UInt(value) => {
                if let Ok(value) = i64::try_from(*value) {
                    Self::Int(value.saturating_neg())
                } else {
                    Self::Int(i64::MIN)
                }
            }
            Self::Float(value) => Self::Float(-*value),
        }
    }

    /// Returns the bitwise not of this value.
    pub fn bitwise_not(&self) -> Result<Self, Fault> {
        match self {
            Self::Nil => Ok(Self::Nil),
            Self::Bool(bool) => Ok(Self::Bool(!bool)),
            Self::Int(value) => Ok(Self::Int(!*value)),
            Self::UInt(value) => Ok(Self::UInt(!*value)),
            Self::Float(_) => Err(Fault::UnsupportedOperation),
        }
    }

    /// Returns the bitwise and of `self` and `rhs`.
    pub fn bitwise_and(&self, rhs: &Self) -> Result<Self, Fault> {
        match (self, rhs) {
            (Self::Int(lhs), Self::Int(rhs)) => Ok(Self::Int(lhs & rhs)),
            #[allow(clippy::cast_possible_wrap)]
            (Self::Int(lhs), Self::UInt(rhs)) => Ok(Self::Int(lhs & *rhs as i64)),
            (Self::Int(lhs), _) => {
                if let Some(rhs) = rhs.as_i64() {
                    Ok(Self::Int(lhs & rhs))
                } else {
                    Err(Fault::UnsupportedOperation)
                }
            }

            (Self::UInt(lhs), Self::UInt(rhs)) => Ok(Self::UInt(lhs & rhs)),
            #[allow(clippy::cast_sign_loss)]
            (Self::UInt(lhs), Self::Int(rhs)) => Ok(Self::UInt(lhs & *rhs as u64)),
            (Self::UInt(lhs), _) => {
                if let Some(rhs) = rhs.to_u64() {
                    Ok(Self::UInt(lhs & rhs))
                } else {
                    Err(Fault::UnsupportedOperation)
                }
            }

            (Self::Float(lhs), _) if lhs.is_sign_negative() => {
                match (self.as_i64(), rhs.as_i64()) {
                    (Some(lhs), Some(rhs)) => Ok(Self::Int(lhs & rhs)),
                    // If either are none, we know the result is 0.
                    _ => Ok(Self::Int(0)),
                }
            }

            _ => match (self.to_u64(), rhs.to_u64()) {
                (Some(lhs), Some(rhs)) => Ok(Self::UInt(lhs & rhs)),
                // If either are none, we know the result is 0.
                _ => Ok(Self::UInt(0)),
            },
        }
    }

    /// Returns the bitwise or of `self` and `rhs`.
    pub fn bitwise_or(&self, rhs: &Self) -> Result<Self, Fault> {
        match (self, rhs) {
            (Self::Int(lhs), Self::Int(rhs)) => Ok(Self::Int(lhs | rhs)),
            #[allow(clippy::cast_possible_wrap)]
            (Self::Int(lhs), Self::UInt(rhs)) => Ok(Self::Int(lhs | *rhs as i64)),
            (Self::Int(lhs), _) => {
                if let Some(rhs) = rhs.to_i64() {
                    Ok(Self::Int(lhs | rhs))
                } else {
                    Err(Fault::UnsupportedOperation)
                }
            }

            (Self::UInt(lhs), Self::UInt(rhs)) => Ok(Self::UInt(lhs | rhs)),
            #[allow(clippy::cast_sign_loss)]
            (Self::UInt(lhs), Self::Int(rhs)) => Ok(Self::UInt(lhs | *rhs as u64)),
            (Self::UInt(lhs), _) => {
                if let Some(rhs) = rhs.to_u64() {
                    Ok(Self::UInt(lhs | rhs))
                } else {
                    Err(Fault::UnsupportedOperation)
                }
            }

            (Self::Float(lhs), _) if lhs.is_sign_negative() => {
                match (self.to_i64(), rhs.to_i64()) {
                    (Some(lhs), Some(rhs)) => Ok(Self::Int(lhs | rhs)),
                    (Some(result), None) | (None, Some(result)) => Ok(Self::Int(result)),
                    (None, None) => Ok(Self::Int(0)),
                }
            }

            _ => match (self.to_u64(), rhs.to_u64()) {
                (Some(lhs), Some(rhs)) => Ok(Self::UInt(lhs | rhs)),
                (Some(result), None) | (None, Some(result)) => Ok(Self::UInt(result)),
                (None, None) => Ok(Self::UInt(0)),
            },
        }
    }

    /// Returns the bitwise xor of `self` and `rhs`.
    pub fn bitwise_xor(&self, rhs: &Self) -> Result<Self, Fault> {
        match (self, rhs) {
            (Self::Int(lhs), Self::Int(rhs)) => Ok(Self::Int(lhs ^ rhs)),
            #[allow(clippy::cast_possible_wrap)]
            (Self::Int(lhs), Self::UInt(rhs)) => Ok(Self::Int(lhs ^ *rhs as i64)),
            (Self::Int(lhs), _) => {
                if let Some(rhs) = rhs.to_i64() {
                    Ok(Self::Int(lhs ^ rhs))
                } else {
                    Err(Fault::UnsupportedOperation)
                }
            }

            (Self::UInt(lhs), Self::UInt(rhs)) => Ok(Self::UInt(lhs ^ rhs)),
            #[allow(clippy::cast_sign_loss)]
            (Self::UInt(lhs), Self::Int(rhs)) => Ok(Self::UInt(lhs ^ *rhs as u64)),
            (Self::UInt(lhs), _) => {
                if let Some(rhs) = rhs.to_u64() {
                    Ok(Self::UInt(lhs ^ rhs))
                } else {
                    Err(Fault::UnsupportedOperation)
                }
            }

            (Self::Float(lhs), _) if lhs.is_sign_negative() => {
                match (self.to_i64(), rhs.to_i64()) {
                    (Some(lhs), Some(rhs)) => Ok(Self::Int(lhs ^ rhs)),
                    (Some(result), None) | (None, Some(result)) => Ok(Self::Int(result)),
                    (None, None) => Ok(Self::Int(0)),
                }
            }

            _ => match (self.to_u64(), rhs.to_u64()) {
                (Some(lhs), Some(rhs)) => Ok(Self::UInt(lhs ^ rhs)),
                (Some(result), None) | (None, Some(result)) => Ok(Self::UInt(result)),
                (None, None) => Ok(Self::UInt(0)),
            },
        }
    }

    /// Returns the bitwise shift left of `self` by `rhs`.
    pub fn shift_left(&self, rhs: &Self) -> Result<Self, Fault> {
        match self {
            Self::Int(lhs) => Ok(Self::Int(
                lhs.checked_shl(rhs.to_u32().ok_or(Fault::UnsupportedOperation)?)
                    .unwrap_or_default(),
            )),
            Self::UInt(lhs) => Ok(Self::UInt(
                lhs.checked_shl(rhs.to_u32().ok_or(Fault::UnsupportedOperation)?)
                    .unwrap_or_default(),
            )),

            #[allow(clippy::cast_possible_truncation)]
            Self::Float(lhs) if lhs.is_sign_negative() => Ok(Self::Int(
                (*lhs as i64)
                    .checked_shl(rhs.to_u32().ok_or(Fault::UnsupportedOperation)?)
                    .unwrap_or_default(),
            )),

            _ => match (self.to_u64(), rhs.to_u32()) {
                (Some(lhs), Some(rhs)) => Ok(Self::UInt(lhs.checked_shl(rhs).unwrap_or_default())),
                _ => Err(Fault::UnsupportedOperation),
            },
        }
    }

    /// Returns the shift right of `self` by `rhs`.
    pub fn shift_right(&self, rhs: &Self) -> Result<Self, Fault> {
        match self {
            Self::Int(lhs) => Ok(Self::Int(
                lhs.checked_shr(rhs.to_u32().ok_or(Fault::UnsupportedOperation)?)
                    .unwrap_or_default(),
            )),
            Self::UInt(lhs) => Ok(Self::UInt(
                lhs.checked_shr(rhs.to_u32().ok_or(Fault::UnsupportedOperation)?)
                    .unwrap_or_default(),
            )),

            #[allow(clippy::cast_possible_truncation)]
            Self::Float(lhs) if lhs.is_sign_negative() => Ok(Self::Int(
                (*lhs as i64)
                    .checked_shr(rhs.to_u32().ok_or(Fault::UnsupportedOperation)?)
                    .unwrap_or_default(),
            )),

            _ => match (self.to_u64(), rhs.to_u32()) {
                (Some(lhs), Some(rhs)) => Ok(Self::UInt(lhs.checked_shr(rhs).unwrap_or_default())),
                _ => Err(Fault::UnsupportedOperation),
            },
        }
    }

    /// Returns this value as a shared string reference.
    pub fn to_string(&self) -> Result<SymbolRef, Fault> {
        match self {
            Self::Nil => Ok(Symbol::empty().downgrade()),
            Self::Bool(bool) => Ok(SymbolRef::from(*bool)),
            Self::Int(value) => Ok(SymbolRef::from(value.to_string())),
            Self::UInt(value) => Ok(SymbolRef::from(value.to_string())),
            Self::Float(value) => Ok(SymbolRef::from(value.to_string())),
        }
    }

    /// Hashes this value into `hasher`.
    pub fn hash_into(&self, hasher: &mut ValueHasher) {
        core::mem::discriminant(self).hash(hasher);
        match self {
            Self::Nil => {}
            Self::Bool(b) => b.hash(hasher),
            Self::Int(i) => i.hash(hasher),
            Self::UInt(i) => i.hash(hasher),
            Self::Float(f) => f.to_bits().hash(hasher),
        }
    }

    /// Returns true if `self` and `other` are equivalent values.
    pub fn equals(&self, other: &Self) -> Result<bool, Fault> {
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

            _ => Ok(false),
        }
    }

    /// Returns an ordering of `self` and `other` that takes into account both
    /// the type of data and the value itself.
    pub fn total_cmp(&self, other: &Self) -> Result<Ordering, Fault> {
        match (self, other) {
            (Self::Nil, Self::Nil) => Ok(Ordering::Equal),

            (Self::Bool(l), Self::Bool(r)) => Ok(l.cmp(r)),

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

            (Self::Int(l), Self::Int(r)) => Ok(l.cmp(r)),
            (Self::Int(l), Self::UInt(r)) => Ok(if let Ok(l) = u64::try_from(*l) {
                l.cmp(r)
            } else {
                Ordering::Less
            }),
            (Self::UInt(l), Self::UInt(r)) => Ok(l.cmp(r)),
            (Self::Float(l), Self::Float(r)) => Ok(l.total_cmp(r)),

            #[allow(clippy::cast_precision_loss)]
            (Self::Int(l_int), Self::Float(r_float)) => Ok((*l_int as f64).total_cmp(r_float)),
            #[allow(clippy::cast_precision_loss)]
            (Self::Float(l_float), Self::Int(r_int)) => Ok(l_float.total_cmp(&(*r_int as f64))),

            (Self::Nil, _) => Ok(Ordering::Less),
            (_, Self::Nil) => Ok(Ordering::Greater),

            (Self::Bool(_), _) => Ok(Ordering::Less),
            (_, Self::Bool(_)) => Ok(Ordering::Greater),
            (_, Self::Int(_)) => Ok(Ordering::Greater),
            (Self::UInt(_), _) => Ok(Ordering::Less),
            (_, Self::UInt(_)) => Ok(Ordering::Greater),
        }
    }

    /// Take the contents of this value, leaving nil behind.
    #[must_use]
    pub fn take(&mut self) -> Primitive {
        std::mem::take(self)
    }

    /// Formats this value for display into `f`.
    pub fn format(&self, context: &mut VmContext<'_, '_>, mut f: impl fmt::Write) -> fmt::Result {
        if let Some(s) = self.to_string().ok().and_then(|s| s.load(context.guard())) {
            f.write_str(s)
        } else {
            todo!("display unformattable value")
        }
    }
}

impl PartialEq for Primitive {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other).unwrap_or(false)
    }
}

/// A Muse virtual machine value.
#[derive(Clone, Copy, Debug)]
pub enum Value {
    Primitive(Primitive),
    /// A symbol.
    Symbol(SymbolRef),
    /// A dynamically allocated, garbage collected type.
    Dynamic(AnyDynamic),
}

impl Value {
    pub const TRUE: Self = Self::Primitive(Primitive::Bool(true));
    pub const FALSE: Self = Self::Primitive(Primitive::Bool(false));
    pub const NIL: Self = Self::Primitive(Primitive::Nil);
    pub const ZERO: Self = Self::Primitive(Primitive::Int(0));

    /// Returns this value with any garbage collected values upgraded to root
    /// references.
    pub fn upgrade(&self, guard: &CollectionGuard<'_>) -> Option<RootedValue> {
        match self {
            Value::Primitive(primitive) => Some(RootedValue::Primitive(*primitive)),
            Value::Symbol(v) => v.upgrade(guard).map(RootedValue::Symbol),
            Value::Dynamic(v) => v.upgrade(guard).map(RootedValue::Dynamic),
        }
    }

    /// Moves `value` into the virtual machine.
    pub fn dynamic<'guard, T>(value: T, guard: &impl AsRef<CollectionGuard<'guard>>) -> Self
    where
        T: DynamicValue + Trace,
    {
        Self::Dynamic(AnyDynamic::new(value, guard))
    }

    /// Returns true if this value is nil.
    #[must_use]
    pub const fn is_nil(&self) -> bool {
        matches!(self, Self::Primitive(Primitive::Nil))
    }

    /// Returns this value as an i64, if possible.
    #[must_use]
    pub fn as_primitive(&self) -> Option<Primitive> {
        match self {
            Value::Primitive(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns this value as an i64, if possible.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        self.as_primitive().and_then(|p| p.as_i64())
    }

    /// Returns this value as an u64, if possible.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        self.as_primitive().and_then(|p| p.as_u64())
    }

    /// Returns this value as an u32, if possible.
    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        self.as_primitive().and_then(|p| p.as_u32())
    }

    /// Returns this value as an u16, if possible.
    #[must_use]
    pub fn as_u16(&self) -> Option<u16> {
        self.as_primitive().and_then(|p| p.as_u16())
    }

    /// Returns this value as an usize, if possible.
    #[must_use]
    pub fn as_usize(&self) -> Option<usize> {
        self.as_primitive().and_then(|p| p.as_usize())
    }

    /// Returns this value as an f64, if possible.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        self.as_primitive().and_then(|p| p.as_f64())
    }

    /// Converts this value to an i64, if possible.
    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        match self {
            Value::Primitive(value) => value.to_i64(),
            Value::Symbol(_) | Value::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    /// Converts this value to an u64, if possible.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        match self {
            Value::Primitive(value) => value.to_u64(),
            Value::Symbol(_) | Value::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    /// Converts this value to an u32, if possible.
    #[must_use]
    pub fn to_u32(&self) -> Option<u32> {
        match self {
            Value::Primitive(value) => value.to_u32(),
            Value::Symbol(_) | Value::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    /// Converts this value to an usize, if possible.
    #[must_use]
    pub fn to_usize(&self) -> Option<usize> {
        match self {
            Value::Primitive(value) => value.to_usize(),
            Value::Symbol(_) | Value::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    /// Converts this value to an f64, if possible.
    #[must_use]
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            Value::Primitive(value) => Some(value.to_f64()),
            Value::Symbol(_) | Value::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    /// Returns this value as a `SymbolRef`, if possible.
    #[must_use]
    pub fn as_symbol_ref(&self) -> Option<&SymbolRef> {
        match self {
            Value::Symbol(value) => Some(value),
            _ => None,
        }
    }

    /// Returns this value as a `Symbol`, if possible.
    #[must_use]
    pub fn as_symbol(&self, guard: &CollectionGuard<'_>) -> Option<Symbol> {
        match self {
            Value::Symbol(value) => value.upgrade(guard),
            _ => None,
        }
    }

    /// Returns this value as an `AnyDynamic`, if possible.
    #[must_use]
    pub fn as_any_dynamic(&self) -> Option<AnyDynamic> {
        match self {
            Value::Dynamic(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns this value as a `Dynamic<T>`, if this value contains a `T`.
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

    /// Returns this value as a `Rooted<T>`, if this value contains a `T`.
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

    /// Returns this value as a`&T`, if this value contains a `T`.
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

    /// Returns true if this value should be considered `true` in a boolean
    /// expression.
    pub fn truthy(&self, vm: &mut VmContext<'_, '_>) -> bool {
        match self {
            Value::Primitive(value) => value.truthy(),
            Value::Symbol(sym) => sym.load(vm.as_ref()).map_or(false, |sym| !sym.is_empty()),
            Value::Dynamic(value) => value.truthy(vm),
        }
    }

    /// Invokes this value as a function.
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
            Value::Primitive(Primitive::Nil) => vm.recurse_current_function(arity.into()),
            _ => Err(Fault::NotAFunction),
        }
    }

    /// Invokes `name` on this value.
    pub fn invoke(
        &self,
        vm: &mut VmContext<'_, '_>,
        name: &SymbolRef,
        arity: impl Into<Arity>,
    ) -> Result<Value, Fault> {
        match self {
            Value::Dynamic(dynamic) => dynamic.invoke(vm, name, arity.into()),
            Value::Primitive(Primitive::Nil) => Err(Fault::OperationOnNil),
            // TODO we should pass through to the appropriate Type
            _ => Err(Fault::UnknownSymbol),
        }
    }

    /// Adds `self` to `rhs`.
    pub fn add(&self, vm: &mut VmContext<'_, '_>, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Primitive(lhs), Value::Primitive(rhs)) => lhs.add(rhs).map(Self::Primitive),

            (Value::Symbol(lhs), rhs) => {
                let lhs = lhs.try_upgrade(vm.guard())?;
                rhs.map_str(vm, |_vm, rhs| Value::Symbol(SymbolRef::from(&lhs + rhs)))
            }
            (lhs, Value::Symbol(rhs)) => {
                let rhs = rhs.try_upgrade(vm.guard())?;
                lhs.map_str(vm, |_vm, lhs| Value::Symbol(SymbolRef::from(lhs + &rhs)))
            }

            (Value::Dynamic(lhs), rhs) => lhs.add(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.add_right(vm, lhs),
        }
    }

    /// Subtracts `rhs` from `self`.
    pub fn sub(&self, vm: &mut VmContext<'_, '_>, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Primitive(lhs), Value::Primitive(rhs)) => lhs.sub(rhs).map(Value::Primitive),

            (Value::Dynamic(lhs), rhs) => lhs.sub(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.sub_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    /// Multiplies `self` by `rhs`.
    pub fn mul(&self, vm: &mut VmContext<'_, '_>, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Primitive(lhs), Value::Primitive(rhs)) => lhs.mul(rhs).map(Value::Primitive),

            // TODO unsigned int
            (Value::Primitive(Primitive::Int(count)), Value::Symbol(string))
            | (Value::Symbol(string), Value::Primitive(Primitive::Int(count))) => {
                let string = string.try_upgrade(vm.guard())?;
                Ok(Value::Symbol(SymbolRef::from(string.repeat(
                    usize::try_from(*count).map_err(|_| Fault::OutOfMemory)?,
                ))))
            }

            (Value::Dynamic(lhs), rhs) => lhs.mul(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.mul_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    /// Raises `self` to the `rhs` power.
    pub fn pow(&self, vm: &mut VmContext<'_, '_>, exp: &Self) -> Result<Value, Fault> {
        match (self, exp) {
            (Value::Primitive(lhs), Value::Primitive(rhs)) => lhs.pow(rhs).map(Value::Primitive),

            (Value::Dynamic(lhs), rhs) => lhs.mul(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.mul_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    /// Divides `self` by `rhs`.
    pub fn div(&self, vm: &mut VmContext<'_, '_>, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Primitive(lhs), Value::Primitive(rhs)) => lhs.div(rhs).map(Value::Primitive),

            (Value::Dynamic(lhs), rhs) => lhs.div(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.div_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    /// Divides `self` by `rhs`, using whole numbers.
    pub fn idiv(&self, vm: &mut VmContext<'_, '_>, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Primitive(lhs), Value::Primitive(rhs)) => lhs.idiv(rhs).map(Value::Primitive),

            (Value::Dynamic(lhs), rhs) => lhs.idiv(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.idiv_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    /// Calcualtes the remainder of dividing `self` by `rhs` using whole
    /// numbers.
    pub fn rem(&self, vm: &mut VmContext<'_, '_>, rhs: &Self) -> Result<Value, Fault> {
        match (self, rhs) {
            (Value::Primitive(lhs), Value::Primitive(rhs)) => lhs.rem(rhs).map(Value::Primitive),

            (Value::Dynamic(lhs), rhs) => lhs.rem(vm, rhs),
            (lhs, Value::Dynamic(rhs)) => rhs.rem_right(vm, lhs),
            _ => Err(Fault::UnsupportedOperation),
        }
    }

    /// Returns the inverse of [`Self::truthy`].
    pub fn not(&self, vm: &mut VmContext<'_, '_>) -> Result<Self, Fault> {
        Ok(Value::Primitive(Primitive::Bool(!self.truthy(vm))))
    }

    /// Negates this value.
    pub fn negate(&self, vm: &mut VmContext<'_, '_>) -> Result<Self, Fault> {
        match self {
            Value::Primitive(p) => Ok(Self::Primitive(p.negate())),
            Value::Dynamic(value) => value.negate(vm),
            Value::Symbol(_) => Err(Fault::UnsupportedOperation),
        }
    }

    /// Returns the bitwise not of this value.
    pub fn bitwise_not(&self, vm: &mut VmContext<'_, '_>) -> Result<Self, Fault> {
        match self {
            Value::Primitive(p) => p.bitwise_not().map(Self::Primitive),
            Value::Dynamic(value) => value.bitwise_not(vm),
            Value::Symbol(_) => Err(Fault::UnsupportedOperation),
        }
    }

    /// Returns the bitwise and of `self` and `rhs`.
    pub fn bitwise_and(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Self, Fault> {
        match (self, rhs) {
            (Value::Dynamic(dymamic), other) | (other, Value::Dynamic(dymamic)) => {
                dymamic.bitwise_and(vm, other)
            }
            (Value::Primitive(lhs), Value::Primitive(rhs)) => {
                lhs.bitwise_and(rhs).map(Value::Primitive)
            }

            _ => match (self.to_u64(), rhs.to_u64()) {
                (Some(lhs), Some(rhs)) => Ok(Value::Primitive(Primitive::UInt(lhs & rhs))),
                // If either are none, we know the result is 0.
                _ => Ok(Value::Primitive(Primitive::UInt(0))),
            },
        }
    }

    /// Returns the bitwise or of `self` and `rhs`.
    pub fn bitwise_or(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Self, Fault> {
        match (self, rhs) {
            (Value::Dynamic(dymamic), other) | (other, Value::Dynamic(dymamic)) => {
                dymamic.bitwise_or(vm, other)
            }
            (Value::Primitive(lhs), Value::Primitive(rhs)) => {
                lhs.bitwise_or(rhs).map(Value::Primitive)
            }

            _ => match (self.to_u64(), rhs.to_u64()) {
                (Some(lhs), Some(rhs)) => Ok(Value::Primitive(Primitive::UInt(lhs | rhs))),
                (Some(result), None) | (None, Some(result)) => {
                    Ok(Value::Primitive(Primitive::UInt(result)))
                }
                (None, None) => Ok(Value::Primitive(Primitive::UInt(0))),
            },
        }
    }

    /// Returns the bitwise xor of `self` and `rhs`.
    pub fn bitwise_xor(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Self, Fault> {
        match (self, rhs) {
            (Value::Dynamic(dymamic), other) | (other, Value::Dynamic(dymamic)) => {
                dymamic.bitwise_xor(vm, other)
            }
            (Value::Primitive(lhs), Value::Primitive(rhs)) => {
                lhs.bitwise_xor(rhs).map(Value::Primitive)
            }

            _ => match (self.to_u64(), rhs.to_u64()) {
                (Some(lhs), Some(rhs)) => Ok(Value::Primitive(Primitive::UInt(lhs ^ rhs))),
                (Some(result), None) | (None, Some(result)) => {
                    Ok(Value::Primitive(Primitive::UInt(result)))
                }
                (None, None) => Ok(Value::Primitive(Primitive::UInt(0))),
            },
        }
    }

    /// Returns the bitwise shift left of `self` by `rhs`.
    pub fn shift_left(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Self, Fault> {
        match (self, rhs) {
            (Value::Dynamic(dymamic), _) => dymamic.shift_left(vm, rhs),

            (Value::Primitive(lhs), Value::Primitive(rhs)) => {
                lhs.shift_left(rhs).map(Value::Primitive)
            }

            _ => match (self.to_u64(), rhs.to_u32()) {
                (Some(lhs), Some(rhs)) => Ok(Value::Primitive(Primitive::UInt(
                    lhs.checked_shl(rhs).unwrap_or_default(),
                ))),
                _ => Err(Fault::UnsupportedOperation),
            },
        }
    }

    /// Returns the shift right of `self` by `rhs`.
    pub fn shift_right(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Self, Fault> {
        match (self, rhs) {
            (Value::Dynamic(dymamic), _) => dymamic.shift_right(vm, rhs),

            (Value::Primitive(lhs), Value::Primitive(rhs)) => {
                lhs.shift_right(rhs).map(Value::Primitive)
            }

            _ => match (self.to_u64(), rhs.to_u32()) {
                (Some(lhs), Some(rhs)) => Ok(Value::Primitive(Primitive::UInt(
                    lhs.checked_shr(rhs).unwrap_or_default(),
                ))),
                _ => Err(Fault::UnsupportedOperation),
            },
        }
    }

    /// Returns this value as a shared string reference.
    pub fn to_string(&self, vm: &mut VmContext<'_, '_>) -> Result<SymbolRef, Fault> {
        match self {
            Value::Primitive(value) => value.to_string(),
            Value::Symbol(value) => Ok(*value),
            Value::Dynamic(value) => value.to_string(vm),
        }
    }

    /// Maps the contents of this value as a `str`, if possible.
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

    /// Hashes this value into `hasher`.
    pub fn hash_into(&self, vm: &mut VmContext<'_, '_>, hasher: &mut ValueHasher) {
        core::mem::discriminant(self).hash(hasher);
        match self {
            Value::Primitive(v) => v.hash_into(hasher),
            Value::Symbol(s) => s.hash(hasher),
            Value::Dynamic(d) => d.hash(vm, hasher),
        }
    }

    /// Calculates the hash of this value.
    pub fn hash(&self, vm: &mut VmContext<'_, '_>) -> u64 {
        let mut hasher = ValueHasher::default();

        core::mem::discriminant(self).hash(&mut hasher);
        match self {
            Value::Primitive(v) => v.hash_into(&mut hasher),
            Value::Symbol(s) => s.hash(&mut hasher),
            Value::Dynamic(d) => d.hash(vm, &mut hasher),
        }

        hasher.finish()
    }

    /// Returns true if `self` and `other` are equivalent values.
    pub fn equals(&self, vm: ContextOrGuard<'_, '_, '_>, other: &Self) -> Result<bool, Fault> {
        match (self, other) {
            (Self::Primitive(l0), Self::Primitive(r0)) => l0.equals(r0),
            (Self::Primitive(Primitive::Nil), _) | (_, Self::Primitive(Primitive::Nil)) => {
                Ok(false)
            }

            (Self::Symbol(l0), Self::Symbol(r0)) => Ok(l0 == r0),
            (Self::Symbol(s), Self::Primitive(Primitive::Bool(b)))
            | (Self::Primitive(Primitive::Bool(b)), Self::Symbol(s)) => {
                Ok(s == &SymbolRef::from(*b))
            }

            (Self::Dynamic(l0), _) => l0.eq(vm, other),
            (_, Self::Dynamic(r0)) => r0.eq(vm, self),

            _ => Ok(false),
        }
    }

    /// Returns true if `self` matches `other`.
    pub fn matches(&self, vm: &mut VmContext<'_, '_>, other: &Self) -> Result<bool, Fault> {
        match (self, other) {
            (Self::Dynamic(l0), _) => l0.matches(vm, other),
            (_, Self::Dynamic(r0)) => r0.matches(vm, self),
            _ => self.equals(ContextOrGuard::Context(vm), other),
        }
    }

    /// Returns an ordering of `self` and `other` that takes into account both
    /// the type of data and the value itself.
    pub fn total_cmp(&self, vm: &mut VmContext<'_, '_>, other: &Self) -> Result<Ordering, Fault> {
        match (self, other) {
            (Value::Primitive(l), Value::Primitive(r)) => l.total_cmp(r),

            (Value::Symbol(l), Value::Symbol(r)) => Ok(l.cmp(r)),

            (Self::Dynamic(l0), _) => l0.cmp(vm, other),
            (_, Self::Dynamic(r0)) => r0.cmp(vm, self).map(Ordering::reverse),

            (Value::Primitive(Primitive::Nil), _) => Ok(Ordering::Less),
            (_, Value::Primitive(Primitive::Nil)) => Ok(Ordering::Greater),

            (Value::Primitive(Primitive::Bool(_)), _) => Ok(Ordering::Less),
            (_, Value::Primitive(Primitive::Bool(_))) => Ok(Ordering::Greater),
            (Value::Primitive(Primitive::Int(_)), _) => Ok(Ordering::Less),
            (_, Value::Primitive(Primitive::Int(_))) => Ok(Ordering::Greater),
            (Value::Primitive(Primitive::UInt(_)), _) => Ok(Ordering::Less),
            (_, Value::Primitive(Primitive::UInt(_))) => Ok(Ordering::Greater),
            (Value::Primitive(Primitive::Float(_)), _) => Ok(Ordering::Less),
            (_, Value::Primitive(Primitive::Float(_))) => Ok(Ordering::Greater),
        }
    }

    /// Take the contents of this value, leaving nil behind.
    #[must_use]
    pub fn take(&mut self) -> Value {
        std::mem::take(self)
    }

    /// Perform a deep-clone on the contents of this value.
    ///
    /// Not all types are able to be deeply cloned. Unsupported types will
    /// result in `None`.
    pub fn deep_clone(&self, guard: &CollectionGuard) -> Option<Value> {
        match self {
            Value::Primitive(p) => Some(Value::Primitive(*p)),
            Value::Symbol(value) => Some(Value::Symbol(*value)),
            Value::Dynamic(value) => value.deep_clone(guard).map(Value::Dynamic),
        }
    }

    /// Formats this value for display into `f`.
    pub fn format(&self, context: &mut VmContext<'_, '_>, mut f: impl fmt::Write) -> fmt::Result {
        if let Some(s) = self
            .to_string(context)
            .ok()
            .and_then(|s| s.load(context.guard()))
        {
            f.write_str(s)
        } else {
            todo!("display unformattable value")
        }
    }

    #[cfg(feature = "dispatched")]
    pub(crate) fn as_source(&self, guard: &CollectionGuard<'_>) -> ValueOrSource {
        match self {
            Value::Primitive(p) => ValueOrSource::Primitive(*p),
            Value::Symbol(value) => value.upgrade(guard).map_or(
                ValueOrSource::Primitive(Primitive::Nil),
                ValueOrSource::Symbol,
            ),
            Value::Dynamic(value) => {
                if let Some(func) = value.downcast_ref::<Function>(guard) {
                    ValueOrSource::Function(BitcodeFunction::from_function(func, guard))
                } else if let Some(func) = value.downcast_ref::<RuntimeStruct>(guard) {
                    ValueOrSource::Struct(func.to_bitcode_type(guard))
                } else if let Some(func) = value.downcast_ref::<RuntimeEnum>(guard) {
                    ValueOrSource::Enum(func.to_bitcode_type(guard))
                } else {
                    // All dynamics generated into a Source must be known by
                    // Muse
                    unreachable!("unexpected dynamic")
                }
            }
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Self::Primitive(Primitive::default())
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

impl_from!(Primitive, f32, Float);
impl_from!(Primitive, f64, Float);
impl_from!(Primitive, i8, Int);
impl_from!(Primitive, i16, Int);
impl_from!(Primitive, i32, Int);
impl_from!(Primitive, i64, Int);
impl_from!(Primitive, u8, UInt);
impl_from!(Primitive, u16, UInt);
impl_from!(Primitive, u32, UInt);
impl_from!(Primitive, u64, UInt);
impl_from!(Primitive, bool, Bool);

#[macro_export]
macro_rules! impl_from_primitive {
    ($on:ident, $from:ty, $variant:ident) => {
        impl From<$from> for $on {
            fn from(value: $from) -> Self {
                Self::Primitive(Primitive::$variant(value.into()))
            }
        }
    };
}
impl_from_primitive!(Value, f32, Float);
impl_from_primitive!(Value, f64, Float);
impl_from_primitive!(Value, i8, Int);
impl_from_primitive!(Value, i16, Int);
impl_from_primitive!(Value, i32, Int);
impl_from_primitive!(Value, i64, Int);
impl_from_primitive!(Value, u8, UInt);
impl_from_primitive!(Value, u16, UInt);
impl_from_primitive!(Value, u32, UInt);
impl_from_primitive!(Value, u64, UInt);
impl_from_primitive!(Value, bool, Bool);
impl_from!(Value, Symbol, Symbol);
impl_from!(Value, &'_ Symbol, Symbol);
impl_from!(Value, SymbolRef, Symbol);
impl_from!(Value, &'_ SymbolRef, Symbol);

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

impl_try_from!(Primitive, u128, UInt);
impl_try_from!(Primitive, usize, UInt);
impl_try_from!(Primitive, isize, Int);
impl_try_from!(Primitive, i128, UInt);

macro_rules! impl_try_from_primitive {
    ($on:ident, $from:ty) => {
        impl TryFrom<$from> for $on {
            type Error = Fault;

            fn try_from(value: $from) -> Result<Self, Self::Error> {
                Primitive::try_from(value).map(Self::Primitive)
            }
        }
    };
}

impl_try_from_primitive!(Value, u128);
impl_try_from_primitive!(Value, usize);
impl_try_from_primitive!(Value, isize);
impl_try_from_primitive!(Value, i128);

/// A weak reference to a [`Rooted<T>`].
#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct AnyDynamic(pub(crate) AnyRef);

impl AnyDynamic {
    /// Returns `value` as a garbage collected value that can be used in Muse.
    pub fn new<'guard, T>(value: T, guard: &impl AsRef<CollectionGuard<'guard>>) -> Self
    where
        T: DynamicValue + Trace,
    {
        Self(Ref::new(Custom(value), guard).as_any())
    }

    /// Returns this dynamic upgraded to a root reference.
    pub fn upgrade(&self, guard: &CollectionGuard<'_>) -> Option<AnyDynamicRoot> {
        self.0.upgrade(guard).map(AnyDynamicRoot)
    }

    /// Upgrades this reference to a [`Dynamic<T>`].
    ///
    /// This function does no type checking. If `T` is the incorrect type,
    /// trying to access the underyling data will return None/an error.
    #[must_use]
    pub fn as_dynamic<T>(&self) -> Dynamic<T>
    where
        T: DynamicValue + Trace,
    {
        Dynamic(self.0.downcast_ref::<Custom<T>>())
    }

    /// Tries to upgrade this reference to a [`Rooted<T>`].
    ///
    /// This function can return None if:
    ///
    /// - `T` is not the correct type.
    /// - The value has been garbage collected.
    #[must_use]
    pub fn as_rooted<T>(&self, guard: &CollectionGuard<'_>) -> Option<Rooted<T>>
    where
        T: DynamicValue + Trace,
    {
        self.0
            .downcast_root::<Custom<T>>(guard)
            .map(|cast| Rooted(cast))
    }

    /// Tries to load a reference to this reference's underlying data.
    ///
    /// This function can return None if:
    ///
    /// - `T` is not the correct type.
    /// - The value has been garbage collected.
    #[must_use]
    pub fn downcast_ref<'guard, T>(&self, guard: &'guard CollectionGuard) -> Option<&'guard T>
    where
        T: DynamicValue + Trace,
    {
        self.0
            .load::<Custom<T>>(guard)
            .and_then(|d| d.0.as_any().downcast_ref())
    }

    /// Invokes `deep_clone` on the underlying type's [`TypeVtable`].
    #[must_use]
    pub fn deep_clone(&self, guard: &CollectionGuard) -> Option<AnyDynamic> {
        (self
            .0
            .load_mapped::<dyn CustomType>(guard)?
            .muse_type()
            .vtable
            .deep_clone)(self, guard)
    }

    /// Invokes `call` on the underlying type's [`TypeVtable`].
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

    /// Invokes `invoke` on the underlying type's [`TypeVtable`].
    pub fn invoke(
        &self,
        vm: &mut VmContext<'_, '_>,
        symbol: &SymbolRef,
        arity: impl Into<Arity>,
    ) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .invoke)(self, vm, symbol, arity.into())
    }

    /// Invokes `add` on the underlying type's [`TypeVtable`].
    pub fn add(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .add)(self, vm, rhs)
    }

    /// Invokes `add_right` on the underlying type's [`TypeVtable`].
    pub fn add_right(&self, vm: &mut VmContext<'_, '_>, lhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .add_right)(self, vm, lhs)
    }

    /// Invokes `sub` on the underlying type's [`TypeVtable`].
    pub fn sub(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .sub)(self, vm, rhs)
    }

    /// Invokes `sub_right` on the underlying type's [`TypeVtable`].
    pub fn sub_right(&self, vm: &mut VmContext<'_, '_>, lhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .sub_right)(self, vm, lhs)
    }

    /// Invokes `mul` on the underlying type's [`TypeVtable`].
    pub fn mul(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .mul)(self, vm, rhs)
    }

    /// Invokes `mul_right` on the underlying type's [`TypeVtable`].
    pub fn mul_right(&self, vm: &mut VmContext<'_, '_>, lhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .mul_right)(self, vm, lhs)
    }

    /// Invokes `div` on the underlying type's [`TypeVtable`].
    pub fn div(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .div)(self, vm, rhs)
    }

    /// Invokes `div_right` on the underlying type's [`TypeVtable`].
    pub fn div_right(&self, vm: &mut VmContext<'_, '_>, lhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .div_right)(self, vm, lhs)
    }

    /// Invokes `rem` on the underlying type's [`TypeVtable`].
    pub fn rem(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .rem)(self, vm, rhs)
    }

    /// Invokes `rem_right` on the underlying type's [`TypeVtable`].
    pub fn rem_right(&self, vm: &mut VmContext<'_, '_>, lhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .rem_right)(self, vm, lhs)
    }

    /// Invokes `idiv` on the underlying type's [`TypeVtable`].
    pub fn idiv(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .idiv)(self, vm, rhs)
    }

    /// Invokes `idiv_right` on the underlying type's [`TypeVtable`].
    pub fn idiv_right(&self, vm: &mut VmContext<'_, '_>, lhs: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .idiv_right)(self, vm, lhs)
    }

    /// Invokes `hash` on the underlying type's [`TypeVtable`].
    pub fn hash(&self, vm: &mut VmContext<'_, '_>, hasher: &mut ValueHasher) {
        let Some(value) = self.0.load_mapped::<dyn CustomType>(vm.as_ref()) else {
            return;
        };
        (value.muse_type().clone().vtable.hash)(self, vm, hasher);
    }

    /// Invokes `bitwise_not` on the underlying type's [`TypeVtable`].
    pub fn bitwise_not(&self, vm: &mut VmContext<'_, '_>) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .bitwise_not)(self, vm)
    }

    /// Invokes `bitwise_and` on the underlying type's [`TypeVtable`].
    pub fn bitwise_and(&self, vm: &mut VmContext<'_, '_>, other: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .bitwise_and)(self, vm, other)
    }

    /// Invokes `bitwise_or` on the underlying type's [`TypeVtable`].
    pub fn bitwise_or(&self, vm: &mut VmContext<'_, '_>, other: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .bitwise_or)(self, vm, other)
    }

    /// Invokes `bitwise_xor` on the underlying type's [`TypeVtable`].
    pub fn bitwise_xor(&self, vm: &mut VmContext<'_, '_>, other: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .bitwise_xor)(self, vm, other)
    }

    /// Invokes `shift_left` on the underlying type's [`TypeVtable`].
    pub fn shift_left(&self, vm: &mut VmContext<'_, '_>, amount: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .shift_left)(self, vm, amount)
    }

    /// Invokes `shift_right` on the underlying type's [`TypeVtable`].
    pub fn shift_right(&self, vm: &mut VmContext<'_, '_>, amount: &Value) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .shift_right)(self, vm, amount)
    }

    /// Invokes `negate` on the underlying type's [`TypeVtable`].
    pub fn negate(&self, vm: &mut VmContext<'_, '_>) -> Result<Value, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .negate)(self, vm)
    }

    /// Invokes `to_string` on the underlying type's [`TypeVtable`].
    pub fn to_string(&self, vm: &mut VmContext<'_, '_>) -> Result<SymbolRef, Fault> {
        (self
            .0
            .load_mapped::<dyn CustomType>(vm.as_ref())
            .ok_or(ValueFreed)?
            .muse_type()
            .clone()
            .vtable
            .to_string)(self, vm)
    }

    /// Invokes `truthy` on the underlying type's [`TypeVtable`].
    pub fn truthy(&self, vm: &mut VmContext<'_, '_>) -> bool {
        let Some(value) = self.0.load_mapped::<dyn CustomType>(vm.as_ref()) else {
            return false;
        };
        (value.muse_type().clone().vtable.truthy)(self, vm)
    }

    /// Invokes `eq` on the underlying type's [`TypeVtable`].
    pub fn eq(&self, vm: ContextOrGuard<'_, '_, '_>, rhs: &Value) -> Result<bool, Fault> {
        match rhs {
            Value::Dynamic(dynamic) if self.0 == dynamic.0 => Ok(true),
            _ => (self
                .0
                .load_mapped::<dyn CustomType>(vm.as_ref())
                .ok_or(ValueFreed)?
                .muse_type()
                .clone()
                .vtable
                .eq)(self, vm, rhs),
        }
    }

    /// Invokes `matches` on the underlying type's [`TypeVtable`].
    pub fn matches(&self, vm: &mut VmContext<'_, '_>, rhs: &Value) -> Result<bool, Fault> {
        match rhs {
            Value::Dynamic(dynamic) if self.0 == dynamic.0 => Ok(true),
            _ => (self
                .0
                .load_mapped::<dyn CustomType>(vm.as_ref())
                .ok_or(ValueFreed)?
                .muse_type()
                .clone()
                .vtable
                .matches)(self, vm, rhs),
        }
    }

    /// Invokes `total_cmp` on the underlying type's [`TypeVtable`].
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

/// A strong reference to a [`Rooted<T>`].
#[derive(Clone)]
pub struct AnyDynamicRoot(pub(crate) AnyRoot);

impl ContainsNoRefs for AnyDynamicRoot {}

impl Debug for AnyDynamicRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = CollectionGuard::acquire();
        let Some(value) = self.0.as_any().load_mapped::<dyn CustomType>(&guard) else {
            return f.write_str("<Deallocated>");
        };
        Debug::fmt(value, f)
    }
}

impl AnyDynamicRoot {
    /// Returns a weak reference to this dynamic.
    pub const fn downgrade(&self) -> AnyDynamic {
        AnyDynamic(self.0.as_any())
    }
    /// Returns `value` as a garbage collected value that can be used in Muse.
    pub fn new<'guard, T>(value: T, guard: &impl AsRef<CollectionGuard<'guard>>) -> Self
    where
        T: DynamicValue + Trace,
    {
        Self(Root::new(Custom(value), guard).into_any_root())
    }

    /// Upgrades this reference to a [`Dynamic<T>`].
    ///
    /// This function does no type checking. If `T` is the incorrect type,
    /// trying to access the underyling data will return None/an error.
    #[must_use]
    pub fn as_dynamic<T>(&self) -> Dynamic<T>
    where
        T: DynamicValue + Trace,
    {
        Dynamic(self.0.downcast_ref::<Custom<T>>())
    }

    /// Tries to upgrade this reference to a [`Rooted<T>`].
    ///
    /// This function can return None if:
    ///
    /// - `T` is not the correct type.
    /// - The value has been garbage collected.
    #[must_use]
    pub fn as_rooted<T>(&self) -> Option<Rooted<T>>
    where
        T: DynamicValue + Trace,
    {
        self.0.downcast_root::<Custom<T>>().map(|cast| Rooted(cast))
    }

    /// Tries to load a reference to this reference's underlying data.
    ///
    /// This function can return None if:
    ///
    /// - `T` is not the correct type.
    /// - The value has been garbage collected.
    #[must_use]
    pub fn downcast_ref<T>(&self) -> Option<&T>
    where
        T: DynamicValue + Trace,
    {
        self.0
            .load::<Custom<T>>()
            .and_then(|d| d.0.as_any().downcast_ref())
    }
}

/// A reference counted pointer to a garbage collected value.
pub struct Rooted<T: CustomType + Trace>(Root<Custom<T>>);
impl<T> Rooted<T>
where
    T: CustomType + Trace,
{
    /// Moves `value` into the garbage collector and returns a rooted reference
    /// to it.
    #[must_use]
    pub fn new<'guard>(value: T, guard: &impl AsRef<CollectionGuard<'guard>>) -> Self {
        Self(Root::new(Custom(value), guard))
    }

    /// Returns a weak, typeless reference to this value.
    #[must_use]
    pub const fn as_any_dynamic(&self) -> AnyDynamic {
        AnyDynamic(self.0.downgrade_any())
    }

    /// Returns a weak reference to this value.
    #[must_use]
    pub const fn downgrade(&self) -> Dynamic<T> {
        Dynamic(self.0.downgrade())
    }

    /// Returns this value as a typeless root reference.
    #[must_use]
    pub fn as_any_root(&self) -> AnyDynamicRoot {
        AnyDynamicRoot(self.0.to_any_root())
    }

    /// Returns this value as a typeless root reference.
    #[must_use]
    pub fn into_any_root(self) -> AnyDynamicRoot {
        AnyDynamicRoot(self.0.into_any_root())
    }
}

impl<T> Trace for Rooted<T>
where
    T: CustomType + Trace,
{
    const MAY_CONTAIN_REFERENCES: bool = false;

    fn trace(&self, _tracer: &mut refuse::Tracer) {}
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

impl<T> Debug for Rooted<T>
where
    T: Debug + CustomType + Trace,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&**self, f)
    }
}

/// A weak reference to a [`Rooted<T>`].
pub struct Dynamic<T: CustomType>(Ref<Custom<T>>);

impl<T> Dynamic<T>
where
    T: CustomType + Trace,
{
    /// Moves `value` into the garbage collector and returns a weak reference to
    /// it.
    #[must_use]
    pub fn new<'guard>(value: T, guard: &impl AsRef<CollectionGuard<'guard>>) -> Self {
        Self(Ref::new(Custom(value), guard))
    }

    /// Loads a reference to the underlying data, if it has not been collected.
    #[must_use]
    pub fn load<'guard>(&self, guard: &'guard CollectionGuard) -> Option<&'guard T> {
        self.0.load(guard).map(|c| &c.0)
    }

    /// Loads a reference to the underlying data, if it has not been collected.
    pub fn try_load<'guard>(
        &self,
        guard: &'guard CollectionGuard,
    ) -> Result<&'guard T, ValueFreed> {
        self.load(guard).ok_or(ValueFreed)
    }

    /// Loads a rooted reference to the underlying data, if it has not been
    /// collected.
    #[must_use]
    pub fn as_rooted(&self, guard: &CollectionGuard<'_>) -> Option<Rooted<T>> {
        self.0.as_root(guard).map(Rooted)
    }

    /// Returns this dynamic as a Value.
    #[must_use]
    pub const fn into_value(self) -> Value {
        Value::Dynamic(self.into_any_dynamic())
    }

    /// Returns this reference with its type erased.
    #[must_use]
    pub const fn into_any_dynamic(self) -> AnyDynamic {
        AnyDynamic(self.0.as_any())
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

impl<T> PartialEq for Dynamic<T>
where
    T: CustomType + Trace,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.as_any() == other.0.as_any()
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

/// A weak reference could not be loaded because the underlying data has been
/// freed.
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

/// A static [`TypeRef`].
pub struct StaticType(OnceLock<TypeRef>, fn() -> Type);

impl StaticType {
    /// creates a new static type from the given type initializer.
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

/// A static [`TypeRef`] defined in Rust.
pub struct RustType<T>(
    OnceLock<TypeRef>,
    &'static str,
    fn(RustTypeBuilder<T>) -> RustTypeBuilder<T>,
);

impl<T> RustType<T> {
    /// Returns a new type of the given name and initializer.
    pub const fn new(
        name: &'static str,
        init: fn(RustTypeBuilder<T>) -> RustTypeBuilder<T>,
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
            .get_or_init(|| self.2(RustTypeBuilder::new(self.1)).seal(&CollectionGuard::acquire()))
    }
}

/// A Muse type definition.
#[derive(Trace)]
pub struct Type {
    /// The name of the type.
    pub name: Symbol,
    /// The virtual function table.
    pub vtable: TypeVtable,
}

impl Type {
    /// Returns a new type with the default virtual table.
    pub fn new(name: impl Into<Symbol>) -> Self {
        Self {
            name: name.into(),
            vtable: TypeVtable::default(),
        }
    }

    /// Replaces the constructor with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
    #[must_use]
    pub fn with_construct<Func>(mut self, func: impl FnOnce(ConstructFn) -> Func) -> Self
    where
        Func: Fn(&mut VmContext<'_, '_>, Arity) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.construct = Box::new(func(self.vtable.construct));
        self
    }

    /// Replaces the call handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the invoke handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the hash handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
    #[must_use]
    pub fn with_hash<Func>(mut self, func: impl FnOnce(HashFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>, &mut ValueHasher) + Send + Sync + 'static,
    {
        self.vtable.hash = Box::new(func(self.vtable.hash));
        self
    }

    /// Replaces the bitwise not handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
    #[must_use]
    pub fn with_bitwise_not<Func>(mut self, func: impl FnOnce(UnaryFn) -> Func) -> Self
    where
        Func:
            Fn(&AnyDynamic, &mut VmContext<'_, '_>) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.bitwise_not = Box::new(func(self.vtable.bitwise_not));
        self
    }

    /// Replaces the bitwise and with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the bitwise or handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the bitwise xor handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the shift left handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the shift right handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the negate handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
    #[must_use]
    pub fn with_negate<Func>(mut self, func: impl FnOnce(UnaryFn) -> Func) -> Self
    where
        Func:
            Fn(&AnyDynamic, &mut VmContext<'_, '_>) -> Result<Value, Fault> + Send + Sync + 'static,
    {
        self.vtable.negate = Box::new(func(self.vtable.negate));
        self
    }

    /// Replaces the eq handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the matches handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the comparison handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the add handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the add-right handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the sub handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the sub-right handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the mul handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the mul-right handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the div handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the div-right handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the idiv handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the idiv-right handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the rem handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the rem-right handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the truthy handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
    #[must_use]
    pub fn with_truthy<Func>(mut self, func: impl FnOnce(TruthyFn) -> Func) -> Self
    where
        Func: Fn(&AnyDynamic, &mut VmContext<'_, '_>) -> bool + Send + Sync + 'static,
    {
        self.vtable.truthy = Box::new(func(self.vtable.truthy));
        self
    }

    /// Replaces the `to_string` handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the `deep_clone` handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces all functions on this table by invoking `mapping` and invoking
    /// the corresponding functionality on the returned value.
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

    /// Finalizes this type for use in Muse.
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

/// A builder for a [`RustType<T>`].
pub struct RustTypeBuilder<T> {
    t: Type,
    _t: PhantomData<T>,
}

impl<T> RustTypeBuilder<T>
where
    T: CustomType + Trace,
{
    fn new(name: &'static str) -> Self {
        Self {
            t: Type::new(name),
            _t: PhantomData,
        }
    }

    /// Replaces the constructor with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the call handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the invoke handler with a handler that invokes the appropriate
    /// function.
    #[must_use]
    pub fn with_function_table(self, functions: RustFunctionTable<T>) -> Self {
        self.with_invoke(Box::new(move |_| {
            move |this, vm: &mut VmContext<'_, '_>, name: &SymbolRef, arity| {
                functions.invoke(vm, name, arity, &this)
            }
        }))
    }

    /// Replaces the invoke handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the hash handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the bitwise not handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the bitwise and with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the bitwise or handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the bitwise xor handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the shift left handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the shift right handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the negate handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the eq handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the matches handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the comparison handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the add handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the add-right handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the sub handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the sub-right handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the mul handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the mul-right handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the div handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the div-right handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the idiv handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the idiv-right handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the rem handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the rem-right handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the truthy handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the `to_string` handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the `deep_clone` handler with `func`.
    ///
    /// `func` is a function that returns the actual handler function. It takes
    /// a single parameter: the existing handler. This design allows for
    /// functions to be overridden while still falling back to the previous
    /// implementation.
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

    /// Replaces the `deep_clone` handler with an implementation that uses
    /// [`Clone`].
    #[must_use]
    pub fn with_clone(self) -> Self
    where
        T: Clone,
    {
        self.with_deep_clone(|_| |this, guard| Some(AnyDynamic::new((*this).clone(), guard)))
    }

    /// Replaces all functions on this table by invoking `mapping` and invoking
    /// the corresponding functionality on the returned value.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn with_fallback<Mapping>(mut self, mapping: Mapping) -> Self
    where
        Mapping: Fn(Rooted<T>, &CollectionGuard) -> Value + Send + Sync + Clone + 'static,
    {
        self.t = self.t.with_fallback(move |dynamic, guard| {
            dynamic
                .as_rooted::<T>(guard)
                .map_or(Value::NIL, |rooted| mapping(rooted, guard))
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

/// A reference to a [`CollectionGuard`] or a [`VmContext`].
///
/// This type is used when a function may need to be invoked with or without a
/// [`VmContext`].
pub enum ContextOrGuard<'a, 'context, 'guard> {
    /// A collection guard.
    Guard(&'a CollectionGuard<'guard>),
    /// An execution context.
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
    /// Returns a reference to the execution context, if available.
    pub fn vm(&mut self) -> Option<&mut VmContext<'context, 'guard>> {
        let Self::Context(context) = self else {
            return None;
        };
        Some(context)
    }

    /// Returns a new [`ContextOrGuard`] that borrows from `self`.
    pub fn borrowed(&mut self) -> ContextOrGuard<'_, 'context, 'guard> {
        match self {
            ContextOrGuard::Guard(guard) => ContextOrGuard::Guard(guard),
            ContextOrGuard::Context(vm) => ContextOrGuard::Context(vm),
        }
    }
}

/// A reference to a [`Type`].
pub type TypeRef = Rooted<Type>;

/// A boxed constructor used in a [`TypeVtable`].
pub type ConstructFn =
    Box<dyn Fn(&mut VmContext<'_, '_>, Arity) -> Result<Value, Fault> + Send + Sync>;
/// A boxed call handler used in a [`TypeVtable`].
pub type CallFn =
    Box<dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>, Arity) -> Result<Value, Fault> + Send + Sync>;
/// A boxed hash handler used in a [`TypeVtable`].
pub type HashFn = Box<dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>, &mut ValueHasher) + Send + Sync>;
/// A boxed single-argument handler used in a [`TypeVtable`].
pub type UnaryFn =
    Box<dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>) -> Result<Value, Fault> + Send + Sync>;
/// A boxed two-argument handler used in a [`TypeVtable`].
pub type BinaryFn =
    Box<dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Value, Fault> + Send + Sync>;
/// A boxed matches handler used in a [`TypeVtable`].
pub type MatchesFn =
    Box<dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<bool, Fault> + Send + Sync>;
/// A boxed eq handler used in a [`TypeVtable`].
pub type EqFn = Box<
    dyn Fn(&AnyDynamic, ContextOrGuard<'_, '_, '_>, &Value) -> Result<bool, Fault> + Send + Sync,
>;
/// A boxed comparison handler used in a [`TypeVtable`].
pub type TotalCmpFn = Box<
    dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>, &Value) -> Result<Ordering, Fault> + Send + Sync,
>;
/// A boxed invoke handler used in a [`TypeVtable`].
pub type InvokeFn = Box<
    dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>, &SymbolRef, Arity) -> Result<Value, Fault>
        + Send
        + Sync,
>;
/// A boxed `deep_clone` handler used in a [`TypeVtable`].
pub type DeepCloneFn =
    Box<dyn Fn(&AnyDynamic, &CollectionGuard) -> Option<AnyDynamic> + Send + Sync>;
/// A boxed truthy handler used in a [`TypeVtable`].
pub type TruthyFn = Box<dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>) -> bool + Send + Sync>;
/// A boxed `to_string` handler used in a [`TypeVtable`].
pub type ToStringFn =
    Box<dyn Fn(&AnyDynamic, &mut VmContext<'_, '_>) -> Result<SymbolRef, Fault> + Send + Sync>;

/// A virtual function table for a [`Type`].
pub struct TypeVtable {
    construct: ConstructFn,
    pub(crate) call: CallFn,
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

/// A mapping from a Rust type to its Muse [`Type`].
pub trait CustomType: Send + Sync + Debug + 'static {
    /// Returns the Muse type for this Rust type.
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

/// A table containing Rust-defined functions for various names and arities.
pub struct RustFunctionTable<T> {
    functions: Map<Symbol, Map<Arity, Arc<dyn RustFn<T>>>>,
}

impl<T> RustFunctionTable<T>
where
    T: CustomType + Trace,
{
    /// Returns an empty function table.
    #[must_use]
    #[allow(clippy::new_without_default)]
    pub const fn new() -> Self {
        Self {
            functions: Map::new(),
        }
    }

    /// Adds a function with `name` and `arity` to this table, and returns self.q
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

    /// Invokes a function named `name` with `arity` arguments.
    ///
    /// # Errors
    ///
    /// - [`Fault::UnknownSymbol`]: Returned if this table has no functions with
    ///       the given name.
    /// - [`Fault::IncorrectNumberOfArguments`]: Returned if no functions with
    ///       the given name accept the given number of arguments.
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

/// A statically defined [`RustFunctionTable`].
pub struct StaticRustFunctionTable<T>(
    OnceLock<RustFunctionTable<T>>,
    fn(RustFunctionTable<T>) -> RustFunctionTable<T>,
);

impl<T> StaticRustFunctionTable<T> {
    /// Returns a new static function table that initializes itself using
    /// `init`.
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

/// A Rust function that can be stored in a [`Value`] and called.
#[derive(Clone)]
pub struct RustFunction(ArcRustFn);

impl RustFunction {
    /// Returns a new function that invokes `function` when called.
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

/// An asynchronous Rust function that can be stored in a [`Value`] and called.
#[derive(Clone)]
pub struct AsyncFunction(ArcAsyncFunction);

impl AsyncFunction {
    /// Returns a new function that invokes `function` and awaits the returned
    /// future when called.
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
                            vm.guard(),
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

/// A [`CustomType`] that can be type-erased.
pub trait DynamicValue: CustomType {
    /// Returns `self` as an [`Any`].
    fn as_any(&self) -> &dyn Any;
    /// Returns `self` as a mut [`Any`].
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
            static TYPE: RustType<usize> = RustType::new("usize", RustTypeBuilder::with_clone);
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
        RustFunction::new(|_vm: &mut VmContext<'_, '_>, _arity| {
            Ok(Value::Primitive(Primitive::Int(1)))
        }),
        &guard,
    ));
    let runtime = crate::vm::Vm::new(&guard);
    let Value::Primitive(Primitive::Int(i)) = func
        .call(&mut VmContext::new(&runtime, &mut guard), 0)
        .unwrap()
    else {
        unreachable!()
    };
    assert_eq!(i, 1);
}

/// A Muse virtual machine value that holds strong references.
#[derive(Clone, Debug)]
pub enum RootedValue {
    Primitive(Primitive),
    /// A symbol.
    Symbol(Symbol),
    /// A dynamically allocated, garbage collected type.
    Dynamic(AnyDynamicRoot),
}

impl RootedValue {
    pub const TRUE: Self = Self::Primitive(Primitive::Bool(true));
    pub const FALSE: Self = Self::Primitive(Primitive::Bool(false));
    pub const NIL: Self = Self::Primitive(Primitive::Nil);
    pub const ZERO: Self = Self::Primitive(Primitive::Int(0));

    /// Returns this value with weak references to any garbage collected data.
    pub fn downgrade(&self) -> Value {
        match self {
            Self::Primitive(p) => Value::Primitive(*p),
            Self::Symbol(v) => Value::Symbol(v.downgrade()),
            Self::Dynamic(v) => Value::Dynamic(v.downgrade()),
        }
    }

    /// Moves `value` into the virtual machine.
    pub fn dynamic<'guard, T>(value: T, guard: &impl AsRef<CollectionGuard<'guard>>) -> Self
    where
        T: DynamicValue + Trace,
    {
        Self::Dynamic(AnyDynamicRoot::new(value, guard))
    }

    /// Returns true if this value is nil.
    #[must_use]
    pub const fn is_nil(&self) -> bool {
        matches!(self, Self::Primitive(Primitive::Nil))
    }

    /// Returns this value as an i64, if possible.
    #[must_use]
    pub fn as_primitive(&self) -> Option<Primitive> {
        match self {
            Self::Primitive(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns this value as an i64, if possible.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        self.as_primitive().and_then(|p| p.as_i64())
    }

    /// Returns this value as an u64, if possible.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        self.as_primitive().and_then(|p| p.as_u64())
    }

    /// Returns this value as an u32, if possible.
    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        self.as_primitive().and_then(|p| p.as_u32())
    }

    /// Returns this value as an u16, if possible.
    #[must_use]
    pub fn as_u16(&self) -> Option<u16> {
        self.as_primitive().and_then(|p| p.as_u16())
    }

    /// Returns this value as an usize, if possible.
    #[must_use]
    pub fn as_usize(&self) -> Option<usize> {
        self.as_primitive().and_then(|p| p.as_usize())
    }

    /// Returns this value as an f64, if possible.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        self.as_primitive().and_then(|p| p.as_f64())
    }

    /// Converts this value to an i64, if possible.
    #[must_use]
    pub fn to_i64(&self) -> Option<i64> {
        match self {
            Self::Primitive(value) => value.to_i64(),
            Self::Symbol(_) | Self::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    /// Converts this value to an u64, if possible.
    #[must_use]
    pub fn to_u64(&self) -> Option<u64> {
        match self {
            Self::Primitive(value) => value.to_u64(),
            Self::Symbol(_) | Self::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    /// Converts this value to an u32, if possible.
    #[must_use]
    pub fn to_u32(&self) -> Option<u32> {
        match self {
            Self::Primitive(value) => value.to_u32(),
            Self::Symbol(_) | Self::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    /// Converts this value to an usize, if possible.
    #[must_use]
    pub fn to_usize(&self) -> Option<usize> {
        match self {
            Self::Primitive(value) => value.to_usize(),
            Self::Symbol(_) | Self::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    /// Converts this value to an f64, if possible.
    #[must_use]
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            Self::Primitive(value) => Some(value.to_f64()),
            Self::Symbol(_) | Self::Dynamic(_) => None, // TODO offer dynamic conversion
        }
    }

    /// Returns this value as a `SymbolRef`, if possible.
    #[must_use]
    pub fn as_symbol_ref(&self) -> Option<SymbolRef> {
        match self {
            Self::Symbol(value) => Some(value.downgrade()),
            _ => None,
        }
    }

    /// Returns this value as a `Symbol`, if possible.
    #[must_use]
    pub fn as_symbol(&self) -> Option<&Symbol> {
        match self {
            Self::Symbol(value) => Some(value),
            _ => None,
        }
    }

    /// Returns this value as an `AnyDynamic`, if possible.
    #[must_use]
    pub fn as_any_dynamic(&self) -> Option<AnyDynamic> {
        match self {
            Self::Dynamic(value) => Some(value.downgrade()),
            _ => None,
        }
    }

    /// Returns this value as a `Dynamic<T>`, if this value contains a `T`.
    #[must_use]
    pub fn as_dynamic<T>(&self) -> Option<Dynamic<T>>
    where
        T: DynamicValue + Trace,
    {
        match self {
            Self::Dynamic(value) => Some(value.as_dynamic()),
            _ => None,
        }
    }

    /// Returns this value as a `Rooted<T>`, if this value contains a `T`.
    #[must_use]
    pub fn as_rooted<T>(&self) -> Option<Rooted<T>>
    where
        T: DynamicValue + Trace,
    {
        match self {
            Self::Dynamic(value) => value.as_rooted(),
            _ => None,
        }
    }

    /// Returns this value as a`&T`, if this value contains a `T`.
    #[must_use]
    pub fn as_downcast_ref<T>(&self) -> Option<&T>
    where
        T: DynamicValue + Trace,
    {
        match self {
            Self::Dynamic(value) => value.downcast_ref(),
            _ => None,
        }
    }

    /// Returns true if this value should be considered `true` in a boolean
    /// expression.
    pub fn truthy(&self, vm: &mut VmContext<'_, '_>) -> bool {
        match self {
            Self::Primitive(value) => value.truthy(),
            Self::Symbol(sym) => !sym.is_empty(),
            Self::Dynamic(value) => value.downgrade().truthy(vm),
        }
    }
}

impl Default for RootedValue {
    fn default() -> Self {
        Self::NIL
    }
}

impl PartialEq for RootedValue {
    fn eq(&self, other: &Self) -> bool {
        let other = other.downgrade();
        self.downgrade()
            .equals(ContextOrGuard::Guard(&CollectionGuard::acquire()), &other)
            .unwrap_or(false)
    }
}

impl_from_primitive!(RootedValue, f32, Float);
impl_from_primitive!(RootedValue, f64, Float);
impl_from_primitive!(RootedValue, i8, Int);
impl_from_primitive!(RootedValue, i16, Int);
impl_from_primitive!(RootedValue, i32, Int);
impl_from_primitive!(RootedValue, i64, Int);
impl_from_primitive!(RootedValue, u8, UInt);
impl_from_primitive!(RootedValue, u16, UInt);
impl_from_primitive!(RootedValue, u32, UInt);
impl_from_primitive!(RootedValue, u64, UInt);
impl_from_primitive!(RootedValue, bool, Bool);
impl_from!(RootedValue, Symbol, Symbol);
impl_from!(RootedValue, &'_ Symbol, Symbol);
impl_try_from_primitive!(RootedValue, u128);
impl_try_from_primitive!(RootedValue, usize);
impl_try_from_primitive!(RootedValue, isize);
impl_try_from_primitive!(RootedValue, i128);
