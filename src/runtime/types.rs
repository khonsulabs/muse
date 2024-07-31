//! Muse-defined custom types.

use kempt::Map;
use refuse::{CollectionGuard, ContainsNoRefs, Trace};
use serde::{Deserialize, Serialize};

use super::symbol::{Symbol, SymbolRef};
use super::value::{CustomType, Rooted, Type, TypeRef, Value};
use crate::vm::bitcode::{Access, Accessable, BitcodeFunction, ValueOrSource};
use crate::vm::{Arity, Fault, Function, ModuleId, Register, VmContext};

/// An IR Muse-defined struct.
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
pub struct BitcodeStruct {
    /// The name of the struct.
    pub name: Symbol,
    /// The functions defined on the struct.
    pub functions: Map<Symbol, Accessable<BitcodeFunction>>,
    /// The fields defined on members of this struct.
    pub fields: Map<Symbol, Access>,
}

impl BitcodeStruct {
    pub(crate) fn load(&self, guard: &CollectionGuard<'_>, module: ModuleId) -> RuntimeStruct {
        let functions = self
            .functions
            .iter()
            .map(|field| (field.key().clone(), field.value.to_function(guard, module)))
            .collect::<Map<_, _>>();

        let mut ty = Type::new(self.name.clone());

        ty = ty.with_invoke(|fallback| {
            let functions = functions.clone();
            move |this, vm, name, arity| {
                if let Some(func) = functions.get(name) {
                    if func.access < vm.caller_access_level_by_index(module) {
                        return Err(Fault::Forbidden);
                    }

                    if arity == 255 {
                        return Err(Fault::InvalidArity);
                    } else if arity.0 > 0 {
                        vm.registers_mut().copy_within(0..usize::from(arity.0), 1);
                    }
                    vm[Register(0)] = Value::Dynamic(*this);
                    (func.accessable.muse_type().vtable.call)(
                        &func.accessable.as_any_dynamic(),
                        vm,
                        Arity(arity.0 + 1),
                    )
                } else if name == Symbol::get_symbol() && arity == 1 {
                    let Some(field_name) = vm[Register(0)].as_symbol_ref() else {
                        return Err(Fault::ExpectedSymbol);
                    };
                    let loaded = this
                        .downcast_ref::<StructInstance>(vm.guard())
                        .ok_or(Fault::ValueFreed)?;
                    if let Some(field) = loaded.fields.get(field_name) {
                        if field.access < dbg!(vm.caller_access_level_by_index(module)) {
                            return Err(Fault::Forbidden);
                        }
                        Ok(field.accessable)
                    } else {
                        Err(Fault::UnknownSymbol)
                    }
                } else {
                    fallback(this, vm, name, arity)
                }
            }
        });

        let instance = ty.seal(guard);

        // TODO the type name for the type itself should probably be distinctive.
        let mut ty = Type::new(self.name.clone()).with_call(|_| {
            let fields = self.fields.clone();
            move |_this, vm, arity| {
                let fields = (0..arity.0 / 2)
                    .map(|index| {
                        let name = vm[Register(index * 2)]
                            .take()
                            .as_symbol(vm.guard())
                            .ok_or(Fault::ExpectedSymbol)?;
                        let access = *fields.get(&name).ok_or(Fault::UnknownSymbol)?;
                        Ok((
                            name.downgrade(),
                            Accessable {
                                access,
                                accessable: vm[Register(index * 2 + 1)].take(),
                            },
                        ))
                    })
                    .collect::<Result<_, Fault>>()?;
                Ok(Value::dynamic(
                    StructInstance {
                        ty: instance.clone(),
                        fields,
                    },
                    vm.guard(),
                ))
            }
        });

        ty = ty.with_invoke(|fallback| {
            let functions = functions.clone();
            move |this, vm, name, arity| {
                if let Some(func) = functions.get(name) {
                    if func.access < vm.caller_access_level_by_index(module) {
                        return Err(Fault::Forbidden);
                    }

                    (func.accessable.muse_type().vtable.call)(
                        &func.accessable.as_any_dynamic(),
                        vm,
                        arity,
                    )
                } else {
                    fallback(this, vm, name, arity)
                }
            }
        });

        let loaded = ty.seal(guard);

        RuntimeStruct {
            loaded,
            functions,
            fields: self.fields.clone(),
        }
    }
}

impl Accessable<BitcodeFunction> {
    fn to_function(
        &self,
        guard: &CollectionGuard<'_>,
        module: ModuleId,
    ) -> Accessable<Rooted<Function>> {
        Accessable {
            access: self.access,
            accessable: Rooted::new(self.accessable.to_function(guard).in_module(module), guard),
        }
    }
}

/// A loaded Muse-defined type.
#[derive(Debug, Clone)]
pub struct RuntimeStruct {
    loaded: TypeRef,
    functions: Map<Symbol, Accessable<Rooted<Function>>>,
    fields: Map<Symbol, Access>,
}

impl RuntimeStruct {
    /// Converts this type back into a [`BitcodeStruct`].
    #[must_use]
    pub fn to_bitcode_type(&self, guard: &CollectionGuard<'_>) -> BitcodeStruct {
        BitcodeStruct {
            name: self.loaded.name.clone(),
            functions: self
                .functions
                .iter()
                .map(|field| {
                    (
                        field.key().clone(),
                        Accessable {
                            access: field.value.access,
                            accessable: BitcodeFunction::from_function(
                                &field.value.accessable,
                                guard,
                            ),
                        },
                    )
                })
                .collect(),
            fields: self.fields.clone(),
        }
    }
}

impl CustomType for RuntimeStruct {
    fn muse_type(&self) -> &TypeRef {
        &self.loaded
    }
}

impl ContainsNoRefs for RuntimeStruct {}

#[derive(Debug, Trace)]
struct StructInstance {
    ty: TypeRef,
    fields: Map<SymbolRef, Accessable<Value>>,
}

impl CustomType for StructInstance {
    fn muse_type(&self) -> &TypeRef {
        &self.ty
    }
}

/// An IR representation of an enum definition.
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
pub struct BitcodeEnum {
    /// The name of this enum.
    pub name: Symbol,
    /// The variants defined in this enum.
    pub variants: Vec<EnumVariant<ValueOrSource>>,
}

impl BitcodeEnum {
    pub(crate) fn load(&self, vm: &VmContext<'_, '_>) -> Result<RuntimeEnum, Fault> {
        let instance = Type::new(self.name.clone())
            .with_total_cmp(|fallback| {
                move |this, vm, rhs| {
                    if let (Some(lhs), Some(rhs)) = (
                        this.downcast_ref::<VariantInstance>(vm.guard()),
                        rhs.as_downcast_ref::<VariantInstance>(vm.guard()),
                    ) {
                        let lhs = lhs.value;
                        let rhs = rhs.value;
                        lhs.total_cmp(vm, &rhs)
                    } else {
                        fallback(this, vm, rhs)
                    }
                }
            })
            .seal(vm.guard());

        let mut variants = Vec::with_capacity(self.variants.len());
        let mut variants_by_name = Map::with_capacity(self.variants.len());

        for (index, variant) in self.variants.iter().enumerate() {
            variants_by_name.insert(variant.name.clone(), index);
            variants.push(EnumVariant {
                name: variant.name.clone(),
                value: Value::dynamic(
                    VariantInstance {
                        ty: instance.clone(),
                        name: variant.name.clone(),
                        value: variant.value.load(vm)?,
                    },
                    vm.guard(),
                ),
            });
        }

        // TODO the type name for the type itself should probably be distinctive.
        let ty = Type::new(self.name.clone()).with_invoke(|fallback| {
            let variants = variants.clone();
            let variants_by_name = variants_by_name.clone();
            move |this, vm, name, arity| {
                if name == Symbol::get_symbol() && arity == 1 {
                    let Some(field_name) = vm[Register(0)].as_symbol_ref() else {
                        return Err(Fault::ExpectedSymbol);
                    };
                    Ok(variants[*variants_by_name
                        .get(field_name)
                        .ok_or(Fault::UnknownSymbol)?]
                    .value)
                } else {
                    fallback(this, vm, name, arity)
                }
            }
        });

        let ty = ty.seal(vm.guard());

        Ok(RuntimeEnum {
            ty,
            variants,
            variants_by_name,
        })
    }
}

/// A Muse enum definition.
#[derive(Debug, Trace, Clone)]
pub struct RuntimeEnum {
    ty: TypeRef,
    variants: Vec<EnumVariant<Value>>,
    variants_by_name: Map<Symbol, usize>,
}

impl RuntimeEnum {
    pub(crate) fn to_bitcode_type(&self, guard: &CollectionGuard) -> BitcodeEnum {
        BitcodeEnum {
            name: self.ty.name.clone(),
            variants: self
                .variants
                .iter()
                .map(|v| EnumVariant {
                    name: v.name.clone(),
                    value: v.value.as_source(guard),
                })
                .collect(),
        }
    }
}

impl CustomType for RuntimeEnum {
    fn muse_type(&self) -> &TypeRef {
        &self.ty
    }
}

/// A variant of an enum.
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize, Trace)]
pub struct EnumVariant<T> {
    /// The name of the variant.
    pub name: Symbol,
    /// The value of the variant.
    pub value: T,
}

#[derive(Debug, Trace)]
struct VariantInstance {
    ty: TypeRef,
    name: Symbol,
    value: Value,
}

impl CustomType for VariantInstance {
    fn muse_type(&self) -> &TypeRef {
        &self.ty
    }
}
