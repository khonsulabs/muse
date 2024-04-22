//! Muse-defined custom types.

use kempt::Map;
use refuse::{CollectionGuard, ContainsNoRefs, Trace};
use serde::{Deserialize, Serialize};

use crate::vm::{
    bitcode::{Access, Accessable, BitcodeFunction},
    Arity, Fault, Function, Register,
};

use super::{
    symbol::{Symbol, SymbolRef},
    value::{CustomType, Rooted, Type, TypeRef, Value},
};

/// An IR Muse-defined type.
#[derive(PartialEq, Clone, Debug, Serialize, Deserialize)]
pub struct BitcodeType {
    /// The name of the type.
    pub name: Symbol,
    /// The functions defined on the type.
    pub functions: Map<Symbol, Accessable<BitcodeFunction>>,
    /// The fields defined on members of this type.
    pub fields: Map<Symbol, Access>,
}

impl BitcodeType {
    pub(crate) fn load(&self, guard: &CollectionGuard<'_>, module: usize) -> DefinedType {
        let functions = self
            .functions
            .iter()
            .map(|field| (field.key().clone(), field.value.to_function(guard, module)))
            .collect::<Map<_, _>>();

        let mut ty = Type::new(self.name.clone());

        if !functions.is_empty() {
            let functions = functions.clone();
            ty = ty.with_invoke(|fallback| {
                move |this, vm, name, arity| {
                    println!("invoking on instance");
                    if let Some(func) = functions.get(name) {
                        // TODO verify access

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
                            .downcast_ref::<Instance>(vm.guard())
                            .ok_or(Fault::ValueFreed)?;
                        if let Some(field) = loaded.fields.get(field_name) {
                            // TODO verify access
                            Ok(field.accessable)
                        } else {
                            Err(Fault::UnknownSymbol)
                        }
                    } else {
                        fallback(this, vm, name, arity)
                    }
                }
            });
        }

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
                    Instance {
                        ty: instance.clone(),
                        fields,
                    },
                    vm.guard(),
                ))
            }
        });

        if !functions.is_empty() {
            let functions = functions.clone();
            ty = ty.with_invoke(|fallback| {
                move |this, vm, name, arity| {
                    if let Some(func) = functions.get(name) {
                        // TODO verify access

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
        }

        let loaded = ty.seal(guard);

        DefinedType {
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
        module: usize,
    ) -> Accessable<Rooted<Function>> {
        Accessable {
            access: self.access,
            accessable: Rooted::new(self.accessable.to_function(guard).in_module(module), guard),
        }
    }
}

/// A loaded Muse-defined type.
#[derive(Debug)]
pub struct DefinedType {
    loaded: TypeRef,
    functions: Map<Symbol, Accessable<Rooted<Function>>>,
    fields: Map<Symbol, Access>,
}

impl DefinedType {
    /// Converts this type back into a [`BitcodeType`].
    pub fn to_bitcode_type(&self, guard: &CollectionGuard<'_>) -> BitcodeType {
        BitcodeType {
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

impl CustomType for DefinedType {
    fn muse_type(&self) -> &TypeRef {
        &self.loaded
    }
}

impl ContainsNoRefs for DefinedType {}

#[derive(Debug, Trace)]
struct Instance {
    ty: TypeRef,
    fields: Map<SymbolRef, Accessable<Value>>,
}

impl CustomType for Instance {
    fn muse_type(&self) -> &TypeRef {
        &self.ty
    }
}
