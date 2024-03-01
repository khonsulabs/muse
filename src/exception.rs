use crate::value::{AnyDynamic, CustomType, RustType, TypeRef, Value};
use crate::vm::{StackFrame, Vm};

#[derive(Debug)]
pub struct Exception {
    value: Value,
    stack_trace: Vec<StackFrame>,
}

impl Exception {
    pub fn new(value: Value, vm: &mut Vm) -> Self {
        let stack_trace = vm.stack_trace();
        Self { value, stack_trace }
    }

    #[must_use]
    pub const fn value(&self) -> &Value {
        &self.value
    }

    #[must_use]
    pub fn backtrace(&self) -> &[StackFrame] {
        &self.stack_trace
    }
}

impl CustomType for Exception {
    fn muse_type(&self) -> &TypeRef {
        static EXCEPTION_TYPE: RustType<Exception> = RustType::new("Exception", |t| {
            t.with_fallback(|this| this.value.clone())
                .with_eq(|_| {
                    |this, vm, rhs| {
                        if let Some(rhs) = rhs.as_downcast_ref::<Exception>() {
                            Ok(this.value.equals(vm, &rhs.value)?
                                && this.stack_trace == rhs.stack_trace)
                        } else {
                            Ok(false)
                        }
                    }
                })
                .with_matches(|_| |this, vm, rhs| this.value.matches(vm, rhs))
                .with_deep_clone(|_| {
                    |this| {
                        this.value.deep_clone().map(|value| {
                            AnyDynamic::new(Exception {
                                value,
                                stack_trace: this.stack_trace.clone(),
                            })
                        })
                    }
                })
        });
        &EXCEPTION_TYPE
    }
}
