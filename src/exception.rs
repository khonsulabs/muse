use crate::symbol::Symbol;
use crate::value::{CustomType, Dynamic, Value};
use crate::vm::{Fault, StackFrame, Vm};

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
    fn eq(&self, vm: Option<&mut Vm>, rhs: &Value) -> Result<bool, Fault> {
        if let Some(rhs) = rhs.as_downcast_ref::<Self>() {
            Ok(self.value.equals(vm, &rhs.value)? && self.stack_trace == rhs.stack_trace)
        } else {
            Ok(false)
        }
    }

    fn matches(&self, vm: &mut Vm, rhs: &Value) -> Result<bool, Fault> {
        self.value.matches(vm, rhs)
    }

    fn total_cmp(&self, vm: &mut Vm, rhs: &Value) -> Result<std::cmp::Ordering, Fault> {
        self.value.total_cmp(vm, rhs)
    }

    fn invoke(&self, vm: &mut Vm, name: &Symbol, arity: crate::vm::Arity) -> Result<Value, Fault> {
        self.value.invoke(vm, name, arity)
    }

    fn to_string(&self, vm: &mut Vm) -> Result<Symbol, Fault> {
        self.value.to_string(vm)
    }

    fn deep_clone(&self) -> Option<Dynamic> {
        self.value.deep_clone().map(|value| {
            Dynamic::new(Self {
                value,
                stack_trace: self.stack_trace.clone(),
            })
        })
    }
}
