use std::fmt::{self, Debug};

use refuse::Trace;

use crate::syntax::Sources;
use crate::value::{AnyDynamic, CustomType, RustType, TypeRef, Value};
use crate::vm::{Code, StackFrame, VmContext};

#[derive(Debug)]
pub struct Exception {
    value: Value,
    stack_trace: Vec<StackFrame>,
}

impl Exception {
    pub fn new(value: Value, vm: &mut VmContext<'_, '_>) -> Self {
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

    pub fn format(
        &self,
        sources: &Sources,
        context: &mut VmContext<'_, '_>,
        mut f: impl fmt::Write,
    ) -> fmt::Result {
        f.write_str("uncaught exception: ")?;
        self.value.format(context, &mut f)?;
        if !self.stack_trace.is_empty() {
            for entry in &self.stack_trace {
                f.write_char('\n')?;
                if let Some(range) = entry.source_range() {
                    if let Some(source) = sources.get(range.source_id) {
                        let (line_no, start) = source.offset_to_line(range.start);
                        write!(f, "in {}:{line_no}:{start}", source.name)?;
                        continue;
                    }
                }
                write!(f, "in {:?}", entry.code() as *const Code)?;
            }
        }
        Ok(())
    }
}

impl CustomType for Exception {
    fn muse_type(&self) -> &TypeRef {
        static EXCEPTION_TYPE: RustType<Exception> = RustType::new("Exception", |t| {
            t.with_fallback(|this, _guard| this.value)
                .with_eq(|_| {
                    |this, vm, rhs| {
                        if let Some(rhs) = rhs.as_rooted::<Exception>(vm.as_ref()) {
                            Ok(this.value.equals(vm, &rhs.value)?
                                && this.stack_trace == rhs.stack_trace)
                        } else {
                            Ok(false)
                        }
                    }
                })
                .with_matches(|_| |this, vm, rhs| this.value.matches(vm, rhs))
                .with_deep_clone(|_| {
                    |this, guard| {
                        this.value.deep_clone(guard).map(|value| {
                            AnyDynamic::new(
                                Exception {
                                    value,
                                    stack_trace: this.stack_trace.clone(),
                                },
                                guard,
                            )
                        })
                    }
                })
        });
        &EXCEPTION_TYPE
    }
}

impl Trace for Exception {
    const MAY_CONTAIN_REFERENCES: bool = true;

    fn trace(&self, tracer: &mut refuse::Tracer) {
        if let Some(dynamic) = self.value.as_any_dynamic() {
            tracer.mark(dynamic);
        }
    }
}
