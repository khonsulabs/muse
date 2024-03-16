use std::ops::{BitOr, Deref, DerefMut};
use std::sync::Mutex;

use cushy::value::{Dynamic, IntoValue, Source, Value as CushyValue};
use cushy::widget::{MakeWidget, WidgetInstance};
use cushy::widgets::slider::Slidable;
use cushy::widgets::{Expand, Slider};
use cushy::window::WindowHandle;
use cushy::{App, Application, Open, PendingApp};
use muse::refuse::{self, CollectionGuard, SimpleType, Trace};
use muse::symbol::Symbol;
use muse::value::{ContextOrGuard, CustomType, RustFunction, RustType, TypeRef, Value};
use muse::vm::{Fault, Register, Vm, VmContext};

pub fn install(vm: &Vm, guard: &mut CollectionGuard<'_>) {
    vm.declare(
        "Dynamic",
        Value::dynamic(
            RustFunction::new(|vm: &mut VmContext<'_, '_>, arity| {
                if arity == 1 {
                    Ok(Value::dynamic(
                        DynamicValue(Dynamic::new(vm[Register(0)].take())),
                        vm,
                    ))
                } else {
                    Err(Fault::IncorrectNumberOfArguments)
                }
            }),
            &guard,
        ),
        guard,
    )
    .unwrap();
}

static APP: Mutex<Option<App>> = Mutex::new(None);

pub fn initialize(app: &PendingApp) {
    *APP.lock().expect("poisoned") = Some(app.as_app());
}

fn muse_app() -> Result<App, Fault> {
    if let Some(app) = APP.lock().expect("poisoned").clone() {
        Ok(app)
    } else {
        Err(Fault::UnsupportedOperation)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DynamicValue(pub Dynamic<Value>);

impl Deref for DynamicValue {
    type Target = Dynamic<Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for DynamicValue {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Trace for DynamicValue {
    const MAY_CONTAIN_REFERENCES: bool = true;

    fn trace(&self, tracer: &mut refuse::Tracer) {
        self.0.map_ref(|value| value.trace(tracer));
    }
}

impl CustomType for DynamicValue {
    fn muse_type(&self) -> &TypeRef {
        static TYPE: RustType<DynamicValue> = RustType::new("DynamicValue", |t| {
            t.with_invoke(|_| {
                |this, vm, name, arity| {
                    if name == Symbol::set_symbol() && arity == 1 {
                        let value = vm[Register(0)].take();
                        if let Ok(mut contents) = this.0.try_lock() {
                            if contents.equals(ContextOrGuard::Context(vm), &value)? {
                                Ok(value)
                            } else {
                                let old_value = std::mem::replace(&mut *contents, value);
                                Ok(old_value)
                            }
                        } else {
                            Ok(Value::Nil)
                        }
                    } else if name == &Symbol::from("slider_between") && arity == 2 {
                        let start = vm[Register(0)].take();
                        let end = vm[Register(1)].take();

                        match this.0.map_ref(|this| numeric_kind(this, vm.as_ref()))
                            | numeric_kind(&end, vm.as_ref())
                            | numeric_kind(&start, vm.as_ref())
                        {
                            NumericKind::Unknown => Err(Fault::UnsupportedOperation),
                            NumericKind::Float => Ok(Value::dynamic(
                                MuseWidget::FloatSlider(
                                    this.0
                                        .linked(
                                            |v| v.as_f64().unwrap_or_default(),
                                            |v| Value::Float(*v),
                                        )
                                        .slider_between(
                                            linked_dynamic_value(
                                                &start,
                                                vm.as_ref(),
                                                |value| value.as_f64().unwrap_or_default(),
                                                |float| Value::Float(*float),
                                            ),
                                            linked_dynamic_value(
                                                &end,
                                                vm.as_ref(),
                                                |value| value.as_f64().unwrap_or_default(),
                                                |float| Value::Float(*float),
                                            ),
                                        ),
                                ),
                                vm,
                            )),
                            NumericKind::Int => Ok(Value::dynamic(
                                MuseWidget::IntSlider(
                                    this.0
                                        .linked(
                                            |v| v.as_i64().unwrap_or_default(),
                                            |v| Value::Int(*v),
                                        )
                                        .slider_between(
                                            linked_dynamic_value(
                                                &start,
                                                vm.as_ref(),
                                                |value| value.as_i64().unwrap_or_default(),
                                                |int| Value::Int(*int),
                                            ),
                                            linked_dynamic_value(
                                                &end,
                                                vm.as_ref(),
                                                |value| value.as_i64().unwrap_or_default(),
                                                |int| Value::Int(*int),
                                            ),
                                        ),
                                ),
                                vm,
                            )),
                        }
                    } else {
                        Err(Fault::UnknownSymbol)
                    }
                }
            })
            .with_call(|_| {
                |this, _vm, arity| {
                    if arity == 0 {
                        Ok(this.0.get())
                    } else {
                        Err(Fault::NotAFunction)
                    }
                }
            })
            .with_deep_clone(|_| {
                |this, guard| {
                    this.0
                        .map_ref(|value| value.deep_clone(guard))
                        .map(|value| muse::value::AnyDynamic::new(Self(Dynamic::new(value)), guard))
                }
            })
        });
        &TYPE
    }
}

fn numeric_kind(value: &Value, guard: &CollectionGuard) -> NumericKind {
    map_dynamic_value(value, guard, |value| match value {
        Value::Int(_) => NumericKind::Int,
        Value::Float(_) => NumericKind::Float,
        _ => NumericKind::Unknown,
    })
}

fn map_dynamic_value<R>(
    value: &Value,
    guard: &CollectionGuard,
    map: impl FnOnce(&Value) -> R,
) -> R {
    if let Some(dynamic) = value.as_downcast_ref::<DynamicValue>(guard) {
        return dynamic.0.map_ref(map);
    }
    map(value)
}

fn linked_dynamic_value<R>(
    value: &Value,
    guard: &CollectionGuard,
    mut map_to: impl FnMut(&Value) -> R + Send + 'static,
    map_from: impl FnMut(&R) -> Value + Send + 'static,
) -> CushyValue<R>
where
    R: PartialEq + Send + 'static,
{
    if let Some(dynamic) = value.as_downcast_ref::<DynamicValue>(guard) {
        return dynamic.0.linked(map_to, map_from).into_value();
    }

    CushyValue::Constant(map_to(value))
}

// fn map_each_dynamic_value<R>(
//     value: &Value,
//     mut map: impl FnMut(&Value) -> R + Send + 'static,
// ) -> CushyValue<R>
// where
//     R: PartialEq + Send + 'static,
// {
//     if let Some(dynamic) = value.as_dynamic() {
//         if let Some(dynamic) = dynamic.downcast_ref::<DynamicValue>() {
//             return dynamic.0.map_each(map).into_value();
//         }
//     }

//     CushyValue::Constant(map(value))
// }

enum NumericKind {
    Unknown,
    Float,
    Int,
}

impl BitOr for NumericKind {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (NumericKind::Unknown, _) | (_, NumericKind::Unknown) => NumericKind::Unknown,
            (NumericKind::Float, _) | (_, NumericKind::Float) => NumericKind::Float,
            (NumericKind::Int, _) => NumericKind::Int,
        }
    }
}

pub trait VmUi: Sized {
    fn with_ui(self, guard: &mut CollectionGuard<'_>) -> Self;
}

impl VmUi for Vm {
    fn with_ui(self, guard: &mut CollectionGuard<'_>) -> Self {
        install(&self, guard);
        self
    }
}

#[derive(Debug)]
pub enum MuseWidget {
    FloatSlider(Slider<f64>),
    IntSlider(Slider<i64>),
    Expand(Expand),
}

impl MakeWidget for &'_ MuseWidget {
    fn make_widget(self) -> WidgetInstance {
        match self {
            MuseWidget::FloatSlider(widget) => widget.clone().make_widget(),
            MuseWidget::IntSlider(widget) => widget.clone().make_widget(),
            MuseWidget::Expand(widget) => widget.clone().make_widget(),
        }
    }
}

impl CustomType for MuseWidget {
    fn muse_type(&self) -> &TypeRef {
        static TYPE: RustType<MuseWidget> = RustType::new("Widget", |t| {
            t.with_invoke(|_| {
                |this, vm, name, arity| {
                    if name == &Symbol::from("open") && arity == 0 {
                        let widget = this.make_widget();
                        Ok(widget
                            .open(&muse_app()?)
                            .unwrap()
                            .map(|handle| Value::dynamic(OpenWindow(handle), vm))
                            .unwrap_or_default())
                    } else if name == &Symbol::from("expand") && arity == 0 {
                        Ok(Value::dynamic(
                            MuseWidget::Expand(this.make_widget().expand()),
                            vm,
                        ))
                    } else {
                        Err(Fault::UnknownSymbol)
                    }
                }
            })
        });
        &TYPE
    }
}

impl SimpleType for MuseWidget {}

#[derive(Debug)]
pub struct OpenWindow(WindowHandle);

impl CustomType for OpenWindow {
    fn muse_type(&self) -> &TypeRef {
        static TYPE: RustType<OpenWindow> = RustType::new("Window", |t| t);
        &TYPE
    }
}

impl SimpleType for OpenWindow {}
