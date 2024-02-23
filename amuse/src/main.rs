use cushy::kludgine::app::winit::keyboard::{Key, NamedKey};
use cushy::styles::components::{ErrorColor, FontFamily, TextColor};
use cushy::styles::FamilyOwned;
use cushy::value::{Destination, Dynamic, Source};
use cushy::widget::{MakeWidget, WidgetInstance, WidgetList, HANDLED, IGNORED};
use cushy::widgets::input::InputValue;
use cushy::widgets::list::ListStyle;
use cushy::widgets::Label;
use cushy::{Open, PendingApp};
use muse::compiler::Compiler;
use muse::syntax::{SourceCode, SourceId, Sources};
use muse::value::Value as MuseValue;
use muse::vm::{Fault, Vm};
use muse_ui::VmUi;

fn main() {
    let runtime = Dynamic::new(Runtime {
        vm: Vm::default().with_ui(),
        compiler: Compiler::default(),
        sources: Sources::default(),
    });
    let history = Dynamic::<Vec<History>>::default();
    let history_list = history.map_each({
        let runtime = runtime.clone();
        move |history| {
            history
                .iter()
                .map(|i| i.make_widget(&runtime))
                .collect::<WidgetList>()
        }
    });

    let input = Dynamic::<String>::default();
    let parsed = input.map_each(|source| {
        muse::syntax::parse(&SourceCode::anonymous(source)).map_err(|err| format!("{err:?}"))
    });
    let expression = parsed.map_each(|parsed| {
        parsed
            .as_ref()
            .map(|expr| format!("{expr:?}"))
            .unwrap_or_default()
    });
    let app = PendingApp::default();
    muse_ui::initialize(&app);

    history_list
        .into_list()
        .style(ListStyle::Decimal)
        .vertical_scroll()
        .expand()
        // TODO no way to detect shift key in on_key
        .and(
            input
                .clone()
                .into_input()
                .on_key(move |key| {
                    if matches!(key.logical_key, Key::Named(NamedKey::Enter)) {
                        if key.state.is_pressed() {
                            execute_input(input.take(), &runtime, &history);
                        }
                        HANDLED
                    } else {
                        IGNORED
                    }
                })
                .with(&FontFamily, FamilyOwned::Monospace)
                .validation(parsed)
                .hint(expression),
        )
        .into_rows()
        .expand()
        .run_in(app)
        .expect("error launching app");
}

struct History {
    source: SourceId,
    result: Result<MuseValue, Fault>,
}

impl History {
    fn make_widget(&self, runtime: &Dynamic<Runtime>) -> WidgetInstance {
        let mut runtime = runtime.lock();
        Label::new(
            runtime
                .sources
                .get(self.source)
                .expect("missing source")
                .to_string(),
        )
        .with(&FontFamily, FamilyOwned::Monospace)
        .align_left()
        .and(
            match &self.result {
                Ok(value) => match value.to_string(&mut runtime.vm) {
                    Ok(formatted) => formatted.to_string(),
                    Err(_) => format!("{:?}", value),
                }
                .with(&FontFamily, FamilyOwned::Monospace)
                .make_widget(),
                Err(err) => format!("{err:?}")
                    .with_dynamic(&TextColor, ErrorColor)
                    .make_widget(),
            }
            .align_left()
            .contain(),
        )
        .into_rows()
        .make_widget()
    }
}

struct Runtime {
    vm: Vm,
    compiler: Compiler,
    sources: Sources,
}

fn execute_input(source: String, runtime: &Dynamic<Runtime>, history: &Dynamic<Vec<History>>) {
    let mut runtime = runtime.lock();
    let runtime = &mut *runtime;
    let source = runtime.sources.push(source);
    runtime.compiler.push(&source);
    match runtime.compiler.build() {
        Ok(code) => {
            let entry = History {
                source: source.id,
                result: runtime.vm.execute(&code),
            };
            history.lock().push(entry);
        }
        Err(errors) => {
            eprintln!("Errors: {errors:?}");
        }
    }
}
