use muse_lang::{
    refuse::CollectionGuard,
    runtime::value::{Primitive, RootedValue, RustFunction, Value},
    vm::{Register, Vm},
};

use crate::{bounded, TrySendError, WithNewChannel};

#[test]
fn basic() {
    let (main_sender, main_receiver) = crate::bounded(1);
    let (thread_sender, thread_receiver) = crate::bounded(1);

    std::thread::spawn(move || match main_receiver.recv() {
        Some(RootedValue::Primitive(Primitive::Int(i))) => {
            // Ensure we disconnect before sending the result to the main thread.
            drop(main_receiver);
            thread_sender
                .send(RootedValue::Primitive(Primitive::Int(i + 1)))
                .unwrap();
        }
        other => unreachable!("unexpected recv: {other:?}"),
    });

    main_sender
        .send(RootedValue::Primitive(Primitive::Int(41)))
        .unwrap();
    match thread_receiver.recv() {
        Some(RootedValue::Primitive(Primitive::Int(42))) => {}
        other => unreachable!("unexpected recv: {other:?}"),
    }

    // Ensure we're disconnected
    main_sender
        .send(RootedValue::Primitive(Primitive::Int(41)))
        .unwrap_err();
}

#[test]
fn capacity() {
    let (in_sender, in_receiver) = bounded(1);
    let (out_sender, out_receiver) = bounded(1);

    std::thread::spawn(move || {
        while let Some(RootedValue::Primitive(Primitive::Int(value))) = in_receiver.recv() {
            out_sender
                .send(RootedValue::Primitive(Primitive::Int(value + 1)))
                .unwrap();
        }
    });
    in_sender
        .send(RootedValue::Primitive(Primitive::Int(0)))
        .unwrap();
    in_sender
        .send(RootedValue::Primitive(Primitive::Int(1)))
        .unwrap();
    match in_sender.try_send(RootedValue::Primitive(Primitive::Int(2))) {
        Err(TrySendError::Full(_)) => {}
        other => unreachable!("unexpected try_send result: {other:?}"),
    }
    assert_eq!(
        out_receiver.recv(),
        Some(RootedValue::Primitive(Primitive::Int(1)))
    );
    in_sender
        .send(RootedValue::Primitive(Primitive::Int(2)))
        .unwrap();
    assert_eq!(
        out_receiver.recv(),
        Some(RootedValue::Primitive(Primitive::Int(2)))
    );
    assert_eq!(
        out_receiver.recv(),
        Some(RootedValue::Primitive(Primitive::Int(3)))
    );
    for i in 3..100 {
        in_sender
            .send(RootedValue::Primitive(Primitive::Int(i)))
            .unwrap();
        assert_eq!(
            out_receiver.recv(),
            Some(RootedValue::Primitive(Primitive::Int(i + 1)))
        );
    }
}

#[test]
fn muse() {
    let mut guard = CollectionGuard::acquire();
    let vm = Vm::new(&guard).with_new_channel(&mut guard);
    vm.declare(
        "spawn_task",
        Value::dynamic(
            RustFunction::new(|vm, arity| {
                assert_eq!(arity.0, 1);
                let sender = vm[Register(0)].take();
                std::thread::spawn(move || {
                    let mut guard = CollectionGuard::acquire();
                    let vm = Vm::new(&guard).with_new_channel(&mut guard);
                    vm.declare("sender", sender, &mut guard).unwrap();
                    vm.compile_and_execute(
                        r"
                            let [my_sender, receiver] = new_channel(1);
                            sender.send(my_sender);
                            for value in receiver {
                                sender.send(value + 1);
                            }
                        ",
                        &mut guard,
                    )
                    .unwrap();
                });
                Ok(Value::NIL)
            }),
            &guard,
        ),
        &mut guard,
    )
    .unwrap();
    vm.compile_and_execute(
        r"
            let [sender, receiver] = new_channel(1);
            spawn_task(sender);
            let sender = receiver.next();
            # The first message will cause our inbox to fill up
            sender.send(1);
            # The second message will cause the task to wake up and wait to send
            sender.send(2);
            # The third message will fill the channel
            sender.send(3);
            # The fourth message will result in a full message.
            let [:full, 4] = sender.try_send(4);
            let 2 = receiver.next();
            let 3 = receiver.next();
            let 4 = receiver.next();
        ",
        &mut guard,
    )
    .unwrap();
}
