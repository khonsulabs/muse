# Nil and Exception Handling

```musebnf
Punctuation: <Prefix> <Call | Index | Lookup | TryOperator | NilCoalesce>*

TryOperator: '?' <Punctuation>?;
NilCoalesce: '??' <Punctuation>;

Try: 'try' <Expression> ('catch' <MatchBlock | SingleCatch | ArrowCatch>)?;
SingleCatch: <GuardedPattern> <Block>;
ArrowCatch: '=>' <Expression>;

Throw: 'throw' <Expression>?;
```

Muse embraces exception handling for propagating errors and aims to make
handling errors straightfoward when desired.

When an exception is thrown, Muse unwinds the execution stack to the nearest
`catch` handler. If the catch handler matches the exception thrown, the
exception handler will be executed. If not, Muse will continue to unwind the
stack until a matching handler is found or execution returns to the embedding
application. In Rust, the result will be `Err(_)`.

Any value can be used as an exception, and handling specific functions is done
using pattern matching. Consider this example:

```muse
fn first(n) {
    try {
        second(n)
    } catch :bad_arg {
        second(1)
    }
};

fn second(n) {
    try {
        100 / n
    } catch :divided_by_zero {
        throw :bad_arg
    }
};

first(0)
```

When the above example is executed, second is invoked with `0`, which causes a
divide by zero exception. The exception occurs within the `try` block, and is
caught by the `catch :divide_by_zero` block.

The catch block proceeds to throw the symbol `:bad_arg` as a new exception.
`first()` is executing `second()` within a try block that catches that value,
and calls `second(1)` instead. This will succeed, and return a result of `100`.

While this is a nonsensical example, it highlights the basic way that exception
handling and pattern matching can be combined when handling errors.

## Catch any error

Catching any error can be done in one of two ways: using an identifier binding
or using an arrow catch block. In the next example, both functions are
identical:

```muse
fn identifier_binding() {
    try {
        1 / 0
    } catch err {
        err
    }
}

fn arrow_catch() {
    try {
        1 / 0
    } catch => {
        # when not provided a name, `it` is bound to the thrown exception.
        it
    }
}
```

## Matching multiple errors

To catch multiple errors raised by inside of a try block, a match block can be
used:

```muse
fn checked_area(width,height) {
    if width == 0 {
        throw :invalid_width
    } else if height == 0 {
        throw :invalid_height
    }

    width * height
};

try {
    checked_area(0, 100)
} catch {
    :invalid_width => {}
    :invalid_height => {}
}
```

## Converting any exception to `nil`

A `try` block without a `catch` block will return `nil` if an exception is
raised. Similarly, the try operator (`?`) can be used to convert any exception
raised to `nil`. These examples produce identical code:

```muse

try {
    1 / 0
};

(1 / 0)?;
```
