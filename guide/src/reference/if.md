# If

```musebnf
If: 'if' <Expression> <ThenExpression | Block> <ElseExpression>?;
ThenExpression: 'then' <Expression>;
ElseExpression: 'else' <Expression>;

InlineIf: <LogicalOr> ('if' <LogicalOr> ('else' <Expression>)?)?;
```

The `if` keyword allows conditionally evaluating code. There are two forms of
the `if` expression: standalone and inline.

## Standalone If

The standalone `if` expression enables executing an expression when a condition
is true:

```muselang
let 42 = if true {
    42
};
let 42 = if true then 42;
let nil = if false {
    42
};
let nil = if false then 42;
```

If the `else` keyword is the next token after the "when true" expression, an
expression can be evaluated when the condition is false.

```muselang
let 42 = if false {
    0
} else {
    42
};
let 42 = if false then 0 else 42;
```

Because `if` is an expression, the expressions can be chained to create more
complex if/else-if expressions:

```muselang
fn clamp_to_ten(n) {
    if n < 0 {
        0
    } else if n > 10 {
        10
    } else {
        n
    }
};
let 0 = clamp_to_ten(-1);
let 1 = clamp_to_ten(1);
let 10 = clamp_to_ten(11);
```

## Inline If

An inline if expression returns the *guarded expression* if the condition is
true, or `nil` when the condition is false:

```muselang
let 42 = 42 if true;
let nil = 42 if false;
```

Similar to the standalone `if` expression, `else` can be used to execute a
different expression when the condition is false:

```muselang
let 42 = 0 if false else 42;
```

Inline if statements can also be chained:

```muselang
fn clamp_to_ten(n) {
    0 if n < 0
        else 10 if n > 10
        else n
};
let 0 = clamp_to_ten(-1);
let 1 = clamp_to_ten(1);
let 10 = clamp_to_ten(11);
```
