# Functions

```musebnf
Fn: 'fn' <Identifier>? <FnDeclaration | FnMatch>;
FnDeclaration: ('(' (<Identifier>)? ')')? <ArrowBody | Block>;
ArrowBody: '=>' <Expression>;
BlockBody: '{' <Chain> '}';
FnMatch: '{' <MatchBody> '}';

Return: 'return' <Expression>?;
```

Functions in Muse are parameterized expressions that can be executed on-demand.
This is the primary way to reuse code and avoid [code duplication][duplicate-code].

## `fn` Functions

There are many forms that `fn` functions can be declared.

### Anonymous Functions

Anonymous functions do not have a name specified at the time of creation. They
can only be called by calling the function returned from the function
expression:

```muse
let square = fn(n) => n ** 2;
let 4 = square(2);

let area = fn(width, height) => width * height;
let 6 = area(2, 3);

# Function bodies can also be specified using a block.
let square = fn(n) {
    n ** 2
};
let 4 = square(2);

let area = fn(width, height) {
    width * height
};
let 6 = area(2, 3);
```

### Named Functions

If an identifier is directly after the `fn` keyword, the function is declared
using the identifier as its name. These examples are identical to the anonymous
function examples, except they utilize named functions instead of `let`
expressions to declare the function.

```muse
fn square(n) => n ** 2;
let 4 = square(2);

fn area(width, height) => width * height;
let 6 = area(2, 3);

# Function bodies can also be specified using a block.
fn square(n) {
    n ** 2
};
let 4 = square(2);

fn area(width, height) {
    width * height
};
let 6 = area(2, 3);
```

### Match Functions

Muse supports function overloading/multiple dispatch using pattern matching. If
an open curly brace (`{`) is found after the `fn` keyword or after the
function's name, the contents of the braces are interpretted as a set of match
patterns:

```muse
fn area {
    (width) => width * width,
    (width, height) => width * height,
};

let 4 = area(2);
let 6 = area(2, 3);

# Anonymous functions can also be match functions
let area = fn {
    (width) => width * width,
    (width, height) => width * height,
};

let 4 = area(2);
let 6 = area(2, 3);
```

## Returning values from a function

When a function executes, the result of the final expression evaluated will be
returned. The `return` expression can be used to exit a function without
executing any additional code.

```muse
fn my_function() {
    return;
    this_expression_wont_be_evaluated
};
my_function()
```

The `return` expression can also be provided an expression to return as the
result of the function. If no value is provided, `nil` is returned.

```muse
fn checked_op(numerator, denominator) {
    if denominator == 0 {
        return numerator // denominator;
    }
};
let nil = checked_op(1, 0);
let 3 = checked_op(6, 2);
```

[duplicate-code]: https://en.wikipedia.org/wiki/Duplicate_code
