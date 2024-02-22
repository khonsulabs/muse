# Boolean Logic

```musebnf
LogicalOr: <LogicalXor> ('or' <LogicalXor>)*;
LogicalXor: <LogicalAnd> ('xor' <LogicalAnd>)*;
LogicalAnd: <Comparison> ('and' <Comparison>)*;

LogicalNot: 'not' <Prefix>;
```

Boolean logic allows combining boolean values by performing logic operations.
The logic operations that Muse supports are `and`, `or`, `xor`, and `not`.

## Truthiness

Muse evaluates each operand's *truthiness* to perform boolean logic. If a value
is said to be *truthy*, it is considered equivalent to `true` in boolean logic.
If a value is said to be *falsy*, it is considered equivalent to `false` in
boolean logic. Each type is responsible for implementing its truthiness
conditions:

- `nil`: Always falsey
- Numbers: Non-zero values are truthy, zero is falsy.
- Strings: Non-empty strings are truthy, empty strings are falsey.
- Lists/Tuples/Maps: Non-empty collections are truthy, empty collections are
  falsey.
- Symbol: Always truthy
- Other types that don't implement `truthy`: Always truthy.

## Logical Or

The logical or expression is a short-circuiting operator that returns true if
either of its operands are truthy.

```muse
let true = true or true;
let true = true or false;
let true = false or true;
let false = false or false;
```

The short-circuiting behavior ensures that once the expression is known to
return `true`, no remaining chained expressions will be evaluated:

```muse
# "error" is not evaluated
let true = true or error;
```

## Logical Xor

The logical exclusive or (xor) expression is an operator that returns true if
one of its operands is truthy, but not both.

```muse
let true = true xor false;
let true = false xor true;
let false = true xor true;
let false = false xor false;
```

This operator can not short-circuit, so both expressions are always evaluated.

## Logical And

The logical and expression is a short-circuiting operator that returns true both
of its operands are truthy.

```muse
let true = true or true;
let true = true or false;
let true = false or true;
let false = false or false;
```

The short-circuiting behavior ensures that once the expression is known to
return `false`, no remaining chained expressions will be evaluated:

```muse
# "error" is not evaluated
let false = false and error;
```
