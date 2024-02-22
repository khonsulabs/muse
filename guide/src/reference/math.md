# Math/Arithmetics

```musebnf
AddSubtract: <MultiplyDivide> <Addition | Subtraction>*;
Addition: '+' <MultiplyDivide>;
Subtraction: '-' <MultiplyDivide>;

MultiplyDivide:
    <Power>
    <Multiply | Divide | Remainder | IntegerDivide>*;
Multiply: '*' <Power>;
Divide: '/' <Power>;
Remainder: '%' <Power>;
IntegerDivide: '//' <Power>;

Power: <Punctuation> ('**' <Punctuation>)*;

Negate: '-' <Prefix>;
```

## Arithmetic Order of Operations

Arithmetic expressions in Muse honor the traditional [order of operations][ooo].
For example, consider these identical pairs of expressions written without and
with parentheses:

```muse
2 * 4 + 2 * 3;
(2 * 4) + (2 * 3);

2 ** 3 * 2 + 1;
((2 ** 3) * 2) + 1;
```

## Floating Point and Integer Arithmetic

With the exception of the division (`/`), integer divide (`//`) and remainder
(`%`) operators, all math operators will follow these rules for type
conversions:

- If any operand is a `Dynamic`, return the result of invoking the associated
  function for the operator on that type.
- If all operands are of the same numeric type, return the result as the same
  numeric type.
- If any operand is a floating point, return the result as a floating point.
- Perform the result as integers, using saturating operations, preserving the
  integer type of the first operand.

The division operator (`/`) always returns a floating point result when both
operands are numbers. Conversely, the remainder (`%`) and integer divide (`//`)
operators always return integer results when both operands are numbers.

[ooo]: https://en.wikipedia.org/wiki/Order_of_operations
