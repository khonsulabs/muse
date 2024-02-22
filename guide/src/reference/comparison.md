# Comparisons

```musebnf
Comparison:
    <BitwiseOr>
    <LessThanOrEqual |
        LessThen |
        Equal |
        NotEqual |
        GreaterThan |
        GreaterThanOrEqual>*;
LessThanOrEqual: '<=' <BitwiseOr>;
LessThan: '<' <BitwiseOr>;
Equal: '=' <BitwiseOr>`;
NotEqual: '!=' <BitwiseOr>;
GreaterThan: '>' <BitwiseOr>;
GreaterThanOrEqual: '>=' <BitwiseOr>;
```

Comparing two values in Muse is done using comparison operators. The types of
comparisons Muse supports are equality and relative comparisons.

## Equality Comparison

When Muse checks if two values are equal or not equal, Muse will try to
approximately compare similar data types. For example, all of these comparisons
are true:

```muse
let true = 1 == 1.0;
let true = 1 == true;
let true = 1.0 == true;
```

`nil` is only considered equal to `nil` and will be not equal to every other
value.

```muse
let true = nil == nil;
let false = nil == false;
```

## Relative Comparison

Muse attempts to create a "total order" of all data types so that sorting a list
of values can provide predictable results even with mixed types. If a type does
not support relative comparisons, its memory address will be used as a unique
value to compare against.

## Range Comparisons

Chaining multiple comparison expressions together creates a *range comparison*.
For example, consider `0 < a < 5`. On its own, `0 < a` results in a boolean,
which traditionally would result in the second part of the expression becoming
`bool_result < 5`. In general, this isn't the programmer's desire.

Muse interprets `0 < a < 5` as `0 < a && a < 5`. This form of chaining
comparisons can mix and match all comparison operators. Consider a longer
example:

```muse
let a = 1;
let b = 2;

# These two statements are practically identical
let true = 0 < a < b < 5;
let true = 0 < a && a < b && b < 5;
```
