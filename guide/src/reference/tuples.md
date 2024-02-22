# Lists/Arrays/Tuples

```musebnf
Parentheses: '(' <ExpressionList> ')';
Brackets: '[' <ExpressionList> ']';
```

A sequence of values stored sequentially is a List in Muse. In other languages,
these structures may also be referred to as arrays or tuples. Because of Muse's
dynamic nature, lists and tuples are treated identically.

```muse
let list = [1, 2, 3];
let tuple = (1, 2, 3);

let true = list == tuple
```
