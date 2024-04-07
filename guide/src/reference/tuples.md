# Lists/Arrays/Tuples

```musebnf
Brackets: '[' <ExpressionList> ']';
```

A sequence of values stored sequentially is a List in Muse. In other languages,
these structures may also be referred to as arrays or tuples.

```muselang
let list = [1, 2, 3];
$assert(list[0] == 1);
$assert(list[1] == 2);
$assert(list[2] == 3);
```
