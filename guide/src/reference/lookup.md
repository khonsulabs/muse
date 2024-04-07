# Lookup/Call/Index

```musebnf
Punctuation: <Prefix> <Call | Index | Lookup | TryOperator | NilCoalesce>*

Call: '(' <ExpressionList> ')' <Punctuation>?;
Index: '[' <ExpressionList> ']' <Punctuation>?;
Lookup: '.' <Identifier> <Punctuation>?;
```

This set of operators acts on a [term](./term.md) followed by a way of
interacting with that term to either call it like a function, look up a value by
index, or lookup a member value by an identifier.

These expression are chainable. For example, this code accesses a value from a
list within a list by chaining the index operator after invoking `list.get(0)`
function:

```muselang
let list = [[1, 2, 3], [4, 5, 6]];
let 2 = list.get(0)[1];
```
