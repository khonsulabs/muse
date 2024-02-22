# Values, Variables, and Assignment

```musebnf
Assignment: <Lookup | Index> ('=' <Assignment>)*;

Let: 'let' <VariablePattern>;
Var: 'var' <VariablePattern>;
VariablePattern: <GuardedPattern> '=' <Expression> <VariableElse>?;
VariableElse: 'else' <Expression>;
```

Named values can be declared using either the `let` or `var` keywords followed
by a pattern binding.

When a `let` expression is evaluated, all bound identifiers will not be able to
be reassigned to. In Muse, these are simply called *values*. While values cannot
be reassigned to, they can be shadowed by another declaration.

```muse
# Basic value declaration
let a = 42;
# Values declared using destructuring
let (a, b, _) = [0, 1, 2];
# Handling pattern mismatches
let (a, b) = [0, 1, 2] else return;

# The contents of a value can still be affected (mutated), if the type supports it.
let a = [];
a.push(42);
```

If the expression does not match the pattern, either the `else` expression will
be evaluated or a pattern mismatch exception will be thrown. The `else`
expression requires that all code paths escape the current block. This can be
done using the `break`, `continue`, `return`, or `throw` expressions.

When the `var` keyword is used, all bound identifiers become *variables*. Unlike
values, variables can be assigned new values.

```muse
# Basic declaration
var a = 42;
# `var`s can have their values updated through assignment.
a = 42;

# Declarations using destructuring
var (a, b, _) = [0, 1, 2];

# Both a and b can have their values updated
a = 42;
b = 43;
```
