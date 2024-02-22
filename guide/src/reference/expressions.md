# Expressions

```musebnf
Program: <Chain>;

Chain: <Expression> (';' <Expression>)*;

Expression: <Assignment> | <ArrowFn> | <InlineIf>;
```

The compiler expects one or more expressions delimited by a ';'. These
expressions are evaluated within the root "module". When executing multiple
compiled programs in the same virtual machine, the root module will contain
declarations from previous programs.

Each expression can be [an assignment](./assignment.md), [an arrow
function](./functions.md#arrow-functions), or an [inline if](./if.md#inline-if).
The precedence of these operators matches the order they are listed in. For
example, these two ways of assigning to `my_function` are identical:

```muse
var my_function = nil;

my_function = n => n * 2 if n > 0;
my_function = (n => (n * 2 if n > 0));
```

The definition of [inline if](./if.md#inline-if) recurses into the remaining
expression types.
