# Expressions

```musebnf
Program: <Chain>;

Chain: <Expression> (';' <Expression>)*;

Expression: <Assignment> | <ArrowFn> | <InlineIf>;
```
