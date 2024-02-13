# Variables and Assignment

```musebnf
Assignment: <Lookup | Index> ('=' <Assignment>)*;

Let: 'let' <VariablePattern>;
Var: 'var' <VariablePattern>;
VariablePattern: <GuardedPattern> '=' <Expression> <VariableElse>?;
VariableElse: 'else' <Expression>;
```
