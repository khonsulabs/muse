# Boolean Logic

```musebnf
LogicalOr: <LogicalXor> ('or' <LogicalXor>)*;
LogicalXor: <LogicalAnd> ('xor' <LogicalAnd>)*;
LogicalAnd: <Comparison> ('and' <Comparison>)*;

LogicalNot: 'not' <Prefix>;
```
