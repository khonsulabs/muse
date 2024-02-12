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
