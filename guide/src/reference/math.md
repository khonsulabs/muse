# Math/Arithmetics

```musebnf
AddSubtract: <MultiplyDivide> <Addition | Subtraction>*;
Addition: '+' <MultiplyDivide>;
Subtraction: '-' <MultiplyDivide>;

MultiplyDivide:
    <Power>
    <Multiply | Divide | Remainder | IntegerDivide>*;
Multiply: '*' <Power>;
Divide: '/' <Power>;
Remainder: '%' <Power>;
IntegerDivide: '//' <Power>;

Power: <Punctuation> ('**' <Punctuation>)*;
```
