# Functions

```musebnf
ArrowFn: <Identifier | Tuple(Identifier*)> '=>' <ArrowFn>;

Fn: 'fn' <Identifier>? <FnDeclaration | FnMatch>;
FnDeclaration: ('(' (<Identifier>)? ')')? <ArrowBody | Block>;
ArrowBody: '=>' <Expression>;
BlockBody: '{' <Chain> '}';
FnMatch: '{' <MatchBody> '}';

Return: 'return' <Expression>?;
```
