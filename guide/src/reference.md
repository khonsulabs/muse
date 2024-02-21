# Language Reference

## Grammar Syntax

To specify the allowed grammar rules, this syntax is used:

- `RuleName: [..];`: A named grammar rule that matches the specified terms.
- `<RuleName>`: A reference to a rule named `RuleName`
- `'text'`: The raw characters `text`
- `( x )`: A grouping of grammar terms
- `x | y`: Either `x` or `y`, with `x` having precedence.
- `<x | y>`: Either rule `x` or rule `y` with equal precedence.
- `x?`: Optionally allow `x`
- `x*`: Zero or more `x` in a row.
- `x+`: One or more `x` in a row.

```musebnf
Program: <Chain>;

Chain: <Expression> (';' <Expression>)*;

Expression: <Assignment>;

Assignment: <Lookup | Index> ('=' <Assignment>)*;

ArrowFn: <Identifier | Tuple(Identifier*)> '=>' <ArrowFn>;

LogicalOr: <LogicalXor> ('or' <LogicalXor>)*;
LogicalXor: <LogicalAnd> ('xor' <LogicalAnd>)*;
LogicalAnd: <Comparison> ('and' <Comparison>)*;

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

BitwiseOr: <BitwiseXor> ('|' <BitwiseXor>)*;
BitwiseXor: <BitwiseAnd> ('^' <BitwiseAnd>)*;
BitwiseAnd: <AddSubtract> ('&' <AddSubtract>)*;

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

Punctuation: <Prefix> <Call | Index | Lookup | TryOperator | NilCoalesce>*
Call: '(' <ExpressionList> ')' <Punctuation>?;
Index: '[' <ExpressionList> ']' <Punctuation>?;
Lookup: '.' <Identifier> <Punctuation>?;
TryOperator: '?' <Punctuation>?;
NilCoalesce: '??' <Punctuation>;

ExpressionList: <Expression> (',' <Expression>)* ','?;

Prefix:
    <BlockOrMap |
        Tuple |
        List |
        LogicalNot |
        BitwiseNot |
        Negate |
        Mod |
        Pub |
        Fn |
        Let |
        Var |
        If |
        Literal |
        Loop |
        While |
        For |
        Labeled |
        Continue |
        Break |
        Return |
        Match |
        Try |
        Throw> | Term;

Literal: 'true' | 'false';

BlockOrMap: '{' <EmptyMap | BlockBody | MapBody | SetBody> '}';
EmptyMap: ',';
BlockBody: <Chain>?;
MapBody: <Mapping> (',' <Mapping>)*;
Mapping: <Expression> ':' <Expression>;
SetBody: <Expression> (',' <Expression>)*;

Parentheses: '(' <ExpressionList> ')';

Brackets: '[' <ExpressionList> ']';

LogicalNot: 'not' <Prefix>;
BitwiseNot: '!' <Prefix>;
Negate: '-' <Prefix>;

Pub: 'pub' <Mod | Fn | Let | Var>;

Mod: 'mod' <Identifier> '{' <Chain> '}';

Fn: 'fn' <Identifier>? <FnDeclaration | FnMatch>;
FnDeclaration: ('(' (<Identifier>)? ')')? <ArrowBody | Block>;
ArrowBody: '=>' <Expression>;
BlockBody: '{' <Chain> '}';
FnMatch: '{' <MatchBody> '}';

Let: 'let' <VariablePattern>;
Var: 'var' <VariablePattern>;
VariablePattern: <GuardedPattern> '=' <Expression> <VariableElse>?;
VariableElse: 'else' <Expression>;

If: 'if' <Expression> <ThenExpression | Block> <ElseExpression>?;
ThenExpression: 'then' <Expression>;
ElseExpression: 'else' <Expression>;

Loop: 'loop' <Block> ('while' <Expression>)?;
While: 'while' <Expression> <Block>;
For: 'for' <GuardedPattern> 'in' <Expression> <Block>;

Labeled: <Label> ':' <Loop | While | For | Block>;

Continue: 'continue' <Label>?;
Break: 'break' <Label>? <Expression>?;
Return: 'return' <Expression>?;

Match: 'match' <Expression> <MatchBlock>;
MatchBlock: '{' <MatchBody> '}';
MatchBody: (<MatchPattern> (',' <MatchPattern>)*)?;
MatchPattern: <GuardedPattern> '=>' <Expression>;
GuardedPattern: <Pattern> ('if' <Expression>)?;
Pattern: <IdentifierPattern | TuplePattern | ListPattern>;
IdentifierPattern: '_' | <Identifier>;
TuplePattern: '(' (<Pattern> (',' <Pattern>)*)? ')';
ListPattern: '[' (<Pattern> (',' <Pattern>)*)? ']';

Try: 'try' <Expression> ('catch' <MatchBlock | SingleCatch | ArrowCatch>)?;
SingleCatch: <GuardedPattern> <Block>;
ArrowCatch: '=>' <Expression>;

Throw: 'throw' <Expression>;

Term: <Identifier | Number | Regex | String | Symbol>;
```
