# Pattern Matching

```musebnf
Match: 'match' <Expression> '{' <MatchBody> '}';
MatchBody: (<MatchPattern> (',' <MatchPattern>)*)?;
MatchPattern: <GuardedPattern> '=>' <Expression>;
GuardedPattern: <Pattern> ('if' <Expression>)?;
Pattern: <IdentifierPattern | TuplePattern | ListPattern>;
IdentifierPattern: '_' | <Identifier>;
TuplePattern: '(' (<Pattern> (',' <Pattern>)*)? ')';
ListPattern: '[' (<Pattern> (',' <Pattern>)*)? ']';
```
