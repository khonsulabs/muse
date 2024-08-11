# Pattern Matching

```musebnf
Match: 'match' <Expression> '{' <MatchBody> '}';
MatchBody: (<MatchPattern> (',' <MatchPattern>)*)?;
MatchPattern: <GuardedPattern> '=>' <Expression>;
GuardedPattern: <Pattern> ('if' <Expression>)?;
Pattern: <PatternKind> ('|' <PatternKind>)*;
PatternKind: <IdentifierPattern | ListPattern | MapPattern | ExpressionPattern>;
IdentifierPattern: '_' | '...' | <Identifier>;
ListPattern: '[' (<Pattern> (',' <Pattern>)*)? ']';
ExpressionPattern: ('<=' | '>=' | '<' | '>' | '=' | '!=') <Expression>;

MapPattern: '{' (<EntryPattern> (',' <EntryPattern>)*)? ','? '}';
EntryPattern: <EntryKeyPattern> ':' <Pattern>;
EntryKeyPattern: <Identifier | Number | String | Symbol>;
```

## `match` expression

The match expression enables concise and powerful ways to inspect and instract
data from the result of an expression.

## Match block

A match block is one or more *guarded patterns* and their associated
expressions. Any identifiers bound in the match pattern will only be accessible
while executing that pattern's guard and associated expression.

When evaluating a match block, the compiler generates code that tries to match
each pattern in sequence. If no match is found, a pattern mismatch exception is
thrown.

## Guarded Pattern

A guarded pattern is a pattern followed by an optional `if` condition. If the
pattern matches, the `if` condition is evaluated. If the result is truthy, the
pattern is considered a match.

## Pattern

A pattern can one one of these kinds:

- Wildcard: `_` will match any value.
- Named wildcard: Any identifier
- Remaining Wildcard: `...` will match all remaining elements in a collection/
- An expression comparison: A comparison operator followed by an expression.
- Tuple pattern: A comma separated list of patterns enclosed in parentheses.
- List pattern: A comma separated list of patterns enclosed in square brackets.

Multiple matching patterns can be used by chaining patterns together with the
vertical bar (`|`). This example uses a match function to try to demonstrate a
lot of the flexibility this feature provides:

```muselang
fn test_match {
    [a, b] if b != 1 => a - b,
    [a, b] => a + b,
    n => n,
    [] => 42,
    _ => "wildcard",
};

let 42 = test_match(44, 2);
let 42 = test_match(41, 1);
let 42 = test_match(42);
let 42 = test_match();
let "wildcard" = test_match(1, 2, 3);
```
