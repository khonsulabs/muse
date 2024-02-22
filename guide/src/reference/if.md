# If

```musebnf
If: 'if' <Expression> <ThenExpression | Block> <ElseExpression>?;
ThenExpression: 'then' <Expression>;
ElseExpression: 'else' <Expression>;

InlineIf: <LogicalOr> ('if' <LogicalOr> ('else' <Expression>)?)?;
```
