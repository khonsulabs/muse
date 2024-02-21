# Nil and Exception Handling

```musebnf
TryOperator: '?' <Punctuation>?;
NilCoalesce: '??' <Punctuation>;

Try: 'try' <Expression> ('catch' <MatchBlock | SingleCatch | ArrowCatch>)?;
SingleCatch: <GuardedPattern> <Block>;
ArrowCatch: '=>' <Expression>;

Throw: 'throw' <Expression>;
```
