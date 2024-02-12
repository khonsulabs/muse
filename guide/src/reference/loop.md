# Loops

```musebnf
Loop: 'loop' <Block> ('while' <Expression>)?;
While: 'while' <Expression> <Block>;
For: 'for' <GuardedPattern> 'in' <Expression> <Block>;
```
