# Maps

```musebnf
BlockOrMap: '{' <EmptyMap | BlockBody | MapBody | SetBody> '}';
EmptyMap: ',';
BlockBody: <Chain>?;
MapBody: <Mapping> (',' <Mapping>)*;
Mapping: <Expression> ':' <Expression>;
SetBody: <Expression> (',' <Expression>)*;
```
