# Maps and Sets

```musebnf
BlockOrMap: '{' <EmptyMap | BlockBody | MapBody | SetBody> '}';
EmptyMap: ',';
BlockBody: <Chain>?;
MapBody: <Mapping> (',' <Mapping>)*;
Mapping: <Expression> ':' <Expression>;
SetBody: <Expression> (',' <Expression>)*;
```

Maps are collections that store key-value pairs. The underlying way maps store
their key-value pairs is considered an implementation detail and not to be
relied upon.

Map literals are created using curly braces (`{` / `}`). An empty map literal is
created by placing a comma (`,`) between the braces: `{,}`. Muse considers `{}`
to be an empty block, which results in `nil`.

Pairs are specified by placing a colon (`:`) between two expressions. For
example, consider this map literal and usage:

```muselang
let map = {
    "a": 1,
    "b": 2,
    "c": 3,
};
let 2 = map["b"];
```

Sets in Muse are implemented under the hood using the Map type. Set literals can
be specified by using a comma separated list of expressions in curly braces:

```muselang
let set = {
    1,
    2,
    3
};
set.contains(1)
```
