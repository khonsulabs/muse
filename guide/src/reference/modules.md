# Modules

```musebnf
Mod: 'mod' <Identifier> '{' <Chain> '}';

Pub: 'pub' <Mod | Fn | Let | Var>;
```

Modules provide a way to encapsulate code behind a namespace. Consider this
example:

```muselang
mod math {
    pub fn square(n) => n * n;
};

math.square(4)
```

The above example creates a module named `math`, which contains a function named
`square`. The function is then invoked with the value `4`.

## Publishing declarations

The `pub` keyword can be used to publish modules, functions, values, and
variables. Without the `pub` keyword in the previous section's example,
attempting to access the `square` function will fail.

The default visibility of a declaration is module-private. Any other module,
including submodules, are not be able to access private declarations.

Published declarations are able to be accessed anywhere that the containing
module is accessible.
