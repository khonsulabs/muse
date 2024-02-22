# Break, Continue, and Labels

```musebnf
Labeled: <Label> ':' <Loop | While | For | Block>;
Continue: 'continue' <Label>?;
Break: 'break' <Label>? <Expression>?;
```

Muse does not have a "go to" capability, but it offers several expressions that
can jump to well-defined execution points: `continue` and `break`.

The well-defined execution points are:

- A loop's next iteration
- The first instruction after a loop body
- A labeled block or loop

## Labels

A label is the `@` character followed by an identifier, e.g., `@label`. Labels
can be applied to blocks or loops to allow controlled execution flow in nested
code.

```muse
var total = 0;
@outer: for x in [1, 2, 3] {
    for y in [1, 2, 3] {
        if x == y {
            continue @outer;
        };

        # Only executed for [(2, 1), (3,1), (3, 2)]
        total = total + x * y
    };
};
let 11 = total;
```

## Continue

The `continue` expression takes an optional label. If no label is provided,
execution jumps to the location to begin the next iteration of the loop
containing the code. If no loop is found, a compilation error will occur.

If a label is provided, the label must belong to a loop. If not, a compilation
error will occur.

## Break

The `break` expression takes an optional label and an optional expression. If no
label is provided, execution jumps to the location after the current loop exits.
If no loop is found, a compilation error will occur.

If a label is provided, execution will jump to the the next instruction after
the block or loop identified by that label. The result of the block or loop will
be the optional expression or `nil`.
