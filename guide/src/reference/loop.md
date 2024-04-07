# Loops

```musebnf
Loop: 'loop' <Block> ('while' <Expression>)?;
While: 'while' <Expression> <Block>;
For: 'for' <GuardedPattern> 'in' <Expression> <Block>;
```

Muse has four loop types:

- Infinite loop: Repeats the loop body unconditionally.
- While loop: Repeats the loop body while a condition is truthy.
- Loop-While loop: Executes the loop body and then repeats the loop while the
  condition is truthy.
- For loop: Evaluates the loop body for each matching element in an iterable
  value.

Each loop type currently requires that the loop body is a block. All loop's
execution can be affected by these operations:

- An uncaught exception
- `return`
- `break`: Exits the loop body, optionally returning a value.
- `continue`: Jumps to the beginning of the next loop iteration, which is just
  prior to the iterator is advanced and any conditions being checked.

# Infinite Loop

```muselang
var n = 0;
let nil = loop {
    if n % 2 == 0 {
        n = n + 1;
    } else if n > 50 {
        break;
    } else {
        n = n * 2;
    }
};
let 63 = n;
```

The above loop executes a series of operations until `n > 50` is true, at which
point the loop exits.

# While Loop

A while loop checks that a condition is truthy before executing the loop body,
and continues repeating until the condition is not truthy.

```muselang
var n = 0;
while n < 10 {
    n = n + 1;
};
let 10 = n;
```

# Loop-While Loop

A Loop-While loop executes the loop body, then checks whether the condition is
truthy. This is different from a While loop in that the condition is only
checked after the first iteration of the loop. The `continue` expression will
continue iteration just prior to the condition evaluation.

```muselang
var n = 1;
loop {
    n = n * 3;
} while n % 4 != 1;
let 9 = n;
```

# For loop

A For loop iterates a value and executes the loop body for each matching element
in the iterator. The syntax for the for loop uses a [guarded
pattern](./pattern-matching.md#guarded-pattern), which means all pattern
matching features can be used in the for loop syntax.

If the iterator returns an item that does not match the pattern, the next
element will be requested and the loop body will not be executed for that
element.

```muselang
var sum = 0;
for n in [1, 2, 3] {
    sum = sum + n
};
let 6 = sum;

for (key, value) in {"a": 1, "b": 2} {
    match key {
        "a" => let 1 = value,
        "b" => let 2 = value,
    }
};

var sum = 0;
for (_, value) if value % 2 == 1 in {"a": 1, "b": 2, "c": 3} {
    sum = sum + value;
};
let 4 = sum;
```
