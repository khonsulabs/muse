# muse

A thoughtful, embeddable programming language.

## Muse is

- Dynamically typed
  - Each value is strongly typed, but the types of each value are only known at
    runtime.
  - Gradual typing is an eventual goal to allow compile-time type checking and
    potentially optimizations.
- Safe for untrusted code execution.
  - No unsafe code.
  - All panics are considered bugs.
  - Each Virtual Machine (VM) is fully isolated.
  - Able to protect against infinite loops.
    - VMs can be given a budget to interrupt tasks that exceed a
    limit of operations performed.
    - Interrupted VMs can be given additional budget and resumed.
- Straightfoward to embed in Rust
  - Create Muse types and functions in Rust
  - Async-friendly
    - `AsyncFunction` allows native `Future` implementations to be invoked from
      Muse.
    - `Vm::execute_async()` returns a `Future` that executes the given code.
    - `Vm::execute()` will block the current thread, parking and waking using a
      custom `Waker`.
    - `Vm::block_on()` uses the virtual machine's own `Future` executor to execute
      a future to completion, blocking the current thread until it is complete.

## Syntax Ideas

```muse
# Single line comments

# Declare a single-assign variable
let a = 1;

# Declare a variable that can be assigned to more than once
var a = 1;
a = 2;

# addition, subtraction
a + 1 - b

# Multiplication, Division, Integer Division, Rem
a * b / c \ d % f

# Pow
a ** b

# Bit and, or, xor
a & b | c ^ d

# Logical and, or, xor (short circuits when possible)
a and b or c xor d

# Blocks
{
  a = b;
  a = c;
}

# Functions, Closures, and Lambdas

# All functions are a mapping of zero or more arguments to an expression.
fn answer() => 42;
fn answer => 42;
let answer = () => 42;
() => 42;

# More complex functions can use a block expression.
fn answer() => { 42 };
fn answer => { 42 };
let answer = () => { 42 };
() => { 42 }

# In both forms, if a function takes a single (untyped?) parameter, the parentheses can be omitted
fn square n => n ** 2;
let square = n => n ** 2;
n => n ** 2



# Lists
let list = [1, 2, 3];

# Set (Hash)
let set = {1, 2, 3};

# Set (Ordered, uses a function named ordered )
let set = ordered{};

# Map
let map = {k: v};

# Invoke
ident(comma list of args)
ident{ map or set } # shorthand for ident({..})
ident[list] # shorthand for ident([..])

# type conversions
Int("1")

# Pattern Matching

match expr with matchexpr;
expr matches


```
