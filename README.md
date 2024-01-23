# muse

A thoughtful programming language.

## Muse is

**None of these things yet.**

- Gradually Typed.
  - Strong runtime typing.
  - Dynamic typing by default
  - In the future, ability to annotate types and have compile time checking and
    potentially more optimizations.
- Virtual Machine-powered
  - Each VM has its own Stack and isolated environment.
  - VMs can be metered to safely execute untrusted code.
  - As long as all Rust helpers have proper error handling, a virtual machine
    should never panic.

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

# Closures/lambdas
{|args| expr}

# functions
fn name(args) expr_or_block;

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


# Pattern Matching

expr -> {
  foo => {

  };
};

```
