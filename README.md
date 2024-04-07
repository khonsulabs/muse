# muse

A thoughtful, designed-to-be-embedded programming language.

## What is Muse?

> *Muse is a work-in-progress.* This section describes what Muse aims to be, but
> it almost certainly does not fulfill its goals.

Muse is a dynamic language designed to allow Rust developers a lightweight way
to **run untrusted scripts**. It is designed to be familiar to users of other
dynamic programming languages, while also feeling inspired by Rust.

Muse aims to be flexible enough to be adopted as a general purpose scripting
language, while remaining minimal in its base implementation to ensure embedders
have ultimate control over what is available when executing code.

Muse uses Rust's standard library, which means while Muse is designed to be
embedded in Rust applications, it is not designed to be used in `no_std` Rust
applications.

For more of what inspires Muse's design, see the [design goals
chapter][design-goals] of Muse's User's Guide.

## Example: Fibonacci

This is a recursive Fibonacci function. This algorithm should never be used in
production, as iterative approaches are exponentially faster, but it provides a
few opportunities to demonstrate Muse's features.

```muselang
fn fib(n) {
  if n <= 2 {
    1
  } else {
    fib(n - 1) + fib(n - 2)
  }
};

fib(10)
```

At first glance, this may look like a generic dynamically typed language. This
is by design: Muse code should be easy to read, regardless of what programming
language you're most familiar with. As a language designed for embedding, this
is an important design goal.

Muse also supports *pattern matching*. Here is the same example written using
pattern matching instead of if/else:

```muselang
fn fib(n) {
  match n {
    n if n <= 2 => 1,
    n => fib(n - 1) + fib(n - 2),
  }
}
```

Muse also supports *function overloading* using the same pattern match syntax.
Here's the same function using function overloading:

```muselang
fn fib {
  n if n <= 2 => 1,
  n => fib(n - 1) + fib(n - 2),
}
```

All of these examples create a function named `fib` that accepts one parameter
and produce identical results.

A full [language reference][language-ref] is available in the [Muse User's Guide][guide] is
available in the User's Guide.

[design-goals]: https://khonsu.dev/muse/main/guide/design.html
[guide]: https://khonsu.dev/muse/main/guide/
[language-ref]: https://khonsu.dev/muse/main/guide/reference.html
