# muse

A thoughtful, designed-to-be-embedded programming language.

## What is Muse?

> *Muse is a work-in-progress.* This section describes what Muse aims to be, but
> it almost certainly does not fulfill its goals.

Muse is a dynamic language designed to allow Rust developers a lightweight way
to **run untrusted scripts**. It is designed to be familiar to users of other
dynamic programming languages, while also feeling inspired by Rust.

Muse aims to be flexible enough to be adopted as a general purpose scripting
langauge, while remaining minimal in its base implementation to ensure embedders
have ultimate control over what is available when executing code.

## Example: Fibonacci

This is a recursive Fibonnaci function. This algorithm should never be used in
production, as iterative approaches are expenentially faster, but it provides a
few opportunities to demonstrate Muse's features.

```muse
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

```muse
fn fib(n) {
  match n {
    n if n <= 2 => 1,
    n => fib(n - 1) + fib(n - 2),
  }
}
```

Muse also supports supports *function overloading* using the same pattern match
syntax. Here's the same function using function overloading:

```muse
fn fib {
  n if n <= 2 => 1,
  n => fib(n - 1) + fib(n - 2),
}
```

All of these examples create a function named `fib` that accepts one parameter
and produce identical results.

## Syntax Overview

*A user's guide will be written to supplement this summary.*

Muse is an expression-based language. This means that every valid piece of code
returns a value. `1 + 2` returns `3`, and `fn (n) => n * 2` returns a function
that multiplies an argument by 2.

To chain multiple expressions together, a `;` is used. When multiple chained
statements are executed, the final expression's result is returned.

The types of expressions Muse supports are listed below in precedence order.

* Program: `<Chain>`
* Chain: `<Expression> (';' <Expression>)*`
* Expression: `<Assignment>`
* Assignment: `<Lookup | Index> ('=' <Assignment>)*`
* ArrowFn: `<Identifier | Tuple(Identifier*)> '=>' <ArrowFn>`
* LogicalOr: `<LogicalXor> ('or' <LogicalXor>)*`
* LogicalXor: `<LogicalAnd> ('xor' <LogicalAnd>)*`
* LogicalAnd: `<Comparison> ('and' <Comparison>)*`
* Comparison: `<BitwiseOr> (('<=' | '<' | '=' | '!=' | '>' | '>=') <BitwiseOr>)*`
  * Less than or equal: `<BitwiseOr> '<=' <BitwiseOr>`
  * Less than: `<BitwiseOr> '<' <BitwiseOr>`
  * Equal: `<BitwiseOr> '=' <BitwiseOr>`
  * Not Equal: `<BitwiseOr> '!=' <BitwiseOr>`
  * Greater than: `<BitwiseOr> '>' <BitwiseOr>`
  * Greater than or equal: `<BitwiseOr> '>=' <BitwiseOr>`
* BitwiseOr: `<BitwiseXor> ('|' <BitwiseXor>)*`
* BitwiseXor: `<BitwiseAnd> ('^' <BitwiseAnd>)*`
* BitwiseAnd: `<AddSubtract> ('&' <AddSubtract>)*`
* AddSubtract: `<MultiplyDivide> ('+' | '-') <MultiplyDivide>`
  * Addition: `<MultiplyDivide> '+' <AddSubtract>`
  * Subtraction: `<MultiplyDivide> '-' <AddSubtract>`
* MultiplyDivide: `<Punctuation> (('*' | '/' | '%' | '//') <Punctuation>)*`
  * Multiply: `<Punctuation> ('*' <Punctuation>)*`
  * Divide: `<Punctuation> ('/' <Punctuation>)*`
  * Remainder: `<Punctuation> ('%' <Punctuation>)*`
  * IntegerDivide: `<Punctuation> ('//' <Punctuation>)*`
* Punctuation
  * Call: `<Prefix> '(' <Expression>* ','? ')' <Punctuation>?`
  * Lookup: `<Prefix> '.' <Identifier> <Punctuation>?`
  * TryOperator: `<Prefix> '?' <Punctuation>?`
  * NilCoalesce: `<Prefix> '??' <Punctuation>`
* Prefix
  * Block: `'{' <Chain>? '}'`
  * EmptyMap: `'{' ',' '}'`
  * Map: `'{' <Mapping> (',' <Mapping>)* '}'`
    * Mapping: `<Expression> ':' <Expression>`
  * Parentheses: `'(' <Expression> (',' <Expression>)* ','? ')'`
  * Brackets: `'[' <Expression> (',' <Expression>)* ','? ']'`
  * LogicalNot: `'not' <Prefix>`
  * BitwiseNot: `'!' <Prefix>`
  * Negate: `'-' <Prefix>`
  * Mod: `'mod' <Identifier> '{' <Chain> '}'`
  * Pub: `'pub' <Mod | Fn | Let | Var>`
  * Fn: `'fn' <Identifier>? <FnDeclaration | FnMatch>`
    * FnDeclaration: `('(' (<Identifier>)? ')')? <ArrowBody | Block>`
      * ArrowBody: `'=>' <Expression>`
      * BlockBody: `'{' <Chain> '}'`
    * FnMatch: `'{' <MatchBody> '}'`
  * Let: `'let' <Pattern> '=' <Expression>`
  * Var: `'var' <Pattern> '=' <Expression>`
  * If: `'if' <Expression> <ThenExpression | Block> ('else' <Expression>)`
  * True: `'true'`
  * False: `'false'`
  * Loop: `'loop' <Block> ('while' <Expression>)?`
  * While: `'while' <Expression> <Block>`
  * Labeled: `<Label> ':' <Loop | While | For | Block>`
  * Continue: `'continue' <Label>?`
  * Break: `'break' <Label>? <Expression>?`
  * Return: `'return' <Expression>?`
  * For: `'for' <GuardedPattern> 'in' <Expression> <Block>`
  * Match: `'match' <Expression> '{' <MatchBody> '}'`
    * MatchBody: `(<MatchPattern> (',' <MatchPattern>)*)?`
    * MatchPattern: `<GuardedPattern> '=>' <Expression>`
    * GuardedPattern: `<Pattern> ('if' <Expression>)?`
    * Pattern: `<IdentifierPattern | TuplePattern | ListPattern>`
      * IdentifierPattern: `('_' | <Identifier>)`
      * TuplePattern: `'(' (<Pattern> (',' <Pattern>)*)? ')'`
      * ListPattern: `'[' (<Pattern> (',' <Pattern>)*)? ']'`
* Term: `<Identifier | Number | Regex | String | Symbol>`

Several rules such as `<Comparison>` specify a grammar at the group level and at
the operator level. When the group grammar is present, the operator-specific
grammar is provided for visualization purposes only.

### Lexical Analysis (Tokenization)

Source code is interpretted as a series of tokens. Whitespace is insignificant
to Muse, and any amount of whitespace is allowed between tokens. Comment tokens
are also ignored by the Muse compiler.

The types of tokens Muse supports are:

* Identifier: Any valid [Unicode
  Identifier](https://www.unicode.org/reports/tr31/) is supported by Muse.
  Underscores are allowed anywhere in an identifier, and an identifier can be
  comprised solely of underscores. Ascii digits can be used anywhere except at
  the beginning of an identifier.
* Number: <Integer | Float | Bits>.
  * Integer: `\\d+u?/`
  * Float: `\(\d+\.\d*|\.\d+)/`
  * Bits: `<Hex | Octal | Binary | Radix>`
    * Hex: `\0u?x[\da-f]+/i`
    * Octal: `\0u?o[0-5]+/`
    * Binary: `\0u?b[01]+/`
    * Radix: `\\d+u?r[0-9a-z]+/i`
* Regex: Regex supports two formats of Regex literals, both which use `\` to
  begin the Regex. The end of the Regex is `/`. E.g., `\hello/` is a Regex that
  matches `hello`.

  The only character that needs to be escaped beyond what Regex normally
  requires is `/`. E.g., `\hello\/world/` will match `hello/world`.

  * Expanded: `w\` followed by an escaped Regex where [whitespace is ignored][regex-whitespace], ended with `/`
  * Normal: `\` followed by an escaped Regex, ended with `/`

  These characters can be added after the trailing `/` to customize the Regex
  options:

  * `i`: Ignore case
  * `u`: Enables Unicode mode for the entire pattern
  * `s`: `.` matches all, including newlines
  * `m`: Enables multiline matching

  Muse uses the [regex][regex] crate, which provides excellent documentation
  about what features it supports.
* String: Begins and ends with `"`. Escapes begin with `\` and match Rust's
  supported escape sequences:
  * `\"`: Quotation mark
  * `\n`: ASCII newline
  * `\r`: ASCII carriage return
  * `\t`: ASCII tab
  * `\\`: Backslash
  * `\0`: ASCII null
  * `\x`: ASCII escape. Must be followed by exactly two hexadecimal digits.
  * `\u`: Unicode escape. Must be followed by up to six hexadecimal digits
    enclodes in curly braces. E.g, `\u{1F980}` is the escape sequence for `🦀`.
* Symbol: `:` followed by a valid `<Identifier>`.
* Comment: `#` followed by any character until the next line.

## Why use Muse over WASM?

Overall, WASM is a very tempting option for safely executing untrusted code
within another application. Embedding WASM allows users to run arbitrary code in
isolated environments incredibly efficiently and safely. Many runtimes also
support budgeted execution to allow developers to defend against inefficient
code or infinite loops. Nowadays, [Extism](https://extism.org/) makes it
incredibly easy.

The downsides might be:

* Large number of dependencies. Let's face it, build times can sometimes be an
  issue. Muse has limited dependencies, and due to being a very focused
  implementation, it means Muse will always be less code than embdding a WASM
  runtime.
* Serialization Overhead. With Muse, your Rust types can be used directly within
  the scripts with no serialization overhead. With WASM, because memory isn't
  guaranteed to be compatible, serialization is often used when passing data
  between native code and WASM.
* Split documentation. With every language that your plugin system supports, you
  might feel the need to provide documentation for those languages. By focusing
  on a single embedded language, the end-user documentation may be less work to
  maintain.

Ultimately if the allure of having user's be able to write code in the language
they are most familiar with, WASM is a great option.

[regex]: https://docs.rs/regex/
[regex-whitespace]: https://docs.rs/regex/latest/regex/struct.RegexBuilder.html#method.ignore_whitespace
