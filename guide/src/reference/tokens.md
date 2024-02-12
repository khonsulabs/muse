# Lexical Analysis

Source code is interpretted as a series of tokens. Whitespace is insignificant
to Muse, and any amount of whitespace is allowed between tokens. Comment tokens
are also ignored by the Muse compiler. Many of these tokens use Muse's own Regex
syntax to represent the allowed characters.

The types of tokens Muse supports are:

* Identifier: Any valid [Unicode
  Identifier][unicode-ident] is supported by Muse.
  Underscores are allowed anywhere in an identifier, and an identifier can be
  comprised solely of underscores. Ascii digits can be used anywhere except at
  the beginning of an identifier.
* Number: `<Integer | Float | Bits>`.
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

  * Expanded: `w\` followed by an escaped Regex where [whitespace is
    ignored][regex-whitespace], ended with `/`
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
* Label: `@` followed by a valid `<Identifier>`.
* Comment: `#` followed by any character until the next line.

[regex]: https://docs.rs/regex/
[regex-whitespace]: https://docs.rs/regex/latest/regex/struct.RegexBuilder.html#method.ignore_whitespace
[unicode-ident]: https://www.unicode.org/reports/tr31/
