//! Source code tokenization.

use std::cmp::Ordering;
use std::collections::VecDeque;
use std::fmt::Display;
use std::hash::Hash;
use std::ops::RangeBounds;
use std::str::CharIndices;

use serde::{Deserialize, Serialize};

use super::{Ranged, SourceCode, SourceId};
use crate::runtime::symbol::Symbol;

/// A Muse token.
///
/// A token is the smallest unit of syntax that forms Muse syntax. A series of
/// tokens can be parsed into an expression using Muse's grammar.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Token {
    /// A sequence of ASCII whitespace.
    ///
    /// Whitespace is insignificant to Muse and is ignored by the compiler.
    /// While being insignificant, it can still serve as a way to ensure a
    /// sequence of characters is broken into multiple tokens instead of
    /// interpreted as a single token (e.g., `forx` parses as a single
    /// identifier token while `for x` parses as two identifier tokens
    /// regardless of if you ignore whitespace tokens).
    Whitespace,
    /// A source code comment.
    ///
    /// Comments are ignored by the Muse compiler.
    Comment,
    /// A name or keyword (`foo`).
    Identifier(Symbol),
    /// A symbol (`:foo`).
    Symbol(Symbol),
    /// A sigil (`$foo`).
    Sigil(Symbol),
    /// A source label name (`@foo`).
    Label(Symbol),
    /// A signed integer.
    Int(i64),
    /// An unsigned integer.
    UInt(u64),
    /// A floating point number.
    Float(f64),
    /// A single character (not a character literal).
    Char(char),
    /// A string literal.
    String(Symbol),
    /// A regular expression literal.
    Regex(RegexLiteral),
    /// A format string.
    FormatString(FormatString),
    /// The power token `**`
    Power,
    /// The less than or equal token `<=`
    LessThanOrEqual,
    /// The greater than or equal token `>=`
    GreaterThanOrEqual,
    /// The equals token `==`
    Equals,
    /// The add assign token `+=`
    AddAssign,
    /// The add assign token `-=`
    SubtractAssign,
    /// The add assign token `*=`
    MultiplyAssign,
    /// The add assign token `/=`
    DivideAssign,
    /// The add assign token `//`
    IntegerDivide,
    /// The add assign token `//=`
    IntegerDivideAssign,
    /// The add assign token `%=`
    RemainderAssign,
    /// The shift left token `<<`
    ShiftLeft,
    /// The shift left asign token `<<=`
    ShiftLeftAssign,
    /// The shift right token `>>`
    ShiftRight,
    /// The shift right token assign `>>=`
    ShiftRightAssign,
    /// The not equal token `!=`
    NotEqual,
    /// The range token `..`
    Range,
    /// The range inclusive token `..=`
    RangeInclusive,
    /// The elipses token `...`
    Ellipses,
    /// The slim arrow token `->`
    SlimArrow,
    /// The fat arrow token `=>`
    FatArrow,
    /// Nil coalesce token `??`
    NilCoalesce,
    /// An opening of a paired token.
    Open(Paired),
    /// A closing of a paired token.
    Close(Paired),
}

impl Token {
    /// Returns true if this token is likely the end of an expression.
    ///
    /// This function is used in some locations to make the grammar not require
    /// as many "hard" keywords. Many grammar keywords only match if the next
    /// token is not a likely end, which allows for "escaping" ambiguous usage
    /// by surrounding the keyword in parentheses.
    #[must_use]
    pub fn is_likely_end(&self) -> bool {
        matches!(self, Token::Close(_) | Token::Char(';'))
    }
}

impl Eq for Token {}

impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Identifier(l0), Self::Identifier(r0))
            | (Self::Symbol(l0), Self::Symbol(r0))
            | (Self::String(l0), Self::String(r0)) => l0 == r0,
            (Self::Regex(l0), Self::Regex(r0)) => l0 == r0,
            (Self::FormatString(l0), Self::FormatString(r0)) => l0 == r0,
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
            (Self::UInt(l0), Self::UInt(r0)) => l0 == r0,
            (Self::Float(l0), Self::Float(r0)) => l0.total_cmp(r0) == Ordering::Equal,
            (Self::Char(l0), Self::Char(r0)) => l0 == r0,
            (Self::Open(l0), Self::Open(r0)) | (Self::Close(l0), Self::Close(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Hash for Token {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Token::Identifier(t) | Token::Label(t) | Token::Symbol(t) | Token::Sigil(t) => {
                t.hash(state);
            }
            Token::Int(t) => t.hash(state),
            Token::UInt(t) => t.hash(state),
            Token::Float(t) => t.to_bits().hash(state),
            Token::Char(t) => t.hash(state),
            Token::Open(t) | Token::Close(t) => t.hash(state),
            Token::String(t) => t.hash(state),
            Token::Regex(t) => t.hash(state),
            Token::FormatString(t) => t.hash(state),
            Token::Whitespace
            | Token::NilCoalesce
            | Token::Comment
            | Token::Power
            | Token::LessThanOrEqual
            | Token::GreaterThanOrEqual
            | Token::AddAssign
            | Token::Equals
            | Token::SubtractAssign
            | Token::MultiplyAssign
            | Token::DivideAssign
            | Token::IntegerDivide
            | Token::IntegerDivideAssign
            | Token::RemainderAssign
            | Token::ShiftLeft
            | Token::ShiftLeftAssign
            | Token::ShiftRight
            | Token::ShiftRightAssign
            | Token::NotEqual
            | Token::Range
            | Token::RangeInclusive
            | Token::Ellipses
            | Token::SlimArrow
            | Token::FatArrow => {}
        }
    }
}

/// A token kind that is expected in pairs.
#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash, Serialize, Deserialize)]
pub enum Paired {
    /// A curly brace (`{` or `}`)
    Brace,
    /// A parenthesis (`(` or `)`)
    Paren,
    /// A square bracket (`[` or `]`)
    Bracket,
}

impl Paired {
    /// Returns the char corresponding to the opening token of this kind.
    #[must_use]
    pub fn as_open(self) -> char {
        match self {
            Paired::Brace => '{',
            Paired::Paren => '(',
            Paired::Bracket => '[',
        }
    }

    /// Returns the char corresponding to the closing token of this kind.
    #[must_use]
    pub fn as_close(self) -> char {
        match self {
            Paired::Brace => '}',
            Paired::Paren => ')',
            Paired::Bracket => ']',
        }
    }
}

struct PeekNChars<'a> {
    peeked: VecDeque<(usize, char)>,
    chars: CharIndices<'a>,
}

impl<'a> PeekNChars<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            peeked: VecDeque::new(),
            chars: source.char_indices(),
        }
    }

    fn peek_n(&mut self, n: usize) -> Option<&(usize, char)> {
        while n >= self.peeked.len() {
            self.peeked.push_back(self.chars.next()?);
        }

        self.peeked.get(n)
    }

    fn peek(&mut self) -> Option<&(usize, char)> {
        self.peek_n(0)
    }
}

impl Iterator for PeekNChars<'_> {
    type Item = (usize, char);

    fn next(&mut self) -> Option<Self::Item> {
        self.peeked.pop_front().or_else(|| self.chars.next())
    }
}

struct Chars<'a> {
    id: SourceId,
    source: PeekNChars<'a>,
    last_index: usize,
}

impl<'a> Chars<'a> {
    fn new(source: &'a str, id: SourceId) -> Self {
        Self {
            id,
            source: PeekNChars::new(source),
            last_index: 0,
        }
    }

    pub const fn source(&self) -> SourceId {
        self.id
    }

    fn peek_n(&mut self, n: usize) -> Option<char> {
        self.source.peek_n(n).map(|(_, ch)| *ch)
    }

    fn advance(&mut self, n: usize) {
        for _ in 0..n {
            self.source.next();
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.source.peek().map(|(_, ch)| *ch)
    }

    fn ranged<T>(&self, range: impl RangeBounds<usize>, value: T) -> Ranged<T> {
        Ranged::bounded(self.id, range, self.last_index, value)
    }
}

impl Iterator for Chars<'_> {
    type Item = (usize, char);

    fn next(&mut self) -> Option<Self::Item> {
        let (offset, char) = self.source.next()?;
        self.last_index = offset + char.len_utf8();
        Some((offset, char))
    }
}

pub(super) struct Tokens<'a> {
    chars: Chars<'a>,
    scratch: String,
    include_whitespace: bool,
    include_comments: bool,
}

impl Iterator for Tokens<'_> {
    type Item = Result<Ranged<Token>, Ranged<LexerError>>;

    #[allow(clippy::too_many_lines)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            break Some(match self.chars.next()? {
                (start, ch) if ch.is_ascii_whitespace() => {
                    if self.include_whitespace {
                        Ok(self.tokenize_whitespace(start))
                    } else {
                        continue;
                    }
                }
                (start, '#') => {
                    let comment = self.tokenize_oneline_comment(start);
                    if self.include_comments {
                        Ok(comment)
                    } else {
                        continue;
                    }
                }

                (start, '"') => self.tokenize_string(start),
                (start, 'r') if matches!(self.chars.peek(), Some('"' | '#')) => {
                    self.tokenize_raw_string(start)
                }
                (start, 'f') if self.chars.peek() == Some('"') => {
                    self.tokenize_format_string(start, false)
                }
                (start, 'f')
                    if self.chars.peek() == Some('r')
                        && matches!(self.chars.peek_n(1), Some('"' | '#')) =>
                {
                    self.chars.next();
                    self.tokenize_format_string(start, true)
                }
                (start, '$') => Ok(self.tokenize_sigil(start)),
                (start, '@') if self.chars.peek().map_or(false, unicode_ident::is_xid_start) => {
                    let ch = self.chars.next().expect("just peekend").1;
                    Ok(self.tokenize_label(start, ch))
                }
                (start, ':') if self.chars.peek().map_or(false, unicode_ident::is_xid_start) => {
                    let ch = self.chars.next().expect("just peekend").1;
                    Ok(self.tokenize_symbol(start, ch))
                }
                (start, '\\') => self.tokenize_regex(start, false),
                (start, 'w') if self.chars.peek().map_or(false, |ch| ch == '\\') => {
                    self.chars.next()?;
                    self.tokenize_regex(start, true)
                }
                (start, '{') => Ok(self.chars.ranged(start.., Token::Open(Paired::Brace))),
                (start, '}') => Ok(self.chars.ranged(start.., Token::Close(Paired::Brace))),
                (start, '(') => Ok(self.chars.ranged(start.., Token::Open(Paired::Paren))),
                (start, ')') => Ok(self.chars.ranged(start.., Token::Close(Paired::Paren))),
                (start, '[') => Ok(self.chars.ranged(start.., Token::Open(Paired::Bracket))),
                (start, ']') => Ok(self.chars.ranged(start.., Token::Close(Paired::Bracket))),
                (start, ch) if ch.is_ascii_digit() => self.tokenize_number(start, ch),
                (start, '.') if self.chars.peek().map_or(false, |ch| ch.is_ascii_digit()) => {
                    self.tokenize_number(start, '.')
                }
                (start, '.') if self.chars.peek() == Some('.') => Ok(self.tokenize_range(start)),
                (start, '*') if self.chars.peek() == Some('*') => {
                    self.chars.next();
                    Ok(self.chars.ranged(start.., Token::Power))
                }
                (start, '/') if self.chars.peek() == Some('/') => {
                    self.chars.next();
                    Ok(self.chars.ranged(start.., Token::IntegerDivide))
                }
                (start, '=') if self.chars.peek() == Some('=') => {
                    self.chars.next();
                    Ok(self.chars.ranged(start.., Token::Equals))
                }
                (start, '<') if self.chars.peek() == Some('=') => {
                    self.chars.next();
                    Ok(self.chars.ranged(start.., Token::LessThanOrEqual))
                }
                (start, '>') if self.chars.peek() == Some('=') => {
                    self.chars.next();
                    Ok(self.chars.ranged(start.., Token::GreaterThanOrEqual))
                }
                (start, '!') if self.chars.peek() == Some('=') => {
                    self.chars.next();
                    Ok(self.chars.ranged(start.., Token::NotEqual))
                }
                (start, '-') if self.chars.peek() == Some('>') => {
                    self.chars.next();
                    Ok(self.chars.ranged(start.., Token::SlimArrow))
                }
                (start, '=') if self.chars.peek() == Some('>') => {
                    self.chars.next();
                    Ok(self.chars.ranged(start.., Token::FatArrow))
                }
                (start, '?') if self.chars.peek() == Some('?') => {
                    self.chars.next();

                    Ok(self.chars.ranged(start.., Token::NilCoalesce))
                }
                (start, ch)
                    if unicode_ident::is_xid_start(ch)
                        || (ch == '_' && self.chars.peek().map_or(false, is_ident_continue)) =>
                {
                    Ok(self.tokenize_identifier(start, ch))
                }
                (start, ch) if ch.is_ascii_punctuation() => {
                    Ok(self.chars.ranged(start.., Token::Char(ch)))
                }
                (start, ch) => Err(self.chars.ranged(start.., LexerError::UnexpectedChar(ch))),
            });
        }
    }
}

fn is_ident_continue(ch: char) -> bool {
    unicode_ident::is_xid_continue(ch) || ch == '_' || ch.is_ascii_digit()
}

impl<'a> Tokens<'a> {
    #[must_use]
    pub fn new(source: impl Into<SourceCode<'a>>) -> Self {
        let source = source.into();
        Self {
            chars: Chars::new(source.code, source.id),
            scratch: String::new(),
            include_whitespace: true,
            include_comments: true,
        }
    }

    #[must_use]
    pub const fn source(&self) -> SourceId {
        self.chars.source()
    }

    #[must_use]
    pub fn excluding_whitespace(mut self) -> Self {
        self.include_whitespace = false;
        self
    }

    #[must_use]
    pub fn excluding_comments(mut self) -> Self {
        self.include_comments = false;
        self
    }

    fn tokenize_sigil(&mut self, start: usize) -> Ranged<Token> {
        let symbol = if self.chars.peek().map_or(false, unicode_ident::is_xid_start) {
            self.tokenize_identifier_symbol(start, '$')
        } else {
            self.chars.ranged(start.., Symbol::sigil_symbol().clone())
        };

        symbol.map(Token::Sigil)
    }

    fn tokenize_range(&mut self, start: usize) -> Ranged<Token> {
        let second_dot = self.chars.next();
        debug_assert!(matches!(second_dot, Some((_, '.'))));

        match self.chars.peek() {
            Some('=') => {
                self.chars.next();
                self.chars.ranged(start.., Token::RangeInclusive)
            }
            Some('.') => {
                self.chars.next();
                self.chars.ranged(start.., Token::Ellipses)
            }
            _ => self.chars.ranged(start.., Token::Range),
        }
    }

    fn tokenize_number(
        &mut self,
        start: usize,
        start_char: char,
    ) -> Result<Ranged<Token>, Ranged<LexerError>> {
        self.scratch.clear();
        self.scratch.push(start_char);
        let has_decimal = if start_char == '.' {
            true
        } else {
            while let Some(ch) = self
                .chars
                .peek()
                .filter(|ch| ch.is_ascii_digit() || *ch == '_')
            {
                if ch != '_' {
                    self.scratch.push(ch);
                }
                self.chars.next();
            }

            if self.chars.peek() == Some('.') {
                self.scratch.push('.');
                self.chars.next();
                true
            } else {
                false
            }
        };

        // TODO e syntax?

        if has_decimal {
            while let Some(ch) = self
                .chars
                .peek()
                .filter(|ch| ch.is_ascii_digit() || *ch == '_')
            {
                if ch != '_' {
                    self.scratch.push(ch);
                }
                self.chars.next();
            }

            let float = self.scratch.parse::<f64>().map_err(|err| {
                self.chars
                    .ranged(start.., LexerError::FloatParse(err.to_string()))
            })?;
            Ok(self.chars.ranged(start.., Token::Float(float)))
        } else {
            let signed = if self.chars.peek() == Some('u') {
                self.chars.next();
                false
            } else {
                true
            };
            match self.chars.peek() {
                Some('x') if self.scratch == "0" => {
                    self.chars.next();
                    self.tokenize_radix(start, 16, signed)
                }
                Some('b') if self.scratch == "0" => {
                    self.chars.next();
                    self.tokenize_radix(start, 2, signed)
                }
                Some('o') if self.scratch == "0" => {
                    self.chars.next();
                    self.tokenize_radix(start, 8, signed)
                }
                Some('r') => {
                    self.chars.next();
                    let radix = self.scratch.parse::<u32>().map_err(|err| {
                        self.chars
                            .ranged(start.., LexerError::IntegerParse(err.to_string()))
                    })?;
                    self.tokenize_radix(start, radix, signed)
                }
                _ => {
                    let token = if signed {
                        self.scratch.parse::<i64>().map(Token::Int)
                    } else {
                        self.scratch.parse::<u64>().map(Token::UInt)
                    }
                    .map_err(|err| {
                        self.chars
                            .ranged(start.., LexerError::IntegerParse(err.to_string()))
                    })?;

                    Ok(self.chars.ranged(start.., token))
                }
            }
        }
    }

    fn tokenize_radix(
        &mut self,
        start: usize,
        radix: u32,
        signed: bool,
    ) -> Result<Ranged<Token>, Ranged<LexerError>> {
        self.scratch.clear();
        if let Some(alpha_radix @ 1..) = radix.checked_sub(10) {
            // Radix > 10. We need to check ascii_digit as well as filter the
            // ascii alphabet.
            while let Some(ch) = self.chars.peek().filter(|ch| {
                ch.is_ascii_digit()
                    || u8::try_from(*ch).map_or(false, |ch| {
                        u32::from(ch.to_ascii_lowercase().wrapping_sub(b'a')) < alpha_radix
                    })
                    || *ch == '_'
            }) {
                if ch != '_' {
                    self.scratch.push(ch);
                }
                self.chars.next();
            }
        } else {
            // Radix <= 10. We only need to filter based on ascii digits.
            while let Some(ch) = self.chars.peek().filter(|ch| {
                u8::try_from(*ch).map_or(false, |ch| {
                    u32::from(ch.to_ascii_lowercase().wrapping_sub(b'0')) < radix
                }) || *ch == '_'
            }) {
                if ch != '_' {
                    self.scratch.push(ch);
                }
                self.chars.next();
            }
        }

        let decoded_bits = u64::from_str_radix(&self.scratch, radix).map_err(|err| {
            self.chars
                .ranged(start.., LexerError::IntegerParse(err.to_string()))
        })?;

        #[allow(clippy::cast_possible_wrap)]
        let token = if signed {
            Token::Int(decoded_bits as i64)
        } else {
            Token::UInt(decoded_bits)
        };

        Ok(self.chars.ranged(start.., token))
    }

    fn tokenize_identifier_symbol(&mut self, start: usize, start_char: char) -> Ranged<Symbol> {
        self.scratch.clear();
        self.scratch.push(start_char);

        while let Some(ch) = self.chars.peek().filter(|ch| is_ident_continue(*ch)) {
            self.scratch.push(ch);
            self.chars.next();
        }

        let symbol = Symbol::from(&self.scratch);
        self.chars.ranged(start.., symbol)
    }

    fn tokenize_identifier(&mut self, start: usize, start_char: char) -> Ranged<Token> {
        self.tokenize_identifier_symbol(start, start_char)
            .map(Token::Identifier)
    }

    fn tokenize_label(&mut self, start: usize, start_char: char) -> Ranged<Token> {
        self.tokenize_identifier_symbol(start, start_char)
            .map(Token::Label)
    }

    fn tokenize_symbol(&mut self, start: usize, start_char: char) -> Ranged<Token> {
        self.tokenize_identifier_symbol(start, start_char)
            .map(Token::Symbol)
    }

    fn tokenize_whitespace(&mut self, start: usize) -> Ranged<Token> {
        while self
            .chars
            .peek()
            .map_or(false, |ch| ch.is_ascii_whitespace())
        {
            self.chars.next();
        }
        self.chars.ranged(start.., Token::Whitespace)
    }

    fn tokenize_oneline_comment(&mut self, start: usize) -> Ranged<Token> {
        while self
            .chars
            .peek()
            .map_or(false, |ch| !matches!(ch, '\r' | '\n'))
        {
            self.chars.next();
        }
        self.chars.ranged(start.., Token::Comment)
    }

    fn tokenize_string_literal_into_scratch(
        &mut self,
        allowed_escapes: &[char],
        fallback: impl Fn(&mut Self, usize, char) -> Result<StringFlow, Ranged<LexerError>>,
    ) -> Result<bool, Ranged<LexerError>> {
        self.scratch.clear();
        loop {
            match self.chars.next() {
                Some((_, '"')) => break Ok(false),
                Some((index, '\\')) => match self.chars.next() {
                    Some((_, '"')) => self.scratch.push('"'),
                    Some((_, 'n')) => self.scratch.push('\n'),
                    Some((_, 'r')) => self.scratch.push('\r'),
                    Some((_, 't')) => self.scratch.push('\t'),
                    Some((_, '\\')) => self.scratch.push('\\'),
                    Some((_, '\0')) => self.scratch.push('\0'),
                    Some((_, 'u')) => self.decode_unicode_escape_into_scratch().map_err(|()| {
                        self.chars
                            .ranged(index.., LexerError::InvalidEscapeSequence)
                    })?,
                    Some((_, 'x')) => self.decode_ascii_escape_into_scratch().map_err(|()| {
                        self.chars
                            .ranged(index.., LexerError::InvalidEscapeSequence)
                    })?,
                    Some((_, ch)) if allowed_escapes.contains(&ch) => self.scratch.push(ch),
                    _ => {
                        return Err(self
                            .chars
                            .ranged(index.., LexerError::InvalidEscapeSequence))
                    }
                },
                Some((offset, ch)) => match fallback(self, offset, ch)? {
                    StringFlow::Break => break Ok(true),
                    StringFlow::Unhandled => {
                        self.scratch.push(ch);
                    }
                },
                None => {
                    return Err(self
                        .chars
                        .ranged(self.chars.last_index.., LexerError::MissingEndQuote))
                }
            }
        }
    }

    fn tokenize_string(&mut self, start: usize) -> Result<Ranged<Token>, Ranged<LexerError>> {
        self.tokenize_string_literal_into_scratch(&[], |_, _, _| Ok(StringFlow::Unhandled))?;

        Ok(self
            .chars
            .ranged(start.., Token::String(Symbol::from(&self.scratch))))
    }

    fn tokenize_raw_string(&mut self, start: usize) -> Result<Ranged<Token>, Ranged<LexerError>> {
        self.tokenize_raw_string_literal_into_scratch(&[], |_, _, _| Ok(StringFlow::Unhandled))?;

        Ok(self
            .chars
            .ranged(start.., Token::String(Symbol::from(&self.scratch))))
    }

    fn decode_unicode_escape_into_scratch(&mut self) -> Result<(), ()> {
        match self.chars.next() {
            Some((_, '{')) => {}
            _ => return Err(()),
        }

        let mut decoded = 0_u32;
        let mut digit_count = 0;
        let mut found_brace = false;
        while digit_count < 6 {
            match self.chars.next() {
                Some((_, '}')) => {
                    found_brace = true;
                    break;
                }
                Some((_, ch)) => {
                    if let Some(digit) = decode_hex_char(ch) {
                        decoded <<= 4;
                        decoded |= u32::from(digit);
                        digit_count += 1;
                    } else {
                        return Err(());
                    }
                }
                None => break,
            }
        }

        if digit_count == 0 {
            return Err(());
        }

        if !found_brace {
            match self.chars.next() {
                Some((_, '}')) => {}
                _ => return Err(()),
            }
        }

        let ch = char::from_u32(decoded).ok_or(())?;

        self.scratch.push(ch);
        Ok(())
    }

    fn decode_ascii_escape_into_scratch(&mut self) -> Result<(), ()> {
        match (self.chars.next(), self.chars.next()) {
            (Some((_, high)), Some((_, low))) => {
                match (decode_hex_char(high), decode_hex_char(low)) {
                    (Some(high), Some(low)) => {
                        let ascii = (high << 4) | low;
                        if ascii <= 127 {
                            self.scratch.push(char::from(ascii));
                            Ok(())
                        } else {
                            Err(())
                        }
                    }
                    _ => Err(()),
                }
            }
            _ => Err(()),
        }
    }

    fn determine_raw_string_thorpeness(&mut self) -> Result<usize, Ranged<LexerError>> {
        let mut octothorpeness = 0;
        while self.chars.peek() == Some('#') {
            octothorpeness += 1;
            self.chars.next();
        }
        match self.chars.next() {
            Some((_, '"')) => Ok(octothorpeness),
            Some((index, _)) => Err(self.chars.ranged(index.., LexerError::ExpectedRawString)),
            None => Err(self
                .chars
                .ranged(self.chars.last_index.., LexerError::ExpectedRawString)),
        }
    }

    fn tokenize_raw_string_literal_into_scratch(
        &mut self,
        allowed_escapes: &[char],
        fallback: impl Fn(&mut Self, usize, char) -> Result<StringFlow, Ranged<LexerError>>,
    ) -> Result<bool, Ranged<LexerError>> {
        let octothorpeness = self.determine_raw_string_thorpeness()?;
        self.tokenize_raw_string_literal_into_scratch_with_thorpeness(
            octothorpeness,
            allowed_escapes,
            fallback,
        )
    }

    fn tokenize_raw_string_literal_into_scratch_with_thorpeness(
        &mut self,
        octothorpeness: usize,
        allowed_escapes: &[char],
        fallback: impl Fn(&mut Self, usize, char) -> Result<StringFlow, Ranged<LexerError>>,
    ) -> Result<bool, Ranged<LexerError>> {
        self.scratch.clear();
        'decoding: loop {
            match self.chars.next() {
                Some((_, '"')) => {
                    for thorp in 0..octothorpeness {
                        if self.chars.peek_n(thorp) != Some('#') {
                            self.scratch.push('"');
                            continue 'decoding;
                        }
                    }
                    self.chars.advance(octothorpeness);
                    break Ok(false);
                }
                Some((_, ch)) if allowed_escapes.contains(&ch) && self.chars.peek() == Some(ch) => {
                    self.chars.next();
                    self.scratch.push(ch);
                }
                Some((offset, ch)) => match fallback(self, offset, ch)? {
                    StringFlow::Break => break Ok(true),
                    StringFlow::Unhandled => {
                        self.scratch.push(ch);
                    }
                },
                _ => {
                    return Err(self
                        .chars
                        .ranged(self.chars.last_index.., LexerError::MissingEndQuote))
                }
            }
        }
    }

    #[allow(clippy::unnecessary_wraps)]
    fn decode_format_string_fallback(
        &mut self,
        _offset: usize,
        ch: char,
    ) -> Result<StringFlow, Ranged<LexerError>> {
        match ch {
            '$' if self.chars.peek() == Some('{') => Ok(StringFlow::Break),
            _ => Ok(StringFlow::Unhandled),
        }
    }

    fn decode_format_string_contents(
        &mut self,
        raw: bool,
        octothorpeness: usize,
    ) -> Result<bool, Ranged<LexerError>> {
        if raw {
            self.tokenize_raw_string_literal_into_scratch_with_thorpeness(
                octothorpeness,
                &['$'],
                Self::decode_format_string_fallback,
            )
        } else {
            self.tokenize_string_literal_into_scratch(&['$'], Self::decode_format_string_fallback)
        }
    }

    fn tokenize_format_string(
        &mut self,
        start: usize,
        raw: bool,
    ) -> Result<Ranged<Token>, Ranged<LexerError>> {
        let octothorpeness = if raw {
            self.determine_raw_string_thorpeness()?
        } else {
            self.chars.next();
            0
        };
        let mut continued = self.decode_format_string_contents(raw, octothorpeness)?;

        let initial = self.chars.ranged(start.., Symbol::from(&self.scratch));
        let mut parts = Vec::new();

        let mut stack = Vec::new();
        while continued {
            let (brace_offset, _brace) = self.chars.next().expect("just peeked");
            stack.clear();
            stack.push(Paired::Brace);
            let mut expression = vec![self
                .chars
                .ranged(brace_offset.., Token::Open(Paired::Brace))];
            while let Some(last_open) = stack.last().copied() {
                let token = self.next().ok_or_else(|| {
                    self.chars
                        .ranged(self.chars.last_index.., LexerError::MissingEndQuote)
                })??;
                expression.push(token);
                match &expression.last().expect("just pushed").0 {
                    Token::Close(kind) => {
                        if *kind == last_open {
                            stack.pop();
                        } else {
                            // TODO change to a MissingEnd, but then refactor the other MissingEnd to use this variant.
                            return Err(self
                                .chars
                                .ranged(self.chars.last_index.., LexerError::MissingEndQuote));
                        }
                    }
                    Token::Open(kind) => {
                        stack.push(*kind);
                    }
                    _ => {}
                }
            }
            let suffix_start = self.chars.last_index;
            continued = self.decode_format_string_contents(raw, octothorpeness)?;
            parts.push(FormatStringPart {
                expression,
                suffix: self
                    .chars
                    .ranged(suffix_start.., Symbol::from(&self.scratch)),
            });
        }

        Ok(self.chars.ranged(
            start..,
            Token::FormatString(FormatString { initial, parts }),
        ))
    }

    fn tokenize_regex(
        &mut self,
        start: usize,
        expanded: bool,
    ) -> Result<Ranged<Token>, Ranged<LexerError>> {
        self.scratch.clear();
        loop {
            match self.chars.next() {
                Some((_, '/')) => break,
                Some((index, ch @ ('\r' | '\n'))) if !expanded => {
                    return Err(self.chars.ranged(index.., LexerError::UnexpectedChar(ch)))
                }
                Some((index, '\\')) => match self.chars.next() {
                    Some((_, '/')) => self.scratch.push('/'),
                    Some((_, ch)) => {
                        self.scratch.push('\\');
                        self.scratch.push(ch);
                    }
                    _ => {
                        return Err(self
                            .chars
                            .ranged(index.., LexerError::InvalidEscapeSequence))
                    }
                },
                Some((_, ch)) => {
                    self.scratch.push(ch);
                }
                None => {
                    return Err(self
                        .chars
                        .ranged(self.chars.last_index.., LexerError::MissingRegexEnd))
                }
            }
        }

        let mut ignore_case = false;
        let mut unicode = false;
        let mut dot_matches_all = false;
        let mut multiline = false;
        loop {
            match self.chars.peek() {
                Some('i') => ignore_case = true,
                Some('u') => unicode = true,
                Some('s') => dot_matches_all = true,
                Some('m') => multiline = true,
                _ => break,
            }

            self.chars.next();
        }

        Ok(self.chars.ranged(
            start..,
            Token::Regex(RegexLiteral {
                pattern: self.scratch.clone(),
                expanded,
                ignore_case,
                unicode,
                dot_matches_all,
                multiline,
            }),
        ))
    }
}

enum StringFlow {
    Break,
    Unhandled,
}

fn decode_hex_char(ch: char) -> Option<u8> {
    const ASCII_CASE_BIT: u8 = 0b10_0000;
    let u8 = u8::try_from(ch).ok()? | ASCII_CASE_BIT;
    match u8 {
        b'0'..=b'9' => Some(u8 - b'0'),
        b'a'..=b'f' => Some(u8 - b'a' + 10),
        _ => None,
    }
}

#[test]
fn decode_hex_char_tests() {
    assert_eq!(decode_hex_char('1'), Some(1));
    assert_eq!(decode_hex_char('a'), Some(10));
    assert_eq!(decode_hex_char('F'), Some(15));
    assert_eq!(decode_hex_char('.'), None);
}

/// An error occurred while tokenizing source code.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum LexerError {
    /// An unexpected character was found.
    UnexpectedChar(char),
    /// An error occurred parsing an integer value.
    IntegerParse(String),
    /// An error occurred parsing a floating point value.
    FloatParse(String),
    /// A string literal is missing its end quote.
    MissingEndQuote,
    /// A regular expression literal is missing its end slash.
    MissingRegexEnd,
    /// A raw string literal was expected.
    ExpectedRawString,
    /// An invalid escape sequence in a string literal.
    InvalidEscapeSequence,
}

impl crate::ErrorKind for LexerError {
    fn kind(&self) -> &'static str {
        match self {
            LexerError::UnexpectedChar(_) => "unexpected char",
            LexerError::IntegerParse(_) => "invalid integer literal",
            LexerError::FloatParse(_) => "invalid float literal",
            LexerError::MissingEndQuote => "missing end quote",
            LexerError::MissingRegexEnd => "missing regex end",
            LexerError::InvalidEscapeSequence => "invalid escape sequence",
            LexerError::ExpectedRawString => "expected raw string",
        }
    }
}

impl std::error::Error for LexerError {}

impl Display for LexerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LexerError::UnexpectedChar(ch) => write!(f, "unexpected character: {ch}"),
            LexerError::IntegerParse(err) => write!(f, "invalid integer literal: {err}"),
            LexerError::FloatParse(err) => write!(f, "invalid floating point literal: {err}"),
            LexerError::MissingEndQuote => f.write_str("missing end quote (\")"),
            LexerError::MissingRegexEnd => f.write_str("missing regular expression end (/)"),
            LexerError::InvalidEscapeSequence => f.write_str("invalid escape sequence"),
            LexerError::ExpectedRawString => f.write_str("expected raw string"),
        }
    }
}

/// A regular expression literal.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, Hash, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct RegexLiteral {
    /// The regular expression source.
    pub pattern: String,
    /// If true, whitespace is ignored by default.
    ///
    /// See the documentation for [ignore_whitespace][regex-whitespace] for more
    /// information.
    ///
    /// [regex-whitespace]:
    ///     https://docs.rs/regex/latest/regex/struct.RegexBuilder.html#method.ignore_whitespace
    #[serde(default)]
    pub expanded: bool,
    /// If true, the regular expression will ignore uppercase/lowercase differences when matching.
    #[serde(default)]
    pub ignore_case: bool,
    /// When true, unicode mode is enabled for the regular expression.
    #[serde(default)]
    pub unicode: bool,
    /// When true, `.` matches all characters including new lines.
    #[serde(default)]
    pub dot_matches_all: bool,
    /// When true, the regular expression will operate in multiline mode.
    #[serde(default)]
    pub multiline: bool,
}

/// A format string literal.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, Hash, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct FormatString {
    /// The initial portion of the string literal.
    pub initial: Ranged<Symbol>,
    /// The remaining parts that are joined to the intitial string.
    pub parts: Vec<FormatStringPart>,
}

/// A part of a format string literal.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, Hash, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct FormatStringPart {
    /// The expression to evaluate and convert to a string.
    pub expression: Vec<Ranged<Token>>,
    /// The literal string that follows the expression.
    pub suffix: Ranged<Symbol>,
}

#[test]
fn basics() {
    use std::num::NonZeroUsize;
    let tokens = Tokens::new("a_09_ + 1 - .2 __reserved")
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        tokens,
        &[
            Ranged::new(
                (SourceId::anonymous(), 0..5),
                Token::Identifier(Symbol::from("a_09_"))
            ),
            Ranged::new((SourceId::anonymous(), 5..6), Token::Whitespace),
            Ranged::new((SourceId::anonymous(), 6..7), Token::Char('+')),
            Ranged::new((SourceId::anonymous(), 7..8), Token::Whitespace),
            Ranged::new((SourceId::anonymous(), 8..9), Token::Int(1)),
            Ranged::new((SourceId::anonymous(), 9..10), Token::Whitespace),
            Ranged::new((SourceId::anonymous(), 10..11), Token::Char('-')),
            Ranged::new((SourceId::anonymous(), 11..12), Token::Whitespace),
            Ranged::new((SourceId::anonymous(), 12..14), Token::Float(0.2)),
            Ranged::new((SourceId::anonymous(), 14..15), Token::Whitespace),
            Ranged::new(
                (SourceId::anonymous(), 15..25),
                Token::Identifier(Symbol::from("__reserved"))
            ),
        ]
    );
    let id = SourceId::new(NonZeroUsize::MIN);
    let tokens = Tokens::new(SourceCode::new("a_09_ + 1 - .2", id))
        .excluding_whitespace()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        tokens,
        &[
            Ranged::new((id, 0..5), Token::Identifier(Symbol::from("a_09_"))),
            Ranged::new((id, 6..7), Token::Char('+')),
            Ranged::new((id, 8..9), Token::Int(1)),
            Ranged::new((id, 10..11), Token::Char('-')),
            Ranged::new((id, 12..14), Token::Float(0.2)),
        ]
    );
}
