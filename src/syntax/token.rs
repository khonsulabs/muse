use std::cmp::Ordering;
use std::hash::Hash;
use std::iter::Peekable;
use std::ops::RangeBounds;
use std::str::CharIndices;

use serde::{Deserialize, Serialize};

use super::Ranged;
use crate::symbol::Symbol;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Token {
    Whitespace,
    Comment,
    Identifier(Symbol),
    Symbol(Symbol),
    Label(Symbol),
    Int(i64),
    Float(f64),
    Char(char),
    String(String),
    RegEx(RegExLiteral),
    Power,
    LessThanOrEqual,
    GreaterThanOrEqual,
    Equals,
    AddAssign,
    SubtractAssign,
    MultiplyAssign,
    DivideAssign,
    IntegerDivide,
    IntegerDivideAssign,
    RemainderAssign,
    ShiftLeft,
    ShiftLeftAssign,
    ShiftRight,
    ShiftRightAssign,
    NotEqual,
    Range,
    RangeInclusive,
    SlimArrow,
    FatArrow,
    Open(Paired),
    Close(Paired),
}

impl Token {
    #[must_use]
    pub fn is_likely_end(&self) -> bool {
        matches!(self, Token::Close(_) | Token::Char(';'))
    }
}

impl Eq for Token {}

impl PartialEq for Token {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Identifier(l0), Self::Identifier(r0)) => l0 == r0,
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
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
            Token::Identifier(t) | Token::Label(t) | Token::Symbol(t) => t.hash(state),
            Token::Int(t) => t.hash(state),
            Token::Float(t) => t.to_bits().hash(state),
            Token::Char(t) => t.hash(state),
            Token::Open(t) | Token::Close(t) => t.hash(state),
            Token::String(t) => t.hash(state),
            Token::RegEx(t) => t.hash(state),
            Token::Whitespace
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
            | Token::SlimArrow
            | Token::FatArrow => {}
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash, Serialize, Deserialize)]
pub enum Paired {
    Brace,
    Paren,
    Bracket,
}

struct Chars<'a> {
    source: Peekable<CharIndices<'a>>,
    last_index: usize,
}

impl<'a> Chars<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source: source.char_indices().peekable(),
            last_index: 0,
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.source.peek().map(|(_, ch)| *ch)
    }

    fn ranged<T>(&self, range: impl RangeBounds<usize>, value: T) -> Ranged<T> {
        Ranged::bounded(range, self.last_index, value)
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

pub struct Tokens<'a> {
    chars: Chars<'a>,
    scratch: String,
    include_whitespace: bool,
    include_comments: bool,
}

impl Iterator for Tokens<'_> {
    type Item = Result<Ranged<Token>, Ranged<Error>>;

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
                (start, ch @ ('.' | '-'))
                    if self.chars.peek().map_or(false, |ch| ch.is_ascii_digit()) =>
                {
                    self.tokenize_number(start, ch)
                }
                (start, '.') if self.chars.peek() == Some('.') => {
                    self.chars.next();
                    if self.chars.peek() == Some('=') {
                        self.chars.next();
                        Ok(self.chars.ranged(start.., Token::RangeInclusive))
                    } else {
                        Ok(self.chars.ranged(start.., Token::Range))
                    }
                }
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
                (start, ch) if ch.is_ascii_punctuation() => {
                    Ok(self.chars.ranged(start.., Token::Char(ch)))
                }
                (start, ch) if unicode_ident::is_xid_start(ch) => {
                    Ok(self.tokenize_identifier(start, ch))
                }
                (start, ch) => Err(self.chars.ranged(start.., Error::UnexpectedChar(ch))),
            });
        }
    }
}

impl<'a> Tokens<'a> {
    #[must_use]
    pub fn new(source: &'a str) -> Self {
        Self {
            chars: Chars::new(source),
            scratch: String::new(),
            include_whitespace: true,
            include_comments: true,
        }
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

    fn tokenize_number(
        &mut self,
        start: usize,
        start_char: char,
    ) -> Result<Ranged<Token>, Ranged<Error>> {
        self.scratch.clear();
        self.scratch.push(start_char);
        let has_decimal = if start_char == '.' {
            true
        } else {
            while let Some(ch) = self.chars.peek().filter(char::is_ascii_digit) {
                self.scratch.push(ch);
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
            while let Some(ch) = self.chars.peek().filter(char::is_ascii_digit) {
                self.scratch.push(ch);
                self.chars.next();
            }

            let float = self.scratch.parse::<f64>().map_err(|err| {
                self.chars
                    .ranged(start.., Error::FloatParse(err.to_string()))
            })?;
            Ok(self.chars.ranged(start.., Token::Float(float)))
        } else {
            let int = self.scratch.parse::<i64>().map_err(|err| {
                self.chars
                    .ranged(start.., Error::IntegerParse(err.to_string()))
            })?;
            Ok(self.chars.ranged(start.., Token::Int(int)))
        }
    }

    fn tokenize_identifier_symbol(&mut self, start: usize, start_char: char) -> Ranged<Symbol> {
        self.scratch.clear();
        self.scratch.push(start_char);

        while let Some(ch) = self
            .chars
            .peek()
            .filter(|ch| unicode_ident::is_xid_continue(*ch) || *ch == '_' || ch.is_ascii_digit())
        {
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

    fn tokenize_string(&mut self, start: usize) -> Result<Ranged<Token>, Ranged<Error>> {
        self.scratch.clear();
        loop {
            match self.chars.next() {
                Some((_, '"')) => break,
                Some((index, '\\')) => match self.chars.next() {
                    Some((_, '"')) => self.scratch.push('"'),
                    Some((_, 'n')) => self.scratch.push('\n'),
                    Some((_, 'r')) => self.scratch.push('\r'),
                    Some((_, 't')) => self.scratch.push('\t'),
                    Some((_, '\\')) => self.scratch.push('\\'),
                    Some((_, '\0')) => self.scratch.push('\0'),
                    Some((_, 'u')) => self
                        .decode_unicode_escape_into_scratch()
                        .map_err(|()| self.chars.ranged(index.., Error::InvalidEscapeSequence))?,
                    Some((_, 'x')) => self
                        .decode_ascii_escape_into_scratch()
                        .map_err(|()| self.chars.ranged(index.., Error::InvalidEscapeSequence))?,
                    _ => return Err(self.chars.ranged(index.., Error::InvalidEscapeSequence)),
                },
                Some((_, ch)) => {
                    self.scratch.push(ch);
                }
                None => {
                    return Err(self
                        .chars
                        .ranged(self.chars.last_index.., Error::MissingEndQuote))
                }
            }
        }

        Ok(self
            .chars
            .ranged(start.., Token::String(self.scratch.clone())))
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

    fn tokenize_regex(
        &mut self,
        start: usize,
        expanded: bool,
    ) -> Result<Ranged<Token>, Ranged<Error>> {
        self.scratch.clear();
        loop {
            match self.chars.next() {
                Some((_, '/')) => break,
                Some((index, ch @ ('\r' | '\n'))) if !expanded => {
                    return Err(self.chars.ranged(index.., Error::UnexpectedChar(ch)))
                }
                Some((index, '\\')) => match self.chars.next() {
                    Some((_, '/')) => self.scratch.push('/'),
                    Some((_, ch)) => {
                        self.scratch.push('\\');
                        self.scratch.push(ch);
                    }
                    _ => return Err(self.chars.ranged(index.., Error::InvalidEscapeSequence)),
                },
                Some((_, ch)) => {
                    self.scratch.push(ch);
                }
                None => {
                    return Err(self
                        .chars
                        .ranged(self.chars.last_index.., Error::MissingRegExEnd))
                }
            }
        }

        Ok(self.chars.ranged(
            start..,
            Token::RegEx(RegExLiteral {
                pattern: self.scratch.clone(),
                expanded,
            }),
        ))
    }
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

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum Error {
    UnexpectedChar(char),
    IntegerParse(String),
    FloatParse(String),
    MissingEndQuote,
    MissingRegExEnd,
    InvalidEscapeSequence,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub struct RegExLiteral {
    pub pattern: String,
    pub expanded: bool,
}

#[test]
fn basics() {
    let tokens = Tokens::new("a_09_ + 1 - .2")
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        tokens,
        &[
            Ranged::new(0..5, Token::Identifier(Symbol::from("a_09_"))),
            Ranged::new(5..6, Token::Whitespace),
            Ranged::new(6..7, Token::Char('+')),
            Ranged::new(7..8, Token::Whitespace),
            Ranged::new(8..9, Token::Int(1)),
            Ranged::new(9..10, Token::Whitespace),
            Ranged::new(10..11, Token::Char('-')),
            Ranged::new(11..12, Token::Whitespace),
            Ranged::new(12..14, Token::Float(0.2)),
        ]
    );
    let tokens = Tokens::new("a_09_ + 1 - .2")
        .excluding_whitespace()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        tokens,
        &[
            Ranged::new(0..5, Token::Identifier(Symbol::from("a_09_"))),
            Ranged::new(6..7, Token::Char('+')),
            Ranged::new(8..9, Token::Int(1)),
            Ranged::new(10..11, Token::Char('-')),
            Ranged::new(12..14, Token::Float(0.2)),
        ]
    );
}
