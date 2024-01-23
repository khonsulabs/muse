use std::cmp::Ordering;
use std::hash::Hash;
use std::iter::Peekable;
use std::ops::RangeBounds;
use std::str::CharIndices;

use super::Ranged;
use crate::symbol::Symbol;

#[derive(Clone, Debug)]
pub enum Token {
    Whitespace,
    Comment,
    Identifier(Symbol),
    Int(i64),
    Float(f64),
    Char(char),
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
            Token::Identifier(t) => t.hash(state),
            Token::Int(t) => t.hash(state),
            Token::Float(t) => t.to_bits().hash(state),
            Token::Char(t) => t.hash(state),
            Token::Open(t) | Token::Close(t) => t.hash(state),
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

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash)]
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

            let float = self
                .scratch
                .parse::<f64>()
                .map_err(|err| self.chars.ranged(start.., Error::FloatParse(err)))?;
            Ok(self.chars.ranged(start.., Token::Float(float)))
        } else {
            let int = self
                .scratch
                .parse::<i64>()
                .map_err(|err| self.chars.ranged(start.., Error::IntegerParse(err)))?;
            Ok(self.chars.ranged(start.., Token::Int(int)))
        }
    }

    fn tokenize_identifier(&mut self, start: usize, start_char: char) -> Ranged<Token> {
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

        // Allow a trailing exclamation of question mark.
        if let Some(ch) = self.chars.peek().filter(|ch| matches!(ch, '!' | '?')) {
            self.scratch.push(ch);
            self.chars.next();
        }

        let symbol = Symbol::from(&self.scratch);
        self.chars.ranged(start.., Token::Identifier(symbol))
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
            .map_or(false, |ch| matches!(ch, '\r' | '\n'))
        {
            self.chars.next();
        }
        self.chars.ranged(start.., Token::Comment)
    }
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
                    if self.include_comments {
                        Ok(self.tokenize_oneline_comment(start))
                    } else {
                        continue;
                    }
                }

                (start, '{') => Ok(self.chars.ranged(start.., Token::Open(Paired::Brace))),
                (start, '}') => Ok(self.chars.ranged(start.., Token::Close(Paired::Brace))),
                (start, '(') => Ok(self.chars.ranged(start.., Token::Open(Paired::Paren))),
                (start, ')') => Ok(self.chars.ranged(start.., Token::Close(Paired::Paren))),
                (start, '[') => Ok(self.chars.ranged(start.., Token::Open(Paired::Bracket))),
                (start, ']') => Ok(self.chars.ranged(start.., Token::Close(Paired::Bracket))),
                (start, ch) if ch.is_ascii_digit() => self.tokenize_number(start, ch),
                (start, '.' | '-') if self.chars.peek().map_or(false, |ch| ch.is_ascii_digit()) => {
                    self.tokenize_number(start, '.')
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

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Error {
    UnexpectedChar(char),
    IntegerParse(std::num::ParseIntError),
    FloatParse(std::num::ParseFloatError),
}

#[test]
fn basics() {
    let tokens = Tokens::new("a_09? + 1 - .2")
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        tokens,
        &[
            Ranged::new(0..5, Token::Identifier(Symbol::from("a_09?"))),
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
    let tokens = Tokens::new("a_09? + 1 - .2")
        .excluding_whitespace()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        tokens,
        &[
            Ranged::new(0..5, Token::Identifier(Symbol::from("a_09?"))),
            Ranged::new(6..7, Token::Char('+')),
            Ranged::new(8..9, Token::Int(1)),
            Ranged::new(10..11, Token::Char('-')),
            Ranged::new(12..14, Token::Float(0.2)),
        ]
    );
}
