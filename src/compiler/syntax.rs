//! The Muse syntax parser.

use core::slice;
use std::collections::VecDeque;
use std::fmt::{self, Debug, Display};
use std::num::NonZeroUsize;
use std::ops::{Bound, Deref, DerefMut, Range, RangeBounds, RangeInclusive};
use std::{option, vec};

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use self::token::{FormatString, FormatStringPart, Paired, RegexLiteral, Token, Tokens};
use crate::runtime::exception::Exception;
use crate::runtime::symbol::Symbol;
use crate::vm::{ExecutionError, VmContext};

pub mod token;

/// An identified chunk of Muse source code.
pub struct SourceCode<'a> {
    /// The Muse source code.
    pub code: &'a str,
    /// The ID of this source.
    pub id: SourceId,
}

impl<'a> SourceCode<'a> {
    /// Returns a new source.
    #[must_use]
    pub const fn new(code: &'a str, id: SourceId) -> Self {
        Self { code, id }
    }

    /// Returns a source using [`SourceId::anonymous()`] as the id.
    #[must_use]
    pub const fn anonymous(code: &'a str) -> Self {
        Self {
            code,
            id: SourceId::anonymous(),
        }
    }
}

impl<'a> From<&'a str> for SourceCode<'a> {
    fn from(value: &'a str) -> Self {
        Self::anonymous(value)
    }
}

impl<'a> From<&'a String> for SourceCode<'a> {
    fn from(value: &'a String) -> Self {
        Self::from(value.as_str())
    }
}

/// An entry in a [`Sources`] collection.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TrackedSource {
    /// The name of this source.
    pub name: String,
    /// The text of this source.
    pub source: String,
    line_lengths: Vec<usize>,
}

impl TrackedSource {
    fn new(name: impl Into<String>, source: impl Into<String>) -> Self {
        let name = name.into();
        let source = source.into();
        let line_lengths = source.split_inclusive('\n').map(str::len).collect();

        Self {
            name,
            source,
            line_lengths,
        }
    }

    /// Returns the line number and column of `offset` in this source.
    ///
    /// If `offset` is out of bounds, this function returns the last line and
    /// column relative to that line's start.
    #[must_use]
    pub fn offset_to_line(&self, mut offset: usize) -> (usize, usize) {
        let mut line_no = 1;
        for &line_len in &self.line_lengths {
            if offset < line_len {
                break;
            }
            offset -= line_len;
            line_no += 1;
        }
        (line_no, offset + 1)
    }
}

/// A collection of [`TrackedSource`].
///
/// This type is used to accumulate a collection of [`SourceCode`] allowing for
/// errors and exceptions to have their locations resolved to the underlying
/// source name, line number, and column offsets.
#[derive(Default)]
pub struct Sources(Vec<TrackedSource>);

impl Sources {
    /// Adds another `source` identified by `name`, returning a `SourceCode`
    /// that can be parsed/compiled.
    pub fn push(&mut self, name: impl Into<String>, source: impl Into<String>) -> SourceCode<'_> {
        let id = self.next_id();
        self.0.push(TrackedSource::new(name, source));
        SourceCode::new(&self.0.last().expect("just pushed").source, id)
    }

    /// Returns the [`TrackedSource`] for a given [`SourceId`], if found.
    #[must_use]
    pub fn get(&self, id: SourceId) -> Option<&TrackedSource> {
        let index = id.0?.get() - 1;
        self.0.get(index)
    }

    #[must_use]
    fn next_id(&self) -> SourceId {
        SourceId::new(NonZeroUsize::new(self.0.len() + 1).expect("always > 0"))
    }

    /// Formats `err` for display into `fmt`
    pub fn format_error(
        &self,
        err: impl Into<crate::Error>,
        context: &mut VmContext<'_, '_>,
        fmt: impl fmt::Write,
    ) -> fmt::Result {
        self.format_error_inner(err.into(), context, fmt)
    }

    fn format_error_inner(
        &self,
        err: crate::Error,
        context: &mut VmContext<'_, '_>,
        mut f: impl fmt::Write,
    ) -> fmt::Result {
        match err {
            crate::Error::Compilation(errors) => {
                for error in errors {
                    if let Some(source) = self.get(error.range().source_id) {
                        let (line_no, start) = source.offset_to_line(error.range().start);
                        write!(
                            f,
                            "compilation error in {}:{line_no}:{start}: ",
                            source.name
                        )?;
                    } else {
                        write!(f, "compilation error: ")?;
                    }
                    write!(f, "{}", error.0)?;
                }
                Ok(())
            }
            crate::Error::Execution(execution) => match execution {
                ExecutionError::NoBudget => f.write_str("execution budget exhausted"),
                ExecutionError::Waiting => f.write_str("blocked waiting for an external task"),
                ExecutionError::Timeout => f.write_str("execution timeout"),
                ExecutionError::Exception(value) => {
                    if let Some(exception) = value.as_rooted::<Exception>(context.guard()) {
                        exception.format(self, context, f)
                    } else {
                        value.format(context, f)
                    }
                }
            },
        }
    }
}

impl Deref for Sources {
    type Target = [TrackedSource];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A value and an associated source range.
#[derive(Default, Clone, Copy, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct Ranged<T>(pub T, pub SourceRange);

impl<T> Ranged<T> {
    /// Returns a new ranged value.
    pub fn new(range: impl Into<SourceRange>, value: T) -> Self {
        Self(value, range.into())
    }

    /// Returns an instance with [`SourceRange::default()`] used for the range.
    pub fn default_for(value: T) -> Self {
        Self::new(SourceRange::default(), value)
    }

    /// Returns an instance from the source id and bounds.
    pub fn bounded(
        source_id: SourceId,
        range: impl RangeBounds<usize>,
        unbounded_end: usize,
        value: T,
    ) -> Ranged<T> {
        let start = match range.start_bound() {
            Bound::Included(start) => *start,
            Bound::Excluded(start) => start + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(end) => end + 1,
            Bound::Excluded(end) => *end,
            Bound::Unbounded => unbounded_end,
        };
        Ranged(
            value,
            SourceRange {
                source_id,
                start,
                length: end.saturating_sub(start),
            },
        )
    }

    /// Returns a new instance with the result of invoking the closure. The
    /// range remains unchainged.
    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> Ranged<U> {
        Ranged(map(self.0), self.1)
    }
}

impl<T> Ranged<T> {
    /// Returns the range of this value.
    pub const fn range(&self) -> SourceRange {
        self.1
    }
}

impl<T> Deref for Ranged<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Ranged<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// The ID of a [`SourceCode`].
#[derive(Default, Clone, Copy, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct SourceId(Option<NonZeroUsize>);

impl SourceId {
    /// Returns a [`SourceId`] that represents an anonymous location.
    #[must_use]
    pub const fn anonymous() -> Self {
        Self(None)
    }

    /// Returns a new id.
    #[must_use]
    pub const fn new(id: NonZeroUsize) -> Self {
        Self(Some(id))
    }

    /// Returns the inner value of this id.
    #[must_use]
    pub const fn get(&self) -> Option<NonZeroUsize> {
        self.0
    }
}

/// A range within a [`SourceCode`].
#[derive(Default, Clone, Copy, Eq, PartialEq, Debug, Serialize, Deserialize, Hash)]
pub struct SourceRange {
    /// The id of the [`SourceCode`] this range belongs to.
    #[serde(default)]
    pub source_id: SourceId,
    /// The start offset of the range.
    pub start: usize,
    /// The length of the range.
    pub length: usize,
}

impl SourceRange {
    /// Returns the end offset of this range.
    #[must_use]
    pub const fn end(&self) -> usize {
        self.start + self.length
    }

    /// Returns a new range with an updated length.
    #[must_use]
    pub fn with_length(mut self, length: usize) -> Self {
        self.length = length;
        self
    }

    /// Returns a new range with a length updated from the end offset.
    #[must_use]
    pub fn with_end(mut self, end: usize) -> Self {
        self.length = end.saturating_sub(self.start);
        self
    }
}

impl From<(SourceId, Range<usize>)> for SourceRange {
    fn from((source_id, range): (SourceId, Range<usize>)) -> Self {
        Self {
            source_id,
            start: range.start,
            length: range.end - range.start,
        }
    }
}

impl From<(SourceId, RangeInclusive<usize>)> for SourceRange {
    fn from((source_id, range): (SourceId, RangeInclusive<usize>)) -> Self {
        Self {
            source_id,
            start: *range.start(),
            length: range.end() - range.start(),
        }
    }
}

/// A sequence of `T`, delimited by `Delimiter`.
#[derive(Debug, Clone, PartialEq)]
pub struct Delimited<T, Delimiter = Ranged<Token>> {
    /// The first entry in this sequence.
    pub first: Option<T>,
    /// The remaining delimiter and entries.
    pub remaining: Vec<(Delimiter, T)>,
}

impl<T, Delimiter> Delimited<T, Delimiter> {
    /// Returns a list from a single value.
    #[must_use]
    pub const fn single(value: T) -> Self {
        Self {
            first: Some(value),
            remaining: Vec::new(),
        }
    }

    /// Returns an empty list.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            first: None,
            remaining: Vec::new(),
        }
    }

    /// Returns a builder with the first element provided.
    #[must_use]
    pub const fn build(first: T) -> DelimitedBuilder<T, Delimiter> {
        DelimitedBuilder::new(first)
    }

    /// Returns an empty builder.
    #[must_use]
    pub const fn build_empty() -> DelimitedBuilder<T, Delimiter> {
        DelimitedBuilder::empty()
    }

    /// Returns the number of elements contained in this collection.
    pub fn len(&self) -> usize {
        self.remaining.len() + usize::from(self.first.is_some())
    }

    /// Returns true if this collection is empty.
    pub const fn is_empty(&self) -> bool {
        self.first.is_none()
    }

    /// Returns an iterator over the values in this list.
    pub fn iter(&self) -> DelimitedIter<'_, T, Delimiter> {
        self.into_iter()
    }

    /// Returns an iterator over exclusive references to values in this list.
    pub fn iter_mut(&mut self) -> DelimitedIterMut<'_, T, Delimiter> {
        self.into_iter()
    }

    /// Returns the last value in this list.
    pub fn last(&self) -> Option<&'_ T> {
        self.remaining
            .last()
            .map(|(_, t)| t)
            .or(self.first.as_ref())
    }
}

impl<T> std::ops::Index<usize> for Delimited<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if let Some(index_in_remaining) = index.checked_sub(1) {
            &self.remaining[index_in_remaining].1
        } else {
            self.first.as_ref().expect("out of bounds")
        }
    }
}

impl<T> Default for Delimited<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T> TokenizeInto for Delimited<T>
where
    T: TokenizeInto,
{
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        if let Some(first) = &self.first {
            first.tokenize_into(tokens);
        }
        for (delimiter, value) in &self.remaining {
            tokens.push_back(delimiter.clone());
            value.tokenize_into(tokens);
        }
    }
}

impl<'a, T, Delimiter> IntoIterator for &'a Delimited<T, Delimiter> {
    type IntoIter = DelimitedIter<'a, T, Delimiter>;
    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter {
        DelimitedIter {
            first: self.first.iter(),
            remaining: self.remaining.iter(),
        }
    }
}

/// An iterator over a [`Delimited`] list's values.
pub struct DelimitedIter<'a, T, Delimiter = Ranged<Token>> {
    first: option::Iter<'a, T>,
    remaining: slice::Iter<'a, (Delimiter, T)>,
}

impl<'a, T, Delimiter> Iterator for DelimitedIter<'a, T, Delimiter> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.first
            .next()
            .or_else(|| self.remaining.next().map(|(_, value)| value))
    }
}

impl<'a, T, Delimiter> IntoIterator for &'a mut Delimited<T, Delimiter> {
    type IntoIter = DelimitedIterMut<'a, T, Delimiter>;
    type Item = &'a mut T;

    fn into_iter(self) -> Self::IntoIter {
        DelimitedIterMut {
            first: self.first.iter_mut(),
            remaining: self.remaining.iter_mut(),
        }
    }
}

/// An iterator over exclusive references to a [`Delimited`] list's values.
pub struct DelimitedIterMut<'a, T, Delimiter = Ranged<Token>> {
    first: option::IterMut<'a, T>,
    remaining: slice::IterMut<'a, (Delimiter, T)>,
}

impl<'a, T, Delimiter> Iterator for DelimitedIterMut<'a, T, Delimiter> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.first
            .next()
            .or_else(|| self.remaining.next().map(|(_, value)| value))
    }
}

/// A builder that produces a [`Delimited`] list.
pub struct DelimitedBuilder<T, Delimiter = Ranged<Token>> {
    delimited: Delimited<T, Delimiter>,
    pending_delimiter: Option<Delimiter>,
}

impl<T, Delimiter> DelimitedBuilder<T, Delimiter> {
    /// Returns a builder that contains an initial element.
    #[must_use]
    pub const fn new(first: T) -> Self {
        Self {
            delimited: Delimited::single(first),
            pending_delimiter: None,
        }
    }

    /// Returns an empt builder.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            delimited: Delimited::empty(),
            pending_delimiter: None,
        }
    }

    /// Returns the built collection.
    #[must_use]
    pub fn finish(self) -> Delimited<T, Delimiter> {
        self.delimited
    }

    /// Sets the delimiter to place between the next element and the current
    /// element.
    ///
    /// # Panics
    ///
    /// This function must only be called once between invocations to `push`.
    pub fn set_delimiter(&mut self, delimiter: Delimiter) {
        assert!(self.pending_delimiter.replace(delimiter).is_none());
    }

    /// Pushes a new value to this list.
    ///
    /// # Panics
    ///
    /// This function panics if a delimiter is expected and
    /// [`set_delimiter`](Self::set_delimiter) was not invoked before pushing a
    /// value.
    pub fn push(&mut self, value: T) {
        if self.delimited.first.is_none() {
            assert!(self.pending_delimiter.is_none());
            self.delimited.first = Some(value);
        } else {
            let delimiter = self.pending_delimiter.take().expect("missing delimiter");
            self.delimited.remaining.push((delimiter, value));
        }
    }
}

/// Converts a value into [`Token`]s.
pub trait TokenizeInto {
    /// Tokenize `self` into `tokens`.
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>);

    /// Returns a list of tokens that could be re-parsed to produce this value.
    fn to_tokens(&self) -> VecDeque<Ranged<Token>> {
        let mut tokens = VecDeque::new();
        self.tokenize_into(&mut tokens);
        tokens
    }
}

impl<T> TokenizeInto for Ranged<T>
where
    T: TokenizeRanged,
{
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.0.tokenize_ranged(self.range(), tokens);
    }
}

/// Converts a value into a series of [`Token`]s with the provided enclosing
/// range.
pub trait TokenizeRanged {
    /// Tokenize `self` into `tokens` within the enclosing `range`.
    fn tokenize_ranged(&self, range: SourceRange, tokens: &mut VecDeque<Ranged<Token>>);
}

impl<T> TokenizeRanged for T
where
    T: TokenizeInto,
{
    fn tokenize_ranged(&self, _range: SourceRange, tokens: &mut VecDeque<Ranged<Token>>) {
        self.tokenize_into(tokens);
    }
}

impl TokenizeRanged for Symbol {
    fn tokenize_ranged(&self, range: SourceRange, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(Ranged::new(range, Token::Symbol(self.clone())));
    }
}

/// A Muse expression.
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// A reference to the root module (`$`).
    RootModule,
    /// A literal value.
    Literal(Literal),
    /// A lookup of a field by a name (`a.b`).
    Lookup(Box<Lookup>),
    /// Conditional expression evaluation.
    If(Box<IfExpression>),
    /// A pattern match expression.
    Match(Box<MatchExpression>),
    /// A try/catch expression.
    Try(Box<TryExpression>),
    /// A try-or-nil expression (`?`).
    TryOrNil(Box<TryOrNil>),
    /// A throw expression.
    Throw(Box<ThrowExpression>),
    /// A map literal.
    Map(Box<MapExpression>),
    /// A list literal.
    List(Box<ListExpression>),
    /// A function invocation (`a(b)`).
    Call(Box<FunctionCall>),
    /// An indexing expression (`a[b]`)
    Index(Box<Index>),
    /// An assignment expression (`a = b`)
    Assign(Box<Assignment>),
    /// An expression with a single expression argument.
    Unary(Box<UnaryExpression>),
    /// An expression with two expression arguments.
    Binary(Box<BinaryExpression>),
    /// A scoped, sequence of expressions.
    Block(Box<Block>),
    /// A loop expression.
    Loop(Box<LoopExpression>),
    /// A break control flow expression.
    Break(Box<BreakExpression>),
    /// A continue control flow expression.
    Continue(Box<ContinueExpression>),
    /// A return control flow expression.
    Return(Box<ReturnExpression>),
    /// A module declaration.
    Module(Box<ModuleDefinition>),
    /// A function declaration.
    Function(Box<FunctionDefinition>),
    /// A structure definition.
    Structure(Box<StructureDefinition>),
    /// A structure literal.
    StructureLiteral(Box<NewStruct>),
    /// A variable declaration.
    SingleMatch(Box<SingleMatch>),
    /// A macro invocation.
    Macro(Box<MacroInvocation>),
    /// An infix macro invocation.
    InfixMacro(Box<InfixMacroInvocation>),
    /// A grouped expression (`(a)`).
    Group(Box<Enclosed<Ranged<Expression>>>),
    /// A format string expression (`f"hello {name}"`).
    FormatString(Box<Delimited<Ranged<Symbol>, Ranged<Expression>>>),
}

impl Expression {
    /// Converts a list of expressions into a chain expression, delimited by
    /// `semicolons`.
    ///
    /// If `semicolons` does not contain enough delimiters, new ones will be
    /// manufactured with a range that sits between the end of the previous
    /// expression and the start of the next.
    #[must_use]
    pub fn chain(
        mut expressions: Vec<Ranged<Expression>>,
        mut semicolons: Vec<Ranged<Token>>,
    ) -> Ranged<Self> {
        let Some(mut expression) = expressions.pop() else {
            return Ranged::new(
                (SourceId::anonymous(), 0..0),
                Expression::Literal(Literal::Nil),
            );
        };

        while let Some(previous) = expressions.pop() {
            let operator = semicolons.pop().unwrap_or_else(|| {
                Ranged::new(
                    SourceRange {
                        start: previous.range().end(),
                        length: 0,
                        source_id: previous.range().source_id,
                    },
                    Token::Char(';'),
                )
            });
            expression = Ranged::new(
                previous.range().with_end(expression.range().end()),
                Expression::Binary(Box::new(BinaryExpression {
                    kind: BinaryKind::Chain,
                    left: previous,
                    operator,
                    right: expression,
                })),
            );
        }

        expression
    }
}

impl Default for Expression {
    fn default() -> Self {
        Self::Literal(Literal::Nil)
    }
}

impl TokenizeRanged for Expression {
    fn tokenize_ranged(&self, range: SourceRange, tokens: &mut VecDeque<Ranged<Token>>) {
        match &self {
            Expression::RootModule => {
                tokens.push_back(Ranged::new(
                    range,
                    Token::Sigil(Symbol::sigil_symbol().clone()),
                ));
            }
            Expression::Literal(it) => it.tokenize_ranged(range, tokens),
            Expression::Lookup(it) => it.tokenize_into(tokens),
            Expression::If(it) => it.tokenize_into(tokens),
            Expression::Match(it) => it.tokenize_into(tokens),
            Expression::Try(it) => it.tokenize_into(tokens),
            Expression::TryOrNil(it) => it.tokenize_into(tokens),
            Expression::Throw(it) => it.tokenize_into(tokens),
            Expression::Map(it) => it.tokenize_into(tokens),
            Expression::List(it) => it.tokenize_into(tokens),
            Expression::Call(it) => it.tokenize_into(tokens),
            Expression::Index(it) => it.tokenize_into(tokens),
            Expression::Assign(it) => it.tokenize_into(tokens),
            Expression::Unary(it) => it.tokenize_into(tokens),
            Expression::Binary(it) => it.tokenize_into(tokens),
            Expression::Block(it) => it.tokenize_into(tokens),
            Expression::Loop(it) => it.tokenize_into(tokens),
            Expression::Break(it) => it.tokenize_into(tokens),
            Expression::Continue(it) => it.tokenize_into(tokens),
            Expression::Return(it) => it.tokenize_into(tokens),
            Expression::Module(it) => it.tokenize_into(tokens),
            Expression::Structure(it) => it.tokenize_into(tokens),
            Expression::StructureLiteral(it) => it.tokenize_into(tokens),
            Expression::Function(it) => it.tokenize_into(tokens),
            Expression::SingleMatch(it) => it.tokenize_into(tokens),
            Expression::Macro(it) => it.tokenize_into(tokens),
            Expression::InfixMacro(it) => it.tokenize_into(tokens),
            Expression::Group(e) => {
                e.tokenize_into(tokens);
            }
            Expression::FormatString(parts) => tokens.push_back(Ranged::new(
                range,
                Token::FormatString(FormatString {
                    initial: parts.first.clone().expect("missing initial format string"),
                    parts: parts
                        .remaining
                        .iter()
                        .map(|(expr, part)| FormatStringPart {
                            expression: Vec::from(expr.to_tokens()),
                            suffix: part.clone(),
                        })
                        .collect(),
                }),
            )),
        }
    }
}

/// A literal value.
#[derive(Default, Debug, Clone, PartialEq)]
pub enum Literal {
    /// The literal `nil`.
    #[default]
    Nil,
    /// A boolean literal.
    Bool(bool),
    /// A signed 64-bit integer literal.
    Int(i64),
    /// An unsigned 64-bit integer literal.
    UInt(u64),
    /// A double-precision floating point number.
    Float(f64),
    /// A string literal.
    String(Symbol),
    /// A symbol literal (`:foo`).
    Symbol(Symbol),
    /// A regular expression literal.
    Regex(RegexLiteral),
}

impl TokenizeRanged for Literal {
    fn tokenize_ranged(&self, range: SourceRange, tokens: &mut VecDeque<Ranged<Token>>) {
        let token = match self {
            Literal::Nil => Token::Identifier(Symbol::nil_symbol().clone()),
            Literal::Bool(false) => Token::Identifier(Symbol::false_symbol().clone()),
            Literal::Bool(true) => Token::Identifier(Symbol::true_symbol().clone()),
            Literal::Int(value) => Token::Int(*value),
            Literal::UInt(value) => Token::UInt(*value),
            Literal::Float(float) => Token::Float(*float),
            Literal::String(string) => Token::String(string.clone()),
            Literal::Symbol(symbol) => Token::Symbol(symbol.clone()),
            Literal::Regex(regex) => Token::Regex(regex.clone()),
        };
        tokens.push_back(Ranged::new(range, token));
    }
}

/// The syntax components of the label part of a labeled expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Label {
    /// The name of the label being defined.
    pub name: Ranged<Symbol>,
    /// The colon after the name.
    pub colon: Ranged<Token>,
}

impl TokenizeInto for Label {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.name.clone().map(Token::Label));
        tokens.push_back(self.colon.clone());
    }
}

/// The syntax components of a block.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    /// The label of the block.
    pub label: Option<Label>,
    /// The contents of the block.
    pub body: Enclosed<Ranged<Expression>>,
}

impl TokenizeInto for Block {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        if let Some(name) = &self.label {
            name.tokenize_into(tokens);
        }
        self.body.tokenize_into(tokens);
    }
}

/// The syntax components of a map literal.
#[derive(Debug, Clone, PartialEq)]
pub struct MapExpression {
    /// The map fields.
    pub fields: Enclosed<Delimited<MapField>>,
}

impl TokenizeInto for MapExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.fields.tokenize_into(tokens);
    }
}

/// The syntax components of a list literal.
#[derive(Debug, Clone, PartialEq)]
pub struct ListExpression {
    /// The values in the list.
    pub values: Enclosed<Delimited<Ranged<Expression>>>,
}

impl TokenizeInto for ListExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.values.tokenize_into(tokens);
    }
}

/// A new structure instantiation.
#[derive(Debug, Clone, PartialEq)]
pub struct NewStruct {
    /// The `new` token.
    pub new: Ranged<Token>,
    /// The name of the struct to instantiate, or the path to it.
    pub name: Ranged<Expression>,
    /// The fields to provide to the constructor.
    pub fields: Option<Enclosed<Delimited<NewStructField>>>,
}

impl TokenizeInto for NewStruct {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.new.clone());
        self.name.tokenize_into(tokens);
        if let Some(fields) = &self.fields {
            fields.tokenize_into(tokens);
        }
    }
}

/// A single field in a [`NewStruct`].
#[derive(Debug, Clone, PartialEq)]
pub struct NewStructField {
    /// The name of the field.
    pub name: Ranged<Symbol>,
    /// The `:` token
    pub colon: Ranged<Token>,
    /// The value of the field.
    pub value: Ranged<Expression>,
}

impl TokenizeInto for NewStructField {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.name.tokenize_into(tokens);

        tokens.push_back(self.colon.clone());
        self.value.tokenize_into(tokens);
    }
}

/// The syntax components of a field in a map literal.
#[derive(Debug, Clone, PartialEq)]
pub struct MapField {
    /// The key of this field.
    pub key: Ranged<Expression>,
    /// The colon separating the key and value.
    pub colon: Option<Ranged<Token>>,
    /// The value of this field.
    pub value: Ranged<Expression>,
}

impl TokenizeInto for MapField {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        if let Some(colon) = &self.colon {
            self.key.tokenize_into(tokens);
            tokens.push_back(colon.clone());
            self.value.tokenize_into(tokens);
        } else {
            // Set literal
            assert_eq!(&self.key, &self.value);
            self.key.tokenize_into(tokens);
        }
    }
}

/// The syntax components of a lookup expression (`foo.bar`).
#[derive(Debug, Clone, PartialEq)]
pub struct Lookup {
    /// The base of the lookup expression (`foo.`).
    pub base: Option<LookupBase>,
    /// The name of the value to look up.
    pub name: Ranged<Symbol>,
}

impl From<Ranged<Symbol>> for Lookup {
    fn from(name: Ranged<Symbol>) -> Self {
        Self { name, base: None }
    }
}

impl TokenizeInto for Lookup {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        if let Some(base) = &self.base {
            base.expression.tokenize_into(tokens);
            tokens.push_back(base.dot.clone());
        }

        tokens.push_back(self.name.clone().map(Token::Identifier));
    }
}

/// The syntax components of the base of a lookup expression.
#[derive(Debug, Clone, PartialEq)]
pub struct LookupBase {
    /// The expression before the dot.
    pub expression: Ranged<Expression>,
    /// The dot separating the base from the lookup.
    pub dot: Ranged<Token>,
}

/// The syntax components of an if expression.
#[derive(Debug, Clone, PartialEq)]
pub struct IfExpression {
    /// The if keyword.
    pub r#if: Ranged<Token>,
    /// The condition being checked.
    pub condition: Ranged<Expression>,
    /// If a `then` keyword is provided instead of `when_true` being a block,
    /// this contains it.
    pub then: Option<Ranged<Token>>,
    /// The expression to evaluate if `condition` is truthy.
    pub when_true: Ranged<Expression>,
    /// The expression to evaluate if `condition` is falsey.
    pub when_false: Option<Else>,
}

impl TokenizeInto for IfExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.r#if.clone());
        self.condition.tokenize_into(tokens);
        if let Some(then) = &self.then {
            tokens.push_back(then.clone());
        }
        self.when_true.tokenize_into(tokens);
        if let Some(when_false) = &self.when_false {
            tokens.push_back(when_false.r#else.clone());
            when_false.expression.tokenize_into(tokens);
        }
    }
}

/// The syntax components of the else part of an if expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Else {
    /// The `else` keyword.
    pub r#else: Ranged<Token>,
    /// The expression to evaluate.
    pub expression: Ranged<Expression>,
}

/// The syntax components of a match statement.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchExpression {
    /// The match keyword.
    pub r#match: Ranged<Token>,
    /// The condition to match against.
    pub condition: Ranged<Expression>,
    /// The list of matches to match against.
    pub matches: Ranged<Matches>,
}

impl TokenizeInto for MatchExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.r#match.clone());
        self.condition.tokenize_into(tokens);
        self.matches.tokenize_into(tokens);
    }
}

/// The syntax components of a try-or-nil expression (`foo?`).
#[derive(Debug, Clone, PartialEq)]
pub struct TryOrNil {
    /// The question mark token.
    pub token: Ranged<Token>,
    /// The expression to try.
    pub body: Ranged<Expression>,
}

impl TokenizeInto for TryOrNil {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.body.tokenize_into(tokens);
        tokens.push_back(self.token.clone());
    }
}

/// The syntax components of a return control flow expression.
#[derive(Debug, Clone, PartialEq)]
pub struct ReturnExpression {
    /// The return keyword.
    pub r#return: Ranged<Token>,
    /// The value to return.
    pub value: Ranged<Expression>,
}

impl TokenizeInto for ReturnExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.r#return.clone());
        self.value.tokenize_into(tokens);
    }
}

/// The syntax components of a throw control flow expression.
#[derive(Debug, Clone, PartialEq)]
pub struct ThrowExpression {
    /// The throw keyword.
    pub throw: Ranged<Token>,
    /// The value to throw.
    pub value: Ranged<Expression>,
}

impl TokenizeInto for ThrowExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.throw.clone());
        self.value.tokenize_into(tokens);
    }
}
/// The syntax components of a try block.
#[derive(Debug, Clone, PartialEq)]
pub struct TryExpression {
    /// The try keyword.
    pub r#try: Ranged<Token>,
    /// The body to try evaluating.
    pub body: Ranged<Expression>,
    /// The catch expression.
    pub catch: Option<Catch>,
}

impl TokenizeInto for TryExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.r#try.clone());
        self.body.tokenize_into(tokens);
        if let Some(catch) = &self.catch {
            tokens.push_back(catch.catch.clone());
            catch.matches.tokenize_into(tokens);
        }
    }
}

/// The syntax components of the catch portion of a try expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Catch {
    /// The catch keyword.
    pub catch: Ranged<Token>,
    /// The pattern matches to apply to the caught exception.
    pub matches: Ranged<Matches>,
}

/// The syntax components of a loop expression.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopExpression {
    /// The keyword of this loop.
    pub token: Ranged<Token>,
    /// The kind of loop.
    pub kind: LoopKind,
    /// The body to repeat executing.
    pub block: Ranged<Block>,
}

impl TokenizeInto for LoopExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        if let Some(label) = &self.block.0.label {
            tokens.push_back(label.name.clone().map(Token::Label));
            tokens.push_back(label.colon.clone());
        }
        tokens.push_back(self.token.clone());

        match &self.kind {
            LoopKind::Infinite | LoopKind::TailWhile { .. } => {}
            LoopKind::While(condition) => {
                condition.tokenize_into(tokens);
            }
            LoopKind::For {
                pattern,
                r#in,
                source,
            } => {
                pattern.tokenize_into(tokens);
                tokens.push_back(r#in.clone());
                source.tokenize_into(tokens);
            }
        }

        self.block.body.tokenize_into(tokens);

        if let LoopKind::TailWhile {
            r#while,
            expression,
        } = &self.kind
        {
            tokens.push_back(r#while.clone());
            expression.tokenize_into(tokens);
        }
    }
}

/// The syntax components of a break control flow expression.
#[derive(Debug, Clone, PartialEq)]
pub struct BreakExpression {
    /// The break keyword.
    pub r#break: Ranged<Token>,
    /// The name of the label to break from.
    pub name: Option<Ranged<Symbol>>,
    /// The value to break with.
    pub value: Ranged<Expression>,
}

impl TokenizeInto for BreakExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.r#break.clone());
        if let Some(name) = &self.name {
            tokens.push_back(name.clone().map(Token::Label));
        }
        self.value.tokenize_into(tokens);
    }
}

/// The syntax components of a continue expression.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinueExpression {
    /// The continue keyword.
    pub r#continue: Ranged<Token>,
    /// The label of the loop to continue.
    pub name: Option<Ranged<Symbol>>,
}

impl TokenizeInto for ContinueExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.r#continue.clone());
        if let Some(name) = &self.name {
            tokens.push_back(name.clone().map(Token::Label));
        }
    }
}

/// The kind of a loop expression.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopKind {
    /// A loop that repeats forever with no conditions.
    Infinite,
    /// A loop that checks the expression each time before the body executes.
    While(Ranged<Expression>),
    /// A loop that checks the expression each time after the body executes.
    TailWhile {
        /// The while keyword.
        r#while: Ranged<Token>,
        /// The expression to check.
        expression: Ranged<Expression>,
    },
    /// A loop over an iterable or indexable.
    For {
        /// The pattern to match each iteration against.
        pattern: Ranged<Pattern>,
        /// The in keyword.
        r#in: Ranged<Token>,
        /// The source of items to iterate.
        source: Ranged<Expression>,
    },
}

/// The syntax components of a module definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleDefinition {
    /// The pub keyword.
    pub publish: Option<Ranged<Token>>,
    /// The mod keyword.
    pub r#mod: Ranged<Token>,
    /// The name of this module.
    pub name: Ranged<Symbol>,
    /// The contents of the module.
    pub contents: Ranged<Block>,
}

impl TokenizeInto for ModuleDefinition {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        if let Some(publish) = &self.publish {
            tokens.push_back(publish.clone());
        }

        tokens.push_back(self.r#mod.clone());
        tokens.push_back(self.name.clone().map(Token::Identifier));
        self.contents.tokenize_into(tokens);
    }
}

/// The syntax components of a function definition.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDefinition {
    /// The pub keyword.
    pub visibility: Option<Ranged<Symbol>>,
    /// The fn keyword.
    pub r#fn: Ranged<Token>,
    /// The name of the function.
    pub name: Option<Ranged<Symbol>>,
    /// The body of the function.
    pub body: Ranged<Matches>,
}

impl TokenizeInto for FunctionDefinition {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        if let Some(publish) = &self.visibility {
            tokens.push_back(publish.clone().map(Token::Symbol));
        }
        tokens.push_back(self.r#fn.clone());
        if let Some(name) = &self.name {
            tokens.push_back(name.clone().map(Token::Identifier));
        }
        self.body.tokenize_into(tokens);
    }
}

/// A custom structure type definition.
#[derive(Debug, Clone, PartialEq)]
pub struct StructureDefinition {
    /// The visibility keyword, if specified.
    pub visibility: Option<Ranged<Symbol>>,
    /// The struct keyword.
    pub r#struct: Ranged<Token>,
    /// The name of the structure.
    pub name: Ranged<Symbol>,
    /// The members of the structure, if present.
    pub members: Option<Enclosed<Delimited<Ranged<StructureMember>>>>,
}

impl TokenizeInto for StructureDefinition {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        if let Some(visibility) = &self.visibility {
            tokens.push_back(visibility.clone().map(Token::Symbol));
        }

        tokens.push_back(self.r#struct.clone());
        tokens.push_back(self.name.clone().map(Token::Symbol));
        if let Some(members) = &self.members {
            members.tokenize_into(tokens);
        }
    }
}

/// A member of ta [`StructureDefinition`].
#[derive(Debug, Clone, PartialEq)]
pub enum StructureMember {
    /// A member field.
    Field {
        /// The visibility of this member.
        visibility: Option<Ranged<Symbol>>,
        /// The name of this field.
        name: Ranged<Symbol>,
    },
    /// A function definition.
    Function(FunctionDefinition),
}

impl TokenizeInto for StructureMember {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        match self {
            StructureMember::Field { visibility, name } => {
                if let Some(visibility) = visibility {
                    tokens.push_back(visibility.clone().map(Token::Symbol));
                }

                tokens.push_back(name.clone().map(Token::Symbol));
            }
            StructureMember::Function(func) => func.tokenize_into(tokens),
        }
    }
}

/// The syntax components of a function call.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    /// The expression being invoked.
    pub function: Ranged<Expression>,
    /// The parameters being passed to the function.
    pub parameters: Enclosed<Delimited<Ranged<Expression>>>,
}

impl TokenizeInto for FunctionCall {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.function.tokenize_into(tokens);
        self.parameters.tokenize_into(tokens);
    }
}

/// The syntax components of an index expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Index {
    /// The target being indexed.
    pub target: Ranged<Expression>,
    /// The list of index arguments.
    pub parameters: Enclosed<Delimited<Ranged<Expression>>>,
}

impl TokenizeInto for Index {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.target.tokenize_into(tokens);
        self.parameters.tokenize_into(tokens);
    }
}

/// The syntax components of an assignment expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    /// The target of the assignment.
    pub target: Ranged<AssignTarget>,
    /// The `=` token.
    pub eq: Ranged<Token>,
    /// The value to assign.
    pub value: Ranged<Expression>,
}

impl TokenizeInto for Assignment {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        match &self.target.0 {
            AssignTarget::Lookup(lookup) => lookup.tokenize_into(tokens),
            AssignTarget::Index(index) => index.tokenize_into(tokens),
        }
    }
}

/// The target of an assignment expression.
#[derive(Debug, Clone, PartialEq)]
pub enum AssignTarget {
    /// Assign to the field described by a lookup expression.
    Lookup(Lookup),
    /// Assign into an indexed expression.
    Index(Index),
}

/// The syntax components of a macro invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct MacroInvocation {
    /// The name of the macro.
    pub name: Ranged<Symbol>,
    /// The tokens, surrounding [`Paired`] included.
    pub tokens: VecDeque<Ranged<Token>>,
}

impl TokenizeInto for MacroInvocation {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.name.clone().map(Token::Sigil));
        tokens.extend(self.tokens.iter().cloned());
    }
}

/// The syntax components of an infix macro expression.
#[derive(Debug, Clone, PartialEq)]
pub struct InfixMacroInvocation {
    /// The expression preceding the macro name.
    pub subject: Ranged<Expression>,
    /// The remainder of the macro.
    pub invocation: MacroInvocation,
}

impl TokenizeInto for InfixMacroInvocation {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.subject.tokenize_into(tokens);
        self.invocation.tokenize_into(tokens);
    }
}

/// The syntax components of a single-argument expression.
#[derive(Debug, Clone, PartialEq)]
pub struct UnaryExpression {
    /// The kind of this expression.
    pub kind: UnaryKind,
    /// The operator token.
    pub operator: Ranged<Token>,
    /// The operand.
    pub operand: Ranged<Expression>,
}

impl TokenizeInto for UnaryExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.operator.clone());
        self.operand.tokenize_into(tokens);
    }
}

/// The kind of a single-argument expression.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum UnaryKind {
    /// A logical not (`not true`)
    LogicalNot,
    /// A bitwise not (`!1`)
    BitwiseNot,
    /// A negate (`-1`)
    Negate,
}

/// The syntax components of a two-argument expression.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryExpression {
    /// The kind of this expression.
    pub kind: BinaryKind,
    /// The left hand side of the expression.
    pub left: Ranged<Expression>,
    /// The operator token.
    pub operator: Ranged<Token>,
    /// The right hand side of the expression.
    pub right: Ranged<Expression>,
}

impl TokenizeInto for BinaryExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.left.tokenize_into(tokens);
        tokens.push_back(self.operator.clone());
        self.right.tokenize_into(tokens);
    }
}

/// The kind of a binary expression.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BinaryKind {
    /// Add two values
    Add,
    /// Subtract two values
    Subtract,
    /// Multiply two values
    Multiply,
    /// Divide two values
    Divide,
    /// Divide two values treating as integers
    IntegerDivide,
    /// The remainder of two values treating as integers
    Remainder,
    /// Raise one value to another value
    Power,
    /// Execute the first expression, then the other.
    Chain,
    /// If the first expression not nil, return it. Otherwise, return the
    /// second.
    NilCoalesce,
    /// A binary bitwise kind.
    Bitwise(BitwiseKind),
    /// A binary logical kind.
    Logical(LogicalKind),
    /// A comparison.
    Compare(CompareKind),
}

/// Comparison expression kinds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CompareKind {
    /// `a <= b`
    LessThanOrEqual,
    /// `a < b`
    LessThan,
    /// `a == b`
    Equal,
    /// `a != b`
    NotEqual,
    /// `a > b`
    GreaterThan,
    /// `a >= b`
    GreaterThanOrEqual,
}

/// Binary bitwise expression kinds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BitwiseKind {
    /// `a & b`
    And,
    /// `a | b`
    Or,
    /// `a ^ b`
    Xor,
    /// `a << b`
    ShiftLeft,
    /// `a >> b`
    ShiftRight,
}

/// Binary logical expression kinds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LogicalKind {
    /// `a and b`
    And,
    /// `a or b`
    Or,
    /// `a xor b`
    Xor,
}

/// The syntax components of a single match expression.
#[derive(Debug, Clone, PartialEq)]
pub struct SingleMatch {
    /// The pub keyword.
    pub publish: Option<Ranged<Token>>,
    /// The kind of variables to declare (`let`/`var`) from the pattern.
    pub kind: Ranged<Symbol>,
    /// The pattern match to evaluate.
    pub pattern: Ranged<Pattern>,
    /// The `=` token.
    pub eq: Option<Ranged<Token>>,
    /// The value to pattern match against.
    pub value: Ranged<Expression>,
    /// The else expression to evaluate if the pattern match fails.
    pub r#else: Option<Else>,
}

impl TokenizeInto for SingleMatch {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        if let Some(publish) = &self.publish {
            tokens.push_back(publish.clone());
        }
        tokens.push_back(self.kind.clone().map(Token::Identifier));
        self.pattern.tokenize_into(tokens);
        if let Some(eq) = &self.eq {
            tokens.push_back(eq.clone());
            self.value.tokenize_into(tokens);
        }
        if let Some(r#else) = &self.r#else {
            tokens.push_back(r#else.r#else.clone());
            r#else.expression.tokenize_into(tokens);
        }
    }
}

/// Parses a series of tokens into an expression.
pub fn parse_tokens(
    source: VecDeque<Ranged<Token>>,
) -> Result<Ranged<Expression>, Ranged<ParseError>> {
    parse_from_reader(TokenReader::from(source))
}

/// Parses source code into an expression.
pub fn parse<'a>(
    source: impl Into<SourceCode<'a>>,
) -> Result<Ranged<Expression>, Ranged<ParseError>> {
    parse_from_reader(TokenReader::new(source))
}

fn parse_from_reader(
    mut tokens: TokenReader<'_>,
) -> Result<Ranged<Expression>, Ranged<ParseError>> {
    let parselets = parselets();
    let config = ParserConfig {
        parselets: &parselets,
        minimum_precedence: 0,
    };
    let mut results = Vec::new();
    let mut semicolons = Vec::new();
    loop {
        if tokens.peek().is_none() {
            // Peeking an error returns None
            match tokens.next_or_eof() {
                Ok(_) | Err(Ranged(ParseError::UnexpectedEof, _)) => {
                    results.push(tokens.ranged(
                        tokens.last_index..tokens.last_index,
                        Expression::Literal(Literal::Nil),
                    ));
                    break;
                }
                Err(err) => return Err(err),
            }
        }

        results.push(config.parse(&mut tokens)?);
        match tokens.next_or_eof() {
            Ok(token) if token.0 == Token::Char(';') => {
                semicolons.push(token);
            }
            Ok(token) => {
                return Err(token.map(|_| ParseError::ExpectedEof));
            }
            Err(Ranged(ParseError::UnexpectedEof, _)) => break,
            Err(other) => return Err(other),
        }
    }

    Ok(Expression::chain(results, semicolons))
}

enum TokenStream<'a> {
    List {
        tokens: VecDeque<Ranged<Token>>,
        range: SourceRange,
    },
    Tokens(Tokens<'a>),
}

impl TokenStream<'_> {
    fn source(&self) -> SourceId {
        match self {
            TokenStream::List { range, .. } => range.source_id,
            TokenStream::Tokens(tokens) => tokens.source(),
        }
    }
}

impl Iterator for TokenStream<'_> {
    type Item = Result<Ranged<Token>, Ranged<token::LexerError>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            TokenStream::List { tokens, .. } => tokens.pop_front().map(Ok),
            TokenStream::Tokens(tokens) => tokens.next(),
        }
    }
}

struct TokenReader<'a> {
    tokens: TokenStream<'a>,
    peeked: VecDeque<Result<Ranged<Token>, Ranged<ParseError>>>,
    last_index: usize,
}

impl<'a> TokenReader<'a> {
    pub fn new(source: impl Into<SourceCode<'a>>) -> Self {
        Self {
            tokens: TokenStream::Tokens(
                Tokens::new(source.into())
                    .excluding_comments()
                    .excluding_whitespace(),
            ),
            peeked: VecDeque::with_capacity(2),
            last_index: 0,
        }
    }

    fn peek_n(&mut self, index: usize) -> Option<Ranged<Token>> {
        while self.peeked.len() < index + 1 {
            self.peeked
                .push_back(self.tokens.next()?.map_err(Ranged::from));
        }

        self.peeked
            .get(index)
            .and_then(|result| result.as_ref().ok())
            .cloned()
    }

    fn peek(&mut self) -> Option<Ranged<Token>> {
        self.peek_n(0)
    }

    fn peek_token(&mut self) -> Option<Token> {
        self.peek_n(0).map(|t| t.0)
    }

    fn next_or_eof(&mut self) -> Result<Ranged<Token>, Ranged<ParseError>> {
        self.next(ParseError::UnexpectedEof)
    }

    fn next(&mut self, err: ParseError) -> Result<Ranged<Token>, Ranged<ParseError>> {
        let token = if let Some(peeked) = self.peeked.pop_front() {
            peeked?
        } else {
            self.tokens
                .next()
                .ok_or_else(|| self.ranged(self.last_index..self.last_index, err))??
        };
        // TODO this should only track if the source matches, which only will
        // differ when macros are involved
        self.last_index = token.1.start + token.1.length;
        Ok(token)
    }

    fn next_identifier(&mut self, err: ParseError) -> Result<Ranged<Symbol>, Ranged<ParseError>> {
        let token = self.next(err.clone())?;
        match token.0 {
            Token::Identifier(ident) => Ok(Ranged::new(token.1, ident)),
            _ => Err(Ranged::new(token.1, err)),
        }
    }

    fn ranged<T>(&self, range: impl RangeBounds<usize>, value: T) -> Ranged<T> {
        Ranged::bounded(self.tokens.source(), range, self.last_index, value)
    }
}

impl From<VecDeque<Ranged<Token>>> for TokenReader<'_> {
    fn from(tokens: VecDeque<Ranged<Token>>) -> Self {
        let mut iter = tokens.iter().map(|t| t.1);
        let range = iter
            .next()
            .map(|first| {
                let start = first.start;
                let end = first.end();
                let (start, end) = iter
                    .filter(|t| t.source_id == first.source_id)
                    .fold((start, end), |(start, end), range| {
                        (start.min(range.start), end.max(range.end()))
                    });
                SourceRange {
                    source_id: first.source_id,
                    start,
                    length: end - start,
                }
            })
            .unwrap_or_default();
        Self {
            tokens: TokenStream::List { tokens, range },
            peeked: VecDeque::new(),
            last_index: 0,
        }
    }
}

/// A syntax parsing error.
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum ParseError {
    /// An unexpected end of file.
    UnexpectedEof,
    /// Expected the end of a file.
    ExpectedEof,
    /// Missing the end of a paired token.
    MissingEnd(Paired),
    /// Missing the end of a paired token.
    MissingEndOr(Paired, char),
    /// An error tokenizing the source code.
    Token(token::LexerError),
    /// An unexpectecd token was encountered in the given context.
    UnexpectedToken,
    /// Expected a declaration.
    ExpectedDeclaration,
    /// Expected a name.
    ExpectedName,
    /// Expected a structure member.
    ExpectedStructureMember,
    /// Expected a block.
    ExpectedBlock,
    /// Expected a module body.
    ExpectedModuleBody,
    /// Expected a function body.
    ExpectedFunctionBody,
    /// Expected the `in` keyword.
    ExpectedIn,
    /// Expected function parameters.
    ExpectedFunctionParameters,
    /// Expected the `then` keyword or a block.
    ExpectedThenOrBrace,
    /// Expected a `:`
    ExpectedColon,
    /// Expected a `,` or the end `}`.
    ExpectedCommaOrBrace,
    /// Expected a match body.
    ExpectedMatchBody,
    /// Expected a pattern.
    ExpectedPattern,
    /// Expected a pattern or the end of the pattern match grouping.
    ExpectedPatternOr(Paired),
    /// Expected a `=>`.
    ExpectedFatArrow,
    /// Expected a catch block.
    ExpectedCatchBlock,
    /// An assignment can only be done to a lookup (`a.b = c`) or an index
    /// (`a[b] = c`).
    InvalidAssignmentTarget,
    /// The label specified can't be found or belongs to an invalid target for
    /// this expression.
    InvalidLabelTarget,
    /// Map keys must be literal values.
    InvalidMapKeyPattern,
}

impl crate::ErrorKind for ParseError {
    fn kind(&self) -> &'static str {
        match self {
            ParseError::UnexpectedEof => "unexpected eof",
            ParseError::ExpectedEof => "expected eof",
            ParseError::MissingEnd(_) | ParseError::MissingEndOr(..) => "missing end",

            ParseError::Token(err) => err.kind(),
            ParseError::UnexpectedToken => "unexpected token",
            ParseError::ExpectedDeclaration => "expected declaration",
            ParseError::ExpectedName => "expected name",
            ParseError::ExpectedBlock => "expected block",
            ParseError::ExpectedModuleBody => "expected module body",
            ParseError::ExpectedFunctionBody => "expected function body",
            ParseError::ExpectedStructureMember => "expected structure member",
            ParseError::ExpectedIn => "expected in",
            ParseError::ExpectedFunctionParameters => "expected function parameters",
            ParseError::ExpectedThenOrBrace => "expected then or brace",
            ParseError::ExpectedColon => "expected colon",
            ParseError::ExpectedCommaOrBrace => "expected comma or closing brace",
            ParseError::ExpectedMatchBody => "expected match body",
            ParseError::ExpectedPattern => "expected pattern",
            ParseError::ExpectedPatternOr(_) => "expected pattern or end",
            ParseError::ExpectedFatArrow => "expected fat arrow",
            ParseError::ExpectedCatchBlock => "expected catch block",
            ParseError::InvalidAssignmentTarget => "invalid assignment target",
            ParseError::InvalidLabelTarget => "invalid label target",
            ParseError::InvalidMapKeyPattern => "invalid map key pattern",
        }
    }
}

impl std::error::Error for ParseError {}

impl Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedEof => f.write_str("unexpected end-of-file"),
            ParseError::ExpectedEof => f.write_str("expected the end of input or \";\""),
            ParseError::MissingEnd(kind) => write!(f, "missing closing \"{}\"", kind.as_close()),
            ParseError::MissingEndOr(kind, token) => write!(
                f,
                "missing closing \"{}\" or \"{}\"",
                kind.as_close(),
                token
            ),
            ParseError::Token(err) => Display::fmt(err, f),
            ParseError::UnexpectedToken => f.write_str("unexpected token"),
            ParseError::ExpectedDeclaration => f.write_str("expected a declaration"),
            ParseError::ExpectedName => f.write_str("expected a name (identifier)"),
            ParseError::ExpectedBlock => f.write_str("expected a block"),
            ParseError::ExpectedModuleBody => f.write_str("expected a module body"),
            ParseError::ExpectedFunctionBody => f.write_str("expected function body"),
            ParseError::ExpectedStructureMember => f.write_str("expected structure member"),
            ParseError::ExpectedIn => f.write_str("expected \"in\""),
            ParseError::ExpectedFunctionParameters => f.write_str("expected function parameters"),
            ParseError::ExpectedThenOrBrace => f.write_str("expected \"then\" or \"{\""),
            ParseError::ExpectedColon => f.write_str("expected \":\""),
            ParseError::ExpectedCommaOrBrace => f.write_str("expected comma or closing brace"),
            ParseError::ExpectedMatchBody => f.write_str("expected match body"),
            ParseError::ExpectedPattern => f.write_str("expected match pattern"),
            ParseError::ExpectedPatternOr(paired) => {
                write!(f, "expected match pattern or {}", paired.as_close())
            }
            ParseError::ExpectedFatArrow => f.write_str("expected fat arrow (=>)"),
            ParseError::ExpectedCatchBlock => f.write_str("expected catch block"),
            ParseError::InvalidAssignmentTarget => f.write_str("invalid assignment target"),
            ParseError::InvalidLabelTarget => f.write_str("invalid label target"),
            ParseError::InvalidMapKeyPattern => f.write_str("invalid map key pattern"),
        }
    }
}

impl From<Ranged<token::LexerError>> for Ranged<ParseError> {
    fn from(err: Ranged<token::LexerError>) -> Self {
        err.map(ParseError::Token)
    }
}

struct ParserConfig<'a> {
    parselets: &'a Parselets,
    minimum_precedence: usize,
}

impl<'a> ParserConfig<'a> {
    fn with_precedence(&self, minimum_precedence: usize) -> Self {
        Self {
            parselets: self.parselets,
            minimum_precedence,
        }
    }

    fn parse(
        &self,
        tokens: &mut TokenReader<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let mut lhs = self.parse_prefix(tokens)?;
        if let Some(start_index) =
            self.parselets
                .infix
                .0
                .iter()
                .enumerate()
                .find_map(|(index, level)| {
                    (level.precedence >= self.minimum_precedence).then_some(index)
                })
        {
            'infix: while let Some(possible_operator) = tokens.peek() {
                for level in &self.parselets.infix.0[start_index..] {
                    if let Some(parselet) = level.find_parselet(&possible_operator, tokens) {
                        tokens.next_or_eof()?;
                        lhs = parselet.parse(
                            lhs,
                            possible_operator,
                            tokens,
                            &self.with_precedence(level.precedence),
                        )?;
                        continue 'infix;
                    }
                }

                // If we reach this location, we didn't find a matching parselet.
                break;
            }
        }

        Ok(lhs)
    }

    fn parse_prefix(
        &self,
        tokens: &mut TokenReader<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let token = tokens.next_or_eof()?;
        for level in self
            .parselets
            .prefix
            .0
            .iter()
            .skip_while(|level| level.precedence < self.minimum_precedence)
        {
            if let Some(parselet) = level.find_parselet(&token, tokens) {
                return parselet.parse_prefix(
                    token,
                    tokens,
                    &self.with_precedence(level.precedence),
                );
            }
        }

        Err(token.map(|_| ParseError::UnexpectedToken))
    }

    fn parse_next(
        &self,
        tokens: &mut TokenReader<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        self.with_precedence(self.minimum_precedence + 1)
            .parse(tokens)
    }

    fn parse_expression(
        &self,
        tokens: &mut TokenReader<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        self.with_precedence(self.parselets.markers.expression)
            .parse(tokens)
    }

    fn parse_conditional(
        &self,
        tokens: &mut TokenReader<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        self.with_precedence(self.parselets.markers.conditional)
            .parse(tokens)
    }
}

struct Break;

impl Parselet for Break {
    fn token(&self) -> Option<Token> {
        Some(Token::Identifier(Symbol::break_symbol().clone()))
    }
}

impl PrefixParselet for Break {
    fn parse_prefix(
        &self,
        r#break: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let name = if tokens
            .peek_token()
            .map_or(false, |token| matches!(token, Token::Label(_)))
        {
            let label_token = tokens.next_or_eof()?;
            let Token::Label(label) = label_token.0 else {
                unreachable!("just matched")
            };
            Some(Ranged::new(label_token.1, label))
        } else {
            None
        };
        let value = if tokens
            .peek_token()
            .map_or(false, |token| !token.is_likely_end())
        {
            config.parse_expression(tokens)?
        } else {
            tokens.ranged(tokens.last_index.., Expression::Literal(Literal::Nil))
        };

        Ok(tokens.ranged(
            r#break.range().start..,
            Expression::Break(Box::new(BreakExpression {
                r#break,
                name,
                value,
            })),
        ))
    }
}

struct Continue;

impl Parselet for Continue {
    fn token(&self) -> Option<Token> {
        Some(Token::Identifier(Symbol::continue_symbol().clone()))
    }
}

impl PrefixParselet for Continue {
    fn parse_prefix(
        &self,
        r#continue: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        _config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let name = if tokens
            .peek_token()
            .map_or(false, |token| matches!(token, Token::Label(_)))
        {
            let label_token = tokens.next_or_eof()?;
            let Token::Label(label) = label_token.0 else {
                unreachable!("just matched")
            };
            Some(Ranged::new(label_token.1, label))
        } else {
            None
        };

        Ok(tokens.ranged(
            r#continue.range().start..,
            Expression::Continue(Box::new(ContinueExpression { r#continue, name })),
        ))
    }
}

struct Return;

impl Parselet for Return {
    fn token(&self) -> Option<Token> {
        Some(Token::Identifier(Symbol::return_symbol().clone()))
    }
}

impl PrefixParselet for Return {
    fn parse_prefix(
        &self,
        r#return: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let value = if tokens
            .peek_token()
            .map_or(false, |token| !token.is_likely_end())
        {
            config.parse_expression(tokens)?
        } else {
            tokens.ranged(tokens.last_index.., Expression::Literal(Literal::Nil))
        };

        Ok(tokens.ranged(
            r#return.range().start..,
            Expression::Return(Box::new(ReturnExpression { r#return, value })),
        ))
    }
}

struct Parselets {
    precedence: usize,
    infix: Precedented<Box<dyn InfixParselet>>,
    prefix: Precedented<Box<dyn PrefixParselet>>,
    markers: ParseletMarkers,
}

#[derive(Default)]
struct ParseletMarkers {
    expression: usize,
    conditional: usize,
}

impl Parselets {
    fn new() -> Self {
        Self {
            precedence: 0,
            infix: Precedented::new(),
            prefix: Precedented::new(),
            markers: ParseletMarkers::default(),
        }
    }

    fn push_infix(&mut self, parselets: Vec<Box<dyn InfixParselet>>) {
        self.infix.push(self.precedence, parselets);
        self.precedence += 1;
    }

    fn push_prefix(&mut self, parselets: Vec<Box<dyn PrefixParselet>>) {
        self.prefix.push(self.precedence, parselets);
        self.precedence += 1;
    }
}

struct Precedented<T>(Vec<SharedPredence<T>>);

impl<T> Precedented<T> {
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    pub fn push(&mut self, precedence: usize, multi: Vec<T>)
    where
        T: Parselet,
    {
        let mut by_token = AHashMap::new();
        let mut wildcard = Vec::new();

        for t in multi {
            if let Some(token) = t.token() {
                by_token.insert(token.clone(), t);
            } else {
                wildcard.push(t);
            }
        }

        self.0.push(SharedPredence {
            precedence,
            by_token,
            wildcard,
        });
    }
}

struct SharedPredence<T> {
    precedence: usize,
    by_token: AHashMap<Token, T>,
    wildcard: Vec<T>,
}

impl<T> SharedPredence<T>
where
    T: Parselet,
{
    fn find_parselet(&self, token: &Token, tokens: &mut TokenReader<'_>) -> Option<&T> {
        if let Some(parselet) = self.by_token.get(token) {
            Some(parselet)
        } else if let Some(parselet) = self
            .wildcard
            .iter()
            .find(|parselet| parselet.matches(token, tokens))
        {
            Some(parselet)
        } else {
            None
        }
    }
}

trait Parselet {
    fn token(&self) -> Option<Token>;

    #[allow(unused_variables)]
    fn matches(&self, token: &Token, tokens: &mut TokenReader<'_>) -> bool {
        matches!(self.token(), Some(token))
    }
}

impl<T> Parselet for Box<T>
where
    T: Parselet + ?Sized,
{
    fn matches(&self, token: &Token, tokens: &mut TokenReader<'_>) -> bool {
        T::matches(self, token, tokens)
    }

    fn token(&self) -> Option<Token> {
        T::token(self)
    }
}

trait PrefixParselet: Parselet {
    fn parse_prefix(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>>;
}

struct Term;

impl Parselet for Term {
    fn token(&self) -> Option<Token> {
        None
    }

    fn matches(&self, token: &Token, _tokens: &mut TokenReader<'_>) -> bool {
        matches!(
            token,
            Token::Int(_)
                | Token::UInt(_)
                | Token::Float(_)
                | Token::Identifier(_)
                | Token::Regex(_)
                | Token::String(_)
                | Token::Symbol(_)
                | Token::Sigil(_)
                | Token::FormatString(_)
        )
    }
}

impl PrefixParselet for Term {
    fn parse_prefix(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        match token.0 {
            Token::Int(value) => Ok(Ranged::new(
                token.1,
                Expression::Literal(Literal::Int(value)),
            )),
            Token::UInt(value) => Ok(Ranged::new(
                token.1,
                Expression::Literal(Literal::UInt(value)),
            )),
            Token::Float(value) => Ok(Ranged::new(
                token.1,
                Expression::Literal(Literal::Float(value)),
            )),
            Token::String(string) => Ok(Ranged::new(
                token.1,
                Expression::Literal(Literal::String(string)),
            )),
            Token::Regex(regex) => Ok(Ranged::new(
                token.1,
                Expression::Literal(Literal::Regex(regex)),
            )),
            Token::Identifier(value) => Ok(Ranged::new(
                token.1,
                Expression::Lookup(Box::new(Lookup::from(Ranged::new(token.1, value)))),
            )),
            Token::Symbol(sym) => Ok(Ranged::new(
                token.1,
                Expression::Literal(Literal::Symbol(sym)),
            )),
            Token::Sigil(sym) => {
                if &sym == Symbol::sigil_symbol() {
                    Ok(Ranged::new(token.1, Expression::RootModule))
                } else {
                    let contents = gather_macro_tokens(tokens)?;
                    Ok(tokens.ranged(
                        token.1.start..,
                        Expression::Macro(Box::new(MacroInvocation {
                            name: Ranged::new(token.1, sym),
                            tokens: VecDeque::from(contents),
                        })),
                    ))
                }
            }
            Token::FormatString(format_string) => {
                if format_string.parts.is_empty() {
                    Ok(format_string
                        .initial
                        .map(|s| Expression::Literal(Literal::String(s))))
                } else {
                    let mut all_strings =
                        Delimited::<_, Ranged<Expression>>::build(format_string.initial);
                    for part in format_string.parts {
                        let mut reader = TokenReader::from(VecDeque::from(part.expression));
                        let expression = config.parse_expression(&mut reader)?;

                        // Ensure the expression was fully consumed
                        match reader.next_or_eof() {
                            Ok(token) => {
                                return Err(token.map(|_| ParseError::ExpectedEof));
                            }
                            Err(Ranged(ParseError::UnexpectedEof, _)) => {}
                            Err(other) => return Err(other),
                        }

                        all_strings.set_delimiter(expression);

                        all_strings.push(part.suffix);
                    }
                    Ok(tokens.ranged(
                        token.1.start..,
                        Expression::FormatString(Box::new(all_strings.finish())),
                    ))
                }
            }
            _ => unreachable!("parse called with invalid token"),
        }
    }
}

struct InfixMacro;

impl Parselet for InfixMacro {
    fn token(&self) -> Option<Token> {
        None
    }

    fn matches(&self, token: &Token, _tokens: &mut TokenReader<'_>) -> bool {
        matches!(token, Token::Sigil(sigil) if sigil != Symbol::sigil_symbol())
    }
}

impl InfixParselet for InfixMacro {
    fn parse(
        &self,
        lhs: Ranged<Expression>,
        name: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        _config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let range = name.range();
        let Token::Sigil(name) = name.0 else {
            unreachable!("parselet matched")
        };
        let name = Ranged::new(range, name);

        let macro_tokens = gather_macro_tokens(tokens)?;

        Ok(tokens.ranged(
            lhs.range().start..,
            Expression::InfixMacro(Box::new(InfixMacroInvocation {
                subject: lhs,
                invocation: MacroInvocation {
                    name,
                    tokens: VecDeque::from(macro_tokens),
                },
            })),
        ))
    }
}

fn gather_macro_tokens(
    tokens: &mut TokenReader<'_>,
) -> Result<Vec<Ranged<Token>>, Ranged<ParseError>> {
    let Some(Token::Open(paired)) = tokens.peek_token() else {
        return Ok(Vec::new());
    };

    let mut stack = vec![paired];
    let mut contents = vec![tokens.next_or_eof()?];

    while let Some(last_open) = stack.last().copied() {
        let token = tokens.next(ParseError::MissingEnd(last_open))?;
        match &token.0 {
            Token::Open(next) => stack.push(*next),
            Token::Close(kind) => {
                if *kind == last_open {
                    stack.pop();
                } else {
                    return Err(token.map(|_| ParseError::MissingEnd(last_open)));
                }
            }
            _ => {}
        }
        contents.push(token);
    }

    Ok(contents)
}

macro_rules! impl_prefix_unary_parselet {
    ($name:ident, $token:expr) => {
        impl_prefix_unary_parselet!($name, $token, UnaryKind::$name);
    };
    ($name:ident, $token:expr, $binarykind:expr) => {
        struct $name;

        impl Parselet for $name {
            fn token(&self) -> Option<Token> {
                Some($token)
            }
        }

        impl PrefixParselet for $name {
            fn parse_prefix(
                &self,
                token: Ranged<Token>,
                tokens: &mut TokenReader<'_>,
                config: &ParserConfig<'_>,
            ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
                let operand = config.parse_prefix(tokens)?;
                Ok(tokens.ranged(
                    token.range().start..,
                    Expression::Unary(Box::new(UnaryExpression {
                        kind: $binarykind,
                        operator: token,
                        operand,
                    })),
                ))
            }
        }
    };
}

impl_prefix_unary_parselet!(LogicalNot, Token::Identifier(Symbol::not_symbol().clone()));
impl_prefix_unary_parselet!(BitwiseNot, Token::Char('!'));
impl_prefix_unary_parselet!(Negate, Token::Char('-'));

macro_rules! impl_prefix_standalone_parselet {
    ($name:ident, $token:expr, $binarykind:expr) => {
        struct $name;

        impl Parselet for $name {
            fn token(&self) -> Option<Token> {
                Some($token)
            }
        }

        impl PrefixParselet for $name {
            fn parse_prefix(
                &self,
                token: Ranged<Token>,
                _tokens: &mut TokenReader<'_>,
                _config: &ParserConfig<'_>,
            ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
                Ok(Ranged::new(token.range(), $binarykind))
            }
        }
    };
}

impl_prefix_standalone_parselet!(
    True,
    Token::Identifier(Symbol::true_symbol().clone()),
    Expression::Literal(Literal::Bool(true))
);
impl_prefix_standalone_parselet!(
    False,
    Token::Identifier(Symbol::false_symbol().clone()),
    Expression::Literal(Literal::Bool(false))
);
impl_prefix_standalone_parselet!(
    Nil,
    Token::Identifier(Symbol::nil_symbol().clone()),
    Expression::Literal(Literal::Nil)
);

struct Braces;

fn parse_block(
    tokens: &mut TokenReader<'_>,
    config: &ParserConfig<'_>,
) -> Result<Ranged<Block>, Ranged<ParseError>> {
    let open_brace = tokens.next(ParseError::ExpectedBlock)?;
    if matches!(open_brace.0, Token::Open(Paired::Brace)) {
        if tokens.peek_token() == Some(Token::Close(Paired::Brace)) {
            let close_brace = tokens.next(ParseError::MissingEnd(Paired::Brace))?;
            return Ok(tokens.ranged(
                open_brace.range().start..,
                Block {
                    label: None,
                    body: Enclosed {
                        enclosed: Ranged::new(
                            (
                                open_brace.range().source_id,
                                open_brace.range().end()..close_brace.range().start,
                            ),
                            Expression::Literal(Literal::Nil),
                        ),
                        open: open_brace,
                        close: close_brace,
                    },
                },
            ));
        }
        let expr = config.parse_expression(tokens)?;

        match tokens.peek() {
            Some(Ranged(Token::Char(';'), _)) => {
                let semicolon = tokens.next_or_eof()?;
                Braces::parse_block(open_brace, expr, semicolon, tokens, config)
            }
            Some(Ranged(Token::Close(Paired::Brace), _)) => {
                let close = tokens.next_or_eof()?;
                Ok(tokens.ranged(
                    open_brace.range().start..,
                    Block {
                        label: None,
                        body: Enclosed {
                            enclosed: expr,
                            open: open_brace,
                            close,
                        },
                    },
                ))
            }
            other => Err(tokens.ranged(
                other.map_or(tokens.last_index, |t| t.range().start)..,
                ParseError::MissingEnd(Paired::Brace),
            )),
        }
    } else {
        Err(open_brace.map(|_| ParseError::ExpectedBlock))
    }
}

impl Braces {
    fn parse_block(
        open: Ranged<Token>,
        expr: Ranged<Expression>,
        mut semicolon: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Block>, Ranged<ParseError>> {
        let mut left = expr;

        let mut ended_in_semicolon = true;

        while tokens
            .peek()
            .map_or(false, |token| token.0 != Token::Close(Paired::Brace))
        {
            let right = config.parse_expression(tokens)?;
            left = tokens.ranged(
                left.range().start..,
                Expression::Binary(Box::new(BinaryExpression {
                    left,
                    right,
                    kind: BinaryKind::Chain,
                    operator: semicolon,
                })),
            );

            if tokens.peek_token() == Some(Token::Char(';')) {
                semicolon = tokens.next_or_eof()?;
                ended_in_semicolon = true;
            } else {
                ended_in_semicolon = false;
                // This value isn't used, but since we consume the final
                // semicolon conditionally, we must re-initialize semicolon in
                // this branch anyways to satisfy the compiler.
                semicolon = Ranged::new(SourceRange::default(), Token::Comment);
                break;
            }
        }

        if ended_in_semicolon {
            left = tokens.ranged(
                left.range().start..,
                Expression::Binary(Box::new(BinaryExpression {
                    left,
                    right: tokens.ranged(tokens.last_index.., Expression::Literal(Literal::Nil)),
                    kind: BinaryKind::Chain,
                    operator: semicolon,
                })),
            );
        }

        match tokens.next_or_eof() {
            Ok(token @ Ranged(Token::Close(Paired::Brace), _)) => Ok(tokens.ranged(
                open.range().start..,
                Block {
                    label: None,
                    body: Enclosed {
                        enclosed: left,
                        open,
                        close: token,
                    },
                },
            )),
            Ok(Ranged(_, range)) | Err(Ranged(_, range)) => {
                Err(Ranged::new(range, ParseError::MissingEnd(Paired::Brace)))
            }
        }
    }

    fn parse_map(
        open: Ranged<Token>,
        key: Ranged<Expression>,
        colon: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let value = config.parse_expression(tokens)?;
        let mut values = Delimited::<_, Ranged<Token>>::build(MapField {
            key,
            colon: Some(colon),
            value,
        });

        match tokens.peek_token() {
            Some(Token::Char(',')) => {
                values.set_delimiter(tokens.next_or_eof()?);
                while tokens
                    .peek()
                    .map_or(false, |token| token.0 != Token::Close(Paired::Brace))
                {
                    let key = config.parse_expression(tokens)?;
                    let colon = match tokens.next(ParseError::ExpectedColon)? {
                        colon @ Ranged(Token::Char(':'), _) => colon,
                        other => return Err(other.map(|_| ParseError::ExpectedColon)),
                    };
                    let value = config.parse_expression(tokens)?;
                    values.push(MapField {
                        key,
                        colon: Some(colon),
                        value,
                    });

                    if tokens.peek_token() == Some(Token::Char(',')) {
                        values.set_delimiter(tokens.next_or_eof()?);
                    } else {
                        break;
                    }
                }
            }
            Some(Token::Close(Paired::Brace)) => {}
            _ => {
                return Err(tokens
                    .next(ParseError::ExpectedCommaOrBrace)?
                    .map(|_| ParseError::ExpectedCommaOrBrace))
            }
        }

        match tokens.next_or_eof() {
            Ok(close @ Ranged(Token::Close(Paired::Brace), _)) => Ok(tokens.ranged(
                open.range().start..,
                Expression::Map(Box::new(MapExpression {
                    fields: Enclosed {
                        open,
                        enclosed: values.finish(),
                        close,
                    },
                })),
            )),
            Ok(Ranged(_, range)) | Err(Ranged(_, range)) => {
                Err(Ranged::new(range, ParseError::MissingEnd(Paired::Brace)))
            }
        }
    }

    fn parse_set(
        open: Ranged<Token>,
        expr: Ranged<Expression>,
        comma: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let mut values = Delimited::<_, Ranged<Token>>::build(MapField {
            key: expr.clone(),
            colon: None,
            value: expr,
        });
        values.set_delimiter(comma);

        while tokens
            .peek()
            .map_or(false, |token| token.0 != Token::Close(Paired::Brace))
        {
            let expr = config.parse_expression(tokens)?;
            values.push(MapField {
                key: expr.clone(),
                colon: None,
                value: expr,
            });

            if tokens.peek_token() == Some(Token::Char(',')) {
                values.set_delimiter(tokens.next_or_eof()?);
            } else {
                break;
            }
        }

        match tokens.next_or_eof() {
            Ok(close @ Ranged(Token::Close(Paired::Brace), _)) => Ok(tokens.ranged(
                open.range().start..,
                Expression::Map(Box::new(MapExpression {
                    fields: Enclosed {
                        open,
                        enclosed: values.finish(),
                        close,
                    },
                })),
            )),
            Ok(Ranged(_, range)) | Err(Ranged(_, range)) => {
                Err(Ranged::new(range, ParseError::MissingEnd(Paired::Brace)))
            }
        }
    }
}

impl Parselet for Braces {
    fn token(&self) -> Option<Token> {
        Some(Token::Open(Paired::Brace))
    }
}

impl PrefixParselet for Braces {
    fn parse_prefix(
        &self,
        open: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        match tokens.peek_token() {
            Some(Token::Close(Paired::Brace)) => {
                tokens.next_or_eof()?;
                return Ok(tokens.ranged(open.range().start.., Expression::Literal(Literal::Nil)));
            }
            Some(Token::Char(',')) => {
                tokens.next_or_eof()?;
                if tokens.peek_token() == Some(Token::Close(Paired::Brace)) {
                    let close = tokens.next_or_eof()?;
                    return Ok(tokens.ranged(
                        open.range().start..,
                        Expression::Map(Box::new(MapExpression {
                            fields: Enclosed {
                                open,
                                enclosed: Delimited::empty(),
                                close,
                            },
                        })),
                    ));
                }
            }
            _ => {}
        }
        let expr = config.parse_expression(tokens)?;

        match tokens.peek() {
            Some(Ranged(Token::Char(':'), _)) => {
                let colon = tokens.next_or_eof()?;
                Self::parse_map(open, expr, colon, tokens, config)
            }
            Some(Ranged(Token::Char(','), _)) => {
                let comma = tokens.next_or_eof()?;
                Self::parse_set(open, expr, comma, tokens, config)
            }
            Some(Ranged(Token::Char(';'), _)) => {
                let semicolon = tokens.next_or_eof()?;
                Self::parse_block(open, expr, semicolon, tokens, config)
                    .map(|ranged| ranged.map(|block| Expression::Block(Box::new(block))))
            }
            Some(Ranged(Token::Close(Paired::Brace), _)) => {
                let close = tokens.next_or_eof()?;
                Ok(tokens.ranged(
                    open.range().start..,
                    Expression::Block(Box::new(Block {
                        label: None,
                        body: Enclosed {
                            enclosed: expr,
                            open,
                            close,
                        },
                    })),
                ))
            }
            other => Err(tokens.ranged(
                other.map_or(tokens.last_index, |t| t.range().start)..,
                ParseError::MissingEnd(Paired::Brace),
            )),
        }
    }
}

struct New;

impl Parselet for New {
    fn token(&self) -> Option<Token> {
        None
    }

    fn matches(&self, token: &Token, tokens: &mut TokenReader<'_>) -> bool {
        matches!(token, Token::Identifier(ident) if ident == Symbol::new_symbol())
            && tokens
                .peek_token()
                .map_or(false, |t| matches!(t, Token::Identifier(_)))
    }
}

impl PrefixParselet for New {
    fn parse_prefix(
        &self,
        new: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let start = new.range().start;
        let name = config.parse_expression(tokens)?;
        let fields = if tokens.peek_token() == Some(Token::Open(Paired::Brace)) {
            let mut fields = Delimited::build_empty();
            let open = tokens.next_or_eof()?;
            let (_trailing_delimiter, close) =
                parse_paired(Paired::Brace, ',', tokens, |delimiter, tokens| {
                    if let Some(delimiter) = delimiter {
                        fields.set_delimiter(delimiter);
                    }
                    let name = tokens.next_identifier(ParseError::ExpectedName)?;

                    let colon = tokens.next(ParseError::ExpectedColon)?;
                    let Token::Char(':') = colon.0 else {
                        return Err(Ranged::new(colon.1, ParseError::ExpectedColon));
                    };
                    let value = config.parse_expression(tokens)?;
                    fields.push(NewStructField { name, colon, value });
                    Ok(())
                })?;
            Some(Enclosed {
                open,
                close,
                enclosed: fields.finish(),
            })
        } else {
            None
        };

        Ok(tokens.ranged(
            start..,
            Expression::StructureLiteral(Box::new(NewStruct { new, name, fields })),
        ))
    }
}

type SeparatorAndEnd = (Option<Ranged<Token>>, Ranged<Token>);

fn parse_paired(
    end: Paired,
    separator: char,
    tokens: &mut TokenReader<'_>,
    mut inner: impl FnMut(Option<Ranged<Token>>, &mut TokenReader<'_>) -> Result<(), Ranged<ParseError>>,
) -> Result<SeparatorAndEnd, Ranged<ParseError>> {
    let mut ending_separator = None;
    let sep_token = Token::Char(separator);
    if tokens.peek().map_or(false, |token| token.0 == sep_token) {
        ending_separator = Some(tokens.next_or_eof()?);
    } else {
        while tokens
            .peek()
            .map_or(false, |token| token.0 != Token::Close(end))
        {
            inner(ending_separator, tokens)?;

            if tokens.peek_token().as_ref() == Some(&sep_token) {
                ending_separator = Some(tokens.next_or_eof()?);
            } else {
                ending_separator = None;
                break;
            }
        }
    }

    let close = tokens.next(ParseError::MissingEnd(end));
    match &close {
        Ok(Ranged(Token::Close(token), _)) if token == &end => {
            Ok((ending_separator, close.expect("just matched")))
        }
        Ok(Ranged(_, range)) => Err(Ranged::new(
            *range,
            ParseError::MissingEndOr(end, separator),
        )),
        Err(Ranged(_, range)) => Err(Ranged::new(*range, ParseError::MissingEnd(end))),
    }
}

struct Parentheses;

impl Parselet for Parentheses {
    fn token(&self) -> Option<Token> {
        Some(Token::Open(Paired::Paren))
    }
}

impl PrefixParselet for Parentheses {
    fn parse_prefix(
        &self,
        open: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let expression = config.parse_expression(tokens)?;

        let end_paren = tokens.next(ParseError::MissingEnd(Paired::Paren))?;
        if end_paren.0 == Token::Close(Paired::Paren) {
            Ok(tokens.ranged(
                open.range().start..,
                Expression::Group(Box::new(Enclosed {
                    open,
                    close: end_paren,
                    enclosed: expression,
                })),
            ))
        } else {
            Err(end_paren.map(|_| ParseError::MissingEnd(Paired::Paren)))
        }
    }
}

impl InfixParselet for Parentheses {
    fn parse(
        &self,
        function: Ranged<Expression>,
        open: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let mut parameters = Delimited::<_, Ranged<Token>>::build_empty();

        let (_, close) = parse_paired(Paired::Paren, ',', tokens, |delimiter, tokens| {
            if let Some(delimiter) = delimiter {
                parameters.set_delimiter(delimiter);
            }
            config
                .parse_expression(tokens)
                .map(|expr| parameters.push(expr))
        })?;

        Ok(tokens.ranged(
            function.range().start..,
            Expression::Call(Box::new(FunctionCall {
                function,
                parameters: Enclosed {
                    open,
                    enclosed: parameters.finish(),
                    close,
                },
            })),
        ))
    }
}

struct Brackets;

impl Parselet for Brackets {
    fn token(&self) -> Option<Token> {
        Some(Token::Open(Paired::Bracket))
    }
}

impl PrefixParselet for Brackets {
    fn parse_prefix(
        &self,
        open: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let mut expressions = Delimited::<_, Ranged<Token>>::build_empty();

        let (_, close) = parse_paired(Paired::Bracket, ',', tokens, |delimiter, tokens| {
            if let Some(delimiter) = delimiter {
                expressions.set_delimiter(delimiter);
            }
            config
                .parse_expression(tokens)
                .map(|expr| expressions.push(expr))
        })?;

        Ok(tokens.ranged(
            open.range().start..,
            Expression::List(Box::new(ListExpression {
                values: Enclosed {
                    open,
                    enclosed: expressions.finish(),
                    close,
                },
            })),
        ))
    }
}

impl InfixParselet for Brackets {
    fn parse(
        &self,
        target: Ranged<Expression>,
        open: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let mut parameters = Delimited::<_, Ranged<Token>>::build_empty();

        let (_, close) = parse_paired(Paired::Bracket, ',', tokens, |delimiter, tokens| {
            if let Some(delimiter) = delimiter {
                parameters.set_delimiter(delimiter);
            }
            config
                .parse_expression(tokens)
                .map(|expr| parameters.push(expr))
        })?;

        Ok(tokens.ranged(
            target.range().start..,
            Expression::Index(Box::new(Index {
                target,
                parameters: Enclosed {
                    open,
                    enclosed: parameters.finish(),
                    close,
                },
            })),
        ))
    }
}

trait InfixParselet: Parselet {
    fn parse(
        &self,
        lhs: Ranged<Expression>,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>>;
}

macro_rules! impl_infix_parselet {
    ($name:ident, $token:expr) => {
        impl_infix_parselet!($name, $token, BinaryKind::$name);
    };
    ($name:ident, $token:expr, $binarykind:expr) => {
        struct $name;

        impl Parselet for $name {
            fn token(&self) -> Option<Token> {
                Some($token)
            }
        }

        impl InfixParselet for $name {
            fn parse(
                &self,
                left: Ranged<Expression>,
                operator: Ranged<Token>,
                tokens: &mut TokenReader<'_>,
                config: &ParserConfig<'_>,
            ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
                let right = config.parse_next(tokens)?;
                Ok(tokens.ranged(
                    left.range().start..right.range().end(),
                    Expression::Binary(Box::new(BinaryExpression {
                        kind: $binarykind,
                        left,
                        operator,
                        right,
                    })),
                ))
            }
        }
    };
}

struct If;

impl Parselet for If {
    fn token(&self) -> Option<Token> {
        Some(Token::Identifier(Symbol::if_symbol().clone()))
    }
}

impl PrefixParselet for If {
    fn parse_prefix(
        &self,
        r#if: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let condition = config.parse_expression(tokens)?;
        let brace_or_then = tokens.next(ParseError::ExpectedThenOrBrace)?;
        let (then, when_true) = match &brace_or_then.0 {
            Token::Open(Paired::Brace) => {
                (None, Braces.parse_prefix(brace_or_then, tokens, config)?)
            }
            Token::Identifier(ident) if ident == Symbol::then_symbol() => {
                (Some(brace_or_then), config.parse_expression(tokens)?)
            }
            _ => return Err(brace_or_then.map(|_| ParseError::ExpectedThenOrBrace)),
        };
        let when_false =
            if tokens.peek_token() == Some(Token::Identifier(Symbol::else_symbol().clone())) {
                let r#else = tokens.next_or_eof()?;
                Some(Else {
                    expression: match tokens.peek_token() {
                        Some(Token::Identifier(ident)) if ident == *Symbol::if_symbol() => {
                            Self.parse_prefix(tokens.next_or_eof()?, tokens, config)?
                        }
                        Some(Token::Open(Paired::Brace)) => {
                            Braces.parse_prefix(tokens.next_or_eof()?, tokens, config)?
                        }
                        _ => config.parse_expression(tokens)?,
                    },
                    r#else,
                })
            } else {
                None
            };

        Ok(tokens.ranged(
            r#if.range().start..,
            Expression::If(Box::new(IfExpression {
                r#if,
                condition,
                then,
                when_true,
                when_false,
            })),
        ))
    }
}

impl InfixParselet for If {
    fn parse(
        &self,
        lhs: Ranged<Expression>,
        r#if: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let condition = config.parse_next(tokens)?;
        let else_expr = if tokens.peek_token().map_or(false, |token| {
            token == Token::Identifier(Symbol::else_symbol().clone())
        }) {
            let r#else = tokens.next_or_eof()?;
            Some(Else {
                expression: config.parse_expression(tokens)?,
                r#else,
            })
        } else {
            None
        };

        Ok(tokens.ranged(
            lhs.range().start..,
            Expression::If(Box::new(IfExpression {
                r#if,
                condition,
                then: None,
                when_true: lhs,
                when_false: else_expr,
            })),
        ))
    }
}

struct Labeled;

impl Parselet for Labeled {
    fn token(&self) -> Option<Token> {
        None
    }

    fn matches(&self, token: &Token, tokens: &mut TokenReader<'_>) -> bool {
        matches!(token, Token::Label(_))
            && tokens
                .peek_token()
                .map_or(false, |token| matches!(token, Token::Char(':')))
    }
}

impl PrefixParselet for Labeled {
    fn parse_prefix(
        &self,
        label_token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let label = label_token.map(|token| {
            let Token::Label(token) = token else {
                unreachable!("matches filters")
            };
            token
        });

        let colon = tokens.next(ParseError::ExpectedColon)?;
        if colon.0 != Token::Char(':') {
            return Err(colon.map(|_| ParseError::ExpectedColon));
        }
        let label = Label { name: label, colon };
        let mut subject = config.parse_expression(tokens)?;
        match &mut subject.0 {
            Expression::Block(block) => block.label = Some(label),
            Expression::Loop(loop_expr) => {
                loop_expr.block.label = Some(label);
            }
            _ => return Err(subject.map(|_| ParseError::InvalidLabelTarget)),
        }

        Ok(subject)
    }
}

struct Loop;

impl Parselet for Loop {
    fn token(&self) -> Option<Token> {
        None
    }

    fn matches(&self, token: &Token, tokens: &mut TokenReader<'_>) -> bool {
        matches!(token, Token::Identifier(ident) if ident == Symbol::loop_symbol())
            && tokens
                .peek_token()
                .map_or(false, |t| matches!(t, Token::Open(Paired::Brace)))
    }
}

impl PrefixParselet for Loop {
    fn parse_prefix(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let block = parse_block(tokens, config)?;
        let kind = if tokens.peek_token().map_or(
            false,
            |token| matches!(token, Token::Identifier(ident) if &ident == Symbol::while_symbol()),
        ) {
            let r#while = tokens.next_or_eof()?;
            LoopKind::TailWhile {
                r#while,
                expression: config.parse_expression(tokens)?,
            }
        } else {
            LoopKind::Infinite
        };

        Ok(tokens.ranged(
            token.range().start..,
            Expression::Loop(Box::new(LoopExpression { token, kind, block })),
        ))
    }
}

struct For;

impl Parselet for For {
    fn token(&self) -> Option<Token> {
        None
    }

    fn matches(&self, token: &Token, tokens: &mut TokenReader<'_>) -> bool {
        matches!(token, Token::Identifier(ident) if ident == Symbol::for_symbol())
            && tokens.peek_token().map_or(false, |t| {
                matches!(t, Token::Open(_) | Token::Identifier(_))
            })
    }
}

impl PrefixParselet for For {
    fn parse_prefix(
        &self,
        for_token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let Some(pattern) = parse_pattern(tokens, config)? else {
            return Err(tokens.ranged(tokens.last_index.., ParseError::ExpectedPattern));
        };

        let r#in = tokens.next(ParseError::ExpectedIn)?;
        if !matches!(&r#in.0, Token::Identifier(ident) if ident == Symbol::in_symbol()) {
            return Err(r#in.map(|_| ParseError::ExpectedIn));
        }

        let source = config.parse_expression(tokens)?;

        let body = parse_block(tokens, config)?;

        Ok(tokens.ranged(
            for_token.range().start..,
            Expression::Loop(Box::new(LoopExpression {
                token: for_token,
                kind: LoopKind::For {
                    pattern,
                    r#in,
                    source,
                },
                block: body,
            })),
        ))
    }
}

struct While;

impl Parselet for While {
    fn token(&self) -> Option<Token> {
        Some(Token::Identifier(Symbol::while_symbol().clone()))
    }
}

impl PrefixParselet for While {
    fn parse_prefix(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let condition = config.parse_expression(tokens)?;

        let block = parse_block(tokens, config)?;
        Ok(tokens.ranged(
            token.range().start..,
            Expression::Loop(Box::new(LoopExpression {
                token,
                kind: LoopKind::While(condition),
                block,
            })),
        ))
    }
}

struct Try;

impl Parselet for Try {
    fn token(&self) -> Option<Token> {
        Some(Token::Identifier(Symbol::try_symbol().clone()))
    }
}

impl PrefixParselet for Try {
    fn parse_prefix(
        &self,
        r#try: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let body = config.parse_expression(tokens)?;

        let catch =
            if tokens.peek_token() == Some(Token::Identifier(Symbol::catch_symbol().clone())) {
                let catch_token = tokens.next_or_eof()?;
                let matches = match tokens.peek_token() {
                    Some(Token::Open(Paired::Brace)) => {
                        // Match catch
                        let open_brace = tokens.next_or_eof()?;

                        parse_match_block_body(open_brace, tokens, config)?
                    }
                    Some(Token::FatArrow) => {
                        let arrow = tokens.next_or_eof()?;
                        let pattern = Ranged::new(
                            catch_token.range(),
                            Pattern {
                                kind: Ranged::new(
                                    catch_token.range(),
                                    PatternKind::Any(Some(Symbol::it_symbol().clone())),
                                ),
                                guard: None,
                            },
                        );

                        let body = config.parse_expression(tokens)?;

                        tokens.ranged(
                            pattern.range().start..,
                            Matches {
                                patterns: Delimited::single(tokens.ranged(
                                    pattern.range().start..,
                                    MatchPattern {
                                        pattern,
                                        arrow,
                                        body,
                                    },
                                )),
                                open_close: None,
                            },
                        )
                    }
                    _ => {
                        // Inline binding
                        let Some(pattern) = parse_pattern(tokens, config)? else {
                            return Err(
                                tokens.ranged(tokens.last_index.., ParseError::ExpectedCatchBlock)
                            );
                        };
                        let body = parse_block(tokens, config)?;

                        tokens.ranged(
                            pattern.range().start..,
                            Matches {
                                patterns: Delimited::single(tokens.ranged(
                                    pattern.range().start..,
                                    MatchPattern {
                                        arrow: tokens.ranged(
                                            pattern.range().end()..pattern.range().end(),
                                            Token::FatArrow,
                                        ),
                                        pattern,
                                        body: body.map(|block| Expression::Block(Box::new(block))),
                                    },
                                )),
                                open_close: None,
                            },
                        )
                    }
                };

                Some(Catch {
                    catch: catch_token,
                    matches,
                })
            } else {
                None
            };

        Ok(tokens.ranged(
            r#try.range().start..,
            Expression::Try(Box::new(TryExpression { r#try, body, catch })),
        ))
    }
}

struct Throw;

impl Parselet for Throw {
    fn token(&self) -> Option<Token> {
        Some(Token::Identifier(Symbol::throw_symbol().clone()))
    }
}

impl PrefixParselet for Throw {
    fn parse_prefix(
        &self,
        throw: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let value = if tokens
            .peek_token()
            .map_or(false, |token| !token.is_likely_end())
        {
            config.parse_expression(tokens)?
        } else {
            tokens.ranged(tokens.last_index.., Expression::Literal(Literal::Nil))
        };

        Ok(tokens.ranged(
            throw.range().start..,
            Expression::Throw(Box::new(ThrowExpression { throw, value })),
        ))
    }
}

struct Match;

impl Parselet for Match {
    fn token(&self) -> Option<Token> {
        Some(Token::Identifier(Symbol::match_symbol().clone()))
    }

    fn matches(&self, token: &Token, tokens: &mut TokenReader<'_>) -> bool {
        matches!(token, Token::Identifier(ident) if ident == Symbol::for_symbol())
            && tokens
                .peek_token()
                .map_or(false, |t| Term.matches(&t, tokens))
    }
}

impl PrefixParselet for Match {
    fn parse_prefix(
        &self,
        r#match: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let condition = config.parse_expression(tokens)?;

        let brace = tokens.next(ParseError::ExpectedMatchBody)?;
        let matches = if brace.0 == Token::Open(Paired::Brace) {
            parse_match_block_body(brace, tokens, config)?
        } else {
            return Err(brace.map(|_| ParseError::ExpectedMatchBody));
        };

        Ok(tokens.ranged(
            r#match.range().start..,
            Expression::Match(Box::new(MatchExpression {
                r#match,
                condition,
                matches,
            })),
        ))
    }
}

struct Mod;

impl Parselet for Mod {
    fn token(&self) -> Option<Token> {
        None
    }

    fn matches(&self, token: &Token, tokens: &mut TokenReader<'_>) -> bool {
        matches!(token, Token::Identifier(ident) if ident == Symbol::mod_symbol())
            && tokens
                .peek_token()
                .map_or(false, |t| matches!(t, Token::Identifier(_)))
    }
}

impl PrefixParselet for Mod {
    fn parse_prefix(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        Self::parse_mod(None, token, tokens, config)
    }
}

impl Mod {
    fn parse_mod(
        publish: Option<Ranged<Token>>,
        r#mod: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let name_token = tokens.next(ParseError::ExpectedName)?;
        let Token::Identifier(name) = name_token.0 else {
            return Err(name_token.map(|_| ParseError::ExpectedName));
        };

        let brace = tokens.next(ParseError::ExpectedModuleBody)?;
        let brace_range = brace.range();
        let contents = if brace.0 == Token::Open(Paired::Brace) {
            Braces.parse_prefix(brace, tokens, config)?
        } else {
            return Err(Ranged::new(brace_range, ParseError::ExpectedModuleBody));
        };
        let Ranged(Expression::Block(block), range) = contents else {
            return Err(Ranged::new(brace_range, ParseError::ExpectedModuleBody));
        };

        Ok(tokens.ranged(
            r#mod.range().start..,
            Expression::Module(Box::new(ModuleDefinition {
                publish,
                r#mod,
                name: Ranged::new(name_token.1, name),
                contents: Ranged::new(range, *block),
            })),
        ))
    }
}

struct Pub;

impl Parselet for Pub {
    fn token(&self) -> Option<Token> {
        Some(Token::Identifier(Symbol::pub_symbol().clone()))
    }
}

impl PrefixParselet for Pub {
    fn parse_prefix(
        &self,
        pub_token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let keyword_token = tokens.next(ParseError::ExpectedDeclaration)?;
        let Token::Identifier(keyword) = &keyword_token.0 else {
            return Err(keyword_token.map(|_| ParseError::ExpectedDeclaration));
        };

        let pub_symbol = Ranged::new(pub_token.range(), Symbol::pub_symbol().clone());

        if keyword == Symbol::fn_symbol() {
            Fn::parse_function(Some(pub_symbol), keyword_token, tokens, config)
        } else if keyword == Symbol::let_symbol() || keyword == Symbol::var_symbol() {
            let start = pub_token.range().start;
            parse_variable(
                Ranged::new(keyword_token.1, keyword.clone()),
                Some(pub_token),
                start,
                tokens,
                config,
            )
        } else if keyword == Symbol::mod_symbol() {
            Mod::parse_mod(Some(pub_token), keyword_token, tokens, config)
        } else if keyword == Symbol::struct_symbol() {
            Struct::parse_struct(Some(pub_symbol), keyword_token, tokens, config)
        } else {
            return Err(keyword_token.map(|_| ParseError::ExpectedDeclaration));
        }
    }
}

// fn pattern_to_function_parameters(pattern: Ranged<Pattern>) -> Vec<Ranged<Pattern>> {
//     match pattern {
//         Ranged(PatternKind::DestructureTuple(tuple), _) => tuple,
//         other => vec![other],
//     }
// }

struct Fn;

impl Fn {
    fn parse_function(
        publish: Option<Ranged<Symbol>>,
        r#fn: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let start = r#fn.range().start;
        let func = Self::parse_function_definition(publish, r#fn, tokens, config)?;
        Ok(tokens.ranged(start.., Expression::Function(Box::new(func))))
    }
    fn parse_function_definition(
        publish: Option<Ranged<Symbol>>,
        r#fn: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<FunctionDefinition, Ranged<ParseError>> {
        let name = if let Some(Token::Identifier(name)) = tokens.peek_token() {
            Some(tokens.next_or_eof()?.map(|_| name))
        } else {
            None
        };

        let pattern: Ranged<Pattern> = match tokens.peek_token() {
            Some(Token::Open(Paired::Paren)) => {
                let start = tokens.next_or_eof()?;

                parse_tuple_destructure_pattern(start, Paired::Paren, tokens)?.into()
            }
            Some(Token::Open(Paired::Brace)) => {
                // Pattern/overloaded function.
                let brace = tokens.next_or_eof()?;
                let body = parse_match_block_body(brace, tokens, config)?;

                return Ok(FunctionDefinition {
                    visibility: publish,
                    r#fn,
                    name,
                    body,
                });
            }
            _ => tokens
                .ranged(
                    tokens.last_index..,
                    PatternKind::DestructureTuple(Box::new(Enclosed {
                        open: tokens.ranged(tokens.last_index.., Token::Open(Paired::Bracket)),
                        close: tokens.ranged(tokens.last_index.., Token::Close(Paired::Bracket)),
                        enclosed: Delimited::empty(),
                    })),
                )
                .into(),
        };

        let body_indicator = tokens.next(ParseError::ExpectedFunctionBody)?;
        let (arrow, body) = match &body_indicator.0 {
            Token::Open(Paired::Brace) => {
                let arrow = tokens.ranged(
                    pattern.range().end()..pattern.range().end(),
                    Token::FatArrow,
                );
                (arrow, Braces.parse_prefix(body_indicator, tokens, config)?)
            }
            Token::FatArrow => (body_indicator, config.parse_expression(tokens)?),
            _ => return Err(body_indicator.map(|_| ParseError::ExpectedFunctionBody)),
        };

        Ok(FunctionDefinition {
            visibility: publish,
            r#fn,
            name,
            body: tokens.ranged(
                pattern.range().start..,
                Matches {
                    patterns: Delimited::single(tokens.ranged(
                        pattern.range().start..,
                        MatchPattern {
                            pattern,
                            arrow,
                            body,
                        },
                    )),
                    open_close: None,
                },
            ),
        })
    }
}

impl Parselet for Fn {
    fn token(&self) -> Option<Token> {
        None
    }

    fn matches(&self, token: &Token, tokens: &mut TokenReader<'_>) -> bool {
        matches!(token, Token::Identifier(ident) if ident == Symbol::fn_symbol())
            && tokens.peek_token().map_or(false, |t| {
                matches!(
                    t,
                    Token::Identifier(_)
                        | Token::Open(Paired::Brace | Paired::Paren)
                        | Token::FatArrow
                )
            })
    }
}

impl PrefixParselet for Fn {
    fn parse_prefix(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        Self::parse_function(None, token, tokens, config)
    }
}

struct Struct;

impl Parselet for Struct {
    fn token(&self) -> Option<Token> {
        None
    }

    fn matches(&self, token: &Token, tokens: &mut TokenReader<'_>) -> bool {
        matches!(token, Token::Identifier(ident) if ident == Symbol::struct_symbol())
            && tokens
                .peek_token()
                .map_or(false, |t| matches!(t, Token::Identifier(_)))
    }
}

impl PrefixParselet for Struct {
    fn parse_prefix(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        Self::parse_struct(None, token, tokens, config)
    }
}

impl Struct {
    fn parse_struct(
        visibility: Option<Ranged<Symbol>>,
        r#struct: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let name_token = tokens.next(ParseError::ExpectedName)?;
        let Token::Identifier(name) = name_token.0 else {
            return Err(name_token.map(|_| ParseError::ExpectedName));
        };

        let brace = tokens.next(ParseError::ExpectedModuleBody)?;
        let members = if brace.0 == Token::Open(Paired::Brace) {
            let mut members = Delimited::build_empty();
            let (_, close) = parse_paired(Paired::Brace, ';', tokens, |delimiter, tokens| {
                if let Some(delimiter) = delimiter {
                    members.set_delimiter(delimiter);
                }
                let start = tokens.last_index;
                let member = Self::parse_member(tokens, config)?;
                members.push(tokens.ranged(start.., member));
                Ok(())
            })?;
            Some(Enclosed {
                open: brace,
                close,
                enclosed: members.finish(),
            })
        } else {
            None
        };

        Ok(tokens.ranged(
            r#struct.range().start..,
            Expression::Structure(Box::new(StructureDefinition {
                visibility,
                r#struct,
                members,
                name: Ranged::new(name_token.1, name),
            })),
        ))
    }

    fn parse_member(
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<StructureMember, Ranged<ParseError>> {
        let mut token = tokens.next(ParseError::ExpectedStructureMember)?;
        let mut visibility = None;
        match &token.0 {
            Token::Identifier(ident) if ident == Symbol::pub_symbol() => {
                let Token::Identifier(ident) = token.0 else {
                    unreachable!("just matched")
                };
                visibility = Some(Ranged::new(token.1, ident));
                token = tokens.next(ParseError::ExpectedStructureMember)?;
            }
            _ => {}
        }

        match token.0 {
            Token::Identifier(r#fn) if &r#fn == Symbol::fn_symbol() => {
                Ok(StructureMember::Function(Fn::parse_function_definition(
                    visibility,
                    Ranged::new(token.1, Token::Identifier(r#fn)),
                    tokens,
                    config,
                )?))
            }
            Token::Identifier(name) => Ok(StructureMember::Field {
                visibility,
                name: Ranged::new(token.1, name),
            }),
            _ => Err(Ranged::new(token.1, ParseError::ExpectedStructureMember)),
        }
    }
}

/// A set of match patterns.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct Matches {
    /// The open and close tokens that surround these patterns.
    pub open_close: Option<[Ranged<Token>; 2]>,
    /// The patterns to match against.
    pub patterns: Delimited<Ranged<MatchPattern>>,
}

impl TokenizeInto for Matches {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        if let Some(open_close) = &self.open_close {
            tokens.push_back(open_close[0].clone());
            self.patterns.tokenize_into(tokens);
            tokens.push_back(open_close[1].clone());
        } else {
            self.patterns.tokenize_into(tokens);
        }
    }
}

/// The syntax components of a pattern match in a match expression.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchPattern {
    /// The pattern to match against.
    pub pattern: Ranged<Pattern>,
    /// The arrow between the pattern and body.
    pub arrow: Ranged<Token>,
    /// The body to execute if the pattern matches.
    pub body: Ranged<Expression>,
}

impl TokenizeInto for MatchPattern {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.pattern.tokenize_into(tokens);
        tokens.push_back(self.arrow.clone());
        self.body.tokenize_into(tokens);
    }
}

impl MatchPattern {
    #[must_use]
    pub(crate) fn arity(&self) -> Option<(u8, bool)> {
        match &self.pattern.kind.0 {
            PatternKind::Any(None) | PatternKind::AnyRemaining => Some((0, true)),
            PatternKind::Any(_)
            | PatternKind::Literal(_)
            | PatternKind::Or(_, _, _)
            | PatternKind::DestructureMap(_) => Some((1, false)),
            PatternKind::DestructureTuple(fields) => {
                let variable = fields
                    .enclosed
                    .last()
                    .map_or(false, |field| matches!(&field.0, PatternKind::AnyRemaining));
                fields
                    .enclosed
                    .len()
                    .try_into()
                    .ok()
                    .map(|count| (count, variable))
            }
        }
    }
}

/// The syntax components of a pattern
#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    /// The pattern match.
    pub kind: Ranged<PatternKind>,
    /// An optional guard to check after the pattern is matched.
    pub guard: Option<Ranged<Expression>>,
}

impl TokenizeInto for Pattern {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.kind.tokenize_into(tokens);
    }
}

impl From<Ranged<PatternKind>> for Pattern {
    fn from(kind: Ranged<PatternKind>) -> Self {
        Self { kind, guard: None }
    }
}

impl From<Ranged<PatternKind>> for Ranged<Pattern> {
    fn from(kind: Ranged<PatternKind>) -> Self {
        Ranged::new(kind.range(), Pattern::from(kind))
    }
}

/// A value enclosed in open and close tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct Enclosed<T> {
    /// The open token.
    pub open: Ranged<Token>,
    /// The close token.
    pub close: Ranged<Token>,
    /// The enclosed value.
    pub enclosed: T,
}

impl<T> TokenizeInto for Enclosed<T>
where
    T: TokenizeInto,
{
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.open.clone());
        self.enclosed.tokenize_into(tokens);
        tokens.push_back(self.close.clone());
    }
}

/// The syntax components of a portion of a pattern match.
#[derive(Debug, Clone, PartialEq)]
pub enum PatternKind {
    /// Match any single expression, optionally assigning it a name.
    Any(Option<Symbol>),
    /// Match any remaining expressions.
    AnyRemaining,
    /// Match a literal value.
    Literal(Literal),
    /// Match by destructuring the value as a tuple.
    DestructureTuple(Box<Enclosed<Delimited<Ranged<PatternKind>>>>),
    /// Match by destructuring the value as a map.
    DestructureMap(Box<Enclosed<Delimited<Ranged<EntryPattern>>>>),
    /// Match by either one pattern or another pattern.
    Or(
        Box<Ranged<PatternKind>>,
        Ranged<Token>,
        Box<Ranged<PatternKind>>,
    ),
}

impl TokenizeRanged for PatternKind {
    fn tokenize_ranged(&self, range: SourceRange, tokens: &mut VecDeque<Ranged<Token>>) {
        match self {
            PatternKind::Any(None) => tokens.push_back(Ranged::new(range, Token::Char('_'))),
            PatternKind::Any(Some(name)) => {
                tokens.push_back(Ranged::new(range, Token::Identifier(name.clone())));
            }
            PatternKind::AnyRemaining => tokens.push_back(Ranged::new(range, Token::Ellipses)),
            PatternKind::Literal(literal) => literal.tokenize_ranged(range, tokens),
            PatternKind::DestructureTuple(tuple) => {
                tuple.tokenize_into(tokens);
            }
            PatternKind::DestructureMap(map) => {
                map.tokenize_into(tokens);
            }
            PatternKind::Or(a, or, b) => {
                a.tokenize_into(tokens);
                tokens.push_back(or.clone());
                b.tokenize_into(tokens);
            }
        }
    }
}

/// The syntax components of a pattern match against a single map field.
#[derive(Debug, Clone, PartialEq)]
pub struct EntryPattern {
    /// The key pattern to match against.
    pub key: Ranged<EntryKeyPattern>,
    /// The `:`  token.
    pub colon: Ranged<Token>,
    /// The value pattern to match against.
    pub value: Ranged<PatternKind>,
}

impl TokenizeInto for EntryPattern {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        let key = match &self.key.0 {
            EntryKeyPattern::Nil => Token::Identifier(Symbol::nil_symbol().clone()),
            EntryKeyPattern::Bool(false) => Token::Identifier(Symbol::false_symbol().clone()),
            EntryKeyPattern::Bool(true) => Token::Identifier(Symbol::true_symbol().clone()),
            EntryKeyPattern::Int(value) => Token::Int(*value),
            EntryKeyPattern::UInt(value) => Token::UInt(*value),
            EntryKeyPattern::Float(float) => Token::Float(*float),
            EntryKeyPattern::String(string) => Token::String(string.clone()),
            EntryKeyPattern::Identifier(symbol) => Token::Identifier(symbol.clone()),
        };
        tokens.push_back(Ranged::new(self.key.range(), key));
        tokens.push_back(self.colon.clone());
        self.value.tokenize_into(tokens);
    }
}

/// The syntax components of the key portion of a pattern match on a map field.
#[derive(Debug, Clone, PartialEq)]
pub enum EntryKeyPattern {
    /// Match if nil
    Nil,
    /// Match if a boolean of a given value.
    Bool(bool),
    /// Match if an integer of a given value.
    ///
    /// While this pattern is unsigned, if an unsigned integer has the same
    /// numerical value, it will still match.
    Int(i64),
    /// Match if an integer of a given value.
    ///
    /// While this pattern is unsigned, if a signed integer has the same
    /// numerical value, it will still match.
    UInt(u64),
    /// Match if a float of a given value.
    ///
    /// Float matching checks if the difference between the pattern and value is
    /// less than [`f64::EPSILON`].
    Float(f64),
    /// Match if a string of a given value.
    String(Symbol),
    /// Match against a value contained in a variable.
    Identifier(Symbol),
}

fn parse_pattern(
    tokens: &mut TokenReader<'_>,
    config: &ParserConfig<'_>,
) -> Result<Option<Ranged<Pattern>>, Ranged<ParseError>> {
    let Some(kind) = parse_pattern_kind(tokens)? else {
        return Ok(None);
    };

    let guard = if tokens.peek_token().map_or(
        false,
        |token| matches!(&token, Token::Identifier(ident) if ident == Symbol::if_symbol()),
    ) {
        tokens.next_or_eof()?;
        Some(config.parse_conditional(tokens)?)
    } else {
        None
    };

    Ok(Some(
        tokens.ranged(kind.range().start.., Pattern { kind, guard }),
    ))
}

#[allow(clippy::too_many_lines)]
fn parse_pattern_kind(
    tokens: &mut TokenReader<'_>,
) -> Result<Option<Ranged<PatternKind>>, Ranged<ParseError>> {
    let Some(indicator) = tokens.peek() else {
        return Ok(None);
    };
    let mut pattern = match &indicator.0 {
        Token::Char('_') => {
            tokens.next_or_eof()?;
            indicator.map(|_| PatternKind::Any(None))
        }
        Token::Ellipses => {
            tokens.next_or_eof()?;
            indicator.map(|_| PatternKind::AnyRemaining)
        }
        Token::Identifier(name) if name == Symbol::true_symbol() => {
            tokens.next_or_eof()?;
            Ranged::new(indicator.range(), PatternKind::Literal(Literal::Bool(true)))
        }
        Token::Identifier(name) if name == Symbol::false_symbol() => {
            tokens.next_or_eof()?;
            Ranged::new(
                indicator.range(),
                PatternKind::Literal(Literal::Bool(false)),
            )
        }
        Token::Identifier(name) if name == Symbol::nil_symbol() => {
            tokens.next_or_eof()?;
            Ranged::new(indicator.range(), PatternKind::Literal(Literal::Nil))
        }
        Token::Identifier(name) => {
            tokens.next_or_eof()?;
            Ranged::new(indicator.range(), PatternKind::Any(Some(name.clone())))
        }
        Token::Symbol(name) => {
            tokens.next_or_eof()?;
            Ranged::new(
                indicator.range(),
                PatternKind::Literal(Literal::Symbol(name.clone())),
            )
        }
        Token::Int(value) => {
            tokens.next_or_eof()?;
            Ranged::new(
                indicator.range(),
                PatternKind::Literal(Literal::Int(*value)),
            )
        }
        Token::UInt(value) => {
            tokens.next_or_eof()?;
            Ranged::new(
                indicator.range(),
                PatternKind::Literal(Literal::UInt(*value)),
            )
        }
        Token::Float(value) => {
            tokens.next_or_eof()?;
            Ranged::new(
                indicator.range(),
                PatternKind::Literal(Literal::Float(*value)),
            )
        }
        Token::Regex(_) => {
            let Ok(Ranged(Token::Regex(regex), _)) = tokens.next_or_eof() else {
                unreachable!("just peeked")
            };
            Ranged::new(
                indicator.range(),
                PatternKind::Literal(Literal::Regex(regex)),
            )
        }
        Token::String(_) => {
            let Ok(Ranged(Token::String(string), _)) = tokens.next_or_eof() else {
                unreachable!("just peeked")
            };
            Ranged::new(
                indicator.range(),
                PatternKind::Literal(Literal::String(string)),
            )
        }
        Token::Open(Paired::Bracket) => {
            tokens.next_or_eof()?;
            parse_tuple_destructure_pattern(indicator, Paired::Bracket, tokens)?
        }
        Token::Open(Paired::Brace) => {
            tokens.next_or_eof()?;

            parse_map_destructure_pattern(indicator, tokens)?
        }
        _ => return Ok(None),
    };

    while tokens.peek_token() == Some(Token::Char('|')) {
        let or = tokens.next_or_eof()?;
        let Some(rhs) = parse_pattern_kind(tokens)? else {
            return Err(tokens.ranged(tokens.last_index.., ParseError::ExpectedPattern));
        };
        pattern = tokens.ranged(
            pattern.range().start..,
            PatternKind::Or(Box::new(pattern), or, Box::new(rhs)),
        );
    }

    Ok(Some(pattern))
}

fn parse_tuple_destructure_pattern(
    open: Ranged<Token>,
    kind: Paired,
    tokens: &mut TokenReader<'_>,
) -> Result<Ranged<PatternKind>, Ranged<ParseError>> {
    let mut patterns = Delimited::<_, Ranged<Token>>::build_empty();
    while let Some(pattern) = parse_pattern_kind(tokens)? {
        patterns.push(pattern);

        if tokens.peek_token() == Some(Token::Char(',')) {
            patterns.set_delimiter(tokens.next_or_eof()?);
        } else {
            break;
        }
    }

    let patterns = patterns.finish();
    // If there were no patterns, still allow a comma for consistency with other
    // empty collection literals.
    if patterns.is_empty() && tokens.peek_token() == Some(Token::Char(',')) {
        tokens.next_or_eof()?;
    }

    let closing_brace = tokens.next(ParseError::ExpectedPatternOr(kind))?;
    if matches!(closing_brace.0, Token::Close(paired) if paired == kind) {
        Ok(tokens.ranged(
            open.range().start..,
            PatternKind::DestructureTuple(Box::new(Enclosed {
                open,
                close: closing_brace,
                enclosed: patterns,
            })),
        ))
    } else {
        Err(closing_brace.map(|_| ParseError::ExpectedPatternOr(kind)))
    }
}

fn parse_map_destructure_pattern(
    open: Ranged<Token>,
    tokens: &mut TokenReader<'_>,
) -> Result<Ranged<PatternKind>, Ranged<ParseError>> {
    let mut entries = Delimited::<_, Ranged<Token>>::build_empty();

    if tokens.peek_token() == Some(Token::Char(',')) {
        // Empty map
        tokens.next_or_eof()?;
    } else {
        while tokens
            .peek_token()
            .map_or(false, |token| token != Token::Close(Paired::Brace))
        {
            let token = tokens.next_or_eof()?;
            let key = match token.0 {
                Token::Int(value) => Ranged::new(token.1, EntryKeyPattern::Int(value)),
                Token::UInt(value) => Ranged::new(token.1, EntryKeyPattern::UInt(value)),
                Token::Float(value) => Ranged::new(token.1, EntryKeyPattern::Float(value)),
                Token::String(string) => Ranged::new(token.1, EntryKeyPattern::String(string)),
                Token::Identifier(sym) | Token::Symbol(sym) => {
                    Ranged::new(token.1, EntryKeyPattern::Identifier(sym))
                }
                _ => return Err(Ranged::new(token.1, ParseError::InvalidMapKeyPattern)),
            };

            let colon = tokens.next(ParseError::ExpectedColon)?;
            if colon.0 != Token::Char(':') {
                return Err(colon.map(|_| ParseError::ExpectedColon));
            }

            let Some(value) = parse_pattern_kind(tokens)? else {
                return Err(tokens.ranged(tokens.last_index.., ParseError::ExpectedPattern));
            };
            entries.push(tokens.ranged(key.range().start.., EntryPattern { key, colon, value }));

            if tokens.peek_token() == Some(Token::Char(',')) {
                entries.set_delimiter(tokens.next_or_eof()?);
            } else {
                break;
            }
        }
    }

    let end_brace = tokens.next(ParseError::MissingEnd(Paired::Brace))?;
    if end_brace.0 != Token::Close(Paired::Brace) {
        return Err(end_brace.map(|_| ParseError::MissingEnd(Paired::Brace)));
    }

    Ok(tokens.ranged(
        open.range().start..,
        PatternKind::DestructureMap(Box::new(Enclosed {
            open,
            close: end_brace,
            enclosed: entries.finish(),
        })),
    ))
}

fn parse_match_block_body(
    open: Ranged<Token>,
    tokens: &mut TokenReader<'_>,
    config: &ParserConfig<'_>,
) -> Result<Ranged<Matches>, Ranged<ParseError>> {
    let mut patterns = Delimited::<_, Ranged<Token>>::build_empty();
    while let Some(pattern) = parse_pattern(tokens, config)? {
        let arrow_or_brace = tokens.next(ParseError::ExpectedPatternOr(Paired::Brace))?;
        let (arrow, body) = match &arrow_or_brace.0 {
            Token::FatArrow => (arrow_or_brace, config.parse_expression(tokens)?),
            Token::Open(Paired::Brace) => {
                let arrow = arrow_or_brace.map(|_| Token::FatArrow);

                (
                    arrow,
                    parse_block(tokens, config)?.map(|block| Expression::Block(Box::new(block))),
                )
            }
            _ => return Err(arrow_or_brace.map(|_| ParseError::ExpectedPatternOr(Paired::Brace))),
        };

        patterns.push(tokens.ranged(
            pattern.range().start..,
            MatchPattern {
                pattern,
                arrow,
                body,
            },
        ));

        if tokens.peek_token() == Some(Token::Char(',')) {
            patterns.set_delimiter(tokens.next_or_eof()?);
        } else {
            break;
        }
    }

    let closing_brace = tokens.next(ParseError::ExpectedPatternOr(Paired::Brace))?;
    if closing_brace.0 == Token::Close(Paired::Brace) {
        Ok(tokens.ranged(
            open.range().start..,
            Matches {
                open_close: Some([open, closing_brace]),
                patterns: patterns.finish(),
            },
        ))
    } else {
        Err(closing_brace.map(|_| ParseError::ExpectedPatternOr(Paired::Brace)))
    }
}

fn parse_variable(
    kind: Ranged<Symbol>,
    publish: Option<Ranged<Token>>,
    start: usize,
    tokens: &mut TokenReader<'_>,
    config: &ParserConfig<'_>,
) -> Result<Ranged<Expression>, Ranged<ParseError>> {
    let Some(pattern) = parse_pattern(tokens, config)? else {
        return Err(tokens.ranged(tokens.last_index.., ParseError::ExpectedPattern));
    };

    let (eq, value) = if tokens.peek_token() == Some(Token::Char('=')) {
        let eq = tokens.next_or_eof()?;
        (Some(eq), config.parse_expression(tokens)?)
    } else {
        (
            None,
            tokens.ranged(tokens.last_index.., Expression::Literal(Literal::Nil)),
        )
    };

    let r#else = if tokens.peek_token() == Some(Token::Identifier(Symbol::else_symbol().clone())) {
        let r#else = tokens.next_or_eof()?;
        Some(Else {
            expression: config.parse_expression(tokens)?,
            r#else,
        })
    } else {
        None
    };

    Ok(tokens.ranged(
        start..,
        Expression::SingleMatch(Box::new(SingleMatch {
            publish,
            kind,
            pattern,
            eq,
            value,
            r#else,
        })),
    ))
}

struct Var;

impl Parselet for Var {
    fn matches(&self, token: &Token, _tokens: &mut TokenReader<'_>) -> bool {
        let Token::Identifier(ident) = token else {
            return false;
        };
        ident == Symbol::let_symbol() || ident == Symbol::var_symbol()
    }

    fn token(&self) -> Option<Token> {
        None
    }
}

impl PrefixParselet for Var {
    fn parse_prefix(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let Token::Identifier(keyword) = &token.0 else {
            unreachable!("parselet match")
        };

        parse_variable(
            Ranged::new(token.1, keyword.clone()),
            None,
            token.range().start,
            tokens,
            config,
        )
    }
}

struct Dot;

impl Parselet for Dot {
    fn token(&self) -> Option<Token> {
        Some(Token::Char('.'))
    }
}

impl InfixParselet for Dot {
    fn parse(
        &self,
        lhs: Ranged<Expression>,
        dot: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        _config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        let name_token = tokens.next(ParseError::ExpectedName)?;
        let Token::Identifier(name) = name_token.0 else {
            return Err(Ranged::new(name_token.1, ParseError::ExpectedName));
        };
        Ok(tokens.ranged(
            lhs.range().start..,
            Expression::Lookup(Box::new(Lookup {
                base: Some(LookupBase {
                    expression: lhs,
                    dot,
                }),
                name: Ranged::new(name_token.1, name),
            })),
        ))
    }
}

struct TryOperator;

impl Parselet for TryOperator {
    fn token(&self) -> Option<Token> {
        Some(Token::Char('?'))
    }
}

impl InfixParselet for TryOperator {
    fn parse(
        &self,
        lhs: Ranged<Expression>,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        _config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        Ok(tokens.ranged(
            lhs.range().start..,
            Expression::TryOrNil(Box::new(TryOrNil { token, body: lhs })),
        ))
    }
}

impl_infix_parselet!(Add, Token::Char('+'));
impl_infix_parselet!(Subtract, Token::Char('-'));
impl_infix_parselet!(Multiply, Token::Char('*'));
impl_infix_parselet!(Divide, Token::Char('/'));
impl_infix_parselet!(IntegerDivide, Token::IntegerDivide);
impl_infix_parselet!(Remainder, Token::Char('%'));
impl_infix_parselet!(Power, Token::Power);
impl_infix_parselet!(
    BitwiseAnd,
    Token::Char('&'),
    BinaryKind::Bitwise(BitwiseKind::And)
);
impl_infix_parselet!(
    BitwiseOr,
    Token::Char('|'),
    BinaryKind::Bitwise(BitwiseKind::Or)
);
impl_infix_parselet!(
    BitwiseXor,
    Token::Char('^'),
    BinaryKind::Bitwise(BitwiseKind::Xor)
);
impl_infix_parselet!(
    BitwiseShiftLeft,
    Token::ShiftLeft,
    BinaryKind::Bitwise(BitwiseKind::ShiftLeft)
);
impl_infix_parselet!(
    BitwiseShiftRight,
    Token::ShiftRight,
    BinaryKind::Bitwise(BitwiseKind::ShiftRight)
);
impl_infix_parselet!(
    And,
    Token::Identifier(Symbol::and_symbol().clone()),
    BinaryKind::Logical(LogicalKind::And)
);
impl_infix_parselet!(
    Or,
    Token::Identifier(Symbol::or_symbol().clone()),
    BinaryKind::Logical(LogicalKind::Or)
);
impl_infix_parselet!(
    Xor,
    Token::Identifier(Symbol::xor_symbol().clone()),
    BinaryKind::Logical(LogicalKind::Xor)
);
impl_infix_parselet!(
    LessThanOrEqual,
    Token::LessThanOrEqual,
    BinaryKind::Compare(CompareKind::LessThanOrEqual)
);
impl_infix_parselet!(
    LessThan,
    Token::Char('<'),
    BinaryKind::Compare(CompareKind::LessThan)
);
impl_infix_parselet!(
    Equal,
    Token::Equals,
    BinaryKind::Compare(CompareKind::Equal)
);
impl_infix_parselet!(
    GreaterThan,
    Token::Char('>'),
    BinaryKind::Compare(CompareKind::GreaterThan)
);
impl_infix_parselet!(
    GreaterThanOrEqual,
    Token::GreaterThanOrEqual,
    BinaryKind::Compare(CompareKind::GreaterThanOrEqual)
);
impl_infix_parselet!(
    NotEqual,
    Token::NotEqual,
    BinaryKind::Compare(CompareKind::NotEqual)
);
impl_infix_parselet!(NilCoalesce, Token::NilCoalesce, BinaryKind::NilCoalesce);

struct Assign;

impl Parselet for Assign {
    fn token(&self) -> Option<Token> {
        Some(Token::Char('='))
    }
}

impl InfixParselet for Assign {
    fn parse(
        &self,
        lhs: Ranged<Expression>,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<ParseError>> {
        match lhs.0 {
            Expression::Lookup(lookup) => {
                let value = config.parse(tokens)?;
                Ok(tokens.ranged(
                    lhs.1.start..,
                    Expression::Assign(Box::new(Assignment {
                        target: Ranged::new(lhs.1, AssignTarget::Lookup(*lookup)),
                        eq: token,
                        value,
                    })),
                ))
            }
            Expression::Index(index) => {
                let value = config.parse(tokens)?;
                Ok(tokens.ranged(
                    lhs.1.start..,
                    Expression::Assign(Box::new(Assignment {
                        target: Ranged::new(lhs.1, AssignTarget::Index(*index)),
                        eq: token,
                        value,
                    })),
                ))
            }
            _ => Err(lhs.map(|_| ParseError::InvalidAssignmentTarget)),
        }
    }
}

macro_rules! parselets {
    ($($name:expr),+ $(,)?) => {
        vec![$(Box::new($name)),+]
    }
}

fn parselets() -> Parselets {
    let mut parser = Parselets::new();
    parser.markers.expression = parser.precedence;
    parser.push_infix(parselets![Assign]);
    parser.markers.conditional = parser.precedence;
    parser.push_infix(parselets![If]);
    parser.push_infix(parselets![Or]);
    parser.push_infix(parselets![Xor]);
    parser.push_infix(parselets![And]);
    parser.push_infix(parselets![
        LessThanOrEqual,
        LessThan,
        Equal,
        NotEqual,
        GreaterThan,
        GreaterThanOrEqual
    ]);
    parser.push_infix(parselets![BitwiseOr]);
    parser.push_infix(parselets![BitwiseXor]);
    parser.push_infix(parselets![BitwiseAnd]);
    parser.push_infix(parselets![BitwiseShiftLeft, BitwiseShiftRight]);
    parser.push_infix(parselets![Add, Subtract]);
    parser.push_infix(parselets![Multiply, Divide, Remainder, IntegerDivide]);
    parser.push_infix(parselets![Power]);
    parser.push_infix(parselets![
        Parentheses,
        Dot,
        Brackets,
        TryOperator,
        NilCoalesce
    ]);
    parser.push_infix(parselets![InfixMacro]);
    parser.push_prefix(parselets![
        Braces,
        Parentheses,
        Brackets,
        LogicalNot,
        BitwiseNot,
        Negate,
        Mod,
        Pub,
        Fn,
        Struct,
        Var,
        If,
        True,
        False,
        Nil,
        Loop,
        While,
        Labeled,
        Continue,
        Break,
        Return,
        For,
        Match,
        Try,
        Throw,
        New,
    ]);
    parser.push_prefix(parselets![Term]);
    parser
}
