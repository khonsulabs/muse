use core::slice;
use std::collections::VecDeque;
use std::fmt::{self, Debug, Display};
use std::num::NonZeroUsize;
use std::ops::{Bound, Deref, DerefMut, Range, RangeBounds, RangeInclusive};
use std::{option, vec};

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use self::token::{FormatString, FormatStringPart, Paired, RegexLiteral, Token, Tokens};
use crate::exception::Exception;
use crate::symbol::Symbol;
use crate::vm::{ExecutionError, VmContext};
pub mod token;

pub struct SourceCode<'a> {
    pub code: &'a str,
    pub id: SourceId,
}

impl<'a> SourceCode<'a> {
    #[must_use]
    pub const fn new(code: &'a str, id: SourceId) -> Self {
        Self { code, id }
    }

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

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TrackedSource {
    pub name: String,
    pub source: String,
}

impl TrackedSource {
    #[must_use]
    pub fn offset_to_line(&self, mut offset: usize) -> (usize, usize) {
        let mut line_no = 1;
        for line in self.source.split_inclusive('\n') {
            if offset < line.len() {
                break;
            }
            offset -= line.len();
            line_no += 1;
        }
        (line_no, offset + 1)
    }
}

#[derive(Default)]
pub struct Sources(Vec<TrackedSource>);

impl Sources {
    pub fn push(&mut self, name: impl Into<String>, source: impl Into<String>) -> SourceCode<'_> {
        let id = self.next_id();
        self.0.push(TrackedSource {
            name: name.into(),
            source: source.into(),
        });
        SourceCode::new(&self.0.last().expect("just pushed").source, id)
    }

    #[must_use]
    pub fn get(&self, id: SourceId) -> Option<&TrackedSource> {
        let index = id.0?.get() - 1;
        self.0.get(index)
    }

    #[must_use]
    pub fn next_id(&self) -> SourceId {
        SourceId::new(NonZeroUsize::new(self.0.len() + 1).expect("always > 0"))
    }

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

#[derive(Default, Clone, Copy, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct Ranged<T>(pub T, pub SourceRange);

impl<T> Ranged<T> {
    pub fn new(range: impl Into<SourceRange>, value: T) -> Self {
        Self(value, range.into())
    }

    pub fn default_for(value: T) -> Self {
        Self::new(SourceRange::default(), value)
    }

    pub fn bounded(
        source_id: SourceId,
        range: impl RangeBounds<usize>,
        end: usize,
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
            Bound::Unbounded => end,
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

    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> Ranged<U> {
        Ranged(map(self.0), self.1)
    }
}

impl<T> Ranged<T> {
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

#[derive(Default, Clone, Copy, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub struct SourceId(Option<NonZeroUsize>);

impl SourceId {
    #[must_use]
    pub const fn anonymous() -> Self {
        Self(None)
    }

    #[must_use]
    pub const fn new(id: NonZeroUsize) -> Self {
        Self(Some(id))
    }

    #[must_use]
    pub const fn get(&self) -> Option<NonZeroUsize> {
        self.0
    }
}

#[derive(Default, Clone, Copy, Eq, PartialEq, Debug, Serialize, Deserialize, Hash)]
pub struct SourceRange {
    #[serde(default)]
    pub source_id: SourceId,
    pub start: usize,
    pub length: usize,
}

impl SourceRange {
    #[must_use]
    pub const fn end(&self) -> usize {
        self.start + self.length
    }

    #[must_use]
    pub fn with_length(mut self, length: usize) -> Self {
        self.length = length;
        self
    }

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

#[derive(Debug, Clone, PartialEq)]
pub struct Delimited<T, Delimiter = Ranged<Token>> {
    pub first: Option<T>,
    pub remaining: Vec<(Delimiter, T)>,
}

impl<T, Delimiter> Delimited<T, Delimiter> {
    #[must_use]
    pub const fn single(value: T) -> Self {
        Self {
            first: Some(value),
            remaining: Vec::new(),
        }
    }

    #[must_use]
    pub const fn empty() -> Self {
        Self {
            first: None,
            remaining: Vec::new(),
        }
    }

    #[must_use]
    pub const fn build(first: T) -> DelimitedBuilder<T, Delimiter> {
        DelimitedBuilder::new(first)
    }

    #[must_use]
    pub const fn build_empty() -> DelimitedBuilder<T, Delimiter> {
        DelimitedBuilder::empty()
    }

    pub fn len(&self) -> usize {
        self.remaining.len() + usize::from(self.first.is_some())
    }

    pub fn is_empty(&self) -> bool {
        self.first.is_none()
    }

    pub fn iter(&self) -> DelimitedIter<'_, T, Delimiter> {
        self.into_iter()
    }

    pub fn iter_mut(&mut self) -> DelimitedIterMut<'_, T, Delimiter> {
        self.into_iter()
    }

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

pub struct DelimitedBuilder<T, Delimiter = Ranged<Token>> {
    delimited: Delimited<T, Delimiter>,
    pending_delimiter: Option<Delimiter>,
}

impl<T, Delimiter> DelimitedBuilder<T, Delimiter> {
    #[must_use]
    pub const fn new(first: T) -> Self {
        Self {
            delimited: Delimited::single(first),
            pending_delimiter: None,
        }
    }

    #[must_use]
    pub const fn empty() -> Self {
        Self {
            delimited: Delimited::empty(),
            pending_delimiter: None,
        }
    }

    #[must_use]
    pub fn finish(self) -> Delimited<T, Delimiter> {
        self.delimited
    }

    pub fn set_delimiter(&mut self, delimiter: Delimiter) {
        assert!(self.pending_delimiter.replace(delimiter).is_none());
    }

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

pub trait TokenizeInto {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>);
}

impl<T> TokenizeInto for Ranged<T>
where
    T: TokenizeRanged,
{
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.0.tokenize_ranged(self.range(), tokens);
    }
}

pub trait TokenizeRanged {
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

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    RootModule,
    Literal(Literal),
    Lookup(Box<Lookup>),
    If(Box<IfExpression>),
    Match(Box<MatchExpression>),
    Try(Box<TryExpression>),
    TryOrNil(Box<TryOrNil>),
    Throw(Box<ThrowExpression>),
    Map(Box<MapExpression>),
    List(Box<ListExpression>),
    Call(Box<FunctionCall>),
    Index(Box<Index>),
    Assign(Box<Assignment>),
    Unary(Box<UnaryExpression>),
    Binary(Box<BinaryExpression>),
    Block(Box<Block>),
    Loop(Box<LoopExpression>),
    Break(Box<BreakExpression>),
    Continue(Box<ContinueExpression>),
    Return(Box<ReturnExpression>),
    Module(Box<ModuleDefinition>),
    Function(Box<FunctionDefinition>),
    Variable(Box<Variable>),
    Macro(Box<MacroInvocation>),
    InfixMacro(Box<InfixMacroInvocation>),
    Group(Box<Enclosed<Ranged<Expression>>>),
    FormatString(Box<Delimited<Ranged<Symbol>, Ranged<Expression>>>),
}

impl Expression {
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

impl Ranged<Expression> {
    #[must_use]
    pub fn to_tokens(&self) -> VecDeque<Ranged<Token>> {
        let mut tokens = VecDeque::new();
        self.tokenize_into(&mut tokens);
        tokens
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
            Expression::Function(it) => it.tokenize_into(tokens),
            Expression::Variable(it) => it.tokenize_into(tokens),
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

#[derive(Default, Debug, Clone, PartialEq)]
pub enum Literal {
    #[default]
    Nil,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    String(Symbol),
    Symbol(Symbol),
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

#[derive(Debug, Clone, PartialEq)]
pub struct Label {
    pub name: Ranged<Symbol>,
    pub colon: Ranged<Token>,
}

impl TokenizeInto for Label {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.name.clone().map(Token::Label));
        tokens.push_back(self.colon.clone());
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub name: Option<Label>,
    pub body: Enclosed<Ranged<Expression>>,
}

impl TokenizeInto for Block {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        if let Some(name) = &self.name {
            name.tokenize_into(tokens);
        }
        self.body.tokenize_into(tokens);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MapExpression {
    pub open: Ranged<Token>,
    pub fields: Delimited<MapField>,
    pub close: Ranged<Token>,
}

impl TokenizeInto for MapExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.open.clone());
        self.fields.tokenize_into(tokens);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ListExpression {
    pub open: Ranged<Token>,
    pub values: Delimited<Ranged<Expression>>,
    pub close: Ranged<Token>,
}

impl TokenizeInto for ListExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.open.clone());
        self.values.tokenize_into(tokens);
        tokens.push_back(self.close.clone());
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MapField {
    pub key: Ranged<Expression>,
    pub colon: Option<Ranged<Token>>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct Lookup {
    pub base: Option<LookupBase>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct LookupBase {
    pub expression: Ranged<Expression>,
    pub dot: Ranged<Token>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfExpression {
    pub r#if: Ranged<Token>,
    pub condition: Ranged<Expression>,
    pub then: Option<Ranged<Token>>,
    pub when_true: Ranged<Expression>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct Else {
    pub r#else: Ranged<Token>,
    pub expression: Ranged<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchExpression {
    pub r#match: Ranged<Token>,
    pub condition: Ranged<Expression>,
    pub matches: Ranged<Matches>,
}

impl TokenizeInto for MatchExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.r#match.clone());
        self.condition.tokenize_into(tokens);
        self.matches.tokenize_into(tokens);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TryOrNil {
    pub token: Ranged<Token>,
    pub body: Ranged<Expression>,
}

impl TokenizeInto for TryOrNil {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.body.tokenize_into(tokens);
        tokens.push_back(self.token.clone());
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReturnExpression {
    pub r#return: Ranged<Token>,
    pub value: Ranged<Expression>,
}

impl TokenizeInto for ReturnExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.r#return.clone());
        self.value.tokenize_into(tokens);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThrowExpression {
    pub throw: Ranged<Token>,
    pub value: Ranged<Expression>,
}

impl TokenizeInto for ThrowExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.throw.clone());
        self.value.tokenize_into(tokens);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TryExpression {
    pub r#try: Ranged<Token>,
    pub body: Ranged<Expression>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct Catch {
    pub catch: Ranged<Token>,
    pub matches: Ranged<Matches>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoopExpression {
    pub token: Ranged<Token>,
    pub kind: LoopKind,
    pub block: Ranged<Block>,
}

impl TokenizeInto for LoopExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        if let Some(label) = &self.block.0.name {
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

#[derive(Debug, Clone, PartialEq)]
pub struct BreakExpression {
    pub r#break: Ranged<Token>,
    pub name: Option<Ranged<Symbol>>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct ContinueExpression {
    pub r#continue: Ranged<Token>,
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

#[derive(Debug, Clone, PartialEq)]
pub enum LoopKind {
    Infinite,
    While(Ranged<Expression>),
    TailWhile {
        r#while: Ranged<Token>,
        expression: Ranged<Expression>,
    },
    For {
        pattern: Ranged<Pattern>,
        r#in: Ranged<Token>,
        source: Ranged<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModuleDefinition {
    pub publish: Option<Ranged<Token>>,
    pub r#mod: Ranged<Token>,
    pub name: Ranged<Symbol>,
    pub contents: Ranged<Expression>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDefinition {
    pub publish: Option<Ranged<Token>>,
    pub r#fn: Ranged<Token>,
    pub name: Option<Ranged<Symbol>>,
    pub body: Ranged<Matches>,
}

impl TokenizeInto for FunctionDefinition {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        if let Some(publish) = &self.publish {
            tokens.push_back(publish.clone());
        }
        tokens.push_back(self.r#fn.clone());
        if let Some(name) = &self.name {
            tokens.push_back(name.clone().map(Token::Identifier));
        }
        self.body.tokenize_into(tokens);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub function: Ranged<Expression>,
    pub open: Ranged<Token>,
    pub parameters: Delimited<Ranged<Expression>>,
    pub close: Ranged<Token>,
}

impl TokenizeInto for FunctionCall {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.function.tokenize_into(tokens);
        tokens.push_back(self.open.clone());
        self.parameters.tokenize_into(tokens);
        tokens.push_back(self.close.clone());
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Index {
    pub target: Ranged<Expression>,
    pub open: Ranged<Token>,
    pub parameters: Delimited<Ranged<Expression>>,
    pub close: Ranged<Token>,
}

impl TokenizeInto for Index {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.target.tokenize_into(tokens);
        tokens.push_back(self.open.clone());
        self.parameters.tokenize_into(tokens);
        tokens.push_back(self.close.clone());
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub target: Ranged<AssignTarget>,
    pub eq: Ranged<Token>,
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

#[derive(Debug, Clone, PartialEq)]
pub enum AssignTarget {
    Lookup(Lookup),
    Index(Index),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MacroInvocation {
    pub name: Ranged<Symbol>,
    pub tokens: VecDeque<Ranged<Token>>,
}

impl TokenizeInto for MacroInvocation {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.name.clone().map(Token::Sigil));
        tokens.extend(self.tokens.iter().cloned());
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InfixMacroInvocation {
    pub subject: Ranged<Expression>,
    pub invocation: MacroInvocation,
}

impl TokenizeInto for InfixMacroInvocation {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.subject.tokenize_into(tokens);
        self.invocation.tokenize_into(tokens);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnaryExpression {
    pub kind: UnaryKind,
    pub operator: Ranged<Token>,
    pub operand: Ranged<Expression>,
}

impl TokenizeInto for UnaryExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        tokens.push_back(self.operator.clone());
        self.operand.tokenize_into(tokens);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum UnaryKind {
    LogicalNot,
    BitwiseNot,
    Negate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryExpression {
    pub kind: BinaryKind,
    pub left: Ranged<Expression>,
    pub operator: Ranged<Token>,
    pub right: Ranged<Expression>,
}

impl TokenizeInto for BinaryExpression {
    fn tokenize_into(&self, tokens: &mut VecDeque<Ranged<Token>>) {
        self.left.tokenize_into(tokens);
        tokens.push_back(self.operator.clone());
        self.right.tokenize_into(tokens);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BinaryKind {
    Add,
    Subtract,
    Multiply,
    Divide,
    IntegerDivide,
    Remainder,
    Power,
    Chain,
    NilCoalesce,
    Bitwise(BitwiseKind),
    Logical(LogicalKind),
    Compare(CompareKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CompareKind {
    LessThanOrEqual,
    LessThan,
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BitwiseKind {
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRight,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LogicalKind {
    And,
    Or,
    Xor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    pub publish: Option<Ranged<Token>>,
    pub kind: Ranged<Symbol>,
    pub pattern: Ranged<Pattern>,
    pub eq: Option<Ranged<Token>>,
    pub value: Ranged<Expression>,
    pub r#else: Option<Else>,
}

impl TokenizeInto for Variable {
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

pub fn parse_tokens(source: VecDeque<Ranged<Token>>) -> Result<Ranged<Expression>, Ranged<Error>> {
    parse_from_reader(TokenReader::from(source))
}

pub fn parse<'a>(source: impl Into<SourceCode<'a>>) -> Result<Ranged<Expression>, Ranged<Error>> {
    parse_from_reader(TokenReader::new(source))
}

fn parse_from_reader(mut tokens: TokenReader<'_>) -> Result<Ranged<Expression>, Ranged<Error>> {
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
                Ok(_) | Err(Ranged(Error::UnexpectedEof, _)) => {
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
                return Err(token.map(|_| Error::ExpectedEof));
            }
            Err(Ranged(Error::UnexpectedEof, _)) => break,
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
    type Item = Result<Ranged<Token>, Ranged<token::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            TokenStream::List { tokens, .. } => tokens.pop_front().map(Ok),
            TokenStream::Tokens(tokens) => tokens.next(),
        }
    }
}

struct TokenReader<'a> {
    tokens: TokenStream<'a>,
    peeked: VecDeque<Result<Ranged<Token>, Ranged<Error>>>,
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

    fn next_or_eof(&mut self) -> Result<Ranged<Token>, Ranged<Error>> {
        self.next(Error::UnexpectedEof)
    }

    fn next(&mut self, err: Error) -> Result<Ranged<Token>, Ranged<Error>> {
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

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum Error {
    UnexpectedEof,
    ExpectedEof,
    MissingEnd(Paired),
    Token(token::Error),
    UnexpectedToken,
    ExpectedDeclaration,
    ExpectedName,
    ExpectedBlock,
    ExpectedModuleBody,
    ExpectedFunctionBody,
    ExpectedIn,
    ExpectedFunctionParameters,
    ExpectedVariableInitialValue,
    ExpectedThenOrBrace,
    ExpectedColon,
    ExpectedCommaOrBrace,
    ExpectedMatchBody,
    ExpectedPattern,
    ExpectedPatternOr(Paired),
    ExpectedFatArrow,
    ExpectedCatchBlock,
    InvalidAssignmentTarget,
    InvalidLabelTarget,
    InvalidMapKeyPattern,
}

impl crate::ErrorKind for Error {
    fn kind(&self) -> &'static str {
        match self {
            Error::UnexpectedEof => "unexpected eof",
            Error::ExpectedEof => "expected eof",
            Error::MissingEnd(_) => "missing end",
            Error::Token(err) => err.kind(),
            Error::UnexpectedToken => "unexpected token",
            Error::ExpectedDeclaration => "expected declaration",
            Error::ExpectedName => "expected name",
            Error::ExpectedBlock => "expected block",
            Error::ExpectedModuleBody => "expected module body",
            Error::ExpectedFunctionBody => "expected function body",
            Error::ExpectedIn => "expected in",
            Error::ExpectedFunctionParameters => "expected function parameters",
            Error::ExpectedVariableInitialValue => "expected variable initial value",
            Error::ExpectedThenOrBrace => "expected then or brace",
            Error::ExpectedColon => "expected colon",
            Error::ExpectedCommaOrBrace => "expected comma or closing brace",
            Error::ExpectedMatchBody => "expected match body",
            Error::ExpectedPattern => "expected pattern",
            Error::ExpectedPatternOr(_) => "expected pattern or end",
            Error::ExpectedFatArrow => "expected fat arrow",
            Error::ExpectedCatchBlock => "expected catch block",
            Error::InvalidAssignmentTarget => "invalid assignment target",
            Error::InvalidLabelTarget => "invalid label target",
            Error::InvalidMapKeyPattern => "invalid map key pattern",
        }
    }
}

impl std::error::Error for Error {}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::UnexpectedEof => f.write_str("unexpected end-of-file"),
            Error::ExpectedEof => f.write_str("expected the end of input or \";\""),
            Error::MissingEnd(kind) => write!(f, "missing closing \"{}\"", kind.as_close()),
            Error::Token(err) => Display::fmt(err, f),
            Error::UnexpectedToken => f.write_str("unexpected token"),
            Error::ExpectedDeclaration => f.write_str("expected a declaration"),
            Error::ExpectedName => f.write_str("expected a name (identifier)"),
            Error::ExpectedBlock => f.write_str("expected a block"),
            Error::ExpectedModuleBody => f.write_str("expected a module body"),
            Error::ExpectedFunctionBody => f.write_str("expected function body"),
            Error::ExpectedIn => f.write_str("expected \"in\""),
            Error::ExpectedFunctionParameters => f.write_str("expected function parameters"),
            Error::ExpectedVariableInitialValue => f.write_str("expected initial value"),
            Error::ExpectedThenOrBrace => f.write_str("expected \"then\" or \"{\""),
            Error::ExpectedColon => f.write_str("expected \":\""),
            Error::ExpectedCommaOrBrace => f.write_str("expected comma or closing brace"),
            Error::ExpectedMatchBody => f.write_str("expected match body"),
            Error::ExpectedPattern => f.write_str("expected match pattern"),
            Error::ExpectedPatternOr(paired) => {
                write!(f, "expected match pattern or {}", paired.as_close())
            }
            Error::ExpectedFatArrow => f.write_str("expected fat arrow (=>)"),
            Error::ExpectedCatchBlock => f.write_str("expected catch block"),
            Error::InvalidAssignmentTarget => f.write_str("invalid assignment target"),
            Error::InvalidLabelTarget => f.write_str("invalid label target"),
            Error::InvalidMapKeyPattern => f.write_str("invalid map key pattern"),
        }
    }
}

impl From<Ranged<token::Error>> for Ranged<Error> {
    fn from(err: Ranged<token::Error>) -> Self {
        err.map(Error::Token)
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

    fn parse(&self, tokens: &mut TokenReader<'_>) -> Result<Ranged<Expression>, Ranged<Error>> {
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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

        Err(token.map(|_| Error::UnexpectedToken))
    }

    fn parse_next(
        &self,
        tokens: &mut TokenReader<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        self.with_precedence(self.minimum_precedence + 1)
            .parse(tokens)
    }

    fn parse_expression(
        &self,
        tokens: &mut TokenReader<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        self.with_precedence(self.parselets.markers.expression)
            .parse(tokens)
    }

    fn parse_conditional(
        &self,
        tokens: &mut TokenReader<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
    ) -> Result<Ranged<Expression>, Ranged<Error>>;
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
                                return Err(token.map(|_| Error::ExpectedEof));
                            }
                            Err(Ranged(Error::UnexpectedEof, _)) => {}
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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

fn gather_macro_tokens(tokens: &mut TokenReader<'_>) -> Result<Vec<Ranged<Token>>, Ranged<Error>> {
    let Some(Token::Open(paired)) = tokens.peek_token() else {
        return Ok(Vec::new());
    };

    let mut stack = vec![paired];
    let mut contents = vec![tokens.next_or_eof()?];

    while let Some(last_open) = stack.last().copied() {
        let token = tokens.next(Error::MissingEnd(last_open))?;
        match &token.0 {
            Token::Open(next) => stack.push(*next),
            Token::Close(kind) => {
                if *kind == last_open {
                    stack.pop();
                } else {
                    return Err(token.map(|_| Error::MissingEnd(last_open)));
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
            ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
            ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
) -> Result<Ranged<Block>, Ranged<Error>> {
    let open_brace = tokens.next(Error::ExpectedBlock)?;
    if matches!(open_brace.0, Token::Open(Paired::Brace)) {
        if tokens.peek_token() == Some(Token::Close(Paired::Brace)) {
            let close_brace = tokens.next(Error::MissingEnd(Paired::Brace))?;
            return Ok(tokens.ranged(
                open_brace.range().start..,
                Block {
                    name: None,
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
                        name: None,
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
                Error::MissingEnd(Paired::Brace),
            )),
        }
    } else {
        Err(open_brace.map(|_| Error::ExpectedBlock))
    }
}

impl Braces {
    fn parse_block(
        open: Ranged<Token>,
        expr: Ranged<Expression>,
        mut semicolon: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Block>, Ranged<Error>> {
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
                    name: None,
                    body: Enclosed {
                        enclosed: left,
                        open,
                        close: token,
                    },
                },
            )),
            Ok(Ranged(_, range)) | Err(Ranged(_, range)) => {
                Err(Ranged::new(range, Error::MissingEnd(Paired::Brace)))
            }
        }
    }

    fn parse_map(
        open: Ranged<Token>,
        key: Ranged<Expression>,
        colon: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
                    let colon = match tokens.next(Error::ExpectedColon)? {
                        colon @ Ranged(Token::Char(':'), _) => colon,
                        other => return Err(other.map(|_| Error::ExpectedColon)),
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
                    .next(Error::ExpectedCommaOrBrace)?
                    .map(|_| Error::ExpectedCommaOrBrace))
            }
        }

        match tokens.next_or_eof() {
            Ok(close @ Ranged(Token::Close(Paired::Brace), _)) => Ok(tokens.ranged(
                open.range().start..,
                Expression::Map(Box::new(MapExpression {
                    open,
                    fields: values.finish(),
                    close,
                })),
            )),
            Ok(Ranged(_, range)) | Err(Ranged(_, range)) => {
                Err(Ranged::new(range, Error::MissingEnd(Paired::Brace)))
            }
        }
    }

    fn parse_set(
        open: Ranged<Token>,
        expr: Ranged<Expression>,
        comma: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
                    open,
                    fields: values.finish(),
                    close,
                })),
            )),
            Ok(Ranged(_, range)) | Err(Ranged(_, range)) => {
                Err(Ranged::new(range, Error::MissingEnd(Paired::Brace)))
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
                            open,
                            fields: Delimited::empty(),
                            close,
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
                        name: None,
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
                Error::MissingEnd(Paired::Brace),
            )),
        }
    }
}

type SeparatorAndEnd = (Option<Ranged<Token>>, Ranged<Token>);

fn parse_paired(
    end: Paired,
    separator: &Token,
    tokens: &mut TokenReader<'_>,
    mut inner: impl FnMut(Option<Ranged<Token>>, &mut TokenReader<'_>) -> Result<(), Ranged<Error>>,
) -> Result<SeparatorAndEnd, Ranged<Error>> {
    let mut ending_separator = None;
    if tokens.peek().map_or(false, |token| &token.0 == separator) {
        ending_separator = Some(tokens.next_or_eof()?);
    } else {
        while tokens
            .peek()
            .map_or(false, |token| token.0 != Token::Close(end))
        {
            inner(ending_separator, tokens)?;

            if tokens.peek_token().as_ref() == Some(separator) {
                ending_separator = Some(tokens.next_or_eof()?);
            } else {
                ending_separator = None;
                break;
            }
        }
    }

    let close = tokens.next(Error::MissingEnd(end));
    match &close {
        Ok(Ranged(Token::Close(token), _)) if token == &end => {
            Ok((ending_separator, close.expect("just matched")))
        }
        Ok(Ranged(_, range)) | Err(Ranged(_, range)) => {
            Err(Ranged::new(*range, Error::MissingEnd(end)))
        }
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let expression = config.parse_expression(tokens)?;

        let end_paren = tokens.next(Error::MissingEnd(Paired::Paren))?;
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
            Err(end_paren.map(|_| Error::MissingEnd(Paired::Paren)))
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let mut parameters = Delimited::<_, Ranged<Token>>::build_empty();

        let (_, close) = parse_paired(
            Paired::Paren,
            &Token::Char(','),
            tokens,
            |delimiter, tokens| {
                if let Some(delimiter) = delimiter {
                    parameters.set_delimiter(delimiter);
                }
                config
                    .parse_expression(tokens)
                    .map(|expr| parameters.push(expr))
            },
        )?;

        Ok(tokens.ranged(
            function.range().start..,
            Expression::Call(Box::new(FunctionCall {
                function,
                open,
                parameters: parameters.finish(),
                close,
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let mut expressions = Delimited::<_, Ranged<Token>>::build_empty();

        let (_, close) = parse_paired(
            Paired::Bracket,
            &Token::Char(','),
            tokens,
            |delimiter, tokens| {
                if let Some(delimiter) = delimiter {
                    expressions.set_delimiter(delimiter);
                }
                config
                    .parse_expression(tokens)
                    .map(|expr| expressions.push(expr))
            },
        )?;

        Ok(tokens.ranged(
            open.range().start..,
            Expression::List(Box::new(ListExpression {
                open,
                values: expressions.finish(),
                close,
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let mut parameters = Delimited::<_, Ranged<Token>>::build_empty();

        let (_, close) = parse_paired(
            Paired::Bracket,
            &Token::Char(','),
            tokens,
            |delimiter, tokens| {
                if let Some(delimiter) = delimiter {
                    parameters.set_delimiter(delimiter);
                }
                config
                    .parse_expression(tokens)
                    .map(|expr| parameters.push(expr))
            },
        )?;

        Ok(tokens.ranged(
            target.range().start..,
            Expression::Index(Box::new(Index {
                target,
                open,
                parameters: parameters.finish(),
                close,
            })),
        ))
    }
}

// impl InfixParselet for Parentheses {
//     fn parse(
//         &self,
//         function: Ranged<Expression>,
//         tokens: &mut TokenReader<'_>,
//         config: &ParserConfig<'_>,
//     ) -> Result<Ranged<Expression>, Ranged<Error>> {
//         let mut parameters = Vec::new();

//         parse_paired(Paired::Paren, &Token::Char(','), tokens, |tokens| {
//             config
//                 .parse_expression(tokens)
//                 .map(|expr| parameters.push(expr))
//         })?;

//         Ok(tokens.ranged(
//             function.range().start..,
//             Expression::Call(Box::new(FunctionCall {
//                 function,
//                 parameters,
//             })),
//         ))
//     }
// }

trait InfixParselet: Parselet {
    fn parse(
        &self,
        lhs: Ranged<Expression>,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>>;
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
            ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let condition = config.parse_expression(tokens)?;
        let brace_or_then = tokens.next(Error::ExpectedThenOrBrace)?;
        let (then, when_true) = match &brace_or_then.0 {
            Token::Open(Paired::Brace) => {
                (None, Braces.parse_prefix(brace_or_then, tokens, config)?)
            }
            Token::Identifier(ident) if ident == Symbol::then_symbol() => {
                (Some(brace_or_then), config.parse_expression(tokens)?)
            }
            _ => return Err(brace_or_then.map(|_| Error::ExpectedThenOrBrace)),
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let label = label_token.map(|token| {
            let Token::Label(token) = token else {
                unreachable!("matches filters")
            };
            token
        });

        let colon = tokens.next(Error::ExpectedColon)?;
        if colon.0 != Token::Char(':') {
            return Err(colon.map(|_| Error::ExpectedColon));
        }
        let label = Label { name: label, colon };
        let mut subject = config.parse_expression(tokens)?;
        match &mut subject.0 {
            Expression::Block(block) => block.name = Some(label),
            Expression::Loop(loop_expr) => {
                loop_expr.block.name = Some(label);
            }
            _ => return Err(subject.map(|_| Error::InvalidLabelTarget)),
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let Some(pattern) = parse_pattern(tokens, config)? else {
            return Err(tokens.ranged(tokens.last_index.., Error::ExpectedPattern));
        };

        let r#in = tokens.next(Error::ExpectedIn)?;
        if !matches!(&r#in.0, Token::Identifier(ident) if ident == Symbol::in_symbol()) {
            return Err(r#in.map(|_| Error::ExpectedIn));
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let body = config.parse_expression(tokens)?;

        let catch = if tokens.peek_token()
            == Some(Token::Identifier(Symbol::catch_symbol().clone()))
        {
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
                        return Err(tokens.ranged(tokens.last_index.., Error::ExpectedCatchBlock));
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let condition = config.parse_expression(tokens)?;

        let brace = tokens.next(Error::ExpectedMatchBody)?;
        let matches = if brace.0 == Token::Open(Paired::Brace) {
            parse_match_block_body(brace, tokens, config)?
        } else {
            return Err(brace.map(|_| Error::ExpectedMatchBody));
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        Self::parse_mod(None, token, tokens, config)
    }
}

impl Mod {
    fn parse_mod(
        publish: Option<Ranged<Token>>,
        r#mod: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let name_token = tokens.next(Error::ExpectedName)?;
        let Token::Identifier(name) = name_token.0 else {
            return Err(name_token.map(|_| Error::ExpectedName));
        };

        let brace = tokens.next(Error::ExpectedModuleBody)?;
        let contents = if brace.0 == Token::Open(Paired::Brace) {
            Braces.parse_prefix(brace, tokens, config)?
        } else {
            return Err(brace.map(|_| Error::ExpectedModuleBody));
        };

        Ok(tokens.ranged(
            r#mod.range().start..,
            Expression::Module(Box::new(ModuleDefinition {
                publish,
                r#mod,
                name: Ranged::new(name_token.1, name),
                contents,
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let keyword_token = tokens.next(Error::ExpectedDeclaration)?;
        let Token::Identifier(keyword) = &keyword_token.0 else {
            return Err(keyword_token.map(|_| Error::ExpectedDeclaration));
        };

        if keyword == Symbol::fn_symbol() {
            Fn::parse_function(Some(pub_token), keyword_token, tokens, config)
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
        } else {
            return Err(keyword_token.map(|_| Error::ExpectedDeclaration));
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
        publish: Option<Ranged<Token>>,
        r#fn: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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

                return Ok(tokens.ranged(
                    r#fn.range().start..,
                    Expression::Function(Box::new(FunctionDefinition {
                        publish,
                        r#fn,
                        name,
                        body,
                    })),
                ));
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

        let body_indicator = tokens.next(Error::ExpectedFunctionBody)?;
        let (arrow, body) = match &body_indicator.0 {
            Token::Open(Paired::Brace) => {
                let arrow = tokens.ranged(
                    pattern.range().end()..pattern.range().end(),
                    Token::FatArrow,
                );
                (arrow, Braces.parse_prefix(body_indicator, tokens, config)?)
            }
            Token::FatArrow => (body_indicator, config.parse_expression(tokens)?),
            _ => return Err(body_indicator.map(|_| Error::ExpectedFunctionBody)),
        };

        Ok(tokens.ranged(
            r#fn.range().start..,
            Expression::Function(Box::new(FunctionDefinition {
                publish,
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
            })),
        ))
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        Self::parse_function(None, token, tokens, config)
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Matches {
    pub open_close: Option<[Ranged<Token>; 2]>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct MatchPattern {
    pub pattern: Ranged<Pattern>,
    pub arrow: Ranged<Token>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    pub kind: Ranged<PatternKind>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct Enclosed<T> {
    pub open: Ranged<Token>,
    pub close: Ranged<Token>,
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

#[derive(Debug, Clone, PartialEq)]
pub enum PatternKind {
    Any(Option<Symbol>),
    AnyRemaining,
    Literal(Literal),
    DestructureTuple(Box<Enclosed<Delimited<Ranged<PatternKind>>>>),
    DestructureMap(Box<Enclosed<Delimited<Ranged<EntryPattern>>>>),
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

#[derive(Debug, Clone, PartialEq)]
pub struct EntryPattern {
    pub key: Ranged<EntryKeyPattern>,
    pub colon: Ranged<Token>,
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

#[derive(Debug, Clone, PartialEq)]
pub enum EntryKeyPattern {
    Nil,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    String(Symbol),
    Identifier(Symbol),
}

fn parse_pattern(
    tokens: &mut TokenReader<'_>,
    config: &ParserConfig<'_>,
) -> Result<Option<Ranged<Pattern>>, Ranged<Error>> {
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
) -> Result<Option<Ranged<PatternKind>>, Ranged<Error>> {
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
            return Err(tokens.ranged(tokens.last_index.., Error::ExpectedPattern));
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
) -> Result<Ranged<PatternKind>, Ranged<Error>> {
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

    let closing_brace = tokens.next(Error::ExpectedPatternOr(kind))?;
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
        Err(closing_brace.map(|_| Error::ExpectedPatternOr(kind)))
    }
}

fn parse_map_destructure_pattern(
    open: Ranged<Token>,
    tokens: &mut TokenReader<'_>,
) -> Result<Ranged<PatternKind>, Ranged<Error>> {
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
                _ => return Err(Ranged::new(token.1, Error::InvalidMapKeyPattern)),
            };

            let colon = tokens.next(Error::ExpectedColon)?;
            if colon.0 != Token::Char(':') {
                return Err(colon.map(|_| Error::ExpectedColon));
            }

            let Some(value) = parse_pattern_kind(tokens)? else {
                return Err(tokens.ranged(tokens.last_index.., Error::ExpectedPattern));
            };
            entries.push(tokens.ranged(key.range().start.., EntryPattern { key, colon, value }));

            if tokens.peek_token() == Some(Token::Char(',')) {
                entries.set_delimiter(tokens.next_or_eof()?);
            } else {
                break;
            }
        }
    }

    let end_brace = tokens.next(Error::MissingEnd(Paired::Brace))?;
    if end_brace.0 != Token::Close(Paired::Brace) {
        return Err(end_brace.map(|_| Error::MissingEnd(Paired::Brace)));
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
) -> Result<Ranged<Matches>, Ranged<Error>> {
    let mut patterns = Delimited::<_, Ranged<Token>>::build_empty();
    while let Some(pattern) = parse_pattern(tokens, config)? {
        let arrow_or_brace = tokens.next(Error::ExpectedPatternOr(Paired::Brace))?;
        let (arrow, body) = match &arrow_or_brace.0 {
            Token::FatArrow => (arrow_or_brace, config.parse_expression(tokens)?),
            Token::Open(Paired::Brace) => {
                let arrow = arrow_or_brace.map(|_| Token::FatArrow);

                (
                    arrow,
                    parse_block(tokens, config)?.map(|block| Expression::Block(Box::new(block))),
                )
            }
            _ => return Err(arrow_or_brace.map(|_| Error::ExpectedPatternOr(Paired::Brace))),
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

    let closing_brace = tokens.next(Error::ExpectedPatternOr(Paired::Brace))?;
    if closing_brace.0 == Token::Close(Paired::Brace) {
        Ok(tokens.ranged(
            open.range().start..,
            Matches {
                open_close: Some([open, closing_brace]),
                patterns: patterns.finish(),
            },
        ))
    } else {
        Err(closing_brace.map(|_| Error::ExpectedPatternOr(Paired::Brace)))
    }
}

fn parse_variable(
    kind: Ranged<Symbol>,
    publish: Option<Ranged<Token>>,
    start: usize,
    tokens: &mut TokenReader<'_>,
    config: &ParserConfig<'_>,
) -> Result<Ranged<Expression>, Ranged<Error>> {
    let Some(pattern) = parse_pattern(tokens, config)? else {
        return Err(tokens.ranged(tokens.last_index.., Error::ExpectedPattern));
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
        Expression::Variable(Box::new(Variable {
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let name_token = tokens.next(Error::ExpectedName)?;
        let Token::Identifier(name) = name_token.0 else {
            return Err(Ranged::new(name_token.1, Error::ExpectedName));
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
            _ => Err(lhs.map(|_| Error::InvalidAssignmentTarget)),
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
    ]);
    parser.push_prefix(parselets![Term]);
    parser
}
