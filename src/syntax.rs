use std::collections::VecDeque;
use std::fmt::{Debug, Display};
use std::num::NonZeroUsize;
use std::ops::{Bound, Deref, DerefMut, Range, RangeBounds, RangeInclusive};

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use self::token::{Paired, RegexLiteral, Token, Tokens};
use crate::symbol::Symbol;
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

#[derive(Default)]
pub struct Sources(Vec<String>);

impl Sources {
    pub fn push(&mut self, source: impl Into<String>) -> SourceCode<'_> {
        let id = self.next_id();
        self.0.push(source.into());
        SourceCode::new(self.0.last().expect("just pushed"), id)
    }

    #[must_use]
    pub fn get(&self, id: SourceId) -> Option<&String> {
        let index = id.0?.get() - 1;
        self.0.get(index)
    }

    #[must_use]
    pub fn next_id(&self) -> SourceId {
        SourceId::new(NonZeroUsize::new(self.0.len() + 1).expect("always > 0"))
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
pub enum Expression {
    RootModule,
    Literal(Literal),
    Lookup(Box<Lookup>),
    If(Box<IfExpression>),
    Match(Box<MatchExpression>),
    Try(Box<TryExpression>),
    TryOrNil(Box<TryOrNil>),
    Throw(Box<Ranged<Expression>>),
    Map(Box<MapExpression>),
    List(Vec<Ranged<Expression>>),
    Call(Box<FunctionCall>),
    Index(Box<Index>),
    Assign(Box<Assignment>),
    Unary(Box<UnaryExpression>),
    Binary(Box<BinaryExpression>),
    Block(Box<Block>),
    Loop(Box<LoopExpression>),
    Break(Box<BreakExpression>),
    Continue(Box<ContinueExpression>),
    Return(Box<Ranged<Expression>>),
    Module(Box<ModuleDefinition>),
    Function(Box<FunctionDefinition>),
    Variable(Box<Variable>),
    Macro(Box<MacroInvocation>),
}

impl Default for Expression {
    fn default() -> Self {
        Self::Literal(Literal::Nil)
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
    String(String),
    Symbol(Symbol),
    Regex(RegexLiteral),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Chain(pub Ranged<Expression>, pub Ranged<Expression>);

impl Chain {
    #[must_use]
    pub fn from_expressions(mut expressions: Vec<Ranged<Expression>>) -> Ranged<Expression> {
        let Some(mut expression) = expressions.pop() else {
            return Ranged::new(
                (SourceId::anonymous(), 0..0),
                Expression::Literal(Literal::Nil),
            );
        };

        while let Some(previous) = expressions.pop() {
            expression = Ranged::new(
                previous.range().with_end(expression.range().end()),
                Expression::Binary(Box::new(BinaryExpression {
                    kind: BinaryKind::Chain,
                    left: previous,
                    right: expression,
                })),
            );
        }

        expression
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub name: Option<Symbol>,
    pub body: Ranged<Expression>,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct MapExpression {
    pub fields: Vec<MapField>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MapField {
    pub key: Ranged<Expression>,
    pub value: Ranged<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lookup {
    pub base: Option<Ranged<Expression>>,
    pub name: Symbol,
}

impl From<Symbol> for Lookup {
    fn from(name: Symbol) -> Self {
        Self { name, base: None }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfExpression {
    pub condition: Ranged<Expression>,
    pub when_true: Ranged<Expression>,
    pub when_false: Option<Ranged<Expression>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchExpression {
    pub condition: Ranged<Expression>,
    pub matches: Ranged<Matches>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TryOrNil {
    pub body: Ranged<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TryExpression {
    pub body: Ranged<Expression>,
    pub catch: Option<Ranged<Matches>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoopExpression {
    pub kind: LoopKind,
    pub block: Ranged<Block>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BreakExpression {
    pub name: Option<Ranged<Symbol>>,
    pub value: Ranged<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContinueExpression {
    pub name: Option<Ranged<Symbol>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoopKind {
    Infinite,
    While(Ranged<Expression>),
    TailWhile(Ranged<Expression>),
    For {
        pattern: Ranged<Pattern>,
        source: Ranged<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModuleDefinition {
    pub publish: bool,
    pub name: Ranged<Symbol>,
    pub contents: Ranged<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDefinition {
    pub publish: bool,
    pub name: Option<Ranged<Symbol>>,
    pub body: Ranged<Matches>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub function: Ranged<Expression>,
    pub parameters: Vec<Ranged<Expression>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Index {
    pub target: Ranged<Expression>,
    pub parameters: Vec<Ranged<Expression>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub target: Ranged<AssignTarget>,
    pub value: Ranged<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignTarget {
    Lookup(Lookup),
    Index(Index),
}

#[derive(Debug, Clone, PartialEq)]
pub struct MacroInvocation {
    pub name: Symbol,
    pub tokens: VecDeque<Ranged<Token>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnaryExpression {
    pub kind: UnaryKind,
    pub operand: Ranged<Expression>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum UnaryKind {
    LogicalNot,
    BitwiseNot,
    Negate,
    Copy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryExpression {
    pub kind: BinaryKind,
    pub left: Ranged<Expression>,
    pub right: Ranged<Expression>,
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
    pub publish: bool,
    pub mutable: bool,
    pub pattern: Ranged<Pattern>,
    pub value: Ranged<Expression>,
    pub r#else: Option<Ranged<Expression>>,
}

pub fn parse_tokens(source: VecDeque<Ranged<Token>>) -> Result<Ranged<Expression>, Ranged<Error>> {
    parse_from_reader(TokenReader::from(source))
}

pub fn parse(source: &SourceCode<'_>) -> Result<Ranged<Expression>, Ranged<Error>> {
    parse_from_reader(TokenReader::new(source))
}

fn parse_from_reader(mut tokens: TokenReader<'_>) -> Result<Ranged<Expression>, Ranged<Error>> {
    let parselets = parselets();
    let config = ParserConfig {
        parselets: &parselets,
        minimum_precedence: 0,
    };
    let mut results = Vec::new();
    loop {
        if tokens.peek().is_none() {
            // Peeking an error returns None
            match tokens.next() {
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
        match tokens.next() {
            Ok(token) if token.0 == Token::Char(';') => {}
            Ok(token) => {
                return Err(token.map(|_| Error::ExpectedEof));
            }
            Err(Ranged(Error::UnexpectedEof, _)) => break,
            Err(other) => return Err(other),
        }
    }

    Ok(Chain::from_expressions(results))
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
    pub fn new(source: &SourceCode<'a>) -> Self {
        Self {
            tokens: TokenStream::Tokens(
                Tokens::new(source)
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

    fn next(&mut self) -> Result<Ranged<Token>, Ranged<Error>> {
        let token = if let Some(peeked) = self.peeked.pop_front() {
            peeked?
        } else {
            self.tokens.next().ok_or_else(|| {
                self.ranged(self.last_index..self.last_index, Error::UnexpectedEof)
            })??
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
    ExpectedMatchBody,
    ExpectedPattern,
    ExpectedPatternOr(Paired),
    ExpectedFatArrow,
    ExpectedCatchBlock,
    InvalidAssignmentTarget,
    InvalidLabelTarget,
    InvalidMapKeyPattern,
}

impl crate::Error for Error {
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::UnexpectedEof => f.write_str("unexpected end-of-file"),
            Error::ExpectedEof => f.write_str("expected the end of input"),
            Error::MissingEnd(kind) => write!(f, "missing closing {}", kind.as_close()),
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
                        tokens.next()?;
                        lhs = parselet.parse(
                            lhs,
                            &possible_operator,
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
        let token = tokens.next()?;
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
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let name = if tokens
            .peek_token()
            .map_or(false, |token| matches!(token, Token::Label(_)))
        {
            let label_token = tokens.next()?;
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
            token.range().start..,
            Expression::Break(Box::new(BreakExpression { name, value })),
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
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        _config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let name = if tokens
            .peek_token()
            .map_or(false, |token| matches!(token, Token::Label(_)))
        {
            let label_token = tokens.next()?;
            let Token::Label(label) = label_token.0 else {
                unreachable!("just matched")
            };
            Some(Ranged::new(label_token.1, label))
        } else {
            None
        };

        Ok(tokens.ranged(
            token.range().start..,
            Expression::Continue(Box::new(ContinueExpression { name })),
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
        token: Ranged<Token>,
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

        Ok(tokens.ranged(token.range().start.., Expression::Return(Box::new(value))))
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
                Expression::Lookup(Box::new(Lookup::from(value))),
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
                            name: sym,
                            tokens: VecDeque::from(contents),
                        })),
                    ))
                }
            }
            Token::FormatString(format_string) => {
                let mut left = Ranged::new(
                    token.1,
                    Expression::Literal(Literal::String(format_string.initial)),
                );

                // TODO this should use a var-arg join operation to avoid extra
                // allocations
                for part in format_string.parts {
                    let mut reader = TokenReader::from(VecDeque::from(part.expression));
                    let right = config.parse_expression(&mut reader)?;

                    // Ensure the expression was fully consumed
                    match reader.next() {
                        Ok(token) => {
                            return Err(token.map(|_| Error::ExpectedEof));
                        }
                        Err(Ranged(Error::UnexpectedEof, _)) => {}
                        Err(other) => return Err(other),
                    }

                    left = tokens.ranged(
                        left.range().start..right.range().end(),
                        Expression::Binary(Box::new(BinaryExpression {
                            left,
                            right,
                            kind: BinaryKind::Add,
                        })),
                    );

                    let right = part
                        .suffix
                        .map(|suffix| Expression::Literal(Literal::String(suffix)));

                    left = tokens.ranged(
                        left.range().start..right.range().end(),
                        Expression::Binary(Box::new(BinaryExpression {
                            left,
                            right,
                            kind: BinaryKind::Add,
                        })),
                    );
                }

                Ok(left)
            }
            _ => unreachable!("parse called with invalid token"),
        }
    }
}

fn gather_macro_tokens(tokens: &mut TokenReader<'_>) -> Result<Vec<Ranged<Token>>, Ranged<Error>> {
    let Some(Token::Open(paired)) = tokens.peek_token() else {
        return Ok(Vec::new());
    };

    let mut stack = vec![paired];
    let mut contents = vec![tokens.next()?];

    while let Some(last_open) = stack.last().copied() {
        let token = tokens.next()?;
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
    let open_brace = tokens.next()?;
    if matches!(open_brace.0, Token::Open(Paired::Brace)) {
        if tokens.peek_token() == Some(Token::Close(Paired::Brace)) {
            let close_brace = tokens.next()?;
            return Ok(tokens.ranged(
                open_brace.range().start..,
                Block {
                    name: None,
                    body: Ranged::new(
                        (
                            open_brace.range().source_id,
                            open_brace.range().end()..close_brace.range().start,
                        ),
                        Expression::Literal(Literal::Nil),
                    ),
                },
            ));
        }
        let expr = config.parse_expression(tokens)?;

        match tokens.peek() {
            Some(Ranged(Token::Char(';'), _)) => {
                tokens.next()?;
                Braces::parse_block(open_brace.range().start, expr, tokens, config)
            }
            Some(Ranged(Token::Close(Paired::Brace), _)) => {
                tokens.next()?;
                Ok(tokens.ranged(
                    open_brace.range().start..,
                    Block {
                        name: None,
                        body: expr,
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
        start: usize,
        expr: Ranged<Expression>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Block>, Ranged<Error>> {
        let mut expressions = vec![expr];

        let mut ended_in_semicolon = true;

        while tokens
            .peek()
            .map_or(false, |token| token.0 != Token::Close(Paired::Brace))
        {
            expressions.push(config.parse_expression(tokens)?);

            if tokens.peek_token() == Some(Token::Char(';')) {
                tokens.next()?;
                ended_in_semicolon = true;
            } else {
                ended_in_semicolon = false;
                break;
            }
        }

        if ended_in_semicolon {
            expressions.push(tokens.ranged(tokens.last_index.., Expression::Literal(Literal::Nil)));
        }

        match tokens.next() {
            Ok(Ranged(Token::Close(Paired::Brace), _)) => Ok(tokens.ranged(
                start..,
                Block {
                    name: None,
                    body: Chain::from_expressions(expressions),
                },
            )),
            Ok(Ranged(_, range)) | Err(Ranged(_, range)) => {
                Err(Ranged::new(range, Error::MissingEnd(Paired::Brace)))
            }
        }
    }

    fn parse_map(
        start: usize,
        key: Ranged<Expression>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let value = config.parse_expression(tokens)?;
        let mut values = vec![MapField { key, value }];

        if tokens.peek_token() == Some(Token::Char(',')) {
            tokens.next()?;
        }

        while tokens
            .peek()
            .map_or(false, |token| token.0 != Token::Close(Paired::Brace))
        {
            let key = config.parse_expression(tokens)?;
            match tokens.next()? {
                Ranged(Token::Char(':'), _) => {}
                other => return Err(other.map(|_| Error::ExpectedColon)),
            }
            let value = config.parse_expression(tokens)?;
            values.push(MapField { key, value });

            if tokens.peek_token() == Some(Token::Char(',')) {
                tokens.next()?;
            } else {
                break;
            }
        }

        match tokens.next() {
            Ok(Ranged(Token::Close(Paired::Brace), _)) => Ok(tokens.ranged(
                start..,
                Expression::Map(Box::new(MapExpression { fields: values })),
            )),
            Ok(Ranged(_, range)) | Err(Ranged(_, range)) => {
                Err(Ranged::new(range, Error::MissingEnd(Paired::Brace)))
            }
        }
    }

    fn parse_set(
        start: usize,
        expr: Ranged<Expression>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let mut values = vec![MapField {
            key: expr.clone(),
            value: expr,
        }];

        while tokens
            .peek()
            .map_or(false, |token| token.0 != Token::Close(Paired::Brace))
        {
            let expr = config.parse_expression(tokens)?;
            values.push(MapField {
                key: expr.clone(),
                value: expr,
            });

            if tokens.peek_token() == Some(Token::Char(',')) {
                tokens.next()?;
            } else {
                break;
            }
        }

        match tokens.next() {
            Ok(Ranged(Token::Close(Paired::Brace), _)) => Ok(tokens.ranged(
                start..,
                Expression::Map(Box::new(MapExpression { fields: values })),
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
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        match tokens.peek_token() {
            Some(Token::Close(Paired::Brace)) => {
                tokens.next()?;
                return Ok(tokens.ranged(token.range().start.., Expression::Literal(Literal::Nil)));
            }
            Some(Token::Char(',')) => {
                tokens.next()?;
                if tokens.peek_token() == Some(Token::Close(Paired::Brace)) {
                    tokens.next()?;
                    return Ok(
                        tokens.ranged(token.range().start.., Expression::Map(Box::default()))
                    );
                }
            }
            _ => {}
        }
        let expr = config.parse_expression(tokens)?;

        match tokens.peek() {
            Some(Ranged(Token::Char(':'), _)) => {
                tokens.next()?;
                Self::parse_map(token.range().start, expr, tokens, config)
            }
            Some(Ranged(Token::Char(','), _)) => {
                tokens.next()?;
                Self::parse_set(token.range().start, expr, tokens, config)
            }
            Some(Ranged(Token::Char(';'), _)) => {
                tokens.next()?;
                Self::parse_block(token.range().start, expr, tokens, config)
                    .map(|ranged| ranged.map(|block| Expression::Block(Box::new(block))))
            }
            Some(Ranged(Token::Close(Paired::Brace), _)) => {
                tokens.next()?;
                Ok(tokens.ranged(
                    token.range().start..,
                    Expression::Block(Box::new(Block {
                        name: None,
                        body: expr,
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

fn parse_paired(
    end: Paired,
    separator: &Token,
    tokens: &mut TokenReader<'_>,
    mut inner: impl FnMut(&mut TokenReader<'_>) -> Result<(), Ranged<Error>>,
) -> Result<bool, Ranged<Error>> {
    let mut ended_in_separator = false;
    if tokens.peek().map_or(false, |token| &token.0 == separator) {
        tokens.next()?;
        ended_in_separator = true;
    } else {
        while tokens
            .peek()
            .map_or(false, |token| token.0 != Token::Close(end))
        {
            inner(tokens)?;

            if tokens.peek_token().as_ref() == Some(separator) {
                tokens.next()?;
                ended_in_separator = true;
            } else {
                ended_in_separator = false;
                break;
            }
        }
    }

    match tokens.next() {
        Ok(Ranged(Token::Close(token), _)) if token == end => Ok(ended_in_separator),
        Ok(Ranged(_, range)) | Err(Ranged(_, range)) => {
            Err(Ranged::new(range, Error::MissingEnd(end)))
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
        _token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let expression = config.parse_expression(tokens)?;

        let end_paren = tokens.next()?;
        if end_paren.0 == Token::Close(Paired::Paren) {
            Ok(expression)
        } else {
            Err(end_paren.map(|_| Error::MissingEnd(Paired::Paren)))
        }
    }
}

impl InfixParselet for Parentheses {
    fn parse(
        &self,
        function: Ranged<Expression>,
        _token: &Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let mut parameters = Vec::new();

        parse_paired(Paired::Paren, &Token::Char(','), tokens, |tokens| {
            config
                .parse_expression(tokens)
                .map(|expr| parameters.push(expr))
        })?;

        Ok(tokens.ranged(
            function.range().start..,
            Expression::Call(Box::new(FunctionCall {
                function,
                parameters,
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
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let mut expressions = Vec::new();

        parse_paired(Paired::Bracket, &Token::Char(','), tokens, |tokens| {
            config
                .parse_expression(tokens)
                .map(|expr| expressions.push(expr))
        })?;

        Ok(tokens.ranged(token.range().start.., Expression::List(expressions)))
    }
}

impl InfixParselet for Brackets {
    fn parse(
        &self,
        target: Ranged<Expression>,
        _token: &Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let mut parameters = Vec::new();

        parse_paired(Paired::Bracket, &Token::Char(','), tokens, |tokens| {
            config
                .parse_expression(tokens)
                .map(|expr| parameters.push(expr))
        })?;

        Ok(tokens.ranged(
            target.range().start..,
            Expression::Index(Box::new(Index { target, parameters })),
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
        token: &Ranged<Token>,
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
                _token: &Ranged<Token>,
                tokens: &mut TokenReader<'_>,
                config: &ParserConfig<'_>,
            ) -> Result<Ranged<Expression>, Ranged<Error>> {
                let right = config.parse_next(tokens)?;
                Ok(tokens.ranged(
                    left.range().start..right.range().end(),
                    Expression::Binary(Box::new(BinaryExpression {
                        kind: $binarykind,
                        left,
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
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let condition = config.parse_expression(tokens)?;
        let brace_or_then = tokens.next()?;
        let when_true = match &brace_or_then.0 {
            Token::Open(Paired::Brace) => Braces.parse_prefix(brace_or_then, tokens, config)?,
            Token::Identifier(ident) if ident == Symbol::then_symbol() => {
                config.parse_expression(tokens)?
            }
            _ => return Err(brace_or_then.map(|_| Error::ExpectedThenOrBrace)),
        };
        let when_false =
            if tokens.peek_token() == Some(Token::Identifier(Symbol::else_symbol().clone())) {
                tokens.next()?;
                Some(match tokens.peek_token() {
                    Some(Token::Identifier(ident)) if ident == *Symbol::if_symbol() => {
                        Self.parse_prefix(tokens.next()?, tokens, config)?
                    }
                    Some(Token::Open(Paired::Brace)) => {
                        Braces.parse_prefix(tokens.next()?, tokens, config)?
                    }
                    _ => config.parse_expression(tokens)?,
                })
            } else {
                None
            };

        Ok(tokens.ranged(
            token.range().start..,
            Expression::If(Box::new(IfExpression {
                condition,
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
        _token: &Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let condition = config.parse_next(tokens)?;
        let else_expr = if tokens.peek_token().map_or(false, |token| {
            token == Token::Identifier(Symbol::else_symbol().clone())
        }) {
            tokens.next()?;
            Some(config.parse_expression(tokens)?)
        } else {
            None
        };

        Ok(tokens.ranged(
            lhs.range().start..,
            Expression::If(Box::new(IfExpression {
                condition,
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
        label: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let Token::Label(label) = label.0 else {
            unreachable!("matches filters")
        };
        let _colon = tokens.next()?;
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
            let _while = tokens.next()?;
            LoopKind::TailWhile(config.parse_expression(tokens)?)
        } else {
            LoopKind::Infinite
        };

        Ok(tokens.ranged(
            token.range().start..,
            Expression::Loop(Box::new(LoopExpression { kind, block })),
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

        let in_token = tokens.next()?;
        if !matches!(&in_token.0, Token::Identifier(ident) if ident == Symbol::in_symbol()) {
            return Err(in_token.map(|_| Error::ExpectedIn));
        }

        let source = config.parse_expression(tokens)?;

        let body = parse_block(tokens, config)?;

        Ok(tokens.ranged(
            for_token.range().start..,
            Expression::Loop(Box::new(LoopExpression {
                kind: LoopKind::For { pattern, source },
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
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let body = config.parse_expression(tokens)?;

        let catch =
            if tokens.peek_token() == Some(Token::Identifier(Symbol::catch_symbol().clone())) {
                let catch_token = tokens.next()?;
                match tokens.peek_token() {
                    Some(Token::Open(Paired::Brace)) => {
                        // Match catch
                        let open_brace = tokens.next()?;

                        Some(parse_match_block_body(open_brace.1.start, tokens, config)?)
                    }
                    Some(Token::FatArrow) => {
                        tokens.next()?;
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

                        Some(tokens.ranged(
                            pattern.range().start..,
                            Matches {
                                patterns: vec![tokens.ranged(
                                    pattern.range().start..,
                                    MatchPattern { pattern, body },
                                )],
                            },
                        ))
                    }
                    _ => {
                        // Inline binding
                        let Some(pattern) = parse_pattern(tokens, config)? else {
                            return Err(
                                tokens.ranged(tokens.last_index.., Error::ExpectedCatchBlock)
                            );
                        };
                        let body = parse_block(tokens, config)?;

                        Some(tokens.ranged(
                            pattern.range().start..,
                            Matches {
                                patterns: vec![tokens.ranged(
                                    pattern.range().start..,
                                    MatchPattern {
                                        pattern,
                                        body: body.map(|block| Expression::Block(Box::new(block))),
                                    },
                                )],
                            },
                        ))
                    }
                }
            } else {
                None
            };

        Ok(tokens.ranged(
            token.range().start..,
            Expression::Try(Box::new(TryExpression { body, catch })),
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
        token: Ranged<Token>,
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

        Ok(tokens.ranged(token.range().start.., Expression::Throw(Box::new(value))))
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
        match_token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let condition = config.parse_expression(tokens)?;

        let brace = tokens.next()?;
        let matches = if brace.0 == Token::Open(Paired::Brace) {
            parse_match_block_body(brace.range().start, tokens, config)?
        } else {
            return Err(brace.map(|_| Error::ExpectedMatchBody));
        };

        Ok(tokens.ranged(
            match_token.range().start..,
            Expression::Match(Box::new(MatchExpression { condition, matches })),
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
        Self::parse_mod(false, &token, tokens, config)
    }
}

impl Mod {
    fn parse_mod(
        publish: bool,
        token: &Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let name_token = tokens.next()?;
        let Token::Identifier(name) = name_token.0 else {
            return Err(name_token.map(|_| Error::ExpectedName));
        };

        let brace = tokens.next()?;
        let contents = if brace.0 == Token::Open(Paired::Brace) {
            Braces.parse_prefix(brace, tokens, config)?
        } else {
            return Err(brace.map(|_| Error::ExpectedModuleBody));
        };

        Ok(tokens.ranged(
            token.range().start..,
            Expression::Module(Box::new(ModuleDefinition {
                publish,
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
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let keyword_token = tokens.next()?;
        let Token::Identifier(keyword) = &keyword_token.0 else {
            return Err(keyword_token.map(|_| Error::ExpectedDeclaration));
        };

        if keyword == Symbol::fn_symbol() {
            Fn::parse_function(true, keyword_token.range().start, tokens, config)
        } else if keyword == Symbol::let_symbol() {
            parse_variable(false, true, token.range().start, tokens, config)
        } else if keyword == Symbol::var_symbol() {
            parse_variable(true, true, token.range().start, tokens, config)
        } else if keyword == Symbol::mod_symbol() {
            Mod::parse_mod(true, &token, tokens, config)
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
        publish: bool,
        start: usize,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let name = if let Some(Token::Identifier(name)) = tokens.peek_token() {
            Some(tokens.next()?.map(|_| name))
        } else {
            None
        };

        let pattern: Ranged<Pattern> = match tokens.peek_token() {
            Some(Token::Open(Paired::Paren)) => {
                let start = tokens.next()?;

                parse_tuple_destructure_pattern(start.range().start, Paired::Paren, tokens)?.into()
            }
            Some(Token::Open(Paired::Brace)) => {
                // Pattern/overloaded function.
                let brace = tokens.next()?;
                let body = parse_match_block_body(brace.range().start, tokens, config)?;

                return Ok(tokens.ranged(
                    start..,
                    Expression::Function(Box::new(FunctionDefinition {
                        publish,
                        name,
                        body,
                    })),
                ));
            }
            _ => tokens
                .ranged(
                    tokens.last_index..,
                    PatternKind::DestructureTuple(Vec::new()),
                )
                .into(),
        };

        let body_indicator = tokens.next()?;
        let body = match &body_indicator.0 {
            Token::Open(Paired::Brace) => Braces.parse_prefix(body_indicator, tokens, config)?,
            Token::FatArrow => config.parse_expression(tokens)?,
            _ => return Err(body_indicator.map(|_| Error::ExpectedFunctionBody)),
        };

        Ok(tokens.ranged(
            start..,
            Expression::Function(Box::new(FunctionDefinition {
                publish,
                name,
                body: tokens.ranged(
                    pattern.range().start..,
                    Matches {
                        patterns: vec![
                            tokens.ranged(pattern.range().start.., MatchPattern { pattern, body })
                        ],
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
        Self::parse_function(false, token.range().start, tokens, config)
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Matches {
    pub patterns: Vec<Ranged<MatchPattern>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchPattern {
    pub pattern: Ranged<Pattern>,
    pub body: Ranged<Expression>,
}

impl MatchPattern {
    #[must_use]
    pub(crate) fn arity(&self) -> Option<(u8, bool)> {
        match &self.pattern.kind.0 {
            PatternKind::Any(None) | PatternKind::AnyRemaining => Some((0, true)),
            PatternKind::Any(_)
            | PatternKind::Literal(_)
            | PatternKind::Or(_, _)
            | PatternKind::DestructureMap(_) => Some((1, false)),
            PatternKind::DestructureTuple(fields) => {
                let variable = fields
                    .last()
                    .map_or(false, |field| matches!(&field.0, PatternKind::AnyRemaining));
                fields.len().try_into().ok().map(|count| (count, variable))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    pub kind: Ranged<PatternKind>,
    pub guard: Option<Ranged<Expression>>,
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
pub enum PatternKind {
    Any(Option<Symbol>),
    AnyRemaining,
    Literal(Literal),
    DestructureTuple(Vec<Ranged<PatternKind>>),
    DestructureMap(Vec<Ranged<EntryPattern>>),
    Or(Box<Ranged<PatternKind>>, Box<Ranged<PatternKind>>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntryPattern {
    pub key: Ranged<EntryKeyPattern>,
    pub value: Ranged<PatternKind>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EntryKeyPattern {
    Nil,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    String(String),
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
        tokens.next()?;
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
            tokens.next()?;
            indicator.map(|_| PatternKind::Any(None))
        }
        Token::Ellipses => {
            tokens.next()?;
            indicator.map(|_| PatternKind::AnyRemaining)
        }
        Token::Identifier(name) if name == Symbol::true_symbol() => {
            tokens.next()?;
            Ranged::new(indicator.range(), PatternKind::Literal(Literal::Bool(true)))
        }
        Token::Identifier(name) if name == Symbol::false_symbol() => {
            tokens.next()?;
            Ranged::new(
                indicator.range(),
                PatternKind::Literal(Literal::Bool(false)),
            )
        }
        Token::Identifier(name) if name == Symbol::nil_symbol() => {
            tokens.next()?;
            Ranged::new(indicator.range(), PatternKind::Literal(Literal::Nil))
        }
        Token::Identifier(name) => {
            tokens.next()?;
            Ranged::new(indicator.range(), PatternKind::Any(Some(name.clone())))
        }
        Token::Symbol(name) => {
            tokens.next()?;
            Ranged::new(
                indicator.range(),
                PatternKind::Literal(Literal::Symbol(name.clone())),
            )
        }
        Token::Int(value) => {
            tokens.next()?;
            Ranged::new(
                indicator.range(),
                PatternKind::Literal(Literal::Int(*value)),
            )
        }
        Token::UInt(value) => {
            tokens.next()?;
            Ranged::new(
                indicator.range(),
                PatternKind::Literal(Literal::UInt(*value)),
            )
        }
        Token::Float(value) => {
            tokens.next()?;
            Ranged::new(
                indicator.range(),
                PatternKind::Literal(Literal::Float(*value)),
            )
        }
        Token::Regex(_) => {
            let Ok(Ranged(Token::Regex(regex), _)) = tokens.next() else {
                unreachable!("just peeked")
            };
            Ranged::new(
                indicator.range(),
                PatternKind::Literal(Literal::Regex(regex)),
            )
        }
        Token::String(_) => {
            let Ok(Ranged(Token::String(string), _)) = tokens.next() else {
                unreachable!("just peeked")
            };
            Ranged::new(
                indicator.range(),
                PatternKind::Literal(Literal::String(string)),
            )
        }
        Token::Open(Paired::Bracket) => {
            tokens.next()?;
            parse_tuple_destructure_pattern(indicator.range().start, Paired::Bracket, tokens)?
        }
        Token::Open(Paired::Brace) => {
            tokens.next()?;

            parse_map_destructure_pattern(indicator.range().start, tokens)?
        }
        _ => return Ok(None),
    };

    while tokens.peek_token() == Some(Token::Char('|')) {
        tokens.next()?;
        let Some(rhs) = parse_pattern_kind(tokens)? else {
            return Err(tokens.ranged(tokens.last_index.., Error::ExpectedPattern));
        };
        pattern = tokens.ranged(
            pattern.range().start..,
            PatternKind::Or(Box::new(pattern), Box::new(rhs)),
        );
    }

    Ok(Some(pattern))
}

fn parse_tuple_destructure_pattern(
    start: usize,
    kind: Paired,
    tokens: &mut TokenReader<'_>,
) -> Result<Ranged<PatternKind>, Ranged<Error>> {
    let mut patterns = Vec::new();
    while let Some(pattern) = parse_pattern_kind(tokens)? {
        patterns.push(pattern);

        if tokens.peek_token() == Some(Token::Char(',')) {
            tokens.next()?;
        } else {
            break;
        }
    }

    // If there were no patterns, still allow a comma for consistency with other
    // empty collection literals.
    if patterns.is_empty() && tokens.peek_token() == Some(Token::Char(',')) {
        tokens.next()?;
    }

    let closing_brace = tokens.next()?;
    if matches!(closing_brace.0, Token::Close(paired) if paired == kind) {
        Ok(tokens.ranged(start.., PatternKind::DestructureTuple(patterns)))
    } else {
        Err(closing_brace.map(|_| Error::ExpectedPatternOr(kind)))
    }
}

fn parse_map_destructure_pattern(
    start: usize,
    tokens: &mut TokenReader<'_>,
) -> Result<Ranged<PatternKind>, Ranged<Error>> {
    let mut entries = Vec::new();

    if tokens.peek_token() == Some(Token::Char(',')) {
        // Empty map
        tokens.next()?;
    } else {
        while tokens
            .peek_token()
            .map_or(false, |token| token != Token::Close(Paired::Brace))
        {
            let token = tokens.next()?;
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

            let colon = tokens.next()?;
            if colon.0 != Token::Char(':') {
                return Err(colon.map(|_| Error::ExpectedColon));
            }

            let Some(value) = parse_pattern_kind(tokens)? else {
                return Err(tokens.ranged(tokens.last_index.., Error::ExpectedPattern));
            };
            entries.push(tokens.ranged(key.range().start.., EntryPattern { key, value }));

            if tokens.peek_token() == Some(Token::Char(',')) {
                tokens.next()?;
            } else {
                break;
            }
        }
    }

    let end_brace = tokens.next()?;
    if end_brace.0 != Token::Close(Paired::Brace) {
        return Err(end_brace.map(|_| Error::MissingEnd(Paired::Brace)));
    }

    Ok(tokens.ranged(start.., PatternKind::DestructureMap(entries)))
}

fn parse_match_block_body(
    start: usize,
    tokens: &mut TokenReader<'_>,
    config: &ParserConfig<'_>,
) -> Result<Ranged<Matches>, Ranged<Error>> {
    let mut matches = Matches::default();
    while let Some(pattern) = parse_pattern(tokens, config)? {
        let arrow_or_brace = tokens.next()?;
        let body = match &arrow_or_brace.0 {
            Token::FatArrow => config.parse_expression(tokens)?,
            Token::Open(Paired::Brace) => {
                parse_block(tokens, config)?.map(|block| Expression::Block(Box::new(block)))
            }
            _ => return Err(arrow_or_brace.map(|_| Error::ExpectedFatArrow)),
        };
        matches
            .patterns
            .push(tokens.ranged(pattern.range().start.., MatchPattern { pattern, body }));

        if tokens.peek_token() == Some(Token::Char(',')) {
            tokens.next()?;
        } else {
            break;
        }
    }

    let closing_brace = tokens.next()?;
    if closing_brace.0 == Token::Close(Paired::Brace) {
        Ok(tokens.ranged(start.., matches))
    } else {
        Err(closing_brace.map(|_| Error::ExpectedPatternOr(Paired::Brace)))
    }
}

fn parse_variable(
    mutable: bool,
    publish: bool,
    start: usize,
    tokens: &mut TokenReader<'_>,
    config: &ParserConfig<'_>,
) -> Result<Ranged<Expression>, Ranged<Error>> {
    let Some(pattern) = parse_pattern(tokens, config)? else {
        return Err(tokens.ranged(tokens.last_index.., Error::ExpectedPattern));
    };

    let value = if tokens.peek_token() == Some(Token::Char('=')) {
        tokens.next()?;
        config.parse_expression(tokens)?
    } else {
        tokens.ranged(tokens.last_index.., Expression::Literal(Literal::Nil))
    };

    let r#else = if tokens.peek_token() == Some(Token::Identifier(Symbol::else_symbol().clone())) {
        tokens.next()?;
        Some(config.parse_expression(tokens)?)
    } else {
        None
    };

    Ok(tokens.ranged(
        start..,
        Expression::Variable(Box::new(Variable {
            publish,
            mutable,
            pattern,
            value,
            r#else,
        })),
    ))
}

struct Let;

impl Parselet for Let {
    fn token(&self) -> Option<Token> {
        Some(Token::Identifier(Symbol::let_symbol().clone()))
    }
}

impl PrefixParselet for Let {
    fn parse_prefix(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        parse_variable(false, false, token.range().start, tokens, config)
    }
}

struct Var;

impl Parselet for Var {
    fn token(&self) -> Option<Token> {
        Some(Token::Identifier(Symbol::var_symbol().clone()))
    }
}

impl PrefixParselet for Var {
    fn parse_prefix(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        parse_variable(true, false, token.range().start, tokens, config)
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
        _token: &Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        _config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let name_token = tokens.next()?;
        let Token::Identifier(name) = name_token.0 else {
            return Err(Ranged::new(name_token.1, Error::ExpectedName));
        };
        Ok(tokens.ranged(
            lhs.range().start..,
            Expression::Lookup(Box::new(Lookup {
                base: Some(lhs),
                name,
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
        _token: &Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        _config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        Ok(tokens.ranged(
            lhs.range().start..,
            Expression::TryOrNil(Box::new(TryOrNil { body: lhs })),
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
        _token: &Ranged<Token>,
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
                        value,
                    })),
                ))
            }
            _ => Err(lhs.map(|_| Error::InvalidAssignmentTarget)),
        }
    }
}

macro_rules! parselets {
    ($($name:ident),+ $(,)?) => {
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
        Let,
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
