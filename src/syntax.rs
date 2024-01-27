use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::{Bound, Deref, Range, RangeBounds, RangeInclusive};

use ahash::AHashMap;
use serde::{Deserialize, Serialize};

use self::token::{Paired, Token, Tokens};
use crate::symbol::Symbol;
pub mod token;

#[derive(Default, Clone, Copy, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub struct Ranged<T>(pub T, pub SourceRange);

impl<T> Ranged<T> {
    pub fn new(range: impl Into<SourceRange>, value: T) -> Self {
        Self(value, range.into())
    }

    pub fn bounded(range: impl RangeBounds<usize>, end: usize, value: T) -> Ranged<T> {
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

#[derive(Default, Clone, Copy, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub struct SourceRange {
    pub start: usize,
    pub length: usize,
}

impl SourceRange {
    #[must_use]
    pub const fn end(&self) -> usize {
        self.start + self.length
    }
}

impl From<Range<usize>> for SourceRange {
    fn from(range: Range<usize>) -> Self {
        Self {
            start: range.start,
            length: range.end - range.start,
        }
    }
}

impl From<RangeInclusive<usize>> for SourceRange {
    fn from(range: RangeInclusive<usize>) -> Self {
        Self {
            start: *range.start(),
            length: range.end() - range.start(),
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub enum Expression {
    #[default]
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    Lookup(Box<Lookup>),
    If(Box<IfExpression>),
    Function(Box<FunctionDefinition>),
    Map(Box<MapExpression>),
    Call(Box<FunctionCall>),
    Variable(Box<Variable>),
    Assign(Box<Assignment>),
    Unary(Box<UnaryExpression>),
    Binary(Box<BinaryExpression>),
    Chain(Box<Chain>),
    Block(Box<Block>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Chain(pub Ranged<Expression>, pub Ranged<Expression>);

impl Chain {
    #[must_use]
    pub fn from_expressions(mut expressions: Vec<Ranged<Expression>>) -> Ranged<Expression> {
        let Some(mut expression) = expressions.pop() else {
            return Ranged::new(0..0, Expression::Nil);
        };

        while let Some(previous) = expressions.pop() {
            expression = Ranged::new(
                previous.range().start..expression.range().end(),
                Expression::Chain(Box::new(Chain(previous, expression))),
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
    pub values: Vec<MapField>,
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
pub struct FunctionDefinition {
    pub name: Ranged<Symbol>,
    pub parameters: Vec<Ranged<Symbol>>,
    pub body: Ranged<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub function: Ranged<Expression>,
    pub parameters: Vec<Ranged<Expression>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub target: Ranged<Lookup>,
    pub value: Ranged<Expression>,
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
    Call,
    Add,
    Subtract,
    Multiply,
    Divide,
    IntegerDivide,
    Remainder,
    Power,
    JumpIf,
    Bitwise(BitwiseKind),
    Logical(LogicalKind),
    Compare(CompareKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CompareKind {
    LessThanOrEqual,
    LessThan,
    Equal,
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
    pub mutable: bool,
    pub name: Symbol,
    pub value: Ranged<Expression>,
}

pub fn parse(source: &str) -> Result<Ranged<Expression>, Ranged<Error>> {
    let mut tokens = TokenReader::new(source);

    let parselets = parselets();
    let config = ParserConfig {
        parselets: &parselets,
        minimum_precedence: 0,
    };
    let mut results = Vec::new();
    loop {
        if tokens.peek().is_none() {
            results.push(Ranged::new(
                tokens.last_index..tokens.last_index,
                Expression::Nil,
            ));
            break;
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

struct TokenReader<'a> {
    tokens: Tokens<'a>,
    peeked: VecDeque<Result<Ranged<Token>, Ranged<Error>>>,
    last_index: usize,
}

impl<'a> TokenReader<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            tokens: Tokens::new(source)
                .excluding_comments()
                .excluding_whitespace(),
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
        self.last_index = token.1.start + token.1.length;
        Ok(token)
    }

    fn ranged<T>(&self, range: impl RangeBounds<usize>, value: T) -> Ranged<T> {
        Ranged::bounded(range, self.last_index, value)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum Error {
    UnexpectedEof,
    ExpectedEof,
    MissingEnd(Paired),
    Token(token::Error),
    UnexpectedToken(Token),
    ExpectedName,
    ExpectedVariableInitialValue,
    ExpectedThenOrBrace,
    InvalidAssignmentTarget,
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
                        lhs =
                            parselet.parse(lhs, tokens, &self.with_precedence(level.precedence))?;
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
                return parselet.parse(token, tokens, &self.with_precedence(level.precedence));
            }
        }

        Err(token.map(Error::UnexpectedToken))
    }

    fn parse_expression(
        &self,
        tokens: &mut TokenReader<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        self.with_precedence(self.parselets.markers.expression)
            .parse(tokens)
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

// a (b) * 2

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
    fn parse(
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
            Token::Int(_) | Token::Float(_) | Token::Identifier(_)
        )
    }
}

impl PrefixParselet for Term {
    fn parse(
        &self,
        token: Ranged<Token>,
        _tokens: &mut TokenReader<'_>,
        _config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        match token.0 {
            Token::Int(value) => Ok(Ranged::new(token.1, Expression::Int(value))),
            Token::Float(value) => Ok(Ranged::new(token.1, Expression::Float(value))),
            Token::Identifier(value) => Ok(Ranged::new(
                token.1,
                Expression::Lookup(Box::new(Lookup::from(value))),
            )),
            _ => unreachable!("parse called with invalid token"),
        }
    }
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
            fn parse(
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

macro_rules! impl_prefix_standalone_parselet {
    ($name:ident, $token:expr, $binarykind:expr) => {
        struct $name;

        impl Parselet for $name {
            fn token(&self) -> Option<Token> {
                Some($token)
            }
        }

        impl PrefixParselet for $name {
            fn parse(
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
    Expression::Bool(true)
);
impl_prefix_standalone_parselet!(
    False,
    Token::Identifier(Symbol::false_symbol().clone()),
    Expression::Bool(false)
);

fn parse_variable(
    mutable: bool,
    start: usize,
    tokens: &mut TokenReader<'_>,
    config: &ParserConfig<'_>,
) -> Result<Ranged<Expression>, Ranged<Error>> {
    let name_token = tokens.next()?;
    let Token::Identifier(name) = name_token.0 else {
        return Err(name_token.map(|_| Error::ExpectedName));
    };
    let value = if tokens.peek_token() == Some(Token::Char('=')) {
        tokens.next()?;
        config.parse_expression(tokens)?
    } else {
        Ranged::new(name_token.1, Expression::Nil)
    };
    Ok(tokens.ranged(
        start..,
        Expression::Variable(Box::new(Variable {
            mutable,
            name,
            value,
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
    fn parse(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        parse_variable(false, token.range().start, tokens, config)
    }
}

struct Var;

impl Parselet for Var {
    fn token(&self) -> Option<Token> {
        Some(Token::Identifier(Symbol::var_symbol().clone()))
    }
}

impl PrefixParselet for Var {
    fn parse(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        parse_variable(true, token.range().start, tokens, config)
    }
}

struct Braces;

impl Braces {
    fn parse_block(
        start: usize,
        expr: Ranged<Expression>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
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
            expressions.push(tokens.ranged(tokens.last_index.., Expression::Nil));
        }

        match tokens.next() {
            Ok(Ranged(Token::Close(Paired::Brace), _)) => Ok(tokens.ranged(
                start..,
                Expression::Block(Box::new(Block {
                    name: None,
                    body: Chain::from_expressions(expressions),
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
    fn parse(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        if tokens.peek_token() == Some(Token::Close(Paired::Brace)) {
            tokens.next()?;
            return Ok(tokens.ranged(token.range().start.., Expression::Map(Box::default())));
        }
        let expr = config.parse_expression(tokens)?;

        match tokens.peek() {
            Some(Ranged(Token::Char(':'), _)) => todo!("map"),
            Some(Ranged(Token::Char(','), _)) => todo!("set"),
            Some(Ranged(Token::Char(';'), _)) => {
                tokens.next()?;
                Self::parse_block(token.range().start, expr, tokens, config)
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
    fn parse(
        &self,
        _token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let mut expressions = Vec::new();

        let ended_in_comma = parse_paired(Paired::Paren, &Token::Char(','), tokens, |tokens| {
            config
                .parse_expression(tokens)
                .map(|expr| expressions.push(expr))
        })?;

        if expressions.len() != 1 || ended_in_comma {
            todo!("tuples")
        } else {
            Ok(expressions.into_iter().next().expect("length checked"))
        }
    }
}

impl InfixParselet for Parentheses {
    fn parse(
        &self,
        function: Ranged<Expression>,
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

trait InfixParselet: Parselet {
    fn parse(
        &self,
        lhs: Ranged<Expression>,
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
                tokens: &mut TokenReader<'_>,
                config: &ParserConfig<'_>,
            ) -> Result<Ranged<Expression>, Ranged<Error>> {
                let right = config.parse(tokens)?;
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
    fn parse(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let condition = config.parse_expression(tokens)?;
        let brace_or_then = tokens.next()?;
        let when_true = match &brace_or_then.0 {
            Token::Open(Paired::Brace) => Braces.parse(brace_or_then, tokens, config)?,
            Token::Identifier(ident) if ident == Symbol::then_symbol() => {
                config.parse_expression(tokens)?
            }
            _ => return Err(brace_or_then.map(|_| Error::ExpectedThenOrBrace)),
        };
        let when_false = if tokens.peek_token()
            == Some(Token::Identifier(Symbol::else_symbol().clone()))
        {
            tokens.next()?;
            Some(match tokens.peek_token() {
                Some(Token::Identifier(ident)) if ident == *Symbol::if_symbol() => {
                    Self.parse(tokens.next()?, tokens, config)?
                }
                Some(Token::Open(Paired::Brace)) => Braces.parse(tokens.next()?, tokens, config)?,
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

struct Fn;

impl Parselet for Fn {
    fn token(&self) -> Option<Token> {
        None
    }

    fn matches(&self, token: &Token, tokens: &mut TokenReader<'_>) -> bool {
        matches!(token, Token::Identifier(ident) if ident == Symbol::fn_symbol())
            && tokens
                .peek_token()
                .map_or(false, |t| matches!(t, Token::Identifier(_)))
    }
}

impl PrefixParselet for Fn {
    fn parse(
        &self,
        token: Ranged<Token>,
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        let name_token = tokens.next()?;
        let Token::Identifier(name) = &name_token.0 else {
            return Err(name_token.map(|_| Error::ExpectedName));
        };
        let mut parameters = Vec::new();
        if tokens.peek_token() == Some(Token::Open(Paired::Paren)) {
            tokens.next()?;
            parse_paired(Paired::Paren, &Token::Char(','), tokens, |tokens| {
                let name_token = tokens.next()?;
                let Token::Identifier(name) = &name_token.0 else {
                    return Err(name_token.map(|_| Error::ExpectedName));
                };
                parameters.push(Ranged::new(name_token.1, name.clone()));
                Ok(())
            })?;
        }

        let body_indicator = tokens.next()?;
        let body = match &body_indicator.0 {
            Token::Open(Paired::Brace) => Braces.parse(body_indicator, tokens, config)?,
            Token::Char('=') => config.parse_expression(tokens)?,
            _ => todo!("expected body"),
        };

        Ok(tokens.ranged(
            token.range().start..,
            Expression::Function(Box::new(FunctionDefinition {
                name: Ranged::new(name_token.1, name.clone()),
                parameters,
                body,
            })),
        ))
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
        tokens: &mut TokenReader<'_>,
        config: &ParserConfig<'_>,
    ) -> Result<Ranged<Expression>, Ranged<Error>> {
        match lhs.0 {
            Expression::Lookup(lookup) => {
                let value = config.parse(tokens)?;
                Ok(tokens.ranged(
                    lhs.1.start..,
                    Expression::Assign(Box::new(Assignment {
                        target: Ranged::new(lhs.1, *lookup),
                        value,
                    })),
                ))
            }
            _ => Err(lhs.map(|_| Error::InvalidAssignmentTarget)),
        }
    }
}

macro_rules! parselets {
    ($($name:ident),+) => {
        vec![$(Box::new($name)),+]
    }
}

fn parselets() -> Parselets {
    let mut parser = Parselets::new();
    parser.markers.expression = parser.precedence;
    parser.push_infix(parselets![Assign]);
    parser.push_infix(parselets![
        LessThanOrEqual,
        LessThan,
        Equal,
        GreaterThan,
        GreaterThanOrEqual
    ]);
    parser.push_infix(parselets![Or]);
    parser.push_infix(parselets![Xor]);
    parser.push_infix(parselets![And]);
    parser.push_infix(parselets![BitwiseOr]);
    parser.push_infix(parselets![BitwiseXor]);
    parser.push_infix(parselets![BitwiseAnd]);
    parser.push_infix(parselets![Add, Subtract]);
    parser.push_infix(parselets![Multiply, Divide, Remainder, IntegerDivide]);
    parser.push_infix(parselets![Parentheses, Dot]);
    parser.push_prefix(parselets![
        Braces,
        Parentheses,
        LogicalNot,
        BitwiseNot,
        Let,
        Var,
        If,
        True,
        False,
        Fn
    ]);
    parser.push_prefix(parselets![Term]);
    parser
}
