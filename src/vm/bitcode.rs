use std::iter::Take;
use std::sync::Arc;
use std::{array, slice, str};

use ahash::AHashMap;
use ordered_varint::Variable as _;

use super::ops::{Add, Allocate, Destination, Instruction, Source};
use super::{Code, InvalidRegister, Register};
use crate::symbol::Symbol;
use crate::vm::ops::Stack;

pub fn decode(mut bitcode: &[u8]) -> Result<Code, DecodeError> {
    let symbol_count =
        usize::decode_variable(&mut bitcode).map_err(|_| DecodeError::IntegerEncoding)?;
    let mut symbols = Vec::with_capacity(symbol_count);
    for _ in 0..symbol_count {
        let len = usize::decode_variable(&mut bitcode).map_err(|_| DecodeError::IntegerEncoding)?;
        let (symbol, remaining) = bitcode.split_at(len);
        bitcode = remaining;

        symbols.push(Symbol::from(str::from_utf8(symbol)?));
    }

    let mut code = Code::default();
    while !bitcode.is_empty() {
        let opcode = Opcode::try_from(bitcode[0])?;
        bitcode = &bitcode[1..];
        code.push_boxed(match opcode {
            Opcode::Add => decode_add(&mut bitcode, &symbols)?,
            Opcode::Allocate => decode_allocate(&mut bitcode, &symbols)?,
            Opcode::Div => todo!(),
            Opcode::Eq => todo!(),
            Opcode::Hash => todo!(),
            Opcode::Mul => todo!(),
            Opcode::Rem => todo!(),
            Opcode::Sub => todo!(),
        });
    }

    Ok(code)
}

macro_rules! decode_ssd {
    ($decode_name:ident, $compile_name:ident) => {
        fn $decode_name(
            bitcode: &mut &[u8],
            symbols: &[Symbol],
        ) -> Result<Arc<dyn Instruction>, DecodeError> {
            fn source<Lhs>(
                bitcode: &mut &[u8],
                symbols: &[Symbol],
                lhs: Lhs,
            ) -> Result<Arc<dyn Instruction>, DecodeError>
            where
                Lhs: Source,
            {
                decode_source!(bitcode, symbols, source_source, lhs)
            }

            fn source_source<Lhs, Rhs>(
                bitcode: &mut &[u8],
                symbols: &[Symbol],
                lhs: Lhs,
                rhs: Rhs,
            ) -> Result<Arc<dyn Instruction>, DecodeError>
            where
                Lhs: Source,
                Rhs: Source,
            {
                decode_dest!(bitcode, symbols, $compile_name, lhs, rhs)
            }

            decode_source!(bitcode, symbols, source)
        }
    };
}

macro_rules! decode_source {
    ($bitcode:ident, $symbols:ident, $next_fn:ident $(, $($arg:tt)*)?) => {{
        if $bitcode.is_empty() {
            return Err(DecodeError::UnexpectedEof);
        }

        match Arg::decode_from($bitcode, $symbols)? {
            Arg::Variable { index } => $next_fn($bitcode, $symbols $(, $($arg)*)?, Stack(index)),
            Arg::Register { index } => $next_fn($bitcode, $symbols $(, $($arg)*)?, Register::try_from(index)?),
            Arg::Int(int) => $next_fn($bitcode, $symbols $(, $($arg)*)?, int),
            Arg::Float(float) => $next_fn($bitcode, $symbols $(, $($arg)*)?, float),
            Arg::Symbol(symbol) => $next_fn($bitcode, $symbols $(, $($arg)*)?, symbol),
        }
    }};
}

macro_rules! decode_dest {
    ($bitcode:ident, $symbols:ident, $next_fn:ident $(, $($arg:tt)*)?) => {{
        if $bitcode.is_empty() {
            return Err(DecodeError::UnexpectedEof);
        }

        match Arg::decode_from($bitcode, $symbols)? {
            Arg::Variable { index } => $next_fn($bitcode, $symbols $(, $($arg)*)?, Stack(index)),
            Arg::Register { index } => $next_fn($bitcode, $symbols $(, $($arg)*)?, Register::try_from(index)?),
            _ => Err(DecodeError::InvalidDestination)
        }
    }};
}

decode_ssd!(decode_add, compile_add);

fn compile_add<Lhs, Rhs, Dest>(
    _bitcode: &mut &[u8],
    _symbols: &[Symbol],
    lhs: Lhs,
    rhs: Rhs,
    dest: Dest,
) -> Result<Arc<dyn Instruction>, DecodeError>
where
    Lhs: Source,
    Rhs: Source,
    Dest: Destination,
{
    Ok(Arc::new(Add { lhs, rhs, dest }))
}

fn decode_allocate(
    bitcode: &mut &[u8],
    symbols: &[Symbol],
) -> Result<Arc<dyn Instruction>, DecodeError> {
    let Arg::Int(count) = Arg::decode_from(bitcode, symbols)? else {
        return Err(DecodeError::InvalidArg);
    };
    let Ok(count) = u16::try_from(count) else {
        return Err(DecodeError::InvalidArg);
    };
    Ok(Arc::new(Allocate(count)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    IntegerEncoding,
    InvalidOpcode,
    UnexpectedEof,
    InvalidArg,
    InvalidRegister,
    InvalidSymbol,
    InvalidDestination,
    Utf8(str::Utf8Error),
}

impl From<str::Utf8Error> for DecodeError {
    fn from(err: str::Utf8Error) -> Self {
        Self::Utf8(err)
    }
}

impl From<InvalidOpcode> for DecodeError {
    fn from(_: InvalidOpcode) -> Self {
        Self::InvalidOpcode
    }
}

impl From<InvalidRegister> for DecodeError {
    fn from(_: InvalidRegister) -> Self {
        Self::InvalidRegister
    }
}

#[derive(Default)]
pub struct BitcodeEncoder {
    symbols: Vec<Symbol>,
    symbols_by_index: AHashMap<Symbol, usize>,
    bitcode: Vec<u8>,
}

impl BitcodeEncoder {
    pub fn encode(&mut self, opcode: Opcode, args: &[Arg]) {
        self.bitcode.push(u8::from(opcode));
        for arg in args {
            arg.encode_into(self);
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.bitcode
            .splice(0..0, EncodedSymbolIter::new(&self.symbols));
        self.bitcode
    }
}

struct EncodedSymbolIter<'a> {
    symbols: slice::Iter<'a, Symbol>,
    current_symbol: slice::Iter<'a, u8>,
    current_symbol_len: Take<array::IntoIter<u8, 15>>,
}

impl<'a> EncodedSymbolIter<'a> {
    fn new(symbols: &'a [Symbol]) -> Self {
        // Due to how the iterator logic is written, we can emit the number of
        // strings first, then iteration across the symbols will begin.
        let count = symbols.len();
        let mut count_bytes = [0; 15];
        let count_length = count
            .encode_variable(&mut count_bytes[..])
            .expect("infallible");
        Self {
            symbols: symbols.iter(),
            current_symbol: [].iter(),
            current_symbol_len: count_bytes.into_iter().take(count_length),
        }
    }
}

impl Iterator for EncodedSymbolIter<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(byte) = self
            .current_symbol_len
            .next()
            .or_else(|| self.current_symbol.next().copied())
        {
            Some(byte)
        } else if let Some(sym) = self.symbols.next() {
            let mut length_bytes = [0; 15];
            let length = sym
                .len()
                .encode_variable(&mut length_bytes[..])
                .expect("infallible");
            self.current_symbol = sym.as_bytes().iter();
            self.current_symbol_len = length_bytes.into_iter().take(length);
            self.current_symbol_len.next()
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (min, max) = self
            .symbols
            .clone()
            .map(|s| s.len())
            .fold((0, 0), |(min, max), len| (min + len + 1, max + 15));

        let (symbol_len, _) = self.current_symbol_len.size_hint();
        let (symbol, _) = self.current_symbol.size_hint();
        let in_progress_hint = symbol_len + symbol;

        (min + in_progress_hint, Some(max + in_progress_hint))
    }
}

pub trait AsArg {
    fn as_arg(&self) -> Arg;
}

impl AsArg for i64 {
    fn as_arg(&self) -> Arg {
        Arg::Int(*self)
    }
}

impl AsArg for f64 {
    fn as_arg(&self) -> Arg {
        Arg::Float(*self)
    }
}

impl AsArg for Symbol {
    fn as_arg(&self) -> Arg {
        Arg::Symbol(self.clone())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    Variable { index: usize },
    Register { index: u8 },
    Int(i64),
    Float(f64),
    Symbol(Symbol),
}

impl Arg {
    const FLOAT: u8 = 3;
    const INT: u8 = 2;
    const REGISTER: u8 = 0;
    const SYMBOL: u8 = 4;
    const VARIABLE: u8 = 1;

    pub fn encode_into(&self, encoder: &mut BitcodeEncoder) {
        match self {
            Arg::Variable { index } => {
                encoder.bitcode.push(Self::VARIABLE);
                index
                    .encode_variable(&mut encoder.bitcode)
                    .expect("infallible");
            }
            Arg::Register { index } => {
                encoder.bitcode.push(Self::REGISTER);
                encoder.bitcode.push(*index);
            }
            Arg::Int(value) => {
                encoder.bitcode.push(Self::INT);
                value
                    .encode_variable(&mut encoder.bitcode)
                    .expect("infallible");
            }
            Arg::Float(value) => {
                encoder.bitcode.push(Self::FLOAT);
                encoder.bitcode.extend_from_slice(&value.to_le_bytes());
            }
            Arg::Symbol(symbol) => {
                encoder.bitcode.push(Self::SYMBOL);
                let index = encoder
                    .symbols_by_index
                    .entry(symbol.clone())
                    .or_insert_with(|| {
                        let index = encoder.symbols.len();
                        encoder.symbols.push(symbol.clone());
                        index
                    });
                index
                    .encode_variable(&mut encoder.bitcode)
                    .expect("infallible");
            }
        }
    }

    pub fn decode_from(bitcode: &mut &[u8], symbols: &[Symbol]) -> Result<Self, DecodeError> {
        let arg_kind = bitcode[0];
        *bitcode = &bitcode[1..];
        match arg_kind {
            Self::FLOAT => {
                if bitcode.len() < 8 {
                    return Err(DecodeError::UnexpectedEof);
                }
                let decoded = f64::from_le_bytes(array::from_fn(|index| bitcode[index]));
                *bitcode = &bitcode[8..];

                Ok(Self::Float(decoded))
            }
            Self::INT => {
                let decoded =
                    i64::decode_variable(bitcode).map_err(|_| DecodeError::IntegerEncoding)?;

                Ok(Self::Int(decoded))
            }
            Self::REGISTER => {
                if bitcode.is_empty() {
                    return Err(DecodeError::UnexpectedEof);
                }
                // TODO this seems wasteful. Seems like we should just move 16
                // values into arg_kind and save a byte.
                let index = bitcode[0];
                *bitcode = &bitcode[1..];

                Ok(Self::Register { index })
            }
            Self::SYMBOL => {
                let index =
                    usize::decode_variable(bitcode).map_err(|_| DecodeError::IntegerEncoding)?;
                let symbol = symbols
                    .get(index)
                    .ok_or(DecodeError::InvalidSymbol)?
                    .clone();

                Ok(Self::Symbol(symbol))
            }
            Self::VARIABLE => {
                let index =
                    usize::decode_variable(bitcode).map_err(|_| DecodeError::IntegerEncoding)?;

                Ok(Self::Variable { index })
            }
            _ => Err(DecodeError::InvalidArg),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
#[repr(u8)]
#[non_exhaustive]
pub enum Opcode {
    Add = 0,
    Allocate,
    Div,
    Eq,
    Hash,
    Mul,
    Rem,
    Sub,
}

// This impl block only contains the list of Opcodes. Keep this impl block
// separate to ensure other constants don't get grouped in the future.
// Adding/updating Opcodes shouldn't be an errorprone operation.
impl Opcode {}

impl TryFrom<u8> for Opcode {
    type Error = InvalidOpcode;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Add),
            1 => Ok(Self::Allocate),
            2 => Ok(Self::Div),
            3 => Ok(Self::Eq),
            4 => Ok(Self::Hash),
            5 => Ok(Self::Mul),
            6 => Ok(Self::Rem),
            7 => Ok(Self::Sub),
            _ => Err(InvalidOpcode(value)),
        }
    }
}

impl From<Opcode> for u8 {
    fn from(value: Opcode) -> Self {
        value as u8
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct InvalidOpcode(u8);
