# Bitwise

```musebnf
BitwiseOr: <BitwiseXor> ('|' <BitwiseXor>)*;
BitwiseXor: <BitwiseAnd> ('^' <BitwiseAnd>)*;
BitwiseAnd: <AddSubtract> ('&' <AddSubtract>)*;

BitwiseShift: <AddSubtract> <ShiftLeft | ShiftRight>*;

ShiftLeft: '<<' <AddSubtract>;
ShiftRight: '>>' <AddSubtract>;

BitwiseNot: '!' <Prefix>;
```

Bitwise operations operate on integers using logic operations on each individual
bit. In Muse, integers are 64-bits and can be signed or unsigned. Because signed
numbers use a bit for determining the sign, it is often preferred to use
unsigned numbers only when performing bitwise operations.

The only operator that performs differently between signed and unsigned integers
is shift right. Muse uses a sign-preserving shift-right operation for signed
integers.

These operators can be overridden by types to perform different non-bitwise
functionality.

## Bitwise Or

The bitwise or expression produces a new value by performing a logical or
operation on each corresponding bit in the two operands.

```muselang
let 0b101 = 0b100 | 0b001
```

## Bitwise Xor

The bitwise excusive or (xor) expression produces a new value by performing a logical xor
operation on each corresponding bit in the two operands.

```muselang
let 0b101 = 0b110 ^ 0b011
```

## Bitwise And

The bitwise and expression produces a new value by performing a logical and
operation on each corresponding bit in the two operands.

```muselang
let 0b100 = 0b110 & 0b101
```

## Bitwise Shifting

The bitwise shift expressions produce a new value by moving bits left or right
by a number of bits, filling in any empty bits with 0.

```muselang
let 0b100 = 0b010 << 1;
let 0b001 = 0b010 >> 1;
```

The shift-right expression is sign-preserving when operating on signed integers:

```muselang
let -2 = -4 >> 1;
```

## Bitwise Not

The bitwise not expression produces a new value by performing a logical not
operation on each bit in the operand.

```muselang
let -1 = !0;
let 0uxffff_ffff_ffff_ffff = !0u;
```
