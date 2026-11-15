# Complex#==, === and != with a Bignum operand compare the value exactly,
# as CRuby does, where the typed Complex answered false.
c = Complex(1, 0) * 2**62 * 4
p c == 2**64
p c != 2**64
p c === 2**64
p c == 2**64 + 1
p c == -(2**64)
p(-c == -(2**64))
d = Complex(3, 0) * 2**62 * 4
p d == 3 * 2**64
p Complex(18446744073709551616.0, 0) == 2**64
p Complex(18446744073709551616.0, 1) == 2**64
p Complex(1, 0) == 2**64
def same?(z, big) = z == big
p same?(c, 2**64), same?(c, 2**65)
big = 2**100
p Complex(Float::INFINITY, 0) == big
p Complex(Float::NAN, 0) == big
