# A Data value has no to_a, so a splat of it is the value itself:
# `[*d]` is [d]. It raised NoMethodError (undefined method 'to_a'),
# because the splat took a Data class for a Struct, which has to_a.
D = Data.define(:a)
d = D.new(a: 1)
p [*d]
p [0, *d, 2]
def count(*a) = a.size
p count(*d)
x = *d
p x
a, b = *d
p a, b
p(*d)

class E < D; end
p [*E.new(a: 2)]

# a Data class with its own to_a is splatted through it
F = Data.define(:a) do
  def to_a = [a, a]
end
p [*F.new(a: 3)]

G = Data.define
p [*G.new]

# a Struct still splats its members
S = Struct.new(:x, :y)
p [*S.new(3, 4)]
