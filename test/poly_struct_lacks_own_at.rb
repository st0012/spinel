# A boxed Struct raises NoMethodError for `at` also when the program
# defines its own `at` in another class or module.
module HasAt
  def at(i) = "mod#{i}"
end

class Foo
  def at(i) = "foo#{i}"
end

S = Struct.new(:a)
T = Struct.new(:a) do
  def at(i) = "T#{i}"
end
U = Struct.new(:a) { include HasAt }
class V < Struct.new(:a, :b); end

def probe(v) = (v.at(1) rescue $!.class)

vals = [S.new(1), Foo.new, [7, 8], T.new(2), U.new(3), V.new(4, 5), "str", nil, :sym]
vals.each do |v|
  p((v.at(0) rescue $!.class))
  p probe(v)
end
