# A merge, merge! or update with several arguments folds into one call per
# argument only on a Hash: a class of the program's own with a method of
# that name gets every argument in one call. update takes a conflict block
# with several Hashes as merge! does.

class Pair
  def merge(a, b) = "merge #{a.size} #{b.size}"
  def merge!(a, b) = "merge! #{a.keys} #{b.keys}"
  def update(*hs) = hs.size
end

class Spread
  def self.merge(a, b) = "class merge #{a}+#{b}"
end

class H < Hash; end

class Own < Hash
  def merge(a, b = nil) = "own #{a.size} #{b.inspect}"
end

pr = Pair.new
p pr.merge({x: 1}, {y: 2})
p pr.merge!({x: 1}, {y: 2})
p pr.update({x: 1}, {y: 2}, {z: 3})
p Spread.merge(1, 2)

# a Hash in the same program still folds
h = {a: 1}
p h.merge({b: 2}, {c: 3})
p h.merge({a: 2}, {a: 3}) { |k, o, n| o + n }
h.update({a: 5}, {z: 1}) { |k, o, n| o * n }
p h
g = {}
p g.merge({1 => "a"}, {"s" => 2})
p g.merge!({"b" => 2}, {"c" => 3}) { |k, o, n| o + n }

# arguments with a side effect run in order
def side(x) = (puts "side #{x}"; {x => 1})
q = {}
p q.update(side(:p), side(:q))

# a Hash subclass instance takes the builtin method, or its own
hs = H.new
hs[:a] = 1
p hs.merge({b: 2}, {c: 3})
p hs.merge({b: 2}, {c: 3}).class
p Own.new.merge({b: 2}, {c: 3})

# a value that is the program's object or a Hash
def pick(n) = n > 0 ? Pair.new : {q: 1}
p pick(1).merge!({r: 1}, {s: 2})
p pick(0).merge!({r: 1}, {s: 2})
[Pair.new, {q: 1}].each { |x| p x.merge!({r: 1}, {s: 2}) }

# update with several Hashes and a block, on a plain Hash
u = {x: 1, y: 2}
u.update({x: 1}, {y: 2}) { |k, o, n| o + n }
p u
