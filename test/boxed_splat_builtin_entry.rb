# A splatted call on a boxed receiver, beside a user method of the same
# name, reaches the builtin with every argument the list holds.
class Reader
  def read(*a) = [:reader, a]
  def first(*a) = [:reader, a]
  def center(*a) = [:reader, a]
  def round(*a) = [:reader, a]
  def prepend(*a) = [:reader, a]
end

def rd(o, *a) = o.read(*a)
def fi(o, *a) = o.first(*a)
def ce(o, *a) = o.center(*a)
def ro(o, *a) = o.round(*a)
def pr(o, *a) = o.prepend(*a)

r, w = IO.pipe
w.write("abcdef")
w.close
p rd([r, Reader.new][0], 2)
p rd([r, Reader.new][0])
p rd([Reader.new, r][0], 1, 2)

p fi([[1, 2, 3], Reader.new][0])
p fi([[1, 2, 3], Reader.new][0], 2)
p fi([Reader.new, [1, 2, 3]][0], 2)

p ce(["ab", Reader.new][0], 6)
p ce(["ab", Reader.new][0], 6, "*")

p ro([3.14159, Reader.new][0])
p ro([3.14159, Reader.new][0], 2)
p ro([1234, Reader.new][0], -2)

p pr([+"abc", Reader.new][0], "x", "y")
p pr([Reader.new, +"abc"][0], "x", "y")

begin
  fi([[1, 2, 3], Reader.new][0], 1, 2)
rescue ArgumentError => e
  p e.message
end
begin
  ce(["ab", Reader.new][0])
rescue ArgumentError => e
  p e.message
end
