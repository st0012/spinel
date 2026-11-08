# A method that calls nothing that can allocate, poll a safe point or run
# Ruby code has no point where a collection can start, so nothing reads its
# GC roots: codegen takes them all back, and with them the method's root
# frame. The first three shapes are the benchmarks where that frame was most
# of the loop's time: a String read through a class method, a small vector's
# dot product, and a grid cell read.
# spinel: decisions
class Holder
  def self.use(s, i) = s.getbyte(i % 64)
end

class Vec2
  attr_reader :x, :y

  def initialize(x, y)
    @x = x
    @y = y
  end

  def dot(o) = @x * o.x + @y * o.y
end

def cell_get(cells, w, h, x, y)
  cells[((y + h) % h) * w + (x + w) % w]
end

# kept: it builds a String, and a collection there reads `s` after it
def tag(s) = s + "!"

# a root spelled inside a String is text: the String stays whole
def spelled(s) = s.getbyte(0) == 0 ? "SP_GC_ROOT(s);" : "SP_GC_ROOT(x); kept"

s = "x" * 64
sum = 0
200.times { |i| sum += Holder.use(s, i) }
p sum

a = Vec2.new(3, 4)
b = Vec2.new(5, 6)
total = 0
100.times { total += a.dot(b) }
p total

cells = Array.new(16) { |i| i % 3 }
acc = 0
4.times { |y| 4.times { |x| acc += cell_get(cells, 4, 4, x - 1, y + 1) } }
p acc

p tag("hi")
p spelled("a")
