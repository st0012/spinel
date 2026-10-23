# A reopened tap that yields and answers self answers what Kernel's does:
# the tail tap's new String is still the method's own.
class Object
  def tap
    yield self
    self
  end
end
class R
  def initialize = (@b = String.new("abcdef"); @pos = 0)
  def take(n) = @b.byteslice(@pos, n).tap { @pos += n }
end
r = R.new
x = r.take(2)
x << "!"
p x, r.take(2)
