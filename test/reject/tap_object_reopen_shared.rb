# spinel: reject-subclass: cannot carry the shared handle yet
# A tail `tap` on a new String is fresh only when it is Kernel's tap. Here
# an Object reopening gives tap another answer, a String another name holds,
# so the method's answer is shared and changed in place: refused.
HELD = String.new("held")
class Object; def tap = HELD; end
class R
  def initialize = (@b = String.new("abcdef"); @pos = 0)
  def take(n) = @b.byteslice(@pos, n).tap { @pos += n }
end
x = R.new.take(2)
x << "!"
p HELD, x
