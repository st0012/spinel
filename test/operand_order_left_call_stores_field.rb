# A call on the left of an operator can store a new value into a field
# that an argument on the right reads. Ruby reads the field after the
# left call has run, so the right side sees the new value.

# 1. The receiver is an Add everywhere.
class TNum
  attr_reader :v
  def initialize(v)
    @v = v
  end
end
class TAdd
  attr_accessor :l, :r
  def initialize(l, r)
    @l = l
    @r = r
  end
end
class TInterp
  def leaf(n)
    $t_target.r = TNum.new(100) if n.v == 0
    n.v
  end
  def visit_add(node)
    (leaf(node.l) + leaf(node.r)) % 1000003
  end
end
a = TAdd.new(TNum.new(0), TNum.new(1))
$t_target = a
p TInterp.new.visit_add(a)

# 2. The children are boxed (a Num or an Add).
class BNum
  attr_reader :v
  def initialize(v)
    @v = v
  end
end
class BAdd
  attr_accessor :l, :r
  def initialize(l, r)
    @l = l
    @r = r
  end
end
def b_leaf(n)
  if n.is_a?(BNum)
    $b_target.r = BNum.new(100) if n.v == 0
    n.v
  else
    -1
  end
end
def b_visit_add(node)
  (b_leaf(node.l) + b_leaf(node.r)) % 1000003
end
inner = BAdd.new(BNum.new(2), BNum.new(3))
b = BAdd.new(BNum.new(0), inner)
$b_target = b
p b_visit_add(b)
p b_visit_add(BAdd.new(inner, BNum.new(4)))

# 3. The receiver is narrowed by is_a?.
class INum
  attr_reader :v
  def initialize(v)
    @v = v
  end
end
class IAdd
  attr_accessor :l, :r
  def initialize(l, r)
    @l = l
    @r = r
  end
end
class IMut
end
def i_visit(node)
  if node.is_a?(IAdd)
    i_visit(node.l) * 1000 + i_visit(node.r)
  elsif node.is_a?(INum)
    node.v
  elsif node.is_a?(IMut)
    $i_target.r = INum.new(9)
    0
  else
    0
  end
end
c = IAdd.new(IMut.new, INum.new(1))
$i_target = c
p i_visit(c)

# 4. The receiver is narrowed by case/when.
class CNum
  attr_reader :v
  def initialize(v)
    @v = v
  end
end
class CAdd
  attr_accessor :l, :r
  def initialize(l, r)
    @l = l
    @r = r
  end
end
class CMut
end
class CInterp
  def visit(node)
    case node
    when CNum then node.v
    when CAdd then (visit(node.l) + visit(node.r)) % 1000003
    when CMut
      $c_target.r = CNum.new(100)
      0
    else 0
    end
  end
end
d = CAdd.new(CMut.new, CNum.new(1))
$c_target = d
p CInterp.new.visit(d)
p CInterp.new.visit(CAdd.new(CNum.new(5), CNum.new(7)))
