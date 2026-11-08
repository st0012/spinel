# A program's own `def method` with `super` (#10719 made a class's super
# Object#method). A module's own one does the same: its super, in a class
# that includes it, is Object#method when nothing of the program stands
# above it. Where a superclass or an included module defines `method`,
# super calls that one, not the builtin.
# a module's own method with super
module Hook
  def method(n)
    puts "Hook#method #{n}"
    super
  end
end
class H; include Hook; def keep(x) = x * 4; end
p H.new.method(:keep).call(2)

# a class whose superclass defines method
class P
  def method(n) = "P#{n}"
end
class Q < P
  def method(n)
    super
  end
end
p Q.new.method(:x)

# a class including a module with method: super reaches the module
module Tag
  def method(n) = "Tag#{n}"
end
class R
  include Tag
  def method(n) = super
end
p R.new.method(:y)

# module chain: H2 includes Hook2, which supers past Base's method
class Base
  def method(n) = "Base#{n}"
end
module Hook2
  def method(n) = "h2:" + super
end
class H2 < Base
  include Hook2
end
p H2.new.method(:z)

# class with method above a module with method above Kernel
module Inner
  def method(n) = (puts "Inner"; super)
end
class Outer
  include Inner
  def method(n) = (puts "Outer"; super)
  def two = 2
end
p Outer.new.method(:two).call

# unrelated classes each with their own method
class U1; def method(n) = super; def a = :a; end
class U2; def method(n) = super; def b = :b; end
p U1.new.method(:a).call, U2.new.method(:b).call

# a module mixed in from outside the class body
module Tag2
  def method(n) = "Tag2#{n}"
end
class R2
  def method(n) = super
end
R2.include(Tag2)
p R2.new.method(:w)
