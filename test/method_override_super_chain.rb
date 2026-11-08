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

# a module that includes a module: its super reaches that module's method
module N3; def method(n) = "N3#{n}"; end
module M3; include N3; def method(n) = "M3:" + super; end
class C3; include M3; end
p C3.new.method(:x)

# a module included through another module, into a class whose superclass has method
class Base4; def method(n) = "Base4#{n}"; end
module Hook4; def method(n) = "h4:" + super; end
module Outer4; include Hook4; end
class C4 < Base4; include Outer4; end
p C4.new.method(:x)

# extend adds to the singleton class only, a nested class of the same leaf
# gets the include: the instance super is still Object#method
module Tag5; def method(n) = "Tag5#{n}"; end
module X5; class R5; end; end
X5::R5.include(Tag5)
class R5; extend Tag5; def method(n) = super; def k = 3; end
p R5.new.method(:k).call
