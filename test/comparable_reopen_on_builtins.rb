# A method the program adds to Comparable is a method of every builtin
# class that includes it: Integer, Float, String and Symbol
# (and of the program's own includers).
module Comparable
  TAG = :cmp
  def ctag = [TAG, self]
  def cmid(lo, hi) = between?(lo, hi) ? :in : :out
end

class Box
  include Comparable
  attr_reader :v
  def initialize(v) = @v = v
  def <=>(o) = v <=> o.v
  def inspect = "#<Box #{v}>"
end

p 5.ctag, 1.5.ctag, "s".ctag, :s.ctag, Box.new(1).ctag
p "m".cmid("a", "z"), 7.cmid(1, 3)

# a boxed receiver
def tag_of(v) = v.ctag
p tag_of(5), tag_of("s"), tag_of(:s)
[5, "s", 1.5, :s].each { |v| p v.ctag }

# a class's own method of the name comes before Comparable's
class String
  def cown = :string
end
module Comparable
  def cown = :cmp
end
p "s".cown, 5.cown

# a builtin's own method comes before Comparable's
module Comparable
  def size = :cmp
end
p "abc".size
