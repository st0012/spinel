# A GC root matters only if its value is read after a point where a
# collection can start: an allocation, a safe point's poll, or a call that
# can reach one. Codegen runs liveness on each method's C and takes back
# the roots that no such point needs (src/codegen_gclive.c).
# spinel: decisions
class Node
  attr_accessor :key, :left, :right

  def initialize(key)
    @key = key
    @left = nil
    @right = nil
  end
end

# `h.left = insert(h.left, key)`: the temp that holds h.left becomes the
# callee's parameter, which the callee roots itself, and the caller does not
# read the temp again. h is read after the call (the store goes into it), so
# its root stays.
def insert(h, key)
  return Node.new(key) if h.nil?
  if key < h.key
    h.left = insert(h.left, key)
  elsif key > h.key
    h.right = insert(h.right, key)
  end
  h
end

def count(h)
  return 0 if h.nil?
  count(h.left) + count(h.right) + 1
end

# l is read after the second call, which can collect: its value is an
# Integer, so it needs no root either way
def depth(h)
  return 0 if h.nil?
  l = depth(h.left)
  r = depth(h.right)
  (l > r ? l : r) + 1
end

def keys(h, out)
  return out if h.nil?
  keys(h.left, out)
  out << h.key
  keys(h.right, out)
end

# kept: s is read after the String that `s + "!"` builds
def tag(s)
  t = s + "!"
  s + t
end

class Bag
  def initialize(items)
    @items = items
  end

  # reads self before anything can collect: the caller's receiver needs no root
  def total = @items.sum
end

root = nil
[50, 30, 70, 20, 40, 60, 80, 35, 45, 65].each { |k| root = insert(root, k) }
300.times { |i| root = insert(root, (i * 37) % 211) }
p count(root)
p depth(root)
p keys(root, []).first(12)
p tag("ab")
bags = (0...40).map { |i| Bag.new([i, i * 2, "x#{i}".size]) }
p bags.sum(&:total)
