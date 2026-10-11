# An empty [] a helper fills through its parameter holds what the helper pushes.
class Meth
  attr_reader :name, :singleton
  attr_accessor :visibility
  def initialize(name, singleton)
    @name = name
    @singleton = singleton
    @visibility = nil
  end
end

class Context
  attr_reader :method_list

  def initialize
    @method_list = []
  end

  def add_to(array, thing)
    array << thing
  end

  def add_method(meth)
    add_to @method_list, meth
  end
end

ctx = Context.new
ctx.add_method Meth.new("new", true)
ctx.add_method Meth.new("each", false)
ctx.add_method Meth.new("size", false)

class_methods, instance_methods = ctx.method_list.partition { |m| m.singleton }
p class_methods.map(&:name)
p instance_methods.map(&:name)

pair = ctx.method_list.partition { |m| m.name.start_with?("s") }
p pair.map { |ms| ms.map(&:name) }

ctx.method_list.first.visibility = :private
ctx.method_list.each do |m|
  m.visibility ||= :public
end
p ctx.method_list.map(&:visibility)

# a local, and Integers
def push_to(list, value)
  list.push value
end

nums = []
push_to nums, 3
push_to nums, 4
small, large = nums.partition { |n| n < 4 }
p small, large, nums.sum
