# A top-level include brings the module's instance methods and its module
# functions, not its own singleton methods: a bare call of one of those
# names raises NameError, as in CRuby.

module M
  def self.f = "cm"
  class << self
    def sc = "sc"
  end
  def g = "im"
  def self.both = "cm"
  def both = "im"
end

module A
  module_function

  def a1 = "a1"
end

module B
  def b1 = "b1"
  module_function :b1
end

include M, A, B

p g, both, a1, b1
p M.f, M.sc

begin
  p f
rescue NameError => e
  puts e.class
end

begin
  p sc
rescue NameError => e
  puts e.class
end

def top_f = f

class K
  def run = [a1, b1, g]

  def bare_f
    f
  rescue NameError => e
    e.class
  end
end

p K.new.run
p K.new.bare_f

begin
  top_f
rescue NameError => e
  puts e.class
end
