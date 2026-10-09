# A builtin's constant read through its path is the builtin's, even when
# a module of the program defines a constant with the same last name. The
# path used to resolve by that last name alone, so Encoding::BINARY read
# prism's IntegerBaseFlags::BINARY (an Integer) and comparing a binary
# String's encoding with it answered false.
module Flags
  BINARY = 1 << 2
  ASCII_8BIT = 1 << 7
  UTF_8 = 1 << 9
  MAX = 3
  MULTILINE = 7
end

s = "ab".b
p s.encoding == Encoding::BINARY
p s.encoding == Encoding::ASCII_8BIT
p "ab".encoding == Encoding::UTF_8
p Encoding::BINARY.name
p Encoding::UTF_8.name
p Float::MAX > 1e300
p Regexp::MULTILINE
p [Flags::BINARY, Flags::ASCII_8BIT, Flags::UTF_8, Flags::MAX, Flags::MULTILINE]

include Flags
p BINARY

# The parent resolves lexically, as in CRuby: inside a module that defines
# its own Encoding the path is that module's, elsewhere it is the builtin's.
module Lib
  module Encoding
    UTF_8 = "lib-utf8"
  end
  def self.inside = Encoding::UTF_8
  def self.outer = ::Encoding::UTF_8.name
end
module Other
  def self.bin?(s) = s.encoding == Encoding::BINARY
end
p Lib.inside
p Lib.outer
p Other.bin?("x".b)
