# A method whose value leaves through `return yield`, called with blocks of
# different types: each call answers its own block's value. The first call
# site used to fix the method's type, so a later site read nil, a raw
# pointer as an Integer, or crashed.
def with
  return yield
end

with { nil }
p(with { "xyz" })
p(with { 42 })
p(with { [1, 2] })

def guarded(n)
  begin
    raise ArgumentError, "negative" if n < 0
    return yield n
  ensure
    n = 0
  end
end

guarded(1) { |x| x.to_s }
p guarded(2) { |x| x * 10 }
p guarded(3) { |x| "#{x}!" }

def early(flag)
  return(yield) if flag
  :fallthrough
end

p early(true) { "taken" }
p early(false) { "unused" }
p early(true) { 7 }
