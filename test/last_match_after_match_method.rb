# `$~` after Regexp#match / String#match is that match, and `$~` in a method
# is the method's own: nil until the method matches.
# Regexp#match built its MatchData from positions of its own, and `$~` read
# the positions of an older match, so its groups came out as "".

m = /(a)(b)?(c)/.match("ac")
p $~
p m
p $~.to_a
p $~[1], $~[2], $~.pre_match, $~.post_match
def f(*a) = a
p f(*$~)

"xay".match(/(a)/)
p $~
p $~.begin(0)
"bca".match(/(c)(a)/, 1)
p $~.to_a
p Regexp.last_match(2)
/(q)/.match("zz")
p $~

"ac" =~ /(a)(c)/
def reads_last = $~
p reads_last
def reads_group = $1
p reads_group
def reads_amp; v = $&; "q" =~ /q/; [v, $&]; end
p reads_amp
def reads_last_match = Regexp.last_match
p reads_last_match
def helper(s)
  if s =~ /(\d+)/
    $1.to_i
  else
    $~
  end
end
p helper("a12")
p helper("zz")
p $~
p $1
