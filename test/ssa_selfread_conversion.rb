def parse(s)
  n = s
  n = n.to_i
  n + 1
end
def run(lines)
  t = 0
  i = 0
  while i < 2_000
    t += parse(lines[i % 3])
    i += 1
  end
  t
end
p run(["12", "7", "300"])
