# A local a multiple assignment types as a String, then rebound to an Array
# of its own parts (RDoc::Markup::Formatter.gen_relative_url). The write
# pass reaches the rebind before the multiple assignment; a read there
# carries the last round's type only when it is concrete, so the local came
# out poly from a String and a String from poly, every round, to the cap.
def relative(target)
  to, to_file = File.split target
  to = to.split "/"
  to.delete "."
  [to, to_file]
end

p relative("a/./d/e.html")
