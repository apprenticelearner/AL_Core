
dictionary = {
  "add" : "+",
  "sum" : "+",
  "total" : "+",
  "count" : "+",
  "plus" : "+",
  "half" : "/2",

  "subtract" : "-",
  "difference": "-",

  "reduction": "-",
  "minus": "-",

  "multiply" : "x", 
  "product" : "x",
  "times" : "x",
  "double" : "x",
  
  "divide" : "/",
  "split": "/",
  "equals" : "=",

  "half" : "/2",

  "ones" : "[0]",
  "ones-digit" : "[0]",
  "tens" : "[1]",
  "tens-digit" : "[1]",

  "square" : "**2",
  "squared" : "**2",
}


special_patterns = {
  r"(\S+)\sdivided\sby\s(\S+)" : "/",
  r"(\S+)\stimes\s(\S+)" : "x",
  r"(\S+)\sminus\s(\S+)" : "-",
  r"(\S+)\splus\s(\S+)" : "+",
  

  r"ones\sdigit" : "[0]",
  r"ones'\sdigit" : "[0]",
  r"one's\sdigit" : "[0]",

  r"tens\sdigit" : "[1]",
  r"tens'\sdigit" : "[1]",
  r"ten's\sdigit" : "[1]",
}

not_main = {
    "take",
    "set",
    "find",
    "calculate",
    "apply"
}

#substitute numbers in the sentence with nouns for more accurate parsing
noun = "dog"
