import spacy
import coreferee
from spacy.symbols import nsubj
from spacy import displacy
from spacy.matcher import Matcher
from spacy.language import Language
from spacy.tokens import Token
from numpy import maximum
import re
import string
import ast
from copy import copy

# ----------------------------------------------------------------
# : Policy building utilities

def join_policy(policy, other_policy):
  if(len(policy) == 0): return [*other_policy]

  # Deep copy the policy
  policy = [[*dp] for dp in policy]
  if(len(policy) < len(other_policy)):
    for i in range(len(other_policy)-len(policy)):
      policy.append([])
  
  for i, arr in enumerate(other_policy):
    policy[i] = [*policy[i],*arr]
  return policy

def ensure_length(policy, length, prepend=False):
  if(len(policy) < length):
    if(prepend):
      for i in range((length)-len(policy)):
          policy.insert(0,[])
    else:
      for i in range((length)-len(policy)):
          policy.append([])
  return policy

# ----------------------------------------------------------------
# : Code/Math detection and parsing

# -----------------------
# : find_code_sections

unary_op_regex = r"([\+|\-|\!|~]+)"
binary_op_regex = r"(\+|\-|\*{1,2}|\/{1,2}|%|==|!=|>{1,2}|<{1,2}|<=|>=|&{1,2}|\|{1,2}|\^)"


def _scan_end_parens(s):
    paren_count = 1
    i = 0
    for i, c in enumerate(s):
        if(c == "("):
            paren_count += 1
        elif(c == ")"):
            paren_count -= 1
        if(paren_count == 0): break
    if(paren_count == 0):
        return i
    else:
        return -1

def _span_overlap(ind, spans):
    for i, (s0,s1,s) in enumerate(spans):
        if(ind >= s0 and ind <= s1):
            return i
    return -1
quotes = []

def operand_okay(s):
  is_str = re.fullmatch(r'[\"\']\w*[\"\']',s) != None
  is_num = re.fullmatch(r'\d*\.?\d*',s) != None
  is_char = re.fullmatch(r'\w',s) != None
  # print(s, ":", is_str, is_num, is_char)
  return (is_str or is_num or is_char) # is single characer

def find_code_sections(s):
    spans = []
    
    # Find beginnings of functions and parentheses.
    offset, _s = 0, copy(s)
    #                            funcs or   unary in paren   or   binary in paren
    while(match:=re.search(fr'''(\w+\()|(\({unary_op_regex})|(\([\w'\."]*\s*{binary_op_regex})''', _s)):
        s0,s1 = match.span()
        end = _scan_end_parens(_s[s1:])
        if(end != -1):
            s0, s1 = offset+s0, offset+s1+end+1
            spans.append((s0, s1, s[s0:s1]))
            _s, offset = s[s1:], s1
        else:
            raise NotImplementedError()

    
    

    # Find unary ops
    offset, _s = 0, copy(s)
    while(match:=re.search(rf'''[\(|\s+]{unary_op_regex}([\w\(\.'"]*)''', _s)):
        s0, _ = match.span(1)
        _, s1 = match.span(2)
        s0 += offset; s1 += offset 
        # print("<<", match.group(0), s0, s1)
        ind = _span_overlap(s1, spans)
        if(ind != -1):
            _s0, s1, _ = spans[ind]
            s0 = min(s0,_s0) # for cases like (~4)
            spans[ind] = (s0, s1, s[s0:s1])
        else:
            spans.append((s0, s1, s[s0:s1]))
        _s, offset = s[s1:], s1

    # Find binary ops
    offset, _s = 0, copy(s)
    while(match:=re.search(rf'''([\w\)\.'"]*)\s*{binary_op_regex}\s*([\w\(\.'"]*)''', _s)):
        ls0, ls1 = match.span(1)
        ls0 += offset; ls1 += offset
        l_ind = _span_overlap(ls0, spans)

        op_str = match.group(2)

        rs0, rs1 = match.span(3)
        rs0 += offset; rs1 += offset
        r_ind = _span_overlap(rs1, spans)

        # print("<< 0", match.group(0))
        # print("<< 1", match.group(1))
        # print("<< 2", match.group(2))
        # print("<< 3", match.group(3))
        # print(repr(s[ls1-1]), repr(s[rs0]), (op_str == "%", s[ls1-1] != " ", s[rs0] == " "))

        # Skip when l_ind == r_ind since that means the match 
        #  is enclosed by an existing span of discovered code.
        if((l_ind != -1 and l_ind == r_ind) or
          # Also skip if either left or right are empty
            ls0 == ls1 or rs0 == rs1 or
          # Skip if operands not string, number, or single character
            (l_ind == -1 and not operand_okay(s[ls0:ls1])) or 
            (r_ind == -1 and not operand_okay(s[rs0:rs1])) or
          # Skip if special case "A% B"
            (op_str == "%" and s[ls1-1] != " " and s[rs0] == " ")):
            _s, offset = s[rs1:], rs1
            continue
                
        if(l_ind != -1 and r_ind != -1):
            s0, s1 = spans[l_ind][0], spans[r_ind][1]
            if(r_ind > l_ind):
                del spans[r_ind]; del spans[l_ind];
            else:
                del spans[l_ind]; del spans[r_ind];
            spans.append((s0,s1,s[s0:s1]))

        elif(l_ind != -1):
            s0, s1 = spans[l_ind][0], rs1
            spans[l_ind] = (s0,s1,s[s0:s1])
        elif(r_ind != -1):
            s0, s1 = ls0, spans[r_ind][1]
            spans[r_ind] = (s0,s1,s[s0:s1]) 
        else:
            s0, s1 = ls0, rs1
            spans.append((s0,s1,s[s0:s1]))            

        _s, offset = s[s1:], s1

    # Sort in left-to-right order
    spans = sorted(spans)

    # for (s0,s1,s) in spans:
    #     print(s)

    return spans

# ---------------------------
# : ast_to_policy

# Give concrete names to various AST objects
ast_to_str = {
  # Unary Ops
  ast.UAdd : "Positive",
  ast.USub : "Negative",
  ast.Not : "Not",
  ast.Invert : "Invert",

  # Binary Ops
  ast.Add : "Add",
  ast.Sub : "Subtract",
  ast.Mult : "Multiply",
  ast.Div : "Divide",
  ast.Mod : "Modulus",
  ast.FloorDiv : "DivideInt",
  ast.Pow : "Power",
  ast.LShift : "BitshiftLeft",
  ast.RShift : "BitshiftRight",
  ast.BitOr : "BitwiseOR",
  ast.BitXor : "Power", # Since XOR is rare make this power instead
  ast.BitAnd : "BitwiseAND",
  ast.MatMult : "MatrixMultiply",

  # Comparators
  ast.Eq : "Equal",
  ast.NotEq : "NotEqual",
  ast.Lt : "LessThan",
  ast.LtE : "LessThanOrEqual",
  ast.Gt : "GreaterThan",
  ast.GtE : "GreaterThanOrEqual",
  ast.Is : "Is",
  ast.IsNot : "IsNot",
  ast.In : "In",
  ast.NotIn : "NotIn",
}


def _ast_to_op_or_const(ast_obj, recurse, *args):
  func_map, var_map,  const_operands = tuple([*args])
  if(isinstance(ast_obj, ast.Call)):
      op = ast_obj.func.id
      args = [recurse(a, *args) for a in ast_obj.args]
  elif(isinstance(ast_obj, ast.UnaryOp)):
      op = ast_to_str[type(ast_obj.op)]
      args = [recurse(ast_obj.operand, *args)]
  elif(isinstance(ast_obj, (ast.BinOp, ast.Compare))):
      op = ast_to_str[type(ast_obj.op)]
      args = [recurse(ast_obj.left, *args),
            recurse(ast_obj.right, *args)]
  elif(isinstance(ast_obj, ast.Constant)):
      print("dD CONST", ast_obj.value)
      const_operands.add(ast_obj.value)
      return ast_obj.value
  elif(isinstance(ast_obj, ast.Name)):
    return var_map.get(ast_obj.id, ast_obj.id)
  else:
    return ""
  op = func_map.get(op.lower(), op)
  return op, args

def _ast_to_policy(ast_obj, func_map, var_map, const_operands):
    val = _ast_to_op_or_const(ast_obj, _ast_to_policy, func_map, var_map, const_operands)

    if(not isinstance(val, tuple)):
      return val

    op, args = val

    policy_args = []
    policy = [[(op, policy_args)]]
    
    for i, arg in enumerate(args):
        if(isinstance(arg, list)):
            child_policy = arg
            policy = ensure_length(policy, len(child_policy)+1, prepend=True)
            policy = join_policy(policy, child_policy)
        else:
            policy_args.append(arg)
    return policy

def ast_to_policy(ast_parse, func_map={}, var_map={}, const_operands=None):
    const_operands = set() if const_operands is None else const_operands
    return _ast_to_policy(ast_parse.body[0].value, func_map, var_map, const_operands)

def _ast_to_func(ast_obj, func_map, var_map):
    val = _ast_to_op_or_const(ast_obj, func_map, var_map, _ast_to_func)
    if(not isinstance(val, tuple)):
      return val

    op, args = val
    if(isinstance(op, str)):
      raise ValueError(f"{op!r} was not found in func_map.")
    if(not hasattr(op, '__call__')):
      raise ValueError(f"{op} not a callable function.")
    return op(*args)

def ast_to_func(ast_parse, func_map, var_map):
    return _ast_to_func(ast_parse.body[0].value, func_map, var_map)


# def extract_code(text):
  

# ----------------------------------------------------------------
# : TextToPolicyParser

def is_number(x):
  try:
    float(x)
  except ValueError:
    return False
  return True

# Replaces numbers with specified noun from resources
# Returns [modified string, indicies of previous numbers]
# def replace_numbers(nlp, text):
#   doc_text = nlp(text)
#   modified_string = ""
#   modified_positions = {}

#   for token in doc_text:
#     if is_number(str(token)):
#       modified_string += noun + " "
#       modified_positions[token.i] = str(token)
#     else:
#       modified_string += str(token) + " "

#   returnvalue = dict()
#   returnvalue['modified_string'] = modified_string
#   returnvalue['modified_positions']  = modified_positions
#   return returnvalue


def first_non_empty(arr):
  if len(arr) == 0: return []
  for x in arr:
    if(len(x) == 0):
      continue
    return x
  return []






class TextToPolicyParser():
  ''' Generates policies from text.'''
  def __init__(self, func_dictionary, special_patterns={},
    spacy_model="en_core_web_trf", use_func_key=True, 
    display_parse=False, display_options={}):
    self.func_dictionary = func_dictionary
    self.special_patterns = special_patterns
    self.use_func_key = use_func_key
    self.display_parse = display_parse
    self.display_options = display_options

    from cre.utils import PrintElapse
    with PrintElapse("spacy load time"):
      self.nlp = spacy.load(spacy_model)
      

    self.nlp.add_pipe('coreferee')

    # Custom pipleine that forces ops to have POS == "VERB"
    #  doesn't seem to have an effect on downstream grammar parsing.
    if(False):
      @Language.component("force_ops_verbs")
      def force_ops_verbs(doc):
          matcher = Matcher(doc.vocab)
          patterns = []
          for lemma in self.func_dictionary:
            patterns.append([{'LEMMA': lemma}])
          matcher.add('OpsAreVerbs', patterns)
          
          matches = matcher(doc)
          
          for match_id, start, end in matches:
            # print(doc[start].text)
            #print("Match found:", doc[start].text)
            doc[start].pos_ = 'VERB'    
              
          return doc
      # print(self.nlp.pipe_names)
      self.nlp.remove_pipe("parser")
      self.nlp.add_pipe("force_ops_verbs", after="lemmatizer")
      self.nlp.add_pipe("parser", after="force_ops_verbs", source=spacy.load(spacy_model))

    d = {}
    for k, v in func_dictionary.items():
      # print("!!", list(self.nlp(k))[0].lemma_.lower())
      token = list(self.nlp(k))[0]
      lemma = token.lemma_.lower()
      d[str(token)] = v
      d[lemma] = v
    self.func_dictionary = d
    # print(self.func_dictionary)

  def replace_code(self, text, const_operands=None):
    const_operands = set() if(const_operands is None) else const_operands
    code_sections = find_code_sections(text)
    code_policies = []
    
    # Loop backward so that replacement locations stay valid
    for (s0, s1, code) in reversed(code_sections):
      try:
        ast_parse = ast.parse(code)
      except Exception:
        continue

      policy = ast_to_policy(ast_parse, self.func_dictionary, const_operands=const_operands)

      text = f"{text[:s0]}xfunc{text[s1:]}"
      code_policies.append(policy)
    return text, code_policies[::-1]

  def is_only_code(self, text):
    pos = 0
    for match in list(re.finditer(r"\W*xfunc\W*", text)):
      s0, s1 = match.span()
      if(pos != s0):
        break
      pos = s1
    return pos >= len(text)-1

  def op_for_token(self, token):
    op = self.func_dictionary.get(str(token).lower(),None)
    if(op is None):
      op = self.func_dictionary.get(token.lemma_.lower(),None)
    return op

  def replace_special(self, text, code_policies, const_operands=None):
    special_ops = []
    const_operands = set() if(const_operands is None) else const_operands
    for pattern, op in self.special_patterns.items():
      # print(pattern)
      matches = list(re.finditer(pattern, text))
      while(len(matches) > 0):
        match = matches[0]
        # print(match)
        g = list(match.groups())
        
        for val in g:
          # print("SPECIAL CONST", val)
          if(val not in ["xfunc", "xlick", "xcat"]):
            const_operands.add(val)        

        if('xfunc' in g):
          policy = [[(op, [])]]
          for i, arg in enumerate(g):
            if(arg == "xfunc"):
              s0,s1 = match.span(i+1)
              code_policy = code_policies[text[:s0].count("xfunc")]
              policy = ensure_length(policy, len(code_policy)+1, prepend=True)
              policy = join_policy(policy, code_policy)
        else:
          policy = [[(op, g)]]

        s0,s1 = match.span()

        # If none of the args are #cat add a special op.
        # print(g, [a for a in g if a not in ["#cat","#lick"]])
        if(len(g) == 0):
          special_ops.insert(text[:s0].count("xlick") + text[:s0].count("xcat"), policy)
          text = f"{text[:s0]}xlick{text[s1:]}"
          matches = list(re.finditer(pattern, text))
        elif(len([a for a in g if a in ["xcat","xlick"]]) == 0):
          special_ops.insert(text[:s0].count("xlick") + text[:s0].count("xcat"), policy)
          text = f"{text[:s0]}xcat{text[s1:]}"
          matches = list(re.finditer(pattern, text))
        else:
          matches = matches[1:]

    return text, special_ops

  def replace_numbers(self, text, const_operands=None):
    gparse = self.nlp(text)
    arg_inds = {}
    words = []
    const_operands = set() if const_operands is None else const_operands
    for token in gparse:
      # if token.like_num:# and len(str(token)) == 1:
      token_str = str(token)
      if is_number(token_str):#token.like_num:# and len(str(token)) == 1:
        # print("SPECIAL NUMBER", token_str)
        const_operands.add(token_str)
        words.append("cat")
        # words.append(str(token))
        # modified_string += "dog" + " "
        arg_inds[token.i] = token
      else:
        words.append(token_str)
        # modified_string += str(token) + " "

    self.arg_inds = arg_inds

    return " ".join(words)

  def replace_funcs(self, text):
    gparse = self.nlp(text)
    func_inds = {}
    words = []
    for token in gparse:
      op = self.op_for_token(token)
      # print("<<",token, token.lemma_, op)
      if(op is not None):        
        # print("**", op, token.pos_)
        if(token.pos_ not in ["VERB", "NOUN", "PROPN"]):
          words.append("fetch")
        else:
          words.append(str(token))
        func_inds[token.i] = token if(self.use_func_key) else op
      else:
        words.append(str(token))

    self.func_inds = func_inds

    return " ".join(words)

  def annotate_special(self, doc, special_ops, code_policies):
    # print(special_ops)
    i = 0
    c = 0
    self.special_inds = {}
    for token in doc:
      # if(str(doc[token.i-1]) =="#"): 
      if(str(token) == 'xlick'): # Recognized Function Case
        self.func_inds[token.i] = special_ops[i][0]
        i += 1
      if(str(token) == 'xcat'): # Special Pattern Case (i.e. 7 times 9)
        self.special_inds[token.i] = special_ops[i]
        i += 1
      if(str(token) == 'xfunc'): # Code Case
        self.special_inds[token.i] = code_policies[c]
        c += 1

  def recurse(self, token, args=[]):
    # Case: Token is operator
    if(token.i in self.func_inds):
      policy, arg_lists = [[]], []
      op = self.func_inds[token.i]

      
      # print("<<", token, token._.special_op)
      # if(token._.special_op is not None):
        
      # print("<<", token)
      dep_policies = []

      for child in token.children:
        if(child.dep_ in ["dep"]): continue # Ignore "dep"
        _args = []
        child_policy = self.recurse(child, _args)

        coref_tokens = self.doc._.coref_chains.resolve(child)
        if(coref_tokens is not None):
          for coref_token in coref_tokens:
            # print("<<", token, child, coref_token)
            if(coref_token.i in self.policy_inds):
              coref_policy = self.policy_inds[coref_token.i]
              policy = ensure_length(policy, len(coref_policy)+1)  

        # If the child emitted a policy (i.e. it has a downstream op) then
        #  add a new arg_set, otherwise fold the arg into the most recent arg_set. 
        if(len(child_policy) > 0 or len(arg_lists) == 0):
          arg_lists.append(_args)
        else:
          arg_lists[-1] = [*arg_lists[-1], *_args]

        # If the child is direct/indirect object-like then increment the policy length.
        if(child.dep_ in ["dobj", "prep", "pobj", "compound", "dative"]):
          policy = ensure_length(policy, len(child_policy)+1)
        # print(i, child, policy)

        policy = join_policy(policy, child_policy)

      # if(token.i in self.special_inds):
      #   policy = join_policy(policy, [self.special_inds[token.i]])

      # If token is in clause like "3 times 4" then treat any preceeding
      #  arguments as belonging to this operator.
      leading_args = []
      if(token.dep_ in ["acl", "compound", "cc", "conj", 'dep']):
        leading_args = [*args]        

      # Pick the first non-empty arg_list among the child arg_lists.
      child_args = first_non_empty(arg_lists)

      # Ensure argument tokens are in sentence order.
      args = sorted(leading_args+child_args, key=lambda x:x.i)

      # Add the operator-arguments pairs at the deepest available depth.
      policy[-1].append((op, args))

      
    # Case: Token is an argument or filler.
    else:
      policy = []
      
      # If token is a special token then use it's current policy
      if(token.i in self.special_inds):
        policy = join_policy(policy, self.special_inds[token.i])
        # print(policy)

      # If token was marked as an argument then add to args
      if(token.i in self.arg_inds):
        args.append(self.arg_inds[token.i])
          
      for child in token.children:
        if(child.dep_ in ["dep"]): continue # Ignore "dep"
        child_policy = self.recurse(child, args)
        policy = join_policy(policy, child_policy)

    # Handle any 'dep' dependencies like "this statement, 'then' that statment."
    #  Add these at one depth greater than the current policy.        
    dp_len = len(policy)+1
    for child in token.children:
      if(child.dep_ not in ["dep"]): continue
      _args = []
      child_policy = self.recurse(child, _args)
      child_policy = ensure_length(child_policy, dp_len, prepend=True)
      policy = join_policy(policy, child_policy)

    self.policy_inds[token.i] = policy

    return policy   

      # print(depth, token, list(token.children))



  def parse(self, text, return_consts=False):
    # Lower Case
    text = text.lower()

    # Remove any trailing puctuation
    # text = text[:-1] + text[-1].translate(str.maketrans("","", string.punctuation))

    # Replace Any Code Sections
    const_operands = set()
    text, code_policies = self.replace_code(text, const_operands)
    print("const_operands", const_operands)
    # if(len(code_policies) > 0):
    #   print("code_policies:")
    #   for i, cp in enumerate(code_policies):
    #     print(i, cp)

    # If turned out to entirely code then skip other steps 
    if(self.is_only_code(text)):
      _policy = [[]]
      for cp in code_policies:
        _policy = join_policy(cp, _policy)

      if(return_consts):
        return _policy, const_operands
      else:
        return _policy  

    # Make special, number and function replacements 
    text, special_ops = self.replace_special(text, code_policies, const_operands)
    text = self.replace_numbers(text, const_operands)
    text = self.replace_funcs(text)

    # print("INERN", const_operands)

    # Final Grammar Parse + Coreference Resolution
    self.doc = doc = self.nlp(text)
    # doc._.coref_chains.print()
    self.annotate_special(doc, special_ops, code_policies)

    # Display
    if(self.display_parse):
      displacy.render(doc, style="dep", jupyter=True, options=self.display_options) # Uncomment on Google Colab to show graph

    # Recurse Through Each Root and Join Policies
    roots = [token for token in doc if token.dep_=="ROOT"]
    self.policy_inds = {}
    _policy = []
    for root in roots:
      _policy = join_policy(self.recurse(root, []), _policy)
      
    # Remove redundant references to args + make args strings.
    policy = []
    covered_inds = set()
    for dp in _policy:
      new_dp = []
      for op, args in dp:
        new_dp.append((op, [str(tk) for tk in args if not isinstance(tk,Token) or tk.i not in covered_inds]))
        for a in args:
          if isinstance(a,Token):
            covered_inds.add(a.i)
      policy.append(new_dp)

    print("BLARG const_operands", const_operands)
    if(return_consts):
      return policy, const_operands
    else:
      return policy

  # Returns [closest word in dictionary, score]
  def get_closest_word(self, word, dic):
    max_score = 0.0
    most = ""
    doc_word = self.nlp(word)

    for reference in dic:
      doc_reference = self.nlp(reference)
      score = doc_word.similarity(doc_reference)

      if (score > max_score):
        max_score = score
        most = reference

    doc_most = self.nlp(most)
    print(doc_word, "<->", doc_most, max_score)

    returnvalue = dict()
    returnvalue['closest_word'] = most
    returnvalue['score']  = max_score
    return returnvalue

  def __call__(self, text, *args, **kwargs):
    return self.parse(text, *args, **kwargs)

# ----------------------------------------------------------------
# : Test Cases

def policy_to_strs(policy):
  return [sorted([(str(op).lower(), [str(x) for x in args]) for op,args in dp]) for dp in policy]

def test_language_cases():
  from colorama import init, Fore, Back
  from resources import dictionary, special_patterns, not_main, noun
  init(autoreset=True)
  parser = TextToPolicyParser(dictionary, special_patterns)

  def test_it(text, out, backup=None):
    print(text)
    policy = parser(text)
    okay = policy_to_strs(policy) == policy_to_strs(out)
    if(okay):
      print(Back.GREEN + str(policy))
    else:
      almost = (policy_to_strs(policy) == policy_to_strs(backup)) if backup else False
      if(almost):
        print(Back.YELLOW + str(policy))
      else:
        print(Back.RED + str(policy))

  # test_it("Subtract the first y-coordinate 4 from the second y-coordinate 8, then subtract the first x-coordinate 5 from the second x-coordinate 7, and then divide the difference of the y-coordinates (4) by the difference of the x-coordinates (2).", [])
  # return 
  test_it(
    "Multiply 3 and 2.",
    [[("multiply", ['3', '2'])]]
  )

  test_it(
    "Subtract the product of 3 and 4 from the sum of 5 and 6",
    [[('product', ['3', '4']), ('sum', ['5', '6'])], [('subtract', [])]]
  )

  test_it(
    "Take the product of 3 and 4 and the sum of 5 and 6 and add them",
    [[('product', ['3', '4']), ('sum', ['5', '6'])], [('add', [])]]
  )

#
  test_it(
    "Add the product of 3 and 4 and the product of 5 and 6",
    [[('product', ['3', '4']), ('product', ['5', '6'])], [('add', [])]]
  )

  test_it(
    "Take the sum of the product of 3 and 5 and the product of 4 and 2.",
    [[('product', ['3', '5']), ('product', ['4', '2'])], [('sum', [])]]
  )

  # test_it(
  #   "Subtract the first digit by the second digit and then multiply it by 3",
  #   [[('subtract', ['digit', 'digit'])], [('multiply', ['3'])]]
  # )

  # test_it(
  #   "Subtract the first one by the second one and then multiply it by 3 and then add 5",
  #   [[('subtract', ['one', 'one'])], [('multiply', ['3'])], [('add', ['5'])]]
  # )

#
  test_it(
    "Set x to the sum of 3 and 4 and then multiply it by the sum of 5 and 6",
    [[('sum', ['3', '4']), ('sum', ['5', '6'])], [('multiply', [])]]
  )

  # test_it(
  #   "Take the product of 3 and 4 and the sum of 5 and 6 and add them and then subtract it from the sum of 6 and 8",
  #   [[('product', ['3', '4']), ('sum', ['5', '6'])], [('add', [])], [('sum', ['6', '8'])], [('subtract', [])]]
  # )



  test_it(
    "Subtract half of 20 times 12 from half of 10 times 6",
    [[("x", ['20','12']), ("x", ['10','6'])],[('half',[]),('half',[])], [("subtract",[])] ]
  )

  test_it(
    "Take the one's digit of 6 plus 4",
    [[("+", ['6', '4'])],[("[0]", [])]],
  )

  # One's Napol came up with

  test_it(
    "3 divided by 12 times 100",
    [[('/', ['3', '12'])], [('times', ['100'])]],
    [[('/', ['3', '12']), ("times", ['100'])]]
  )

  test_it(
    "Subtract 2 times 209 from 836.5 and the divide it by 6.5 minus 2.",
    [[("x", ['2', '209']), ("-", ['6.5', '2'])],[("subtract", ["836.5"])], [("divide", [])]],
    [[('x', ['2', '209']), ('-', ['6.5', '2.'])], [("divide", []), ("subtract", ['836.5'])]]
  )

  test_it(
    "3 times 2",
    [[("x", ['3','2'])]]
  )

  test_it(
    "5 times 3 plus 4 times 2",
    [[('x', ['5', '3']), ('x', ['4', '2'])], [('plus', [])]]
  )

  test_it(
    "Add 9 and 8",
    [[('add', ['9', '8'])]]
  )

  test_it(
    "135 divided by 360 times 6 squared",
    [[('/', ['135', '360']),("squared", ['6'])], [("times", [])]],
    [[('/', ['135', '360'])], [("times", [])], [("squared", ['6'])]]
  )

  test_it(
    "8 divided by 2 times 1",
    [[('/', ['8', '2'])], [("times", ['1'])]]
  )



def test_code_cases():
  from colorama import init, Fore, Back
  from resources import dictionary, special_patterns, not_main, noun
  init(autoreset=True)
  parser = TextToPolicyParser(dictionary, special_patterns)

  def test_it(text, out, backup=None):
    print(text)
    policy = parser(text)
    okay = policy_to_strs(policy) == policy_to_strs(out)
    if(okay):
      print(Back.GREEN + str(policy))
    else:
      almost = (policy_to_strs(policy) == policy_to_strs(backup)) if backup else False
      if(almost):
        print(Back.YELLOW + str(policy))
      else:
        print(Back.RED + str(policy))


  test_it(
    "(135 / 360) * (6**2)",
    [[('Divide', ['135', '360']),("Power", ['6', '2'])], [("Multiply", [])]]
  )

  test_it(
    "Double(135)*77 + (9%2).",
    [[('Double', ['135']), ('Modulus', ['9', '2'])], [('Multiply', ['77'])], [('Add', [])]]
  )

def test_mixed_cases():
  from colorama import init, Fore, Back
  from resources import dictionary, special_patterns, not_main, noun
  init(autoreset=True)
  parser = TextToPolicyParser(dictionary, special_patterns)

  def test_it(text, out, backup=None):
    print(text)
    policy = parser(text)
    okay = policy_to_strs(policy) == policy_to_strs(out)
    if(okay):
      print(Back.GREEN + str(policy))
    else:
      almost = (policy_to_strs(policy) == policy_to_strs(backup)) if backup else False
      if(almost):
        print(Back.YELLOW + str(policy))
      else:
        print(Back.RED + str(policy))

  

  test_it(
    "Subtract 1/2 from 5*2",
    [[('/', ['1', '2']), ('x', ['5', '2'])], [('Subtract', [])]]
  )

  test_it(
    "4^2 times 3^2 plus 4",
    [[('Power', ['4', '2']), ('Power', ['3', '2'])], [('x', []), ('plus', ['4'])]]
  )

  # These won't produce valid policies but cover some code parsing edge cases 

  test_it(
    "I'm talking about !1 and 1+f(x) and doof(blouble(1+2)))+2 and f(x)**g(x) then +-~(1+2), moreover 7+7.",
    [[]]
  )

  test_it(
    "In order to simplify the fractions, 3/4 will be multiplied by 3 to get 9/12. 2/3 will be multiplied by 4 to get 8/12. Now that the denominator of both fractions is 12, they are ready to be simplified.",
    [[]]
  )

  test_it(
    "The value 'ab' can by obtained by 'a'+'b'",
    [[]]
  )

  test_it(
    'The value "ab" can by obtained by "a"+"b" ',
    [[]]
  )

  test_it(
    'f(4) is a func (1) and (2) are not but (1+2), 1+(2*4) and (~4).',
    [[]]
  )

  test_it(
    '45% and 30% but 1%2 and 1 % 2, y-coordinate',
    [[]]
  )

def test_ast_to_func():
  from numba import f8
  from cre import Var
  from cre.default_funcs import Add, Multiply, Power

  func_map = {
    "add" : Add,
    "multiply" : Multiply,
    "power" : Power,
  }

  var_map = {
    "A" : Var(f8, "A"),
    "B" : Var(f8, "B"),
    "C" : Var(f8, "C"),
    "D" : Var(f8, "D"),
  }

  print(ast_to_func(ast.parse("A*B+C**D"), func_map, var_map))
  print(ast_to_func(ast.parse("(A^B)*C+D"), func_map, var_map))


if __name__ == "__main__":
  import sys

  # parser = TextToPolicyParser(dictionary)

  

  # test_language_cases()
  # test_code_cases()
  # test_mixed_cases()
  test_ast_to_func()

  #default input
  # test_string = "Subtract the product of 3 and 4 from the sum of 5 and 6 and 7"

  # if len(sys.argv) > 1:
  #   test_string = sys.argv[1]

  # print(test_string + " parsed is: ")
  # math_parsing = parser(test_string)
  # print(math_parsing)
  # print(parser("Multiply the sum of 1 and 2 with the product of 3 and 4."))
  # print(parser("Multiply the sum of 1 and 2 with the product of 3 and 4."))

  # parser("Subtract half of 20 times 12, from half of 10 times 6.")

  # print(parser("The sum of 1 multiplied by 2 and 3 multiplied by 4."))
  # print(parser("The sum of 1 times 2 and 3 times 4."))
  # print(parser("The sum of 1 plus 2 and 3 plus 4."))
  # print(parser("Multiply the sum of 1 and 2 with the product of 3 and 4."))

  # test_string = "Take the product of the sum of the numerator 7 and the numerator 9 and the sum of the denominator 15 and the denominator 19."
  # print(parser(test_string))

  # test_string = "Add the left converted numerator 9 and the right converted numerator 8."
  # print(parser(test_string))

  

#Testing



  # test_string = "Subtract the product of 3 and 4 from the sum of 5 and 6"
  # print(parser(test_string))
  # assert parser(test_string) == [[('product', ['3', '4']), ('sum', ['5', '6'])], [('subtract', [])]]

  # test_string = "Take the product of 3 and 4 and the sum of 5 and 6 and add them"
  # assert parser(test_string) == [[('product', ['3', '4']), ('sum', ['5', '6'])], [('add', [])]]

  # test_string = "Add the product of 3 and 4 and the product of 5 and 6"
  # assert parser(test_string) == [[('product', ['3', '4']), ('product', ['5', '6'])], [('add', [])]]

  # test_string = "Subtract the first digit by the second digit and then multiply it by 3"
  # assert parser(test_string) == [[('subtract', ['digit', 'digit'])], [('multiply', ['3'])]]

  # test_string = "Subtract the first one by the second one and then multiply it by 3 and then add 5"
  # print(parser(test_string))
  # # assert parser(test_string) == [[('subtract', ['one', 'one'])], [('multiply', ['3'])], [('add', ['5'])]]

  # test_string = "Set x to the sum of 3 and 4 and then multiply it by the sum of 5 and 6"
  # assert parser(test_string) == [[('sum', ['3', '4']), ('sum', ['5', '6'])], [('multiply', [])]]

  # test_string = "Take the product of 3 and 4 and the sum of 5 and 6 and add them and then subtract it from the sum of 6 and 8"
  # assert parser(test_string) == [[('product', ['3', '4']), ('sum', ['5', '6'])], [('add', [])], [('sum', ['6', '8'])], [('subtract', [])]]

  

# CASES
## 



# test_cases()
  # test_string =  "Divide 3 by 4 and then add the product of that and 5 and the product of 5 and 6"
  # assert parser(test_string) == [["divide"], ["product", "product"], ["add"]]

  # test_string = "Divide x by y and then take the answer and divide it by the sum of 7 and 8 and the product of 5 and 7"
  # assert parser(test_string) == [["divide"], ["sum", "product"], ["divide"]]
  # "divide" -> ("divide", "5", "6")
  # divide("5", "6") -- make a class?
  # &1 = sum("3", "4")
  # &2 = sum("4", "5")
  # multiply("&1", "&2")


  #Instances where it doesn't work
  # test_string = "Set x to the sum of 3 and 4 after multiplying it by the product of 5 and 6" #do a special case with after?
  # assert parser(test_string) == [["muliply", "product"], ["sum"]]

  # test_string =  "Find the sum of 3 by 4 and then take that answer and divide that by the sum of the first and second column"
  # assert parser(test_string) == [["sum"], ["sum"], ["divide"]] #limitation in parsing


# """Pytest test"""

# def test_answer():
#   assert inc(3) == 5
#   assert 4 == 5
#   where 4 = inc(3)





#pull out operators
# def update_operators(dict, text):
#   #replace every word with the words and if the dependencies dont change then add it to the dictionary
#   #go through dictionary and find something with the same department
#   to_replace = ""

#   doc_text = nlp(text)

#   for token in doc_text:
#     word_type = token.pos_
#     position = token.i

#     if str(token) not in dictionary and str(token) not in not_main and (word_type == "NOUN" or word_type == "VERB"):
#       for word in dictionary:
#         dict_text = nlp(text)

#         sentence1 = text
#         array = str(sentence1).split(" ")
#         array[position] = word
#         sentence2 = " ".join(array)

#         # print(sentence1)
#         # print(sentence2)
        
#         if (dict_text[0].dep_ == word_type and are_dependencies_same(sentence1, sentence2)):
#           print("word that could be important: " + str(token))



# def are_dependencies_same(text, text1):
#   doc_text = nlp(text)
#   doc_text2 = nlp(text1)

#   if (len(doc_text) != len(doc_text2)):
#     return False
  
#   for i in range(len(doc_text)):
#     word1 = doc_text[i]
#     word2 = doc_text2[i]
#     if word1.pos_ != word2.pos_ or word1.dep_ != word2.dep_ or word1.head.text != word2.head.text:
#       return False
  
#   return True
