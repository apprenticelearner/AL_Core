import re

# from nltk import ViterbiParser

from planners.fo_planner import Operator
from planners.fo_planner import FoPlanner
# from learners.Grammar import grammar
# from learners.WhatLearner import ViterbiParser

_then_gensym_counter = 0


def custom_feature_set():
    """
    Return a custom set of feature operators.

    Use this function to expose any custom ruleset logic you want to add that
    doesn't rely on the Operator model system. This function will be called by
    all agents created by the system so you can use it for custom logic but we
    should probably turn it off if we ever have a production server.
    """
    return []


def custom_function_set():
    """
    Return a custom set of function operators.

    Use this function to expose any custom ruleset logic you want to add that
    doesn't rely on the Operator model system. This function will be called by
    all agents created by the system so you can use it for custom logic but we
    should probably turn it off if we ever have a production server.
    """
    # return [add_rule, sub_rule, mult_rule, div_rule,ones,tens]
    return []
    # return [add_then_tens, add_then_ones,add_then_tens3,add_then_ones3]


def gensym():
    global _then_gensym_counter
    _then_gensym_counter += 1
    return 'QMthengensym%i' % _then_gensym_counter


def is_str_and_not_number(s):
    if not isinstance(s, str):
        return False
    if s == "":
        return False

    try:
        float(s)
        return False
    except ValueError:
        return True



def is_str_number(s):
    if not isinstance(s, str):
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def int_float_add(x, y):
    z = float(x) + float(y)
    if z.is_integer():
        z = int(z)
    return str(z)


def int_float_subtract(x, y):
    z = float(x) - float(y)
    if z.is_integer():
        z = int(z)
    return str(z)


def int_float_multiply(x, y):
    z = float(x) * float(y)
    if z.is_integer():
        z = int(z)
    return str(z)


def int_float_divide(x, y):
    z = round(float(x) / float(y), 6)
    if z.is_integer():
        z = int(z)
    return str(z)


def sig_figs(x, y):
    x = float(x)
    y = float(y)
    if not y.is_integer():
        raise Exception("Cannot round to a fractional precision")
    else:
        y = int(y)
    output_s = '%.' + str(y) + 'g'
    out = float(output_s % x)
    if out.is_integer():
        out = int(out)
    out = str(out)
    curr_digits = len(out.replace(".", ""))
    if curr_digits < y:
        if "." not in out:
            out += "."
        out += "0" * (y - curr_digits)
    return out


def is_unit(unit):
    units = {'L', 'g', 'gal', 'kL', 'kg', 'lb', 'mL', 'mg'}
    return unit in units


def convert_units(val, from_unit, to_unit):
    print('converting %s from %s to %s.' % (val, from_unit, to_unit))
    val = float(val)
    conversions = {}
    conversions[('lb', 'kg')] = 2.2046
    conversions[('mg', 'g')] = 1000
    conversions[('mL', 'L')] = 1000
    conversions[('L', 'kL')] = 1000
    conversions[('g', 'kg')] = 1000
    conversions[('L', 'gal')] = 3.7854

    result = None
    if (from_unit, to_unit) in conversions:
        result = val * conversions[(from_unit, to_unit)]
    elif (to_unit, from_unit) in conversions:
        result = val * 1.0 / conversions[(to_unit, from_unit)]

    if result is None:
        raise Exception("Unknown units for conversion")

    if result.is_integer():
        return str(int(result))
    else:
        return str(result)


def tokenize_text(attr, val):
    ret = []
    words = re.findall(r"[0-9]+|[a-zA-Z]+|[^\w ]", val)
    prev_word_obj = None
    for w in words:
        w = w.lower()
        if w == '':
            continue

        word_obj = gensym()
        print(word_obj)
        ret.append((('contains-token', attr, word_obj), True))
        ret.append((('value', word_obj), w))

        if prev_word_obj is not None:
            ret.append((('token-adj', attr, prev_word_obj, word_obj), True))

        prev_word_obj = word_obj

    return ret


def tree_features(tree, path):
    ret = []
    node_str = ""
    for l in tree.leaves():
        node_str += l

    ret.append((('tree-label', path), tree.label()))
    ret.append((('value', path), node_str))

    # print(len(tree))
    if len(tree) < 2:
        return ret

    left_rt = tree_features(tree[0], ('left-tree', path))
    right_rt = tree_features(tree[1], ('right-tree', path))
    # left_rt = tree_features(tree[0], (tree[0].label(), path))
    # right_rt = tree_features(tree[1], (tree[1].label(), path))

    # print(ret + left_rt + right_rt)
    return ret + left_rt + right_rt


def grammar_features(attr, val):
    if not isinstance(val, str):
        raise Exception("Can only parse strings")

    parser = ViterbiParser(grammar)
    sent = [c for c in val.replace(" ", "").lower()]
    for tree in parser.parse(sent):
        return tree_features(tree, attr)

    raise Exception("Unable to parse val with grammar")


def unigramize(attr, val):
    ret = []
    words = re.findall("[a-zA-Z0-9_']+|[^a-zA-Z0-9_\s]+",
                       val.replace('QM', '?'))
    # words = val.split(' ')

    for w in words:
        if w == '':
            continue
        w = w.lower()
        w = w.replace('?', 'QM')

        ret.append((('unigram', attr, w), True))

    return ret


def bigramize(attr, val):
    ret = []
    words = re.findall("[a-zA-Z0-9_']+|[^a-zA-Z0-9_\s]",
                       val.replace('QM', '?'))
    # words = val.split(' ')
    prev_w = "<START>"

    for w in words:
        if w == '':
            continue
        w = w.lower()
        w = w.replace('?', 'QM')
        ret.append((('bigram', attr, prev_w, w), True))
        prev_w = w

    ret.append((('bigram', attr, prev_w, "<END>"), True))

    return ret


def subtract_strings(x, y):
    result = x.replace(y, "")

    assert result != ""
    assert result != x

    return result


def concatenate_with_space(x, y):
    return "%s %s" % (x, y)


def concatenate_without_space(x, y):
    return "%s%s" % (x, y)

def ones_digit(x):
  return x[-1]

def tens_digit(x):
  return x[-2]

def int2_float_add_then_ones(x, y):
  z = float(x) + float(y)
  z = z % 10
  if z.is_integer():
    z = int(z)
  return str(z)

def int2_float_add_then_tens(x, y):
  z = float(x) + float(y)
  z = z // 10
  if z.is_integer():
    z = int(z)
  return str(z)


def int3_float_add_then_ones(x, y, w):
  z = float(x) + float(y) + float(w)
  z = z % 10
  if z.is_integer():
    z = int(z)
  return str(z)

def int3_float_add_then_tens(x, y, w):
  z = float(x) + float(y) + float(w)
  z = z // 10
  if z.is_integer():
    z = int(z)
  return str(z)

add_rule = Operator(('Add', '?x', '?y'),
                    [(('value', '?x'), '?xv'),
                     (('value', '?y'), '?yv'),
                     # (lambda x, y: x <= y, '?x', '?y')
                     ],
                    [(('value', ('Add', ('value', '?x'), ('value', '?y'))),
                      (int_float_add, '?xv', '?yv'))])
Operator.register("add", add_rule)

add_then_ones = Operator(('Add_Then_Ones', '?x', '?y'),
                    [(('value', '?x'), '?xv'),
                     (('value', '?y'), '?yv'),
                     # (lambda x, y: x <= y, '?x', '?y')
                     ],
                    [(('value', ('Add_Then_Ones', ('value', '?x'), ('value', '?y'))),
                      (int2_float_add_then_ones, '?xv', '?yv'))])

add_then_tens = Operator(('Add_Then_Tens', '?x', '?y'),
                    [(('value', '?x'), '?xv'),
                     (('value', '?y'), '?yv'),
                     # (lambda x, y: x <= y, '?x', '?y')
                     ],
                    [(('value', ('Add_Then_Tens', ('value', '?x'), ('value', '?y'))),
                      (int2_float_add_then_tens, '?xv', '?yv'))])

add_then_ones3 = Operator(('Add_Then_Ones3', '?x', '?y', '?w'),
                    [(('value', '?x'), '?xv'),
                     (('value', '?y'), '?yv'),
                     (('value', '?w'), '?wv'),
                     # (lambda x, y: x <= y, '?x', '?y')
                     ],
                    [(('value', ('Add_Then_Ones3', ('value', '?x'), ('value', '?y'), ('value', '?w'))),
                      (int3_float_add_then_ones, '?xv', '?yv', '?wv'))])

add_then_tens3 = Operator(('Add_Then_Tens3', '?x', '?y', '?w'),
                    [(('value', '?x'), '?xv'),
                     (('value', '?y'), '?yv'),
                     (('value', '?w'), '?wv'),
                     # (lambda x, y: x <= y, '?x', '?y')
                     ],
                    [(('value', ('Add_Then_Tens3', ('value', '?x'), ('value', '?y'),('value', '?w'))),
                      (int3_float_add_then_tens, '?xv', '?yv', '?wv'))])


ones =    Operator(('Ones', '?x'),
                    [(('value', '?x'), '?xv'),
                    (lambda x: x != "" and float(x) > 1, '?xv'),
                     ],
                    [(('value', ('Ones', ('value', '?x'))),
                      (ones_digit, '?xv'))])

tens =    Operator(('Tens', '?x'),
                    [(('value', '?x'), '?xv'),
                    (lambda x: x != "" and float(x) > 10, '?xv'),
                     ],
                    [(('value', ('Tens', ('value', '?x'))),
                      (tens_digit, '?xv'))])


update_rule = Operator(('sai', '?sel', 'UpdateTable', '?val', '?ele'),
                       [(('value', '?ele'), '?val'),
                        (lambda x: x != "", '?val'),
                        (('name', '?ele2'), '?sel'),
                        (('type', '?ele2'), 'MAIN::cell'),
                        (('value', '?ele2'), '')],
                       [('sai', '?sel', 'UpdateTable', '?val', '?ele')])


done_rule = Operator(('sai', 'done', 'ButtonPressed', '-1'),
                     [],
                     [('sai', 'done', 'ButtonPressed', '-1', 'done-button')])


sub_rule = Operator(('Subtract', '?x', '?y'),
                    [(('value', '?x'), '?xv'),
                     (('value', '?y'), '?yv')],
                    [(('value', ('Subtract', ('value', '?x'),
                                 ('value', '?y'))),
                      (int_float_subtract, '?xv', '?yv'))])
Operator.register("subtract", sub_rule)

convert_units_rule = Operator(('ConvertUnits', '?x', '?y', '?z'),
                              [(('value', '?x'), '?val'),
                               (('value', '?y'), '?from-unit'),
                               (('value', '?z'), '?to-unit'),
                               (is_str_number, '?val'),
                               (is_str_and_not_number, '?from-unit'),
                               (is_str_and_not_number, '?to-unit'),
                               (is_unit, '?from-unit'),
                               (is_unit, '?to-unit'),
                               (lambda x, y: x != y, '?from-unit',
                                                     '?to-unit')
                               ],
                              [(('value', ('ConvertUnits', ('value', '?x'),
                               ('value', '?y'), ('value', '?z'))),
                               (convert_units, '?val', '?from-unit',
                                '?to-unit'))])


sig_fig_rule = Operator(('SigFig', '?x', '?y'),
                        [(('value', '?x'), '?xv'),
                         (('value', '?y'), '?yv'),
                         (('type', '?y'), 'MAIN::cell'),
                         (lambda x: float(x) < 10, '?yv')],
                        [(('value', ('SigFig', ('value', '?x'),
                         ('value', '?y'))), (sig_figs, '?xv', '?yv'))])


mult_rule = Operator(('Multiply', '?x', '?y'),
                     [(('value', '?x'), '?xv'),
                      (('value', '?y'), '?yv')],
                     [(('value', ('Multiply', ('value', '?x'),
                                  ('value', '?y'))),
                       (int_float_multiply, '?xv', '?yv'))])
Operator.register("multiply", mult_rule)

div_rule = Operator(('Divide', '?x', '?y'),
                    [(('value', '?x'), '?xv'),
                     (('value', '?y'), '?yv')],
                    [(('value', ('Divide', ('value', '?x'), ('value', '?y'))),
                      (int_float_divide, '?xv', '?yv'))])
Operator.register("divide", div_rule)

equal_rule = Operator(('Equal', '?x', '?y'),
                      [(('value', '?x'), '?xv'), (('value', '?y'), '?yv'),
                       (lambda x, y: x < y, '?x', '?y'),
                       (lambda x: x != '', '?xv'),
                       (lambda x: x != '', '?yv'),
                       (is_str_number, '?xv'),
                       (is_str_number, '?yv'),
                       # (lambda x, y: x == y, '?xv', '?yv')
                       ],
                      # [(('eq', ('value', '?x'), ('value', '?y')), True)])
                      [(('eq', ('value', '?x'), ('value', '?y')),
                        (lambda x, y: x == y, '?xv', '?yv'))])
Operator.register("equals", equal_rule)
Operator.register("equal", equal_rule)

is_number_rule = Operator(('IsNumber', '?x'),
                          [(('value', '?x'), '?xv')],
                          [(('IsNumber', '?x'), (is_str_number, '?xv'))])

editable_rule = Operator(('Editable', '?x'),
                         [(('value', '?x'), '?xv'),
                          (('type', '?x'), 'MAIN::cell'),
                         # (lambda x: x == "", '?xv')
                          ],
                         # [(('editable', '?x'), True)])
                         [(('editable', '?x'), (lambda x: x == "", '?xv'))])

half_val = Operator(('Half', '?x'),
                    [(('value', '?x'), '?xv'),
                     (lambda x: x % 2 == 0, '?xv')],
                    [(('value', ('Half', '?x')),
                      (lambda x: str(int(x) // 2), '?xv'))])


grammar_parser_rule = Operator(('GrammarParse', '?x'),
                               [(('value', '?x'), '?xv')],
                               [(grammar_features, '?x', '?xv')])


tokenize_rule = Operator(('Tokenize', '?x'),
                         [(('value', '?x'), '?xv')],
                         [(tokenize_text, '?x', '?xv')])

concatenate_rule = Operator(('concatenate-rule', '?x', '?y'),
                            [(('value', '?x'), '?xv'),
                             (('value', '?y'), '?yv')],
                            [(('value', ('concatenate-rule',
                                         '?x', '?y')),
                                (concatenate_without_space,
                                    '?xv', '?yv'))])

string_subtract_rule = Operator(('string-subtract-rule',
                                 '?x', '?y'),
                                [(('value', '?x'), '?xv'),
                                 (('value', '?y'), '?yv')],
                                [(('value', ('string-subtract-rule',
                                             '?x', '?y')),
                                 (subtract_strings, '?xv', '?yv'))])

unigram_rule = Operator(('Unigram-rule', '?x'),
                        [(('value', '?x'), '?xv')],
                        [(unigramize, '?x', '?xv')])

bigram_rule = Operator(('Bigram-rule', '?x'),
                       [(('value', '?x'), '?xv'),
                        (lambda x: ' ' in x, '?xv')],
                       [(bigramize, '?x', '?xv')])

half = Operator(('Half', '?x'),
                [(('y', '?x'), '?xv')],
                [(('y', ('Half', '?x')),
                  (lambda x: x / 2, '?xv'))])

add_y = Operator(('Add', '?y1', '?y2'),
                 [(('y', '?y1'), '?yv1'),
                  (('y', '?y2'), '?yv2')],
                 [(('y', ('Add', '?y1', '?y2')),
                  (lambda y1, y2: y1 + y2, '?yv1', '?yv2'))])

sub_y = Operator(('Subtract', '?y1', '?y2'),
                 [(('y', '?y1'), '?yv1'),
                  (('y', '?y2'), '?yv2')],
                 [(('y', ('Subtract', '?y1', '?y2')),
                  (lambda y1, y2: y1 - y2, '?yv1', '?yv2'))])

add_x = Operator(('Add', '?x1', '?x2'),
                 [(('x', '?x1'), '?xv1'),
                  (('x', '?x2'), '?xv2')],
                 [(('x', ('Add', '?x1', '?x2')),
                  (lambda x1, x2: x1 + x2, '?xv1', '?xv2'))])

sub_x = Operator(('Subtract', '?x1', '?x2'),
                 [(('x', '?x1'), '?xv1'),
                  (('x', '?x2'), '?xv2')],
                 [(('x', ('Subtract', '?x1', '?x2')),
                  (lambda x1, x2: x1 - x2, '?xv1', '?xv2'))])

rotate = Operator(('Rotate', '?b1'),
                  [(('x', ('bound', '?b1')), '?xv'),
                   (('y', ('bound', '?b1')), '?yv'),
                   (lambda x: not isinstance(x, tuple) or not x[0] == 'Rotate',
                    '?b1')],
                  [(('y', ('bound', ('Rotate', '?b1'))), '?yv'),
                   (('x', ('bound', ('Rotate', '?b1'))), '?xv')])


def int_float_square(x):
    z = float(x) * float(x)
    if z.is_integer():
        z = int(z)
    return str(z)

square_rule = Operator(('Square', '?x'),
                     [(('value', '?x'), '?xv')],
                     [(('value', ('Square', ('value', '?x') )),
                       (int_float_square, '?xv'))])

const_pi_rule = Operator(('Pi', '?x'),
                     [],
                     [(('value', ('Pi')),
                       ('pi', '?xv'))])

const_half_rule = Operator(('Half', '?x'),
                     [],
                     [(('value', ('Half')),
                       ('1/2', '?xv'))])





# ^^^^^^^^^^^^^^^^^ Define All Operators Above This Line ^^^^^^^^^^^^^^^^^^
for name,op in locals().copy().items():
  if(isinstance(op, Operator)):
    Operator.register(name,op)    


# Operator.register([add_rule, add_then_ones, add_then_tens,  add_then_ones3,
#     add_then_tens3, ones,tens, update_rule, done_rule,sub_rule,
#     convert_units_rule, sig_fig_rule, mult_rule,div_rule, equal_rule,
#     is_number_rule, half_val, grammar_parser_rule, tokenize_rule,
#     concatenate_rule, string_subtract_rule, unigram_rule, bigram_rule,
#     half, add_y, add_rule, add_then_ones, add_then_tens, add_then_ones3,
#     add_then_tens3, ones, tens, update_rule, done_rule, sub_rule, 
#     convert_units_rule, sig_fig_rule, mult_rule, div_rule, equal_rule,
#     is_number_rule, half_val, grammar_parser_rule, tokenize_rule,
#     concatenate_rule,string_subtract_rule,unigram_rule,bigram_rule, 
#     half, add_y,sub_y, add_x, sub_x, rotate])


rb_rules = [add_x, add_y, sub_x, sub_y, half, rotate]
arith_rules = [add_rule, sub_rule, mult_rule, div_rule, sig_fig_rule,
               concatenate_rule]
stoichiometry_rules = [sig_fig_rule, div_rule, mult_rule]

# arith_rules = [add_rule, sub_rule, mult_rule, div_rule, update_rule,
#                done_rule]
# arith_rules = [add_rule, mult_rule, update_rule, done_rule]

functionsets = {'tutor knowledge': arith_rules,
                'stoichiometry': stoichiometry_rules,
                'rumbleblocks': rb_rules, 'article selection': []}

featuresets = {'tutor knowledge': [equal_rule,
                                   grammar_parser_rule,
                                   editable_rule],
               'stoichiometry': [editable_rule, equal_rule],
               'rumbleblocks': [], 'article selection': [unigram_rule,
                                                         bigram_rule,
                                                         equal_rule,
                                                         editable_rule]}

if __name__ == "__main__":

    # facts = [(('value', 'a'), '3'),
    #          (('value', 'b'), '3x')]

    facts = [
              # (('x', ('bound', '?b1')), '?xv')
              (('x', ('pos','rect1')),2),
              (('y', ('pos','rect1')),1),
              (('x', ('bound','rect1')),3),
              (('y', ('bound','rect1')),1),

              (('x', ('pos','block1')),1),
              (('y', ('pos','block1')),2),
              (('x', ('bound','block1')),1),
              (('y', ('bound','block1')),1),

              (('x', ('bound','block2')),1),
              (('y', ('bound','block2')),1),

              (('x', ('pos','check1')),1),
              (('y', ('pos','check1')),1),

              (('x', ('pos','check2')),1),
              (('y', ('pos','check2')),3),

              (('x', ('pos','check3')),3),
              (('y', ('pos','check3')),1),

              (('x', ('pos','check4')),3),
              (('y', ('pos','check4')),3),

              ]


    # kb = FoPlanner(facts, [string_subtract_rule])
    # kb.fc_infer()
    # print(kb.facts)

    # facts = [(('value', 'x'), '17'),
    #          (('value', 'y'), '7')]
    kb = FoPlanner(facts, rb_rules)
    from pprint import pprint
    

    print('X')
    i = 0
    for sol in kb.fc_query([(('x', '?a'),3)], 2):
      if i % 5 == 0:
        pprint(sol)
      if i > 5 * 10:
        break
      i += 1


    print('Y')
    i = 0
    for sol in kb.fc_query([(('y', '?a'),2)], 2):
      if i % 5 == 0:
        pprint(sol)
      if i > 5 * 10:
        break
      i += 1


    
    # print(len(xs))
    # pprint(xs[:10])
    
    # ys = [sol for sol in kb.fc_query([(('y', '?a'),2)], 2)]
    # print('Y')
    # print(len(ys))
    # pprint(ys[:10])    

    # print('Y')
    # for sol in kb.fc_query([(('y', '?a'),2)], 2):
    #     pprint(sol)
    # kb.fc_infer()
    # pprint(kb.facts)

