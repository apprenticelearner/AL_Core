import re

from planners.fo_planner import Operator
from planners.fo_planner import FoPlanner

_then_gensym_counter = 0


def gensym():
    global _then_gensym_counter
    _then_gensym_counter += 1
    return 'QMthengensym%i' % _then_gensym_counter


add_rule = Operator(('Add', '?x', '?y'),
                    [(('value', '?x'), '?xv'),
                     (('value', '?y'), '?yv'),
                     (lambda x, y: x <= y, '?x', '?y')],
                    [(('value', ('Add', ('value', '?x'), ('value', '?y'))),
                      (lambda x, y: str(int(x) + int(y)), '?xv', '?yv'))])

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
                      (lambda x, y: str(int(x) - int(y)), '?xv', '?yv'))])

mult_rule = Operator(('Multiply', '?x', '?y'),
                     [(('value', '?x'), '?xv'),
                      (('value', '?y'), '?yv'),
                      (lambda x, y: x <= y, '?x', '?y')],
                     [(('value', ('Multiply', ('value', '?x'),
                                  ('value', '?y'))),
                       (lambda x, y: str(int(x) * int(y)), '?xv', '?yv'))])

div_rule = Operator(('Divide', '?x', '?y'),
                    [(('value', '?x'), '?xv'),
                     (('value', '?y'), '?yv')],
                    [(('value', ('Divide', ('value', '?x'), ('value', '?y'))),
                      (lambda x, y: str(int(x) / int(y)), '?xv', '?yv'))])

equal_rule = Operator(('Equal', '?x', '?y'),
                      [(('value', '?x'), '?xv'), (('value', '?y'), '?yv'),
                       (lambda x, y: x < y, '?x', '?y'),
                       (lambda x: x != '', '?xv'),
                       (lambda x: x != '', '?yv'),
                       (lambda x, y: x == y, '?xv', '?yv')],
                      [(('eq', '?x', '?y'), True)])

editable_rule = Operator(('Editable', '?x'),
                         [(('value', '?x'), '?xv'),
                          (lambda x: x == "", '?xv')],
                         [(('editable', '?x'), True)])
                         # [(('editable', '?x'), (lambda x: x == "", '?xv'))])

def structurize_text(attr, val):
    ret = []
    words = val.split(' ')
    prev_word_obj = None
    for w in words:
        w = w.lower()
        if w == '':
            continue

        word_obj = gensym()
        print(word_obj)
        ret.append((('contains-word', attr, word_obj), True))
        ret.append((('word-value', word_obj, w), True))

        if prev_word_obj is not None:
            ret.append((('word-adj', attr, prev_word_obj, word_obj), True))

        prev_word_obj = word_obj

    return ret


def unigramize(attr, val):
    ret = []
    words = re.findall("[a-zA-Z0-9_']+|[^a-zA-Z0-9_\s]",
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


structurize = Operator(('Structurize', '?x'),
                       [(('value', '?x'), '?xv')],
                       [(structurize_text, '?x', '?xv')])

unigram_rule = Operator(('Unigram-rule', '?x'),
                        [(('value', '?x'), '?xv')],
                        [(unigramize, '?x', '?xv')])

bigram_rule = Operator(('Bigram-rule', '?x'),
                       [(('value', '?x'), '?xv'),
                        (lambda x: ' ' in x, '?xv')],
                       [(bigramize, '?x', '?xv')])

arith_rules = [add_rule, sub_rule, mult_rule, div_rule]
# arith_rules = [add_rule, sub_rule, mult_rule, div_rule, update_rule,
#                done_rule]
# arith_rules = [add_rule, mult_rule, update_rule, done_rule]


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


rb_rules = [add_x, add_y, sub_x, sub_y, half, rotate]

functionsets = {'fraction arithmetic prior knowledge': arith_rules,
                'rumbleblocks': rb_rules, 'article selection': []}

featuresets = {'fraction arithmetic prior knowledge': [equal_rule,
                                                       editable_rule],
               'rumbleblocks': [], 'article selection': [unigram_rule,
                                                         bigram_rule,
                                                         equal_rule,
                                                         editable_rule]}

if __name__ == "__main__":

    facts = [(('value', 'a'), 'This is Chris\'s first sentence.'),
             (('value', 'b'), 'This is the secondQM sentence.')]
    kb = FoPlanner(facts, [bigram_rule])
    kb.fc_infer()
    print(kb.facts)
