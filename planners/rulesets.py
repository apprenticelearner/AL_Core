from planners.fo_planner import Operator


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
                       (lambda x, y: x == y, '?xv', '?yv')],
                      [('Equal', '?x', '?y')])

unigramize = Operator(('Unigramize', '?x'),
                      [(('value', '?x'), '?xv')],
                      [(lambda x, y: [('unigram', y, e) for e in
                                      x.split(' ')], '?xv', '?x')])

bigramize = Operator(('Bigramize', '?x'),
                     [(('value', '?x'), '?xv'),
                      (lambda x: ' ' in x, '?xv')],
                     [(lambda x, y: [('bigram', y, x.split(' ')[i],
                                      x.split(' ')[i+1]) for i in
                                     range(len(x.split(' '))-1)], '?xv', '?x')])

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
                  (('y', '?y2'), '?yv2'),
                  (lambda y1, y2: y1 <= y2, '?yv1', '?yv2')],
                 [(('y', ('Add', '?y1', '?y2')),
                  (lambda y1, y2: y1 + y2, '?yv1', '?yv2'))])

sub_y = Operator(('Subtract', '?y1', '?y2'),
                 [(('y', '?y1'), '?yv1'),
                  (('y', '?y2'), '?yv2')],
                 [(('y', ('Subtract', '?y1', '?y2')),
                  (lambda y1, y2: y1 - y2, '?yv1', '?yv2'))])

add_x = Operator(('Add', '?x1', '?x2'),
                 [(('x', '?x1'), '?xv1'),
                  (('x', '?x2'), '?xv2'),
                  (lambda x1, x2: x1 <= x2, '?xv1', '?xv2')],
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

rulesets = {'fraction arithmetic prior knowledge': arith_rules,
            'rumbleblocks': rb_rules,
            'article selection': []}
