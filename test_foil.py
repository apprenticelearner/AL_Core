from ilp.foil import Foil

x = Foil(closed_world=False)

e1 = {}
e1[('value', '?c1')] = '7'
e1[('value', '?c2')] = '7'

e2 = {}
e2[('value', '?c1')] = '7'
e2[('value', '?c2')] = '8'

e3 = {}
e3[('value', '?c1')] = '8'
e3[('value', '?c2')] = '8'

e4 = {}
e4[('value', '?c1')] = '7'
e4[('value', '?c2')] = ''
e4[('value', '?c3')] = '8'
e4[('value', '?c4-Test')] = '8'

e5 = {}
e5[('value', '?c1-Test')] = '7'

x.fit([('?c1', '?c2'), ('?c1', '?c2'), ('?c1', '?c2'), ('?c3', '?c4-Test'), 
       ('?c1', '?c2'), ('?c1', '?c2'), ('?c1', '?c2'), ('?c1-Test', '?c1-Test')], 
      [e1, e2, e3, e4, e1, e1, e1, e5], 
      [1, 0, 1, 1, 1, 1, 1, 0])

print(x.get_matches(e4))
