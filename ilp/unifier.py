
class PatternMatcher:

    def __init__(self, variables, pattern):
        self.variables = variables
        self.pattern = set(pattern)
        self.var_pattern = set()

        for v in self.variables:
            for p in pattern:
                if self.occur_check(v, p, {}):
                    self.var_pattern.add(p)

        self.not_var_pattern = self.pattern - self.var_pattern

    def is_variable(self, x):
        """
        Checks if the provided expression x is a variable, i.e., a string that
        starts with ?.

        >>> pm = PatternMatcher({}, [])
        >>> pm.is_variable('?x')
        True
        >>> pm.is_variable('x')
        False
        """
        return isinstance(x, str) and len(x) > 0 and x[0] == "?"

    def occur_check(self, var, x, s):
        """
        Check if x contains var, after using substition s. This prevents
        binding a variable to an expression that contains the variable in an
        infinite loop.

        >>> pm = PatternMatcher({}, [])
        >>> pm.occur_check('?x', '?x', {})
        True
        >>> pm.occur_check('?x', '?y', {'?y':'?x'})
        True
        >>> pm.occur_check('?x', '?y', {'?y':'?z'})
        False
        >>> pm.occur_check('?x', '?y', {'?y':'?z', '?z':'?x'})
        True
        >>> pm.occur_check('?x', ('relation', '?x'), {})
        True
        >>> pm.occur_check('?x', ('relation', ('relation2', '?x')), {})
        True
        >>> pm.occur_check('?x', ('relation', ('relation2', '?y')), {})
        False
        """
        if var == x:
            return True
        elif self.is_variable(x) and x in s:
            return self.occur_check(var, s[x], s)
        elif isinstance(x, (list, tuple)):
            for e in x:
                if self.occur_check(var, e, s):
                    return True
            return False
        else:
            return False

    def subst(self, s, x):
        """
        Substitute the substitution s into the expression x.

        >>> pm = PatternMatcher({}, [])
        >>> pm.subst({'?x': 42, '?y':0}, ('+', ('F', '?x'), '?y'))
        ('+', ('F', 42), 0)
        """
        if isinstance(x, tuple):
            return tuple(self.subst(s, xi) for xi in x)
        elif self.is_variable(x):
            return s.get(x, x)
        else:
            return x

    def extend(self, s, var, val):
        """
        Returns a new dict with var:val added.
        """
        s2 = {a: s[a] for a in s}
        s2[var] = val
        return s2

    def unify_var(self, var, x, s):
        """
        Unify var with x, using the mapping s.
        """
        if var in s:
            return self.unify(s[var], x, s)
        elif self.occur_check(var, x, s):
            return None
        else:
            return self.extend(s, var, x)

    def unify(self, x, y, s):
        """
        Unify expressions x and y. Return a mapping (a dict) that will make x
        and y equal or, if this is not possible, then it returns None.
        """
        if s is None:
            return None
        if x == y:
            return s
        elif self.is_variable(x):
            return self.unify_var(x, y, s)
        elif self.is_variable(y):
            return self.unify_var(y, x, s)
        elif (isinstance(x, tuple) and
              isinstance(y, tuple) and len(x) == len(y)):
            if not x:
                return s
            return self.unify(x[1:], y[1:], self.unify(x[0], y[0], s))
        else:
            return None

    def extract_first_string(self, s):
        """
        Extracts the first string from a tuple, it wraps it with parens to keep
        track of the depth of the constant within the relation.
        """
        if isinstance(s, tuple):
            return '(' + str(self.extract_first_string(s[0])) + ")"
        return s

    def match(self, state):
        """
        Given a state, return all of the bindings of the current pattern that
        produce a valid match.
        """
        # from pprint import pprint
        # pprint(state)

        index = {}
        for ele in state:
            pred = ele[0][0]
            first = self.extract_first_string(ele[0][1])

            if pred not in index:
                index[pred] = {}
                index[pred]['?'] = []
            if first not in index[pred]:
                index[pred][first] = []
            index[pred]['?'].append(ele)
            index[pred][first].append(ele)

        # returned = set()
        for m in self.pattern_match(self.var_pattern, index, {}):
            # print('OUTER', m)
            # print()
            for m2 in self.pattern_match(self.not_var_pattern, index, m):
                # print('INNER', m2)
                # print()
                result = {v: m[v] for v in m if v in self.variables}
                yield result
                # fz = frozenset(result)
                # if fz not in returned:
                #     returned.add(fz)
                #     yield result
                break

    def pattern_match(self, pattern, index, substitution):
        # print(len(pattern))
        # print(pattern, substitution)

        if len(pattern) > 0:
            ps = []
            for p in pattern:
                pred = p[0][0]
                first = self.extract_first_string(self.subst(substitution, p[0][1]))
                if self.is_variable(first):
                    first = '?'
                count = len(index[pred].get(first, []))
                ps.append((count, p))
            ps.sort()
            # print(ps[0], substitution)

            ele = ps[0][1]
            pred = ele[0][0]
            first = self.extract_first_string(self.subst(substitution, ele[0][1]))
            if self.is_variable(first):
                first = '?'

            # print(len(pattern), len(index[key]), pattern[0], substitution)
            if pred in index and first in index[pred]:
                for s in index[pred][first]:
                    new_sub = self.unify(ele, s, substitution)
                    if new_sub is not None:
                        for inner in self.pattern_match([p for p in pattern
                                                         if p != ele],
                                                        index, new_sub):
                            yield inner
        else:
            yield substitution


if __name__ == "__main__":

    rule = PatternMatcher({'?foa0', '?foa1'},
                          [(('haselement', '?o17', '?o18'), True),
                           (('haselement', '?o18', '?foa0'), True),
                           (('haselement', '?o18', '?foa1'), True),
                           (('haselement', '?o20', '?o17'), True),

                           (('haselement', '?o20', '?o19'), True),
                           (('haselement', '?o20', '?o21'), True),
                           (('haselement', '?o20', '?o22'), True),
                           (('haselement', '?o20', '?o23'), True),
                           (('haselement', '?o20', '?o24'), True),
                           (('haselement', '?o20', '?o25'), True),
                           (('haselement', '?o20', '?o26'), True),
                           (('haselement', '?o20', '?o27'), True),
                           (('haselement', '?o20', '?o28'), True),
                           (('haselement', '?o20', '?o29'), True),
                           (('haselement', '?o20', '?o30'), True),
                           (('haselement', '?o20', '?o31'), True),
                           (('haselement', '?o20', '?o32'), True), 

                           (('name', '?o20'), 'init'),
                           (('type', ('a', ('c', '?b')), '?foa0'), 'MAIN::cell'), 
                           (('type', '?foa1'), 'MAIN::cell'),
                           (('type', '?o17'), 'MAIN::table'), 
                           (('type', '?o18'), 'MAIN::column'),
                           (('type', '?o20'), 'MAIN::problem')])
    #print(uf.unify(('r1', '?x', 'b'), ('r1', 'a', '?y'), {}))
    state = [(('haselement', 'obj-init', 'obj-JCommTable7'), True),
             (('type', 'obj-hint'), 'MAIN::button'),
             (('type', 'obj-JCommTable'), 'MAIN::table'),
             (('type', 'obj-JCommLabel'), 'MAIN::label'),
             (('haselement', 'obj-JCommTable4_Column1', 'obj-JCommTable4_C1R1'), True),
             (('type', 'obj-JCommTable3_Column1'), 'MAIN::column'),
             (('haselement', 'obj-init', 'obj-JCommTable8'), True),
             (('haselement', 'obj-JCommTable8_Column1', 'obj-JCommTable8_C1R1'), True),
             (('type', 'obj-JCommTable5'), 'MAIN::table'),
             (('haselement', 'obj-JCommTable4', 'obj-JCommTable4_Column1'), True),
             (('haselement', 'obj-JCommTable5_Column1', 'obj-JCommTable5_C1R1'), True),
             (('name', 'obj-JCommLabel'), 'JCommLabel'),
             (('haselement', 'obj-JCommTable5_Column1', 'obj-JCommTable5_C1R2'), True),
             (('type', 'obj-JCommTable3_C1R2'), 'MAIN::cell'),
             (('name', 'obj-done'), 'done'),
             (('haselement', 'obj-init', 'obj-JCommLabel2'), True),
             (('name', 'obj-JCommTable4'), 'JCommTable4'),
             (('haselement', 'obj-init', 'obj-JCommLabel3'), True),
             (('type', 'obj-JCommTable6_C1R1'), 'MAIN::cell'),
             (('name', 'obj-JCommTable_C1R2'), 'JCommTable_C1R2'),
             (('type', 'obj-JCommTable6'), 'MAIN::table'),
             (('type', 'obj-JCommTable5_C1R2'), 'MAIN::cell'),
             (('name', 'obj-JCommTable2_C1R1'), 'JCommTable2_C1R1'),
             (('haselement', 'obj-init', 'obj-JCommTable5'), True),
             (('haselement', 'obj-JCommTable8', 'obj-JCommTable8_Column1'), True),
             (('haselement', 'obj-init', 'obj-done'), True),
             (('type', 'obj-JCommTable6_Column1'), 'MAIN::column'),
             (('type', 'obj-JCommTable7_Column1'), 'MAIN::column'),
             (('haselement', 'obj-JCommTable', 'obj-JCommTable_Column1'), True),
             (('name', 'obj-JCommTable4_C1R1'), 'JCommTable4_C1R1'),
             (('haselement', 'obj-init', 'obj-JCommTable2'), True),
             (('name', 'obj-JCommTable5_C1R1'), 'JCommTable5_C1R1'),
             (('type', 'obj-JCommTable7'), 'MAIN::table'),
             (('type', 'obj-JCommLabel3'), 'MAIN::label'),
             (('name', 'obj-JCommTable2_Column1'), 'JCommTable2_Column1'),
             (('name', 'obj-JCommTable8_C1R1'), 'JCommTable8_C1R1'),
             (('type', 'obj-JCommTable8_C1R1'), 'MAIN::cell'),
             (('haselement', 'obj-JCommTable7_Column1', 'obj-JCommTable7_C1R1'), True),
             (('name', 'obj-JCommLabel3'), 'JCommLabel3'),
             (('name', 'obj-JCommTable7'), 'JCommTable7'),
             (('name', 'obj-JCommTable3'), 'JCommTable3'),
             (('name', 'obj-JCommTable7_C1R1'), 'JCommTable7_C1R1'),
             (('haselement', 'obj-JCommTable6_Column1', 'obj-JCommTable6_C1R2'), True),
             (('type', 'obj-done'), 'MAIN::button'),
             (('name', 'obj-JCommTable8'), 'JCommTable8'),
             (('type', 'obj-JCommLabel2'), 'MAIN::label'),
             (('haselement', 'obj-init', 'obj-JCommTable4'), True),
             (('name', 'obj-JCommTable6_Column1'), 'JCommTable6_Column1'),
             (('name', 'obj-JCommTable3_C1R1'), 'JCommTable3_C1R1'),
             (('type', 'obj-JCommTable8_Column1'), 'MAIN::column'),
             (('name', 'obj-JCommTable_C1R1'), 'JCommTable_C1R1'),
             (('type', ('a', ('c', 'b')), 'obj-JCommTable4_C1R1'), 'MAIN::cell'),
             (('name', 'obj-JCommTable5_C1R2'), 'JCommTable5_C1R2'),
             (('type', 'obj-JCommTable_C1R2'), 'MAIN::cell'),
             (('name', 'obj-init'), 'init'),
             (('name', 'obj-JCommTable4_Column1'), 'JCommTable4_Column1'),
             (('haselement', 'obj-init', 'obj-JCommLabel'), True),
             (('haselement', 'obj-init', 'obj-JCommLabel4'), True),
             (('type', 'obj-JCommTable4'), 'MAIN::table'),
             (('type', 'obj-JCommLabel4'), 'MAIN::label'),
             (('name', 'obj-JCommTable_Column1'), 'JCommTable_Column1'),
             (('name', 'obj-JCommTable5_Column1'), 'JCommTable5_Column1'),
             (('name', 'obj-JCommTable3_C1R2'), 'JCommTable3_C1R2'),
             (('haselement', 'obj-JCommTable2', 'obj-JCommTable2_Column1'), True),
             (('name', 'obj-JCommTable5'), 'JCommTable5'),
             (('haselement', 'obj-JCommTable3_Column1', 'obj-JCommTable3_C1R2'), True),
             (('type', 'obj-JCommTable7_C1R1'), 'MAIN::cell'),
             (('name', 'obj-JCommTable'), 'JCommTable'),
             (('name', 'obj-hint'), 'hint'),
             (('haselement', 'obj-JCommTable_Column1', 'obj-JCommTable_C1R2'), True),
             (('name', 'obj-JCommLabel4'), 'JCommLabel4'),
             (('name', 'obj-JCommTable4_C1R2'), 'JCommTable4_C1R2'),
             (('haselement', 'obj-init', 'obj-JCommTable6'), True),
             (('haselement', 'obj-JCommTable4_Column1', 'obj-JCommTable4_C1R2'), True),
             (('type', 'obj-init'), 'MAIN::problem'),
             (('type', 'obj-JCommTable6_C1R2'), 'MAIN::cell'),
             (('type', 'obj-JCommTable2'), 'MAIN::table'),
             (('haselement', 'obj-init', 'obj-JCommTable'), True),
             (('type', 'obj-JCommTable_C1R1'), 'MAIN::cell'),
             (('haselement', 'obj-JCommTable3', 'obj-JCommTable3_Column1'), True),
             (('haselement', 'obj-JCommTable2_Column1', 'obj-JCommTable2_C1R1'), True),
             (('haselement', 'obj-JCommTable6', 'obj-JCommTable6_Column1'), True),
             (('name', 'obj-JCommLabel2'), 'JCommLabel2'),
             (('type', 'obj-JCommTable8'), 'MAIN::table'),
             (('haselement', 'obj-JCommTable6_Column1', 'obj-JCommTable6_C1R1'), True),
             (('type', 'obj-JCommTable3'), 'MAIN::table'),
             (('haselement', 'obj-init', 'obj-hint'), True),
             (('type', 'obj-JCommTable4_Column1'), 'MAIN::column'),
             (('type', 'obj-JCommTable3_C1R1'), 'MAIN::cell'),
             (('haselement', 'obj-init', 'obj-JCommTable3'), True),
             (('name', 'obj-JCommTable8_Column1'), 'JCommTable8_Column1'),
             (('type', 'obj-JCommTable_Column1'), 'MAIN::column'),
             (('type', 'obj-JCommTable5_Column1'), 'MAIN::column'),
             (('haselement', 'obj-JCommTable_Column1', 'obj-JCommTable_C1R1'), True),
             (('haselement', 'obj-JCommTable3_Column1', 'obj-JCommTable3_C1R1'), True),
             (('type', 'obj-JCommTable5_C1R1'), 'MAIN::cell'),
             (('name', 'obj-JCommTable7_Column1'), 'JCommTable7_Column1'),
             (('type', 'obj-JCommTable2_Column1'), 'MAIN::column'),
             (('name', 'obj-JCommTable3_Column1'), 'JCommTable3_Column1'),
             (('type', 'obj-JCommTable2_C1R1'), 'MAIN::cell'),
             (('name', 'obj-JCommTable6'), 'JCommTable6'),
             (('name', 'obj-JCommTable2'), 'JCommTable2'),
             (('name', 'obj-JCommTable6_C1R2'), 'JCommTable6_C1R2'),
             (('haselement', 'obj-JCommTable5', 'obj-JCommTable5_Column1'), True),
             (('haselement', 'obj-JCommTable7', 'obj-JCommTable7_Column1'), True),
             (('name', 'obj-JCommTable6_C1R1'), 'JCommTable6_C1R1'),
             (('type', 'obj-JCommTable4_C1R2'), 'MAIN::cell')]

    for m in rule.match(state):
        print(m)
