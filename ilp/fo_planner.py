from pprint import pprint
from random import random

from py_search.base import Problem
from py_search.base import Node
from py_search.uninformed import iterative_deepening_search
from py_search.uninformed import depth_first_search
from py_search.uninformed import breadth_first_search
from py_search.informed import iterative_deepening_best_first_search
from py_search.informed import best_first_search
from py_search.utils import compare_searches


def index_key(fact):
    """
    Generates the key values necessary to look up the given fact in the index.
    """
    if not isinstance(fact, tuple):
        return fact, None
    if len(fact) == 1:
        return extract_first_string(fact[0]), None
    return extract_first_string(fact[0]), extract_first_string(fact[1])


def extract_first_string(s):
    """
    Extracts the first string from a tuple, it wraps it with parens to keep
    track of the depth of the constant within the relation.
    """
    if isinstance(s, tuple):
        return '(' + str(extract_first_string(s[0])) + ")"
    if is_variable(s):
        return '?'
    return s


def extract_strings(s):
    """
    Gets all of the string elements via iterator and depth first traversal.
    """
    if isinstance(s, (tuple, list)):
        for ele in s:
            for inner in extract_strings(ele):
                yield '%s' % inner
    else:
        yield s


def extend(s, var, val):
    """
    Returns a new dict with var:val added.
    """
    s2 = {a: s[a] for a in s}
    s2[var] = val
    return s2


def is_variable(x):
    """
    Checks if the provided expression x is a variable, i.e., a string that
    starts with ?.

    >>> is_variable('?x')
    True
    >>> is_variable('x')
    False
    """
    return isinstance(x, str) and len(x) > 0 and x[0] == "?"


def occur_check(var, x, s):
    """
    Check if x contains var, after using substition s. This prevents
    binding a variable to an expression that contains the variable in an
    infinite loop.

    >>> occur_check('?x', '?x', {})
    True
    >>> occur_check('?x', '?y', {'?y':'?x'})
    True
    >>> occur_check('?x', '?y', {'?y':'?z'})
    False
    >>> occur_check('?x', '?y', {'?y':'?z', '?z':'?x'})
    True
    >>> occur_check('?x', ('relation', '?x'), {})
    True
    >>> occur_check('?x', ('relation', ('relation2', '?x')), {})
    True
    >>> occur_check('?x', ('relation', ('relation2', '?y')), {})
    False
    """
    if var == x:
        return True
    elif is_variable(x) and x in s:
        return occur_check(var, s[x], s)
    elif isinstance(x, (list, tuple)):
        for e in x:
            if occur_check(var, e, s):
                return True
        return False
    else:
        return False


def subst(s, x):
    """
    Substitute the substitution s into the expression x.

    >>> subst({'?x': 42, '?y':0}, ('+', ('F', '?x'), '?y'))
    ('+', ('F', 42), 0)
    """
    if isinstance(x, tuple):
        return tuple(subst(s, xi) for xi in x)
    elif is_variable(x):
        return s.get(x, x)
    else:
        return x


def unify_var(var, x, s):
    """
    Unify var with x, using the mapping s.
    """
    # TODO write tests
    if var in s:
        return unify(s[var], x, s)
    elif occur_check(var, x, s):
        return None
    else:
        return extend(s, var, x)


def unify(x, y, s):
    """
    Unify expressions x and y. Return a mapping (a dict) that will make x
    and y equal or, if this is not possible, then it returns None.

    >>> unify((('Value', '?a'), '8'), (('Value', 'cell1'), '8'), {})
    {'?a': 'cell1'}
    """
    # TODO write tests
    if s is None:
        return None
    if x == y:
        return s
    elif is_variable(x):
        return unify_var(x, y, s)
    elif is_variable(y):
        return unify_var(y, x, s)
    elif (isinstance(x, tuple) and
          isinstance(y, tuple) and len(x) == len(y)):
        if not x:
            return s
        return unify(x[1:], y[1:], unify(x[0], y[0], s))
    else:
        return None


def pattern_match(pattern, index, substitution):
    """
    Find substitutions that yield a match of the pattern against the provided
    index. If no match is found then it returns None.
    """
    if len(pattern) > 0:
        ps = []
        for p in pattern:
            new_p = subst(substitution, p)
            first, second = index_key(new_p)
            # print(first, second)

            count = 0
            if first in index:
                count = len(index[first].get(second, []))
            ps.append((count, random(), p))

        ps.sort()
        ele = ps[0][2]
        new_ele = subst(substitution, ps[0][2])
        first, second = index_key(new_ele)

        if first in index and second in index[first]:
            for s in index[first][second]:
                new_sub = unify(ele, s, substitution)
                if new_sub is not None:
                    for inner in pattern_match([p for p in pattern
                                                if p != ele], index, new_sub):
                        yield inner
    else:
        yield substitution


def build_index(facts):
    """
    Given an iterator of facts returns a dict index.
    """
    index = {}
    for fact in facts:

        first, second = index_key(fact)

        if '?' not in index:
            index['?'] = {}
        if '?' not in index['?']:
            index['?']['?'] = []

        if first not in index:
            index[first] = {}
            index[first]['?'] = []

        if second not in index['?']:
            index['?'][second] = []

        if second not in index[first]:
            index[first][second] = []

        index[first][second].append(fact)
        index['?'][second].append(fact)
        if second is not None:
            index[first]['?'].append(fact)
            index['?']['?'].append(fact)

    return index


def execute_functions(fact):
    """
    Traverses a fact executing any functions present within. Returns a fact
    where functions are replaced with the function return value.
    """
    if isinstance(fact, tuple) and len(fact) > 1:
        if callable(fact[0]):
            return fact[0](*[execute_functions(ele) for ele in fact[1:]])
        else:
            return tuple(execute_functions(ele) for ele in fact)
    elif is_variable(fact):
        raise Exception("Cannot execute functions on variables")

    return fact


class FC_Problem(Problem):
    """
    A problem object that allows for the use of the py_search library to employ
    different search techniques for forward chaining.
    """

    def compute_action_cost(self, action):
        if isinstance(action, (tuple, list)):
            return 1 + sum([self.compute_action_cost(a) for a in action])
        return 0

    def set_level_heuristic(self, node, max_depth=4):
        """
        Use a breadth-first expansion without delete effects to compute the
        distance until all of the goals are reachable.
        """
        state, goals = node.state
        operators, _ = node.extra

        facts = set(state)
        index = build_index(facts)

        for m in pattern_match(goals, index, {}):
            return 0

        depth = 0
        new = set([1])
        count = 0
        relevant_facts = set()

        while len(new) > 0 and depth < max_depth:
            depth += 1
            new = set()
            # could optimize here to only iterate over operators that bind with
            # facts in prev.
            for o in operators:
                for m in o.match(index):
                    count += 1
                    try:
                        effects = set([execute_functions(subst(m, f))
                                       for f in o.add_effects]) - facts
                    except:
                        continue

                    for e in effects:
                        for g in goals:
                            sub = unify(g, e, {})
                            if sub is not None:
                                relevant_facts.add(e)
                                for m in pattern_match(goals,
                                        build_index(relevant_facts), sub):
                                    # print("Level: %i, # Operators: %i" %
                                    # (depth, count))
                                    return depth

                    new.update(effects)

            facts = new | facts
            index = build_index(facts)

        return max_depth

    # def node_value(self, node):
    #     return node.cost() + self.set_level_heuristic(node)

    def successors(self, node):
        state, goals = node.state
        operators, index = node.extra

        for o in operators:
            for m in o.match(index):
                try:
                    adds = frozenset([execute_functions(subst(m, f))
                                      for f in o.add_effects])
                    dels = frozenset([execute_functions(subst(m, f))
                                      for f in o.delete_effects])
                except:
                    continue

                new_state = state - dels
                new_state = new_state | adds

                if len(new_state) == len(state):
                    # added nothing
                    continue

                new_index = build_index(new_state)
                action = subst(m, o.name)

                yield Node((new_state, goals), node, action,
                           node.cost() + 1,
                           (operators, new_index))

    def goal_test(self, node):
        _, index = node.extra
        state, goals = node.state
        for m in pattern_match(goals, index, {}):
            return True
        return False


class FoPlanner:

    def __init__(self, facts, operators):
        """
        facts - the state
        operators - a list of operators
        """
        self.facts = set()
        self.operators = []
        self.index = {}
        self._gensym_counter = 0

        for fact in facts:
            self.add_fact(fact)

        for o in operators:
            self.add_operator(o)

    def gensym(self):
        self._gensym_counter += 1
        return '?gensym%i' % self._gensym_counter

    def add_fact(self, fact):

        self.facts.add(fact)
        first, second = index_key(fact)

        if '?' not in self.index:
            self.index['?'] = {}
        if '?' not in self.index['?']:
            self.index['?']['?'] = []

        if first not in self.index:
            self.index[first] = {}
            self.index[first]['?'] = []

        if second not in self.index['?']:
            self.index['?'][second] = []

        if second not in self.index[first]:
            self.index[first][second] = []

        self.index[first][second].append(fact)
        self.index['?'][second].append(fact)
        if second is not None:
            self.index[first]['?'].append(fact)
            self.index['?']['?'].append(fact)

    def add_operator(self, operator):
        """
        Adds an operator to the knowledge base. Should be upgraded to include
        additional preprocessing steps. For example, need to standardize apart
        variable names.
        """
        variables = set([s for c in operator.conditions
                         for s in extract_strings(c) if is_variable(s)])
        m = {v: self.gensym() for v in variables}
        new_name = tuple(subst(m, ele) for ele in operator.name)
        new_effects = [subst(m, ele) for ele in operator.effects]
        new_conditions = [subst(m, ele) for ele in operator.conditions]

        new_operator = Operator(new_name, new_conditions, new_effects)
        self.operators.append(new_operator)

    def __str__(self):
        result = "=====\nFACTS\n=====\n"
        for f in self.facts:
            result += str(f) + '\n'
        result += '\n'

        result += "=====\nOPERATORS\n=====\n"
        for o in self.operators:
            result += str(o) + '\n'

        return result

    def fact_exists(self, fact):
        """
        Returns true if a fact exists in the knowledge base. This might be
        augmented in the future to check renamings of variables too.
        """
        return fact in self.facts

    def fc_infer(self, depth=1):
        for o in self.operators:
            if len(o.delete_effects) > 0:
                raise Exception("Cannot fc_infer with delete effects.")

        new = set([1])
        count = 0

        while len(new) > 0 and depth > 0:
            new = set()
            # could optimize here to only iterate over operators that bind with
            # facts in prev.
            for o in self.operators:
                for m in o.match(self.index):
                    count += 1
                    try:
                        effects = set([execute_functions(subst(m, f))
                                       for f in o.add_effects]) - self.facts
                    except:
                        continue

                    new.update(effects)

            for f in new:
                self.add_fact(f)
            depth -= 1

    def fc_query(self, goals, max_depth=5):
        """
        Accepts a single goal and querys for it using breadth-first forward
        chaining.
        """
        for o in self.operators:
            if len(o.delete_effects) > 0:
                raise Exception("Cannot fc_query with delete effects.")

        for m in pattern_match(goals, self.index, {}):
            yield m

        depth = 0
        new = set([1])
        count = 0
        relevant_facts = set()

        while len(new) > 0 and depth < max_depth:
            new = set()
            # could optimize here to only iterate over operators that bind with
            # facts in prev.
            for o in self.operators:
                for m in o.match(self.index):
                    count += 1
                    try:
                        effects = set([execute_functions(subst(m, f))
                                       for f in o.add_effects]) - self.facts
                    except:
                        continue

                    for e in effects:
                        for g in goals:
                            sub = unify(g, e, {})
                            if sub is None:
                                break
                            relevant_facts.add(e)
                            # print(sub)
                            for m in pattern_match(goals,
                                                   build_index(relevant_facts),
                                                   sub):
                                yield m

                    new.update(effects)

            for f in new:
                self.add_fact(f)
            depth += 1

        return

    def fc_plan(self, goals, max_depth=5):
        """
        A strategy of expanding forward from the initial state.
        """
        index = build_index(self.facts)

        fc_problem = FC_Problem(initial=(frozenset(self.facts),
                                         frozenset(goals)),
                                extra=(self.operators, index))

        # compare_searches([fc_problem], [iterative_deepening_search,
        #                                 best_first_search])

        for solution in iterative_deepening_search(fc_problem,
                                                   max_depth_limit=max_depth):
            yield solution


class Operator:

    def __init__(self, name, conditions, effects):
        self.name = name
        self.conditions = set(conditions)
        self.effects = set(effects)
        self.variables = {s for s in extract_strings(name) if
                          is_variable(s)}

        self.head_conditions = set()
        self.non_head_conditions = set()
        self.function_conditions = set()
        self.negative_conditions = set()

        self.add_effects = set()
        self.delete_effects = set()

        for c in self.conditions:
            if isinstance(c, tuple) and len(c) > 0 and c[0] == 'not':
                self.negative_conditions.add(c[1])
            elif isinstance(c, tuple) and len(c) > 0 and callable(c[0]):
                self.function_conditions.add(c)
            else:
                found = False
                for v in self.variables:
                    if occur_check(v, c, {}):
                        self.head_conditions.add(c)
                        found = True
                        break
                if not found:
                    self.non_head_conditions.add(c)

        for e in self.effects:
            if isinstance(e, tuple) and len(e) > 0 and e[0] == 'not':
                self.delete_effects.add(e[1])
            else:
                self.add_effects.add(e)

    def __str__(self):
        pprint(self.conditions)
        pprint(self.effects)
        return "%s" % (str(self.name))

    def __repr__(self):
        return str(self)

    def match(self, index):
        """
        Given a state, return all of the bindings of the current pattern that
        produce a valid match.
        """
        for head_match in pattern_match(self.head_conditions, index, {}):

            for full_match in pattern_match(self.non_head_conditions, index,
                                            head_match):

                neg_fun_fail = False
                for foo in self.function_conditions:
                    try:
                        if execute_functions(tuple(subst(full_match, ele)
                                                   for ele in foo)) is False:
                            neg_fun_fail = True
                            break

                    except:
                        neg_fun_fail = True
                        break

                if neg_fun_fail is True:
                    break

                for neg_ele in self.negative_conditions:
                    for nm in pattern_match([neg_ele], index, full_match):
                        neg_fun_fail = True
                        break
                    if neg_fun_fail is True:
                        break

                if neg_fun_fail is True:
                    break

                # result = {v: head_match[v] for v in head_match
                #           if v in self.variables}
                yield full_match
                # yield result
                break


add_rule = Operator(('Add', '?x', '?y'),
                    [(('value', '?x'), '?xv'),
                     (('value', '?y'), '?yv'),
                     (lambda x, y: int(x) <= int(y), '?xv', '?yv')],
                    [(('value', ('Add', ('value', '?x'), ('value', '?y'))),
                      (lambda x, y: str(int(x) + int(y)), '?xv', '?yv'))])

sub_rule = Operator(('Subtract', '?x', '?y'),
                    [(('value', '?x'), '?xv'),
                     (('value', '?y'), '?yv')],
                    [(('value', ('Subtract', ('value', '?x'),
                                 ('value', '?y'))),
                      (lambda x, y: str(int(x) - int(y)), '?xv', '?yv'))])

mult_rule = Operator(('Multiply', '?x', '?y'),
                     [(('value', '?x'), '?xv'),
                      (('value', '?y'), '?yv'),
                      (lambda x, y: int(x) <= int(y), '?xv', '?yv')],
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

arith_rules = [equal_rule, add_rule, sub_rule, mult_rule, div_rule]


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
                    (lambda x: not isinstance(x, tuple) or not x[0] == 'Rotate', '?b1')],
                    
                    [(('y', ('bound', ('Rotate', '?b1'))), '?yv'),
                    (('x', ('bound', ('Rotate', '?b1'))), '?xv')])


rb_rules = [add_x,add_y,sub_x,sub_y,half,rotate]

if __name__ == "__main__":

    # criminal_rule = Operator(('Criminal', '?x'), [('Criminal', '?x')], [],
    #                         [('American', '?x'),
    #                          ('Weapon', '?y'),
    #                          ('Sells', '?x', '?y', '?z'),
    #                          ('Hostile', '?z')])

    # sells_west_rule = Operator(('Sells', 'West', '?x', 'Nono'),
    #                           [('Sells', 'West', '?x', 'Nono')], [],
    #                           [('Missle', '?x'),
    #                            ('Owns', 'Nono', '?x')])

    # weapons_rule = Operator(('Weapon', '?x'),
    #                        [('Weapon', '?x')], [],
    #                        [('Missle', '?x')])

    # hostile_rule = Operator(('Hostile', '?x'),
    #                        [('Hostile', '?x')], [],
    #                        [('Enemy', '?x', 'America')])

    # facts = [('Owns', 'Nono', 'M1'),
    #          ('Missle', 'M1'),
    #          ('American', 'West'),
    #          ('Enemy', 'Nono', 'America')]

    # kb = FolKb(facts, [criminal_rule, sells_west_rule, weapons_rule,
    #                    hostile_rule])

    # print(kb)

    # for solution in kb.forward_chain([('Criminal', '?x')]):
    #     print("Found solution", solution)
    #     break

    # rule1 = Operator('A', ['A'], [], ['B', ('not', 'C')])
    # kb2 = FolKb(['B', 'C'], [rule1])
    # # kb2 = FolKb(['B'], [rule1])

    # print(kb2)

    # for solution in kb2.forward_chain(['A']):
    #     print("Found solution", solution)
    #     break

    facts = [(('value', 'cell1'), '1'),
             (('value', 'cell2'), '2'),
             (('value', 'cell3'), '3'),
             (('value', 'cell4'), '5'),
             (('value', 'cell5'), '7'),
             (('value', 'cell6'), '11'),
             (('value', 'cell7'), '13'),
             (('value', 'cell8'), '17')]
    kb = FoPlanner(facts, arith_rules)

    import timeit

    start_time = timeit.default_timer()

    found = False
    for solution in kb.fc_plan([('?a', '105')]):
        for m in pattern_match(solution.state[1], solution.extra[1], {}):
            elapsed = timeit.default_timer() - start_time
            found = True
            print("Found solution, %0.4f" % elapsed)
            print(m)
            print(solution.path())
            print()
            break
        if found:
            break

    start_time = timeit.default_timer()
    for m in kb.fc_query([('?a', '105')]):
        elapsed = timeit.default_timer() - start_time
        print("Found solution %0.4f" % elapsed)
        print(m)
        break

    # op = Operator(('Something', '?foa0', '?foa1'),
    #                   [(('haselement', '?o17', '?o18'), True),
    #                    (('haselement', '?o18', '?foa0'), True),
    #                    (('haselement', '?o18', '?foa1'), True),
    #                    (('haselement', '?o20', '?o17'), True),

    #                    (('haselement', '?o20', '?o19'), True),
    #                    (('haselement', '?o20', '?o21'), True),
    #                    (('haselement', '?o20', '?o22'), True),
    #                    (('haselement', '?o20', '?o23'), True),
    #                    (('haselement', '?o20', '?o24'), True),
    #                    (('haselement', '?o20', '?o25'), True),
    #                    (('haselement', '?o20', '?o26'), True),
    #                    (('haselement', '?o20', '?o27'), True),
    #                    (('haselement', '?o20', '?o28'), True),
    #                    (('haselement', '?o20', '?o29'), True),
    #                    (('haselement', '?o20', '?o30'), True),
    #                    (('haselement', '?o20', '?o31'), True),
    #                    (('haselement', '?o20', '?o32'), True),

    #                    (('name', '?o20'), 'init'),
    #                    (('type', ('a', ('c', '?b')), '?foa0'), 'MAIN::cell'),
    #                    (('type', '?foa1'), 'MAIN::cell'),
    #                    (('type', '?o17'), 'MAIN::table'),
    #                    (('type', '?o18'), 'MAIN::column'),
    #                    (('type', '?o20'), 'MAIN::problem')],['match'])
    # #print(uf.unify(('r1', '?x', 'b'), ('r1', 'a', '?y'), {}))
    # state = [(('haselement', 'obj-init', 'obj-JCommTable7'), True),
    #          (('type', 'obj-hint'), 'MAIN::button'),
    #          (('type', 'obj-JCommTable'), 'MAIN::table'),
    #          (('type', 'obj-JCommLabel'), 'MAIN::label'),
    #          (('haselement', 'obj-JCommTable4_Column1', 'obj-JCommTable4_C1R1'), True),
    #          (('type', 'obj-JCommTable3_Column1'), 'MAIN::column'),
    #          (('haselement', 'obj-init', 'obj-JCommTable8'), True),
    #          (('haselement', 'obj-JCommTable8_Column1', 'obj-JCommTable8_C1R1'), True),
    #          (('type', 'obj-JCommTable5'), 'MAIN::table'),
    #          (('haselement', 'obj-JCommTable4', 'obj-JCommTable4_Column1'), True),
    #          (('haselement', 'obj-JCommTable5_Column1', 'obj-JCommTable5_C1R1'), True),
    #          (('name', 'obj-JCommLabel'), 'JCommLabel'),
    #          (('haselement', 'obj-JCommTable5_Column1', 'obj-JCommTable5_C1R2'), True),
    #          (('type', 'obj-JCommTable3_C1R2'), 'MAIN::cell'),
    #          (('name', 'obj-done'), 'done'),
    #          (('haselement', 'obj-init', 'obj-JCommLabel2'), True),
    #          (('name', 'obj-JCommTable4'), 'JCommTable4'),
    #          (('haselement', 'obj-init', 'obj-JCommLabel3'), True),
    #          (('type', 'obj-JCommTable6_C1R1'), 'MAIN::cell'),
    #          (('name', 'obj-JCommTable_C1R2'), 'JCommTable_C1R2'),
    #          (('type', 'obj-JCommTable6'), 'MAIN::table'),
    #          (('type', 'obj-JCommTable5_C1R2'), 'MAIN::cell'),
    #          (('name', 'obj-JCommTable2_C1R1'), 'JCommTable2_C1R1'),
    #          (('haselement', 'obj-init', 'obj-JCommTable5'), True),
    #          (('haselement', 'obj-JCommTable8', 'obj-JCommTable8_Column1'), True),
    #          (('haselement', 'obj-init', 'obj-done'), True),
    #          (('type', 'obj-JCommTable6_Column1'), 'MAIN::column'),
    #          (('type', 'obj-JCommTable7_Column1'), 'MAIN::column'),
    #          (('haselement', 'obj-JCommTable', 'obj-JCommTable_Column1'), True),
    #          (('name', 'obj-JCommTable4_C1R1'), 'JCommTable4_C1R1'),
    #          (('haselement', 'obj-init', 'obj-JCommTable2'), True),
    #          (('name', 'obj-JCommTable5_C1R1'), 'JCommTable5_C1R1'),
    #          (('type', 'obj-JCommTable7'), 'MAIN::table'),
    #          (('type', 'obj-JCommLabel3'), 'MAIN::label'),
    #          (('name', 'obj-JCommTable2_Column1'), 'JCommTable2_Column1'),
    #          (('name', 'obj-JCommTable8_C1R1'), 'JCommTable8_C1R1'),
    #          (('type', 'obj-JCommTable8_C1R1'), 'MAIN::cell'),
    #          (('haselement', 'obj-JCommTable7_Column1', 'obj-JCommTable7_C1R1'), True),
    #          (('name', 'obj-JCommLabel3'), 'JCommLabel3'),
    #          (('name', 'obj-JCommTable7'), 'JCommTable7'),
    #          (('name', 'obj-JCommTable3'), 'JCommTable3'),
    #          (('name', 'obj-JCommTable7_C1R1'), 'JCommTable7_C1R1'),
    #          (('haselement', 'obj-JCommTable6_Column1', 'obj-JCommTable6_C1R2'), True),
    #          (('type', 'obj-done'), 'MAIN::button'),
    #          (('name', 'obj-JCommTable8'), 'JCommTable8'),
    #          (('type', 'obj-JCommLabel2'), 'MAIN::label'),
    #          (('haselement', 'obj-init', 'obj-JCommTable4'), True),
    #          (('name', 'obj-JCommTable6_Column1'), 'JCommTable6_Column1'),
    #          (('name', 'obj-JCommTable3_C1R1'), 'JCommTable3_C1R1'),
    #          (('type', 'obj-JCommTable8_Column1'), 'MAIN::column'),
    #          (('name', 'obj-JCommTable_C1R1'), 'JCommTable_C1R1'),
    #          (('type', ('a', ('c', 'b')), 'obj-JCommTable4_C1R1'), 'MAIN::cell'),
    #          (('name', 'obj-JCommTable5_C1R2'), 'JCommTable5_C1R2'),
    #          (('type', 'obj-JCommTable_C1R2'), 'MAIN::cell'),
    #          (('name', 'obj-init'), 'init'),
    #          (('name', 'obj-JCommTable4_Column1'), 'JCommTable4_Column1'),
    #          (('haselement', 'obj-init', 'obj-JCommLabel'), True),
    #          (('haselement', 'obj-init', 'obj-JCommLabel4'), True),
    #          (('type', 'obj-JCommTable4'), 'MAIN::table'),
    #          (('type', 'obj-JCommLabel4'), 'MAIN::label'),
    #          (('name', 'obj-JCommTable_Column1'), 'JCommTable_Column1'),
    #          (('name', 'obj-JCommTable5_Column1'), 'JCommTable5_Column1'),
    #          (('name', 'obj-JCommTable3_C1R2'), 'JCommTable3_C1R2'),
    #          (('haselement', 'obj-JCommTable2', 'obj-JCommTable2_Column1'), True),
    #          (('name', 'obj-JCommTable5'), 'JCommTable5'),
    #          (('haselement', 'obj-JCommTable3_Column1', 'obj-JCommTable3_C1R2'), True),
    #          (('type', 'obj-JCommTable7_C1R1'), 'MAIN::cell'),
    #          (('name', 'obj-JCommTable'), 'JCommTable'),
    #          (('name', 'obj-hint'), 'hint'),
    #          (('haselement', 'obj-JCommTable_Column1', 'obj-JCommTable_C1R2'), True),
    #          (('name', 'obj-JCommLabel4'), 'JCommLabel4'),
    #          (('name', 'obj-JCommTable4_C1R2'), 'JCommTable4_C1R2'),
    #          (('haselement', 'obj-init', 'obj-JCommTable6'), True),
    #          (('haselement', 'obj-JCommTable4_Column1', 'obj-JCommTable4_C1R2'), True),
    #          (('type', 'obj-init'), 'MAIN::problem'),
    #          (('type', 'obj-JCommTable6_C1R2'), 'MAIN::cell'),
    #          (('type', 'obj-JCommTable2'), 'MAIN::table'),
    #          (('haselement', 'obj-init', 'obj-JCommTable'), True),
    #          (('type', 'obj-JCommTable_C1R1'), 'MAIN::cell'),
    #          (('haselement', 'obj-JCommTable3', 'obj-JCommTable3_Column1'), True),
    #          (('haselement', 'obj-JCommTable2_Column1', 'obj-JCommTable2_C1R1'), True),
    #          (('haselement', 'obj-JCommTable6', 'obj-JCommTable6_Column1'), True),
    #          (('name', 'obj-JCommLabel2'), 'JCommLabel2'),
    #          (('type', 'obj-JCommTable8'), 'MAIN::table'),
    #          (('haselement', 'obj-JCommTable6_Column1', 'obj-JCommTable6_C1R1'), True),
    #          (('type', 'obj-JCommTable3'), 'MAIN::table'),
    #          (('haselement', 'obj-init', 'obj-hint'), True),
    #          (('type', 'obj-JCommTable4_Column1'), 'MAIN::column'),
    #          (('type', 'obj-JCommTable3_C1R1'), 'MAIN::cell'),
    #          (('haselement', 'obj-init', 'obj-JCommTable3'), True),
    #          (('name', 'obj-JCommTable8_Column1'), 'JCommTable8_Column1'),
    #          (('type', 'obj-JCommTable_Column1'), 'MAIN::column'),
    #          (('type', 'obj-JCommTable5_Column1'), 'MAIN::column'),
    #          (('haselement', 'obj-JCommTable_Column1', 'obj-JCommTable_C1R1'), True),
    #          (('haselement', 'obj-JCommTable3_Column1', 'obj-JCommTable3_C1R1'), True),
    #          (('type', 'obj-JCommTable5_C1R1'), 'MAIN::cell'),
    #          (('name', 'obj-JCommTable7_Column1'), 'JCommTable7_Column1'),
    #          (('type', 'obj-JCommTable2_Column1'), 'MAIN::column'),
    #          (('name', 'obj-JCommTable3_Column1'), 'JCommTable3_Column1'),
    #          (('type', 'obj-JCommTable2_C1R1'), 'MAIN::cell'),
    #          (('name', 'obj-JCommTable6'), 'JCommTable6'),
    #          (('name', 'obj-JCommTable2'), 'JCommTable2'),
    #          (('name', 'obj-JCommTable6_C1R2'), 'JCommTable6_C1R2'),
    #          (('haselement', 'obj-JCommTable5', 'obj-JCommTable5_Column1'), True),
    #          (('haselement', 'obj-JCommTable7', 'obj-JCommTable7_Column1'), True),
    #          (('name', 'obj-JCommTable6_C1R1'), 'JCommTable6_C1R1'),
    #          (('type', 'obj-JCommTable4_C1R2'), 'MAIN::cell')]

    # kb = FoPlanner(state, [op])

    # for solution in kb.forward_chain(['match']):
    #     for m in pattern_match(solution.state[1], solution.extra[1], {}):
    #         found = True
    #         print("Found solution")
    #         print(m)
    #         print(solution.path())
    #         break
    #     if found:
    #         break
