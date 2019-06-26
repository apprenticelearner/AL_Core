from math import isclose
from pprint import pprint
from random import random
from random import shuffle
from itertools import product
from multiprocessing import cpu_count
from multiprocess import Pool

from concept_formation.utils import isNumber
from py_search.base import Problem
from py_search.base import Node
from py_search.uninformed import iterative_deepening_search
from planners.base_planner import BasePlanner
# from py_search.uninformed import depth_first_search
# from py_search.uninformed import breadth_first_search
# from py_search.informed import iterative_deepening_best_first_search
# from py_search.informed import best_first_search
# from py_search.utils import compare_searches

pool = None


def get_pool():
    global pool
    if pool is None:
        pool = Pool(cpu_count())
        # pool = Pool(1)
    return pool


def index_key(fact):
    """
    A new total indexing of the fact. Just build the whole damn thing, assuming
    it doesn't explode the memory usage.

    >>> index_key('cell')
    'cell'

    >>> index_key(('cell',))
    ('cell',)

    >>> index_key(('cell', '5'))
    ('cell', '5')

    >>> index_key((('value', '?x'), '5'))
    (('value', '?'), '5')

    >>> index_key((('X',('Position','Block1')), 10))
    (('X',('Position','Block1')), '#NUM')

    >>> index_key((('value', ('Add', ('value', '?x'),
    ...                              ('value', '?y'))), '5'))
    (('value', ('Add', ('value', '?'), ('value', '?'))), '5')
    """
    if isinstance(fact, tuple):
        return tuple(index_key(ele) for ele in fact)
    elif is_variable(fact):
        return '?'
    elif isNumber(fact):
        return '#NUM'
    else:
        return fact


def old_index_key(fact):
    """
    Generates the key values necessary to look up the given fact in the index.

    Takes a fact, represented as a tuple of elements and returns a tuple key.
    In general tuples have a particular structure: ``((fact), value)``, so when
    developing the index, it should use some combination of fact and values.

    Currently, I want the index to return (fact[0], fact[1], value); i.e., a
    triple element index.

    >>> old_index_key(('cell'))
    ('cell', None, None)

    >>> old_index_key(('cell', '5'))
    ('cell', None, '5')

    >>> old_index_key((('value', 'TableCell'), '5'))
    ('value', 'TableCell', '5')

    >>> old_index_key((('X',('Position','Block1')), 10))
    ('X', 'Position', '#NUM')

    >>> old_index_key((('value', ('Add', ('value', 'TableCell'),
    ...                              ('value', 'TableCell'))), '5'))
    ('value', 'Add', '5')
    """
    if not isinstance(fact, tuple):
        return extract_first_string(fact), None, None

    first = fact[0]
    second = fact[1]

    if not isinstance(first, tuple):
        return extract_first_string(first), None, extract_first_string(second)

    return (extract_first_string(first[0]), extract_first_string(first[1]),
            extract_first_string(second))


def get_variablized_keys(key):
    """
    Takes the triple key given above (fact[0], fact[1], value) and returns all
    the possible variablizations of it.

    >>> [k for k in get_variablized_keys(('value', 'cell', '5'))]
    [('value', 'cell', '5'), ('value', 'cell', '?'), \
('value', '?', '5'), ('value', '?', '?'), '?']

    >>> [k for k in get_variablized_keys(('value', '?', '5'))]
    [('value', '?', '5'), ('value', '?', '?'), '?']

    >>> [k for k in get_variablized_keys((('value',
    ...                                    ('Add', ('value', 'TableCell'),
    ...                                            ('value', 'TableCell'))),
    ...                                    '5'))]
    [(('value', ('Add', ('value', 'TableCell'), ('value', 'TableCell'))), \
'5'), (('value', ('Add', ('value', 'TableCell'), ('value', 'TableCell'))), \
'?'), (('value', ('Add', ('value', 'TableCell'), ('value', '?'))), '5'), \
(('value', ('Add', ('value', 'TableCell'), ('value', '?'))), '?'), (('value', \
('Add', ('value', 'TableCell'), '?')), '5'), (('value', ('Add', ('value', \
'TableCell'), '?')), '?'), (('value', ('Add', ('value', '?'), ('value', \
'TableCell'))), '5'), (('value', ('Add', ('value', '?'), ('value', \
'TableCell'))), '?'), (('value', ('Add', ('value', '?'), ('value', '?'))), \
'5'), (('value', ('Add', ('value', '?'), ('value', '?'))), '?'), (('value', \
('Add', ('value', '?'), '?')), '5'), (('value', ('Add', ('value', '?'), \
'?')), '?'), (('value', ('Add', '?', ('value', 'TableCell'))), '5'), ((\
'value', ('Add', '?', ('value', 'TableCell'))), '?'), (('value', ('Add', \
'?', ('value', '?'))), '5'), (('value', ('Add', '?', ('value', '?'))), '?'), \
(('value', ('Add', '?', '?')), '5'), (('value', ('Add', '?', '?')), '?'), \
(('value', '?'), '5'), (('value', '?'), '?'), ('?', '5'), ('?', '?'), '?']
    """
    yield key

    if isinstance(key, tuple):

        if isinstance(key[0], tuple):
            head = None
            body = key
        else:
            head = key[0]
            body = key[1:]

        possible_bodies = [list(get_variablized_keys(e)) for e in
                           body]
        for body in product(*possible_bodies):
            if head is None:
                new = tuple(body)
            else:
                new = (head,) + tuple(body)
            if new != key:
                yield new

    if not is_variable(key):
        yield '?'


def old_get_variablized_keys(key):
    """
    Takes the triple key given above (fact[0], fact[1], value) and returns all
    the possible variablizations of it.

    >>> [k for k in old_get_variablized_keys(('value', 'cell', '5'))]
    [('value', 'cell', '5'), ('?', 'cell', '5'), ('value', '?', '5'),\
 ('?', '?', '5'), ('value', 'cell', '?'), ('?', 'cell', '?'),\
 ('value', '?', '?'), ('?', '?', '?'), ('?', None, '5'), ('?', None, '?')]

    >>> [k for k in old_get_variablized_keys(('value', '?', '5'))]
    [('value', '?', '5'), ('?', '?', '5'), ('value', '?', '?'),\
 ('?', '?', '?'), ('?', None, '5'), ('?', None, '?')]

    """
    for k in get_variablized_keys_rec(key):
        yield k

    if key[0] is not None and key[0] != '?':
        for k in get_variablized_keys_rec(key[2:]):
            yield ('?', None) + k


def get_variablized_keys_rec(key):
    if len(key) > 0:
        for sub_key in get_variablized_keys_rec(key[1:]):
            yield key[:1] + sub_key

            if key[0] is not None and key[0] != '?':
                yield ('?',) + sub_key
    else:
        yield tuple()


def extract_first_string(s):
    """
    Extracts the first string from a tuple, it wraps it with parens to keep
    track of the depth of the constant within the relation.
    """
    if isinstance(s, tuple):
        return str(extract_first_string(s[0]))
    if is_variable(s):
        return '?'
    if isinstance(s, (int, float)):
        return '#NUM'
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


def unify(x, y, s, epsilon=0.0):
    """
    Unify expressions x and y. Return a mapping (a dict) that will make x
    and y equal or, if this is not possible, then it returns None.

    >>> unify((('Value', '?a'), '8'), (('Value', 'cell1'), '8'), {})
    {'?a': 'cell1'}
    """
    # TODO write tests
    # print(x, y, epsilon)

    if s is None:
        return None
    if (x == y or (isinstance(x, (int, float)) and isinstance(y, (int, float))
                   and isclose(x, y, abs_tol=epsilon))):
        return s
    elif is_variable(x):
        return unify_var(x, y, s)
    elif is_variable(y):
        return unify_var(y, x, s)
    elif (isinstance(x, tuple) and
          isinstance(y, tuple) and len(x) == len(y)):
        if not x:
            return s
        return unify(x[1:], y[1:], unify(x[0], y[0], s, epsilon), epsilon)
    else:
        return None


def pattern_match(pattern, index, substitution, epsilon=0.0):
    """
    Find substitutions that yield a match of the pattern against the provided
    index. If no match is found then it returns None.
    """
    # print("SIZE OF PATTERN BEING MATCHED", len(pattern))
    # pprint(pattern)
    ps = []
    remove_negs = []

    for p in pattern:
        if isinstance(p, tuple) and len(p) > 0 and p[0] == 'not':
            new_p = subst(substitution, p)
            args = [s for s in extract_strings(new_p[1]) if is_variable(s)]

            if len(args) == 0:
                # print("Checking Fully Bound Negation", new_p)

                key = index_key(new_p[1])

                if key in index:
                    for f in index[key]:
                        if new_p[1] == f:
                            # print("Early Negation FAILURE!!!!!")
                            return

                remove_negs.append(p)

        elif isinstance(p, tuple) and len(p) > 0 and callable(p[0]):
            pass
            # self.function_conditions.add(c)
        else:
            new_p = subst(substitution, p)
            key = index_key(new_p)

            count = 0
            if key in index:
                count = len(index[key])
            ps.append((count, random(), p))

    if len(ps) == 0:
        yield substitution

    else:
        ps.sort()
        ele = ps[0][2]
        new_ele = subst(substitution, ps[0][2])
        key = index_key(new_ele)

        if key in index:
            elements = [e for e in index[key]]
            shuffle(elements)

            for s in elements:
                new_sub = unify(ele, s, substitution, epsilon)
                if new_sub is not None:
                    for inner in pattern_match([p for p in pattern
                                                if p != ele and p not in
                                                remove_negs], index, new_sub,
                                               epsilon):
                        yield inner


def build_index(facts):
    """
    Given an iterator of facts returns a dict index.
    """
    index = {}
    for fact in facts:
        key = index_key(fact)
        # print('KEY', key)
        for k in get_variablized_keys(key):
            # print('VB KEY', k)
            if k not in index:
                index[k] = []
            index[k].append(fact)
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
    # elif is_variable(fact):
    #     raise Exception("Cannot execute functions on variables")

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
        operators, _, epsilon = node.extra

        facts = set(state)
        index = build_index(facts)

        for m in pattern_match(goals, index, {}, epsilon):
            return 0

        depth = 0
        new = set([1])
        count = 0
        rfacts = set()

        while len(new) > 0 and depth < max_depth:
            depth += 1
            new = set()
            # could optimize here to only iterate over operators that bind with
            # facts in prev.
            for o in operators:
                for m in o.match(index, epsilon):
                    count += 1
                    try:
                        effects = set([execute_functions(subst(m, f))
                                       for f in o.add_effects]) - facts
                    except:
                        continue

                    for e in effects:
                        for g in goals:
                            sub = unify(g, e, {}, epsilon)
                            if sub is not None:
                                rfacts.add(e)
                                for m in pattern_match(goals,
                                                       build_index(rfacts),
                                                       sub,
                                                       epsilon):
                                    return depth

                    new.update(effects)

            facts = new | facts
            index = build_index(facts)

        return max_depth

    # def node_value(self, node):
    #     return node.cost() + self.set_level_heuristic(node)

    def successors(self, node):
        state, goals = node.state
        operators, index, epsilon = node.extra

        for o in operators:
            for m in o.match(index, epsilon):
                try:
                    adds = frozenset([execute_functions(subst(m, f))
                                      for f in o.add_effects])
                    dels = frozenset([execute_functions(subst(m, f))
                                      for f in o.delete_effects])
                except:
                    continue

                new_state = state - dels
                dl = len(new_state)
                new_state = new_state | adds
                al = len(new_state)

                if dl == len(state) and al == dl:
                    # added nothing
                    continue

                new_index = build_index(new_state)
                action = subst(m, o.name)

                yield Node((new_state, goals), node, action,
                           node.cost() + 1,
                           (operators, new_index, epsilon))

    def goal_test(self, node):
        _, index, epsilon = node.extra
        state, goals = node.state
        for m in pattern_match(goals, index, {}, epsilon):
            return True
        return False

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer

def ground(arg):
    """
    Doc String
    """
    if isinstance(arg, tuple):
        return tuple(ground(e) for e in arg)
    elif isinstance(arg, str):
        return arg.replace('?', 'QM')
    else:
        return arg

def unground(arg):
    """
    Doc String
    """
    if isinstance(arg, tuple):
        return tuple(unground(e) for e in arg)
    elif isinstance(arg, str):
        return arg.replace('QM', '?')
    else:
        return arg

def get_vars(arg):
    """
    Doc String
    """
    if isinstance(arg, tuple):
        lis = []
        for elem in arg:
            for val in get_vars(elem):
                if val not in lis:
                    lis.append(val)
        return lis
        # return [v for e in arg for v in self.get_vars(e)]
    elif isinstance(arg, str) and len(arg) > 0 and arg[0] == '?':
        return [arg]
    else:
        return []

def replace_vars(arg, i=0):
    """
    Doc String
    """
    if isinstance(arg, tuple):
        ret = []
        for elem in arg:
            replaced, i = replace_vars(elem, i)
            ret.append(replaced)
        return tuple(ret), i
    elif isinstance(arg, str) and len(arg) > 0 and arg[0] == '?':
        return '?arg%s' % (str(i)), i+1
    else:
        return arg, i

def apply_operators(ele, operators, knowledge_base,epsilon):
    operator_output = None
    for operator in operators:
        effect = list(operator.effects)[0]
        pattern = effect[0][1]
        u_mapping = unify(pattern, ground(ele), {}) #Returns a mapping to name the values in the expression
        
        if(u_mapping):
            condition_sub = [subst(u_mapping,x) for x in operator.conditions]
            value_map = next(knowledge_base.fc_query(condition_sub, max_depth=0, epsilon=epsilon))
            
            try:
                operator_output = execute_functions(subst(value_map,effect[1]))
            except:
                continue

            return operator_output

from concept_formation.structure_mapper import rename_relation
class FoPlannerModule(BasePlanner):
    def __init__(self,search_depth,function_set,feature_set,epsilon=0.0):
        self.search_depth = search_depth
        self.function_set = function_set
        self.feature_set = feature_set
        self.epsilon = epsilon

    def how_search(self,state,sai,
                    operators=None,
                    foci_of_attention=None,
                    search_depth=None,
                    allow_bottomout=True,
                    allow_copy=True,
                    epsilon=0.0):

        if(operators == None and sai.action == "ButtonPressed"):
            yield -1,{}
            return

        _ = state.get_view("flat_ungrounded")

        if(not state.contains_view("func_knowledge_base")):
            key_vals_grounded = state.compute_from("key_vals_grounded","flat_ungrounded")
            knowledge_base = FoPlanner(key_vals_grounded,
                                        self.function_set)
            
            knowledge_base.fc_infer(depth=self.search_depth, epsilon=self.epsilon)

            state.set_view("func_knowledge_base",knowledge_base)
        else:
            knowledge_base = state.get_view("func_knowledge_base")            
        

        for attr,input_val in sai.inputs.items():
            #Danny: This populates a list of explanations found earlier in How search that work"
            possible = []
            at_least_one = False
            for iv_m in knowledge_base.fc_query([((attr, '?input'), input_val)],
                                                max_depth=0,
                                                epsilon=self.epsilon):

                input_rule = unground(iv_m['?input'])
                args = get_vars(input_rule)
                mapping = {"?arg%d"%i: arg for i,arg in enumerate(args)}
                input_rule = replace_vars(input_rule)[0]

                #Special case to handle copies
                if(isinstance(input_rule,str)):
                    input_rule = (attr, input_rule)

                if(operators != None):
                    if(not True in [input_rule == o for o in operators]):
                        continue

                if(len(get_vars(input_rule)) != len(args)):
                    continue

                at_least_one = True

                yield input_rule, mapping

            if(not at_least_one and allow_bottomout):
                yield input_val, {}




    def apply_featureset(self, state):
        _ = state.get_view("flat_ungrounded")

        key_vals_grounded = state.compute_from("key_vals_grounded","flat_ungrounded")
        knowledge_base = FoPlanner(key_vals_grounded,
                                    self.feature_set)
                                 
        knowledge_base.fc_infer(depth=1, epsilon=self.epsilon)

        state.set_view("feat_knowledge_base",knowledge_base)
        state.compute_from("flat_ungrounded","feat_knowledge_base")
        return state


    def eval_expression(self,x,mapping,state):
        rg_exp = []
        try:
            knowledge_base = state.get_view("func_knowledge_base")
        except:
            knowledge_base = state.get_view("feat_knowledge_base")

        x = rename_relation(x,mapping)
        for ele in x:
            if isinstance(ele, tuple):

                ok = False
                for var_match in knowledge_base.fc_query([(ground(ele), '?v')],
                                                         max_depth=0,
                                                          epsilon=self.epsilon):
                    if var_match['?v'] != '':
                        rg_exp.append(var_match['?v'])
                        ok = True
                    break

                if(not ok):
                    operator_output = apply_operators(ele,self.function_set,knowledge_base,self.epsilon)
                    if(operator_output != None and operator_output != ""):
                        rg_exp.append(operator_output)
                        ok = True

                if(not ok):
                    rg_exp.append(None)

            else:
                rg_exp.append(ele)
        return rg_exp

from planners.base_planner import PLANNERS
PLANNERS["foplanner"] = FoPlannerModule



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
        self.curr_depth = 0

        for fact in facts:
            self.add_fact(fact)

        for o in operators:
            self.add_operator(o)

    def gensym(self):
        self._gensym_counter += 1
        return '?gensym%i' % self._gensym_counter

    def add_fact(self, fact):
        self.facts.add(fact)
        key = index_key(fact)
        for k in get_variablized_keys(key):
            if k not in self.index:
                self.index[k] = []
            self.index[k].append(fact)

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

    def fc_get_actions(self, epsilon=0.0, must_match=None):
        for o in self.operators:
            for m in o.match(self.index, epsilon):
                if must_match is not None:
                    conditions = set([subst(m, c) for c in o.conditions])
                    if len(must_match.intersection(conditions)) == 0:
                        continue

                try:
                    effects = set([execute_functions(subst(m, f))
                                   for f in o.add_effects]) - self.facts
                    yield (o, m, effects)
                except:
                    continue

#     def fc_infer_op(self, operator, mapping):
#         try:
#             effects = set([execute_functions(subst(mapping, f))
#                            for f in operator.add_effects]) - self.facts
#         except:
#             effects = set()
#
#         for f in effects:
#             self.add_fact(f)
#
#         return effects

    def get_effects(self, op_eps):
        operator, epsilon = op_eps
        # print(operator.effects)
        ret = []
        for m in operator.match(self.index, epsilon):
            try:
                unprocessed = [f for f in operator.add_effects]
                effects = set()
                while len(unprocessed) > 0:
                    ele = unprocessed.pop()
                    f = execute_functions(subst(m, ele))
                    # print(ele)
                    if isinstance(f, list):
                        unprocessed.extend(f)
                    else:
                        effects.add(f)
                effects = effects - self.facts
                ret.append(effects)
            except Exception as e:
                continue
        return ret

    def fc_infer(self, depth=1, epsilon=0.0):
        for o in self.operators:
            if len(o.delete_effects) > 0:
                raise Exception("Cannot fc_infer with delete effects.")

        new = set([1])
        # count = 0

        while len(new) > 0 and self.curr_depth < depth:
            new = set()
            # could optimize here to only iterate over operators that bind with
            # facts in prev.

            pool = get_pool()
            all_effects = pool.map(self.get_effects,
                                   [(o, epsilon) for o in
                                       self.operators])

            # all_effects = [self.get_effects((o, epsilon)) for o in
            #                self.operators]

            for match_effects in all_effects:
                for effects in match_effects:
                    new.update(effects)

            # print(match_effects)

            # for o in self.operators:
            #     match_effects = self.get_effects((o, epsilon))

            #     for effects in match_effects:
            #         new.update(effects)

            for f in new:
                self.add_fact(f)

            self.curr_depth += 1

    def fc_query(self, goals, max_depth=5, epsilon=0.0):
        """
        Accepts a single goal and querys for it using breadth-first forward
        chaining.
        """
        for o in self.operators:
            if len(o.delete_effects) > 0:
                raise Exception("Cannot fc_query with delete effects.")

        for m in pattern_match(goals, self.index, {}, epsilon):
            yield m

        depth = 0
        new = set([1])
        count = 0
        # relevant_facts = set()

        while len(new) > 0 and depth < max_depth:
            new = set()
            # could optimize here to only iterate over operators that bind with
            # facts in prev.
            for o in self.operators:
                for m in o.match(self.index, epsilon):
                    count += 1
                    try:
                        effects = set([execute_functions(subst(m, f))
                                       for f in o.add_effects]) - self.facts
                    except:
                        continue

                    new.update(effects)

                    for e in effects:
                        for g in goals:
                            sub = unify(g, e, {}, epsilon)
                            if sub is None:
                                continue

                            # relevant_facts.add(e)

                            for m in pattern_match(goals,
                                                   build_index(self.facts |
                                                               new), sub,
                                                   epsilon):
                                yield m

            for f in new:
                self.add_fact(f)
            depth += 1

        return

    def fc_plan(self, goals, max_depth=5, epsilon=0.0):
        """
        A strategy of expanding forward from the initial state.
        """
        index = build_index(self.facts)

        fc_problem = FC_Problem(initial=(frozenset(self.facts),
                                         frozenset(goals)),
                                extra=(self.operators, index, epsilon))

        for solution in iterative_deepening_search(fc_problem,
                                                   max_depth_limit=max_depth +
                                                   1):
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
                    # print('NON HEAD CONDITION', c)

        for e in self.effects:
            if isinstance(e, tuple) and len(e) > 0 and e[0] == 'not':
                self.delete_effects.add(e[1])
            else:
                self.add_effects.add(e)

    def __str__(self):
        pprint(self.conditions)
        pprint(self.effects)
        return "n:%s\nc:%s\ne:%s" % (str(self.name), self.conditions,self.effects)

    def __repr__(self):
        return str(self)

    def match(self, index, epsilon=0.0, initial_mapping=None):
        """
        Given a state, return all of the bindings of the current pattern that
        produce a valid match.
        """
        if initial_mapping is None:
            initial_mapping = {}

        # print("HEAD CONDITIONS")
        # print(self.head_conditions)
        # print("NEG CONDITIONS")
        # print(self.negative_conditions)

        for head_match in pattern_match(self.head_conditions.union(
                                        set([('not', c) for c in
                                             self.negative_conditions])),
                                        index, initial_mapping,
                                        epsilon):
            # if initial_mapping is not None:
            # print("HEAD MAPPING", head_match)
            # print("INITIAL VS. HEAD", initial_mapping, head_match)
            # print("non_head_conditions", self.non_head_conditions)
            for full_match in pattern_match(self.non_head_conditions.union(
                                            set([('not', c) for c in
                                                 self.negative_conditions])),
                                            index,
                                            head_match, epsilon):

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
                    for nm in pattern_match([neg_ele], index, full_match,
                                            epsilon):
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


if __name__ == "__main__":

    import sys
    sys.path.insert(0, './')

    from planners.rulesets import arith_rules

    facts = [(('value', 'ul'), '6'),
             (('value', 'ur'), '8'),
             (('value', 'll'), '-3')]
    # kb = FoPlanner(facts, [unigramize, bigramize])
    kb = FoPlanner(facts, arith_rules)
    # kb.fc_infer()
    for m in kb.fc_query([(('value', '?x'), '9')], max_depth=1):
        print(m)
    # print(kb.facts)

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

    # facts = [ (('value', 'cell1'), '1'),
    #           (('value', 'cell2'), '2'),
    #           (('value', 'cell4'), '3'),
    #           (('value', 'cell3'), '5'),
    #           (('value', 'cell5'), '7'),
    #           (('value', 'cell6'), '11'),
    #           (('value', 'cell7'), '13'),
    #           (('value', 'cell8'), '17')]
    # kb = FoPlanner(facts, arith_rules)

    # import timeit

    # start_time = timeit.default_timer()

    # found = False
    # print("plan")
    # # for solution in kb.fc_plan([(('value', '?a'), '385')],
    # #                            # epsilon=0.101, max_depth=1
    # #                           ):
    # #     elapsed = timeit.default_timer() - start_time
    # #     print("Found solution, %0.4f" % elapsed)
    # #     print(solution.path())
    # #     for m in pattern_match(solution.state[1], solution.extra[1], {},
    # #                            epsilon=0.101):
    # #         found = True
    # #         print(m)
    # #         print()
    # #         break
    # #     if found:
    # #         break

    # print("query")
    # start_time = timeit.default_timer()
    # for m in kb.fc_query([(('value', '?a'), '385')],
    #                      # epsilon=0.101, max_depth=1
    #                     ):
    #     elapsed = timeit.default_timer() - start_time
    #     print("Found solution %0.4f" % elapsed)
    #     print(m)
    #     break

    # print(kb)

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
    #          (('haselement', 'obj-JCommTable4_Column1',
    #          'obj-JCommTable4_C1R1'), True),
    #          (('type', 'obj-JCommTable3_Column1'), 'MAIN::column'),
    #          (('haselement', 'obj-init', 'obj-JCommTable8'), True),
    #          (('haselement', 'obj-JCommTable8_Column1',
    #          'obj-JCommTable8_C1R1'), True),
    #          (('type', 'obj-JCommTable5'), 'MAIN::table'),
    #          (('haselement', 'obj-JCommTable4', 'obj-JCommTable4_Column1'),
    #          True),
    #          (('haselement', 'obj-JCommTable5_Column1',
    #          'obj-JCommTable5_C1R1'), True),
    #          (('name', 'obj-JCommLabel'), 'JCommLabel'),
    #          (('haselement', 'obj-JCommTable5_Column1',
    #          'obj-JCommTable5_C1R2'), True),
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
    #          (('haselement', 'obj-JCommTable8', 'obj-JCommTable8_Column1'),
    #          True),
    #          (('haselement', 'obj-init', 'obj-done'), True),
    #          (('type', 'obj-JCommTable6_Column1'), 'MAIN::column'),
    #          (('type', 'obj-JCommTable7_Column1'), 'MAIN::column'),
    #          (('haselement', 'obj-JCommTable', 'obj-JCommTable_Column1'),
    #          True),
    #          (('name', 'obj-JCommTable4_C1R1'), 'JCommTable4_C1R1'),
    #          (('haselement', 'obj-init', 'obj-JCommTable2'), True),
    #          (('name', 'obj-JCommTable5_C1R1'), 'JCommTable5_C1R1'),
    #          (('type', 'obj-JCommTable7'), 'MAIN::table'),
    #          (('type', 'obj-JCommLabel3'), 'MAIN::label'),
    #          (('name', 'obj-JCommTable2_Column1'), 'JCommTable2_Column1'),
    #          (('name', 'obj-JCommTable8_C1R1'), 'JCommTable8_C1R1'),
    #          (('type', 'obj-JCommTable8_C1R1'), 'MAIN::cell'),
    #          (('haselement', 'obj-JCommTable7_Column1',
    #          'obj-JCommTable7_C1R1'), True), (('name', 'obj-JCommLabel3'),
    #          'JCommLabel3'), (('name', 'obj-JCommTable7'), 'JCommTable7'),
    #          (('name', 'obj-JCommTable3'), 'JCommTable3'), (('name',
    #          'obj-JCommTable7_C1R1'), 'JCommTable7_C1R1'), (('haselement',
    #          'obj-JCommTable6_Column1', 'obj-JCommTable6_C1R2'), True),
    #          (('type', 'obj-done'), 'MAIN::button'),
    #          (('name', 'obj-JCommTable8'), 'JCommTable8'),
    #          (('type', 'obj-JCommLabel2'), 'MAIN::label'),
    #          (('haselement', 'obj-init', 'obj-JCommTable4'), True),
    #          (('name', 'obj-JCommTable6_Column1'), 'JCommTable6_Column1'),
    #          (('name', 'obj-JCommTable3_C1R1'), 'JCommTable3_C1R1'),
    #          (('type', 'obj-JCommTable8_Column1'), 'MAIN::column'), (('name',
    #          'obj-JCommTable_C1R1'), 'JCommTable_C1R1'), (('type', ('a',
    #          ('c', 'b')), 'obj-JCommTable4_C1R1'), 'MAIN::cell'),
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
    #          (('haselement', 'obj-JCommTable2', 'obj-JCommTable2_Column1'),
    #          True), (('name', 'obj-JCommTable5'), 'JCommTable5'),
    #          (('haselement', 'obj-JCommTable3_Column1',
    #          'obj-JCommTable3_C1R2'), True), (('type',
    #          'obj-JCommTable7_C1R1'), 'MAIN::cell'), (('name',
    #          'obj-JCommTable'), 'JCommTable'), (('name', 'obj-hint'),
    #          'hint'), (('haselement', 'obj-JCommTable_Column1',
    #          'obj-JCommTable_C1R2'), True), (('name', 'obj-JCommLabel4'),
    #          'JCommLabel4'), (('name', 'obj-JCommTable4_C1R2'),
    #          'JCommTable4_C1R2'), (('haselement', 'obj-init',
    #          'obj-JCommTable6'), True), (('haselement',
    #          'obj-JCommTable4_Column1', 'obj-JCommTable4_C1R2'), True),
    #          (('type', 'obj-init'), 'MAIN::problem'), (('type',
    #          'obj-JCommTable6_C1R2'), 'MAIN::cell'), (('type',
    #          'obj-JCommTable2'), 'MAIN::table'), (('haselement', 'obj-init',
    #          'obj-JCommTable'), True), (('type', 'obj-JCommTable_C1R1'),
    #          'MAIN::cell'), (('haselement', 'obj-JCommTable3',
    #          'obj-JCommTable3_Column1'), True), (('haselement',
    #          'obj-JCommTable2_Column1', 'obj-JCommTable2_C1R1'), True),
    #          (('haselement', 'obj-JCommTable6', 'obj-JCommTable6_Column1'),
    #          True), (('name', 'obj-JCommLabel2'), 'JCommLabel2'), (('type',
    #          'obj-JCommTable8'), 'MAIN::table'), (('haselement',
    #          'obj-JCommTable6_Column1', 'obj-JCommTable6_C1R1'), True),
    #          (('type', 'obj-JCommTable3'), 'MAIN::table'), (('haselement',
    #          'obj-init', 'obj-hint'), True), (('type',
    #          'obj-JCommTable4_Column1'), 'MAIN::column'), (('type',
    #          'obj-JCommTable3_C1R1'), 'MAIN::cell'), (('haselement',
    #          'obj-init', 'obj-JCommTable3'), True), (('name',
    #          'obj-JCommTable8_Column1'), 'JCommTable8_Column1'), (('type',
    #          'obj-JCommTable_Column1'), 'MAIN::column'), (('type',
    #          'obj-JCommTable5_Column1'), 'MAIN::column'), (('haselement',
    #          'obj-JCommTable_Column1', 'obj-JCommTable_C1R1'), True),
    #          (('haselement', 'obj-JCommTable3_Column1',
    #          'obj-JCommTable3_C1R1'), True), (('type',
    #          'obj-JCommTable5_C1R1'), 'MAIN::cell'),
    #          (('name', 'obj-JCommTable7_Column1'), 'JCommTable7_Column1'),
    #          (('type', 'obj-JCommTable2_Column1'), 'MAIN::column'),
    #          (('name', 'obj-JCommTable3_Column1'), 'JCommTable3_Column1'),
    #          (('type', 'obj-JCommTable2_C1R1'), 'MAIN::cell'),
    #          (('name', 'obj-JCommTable6'), 'JCommTable6'),
    #          (('name', 'obj-JCommTable2'), 'JCommTable2'),
    #          (('name', 'obj-JCommTable6_C1R2'), 'JCommTable6_C1R2'),
    #          (('haselement', 'obj-JCommTable5', 'obj-JCommTable5_Column1'),
    #          True), (('haselement', 'obj-JCommTable7',
    #          'obj-JCommTable7_Column1'), True), (('name',
    #          'obj-JCommTable6_C1R1'), 'JCommTable6_C1R1'), (('type',
    #          'obj-JCommTable4_C1R2'), 'MAIN::cell')]

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

    # operator = Operator(tuple(('Rule',)), h, [])
    # index = build_index(x)

    # pprint(x)

    # # pprint(index)

    # found = False
    # for m in operator.match(index, epsilon=0.001):
    #     print('match', m, operator.name)
    #     found = True

    # if not found:
    #     print("NOTHING FOUND")
