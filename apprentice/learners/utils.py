"""
Standard functions used to support relational learning
"""
from random import uniform
from itertools import product
from multiprocessing import Pool
from multiprocessing import cpu_count

import numpy as np

from apprentice.planners.fo_planner import Operator
from apprentice.planners.fo_planner import build_index
# from planners.fo_planner import subst
from apprentice.planners.fo_planner import is_variable
from apprentice.planners.fo_planner import extract_strings

pool = None


def weighted_choice(choices):
    """
    A weighted version of choice.
    """
    total = sum(w for w, c in choices)
    r = uniform(0, total)
    upto = 0
    for w, c in choices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"


def get_variablizations(literal):
    """
    Takes a literal and returns all possible variablizations of it. Currently,
    this replaces constants only. Also, it replaces them with a variable that
    is generated based on the hash of the constant, so that similar constants
    map to the same variable.
    """
    if isinstance(literal, tuple):
        head = literal[0]
        possible_bodies = [[e] + list(get_variablizations(e)) for e in
                           literal[1:]]
        for body in product(*possible_bodies):
            new = (head,) + tuple(body)
            if new != literal:
                yield new

    elif not is_variable(literal):
        yield '?gen%s' % repr(literal)


def count_occurances(var, h):
    return len([s for x in h for s in extract_strings(x) if s == var])


def parallel_covers(x):
    h, constraints, x, xm = x
    return covers(h.union(constraints), x, xm)


def test_coverage(h, constraints, pset, nset):
    global pool
    if pool is None:
        pool = Pool(cpu_count())

    xset = [(h, constraints, p, pm) for p, pm in pset]
    pset_covers = pool.map(parallel_covers, xset)
    new_pset = [pset[i] for i, v in enumerate(pset_covers) if v is True]

    xset = [(h, constraints, n, nm) for n, nm in nset]
    nset_covers = pool.map(parallel_covers, xset)
    new_nset = [nset[i] for i, v in enumerate(nset_covers) if v is True]

    # print("TESTING MULTICORE!")
    # print(covers)
    # new_pset = [(p, pm) for p, pm in pset if
    #             covers(h.union(constraints), p, pm)]
    # new_nset = [(n, nm) for n, nm in nset if
    #             covers(h.union(constraints), n, nm)]
    return new_pset, new_nset


def covers(h, x, initial_mapping):
    """
    Returns true if h covers x
    """
    index = build_index(x)
    operator = Operator(tuple(['Rule']), h, [])
    for m in operator.match(index, initial_mapping=initial_mapping):
        return True
    return False


def rename(mapping, literal):
    """
    Given a mapping, renames the literal. Unlike subst, this works with
    constants as well as variables.
    """
    return tuple(mapping[ele] if ele in mapping else rename(mapping, ele) if
                 isinstance(ele, tuple) else ele for ele in literal)


def generate_literal(relation, arity, gensym):
    """
    Returns a new literal with novel variables.
    """
    return (relation,) + tuple(gensym() for i in range(arity))


def generalize_literal(literal, gensym):
    """
    This takes a literal and returns the most general version of it possible.
    i.e., a version that has all the values replaced with new veriables.
    """
    return (literal[0],) + tuple(ele if is_variable(ele) else
                                 # '?gen%s' % hash(ele)
                                 gensym()
                                 for ele in literal[1:])


def remove_vars(literal):
    """
    This removes all variables by putting XXX at the front of the string, so it
    cannot be unified anymore.
    """
    return tuple('XXX' + ele if is_variable(ele) else remove_vars(ele) if
                 isinstance(ele, tuple) else ele for ele in literal)


def clause_length(clause):
    """
    Counts the length of a clause. In particular, it counts number of
    relations, constants, and variable equality relations.
    """
    var_counts = {}
    count = 0

    for c in clause:
        count += count_elements(c, var_counts)

    for v in var_counts:
        count += var_counts[v] - 1

    return count


def count_elements(x, var_counts):
    """
    Counts the number of constants and keeps track of variable occurnaces.
    """
    if x is None:
        return 0

    c = 0
    if isinstance(x, tuple):
        c = sum([count_elements(ele, var_counts) for ele in x])
    elif is_variable(x):
        if x not in var_counts:
            var_counts[x] = 0
        var_counts[x] += 1
    else:
        c = 1
    return c


class OnlineDictVectorizer():
    def __init__(self, n_features):
        self.n_features = n_features
        self.separator = '='
        self.dtype = np.float32
        self.reset()
        self.counter = 0

    def reset(self):
        self.key = {}

    def fit(self, X):
        """
        Given a set of X, it updates the key with any new values.
        """

        for x in X:
            for f, v in x.items():
                if isinstance(v, str):
                    f = "%s%s%s" % (f, self.separator, v)
                if f not in self.key:
                    if len(self.key) < self.n_features:
                        self.key[f] = len(self.key)
                    else:
                        print("Exceeded available features")

        return self

    def transform(self, X):
        """
        Transforms the data using existing key mappings.
        """
        new_X = np.zeros((len(X), self.n_features), dtype=self.dtype)

        for i, x in enumerate(X):
            for f, v in x.items():
                if isinstance(v, str):
                    f = "%s%s%s" % (f, self.separator, v)
                    v = 1
                try:
                    new_X[i, self.key[f]] = self.dtype(v)
                except KeyError:
                    pass

        return new_X

    def fit_transform(self, X):
        """
        Similar to two calls of fit and transform, but does it all in
        one iteration rather than two through the data.
        """
        self.counter += 1
        new_X = np.zeros((len(X), self.n_features), dtype=self.dtype)

        for i, x in enumerate(X):
            for f, v in x.items():
                if isinstance(v, str):
                    f = "%s%s%s" % (f, self.separator, v)
                    v = 1

                if f not in self.key:
                    if len(self.key) < self.n_features:
                        self.key[f] = len(self.key)
                    else:
                        print("Exceeded available features")

                try:
                    new_X[i, self.key[f]] = self.dtype(v)
                except KeyError:
                    pass

        if self.counter % 200 == 0:
            print("Currently using {} features of {}".format(len(self.key),
                                                             self.n_features))
        return new_X
