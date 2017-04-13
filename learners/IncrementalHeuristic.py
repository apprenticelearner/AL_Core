"""
Classes of Relational Learners that learn in a General to Specific Fashion.
"""
from pprint import pprint
from random import choice

from py_search.base import Problem
from py_search.base import Node
from py_search.optimization import simulated_annealing
from py_search.optimization import hill_climbing

from planners.fo_planner import Operator
from planners.fo_planner import build_index

from learners.utils import rename
from learners.utils import clause_length
from learners.utils import test_coverage
from learners.utils import get_variablizations
from learners.utils import weighted_choice

clause_accuracy_weight = 0.95


def clause_score(accuracy_weight, p_covered, p_uncovered, n_covered,
                 n_uncovered, length):
    w = accuracy_weight
    accuracy = ((p_covered + n_uncovered) / (p_covered + p_uncovered +
                                             n_covered + n_uncovered))
    return w * accuracy + (1-w) * 1/(1+length)


def build_clause(v, possible_literals):
    return frozenset([possible_literals[i][j] for i, j in enumerate(v) if
                      possible_literals[i][j] is not None])


def clause_vector_score(v, possible_literals, constraints, pset, nset):
    h = build_clause(v, possible_literals)
    l = clause_length(h)
    p_covered, n_covered = test_coverage(h, constraints, pset, nset)
    return clause_score(clause_accuracy_weight, len(p_covered), len(pset) -
                        len(p_covered), len(n_covered), len(nset) -
                        len(n_covered), l)


def compute_bottom_clause(x, mapping):
    reverse_m = {mapping[a]: a for a in mapping}
    # print("REVERSEM", reverse_m)
    partial = set([rename(reverse_m, l) for l in x])
    return frozenset(partial)


def optimize_clause(h, constraints, pset, nset):
    """
    Returns the set of most specific generalization of h that do NOT
    cover x.
    """
    c_length = clause_length(h)
    p_covered, n_covered = test_coverage(h, constraints, pset, nset)
    p_uncovered = [p for p in pset if p not in p_covered]
    n_uncovered = [n for n in nset if n not in n_covered]
    initial_score = clause_score(clause_accuracy_weight, len(p_covered),
                                 len(p_uncovered), len(n_covered),
                                 len(n_uncovered), c_length)
    p, pm = choice(p_covered)
    pos_partial = list(compute_bottom_clause(p, pm))
    print('POS PARTIAL', pos_partial)

    # TODO if we wanted we could add the introduction of new variables to the
    # get_variablizations function.
    possible_literals = {}
    for i, l in enumerate(pos_partial):
        possible_literals[i] = [None, l] + [v for v in get_variablizations(l)]
    partial_literals = set([l for i in possible_literals for l in
                            possible_literals[i]])

    additional_literals = h - partial_literals

    if len(additional_literals) > 0:
        p_index = build_index(p)
        operator = Operator(tuple(('Rule',)),
                            h.union(constraints), [])
        for add_m in operator.match(p_index, initial_mapping=pm):
            break
        additional_lit_mapping = {rename(add_m, l): l for l in
                                  additional_literals}
        for l in additional_lit_mapping:
            new_l = additional_lit_mapping[l]
            print(pos_partial)
            print(add_m)
            print(l)
            print(new_l)
            possible_literals[pos_partial.index(l)].append(new_l)

    pprint(possible_literals)
    reverse_pl = {l: (i, j) for i in possible_literals for j, l in
                  enumerate(possible_literals[i])}

    clause_vector = [0 for i in range(len(possible_literals))]
    for l in h:
        i, j = reverse_pl[l]
        clause_vector[i] = j
    clause_vector = tuple(clause_vector)

    print("INITIAL CLAUSE VECTOR")
    print(clause_vector)

    flip_weights = [(len(possible_literals[i])-1, i) for i in
                    possible_literals]
    size = 1
    for w, _ in flip_weights:
        size *= (w + 1)
    print("SIZE OF SEARCH SPACE:", size)

    num_successors = sum([w for w, c in flip_weights])
    temp_length = num_successors
    temp_length = 10
    initial_temp = 0.16
    print("TEMP LENGTH", temp_length)
    print('INITIAL SCORE', initial_score)
    problem = ClauseOptimizationProblem(clause_vector,
                                        initial_cost=-1*initial_score,
                                        extra=(possible_literals, flip_weights,
                                               constraints, pset, nset))
    # for sol in hill_climbing(problem):
    for sol in simulated_annealing(problem, initial_temp=initial_temp,
                                   temp_length=temp_length):
        # print("SOLUTION FOUND", sol.state)
        return build_clause(sol.state, possible_literals)


class ClauseOptimizationProblem(Problem):

    def goal_test(self, node):
        """
        This is an optimization, so no early termination
        """
        print("GOAL TESTING", node.state, node.cost())
        return False

    def random_successor(self, node):
        clause_vector = node.state
        print("EXPANDING", clause_vector, node.cost())

        possible_literals, flip_weights, constraints, pset, nset = node.extra
        index = weighted_choice(flip_weights)
        new_j = choice([j for j in range(len(possible_literals[index]))
                        if j != clause_vector[index]])
        new_clause_vector = tuple(new_j if i == index else j for i, j in
                                  enumerate(clause_vector))
        print("SCORING NEW CLAUSE")
        score = clause_vector_score(new_clause_vector, possible_literals,
                                    constraints, pset, nset)
        print("DONE SCORING")
        return Node(new_clause_vector, None, None, -1 * score,
                    extra=node.extra)

    def successors(self, node):
        clause_vector = node.state
        possible_literals, flip_weights, constraints, pset, nset = node.extra

        for index in possible_literals:
            for new_j in range(len(possible_literals[index])):
                if new_j == clause_vector[index]:
                    continue

                new_clause_vector = tuple(new_j if i == index else j for i, j
                                          in enumerate(clause_vector))
                score = clause_vector_score(new_clause_vector,
                                            possible_literals, constraints,
                                            pset, nset)
                yield Node(new_clause_vector, None, None, -1 * score,
                            extra=node.extra)


class IncrementalHeuristic(object):

    def __init__(self, args=None, constraints=None):
        """
        A relational learner that searches the space of hypotheses locally.
        Whenever it receives a new positive or negative example it tries to
        further optimize its hypothesis.

        args - a tuple of arguments to the learner, these are the args in the
        head of the rule.
        constraints - a set of constraints that cannot be removed. These can be
        used to ensure basic things like an FOA must have a value that isn't an
        empty string, etc.
        """
        if args is None:
            args = tuple([])
        if constraints is None:
            constraints = frozenset([])

        self.args = args
        self.constraints = constraints
        self.pset = []
        self.nset = []
        self.h = None
        self.h = frozenset([])
        self.gen_counter = 0

    def gensym(self):
        self.gen_counter += 1
        return '?new_gen%i' % self.gen_counter

    def get_hset(self):
        """
        Gets a list of hypotheses. This is essentially a disjunction of
        conjunctions. Each hypothesis can be fed into a pattern matcher to
        perform matching.
        """
        if self.h is None:
            return []
        return [self.h.union(self.constraints)]

    def compute_bottom_clause(self, x, mapping):
        reverse_m = {mapping[a]: a for a in mapping}
        # print("REVERSEM", reverse_m)
        partial = set([rename(reverse_m, l) for l in x])
        return frozenset(partial)

    def ifit(self, t, x, y):
        """
        Incrementally specializes the hypothesis set.
        """
        mapping = {a: t[i] for i, a in enumerate(self.args)}

        if y == 1:
            self.pset.append((x, mapping))
        elif y == 0:
            self.nset.append((x, mapping))
        else:
            raise Exception("y must be 0 or 1")

        if self.h is None and y == 1:
            self.h = self.compute_bottom_clause(x, mapping)
            # print("ADDING BOTTOM", self.h)

        if self.h is not None:
            self.h = optimize_clause(self.h, self.constraints, self.pset,
                                     self.nset)
            c_length = clause_length(self.h)
            p_covered, n_covered = test_coverage(self.h, self.constraints,
                                                 self.pset, self.nset)
            p_uncovered = [p for p in self.pset if p not in p_covered]
            n_uncovered = [n for n in self.nset if n not in n_covered]
            score = clause_score(clause_accuracy_weight, len(p_covered),
                                 len(p_uncovered), len(n_covered),
                                 len(n_uncovered), c_length)

            print("OVERALL SCORE", score)


if __name__ == "__main__":

    p1 = {('color', 'dark'),
          ('tails', '2'),
          ('nuclei', '2'),
          ('wall', 'thin')}
    n1 = {('color', 'light'),
          ('tails', '2'),
          ('nuclei', '1'),
          ('wall', 'thin')}
    p2 = {('color', 'light'),
          ('tails', '2'),
          ('nuclei', '2'),
          ('wall', 'thin')}
    n2 = {('color', 'dark'),
          ('tails', '1'),
          ('nuclei', '2'),
          ('wall', 'thick')}

    X = [p1, n1, p2, n2]
    y = [1, 0, 1, 0]

    learner = IncrementalHeuristic()

    for i, x in enumerate(X):
        print("Adding the following instance (%i):" % y[i])
        pprint(x)
        learner.ifit(tuple([]), x, y[i])
        print("Resulting hset")
        print(learner.get_hset())
        print(len(learner.get_hset()))

    p1 = {('person', 'a'),
          ('person', 'b'),
          ('person', 'c'),
          ('parent', 'a', 'b'),
          ('parent', 'b', 'c')}

    n1 = {('person', 'a'),
          ('person', 'b'),
          ('person', 'f'),
          ('person', 'g'),
          ('parent', 'a', 'b'),
          ('parent', 'f', 'g')}

    p2 = {('person', 'f'),
          ('person', 'g'),
          ('person', 'e'),
          ('parent', 'e', 'f'),
          ('parent', 'f', 'g')}

    X = [p1, n1, p2]
    y = [1, 0, 1]
    t = [('a', 'c'), ('a', 'g'), ('e', 'g')]

    learner = IncrementalHeuristic(args=('?A', '?B'),
                                   constraints=frozenset([('person', '?A'),
                                                          ('person', '?B')]))

    for i, x in enumerate(X):
        print("Adding the following instance (%i):" % y[i])
        pprint(x)
        learner.ifit(t[i], x, y[i])
        print("Resulting hset")
        print(learner.get_hset())
        print(len(learner.get_hset()))
