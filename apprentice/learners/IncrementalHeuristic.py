"""
Classes of Relational Learners that learn in a General to Specific Fashion.
"""
from pprint import pprint
from random import choice

from py_search.base import Problem
from py_search.base import Node
from py_search.optimization import simulated_annealing
# from py_search.optimization import hill_climbing

from planners.fo_planner import Operator
from planners.fo_planner import build_index

from learners.utils import rename
from learners.utils import clause_length
from learners.utils import count_elements
from learners.utils import test_coverage
from learners.utils import get_variablizations
from learners.utils import weighted_choice

clause_accuracy_weight = 0.95
max_literal_length = 300


def clause_score(accuracy_weight, p_covered, p_uncovered, n_covered,
                 n_uncovered, length):

    # print(length)
    return p_covered - 20 * n_covered - 1/10 * length

    # w = accuracy_weight
    # # accuracy = (p_covered / (p_covered + n_covered))

    # n_uncovered = 4 * n_uncovered
    # n_covered = 4 * n_covered

    # accuracy = ((p_covered + n_uncovered) / (p_covered + p_uncovered +
    #                                          n_covered + n_uncovered))
    # return w * accuracy + (1-w) * 1/(1+length)


def build_clause(v, possible_literals):
    return frozenset([possible_literals[i][j] for i, j in enumerate(v) if
                      possible_literals[i][j] is not None])


def clause_vector_score(v, possible_literals, constraints, pset, nset):
    h = build_clause(v, possible_literals)
    l = clause_length(h)
    print("length", l)
    p_covered, n_covered = test_coverage(h, constraints, pset, nset)
    # if len(p_covered) < 1:
    #     pprint(pset)
    #     pprint(h)
    #     import time
    #     print("NO POSITIVES COVERED!!!!!!")
    #     time.sleep(1000)
    #     assert False

    return (clause_score(clause_accuracy_weight, len(p_covered), len(pset) -
                         len(p_covered), len(n_covered), len(nset) -
                         len(n_covered), l), len(pset) - len(p_covered),
            len(n_covered))


def compute_bottom_clause(x, mapping):
    reverse_m = {mapping[a]: a for a in mapping}
    # print("REVERSEM", reverse_m)
    partial = set([rename(reverse_m, l) for l in x])
    return frozenset(partial)


def optimize_clause(h, constraints, seed, pset, nset):
    """
    Returns the set of most specific generalization of h that do NOT
    cover x.
    """
    c_length = clause_length(h)
    print('# POS', len(pset))
    print('# NEG', len(nset))
    print('length', c_length)

    p_covered, n_covered = test_coverage(h, constraints, pset, nset)
    p_uncovered = [p for p in pset if p not in p_covered]
    n_uncovered = [n for n in nset if n not in n_covered]
    print("P COVERED", len(p_covered))
    print("N COVERED", len(n_covered))
    initial_score = clause_score(clause_accuracy_weight, len(p_covered),
                                 len(p_uncovered), len(n_covered),
                                 len(n_uncovered), c_length)

    p, pm = seed
    pos_partial = list(compute_bottom_clause(p, pm))
    # print('POS PARTIAL', pos_partial)

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
            reverse_m = {pm[a]: a for a in pm}
            l = rename(reverse_m, l)
            print(pos_partial)
            print(add_m)
            print(l)
            print(new_l)
            print(additional_literals)
            if l not in pos_partial:
                print("ERROR l not in pos_partial")
                import time
                time.sleep(1000)
            possible_literals[pos_partial.index(l)].append(new_l)

    # pprint(possible_literals)
    reverse_pl = {l: (i, j) for i in possible_literals for j, l in
                  enumerate(possible_literals[i])}

    clause_vector = [0 for i in range(len(possible_literals))]
    for l in h:
        if l not in reverse_pl:
            print("MISSING LITERAL!!!")
            import time
            time.sleep(1000)

        i, j = reverse_pl[l]
        clause_vector[i] = j
    clause_vector = tuple(clause_vector)

    # print("INITIAL CLAUSE VECTOR")
    # print(clause_vector)

    flip_weights = [(len(possible_literals[i])-1, i) for i in
                    possible_literals]
    size = 1
    for w, _ in flip_weights:
        size *= (w + 1)
    print("SIZE OF SEARCH SPACE:", size)

    num_successors = sum([w for w, c in flip_weights])
    print("NUM SUCCESSORS", num_successors)
    temp_length = num_successors
    temp_length = 10
    # initial_temp = 0.15
    # initial_temp = 0.0
    print("TEMP LENGTH", temp_length)
    print('INITIAL SCORE', initial_score)
    problem = ClauseOptimizationProblem(clause_vector,
                                        initial_cost=-1*initial_score,
                                        extra=(possible_literals, flip_weights,
                                               constraints, pset, nset,
                                               len(p_uncovered),
                                               len(n_covered)))
    # for sol in hill_climbing(problem):
    for sol in simulated_annealing(problem,
                                   # initial_temp=initial_temp,
                                   temp_length=temp_length
                                   ):
        # print("SOLUTION FOUND", sol.state)
        print('FINAL SCORE', -1 * sol.cost())
        return build_clause(sol.state, possible_literals)


class ClauseOptimizationProblem(Problem):

    def goal_test(self, node):
        """
        This is an optimization, so no early termination

        Early terminate at the minimum possible cost.
        """
        return False

        return node.cost() < 0
        # return False

        (possible_literals, flip_weights, constraints, pset, nset, omissions,
         comissions) = node.extra
        # return False
        return omissions + comissions == 0
        # return node.cost() < -len(pset) + 30/100
        # return node.cost() < -0.75
        # return False
        # return node.cost() < - 0.95
        # return node.cost() == -1

    def random_successor(self, node):
        clause_vector = node.state
        # print("EXPANDING", clause_vector, node.cost())

        (possible_literals, flip_weights, constraints, pset, nset, omissions,
         comissions) = node.extra

        # omissions += 1.5
        # comissions += 1.5

        # print()
        # print("OMISSIONS", omissions)
        # print("COMISSIONS", comissions)

        # if omissions > comissions:
        #     print("LEN 1", 1 / ((comissions / omissions) *
        #                   (max_literal_length
        #                                                     - 1 + 0.1)))
        #     print("LEN 3", 1 / ((comissions / omissions) *
        #                   (max_literal_length               - 3 + 0.1)))

        # if comissions > omissions:
        #     print("LEN 1", 1 / ((comissions / omissions) * (1 + 0.1)))
        #     print("LEN 3", 1 / ((comissions / omissions) * (3 + 0.1)))

        # gen_bias = (omissions) / (omissions + comissions)
        # gen_bias = (omissions - comissions) / max(omissions, comissions)
        # print("Gen Bias", omissions / (omissions + comissions))

        flip_weights = [(-1 * (2 + comissions) *
                         count_elements(possible_literals[i][clause_vector[i]],
                                        {}), i) if comissions > 0 else
                        ((2 + omissions) *
                         count_elements(possible_literals[i][clause_vector[i]],
                                        {}), i)# if omissions > comissions else
                        # (0.1 + (omissions / comissions) * (max_literal_length -
                        #  count_elements(possible_literals[i][clause_vector[i]],
                        #                 {})), i) if omissions > comissions else
                        #(1, i)
                        for i in range(len(clause_vector))]

        smallest = min([w for w, i in flip_weights])
        flip_weights = [(0.001 + (w - smallest), i) for w, i in flip_weights]

        # output = [(w, possible_literals[i][clause_vector[i]]) for w, i in flip_weights]
        # pprint(output)

        #flip_weights = [(1, i) for i in range(len(clause_vector))]

        index = weighted_choice(flip_weights)

        # curr_l_size = count_elements(possible_literals[index][clause_vector[index]], {})
        # print("CURR L", possible_literals[index][clause_vector[index]])

        # print("CURRENT SIZE", curr_l_size)

        # print("GEN BIAS", gen_bias)

        # # print(possible_literals[index])
        # weighted_pl = [(gen_bias * (curr_l_size - count_elements(l, {})), j)
        #                for j, l in enumerate(possible_literals[index]) if j !=
        #                clause_vector[index]]
        # min_weight = min([w for w, _ in weighted_pl])
        # if min_weight < 0:
        #     weighted_pl = [(w + abs(min_weight) + 1, j) for w, j in weighted_pl]
        # # weighted_pl.sort()

        # # print("WPL", weighted_pl)

        # new_j = weighted_choice(weighted_pl)

        # print("NEW L", possible_literals[index][new_j])
        # # print(new_j)

        new_j = choice([j for j in range(len(possible_literals[index]))
                        if j != clause_vector[index]])

        new_clause_vector = tuple(new_j if i == index else j for i, j in
                                  enumerate(clause_vector))
        # print("SCORING")
        score, om, cm = clause_vector_score(new_clause_vector,
                                            possible_literals, constraints,
                                            pset, nset)
        # print("Done - Score =", score)
        print("Score = %0.4f, Omissions = %i, Comissions = %i" % (score, om, cm))
        return Node(new_clause_vector, None, None, -1 * score,
                    extra=(possible_literals, flip_weights, constraints, pset,
                          nset, om, cm))

    def successors(self, node):
        clause_vector = node.state
        (possible_literals, flip_weights, constraints, pset, nset, omissions,
         comissions) = node.extra

        for index in possible_literals:
            for new_j in range(len(possible_literals[index])):
                if new_j == clause_vector[index]:
                    continue

                new_clause_vector = tuple(new_j if i == index else j for i, j
                                          in enumerate(clause_vector))
                score, om, cm = clause_vector_score(new_clause_vector,
                                                    possible_literals,
                                                    constraints, pset, nset)
                yield Node(new_clause_vector, None, None, -1 * score,
                           extra=(possible_literals, flip_weights, constraints,
                                  pset, nset, om, cm))


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
        self.hset = []
        # self.hset.append(frozenset([]))
        self.gen_counter = 0
        self.out_of_date = False

    def gensym(self):
        self.gen_counter += 1
        return '?new_gen%i' % self.gen_counter

    def get_hset(self):
        """
        Gets a list of hypotheses. This is essentially a disjunction of
        conjunctions. Each hypothesis can be fed into a pattern matcher to
        perform matching.
        """
        self.optimize_hypotheses()
        return [h.union(self.constraints) for h, seed in self.hset]
        # if self.h is None:
        #     return []
        # return [self.h.union(self.constraints)]

    def compute_bottom_clause(self, x, mapping):
        reverse_m = {mapping[a]: a for a in mapping}
        # print("REVERSEM", reverse_m)
        partial = set([rename(reverse_m, l) for l in x])
        return frozenset(partial)

    def optimize_hypotheses(self):

        if self.out_of_date is False:
            return

        if len(self.pset) == 0:
            self.hset = []

        # self.hset = [optimize_clause(h, self.constraints, self.pset,
        # self.nset) for h in self.hset]
        remaining_p = [p for p in self.pset]

        new_hset = []
        for h, seed in self.hset:
            # new_h = optimize_clause(h, self.constraints, self.pset,
            # self.nset)
            new_h = optimize_clause(h, self.constraints, seed, self.pset,
                                    self.nset)
            c_length = clause_length(new_h)
            p_covered, n_covered = test_coverage(new_h, self.constraints,
                                                 self.pset, self.nset)
            p_uncovered = [p for p in self.pset if p not in p_covered]
            n_uncovered = [n for n in self.nset if n not in n_covered]
            score = clause_score(clause_accuracy_weight, len(p_covered),
                                 len(p_uncovered), len(n_covered),
                                 len(n_uncovered), c_length)
            new_hset.append((score, p_covered, new_h, seed))
            remaining_p = [p for p in remaining_p if p not in p_covered]

            if len(remaining_p) == 0:
                break

        while len(remaining_p) > 0:
            seed = choice(remaining_p)
            # new_h = self.compute_bottom_clause(p, pm)
            new_h = frozenset([])
            print("ADDING NEW!")
            new_h = optimize_clause(new_h, self.constraints, seed, self.pset,
                                    self.nset)

            c_length = clause_length(new_h)
            p_covered, n_covered = test_coverage(new_h, self.constraints,
                                                 self.pset, self.nset)
            p_uncovered = [p for p in self.pset if p not in p_covered]
            n_uncovered = [n for n in self.nset if n not in n_covered]
            score = clause_score(clause_accuracy_weight, len(p_covered),
                                 len(p_uncovered), len(n_covered),
                                 len(n_uncovered), c_length)
            new_hset.append((score, p_covered, new_h, seed))
            remaining_p = [p for p in remaining_p if p not in p_covered]

        # print(new_hset)
        new_hset.sort(key=lambda x: -x[0])

        print(new_hset)
        self.hset = []
        # remove subsumed and inaccurate?
        for i, (_, outer_covered, new_h, seed) in enumerate(new_hset):
            contained = False
            for (_, inner_covered, _, _) in new_hset[i+1:]:
                if contained:
                    continue
                outer_set = set([(frozenset(p), frozenset(pm.items())) for p, pm in outer_covered])
                inner_set = set([(frozenset(p), frozenset(pm.items())) for p, pm in inner_covered])
                if inner_set >= outer_set:
                    print("FOUND CONTAINED")
                    contained = True
            if not contained:
                self.hset.append((new_h, seed))

        # self.hset = [(h, seed) for _, h, seed in new_hset]
        self.out_of_date = False

    def ifit(self, t, x, y):
        """
        Incrementally specializes the hypothesis set.
        """
        mapping = {a: t[i] for i, a in enumerate(self.args)}

        if y == 1:
            self.pset.append((x, mapping))

            # if the new x is not covered then mark out of date
            # covered = False
            # for h, seed in self.hset:
            #     pc, _ = test_coverage(h, self.constraints, [(x, mapping)], [])
            #     if len(pc) == 0:
            #         covered = True
            #         break

            # if not covered:
            #     self.out_of_date = True

        elif y == 0:
            self.nset.append((x, mapping))

            # if the new x is covered then mark out of date
            # covered = False
            # for h, seed in self.hset:
            #     _, nc = test_coverage(h, self.constraints, [], [(x, mapping)])
            #     if len(nc) > 0:
            #         covered = True
            #         break

            # if covered:
            #     self.out_of_date = True

        else:
            raise Exception("y must be 0 or 1")

        self.out_of_date = True

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

    p1 = {('tails', '1'),
          ('nuclei', '2')}
    p2 = {('tails', '2'),
          ('nuclei', '1')}
    n1 = {('tails', '1'),
          ('nuclei', '1')}

    X = [p1, n1, p2]
    y = [1, 0, 1]

    learner = IncrementalHeuristic()

    for i, x in enumerate(X):
        print("Adding the following instance (%i):" % y[i])
        pprint(x)
        learner.ifit(tuple([]), x, y[i])
        print("Resulting hset")
        print(learner.get_hset())
        print(len(learner.get_hset()))

    # p1 = {('person', 'a'),
    #       ('person', 'b'),
    #       ('person', 'c'),
    #       ('parent', 'a', 'b'),
    #       ('parent', 'b', 'c')}

    # n1 = {('person', 'a'),
    #       ('person', 'b'),
    #       ('person', 'f'),
    #       ('person', 'g'),
    #       ('parent', 'a', 'b'),
    #       ('parent', 'f', 'g')}

    # p2 = {('person', 'f'),
    #       ('person', 'g'),
    #       ('person', 'e'),
    #       ('parent', 'e', 'f'),
    #       ('parent', 'f', 'g')}

    # X = [p1, n1, p2]
    # y = [1, 0, 1]
    # t = [('a', 'c'), ('a', 'g'), ('e', 'g')]

    # learner = IncrementalHeuristic(args=('?A', '?B'),
    #                                constraints=frozenset([('person', '?A'),
    #                                                       ('person', '?B')]))

    # for i, x in enumerate(X):
    #     print("Adding the following instance (%i):" % y[i])
    #     pprint(x)
    #     learner.ifit(t[i], x, y[i])
    #     print("Resulting hset")
    #     print(learner.get_hset())
    #     print(len(learner.get_hset()))
