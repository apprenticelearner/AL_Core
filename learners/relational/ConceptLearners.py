"""
Functions for performing antiunification.
"""

from pprint import pprint
from itertools import combinations
from itertools import product
from itertools import chain
from random import random
from random import choice

from scipy.optimize import linear_sum_assignment

from py_search.base import Problem
from py_search.base import Node
from py_search.uninformed import depth_first_search
from py_search.uninformed import breadth_first_search
from py_search.informed import best_first_search
from py_search.optimization import hill_climbing
from py_search.optimization import simulated_annealing
from py_search.utils import compare_searches

from planners.fo_planner import Operator
from planners.fo_planner import build_index
from planners.fo_planner import subst
from planners.fo_planner import extract_strings
from planners.fo_planner import is_variable

from learners.relational.utils import covers
from learners.relational.utils import rename


def simple_clause_antiunification(h, x):
    """
    This is a simple antiunification of clauses (note the length will grow
    exponentially). Could consider adding some kind of clause reduction step
    either using ij-determinacy to remove literals OR somehow using negative
    examples? I don't really understand how either of these approaches work.

    For more details see:
        Muggleton, S., & Feng, C. (1990). Efficient induction of logic
        programs. Turing Institute.

    """
    au_table = build_antiunify_table(h, x)
    new_h = set(au_table[p] for p in au_table
                if not isinstance(au_table[p], frozenset))
    new_h = variablize_hypothesis(new_h)

    return new_h


def local_generalization_search(h, x):
    """
    This tries to find a maximal partial match between h and x. To do it it
    antiunifies each of the terms in h and x, then uses the hungarian algorithm
    to compute the best bipartite match between the literals in h and x. This
    constitutes an initial match. Then hill climbing is performed over the
    space of all possible flips of the bipartite matches. The hungarian
    algorithm doesn't consider the benefit of creating antiunifications
    that share variables. The local search is an effort to overcome this
    limitation.

    Note, this is not guranteed to return a maximal partial match, but it is
    probably good enough.
    """

    ### CODE FOR AD HOC HANDLING OF NEGATIONS
    # x = set(l for l in x)

    # # Need to compute any implicit h negations
    # neg_relations = {}
    # for l in h:
    #     if l[0] == 'not':
    #         key = "%s/%i" % (l[1][0], len(l[1][1:]))
    #         if key not in neg_relations:
    #             neg_relations[key] = {}
    #         for i, v in enumerate(l[1][1:]):
    #             if i not in neg_relations[key]:
    #                 neg_relations[key][i] = set()
    #             if not is_variable(v):
    #                 neg_relations[key][i].add(v)

    # for l in x:
    #     key = "%s/%i" % (l[0], len(l[1:]))
    #     if key not in neg_relations:
    #         continue
    #     for i, v in enumerate(l[1:]):
    #         if i not in neg_relations[key]:
    #             neg_relations[key][i] = set()
    #         neg_relations[key][i].add(v)

    # print(neg_relations)

    # # get neg relations
    # # compute domain of each arg of neg relation
    # # add all possible negs given these domains
    # for key in neg_relations:
    #     head, arity = key.split("/")
    #     arity = int(arity)
    #     args = [neg_relations[key][i] for i in range(arity)]
    #     for t in product(*args):
    #         new_lit = (head,) + t
    #         if new_lit not in x:
    #             x.add(('not', new_lit))

    # # print("NEG AUGMENTED X", x)

    au_table = build_antiunify_table(h, x)
    # Below is a guided search for a single specialization. It doesn't produce
    # the true antiunification, just a single possible specialization.

    if len(x) < len(h):
        temp = h
        h = x
        x = temp

    h = tuple(e for e in h)
    x = tuple(e for e in x)
    # m = tuple(i for i in range(len(h)))
    m = hungarian_mapping(h, x, au_table)

    r = evaluate_reward(m, h, x, au_table)
    const_count, var_counts = get_counts(m, h, x, au_table)

    # print(h)
    # print(x)
    # print("REWARD", r)

    problem = LocalAntiUnifyProblem(m, initial_cost=-r,
                                    extra=(h, x, [len(h) + i for i in
                                                  range(len(x) - len(h))],
                                           au_table, const_count,
                                           var_counts))

    sol = next(hill_climbing(problem))
    # print("FINAL SOLUTION", sol.state, sol.cost())

    new_h = []
    for i, a in enumerate(h):
        # print(a, 'with', x[m[i]])
        new_l = au_table[frozenset([a, x[sol.state[i]]])]
        if not isinstance(new_l, frozenset):
            new_h.append(new_l)
    # print(antiunify_reward(new_h))

    # print(variablize_hypothesis(new_h))
    return variablize_hypothesis(new_h)


def exhaustive_generalization_search(h, x):
    """
    This computes all possible partial antiunifications. I'm not 100% sure, but
    I'm pretty sure that It is guranteed to return all possible minimal
    specializations of the hypothesis. However, since it has to build up a
    match from nothing and since the solutions are at the leaves of the tree
    and since there are many possible paths to the same solution, this can be
    an expensive approach. 
    """

    au_table = build_antiunify_table(h, x)
    problem = AntiUnifyProblem(frozenset(), extra=(h, frozenset(x), au_table,
                                                   frozenset()))
    for sol in best_first_search(problem):
        return set([variablize_hypothesis(sol.state)])
        print(sol.state)
        break
    return set(variablize_hypothesis(sol.state) for sol in
               best_first_search(problem))


def generalize(h, x):
    # return simple_clause_antiunification(h, x)
    # return exhaustive_generalization_search(h, x)
    return local_generalization_search(h, x)


def specialize(h, constraints, args, pset, neg, neg_mapping, gensym,
               depth_limit=10):
    """
    Returns the set of most general specializations of h that does NOT
    cover x.
    """
    problem = SpecializationProblem(h, extra=(args, constraints, pset, neg,
                                              neg_mapping, gensym))
    sol_set = set()
    for sol in breadth_first_search(problem, depth_limit=depth_limit):
        sol_set.add(sol.state)
        if len(sol_set) >= 25:
            break
    return sol_set


def get_counts(m, h, x, au_table):
    c = 0
    var_counts = {}
    new_h = []
    for i, a in enumerate(h):
        new_l = au_table[frozenset([a, x[m[i]]])]
        if not isinstance(new_l, frozenset):
            new_h.append(new_l)
    for ele in new_h:
        c += count_term(ele, var_counts)
    return c, var_counts


def evaluate_reward(m, h, x, au_table):
    new_h = []
    for i, a in enumerate(h):
        new_l = au_table[frozenset([a, x[m[i]]])]
        if not isinstance(new_l, frozenset):
            new_h.append(new_l)
    r = antiunify_reward(new_h)
    # print("H", new_h, r)
    return r


def count_term(x, var_counts):
    c = 0
    if isinstance(x, tuple):
        c = sum([count_term(ele, var_counts) for ele in x])
    elif isinstance(x, frozenset):
        if x not in var_counts:
            var_counts[x] = 0
        var_counts[x] += 1
    else:
        c = 1
    return c


def antiunify_reward(h):
    c = 0
    var_counts = {}
    for ele in h:
        c += count_term(ele, var_counts)
    for v in var_counts:
        c += var_counts[v] - 1
    return c




def count_occurances(var, h):
    return len([s for x in h for s in extract_strings(x) if s == var])


def get_elements(literal):
    """
    Given a literal returns all of the values (no relations).
    """
    if isinstance(literal, tuple):
        for ele in literal[1:]:
            for inner in get_elements(ele):
                yield inner
    else:
        yield literal


def get_variablizations(literal, gensym):
    for i, ele in enumerate(literal[1:]):
        if isinstance(ele, tuple):
            for inner in get_variablizations(ele, gensym):
                yield tuple([literal[0]] + [inner if j == i else iele for j,
                                            iele in enumerate(literal[1:])])
        elif not is_variable(ele):
            yield tuple([literal[0]] + [gensym() if j == i else iele for j,
                                        iele in enumerate(literal[1:])])


def antiunify(x, y):
    if x == y:
        return x
    elif isinstance(x, tuple) and isinstance(y, tuple):
        if x[0] != y[0]:
            return frozenset([x, y])
        else:
            return (x[0],) + tuple(antiunify(x[i+1], y[i+1]) for i in
                                   range(len(x)-1))
    else:
        return frozenset([x, y])


def build_antiunify_table(e1, e2):
    return {frozenset([l1, l2]): antiunify(l1, l2) for l1 in e1 for l2 in e2}


def variablize_literal(l, get_var):
    new_l = []
    for ele in l:
        if isinstance(ele, frozenset):
            new_l.append(get_var(ele))
        elif isinstance(ele, tuple):
            new_l.append(variablize_literal(ele, get_var))
        else:
            new_l.append(ele)
    return tuple(new_l)


def variablize_hypothesis(h):
    """
    Takes a hypothesis with frozen sets and converts them to
    variables.
    """
    variable_table = {}
    count = 0

    def get_var(fz):
        nonlocal count

        if fz in variable_table:
            return variable_table[fz]
        else:
            count += 1
            var = '?%s%i' % ('var', count)
            variable_table[fz] = var
            return var

    new_h = frozenset([variablize_literal(l, get_var) for l in h])
    return new_h


def hungarian_mapping(h, x, au_table):
    """
    Uses the hungarian algorithm to compute the best 1 to 1 mapping of the
    literals between two examples (or a hypothesis and an example).
    """
    cost_matrix = []
    for he in h:
        row = []
        for xe in x:
            var_counts = {}
            reward = count_term(au_table[frozenset([he, xe])], var_counts)
            for var in var_counts:
                reward += var_counts[var] - 1
            cost = -1 * reward
            row.append(cost)

        cost_matrix.append(row)

    # print("HUNGARIAN")
    # print('h', h)
    # print('x', x)
    # print(cost_matrix)

    # Use Scipy because it is much faster than munkres
    indices = linear_sum_assignment(cost_matrix)

    # print("SOLUTION")
    # print(tuple(indices[1]))
    # print()

    # print("SHOULD DELETE THIS ASSERTION BUT JUST RUN IT TO TEST")
    # assert [i for i in range(len(h))] == list(indices[0])
    return tuple(indices[1])


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def contains_variables(literal):
    for ele in literal:
        if is_variable(ele):
            return True
        elif isinstance(ele, tuple) and contains_variables(ele):
            return True
    return False


def remove_vars(literal):
    return tuple('XXX' + ele if is_variable(ele) else remove_vars(ele) if
                 isinstance(ele, tuple) else ele for ele in literal)


def generalize_literal(literal, gensym):
    return (literal[0],) + tuple(ele if is_variable(ele) else
                                 # '?gen%s' % hash(ele)
                                 gensym()
                                 for ele in literal[1:])
    # if isinstance(literal, tuple):
    #     for s in powerset([i+1 for i in range(len(literal)-1)]):
    #         s = set(s)
    #         yield tuple(generalize_literal(ele) if isinstance(ele, tuple) else
    #                     '?gen%s' % hash(ele) if j in s else ele for j, ele in
    #                     enumerate(literal))

    #      for i, v in enumerate(literal):
    #          if i == 0:
    #              continue
    #          for gen in generalize_literal(v):
    #              yield tuple(gen if i == j else ele for j, ele in
    #                          enumerate(literal))
    #  else:
    #      yield '?gen%s' % hash(literal)
    #      yield literal


class LocalAntiUnifyProblem(Problem):

    def random_successor(self, node):
        m = node.state
        h, example, unassigned, au_table, const_count, var_counts = node.extra

        # print(m)

        flips = [(a, b) for a, b in combinations(range(len(h)), 2)]
        una = [(a, ua) for a in range(len(h)) for ua in unassigned]

        if random() <= len(flips) / (len(flips) + len(una)):
            a, b = choice(flips)
            new_var_counts = {a: var_counts[a] for a in var_counts}
            new_m = tuple(m[b] if i == a else m[a] if i == b else v for i, v in
                          enumerate(m))
            old_a = au_table[frozenset([h[a], example[m[a]]])]
            old_b = au_table[frozenset([h[b], example[m[b]]])]
            new_a = au_table[frozenset([h[a], example[new_m[a]]])]
            new_b = au_table[frozenset([h[b], example[new_m[b]]])]

            var_old_a = {}
            if isinstance(old_a, frozenset):
                c_old_a = 0
            else:
                c_old_a = count_term(old_a, var_old_a)
            var_new_a = {}
            if isinstance(new_a, frozenset):
                c_new_a = 0
            else:
                c_new_a = count_term(new_a, var_new_a)

            var_old_b = {}
            if isinstance(old_b, frozenset):
                c_old_b = 0
            else:
                c_old_b = count_term(old_b, var_old_b)
            var_new_b = {}
            if isinstance(new_b, frozenset):
                c_new_b = 0
            else:
                c_new_b = count_term(new_b, var_new_b)

            # print()
            # print()
            # print(node.cost())
            # print(new_var_counts)
            # print(old_a, '->', new_a)
            # print(c_old_a, var_old_a)
            # print(c_new_a, var_new_a)
            # print()
            # print(old_b, '->', new_b)
            # print(c_old_b, var_old_b)
            # print(c_new_b, var_new_b)

            new_reward = -1 * node.cost()
            new_reward -= c_old_a
            new_reward -= c_old_b
            new_reward += c_new_a
            new_reward += c_new_b

            for var in var_old_a:
                if var in new_var_counts and new_var_counts[var] > 1:
                    new_reward -= 1
                    new_var_counts[var] -= 1
                if var in new_var_counts and new_var_counts[var] == 1:
                    del new_var_counts[var]

            for var in var_old_b:
                if var in new_var_counts and new_var_counts[var] > 1:
                    new_reward -= 1
                    new_var_counts[var] -= 1
                if var in new_var_counts and new_var_counts[var] == 1:
                    del new_var_counts[var]

            for var in var_new_a:
                if var in new_var_counts:
                    new_reward += 1
                    new_var_counts[var] += 1
                else:
                    new_var_counts[var] = 1

            for var in var_new_b:
                if var in new_var_counts:
                    new_reward += 1
                    new_var_counts[var] += 1
                else:
                    new_var_counts[var] = 1

            # print()
            # print((a, b), m, new_m)
            # new_cost = -1 * evaluate_reward(new_m, h, example, au_table)
            new_cost = -1 * new_reward
            # print("NEW VARCOUNTS", new_var_counts)
            # print("COMPARISON")
            # print(-1 * new_reward, new_cost)
            # print()
            # print("NODE COST", new_node.cost())
            return Node(new_m, node, ('swap', a, b), new_cost,
                        (h, example, unassigned, au_table, const_count,
                         new_var_counts))

        else:
            a, ua = choice(una)
            new_var_counts = {a: var_counts[a] for a in var_counts}
            new_m = tuple(ua if i == a else v for i, v in enumerate(m))
            new_unassigned = [m[a] if x == ua else x for x in unassigned]

            old_a = au_table[frozenset([h[a], example[m[a]]])]
            new_a = au_table[frozenset([h[a], example[new_m[a]]])]

            var_old_a = {}
            if isinstance(old_a, frozenset):
                c_old_a = 0
            else:
                c_old_a = count_term(old_a, var_old_a)
            var_new_a = {}
            if isinstance(new_a, frozenset):
                c_new_a = 0
            else:
                c_new_a = count_term(new_a, var_new_a)

            new_reward = -1 * node.cost()
            new_reward -= c_old_a
            new_reward += c_new_a

            for var in var_old_a:
                if var in new_var_counts and new_var_counts[var] > 1:
                    new_reward -= 1
                    new_var_counts[var] -= 1
                if var in new_var_counts and new_var_counts[var] == 1:
                    del new_var_counts[var]

            for var in var_new_a:
                if var in new_var_counts:
                    new_reward += 1
                    new_var_counts[var] += 1
                else:
                    new_var_counts[var] = 1

            new_cost = -1 * new_reward

            # new_cost = -1 * evaluate_reward(new_m, h, example, au_table)

            return Node(new_m, node, ('swap unassigned', a, ua),
                       new_cost, (h, example, new_unassigned, au_table,
                                  const_count, var_counts))

    def successors(self, node):
        m = node.state
        h, example, unassigned, au_table, const_count, var_counts = node.extra

        # print(m)
        # print(h)
        # print(example)

        for a, b in combinations(range(len(h)), 2):
            new_var_counts = {a: var_counts[a] for a in var_counts}
            new_m = tuple(m[b] if i == a else m[a] if i == b else v for i, v in
                          enumerate(m))
            old_a = au_table[frozenset([h[a], example[m[a]]])]
            old_b = au_table[frozenset([h[b], example[m[b]]])]
            new_a = au_table[frozenset([h[a], example[new_m[a]]])]
            new_b = au_table[frozenset([h[b], example[new_m[b]]])]

            var_old_a = {}
            if isinstance(old_a, frozenset):
                c_old_a = 0
            else:
                c_old_a = count_term(old_a, var_old_a)
            var_new_a = {}
            if isinstance(new_a, frozenset):
                c_new_a = 0
            else:
                c_new_a = count_term(new_a, var_new_a)

            var_old_b = {}
            if isinstance(old_b, frozenset):
                c_old_b = 0
            else:
                c_old_b = count_term(old_b, var_old_b)
            var_new_b = {}
            if isinstance(new_b, frozenset):
                c_new_b = 0
            else:
                c_new_b = count_term(new_b, var_new_b)

            # print()
            # print()
            # print(node.cost())
            # print(new_var_counts)
            # print(old_a, '->', new_a)
            # print(c_old_a, var_old_a)
            # print(c_new_a, var_new_a)
            # print()
            # print(old_b, '->', new_b)
            # print(c_old_b, var_old_b)
            # print(c_new_b, var_new_b)

            new_reward = -1 * node.cost()
            new_reward -= c_old_a
            new_reward -= c_old_b
            new_reward += c_new_a
            new_reward += c_new_b

            for var in var_old_a:
                if var in new_var_counts and new_var_counts[var] > 1:
                    new_reward -= 1
                    new_var_counts[var] -= 1
                if var in new_var_counts and new_var_counts[var] == 1:
                    del new_var_counts[var]

            for var in var_old_b:
                if var in new_var_counts and new_var_counts[var] > 1:
                    new_reward -= 1
                    new_var_counts[var] -= 1
                if var in new_var_counts and new_var_counts[var] == 1:
                    del new_var_counts[var]

            for var in var_new_a:
                if var in new_var_counts:
                    new_reward += 1
                    new_var_counts[var] += 1
                else:
                    new_var_counts[var] = 1

            for var in var_new_b:
                if var in new_var_counts:
                    new_reward += 1
                    new_var_counts[var] += 1
                else:
                    new_var_counts[var] = 1

            # print()
            # print((a, b), m, new_m)
            # new_cost = -1 * evaluate_reward(new_m, h, example, au_table)
            new_cost = -1 * new_reward
            # print("NEW VARCOUNTS", new_var_counts)
            # print("COMPARISON")
            # print(-1 * new_reward, new_cost)
            # print()
            # print("NODE COST", new_node.cost())
            yield Node(new_m, node, ('swap', a, b), new_cost, (h, example,
                                                               unassigned,
                                                               au_table,
                                                               const_count,
                                                               new_var_counts))

        for a in range(len(h)):
            for ua in unassigned:
                new_var_counts = {a: var_counts[a] for a in var_counts}
                new_m = tuple(ua if i == a else v for i, v in enumerate(m))
                new_unassigned = [m[a] if x == ua else x for x in unassigned]

                old_a = au_table[frozenset([h[a], example[m[a]]])]
                new_a = au_table[frozenset([h[a], example[new_m[a]]])]

                var_old_a = {}
                if isinstance(old_a, frozenset):
                    c_old_a = 0
                else:
                    c_old_a = count_term(old_a, var_old_a)
                var_new_a = {}
                if isinstance(new_a, frozenset):
                    c_new_a = 0
                else:
                    c_new_a = count_term(new_a, var_new_a)

                new_reward = -1 * node.cost()
                new_reward -= c_old_a
                new_reward += c_new_a

                for var in var_old_a:
                    if var in new_var_counts and new_var_counts[var] > 1:
                        new_reward -= 1
                        new_var_counts[var] -= 1
                    if var in new_var_counts and new_var_counts[var] == 1:
                        del new_var_counts[var]

                for var in var_new_a:
                    if var in new_var_counts:
                        new_reward += 1
                        new_var_counts[var] += 1
                    else:
                        new_var_counts[var] = 1

                new_cost = -1 * new_reward

                # new_cost = -1 * evaluate_reward(new_m, h, example, au_table)

                yield Node(new_m, node, ('swap unassigned', a, ua),
                           new_cost, (h, example, new_unassigned, au_table,
                                      const_count, var_counts))


class AntiUnifyProblem(Problem):

    def count_terms(self, literal):
        if isinstance(literal, tuple):
            c = sum([self.count_terms(ele) if isinstance(ele, tuple) else 1
                     for ele in literal])
        else:
            c = 1
        return c

    def antiunify_cost(self, h, prev_vars):
        c = 0
        for ele in h:
            if isinstance(ele, frozenset):
                c += max([self.count_terms(t) for t in ele])
            elif isinstance(ele, tuple):
                c += self.antiunify_cost(ele, prev_vars)
        return c

    def get_vars(self, literal):
        vs = set()
        for ele in literal:
            if isinstance(ele, frozenset):
                vs.add(ele)
            elif isinstance(ele, tuple):
                vs.update(self.get_vars(ele))
        return vs

    def possible_mismatches(self, node):
        # h = node.state
        old_h_literals, e_literals, au_table, prev_vars = node.extra

        ohl_counts = {}
        el_counts = {}

        for ohl in old_h_literals:
            if isinstance(ohl, tuple):
                if ohl[0] not in ohl_counts:
                    ohl_counts[ohl[0]] = 0
                ohl_counts[ohl[0]] += 1
            else:
                if ohl not in ohl_counts:
                    ohl_counts[ohl] = 0
                ohl_counts[ohl] += 1

        for el in e_literals:
            if isinstance(el, tuple):
                if el[0] not in el_counts:
                    el_counts[el[0]] = 0
                el_counts[el[0]] += 1
            else:
                if el not in el_counts:
                    el_counts[el] = 0
                el_counts[el] += 1

        c = 0
        for x in set(ohl_counts).union(set(el_counts)):
            if x in ohl_counts and x not in el_counts:
                c += ohl_counts[x]
            elif x in el_counts and x not in ohl_counts:
                c += el_counts[x]
            else:
                c += abs(el_counts[x] - ohl_counts[x])

        return c

    def count_possible_elements(self, node):
        old_h_literals, e_literals, au_table, prev_vars = node.extra
        oh_count = 0
        for ele in old_h_literals:
            oh_count += self.count_elements(ele, set())
        e_count = 0
        for ele in e_literals:
            e_count += self.count_elements(ele, set())
        # print(node.state, max(oh_count, e_count))
        return max(oh_count, e_count)

    def count_elements(self, l, prev_vars):
        c = 0
        for ele in l:
            if isinstance(ele, tuple):
                c += self.count_elements(ele, prev_vars)
            elif isinstance(ele, frozenset) and ele not in prev_vars:
                pass
            else:
                c += 1
        return c

    def node_value(self, node):
        """
        This is the value being minimized.
        """
        return node.cost() - self.count_possible_elements(node)

    def successors(self, node):
        h = node.state
        old_h_literals, e_literals, au_table, prev_vars = node.extra

        for ohl in old_h_literals:
            for el in e_literals:
                new_l = au_table[frozenset([ohl, el])]
                if isinstance(new_l, frozenset):
                    continue

                ele_count = self.count_elements(new_l, prev_vars)
                # au_cost = self.antiunify_cost(new_l, prev_vars)
                new_vars = self.get_vars(new_l)

                yield Node(h.union([new_l]), node, ('antiunify', ohl, el),
                           node.cost() - ele_count,
                           (old_h_literals.difference([ohl]),
                            e_literals.difference([el]),
                            au_table,
                            prev_vars.union(new_vars)))

    def goal_test(self, node):
        # h = node.state
        old_h_literals, e_literals, au_table, prev_vars = node.extra
        return len(old_h_literals) == 0 or len(e_literals) == 0


class GeneralizationProblem(Problem):

    def successors(self, node):
        h = node.state
        args, pos, pos_mapping, gensym = node.extra

        print("H", h)

        # remove literals
        for literal in h:
            removable = True
            for ele in literal[1:]:
                if not is_variable(ele):
                    removable = False
                    break
                if (ele in args or count_occurances(ele, h) > 1):
                    removable = False
                    break

            if removable:
                new_h = frozenset(x for x in h if x != literal)
                # yield Node(new_h, node, ('remove', literal), node.cost()+1,
                #            node.extra)

        # replace constants with variables.
        for literal in h:
            for new_l in get_variablizations(literal, gensym):
                new_h = frozenset([x if x != literal else new_l for
                                   x in h])
                yield Node(new_h, node, ('variablize', literal, new_l),
                           node.cost()+1, node.extra)

        # replace instances of repeating variable with new variable

    def goal_test(self, node):
        h = node.state
        args, pos, pos_mapping, gensym = node.extra
        return covers(h, pos, pos_mapping)


class SpecializationProblem(Problem):

    def successors(self, node):
        h = node.state
        # print("EXPANDING H", h)
        args, constraints, pset, neg, neg_mapping, gensym = node.extra

        all_args = set(s for x in h.union(constraints) for s in
                       extract_strings(x) if is_variable(s))

        if len(pset) == 0:
            return

        p, pm = choice(pset)
        p_index = build_index(p)

        operator = Operator(tuple(('Rule',) + tuple(all_args)),
                            h.union(constraints), [])

        # operator = Operator(tuple(('Rule',) + args), h, [])

        found = False
        for m in operator.match(p_index, initial_mapping=pm):
            reverse_m = {m[a]: a for a in m}
            pos_partial = set([rename(reverse_m, x) for x in p])
            found = True
            break

        if not found:
            return

        n_index = build_index(neg)
        found = False
        for nm in operator.match(n_index, initial_mapping=neg_mapping):
            # print(nm)
            reverse_nm = {nm[a]: a for a in nm}
            neg_partial = set([rename(reverse_nm, x) for x in neg])
            found = True
            break

        if not found:
            return

        unique_pos = pos_partial - neg_partial
        unique_neg = neg_partial - pos_partial

        # print("UNIQUE POS", unique_pos)
        # print("UNIQUE NEG", unique_neg)

        # Yield all minimum specializations of current vars
        for a in m:
            # TODO make sure m[a] is a minimum specialization
            sub_m = {a: m[a]}
            new_h = frozenset([subst(sub_m, ele) for ele in h])
            # print("SPECIALIZATION", new_h, sub_m)
            # print()
            yield Node(new_h, node, ('specializing', (a, m[a])),
                       node.cost()+1, node.extra)

        # Add Negations for all neg specializations
        # for a in nm:
        #     sub_nm = {a: nm[a]}
        #     new_nh = set()
        #     for ele in h:
        #         new = subst(sub_nm, ele)
        #         if new != ele and new not in h:
        #             new_nh.add(('not', new))
        #     new_h = h.union(new_nh)
        #     print("NEGATION SPECIALIZATION", new_nh)
        #     yield Node(new_h, node, ('negation specialization', (a, nm[a])),
        #                node.cost()+1, node.extra)

        # if current vars then add all relations that include current vars
        if len(all_args) > 0:
            added = set()
            for literal in unique_pos:
                if literal in h or literal in constraints:
                    continue
                args = set(s for s in extract_strings(literal) if
                           is_variable(s))
                if len(args.intersection(all_args)) > 0:
                    key = (literal[0],) + tuple(ele if is_variable(ele) else
                                                '?' for ele in literal[1:])
                    if key in added:
                        continue
                    added.add(key)

                    literal = generalize_literal(literal, gensym)
                    new_h = h.union(frozenset([literal]))
                    # print("ADD CURRENT", new_h)
                    # print()
                    yield Node(new_h, node, ('adding current', literal),
                               node.cost()+1, node.extra)

        else:
            added = set()
            for literal in unique_pos:
                if literal in h or literal in constraints:
                    continue
                if literal[0] in added:
                    continue
                added.add(literal[0])
                literal = generalize_literal(literal, gensym)
                new_h = h.union(frozenset([literal]))
                # print("ADD NEW", new_h)
                # print()
                yield Node(new_h, node, ('adding', literal),
                           node.cost()+1, node.extra)

        ### CODE FOR ADDING NEGATIONS ###
        # # Add negation of all neg partial literals not in pos partial.
        # # print(pos_partial)
        # # print(neg_partial)
        # print('NEG PARTIAL LITERAL', neg_partial - pos_partial)
        # print('NEG LITERALS', set(neg) - set(p))
        # for literal in (set(neg)-set(p)).union(neg_partial - pos_partial):
        #     new_h = h.union(frozenset([('not', literal)]))
        #     yield Node(new_h, node, ('adding negated', ('not', literal)),
        #                node.cost()+1, node.extra)

    def goal_test(self, node):
        h = node.state
        args, constraints, pset, neg, neg_mapping, gensym = node.extra
        return not covers(h.union(constraints), neg, neg_mapping)


class IncrementalSpecificToGeneral(object):

    def __init__(self, args=None, constraints=None):
        """
        A relational learner that searches in a specific to general fashion.
        Currently this uses local search to perfrom the anti-unification. It
        isn't guranteed to yield the best antiunification, but hey, it does
        pretty good.

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
        self.h = None

    def get_hset(self):
        """
        Gets a list of hypotheses. This is essentially a disjunction of
        conjunctions. Each hypothesis can be fed into a pattern matcher to
        perform matching.
        """
        if self.h is None:
            return []
        return [self.h.union(self.constraints)]

    def ifit(self, t, x, y):
        """
        Incrementally specializes the hypothesis set. When a positive example
        is encountered that is not covered, then it utilizes antiunification to
        find the least general generalization (lgg). Note, this ignores
        negative examples.
        """
        mapping = {a: t[i] for i, a in enumerate(self.args)}
        reverse_mapping = {t[i]: a for i, a in enumerate(self.args)}
        renamed_x = set([rename(reverse_mapping, ele) for ele in x])

        if y == 1:
            if self.h is None:
                self.h = renamed_x
            elif not covers(self.h.union(self.constraints), x, mapping):
                self.h = generalize(self.h, renamed_x)

        elif y != 0:
            raise Exception("y must be 0 or 1")


class IncrementalGeneralToSpecific(object):

    def __init__(self, args=None, constraints=None):
        """
        A relational learner that searches in a general to specific fashion.
        I try to limit the specialization to keep things tractable...

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
        self.hset = set([frozenset([])])
        self.gen_counter = 0

    def get_hset(self):
        """
        Gets a list of hypotheses. This is essentially a disjunction of
        conjunctions. Each hypothesis can be fed into a pattern matcher to
        perform matching.
        """
        return [h.union(self.constraints) for h in self.hset]

    def ifit(self, t, x, y):
        """
        Incrementally specializes the hypothesis set.
        """
        mapping = {a: t[i] for i, a in enumerate(self.args)}

        if y == 1:
            self.pset.append((x, mapping))
            bad_h = set([h for h in self.hset
                         if not covers(h.union(self.constraints), x, mapping)])
            # print("POS BAD", bad_h)
            self.hset -= bad_h

        elif y == 0:
            bad_h = set([h for h in self.hset
                         if covers(h.union(self.constraints), x, mapping)])
            # print("NEG BAD", bad_h)
            for h in bad_h:
                self.hset.remove(h)
                gset = specialize(h, self.constraints, self.args, self.pset, x,
                                  mapping, lambda: self.gensym())
                for p, pm in self.pset:
                    bad_g = set([g for g in gset if not
                                 covers(g.union(self.constraints), p, pm)])
                    gset -= bad_g
                # print("WORKABLE GSET", gset)
                self.hset.update(gset)

            self.remove_subsumed()

            # impose a limit on the number of hypotheses
            self.hset = set(list(self.hset)[:10])

        else:
            raise Exception("y must be 0 or 1")

    def remove_subsumed(self):
        """
        Removes hypotheses from the hset that are generalizations of other
        hypotheses in hset.
        """
        bad_h = set()
        hset = list(self.hset)
        for i, h in enumerate(hset):
            if h in bad_h:
                continue
            for g in hset[i+1:]:
                if g in bad_h:
                    continue

                rename_negation = {'not': "--NOT--"}
                rh = frozenset(rename(rename_negation, ele) for ele in h)
                rg = frozenset(rename(rename_negation, ele) for ele in g)

                h_specializes_g = self.is_specialization(rh, rg)
                g_specializes_h = self.is_specialization(rg, rh)

                if h_specializes_g and g_specializes_h:
                    print(h, 'equals', g)
                    if len(h) < len(g):
                        bad_h.add(g)
                    else:
                        bad_h.add(h)

                elif h_specializes_g:
                    bad_h.add(h)
                elif g_specializes_h:
                    bad_h.add(g)

        self.hset -= bad_h

    def is_specialization(self, s, h):
        """
        Takes two hypotheses s and g and returns True if s is a specialization
        of h. Note, it returns False if s and h are equal (s is not a
        specialization in this case).
        """
        if s == h:
            return False

        # remove vars, so the unification isn't going in both directions.
        s = set(remove_vars(l) for l in s)

        # check if h matches s (then s specializes h)
        index = build_index(s)
        operator = Operator(tuple(['Rule']), h, [])
        for m in operator.match(index):
            return True
        return False

    def gensym(self):
        self.gen_counter += 1
        return '?new_gen%i' % self.gen_counter


class IncrementalHillClimbing(object):

    def __init__(self, k=10, args=None, constraints=None):
        if args is None:
            args = tuple([])

        if constraints is None:
            constraints = frozenset([])

        self.k = k
        self.args = args
        self.constraints = constraints
        # self.h = frozenset([])
        self.h = None
        self.kset = []
        self.last_pos = None
        self.gen_counter = 0

    def gensym(self):
        self.gen_counter += 1
        return '?new_gen%i' % self.gen_counter

    def get_hset(self):
        if self.h is None:
            return []

        return [self.h.union(self.constraints)]

    def score(self, h):
        """
        Currently using the simple accuracy measure from page 41 of Langley's
        ML book.
        """
        correct = 0

        for mapping, x, y in self.kset:
            if (((y == 1 and covers(h, x, mapping)) or
                 (y == 0 and not covers(h, x, mapping)))):
                correct += 1

        return correct / len(self.kset)

    def ifit(self, t, x, y):
        mapping = {a: t[i] for i, a in enumerate(self.args)}

        if self.h is None and y == 1:
            rm = {t[i]: a for i, a in enumerate(self.args)}
            self.h = set([rename(rm, l) for l in x])

        self.kset.append((mapping, x, y))
        while len(self.kset) > self.k:
            self.kset.pop(0)

        hset = None

        if y == 1:
            if not covers(self.h.union(self.constraints), x, mapping):
                reverse_mapping = {t[i]: a for i, a in enumerate(self.args)}
                renamed_x = set([rename(reverse_mapping, ele) for ele in x])
                hset = set([generalize(self.h, renamed_x)])
            self.last_pos = (x, mapping)
        else:
            if covers(self.h.union(self.constraints), x, mapping):
                hset = specialize(self.h, self.constraints, self.args,
                                  [self.last_pos], x,
                                  mapping, lambda: self.gensym())

        if hset is None or len(hset) == 0:
            return

        hset = [(self.score(new_h.union(self.constraints)), random(), new_h)
                for new_h in hset]
        curr_score = self.score(self.h.union(self.constraints))
        hset.append((curr_score, random(), self.h))
        hset.sort(reverse=True)
        print(hset)

        new_score, _, new_h = hset[0]
        self.h = new_h


if __name__ == "__main__":

    # Cell example from page 46 of pat's ML book.
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

    p1 = {('nuclei', '1')}
    n1 = {('nuclei', '2')}
    p2 = {('nuclei', '3')}
    p3 = {('walls', 'thin')}

    X = [p1, n1, p2, p2, n1, p3]
    y = [1, 0, 1, 1, 0, 1]

    # X = [{('tails=', 'c', 'd'),
    #       ('tails=', 'd', 'c'),
    #       ('nuclei>', 'c', 'd'),
    #       ('shade>', 'c', 'd')}]
    # y = [1]

    # IGS = IncrementalSpecificToGeneral()

    # IGS.h = frozenset([('tails=', '?x', '?y'), ('tails=', '?y', '?x'),
    #                    ('nuclei>', '?y', '?x'), ('shade>', '?x', '?y')])
    # print(IGS.h)

    # IGS = IncrementalGeneralToSpecific()

    IGS = IncrementalHillClimbing()

    for i, x in enumerate(X):
        print("Adding the following instance (%i):" % y[i])
        pprint(x)
        IGS.ifit(tuple([]), x, y[i])
        print("Resulting hset")
        print(IGS.get_hset())
        print(len(IGS.get_hset()))

    # x = frozenset({('color', 'dark')})
    # y = frozenset({('wall', 'thin'), ('nuclei', '?gensym304'),
    #                ('tails', '2'), ('color', 'dark')})

    # print(IGS.is_specialization(x,y))
    # print(IGS.is_specialization(y,x))

    # IGS.hset = set([x, y])
    # IGS.remove_subsumed()
    # print(IGS.hset)

    # p1 = {('on', 'A', 'B'),
    #       ('on', 'B', 'C'),
    #       ('on', 'C', 'D'),
    #       ('on', 'D', 'E')}
    # p2 = {('on', 'X', 'Y'),
    #       ('on', 'Y', 'Z'),
    #       ('on', 'Z', 'Q')}

    # # p1 = {('on', '?o1', '?o2', True),
    # #       ('name', '?o1', "Block 1"),
    # #       ('name', '?o2', "Block 7"),
    # #       ('valuea', '?o2', 99)}

    # # p2 = {('on', '?o3', '?o4', True),
    # #       ('name', '?o3', "Block 3"),
    # #       ('name', '?o4', "Block 4"),
    # #       ('valuea', '?o4', 2)}

    # au_table = build_antiunify_table(p1, p2)

    # AUP = AntiUnifyProblem(frozenset(), extra=(frozenset(p1), frozenset(p2),
    #                                            au_table,
    #                                            frozenset()))

    # compare_searches(problems=[AUP],
    #                  searches=[breadth_first_search,
    #                            best_first_search])

    # for sol in best_first_search(AUP):
    #     print(sol.state)
    #     print(variablize_hypothesis(sol.state))
    #     print(AUP.node_value(sol))
    #     print()

    # print(AUP.antiunify(('add', ('name', 'cell3'), '2'), ('add', ('value',
    #                                                               'cell1'), '3')))

    print()
    print()
    print()
    print()

    print(generalize(frozenset([('name', '?x', 'Block1'),
                                ('not', ('value', '?x', 4))]),
                     frozenset([('name', 'b2', 'Block1'),
                                ('name', 'b3', 'Block1'),
                                ('value', 'b2', 2),
                                ('value', 'b3', 4)])))
