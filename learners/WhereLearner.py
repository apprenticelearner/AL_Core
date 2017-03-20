from pprint import pprint

from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.cobweb import CobwebNode
from concept_formation.preprocessor import NameStandardizer
from concept_formation.structure_mapper import StructureMapper
from concept_formation.structure_mapper import rename_flat
from concept_formation.structure_mapper import contains_component
from planners.fo_planner import Operator
from planners.fo_planner import build_index
from planners.fo_planner import subst


global my_gensym_counter
my_gensym_counter = 0


class Counter(object):

    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1


class BaseILP(object):

    def __init__(self):
        pass

    def check_match(self, t, x):
        pass

    def get_match(self, X):
        pass

    def get_all_matches(self, X):
        pass

    def ifit(self, X, y):
        pass

    def fit(self, X, y):
        """
        Assume that X is a list of dictionaries that represent FOL using
        TRESTLE notation.

        y is a vector of 0's or 1's that specify whether the examples are
        positive or negative.
        """
        pass


def gensym():
    """
    Generates unique names for naming renaming apart objects.

    :return: a unique object name
    :rtype: '?o'+counter
    """
    global my_gensym_counter
    my_gensym_counter += 1
    return '?o' + str(my_gensym_counter)


def get_vars(arg):
    if isinstance(arg, tuple):
        l = []
        for e in arg:
            for v in get_vars(e):
                if v not in l:
                    l.append(v)
        return l
    elif isinstance(arg, str) and len(arg) > 0 and arg[0] == '?':
        return [arg]
    else:
        return []


class MostSpecific(BaseILP):
    """
    This just memorizes pairs of states and matches. Then given a state it
    looks up a match.
    """

    def __init__(self, remove_values=True):
        self.remove_values = remove_values
        self.states = {}
        self.target_types = []
        self.pos_concept = Counter()
        self.neg_concept = Counter()

    def ifit(self, t, x, y):
        if y == 0:
            self.neg_concept.increment()
            return
        self.pos_concept.increment()

        if self.remove_values:
            # x = {a: "EMPTY" if x[a] == "" 
            #      else str(type(x[a])) for a in x}
            x = {a: str(type(x[a])) for a in x}
        x = frozenset(x.items())
        pprint(x)
        if x not in self.states:
            self.states[x] = set()
        self.states[x].add(tuple(t))

    def get_matches(self, x, epsilon=0.0):
        if self.remove_values:
            x = {a: str(type(x[a])) for a in x}
        x = frozenset(x.items())
        if x not in self.states:
            return
        for t in self.states[x]:
            yield t

    def check_match(self, t, x):
        if self.remove_values:
            x = {a: str(type(x[a])) for a in x}
        x = frozenset(x.items())
        return x in self.states and t in self.states[x]

    def __len__(self):
        return sum(len(self.states[x]) for x in self.states)

    def fit(self, T, X, y):
        for i, t in enumerate(T):
            self.ifit(T[i], X[i], y[i])


class SpecificToGeneral(BaseILP):

    def __init__(self):
        self.pos = set()
        self.operator = None
        self.concept = Cobweb3Node()
        self.pos_concept = CobwebNode()
        self.neg_concept = CobwebNode()

    def __len__(self):
        return self.count

    # def __repr__(self):
    #     return repr(self.operator)

    def check_match(self, t, x):
        """
        if y is predicted to be 1 then returns True
        else returns False
        """
        if self.operator is None:
            return

        mapping = {a: t[i] for i, a in enumerate(self.operator.name[1:])}

        print("MY MAPPING", mapping)

        # print("GETTING MATCHES")
        grounded = [(ground(a), x[a]) for a in x if (isinstance(a, tuple))]
        index = build_index(grounded)

        for m in self.operator.match(index, initial_mapping=mapping):
            return True
        return False

    def get_matches(self, x, constraints=None, epsilon=0.0):
        if self.operator is None:
            return

        print("GETTING MATCHES")
        grounded = [(ground(a), x[a]) for a in x if (isinstance(a, tuple))]
        # print("FACTS")
        # pprint(grounded)
        index = build_index(grounded)

        # print("INDEX")
        # pprint(index)

        # print("OPERATOR")
        # pprint(self.operator)

        for m in self.operator.match(index, epsilon=epsilon):
            print('match', m, self.operator.name)
            result = tuple(['?' + subst(m, ele)
                            for ele in self.operator.name[1:]])
            # result = tuple(['?' + m[e] for e in self.target_types])
            print(result)
            yield result

        print("GOT ALL THE MATCHES!")

        # for t in self.pos:
        #     print(t)
        #     yield t
        # X = {a: X[a] for a in X if not isinstance(a, tuple)
        #      or a[0] != 'value'}
        # pprint(X)
        # print('inferring match')
        # temp = self.tree.infer_missing(X)
        # print('done')

        # foas = []
        # for attr in temp:
        #     if isinstance(attr, tuple) and 'foa' in attr[0]:
        #         foas.append((float(attr[0][3:]), attr[1]))
        # foas.sort()
        # return tuple([e for _,e in foas])

    def matches(self, eles, attribute):
        for e in eles:
            if contains_component(e, attribute):
                return True
        return False

    def is_structural_feature(self, attr, value):
        if not isinstance(attr, tuple) or attr[0] != 'value':
            return True
        print('REMOVING: ', attr, value)
        return False

    def ifit(self, t, x, y):
        # if y == 0:
        #     return

        x = {a: x[a] for a in x if self.is_structural_feature(a, x[a])}
        # x = {a: x[a] for a in x}

        # eles = set([field for field in t])
        # prior_count = 0
        # while len(eles) - prior_count > 0:
        #     prior_count = len(eles)
        #     for a in x:
        #         if isinstance(a, tuple) and a[0] == 'haselement':
        #             if a[2] in eles:
        #                 eles.add(a[1])
        #             # if self.matches(eles, a):
        #             #     names = get_attribute_components(a)
        #             #     eles.update(names)

        # x = {a: x[a] for a in x
        #      if self.matches(eles, a)}

        # foa_mapping = {field: 'foa%s' % j for j, field in enumerate(t)}
        foa_mapping = {}
        for j, field in enumerate(t):
            if field not in foa_mapping:
                foa_mapping[field] = 'foa%s' % j

        # for j,field in enumerate(t):
        #     x[('foa%s' % j, field)] = True
        x = rename_flat(x, foa_mapping)
        # pprint(x)

        # print("adding:")

        ns = NameStandardizer()
        sm = StructureMapper(self.concept)
        x = sm.transform(ns.transform(x))
        # pprint(x)
        self.concept.increment_counts(x)

        if y == 1:
            self.pos_concept.increment_counts(x)
        else:
            self.neg_concept.increment_counts(x)

        # print()
        # print('POSITIVE')
        # pprint(self.pos_concept.av_counts)
        # print('NEGATIVE')
        # pprint(self.neg_concept.av_counts)

        # pprint(self.concept.av_counts)

        pos_instance = {}
        pos_args = set()
        for attr in self.pos_concept.av_counts:
            if len(self.pos_concept.av_counts[attr]) == 1:
                for val in self.pos_concept.av_counts[attr]:
                    if ((self.pos_concept.av_counts[attr][val] ==
                         self.pos_concept.count)):

                        args = get_vars(attr)
                        pos_args.update(args)
                        pos_instance[attr] = val

        # print('POS ARGS', pos_args)

        neg_instance = {}
        for attr in self.neg_concept.av_counts:
            args = set(get_vars(attr))
            if not args.issubset(pos_args):
                continue

            for val in self.neg_concept.av_counts[attr]:
                if ((attr not in self.pos_concept.av_counts or val not in
                     self.pos_concept.av_counts[attr])):
                    neg_instance[attr] = val

        foa_mapping = {'foa%s' % j: '?foa%s' % j for j in range(len(t))}
        pos_instance = rename_flat(pos_instance, foa_mapping)
        neg_instance = rename_flat(neg_instance, foa_mapping)

        conditions = ([(a, pos_instance[a]) for a in pos_instance] +
                      [('not', (a, neg_instance[a])) for a in neg_instance])

        # print("========CONDITIONS======")
        # pprint(conditions)
        # print("========CONDITIONS======")

        self.target_types = ['?foa%s' % i for i in range(len(t))]
        self.operator = Operator(tuple(['Rule'] + self.target_types),
                                 conditions, [])

    def fit(self, T, X, y):
        self.count = len(X)
        self.target_types = ['?foa%s' % i for i in range(len(T[0]))]

        # ignore X and save the positive T's.
        for i, t in enumerate(T):
            if y[i] == 1:
                self.pos.add(t)

        self.concept = Cobweb3Node()

        for i, t in enumerate(T):
            self.ifit(t, X[i], y[i])


if __name__ == "__main__":

    ssw = SpecificToGeneral()

    p1 = {('on', '?o1', '?o2'): True,
          ('name', '?o1'): "Block 1",
          ('name', '?o2'): "Block 7",
          ('value', '?o2'): 99}

    p2 = {('on', '?o3', '?o4'): True,
          ('name', '?o3'): "Block 3",
          ('name', '?o4'): "Block 4"}

    n1 = {('on', '?o5', '?o6'): True,
          ('on', '?o6', '?o7'): True,
          ('name', '?o5'): "Block 5",
          ('name', '?o6'): "Block 6",
          ('name', '?o7'): "Block 7",
          ('value', '?o7'): 100}

    ssw.ifit(['?o1'], p1, 1)
    ssw.ifit(['?o3'], p2, 1)
    ssw.ifit(['?o5'], n1, 0)

    test1 = {('on', '?o5', '?o6'): True,
             ('on', '?o6', '?o7'): True,
             ('name', '?o5'): "Block 1",
             ('name', '?o6'): "Block 7",
             ('name', '?o7'): "Block 5"}

    for m in ssw.get_matches(test1):
        print('OP MATCH', m)
