from pprint import pprint
from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.cobweb import CobwebNode
from concept_formation.preprocessor import NameStandardizer
from concept_formation.structure_mapper import StructureMapper
from concept_formation.structure_mapper import rename_flat
from concept_formation.structure_mapper import contains_component
from learners.IncrementalHeuristic import IncrementalHeuristic
from planners.fo_planner import Operator
from planners.fo_planner import build_index
from planners.fo_planner import subst

global my_gensym_counter
my_gensym_counter = 0


def ground(arg):
    if isinstance(arg, tuple):
        return tuple(ground(e) for e in arg)
    elif isinstance(arg, str):
        return arg.replace('?', 'QM')
    else:
        return arg


def unground(arg):
    if isinstance(arg, tuple):
        return tuple(unground(e) for e in arg)
    elif isinstance(arg, str):
        return arg.replace('QM', '?')
    else:
        return arg


class Counter(object):

    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1


class BaseILP(object):

    def __init__(self, args, G):
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


def value_gensym():
    """
    Generates unique names for naming renaming apart objects.

    :return: a unique object name
    :rtype: '?val'+counter
    """
    global my_gensym_counter
    my_gensym_counter += 1
    return '?val' + str(my_gensym_counter)


# def gensym():
#     """
#     Generates unique names for naming renaming apart objects.
#
#     :return: a unique object name
#     :rtype: '?o'+counter
#     """
#     global my_gensym_counter
#     my_gensym_counter += 1
#     return '?o' + str(my_gensym_counter)


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
    This learner always returns the tuples it was trained with, after ensuring
    they meet any provided constraints.
    """
    def __init__(self, args, constraints=None):
        self.pos_count = 0
        self.neg_count = 0
        self.args = args
        self.tuples = set()

        if constraints is None:
            self.constraints = frozenset()
        else:
            self.constraints = constraints

    def num_pos(self):
        return self.pos_count

    def num_neg(self):
        return self.neg_count

    def __len__(self):
        return self.pos_count + self.neg_count

    def __repr__(self):
        return repr(self.tuples)

    def __str__(self):
        return str(self.tuples)

    def ground_example(self, x):
        grounded = [tuple(list(ground(a)) + [ground(x[a])])
                    if isinstance(a, tuple)
                    else tuple([ground(a), ground(x[a])]) for a in x]
        return grounded

    def check_match(self, t, x):
        # print("CHECK MATCHES T", t)

        t = tuple(ground(ele) for ele in t)

        # Update to include initial args
        mapping = {a: t[i] for i, a in enumerate(self.args)}

        # print("MY MAPPING", mapping)

        # print("CHECKING MATCHES")

        if t not in self.tuples:
            return False

        grounded = self.ground_example(x)
        # grounded = [(ground(a), x[a]) for a in x if (isinstance(a, tuple))]
        # pprint(grounded)
        # pprint(mapping)
        index = build_index(grounded)

        # Update to include initial args
        operator = Operator(tuple(('Rule',) + self.args),
                            frozenset().union(self.constraints), [])
        for m in operator.match(index, initial_mapping=mapping):
            return True
        return False

    def get_matches(self, x, epsilon=0.0):

        # print("GETTING MATCHES")
        # pprint(self.tuples)
        grounded = self.ground_example(x)
        # grounded = [(ground(a), x[a]) for a in x if (isinstance(a, tuple))]
        # print("FACTS")

        # pprint(grounded)

        index = build_index(grounded)

        # print("INDEX")
        # pprint(index)
        print("ARGS")
        print(self.args)

        print("Tuples")
        print(self.tuples) 

        print("Constraints")
        print(self.constraints)        
        # print("OPERATOR")
        # pprint(self.operator)
        # print(self.learner.get_hset())

        for t in self.tuples:
            # print("TUPLE", t)
            mapping = {a: t[i] for i, a in enumerate(self.args)}
            operator = Operator(tuple(('Rule',) + self.args),
                                frozenset().union(self.constraints), [])
            # print("OP", str(operator))

            for m in operator.match(index, epsilon=epsilon,
                                    initial_mapping=mapping):
                print("M", m)
                result = tuple(ele.replace("QM", '?') for ele in t)
                # print('GET MATCHES T', result)
                yield result
                break

        # print("GOT ALL THE MATCHES!")

    def ifit(self, t, x, y):

        if y > 0:
            self.pos_count += 1
        else:
            self.neg_count += 1

        t = tuple(ground(e) for e in t)
        self.tuples.add(t)


class StateResponseLearner(BaseILP):
    """
    This just memorizes pairs of states and matches. Then given a state it
    looks up a match.
    """

    def __init__(self, remove_attrs=None):
        self.states = {}
        self.target_types = []
        self.pos_concept = Counter()
        self.neg_concept = Counter()

        if remove_attrs is None:
            remove_attrs = ['value']
        self.remove_attrs = remove_attrs

    def num_pos(self):
        return self.pos_concept.count()

    def num_neg(self):
        return self.neg_concept.count()

    def ifit(self, t, x, y):
        if y == 0:
            self.neg_concept.increment()
            return
        self.pos_concept.increment()

        x = {a: x[a] for a in x if (isinstance(a, tuple) and a[0] not in
                                    self.remove_attrs) or
                                   (not isinstance(a, tuple) and a not in
                                    self.remove_attrs)}

        x = frozenset(x.items())
        # pprint(x)
        if x not in self.states:
            self.states[x] = set()
        self.states[x].add(tuple(t))

    def get_matches(self, x, epsilon=0.0):
        x = {a: x[a] for a in x if (isinstance(a, tuple) and a[0] not in
                                    self.remove_attrs) or
                                   (not isinstance(a, tuple) and a not in
                                    self.remove_attrs)}
        x = frozenset(x.items())
        if x not in self.states:
            return
        for t in self.states[x]:
            yield t

    def check_match(self, t, x):
        # t = tuple('?' + e for e in t)
        # print("CHECKING", t)
        # print("against", [self.states[a] for a in self.states])

        x = {a: x[a] for a in x if (isinstance(a, tuple) and a[0] not in
                                    self.remove_attrs) or
                                   (not isinstance(a, tuple) and a not in
                                    self.remove_attrs)}
        x = frozenset(x.items())

        # if t in [self.states[a] for a in self.states][0]:
        #     print("KEY")
        #     pprint([a for a in self.states][0])
        #     print("THE X")
        #     pprint(x)

        return x in self.states and t in self.states[x]

    def __len__(self):
        return sum(len(self.states[x]) for x in self.states)

    def fit(self, T, X, y):
        for i, t in enumerate(T):
            self.ifit(T[i], X[i], y[i])


class RelationalLearner(BaseILP):

    def __init__(self, args, constraints=None, remove_attrs=None):
        self.pos_count = 0
        self.neg_count = 0
        self.args = args

        # self.learner = IncrementalGeneralToSpecific(args=self.args,
        #                                             constraints=constraints)
        # self.learner = IncrementalSpecificToGeneral(args=self.args,
        #                                             constraints=constraints)
        self.learner = IncrementalHeuristic(args=self.args,
                                            constraints=constraints)

        if remove_attrs is None:
            remove_attrs = ['value']
        self.remove_attrs = remove_attrs

    def num_pos(self):
        return self.pos_count

    def num_neg(self):
        return self.neg_count

    def __len__(self):
        return self.pos_count + self.neg_count

    def __repr__(self):
        return repr(self.learner.get_hset())

    def __str__(self):
        return str(self.learner.get_hset())

    def ground_example(self, x):
        grounded = [tuple(list(ground(a)) + [ground(x[a])])
                    if isinstance(a, tuple)
                    else tuple([ground(a), ground(x[a])]) for a in x]
        return grounded

    def check_match(self, t, x):
        # print("CHECK MATCHES T", t)

        t = tuple(ele.replace('?', "QM") for ele in t)

        # Update to include initial args
        mapping = {a: t[i] for i, a in enumerate(self.args)}

        # print("MY MAPPING", mapping)

        # print("CHECKING MATCHES")
        grounded = self.ground_example(x)
        # grounded = [(ground(a), x[a]) for a in x if (isinstance(a, tuple))]
        # pprint(grounded)
        # pprint(mapping)
        index = build_index(grounded)

        for h in self.learner.get_hset():
            # Update to include initial args
            operator = Operator(tuple(('Rule',) + self.args), h, [])
            for m in operator.match(index, initial_mapping=mapping):
                return True
        return False

    def get_matches(self, x, epsilon=0.0):

        # print("GETTING MATCHES")
        # pprint(self.learner.get_hset())
        grounded = self.ground_example(x)
        # grounded = [(ground(a), x[a]) for a in x if (isinstance(a, tuple))]
        # print("FACTS")

        # pprint(grounded)

        index = build_index(grounded)

        # print("INDEX")
        # pprint(index)

        # print("OPERATOR")
        # pprint(self.operator)
        # print(self.learner.get_hset())

        for h in self.learner.get_hset():
            # print('h', h)
            # Update to include initial args
            operator = Operator(tuple(('Rule',) + self.args), h, [])
            # print("OPERATOR", h)

            for m in operator.match(index, epsilon=epsilon):
                # print('match', m, operator.name)
                result = tuple([unground(subst(m, ele))
                                for ele in operator.name[1:]])
                # result = tuple(['?' + subst(m, ele)
                #                 for ele in self.operator.name[1:]])
                # result = tuple(['?' + m[e] for e in self.target_types])
                # print('GET MATCHES T', result)
                yield result

        # print("GOT ALL THE MATCHES!")

    def ifit(self, t, x, y):

        if y == 1:
            self.pos_count += 1
        else:
            self.neg_count += 1

        t = tuple(ground(e) for e in t)
        grounded = self.ground_example(x)

        # print("BEFORE IFIT %i" % y)
        # print(self.learner.get_hset())
        self.learner.ifit(t, grounded, y)
        # print("AFTER IFIT")
        # print(self.learner.get_hset())


class SpecificToGeneral(BaseILP):

    def __init__(self, args=None, constraints=None, remove_attrs=None):
        self.pos = set()
        self.operator = None
        self.concept = Cobweb3Node()
        self.pos_concept = CobwebNode()
        self.neg_concept = CobwebNode()

        if remove_attrs is None:
            remove_attrs = ['value']
        self.remove_attrs = remove_attrs

    def num_pos(self):
        return self.pos_concept.count

    def num_neg(self):
        return self.neg_concept.count

    def __len__(self):
        return self.count

    # def __repr__(self):
    #     return repr(self.operator)

    def check_match(self, t, x):
        """
        if y is predicted to be 1 then returns True
        else returns False
        """
        # print("CHECK MATCHES T", t)

        if self.operator is None:
            return

        t = tuple(ele.replace('?', "QM") for ele in t)

        mapping = {a: t[i] for i, a in enumerate(self.operator.name[1:])}

        # print("MY MAPPING", mapping)

        # print("CHECKING MATCHES")
        grounded = [(ground(a), x[a]) for a in x if (isinstance(a, tuple))]
        # pprint(grounded)
        # pprint(mapping)
        index = build_index(grounded)

        for m in self.operator.match(index, initial_mapping=mapping):
            return True
        return False

    def get_matches(self, x, constraints=None, epsilon=0.0):
        if self.operator is None:
            return

        # print("GETTING MATCHES")
        grounded = [(ground(a), x[a]) for a in x if (isinstance(a, tuple))]
        # print("FACTS")
        # pprint(grounded)
        index = build_index(grounded)

        # print("INDEX")
        # pprint(index)

        # print("OPERATOR")
        # pprint(self.operator)

        for m in self.operator.match(index, epsilon=epsilon):
            # print('match', m, self.operator.name)
            result = tuple([unground(subst(m, ele))
                            for ele in self.operator.name[1:]])
            # result = tuple(['?' + subst(m, ele)
            #                 for ele in self.operator.name[1:]])
            # result = tuple(['?' + m[e] for e in self.target_types])
            # print('GET MATCHES T', result)
            yield result

        # print("GOT ALL THE MATCHES!")

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
        # print('REMOVING: ', attr, value)
        return False

    def ifit(self, t, x, y):
        # print("IFIT T", t)
        # if y == 0:
        #     return

        x = {a: x[a] for a in x if (isinstance(a, tuple) and a[0] not in
                                    self.remove_attrs) or
                                   (not isinstance(a, tuple) and a not in
                                    self.remove_attrs)}

        # x = {a: x[a] for a in x if self.is_structural_feature(a, x[a])}
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

        if y > 0:
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
            attr_count = 0
            for val in self.pos_concept.av_counts[attr]:
                attr_count += self.pos_concept.av_counts[attr][val]
            if attr_count == self.pos_concept.count:
                if len(self.pos_concept.av_counts[attr]) == 1:
                    args = get_vars(attr)
                    pos_args.update(args)
                    pos_instance[attr] = val
                else:
                    args = get_vars(attr)
                    val_gensym = value_gensym()
                    args.append(val_gensym)
                    pos_instance[attr] = val_gensym

            # if len(self.pos_concept.av_counts[attr]) == 1:
            #     for val in self.pos_concept.av_counts[attr]:
            #         if ((self.pos_concept.av_counts[attr][val] ==
            #              self.pos_concept.count)):
            #             args = get_vars(attr)
            #             pos_args.update(args)
            #             pos_instance[attr] = val

        # print('POS ARGS', pos_args)

        neg_instance = {}
        for attr in self.neg_concept.av_counts:
            # print("ATTR", attr)
            args = set(get_vars(attr))
            if not args.issubset(pos_args):
                continue

            for val in self.neg_concept.av_counts[attr]:
                # print("VAL", val)
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


def get_where_learner(name):
    return WHERE_LEARNERS[name.lower().replace(' ', '').replace('_', '')]


WHERE_LEARNERS = {
    'mostspecific': MostSpecific,
    'stateresponselearner': StateResponseLearner,
    'relationallearner': RelationalLearner,
    'specifictogeneral': SpecificToGeneral
}


if __name__ == "__main__":

    ssw = RelationalLearner(args=('?foa0',),
                            constraints=frozenset([('name', '?foa0',
                                                  '?foa0val')]))

    p1 = {('on', '?o1'): '?o2',
          ('name', '?o1'): "Block 1",
          ('name', '?o2'): "Block 7",
          ('valuea', '?o2'): 99}

    p2 = {('on', '?o3'): '?o4',
          ('name', '?o3'): "Block 3",
          ('name', '?o4'): "Block 4",
          ('valuea', '?o4'): 2}

    n1 = {('on', '?o5'): '?o6',
          ('on', '?o6'): '?o7',
          ('name', '?o5'): "Block 5",
          ('name', '?o6'): "Block 6",
          ('name', '?o7'): "Block 7",
          ('valuea', '?o7'): 100}

    ssw.ifit(['?o1'], p1, 1)
    ssw.ifit(['?o5'], n1, 0)
    ssw.ifit(['?o3'], p2, 1)

    test1 = {('on', '?o5'): '?o6',
             ('on', '?o6'): '?o7',
             ('name', '?o5'): "Block 1",
             ('name', '?o6'): "Block 7",
             ('name', '?o7'): "Block 5",
             ('valuea', '?o6'): 3}

    # print()
    # print("FINAL HYPOTHESIS")
    # print(ssw.learner.get_hset())

    # for m in ssw.get_matches(test1):
    #    print('OP MATCH', m)
