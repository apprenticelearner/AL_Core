if(__name__ == "__main__"):
    import sys, os
    sys.path.append("../")
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

    def get_matches(self, X):
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


class WhereLearner(object):

    def __init__(self, learner_name,learner_kwargs={}):
        
        self.learner_name = learner_name
        self.learner_kwargs = learner_kwargs
        self.rhs_by_label = {}
        self.learners = {}

    def add_rhs(self,rhs, constraints):
        # args = [skill.selection_var] + skill.input_vars
        self.learners[rhs] = get_where_agent(self.learner_name,
            args=tuple(rhs.all_vars),constraints=constraints,**self.learner_kwargs)
        
        rhs_list = self.rhs_by_label.get(rhs.label,[])
        rhs_list.append(rhs)
        self.rhs_by_label[rhs.label] = rhs_list

    def check_match(self, rhs, t, x):
        return self.learners[rhs].check_match(t,x)

    def get_match(self,rhs, X):
        # args = [skill.selection_var] + skill.input_vars
        return self.learners[rhs].get_match(X)

    def get_matches(self,rhs, X):
        # args = [skill.selection_var] + skill.input_vars
        return self.learners[rhs].get_matches(X)

    def ifit(self,rhs, t, X, y):
        # args = [skill.selection_var] + skill.input_vars
        return self.learners[rhs].ifit(t,X,y)

    def fit(self,rhs, T, X, y):
        return self.learners[rhs].fit(T,X,y)


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
        # print("ARGS")
        # print(self.args)

        # print("Tuples")
        # print(self.tuples) 

        # print("Constraints")
        # print(self.constraints)        
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
                # print("M", m)
                result = tuple(ele.replace("QM", '?') for ele in t)
                # print('GET MATCHES T', result)
                yield result
                break

        # print("GOT ALL THE MATCHES!")

    def ifit(self, t, x, y):
        # print("TUPIN", t)

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
        # pprint(x)

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

        # print(foa_mapping)



        # for j,field in enumerate(t):
        #     x[('foa%s' % j, field)] = True
        x = rename_flat(x, foa_mapping)
        

        # print("adding:")

        ns = NameStandardizer()
        sm = StructureMapper(self.concept)
        x = sm.transform(ns.transform(x))

        pprint(x)
        # pprint(x)
        self.concept.increment_counts(x)

        if y > 0:
            self.pos_concept.increment_counts(x)
        else:
            self.neg_concept.increment_counts(x)

        print(self.pos_concept)

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

        pprint(pos_instance)
        pprint(pos_args)
        print("av_counts")
        pprint(self.pos_concept.av_counts)
        


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

        print(conditions)

        # print("========CONDITIONS======")
        # pprint(conditions)
        # print("========CONDITIONS======")

        self.target_types = ['?foa%s' % i for i in range(len(t))]
        self.operator = Operator(tuple(['Rule'] + self.target_types),
                                 conditions, [])
        print("_-----------------------------------")

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


def get_where_agent(name,**learner_kwargs):
    return WHERE_LEARNER_AGENTS[name.lower().replace(' ', '').replace('_', '')](**learner_kwargs)

def get_where_learner(name,learner_kwargs={}):
    return WhereLearner(name,learner_kwargs)

WHERE_LEARNER_AGENTS = {
    'mostspecific': MostSpecific,
    'stateresponselearner': StateResponseLearner,
    'relationallearner': RelationalLearner,
    'specifictogeneral': SpecificToGeneral
}




# if __name__ == "__main__":

    # ssw = RelationalLearner(args=('?foa0',),
    #                         constraints=frozenset([('name', '?foa0',
    #                                               '?foa0val')]))
    # ssw = SpecificToGeneral(args=('?foa0',),
    #                         constraints=frozenset([('name', '?foa0',
    #                                               '?foa0val')]))

    # ssw = MostSpecific(args=('?foa0',),
    #                         constraints=frozenset([('name', '?foa0',
    #                                               '?foa0val')]))
    # p1 = {('on', '?o1'): '?o2',
    #       ('name', '?o1'): "Block 1",
    #       ('name', '?o2'): "Block 7",
    #       ('valuea', '?o2'): 99}

    # p2 = {('on', '?o3'): '?o4',
    #       ('name', '?o3'): "Block 3",
    #       ('name', '?o4'): "Block 4",
    #       ('valuea', '?o4'): 2}

    # n1 = {('on', '?o5'): '?o6',
    #       ('on', '?o6'): '?o7',
    #       ('name', '?o5'): "Block 5",
    #       ('name', '?o6'): "Block 6",
    #       ('name', '?o7'): "Block 7",
    #       ('valuea', '?o7'): 100}

    # ssw.ifit(['?o1'], p1, 1)
    # ssw.ifit(['?o5'], n1, 0)
    # ssw.ifit(['?o3'], p2, 1)

    # test1 = {('on', '?o5'): '?o6',
    #          ('on', '?o6'): '?o7',
    #          ('name', '?o5'): "Block 1",
    #          ('name', '?o6'): "Block 7",
    #          ('name', '?o7'): "Block 5",
    #          ('valuea', '?o6'): 3}

    # print()
    # print("FINAL HYPOTHESIS")
    # print(ssw.learner.get_hset())

    # ms = SpecificToGeneral(args=('foa0','foa1'))#,
    #                         # constraints=frozenset([('name', '?foa0',
    #                         #                       '?foa0val')]))

    # state = {'columns' : { 
    #             "0": {
    #                 "0":{"?ele1": {"id": "ele1", "value":7}},
    #                 "1":{"?ele2": {"id": "ele2", "value":8}}
    #             },
    #             "1": {
    #                 "0":{"?ele3": {"id": "ele3", "value":7}},
    #                 "1":{"?ele4": {"id": "ele4", "value":8}}
    #             }
    #         }
    #      }
    # from concept_formation.preprocessor import Flattener
    # from pprint import pprint
    # fl = Flattener()
    # state = fl.transform(state)

    # pprint(state)

    # ms.ifit(['?ele1','?ele2'], state, 1)
    # ms.ifit(['?ele3','?ele4'], state, 1)
    # ms.ifit(['?ele1','?ele3'], state, 0)
    # # ms.ifit(['?o3'], state, 1)

    # for m in ms.get_matches(state):
    #    print('OP MATCH', m)

import torch

def mask_select(x,m):
    return x[m.nonzero().view(-1)]

class VersionSpaceILP(BaseILP):
    def __init__(self):
        self.spec_concepts = None
        self.gen_concepts = None
        self.ZERO = None

    def ifit(self,x,y):

        x = torch.tensor(x,dtype=torch.uint8).view(1,-1)
        print("x", y)
        print(x)
        clen = self.clen = x.shape[-1]
        if(y > 0):
            if(isinstance(self.spec_concepts, type(None))):
                self.spec_concepts = x
                self.gen_concepts = torch.zeros(x.shape,dtype=torch.uint8)
                self.ZERO = torch.tensor(0,dtype=torch.uint8)
            else:
                #Generalize the specific concepts to incorperate the positive example
                self.spec_concepts = torch.where(x != self.spec_concepts, self.ZERO,self.spec_concepts)
                
                #Prune the general concepts that are inconsistent with the positive example
                gen_consistency = (self.gen_concepts == self.ZERO) | (self.gen_concepts == x)
                self.gen_concepts = mask_select(self.gen_concepts,gen_consistency.all(dim=1))
                

        else:
            if(isinstance(self.spec_concepts, type(None))):
                return
            else:
                #### Start: Make the General Concepts more specific #### 
                spec_inconsistency = ~((self.spec_concepts == self.ZERO) | (self.spec_concepts == x))
                gen_consistency = (self.gen_concepts == self.ZERO) | (self.gen_concepts == x)

                #The set of general concepts which are contain the negative example and must be specialized 
                to_specilz_concepts = mask_select(self.gen_concepts,gen_consistency.all(dim=1))
                #The set of general concepts which do not contain the negative example an so we leave alone
                to_keep_concepts = mask_select(self.gen_concepts,(~gen_consistency).any(dim=1))

                #The set of concepts with only one specific element which are present in
                    #the most specific concept but not this negative example.
                specialization_candidates = torch.eye(clen,dtype=torch.uint8).view(1,clen,clen)*(spec_inconsistency*self.spec_concepts).view(-1,1,clen)
                specialization_candidates = mask_select(specialization_candidates.view(-1,clen),spec_inconsistency.view(-1))

                #Specialize each concept (among those we found earlier that contain the negative example)
                #   creating all new possible general concepts for each element which is still general
                sc,tsc, = specialization_candidates.view(1,-1,clen), to_specilz_concepts.view(-1,1,clen)
                possible_specialized_gens = torch.where((tsc == self.ZERO), sc,tsc).view(-1,clen)
                # specialized_gens = torch.where((tsc == self.ZERO) & gen_consistency.view(1,-1,clen), sc,tsc).view(-1,clen)
                
                #If a possible specialization is consistent with the non-specialized general concepts then keep it
                new_gen_consistency = (to_keep_concepts.view(1,-1,clen) == self.ZERO) | (to_keep_concepts.view(1,-1,clen) == possible_specialized_gens.view(-1,1,clen))
                inconsistent_w_keepers = (~new_gen_consistency).any(dim=-1).all(dim=-1) 
                specialized_gens = mask_select(possible_specialized_gens,inconsistent_w_keepers)
                
                #Join the tensors of non-specialized and specialized concepts
                self.gen_concepts = torch.cat([to_keep_concepts, specialized_gens],dim=0)
                
                #### Start: Prune Specific Concepts ####
                #TODO: Never happens? What does it look like if we allow for multiple specific concepts?
                self.spec_concepts = mask_select(self.spec_concepts,spec_inconsistency.any(dim=1))


        print("GENERAL")
        print(self.gen_concepts)
        print("SPECIFIC")
        print(self.spec_concepts,self.spec_concepts.shape)
        
        print("---------------------------------")

    def check_match(self,x):
        '''Returns true if x is consistent with all general concepts and any specific concept'''
        x = torch.tensor(x,dtype=torch.uint8).view(1,-1)
        spec_consistency = ((self.spec_concepts == self.ZERO) | (self.spec_concepts == x)).all(dim=-1)
        gen_consistency = ((self.gen_concepts == self.ZERO) | (self.gen_concepts == x)).all(dim=-1)
        return (spec_consistency.any() & gen_consistency.all()).item()


from concept_formation.preprocessor import Preprocessor
class Enumerizer(Preprocessor):
    def __init__(self,start_num=0):
        self.start_num = start_num
        self.attr_maps = {}
        # self.attr_counts = {}
        self.back_maps = {}
        self.keys = []

    '''Maps nominals to unsigned integers'''
    def transform(self, instance):
        """
        Transforms an instance.
        """
        as_list = []
        print(instance)
        for key,value in instance.items():
            if(not key in self.attr_maps):
                self.attr_maps[key] = {}
                # self.attr_counts[key] = self.start_num
                self.back_maps[key] = [None for i in range(self.start_num)]
                self.keys.append(key)
            attr_map = self.attr_maps[key]
            if(not value in attr_map):
                attr_map[value] = len(self.back_maps[key])
                self.back_maps[key].append(value)
                # self.attr_counts[key] += 1
            as_list.append(attr_map[value])

        return as_list


        

    def undo_transform(self, instance):
        """
        Undoes a transformation to an instance.
        """
        assert len(instance) == len(self.keys)

        d = {}

        for enum,key in zip(instance,self.keys):
            d[key] = self.back_maps[key][enum]

        return d



# def version_space():


if __name__ == "__main__":
    vthang = VersionSpaceILP()

    vthang.ifit([1,1,1,2,1],1)
    vthang.ifit([1,2,2,1,2],0)
    vthang.ifit([1,2,1,3,1],1)
    vthang.ifit([2,3,3,2,1],0)
    vthang.ifit([1,1,4,2,1],1)
    print(vthang.check_match([1,1,1,2,1]))
    print(vthang.check_match([1,2,2,1,2]))




    vthang = VersionSpaceILP()

    print("@@@@@@@")
    vthang.ifit([1,2,2,1],1)
    vthang.ifit([2,2,2,1],0)
    vthang.ifit([1,2,2,2],1)
    vthang.ifit([2,2,2,2],0)
    vthang.ifit([1,2,1,2],1)
    vthang.ifit([2,2,1,2],0)
    vthang.ifit([1,1,1,1],0)
    vthang.ifit([1,1,2,1],0)
    vthang.ifit([1,1,2,2],0)
    vthang.ifit([1,1,1,2],0)
    
    # vthang.ifit([1,1,1,1,1,1,1],1)
    # vthang.ifit([1,0,0,0,1,1,1],1)
    # vthang.ifit([1,1,1,0,0,0,1],1)
    # vthang.ifit([1,1,1,0,0,0,0],0)


    state = {
        "A1" : {
            "value" : 1,
            "above" : None,
            "below" : "B1",
            "left"  : "A2",
            "right" : None,
        },
        "A2" : {
            "value" : 2,
            "above" : None,
            "below" : "B2",
            "left"  : "A3",
            "right" : "A1",
        },
        "A3" : {
            "value" : 3,
            "above" : None,
            "below" : "B3",
            "left"  : None,
            "right" : "A2",
        },
        "B1" : {
            "value" : 4,
            "above" : "A1",
            "below" : "C1",
            "left"  : "B2",
            "right" : None,
        },
        "B2" : {
            "value" : 5,
            "above" : "A2",
            "below" : "C2",
            "left"  : "B3",
            "right" : "B1",
        },
        "B3" : {
            "value" : 6,
            "above" : "A3",
            "below" : "C3",
            "left"  : None,
            "right" : "B2",
        },
        "C1" : {
            "value" : 7,
            "above" : "B1",
            "below" : None,
            "left"  : "C2",
            "right" : None,
        },
        "C2" : {
            "value" : 8,
            "above" : "B2",
            "below" : None,
            "left"  : "C1",
            "right" : "C3",
        },
        "C3" : {
            "value" : 9,
            "above" : "B3",
            "below" : None,
            "left"  : "C2",
            "right" : None,
        }
    }

    enumer = Enumerizer(start_num=1)

    Av = VersionSpaceILP()
    Bv = VersionSpaceILP()

    for key,ele in state.items():
        ele_tensor = enumer.transform(ele)
        print("G",ele_tensor)
        if("A" in key):
            # Av.ifit(ele_tensor,1)
            Bv.ifit(ele_tensor,0)
        elif("B" in key):
            # Av.ifit(ele_tensor,0)
            Bv.ifit(ele_tensor,1)
    # print()
    # pprint(enumer.batch_undo(enumer.batch_transform(state.values())))