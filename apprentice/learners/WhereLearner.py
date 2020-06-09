if(__name__ == "__main__"):
    import sys
    sys.path.append("../")

from pprint import pprint
from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.cobweb import CobwebNode
from concept_formation.preprocessor import NameStandardizer
from concept_formation.structure_mapper import StructureMapper
from concept_formation.structure_mapper import rename_flat
from concept_formation.structure_mapper import contains_component
from apprentice.learners.IncrementalHeuristic import IncrementalHeuristic
from apprentice.planners.fo_planner import Operator
from apprentice.planners.fo_planner import build_index
from apprentice.planners.fo_planner import subst
import numpy as np
import itertools
from timeit import repeat
import numba
from numba import types, njit,guvectorize,uint32,vectorize,prange
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.types import UniTuple, ListType

global my_gensym_counter
my_gensym_counter = 0

import logging
logging.disable(logging.CRITICAL)

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
        lst = []
        for e in arg:
            for v in get_vars(e):
                if v not in lst:
                    lst.append(v)
        return lst
    elif isinstance(arg, str) and len(arg) > 0 and arg[0] == '?':
        return [arg]
    else:
        return []


class WhereLearner(object):

    def __init__(self, learner_name, **learner_kwargs):
        self.learner_name = learner_name
        self.learner_kwargs = learner_kwargs
        self.rhs_by_label = {}
        self.learners = {}

    def add_rhs(self, rhs, constraints):
        # args = [skill.selection_var] + skill.input_vars
        self.learners[rhs] = get_where_sublearner(self.learner_name,
            args=tuple(rhs.all_vars), constraints=constraints, **self.learner_kwargs)

        rhs_list = self.rhs_by_label.get(rhs.label, [])
        rhs_list.append(rhs)
        self.rhs_by_label[rhs.label] = rhs_list

    def check_match(self, rhs, t, x):
        return self.learners[rhs].check_match(t, x)

    def get_match(self, rhs, X):
        # args = [skill.selection_var] + skill.input_vars
        return self.learners[rhs].get_match(X)

    def get_matches(self, rhs, X):
        # args = [skill.selection_var] + skill.input_vars
        return self.learners[rhs].get_matches(X)

    def ifit(self, rhs, t, X, y):
        # args = [skill.selection_var] + skill.input_vars
        return self.learners[rhs].ifit(t, X, y)

    def fit(self, rhs, T, X, y):
        return self.learners[rhs].fit(T, X, y)

    def get_strategy(self):
        return WHERE_STRATEGY[self.learner_name.lower().replace(" ","").replace("_","")]

    def skill_info(self,rhs):
        return self.learners[rhs].skill_info()


class FastMostSpecific(BaseILP):
    """
    This learner always returns the tuples it was trained with, after ensuring
    they meet any provided constraints.
    """
    def __init__(self, args,constraints=None):
        self.args = args
        self.tuples = set()

        if constraints is None:
            self.constraints = frozenset()
        else:
            self.constraints = constraints

    def __repr__(self):
        return repr(self.tuples)

    def __str__(self):
        return str(self.tuples)

    def check_constraints(self,t,x):
        if(self.constraints is None):
            return True
        for i,part in enumerate(t):
            c = 0 if i == 0 else 1
            if(not self.constraints[c](x[part])):
                return False
        return True

    def check_match(self, t, x):
        x = x.get_view("object")
        t = tuple(t)
        if(self.check_constraints(t,x)):
            if(t in self.tuples):
                return True
        
        return False


    def get_matches(self, x, epsilon=0.0):
        x = x.get_view("object")

        for t in self.tuples:
            if(self.check_constraints(t,x)):
                print("T",t)
                yield t


    def ifit(self, t, x, y):
        # t = tuple(ground(e) for e in t)
        if(y > 0):
            self.tuples.add(tuple(t))

    def skill_info(self):
        return self.tuples

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
        x = x.get_view("flat_ungrounded")
        # print("CHECK MATCHES T", t)

        t = tuple(ground(ele) for ele in t)

        # Update to include initial args
        mapping = {a: t[i] for i, a in enumerate(self.args)}


        if t not in self.tuples:
            return False

        grounded = self.ground_example(x)
        index = build_index(grounded)

        # Update to include initial args
        operator = Operator(tuple(('Rule',) + self.args),
                            frozenset().union(self.constraints), [])
        for m in operator.match(index, initial_mapping=mapping):
            return True
        return False

    def get_matches(self, x, epsilon=0.0):
        x = x.get_view("flat_ungrounded")

        grounded = self.ground_example(x)

        index = build_index(grounded)

        for t in self.tuples:
            mapping = {a: t[i] for i, a in enumerate(self.args)}
            operator = Operator(tuple(('Rule',) + self.args),
                                frozenset().union(self.constraints), [])

            for m in operator.match(index, epsilon=epsilon,
                                    initial_mapping=mapping):
                result = tuple(ele.replace("QM", '?') for ele in t)
                yield result
                break


    def ifit(self, t, x, y):
        # print("TUPIN", t)

        if y > 0:
            self.pos_count += 1
        else:
            self.neg_count += 1

        t = tuple(ground(e) for e in t)
        self.tuples.add(t)
        # print(len(self.tuples))


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
        x = x.get_view("flat_ungrounded")

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
        x = x.get_view("flat_ungrounded")

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
        x = x.get_view("flat_ungrounded")
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


def get_where_sublearner(name, **learner_kwargs):
    return WHERE_SUBLEARNERS[name.lower().replace(' ', '').replace('_', '')](**learner_kwargs)


def get_where_learner(name, **kwargs):
    return WhereLearner(name, **kwargs)


@njit(nogil=False, parallel=False,fastmath=False,cache=True)
def fill_pairs_at(partial_matches,i,pair_matches):
    ''' 
    Adds new bindings to each partial match in list partial_matches, by
    trying to apply consistent pairs of elements associated with concept_i.
    '''
    #For every concept_j adjacent to concept_i
    for j, pair_matches_ij in pair_matches[i].items():
        new_pms = List()
        #Apply the following to each partial_match "pm" so far:
        for pm in partial_matches:
            altered = False
            
            #For each consistent pair (e_i, e_j) of elements matching concepts
            #   i and j respectively:
            for (k, e_i, e_j) in pair_matches_ij:
                #Bind the pair to partial match "pm" if:
                okay = True
                for p in range(len(pm)):
                    #concept_i is unbound or concept_i is bound to e_i
                    if(p == i):
                        if(not (pm[i] == 0 or pm[i] == e_i)): okay = False; break;
                    #and concept_j is unbound or concept_j is bound to e_j
                    elif(p == j):
                        if(not (pm[j] == 0 or pm[j] == e_j)): okay = False; break;
                    #and e_i and e_j are not bound elsewhere in partial match "pm"
                    elif(pm[p] == e_i or pm[p] == e_j):
                        okay = False; break;
                
                #Then append the new partial match to the set of partial matches
                if(okay):
                    new_pm = pm.copy()
                    new_pm[i], new_pm[j] = e_i, e_j
                    new_pms.append(new_pm)
                    altered = True

            #If partial match "pm" hasn't picked up any new bindings to concepts i and j
            #   but at least one was not yet bound, then no further binding can be found 
            #   for "pm" and it is rejected. Otherwise keep pm for the next iteration.
            if(not altered and pm[i] != 0 and pm[j] != 0):
                new_pms.append(pm)
        partial_matches = new_pms
    return partial_matches

@njit(nogil=False, parallel=False,fastmath=False,cache=True)
def fill_singles_at(partial_matches,i,cand_names):
    '''
    For concept_i which is free floating and without a concept
    tracked neighbor, bind the candidates for this concept
    to every partial match in partial_matches. 
    '''
    new_pms = List()
    for pm in partial_matches:
        if(pm[i] == 0):
            for cn in cand_names:
                if(not (cn == pm).any()):
                    new_pm = pm.copy()
                    new_pm[i] = cn
                    new_pms.append(new_pm)
    return new_pms


@njit(nogil=False, parallel=False,fastmath=False,cache=True)
def find_consistent_pairs(concepts, elems, candidates,
                        elem_names, where_vars_to_inds):
    '''
    Builds the list "pair_matches" which holds a dictionary for each
    concept_i with keys for the indicies of adjacent concepts_j and 
    values are lists of pairs of elements which are mutually 
    consistent candidates for concept_i and concept_j respectively.
    '''
    pair_matches = List()
    for i,(concept_i, cands_inds_i) in enumerate(zip(concepts,candidates)):
        pair_matches_i = Dict.empty(i8,list_of_u4_triple)
        #Loop through every feature of concept_i
        for k, f_k in enumerate(concept_i):

            #If a reference to concept_j is present at feature k of concept_i
            if(f_k in where_vars_to_inds):
                j = where_vars_to_inds[f_k]
                elem_names_j = elem_names[candidates[j]]
                pair_matches_ij = List.empty_list(u4_triple)

                #Find pairs of consistent candidates between the two concepts
                for c1_ind in cands_inds_i:
                    f_k_of_c1 = elems[c1_ind][k]
                    for c2_val in elem_names_j:
                        if(f_k_of_c1 == c2_val):
                            #Add the triple (k, e_i,e_j) as a candidate pair
                            pair_matches_ij.append((k,elem_names[c1_ind],c2_val))
                pair_matches_i[j] = pair_matches_ij
        pair_matches.append(pair_matches_i)
    return pair_matches


u4_triple = UniTuple(u4,3)
list_of_u4_triple = ListType(u4_triple)
@njit(nogil=False, parallel=False,fastmath=False,cache=True)
def match_iterative(split_ps, concept_slices,
                    _elems,elems_slices,
                    concept_cands,cand_slices,
                    elem_names,
                    where_part_vals):
    '''
    Produces every consistent matching of elems to concepts.
    '''

    #----To be new inputs-----
    where_vars_to_inds = Dict.empty(u4,i8)
    for j,v in enumerate(where_part_vals):
        where_vars_to_inds[v] = j
    concepts = List()
    elems = List()
    candidates = List()
    for i in range(len(concept_slices)-1):
        concepts.append(split_ps[concept_slices[i]:concept_slices[i+1]])
    for i in range(len(elems_slices)-1):
        elems.append(_elems[elems_slices[i]:elems_slices[i+1]])
    for i in range(len(cand_slices)-1):
        candidates.append(concept_cands[cand_slices[i]:cand_slices[i+1]])
    #----END new inputs-----

    n_concepts = len(concepts)
    pair_matches = find_consistent_pairs(concepts, elems, candidates,
                        elem_names, where_vars_to_inds)

    partial_matches = List()
    partial_matches.append(np.zeros((n_concepts,),dtype=np.uint32))

    for i in range(n_concepts):
        partial_matches = fill_pairs_at(partial_matches,i,pair_matches)

    for i in range(n_concepts):
        if(len(pair_matches[i]) == 0):
            partial_matches = fill_singles_at(partial_matches,i,[elem_names[c] for c in candidates[i]])
    
    return partial_matches

def flatten_n_slice(lst):
    flat = [x if isinstance(x,list) else np.array(x).reshape(-1) for x in lst]
    lens = [0] + [len(x)  for x in flat]
    slices = np.cumsum(lens)
    out = np.array([x for x in itertools.chain(*flat)])
    return out,slices

try:
    import torch
    from torch.nn import functional as F
except:
    pass
def mask_select(x, m):
    return x[m.nonzero().view(-1)]


class VersionSpace(BaseILP):
    def __init__(self, args=None, constraints=None, use_neg=False, use_gen=False,
                       propose_gens=True, use_neighbor_concepts=True,
                       non_literal_attrs=["to_right","to_left","right","left","above","below","value","contentEditable"],
                       remove_attrs=[],
                       null_types=[None,""]):
        self.pos_concepts = VersionSpaceILP(use_gen=use_gen)
        self.neg_concepts = VersionSpaceILP(use_gen=use_gen) if use_neg else None
        self.propose_gens = propose_gens
        self.constraints = constraints
        self.use_neg = use_neg
        self.elem_slices = None
        self.elem_types = None
        self.initialized = False
        self.pos_ok = False
        self.neg_ok = False
        self.use_neighbor_concepts = use_neighbor_concepts
        self.expected_neighbors = None 
        self.use_gen = use_gen
        self.non_literal_attrs= non_literal_attrs
        self.null_types = null_types
        self.remove_attrs = remove_attrs

    def initialize(self, n):
        assert n >= 1, "not enough elements"
        self.num_elems = n
        self.enumerizer = Enumerizer(start_num=0,
                                     force_add=["<#ANY>",None,'?sel'] + ['?arg%d' % i for i in range(n-1)],
                                     remove_attrs=self.remove_attrs)
        self.initialized = True

    def remove_concepts(self,indicies):
        # print("remove_concepts!!!!!!!!", indicies)
        # print(indicies)
        
        # print("CONCEPT BEFORE", self.pos_concepts.spec_concepts.size())                    
        s = self.elem_slices
        for i in reversed(indicies):
            # print(i)
            ps = self.pos_concepts.spec_concepts
            # print(s[i],s[i+1])
            self.pos_concepts.spec_concepts = np.concatenate([ps[:,:s[i]],ps[:,s[i+1]:]],axis=1)
            # print("CONCEPT", self.pos_concepts.spec_concepts.size())
            if(self.neg_ok):
                ns = self.neg_concepts.spec_concepts
                self.neg_concepts.spec_concepts = np.concatenate([ns[:,:s[i]],ns[:,s[i+1]:]],axis=1)

        sl = np.array(self.elem_slices)
        # print(indicies)
        # print(sl)
        for i in indicies:
            sl[i+1:] -= sl[i+1]-sl[i]
        self.elem_slices = sl[[i for i in range(len(sl)) if i not in indicies]].tolist()
        # print("BEEP",[x for i,x in enumerate(self.elem_types) if i not in indicies])
        # print(self.elem_slices)
        # print(self.pos_concepts.spec_concepts.view(-1,5))
            
    def _resolve_neighbor_relations(self,t,x,relations):
        rename_dict = {ele: "?sel" if i == 0 else "?arg%d" % (i-1) for i, ele in enumerate(t)}
        inv_rename_dict = {v:k for k,v in rename_dict.items()}
        relation_map = {}

        for v_i in relations:
            name = x[inv_rename_dict[v_i[0]]][v_i[1]]
            if(isinstance(name,list)): name = name[v_i[2]]
            relation_map[v_i] = name
        return relation_map

    def get_neighborized_vs_elems(self,t,x,gen_literals=False):
        rename_dict = {ele: "?sel" if i == 0 else "?arg%d" % (i-1) for i, ele in enumerate(t)}
        # print(self.expected_neighbors)
        relation_map = self._resolve_neighbor_relations(t,x,itertools.chain(*self.expected_neighbors.values()))
        # print(relation_map)
        neigh_rename_dict = {}
        for i,(k,v) in enumerate(self.expected_neighbors.items()):
            for v_i in v:
                name = relation_map[v_i]
                if(name != "" and name is not None and x[name]['type'] == self.elem_types[i+self.num_elems]):
                    # assert name not in neigh_rename_dict or neigh_rename_dict[name] == k, "%s != %s" % (neigh_rename_dict.get(name,None),k)
                    if(name not in  neigh_rename_dict or neigh_rename_dict[name] == k):
                        neigh_rename_dict[name] = k
                    else:
                        return None, None    
                else:
                    return None, None


        # pprint(relation_map)
        # pprint(instances)
        # pprint(neigh_rename_dict)
        # print([t_name for t_name in [*t,*neigh_rename_dict.keys()]])
        ##TODO: IMPURE
        
        # print("VS_TYPES", self.elem_types)
        # pprint(rename_dict)
        # pprint(neigh_rename_dict)
        rename_dict = {**rename_dict,**neigh_rename_dict}
        instances = [rename_values(x[t_name],rename_dict,exclude=[t_name]) 
                     for t_name in [*t,*neigh_rename_dict.keys()]]

        
        # pprint("instances")
        # pprint(instances)
        if(gen_literals):
            non_literals = set([*self.null_types,*rename_dict.values()])
            def genrl(k,v):
                if(isinstance(v,list)):
                    return [genrl(k,v_i) for v_i in v]
                elif(v not in non_literals and k in self.non_literal_attrs):
                    # print(v,rename_dict.get(v,None))
                    return "<#ANY>"
                else:
                    return v

            instances = [{k:genrl(k,v) for k,v in inst.items()} 
                          for inst in instances]
            # pprint(instances)

        vs_elems = self.enumerizer.transform(instances)
        # print("VS_ELM", len(vs_elems))
        return instances,vs_elems

    def get_vs_elems(self,t,x,gen_literals=False):
        rename_dict = {ele: "?sel" if i == 0 else "?arg%d" % (i-1) for i, ele in enumerate(t)}
        instances = [rename_values(x[t_name],rename_dict, exclude=[t_name]) for t_name in t]
        if(gen_literals):
            non_literals = set([*self.null_types,*rename_dict.values()])
            def genrl(k,v):
                if(isinstance(v,list)):
                    return [genrl(k,v_i) for v_i in v]
                elif(v not in non_literals and k in self.non_literal_attrs):
                    # print(v,rename_dict.get(v,None))
                    return "<#ANY>"
                else:
                    return v

            instances = [{k:genrl(k,v) for k,v in inst.items()} 
                          for inst in instances]
        # print(self.initialized)
        # print(len(t), self.num_elems)
        # print(self.elem_types)
        # print([z['type'] for z in instances])
        # print([instance['type'] == elm_t for instance,elm_t in zip(instances,self.elem_types)])
        # print((not self.initialized or len(t) == self.num_elems))
        # print((self.elem_types is None or all([instance['type'] == elm_t for instance,elm_t in zip(instances,self.elem_types)] )))
        if((not self.initialized or len(t) == self.num_elems) and (self.elem_types is None
            or all([instance['type'] == elm_t for instance,elm_t in zip(instances,self.elem_types)] ))):
            vs_elems = self.enumerizer.transform(instances)
            return instances,vs_elems
        else:
            # print("NONE!")
            return None,None

    def fit_expected_neighbors(self,t,x,y):
        # print(t)
        instances,vs_elems = self.get_vs_elems(t,x)

        neigh_relations = {}
        varz = ["?sel" if i == 0 else "?arg%d" % (i-1) for i in range(len(t))]
        for inst,var, inst_id in zip(instances,varz, t):
            for k,v in inst.items():
                # print(inst_id, v,k)
                if(isinstance(v,list)):
                    for i,v_i in enumerate(v):
                        if(v_i not in varz and v_i in x and v_i != inst_id):
                            neigh_relations[v_i] = neigh_relations.get(v_i,[])+[(var,k,i)]
                else:
                    if(v not in varz and v in x and v != inst_id):
                        neigh_relations[v] = neigh_relations.get(v,[])+[(var,k)]
                # print("%s.%s"%(var,k),v)
        # print("neigh_relations")
        # print(neigh_relations)
    # neigh_rename_dict = {k:("?neigh%d"%i) for i,k in enumerate(neigh_relations)}
    # neigh_instances = [rename_values(x[n_name],t_neigh_relations) for n_name in neigh_relations.keys()]
    # neigh_elems = self.enumerizer.transform(neigh_instances)

        # print("neigh_relations")
        # print(neigh_relations)
        if(self.expected_neighbors is None):
            self.elem_types = [x[t_name]["type"] for t_name in [*t,*neigh_relations.keys()]]
            self.expected_neighbors = {("?neigh%d"%i): vals for i,vals in enumerate(neigh_relations.values())}
            # neigh_rename_dict = {k:("?neigh%d"%i) for i,k in enumerate(neigh_relations)}
            # pprint(neigh_rename_dict)
            # print("Init expected_neighbors:")
            # pprint(self.expected_neighbors)

            # if(self.elem_types is not None):
            #     print(len(self.elem_types),len(list(self.expected_neighbors.keys()))+len(t))
            # assert self.elem_types is None or len(self.elem_types)==len(list(self.expected_neighbors.keys()))+len(t), "BUG0!!"
        elif(y > 0):
            to_remove = []
            neighbor_assignments = []
            relations = []

            # print("VVVVVVVVVVVVVV", len(self.elem_types),len(list(self.expected_neighbors.keys())))

            for i,(key,neigh_i) in enumerate(self.expected_neighbors.items()):
                # print(key)
                intersections = [[x for x in n_j if x in neigh_i] for n_j in neigh_relations.values()]
                best_match = max(intersections, key=lambda x: len(x))
                num_relations_removed = len(neigh_i)- len(best_match) 
                # print("num_relations_removed",num_relations_removed)
                
                # print(intersections)
                

                # if(len(best_match) == 0):
                #     to_remove.append(i+len(t))
                # else:
                neighbor_assignments.append((-num_relations_removed,best_match,key,i+len(t)))
                # print(neighbor_assignments[-1])
                # print(neighbor_assignments[-1])
                relations.append(best_match)
            # print("^^^^^^^^^^^^^^")

            relation_map = self._resolve_neighbor_relations(t,x,itertools.chain(*relations))
            neighbor_assignments = sorted(neighbor_assignments)
            # print(neighbor_assignments)
            assigned = set()
            assignments = []
            for _, relations, key,i in neighbor_assignments:
                if(len(relations) != 0):
                    elms = [relation_map[x] for x in relations]
                    assert len(set(elms)) == 1, "BAAAAAD!"
                    elm = elms[0]
                    # print(i,self.elem_types)
                    # print(x[elm]["type"] , self.elem_types[i])
                    if(x[elm]["type"] == self.elem_types[i] and elm not in assigned):
                        assigned.add(elm) 
                        assignments.append((i,elm,key,relations))
                        continue
                to_remove.append(i)

            assignments = sorted(assignments)
            # inv_new_expected_neighbors = {relation_map[x[0]]:}
            # print(inv_new_expected_neighbors)


            
            
            
            # pprint(neighbor_assignment)
            new_expected_neighbors = {key:relations for (_,_,key,relations) in sorted(assignments)}
            # pprint("expected_neighbors")
            # pprint(new_expected_neighbors)
            # to_remove = [i+len(t) for i,k in enumerate(self.expected_neighbors.keys()) if k not in new_expected_neighbors]
            # print(to_remove)
            # raise(ValueError)
            if(len(to_remove) > 0): self.remove_concepts(sorted(to_remove))

            neigh_literals = [elm for _,elm,_,_ in assignments ]
            self.elem_types = [x[t_name]["type"] for t_name in [*t,*neigh_literals]]
            self.expected_neighbors = new_expected_neighbors
            # print("BOOP",self.elem_types)

            # if(self.elem_types is not None):
            #     print(len(self.elem_types),len(list(self.expected_neighbors.keys())))
            # assert self.elem_types is None or len(self.elem_types)==len(list(self.expected_neighbors.keys()))+len(t), "BUG1!!"
            

        
        # return instances, vs_elems


    def ifit(self, t, x, y):
        print("FIT B", t,y)
        # with torch.no_grad():
        if(len(t) != len(set(t))):
            raise ValueError("Where elements must be unique; recieved : %s" % t)
            return
        # if(self.elem_types is not None):
        #     print(len(self.elem_types),len(list(self.expected_neighbors.keys())))

        
        # print("I AM VERSIONSPACE",y)
        x = x.get_view("object")
        # print([x[t_name] for t_name in t])

        #Do this just so everything is in the map
        
        # x = rename_values(x,{ele:"sel" if i == 0 else ele:"arg%d" % i-1 for i,ele in enumerate(t)})
        assert False not in ["type" in x[t_name] for t_name in t], "All interface elements must have a type and a static set of attributes."

        if(not self.initialized):
            self.initialize(len(t))
            self.elem_types = [x[t_name]["type"] for t_name in t]
        assert len(t) == self.num_elems, "incorrect number of arguments for this rhs"

        if(self.use_neighbor_concepts):
            assert self.elem_types is None or self.expected_neighbors is None or len(self.elem_types)==len(list(self.expected_neighbors.keys()))+len(t), "BUG!!"
            self.fit_expected_neighbors(t,x,y)
            instances, vs_elems = self.get_neighborized_vs_elems(t,x,gen_literals=True)
        else:
            instances, vs_elems = self.get_vs_elems(t,x,gen_literals=True)

        if(vs_elems is None):
            return

        # print(self.use_neighbor_concepts)
        print("VS_ELMS",vs_elems)
            # self.elem_types = [x[t_name]["type"] for t_name in t]

            # inv_rename_dict = {v:k for k,v in rename_dict.items()}
            # neigh_rename_dict = {}
            # for k,v in self.expected_neighbors.items():
            #     for v_i in v:
            #         name = x[inv_rename_dict[v_i[0]]][v_i[1]]
            #         if(isinstance(name,list)): name = name[v_i[2]]
            #         neigh_rename_dict[name] = k 
                
            # # pprint(rename_dict)
            # pprint(instances)
            # # pprint(neigh_rename_dict)

            # instances = [rename_values(x[t_name],{**rename_dict,**neigh_rename_dict}) 
            #              for t_name in [*t,*neigh_rename_dict.keys()]]
            # vs_elems = self.enumerizer.transform(instances)
        # neigh_elems = []
        # pprint(instances)
                # print(" ",intersections)
                # intersections = neigh_relations.values()
                # print("expected_neighbors:",)
                # pprint(neigh_relations)
            # removable_neighbors = [x for x in self.expected_neighbors if x not in neigh_relations.values()]
            # pprint("remaining_neighbors",neigh_relations.values())
            # pprint("removable_neighbors",removable_neighbors)

        # pprint(instances)
        # pprint(neigh_instances)
        # pprint(neigh_relations)
        
        

        #Do this just so everything is in the map
        self.enumerizer.transform({i:x for i,x in enumerate(x.keys())})
        self.enumerizer.transform(list(x.values()))

        # print("VS")
        # print(self.enumerizer.transform(list(x.values())))

        if(self.elem_slices == None):
            self.elem_slices = [0] + np.cumsum([len(vs_elem) for vs_elem in vs_elems]).tolist()
            # self.elem_slices += 
        # print(t)
        # print(self.enumerizer.attr_maps[0])
        # print(x.keys())
        # print("vs_elems:")
        # pprint(vs_elems)
        # if(self.pos_concepts.spec_concepts is not None):
            # pprint(self.skill_info())
            # print("BEFORE: ",self.pos_concepts.spec_concepts.view(-1,5))

        flat_vs_elems = list(itertools.chain(*vs_elems))
        self.pos_concepts.ifit(flat_vs_elems, y)
        self.pos_ok = self.pos_ok or y > 0
        if(self.use_neg):
            self.neg_concepts.ifit(flat_vs_elems, 0 if y > 0 else 1)
            self.neg_ok = self.neg_ok or y < 0

            # if(self.elem_types is not None):
            #     print(len(self.elem_types),len(list(self.expected_neighbors.keys())))
            # assert self.elem_types is None or len(self.elem_types)==len(list(self.expected_neighbors.keys()))+len(t), "BUG!!"

            # print("AFTER: ")
            # print(self.pos_concepts.spec_concepts.view(-1,5))
            # print(self.elem_types)

            # for i in range(self.num_elems):

            # self.positive_concepts.ifit()

    # def match_elem(self,t_name):
    def check_constraints(self,t,x):
        if(self.constraints is None):
            return True
        for i,part in enumerate(t):
            c = 0 if i == 0 else 1
            if(not self.constraints[c](x[part])):
                return False
        return True

    def check_match(self, t, x):
        # with torch.no_grad():
        if(not self.pos_ok):
            return False

        x = x.get_view("object")

        if(not self.check_constraints(t,x)):
            return False

        # def _rename_values(x):
        #     return rename_values(x, {ele: "?sel" if i == 0 else "?arg%d" % (i-1) for i, ele in enumerate(t)})

        # instances = [_rename_values(x[t_name],) for t_name in t]
        if(self.use_neighbor_concepts):
            instances,vs_elems  = self.get_neighborized_vs_elems(t,x)
            if(vs_elems is None or len(instances) != len(self.elem_slices)-1):
                return False
        else:  #self.enumerizer.transform(instances)
            instances,vs_elems  = self.get_vs_elems(t,x)
        # print(vs_elems,self.use_neighbor_concepts,self.expected_neighbors)
        # print(torch.tensor(vs_elem).view(3, -1))
        if(self.use_neg):
            x = np.array(vs_elems, dtype=np.uint8).reshape((1, -1))
            ps, pg = self.pos_concepts.spec_concepts, self.pos_concepts.gen_concepts
            ns, ng = self.neg_concepts.spec_concepts, self.neg_concepts.gen_concepts
            ZERO = self.pos_concepts.ZERO
            spec_consistency = (((ps == ZERO)) | (ps == x)).all(axis=-1)

            out = spec_consistency.any()
            if(self.use_gen):
                gen_consistency = (((pg == ZERO)) | (pg == x)).all(axis=-1)
                out = out & gen_consistency.all()

            # print(self.enumerizer.attr_maps[0])
            # print(self.enumerizer.back_maps[0])
            # print(x)
            # print(self.enumerizer.back_maps[0][x.view(-1).tolist()[0]], self.enumerizer.back_maps[0][5])
            # print(ps)
            # print((((ps == ZERO)) | (ps == x)))

            if(self.neg_ok):
                neg_spec_consistency = ((ns == ZERO) | (ns == ps) | (ns != x)).all(axis=-1)
                out = out & neg_spec_consistency.any()
                if(self.use_gen):
                    neg_gen_consistency = ((ng == ZERO) | (ng == ps) | (ng != x)).all(axis=-1)
                    out = out & neg_gen_consistency.any()

            
            return out.item()
        else:
            return self.pos_concepts.check_match(vs_elems) > 0


    def get_matches(self, x):
        # with torch.no_grad():
        if(not self.pos_ok):
            return

        state = x.get_view("object")
        # assert False not in [for x in range(self.num_elems), \
        # "It is not the case that the enum for argX == X+2"
        # print(self.enumerizer.transform_values(["?sel","?arg0","?arg1"]))

        if(self.elem_slices == None):
            return
        # print(self.pos_concepts.spec_concepts)

        #Do this just so everything is in the map
        self.enumerizer.transform({i:x for i,x in enumerate(state.keys())})
        self.enumerizer.transform(list(state.values()))

        # Create a tensor which contains the enum values associated with the when parts
        #  i.e. "?sel", "?arg0", "?arg1" would probably map to 2, 3, 4
        where_part_vars = ["?sel"] + ['?arg%d' % i for i in range(self.num_elems-1)]
        if(self.use_neighbor_concepts): where_part_vars += list(self.expected_neighbors.keys())
        where_part_vals = self.enumerizer.transform_values(where_part_vars)
        where_part_vals = np.array(where_part_vals,dtype=np.uint8)

        # Make a tensor that has all of the enum values for the names of the elements/objects
        #  in the state in order that they appear
        elem_names_list = [key for key in state.keys()]
        elem_names = self.enumerizer.transform_values(elem_names_list)
        elem_names = np.array(elem_names,dtype=np.uint8)

        # Make a list that has all of the enumerized elements/objects
        elems_list = [val for val in state.values()]
        elems = self.enumerizer.transform(elems_list)
        # pprint(elems_list)

        # Make a list that has all of the enumerized elements/objects, but zeros in
        #  all of the slot for attributes which have elements names as values. 
        #  Only make this replacement if the element is of a type that we match to
        zero_map = {ele_name: 0 for ele_name,ele in state.items() 
                    if 'type' in ele and ele['type'] in self.elem_types}
        elems_scrubbed = self.enumerizer.transform(elems_list, zero_map)
        # print(elems_scrubbed)
        
        # Split up the concepts by the where parts that each slice selects on
        ps, pg = self.pos_concepts.spec_concepts, self.pos_concepts.gen_concepts
        
        s = self.elem_slices
        split_ps = [ps[:, s[i]:s[i+1]] for i in range(len(self.elem_slices)-1)]
        # pprint(split_ps)
        if(self.neg_ok):
            ns, ng = self.neg_concepts.spec_concepts, self.neg_concepts.gen_concepts
            split_ns = [ns[:, s[i]:s[i+1]] for i in range(len(self.elem_slices)-1)]
        else:
            split_ns = [None] * len(split_ps)

        # Initialize a bunch of containers that we will fill
        inds_by_type = {}
        candidates_by_type = {}
        scrubbed_by_type = {}
        elem_names_by_type = {}
        consistencies = []
        concept_cand_indicies = []
        concept_candidates = []
        # concept_cand_replacements = []
        

        ZERO = self.pos_concepts.ZERO
        # all_consistencies = []
        # all_concepts
        
        # Loop over the concepts for each where part and cull down the list of
        #   possible matches for them. This phase only filters out elements we can
        #   rule out without committing to any part of the where assignment. 
        for typ, ps_i, ns_i in zip(self.elem_types, split_ps, split_ns):


            # If the type for this concept is new then populate the list of candidates in various formats
            if(typ not in inds_by_type):
                candidate_indices = [i for i, e in enumerate(elem_names_list) if state[e]["type"] == typ]
                
                cnd = np.array([elems[i] for i in candidate_indices], dtype=np.uint8)
                candidates_by_type[typ] = cnd

                cnd_s = np.array([elems_scrubbed[i] for i in candidate_indices], dtype=np.uint8)
                cnd_s = cnd_s.reshape(len(candidate_indices), 1, ps_i.shape[-1])
                scrubbed_by_type[typ] = cnd_s

                inds_by_type[typ] = np.array(candidate_indices,dtype=np.long)
                elem_names_by_type[typ] = elem_names[inds_by_type[typ]]
                
            else:
                candidate_indices = inds_by_type[typ]
                cnd_s = scrubbed_by_type[typ]

            # Check the consistency of each scrubbed candidate with the positive and negative
            #   concepts. This is a normal concept comparison except that any attribute slot where  
            #   the scrubbed candidate has a zero is always considered consistent. 
            ps_i = np.expand_dims(ps_i,axis=0)
            if(self.neg_ok):
                ns_i = np.expand_dims(ns_i,axis=0)
            # ZERO = ZERO.numpy()
            # pss = torch.tensor([[0 if (x in elem_names or x in where_part_vals) else x for x in ps_i.view(-1) ]],dtype=torch.uint8)
            # print(pss)
            # print(cnd_s)
            # print(ps_i)
            # consistency = ((pss == ZERO) | (pss == cnd_s)).all(dim=-1).any(dim=-1)
            consistency = ((ps_i == ZERO) | (ps_i == cnd_s) | (cnd_s == ZERO)).all(axis=-1).any(axis=-1)
            if(self.neg_ok):
                nss = torch.tensor([[0 if (x in elem_names or x in where_part_vals) else x for x in ns_i.view(-1) ]],dtype=torch.uint8)
                # ns_consistency = ((nss == ZERO) | (nss != cnd_s) | (ps_i == nss)).all(dim=-1).any(dim=-1)
                ns_consistency = ((nss == ZERO) | (nss != cnd_s) | (ps_i == nss)).all(axis=-1).any(axis=-1)
                consistency = consistency & ns_consistency

            #Store our culled down set of candidates in various formats
            consistencies.append(consistency)
            # scooby_doo = np.zeros(len(elem_names))
            # torch.gather(scooby_doo, 1, torch.tensor([[0,0],[1,0]]))
            # print(inds_by_type[typ])
            # print(consistency)
            # scooby_doo[inds_by_type[typ]] = consistency.numpy()
            # print("scooby_doo")
            # print(scooby_doo)
            # all_consistencies.append(scooby_doo)
            # print(ps_i)
            # print(consistency)
            # print(consistency)
            # print(self.enumerizer.back_maps[0])
            # print(ps_i)
            # print(((ps_i == ZERO) | (ps_i == cnd_s) | (cnd_s == ZERO)))
            # print(consistency.nonzero().tolist())

            # print([elem_names_list[j] for j in inds_by_type[typ].view(-1).tolist()])
            # print([self.enumerizer.back_maps[0][j] for j in elem_names_by_type[typ].view(-1).tolist()])
            # print()
            # print([elem_names_list[j] for j in torch.masked_select(inds_by_type[typ],consistency).tolist()])
            # concept_cand_indicies.append(torch.ones_like(consistency).nonzero())
            # print(candidate_indices)
            concept_cand_indicies.append([x for i,x in enumerate(candidate_indices) if consistency[i] == 1])
            # concept_candidates.append( (consistency, torch.masked_select(elem_names_by_type[typ],consistency)) )

        # all_consistencies = torch.tensor(all_consistencies)


        split_ps_flat, concept_slices = flatten_n_slice(split_ps)
        elems_flat, elems_slices = flatten_n_slice(elems)
        concept_cands_flat, cand_slices = flatten_n_slice(concept_cand_indicies)

        # print("split_ps")
        # print(split_ps_flat)
        # print("concept_slices")
        # print(concept_slices)
        # print("elems_flat")
        # print(elems_flat)
        # print("elems_slices")
        # print(elems_slices)
        # print("concept_cands_flat")
        # print(concept_cands_flat)
        # print("cand_slices")
        # print(cand_slices)
        # print("elem_names")
        # print(elem_names)
        # print("where_part_vals")
        # print(where_part_vals)


        # print(type(split_ps_flat), type(concept_slices))
        # print(type(elems_flat), type(elems_slices))
        # print(type(concept_cands_flat), type(cand_slices))
        # timefunc("match_iterative",match_iterative,
        #                     split_ps, concept_slices,
        #                     elems,elems_slices,
        #                     concept_cands,cand_slices,
        #                     elem_names,
        #                     where_part_vals)
        # timefunc("numpy_mi",numpy_mi,og_split_ps,elems,elem_names,concept_cand_indicies,where_part_vals)
        # translated = np.array([[]])
        # print(concept_cand_indicies)
        # for c in concept_cand_indicies:
        #     nms = elem_names.numpy()[np.array(c)]
        #     print([self.enumerizer.back_maps[0][z] for z in nms])
        # print()
        # print("Before")
        translated = match_iterative(split_ps_flat, concept_slices,
                            elems_flat,elems_slices,
                            concept_cands_flat,cand_slices,
                            elem_names,
                            where_part_vals)
        # translated = np.array([])
        # matches = self._match_iterative(split_ps,split_ns,candidates_by_type,concept_candidates,concept_cand_indicies,where_part_vals,elem_names,consistencies,all_consistencies,elems)
        # matches = self._match_naive(split_ps,split_ns,candidates_by_type,concept_candidates,concept_cand_indicies,where_part_vals)

        # Translate these indicies so that they select from the original state
        # translated = []
        # for i,(typ,indicies) in enumerate(zip(self.elem_types,concept_cand_indicies)):
        #     rel_to_type = torch.index_select(indicies,0,matches[:,i])
        #     # print(rel_to_type)
        #     # print(inds_by_type[typ])
        #     # print(torch.index_select(inds_by_type[typ],0,rel_to_type.view(-1)))
        #     translated.append(torch.index_select(inds_by_type[typ],0,rel_to_type.view(-1)).view(-1,1))
        # translated = torch.cat(translated,dim=1)
        # print(translated)
        
        #Yield each consistent 'where' assignments (i.e. the set of matches) by their original names
        # print(elem_names_list)
        pprint(self.skill_info())
        for out_names in translated:
            out = [self.enumerizer.back_maps[0][y] for i,y in enumerate(out_names) if i < self.num_elems]
            # out = [elem_names_list[y] for i,y in enumerate(out_inds) if i < self.num_elems]
            print(out)
            
            # print(out,self.check_constraints(out,state))
            # print([state[x].get('value',None) for x in out])
            if(self.check_constraints(out,state)):
                yield out
            else:
                print("FAIL CONSTR", out)

    def skill_info(self):
        out = {}
        
        s = self.elem_slices

        ps, pg = self.pos_concepts.spec_concepts, self.pos_concepts.gen_concepts
        split_ps = [ps[:, s[i]:s[i+1]] for i in range(len(self.elem_slices)-1)]
        if(self.neg_ok):
            ns, ng = self.neg_concepts.spec_concepts, self.neg_concepts.gen_concepts
            split_ns = [ns[:, s[i]:s[i+1]] for i in range(len(self.elem_slices)-1)]

        for i, t in enumerate(self.elem_types):
            # print(i,i-self.num_elems,len(self.elem_types),list(self.expected_neighbors))
            key = "?sel" if i==0 else ("?arg%d" % (i-1) if i < self.num_elems else list(self.expected_neighbors)[i-self.num_elems]) 
            out[key] = {}
            out[key]["pos"] = self.enumerizer.undo_transform(split_ps[i].tolist()[0],t)
            if(self.neg_ok):
                out[key]["neg"] = self.enumerizer.undo_transform(split_ns[i].tolist()[0],t)

        return out
            



class VersionSpaceILP(object):
    def __init__(self,use_gen=False):
        self.spec_concepts = None
        self.gen_concepts = None
        self.ZERO = None
        self.unused_concepts = []
        self.use_gen = use_gen

    def ifit(self, x, y):
        # print("MOO",x,y)
        # print(x)
        # print(torch.tensor(x, dtype=torch.uint8))
        # x = torch.tensor(x, dtype=torch.uint8).view(1, -1)
        x = np.array(x, dtype=np.uint8).reshape((1, -1))
        # print("x", y)
        # print(x)
        clen = self.clen = x.shape[-1]
        if(y > 0):
            if(isinstance(self.spec_concepts, type(None))):
                self.spec_concepts = x
                self.gen_concepts = np.zeros(x.shape, dtype=np.uint8)
                self.ZERO = 0
                for _x, _y in self.unused_concepts:
                    self.ifit(_x, _y)
                self.unused_concepts = None
            else:
                # Generalize the specific concepts to incorperate the positive example
                self.spec_concepts[np.where(x != self.spec_concepts)] = self.ZERO

                if(self.use_gen):
                    # Prune the general concepts that are inconsistent with the positive example
                    gen_consistency = (self.gen_concepts == self.ZERO) | (self.gen_concepts == x)
                    self.gen_concepts = self.gen_concepts[gen_consistency.all(axis=1)]

        else:
            if(isinstance(self.spec_concepts, type(None))):
                self.unused_concepts.append((x, y))
                return
            elif(self.use_gen):
                # ### Start: Make the General Concepts more specific ####
                spec_inconsistency = ~((self.spec_concepts == self.ZERO) | (self.spec_concepts == x))
                gen_consistency = (self.gen_concepts == self.ZERO) | (self.gen_concepts == x)

                # The set of general concepts which are contain the negative example and must be specialized 
                to_specilz_concepts = self.gen_concepts[gen_consistency.all(axis=1)]
                # The set of general concepts which do not contain the negative example an so we leave alone
                to_keep_concepts = self.gen_concepts[(~gen_consistency).any(axis=1)]

                # The set of concepts with only one specific element which are present in
                # the most specific concept but not this negative example.
                specialization_candidates = torch.eye(clen, dtype=torch.uint8).view(1, clen, clen)*(spec_inconsistency*self.spec_concepts).view(-1, 1, clen)
                specialization_candidates = mask_select(specialization_candidates.view(-1, clen), spec_inconsistency.view(-1))

                # Specialize each concept (among those we found earlier that contain the negative example)
                #   creating all new possible general concepts for each element which is still general
                sc, tsc, = specialization_candidates.view(1, -1, clen), to_specilz_concepts.view(-1, 1, clen)
                possible_specialized_gens = torch.where((tsc == self.ZERO), sc, tsc).view(-1, clen)
                # specialized_gens = torch.where((tsc == self.ZERO) & gen_consistency.view(1,-1,clen), sc,tsc).view(-1,clen)

                # If a possible specialization is consistent with the non-specialized general concepts then keep it
                new_gen_consistency = (to_keep_concepts.view(1, -1, clen) == self.ZERO) | (to_keep_concepts.view(1, -1, clen) == possible_specialized_gens.view(-1, 1, clen))
                inconsistent_w_keepers = (~new_gen_consistency).any(dim=-1).all(dim=-1)
                specialized_gens = mask_select(possible_specialized_gens, inconsistent_w_keepers)

                # Join the tensors of non-specialized and specialized concepts
                self.gen_concepts = torch.cat([to_keep_concepts, specialized_gens], dim=0)

                # ### Start: Prune Specific Concepts ####
                # TODO: Never happens? What does it look like if we allow for multiple specific concepts?
                # self.spec_concepts = mask_select(self.spec_concepts, spec_inconsistency.any(dim=1))

        # print("GENERAL")
        # print(self.gen_concepts)
        # print("SPECIFIC")
        # print(self.spec_concepts, self.spec_concepts.shape)

        # print("---------------------------------")

    def check_match(self, x):
        '''Returns true if x is consistent with all general concepts and any specific concept'''
        # print(x)
        # print([len(z) for z in x])
        # print(len())
        # print(self.spec_concepts.size())
        # print(torch.tensor(x, dtype=torch.uint8))
        flat_x = list(itertools.chain(*x)) if isinstance(x[0],list) else x
        x = np.array(flat_x, dtype=np.uint8).reshape((1, -1))
        spec_consistency = ((self.spec_concepts == self.ZERO) | (self.spec_concepts == x)).all(axis=-1)
        out = spec_consistency.any()
        if(self.use_gen):
            gen_consistency = ((self.gen_concepts == self.ZERO) | (self.gen_concepts == x)).all(axis=-1)
            out = out & gen_consistency.all()
        return out.item()


def rename_values(x, mapping, exclude=[],rename_keys=False): 
    if(isinstance(x, dict)):
        if(rename_keys):
            return {rename_values(name, mapping,exclude): rename_values(val, mapping, exclude) for name, val in x.items()}
        else:
            return {name: rename_values(val, mapping,exclude) for name, val in x.items()}
    elif(isinstance(x, list)):
        return [rename_values(val, mapping,exclude) for val in x]
    elif(isinstance(x, tuple)):
        return tuple([rename_values(val, mapping,exclude) for val in x])
    elif(x in mapping and x not in exclude):
        return mapping[x]
    else:
        return x

from concept_formation.preprocessor import Preprocessor
class Enumerizer(Preprocessor):
    def __init__(self, start_num=0, force_add=[],attrs_independant=False,remove_attrs=[]):
        self.start_num = start_num
        self.attr_maps = {}
        self.force_add = force_add
        self.attrs_independant = attrs_independant
        self.type_lengths = {}
        self.remove_attrs = remove_attrs
        # self.attr_counts = {}
        self.back_maps = {}
        self.back_keys = {}
        self.keys = []

    def transform_values(self,values):
        assert not self.attrs_independant, \
            "attrs_independant, must be false. Mapping ambiguous."
        mapping = self.attr_maps[0]
        return [mapping[x] for x in values]

    '''Maps nominals to unsigned integers'''
    #TODO: MAKE IT ASSERT A FIXED REPRESENTATION SOMEHOW
    def transform(self, instances, force_map=None):
        """
        Transforms an list of instances.
        """
        if(isinstance(instances, dict)):
            instances = [instances]

        def instance_iter(instance):
            for key,value in instance.items():
                if(isinstance(value,(list,tuple))):
                    for i,v in enumerate(value):
                        yield (key,i), v
                else:
                    yield key,value
        
        # print(instances)
        enumerized_instances = []
        for i, instance in enumerate(instances):

            if("type" in instance):
                back_keys = self.back_keys[instance['type']] = [] 
            else:
                back_keys = None

            enumerized_instance = []
            for j,(k, value) in enumerate(instance_iter(instance)):
                # print(instance,k)
                if(k in self.remove_attrs):
                    continue

                if(back_keys is not None and j >= len(back_keys)):
                    back_keys.append(k)
                # print(value)
                if(force_map != None and value in force_map):
                    enumerized_instance.append(force_map[value])
                else:
                    key = (instance["type"],k) if self.attrs_independant else 0
                    if(key not in self.attr_maps):
                        # if(fail_on_grow):
                        #     raise ValueError("Dynamic")
                        self.attr_maps[key] = {x: i+self.start_num for i, x in enumerate(self.force_add)}
                        # self.attr_counts[key] = self.start_num
                        self.back_maps[key] = [None for i in range(self.start_num)] + self.force_add
                        self.keys.append(key)
                    attr_map = self.attr_maps[key]
                    if(value not in attr_map):
                        attr_map[value] = len(self.back_maps[key])
                        self.back_maps[key].append(value)
                        # self.attr_counts[key] += 1
                    enumerized_instance.append(attr_map[value])
            # print()
            if("type" in instance):
                inst_type = instance['type']
                if(inst_type in self.type_lengths):
                    assert self.type_lengths[inst_type] == len(enumerized_instance), \
                    "Got type %s with number of attributes %s (should be %s)" % \
                    (inst_type,len(enumerized_instance),self.type_lengths[inst_type])
                else:
                    self.type_lengths[inst_type] = len(enumerized_instance)

            enumerized_instances.append(enumerized_instance)
        return enumerized_instances

    def undo_transform(self, instance,typ):
        """
        Undoes a transformation to an instance.
        """
        

        d = {}

        if(self.attrs_independant):
            assert len(instance) == len(self.keys)
            for enum, key in zip(instance, self.keys):
                d[key] = self.back_maps[key][enum]
        else:
            # for key in self.back_maps[0]:
            #     print(key)
            # print("---------------------")
            back_map = self.back_maps[0]
            back_keys = self.back_keys[typ]
            # print(self.back_keys)
            # print(back_keys)
            # list_keys = {}
            for i,key in enumerate(back_keys):
                # print(back_keys[instance[i]])
                if(isinstance(key,tuple)):
                    lst = d.get(key[0],[None]*(key[1]+1))
                    if(len(lst) <= key[1]):
                        lst += [None]*(key[1]+1-len(lst))
                    # print(lst,key[1])
                    lst[key[1]] = back_map[instance[i]]
                    d[key[0]] = lst
                else:
                    # print(i,instance,back_keys,typ,len(instance))
                    # print(instance[i])
                    d[key] = back_map[instance[i]]


        return d

# def version_space():

WHERE_SUBLEARNERS = {
    'versionspace': VersionSpace,
    'fastmostspecific': FastMostSpecific,
    'mostspecific': MostSpecific,
    'stateresponselearner': StateResponseLearner,
    'relationallearner': RelationalLearner,
    'specifictogeneral': SpecificToGeneral
}

WHERE_STRATEGY = {
    'versionspace': "object",
    'fastmostspecific': "object",
    'mostspecific': "first_order",
    'stateresponselearner': "first_order",
    'relationallearner': "first_order",
    'specifictogeneral': "first_order"   
}


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

    # A4 A3 A2 A1
    #    B3 B2 B1
    #    --------
    # C4 C3 C2 C1

    state = {
        "line" : {
            "type" : "bloop",
            # 'value': "-",
            "above": ["B1","A1"],
            "below": ["C1",None],
            "left" : None,
            "right": None,
        },
        "A1": {
            "type" : "text",
            "value": 1,
            "above": [None,None],
            "below": ["B1","line"],
            "left" : "A2",
            "right": None,
        },
        "A2": {
            "type" : "text",
            "value": 2,
            "above": [None,None],
            "below": ["B2","line"],
            "left" : "A3",
            "right": "A1",
        },
        "A3": {
            "type" : "text",
            "value": 3,
            "above": [None,None],
            "below": ["B3","line"],
            "left" : "A4",
            "right": "A2",
        },
        "A4": {
            "type" : "text",
            "value": 3,
            "above": [None,None],
            "below": ["line","C4"],
            "left" : None,
            "right": "A3",
        },
        "B1": {
            "type" : "text",
            "value": 4,
            "above": ["A1", None],
            "below": ["line","C1"],
            "left" : "B2",
            "right": None,
        },
        "B2": {
            "type" : "text",
            "value": 5,
            "above": ["A2", None],
            "below": ["line","C2"],
            "left" : "B3",
            "right": "B1",
        },
        "B3": {
            "type" : "text",
            "value": 6,
            "above": ["A3", None],
            "below": ["line","C3"],
            "left" : None,
            "right": "B2",
        },
        "C1": {
            "type" : "text",
            "value": 7,
            "above": ["line","B1"],
            "below": [None,None],
            "left" : "C2",
            "right": None,
        },
        "C2": {
            "type" : "text",
            "value": 8,
            "above": ["line","B2"],
            "below": [None,None],
            "left" : "C3",
            "right": "C1",
        },
        "C3": {
            "type" : "text",
            "value": 9,
            "above": ["line","B3"],
            "below": [None,None],
            "left" : "C4",
            "right": "C2",
        },
        "C4": {
            "type" : "text",
            "value": 9,
            "above": ["line","A4"],
            "below": [None,None],
            "left" : None,
            "right": "C3",
        }
        
    }
    from agents.ModularAgent import StateMultiView
    state = StateMultiView("object",state)

    # enumer = Enumerizer(start_num=1, force_add=[None])

    # Av = VersionSpaceILP()
    # Bv = VersionSpaceILP()

    # inv = [(k, e) for k, e in state.items()][::-1]
    # for key, ele in inv:
    #     # for key,ele in state.items():
    #     ele_tensor = enumer.transform(ele)
    #     print("G", ele_tensor)
    #     if("A" in key):
    #         # Av.ifit(ele_tensor,1)
    #         Bv.ifit(ele_tensor, 0)
    #     elif("B" in key):
    #         # Av.ifit(ele_tensor,0)
    #         Bv.ifit(ele_tensor, 1)
    vs = VersionSpace(use_neg=True,use_neighbor_concepts=False)
    vs.ifit(["C1","A1","B1"],state,1)
    print(vs.check_match(["C1","A1","B1"],state), 1)
    print(vs.check_match(["C2","A2","B2"],state), 1)
    print(vs.check_match(["C3","A3","B3"],state), 1)
    vs.ifit(["C3","A3","B3"],state,1)
    print(vs.check_match(["C1","A1","B1"],state), 1)
    print(vs.check_match(["C2","A2","B2"],state), 1)
    print(vs.check_match(["C3","A3","B3"],state), 1)
    
    vs.ifit(["C1","B1","A1"],state,0)
    print("HERE0")
    for match in vs.get_matches(state):
        print(match)
    
    print("END0")
    # raise ValueError("END0")
    # vs.ifit(["C2","A2","B2"],state,1)
    # for match in vs.get_matches(state):
    #     print(match)
    vs.ifit(["C1","B2","A2"],state,0)
    vs.ifit(["C3","A3","B3"],state,1)
    vs.ifit(["C3","A3","A2"],state,0)

    print(vs.check_match(["C1","A1","B1"],state), 1)
    print(vs.check_match(["C1","B1","A1"],state), 0)
    print(vs.check_match(["C2","A2","B2"],state), 1)
    print(vs.check_match(["C1","B2","A2"],state), 0)
    print(vs.check_match(["C3","A3","B3"],state), 1)
    print(vs.check_match(["C3","A3","A2"],state), 0)
    print(vs.check_match(["C3","A1","A3"],state), 0)
    print(vs.check_match(["A2","A3","B3"],state), 0)
    print(vs.check_match(["C1","A3","C3"],state), 0)
    print(vs.check_match(["C1","A2","B2"],state), 0)
    print(vs.check_match(["C1","B2","B1"],state), 0)
    print(vs.check_match(['C1','A1','B3'],state), 0)
    print(vs.check_match(['B1','A2','B2'],state), 0)

    # print(rename_values(state,{"C1": "sel", "A1" : "arg0", "B1": "arg1"},False))
    print("HERE1")
    for match in vs.get_matches(state):
        print(match)

    # pprint._
    pprint(vs.skill_info())


    def timefunc(s, func, *args, **kwargs):
        """
        Benchmark *func* and print out its runtime.
        """
        print(s.ljust(20), end=" ")
        
        # time it
        print('{:>5.0f} ms'.format(min(repeat(lambda: func(*args, **kwargs),
                                              number=5, repeat=2)) * 1000))
        # return res

    def gen_test_get_matches(vs):
        def test_get_matches():
            l = [match for match in vs.get_matches(state)]
        return test_get_matches        


    
    

    # state_no_line = StateMultiView("object",{k:v for k,v in state.get_view("object").items() if k != "line"})
    state_l1 = StateMultiView("object",{name: {k:((v[0] if v[0] != "line" else v[1]) if isinstance(v,list) else v) for k,v in itm.items()}  for name,itm in state.get_view("object").items() if name != "line"})
    # state_l1 = state_l1.get_view("object")
    
    vs_tricky = VersionSpace(use_neg=False,use_neighbor_concepts=True)
    vs_tricky.ifit(["C2","A1","B1"],state_l1,1)
    # vs_tricky.ifit(["C2","A1","B1"],state_l1,1)
    # vs_tricky.ifit(["C2","A1","B1"],state_l1,1)
    pprint(vs_tricky.skill_info())
    # vs_tricky.ifit(["C3","A2","B2"],state_l1,1)
    vs_tricky.ifit(["C4","A3","B3"],state_l1,1)
    pprint(vs_tricky.skill_info())
    for match in vs_tricky.get_matches(state_l1):
        print("match",match)
    # raise ValueError("END")

    vs_tricky = VersionSpace(use_neg=False,use_neighbor_concepts=False)
    vs_tricky.ifit(["C2","A1"],state_l1,1)
    # vs_tricky.ifit(["C2","A1","B1"],state_l1,1)
    # vs_tricky.ifit(["C2","A1","B1"],state_l1,1)
    # pprint(vs_tricky.skill_info())
    # vs_tricky.ifit(["C3","A2","B2"],state_l1,1)
    vs_tricky.ifit(["C4","A3"],state_l1,1)
    pprint(vs_tricky.skill_info())
    for match in vs_tricky.get_matches(state_l1):
        print("match",match)

    # raise ValueError("")
    if(True):
        timefunc("match-3", gen_test_get_matches(vs))
        for match in vs.get_matches(state):
            print(match)

        vs2 = VersionSpace(use_neg=True)
        vs2.ifit(["C1","A1","B1","A2","C2"],state,1)
        vs2.ifit(["C1","B1","A1","A2","C2"],state,0)
        vs2.ifit(["C2","A2","B2","A3","C3"],state,1)
        vs2.ifit(["C1","B2","A2","A3","C3"],state,0)
        # vs2.ifit(["C3","A3","A2","A3","C3"],state,0)
        for match in vs2.get_matches(state):
            print(match)
        timefunc("match-5", gen_test_get_matches(vs2))

        
        vs3 = VersionSpace(use_neg=True)
        vs3.ifit(["A1","B1","C1","A2","B2","C2","A3","B3","C3",],state,1)
        for match in vs3.get_matches(state):
            print(match)
        timefunc("match-9", gen_test_get_matches(vs3))




        vs4 = VersionSpace(use_neg=True)
        vs4.ifit(["A1","B1","C1","A2","B2","C2","A3","B3","C3","A4","C4"],state,1)
        for match in vs4.get_matches(state):
            print(match)
        timefunc("match-11", gen_test_get_matches(vs4))

    # vs4 = VersionSpace(use_neg=True)
    # vs4.ifit(["A1","B1","C1","A2","B2","C2","A3","B3","C3","A4","C4"],state,1)
    # for match in vs4.get_matches(state):
    #     print(match)
    # pprint(vs_tricky.skill_info())
    # pprint(state_l1.get_view("object"))

    # print(vs.enumerizer.undo_transform(vs.pos_concepts.spec_concepts.tolist()[0]))
    # print(vs.enumerizer.undo_transform(vs.pos_concepts.spec_concepts.tolist()[1]))
    # print(vs.enumerizer.undo_transform(vs.pos_concepts.spec_concepts.tolist()[2]))
    # print()
    # print(vs.pos_concepts.spec_concepts.view(3,-1))
    # print(vs.pos_concepts.gen_concepts.view(3,-1))
    # print(vs.neg_concepts.spec_concepts.view(3,-1))
    # print(vs.neg_concepts.gen_concepts.view(3,3,-1))
    # print()
    # print()
    # pprint(enumer.batch_undo(enumer.batch_transform(state.values())))
