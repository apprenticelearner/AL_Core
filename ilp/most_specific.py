from pprint import pprint

from concept_formation.cobweb3 import Cobweb3Node
from concept_formation.structure_mapper import StructureMapper
from concept_formation.structure_mapper import rename_flat
from concept_formation.structure_mapper import contains_component
from ilp.fo_planner import Operator
from ilp.fo_planner import build_index
from ilp.fo_planner import subst

from ilp.base_ilp import BaseILP

global my_gensym_counter
my_gensym_counter = 0


def gensym():
    """
    Generates unique names for naming renaming apart objects.

    :return: a unique object name
    :rtype: '?o'+counter
    """
    global my_gensym_counter
    my_gensym_counter += 1
    return '?o' + str(my_gensym_counter)


class MostSpecific(BaseILP):

    def __init__(self):
        self.pos = set()
        self.target_types = []

    def get_matches(self, X, constraints=None):
        for t in self.pos:
            print(t)
            yield t
        
    def fit(self, T, X, y):

        self.target_types = T[0]

        #ignore X and save the positive T's.
        for i,t in enumerate(T):
            if y[i] == 1:
                self.pos.add(t)

    def __repr__(self):
        tt = []
        try:
            for t in self.pos:
                t = ["('name', '?foa%i')=='%s'" % (i,v) for i,v in enumerate(t)]
                tt.append(t)
        except:
            pass

        tt = ["(" + ", ".join(t) + ")" for t in tt]
        return repr(" or ".join(tt))

def ground(attr):
    new = []
    for ele in attr:
        if isinstance(ele, tuple):
            new.append(ground(ele))
        elif isinstance(ele, str):
            if ele[0] == '?':
                new.append(ele[1:])
            else:
                new.append(ele)
    return tuple(new)


class SimStudentWhere(BaseILP):

    def __init__(self):
        self.pos = set()
        self.operator = None

    def get_matches(self, x, constraints=None):
        if self.operator is None:
            return

        # print("GETTING MATCHES")
        grounded = [(ground(a), x[a]) for a in x if (isinstance(a, tuple))]
        index = build_index(grounded)

        for m in self.operator.match(index):
            result = tuple(['?' + subst(m, ele)
                            for ele in self.operator.name[1:]])
            # result = tuple(['?' + m[e] for e in self.target_types])
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

    def fit(self, T, X, y):
        self.target_types = ['?foa%s' % i for i in range(len(T[0]))]

        # ignore X and save the positive T's.
        for i, t in enumerate(T):
            if y[i] == 1:
                self.pos.add(t)

        # pprint(X)
        concept = Cobweb3Node()
        # self.tree = TrestleTree(structure_map_internally=False)
        # self.target_types = T[0]

        # print("BUILDING UP")
        # print(len(T))
        for i, t in enumerate(T):
            if y[i] == 1:

                x = {a: X[i][a] for a in X[i]}
                # print("X BEFORE FILTERING OUT")
                # pprint(x)

                # x = {a: X[i][a] for a in X[i] if not isinstance(a, tuple) or
                #     a[0] == 'orderedlist' or a[0] == 'value' or
                #     a[0] == 'name' or a[0] == 'type' or a[0] == 'haselement'}

                # need to extract only elements that are parents of one of the
                # foas. All other elements and their relations should be
                # removed.
                eles = set([field for field in t])
                prior_count = 0
                while len(eles) - prior_count > 0:
                    prior_count = len(eles)
                    for a in x:
                        if isinstance(a, tuple) and a[0] == 'haselement':
                            if a[2] in eles:
                                eles.add(a[1])
                            # if self.matches(eles, a):
                            #     names = get_attribute_components(a)
                            #     eles.update(names)

                x = {a: x[a] for a in x
                     if self.matches(eles, a)}

                foa_mapping = {field: 'foa%s' % j for j, field in enumerate(t)}
                # for j,field in enumerate(t):
                #     x[('foa%s' % j, field)] = True
                x = rename_flat(x, foa_mapping)
                # pprint(x)

                # print("adding:")

                sm = StructureMapper(concept, gensym=gensym)
                x = sm.transform(x)
                # pprint(x)
                concept.increment_counts(x)
                # pprint(concept.av_counts)
                remove = []
                for attr in concept.av_counts:
                    if len(concept.av_counts[attr]) > 1:
                        remove.append(attr)
                    else:
                        for val in concept.av_counts[attr]:
                            if concept.av_counts[attr][val] != concept.count:
                                remove.append(attr)

                for attr in remove:
                    del concept.av_counts[attr]

                # print("done!")

        pattern_instance = {attr: val for attr in concept.av_counts for val in
                            concept.av_counts[attr]}
        foa_mapping = {'foa%s' % j: '?foa%s' % j for j in range(len(t))}
        pattern_instance = rename_flat(pattern_instance, foa_mapping)
        # pprint(pattern_instance)
        self.operator = Operator(tuple(['Rule'] + self.target_types),
                                 [(a, pattern_instance[a])
                                  for a in pattern_instance], [])
        # self.pattern = []
        # for attr in concept.av_counts:
        #     for val in concept.av_counts[attr]:
        #         self.pattern.append((attr, val))
        # pprint(self.pattern)
