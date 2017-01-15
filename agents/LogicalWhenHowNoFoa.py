from pprint import pprint
from random import random

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
from concept_formation.trestle import TrestleTree
from concept_formation.structure_mapper import rename_flat

from agents.BaseAgent import BaseAgent
from ilp.most_specific import SimStudentWhere
from ilp.most_specific import MostSpecific
from ilp.fo_planner import FoPlanner
# from ilp.fo_planner import Operator
from ilp.fo_planner import rb_rules

search_depth = 3
epsilon = .9

class LogicalWhenHow(BaseAgent):
    """
    This is the basis for the 2 mechanism model.
    """
    def __init__(self, action_set):
        self.where = SimStudentWhere
        # self.where = MostSpecific
        self.when = TrestleTree
        self.skills = {}
        self.examples = {}

    def request(self, state):
        tup = Tuplizer()
        flt = Flattener()

        state = flt.transform(flt.transform(tup.transform(state)))
        new = {}
        for attr in state:
            if (isinstance(attr, tuple) and attr[0] == 'value'):
                new[('editable', attr[1])] = state[attr] == ''
                for attr2 in state:
                    if (isinstance(attr2, tuple) and attr2[0] == 'value'):
                        if (attr2 == attr or attr < attr2 or state[attr] == ""
                            or state[attr2] == ""):
                            continue
                        if (state[attr] == state[attr2]):
                            new[('eq', attr, attr2)] = True
        state.update(new)

        # compute features

        # for attr, value in self.compute_features(state):
        #     state[attr] = value

        skillset = []
        for label in self.skills:
            for exp in self.skills[label]:
                skillset.append((len(self.skills[label][exp]['where']),
                                 random(), label, exp,
                                 self.skills[label][exp]))
        skillset.sort(reverse=True)

        pprint(skillset)

        # used for grounding out plans, don't need to build up each time.
        kb = FoPlanner([(self.ground(a),
                         state[a].replace('?', 'QM') if
                         isinstance(state[a], str) else
                         state[a])
                        for a in state], rb_rules)
        # print(kb)

        for _, _, label, (exp, input_args), skill in skillset:
            print(skill['where'].operator)
            for m in skill['where'].get_matches(state):
                if len(m) != len(set(m)):
                    print("GENERATED MATCH WITH TWO VARS BOUND TO SAME THING")
                    continue
                    
                print("MATCH FOUND", label, exp, m)
                vmapping = {'?foa' + str(i): ele for i, ele in enumerate(m)}
                mapping = {'foa' + str(i): ele for i, ele in enumerate(m)}

                r_exp = list(rename_flat({exp: True}, vmapping))[0]
                r_state = rename_flat(state, {mapping[a]: a for a in mapping})

                # pprint(r_state)

                # pprint(r_state)

                print("predicting")
                c = skill['when'].categorize(r_state)
                # pprint(c.av_counts)
                print(c.predict('correct'))

                if not c.predict('correct'):
                    print("predicting FAIL")
                    continue
                print("predicting FIRE")

                rg_exp = []
                for ele in r_exp:
                    if isinstance(ele, tuple):
                        kb = FoPlanner([(self.ground(a),
                                         state[a].replace('?', 'QM') if
                                         isinstance(state[a], str) else
                                         state[a])
                                        for a in state], rb_rules)
                        for vm in kb.fc_query([(self.ground(ele), '?v')],
                                              max_depth=search_depth, epsilon=epsilon):
                            rg_exp.append(vm['?v'])
                            break
                    else:
                        rg_exp.append(ele)

                if len(rg_exp) != len(r_exp):
                    print("FAILED TO FIRE RULE")
                    print(rg_exp, 'from', r_exp)
                    continue

                # pprint(r_exp)
                # pprint(rg_exp)

                # assert self.explains_sai(kb, r_exp, rg_exp)

                response = {}
                response['label'] = label
                response['selection'] = rg_exp[1]
                response['action'] = rg_exp[2]
                response['inputs'] = {a: rg_exp[3+i] for i,a in enumerate(input_args)}
                # response['inputs'] = list(rg_exp[3:])
                response['foas'] = []
                # pprint(response)
                return response

        return {}

    def ground(self, arg):
        if isinstance(arg, tuple):
            return tuple(self.ground(e) for e in arg)
        elif isinstance(arg, str):
            return arg.replace('?', 'QM')
        else:
            return arg

    def unground(self, arg):
        if isinstance(arg, tuple):
            return tuple(self.unground(e) for e in arg)
        elif isinstance(arg, str):
            return arg.replace('QM', '?')
        else:
            return arg

    def replace_vars(self, arg, i=0):
        if isinstance(arg, tuple):
            l = []
            for e in arg:
                replaced, i = self.replace_vars(e, i)
                l.append(replaced)
            return tuple(l), i
        elif isinstance(arg, str) and len(arg) > 0 and arg[0] == '?':
            return '?foa%s' % (str(i)), i+1
        else:
            return arg, i

    def get_vars(self, arg):
        if isinstance(arg, tuple):
            l = []
            for e in arg:
                for v in self.get_vars(e):
                    if v not in l:
                        l.append(v)
            return l
            # return [v for e in arg for v in self.get_vars(e)]
        elif isinstance(arg, str) and len(arg) > 0 and arg[0] == '?':
            return [arg]
        else:
            return []

    def explains_sai(self, kb, exp, sai):
        if len(exp) != len(sai):
            return
        # print('trying', exp, 'with', sai)

        goals = []
        for i, e in enumerate(exp):
            if e == sai[i]:
                continue
            else:
                goals.append((e, sai[i]))

        # print(goals)

        for m in kb.fc_query(goals, max_depth=search_depth, epsilon=epsilon):
            yield m

        # for f in kb.facts:
        #     if isinstance(f[0], tuple) and f[0][0] == 'value':
        #         print(f)
        # print(kb.facts)

    def train(self, state, label, foas, selection, action, inputs, correct):

        # create example dict
        example = {}
        example['state'] = state
        example['label'] = label
        example['selection'] = selection
        example['action'] = action
        example['inputs'] = inputs
        example['correct'] = correct

        tup = Tuplizer()
        flt = Flattener()
        example['flat_state'] =flt.transform(flt.transform(tup.transform(state)))
        print('SAI:',selection,action,inputs)
        print('State:')
        pprint(example['state'])
        print('Flat State:')
        pprint(example['flat_state'])


        new = {}
        for attr in example['flat_state']:
            if (isinstance(attr, tuple) and attr[0] == 'value'):
                new[('editable', attr[1])] = example['flat_state'][attr] == ''

                for attr2 in example['flat_state']:
                    if (isinstance(attr2, tuple) and attr2[0] == 'value'):
                        if (attr2 == attr or attr < attr2 or
                            example['flat_state'][attr] == "" or
                            example['flat_state'][attr2] == ""):
                            continue
                        if (example['flat_state'][attr] == example['flat_state'][attr2]):
                            new[('eq', attr, attr2)] = True

        example['flat_state'].update(new)

        # pprint(example['flat_state'])

        if label not in self.skills:
            self.skills[label] = {}

        explainations = []

        # the base explaination (the constants)
        input_args = tuple(sorted([arg for arg in inputs]))
        sai = ('sai', selection, action, *[inputs[a] for a in input_args])

        # Need to do stuff with features here too.

        # kb = FoPlanner([(self.ground(a),
        #                  example['flat_state'][a].replace('?', 'QM') if
        #                  isinstance(example['flat_state'][a], str) else
        #                  example['flat_state'][a])
        #                 for a in example['flat_state']], rb_rules)

        for exp in self.skills[label]:
            kb = FoPlanner([(self.ground(a),
                             example['flat_state'][a].replace('?', 'QM') if
                             isinstance(example['flat_state'][a], str) else
                             example['flat_state'][a])
                            for a in example['flat_state']], rb_rules)
            for m in self.explains_sai(kb, exp, sai):
                print("COVERED", exp, m)
                r_exp = self.unground(list(rename_flat({exp: True}, m))[0])
                explainations.append(r_exp)

        if len(explainations) == 0:
            kb = FoPlanner([(self.ground(a),
                             example['flat_state'][a].replace('?', 'QM') if
                             isinstance(example['flat_state'][a], str) else
                             example['flat_state'][a])
                            for a in example['flat_state']], rb_rules)

            selection_exp = selection
            for sel_match in kb.fc_query([('?selection', selection)],
                                         max_depth=search_depth, epsilon=epsilon):
                selection_exp = sel_match['?selection']
                break

            input_exps = []
            for a in input_args:
                iv = inputs[a]
                kb = FoPlanner([(self.ground(a),
                                 example['flat_state'][a].replace('?', 'QM') if
                                 isinstance(example['flat_state'][a], str) else
                                 example['flat_state'][a])
                                for a in example['flat_state']], rb_rules)
                input_exp = iv
                for iv_m in kb.fc_query([((a, '?input'), iv)],
                                        max_depth=search_depth, epsilon=epsilon):
                    input_exp = (a, iv_m['?input'])
                    break
                input_exps.append(input_exp)

            explainations.append(self.unground(('sai', selection_exp, action,
                                                *input_exps)))

        for exp in explainations:
            args = self.get_vars(exp)
            foa_vmapping = {field: '?foa%s' % j
                            for j, field in enumerate(args)}
            foa_mapping = {field: 'foa%s' % j for j, field in enumerate(args)}
            x = rename_flat({exp: True}, foa_vmapping)
            r_exp = (list(x)[0], input_args)
            # r_exp = self.replace_vars(exp)
            # print("REPLACED")
            # print(exp)
            # print(r_exp)

            if r_exp not in self.skills[label]:
                self.skills[label][r_exp] = {}
                self.skills[label][r_exp]['where'] = self.where()
                self.skills[label][r_exp]['when'] = self.when()

            print('where learning')
            self.skills[label][r_exp]['where'].ifit(args,
                                                    example['flat_state'],
                                                    example['correct'])
            print('done where learning')

            # TODO
            # Need to add computed features.
            # need to rename example with foa's that are not variables
            x = rename_flat(example['flat_state'], foa_mapping)
            x['correct'] = example['correct']

            print('ifitting')
            self.skills[label][r_exp]['when'].ifit(x)
            print('done ifitting')

        # check for subsuming explainations (alternatively we could probably
        # just order the explainations by how many examples they cover

    def check(self, state, selection, action, inputs):
        return False
