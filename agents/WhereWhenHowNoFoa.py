from pprint import pprint
from random import random
from random import choice

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
from concept_formation.trestle import TrestleTree
from concept_formation.structure_mapper import rename_flat

from agents.BaseAgent import BaseAgent
from learners.WhereLearner import SpecificToGeneral
from learners.WhereLearner import RelationalLearner
from learners.WhereLearner import MostSpecific
from learners.WhenLearner import when_learners
from planners.fo_planner import FoPlanner
# from ilp.fo_planner import Operator

from planners.rulesets import functionsets
from planners.rulesets import featuresets

search_depth = 1
epsilon = .9


class WhereWhenHowNoFoa(BaseAgent):
    """
    This is the basis for the 2 mechanism model.
    """
    def __init__(self, action_set):
        # self.where = SpecificToGeneral
        self.where = RelationalLearner
        # self.where = MostSpecific
        # self.when = 'naive bayes'
        self.when = 'always true'
        self.skills = {}
        self.examples = {}
        self.action_set = action_set.name

    def request(self, state):
        print("REQUEST RECEIVED")
        tup = Tuplizer()
        flt = Flattener()

        state = flt.transform(tup.transform(state))

        # new = {}
        # for attr in state:
        #     if (isinstance(attr, tuple) and attr[0] == 'value'):
        #         new[('editable', attr[1])] = state[attr] == ''
        #         for attr2 in state:
        #             if (isinstance(attr2, tuple) and attr2[0] == 'value'):
        #                 if (attr2 == attr or attr < attr2 or (state[attr] == ""
        #                                                       or state[attr2]
        #                                                       == "")):
        #                     continue
        #                 if (state[attr] == state[attr2]):
        #                     new[('eq', attr, attr2)] = True
        # state.update(new)

        kb = FoPlanner([(self.ground(a),
                         state[a].replace('?', 'QM') if
                         isinstance(state[a], str) else
                         state[a])
                        for a in state], featuresets[self.action_set])
        kb.fc_infer(depth=1, epsilon=epsilon)
        state = {self.unground(a): v.replace("QM", "?") if isinstance(v, str)
                 else v for a, v in kb.facts}

        # pprint(state)

        # compute features

        # for attr, value in self.compute_features(state):
        #     state[attr] = value

        skillset = []
        for label in self.skills:
            for exp in self.skills[label]:
                pos = self.skills[label][exp]['where'].num_pos()
                neg = self.skills[label][exp]['where'].num_neg()

                skillset.append((pos / (pos + neg), pos + neg,
                                 random(), label, exp,
                                 self.skills[label][exp]))
        skillset.sort(reverse=True)

        print('####SKILLSET####')
        pprint(skillset)
        print('####SKILLSET####')

        # used for grounding out plans, don't need to build up each time.
        kb = FoPlanner([(self.ground(a),
                         state[a].replace('?', 'QM') if
                         isinstance(state[a], str) else
                         state[a])
                        for a in state], functionsets[self.action_set])
        kb.fc_infer(depth=search_depth, epsilon=epsilon)

        # print(kb)

        for _, _, _, label, (exp, input_args), skill in skillset:

            print()
            print("TRYING:", label, exp)
            # print("Conditions:")
            # pprint(skill['where'].operator.conditions)

            # Continue until we either do something with the rule. For example,
            # generate an SAI or determine that the rule doesn't match. If we
            # get some kind of failure, such as being unable to execute an
            # action sequence, then we want to learn from that and try again.
            failed = True

            while failed:

                failed = False
                for m in skill['where'].get_matches(state, epsilon=epsilon):
                    if len(m) != len(set(m)):
                        print("GENERATED MATCH WITH TWO VARS BOUND TO ",
                              "SAME THING")
                        continue

                    print("MATCH FOUND", label, exp, m)
                    vmapping = {'?foa' + str(i): ele for i, ele in enumerate(m)}
                    mapping = {'foa' + str(i): ele for i, ele in enumerate(m)}

                    r_exp = list(rename_flat({exp: True}, vmapping))[0]
                    r_state = rename_flat(state, {mapping[a]: a for a in mapping})

                    # pprint(r_state)

                    # pprint(r_state)

                    rg_exp = []
                    for ele in r_exp:
                        if isinstance(ele, tuple):
                            # kb = FoPlanner([(self.ground(a),
                            #                  state[a].replace('?', 'QM') if
                            #                  isinstance(state[a], str) else
                            #                  state[a])
                            #                 for a in state], functionsets[self.action_set])
                            for vm in kb.fc_query([(self.ground(ele), '?v')],
                                                  max_depth=0,
                                                  epsilon=epsilon):
                                # if vm['?v'] == '':
                                #     raise Exception("Should not be an empty str")
                                if vm['?v'] != '':
                                    rg_exp.append(vm['?v'])
                                break
                        else:
                            rg_exp.append(ele)

                    if len(rg_exp) != len(r_exp):
                        print("FAILED TO FIRE RULE")
                        print(rg_exp, 'from', r_exp)
                        continue

                        # # add neg to where
                        # skill['where'].ifit(m, state, 0)

                        # # add neg to when
                        # foa_mapping = {field: 'foa%s' % j for j, field in
                        #                enumerate(m)}
                        # neg_x = rename_flat(state, foa_mapping)
                        # skill['when'].ifit(neg_x)

                        # failed = True
                        # break

                    print("predicting")
                    # pprint(r_state)

                    # c = skill['when'].categorize(r_state)
                    p = skill['when'].predict([r_state])[0]

                    # print("###CATEGORIZED CONCEPT###")
                    # print(c)
                    # pprint(c.av_counts)
                    # print(c.predict('correct'))

                    if p == 0:
                        print("predicting FAIL")
                        continue
                    print("predicting FIRE")

                    # if not c.predict('correct'):
                    #     print("predicting FAIL")
                    #     continue
                    # print("predicting FIRE")

                    # print("###TREE###")
                    # print(skill['when'])

                    # pprint(r_exp)
                    # pprint(rg_exp)

                    # assert self.explains_sai(kb, r_exp, rg_exp)

                    response = {}
                    response['label'] = label
                    response['selection'] = rg_exp[1]
                    response['action'] = rg_exp[2]
                    response['inputs'] = {a: rg_exp[3+i] for i, a in
                                          enumerate(input_args)}
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
        print('trying', exp, 'with', sai)
        if len(exp) != len(sai):
            return

        goals = []
        for i, e in enumerate(exp):
            if e == sai[i]:
                continue
            else:
                goals.append((e, sai[i]))

        # print(goals)

        for m in kb.fc_query(goals, max_depth=0, epsilon=epsilon):
            yield m

        # for f in kb.facts:
        #     if isinstance(f[0], tuple) and f[0][0] == 'value':
        #         print(f)
        # print(kb.facts)

    def compute_exp_depth(self, exp):
        if isinstance(exp, tuple):
            return 1 + max([self.compute_exp_depth(sub) for sub in exp])
        return 0

    def train(self, state, label, foas, selection, action, inputs, correct):

        # label = 'math'

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
        example['flat_state'] = flt.transform(tup.transform(state))
        print('SAI:', selection, action, inputs)

        # print('State:')
        # pprint(example['state'])
        # print('Flat State:')
        # pprint(example['flat_state'])

        # new = {}
        # for attr in example['flat_state']:
        #     if (isinstance(attr, tuple) and attr[0] == 'value'):
        #         new[('editable', attr[1])] = example['flat_state'][attr] == ''

        #         for attr2 in example['flat_state']:
        #             if (isinstance(attr2, tuple) and attr2[0] == 'value'):
        #                 if (attr2 == attr or attr < attr2 or
        #                     (example['flat_state'][attr] == "" or
        #                      example['flat_state'][attr2] == "")):
        #                     continue
        #                 if ((example['flat_state'][attr] ==
        #                      example['flat_state'][attr2])):
        #                     new[('eq', attr, attr2)] = True

        # example['flat_state'].update(new)

        kb = FoPlanner([(self.ground(a),
                         example['flat_state'][a].replace('?', 'QM') if
                         isinstance(example['flat_state'][a], str) else
                         example['flat_state'][a])
                        for a in example['flat_state']],
                       featuresets[self.action_set])
        kb.fc_infer(depth=1, epsilon=epsilon)
        example['flat_state'] = {self.unground(a): v.replace("QM", "?") if
                                 isinstance(v, str) else v for a, v in
                                 kb.facts}

        pprint(example['flat_state'])

        if label not in self.skills:
            self.skills[label] = {}

        explainations = []
        secondary_explainations = []

        # the base explaination (the constants)
        input_args = tuple(sorted([arg for arg in inputs]))
        sai = ('sai', selection, action, *[inputs[a] for a in input_args])

        # Need to do stuff with features here too.

        # used for grounding out plans, don't need to build up each time.
        kb = FoPlanner([(self.ground(a),
                         example['flat_state'][a].replace('?', 'QM') if
                         isinstance(example['flat_state'][a], str) else
                         example['flat_state'][a])
                        for a in example['flat_state']], functionsets[self.action_set])
        kb.fc_infer(depth=search_depth, epsilon=epsilon)

        for exp, iargs in self.skills[label]:
            # kb = FoPlanner([(self.ground(a),
            #                  example['flat_state'][a].replace('?', 'QM') if
            #                  isinstance(example['flat_state'][a], str) else
            #                  example['flat_state'][a])
            #                 for a in example['flat_state']], functionsets[self.action_set])
            for m in self.explains_sai(kb, exp, sai):
                print("COVERED", exp, m)

                # Need to check if it would have been actully generated
                # under where and when.

                r_exp = self.unground(list(rename_flat({exp: True}, m))[0])

                args = self.get_vars(exp)

                if len(args) != len(m):
                    print("EXP not same length")
                    continue

                grounded = True
                for ele in m:
                    if not isinstance(m[ele], str):
                        grounded = False
                        break
                if not grounded:
                    print("Pattern not fully grounded")
                    continue

                # foa_vmapping = {field: '?foa%s' % j
                #                 for j, field in enumerate(args)}
                # foa_mapping = {field: 'foa%s' % j for j, field in
                #                enumerate(args)}

                t = tuple([m["?foa%s" % i].replace("QM", "?") for i in
                           range(len(m))])

                if len(t) != len(set(t)):
                    print("TWO VARS BOUND TO SAME")
                    continue

                secondary_explainations.append(r_exp)

                print("This is my T:", t)

                if not self.skills[label][(exp,
                                           iargs)]['where'].check_match(t,
                                                                     example['flat_state']):
                    continue

                print("####### SUCCESSFUL WHERE MATCH########")

                # x = rename_flat(example['flat_state'], foa_mapping)
                # c = self.skills[label][(exp, iargs)]['when'].categorize(x)
                # if not c.predict('correct'):
                #     continue

                print("ADDING", r_exp)
                explainations.append(r_exp)

        if len(explainations) == 0 and len(secondary_explainations) > 0:
            explainations.append(choice(secondary_explainations))
            # explainations.append(secondary_explanation)

        elif len(explainations) == 0:
            # kb = FoPlanner([(self.ground(a),
            #                  example['flat_state'][a].replace('?', 'QM') if
            #                  isinstance(example['flat_state'][a], str) else
            #                  example['flat_state'][a])
            #                 for a in example['flat_state']], functionsets[self.action_set])

            selection_exp = selection
            for sel_match in kb.fc_query([('?selection', selection)],
                                         max_depth=0,
                                         epsilon=epsilon):
                selection_exp = sel_match['?selection']
                break

            input_exps = []

            for a in input_args:
                iv = inputs[a]
                # kb = FoPlanner([(self.ground(a),
                #                  example['flat_state'][a].replace('?', 'QM') if
                #                  isinstance(example['flat_state'][a], str) else
                #                  example['flat_state'][a])
                #                 for a in example['flat_state']], functionsets[self.action_set])
                input_exp = iv
                print('trying to explain', [((a, '?input'), iv)])


                # TODO not sure what the best approach is for choosing among
                # the possible explanations. Perhaps we should choose more than
                # one. Maybe the shortest (less deep). 

                # f = False
                possible = []
                for iv_m in kb.fc_query([((a, '?input'), iv)],
                                        max_depth=0,
                                        epsilon=epsilon):

                    # input_exp = (a, iv_m['?input'])
                    possible.append((a, iv_m['?input']))
                    # print("FOUND!", input_exp)
                    # f = True
                    # break

                possible = [(self.compute_exp_depth(p), random(), p) for p in
                            possible]
                possible.sort()
                print("FOUND!")
                pprint(possible)


                if len(possible) > 0:
                    _, _, input_exp = possible[0]
                    # input_exp = choice(possible)

                # if not f:
                #     print()
                #     print("FAILED TO EXPLAIN INPUT PRINTING GOAL AND FACTS")
                #     print("GOAL:", ((a, '?input'), iv))

                #     for f in kb.facts:
                #         if f[0][0] == 'value':
                #             print(f)
                #     from time import sleep
                #     sleep(30)

                #     # pprint(kb.facts)

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
                mg_h = self.extract_mg_h(r_exp[0])
                w_args = tuple(['?foa%s' % j for j, _ in enumerate(args)])

                self.skills[label][r_exp] = {}
                self.skills[label][r_exp]['where'] = self.where(args=w_args,
                                                                constraints=mg_h)
                                                                # initial_h=mg_h)
                self.skills[label][r_exp]['when'] = when_learners[self.when]()

            print('where learning for ', exp)
            self.skills[label][r_exp]['where'].ifit(args,
                                                    example['flat_state'],
                                                    example['correct'])
            print('done where learning')

            # TODO
            # Need to add computed features.
            # need to rename example with foa's that are not variables
            x = rename_flat(example['flat_state'], foa_mapping)
            # x['correct'] = example['correct']

            print('ifitting')
            # pprint(x)
            # self.skills[label][r_exp]['when'].ifit(x)
            self.skills[label][r_exp]['when'].ifit(x, example['correct'])
            print('done ifitting')

            # print("###UPDATED TREE###")
            # print(self.skills[label][r_exp]['when'])

        # check for subsuming explainations (alternatively we could probably
        # just order the explainations by how many examples they cover

    def extract_mg_h(self, sai):
        """
        Given an SAI, this find the most general pattern that will generate a
        match.

        E.g., ('sai', ('name', '?foa0'), 'UpdateTable', ('value', '?foa1'))
        will yield: {('name', '?foa0'), ('value', '?foa1')}
        """
        h = set()
        for ele in sai:
            if isinstance(ele, tuple) and len(ele) == 2 and ele[1][0] == '?':
                h.add(tuple(list(ele) + [ele[1] + 'val']))
            elif isinstance(ele, tuple):
                h.update(self.extract_mg_h(ele))

        return frozenset(h)

    def check(self, state, selection, action, inputs):
        return False
