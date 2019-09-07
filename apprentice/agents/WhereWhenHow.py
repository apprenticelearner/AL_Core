from pprint import pprint
from itertools import combinations
from random import choice

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer

from agents.utils import gen_varnames
from agents.utils import compute_features
from agents.utils import weighted_choice
from agents.BaseAgent import BaseAgent
from planners.action_planner import ActionPlanner
from learners.WhenLearner import get_when_learner
from learners.HowLearner import get_how_learner
# from agents.WhatLearner import what_learners
# from agents.WhatLearner import GrammarLearner
# from agents.WhatLearner import NeuralNetLearnedFeatures
# from agents.WhatLearner import GrammarLearnedFeatures
from learners.WhereLearner import MostSpecific
from learners.WhereLearner import SpecificToGeneral


class WhereWhenHow(BaseAgent):
    """
    This is the basis for the 3 learning mechanism model. It accepts three classes
    for the where, when, and how learning. Where and and When are both
    classifiers. How learning is a form of planner.
    """
    def __init__(self, action_set, when="decision tree", how="simstudent",
                 when_params=None, how_params=None):

        self.action_set = action_set
        self.planner = ActionPlanner(action_set, act_params=how_params)

        self.what = None

        #  Use NN features
        # self.what = NeuralNetLearnedFeatures()

        #  Use Grammar features
        # self.what = GrammarLearnedFeatures()

        # # print hand features
        # with open("/Users/cmaclell/Projects/simstudent/AuthoringTools/java/Projects/articleSelectionTutor/massproduction-templates/article_sentences.csv") as fin:
        #     lines = {}
        #     for line in fin:
        #         f = {}
        #         f.update(compute_features({"?val": line},
        #                                     self.action_set.get_feature_dict()))
        #         lines[line] = f
        #     pprint(lines)

        # # Learn grammar
        # self.what = GrammarLearner()
        # with open("/Users/cmaclell/Projects/simstudent/AuthoringTools/java/Projects/articleSelectionTutor/massproduction-templates/article_sentences.csv") as fin:
        #     lines = [line for line in fin]
        # words = [[c for c in w] for line in lines
        #          for w in re.split(r'[ ,;.\?\-"]+', line.strip().lower())
        #          if w != ""]
        # test_seqs = words + [[c for c in line.strip().lower()] for line in lines]
        # self.what.fit(test_seqs)

        # with open("/Users/cmaclell/Downloads/article_grammar.pickle", 'wb') as fout:
        #     pickle.dump(self.what, fout)
        # print("PICKLED")

        #  Load grammar
        # with open("/Users/cmaclell/Downloads/article_grammar.pickle", 'rb') as f:
        #     self.what = pickle.load(f)

        self.how = get_how_learner(how)
        self.how_instances = {}

        # self.where = Foil
        self.where = MostSpecific
        # self.where = SimStudentWhere
        # self.where = iFoil

        # self.where = Aleph

        self.when = when
        self.when_params = when_params

        self.skills = {}
        self.examples = {}

    def request(self, state):
        ff = Flattener()
        tup = Tuplizer()

        state = tup.transform(state)
        state = ff.transform(state)

        pprint(state)

        # foa_state = {attr: state[attr] for attr in state
        #              if (isinstance(attr, tuple) and attr[0] != "value") or
        #              not isinstance(attr, tuple)}

        # This is basically conflict resolution here.
        skills = []
        for label in self.skills:
            for seq in self.skills[label]:
                corrects = self.skills[label][seq]['correct']
                accuracy = sum(corrects) / len(corrects)
                s = ((label, seq), accuracy)
                # s = (random(), len(corrects), label, seq)
                skills.append(s)
        # skills.sort(reverse=True)

        # for _,_,label,seq in skills:
        #     s = self.skills[label][seq]
        #     print(str(seq))

        while len(skills) > 0:
            # probability matching (with accuracy) conflict resolution
            # (label, seq), accuracy = weighted_choice(skills)

            # random conflict resolution
            (label, seq), accuracy = choice(skills)

            skills.remove(((label, seq), accuracy))

            s = self.skills[label][seq]

            for m in s['where_classifier'].get_matches(state):
                print("MATCH", label, m)
                if isinstance(m, tuple):
                    mapping = {"?foa%i" % i: str(ele)
                               for i, ele in enumerate(m)}
                else:
                    mapping = {'?foa0': m}
                # print('trying', m)

                if state[('value', mapping['?foa0'])] != "":
                    # print('no selection')
                    continue

                limited_state = {}
                for foa in mapping:
                    limited_state[('name', foa)] = state[('name', mapping[foa])]
                    limited_state[('value', foa)] = state[('value', mapping[foa])]

                try:
                    print('SEQ:', seq)
                    grounded_plan = tuple([self.planner.execute_plan(ele,
                                           limited_state) for ele in seq])
                except Exception as e:
                    # print('plan could not execute')
                    # pprint(limited_state)
                    # print("EXECPTION WITH", e)
                    continue

                vX = {}
                for foa in mapping:
                    vX[('value', foa)] = state[('value', mapping[foa])]

                if self.what:
                    what_features = {}
                    for attr in vX:
                        if isinstance(vX[attr], str) and vX[attr] != "":
                            seq = [c for c in vX[attr].lower().replace('"', "").replace("\\","")]
                            # print(seq)
                            # print(self.what.parse(seq))
                            new_what_f = self.what.get_features(attr, seq)
                            for attr in new_what_f:
                                what_features[attr] = new_what_f[attr]
                # what_training += [x[attr] for attr in x if isinstance(x[attr],
                #                                                       str) and
                #                  x[attr] != ""]

                vX.update(compute_features(vX,
                                           self.action_set.get_feature_dict()))

                if self.what:
                    vX.update(what_features)
                # for attr, value in self.compute_features(vX, features):
                #     vX[attr] = value
                # for foa in mapping:
                #     vX[('name', foa)] = state[('name', mapping[foa])]

                vX = tup.undo_transform(vX)

                print("WHEN PREDICTION STATE")
                pprint(vX)
                when_pred = s['when_classifier'].predict([vX])[0]
                # print(label, seq, s['when_classifier'])
                # pprint(when_pred)

                if when_pred == 0:
                    continue
               
                # pprint(limited_state)
                print("FOUND SKILL MATCH!")
                # pprint(limited_state)
                # pprint(seq)
                pprint(grounded_plan)
                    
                response = {}
                response['label'] = label
                response['selection'] = grounded_plan[2]
                response['action'] = grounded_plan[1]
                # response['inputs'] = list(grounded_plan[3:])

                # TODO replace value here with input_args, which need to be
                # tracked.
                if grounded_plan[2] == 'done':
                    response['inputs'] = {}
                else:
                    response['inputs'] = {a: grounded_plan[3+i] for i, a in
                                          enumerate(['value'])}
                response['foas'] = []
                # response['foas'].append("|" +
                #                         limited_state[("name", "?foa0")] +
                #                         "|" + grounded_plan[3])
                for i in range(1, len(mapping)):
                    response['foas'].append(limited_state[("name", "?foa%i" %
                                                           i)])
                    # response['foas'].append("|" +
                    #                         limited_state[("name", "?foa%i" % i)]
                    #                         +
                    #                         "|" +
                    #                         limited_state[('value', "?foa%i" % i)]) 

                pprint(response)
                return response

            # import time
            # time.sleep(5)

        return {}

    def train(self, state, label, foas, selection, action,
              inputs, correct):

        # create example dict
        example = {}
        example['state'] = state
        example['label'] = label
        example['selection'] = selection
        example['action'] = action
        example['inputs'] = inputs
        example['correct'] = correct
        example['foa_args'] = tuple(['?obj-' + foa for foa in foas])
        example['foa_names'] = {("name", "?foa%i" % i): foa
                                for i, foa in enumerate(foas)}

        example['foa_values'] = {("value", "?foa%i" % i): state['?obj-' +
                                                                foa]['value']
                                 for i, foa in enumerate(foas)}
        example['foa_values'][("value", "?foa0")] = ""
        example['limited_state'] = {("value", "?foa%i" % i):
                                    state['?obj-' + foa]['value']
                                    for i, foa in enumerate(foas)}
        example['limited_state'][("value", "?foa0")] = ""
        for attr in example['foa_names']:
            example['limited_state'][attr] = example['foa_names'][attr]

        tup = Tuplizer()
        flt = Flattener()
        # pprint(state)
        example['flat_state'] = flt.transform(tup.transform(state))

        # pprint(example['flat_state'])
        # import time
        # time.sleep(1000)

        # pprint(example)

        if label not in self.skills:
            self.skills[label] = {}

        # add example to examples
        if label not in self.examples:
            self.examples[label] = []
        self.examples[label].append(example)

        if label not in self.how_instances:
            self.how_instances[label] = self.how(planner=self.planner)
        # how = self.how(functions=functions, how_params=self.how_params)
        how_result = self.how_instances[label].ifit(example)

        # print(len(self.examples[label]))
        # for exp in how_result:
        #     correctness = [e['correct'] for e in how_result[exp]]
        #     print(label, len(correctness), sum(correctness) /
        #           len(correctness) , exp)
        # print()

        # act_plan = ActionPlanner(actions=functions,
        #                          act_params=self.how_params)
        # explanations = []

        # for exp in self.skills[label]:
        #     #print("CHECKING EXPLANATION", exp)
        #     try:
        #         grounded_plan = tuple([act_plan.execute_plan(ele,
        #                                 example['limited_state'])
        #                                for ele in exp])
        #         if act_plan.is_sais_equal(grounded_plan, sai):
        #             #print("found existing explanation")
        #             explanations.append(exp)
        #     except Exception as e:
        #         #print("EXECPTION WITH", e)
        #         continue

        # if len(explanations) == 0:
        #     explanations = act_plan.explain_sai(example['limited_state'],
        #                                         sai)

        # #print("EXPLANATIONS")
        # #pprint(explanations)

        # first delete old skill description
        del self.skills[label]
        self.skills[label] = {}

        # build new skill descriptions
        for exp in how_result:
            print('EXP', exp, correct)
            self.skills[label][exp] = {}
            self.skills[label][exp]['args'] = []
            self.skills[label][exp]['foa_states'] = []
            self.skills[label][exp]['examples'] = []
            self.skills[label][exp]['correct'] = []
            where = self.where()
            when = get_when_learner(self.when)(self.when_params)
            # when = Pipeline([('dict_vect', DictVectorizer(sparse=False)),
            #                  ('clf', self.when())])
            self.skills[label][exp]['where_classifier'] = where
            self.skills[label][exp]['when_classifier'] = when

            for e in how_result[exp]:
                self.skills[label][exp]['args'].append(e['foa_args'])
                self.skills[label][exp]['examples'].append(e)
                self.skills[label][exp]['correct'].append(int(e['correct']))

            T = self.skills[label][exp]['args']
            y = self.skills[label][exp]['correct']

            # foa_state = {attr: example['flat_state'][attr]
            #              for attr in example['flat_state']
            #              #if (isinstance(attr, tuple) and attr[0] != "value") or
            #              #not isinstance(attr, tuple)
            #             }
            # print("FOA STATE")
            # pprint(T[0])
            # pprint(foa_state)

            # structural_X = [e['flat_state'] for e in
            #                 self.skills[label][exp]['examples']]

            #  Should rewrite this so that I use the right values.
            # structural_X = [foa_state for t in T]
            # structural_X = self.skills[label][exp]['foa_states']

            structural_X = []
            for i, e in enumerate(self.skills[label][exp]['examples']):
                x = {attr: e['flat_state'][attr] for attr in e['flat_state']}
                # x_vals = {a: x[a] for a in x if isinstance(a, tuple) and a[0] ==
                #         "value" and a[1] in self.skills[label][exp]['args'][i]}

                # print("COMPUTED FEATURES")
                # pprint([a for a in
                #         compute_features(x_vals,self.action_set.get_feature_dict())])
                # x.update(compute_features(x_vals,self.action_set.get_feature_dict()))
                # pprint(x)
                structural_X.append(x)

            value_X = []
            for e in self.skills[label][exp]['examples']:
                x = {attr: e['foa_values'][attr] for attr in e['foa_values']}

                if self.what:
                    what_features = {}
                    for attr in x:
                        if isinstance(x[attr], str) and x[attr] != "":
                            seq = [c for c in x[attr].lower().replace('"', "").replace("\\","")]
                            # print(seq)
                            # print(self.what.parse(seq))
                            new_what_f = self.what.get_features(attr, seq)
                            for attr in new_what_f:
                                what_features[attr] = new_what_f[attr]

                x.update(compute_features(x, self.action_set.get_feature_dict()))

                if self.what:
                    x.update(what_features)

                # for attr, value in self.compute_features(x, features):
                #     x[attr] = value
                # for attr in e['foa_names']:
                #     x[attr] = e['foa_names'][attr]
                x = tup.undo_transform(x)

                pprint(x)
                value_X.append(x)

                #if example['label'] == "convert-different-num2":
                #    print("CORRECTNESS:", e['correct'])
                #    pprint(x)

            self.skills[label][exp]['where_classifier'].fit(T, structural_X, y)
            self.skills[label][exp]['when_classifier'].fit(value_X, y)
            # print(self.skills[label][exp]['when_classifier'])
            # self.skills[label][exp]['when_classifier'].ifit(value_X[-1], y[-1])

    def check(self, state, selection, action, inputs):
        return False
