import inspect
from pprint import pprint
from itertools import permutations
from itertools import product

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
#from concept_formation.structure_mapper import rename_flat
from concept_formation.structure_mapper import get_component_names

from agents.BaseAgent import BaseAgent
from agents.action_planner import ActionPlanner
from agents.action_planner import math_actions
#from ilp.foil import Foil
from ilp.aleph import Aleph

#import sys
#sys.stdout = open('/home/anant/Documents/output.txt', 'w')
class LogicalWhenHow(BaseAgent):
    """
    This is the basis for the 3 learning phase model. It accepts three classes
    for the where, when, and how learning. Where and and When are both
    classifiers. How learning is a form of planner. 
    """
    def __init__(self):
        self.when = Aleph 
        self.how = ActionPlanner(math_actions)
        self.skills = {}
        self.examples = {}

    def compute_features(self, state, features):
        for feature in features:
            num_args = len(inspect.getargspec(features[feature]).args)
            if num_args < 1:
                raise Exception("Features must accept at least 1 argument")
            possible_args = [attr for attr in state]

            for tupled_args in product(possible_args, repeat=num_args):
                new_feature = (feature,) + tupled_args
                values = [state[attr] for attr in tupled_args]
                yield new_feature, features[feature](*values)

    def request(self, state, features, functions):
        #print("REQUEST")
        #pprint(self.skills)
        ff = Flattener()
        tup = Tuplizer()

        state = tup.transform(state)
        state = ff.transform(state)
        for attr, value in self.compute_features(state, features):
            state[attr] = value
        
        for label in self.skills:
            for grounded_plan, plan in self.get_plan(label, state, functions):
                response = {}
                response['label'] = label
                response['selection'] = grounded_plan[2]
                response['action'] = grounded_plan[1]
                response['inputs'] = list(grounded_plan[3:])
                response['foas'] = []
                return response

        return {}

    def get_plan(self, label, state, functions):
        act_plan = ActionPlanner(functions)

        for seq in self.skills[label]:
            s = self.skills[label][seq]
            for m in s['when_classifier'].get_matches(state):
                if isinstance(m, tuple):
                    mapping = {"?foa%i" % (i): v for i,v in enumerate(m)}
                else:
                    mapping = {"?foa0": m}
                plan = self.rename_exp(seq, mapping)
                #pprint(state)
                #print(seq)
                print(m)
                #print(mapping)
                #print(plan)

                if state[('value', plan[2][1])] != "":
                    print("SELECTION VALUE NOT NIL")
                    continue

                try:
                    grounded_plan = tuple([act_plan.execute_plan(ele, state)
                                               for ele in plan])
                    yield grounded_plan, plan

                except Exception as e:
                    #print("EXECPTION WITH", e)
                    continue

    def train(self, state, features, functions, label, foas, selection, action,
              inputs, correct):

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
        #pprint(example)

        if label not in self.skills:
            self.skills[label] = {}

        # add example to examples
        if label not in self.examples:
            self.examples[label] = []
        self.examples[label].append(example)

        sai = []
        sai.append('sai')
        sai.append(action)
        sai.append(selection)

        if inputs is None:
            pass
        elif isinstance(inputs, list):
            sai.extend(inputs)
        else:
            sai.append(inputs)

        # mark selection (so that we can identify it as empty 
        sai = tuple(sai)

        act_plan = ActionPlanner(actions=functions)
        explanations = []

        # TODO need to update code for checking existing explanations
        print("TRYING PREV EXP", label, [l for l in self.skills])

        temp_state = {attr: example['flat_state'][attr] 
                      for attr in example['flat_state']}
        for attr, value in self.compute_features(temp_state, features):
            temp_state[attr] = value

        for grounded_plan, plan in self.get_plan(label, temp_state,
                                                functions):
            print("trying: ", plan)

            if grounded_plan == sai:
                print("found existing explanation")
                explanations.append(plan)

        if len(explanations) == 0:
            explanations = act_plan.explain_sai(example['flat_state'], sai)

        pprint(explanations)

        for exp in explanations:
            new_exp, args = self.arg_exp(exp)
            print(exp)
            print(new_exp)
            #print(args)

            if new_exp not in self.skills[label]:
                self.skills[label][new_exp] = {}
                self.skills[label][new_exp]['args'] = []
                self.skills[label][new_exp]['examples'] = []
                self.skills[label][new_exp]['correct'] = []
                when = self.when()
                self.skills[label][new_exp]['when_classifier'] = when

            self.skills[label][new_exp]['args'].append(args)
            self.skills[label][new_exp]['examples'].append(example)
            self.skills[label][new_exp]['correct'].append(int(example['correct']))

            print("exp pos: ", (sum(self.skills[label][new_exp]['correct'])))
            print("exp neg: ", (len(self.skills[label][new_exp]['correct']) -
                                sum(self.skills[label][new_exp]['correct'])))
            print()
            
            T = self.skills[label][new_exp]['args']

            X = []
            for e in self.skills[label][new_exp]['examples']:
                x = {attr: e['flat_state'][attr] for attr in e['flat_state']}
                for attr, value in self.compute_features(x, features):
                    x[attr] = value
                X.append(x)

            y = self.skills[label][new_exp]['correct']

            #print(T)
            #print(X)
            #print(y)
            
            self.skills[label][new_exp]['when_classifier'].fit(T, X, y)

    def check(self, state, features, functions, selection, action, inputs):
        return False

    def rename_exp(self, exp, mapping):
        new_exp = []
        for ele in exp:
            if isinstance(ele, tuple):
                new_exp.append(self.rename_exp(ele, mapping))
            elif ele in mapping:
                new_exp.append(mapping[ele])
            else:
                new_exp.append(ele)
        return tuple(new_exp)

    def arg_exp(self, exp, count=0):
        # TODO need to check if FOA already exists for eahc obj and use prev
        # foa
        new_exp = []
        args = []
        for ele in exp:
            if isinstance(ele, tuple):
                sub_exp, sub_args = self.arg_exp(ele, count)
                new_exp.append(sub_exp)
                args += list(sub_args)
                count += len(sub_args)
            elif isinstance(ele, str) and ele[0] == "?":
                new_exp.append("?foa%i" % count)
                args.append(ele)
                count += 1
            else:
                new_exp.append(ele)

        return tuple(new_exp), tuple(args)

