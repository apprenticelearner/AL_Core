from pprint import pprint
import inspect
from itertools import permutations
from itertools import product

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
#from concept_formation.structure_mapper import rename_flat
from concept_formation.structure_mapper import get_component_names

from agents.BaseAgent import BaseAgent
from agents.action_planner import ActionPlanner
from agents.action_planner import math_actions
from ilp.most_specific import MostSpecific

def equals(x, y):
    return x == y

class LogicalWhereWhenHow(BaseAgent):
    """
    This is the basis for the 3 learning phase model. It accepts three classes
    for the where, when, and how learning. Where and and When are both
    classifiers. How learning is a form of planner. 
    """
    #def __init__(self, where, when, how):
    def __init__(self):
        #self.where = where
        #self.when = when
        #self.how = how
        #self.where = Foil
        self.where = MostSpecific
        #self.when = DecisionTreeClassifier
        #self.when = LogisticRegression
        self.when = GaussianNB
        self.how = ActionPlanner(math_actions)
        self.features = {'equals': equals}
        self.skills = {}
        self.examples = {}

    def compute_features(self, state):
        for feature in self.features:
            num_args = len(inspect.getargspec(self.features[feature]).args)
            if num_args < 1:
                raise Exception("Features must accept at least 1 argument")
            possible_args = [attr for attr in state]

            for tupled_args in product(possible_args, repeat=num_args):
                new_feature = (feature,) + tupled_args
                values = [state[attr] for attr in tupled_args]
                yield new_feature, self.features[feature](*values)

    def request(self, state):
        #print("REQUEST")
        #pprint(self.skills)
        ff = Flattener()
        tup = Tuplizer()
        act_plan = ActionPlanner(math_actions)

        state = tup.transform(state)
        state = ff.transform(state)

        foa_state = {attr: state[attr] for attr in state 
                     if (isinstance(attr, tuple) and attr[0] != "value") or 
                     not isinstance(attr, tuple)}
        
        for label in self.skills:
            for seq in self.skills[label]:
                s = self.skills[label][seq]
                for m in s['where_classifier'].get_matches(foa_state):
                    if isinstance(m, tuple):
                        mapping = {"?foa%i" % i: "?obj-" + str(ele) for i, ele in enumerate(m)}
                    else:
                        mapping = {'?foa0': m}

                    if state[('value', mapping['?foa0'])] != "":
                        continue

                    limited_state = {}
                    for foa in mapping:
                        limited_state[('name', foa)] = state[('name', mapping[foa])]
                        limited_state[('value', foa)] = state[('value', mapping[foa])]
                                
                    try:
                        grounded_plan = tuple([act_plan.execute_plan(ele, limited_state)
                                                   for ele in seq])
                    except Exception as e:
                        #print("EXECPTION WITH", e)
                        continue

                    vX = {}
                    for foa in mapping:
                        vX[('value', foa)] = state[('value', mapping[foa])]
                    for attr, value in self.compute_features(vX):
                        vX[attr] = value
                    vX = tup.undo_transform(vX)

                    #print("WHEN PREDICTION STATE")
                    #pprint(vX)
                    #pprint(s['when_classifier'].predict([vX]))

                    if s['when_classifier'].predict([vX])[0] == 0:
                        continue
                   
                    #print("FOUND SKILL MATCH!")
                    #pprint(limited_state)
                    #pprint(seq)
                    #pprint(grounded_plan)
                        
                    response = {}
                    response['label'] = label
                    response['selection'] = grounded_plan[2]
                    response['action'] = grounded_plan[1]
                    response['inputs'] = list(grounded_plan[3:])
                    response['foas'] = []
                    #response['foas'].append("|" +
                    #                        limited_state[("name", "?foa0")] +
                    #                        "|" + grounded_plan[3]) 
                    for i in range(1,len(mapping)):
                        response['foas'].append("|" +
                                                limited_state[("name", "?foa%i" % i)]
                                                +
                                                "|" +
                                                limited_state[('value', "?foa%i" % i)]) 

                    #pprint(response)
                    return response

        return {}

    def train(self, state, label, foas, selection, action, inputs, correct):

        # create example dict
        example = {}
        example['state'] = state
        example['label'] = label
        example['selection'] = selection
        example['action'] = action
        example['inputs'] = inputs
        example['correct'] = correct
        example['foa_args'] = tuple([foa.split("|")[1] for foa in foas])
        example['foa_names'] = {("name", "?foa%i" % i): foa.split("|")[1] for i,
                                 foa in enumerate(foas)}
        example['foa_values'] =  {("value", "?foa%i" % i): val.split("|")[2] for i, val in
                                    enumerate(foas)}
        example['foa_values'][("value", "?foa0")] = ""
        example['limited_state'] = {("value", "?foa%i" % i): val.split("|")[2] for i, val in
                                    enumerate(foas)}
        example['limited_state'][("value", "?foa0")] = ""
        for attr in example['foa_names']:
            example['limited_state'][attr] = example['foa_names'][attr]

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

        act_plan = ActionPlanner(actions=math_actions)
        explanations = []

        for exp in self.skills[label]:
            try:
                grounded_plan = tuple([act_plan.execute_plan(ele, example['limited_state'])
                                           for ele in exp])
                if grounded_plan == sai:
                    print("found existing explanation")
                    explanations.append(exp)
            except Exception as e:
                print("EXECPTION WITH", e)
                continue
            pass

        if len(explanations) == 0:
            explanations = act_plan.explain_sai(example['limited_state'], sai)

        print("EXPLANATIONS")
        pprint(explanations)

        for exp in explanations:
            if exp not in self.skills[label]:
                self.skills[label][exp] = {}
                self.skills[label][exp]['args'] = []
                self.skills[label][exp]['examples'] = []
                self.skills[label][exp]['correct'] = []
                where = self.where()
                when = Pipeline([('dict_vect', DictVectorizer(sparse=False)), 
                                  ('clf', self.when())])
                self.skills[label][exp]['where_classifier'] = where
                self.skills[label][exp]['when_classifier'] = when

            self.skills[label][exp]['args'].append(example['foa_args'])
            self.skills[label][exp]['examples'].append(example)
            self.skills[label][exp]['correct'].append(int(example['correct']))

            T = self.skills[label][exp]['args']
            y = self.skills[label][exp]['correct']

            foa_state = {attr: example['flat_state'][attr] 
                         for attr in example['flat_state']
                        if (isinstance(attr, tuple) and attr[0] != "value") or
                           not isinstance(attr, tuple)}
            #print("FOA STATE")
            #pprint(T[0])
            #pprint(foa_state)

            structural_X = [foa_state for t in T]
            
            value_X = []
            for e in self.skills[label][exp]['examples']:
                x = {attr: e['foa_values'][attr] for attr in e['foa_values']}
                for attr, value in self.compute_features(x):
                    x[attr] = value
                x = tup.undo_transform(x)
                value_X.append(x)

                if example['label'] == "convert-different-num2":
                    print("CORRECTNESS:", e['correct'])
                    pprint(x)

            self.skills[label][exp]['where_classifier'].fit(T, structural_X, y)
            self.skills[label][exp]['when_classifier'].fit(value_X, y)

    def check(self, state, selection, action, inputs):
        return False

