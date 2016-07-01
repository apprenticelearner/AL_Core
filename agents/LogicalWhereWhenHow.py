from pprint import pprint

import numpy as np
import inspect
from itertools import product
from itertools import combinations
import re

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
#from concept_formation.structure_mapper import rename_flat
from concept_formation.structure_mapper import get_component_names

from agents.utils import tup_sai
from agents.BaseAgent import BaseAgent
from agents.action_planner import ActionPlanner
from agents.WhenLearner import when_learners
from ilp.most_specific import MostSpecific
from ilp.aleph import Aleph
from ilp.foil import Foil
from ilp.ifoil import iFoil

#decorator for pipeline


#dictionary for when with pipeline stuff
#logistic regression(error check)
#svm(basically svc with kernels)
#SGD classifier
#nearest neighbours classification
#Any Ensemble classifier

#Pyibl prepopulate, chose

#Concept_formation(Cobweb)

class LogicalWhereWhenHow(BaseAgent):
    """
    This is the basis for the 3 learning phase model. It accepts three classes
    for the where, when, and how learning. Where and and When are both
    classifiers. How learning is a form of planner. 
    """
    #def __init__(self, where, when, how):
    def __init__(self, when="pyibl", when_params=None, how_params=None):

        #self.where = where
        #self.when = when
        #self.how = how
        #self.where = Foil
        #self.where = MostSpecific
        self.where = iFoil
        #self.where = Aleph
        #self.when = DecisionTreeClassifier
        self.when = when
        self.when_params = when_params
        self.how_params = how_params
        #self.when = GaussianNB
        #self.how = ActionPlanner(math_actions)
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
        act_plan = ActionPlanner(actions=functions,act_params=self.how_params)

        state = tup.transform(state)
        state = ff.transform(state)

        #foa_state = {attr: state[attr] for attr in state 
        #             if (isinstance(attr, tuple) and attr[0] != "value") or 
        #             not isinstance(attr, tuple)}
        
        for label in self.skills:
            for seq in self.skills[label]:
                s = self.skills[label][seq]
                #print(str(seq))

                constraints = []
                constraints.append("avalue(A,B,anil)")

                #print(str(seq))
                #for match in re.finditer(r"'\?foa(?P<M>[0-9]+)'", str(seq)):
                #    v = chr(int(match.group("M")) + ord("B"))
                #    if v == "B":
                #        continue
                #    constraints.append("not(avalue(A," + v + ",anil))")
                #    constraints.append("not(avalue(A," + v + ",aplussign))")
                #    constraints.append("not(avalue(A," + v + ",amultsign))")
                #    constraints.append("not(avalue(A," + v + ",aequalsign))")
                #    constraints.append("not(avalue(A," + v + ",aquestionmark))")
                #    constraints.append("not(avalue(A," + v + ",aIspaceneedspacetospaceconvertspacethesespacefractionsspacebeforespacesolving))")
                #    

                args = [chr(i + ord("B")) for i in 
                        range(len(s['where_classifier'].target_types)-1)]
                for p in combinations(args, 2):
                    constraints.append("dif(" + p[0] + "," + p[1] + ")")

                for m in s['where_classifier'].get_matches(state,
                                                           constraints):
                    #print("MATCH", m)
                    if isinstance(m, tuple):
                        mapping = {"?foa%i" % i: str(ele) for i, ele in enumerate(m)}
                    else:
                        mapping = {'?foa0': m}
                    #print('trying', m)

                    if state[('value', mapping['?foa0'])] != "":
                        #print('no selection')
                        continue

                    limited_state = {}
                    for foa in mapping:
                        limited_state[('name', foa)] = state[('name', mapping[foa])]
                        limited_state[('value', foa)] = state[('value', mapping[foa])]
                                
                    try:
                        grounded_plan = tuple([act_plan.execute_plan(ele, limited_state)
                                                   for ele in seq])
                    except Exception as e:
                        print('plan could not execute')
                        pprint(limited_state)
                        #print("EXECPTION WITH", e)
                        continue

                    vX = {}
                    for foa in mapping:
                        vX[('value', foa)] = state[('value', mapping[foa])]
                    for attr, value in self.compute_features(vX, features):
                        vX[attr] = value
                    #for foa in mapping:
                    #    vX[('name', foa)] = state[('name', mapping[foa])]

                    vX = tup.undo_transform(vX)

                    #print("WHEN PREDICTION STATE")
                    #pprint(vX)
                    when_pred = s['when_classifier'].predict([vX])[0]
                    #pprint(when_pred)

                    if when_pred == 0:
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

                #import time
                #time.sleep(5)

        return {}

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
        example['foa_args'] = tuple(['?obj-' + foa.split("|")[1] for foa in foas])
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
        #pprint(state)
        example['flat_state'] = flt.transform(tup.transform(state))

        #pprint(example['flat_state'])
        #import time
        #time.sleep(1000)

        #pprint(example)

        if label not in self.skills:
            self.skills[label] = {}

        # add example to examples
        if label not in self.examples:
            self.examples[label] = []
        self.examples[label].append(example)

        sai = tup_sai(selection,action,inputs)

        act_plan = ActionPlanner(actions=functions,act_params=self.how_params)
        explanations = []

        for exp in self.skills[label]:
            try:
                grounded_plan = tuple([act_plan.execute_plan(ele, example['limited_state'])
                                           for ele in exp])
                if grounded_plan == sai:
                    #print("found existing explanation")
                    explanations.append(exp)
            except Exception as e:
                #print("EXECPTION WITH", e)
                continue
            pass

        if len(explanations) == 0:
            explanations = act_plan.explain_sai(example['limited_state'], sai,act_params=self.how_params)

        #print("EXPLANATIONS")
        #pprint(explanations)

        for exp in explanations:
            if exp not in self.skills[label]:
                self.skills[label][exp] = {}
                self.skills[label][exp]['args'] = []
                self.skills[label][exp]['examples'] = []
                self.skills[label][exp]['correct'] = []
                where = self.where()
                when = when_learners[self.when](self.when_params)
                #when = Pipeline([('dict_vect', DictVectorizer(sparse=False)), 
                #                  ('clf', self.when())])
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

            #structural_X = [e['flat_state'] for e in
            #                self.skills[label][exp]['examples']]
            structural_X = [foa_state for t in T]
            
            value_X = []
            for e in self.skills[label][exp]['examples']:
                x = {attr: e['foa_values'][attr] for attr in e['foa_values']}
                for attr, value in self.compute_features(x, features):
                    x[attr] = value
                #for attr in e['foa_names']:
                #    x[attr] = e['foa_names'][attr]
                x = tup.undo_transform(x)
                value_X.append(x)

                #if example['label'] == "convert-different-num2":
                #    print("CORRECTNESS:", e['correct'])
                #    pprint(x)
            self.skills[label][exp]['where_classifier'].fit(T, structural_X, y)
            self.skills[label][exp]['when_classifier'].fit(value_X, y)
            #self.skills[label][exp]['when_classifier'].ifit(value_X[-1], y[-1])
 	
		
    def check(self, state, features, functions, selection, action, inputs):
        return False

