from pprint import pprint
from itertools import permutations

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
from ilp.foil import Foil

class WhereWhenHow(BaseAgent):
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
        self.where = Foil
        self.when = DecisionTreeClassifier
        self.how = ActionPlanner(math_actions)
        self.skills = {}
        self.examples = {}

    def request(self, state):
        #print("REQUEST")
        #pprint(self.skills)
        ff = Flattener()
        tup = Tuplizer()
        act_plan = ActionPlanner(math_actions)

        state = tup.transform(state)
        state = ff.transform(state)
        state_comps = get_component_names(state)
        
        for label in self.skills:
            for seq in self.skills[label]:
                s = self.skills[label][seq]
                c = get_component_names(s['examples'][0]['limited_state'])

                nil_c = set([comp for comp in state_comps 
                             if state[('value', comp)] == ''])
                non_nil_c = state_comps - nil_c

                for sel in nil_c:
                    mapping = {}
                    limited_state = {}

                    mapping['?foa0'] = sel
                    limited_state[('name', '?foa0')] = state[('name', sel)]
                    limited_state[('value', '?foa0')] = state[('value', sel)]

                    seq_c = []
                    if len(seq) > 3:
                        seq_c = list(get_component_names({seq[3]: True}))

                    available_c = state_comps - set([sel])
                    sub_c = list(c - set(['?foa0']))

                    if len(seq_c) > 0:
                        for p_inp in permutations(non_nil_c, len(seq_c)):

                            for i, v in enumerate(p_inp):
                                mapping[seq_c[i]] = v
                                limited_state[('name', seq_c[i])] = state[('name', v)]
                                limited_state[('value', seq_c[i])] = state[('value', v)]
                                
                            try:
                                grounded_plan = tuple([act_plan.execute_plan(ele, limited_state)
                                                           for ele in seq])
                            except Exception as e:
                                #print("EXECPTION WITH", e)
                                continue

                            available_c = state_comps - set([sel]) - set(p_inp)
                            sub_c = list(c - set(['?foa0']) - set(seq_c))

                            for p_other in permutations(available_c, len(sub_c)):
                                mapping = {} 
                                mapping['?foa0'] = sel
                                for i, v in enumerate(p_inp):
                                    mapping[seq_c[i]] = v
                                for i, v in enumerate(p_other):
                                    mapping[sub_c[i]] = v

                                sX = {}
                                vX = {}
                                for foa in mapping:
                                    sX[('name', foa)] = state[('name', mapping[foa])]
                                    vX[('value', foa)] = state[('value', mapping[foa])]

                                if s['where_classifier'].predict([sX])[0] == 0:
                                    continue
                                if s['when_classifier'].predict([vX])[0] == 0:
                                    continue

                                limited_state = {}
                                for a in sX:
                                    limited_state[a] = sX[a]
                                for a in vX:
                                    limited_state[a] = vX[a]
                               
                                print("FOUND SKILL MATCH!")
                                pprint(limited_state)
                                pprint(seq)
                                pprint(grounded_plan)
                                    
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

                                pprint(response)
                                return response

                    else:
                        print("NO SEQUENCE COMPONENTS!")

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
        example['foa_args'] = [foa.split("|")[1] for foa in foas]
        example['foa_names'] = {("name", "?foa%i" % i): foa.split("|")[1] for i,
                                 foa in enumerate(foas)}
        example['foa_values'] =  {("value", "?foa%i" % i): val.split("|")[2] for i, val in
                                    enumerate(foas)}
        example['limited_state'] = {("value", "?foa%i" % i): val.split("|")[2] for i, val in
                                    enumerate(foas)}
        example['limited_state'][("value", "?foa0")] = ""
        for attr in example['foa_names']:
            example['limited_state'][attr] = example['foa_names'][attr]

        tup = Tuplizer()
        flt = Flattener()
        example['flat_state'] = flt.transform(tup.transform(state))

        pprint(example)

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

            foa_state = {}

            structural_X = [e['foa_names'] for e in
                            self.skills[label][exp]['examples']]
            value_X = [e['foa_values'] for e in
                       self.skills[label][exp]['examples']]

            self.skills[label][exp]['where_classifier'].fit(T, structural_X, y)
            self.skills[label][exp]['when_classifier'].fit(value_X, y)

    def check(self, state, selection, action, inputs):
        return False
