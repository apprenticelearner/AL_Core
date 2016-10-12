from copy import deepcopy
from random import shuffle

from concept_formation.trestle import TrestleTree
from concept_formation.preprocessor import Flattener

from agents.utils import tup_sai
from agents.utils import compute_features
from agents.utils import parse_foas
from agents.BaseAgent import BaseAgent
from agents.action_planner import ActionPlanner

class TrestleHow(BaseAgent):
    """
    This is an agent based on a single Trestle knowledge base contianing both
    the contexts of actions and the explained parameters of actions. It is
    based on the original implementation of the trestle_api.
    """

    def __init__(self,how_params=None):
        self.tree = TrestleTree()
        self.how_params = how_params

    def request(self, state, features, functions):
        """
        """
        tree = self.tree
        aug_state = deepcopy(state)
        aug_state.update(compute_features(aug_state,features))
        inf_state = tree.infer_missing(aug_state)
        sais = [k for k in inf_state if isinstance(k,tuple) and k[0]=='sai']
        foas = [k for k in inf_state if isinstance(k,tuple) and k[0]=='foa']

        act_plan = ActionPlanner(actions=functions,act_params=self.how_params)

        shuffle(sais)
        sai = sais[0]

        ft = Flattener()
        flat_state = ft.transform(inf_state)

        grounded_plan = act_plan.execute_plan(sai,flat_state)

        response = {}
        response['label'] = inf_state['skill_label']
        response['selection'] = grounded_plan[2]
        response['action'] = grounded_plan[1]
        response['inputs'] = list(grounded_plan[3:])
        response['foas'] = ['|'+f+'|' for f in foas]

        return response

    def train(self,state,features,functions,label,foas,selection,action,inputs,correct):
        """
        This agent makes use of a limited collection of the normal arguments
        """
        tree = self.tree
        sai = tup_sai(selection,action,inputs)
        act_plan = ActionPlanner(actions=functions,act_params=self.how_params)

        aug_state = deepcopy(state)

        aug_state.update(compute_features(aug_state,features))

        foas = parse_foas(foas)
        for foa in foas:
            aug_state[('foa',foa['name'])] = True

        aug_state['skill_label'] = label

        inf_state = tree.infer_missing(aug_state)
        plans = [k for k in inf_state if isinstance(k,tuple) and k[0] == 'sai']

        ft = Flattener()
        flat_state = ft.transform(aug_state)

        plans = [p for p in plans if act_plan.compare_plan(p,sai,flat_state)]

        if len(plans) == 0:
            plans = act_plan.explain_sai(flat_state,sai,act_params=self.how_params)

        for p in plans:
            aug_state[p] = correct

        tree.ifit(aug_state)

    def check(self,state,features,functions,selection,action,inputs):
        return False