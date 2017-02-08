from pprint import pprint
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

    def __init__(self, action_set, how_params=None):
        self.tree = TrestleTree()
        self.action_set = action_set
        self.how_params = how_params
        self.noop = {}

    def request(self, state):
        """
        """
        tree = self.tree

        ft = Flattener()
        flat_state = ft.transform(state)

        #pprint(flat_state)

        state = {a: flat_state[a] for a in flat_state if isinstance(a, tuple)
                 and (a[0] == 'name' or a[0] == 'value')}

        #pprint(state)

        aug_state = deepcopy(state)
        aug_state.update(compute_features(state,
                                          self.action_set.get_feature_dict()))
        inf_state = tree.infer_missing(aug_state)

        sais = [k for k in inf_state if isinstance(k,tuple) and k[0]=='sai' and
               inf_state[k] == True]
        foas = [k for k in inf_state if isinstance(k,tuple) and k[0]=='foa']

        shuffle(sais)
        act_plan = ActionPlanner(self.action_set, act_params=self.how_params)

        for sai in sais:
            try:
                grounded_plan = tuple([act_plan.execute_plan(ele, state)
                                       for ele in sai])
            except:
                continue

            # can't update something that isn't updatable.
            if flat_state[('value', '?obj-' + grounded_plan[2])] != "":
                continue

            print('GROUND PLAN: ', grounded_plan)

            response = {}
            response['label'] = inf_state['skill_label']
            response['selection'] = grounded_plan[2]
            response['action'] = grounded_plan[1]
            response['inputs'] = list(grounded_plan[3:])
            response['foas'] = ['|'+f+'|' for f in foas]

            return response

        return {}

    def train(self, state, label, foas, selection, action, inputs, correct):
        """
        This agent makes use of a limited collection of the normal arguments
        """
        tree = self.tree
        sai = tup_sai(selection, action, inputs)
        act_plan = ActionPlanner(self.action_set, act_params=self.how_params)

        # state = {a: state[a] for a in state if isinstance(a, tuple) and
        #         (a[0] == 'name' or a[0] == 'value')}

        aug_state = deepcopy(state)
        aug_state.update(compute_features(aug_state,
                                          self.action_set.get_feature_dict()))

        for i, foa in enumerate(foas):
            aug_state[('foa' + str(i), foa)] = True

        aug_state['skill_label'] = label

        inf_state = tree.infer_missing(aug_state)
        plans = [k for k in inf_state
                 if isinstance(k, tuple) and k[0] == 'sai']

        ft = Flattener()
        flat_state = ft.transform(state)
        flat_state = {a: flat_state[a] for a in flat_state
                      if (isinstance(a, tuple) and
                          (a[0] == 'name' or a[0] == 'value'))}

        s = frozenset(flat_state.items())
        if s not in self.noop:
            self.noop[s] = []
        self.noop[s].append((selection, action, *inputs))

        plans = [p for p in plans if act_plan.compare_plan(p, sai, flat_state)]

        if len(plans) == 0:
            plans = act_plan.explain_sai(flat_state, sai)

        for p in plans:
            print(sai)
            print(p)
            print(tuple([act_plan.execute_plan(ele, flat_state) for ele in p]))
            aug_state[p] = correct

        print("about to ifit")
        tree.ifit(aug_state)
        print("done ifitting")

    def check(self, state, selection, action, inputs):
        return False
