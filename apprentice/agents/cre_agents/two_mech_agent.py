from numba.types import f8, string, boolean
import numpy as np
from cre import MemSet, CREFunc, UntypedCREFunc, Fact, FactProxy
from apprentice.agents.base import BaseAgent
from apprentice.agents.cre_agents.state import State, encode_neighbors
from apprentice.agents.cre_agents.dipl_base import BaseDIPLAgent
from apprentice.agents.cre_agents.cre_agent import CREAgent, SAI, SkillApplication, minimal_func_str
from apprentice.agents.cre_agents.when import VectorTransformMixin
from apprentice.shared import SAI as BaseSAI, rand_skill_uid, rand_skill_app_uid, rand_state_uid
from cre.transform import MemSetBuilder, Flattener, FeatureApplier, RelativeEncoder, Vectorizer, Enumerizer

from cre.utils import PrintElapse
from cre import TF
from cre.gval import new_gval
from itertools import chain
from copy import copy

from numba.core.runtime.nrt import rtsys
import gc
import hashlib
import base64
from datetime import datetime


class LHSLearner(VectorTransformMixin):
    def __init__(self, skill, **kwargs):
        kwargs['one_hot'] = kwargs.get('one_hot', True)

        self.skill = skill
        self.agent = skill.agent

        VectorTransformMixin.__init__(self, skill, encode_relative=False, **kwargs)

        from sklearn.tree import DecisionTreeClassifier
        self.classifier = DecisionTreeClassifier()
        self.match_map = {}
        self.inv_match_map = {}

    def ifit(self, state, skill_app, match, reward):
        match_ids = tuple([m.id for m in match])
        tup = (match_ids, reward)
        if(tup not in self.match_map):
            y = len(self.match_map)
            self.match_map[tup] = y
            self.inv_match_map[y] = tup
        y = self.match_map[tup]
        self.add_example(state, skill_app, y) # Insert into X_nom, Y
        self.classifier.fit(self.X_nom, self.Y) # Re-fit

    def predict(self, state):
        continuous, nominal = self.transform(state, None)
        X_nom_subset = nominal[:self.X_nom.shape[1]].reshape(1,-1)
        prediction = self.classifier.predict(X_nom_subset)[0]        
        match_ids, reward = self.inv_match_map[prediction]

        if(reward > 0):
            wm = state.get("working_memory")
            match = [wm.get_fact(id=_id) for _id in match_ids]
            return match
        return None


class Skill(object):
    def __init__(self, agent, action_type, how_part, input_attr,
                 uid=None, label=None, explanation_set=None):
        self.agent = agent
        self.label = label
        self.explanation_set = explanation_set
        self.how_part = how_part
        self.action_type = action_type
        self.input_attr = input_attr
        self.uid = rand_skill_uid() if(uid is None) else uid

        # self.where_lrn_mech = agent.where_cls(self,**agent.where_args)
        self.lhs_learner = LHSLearner(self)
        self.skill_apps = {}

    def get_applications(self, state, skip_when=False):
        match = self.lhs_learner.predict(state)
        if(match is not None):
            skill_app = SkillApplication(self, state, match)
            return [skill_app] if skill_app is not None else []
        else:
            return []

    def get_info(self, where_kwargs={}, when_kwargs={},
                 which_kwargs={}, **kwargs):
        info = {  "uid" : self.uid,
                  "how": {
                    "func" : func_get_info(self.how_part, ignore_funcs=self.agent.conversions)
                  },
                  "where": self.where_lrn_mech.get_info(**where_kwargs),
                  "when": self.when_lrn_mech.get_info(**when_kwargs),
                  "which": self.which_lrn_mech.get_info(**which_kwargs),
                }
        return info

    def __call__(self, *match):
        args = match[1:]
        if(hasattr(self.how_part, '__call__')):
            if(len(args) != self.how_part.n_args):
                raise ValueError(f"Incorrect number of args: {len(args)}, for skill how-part {self.how_part} with {self.how_part.n_args} positional arguments.")
            try:
                val = self.how_part(*args)
            except Exception as e:
                return None
        else:
            val = self.how_part
        inp = {self.input_attr : val}
        return SAI(match[0], self.action_type, inp)


    def ifit(self, state, skill_app, reward):
        skill_app.skill.skill_apps[skill_app.uid] = skill_app
        self.lhs_learner.ifit(state, skill_app, skill_app.match, reward)

    def __repr__(self):
        return f"Skill({self.how_part}, uid={self.uid!r})"

    def __str__(self):
        min_str = minimal_func_str(self.how_part, ignore_funcs=self.agent.conversions)
        return f"Skill_{self.uid[3:8]}({min_str})"

# -----------------------
# : CREAgent


class TwoMech(CREAgent):
    def __init__(self, **config):
        # Parent defines learning-mechanism classes and args + action_chooser
        super().__init__(**config)
        self.action_chooser = lambda s,sas : sas[0]
        self.prev_incorrect = False

# ------------------------------------------------
# : Act, Act_All
    def get_skill_applications(self, state):
        # Force to ask for hint if prev action was incorrect
        if(self.prev_incorrect):
            return []

        skill_apps = []
        for skill in self.skills.values():
            for skill_app in skill.get_applications(state):
                skill_apps.append(skill_app)
        print("SAS", skill_apps)
        # skill_apps = self.which_cls.sort(state, skill_apps)
        skill_apps = self.action_filter(state, skill_apps)
        

        return skill_apps

    def induce_skill(self, state, sai, explanation_set, label=None, explanation_selected=None):
        # TODO: Make this not CTAT specific
        input_attr = list(sai.inputs.keys())[0]
        if(sai.selection.id == "done"):
            how_part, explanation_set, args = -1, None, []
        else:
            if(explanation_selected is not None):
                esd = explanation_selected['data']
                choice_repr = esd['func']['repr']
                choice_args = tuple(esd['args'])
                how_part, args = None, None
                for func, args in explanation_set:
                    if(choice_repr == repr(func) and
                       choice_args == tuple([a.id for a in args])
                    ):
                        how_part, args = func, args
                        break
                if(how_part is None and args is None):
                    raise ValueError("")
            else:
                # TODO: does not currently support multiple inputs per SAI.
                how_part, args = explanation_set.choose()

        # Make new skill.
        skill = Skill(self, sai.action_type, how_part, input_attr, 
            label=label, explanation_set=explanation_set)
        print("INDUCE SKILL", skill, skill.how_part)

        # print("INDUCE SKILL", skill)

        # Add new skill to various collections.
        self.skills[skill.uid] = skill
        if(label is not None):
            label_lst = self.skills_by_label.get(label,[])
            label_lst.append(skill)
            self.skills_by_label[label] = label_lst

        return SkillApplication(skill, state, [sai.selection,*args])

# ------------------------------------------------
# : Train

    def best_skill_explanations(self, state, skill_apps, alpha=0.1):
        def get_score(skill_app):
            match = skill_app.skill.lhs_learner.predict(state)
            if(match is None):
                return -1
            return tuple([m.id for m in match]) == tuple([m.id for m in skill_app.match])

        scored_apps = [x for x in [(get_score(sa), sa) for sa in skill_apps]]# if x[0] > 0.0]
        if(len(scored_apps) > 0):
            srted = sorted(scored_apps, key=lambda x: -x[0])

            # Keep only explanations with scores within alpha% of the max.
            max_score = srted[0][0]
            thresh = max_score*(1-alpha)
            k = 1
            for k in range(1,len(srted)):
                if(srted[k][0] < thresh):
                    break

            return [x[1] for x in srted[:k]]
        else:
            return []













                















##### Thoughts on optimization for later ##### 
'''
Time Breakdown:
    52% Act
        40% predict 
            34% transform
            65% predict
        12% get_matches
            encode_relative
    48% Train
        16% ifit
            10% transform
            6% fit
        13% explain from skills
        5% induce skill
        3% standardize SAI
        5% standardize state


-Need to make input state update incrementally

-Vectorizer: Could avoid re-fetching from dictionaries in Vectorizer by 
caching idrecs of the input memset similar to fact_ptrs data
structure in memset.
    -Note wouldn't help unless featurized state could be made to update incrementally


Thoughts on what is going on w/ When:
1) Lack of explicit null
2) Inclusion of "value":  maybe should be decremented


'''


