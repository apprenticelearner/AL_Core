from numba.types import f8, string, boolean
import numpy as np
from cre import MemSet, CREFunc, UntypedCREFunc, Fact, FactProxy
from apprentice.agents.base import BaseAgent
from apprentice.agents.cre_agents.state import State, encode_neighbors
from apprentice.agents.cre_agents.dipl_base import BaseDIPLAgent
from apprentice.shared import SAI as BaseSAI, rand_skill_uid, rand_skill_app_uid, rand_state_uid
from cre.transform import MemSetBuilder, Flattener, FeatureApplier, RelativeEncoder, Vectorizer, Enumerizer

from cre.utils import PrintElapse
from cre import TF
from cre.gval import new_gval
from itertools import chain
from copy import copy

from numba.core.runtime.nrt import rtsys
import gc

def used_bytes(garbage_collect=True):
    # if(garbage_collect): gc.collect()
    stats = rtsys.get_allocation_stats()
    # print(stats)
    return stats.alloc-stats.free

# -----------------------
# : Function minimal_str + get_info

def minimal_func_str(func, ignore_funcs=[]):
        if(isinstance(func, CREFunc)):
            return func.minimal_str(ignore_funcs=ignore_funcs)
        else:
            return str(func)

def func_get_info(func, ignore_funcs=[]):
    var_infos = []
    if(isinstance(func, CREFunc)):
        for alias, typ in zip(func.base_var_aliases, func.arg_types):
            var_infos.append({"alias" : alias, "type" : str(typ)})
        min_str = minimal_func_str(func, ignore_funcs)
    else:
        min_str = str(func)

    return {
        "repr" : repr(func),
        "vars" : var_infos,
        "minimal_str" : min_str
    }

# -----------------------
# : SAI

class SAI(BaseSAI):
    ''' Same as shared.SAI but when used internally we expect
        'selection' to be a cre.Fact instead of str, and 'action_type' 
        to be an ActionType instance instead of str. '''
    def as_tuple(self):
        sel_str = self.selection.id if(isinstance(self.selection,FactProxy)) else self.selection
        at_str = self.action_type.name if(not isinstance(self.action_type, str)) else self.action_type
        return (sel_str, at_str, self.inputs)

# -----------------------
# : Skill

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

        self.where_lrn_mech = agent.where_cls(self,**agent.where_args)
        self.when_lrn_mech = agent.when_cls(self,**agent.when_args)
        self.which_lrn_mech = agent.which_cls(self,*agent.which_args)

    def get_applications(self, state, skip_when=False):
        applications = []
        # print(self.how_part,":")
        # print(self.where_lrn_mech.conds)

        # with PrintElapse("get_matches"):
        matches = list(self.where_lrn_mech.get_matches(state))


        # with PrintElapse("iter_matches"):
        for match in matches:
            when_predict = 1 if skip_when else self.when_lrn_mech.predict(state, match)

            # print("1" if when_predict else "0",  match[0].id,"\t" , [m.id for m in match][1:], self.id, self.how_part)
            if(when_predict > 0):
                skill_app = SkillApplication(self, match)
                if(skill_app is not None):
                    applications.append(skill_app)
        return applications

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


    def ifit(self, state, match, reward):
        reward = float(reward)
        # with PrintElapse("fit where"):
        self.where_lrn_mech.ifit(state, match, reward)
        # with PrintElapse("fit when"):
        self.when_lrn_mech.ifit(state, match, reward)
        # with PrintElapse("fit which"):
        self.which_lrn_mech.ifit(state, match, reward)

        # print("FIT", reward, self, [m.id for m in match])

        # if(not hasattr(self.how_part,'__call__') and self.how_part == -1):
        #     print("<<", self.how_part, match)
        #     raise ValueError()

    def __repr__(self):
        return f"Skill({self.how_part}, uid={self.uid!r})"

    def __str__(self):
        min_str = minimal_func_str(self.how_part, ignore_funcs=self.agent.conversions)
        return f"Skill_{self.uid[4:9]}({min_str})"

# -----------------------
# : SkillApplication

class SkillApplication(object):
    # __slots__ = ("skill", "match", "sai")
    def __new__(cls, skill, match, uid=None):
        sai = skill(*match)
        if(sai is None):
            return None
        self = super().__new__(cls)

        self.skill = skill
        self.match = match
        self.sai = sai
        self.uid = rand_skill_app_uid() if(uid is None) else uid
        return self

    def get_info(self):
        sai = self.sai
        info = {
            'uid' :  self.uid,
            'skill_uid' :  self.skill.uid,
            'skill_label' : self.skill.label,
            'selection' :  sai.selection.id,
            'action' :  sai.action_type.name,
            'action_type' :  sai.action_type.name,
            'inputs' :  sai.inputs,
            'args' : [m.id for m in self.match][1:],
            # 'mapping' :  {f"arg{i-1}" if i else "sel" : x.id for i,x in enumerate(self.match)}
        }
        return info

    def __repr__(self):
        return f'{self.skill}({", ".join([m.id for m in self.match])}) -> {self.sai}'

# -----------------------
# : CREAgent


class CREAgent(BaseDIPLAgent):
# ------------------------------------------------
# : __init__
    def init_processesors(self):
        # The types that visible attributes / features can take.
        val_types = set([f8,string,boolean])
        for fact_type in self.fact_types:
            for _, attr_spec in fact_type.filter_spec("visible").items():
                # print(attr_spec)
                val_types.add(attr_spec['type'])

        for func in self.feature_set:
            # print(op, type(op), isinstance(op, Op))
            if(isinstance(func, UntypedCREFunc)):
                raise ValueError(
                "Feature functions must be typed. Specify signature in definition. " +
                "For instance @CREFunc(signature = unicode_type(unicode_type,unicode_type))."
                )
            val_types.add(func.signature.return_type)

        self.memset_builder = MemSetBuilder()
        self.enumerizer = Enumerizer()
        self.flattener = Flattener(self.fact_types,
            in_memset=None, id_attr="id", enumerizer=self.enumerizer)
        self.feature_applier = FeatureApplier(self.feature_set)

        # self.relative_encoder = RelativeEncoder(self.fact_types, in_memset=None, id_attr='id')
        # self.vectorizer = Vectorizer(val_types)

        state_cls = self.state_cls = State(self)


        @state_cls.register_transform(is_incremental=True, prereqs=['working_memory'])
        def flat(state):
            wm = state.get('working_memory')
            flattener = self.flattener
            return flattener(wm)

        @state_cls.register_transform(is_incremental=len(self.extra_features)==0, prereqs=['flat'])
        def flat_featurized(state):
            flat = state.get('flat')
            feature_applier = self.feature_applier
            featurized_state = feature_applier(flat)

            featurized_state = featurized_state.copy()
            for extra_feature in self.extra_features:
                featurized_state = extra_feature(self, state, featurized_state)

            return featurized_state
        

    def __init__(self, encode_neighbors=True, **config):
        # Parent defines learning-mechanism classes and args + action_chooser
        super().__init__(**config)

        self.how_lrn_mech = self.how_cls(self, **self.how_args)
        self.working_memory = MemSet()
        self.init_processesors()
        self.skills = {}
        self.skills_by_label = {}
        self.prev_skill_app = None


    def standardize_state(self, state, **kwargs):
        if(not isinstance(state, self.state_cls)):
            state_uid = kwargs.get('state_uid', None)
            if(isinstance(state, dict)):
                for k,obj in state.items():
                    if('contentEditable' in obj):
                        obj['locked'] = not obj['contentEditable']
                        del obj['contentEditable']

                state_uid = state.get("__uid__", None) if(state_uid is None) else state_uid
                if self.should_find_neighbors:
                    state = encode_neighbors(state)
                wm = self.memset_builder(state, MemSet())
            elif(isinstance(state, MemSet)):
                wm = state
            else:
                raise ValueError(f"Unrecognized State Type: \n{state}")

            if(state_uid is None):
                state_uid = f"ST_{wm.long_hash()}"

            state = self.state_cls({'__uid__' : state_uid, 'working_memory' : wm})

        # Ensure if prev_skill_app references current state's working_memory
        prev_skill_app = getattr(self,'prev_skill_app',None)
        if(prev_skill_app):
            wm = state.get("working_memory")
            self.prev_skill_app.match =[wm.get_fact(id=m.id) for m in prev_skill_app.match]

        self.state = state
        return state

    def standardize_SAI(self, sai):
        if(not isinstance(sai, SAI)):
            sai = SAI(sai)
        if(isinstance(sai.selection, str)):
            try:
                sai.selection = self.state.get('working_memory').get_fact(id=sai.selection)
            except KeyError:
                print(self.state.get('working_memory'))
                raise KeyError(f"Bad SAI: Element {sai.selection!r} not found in state.")
        if(isinstance(sai.action_type, str)):
            sai.action_type = self.action_types[sai.action_type]
        return sai

    def standardize_arg_foci(self, arg_foci, kwargs={}):
        # Allow for legacy name 'foci_of_attention'
        if(arg_foci is None):
            arg_foci = kwargs.get('foci_of_attention', None)
        if(arg_foci is None): 
            return None
        new_arg_foci = []
        wm = self.state.get('working_memory')
        for fact in arg_foci:
            if(isinstance(fact, str)):
                fact = wm.get_fact(id=fact)
            new_arg_foci.append(fact)
        return new_arg_foci

    def standardize_halt_policy(self, halt_policy):
        pass


# ------------------------------------------------
# : Act, Act_All
    def get_skill_applications(self, state):
        skill_apps = []
        for skill in self.skills.values():
            for skill_app in skill.get_applications(state):
                skill_apps.append(skill_app)

        skill_apps = self.which_cls.sort(state, skill_apps)
        return skill_apps

    def act(self, state, 
            return_kind='sai', # 'sai' | 'skill_app'
            json_friendly=False,
            **kwargs):

        state = self.standardize_state(state)
        skill_apps = self.get_skill_applications(state)

        # Apply action_chooser to pick from conflict set
        output = None
        if(len(skill_apps) > 0):
            skill_app = self.action_chooser(state, skill_apps)
            self.prev_skill_app = skill_app

            output = skill_app.sai if(return_kind == 'sai') else skill_app
                            
            if(json_friendly):
                output = output.get_info()
                
        return output

    def act_all(self, state,
        max_return=-1,
        return_kind='sai',
        json_friendly=False,
        **kwargs):

        state = self.standardize_state(state)
        skill_apps = self.get_skill_applications(state)

        # TODO: Not sure if necessary.
        # self.state.clear()

        if(max_return >= 0):
            skill_apps = skill_apps[:max_return]

        output = [sa.sai for sa in skill_apps] if(return_kind == 'sai') else skill_apps
            
        if(json_friendly):
            output = [x.get_info() for x in output]

        return output

            


# -----------------------------------------
# : Explain Demo

    def _skill_subset(self, sai, arg_foci=None, skill_label=None, skill_uid=None):
        # skill_uid or skill_label can cut down possible skill candidates 
        # print()
        subset = list(self.skills.values())
        if(skill_uid is not None):
            # print("A")
            subset = [self.skills[skill_uid]]

        # TODO: choose "NO_LABEL" or None to be standard
        elif(skill_label is not None and skill_label != "NO_LABEL"):
            # print("B")
            subset = self.skills_by_label.get(skill_label, subset)

        # print(sai, arg_foci, skill_label, skill_uid)
        # print(self.skills, )
        # print(subset)

        subset = [x for x in subset if x.input_attr == list(sai.inputs.keys())[0]]
        if(arg_foci is not None):
            pass
            # TODO: can probably reduce by matching n_args
            
        return subset
    

    def explain_from_skills(self, state, sai, 
        arg_foci=None, skill_label=None, skill_uid=None):

        skills_to_try = self._skill_subset(sai, arg_foci, skill_label, skill_uid)
        skill_apps = []
        
        # Try to find an explanation from the existing skills that matches
        #  the how + where parts. 
        for skill in skills_to_try:
            for candidate in skill.get_applications(state, skip_when=True):
                if(candidate.sai == sai):
                    # If foci are given make sure candidate has the 
                    #  same arguments in it's match.
                    if(arg_foci is not None and 
                        candidate.match[:1] != arg_foci):
                        continue

                    skill_apps.append(candidate)

        # If that doesn't work try to find an explanation from the existing
        #  skills that matches just the how-parts.
        if(len(skill_apps) == 0):
            input_attr, inp = list(sai.inputs.items())[0]

            for skill in skills_to_try:
                # Execute how-search to depth 1 with each skill's how-part
                if(hasattr(skill.how_part,'__call__')):
                    if(arg_foci is not None and skill.how_part.n_args != len(arg_foci)):
                        continue 
                    
                    explanation_set = self.how_lrn_mech.get_explanations(
                        state, inp, arg_foci, function_set=[skill.how_part],
                        search_depth=1, min_stop_depth=1)

                    for _, match in explanation_set:
                        if(len(match) != skill.how_part.n_args):
                            continue

                        match = [sai.selection, *match]
                        # print("<<", _, f'[{", ".join([x.id for x in match])}])')
                        skill_app = SkillApplication(skill, match)
                        if(skill_app is not None):
                            skill_apps.append(skill_app)

                # For skills with constant how-parts just check equality
                else:
                    if(skill.how_part == inp):
                        skill_apps.append(SkillApplication(skill, [sai.selection]))                        

            # if(len(skill_apps) > 0): print("EXPL HOW")
        return skill_apps
        # best_expl = self.choose_best_explanation(state, skill_apps)
        # if(best_expl is not None):
        #     print(best_expl.skill.where_lrn_mech.conds)
        # print("BEST EXPLANATION", best_expl)
        # return best_expl
    def explain_from_funcs(self, state, sai, 
        arg_foci=None, skill_label=None, skill_uid=None):

        # TODO: does not currently support multiple inputs per SAI.
        inp_attr, inp = list(sai.inputs.items())[0]
        if(not sai.action_type.get(inp_attr,{}).get('semantic',False)):
            return self.how_lrn_mech.new_explanation_set([(inp, [])])
            

        # Use how-learning mechanism to produce a set of candidate how-parts
        explanation_set = self.how_lrn_mech.get_explanations(
                state, inp, arg_foci)

        # If failed yield a set with just the how-part as a constant.
        if(len(explanation_set) == 0):
            explanation_set = self.how_lrn_mech.new_explanation_set([(inp, [])])

        return explanation_set

    def _as_json_friendly_expls(self, skill_explanations, func_explanations):
        if(skill_explanations is not None):
            skill_explanations = [exp.get_info() for exp in skill_explanations]

        if(func_explanations is not None):
            _func_expls = []
            for func, args in func_explanations:
                _func_expls.append({
                    "func" : func_get_info(func,ignore_funcs=self.conversions),
                    "args" : [a.id for a in args]
                })
            func_explanations = _func_expls
        return {
            "skill_explanations" : skill_explanations,
            "func_explanations" :  func_explanations
        }


    def explain_demo(self, state, sai, arg_foci=None, skill_label=None, skill_uid=None,
             json_friendly=False, force_use_funcs=False, **kwargs):
        ''' Explains an action 'sai' first using existing skills then using function_set''' 
        state = self.standardize_state(state)
        sai = self.standardize_SAI(sai)
        arg_foci = self.standardize_arg_foci(arg_foci, kwargs)

        # print("explain_demo:", sai, arg_foci)

        func_explanations = None
        skill_explanations = self.explain_from_skills(state, sai, arg_foci, skill_label, skill_uid)
        if(force_use_funcs or len(skill_explanations) == 0):
            func_explanations = self.explain_from_funcs(state, sai, arg_foci, skill_label)

        if(json_friendly):
            return self._as_json_friendly_expls(skill_explanations, func_explanations)

        # returns either how's implmentation of ExplanationSet or a SkillApplicationSet
        return skill_explanations, func_explanations

# ------------------------------------------------
# : Train
    def best_skill_explanation(self, state, skill_apps):
        def get_score(skill_app):
            score = skill_app.skill.where_lrn_mech.score_match(state, skill_app.match)
            # print("SCORE", score, skill_app)
            return score

        scored_apps = [x for x in [(get_score(sa), sa) for sa in skill_apps]]# if x[0] > 0.0]
        if(len(scored_apps) > 0):
            return sorted(scored_apps, key=lambda x: x[0])[-1][1]
        else:
            return None


    def induce_skill(self, sai, explanation_set, label=None):
        # TODO: Make this not CTAT specific
        input_attr = list(sai.inputs.keys())[0]
        if(sai.selection.id != "done"):
            # TODO: does not currently support multiple inputs per SAI.
            how_part, args = explanation_set.choose()
        else:
            how_part, explanation_set, args = -1, None, []

        # Make new skill.
        skill = Skill(self, sai.action_type, how_part, input_attr, 
            label=label, explanation_set=explanation_set)

        # print("INDUCE SKILL", skill)

        # Add new skill to various collections.
        self.skills[skill.uid] = skill
        if(label is not None):
            label_lst = self.skills_by_label.get(label,[])
            label_lst.append(skill)
            self.skills_by_label[label] = label_lst

        return SkillApplication(skill, [sai.selection,*args])


    def train(self, state, sai=None, reward:float=None,
              arg_foci=None, skill_label=None, skill_uid=None, mapping=None,
              ret_train_expl=False, add_skill_info=False,**kwargs):
        # print("SAI", sai)
        if(skill_label == "NO_LABEL"): skill_label = None

        state = self.standardize_state(state)
        sai = self.standardize_SAI(sai)
        arg_foci = self.standardize_arg_foci(arg_foci, kwargs)
        skill_apps = None

        # print("--TRAIN:", sai.selection.id, sai.inputs['value'])

        # Feedback Case : just train according to the last skill application.
        if(self.prev_skill_app != None and self.prev_skill_app.sai == sai):
            skill_app = self.prev_skill_app
        # Demonstration Case : try to explain the sai from existing skills.
        else:
            skill_explanations, func_explanations = \
                self.explain_demo(state, sai, arg_foci, skill_label, skill_uid)

            if(len(skill_explanations) > 0):
                skill_app = self.best_skill_explanation(state, skill_explanations)
            else:
                skill_app = self.induce_skill(sai, func_explanations, skill_label)

        skill_app.skill.ifit(state, skill_app.match, reward)
        # self.state.clear()

        # Return the unique id of the skill that was updated
        return skill_app.skill.uid

# -----------------------------------------------
# get_skills ()

    def get_skills(self,  skill_uids=None,  skill_labels=None,
                    states=None, json_friendly=False, **kwargs):
        ''' Gets all skills from a list of uids, labels, or states. If given states,
            then the agent is applied on the list of states and the set of applied skills
            are returned (e.g. for identifying the knowledge components of a domain).
            When json_friendly=True returns the get_info() of skills instead of their instances.
            Any remaining keyword arguments are passed to get_info().
        '''
        if(skill_uids):
            skills = [self.skills[uid] for uid in skill_uids]
        elif(skill_labels):
            skills = chain([skills_by_label.get(label,[]) for label in skill_labels])
        elif(states):
            raise NotImplemented()
        else:
            skills = list(self.skills.values())

        if(json_friendly):
            return [s.get_info(**kwargs) for s in skills]
        else:
            return skills

    def get_skill(self, skill_uid=None, skill_label=None, state=None, json_friendly=False, **kwargs):
        ''' Same as get_skills() but only returns the first result'''
        skill_uids = [skill_uid] if skill_uid is not None else None
        skill_labels = [skill_label] if skill_label is not None else None
        states = [state] if state is not None else None
        return self.get_skills(skill_uids, skill_labels, states, json_friendly, **kwargs)[0]
    

# ------------------------------------------------
# act_rollout()

    def _insert_rollout_skill_app(self, state, next_state, skill_app, states, actions, uid_stack, depth_counts, depth):
        nxt_uid = next_state.get('__uid__')
        
        if(skill_app is not None):
            uid = state.get('__uid__')    
            action_obj = {
                "skill_app_uid" : skill_app.uid,
                "state_uid" : uid,
                "next_state_uid" : nxt_uid,
                "skill_app" : skill_app,
            }
            actions[skill_app.uid] = action_obj
        if(nxt_uid not in states):
            # Ensure Depth Counts long enought
            while(depth >= len(depth_counts)):
                depth_counts.append(0)

            depth_index = depth_counts[depth]
            state_obj = {"state": next_state, "uid" : nxt_uid, "depth" : depth, "depth_index" : depth_index}
            states[nxt_uid] = state_obj
            uid_stack.append(nxt_uid)
            depth_counts[depth] += 1
            return True
        
        if(skill_app is not None):
            state_obj = states[uid]
            out_uids = state_obj.get('out_skill_app_uids', [])
            out_uids.append(skill_app.uid)            

            nxt_state_obj = states[nxt_uid]
            in_uids = state_obj.get('in_skill_app_uids', [])
            in_uids.append(skill_app.uid)

        return False

    def act_rollout(self, state, max_depth=-1, halt_policies=[], json_friendly=False, **kwargs):
        ''' 
        Applies act_all() repeatedly starting from 'state', and fanning out to create at 
        tree of all action rollouts up to some depth. At each step in this process the agent's 
        actions produce subsequent states based on the default state change defined by each 
        action's ActionType object. A list of 'halt_policies' specifies a set of functions that 
        when evaluated to false prevent further actions. Returns a tuple (states, action_infos).
        '''
        print("START ACT ROLLOUT")
        state = self.standardize_state(state)
        print(state.get('working_memory'))
        halt_policies = [self.standardize_halt_policy(policy) for policy in halt_policies]

        states = {}
        actions = {}
        uid_stack = []
        depth_counts = []
        self._insert_rollout_skill_app(None, state, None,
                     states, actions, uid_stack, depth_counts, 0)

        print("BEGIN RECRUSE", uid_stack)
        while(len(uid_stack) > 0):
            # for _ in range(len(uid_stack)):
            uid = uid_stack.pop()
            depth = states[uid]['depth']+1
            state = states[uid]['state']
            src_wm = state.get("working_memory")
            # print("---source---")
            # print(repr(src_wm))
            skill_apps = self.act_all(state, return_kind='skill_app')

            for skill_app in skill_apps:
                at = skill_app.sai.action_type
                # print("---skill_app :", skill_app)
                # print("---action_type :", at)
                next_wm = at.predict_state_change(src_wm, skill_app.sai)
                # print("---dest---")
                # print(repr(dest_wm))
                next_state = self.standardize_state(next_wm)
                
                self._insert_rollout_skill_app(state, next_state, skill_app,
                                states, actions, uid_stack, depth_counts, depth)
            # print(uid_stack)

        from pprint import pprint

        # pprint(states)
        for uid, obj in states.items():
            wm = obj['state'].get('working_memory')
            print(uid)
            print(repr(wm))
        for uid, obj in actions.items():
            print(uid)
            pprint(obj)
        # print(actions.key)









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


