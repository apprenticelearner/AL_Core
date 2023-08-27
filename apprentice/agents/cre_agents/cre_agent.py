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
import hashlib
import base64
from datetime import datetime

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

    def long_hash(self):
        # A very explicit long hash, but conflict safe one 
        #  TODO: consider a fixed-width, perhaps more efficient method.
        sel = self.selection.id
        atn = self.action_type.name
        inp_summary = ",".join([f"{k}:{v}" for k,v in self.inputs.items()])
        return f'{sel}|{atn}|{inp_summary}'

def action_uid(state, sai):
    h = hashlib.sha224()
    h.update(state.get("__uid__", None).encode('utf-8'))
    h.update(repr(sai).encode('utf-8'))
    # Limit length to 30 chars to be consistent with other hashes 
    return f"AC_{base64.b64encode(h.digest(), altchars=b'AB')[:30].decode('utf-8')}"

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

        self.skill_apps = {}

    def get_applications(self, state, skip_when=False):        
        # print(str(self.how_part),":")
        # print(self.where_lrn_mech.conds)
        # with PrintElapse("get_matches"):

        applications = []
        matches = list(self.where_lrn_mech.get_matches(state))

        # with PrintElapse("iter_matches"):
        for match in matches:
            when_pred = 1 if skip_when else self.when_lrn_mech.predict(state, match)                
            # print(when_pred, match)
            if(when_pred > 0 or self.agent.suggest_uncert_neg and when_pred != -1.0):
                skill_app = SkillApplication(self, state, match, when_pred=when_pred)
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


    def ifit(self, state, skill_app, reward):

        skill_app.skill.skill_apps[skill_app.uid] = skill_app

        self.where_lrn_mech.ifit(state, skill_app.match, reward)
        # with PrintElapse("fit when"):
        self.when_lrn_mech.ifit(state, skill_app, reward)
        # with PrintElapse("fit which"):
        self.which_lrn_mech.ifit(state, skill_app, reward)

        # print("FIT", reward, self, [m.id for m in match])

        # if(not hasattr(self.how_part,'__call__') and self.how_part == -1):
        #     print("<<", self.how_part, match)
        #     raise ValueError()

    def __repr__(self):
        return f"Skill({self.how_part}, uid={self.uid!r})"

    def __str__(self):
        min_str = minimal_func_str(self.how_part, ignore_funcs=self.agent.conversions)
        return f"Skill_{self.uid[3:8]}({min_str})"

# -----------------------
# : SkillApplication

class SkillApplication(object):
    # __slots__ = ("skill", "match", "sai")
    def __new__(cls, skill, state, match, uid=None, when_pred=None):
        sai = skill(*match)

        # Find the unique id for this skill_app
        state_uid = state.get("__uid__", None)
        h = hashlib.sha224()
        h.update(skill.uid.encode('utf-8'))
        h.update(state_uid.encode('utf-8'))
        h.update(",".join([m.id for m in match]).encode('utf-8'))
        uid = f"A_{base64.b64encode(h.digest(), altchars=b'AB')[:30].decode('utf-8')}"

        # If this skill has been fit with this skill_app then return the previous instance
        if(uid in skill.skill_apps):
            self = skill.skill_apps[uid]
            # If given a prediction probability then overwrite the old one.
            if(self.when_pred is not None):
                self.when_pred = when_pred
            return self

        if(sai is None):
            return None
        self = super().__new__(cls)

        self.skill = skill
        self.state_uid = state_uid
        self.match = match
        self.args = match[1:]
        self.sai = sai
        self.uid = uid
        self.when_pred = when_pred

        return self

    def annotate_train_data(self, reward, arg_foci, skill_label, skill_uid,
                            how_help, explanation_selected, is_demo=False, **kwargs):
        self.train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.reward = reward
        self.arg_foci = arg_foci
        self.skill_label = skill_label
        self.skill_uid = skill_uid
        self.how_help = how_help
        self.explanation_selected = explanation_selected
        self.is_demo = is_demo

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
            'args' : [m.id for m in self.args],
        }
        if(hasattr(self, 'train_time')):
            train_data = {}
            train_data['train_time'] = getattr(self, 'train_time', None)
            train_data['reward'] = getattr(self, 'reward', None)
            train_data['arg_foci'] = getattr(self, 'arg_foci', None)
            train_data['skill_label'] = getattr(self, 'skill_label', None)
            train_data['skill_uid'] = getattr(self, 'skill_uid', None)
            train_data['how_help'] = getattr(self, 'how_help', None)
            train_data['explanation_selected'] = getattr(self, 'explanation_selected', None)
            train_data['is_demo'] = getattr(self, 'is_demo', False)
            train_data = {k:v for k,v in train_data.items() if v is not None}
            if('arg_foci' in train_data):
                print("ARG FOCI", train_data['arg_foci'])

                # TODO: Fix whatever causing this to be necessary
                # arg_foci = train_data['arg_foci']
                # if(len(arg_foci) > 0 and isinstance(arg_foci[0], list)):
                #     arg_foci = arg_foci[0]
                train_data['arg_foci'] = [m.id for m in train_data['arg_foci']]
            if('reward' in train_data):
                train_data['confirmed'] = True
            info.update(train_data)

        return info

    def as_train_kwargs(self):
        return {'sai': BaseSAI(*self.sai.as_tuple()),
                'arg_foci' : [m.id for m in self.args],
                'how_str' : "???"}

    def __repr__(self):
        return f'{self.skill}({", ".join([m.id for m in self.match])}) -> {self.sai}'

    def __eq__(self, other):
        return getattr(self, 'uid', None) == getattr(other, 'uid', None)

    def __hash__(self):
        return hash(self.uid)

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

        @state_cls.register_transform(is_incremental=False, prereqs=['working_memory'])
        def py_dict(state):
            wm = state.get('working_memory')
            return wm.as_dict(key_attr='id')
        

    def __init__(self, encode_neighbors=True, **config):
        # Parent defines learning-mechanism classes and args + action_chooser
        super().__init__(**config)

        self.how_lrn_mech = self.how_cls(self, **self.how_args)
        self.working_memory = MemSet()
        self.init_processesors()
        self.skills = {}
        self.skills_by_label = {}
        self.prev_skill_app = None
        self.episodic_memory = {}


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
                state_uid = f"S_{wm.long_hash()}"

            state = self.state_cls({'__uid__' : state_uid, 'working_memory' : wm})

        # Ensure if prev_skill_app references current state's working_memory
        prev_skill_app = getattr(self,'prev_skill_app',None)
        if(prev_skill_app):
            wm = state.get("working_memory")

            # Try to recover the facts matched by the previous skill app in the new state
            try:
                self.prev_skill_app.match =[wm.get_fact(id=m.id) for m in prev_skill_app.match]
            # However if that is impossible then ignore prev_skill_app
            except:
                # print([m.id for m in  prev_skill_app.match])
                # print(wm)
                self.prev_skill_app = None
        self.state = state
        return state

    def standardize_SAI(self, sai):
        if(isinstance(sai, BaseSAI)):
            # Always copy the SAI to avoid side effects in the caller
            sai = SAI(sai.selection, sai.action_type, sai.inputs)
        else:
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
            # if("Qr9" in self.state.get('__uid__')):
            #     if(repr(skill.how_part) == "NumericalToStr(TensDigit(Add3(CastFloat(a.value), CastFloat(b.value), CastFloat(c.value))))"):
            #         print(repr(self.state.get("flat_featurized")))
            #         print(skill.when_lrn_mech.classifier)

            for skill_app in skill.get_applications(state):
                skill_apps.append(skill_app)

        skill_apps = self.which_cls.sort(state, skill_apps)
        # print('---')
        for skill_app in skill_apps:
            skill, match, when_pred = skill_app.skill, skill_app.match, skill_app.when_pred
            when_pred = 1 if when_pred is None else when_pred
            print(f"{when_pred:.2f} {match[0].id} -> {skill(*match)}",
                [(m.id,getattr(m, 'value', None)) for m in match][1:], str(skill.how_part))

        skill_apps = self.action_filter(state, skill_apps)

        # print("N SKILL APPS", len(skill_apps))
        # print('-^-')
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

            # Append to Skill 
            skill_app.skill.skill_apps[skill_app.uid] = skill_app
            
            self.prev_skill_app = skill_app

            output = skill_app.sai if(return_kind == 'sai') else skill_app
                            
            if(json_friendly):
                output = output.get_info()
        
        # print(action_uid(state, output))
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

        for skill_app in skill_apps:
            skill_app.skill.skill_apps[skill_app.uid] = skill_app

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
        arg_foci=None, skill_label=None, skill_uid=None, how_help=None):
        

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
            # print("EXPLANATION REQUIRES GENERALIZATION")
            input_attr, inp = list(sai.inputs.items())[0]

            for skill in skills_to_try:
                # Execute how-search to depth 1 with each skill's how-part
                if(hasattr(skill.how_part,'__call__')):
                    if(arg_foci is not None and skill.how_part.n_args != len(arg_foci)):
                        continue 
                    
                    explanation_set = self.how_lrn_mech.get_explanations(
                        state, inp, arg_foci=arg_foci, function_set=[skill.how_part],
                        how_help=how_help, search_depth=1, min_stop_depth=1, min_solution_depth=1)


                    for _, match in explanation_set:
                        if(len(match) != skill.how_part.n_args):
                            continue

                        match = [sai.selection, *match]
                        # print("<<", _, f'[{", ".join([x.id for x in match])}])')
                        skill_app = SkillApplication(skill, state, match)
                        # print(inp, skill.how_part, match)
                        print("CAND", skill_app.sai, "Target", sai)

                        if(skill_app is not None):
                            skill_apps.append(skill_app)

                # For skills with constant how-parts just check equality
                else:
                    if(skill.how_part == inp):
                        skill_apps.append(SkillApplication(skill, state, [sai.selection]))                        

            # if(len(skill_apps) > 0): print("EXPL HOW")
        # for sa in skill_apps:
        #     print("::", sa) # TODO
        return skill_apps
        # best_expl = self.choose_best_explanation(state, skill_apps)
        # if(best_expl is not None):
        #     print(best_expl.skill.where_lrn_mech.conds)
        # print("BEST EXPLANATION", best_expl)
        # return best_expl
    def explain_from_funcs(self, state, sai, 
        arg_foci=None, skill_label=None, skill_uid=None, how_help=None):

        # TODO: does not currently support multiple inputs per SAI.
        inp_attr, inp = list(sai.inputs.items())[0]
        if(not sai.action_type.get(inp_attr,{}).get('semantic',False)):
            return self.how_lrn_mech.new_explanation_set([(inp, [])])
            

        # Use how-learning mechanism to produce a set of candidate how-parts
        explanation_set = self.how_lrn_mech.get_explanations(
                state, inp, arg_foci=arg_foci, how_help=how_help)

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
                expl = {
                    "func" : func_get_info(func, ignore_funcs=self.conversions),
                    "args" : [a.id for a in args],
                }

                # If is not constant add the values that were used 
                if(isinstance(func, CREFunc)):
                    head_vals = []
                    for i, hvs in enumerate(func.head_vars):
                        hv = hvs[0] #TODO: currently only supports single head for each base var
                        head_vals.append(args[i].resolve_deref(hv))
                    expl['head_vals'] = head_vals

                _func_expls.append(expl)
            func_explanations = _func_expls

        out = {
            "skill_explanations" : skill_explanations,
            "func_explanations" :  func_explanations,
        }
        return out

    def explain_demo(self, state, sai, arg_foci=None, skill_label=None, skill_uid=None, how_help=None,
             json_friendly=False, force_use_funcs=False, **kwargs):
        ''' Explains an action 'sai' first using existing skills then using function_set''' 
        state = self.standardize_state(state)
        sai = self.standardize_SAI(sai)
        arg_foci = self.standardize_arg_foci(arg_foci, kwargs)

        with PrintElapse("EXPLAIN TIME"):
            skill_explanations = self.explain_from_skills(state, sai, arg_foci,
                skill_label, skill_uid, how_help=how_help)

            skill_explanations = self.best_skill_explanations(state, skill_explanations)

            func_explanations = None
            if(force_use_funcs or len(skill_explanations) == 0):
                func_explanations = self.explain_from_funcs(state, sai, arg_foci,
                 skill_label, how_help=how_help)

            if(skill_explanations):
                print("--SKILLS--")
                for sa in skill_explanations:
                    print(sa.skill.how_part, sa.match[0].id, [m.id for m in sa.match[1:]])

        
            if(func_explanations):
                print("--FUNCS--")
                for f,match in func_explanations:
                    print(f, [m.id for m in match])

            # import time
            # time.sleep(5)

            if(json_friendly):
                return self._as_json_friendly_expls(skill_explanations, func_explanations)
            
        
        return skill_explanations, func_explanations

# ------------------------------------------------
# : Get state UID

    def get_state_uid(self, state, **kwargs):
        state = self.standardize_state(state)
        return state.get('__uid__')
# ------------------------------------------------
# : Predict Next State

    def predict_next_state(self, state, sai, json_friendly=False, **kwargs):
        ''' Given a 'state' and 'sai' use the registered ActionType definitions to produce
             the new state as a result of applying 'sai' on 'state'. '''
        state = self.standardize_state(state)
        sai = self.standardize_SAI(sai)
        at = sai.action_type
        # print("PREDICT NEXT STATE", sai)

        # Special null state for done
        if(sai.selection.id == "done"):
            # print("SEL IS DONE!")
            next_wm = MemSet()
            next_state = self.standardize_state(next_wm)
            next_state.is_done = True
        # Otherwise standarize the input state
        else:
            next_wm = at.predict_state_change(state.get('working_memory'), sai)
            next_state = self.standardize_state(next_wm)
        
        if(json_friendly):
            state_uid = state.get('__uid__')
            next_state_uid = next_state.get('__uid__')
            next_state = {'state_uid' : state_uid, 'next_state_uid' : next_state_uid,
                          'next_state': next_wm.as_dict(key_attr='id'),
                         }
            # print("next_state RESP:", next_state['next_state_uid'])
            # for _id, d in next_state['next_state'].items():
            #     print(_id, d.get('value',None))
        return next_state

# ------------------------------------------------
# : Train

    def best_skill_explanations(self, state, skill_apps, alpha=0.1):
        def get_score(skill_app):
            score = skill_app.skill.where_lrn_mech.score_match(state, skill_app.match)
            # print("SCORE", score, skill_app)
            return score

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

    def train(self, state, sai=None, reward:float=None, arg_foci=None, how_help=None, 
              skill_label=None, skill_uid=None, uid=None, explanation_selected=None,
              ret_train_expl=False, add_skill_info=False, **kwargs):
        # print("SAI", sai)
        if(skill_label == "NO_LABEL"): skill_label = None
        state = self.standardize_state(state)
        sai = self.standardize_SAI(sai)        
        arg_foci = self.standardize_arg_foci(arg_foci, kwargs)

        # print("---------------------------")
        # for fact in state.get('working_memory').get_facts():
        #     print(repr(fact))
        # print("---------------------------")


        skill_app = None

        # print("--TRAIN:", sai.selection.id, sai.inputs['value'])

        # Feedback Case : Train according to uid of the previous skill_app        
        # print("::", skill_uid, uid, kwargs)
        if(uid is not None):
            if(skill_uid is not None):
                skill_app = self.skills[skill_uid].skill_apps[uid]

            for skill in self.skills.values():
                if(uid in skill.skill_apps):
                    skill_app = skill.skill_apps[uid]
                    break

        if(skill_app is None):
            # Feedback Case : just train according to the last skill application.
            if(self.prev_skill_app is not None and self.prev_skill_app.sai == sai):

                skill_app = self.prev_skill_app
            # Demonstration Case : try to explain the sai from existing skills.
            else:
                skill_explanations, func_explanations = \
                    self.explain_demo(state, sai, arg_foci, skill_label, skill_uid, how_help)

                # print("-->", explanation_selected)
                if(len(skill_explanations) > 0):
                    skill_app = None
                    if(explanation_selected is not None):
                        esd = explanation_selected['data']
                        choice_args = tuple(esd.get('args',[]))
                        for sa in skill_explanations:
                            sa_args = tuple([m.id for m in sa.match[1:]])
                            if( (sa.uid == esd.get('uid',None) or 
                                 repr(sa.skill.how_part) == esd.get('repr', None)
                                 ) and sa_args==choice_args):
                                skill_app = sa
                                break
                    if(skill_app is None):
                        skill_app = skill_explanations[0]                        
                else:
                    skill_app = self.induce_skill(state, sai, func_explanations, skill_label,
                        explanation_selected=explanation_selected)

        # print("ANNOTATE ARG FOCI:", arg_foci)
        skill_app.annotate_train_data(reward, arg_foci, skill_label, skill_uid, 
            how_help, explanation_selected, **kwargs)

        skill_app.skill.ifit(state, skill_app, reward)
        # self.state.clear()

        # Return the unique id of the skill that was updated
        return skill_app.skill.uid

    def train_all(self, training_set, states={}, **kwargs):
        skill_app_uids = []
        for example in training_set:
            state = example['state']
            del example['state']

            # If 'state' is a uid find it's object from 'states'.
            if(isinstance(state,str)):
                state = states[state]

            uid = self.train(state, **example, **kwargs)
            skill_app_uids.append(uid)
        return skill_app_uids


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
                "uid" : skill_app.uid,
                "state_uid" : uid,
                "next_state_uid" : nxt_uid,
                "skill_app" : skill_app,
            }
            actions[skill_app.uid] = action_obj

        # Ensure Depth Counts long enought
        while(depth >= len(depth_counts)):
            depth_counts.append(0)
        depth_index = depth_counts[depth]

        
        if(nxt_uid not in states):
            state_obj = {"state": next_state, "uid" : nxt_uid, "depth" : depth, "depth_index" : depth_index}
            states[nxt_uid] = state_obj
            uid_stack.append(nxt_uid)
            depth_counts[depth] += 1
            if(getattr(next_state, 'is_done', False)):
                print("--------IS DONE!!!---------", nxt_uid)
                state_obj['is_done'] = True
        else:
            state_obj = states[nxt_uid]
            if(depth > state_obj['depth']):
                state_obj['depth'] = depth
                state_obj['depth_index'] = depth_index
                depth_counts[depth] += 1
        
        
        # if(skill_app is not None):
        #     state_obj = states[uid]
        #     out_uids = state_obj.get('out_skill_app_uids', [])
        #     out_uids.append(skill_app.uid)            
        #     state_obj['out_skill_app_uids'] = out_uids

        #     nxt_state_obj = states[nxt_uid]
        #     in_uids = state_obj.get('in_skill_app_uids', [])
        #     in_uids.append(skill_app.uid)
        #     nxt_state_obj['in_skill_app_uids'] = in_uids

    def annotate_verified(self, states, actions):
        verified_state_uids = set()
        certain_state_uids = set()

        verified_state_uids.add(list(states.values())[0]['uid'])
        certain_state_uids.add(list(states.values())[0]['uid'])

        verified_action_uids = set()
        certain_action_uids = set()
        out_uids = {}        

        # Actions should already be in breadth-first order 
        for a_uid, action in actions.items():
            state_uid = action['state_uid']
            next_state_uid = action['next_state_uid']
            skill_app = action['skill_app']

            out_uids[state_uid] = out_uids.get(state_uid,[])
            out_uids[state_uid].append(a_uid)

            print(":", state_uid in verified_state_uids, hasattr(skill_app, 'train_time'), getattr(skill_app, 'reward', 0))
            if(hasattr(skill_app, 'train_time') and
               getattr(skill_app, 'reward', 0) > 0
               ):
                verified_action_uids.add(a_uid)
                verified_state_uids.add(state_uid)
                # Don't add done state
                if(not states[next_state_uid].get('is_done', False)):
                    verified_state_uids.add(next_state_uid)

                skill_app.skill.ifit(states[action['state_uid']]['state'], skill_app, 1)
                

            if(getattr(skill_app, 'reward', 0) == 1 or
               getattr(skill_app, 'when_pred', 0) == 1):

                certain_action_uids.add(a_uid)
                certain_state_uids.add(state_uid)
                # Don't add done state
                if(not states[next_state_uid].get('is_done', False)):
                    certain_state_uids.add(next_state_uid)

        on_verified_path = copy(verified_action_uids)
        for s_uid, s_obj in states.items():
            outs = out_uids.get(s_uid, [])
            if(s_uid in verified_state_uids and len(outs) == 1):
                print("Add VERIFIED", actions[outs[0]])
                on_verified_path.add(outs[0])


        # print("VERIFIED:")
        # print([x[:8] for x in verified_state_uids])

        # print("CERTAIN:")
        # print([x[:8] for x in certain_state_uids])

        # for a_uid, action in actions.items():
        #     if(#action['next_state_uid'] not in verified_state_uids and
        #        action['state_uid'] not in verified_state_uids and
        #        a_uid not in on_verified_path
        #         ):
        #         print("IMPLICIT NEGATIVE", action)
        #         skill_app = action['skill_app']
        #         skill_app.skill.ifit(states[action['state_uid']]['state'], skill_app, -1)



    def act_rollout(self, state, max_depth=-1, halt_policies=[], json_friendly=False,
                    base_depth=0, **kwargs):
        ''' 
        Applies act_all() repeatedly starting from 'state', and fanning out to create at 
        tree of all action rollouts up to some depth. At each step in this process the agent's 
        actions produce subsequent states based on the default state change defined by each 
        action's ActionType object. A list of 'halt_policies' specifies a set of functions that 
        when evaluated to false prevent further actions. Returns a tuple (states, action_infos).
        '''
        with PrintElapse("ACT ROLLOUT"):
            print("\n\t\tSTART ACT ROLLOUT\t\t", base_depth)
            state = self.standardize_state(state)
            curr_state_uid = state.get('__uid__')
            # print(state.get('working_memory'))
            halt_policies = [self.standardize_halt_policy(policy) for policy in halt_policies]

            states = {}
            actions = {}
            uid_stack = []
            depth_counts = []
            self._insert_rollout_skill_app(None, state, None,
                         states, actions, uid_stack, depth_counts, base_depth)

            
            while(len(uid_stack) > 0):
                print("RECURSE", uid_stack)
                # for _ in range(len(uid_stack)):
                uid = uid_stack.pop()
                depth = states[uid]['depth']+1
                state = states[uid]['state']
                src_wm = state.get("working_memory")
                print("---source---", uid)
                # print(repr(src_wm))
                skill_apps = self.act_all(state, return_kind='skill_app')

                for skill_app in skill_apps:
                    skill_app.skill.skill_apps[skill_app.uid] = skill_app
                    at = skill_app.sai.action_type
                    # print("---skill_app :", skill_app)
                    # print("---action_type :", at)

                    next_state = self.predict_next_state(src_wm, skill_app.sai)
                    # next_wm = at.predict_state_change(src_wm, skill_app.sai)
                    # print("---dest---")
                    # print(repr(dest_wm))
                    # next_state = self.standardize_state(next_wm)
                    
                    self._insert_rollout_skill_app(state, next_state, skill_app,
                                    states, actions, uid_stack, depth_counts, depth)
                # print(uid_stack)
            # self.annotate_verified(states, actions)
            
            if(json_friendly):
                for state_obj in states.values():
                    state_obj['state'] = state_obj['state'].get("py_dict")
                # states = {uid : state for uid, state in states.items()}
                for action in actions.values():
                    print("ROLLOUT APP:")
                    action['skill_app'] = action['skill_app'].get_info()

            from pprint import pprint

            # print("curr_state_uid:",  curr_state_uid)
            # print(states[curr_state_uid])
            # print("\nstates:")
            # for i,(uid, obj) in enumerate(states.items()):
            #     if(not obj.get('uid',False)):
            #         print(obj)
            #         raise ValueError("THIS HAPPENED.... WHY!!")
            #     print(i, uid) #obj.get('out_skill_app_uids',[]))

            # print("\nactions:")
            # for i,(uid, obj) in enumerate(actions.items()):
            #     print(i, uid, obj['skill_app'])
            #     pprint(obj)
            # import time
            # time.sleep(.5)

            return {"curr_state_uid" :  curr_state_uid, 
                    "states" : states,
                    "actions" : actions,
                    }
            # print(actions.key)

    def gen_completeness_profile(self, start_states, output_file, **kwargs):
        '''Generates a ground-truth completeness profile. Should be called on agents known 
            to exhibit a particular definition of model-complete behavior. The profile is
            generated by building rollouts from each state in the list of given start_states.'''
            
        import json, os

        # print("WRITING TO", os.path.abspath(output_file))
        with open(output_file, 'w') as profile:
            for state in start_states:
                ro = self.act_rollout(state)
                states, actions = ro['states'], ro['actions']
                for state_uid, state_obj in states.items():
                    if(not state_obj.get('is_done', False)):
                        sais = []
                        for a_uid, action in actions.items():
                            if(action['state_uid'] == state_uid):
                                sai = action['skill_app'].sai
                                sais.append(sai.get_info())    
                        profile.write(json.dumps({'state' : state_obj['state'].get("py_dict"), 'sais' : sais})+"\n")
        # print("PROFILE DONE", os.path.abspath(output_file))

    def eval_completeness(self, profile, partial_credit=False,
                          print_diff=True,
                         **kwargs):
        ''' Evaluates an agent's correctness and completeness against a completeness profile.'''
        import json, os

        n_correct, total = 0, 0
        n_first_correct, total_states = 0,0
        with open(profile, 'r') as profile_f:
            for line_ind, line in enumerate(profile_f):
                # Read a line from the profile
                item = json.loads(line)

                
                    
                # Get the ground-truth sais
                profile_sais = item['sais']
                state = item['state']
                agent_sais = self.act_all(state, return_kind='sai')

                # Find the difference of the sets 
                profile_sai_strs = set([str(s) for s in profile_sais])
                agent_sai_strs = [str(s.get_info()) for s in agent_sais]
                diff = profile_sai_strs.symmetric_difference(set(agent_sai_strs))
                if(print_diff and len(diff) > 0):
                    # Print Problem
                    gv = lambda x : item['state'][x]['value']
                
                    # print(list(item['state'].keys()))
                    print(line_ind+1, ">>",f"{gv('inpA3')}{gv('inpA2')}{gv('inpA1')} + {gv('inpB3')}{gv('inpB2')}{gv('inpB1')}")
                    # print(list(state.keys()))
                    # print("----------------------")
                    # for key, obj in state.items():
                    #     print(key, obj)
                    print("AGENT:")
                    for x in agent_sai_strs:
                        print(x)
                    print("TRUTH:")
                    for x in profile_sai_strs:
                        print(x)
                    print("----------------------")
                    print()

                # print("LINE", total_states, len(diff) == 0)
                if(partial_credit):
                    total += len(profile_sai_strs)
                    n_correct += max(0, len(profile_sai_strs)-len(diff))
                else:
                    total += 1
                    n_correct += len(diff) == 0

                total_states += 1
                if(len(agent_sai_strs) > 0 and agent_sai_strs[0] in profile_sai_strs):
                    n_first_correct += 1
            
        completeness = n_correct / total
        correctness = n_first_correct / total_states

        print(f"Correctness : {correctness*100:.2f}%")
        print(f"Completeness : {completeness*100:.2f}%")
        return {"completeness" : completeness, "correctness" : correctness}












                















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


