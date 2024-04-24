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
import itertools
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
                skill_app = SkillApplication(self, match, state, when_pred=when_pred)
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


        # print("FIT", reward)

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


# TODO: Need to think about whether it makes sense for SkillApps
#  to keep a reference to their states and next_states,
#  is a possible memory leak opportunity.
KEEP_STATE_REFS = True


class SkillApplication(object):
    # __slots__ = ("skill", "match", "sai")
    def __new__(cls, skill, match, state, uid=None,
                next_state=None, prob_uid=None, short_name=None,
                reward=None, when_pred=None):
        # print(skill, [m.id for m in match])
        sai = skill(*match)

        # Find the unique id for this skill_app
        state_uid = state.get("__uid__", None)
        h = hashlib.sha224()
        h.update(skill.uid.encode('utf-8'))
        h.update(state_uid.encode('utf-8'))
        h.update(",".join([m.id for m in match]).encode('utf-8'))
        uid = f"A_{base64.b64encode(h.digest(), altchars=b'AB')[:30].decode('utf-8')}"

        agent = getattr(skill,'agent', None)

        # If this skill has been fit with this skill_app then return the previous instance
        # if(uid in skill.skill_apps):
        #     self = skill.skill_apps[uid]
        if(agent):
            if(agent and uid in agent.skill_apps_by_uid):
                self = agent.skill_apps_by_uid[uid]
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
        self.implicit_rewards = {}
        self.implicit_dependants = {}
        self.explicit_reward = reward
        self.implicit_reward = None
        
        if(not hasattr(self, "when_pred") or when_pred is not None):
            self.when_pred = when_pred
        if(not hasattr(self, "prob_uid") or prob_uid is not None):
            self.prob_uid = prob_uid

        if(KEEP_STATE_REFS):
            self.state = state
            self.next_state = next_state

        return self

    @property
    def reward(self):
        explicit_reward = getattr(self, 'explicit_reward', None)
        implicit_reward = getattr(self, 'implicit_reward', None)
        if(explicit_reward is not None):
            return explicit_reward
        elif(implicit_reward is not None):
            return implicit_reward
        return None
    
    def annotate_train_data(self, reward, arg_foci, skill_label, skill_uid,
                            how_help, explanation_selected, is_demo=False, **kwargs):
        
        # self.reward = reward
        self.explicit_reward = reward
        self.arg_foci = arg_foci
        self.skill_label = skill_label
        self.skill_uid = skill_uid
        self.how_help = how_help
        self.explanation_selected = explanation_selected
        self.is_demo = is_demo

    def add_seq_tracking(self, prob_uid=None):
        if(prob_uid is not None):
            self.prob_uid = prob_uid

        agent = self.skill.agent
        if(agent and self.uid not in agent.skill_apps_by_uid):
            agent.skill_apps_by_uid[self.uid] = self     
            by_s_uid = agent.skill_apps_by_state_uid.get(self.state_uid, set())   
            by_s_uid.add(self)
            agent.skill_apps_by_state_uid[self.state_uid] = by_s_uid        

            if(hasattr(agent, 'rollout_preseq_tracker')):
                # print(self)
                if(self.next_state is None):
                    # If cannot predict next state or skill_app doesn't change the state
                    #  then don't keep
                    try:
                        self.next_state = agent.predict_next_state(self.state, self.sai)
                    except:
                        print("DID FAIL", self)
                        return False

                    # print("NEXT STATE", self.sai[0], self.sai[2]['value'], self.next_state.get('__uid__')[:5])
                    # print(repr(self.next_state.get("working_memory")))
                # print("THIS IS START", state.is_start, getattr(state,'is_start', None))
                agent.rollout_preseq_tracker.add_skill_app(self, getattr(self.state,'is_start', None), do_update=False)
        return True

    def remove_seq_tracking(self):
        agent = self.skill.agent
        if(self.state_uid in agent.skill_apps_by_state_uid):
            by_s_uid = agent.skill_apps_by_state_uid[self.state_uid]
            by_s_uid.remove(self)
        del agent.skill_apps_by_uid[self.uid]
        agent.rollout_preseq_tracker.remove_skill_app(self, getattr(self.state,'is_start', None), do_update=False)

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
            'when_pred' : self.when_pred,
            'in_process' : getattr(self, 'in_process', False),
            
        }
        if(self.skill and len(self.match) > 1):
            hvs = self.skill.how_part.head_vars
            head_vals = [hv[0](m) for hv, m in zip(hvs, self.match[1:])]
            info['head_vals'] = head_vals

        if(getattr(self, 'path', None)):
            info['path'] = self.path.get_info()
            info['internal_unordered'] = self.path.is_internal_unordered
            info['initial_unordered'] = self.path.is_initial_unordered

        if(getattr(self, 'certainty', None)):
            info['certainty'] = self.certainty
            info['cert_diff'] = self.cert_diff

        if(getattr(self, 'removed', None)):
            info['removed'] = self.removed

        if(getattr(self, 'unordered_group', None)):
            info['unordered_group'] = self.unordered_group
            # info['group_next_state_uid'] = self.group_next_state_uid

        if(hasattr(self, 'train_time')):
            train_data = {}
            train_data['train_time'] = getattr(self, 'train_time', None)
            train_data['explicit_reward'] = getattr(self, 'explicit_reward', None)
            train_data['implicit_reward'] = getattr(self, 'implicit_reward', None)
            train_data['reward'] = self.reward #getattr(self, 'reward', None)
            train_data['arg_foci'] = getattr(self, 'arg_foci', None)
            train_data['skill_label'] = getattr(self, 'skill_label', None)
            train_data['skill_uid'] = getattr(self, 'skill_uid', None)
            train_data['how_help'] = getattr(self, 'how_help', None)
            train_data['explanation_selected'] = getattr(self, 'explanation_selected', None)
            train_data['is_demo'] = getattr(self, 'is_demo', False)
            train_data = {k:v for k,v in train_data.items() if v is not None}
            if('arg_foci' in train_data):
                # print("ARG FOCI", train_data['arg_foci'])

                # TODO: Fix whatever causing this to be necessary
                # arg_foci = train_data['arg_foci']
                # if(len(arg_foci) > 0 and isinstance(arg_foci[0], list)):
                #     arg_foci = arg_foci[0]
                train_data['arg_foci'] = [m if isinstance(m,str) else m.id for m in train_data['arg_foci']]
            if('explicit_reward' in train_data):
                train_data['confirmed'] = True
            info.update(train_data)

        return info

    def as_train_kwargs(self):
        return {'sai': BaseSAI(*self.sai.as_tuple()),
                'arg_foci' : [m.id for m in self.args],
                'how_str' : "???"}

    def __repr__(self, add_sai=True):
        app_str = f'{self.skill}({", ".join([m.id for m in self.args])})'
        if(add_sai): 
            return f'{app_str} -> {self.sai}'
        else:
            return app_str

    def __eq__(self, other):
        return getattr(self, 'uid', None) == getattr(other, 'uid', None)

    def __hash__(self):
        return hash(self.uid)

    def update_implicit_reward(self):
        old_implicit_reward = self.implicit_reward
        if(len(self.implicit_rewards) == 0):
            self.implicit_reward = None
        else:
            max_rew = None
            for src, (_, r) in self.implicit_rewards.items():
                if( src.explicit_reward is None or
                    src.explicit_reward <= 0):
                    continue
                if(max_rew is None or r > max_rew): 
                    max_rew = r
            self.implicit_reward = max_rew
        return old_implicit_reward == self.implicit_reward, self.implicit_reward

    def add_implicit_reward(self, depends, reward, update=False):
        for other in depends:
            self.implicit_rewards[other] = (depends, reward)
            other.implicit_dependants[self] = (depends, reward)

        # Recalculate the value of implicit_reward 
        if(update):
            return self.update_implicit_reward()
        return False

    def remove_implicit_reward(self, other, update=False):
        if(other in self.implicit_rewards):
            depends, reward = self.implicit_rewards[other]
            for dep in depends:
                del self.implicit_rewards[dep]
                if(self in dep.implicit_dependants):
                    del dep.implicit_dependants[self]

        # Recalculate the value of implicit_reward 
        if(update):
            return self.update_implicit_reward()
        return False
        
    def clear_implicit_dependants(self, update=True):
        changed_depends = []
        for dep in [*self.implicit_dependants]:
            did_change = dep.remove_implicit_reward(self, update)
            if(did_change):
                changed_depends.append(dep)


        # if(clear_fit):
        #     for sa, (depends,reward) in self.implicit_rewards.items():

        #         sa.skill.ifit(sa.state, sa, None)
        # old_implicit_rewards = self.implicit_rewards
        # self.implicit_rewards = {}
        return changed_depends

    def ifit_implicit_dependants(self):
        agent = self.skill.agent
        for dep in self.implicit_dependants:
            did_update, impl_rew = dep.update_implicit_reward()
            if(did_update and dep.explicit_reward is None):
                print("IMPLICIT", dep.state.get("__uid__")[:5], dep, impl_rew)
                agent._ifit_skill_app(dep, impl_rew)
        # if(self.explicit_reward is not None and self.explicit_reward > 0):
            
        # elif(self.explicit_reward is None or self.explicit_reward <= 0):
        #     for dep in self.implicit_dependants:
        #         did_update, impl_rew = dep.update_implicit_reward()
        #         if(did_update and dep.explicit_reward is None):
        #             agent._ifit_skill_app(dep, impl_rew)

    #     for sa, (depends, reward) in self.implicit_rewards.items():
            # # Don't override explicit rewards
            # if(hasattr(sa, 'explicit_reward')):
            #     continue
            # if(old_implicit_rewards.get(sa, None) == reward):
            #     continue

            # self.agent._ifit_skill_app(self.state, self, reward)
            # if(getattr(imp_neg_sa, "reward", 0) == 0):
            # sa.skill.ifit(sa.state, sa, reward)
                # self.train(imp_neg_sa.state, reward=-1, skill_app=imp_neg_sa)


    def ensure_when_pred(self):
        # print("ENSURE", self)
        self.when_pred = self.skill.when_lrn_mech.predict(self.state, self.match)


# ----------------------
# : AppGroupAnnotation
class AppGroupAnnotation():
    def __init__(self, app_group, kind):
        self.app_group = app_group
        self.kind = kind

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

        self.process_lrn_mech = None
        if(self.process_cls):
            self.process_lrn_mech = self.process_cls(self, **self.process_args)
            if(self.track_rollout_preseqs):
                from .learning_mechs.process.process import PreseqTracker
                self.rollout_preseq_tracker = PreseqTracker()

        self.working_memory = MemSet()
        self.init_processesors()
        self.skills = {}
        self.skill_apps_by_uid = {}
        self.skill_apps_by_state_uid = {}
        # self.implicit_negs = {}
        self.skills_by_label = {}
        self.prev_skill_app = None
        self.episodic_memory = {}
        self.group_annotations = []


    def standardize_state(self, state, is_start=None, **kwargs):
        if(not isinstance(state, self.state_cls)):
            if(isinstance(state, State)):
                state = self.state_cls(state.state_formats)
            else:
                state_uid = kwargs.get('state_uid', None)
                if(isinstance(state, dict)):

                    # NOTE This is a fix for legacy attribute 'contentEditable'
                    for k,obj in state.items():
                        if('contentEditable' in obj):
                            obj['locked'] = not obj['contentEditable']
                            del obj['contentEditable']

                    state_uid = state.get("__uid__", None) if(state_uid is None) else state_uid
                    if self.should_find_neighbors:
                        state = encode_neighbors(state)
                        # print()
                        # for _id, obj in state.items():
                        #     print(obj)
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
        if(getattr(state, 'is_start',None) is None):
            state.is_start = is_start 
        return state

    def standardize_SAI(self, sai):
        if(isinstance(sai, SkillApplication)):
            sai = sai.sai
        if(isinstance(sai, BaseSAI)):
            # Always copy the SAI to avoid side effects in the caller
            sai = SAI(sai.selection, sai.action_type, sai.inputs)
        else:
            sai = SAI(sai)
        if(isinstance(sai.selection, str)):
            try:
                sai.selection = self.state.get('working_memory').get_fact(id=sai.selection)
            except KeyError:
                # print(self.state.get('working_memory'))
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

    def _organize_mutl_excl(self, in_process_grps):
        mut_excl_grps = []
        for i, grp in enumerate(in_process_grps):
            # print("<<", i)

            filtered_grp = []
            for sa in grp:
                # Weh
                if(not sa.skill.where_lrn_mech.check_match(sa.state, sa.match)):
                    continue

                sa.in_process = True
                sa.ensure_when_pred()
                filtered_grp.append(sa)

            prefix_groups = group_by_path(filtered_grp)


            me_grp = []
            for pre_grp in prefix_groups.values():
                me_grp.append(pre_grp)
            mut_excl_grps.append(me_grp)
        return mut_excl_grps

    def _add_process_implicit_rewards(self, mut_excl_grps):
        # for i, disj_grps in enumerate(mut_excl_grps):
        #     for j, grp in enumerate(disj_grps):
        #         # Reset implicit rewards of this skill app on other
        #         #  skill apps
        #         for sa_a in grp:
        #             sa_a.clear_implicit_rewards()

        apps_so_far = []
        for i, disj_grps in enumerate(mut_excl_grps):
            for j, grp in enumerate(disj_grps):
                
                # Add implicit negatives between skill_apps
                #  that are part of disjoint groups.
                for k, other_grp in enumerate(disj_grps):
                    if(j == k):
                        continue
                    # other_grp = disj_grps[k]
                    for sa_a in grp:
                        for sa_b in other_grp:
                            # print("This doesn't go with this:")
                            # print("\t", a)
                            # print("\t", b)
                            sa_a.add_implicit_reward(sa_b, -1)

                
                for sa_a in grp:
                    # Add implicit negatives for all skill_apps preceeding
                    #  those in the current group, and for each of the 
                    #  preceeding skill_apps add implicit no reward on
                    for sa_b in apps_so_far:
                        sa_a.add_implicit_reward(sa_b, -1)
                        sa_b.add_implicit_reward(sa_a, None)                        

            # print("DISJ", disj_grps)
            # print("CHAIN", list(chain(disj_grps)))
            apps_so_far += chain(*disj_grps)

        # for i, disj_grps in enumerate(mut_excl_grps):
        #     for j, grp in enumerate(disj_grps):
        #         # If a skill app already has reward == 1 then
        #         #  apply it's implicit rewards.
        #         for sa_a in grp:
        #             if(getattr(sa_a, 'reward', 0) == 1):
        #                 sa_a.apply_implicit_rewards()

    def _add_conflict_certainty(self, skill_apps):
        # Calculate certainty of each skill_app in a conflict set of possible  
        #  next skill_apps. Helpful for choosing apps to show to user.
        if(len(skill_apps) == 0):
            return

        # Add n_path_apps 
        for skill_app in skill_apps:
            path = getattr(skill_app, 'path', None)
            if(path is not None):
                skill_app.n_path_apps = len(path.get_item().skill_apps)

        # Collect when_preds, when_preds, n_apps
        when_preds = np.array([sa.when_pred for sa in skill_apps], dtype=np.float64)
        in_process = np.array([getattr(sa,'in_process', False) for sa in skill_apps],dtype=np.bool_)
        n_apps = np.array([getattr(sa,'n_path_apps', False) for sa in skill_apps],dtype=np.bool_)

        def is_mid_unordered_grp(sa):
            path = getattr(sa, "path", None)
            if(path):
                return path.is_initial_unordered
                # macro, meth_ind, item_ind, cov = path.steps[-1]
                # return cov and len(cov) > 0
            return False


        mid_unord = np.array([is_mid_unordered_grp(sa) for sa in skill_apps],dtype=np.bool_)
        print("MID UNORD", mid_unord)
        avg_iproc_pred, avg_iproc_n_apps = 0, 0
        mask = in_process & (when_preds > 0)
        if(np.sum(mask) > 0):
            avg_iproc_pred = np.average(when_preds[mask])
            avg_iproc_n_apps = np.average(n_apps[mask])

        # Initial certainty is the when predictions
        certainty = when_preds.copy()

        # Reduce out-of-process certainties by a function of the certainties of
        #  in-process action's and their numbers of supporting skill_apps.
        #  If there are any mid-unordered group apps then double the reduction.
        certainty[~in_process] /= (1.0 + max(avg_iproc_pred-1/(1+avg_iproc_n_apps), 0))*(1+mid_unord.any())



        max_cert = np.max(certainty)
        cert_diffs = max_cert-certainty

        for sa, cert, cert_diff in zip(skill_apps, certainty, cert_diffs):
            sa.certainty = np.nan_to_num(cert, 0)
            sa.cert_diff = np.nan_to_num(cert_diff, 1)

    def get_skill_applications(self, state,
            is_start=None,
            prob_uid=None, 
            eval_mode=False,
            add_out_of_process=False,
            ignore_filter=False,
            add_conflict_certainty=False,
            add_known=True,
            hard_cert_thresh=None,
            **kwargs):
        skill_apps = set()

        if(prob_uid is None and is_start):
            prob_uid = state.get("__uid__")
        
        # print("\nGET SKILL APPS", state.get('__uid__')[:5])
        
        

        # If there is a process learning mechanism then use it to 
        #  generate "in-process" skill applications from its grammar.
        apps_in_process = False 
        if(self.process_lrn_mech):
            
            preseq_tracker = getattr(self,'rollout_preseq_tracker', None)
            # try:
            #     preseq = preseq_tracker.get_good_preseq(state)
            #     print("--", preseq)
            # except RuntimeError as e:
            #     print("--FAIL")
            # print("PROB UID", prob_uid)
            try:
                in_process_grps = self.process_lrn_mech.get_next_skill_apps(
                    state, preseq_tracker,
                    prob_uid=prob_uid, group_by_depends=True)
            except:
                in_process_grps = []

            filtered_skill_apps = []

            # Regroup in_process_grps into mut_excl_grps of form
            #  [ [[...skill_apps for items0],[...skill_apps for items1]] , [...skill_apps for items2] ] 
            #  assuming items0 and items1 are part of disjoint methods that share a macro and items2
            #  are another set of items contiguous with those.
            mut_excl_grps = self._organize_mutl_excl(in_process_grps)
            if("process" in self.implicit_reward_kinds):
                self._add_process_implicit_rewards(mut_excl_grps)
        
            apps_in_process = len(in_process_grps) > 0
            
            if(not apps_in_process):
                print("NOT IN PROCESS", state.get("__uid__")[:5], f"PROB={prob_uid[:5] if prob_uid else prob_uid}")
                # print(self.rollout_preseq_tracker.get_good_preseq(state))
            # if(len(in_process_grps) > 1):
                # print("in_process_grps", in_process_grps)

            # Find the best skill_apps in each disjunction and record the
            #  maximum value among them.
            vals = [-2]*len(mut_excl_grps)
            contig_grps = [None]*len(mut_excl_grps)
            for i, disj_grps in enumerate(mut_excl_grps):
                
                # Find the best group among disj_grps on the basis of the
                #  maximum skill_app reward (or predicted reward) in the group.
                best_d_ind = -1
                best_d_val = -2
                
                disj_apps = []
                for j, grp in enumerate(disj_grps):

                    # Prefer to compute val on basis of the non-optional
                    #  members of the group
                    # def is_optional(sa):
                    #     macro, meth_ind, item_ind, _ = sa.path.steps[-1]
                    #     return macro.methods[meth_ind].optional_mask[item_ind]

                    # if(any([not is_optional(sa) for sa in grp])):
                    #     print("ONLY NO OPTS")
                    #     itr_grp = [sa for sa in grp if not is_optional(sa)]
                    # else:
                    #     itr_grp = grp

                    max_val = -1
                    for skill_app in grp:
                        val = getattr(skill_app, 'reward', None)
                        if(val is None):
                            val = getattr(skill_app, 'when_pred', None)
                        else:
                            # Prefer verified skill_apps over predictions
                            val *= 2 
                        max_val = max(val, max_val)

                    # If max_val is not that different than the best one
                    #  then keep skill_apps from both disjoint groups so
                    #  that the user can resolve the ambiguity. 
                    diff = max_val-best_d_val
                    if(abs(diff) < .25):
                        disj_apps += grp
                    elif(diff > 0):
                        disj_apps = [*grp]

                    if(max_val > best_d_val):
                        best_d_val = max_val
                        best_d_ind = j
                vals[i] = best_d_val
                contig_grps[i] = disj_apps

            # Determine the subset of skill_apps which should be presented
            #  present just the highest prediction group unless there is
            #  some ambiguity (i.e. delta_val < .25) as to which group is best. 
            best_val = -2
            skill_apps = set()
            for i, val in enumerate(vals):
                diff = val-best_val
                if(abs(val) < .25):
                    skill_apps = skill_apps.intersection(contig_grps[i])
                elif(diff > 0):
                    skill_apps = set([*contig_grps[i]])

                if(val > best_val):
                    best_val = val

                # When in eval mode just apply the first skill_app group
                #   which has positive reward
                if(eval_mode and val > 0):
                    break



            # if(len(in_process_grps) >= 1):
            #     best_grp_pred = -2
            #     best_grp_ind = -1
            #     for i, in_process_apps in enumerate(in_process_grps):
            #         if(len(in_process_apps) == 0):
            #             continue
            #         total = 0
            #         for skill_app in in_process_apps:
            #             val = getattr(skill_app, 'reward', None)
            #             if(val is None):
            #                 val = getattr(skill_app, 'when_pred', None)
            #             total += val
            #             # if(reward != -1):
            #             #     filtered_skill_apps.append(skill_app)
            #         avg = total / len(in_process_apps)
            #         if(avg > best_grp_pred):
            #             best_grp_pred = avg
            #             best_grp_ind = i

            #     # if(best_grp_pred > 0):
            #     # print("BEST GRP IND", best_grp_ind, "/", len(in_process_grps), best_grp_pred)
            #     if(best_grp_ind >= 0):
            #         skill_apps = list(in_process_grps[best_grp_ind])



            # if(len(filtered_skill_apps) > 0):
            #     # if(len(in_process_grps) > 1):
            #         # print("<<", filtered_skill_apps)
            #     skill_apps = filtered_skill_apps
            # skills = {sa.skill for sa in in_process_apps}
            # for skill in skills:    
            #     for skill_app in skill.get_applications(state):
            #         if(skill_app in in_process_apps):
            #             path = in_process_apps[skill_app]
            #             skill_app.path = path
            #             skill_apps.append(skill_app)


        if(len(skill_apps) == 0 or add_out_of_process):
            # print("BACKUP", len(skill_apps), add_out_of_process)
            for skill in self.skills.values():
                for skill_app in skill.get_applications(state):
                    if (skill_app not in skill_apps):
                        skill_apps.add(skill_app)
                        skill_app.in_process = False
                        
                    if(prob_uid is not None):
                        skill_app.prob_uid = prob_uid
        
        # print('---')
        # for skill_app in skill_apps:
        #     skill, match, when_pred = skill_app.skill, skill_app.match, skill_app.when_pred
        #     when_pred = 1 if when_pred is None else when_pred
        #     print(f"{' ' if (when_pred >= 0) else ''}{when_pred:.2f} {skill_app}")
        # if(not apps_in_process):
        # print("AN SKILL APPS", len(skill_apps))
        
        # print("BN SKILL APPS", len(skill_apps))
        if(add_known):
            s_uid = state.get('__uid__')
            known_sas = self.skill_apps_by_state_uid.get(s_uid, [])
            for skill_app in known_sas:

                rew = skill_app.reward
                if(rew is None):
                    if(getattr(skill_app, "removed", False) and 
                        skill_app in skill_apps):
                        skill_apps.remove(skill_app)    
                    continue

                skill_apps.add(skill_app)
                skill_app.ensure_when_pred()
            # # Always show skill apps which have positive reward
            # if(rew > 0):
            #     skill_apps.add(skill_app)
            # # Don't keep skill apps which have negative reward
            # elif(rew < 0 and
            #      skill_app in skill_apps):
            #     skill_apps.remove(skill_app)
        if(add_conflict_certainty):
            self._add_conflict_certainty(skill_apps)
            if(hard_cert_thresh is not None):
                skill_apps = [sa for sa in skill_apps if getattr(sa,"explicit_reward", None) is not None or sa.certainty >= hard_cert_thresh]

        if(not ignore_filter):
            # print("DO FILTER", ignore_filter)
            # print("BEFORE FILTER", [getattr(sa,'when_pred', None) for sa in skill_apps])
            skill_apps = self.action_filter(state, skill_apps, **self.action_filter_args)

        
            # print("AFTER FILTER:")
            # for sa in skill_apps:
            #     print(sa.reward, getattr(sa,'when_pred', None), skill_app)
            


        # print("CN SKILL APPS", len(skill_apps))
        skill_apps = self.which_cls.sort(state, skill_apps)

        out = []
        for sa in skill_apps:
            okay = sa.add_seq_tracking(prob_uid)
            if(okay):
                sa.skill.skill_apps[sa.uid] = sa
                out.append(sa)

        return skill_apps

    def act(self, state, 
            return_kind='sai', # 'sai' | 'skill_app'
            json_friendly=False,
            is_start=None,
            prob_uid=None,
            eval_mode=False,
            **kwargs):

        state = self.standardize_state(state, is_start)
        skill_apps = self.get_skill_applications(state,
            is_start=is_start, 
            prob_uid=prob_uid, eval_mode=eval_mode,
            **kwargs)

        # if(self.track_rollout_preseqs):
        #     for skill_app in skill_apps:
                

        # Apply action_chooser to pick from conflict set
        output = None
        if(len(skill_apps) > 0):
            skill_app = self.action_chooser(state, skill_apps)
            # print(">>", skill_app)

            # Append to Skill 
            # skill_app.skill.skill_apps[skill_app.uid] = skill_app
            
            self.prev_skill_app = skill_app

            output = skill_app.sai if(return_kind == 'sai') else skill_app
                            
            if(json_friendly):
                output = output.get_info()
        
        
        return output

    def act_all(self, state,
        max_return=-1,
        return_kind='sai',
        json_friendly=False,
        is_start=None,
        prob_uid=None,
        eval_mode=False,
        add_out_of_process=False,
        add_conflict_certainty=False,
        ignore_filter=False,
        **kwargs):

        state = self.standardize_state(state, is_start)
        skill_apps = self.get_skill_applications(state,
            is_start=is_start,  prob_uid=prob_uid, 
            eval_mode=eval_mode, 
            add_conflict_certainty=add_conflict_certainty,
            add_out_of_process=add_out_of_process,
            ignore_filter=ignore_filter,
            **kwargs)

        # if(self.track_rollout_preseqs):
        #     for skill_app in skill_apps:
        #         self.rollout_preseq_tracker.add_skill_app(skill_app, is_start)#, do_update=False)
        
        # TODO: Not sure if necessary.
        # self.state.clear()

        if(max_return >= 0):
            skill_apps = skill_apps[:max_return]

        # for skill_app in skill_apps:
        #     skill_app.skill.skill_apps[skill_app.uid] = skill_app

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
                        skill_app = SkillApplication(skill, match, state)
                        # print(inp, skill.how_part, match)
                        # print("CAND", skill_app.sai, "Target", sai)

                        if(skill_app is not None):
                            skill_apps.append(skill_app)

                # For skills with constant how-parts just check equality
                else:
                    if(skill.how_part == inp):
                        skill_apps.append(SkillApplication(skill, [sai.selection], state))                        

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

        # If failed bottom-out with a constant how-part.
        if(len(explanation_set) == 0):
            if(self.error_on_bottom_out and not self.is_bottom_out_exception(sai)):
                raise RuntimeError(f"No explanation found for demonstration:\n" +
                     f"\tsai={sai}\n" +
                    (f"\targ_foci={[a.id for a in arg_foci]} with values {[a.value for a in arg_foci]} \n" if arg_foci is not None else "") +
                    f"Set error_on_bottom_out=False in agent config to remove this message"
                    )
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
                        head_vals.append(hv(args[i]))#.resolve_deref())
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
        # arg_foci = list(reversed(arg_foci)) if arg_foci else arg_foci

        print("EXPLAIN DEMO!", sai['selection'], sai['inputs'], [m.id for m in arg_foci] if arg_foci else arg_foci)
        # with PrintElapse("EXPLAIN TIME"):
        skill_explanations = self.explain_from_skills(state, sai,
            arg_foci, 
            # list(reversed(arg_foci)) if arg_foci else arg_foci,
            skill_label, skill_uid, how_help=how_help)

        skill_explanations = self.best_skill_explanations(state, skill_explanations)

        func_explanations = None
        if(force_use_funcs or len(skill_explanations) == 0):
            func_explanations = self.explain_from_funcs(state, sai,
            arg_foci,
             # list(reversed(arg_foci)) if arg_foci else arg_foci,
             skill_label, how_help=how_help)

        # if(skill_explanations):
        #     print("--SKILLS--")
        #     for sa in skill_explanations:
        #         print(sa.skill.how_part, sa.match[0].id, [m.id for m in sa.match[1:]])

    
        # if(func_explanations):
        #     print("--FUNCS--")
        #     for f,match in func_explanations:
        #         print(f, [m.id for m in match])

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
        ''' Given a 'state' and 'sai' or list of sais use the registered ActionType definitions 
             to produce the new state as a result of applying 'sai' on 'state'. '''
        state = self.standardize_state(state)
        state_uid = state.get('__uid__')

        if(isinstance(sai, list)):
            sai_list = sai
            next_state = state
            for sai in sai_list:
                next_state = self.predict_next_state(next_state, sai)
            next_wm = next_state.get('working_memory')
        else:
            
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
                next_state.is_done = False

        next_state_uid = next_state.get('__uid__')
        if(state_uid == next_state_uid):
            print(sai)
            raise ValueError("BAD ACTION")
        # print("PNS", state.get('__uid__')[:5], next_state.get('__uid__')[:5])
        if(json_friendly):
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
            max_score = srted[0][0]

            # If this is turned on then only accept skill explanations
            #  which are 100% matches. This causes the agent induce
            #  more skills, which can avoid cross attribution of examples.
            if(self.one_skill_per_match and max_score < 1.0):
                return []

            # Keep only explanations with scores within alpha% of the max.
            thresh = max_score*(1-alpha)
            k = 1
            for k in range(1,len(srted)):
                if(srted[k][0] < thresh):
                    break

            return [x[1] for x in srted[:k]]
        else:
            return []


    def induce_skill(self, state, sai, explanation_set, label=None, explanation_selected=None):
        # print("INDUCE STATE IS START", state.is_start)

        # TODO: Make this not CTAT specific
        input_attr = list(sai.inputs.keys())[0]
        if(sai.selection.id == "done"):
            how_part, explanation_set, args = -1, None, []
        else:
            # NOTE: This would be simpler with some kind of null-coalesce
            if(explanation_selected is not None and
                'data' in explanation_selected and
                'func' in explanation_selected['data'] and
                'args' in explanation_selected['data'] and
                'repr' in explanation_selected['data']['func']):
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
        # print("INDUCE SKILL", skill, skill.how_part)

        # print("INDUCE SKILL", skill)

        # Add new skill to various collections.
        self.skills[skill.uid] = skill
        if(label is not None):
            label_lst = self.skills_by_label.get(label,[])
            label_lst.append(skill)
            self.skills_by_label[label] = label_lst

        return SkillApplication(skill, [sai.selection,*args], state)

    def _recover_prev_skill_app(self, sai=None,
            uid=None, skill_uid=None, **kwargs):
        skill_app = None
        if(skill_uid is not None and uid is not None):
            if(skill_uid in self.skills):
                if(uid in getattr(self.skills[skill_uid],"skill_apps", [])):
                    skill_app = self.skills[skill_uid].skill_apps[uid]
                    return skill_app

            for skill in self.skills.values():
                if(uid in skill.skill_apps):
                    skill_app = skill.skill_apps[uid]
                    break

        # If sai matches prev_skill_app then use that
        elif(self.prev_skill_app is not None and self.prev_skill_app.sai == sai):

            skill_app = self.prev_skill_app
        return skill_app

    def _find_skill_app(self, state, sai=None, arg_foci=None, 
                how_help=None, uid=None, skill_label=None, skill_uid=None,
                explanation_selected=None, **kwargs):
        if(skill_label == "NO_LABEL"): skill_label = None

        # if(skill_app is None):            
            
        
        # Case: Feedback on Previous Action (as sai or uid)
        skill_app = self._recover_prev_skill_app(sai, uid, skill_uid, **kwargs)
        # else:
            # Case: Feedback on Previous Action (as skill_app)
            # pass


        if(skill_app is None):
            # Cases : 1) SAI is a Demo 
            #             - Explained by existing skills
            #             - Explained from prior knowledge functions
            #            or            
            #         2) SAI which must be re-explained from existing skills             
            # print("-ITRAIN IS START", is_start, state.is_start)
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
                # print("ITRAIN IS START", is_start, state.is_start)
                skill_app = self.induce_skill(state, sai, func_explanations, skill_label,
                    explanation_selected=explanation_selected)

        return skill_app

    def _remove_skill_app(self, skill_app, is_start=None, **kwargs):
        state = skill_app.state
        skill = skill_app.skill

        # Remove form  skill-specific learning mechanisms where, when, which
        skill.where_lrn_mech.remove(state, skill_app.match)
        skill.when_lrn_mech.remove(state, skill_app)
        skill.which_lrn_mech.remove(state, skill_app)

        # TODO: Should remove from seq_tracking?? will ignoring cause issues?
        # prob_uid = None if not is_start else state.get("__uid__")
        # skill_app.add_seq_tracking(prob_uid)

        # Remove global process-learning mechanism
        if(self.process_lrn_mech):
            self.process_lrn_mech.remove(state, skill_app, is_start=is_start)

        # If a skill has no supporting skill_apps then delete it
        skill_app.remove_seq_tracking()
        if(skill_app.uid in skill.skill_apps):
            del skill.skill_apps[skill_app.uid]
            if(len(skill.skill_apps) == 0):
                self.remove_skill(skill)

    def _ifit_skill_app(self, skill_app, reward, is_start=None, **kwargs):
        # If reward==None then remove skill_app. If it is the
        #  only SkillApp for 'skill' then completely remove 'skill'.
        if(reward is None):
            self._remove_skill_app(skill_app, is_start, **kwargs)
            return

        state = skill_app.state
        skill = skill_app.skill

        # Add skill_app to the skill's supporting skill_apps
        skill.skill_apps[skill_app.uid] = skill_app

        # Fit the skill-specific learning mechanisms where, when, which
        skill.where_lrn_mech.ifit(state, skill_app.match, reward)
        skill.when_lrn_mech.ifit(state, skill_app, reward)
        skill.which_lrn_mech.ifit(state, skill_app, reward)

        # Fit global process-learning mechanism
        prob_uid = None if not is_start else state.get("__uid__") 
        skill_app.add_seq_tracking(prob_uid)
        if(self.process_lrn_mech):
            self.process_lrn_mech.ifit(state, skill_app, is_start=is_start, reward=reward)

        skill_app.train_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def train(self, state, sai=None, reward:float=None, arg_foci=None, how_help=None, 
              skill_app=None, uid=None, skill_label=None, skill_uid=None,
              explanation_selected=None, ret_train_expl=False, add_skill_info=False,
              is_start=None, _ifit_implict=True, remove=False, **kwargs):
        # print("SAI", sai, type(sai))
        
        state = self.standardize_state(state, is_start)

        # Find a SkillApp which explains the provided SAI, and other annotations
        if(skill_app is None):
            sai = self.standardize_SAI(sai)        
            arg_foci = self.standardize_arg_foci(arg_foci, kwargs)
            skill_app = self._find_skill_app(state, sai=sai, arg_foci=arg_foci, 
                    how_help=how_help, uid=uid, skill_label=skill_label, skill_uid=skill_uid,
                    explanation_selected=explanation_selected, **kwargs)        

        
        
        # print("ANNOTATE ARG FOCI:", arg_foci)

        if(remove or reward is None):
            print("REMOVE", skill_app)
            # Remove skill_app from various learning mechanisms
            self._remove_skill_app(skill_app, is_start, **kwargs)
            # Remove any implicit second-order effects
            skill_app.clear_implicit_dependants()
            skill_app.removed = True
        else:
            print("TRAIN", skill_app, reward)
            # Annotate explicit calls to train() with a 
            #  time-stamp, explicit_reward, and other info.
            skill_app.annotate_train_data(reward, arg_foci, skill_label, skill_uid, 
                how_help, explanation_selected, **kwargs)

            # Pass the reward to various learning mechanisms
            self._ifit_skill_app(skill_app, reward, is_start, **kwargs)

            # Apply any implicit second-order effects
            if(_ifit_implict):
                skill_app.ifit_implicit_dependants()

        

        # Return the skill_app that was updated
        return skill_app

    def _add_unorderd_group_implicit_rewards(self, state, skill_apps, **kwargs):
        print("START UNORDER GRP")
        pos_skill_apps = []
        for sa in skill_apps:
            rew = getattr(sa,"reward", 0)
            if(rew is not None and rew > 0 and sa.sai.selection.id != "done"):
                pos_skill_apps.append(sa)
        pos_skill_apps = tuple(sorted(pos_skill_apps, key=lambda x: x.uid))

        start_state = state
        
        print("POS", pos_skill_apps)
        # already_exists = False
        # for skill_app in pos_skill_apps:
        #     skill_app.implicit_dependants


        # Nothing more to do if only 1.
        if(len(pos_skill_apps) <= 1):
            return 

        prob_uid = pos_skill_apps[0].prob_uid
        print("PROB UID", prob_uid)

        # Go through the skill apps with positive reward, 
        #  applying them one at a time forward and backward.

        # for order in itertools.permutations(pos_skill_apps):
        new_skill_apps = set()
        for order in [pos_skill_apps, pos_skill_apps[::-1]]:
            # print(order)
            state = start_state
            for i, sa in enumerate(order):
                if(i != 0):
                    wm = state.get("working_memory")

                    match = [wm.get_fact(id=m.id) for m in sa.match]
                    
                    # Restate the skill app using predicted next state
                    sa = SkillApplication(sa.skill, match, state, prob_uid=prob_uid)
                    if(sa not in new_skill_apps):
                        new_skill_apps.add(sa)

                state = self.predict_next_state(state, sa.sai) 

        # TODO: This clears things out but prevents
        #  mixing with other kinds of implicit reward
        for sa in pos_skill_apps:
            sa.clear_implicit_dependants()


        # kwargs = {**kwargs}
        # if('is_start' in kwargs):
        #     del kwargs['is_start']
        for sa in new_skill_apps:
            sa.add_implicit_reward(pos_skill_apps, 1)
            # print("\t", sa)
            # self.train(state, skill_app=sa, reward=1)
        # print(new_skill_apps)
        print("-----------")

    def train_all(self, training_set, states={}, **kwargs):
        skill_apps = []
        for example in training_set:
            state = example['state']

            example = {k:example[k] for k in example.keys() - {'state'}}
            # del example['state']

            # If 'state' is a uid find it's object from 'states'.
            if(isinstance(state,str)):
                state = states[state]

            skill_app = self.train(state, **example, **kwargs, _ifit_implict=False)
            skill_apps.append(skill_app)

        if("unordered_groups" in self.implicit_reward_kinds):
            apps_by_state = {}            
            for sa in skill_apps:
                uid = sa.state.get("__uid__")
                _, arr = apps_by_state.get(uid, (sa.state,[]))
                arr.append(sa)
                apps_by_state[uid] = (sa.state, arr)
            for uid, (state,apps) in apps_by_state.items():
                self._add_unorderd_group_implicit_rewards(state, skill_apps, **kwargs)

        for skill_app in skill_apps:
            skill_app.ifit_implicit_dependants()

        print("---------------")
        for skill in self.skills.values():
            print()
            print(skill)
            print(skill.when_lrn_mech)
        print("---------------")


        return skill_apps

# ----------------------------------------------
#  annotate_group()

    def annotate_group(self, skill_app_group, kind):
        group = []
        for sa in skill_app_group:
            if(isinstance(sa, str)):
                sa = self._recover_prev_skill_app(uid=sa)
            group.append(sa)

        ann = AppGroupAnnotation(skill_app_group, kind)

        if(kind == "no_others"):
            state_uid = group[0].state_uid
            for i in range(1, len(group)):
                assert group[i].state_uid == state_uid

        self.group_annotations.append(ann)

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
            skills = [self.skills[uid] for uid in skill_uids if uid in self.skills]
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
    def _rollout_expand_policy(self, s_uid, states, actions):
        state_obj = states[s_uid]
        # print("POLICY", s_uid[:5], state_obj['in_skill_app_uids'], state_obj['out_skill_app_uids'])

        # Edge Case: s_uid refers to the start state
        if(len(state_obj['in_skill_app_uids']) == 0):
            return True

        # Otherwise check if at least one explicit positive or 
        #  in_process skill_app.
        for in_uid in state_obj['in_skill_app_uids']:
            sa = actions[in_uid]['skill_app']

            if((sa.reward is not None and
                sa.reward > 0) or
                getattr(sa, 'in_process', False) == True):
                return True

        return False

    def _insert_rollout_skill_app(self, state, next_state, skill_app, states, actions, depth_counts, depth):
        nxt_uid = next_state.get('__uid__')
            
        # Make Action Object
        action_is_new = False
        if(skill_app is not None):
            uid = state.get('__uid__')    
            action_obj = {
                "uid" : skill_app.uid,
                "state_uid" : uid,
                "next_state_uid" : nxt_uid,
                "skill_app" : skill_app,
            }
            action_is_new = skill_app.uid not in actions
            actions[skill_app.uid] = action_obj

        # Ensure Depth Counts long enough
        while(depth >= len(depth_counts)):
            depth_counts.append(0)
        depth_index = depth_counts[depth]

        # Make State Object
        state_is_new = nxt_uid not in states
        if(state_is_new):
            n_state_obj = {"state": next_state, "uid" : nxt_uid,
                         "depth" : depth, "depth_index" : depth_index,
                         "in_skill_app_uids" : [],
                         "out_skill_app_uids" : []
                         }
            states[nxt_uid] = n_state_obj
            # uid_stack.append(nxt_uid)
            depth_counts[depth] += 1
            if(getattr(next_state, 'is_done', False)):
                # print("--------IS DONE!!!---------", nxt_uid)
                n_state_obj['is_done'] = True
        else:
            n_state_obj = states[nxt_uid]
            if(depth > n_state_obj['depth']):
                n_state_obj['depth'] = depth
                n_state_obj['depth_index'] = depth_index
                depth_counts[depth] += 1
        
        # Fill in/out connections
        # print("CONNECT? ", action_is_new, skill_app is not None, state is not None)
        if(action_is_new and
           skill_app is not None and
           state is not None):

            state_obj = states[state.get("__uid__")]
            state_obj['out_skill_app_uids'].append(skill_app.uid)
            n_state_obj['in_skill_app_uids'].append(skill_app.uid)

            
                
            

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

    def _annotate_unordered_groups(self, states, actions):
        unordered_groups = {}
        for s_uid, s_obj in states.items():
            state = s_obj['state']
            apps = [actions[uid]['skill_app'] for uid in s_obj['out_skill_app_uids']]
            if(len(apps) <= 1):
                continue

            apps_not_in_grp = set(apps)
            path_grps = group_by_path(apps)
            if(len(path_grps) > 0):
                for path, grp in path_grps.items():

                    non_neg_apps = [sa for sa in grp if sa.reward is None or sa.reward > 0]
                    if(len(non_neg_apps) <= 1 or non_neg_apps[0].path.is_internal_unordered):
                        continue


                    try:
                        # Try/except since sai seq can be invalid,
                        #  for instance if mixed w/ "press done" 
                        end_state = self.predict_next_state(state, [sa.sai for sa in non_neg_apps])
                    except:
                        continue
                    start_uid = state.get('__uid__')
                    end_uid = end_state.get('__uid__')
                    grp_str = f"{start_uid}-{end_uid}"

                    print("UNORDERED GROUP", start_uid[:5], end_uid[:5], [sa.sai for sa in grp])
                    for sa in grp:
                        sa.unordered_group = grp_str
                        # sa.group_next_state_uid = end_uid
                        if(sa.uid in actions):
                            actions[sa.uid]['group_next_state_uid'] = end_uid
                        # print(sa)
                        apps_not_in_grp.remove(sa)

                    unordered_groups[grp_str] = {
                        "skill_app_uids" : [sa.uid for sa in grp],
                        "start_state_uid" : start_uid,
                        "end_state_uid" : end_uid,
                    }


            # Even if not yet an unordered group give it a group_next_state_uid
            if(len(apps_not_in_grp) > 0):
                pos_apps = [sa for sa in apps_not_in_grp if sa.reward is not None and sa.reward > 0]
                print("POS APPS", state.get('__uid__')[:5])

                if(len(pos_apps) <= 1):
                    continue

                try:
                    # Try/except since sai seq can be invalid,
                    #  for instance if mixed w/ "press done" 
                    end_state = self.predict_next_state(state, [sa.sai for sa in pos_apps])
                except:
                    continue

                end_uid = end_state.get('__uid__')
                # print("POS APPS", state.get('__uid__')[:5], "->", end_uid[:5])
                for sa in pos_apps:
                    if(sa.uid in actions):
                        print(">>", sa.uid)
                        actions[sa.uid]['group_next_state_uid'] = end_uid


        return unordered_groups

    # def _annotate_group_next_state(self, states, actions):
    #     for s_uid, s_obj in states.items():
    #         state = s_obj['state']
    #         apps = [actions[uid]['skill_app'] for uid in s_obj['out_skill_app_uids']]
    #         if(len(apps) <= 1):
    #             continue 
    #         pos_apps = [sa for sa in apps if sa.reward > 0 or sa.in_process]



    def act_rollout(self, state, max_depth=-1, halt_policies=[], json_friendly=False,
                    base_depth=0, ret_avg_certainty=True, is_start=None, prob_uid=None,
                    add_out_of_process=False, ignore_filter=False, add_conflict_certainty=True,
                    annotate_unordered_groups=True, annotate_group_next_state=True,
                     **kwargs):
        # print("IS START", is_start)
        ''' 
        Applies act_all() repeatedly starting from 'state', and fanning out to create at 
        tree of all action rollouts up to some depth. At each step in this process the agent's 
        actions produce subsequent states based on the default state change defined by each 
        action's ActionType object. A list of 'halt_policies' specifies a set of functions that 
        when evaluated to false prevent further actions. Returns a tuple (states, action_infos).
        '''
        # with PrintElapse("ACT ROLLOUT"):
        # print("\n\t\tSTART ACT ROLLOUT\t\t", base_depth)
        curr_state = state = self.standardize_state(state, is_start)
        curr_state_uid = state.get('__uid__')
        if(is_start):
            prob_uid = curr_state_uid 
            curr_state.is_start = is_start

        # print(state.get('working_memory'))
        halt_policies = [self.standardize_halt_policy(policy) for policy in halt_policies]

        states = {}
        actions = {}
        
        depth_counts = []
        self._insert_rollout_skill_app(None, state, None,
                     states, actions, depth_counts, base_depth)

        uid_stack = [curr_state_uid]
        print("ADD:OUT OF PROCESS", add_out_of_process)
        while(len(uid_stack) > 0):
            # print("RECURSE", uid_stack)
            # for _ in range(len(uid_stack)):
            new_uids = set()
            for uid in uid_stack:

                # Prevent cycles 
                if(len(getattr(states[uid], 'out_skill_app_uids', [])) > 0):
                    continue

                depth = states[uid]['depth']+1
                state = states[uid]['state']
                
                src_wm = state.get("working_memory")
                # print("---source---", uid)
                # print(repr(src_wm))
                if(getattr(state,"is_done", False) == True):
                    continue 

                # print("ACT ALL", ignore_filter)
                skill_apps = self.act_all(state, 
                    return_kind='skill_app', prob_uid=prob_uid,
                    add_out_of_process=add_out_of_process,
                    add_conflict_certainty=add_conflict_certainty,
                    ignore_filter=ignore_filter,
                    **kwargs)

                for skill_app in skill_apps:
                    # skill_app.skill.skill_apps[skill_app.uid] = skill_app

                    if(is_start):
                        skill_app.prob_uid = curr_state_uid

                    at = skill_app.sai.action_type
                    # print("---skill_app :", skill_app)
                    # print("---action_type :", at)

                    if(getattr(skill_app, 'next_state', None) is not None):
                        next_state = skill_app.next_state
                    else:
                        next_state = self.predict_next_state(src_wm, skill_app.sai)
                        skill_app.next_state = next_state

                        
                    # next_wm = at.predict_state_change(src_wm, skill_app.sai)
                    # print("---dest---")
                    # print(repr(dest_wm))
                    # next_state = self.standardize_state(next_wm)
                    
                    self._insert_rollout_skill_app(state, next_state, skill_app,
                                    states, actions, depth_counts, depth)
                    new_uids.add(next_state.get('__uid__'))

            uid_stack = [uid for uid in new_uids 
                         if self._rollout_expand_policy(uid, states, actions)]
            # if(not ):
                # self._insert_rollout_skill_app(None, state, None,
                #      states, actions, uid_stack, depth_counts, depth)
                # continue
            # print(uid_stack)
        # self.annotate_verified(states, actions)



        # for state_obj in states.values():
        #     state_obj['out_skill_app_uids'] = []
        #     state_obj['in_skill_app_uids'] = []

        # for uid, action in actions.items():
        #     s_obj = states[action['state_uid']];
        #     s_obj['out_skill_app_uids'].append(uid)
        #     ns_obj = states[action['next_state_uid']];
        #     ns_obj['in_skill_app_uids'].append(uid)

        if(getattr(self, "process_lrn_mech", None) is not None):
            print()
            print(self.process_lrn_mech.grammar)
        # if(self.track_rollout_preseqs and is_start):
        #     start_sa_uids = states[curr_state_uid]['out_skill_app_uids']
        #     if(len(start_sa_uids) > 0):
        #         print(start_sa_uids)
        #         start_sa = actions[start_sa_uids[0]]['skill_app']
        #         prob_uid = self.rollout_preseq_tracker.resolve_prob_uid(start_sa, is_start=is_start)
        #         # self.rollout_preseq_tracker.update_subseqs(prob_uid)

        #         print(">----------------<")
        #         print(self.process_lrn_mech.grammar)
        #         print()
        #         last_state_obj = list(states.values())[-1]
        #         last_state = last_state_obj['state']
        #         last_state_uid = last_state_obj['uid']
        #         print(prob_uid[:5], last_state_uid[:5])
        #         print(self.rollout_preseq_tracker.get_preseqs(last_state, prob_uid))
        #         print(">----------------<")

        # if(add_conflict_certainty):
        #     self._add_conflict_certainty(states, actions)

        if(ret_avg_certainty):
            avg_certainty = 0.0
            for action in actions.values():
                when_pred = getattr(action['skill_app'],'when_pred', None)
                cert = abs(when_pred if when_pred is not None else 1.0)
                avg_certainty += cert
        avg_certainty = 0.0 if len(actions) == 0 else avg_certainty / len(actions)
        

        out = {"curr_state_uid" :  curr_state_uid, 
                "states" : states,
                "actions" : actions,
                }

        if(annotate_unordered_groups):
            unord_grps = self._annotate_unordered_groups(states, actions)
            out["unordered_groups"] = unord_grps
            
        if(json_friendly):
            for state_obj in states.values():
                state_obj['state'] = state_obj['state'].get("py_dict")
            # states = {uid : state for uid, state in states.items()}
            for action in actions.values():
                # print("ROLLOUT APP:")
                action['skill_app'] = action['skill_app'].get_info()

        from pprint import pprint

        

        

        if(ret_avg_certainty):
            out['avg_certainty'] = avg_certainty

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


        return out
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
                          print_diff=True, print_correct=False, return_diffs=False,
                         **kwargs):
        ''' Evaluates an agent's correctness and completeness against a completeness profile.'''
        import json, os
        print("START EVAL COMPLETENESS")

        n_correct, total = 0, 0
        n_first_correct, total_states = 0,0
        diffs = []
        prob_uid = None
        with open(profile, 'r') as profile_f:
            for line_ind, line in enumerate(profile_f):
                # Read a line from the profile
                item = json.loads(line)
                
                # Get the ground-truth sais
                profile_sais = [SAI(x) for x in item['sais']]
                state = item['state']
                is_start = len(item['hist'])==0
                if(is_start):
                    state = self.standardize_state(state)
                    prob_uid = state.get("__uid__")
                    # print()
                    # print("prob_uid", prob_uid)
                # print("IS START", len(item['hist'])==0)
                

                # print(item['state'])
                # print("\t",item['problem'], item['hist'], prob_uid[:5])
                agent_sais = self.act_all(state, return_kind='sai',
                 is_start=is_start, prob_uid=prob_uid, eval_mode=True)

                # Find the difference of the sets 
                # profile_sai_strs = set([str(s) for s in profile_sais])
                # agent_sai_strs = [str(s.get_info()) for s in agent_sais]
                set_agent_sais = set(agent_sais)
                set_profile_sais = set(profile_sais)
                missing = set_profile_sais - set_agent_sais
                extra = set_agent_sais - set_profile_sais
                correct = set_agent_sais.intersection(set_profile_sais)
                n_diff = len(missing) + len(extra)

                if(n_diff > 0):
                    print("\t",item['problem'], item['hist'], n_diff)
                    apps = self.act_all(state, return_kind='skill_app', is_start=is_start, prob_uid=prob_uid, eval_mode=True)
                    for app in apps:
                        when_pred = getattr(app, 'when_pred', None)
                        print(f"{app.when_pred:.2f}" if when_pred is not None else None, app)

                # diff = profile_sai_strs.symmetric_difference(set_agent_sai_strs)
                # if(return_diffs):
            # print()
                diffs.append({"problem": item['problem'], 'hist' : item['hist'], "-": list(missing),"+": list(extra), "=" : list(correct)})
                # if(len(item['hist']) == 0):
                uid = self.standardize_state(state).uid
                
                # print("OUTS", self.rollout_preseq_tracker.states[uid].get('outs',[]))
                # if(n_diff == 0):
                #     print("INS", self.rollout_preseq_tracker.states[uid].get('ins',[]))
                
                    # dp, da = [], []
                    # for d in diff:
                    #     if(d in profile_sais):
                    #         dp.append(json.loads(d))
                    #     elif(d in set_agent_sai_strs):
                    #         da.append(json.loads(d))
                    
                # if(print_diff and len(diff) > 0):
                    # Print Problem
                    # gv = lambda x : item['state'][x]['value']
                
                    # print(list(item['state'].keys()))
                    #print(line_ind+1, ">>",f"{gv('inpA3')}{gv('inpA2')}{gv('inpA1')} + {gv('inpB3')}{gv('inpB2')}{gv('inpB1')}")
                    # print(list(state.keys()))
                    # print("----------------------")
                    # for key, obj in state.items():
                    #     print(key, obj)
                    # print("AGENT:")
                    # for x in agent_sai_strs:
                    #     print(x)
                    # print("TRUTH:")
                    # for x in profile_sai_strs:
                    #     print(x)
                    # print("----------------------")
                    # print()

                # print("LINE", total_states, len(diff) == 0)
                if(partial_credit):
                    total += len(set_profile_sais)
                    n_correct += max(0, len(set_profile_sais)-n_diff)
                else:
                    total += 1
                    n_correct += n_diff == 0

                total_states += 1
                if(len(agent_sais) > 0 and agent_sais[0] in set_profile_sais):
                    n_first_correct += 1
        
        if(print_diff):
            for diff in diffs:
                n_diffs = len(diff['-']) + len(diff['+'])
                
                if(n_diffs != 0):
                    print(f"--DIFF: {diff['problem']} {diff['hist']} --")
                    for m in diff['-']:
                        print("  -", m['selection'], m['inputs']['value'])
                    for m in diff['+']:
                        print("  +", m['selection'], m['inputs']['value'])
                if(print_correct == True or 
                   print_correct=="when_diff" and n_diffs > 0):
                    for m in diff['=']:
                        print("  =", m['selection'], m['inputs']['value'])    


        completeness = n_correct / total
        correctness = n_first_correct / total_states

        print(f"Correctness : {correctness*100:.2f}%",print_diff)
        print(f"Completeness : {completeness*100:.2f}%")
        out = {"completeness" : completeness, "correctness" : correctness}
        # print("return_diffs", return_diffs)
        # print(diffs)
        if(return_diffs):
            out['diffs'] = diffs
        return out

    def remove_skill(self, skill_or_uid):
        uid = None
        if(isinstance(skill_or_uid, str)):
            if(skill_or_uid in self.skills):
                uid = self.skills[skill_or_uid]
        else:
            uid = skill_or_uid.uid

        if(uid in self.skills):
            print("REMOVE SKILL", uid)
            del self.skills[uid]
        else:
            print("NO SUCH SKILL", uid)




def group_by_path(skill_apps):
    prefix_groups = {}
    for sa in skill_apps:
        path = getattr(sa, 'path', None)
        if(not path):
            continue
        prefix = path_prefix(sa.path)
        pre_grp = prefix_groups.get(prefix, [])
        pre_grp.append(sa)
        prefix_groups[prefix] = pre_grp
    return prefix_groups


def path_prefix(path):
    '''A helper function which converts a parse path into a hashable object'''
    pp = []
    for i, (macro, meth_ind, item_ind, cov) in enumerate(path.steps):
        if(cov is not None):
            item_ind = 0
            cov = tuple(sorted([x for x in cov]))
        pp.append((macro._id, meth_ind, item_ind, cov))
    return tuple(pp)







                















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


