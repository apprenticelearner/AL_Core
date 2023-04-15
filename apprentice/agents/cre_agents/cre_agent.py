from numba.types import f8, string, boolean
import numpy as np
from cre import MemSet, CREFunc, UntypedCREFunc, Fact, FactProxy
from apprentice.agents.base import BaseAgent
from apprentice.agents.cre_agents.state import State, encode_neighbors
from apprentice.agents.cre_agents.dipl_base import BaseDIPLAgent
from apprentice.agents.cre_agents.logger import Logger
from cre.transform import MemSetBuilder, Flattener, FeatureApplier, RelativeEncoder, Vectorizer, Enumerizer

from cre.utils import PrintElapse
from cre import TF
from cre.gval import new_gval

from numba.core.runtime.nrt import rtsys
import gc
def used_bytes(garbage_collect=True):
    # if(garbage_collect): gc.collect()
    stats = rtsys.get_allocation_stats()
    # print(stats)
    return stats.alloc-stats.free

logger = Logger().logger
EMPTY_RESPONSE = {}

class SAI(object):
    slots = ('selection', 'action_type', 'inputs')
    def __init__(self, *args):
        if(len(args) == 1):
            inp = args[0]
            if(hasattr(inp, 'selection')):
                selection = inp.selection
                action_type = getattr(inp, 'action_type', inp.action)
                inputs = inp.inputs
            elif(isinstance(inp, (list,tuple))):
                selection, action_type, inputs = inp
            elif(isinstance(inp, dict)):
                selection = inp['selection']
                action_type = inp.get('action_type', inp['action'])
                inputs = inp['inputs']
            else:
                raise ValueError(f"Unable to translate {inp} to SAI.")
        else:
            selection, action_type, inputs = args

        self.selection = selection
        self.action_type = action_type
        self.inputs = inputs


    def __repr__(self):
        sel_str = self.selection.id if(isinstance(self.selection,FactProxy)) else f'{self.selection!r}'
        return f"SAI({sel_str},{self.action_type},{self.inputs})"

    def __iter__(self):
        return (self.selection, self.action_type, self.inputs)

    def __eq__(self, other):
        if(not isinstance(other,SAI)): return False
        return (self.selection.id, self.action_type, self.inputs) == \
               (other.selection.id, other.action_type, other.inputs)

    def __getitem__(self, i):
        return iter(self)[i]

    def get_info(self):
        sel_str = self.selection.id if(isinstance(self.selection,FactProxy)) else self.selection
        return {
            'selection' :  sel_str,
            'action_type' :  self.action_type,
            'inputs' :  self.inputs,
        }

    def __str__(self):
        # print(self.selection, isinstance(self.selection,FactProxy))
        sel_str = self.selection.id if(isinstance(self.selection, FactProxy)) else self.selection
        return f"SAI({sel_str}, {self.action_type!r}, {self.inputs!r})"


class Skill(object):
    def __init__(self, agent, id_num, action_type, how_part, input_attr,
                 label=None, explanation_set=None):
        self.agent = agent
        self.label = label
        self.explanation_set = explanation_set
        self.how_part = how_part
        self.action_type = action_type
        self.input_attr = input_attr
        self.id_num = id_num 

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

            # print("1" if when_predict else "0",  match[0].id,"\t" , [m.id for m in match][1:], "  ", self.how_part)
            if(when_predict > 0):
                skill_app = SkillApplication(self, match)
                if(skill_app is not None):
                    applications.append(skill_app)
        return applications

    def get_info(self):
        info = {  "how": self.how_part,
                  "where": self.where_lrn_mech.get_info(),
                  "when": self.when_lrn_mech.get_info(),
                  "which": self.which_lrn_mech.get_info(),
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
        return f"Skill({self.how_part}, id: {self.id_num})"


class SkillApplication(object):
    # __slots__ = ("skill", "match", "sai")
    def __new__(cls, skill, match):
        sai = skill(*match)
        if(sai is None):
            return None
        self = super().__new__(cls)

        self.skill = skill
        self.match = match
        self.sai = sai
        return self

    def get_info(self):
        sai = self.sai
        info = {
            'skill_label' : self.skill.label,
            'skill_id' :  getattr(self,'id_num', None),
            'selection' :  sai.selection.id,
            'action' :  sai.action_type,
            'action_type' :  sai.action_type,
            'inputs' :  sai.inputs,
            'mapping' :  {f"arg{i-1}" if i else "sel" : x.id for i,x in enumerate(self.match)}
        }
        return info

    def __repr__(self):
        return f'{self.skill}({", ".join([m.id for m in self.match])}) -> {self.sai}'

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

        state = self.state = State(self)

        @state.register_transform(is_incremental=True, prereqs=['working_memory'])
        def flat(state):
            wm = state.get('working_memory')
            flattener = state.agent.flattener
            x = flattener(wm)
            return x

        @state.register_transform(is_incremental=len(self.extra_features)==0, prereqs=['flat'])
        def flat_featurized(state):
            flat = state.get('flat')
            feature_applier = state.agent.feature_applier
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
        self.skills = []
        self.skills_by_label = []
        self.prev_skill_app = None


    def standardize_state(self, state):
        #NOTE: Should just change to locked at interface
        for k,obj in state.items():
            if('contentEditable' in obj):
                obj['locked'] = not obj['contentEditable']
                del obj['contentEditable']
        
        # Clean up old wm
        # if("working_memory" in self.state):
        #     print("HI")
        #     wm = self.state.get("working_memory")
        #     wm.free()
        # print(used_bytes())

        if(isinstance(state, dict)):
            if self.should_find_neighbors:
                state = encode_neighbors(state)
            wm = self.memset_builder(state, MemSet())
        elif(isinstance(state, MemSet)):
            wm = state
        else:
            raise ValueError(f"Unrecognized State Type: \n{state}")

        # print(wm)
        # if('working_memory' in self.state):
        #     self.state.get('working_memory').free()
        # print("WM INITIAL RECOUNT", wm._meminfo.refcount)
        self.state.set('working_memory', wm)
        prev_skill_app = getattr(self,'prev_skill_app',None)
        if(prev_skill_app):
            prev_sel = prev_skill_app.match[0]
            self.prev_skill_app.match =[wm.get_fact(id=m.id) for m in prev_skill_app.match]
            # wm = None
            # used_bytes()
            # print("REF COUNT", prev_sel._meminfo.refcount)
        return self.state

    def standardize_SAI(self, sai):
        # NOTE: does an SAI hold the selection object or the name?
        sai = SAI(sai)
        if(isinstance(sai.selection, str)):
            sai.selection = self.state.get('working_memory').get_fact(id=sai.selection)
        return sai

    def standardize_arg_foci(self, arg_foci):
        if(arg_foci is None): return None
        new_arg_foci = []
        wm = self.state.get('working_memory')
        for fact in arg_foci:
            if(isinstance(fact, str)):
                fact = wm.get_fact(id=fact)
            new_arg_foci.append(fact)
        return new_arg_foci

# ------------------------------------------------
# : Act
    def get_skill_applications(self, state):
        skill_applications = []
        for skill in self.skills:
            for skill_app in skill.get_applications(state):
                skill_applications.append(skill_app)

        skill_applications = self.which_cls.sort(state, skill_applications)
        return skill_applications

    def act(self, state, add_skill_info=False, n=1, **kwargs):  # -> Returns sai
        self.prev_skill_app = None
        state = self.standardize_state(state)
        # with PrintElapse("self.get_skill_applications"):
        skill_applications = self.get_skill_applications(state)

        if(len(skill_applications) > 0):
            skill_app = self.action_chooser(state, skill_applications)

            # print("--ACT: ", skill_app)
            # print()
            # print("--ACT: ")
            # print(skill_app)
            # print(skill_app.skill.when_lrn_mech.predict(state,skill_app.match))

            # for skill_app in skill_applications:
            #     print(skill_app)

            self.prev_skill_app = skill_app
            response = skill_app.get_info()
            if(n != 1):
                response['responses'] = [x.get_info() for x in skill_applications]
        else:
            # print("--NO ACTION")
            self.prev_skill_app = None
            response = EMPTY_RESPONSE

        prev_skill_app = getattr(self,'prev_skill_app',None)
        # if(prev_skill_app): print("BEF", getattr(self,'prev_skill_app',None).match[0]._meminfo.refcount)
        self.state.clear()
        # if(prev_skill_app): print("AFT", getattr(self,'prev_skill_app',None).match[0]._meminfo.refcount)
        return response

# ------------------------------------------------
# : Train
    def _skill_subset(self, sai, arg_foci=None, skill_label=None, skill_id=None):
        # Skill_id or skill_label can cut down possible skill candidates 
        subset = self.skills 
        if(skill_id is not None):
            subset = skills[skill_id]

        # TODO: choose "NO_LABEL" or None to be standard
        elif(skill_label is not None and skill_label != "NO_LABEL"):
            subset = self.skills_by_label.get(skill_label, self.skills)

        # TODO: else

        subset = [x for x in subset if x.input_attr == list(sai.inputs.keys())[0]]
        if(arg_foci is not None):
            pass
            # TODO: can probably reduce by matching n_args
            
        return subset

    def choose_best_explanation(self, state, skill_apps):
        def get_score(skill_app):
            score = skill_app.skill.where_lrn_mech.score_match(state, skill_app.match)
            # print("SCORE", score, skill_app)
            return score

        scored_apps = [x for x in [(get_score(sa), sa) for sa in skill_apps]]# if x[0] > 0.0]
        if(len(scored_apps) > 0):
            return sorted(scored_apps, key=lambda x: x[0])[-1][1]
        else:
            return None



    def explain_from_skills(self, state, sai, 
        arg_foci=None, skill_label=None, skill_id=None):

        skills_to_try = self._skill_subset(sai, arg_foci, skill_label, skill_id)
        skill_apps = []
        
        # Try to find an explanation from the existing skills that matches
        #  the how + where parts. 
        for skill in skills_to_try:
            for candidate in skill.get_applications(state, skip_when=True):
                # print(candidate, candidate.sai, sai, candidate.sai == sai)
                if(candidate.sai == sai):
                    # If foci are given make sure candidate has the 
                    #  same arguments in it's match.
                    if(arg_foci is not None and 
                        candidate.match[:1] != arg_foci):
                        continue

                    skill_apps.append(candidate)

        # if(len(skill_apps) > 0): print("EXPL HOW + WHERE")

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

        best_expl = self.choose_best_explanation(state, skill_apps)
        # if(best_expl is not None):
        #     print(best_expl.skill.where_lrn_mech.conds)
        # print("BEST EXPLANATION", best_expl)
        return best_expl


    def induce_skill(self, state, sai, arg_foci=None, label=None):
        # print("INDUCE SKILL")
        # Does not currently support multiple inputs per SAI.
        input_attr, inp = list(sai.inputs.items())[0]
        
        # TODO: Make this not CTAT specific
        if(sai.selection.id != "done"):
            # Use how-learning mechanism produce a set of candidate how-parts
            explanation_set = self.how_lrn_mech.get_explanations(
                    state, inp, arg_foci)

            # If any candidates then choose one, otherwise treat how-part
            #  as a constant.
            if(len(explanation_set) > 0):
                how_part, args = explanation_set.choose()
                # print("CHOICE", how_part, args)
                if(how_part is None): how_part = inp
            else:
                how_part, args = inp, []
        else:
            how_part = -1
            explanation_set = None
            args = []

        # Make new skill.
        skill = Skill(self, len(self.skills), 
            sai.action_type, how_part, input_attr, 
            label=label, explanation_set=explanation_set)

        # print("INDUCE SKILL", skill)

        # Add new skill to various collections.
        self.skills.append(skill)
        if(label is not None):
            label_lst = self.skills_by_label.get(label,[])
            label_lst.append(skill)
            self.skills_by_label[label] = label_lst

        return SkillApplication(skill, [sai.selection,*args])


    def train(self, state, sai=None, reward:float=None,
              arg_foci=None, skill_label=None, skill_id=None, mapping=None,
              ret_train_expl=False, add_skill_info=False,**kwargs):
        if(skill_label == "NO_LABEL"): skill_label = None

        if('foci_of_attention' in kwargs and arg_foci is None):
            arg_foci = kwargs['foci_of_attention'] 
        # print("arg_foci:", arg_foci)

        # with PrintElapse("standardize"):
        state = self.standardize_state(state)
        sai = self.standardize_SAI(sai)
        arg_foci = self.standardize_arg_foci(arg_foci)
        skill_apps = None

        # print("--TRAIN:", sai.selection.id, sai.inputs['value'])

        # with PrintElapse("explain_from_skills"):
        # Feedback Case : just train according to the last skill application.
        # if(self.prev_skill_app != None):
        #     print(self.prev_skill_app.sai, sai, self.prev_skill_app.sai == sai)
        if(self.prev_skill_app != None and self.prev_skill_app.sai == sai):
            print("PREV SKILL APP", self.prev_skill_app)
            skill_app = self.prev_skill_app
        # Demonstration Case : try to explain the sai from existing skills.
        else:
            # with PrintElapse("explain_from_skills"):
            skill_app = self.explain_from_skills(state, sai, arg_foci, skill_label)

        # with PrintElapse("induce_skill"):
        # If existing skills fail then induce a new one with how-learning.
        if(skill_app is None):
            print("INDUCE SKILL")
            skill_app = self.induce_skill(state, sai, arg_foci, skill_label)

        # print("WM")
        # print(state.get("working_memory"))
        # with PrintElapse("ifit"):
            # skill_app = skill_apps[0]
            # print("Update", skill_app)
            # for skill_app in skill_apps:
        # with PrintElapse("self.ifit"):
        skill_app.skill.ifit(state, skill_app.match, reward)


        self.state.clear()
        # print()




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


