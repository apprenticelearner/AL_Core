import warnings
from abc import ABCMeta
from abc import abstractmethod
from .registers import register_where
from cre.utils import PrintElapse
import numpy as np

# ------------------------------------------------------------------------
# : BaseWhere


# TODO: COMMENTS
class BaseWhere(metaclass=ABCMeta):
    def __init__(self, skill):
        self.skill = skill
        self.agent = getattr(skill, 'agent', None)
        self.constraint_builder = self.agent.constraints if self.agent else None

    @abstractmethod
    def ifit(self, state, match, reward=1):
        """
        
        :param state: 
        """
        pass

    def fit(self, states, matches, rewards=1):
        if(not isinstance(rewards,(list,tuple))): 
            rewards = [rewards]*len(matches)
        for match, reward in zip(matches,rewards):
            self.ifit(match, reward)

    def score_match(self, match):
        """
        
        """
        return float(self.check_match(match))

    def as_conditions(self):
        """
        
        """
        raise NotImplemented()

    @abstractmethod
    def get_matches(self, state):
        """
        
        """
        pass

    @abstractmethod
    def check_match(self, state, match):
        """
        
        """
        pass

# ------------------------------------------------------------------------
# : Where Learning Mechanisms


# --------------------------------------------
# : AntiUnify

from cre import Conditions, Var

class BaseCREWhere(BaseWhere):
    def __init__(self, skill, **kwargs):
        super().__init__(skill, **kwargs)
        self.conds = None
        self.fit_record = []
        self.fit_count = 0
        self.sanity_check = kwargs.get('sanity_check', False)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Inject sanity checks after ifit
        ifit = cls.ifit
        def ifit_w_sanity_check(self, state, match, reward=1, *args, **kwargs):
            ifit(self, state, match, reward, *args, **kwargs)
            self.sanity_check_ifit(state, match)
            self.fit_count += 1
        setattr(cls, 'ifit', ifit_w_sanity_check)

    def sanity_check_ifit(self, state, match):
        if(self.sanity_check):
            if(not self.check_match(state, match)):
                raise Exception("(Sanity Check Error): Match used"
                " for generlization fails ")

    def _ensure_base_conds(self):
        if(self.constraint_builder):
            self._base_conds = self.constraint_builder(self.vars)
        else:
            from cre.conditions import AND
            self._base_conds = AND(*self.vars)

    def _ensure_vars(self, match, var_names=None):
        if(not hasattr(self,'vars')):
            if(var_names is None):
                var_names = []
                for i in range(len(match)):
                    var_names.append(f"Arg{i-1}" if i else "Sel")
            self.vars = [Var(fact._fact_type, alias) for fact, alias in zip(match, var_names)]
            self._ensure_base_conds()

        return self.vars


    def score_match(self, state, match):
        wm = state.get('working_memory')
        if(len(self.vars) != len(match)):
            # print(self.skill.how_part)
            # print(match)
            raise ValueError()
        if(not self._base_conds.check_match(match, wm)):
            return 0.0 
        return self.conds.score_match(match, wm)

    def as_conditions(self):
        return self.conds

    def get_matches(self, state):
        wm = state.get('working_memory')        
        matches = []

        if(self.conds is not None):
            matches = self.conds.get_matches(wm, 
                match_len=len(self.vars))
        matches = list(matches)
        return matches

    def get_partial_matches(self, state, 
            match=None, tolerance=0.0):
        wm = state.get('working_memory')        
        matches = []
        if(self.conds is not None):
            matches = self.conds.get_partial_matches(
                wm, match, tolerance=tolerance,
                return_scores=True, match_len=len(self.vars))

            # matches = [(s, m[:len(self.vars)]) for s,m in matches]
        return matches

    def check_match(self, state, match):
        # TODO: implement in CRE
        if(self.conds is None):
            return False
        wm = state.get("working_memory")
        return self.conds.check_match(match, wm)

    def get_info(self,**kwargs):
        return {
            "conds" : str(self.conds),
            "fit_record" : self.fit_record
        }

    def antiunify(self, other, return_remap=False, fixed_inds=None,
             drop_no_beta=True):
        new_mech = type(self)(self.skill)

        remap_info = None
        _vars = getattr(self,'vars', other.vars)
        if(self.conds is None):
            new_conds = other.conds
        elif(other.conds is None):
            new_conds = self.conds
        elif(return_remap):
            from cre.conditions import AND
            # print(self.conds)
            # print("+++++")
            # print(other.conds)
            remap_info = self.conds.structure_map(other.conds,
                fixed_inds=fixed_inds,
                drop_unconstr=True, drop_no_beta=drop_no_beta)

            _, _, remap, keep_mask_a, _ = remap_info
            new_conds = self.conds.masked_copy(keep_mask_a, remap)

            # If drop_no_beta then remove variables 
            #  which do not have any 
            # if(drop_no_beta):


            # print("REMAP", remap)
            # print(self.conds)
            # print("+++++")
            # print(other.conds)
            # print("----------")
            # print(new_conds)
            # print(remap)
            mask = (remap != -1) & (remap < len(other.vars))
            # print("MASK", mask)
            _vars = [v for i,v in enumerate(_vars) if mask[i]]

            # Ensure that the new conds are reordered to begin with
            #  the main variables of the pattern
            if(len(_vars) > 0  and new_conds is not None):
                new_conds = AND(*_vars) & new_conds
            # print("------------")
            # print(remap, len(self.vars), len(other.vars), len(_vars))
            # print("------")
            # print("THIS ONE", remap)
        else:
            # print(self.conds)
            # print(other.conds)
            # print("------")
            # with PrintElapse("Antiunify"):
            new_conds = self.conds.antiunify(other.conds, drop_unconstr=True)

            # print(new_conds)

        new_mech.conds = new_conds
        if(new_conds is not None):
            new_mech.vars = _vars
            new_mech._ensure_base_conds()

        new_mech.fit_record = self.fit_record + other.fit_record
        new_mech.fit_count = self.fit_count + other.fit_count
        new_mech.vars = _vars

        if(return_remap):
            if(remap_info is None):
                remap = np.arange(len(new_mech.vars))
                alignment = np.zeros((1,1),dtype=np.int64)
                remap_info = (1.0, alignment, remap, None, None)
            return new_mech, remap_info
        return new_mech

    def remove(self, state, match):
        match_ids = tuple([getattr(m, 'id', None) for m in match])
        id_sets = self.id_sets
        if(match_ids in self.id_sets):
            del id_sets[match_ids]

            # Refit everything
            super().__init__(self.skill) 
            for args in id_sets.values():
                self.ifit(*args)




@register_where
class AntiUnify(BaseCREWhere):
    def __init__(self, skill):
        super().__init__(skill)

    # def will_learn(self, state, match, reward=1):
    #     ''' Returns true if the state match pair will  
    #         cause the current pattern to generalize'''
    #     wm = state.get("working_memory")
    #     return (self.conds and self.conds.check_match(match, wm))

    def ifit(self, state, match, reward=1, var_names=None):
        # print("IFIT", match)
        # Only fit on positive reward
        if(reward <= 0): return
        
        wm = state.get("working_memory")

        # print(self.conds)
        # Skip generalizing if check_match already succeeds.
        if(self.conds and self.conds.check_match(match, wm)):
            return

        self.fit_record.append({
            "state_id" : state.get("__uid__"), 
            "match" : [m.id for m in match]
        })

        _vars = self._ensure_vars(match, var_names=var_names)
        if(len(_vars) == 0):
            return 

        match_ids = tuple([getattr(m, 'id', None) for m in match])
        self.id_sets[match_ids] = (state, match, reward, var_names)

        # print(match, _vars)
        conds = Conditions.from_facts(match, _vars, 
            alpha_flags=[("visible","~semantic"), ('unique_id',)],
            beta_weight=10.0
        )
        # print("vvvvvvvvvvvvvvvvv")
        # print(repr(wm))
        # print(self.conds)
        # print("++++")
        # print(conds)

        if(self.conds is None):
            self.conds = self._base_conds & conds
        else:
            with PrintElapse("Antiunify"):
                # print("ENTER ANTI")
                # print(self.conds)
                # print("++++++++++")
                # print(conds)
                # print('---')
                self.conds = self._base_conds & self.conds.antiunify(conds, fix_same_var=True, drop_unconstr=True)
                # print(self.conds)

        # if(repr(self.skill.how_part) == "NumericalToStr(TensDigit(Add3(CastFloat(a.value), CastFloat(b.value), CastFloat(c.value))))"):
        #     print(repr(self.skill.how_part))
        #     print('------------------v-------------------')
        #     print(self.conds)
        #     print('------------------^-------------------')
        # print("-------------------")
        # print(self.conds)
        
        # if(not self.check_match(state, match)):
        #     raise ValueError("BAD BAD BAD")
        # print(self.conds)
        # print("^^^^^^^^^^^^^^^^^^")
        # s = str(self.conds)
        # if("Sel.above" not in s and "Sel.below" not in s):
        #     raise ValueError()

        # print("---------------")
        # if(len(match) > 1):
        #     print(match[1])
        #     print(match[1].left)
        # print([_m.id for _m in match])

        # SANITY CHECK -- Don't remove
        # is_there = False
        # for m in self.conds.get_matches(wm):
        #     print(":", [_m.id for _m in m[:len(match)]])
        #     if([_m.id for _m in m[:len(match)]] == [_m.id for _m in match]):
        #         is_there = True
        # print("IS_THERE", is_there)
        # print(self.conds.score_match(match, wm), self.conds.check_match(match, wm))
        # if(not is_there):
        #     raise ValueError()

            
    # def score_match(self, state, match):
    #     wm = state.get('working_memory')
    #     if(len(self.vars) != len(match)):
    #         # print(self.skill.how_part)
    #         # print(match)
    #         raise ValueError()
    #     if(not self._base_conds.check_match(match, wm)):
    #         return 0.0 
    #     return self.conds.score_match(match, wm)

    # def as_conditions(self):
    #     return self.conds

    # def get_matches(self, state):
    #     wm = state.get('working_memory')
    #     # print("WM BEF MATCH RECOUNT", wm._meminfo.refcount)
        
        
    #     matches = []
    #     if(self.conds is not None):
    #         # with PrintElapse("get_matches"):
    #         #     print("ENTER Get Matches")
    #         matches = self.conds.get_matches(wm)
    #         # print("<<", self.vars)
    #         matches = [m[:len(self.vars)] for m in matches]
    #         # print("WM AFT MATCH RECOUNT", wm._meminfo.refcount)
    #     # if(repr(self.skill.how_part) == "NumericalToStr(TensDigit(Add3(CastFloat(a.value), CastFloat(b.value), CastFloat(c.value))))" and
    #     #    "S_Qr9" in state.get('__uid__')):
    #     #     # print('--------------------------------------')
    #         # print(self.conds)
    #         # print(matches)
    #         # print('--------------------------------------')
    #     return matches

    # def check_match(self, state, match):
    #     # TODO: implement in CRE
    #     if(self.conds is None):
    #         return False
    #     wm = state.get("working_memory")
    #     return self.conds.check_match(match, wm)

    
        

@register_where
class Generalize(BaseCREWhere):
    def __init__(self, skill):
        super().__init__(skill)
        self.id_sets = {}

    # def will_learn(self, state, match, reward=1):
    #     ''' Returns true if the state match pair will  
    #         cause the current pattern to generalize'''
    #     wm = state.get("working_memory")
    #     return (self.conds and self.conds.check_match(match, wm))

    def ifit(self, state, match, reward=1, var_names=None):
        # Only fit on positive reward
        if(reward is None or reward <= 0):
            self.remove(match)
            return

            
        wm = state.get("working_memory")

        # Skip generalizing if check_match already succeeds.
        if(self.conds and self.conds.check_match(match, wm)):
            return

        self.fit_record.append({
            "state_id" : state.get("__uid__"), 
            "match" : [getattr(m, 'id', None) for m in match]
        })

        _vars = self._ensure_vars(match, var_names=var_names)
        if(len(_vars) == 0):
            return 

        match_ids = tuple([getattr(m, 'id', None) for m in match])
        self.id_sets[match_ids] = (state, match, reward, var_names)

        if(self.conds is None):
            conds = Conditions.from_facts(match, _vars, 
                alpha_flags=[("visible","~semantic"), ('unique_id',)],
                beta_weight=10.0
            )
            self.conds = self._base_conds & conds
        else:
            from cre.matching import (WN_VAR_TYPE, WN_BAD_DEREF,
                 WN_INFER_UNPROVIDED, WN_FAIL_MATCH)
            why_nots = self.conds.why_not_match(match)
            removals = []
            for obj, wn in why_nots:
                kind = wn['kind']
                v_ind0 = wn['var_ind0'] 
                v_ind1 = wn['var_ind1']

                # Allow removals of neighbor/parent
                #  variables which don't occur in
                #  the usual matching pattern
                has_adj_var = (v_ind0 >= len(match) or
                               v_ind1 >= len(match))

                if(kind == WN_VAR_TYPE):
                    raise NotImplementedError("VAR TYPE")
                    # Should try to upcast type here
                elif(kind == WN_BAD_DEREF):
                    raise NotImplementedError("BAD DEREF")

                elif(kind == WN_INFER_UNPROVIDED):
                    # print("REMOVE VAR", obj, wn)
                    if(has_adj_var or
                       match[v_ind0] is not None
                     ):
                        removals.append(wn['var_ind0'])
                elif(kind == WN_FAIL_MATCH):
                    # print("REMOVE LIT", obj, wn)
                    if( has_adj_var or
                        match[v_ind0] is not None and
                        (v_ind1 == -1 or 
                        match[v_ind1] is not None)
                       ):
                        removals.append((wn['d_ind'], wn['c_ind']))
            # print(self.conds)
            # print(removals)     
            conds = self.conds.remove(removals)
            # print(conds)
            # raise ValueError()
            
            self.conds = self._base_conds & conds

        # if(repr(self.skill.how_part) == "NumericalToStr(TensDigit(Add3(CastFloat(a.value), CastFloat(b.value), CastFloat(c.value))))"):
        #     print(repr(self.skill.how_part))
        #     print('------------------v-------------------')
        #     print(self.conds)
        #     print('------------------^-------------------')

        # print(self.conds)
        # print("-------------------")
        # if(not self.check_match(state, match)):
        #     raise ValueError("BAD BAD BAD")
        # print(self.conds)
        # print("^^^^^^^^^^^^^^^^^^")
        # s = str(self.conds)
        # if("Sel.above" not in s and "Sel.below" not in s):
        #     raise ValueError()

        # print("---------------")
        # if(len(match) > 1):
        #     print(match[1])
        #     print(match[1].left)
        # print([_m.id for _m in match])

        # SANITY CHECK -- Don't remove
        # is_there = False
        # for m in self.conds.get_matches(wm):
        #     print(":", [_m.id for _m in m[:len(match)]])
        #     if([_m.id for _m in m[:len(match)]] == [_m.id for _m in match]):
        #         is_there = True
        # print("IS_THERE", is_there)
        # print(self.conds.score_match(match, wm), self.conds.check_match(match, wm))
        # if(not is_there):
        #     raise ValueError()

            
    

    # def antiunify(self, other):
    #     new_mech = AntiUnify(self.skill)
    #     if(self.conds is None):
    #         new_conds = other.conds
    #     elif(other.conds is None):
    #         new_conds = self.conds
    #     else:
    #         print(self.conds)
    #         print(other.conds)
    #         print("------")
    #         with PrintElapse("Antiunify"):
    #             new_conds = self.conds.antiunify(other.conds, drop_unconstr=True)
    #         print(new_conds)

    #     new_mech.conds = new_conds
    #     if(new_conds is not None):
    #         new_mech.vars = new_conds.vars
    #         new_mech._ensure_base_conds()

    #     new_mech.fit_record = self.fit_record + other.fit_record
    #     new_mech.fit_count = self.fit_count + other.fit_count
    #     new_mech.vars = getattr(self,'vars', other.vars)
    #     return new_mech

# --------------------------------------------
# : MostSpecific

@register_where
class MostSpecific(BaseCREWhere):
    def __init__(self, agent, **kwargs):
        super().__init__(agent, **kwargs)
        self.id_sets = {}

    def will_learn(self, state, match, reward=1):
        if(reward <= 0): return False
        match_ids = tuple([x.id for x in match])
        return match_ids not in self.id_sets

    def ifit(self, state, match, reward=1):
        # Only fit on positive reward
        match_ids = tuple([x.id for x in match])
        if(reward is None or reward <= 0):
            if(match_ids in self.id_sets):
                del self.id_sets[match_ids]    

        if(match_ids not in self.id_sets):
            self._ensure_vars(match)
            print(">>", self.skill, match_ids)
            self.id_sets[match_ids] = (state, match, reward)
            
    def score_match(self, state, match):
        return float(self.check_match(state, match))

    def as_conditions(self):
        # TODO:
        raise NotImplemented()

    def get_matches(self, state):
        wm = state.get('working_memory')
        matches = []
        
        # print("MATCHES:",self.skill)
        for id_set in self.id_sets:
            try:
                match = [wm.get_fact(id=_id) for _id in id_set]
                match_base = self._base_conds.check_match(match, wm)
                if(match_base):
                    matches.append(match)
            except KeyError:
                continue
        # print()

        return matches

    def check_match(self, state, match):
        match_ids = [x.id for x in match]
        okay = False
        for id_set in self.id_sets:
            # print([(x.id, _id) for x,_id in zip(match,id_set)], all([x.id == _id for x,_id in zip(match,id_set)]))
            if(all([id0 == id1 for id0, id1 in zip(match_ids,id_set)])):
                okay = True

        wm = state.get('working_memory')
        if(okay and self._base_conds.check_match(match, wm)):
            return True
        return False
