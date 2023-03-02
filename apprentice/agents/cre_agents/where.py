import warnings
from abc import ABCMeta
from abc import abstractmethod
from .extending import new_register_decorator
from cre.utils import PrintElapse

# ------------------------------------------------------------------------
# : BaseWhere

register_where = new_register_decorator("where", full_descr="where-learning mechanism")

# TODO: COMMENTS
class BaseWhere(metaclass=ABCMeta):
    def __init__(self, skill):
        self.skill = skill
        self.agent = skill.agent
        self.constraint_builder = skill.agent.constraints

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
    def _ensure_vars(self, match):
        if(not hasattr(self,'vars')):

            _vars = []
            for i, fact in enumerate(match):
                alias = f"Arg{i-1}" if i else "Sel"
                _vars.append(Var(fact._fact_type, alias))
            self.vars = _vars
            self._base_conds = self.constraint_builder(_vars)
            # print(type(self._base_conds))

        return self.vars


@register_where
class AntiUnify(BaseCREWhere):
    def __init__(self, skill):
        super().__init__(skill)
        self.conds = None


    def ifit(self, state, match, reward=1):

        # Only fit on positive reward
        if(reward <= 0): return

        wm = state.get("working_memory")

        # TODO: For efficiency should really guard with check_match here,
        #    but need to change check_match so that it will ensure the 
        #    existence of any guarded unprovided facts, like neighbors.
        _vars = self._ensure_vars(match)

        conds = Conditions.from_facts(match, _vars, 
            alpha_flags=[("visible", "few_valued"), ('unique_id',)],
            beta_weight=10.0
        )
        # print("vvvvvvvvvvvvvvvvv")
        # print(self.conds)
        # print(conds)
        # print("-------------------")

        if(self.conds is None):
            self.conds = self._base_conds & conds
        else:
            self.conds = self._base_conds & self.conds.antiunify(conds, fix_same_var=True)

        # print(self.conds)
        # print("^^^^^^^^^^^^^^^^^^")
        # s = str(self.conds)
        # if("Sel.above" not in s and "Sel.below" not in s):
        #     raise ValueError()

            
    def score_match(self, state, match):
        wm = state.get('working_memory')
        if(len(self.vars) != len(match)):
            print(self.skill.how_part)
            print(match)
            raise ValueError()
        if(not self._base_conds.check_match(match, wm)):
            return 0.0 
        return self.conds.score_match(match, wm)

    def as_conditions(self):
        return self.conds

    def get_matches(self, state):
        wm = state.get('working_memory')
        # print("WM BEF MATCH RECOUNT", wm._meminfo.refcount)
        
        if(self.conds is not None):
            # with PrintElapse("get_matches"):
            matches = self.conds.get_matches(wm)
            print("<<", self.vars)
            matches = [m[:len(self.vars)] for m in matches]
            # print("WM AFT MATCH RECOUNT", wm._meminfo.refcount)
            return matches
        else:
            return []

    def check_match(self, state, match):
        # TODO: implement in CRE
        if(self.conds is None):
            return False
        wm = state.get("working_memory")
        return self.conds.check_match(match, wm)
        

# --------------------------------------------
# : MostSpecific

@register_where
class MostSpecific(BaseCREWhere):
    def __init__(self, agent):
        super().__init__(agent)
        self.id_sets = set()

    def ifit(self, state, match, reward=1):
        # Only fit on positive reward
        if(reward <= 0): return

        self._ensure_vars(match)
        match_ids = tuple([x.id for x in match])
        if(match_ids not in self.id_sets):
            self.id_sets.add(match_ids)

        # print(self.check_match(state, match))
            
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
        for id_set in self.id_sets:
            if(all([x.id == _id for x,_id in zip(match,id_set)])):
                wm = state.get('working_memory')
                # print("Base Conds", self._base_conds)
                if(self._base_conds.check_match(match, wm)):
                    return True
        return False
