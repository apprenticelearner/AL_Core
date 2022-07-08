import warnings
from abc import ABCMeta
from abc import abstractmethod
from .extending import new_register_decorator

# ------------------------------------------------------------------------
# : BaseWhere

register_where = new_register_decorator("where", full_descr="where-learning mechanism")

# TODO: COMMENTS
class BaseWhere(metaclass=ABCMeta):
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
        return self.vars


@register_where
class AntiUnify(BaseCREWhere):
    def __init__(self, agent):
        self.agent = agent
        self.conds = None

    def ifit(self, state, match, reward=1):
        if(self.conds is None or 
           not self.check_match(state, match)):

            _vars = self._ensure_vars(match)
            conds = Conditions.from_facts(match, _vars)
            if(self.conds is None):
                self.conds = conds
            else:
                self.conds = self.conds.antiunify(conds)
            
    def score_match(self, match):
        # TODO: implement in CRE
        return float(self.check_match(match))

    def as_conditions(self):
        return self.conds

    def get_matches(self, state):
        wm = state.get('working_memory')
        if(self.conds is not None):
            matches = self.conds.get_matches(wm)
            return [m[:len(self.vars)] for m in matches]
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
        self.agent = agent
        self.id_sets = []

    def ifit(self, state, match, reward=1):
        if(not self.check_match(state, match)):
            self.id_sets.append([x.id for x in match])
            
    def score_match(self, state, match):
        return float(self.check_match(match))

    def as_conditions(self):
        # TODO:
        raise NotImplemented()

    def get_matches(self, state):
        wm = state.get('working_memory')
        matches = []
        for id_set in self.id_sets:
            try:
                matches.append([wm.get_fact(id=_id) for _id in id_set])
            except KeyError:
                continue
        return matches

    def check_match(self, state, match):
        for id_set in self.id_sets:
            if(all([x.id == _id for x,_id in zip(match,id_set)])):
                return True
        return False
