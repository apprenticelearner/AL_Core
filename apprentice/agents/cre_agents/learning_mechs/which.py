import warnings
from abc import ABCMeta
from abc import abstractmethod
from .registers import register_which

# ------------------------------------------------------------------------
# : BaseWhich

# TODO: COMMENTS
class BaseWhich(metaclass=ABCMeta):
    def __init__(self, skill,**kwargs):
        self.skill = skill
        self.agent = skill.agent

    @staticmethod
    def sort(state, skill_applications):
        # Sort in descending order of utility (i.e. best first)
        def key_func(skill_app):
            which_lrn_mech = skill_app.skill.which_lrn_mech
            return which_lrn_mech.get_utility(state, skill_app)

        return sorted(skill_applications, key=key_func, reverse=True)

    def ifit(self, state, skill_app, reward):
        """
        
        :param state: 
        """
        raise NotImplemented()

    

    def get_utility(self, state, skill_app):
        """
        
        """
        raise NotImplemented()

    def get_info(self, **kwargs):
        return {}


@register_which
class TotalCorrect(BaseWhich):
    def __init__(self, skill, **kwargs):
        super().__init__(skill, **kwargs)
        self.num_correct = 0
        self.num_incorrect = 0
    def ifit(self, state, skill_app, reward):
        if(reward > 0):
            self.num_correct += 1
        else:
            self.num_incorrect += 1

    def get_utility(self, state, skill_app):
        return self.num_correct

    

@register_which
class WhenPrediction(BaseWhich):
    def __init__(self, skill, **kwargs):
        super().__init__(skill, **kwargs)
    def ifit(self, state, skill_app, reward):
        pass

    def get_utility(self, state, skill_app):
        when_pred = getattr(skill_app,'when_pred', 0)
        return when_pred if when_pred is not None else 0
        
    # @staticmethod
    # def sort(state, skill_applications):
    #     return sorted(skill_applications, key=lambda s: getattr(s,'when_prob', 0))


@register_which
class ProportionCorrect(TotalCorrect):
    def get_utility(self, state, skill_app):
        p,n = self.num_correct, self.num_incorrect
        s = p + n
        return (p / s if s > 0 else 0,  s)

@register_which
class WeightedProportionCorrect(TotalCorrect):
    def get_utility(self,state, skill_app, w=2.0):
        p,n = self.num_correct, w*self.num_incorrect
        s = p + n
        return (p / s if s > 0 else 0,  s)

@register_which
class NonLinearProportionCorrect(TotalCorrect):
    def get_utility(self,state, skill_app, a=1.0,b=.25):
        p,n = self.num_correct, self.num_incorrect
        n = a*n + b*(n*n)
        s = p + n
        return (p / s if s > 0 else 0,  s)

