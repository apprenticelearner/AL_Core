import warnings
from abc import ABCMeta
from abc import abstractmethod
from .extending import new_register_decorator

# ------------------------------------------------------------------------
# : BaseWhich

register_which = new_register_decorator("which", full_descr="which-learning mechanism")

# TODO: COMMENTS
class BaseWhich(metaclass=ABCMeta):
    def __init__(self, skill,**kwargs):
        self.skill = skill
        self.agent = skill.agent

    def ifit(self, state, match, reward):
        """
        
        :param state: 
        """
        raise NotImplemented()

    def sort(self, skill_applications):
        """
        
        :param state: 
        """
        raise NotImplemented()

    def get_utility(self, state, match):
        """
        
        """
        raise NotImplemented()


@register_which
class TotalCorrect(BaseWhich):
    def __init__(self, skill, **kwargs):
        super().__init__(skill, **kwargs)
        self.num_correct = 0
        self.num_incorrect = 0
    def ifit(self, state, match, reward):
        if(reward > 0):
            self.num_correct += 1
        else:
            self.num_incorrect += 1

    def get_utility(self, state, match):
        return self.num_correct

    @staticmethod
    def sort(state, skill_applications):
        def key_func(skill_app):
            which_lrn_mech = skill_app.skill.which_lrn_mech
            return which_lrn_mech.get_utility(state, skill_app.match)

        return sorted(skill_applications, key=key_func)


@register_which
class ProportionCorrect(TotalCorrect):
    def get_utility(self,state, match):
        p,n = self.num_correct, self.num_incorrect
        s = p + n
        return (p / s if s > 0 else 0,  s)

@register_which
class WeightedProportionCorrect(TotalCorrect):
    def get_utility(self,state, match, w=2.0):
        p,n = self.num_correct, w*self.num_incorrect
        s = p + n
        return (p / s if s > 0 else 0,  s)

@register_which
class NonLinearProportionCorrect(TotalCorrect):
    def get_utility(self,state, match, a=1.0,b=.25):
        p,n = self.num_correct, self.num_incorrect
        n = a*n + b*(n*n)
        s = p + n
        return (p / s if s > 0 else 0,  s)

