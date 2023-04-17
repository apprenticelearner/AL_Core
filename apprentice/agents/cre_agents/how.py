import warnings
from abc import ABCMeta
from abc import abstractmethod
from .extending import new_register_decorator
from cre.utils import PrintElapse


# ------------------------------------------------------------------------
# : How Base

register_how = new_register_decorator("how", full_descr="how-learning mechanism")

# TODO: COMMENTS
class BaseHow(metaclass=ABCMeta):
    @abstractmethod
    def get_explanations(self, state, value):
        """
        
        :param state: 
        """
        pass


# ------------------------------------------------------------------------
# : How Learning Mechanisms


# --------------------------------------------------
# : SetChaining

from numba.types import string, f8
from cre.sc_planner import SetChainingPlanner
from cre.func import CREFunc
from .extending import registries
from .funcs import register_conversion

func_registry = registries['func']

class ExplanationSet():
    def __init__(self, explanation_tree, arg_foci=None,
                 post_func=None, choice_func=None, max_expls=200):
        self.explanation_tree = explanation_tree
        self.choice_func = choice_func
        self.post_func = post_func

        if(explanation_tree is not None):
            if(isinstance(explanation_tree, list)):
                self.explanations = explanation_tree
            else:
                self.explanations = []
                for i, (func_comp, match) in enumerate(explanation_tree):
                    if(max_expls != -1 and i >= max_expls-1): break

                    # Skip 
                    if(func_comp.n_args != len(match) or
                       (arg_foci is not None and func_comp.n_args != len(arg_foci))):
                        continue

                    if(self.post_func is not None):
                        func_comp = self.post_func(func_comp)

                    self.explanations.append((func_comp, match))
            
                # Sort by min depth, degree to which variables match unique values,
                #  and total funcs in the composition.
                def expl_key(tup):
                    func_comp, match = tup
                    tup = (func_comp.depth, abs(func_comp.n_args-len(match)), func_comp.n_funcs)
                    return tup 
                self.explanations = sorted(self.explanations, key=expl_key)
        else:
            self.explanations = []


    def __len__(self):
        return len(self.explanations)

    def choose(self):
        if(len(self.explanations) > 0):
            return self.explanations[0]
        return None, []


    def __iter__(self):
        for func, match in self.explanations:
            yield (func, match) 

@register_conversion(name="NumericalToStr")
@CREFunc(signature=string(f8),
    shorthand="s({0})")
def NumericalToStr(x):
    if(int(x) == x):
        return str(int(x))
    else:
        return str(x)


@register_how
class SetChaining(BaseHow):
    def __init__(self, agent, search_depth=2, function_set=[], **kwargs):
        self.agent = agent
        self.search_depth = search_depth
        self.function_set = function_set
        for fn in function_set:
            assert isinstance(fn, CREFunc), \
"function_set must consist of Op intances for SetChaining how-learning mechanism." 

    def get_explanations(self, state, value, arg_foci=None,
            function_set=None, min_stop_depth=-1, search_depth=None):

        wm = state.get("working_memory")

        if(search_depth is None):
            search_depth = self.search_depth

        if(function_set is None):
            function_set = self.function_set

        # Prevent from learning too shallow when multiple foci
        if(arg_foci is not None and len(arg_foci) > 1):
            arg_foci = list(reversed(arg_foci))
            min_stop_depth = search_depth
        
        # with PrintElapse("get_facts"):
        facts = wm.get_facts() if arg_foci is None else arg_foci
        # print("how get_explanations:", function_set, search_depth)

        # with PrintElapse("declare"):
        planner = SetChainingPlanner(self.agent.fact_types)
        for fact in facts:
            planner.declare(fact)
            # print(fact)

        # Try to find the value as a string
        # with PrintElapse("search_for_explanations_str"):
        explanation_tree = planner.search_for_explanations(
            value, function_set, search_depth=search_depth, 
            min_stop_depth=min_stop_depth)

        # If fail try float; NOTE: Still a little bit of a kludge
        if(explanation_tree is None):
            planner = SetChainingPlanner(self.agent.fact_types)
            for fact in facts:
                planner.declare(fact)
            try:
                # with PrintElapse("search_for_explanation_float"):
                flt_val = float(value)
                explanation_tree = planner.search_for_explanations(
                    flt_val, function_set, search_depth=search_depth, 
                    min_stop_depth=min_stop_depth)

                expl_set = ExplanationSet(explanation_tree, arg_foci, post_func=NumericalToStr)
            except:
                # with PrintElapse("expl_set_float"):
                expl_set = ExplanationSet(None, arg_foci)
        else:
            # with PrintElapse("expl_set_str"):
            expl_set = ExplanationSet(explanation_tree, arg_foci)

        # if(expl_set is not None):
        #     for op_comp, match in expl_set:
        #         print("<<", op_comp, [m.id for m in match])


        return expl_set

    def new_explanation_set(self, explanations, *args, **kwargs):
        '''Takes a list of explanations i.e. (func, match) and yields an ExplanationSet object'''
        return ExplanationSet(explanations,*args, **kwargs)
