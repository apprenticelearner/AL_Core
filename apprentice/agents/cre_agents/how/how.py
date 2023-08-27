import warnings
from abc import ABCMeta
from abc import abstractmethod
from ..extending import new_register_decorator, registries
from cre.utils import PrintElapse


# ------------------------------------------------------------------------
# : How Base

register_how = new_register_decorator("how", full_descr="how-learning mechanism")

# TODO: COMMENTS
class BaseHow(metaclass=ABCMeta):
    @abstractmethod
    def get_explanations(self, state, goal):
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
from ..conv_funcs import register_conversion

func_registry = registries['func']

class ExplanationSet():
    def __init__(self, explanation_tree, arg_foci=None,
                 post_func=None, choice_func=None, max_expls=1000):
        self.explanation_tree = explanation_tree
        self.choice_func = choice_func
        self.post_func = post_func

        if(explanation_tree is not None):
            if(isinstance(explanation_tree, list)):
                self.explanations = explanation_tree
            else:
                self.explanations = []
                for i, (func_comp, match) in enumerate(explanation_tree):
                    print(func_comp, match)
                    if(max_expls != -1 and i >= max_expls-1): break

                    # Skip 
                    if(func_comp.n_args != len(match) or
                       (arg_foci is not None and func_comp.n_args != len(arg_foci))):
                        continue

                    if(self.post_func is not None):
                        func_comp = self.post_func(func_comp)

                    self.explanations.append((func_comp, match))
            
                # Sort by min depth, degree to which variables match unique goals,
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
    def __init__(self, 
            agent=None,
            search_depth=2, 
            function_set=[], 
            float_to_str=True, 
            **kwargs):
        # print("SC", kwargs)
        self.agent = agent
        self.function_set = function_set
        self.search_depth = search_depth
        self.float_to_str = float_to_str
        self.fact_types = kwargs.get('fact_types', self.agent.fact_types if (self.agent) else [])
        for fn in function_set:
            assert isinstance(fn, CREFunc), \
"function_set must consist of CREFunc intances for SetChaining how-learning mechanism." 


    def _search_for_explanations(self, goal, values, extra_consts=[],  **kwargs):
        # Fallback on any parameters set in __init__()
        kwargs['funcs'] = kwargs.get('function_set', self.function_set)
        if('function_set' in kwargs): del kwargs['function_set']
        kwargs['search_depth'] = kwargs.get('search_depth', self.search_depth)
        
        
        # Make a new planner instance and fill it with values 
        planner = SetChainingPlanner(self.fact_types)
        for v in values:
            planner.declare(v)

        for v in extra_consts:
            planner.declare(v,is_const=True)

        # Search for explanations
        explanation_tree = planner.search_for_explanations(goal, **kwargs)
        self.num_forward_inferences = planner.num_forward_inferences
        return explanation_tree

    def get_explanations(self, state, goal, arg_foci=None, float_to_str=None,
                            extra_consts=[], **kwargs):
        # Prevent from learning too shallow when multiple foci
        if(arg_foci is not None and len(arg_foci) > 1):
            # arg_foci = list(reversed(arg_foci))
            # Don't allow fallback to constant 
            kwargs['min_stop_depth'] = kwargs.get('min_stop_depth', kwargs.get('search_depth',getattr(self, 'search_depth', 2)))
            kwargs['min_solution_depth'] = 1

        if(isinstance(state, list)):
            values = state
        else:
            wm = state.get("working_memory")
            values = list(wm.get_facts()) if arg_foci is None else arg_foci

        float_to_str = float_to_str if float_to_str is not None else self.float_to_str

        try:
            flt_goal = float(goal)
        except ValueError:
            explanation_tree = None
        else:
            explanation_tree = self._search_for_explanations(flt_goal, values, extra_consts, **kwargs)
            post_func = NumericalToStr if (float_to_str) else None
        

        # Try to find the goal as a string
        if(explanation_tree is None):
            # TODO: Shouldn't full reset and run a second time here, should just query.
            explanation_tree = self._search_for_explanations(goal, values, extra_consts, **kwargs)        
            post_func = None
        
        expl_set = ExplanationSet(explanation_tree, arg_foci, post_func=post_func)

        # if(expl_set is not None):
        #     for op_comp, match in expl_set:
        #         print("<<", op_comp, [m.id for m in match])


        return expl_set

    def new_explanation_set(self, explanations, *args, **kwargs):
        '''Takes a list of explanations i.e. (func, match) and yields an ExplanationSet object'''
        return ExplanationSet(explanations,*args, **kwargs)
