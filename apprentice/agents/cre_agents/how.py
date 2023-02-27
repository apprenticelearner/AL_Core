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

func_registry = registries['func']

class ExplanationSet():
    def __init__(self, explanation_tree, arg_foci=None, post_func=None, choice_func=None):
        self.explanation_tree = explanation_tree
        self.choice_func = choice_func
        self.post_func = post_func

        if(explanation_tree is not None):
            if(arg_foci is not None):
                self.explanations = []
                for func_comp, match in explanation_tree:
                    if(func_comp.n_args == len(arg_foci)):
                        self.explanations.append((func_comp, match))
            else:
                self.explanations = list(iter(self.explanation_tree))

            # print("ARG FOCI", [f.id for f in arg_foci])
            def expl_key(tup):
                func_comp, match = tup
                tup = (func_comp.depth, abs(func_comp.n_args-len(match)), func_comp.n_funcs)
                return tup 

            self.explanations = sorted(self.explanations, key=expl_key)
        else:
            self.explanations = []


    def __len__(self):
        # TODO: write a way to efficiently estimate the size of an expl 
        #  tree in CRE
        # return 1 if self.explanation_tree is not None else 0
        return len(self.explanations)

    def choose(self):
        # if(self.choice_func is not None):
        # tree_iter = iter(self.explanation_tree)
        for func, match in self.explanations:
            if(func.n_args != len(match)):
                continue
        # op_comp, match = next(tree_iter)
            if(self.post_func is not None):
                func = self.post_func(func)
                # op_comp = OpComp(self.post_func,op_comp)
            # with PrintElapse("flatten"):
            #     op = op_comp.flatten()
            return func, match
        return None, []

    def __iter__(self):
        for func, match in self.explanations:
            yield (func, match) 

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

                expl_set = ExplanationSet(explanation_tree, arg_foci)
                expl_set.post_func = NumericalToStr
                # print(flt_val, self.function_set, self.search_depth)
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
