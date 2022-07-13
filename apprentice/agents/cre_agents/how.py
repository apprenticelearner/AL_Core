import warnings
from abc import ABCMeta
from abc import abstractmethod
from .extending import new_register_decorator


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
from cre.op import Op, OpComp
from .extending import registries

op_registry = registries['op']

class ExplanationSet():
    def __init__(self, explanation_tree, arg_foci=None, post_op=None, choice_func=None):
        self.explanation_tree = explanation_tree
        self.choice_func = choice_func
        self.post_op = post_op

        if(explanation_tree is not None):
            if(arg_foci is not None):
                self.explanations = []
                for op_comp, match in explanation_tree:
                    if(len(op_comp.base_vars) == len(arg_foci)):
                        self.explanations.append((op_comp, match))
            else:
                self.explanations = list(iter(self.explanation_tree))

            def expl_key(tup):
                op_comp, match = tup
                return (op_comp.depth, abs(op_comp.n_terms-len(match)))

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
        

        for op_comp, match in self.explanations:
            if(op_comp.n_terms != len(match)):
                continue
        # op_comp, match = next(tree_iter)
            if(self.post_op is not None):
                op_comp = OpComp(self.post_op,op_comp)
            op = op_comp.flatten()
            return op, match
        return None, []

    def __iter__(self):
        for op, match in self.explanations:
            yield (op, match) 

@Op(signature=string(f8),
    shorthand="{0}")
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
            assert isinstance(fn, Op), \
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
            min_stop_depth = search_depth
        
        facts = wm.get_facts() if arg_foci is None else arg_foci
        # print("how get_explanations:", function_set, search_depth)

        planner = SetChainingPlanner(self.agent.fact_types)
        for fact in facts:
            planner.declare(fact)
            # print(fact)

        # Try to find the value as a string
        explanation_tree = planner.search_for_explanations(
            value, function_set, search_depth, 
            min_stop_depth=min_stop_depth)

        # If fail try float; NOTE: Still a little bit of a kludge
        if(explanation_tree is None):
            planner = SetChainingPlanner(self.agent.fact_types)
            for fact in facts:
                planner.declare(fact)
            try:
                flt_val = float(value)
                explanation_tree = planner.search_for_explanations(
                    flt_val, function_set, search_depth, 
                    min_stop_depth=min_stop_depth)

                expl_set = ExplanationSet(explanation_tree, arg_foci)
                expl_set.post_op = NumericalToStr
                # print(flt_val, self.function_set, self.search_depth)
            except:
                expl_set = ExplanationSet(None, arg_foci)
        else:
            expl_set = ExplanationSet(explanation_tree, arg_foci)

        # if(expl_set is not None):
        #     for op_comp, match in expl_set:
        #         print("<<", op_comp, [m.id for m in match])

        return expl_set
