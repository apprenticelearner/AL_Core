import inspect
from dataclasses import dataclass
from typing import Collection, Callable, Any

import experta
from experta import Fact
from experta.conditionalelement import ConditionalElement as Condition

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer


# class Condition(tuple):
#     def __new__(cls, *args):
#         return tuple.__new__(Condition, args)


# class Fact(dict):
#     def __new__(cls, *args):
#         return dict.__new__(Fact, args)


@dataclass  # (frozen=True) #
class Sai:
    selection: Any
    action: Any
    inputs: Any

    def __post_init__(self):
        self.__source__ = None
        try:
            activation_frame = inspect.currentframe().f_back
            for i in range(7):
                if 'self' in activation_frame.f_locals:
                    if type(activation_frame.f_locals[
                                'self']) == experta.activation.Activation:

                        self.__source__ = activation_frame.f_locals['self']
                        #print("!!!Source assigned: ", self.__source__)
                        break
                activation_frame = activation_frame.f_back
        except (AssertionError, AttributeError, KeyError) as e:
            #print("!!!Error assingning source: ", e)
            pass


@dataclass(frozen=True)
class Skill:
    conditions: Collection[Condition]
    function_: Callable
    # name: str = "skill_" + str(uuid.uuid1())


@dataclass(frozen=True)
class Activation:
    skill: Skill
    context: dict

    @property
    def fire(self) -> Any:
        raise NotImplementedError

    def __hash__(self):
        return hash(self.as_hash_repr())

    #     from experta import Fact
    #     c = {}
    #     for k,v in self.context.items():
    #         if isinstance(v, Fact):
    #             c[k] = v.as_frozenset()
    #         else:
    #             c[k] = v

    #     return hash((self.skill, frozenset(c)))

    def get_rule_name(self):
        return self.skill.function_.__name__

    def get_rule_bindings(self):
        bindings = {}

        facts = sorted([(k, v) for k, v in self.context.items() if
                        isinstance(v, Fact)])
        facts = [v for k, v in facts]

        for i, v in enumerate(facts):
            for fk, fv in v.items():
                if Fact.is_special(fk):
                    continue
                bindings['fact-%i: %s' % (i, fk)] = fv

        # print(bindings)
        return bindings

    def as_hash_repr(self):
        c = {}
        for k, v in self.context.items():
            if isinstance(v, Fact):
                c[k] = frozenset([(fk, fv) for fk, fv in v.items() if not
                Fact.is_special(fk)])
            else:
                c[k] = v

        return self.skill, frozenset(c.items())

def compute_exp_depth(exp):
    """
    Doc String
    """
    if isinstance(exp, tuple):
        return 1 + max([compute_exp_depth(sub) for sub in exp])
    return 0


class RHS(object):
    def __init__(self, selection_expr, action, input_rule, selection_var,
                 input_vars, input_attrs, conditions=[], label=None):
        self.selection_expr = selection_expr
        self.action = action
        self.input_rule = input_rule
        self.selection_var = selection_var
        self.input_vars = input_vars
        self.input_attrs = input_attrs
        self.all_vars = tuple([self.selection_var] + self.input_vars)
        self.as_tuple = (self.selection_expr, self.action, self.input_rule)

        self.conditions = conditions
        self.label = label
        self._how_depth = None
        self._id_num = None

        self.where = None
        self.when = None
        self.which = None

    def to_xml(self, agent=None):  # -> needs some way of representing itself including its when/where/how parts
        raise NotImplementedError()

    def get_how_depth(self):
        if(self._how_depth == None):
            self._how_depth = compute_exp_depth(self.input_rule)
        return self._how_depth

    def __hash__(self):
        return self._id_num

    def __eq__(self, other):
        a = self._id_num == other._id_num
        b = self._id_num is not None
        c = other._id_num is not None
        return a and b and c
    def __str__(self):
        return str(self.input_rule)
    def __repr__(self):
        return self.__str__()

def ground(arg):
    """
    Doc String
    """
    if isinstance(arg, tuple):
        return tuple(ground(e) for e in arg)
    elif isinstance(arg, str):
        return arg.replace('?', 'QM')
    else:
        return arg


def unground(arg):
    """
    Doc String
    """
    if isinstance(arg, tuple):
        return tuple(unground(e) for e in arg)
    elif isinstance(arg, str):
        return arg.replace('QM', '?')
    else:
        return arg


def flatten_state(state):
    tup = Tuplizer()
    flt = Flattener()
    state = flt.transform(tup.transform(state))
    return state


def grounded_key_vals_state(state):
    return [(ground(a), state[a].replace('?', 'QM')
            if isinstance(state[a], str)
            else state[a])
            for a in state]


def kb_to_flat_ungrounded(knowledge_base):
    state = {unground(a): v.replace("QM", "?")
             if isinstance(v, str)
             else v
             for a, v in knowledge_base.facts}
    return state


from numbert.numbalizer import Numbalizer
numbalizer = Numbalizer()



# numbalizer.register_specification("CTATTable--cell",ctattextinput)
# numbalizer.register_specification("CTATHintWindow",ctathintwindow)
# numbalizer.register_specification("CTATHintButton",ctathintwindow)
# numbalizer.register_specification("CTATDoneButton",ctathintwindow)

class StateMultiView(object):
    def __init__(self, view, state):
        self.views = {"*":self}
        self.set_view(view, state)
        self.transform_dict = {}
        self.register_transform("object", "flat_ungrounded", flatten_state)
        self.register_transform("object", "nb_object", numbalizer.state_to_nb_objects)


        self.register_transform("flat_ungrounded", "key_vals_grounded",
                                grounded_key_vals_state)
        self.register_transform("feat_knowledge_base", "flat_ungrounded",
                                kb_to_flat_ungrounded)

    def set_view(self, view, state):
        self.views[view] = state

    def get_view(self, view):
        out = self.views.get(view, None)
        if(out is None):
            return self.compute(view)
        else:
            return out

    def contains_view(self, view):
        return view in self.views

    def compute(self, view):
        if(isinstance(view,tuple)):
            view_key = view[0]
            view_args = view[1:]
        else:
            view_key = view
            view_args = []

        for key in self.transform_dict[view_key]:
            if(key in self.views):
                out = self.transform_dict[view_key][key](self.views[key],*view_args)
                self.set_view(view, out)
                return out
        raise Exception("No transform possible from %s to %r" %
                        (list(self.views.keys()), view))

    def compute_from(self, to, frm):
        assert to in self.transform_dict
        assert frm in self.transform_dict[to]
        out = self.transform_dict[to][frm](self.views[frm])
        self.set_view(to, out)
        return out

    def register_transform(self, frm, to, function):
        transforms = self.transform_dict.get(to, {})
        transforms[frm] = function
        self.transform_dict[to] = transforms

class Explanation(object):
    def __init__(self, rhs, mapping):
        assert isinstance(mapping, dict), \
               "Mapping must be type dict got type %r" % type(mapping)
        self.rhs = rhs
        self.mapping = mapping
        self.selection_literal = mapping[rhs.selection_var]
        self.input_literals = [mapping[s] for s in rhs.input_vars]

    def compute(self, state, agent):
        #Note: I have no recollection of why this passes a list
        v = agent.planner.eval_expression([self.rhs.input_rule],
                                          self.mapping, state)[0]

        return {self.rhs.input_attrs[0]: v}

    def conditions_apply(self):
        return True

    def to_response(self, state, agent):
        response = {}
        response['skill_label'] = self.rhs.label
        response['selection'] = self.selection_literal.replace("?ele-", "")
        response['action'] = self.rhs.action
        response['inputs'] = self.compute(state, agent)
        response['rhs_id'] = self.rhs._id_num
        return response

    def get_skill_info(self,agent,when_state=None):
        if(when_state is None):
            when_info = None
        else:    
            when_info = tuple(agent.when_learner.skill_info(self.rhs, when_state))
        skill_info = {"when": when_info,
                      "where": agent.where_learner.skill_info(self.rhs),
                      "how": str(self.rhs.input_rule),
                      "which": 0.0,
                      "mapping" : self.mapping}
        return skill_info

    def to_xml(self, agent=None):  # -> needs some way of representing itself including its when/where/how parts
        pass

    def get_how_depth(self):
        return self.rhs.get_how_depth()

    def __str__(self):
        r = str(self.rhs.input_rule)
        args = ",".join([x.replace("?ele-", "")
                        for x in self.input_literals])
        sel = self.selection_literal.replace("?ele-", "")
        return r + ":(" + args + ")->" + sel
