from cre import define_fact, Fact, Conditions
# from cre.default_funcs import CastFloat
from apprentice.agents.cre_agents.extending import new_register_decorator, new_register_all
from .funcs import CastFloat, CastStr
from copy import copy


# NOTE : env_config might be unecessary
# Env Config 
register_env_config = new_register_decorator("env_config", full_descr="environment configuration")



# Fact
register_fact = new_register_decorator("fact", full_descr="fact type")
register_all_facts = new_register_all("fact", types=[Fact], full_descr="fact type")

# Fact set
register_fact_set = new_register_decorator("fact_set", full_descr='fact set')

# Base Constraints
register_constraints = new_register_decorator("constraint", full_descr="base constraint")



with register_all_facts as HTML_fact_types:
    Component = define_fact("Component", {
        "id" : str,
        # "x" : {"type" : float, "visible" : False},
        # "y" : {"type" : float, "visible" : False},
        # "width" : {"type" : float, "visible" : False,},
        # "height" : {"type" : float, "visible" : False},
        "above" : "Component", 
        "below" : "Component",
        "left": "Component", 
        "right" : "Component",
        "parents" : "List(Component)"
    })

    TextField = define_fact("TextField", {
        "inherit_from" : "Component",
        "value" : {"type" : str, "visible" : True, "semantic" : True,
                    'conversions' : {float : CastFloat}},
        "locked" : {"type" : bool, "visible" : True},
    })

    Button = define_fact("Button", {
        "inherit_from" : "Component",
    })

    Container = define_fact("Container", {
        "inherit_from" : "Component",
        "children" : "List(Component)"
    })

    def str_as_id(x):
        return x.id

    # Redefine __repr__ to be more concise
    def text_field_repr(x):
        return f"TextField(id={x.id!r}, value={x.value!r}, locked={x.locked!r})"

    TextField._fact_proxy.__str__ = str_as_id
    TextField._fact_proxy.__repr__ = text_field_repr

    def button_repr(x):
        return f"Button(id={x.id!r})"

    Button._fact_proxy.__str__ = str_as_id
    Button._fact_proxy.__repr__ = button_repr

    def component_repr(x):
        return f"Component(id={x.id!r})"

    Component._fact_proxy.__str__ = str_as_id
    Component._fact_proxy.__repr__ = component_repr

    # TextField._fact_proxy.__repr__ = text_field_str

register_fact_set(name='html')(HTML_fact_types)
# with register_all_actions as HTML_action_types:


@register_constraints(name='none')
def default_constraints(_vars):
    sel, args = _vars[0], _vars[1:]

    conds = Conditions(sel)
    for arg in args:
        conds &= arg
    return conds

@register_constraints(name='html')
def html_constraints(_vars):
    sel, args = _vars[0], _vars[1:]
    conds = default_constraints(_vars)
        
    if(sel.base_type._fact_name == "TextField"):
        conds &= (sel.locked == False)

    for arg in args:
        if(arg.base_type._fact_name == "TextField"):
            conds &= (arg.value != '')        

    return conds


# -------------------------
#  : ActionType
# NOTE: work in progress 

class ActionType(object):
    def __init__(self, name, input_spec, apply_expected_change):
        self.name = name
        self.input_spec = input_spec
        self.apply_expected_change = apply_expected_change

    def predict_state_change(self, state, sai):
        cpy = copy(state)
        selection = cpy.get_fact(id=sai.selection.id)
        self.apply_expected_change(cpy, selection, sai.inputs)
        return cpy

    def __getitem__(self, attr):
        return self.input_spec[attr]

    def get(self,attr,default):
        return self.input_spec.get(attr,default)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"ActionType(name={self.name}, spec={self.input_spec})"

def define_action_type(name, input_spec, *args):
    def wrapper(apply_expected_change):
        return ActionType(name, input_spec, apply_expected_change)
    if(len(args) > 0):
        return wrapper(*args)
    else:
        return wrapper


# NOTE : action might be unecessary
# Action
register_action_type = new_register_decorator("action_type", full_descr="action type")
register_all_action_types = new_register_all("action_type", types=[ActionType], full_descr="action type")

# Action Set
register_action_type_set = new_register_decorator("action_type_set", full_descr='action type set')



with register_all_action_types as HTML_action_type_set:
    # NOTE need to 

    @define_action_type("PressButton", {
        'value' : {'type' : int, "semantic" : False}
        })
    def PressButton(wm, selection, inputs):
        pass

    @define_action_type("ButtonPressed", {
        'value' : {'type' : int, "semantic" : False}
        })
    def ButtonPressed(wm, selection, inputs):
        pass

    @define_action_type("UpdateTextArea", {
        'value' : {'type' : str, "semantic" : True}
        })
    def UpdateTextArea(wm, selection, inputs):
        wm.modify(selection, 'value', inputs['value'])
        wm.modify(selection, 'locked', True)

    @define_action_type("UpdateTextField", {
        'value' : {'type' : str, "semantic" : True}
        })
    def UpdateTextField(wm, selection, inputs):
        wm.modify(selection, 'value', inputs['value'])
        wm.modify(selection, 'locked', True)

    @define_action_type("UpdateField", {
        'value' : {'type' : str, "semantic" : True}
        })
    def UpdateField(wm, selection, inputs):
        wm.modify(selection, 'value', inputs['value'])
        wm.modify(selection, 'locked', True)

HTML_action_type_set = {x.name: x for x in HTML_action_type_set}
# HTML_action_type_set = {
#     "UpdateTextField" : UpdateTextField,
#     "UpdateField" : UpdateTextField,

#     "PressButton" : PressButton,
#     "ButtonPressed" : PressButton,
# }
register_action_type_set(name='html')(HTML_action_type_set)
