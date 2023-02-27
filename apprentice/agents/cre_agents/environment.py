from cre import define_fact, Fact, Conditions
from cre.default_ops import CastFloat
from apprentice.agents.cre_agents.extending import new_register_decorator, new_register_all
from apprentice.agents.cre_agents.ops import CastFloat, CastStr


# NOTE : env_config might be unecessary
# Env Config 
register_env_config = new_register_decorator("env_config", full_descr="environment configuration")

# NOTE : action might be unecessary
# Action
register_action = new_register_decorator("action", full_descr="action type")
register_all_actions = new_register_all("action", types=[], full_descr="action type")

# Fact
register_fact = new_register_decorator("fact", full_descr="fact type")
register_all_facts = new_register_all("fact", types=[Fact], full_descr="fact type")

# Fact set
register_fact_set = new_register_decorator("fact_set", full_descr='fact set')

# Base Constraints
register_constraints = new_register_decorator("constraint", full_descr="base constraint")


with register_all_facts as Tree_fact_types:
    Node = define_fact("Node", {
        "id": str,
        "parent": "Node",
        "value": str,
    })

register_fact_set(name='tree')(Tree_fact_types)

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

    # NOTE: For now TextField will be the base class 
    # TextField = define_fact("TextField", {
    #     "id" : str,
    #     "above" : "TextField", 
    #     "below" : "TextField",
    #     "left": "TextField", 
    #     "right" : "TextField",
    #     "parents" : "List(TextField)",
    #     "value" : {"type" : str, "visible" : True, "semantic" : True,
    #                 'conversions' : {float : CastFloat}},
    #     "locked" : {"type" : bool, "visible" : True},
    # })

    # Button = define_fact("Button", {
    #     "inherit_from" : "TextField",
    # })

    # Container = define_fact("Container", {
    #     "inherit_from" : "TextField",
    #     "children" : "List(TextField)"
    # })


    # Redefine __str__ to be more concise
    def text_field_str(x):
        return f"TextField(id={x.id!r}, value={x.value!r}, locked={x.locked!r})"

    TextField._fact_proxy.__str__ = text_field_str
    TextField._fact_proxy.__repr__ = text_field_str

    def button_str(x):
        return f"Button(id={x.id!r})"

    Button._fact_proxy.__str__ = button_str
    Button._fact_proxy.__repr__ = button_str

    def component_str(x):
        return f"Component(id={x.id!r})"

    Component._fact_proxy.__str__ = component_str
    Component._fact_proxy.__repr__ = component_str

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


