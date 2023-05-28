from apprentice.agents.cre_agents.how.nlp.nlp_sc_planner import NLPSetChaining, func_to_policy
from apprentice.agents.cre_agents.funcs import Add, Add3, Subtract, Multiply, OnesDigit, TensDigit, CastFloat
from cre import CREFunc, define_fact, MemSet, Var
from numba.types import f8, unicode_type
import numpy as np

IE = define_fact("IE", {
    "id" : str,
    "value" : {"type" : str, "visible" : True, "semantic" : True,
                'conversions' : {float : CastFloat}},
})
IE._fact_proxy.__str__ = lambda x: f"{x.value}@{x.id})"


def state_w_values(vals):
    state = MemSet()
    for i, val in enumerate(vals):
        state.declare(IE(str(i), str(val)))
    return {"working_memory" : state}

def test_func_to_policy():
    # 1
    a, b = Var(f8,'a'), Var(f8,'b')
    cf =  Add(Subtract(a,7.0),b)
    policy = func_to_policy(cf, [1,3])
    assert str(policy) == "[[(Subtract(a, b), [1, 7.0])], [(Add(a, b), [3])]]"

    # 2
    cf = Add(a,a)
    policy = func_to_policy(cf, [1])
    assert str(policy) == "[[(Add(a, a), [1])]]"

    
    # 3
    BOOP = define_fact("BOOP", {"A" :unicode_type, "B" :f8})
    v = Var(BOOP, 'v')
    cf = Add(v.B,v.B)
    policy = func_to_policy(cf, [BOOP("1", 1)])
    assert str(policy) == "[[(Add(a, a), [1.0])]]"

    Float = CastFloat(unicode_type)

    # 4
    a, b = Var(IE,'a'), Var(IE,'b')
    cf = TensDigit((Float(a.value) + Float(b.value)))
    policy = func_to_policy(cf, [IE("7", "7"), IE("6", "6")], conv_funcs=[Float])
    assert str(policy) == "[[(Add(a, b), [7.0, 6.0])], [(TensDigit(a), [])]]"

    # 5
    cf = TensDigit((Float(a.value) + Float(a.value)))
    policy = func_to_policy(cf, [IE("7", "7")], conv_funcs=[Float])
    assert str(policy) == "[[(Add(a, a), [7.0])], [(TensDigit(a), [])]]"

def test_basic_searches():
    planner = NLPSetChaining(
        fact_types=(IE,), float_to_str=False,
        function_set=[Add, Add3, OnesDigit, TensDigit]
        )    
    # If multiple args are stated then expl1 should not include
    #  any explanations like Add(a,a), only Add(a,b)

    expls1 = planner.get_explanations(
        state_w_values([7,7,7,7,7,7]),
        1, 
        "Add 7 and 7"
    )

    expls2 = planner.get_explanations(
        state_w_values([7,7,7,7,7,7]),
        1, 
        "Add"
    )

    assert len(expls1) < len(expls2)

    # If arg_foci are given either of these cases then there should only 
    #  be one explanation 

    state = state_w_values([7,7,7,7,7,7])
    wm = state.get('working_memory')
    arg_foci = [wm.get_fact(id="0"), wm.get_fact(id="1")]
    expls1 = planner.get_explanations(
        state,
        1, 
        "Add 7 and 7",
        arg_foci=arg_foci
    )
    assert len(expls1) == 1

    expls2 = planner.get_explanations(
        state,
        1, 
        "Add",
        arg_foci=arg_foci
    )    
    assert len(expls2) == 1


if __name__ == "__main__":
    test_func_to_policy()
    test_basic_searches()



