from apprentice.agents.cre_agents.environment import Component, Button, TextField, Container
from apprentice.agents.cre_agents.state import encode_neighbors, State

from numba.types import unicode_type
from cre import MemSet
from cre.transform import MemSetBuilder, Flattener, FeatureApplier, RelativeEncoder, Vectorizer
from cre.default_funcs import Equals

def new_mc_addition_state(upper, lower):
    upper, lower = str(upper), str(lower)
    n = max(len(upper),len(lower))

    tf_config = {"type": "TextField", "width" : 100, "height" : 100, "value" : ""}
    # comp_config = tf_config
    # comp_config = {"type": "Component", "width" : 100, "height" : 100}
    hidden_config = {**tf_config, 'locked' : True}
    comp_config = hidden_config

    d_state = {
        "operator" : {"id" : "operator", "x" :-110,"y" : 220 , **comp_config},
        # "line" :     {"id" : "line", "x" :0,   "y" : 325 , **comp_config, "height" : 5},
        "done" :     {"id" : "done", "x" :0, "y" : 440 , **comp_config, "type": "Button"},
        "hidey1" :   {"id" : "hidey1", "x" :n * 110, "y" : 0 , **hidden_config},
        "hidey2" :   {"id" : "hidey2", "x" :0,   "y" : 110 , **hidden_config},
        "hidey3" :   {"id" : "hidey3", "x" :0,   "y" : 220 , **hidden_config},
    }

    for i in range(n):
        offset = (n - i) * 110
        d_state.update({
            f"{i}_carry": {"id" : f"{i}_carry", "x" :offset,   "y" : 0 , **tf_config},
            f"{i}_upper": {"id" : f"{i}_upper", "x" :offset,   "y" : 110 , "locked" : True, **tf_config},
            f"{i}_lower": {"id" : f"{i}_lower", "x" :offset,   "y" : 220 , "locked" : True, **tf_config},
            f"{i}_answer": {"id" : f"{i}_answer", "x" :offset,   "y" : 330 , **tf_config},
        })

    del d_state["0_carry"]

    d_state.update({
        f"{n}_carry": {"id" : f"{n}_carry", "x" :0,   "y" : 0 , **tf_config},
        f"{n}_answer": {"id" : f"{n}_answer", "x" :0,   "y" : 330 , **tf_config},
    })

    for i,c in enumerate(reversed(upper)):
        d_state[f'{i}_upper']['value'] = c

    for i,c in enumerate(reversed(lower)):
        d_state[f'{i}_lower']['value'] = c


    # d_state = encode_neighbors(d_state)

    # pprint(d_state)
    return d_state


def test_encode_neighbors():
    pass

def test_flatten_featurize():
    agent = object()
    state = State(agent)

    fl = Flattener((Component, Button, TextField, Container))
    fe = FeatureApplier([Equals(unicode_type, unicode_type)])

    @state.register_transform(is_incremental=True, prereqs=['working_memory'])
    def flat(state):
        wm = state.get('working_memory')
        return fl(wm)

    @state.register_transform(is_incremental=True, prereqs=['flat'])
    def flat_featurized(state):
        flat = state.get('flat')
        return fe(flat)

    # print(transfrom_registry)

    a = TextField(id="a",value="a")
    b = TextField(id="b",value="b")
    c = TextField(id="c",value="c")
    d = TextField(id="d",value="a")
    wm = MemSet()
    wm.declare(a)
    wm.declare(b)
    wm.declare(c)
    wm.declare(d)

    state.set("working_memory", wm)
    flat = state.get("flat")
    feat = state.get("flat_featurized")
    
    assert set([fact.val for fact in flat]) == {False, "a", "b", "c"}
    assert set([fact.val for fact in feat]) == {True, False, "a", "b", "c"}
    
    print(flat)
    print(feat)

    a = TextField(id="a",value="A")
    b = TextField(id="b",value="B")
    c = TextField(id="c",value="A")
    d = TextField(id="d",value="A")
    wm = MemSet()
    wm.declare(a)
    wm.declare(b)
    wm.declare(c)
    wm.declare(d)

    state.set("working_memory", wm)
    flat = state.get("flat")
    feat = state.get("flat_featurized")
    
    assert set([fact.val for fact in flat]) == {False, "A", "B"}
    assert set([fact.val for fact in feat]) == {True, False, "A", "B"}

    print(flat)
    print(feat)


def test_full_when_pipeline():
    from numba.types import f8, boolean, string
    from cre import Var
    py_dicts = new_mc_addition_state(567,891)
    print(py_dicts)

    agent = object()
    state = State(agent)

    fact_types = (Component, Button, TextField, Container)
    val_types = [f8, string, boolean]

    msb = MemSetBuilder()
    fl = Flattener(fact_types)
    fe = FeatureApplier([Equals(string, string)])
    re = RelativeEncoder(fact_types)
    ve = Vectorizer(val_types)

    @state.register_transform(is_incremental=True, prereqs=['working_memory'])
    def flat(state):
        wm = state.get('working_memory')
        return fl(wm)

    @state.register_transform(is_incremental=True, prereqs=['flat'])
    def flat_featurized(state):
        flat = state.get('flat')
        return fe(flat)

    py_dicts = encode_neighbors(py_dicts)
    wm = msb(py_dicts)
    state.set("working_memory", wm)

    wm = state.get('working_memory')
    flat_featurized = state.get('flat_featurized')

    _vars = [Var(TextField,"sel")]
    facts = [wm.get_fact(id='0_answer')]

    re.set_in_memset(wm)
    rel = re.encode_relative_to(flat_featurized, facts, _vars)

    vec = ve(rel)

    print(vec)



if __name__ == "__main__":
    # test_flatten_featurize()
    test_full_when_pipeline()
