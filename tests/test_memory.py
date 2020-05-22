from apprentice.working_memory import ExpertaWorkingMemory
from experta import Fact
import jsondiff


def test_update_simple():
    class Fruit(Fact):
        pass
    wm = ExpertaWorkingMemory()
    facts = {1: {'color': 'red', 'name': 'apple_1', '__class__' : Fruit}, 2: {'color': 'yellow', 'name': 'banana', '__class__' : Fruit}}
    wm.update(facts)
    fl = wm.facts
    assert len(fl.keys()) == 3

def test_update_diff():
    class Fruit(Fact):
        pass
    wm = ExpertaWorkingMemory()
    facts1 = {1: {'color': 'red', 'name': 'apple', '__class__' : Fruit}, 2: {'color': 'yellow', 'name': 'banana', '__class__' : Fruit}}
    wm.update(facts1)
    fl = wm.facts

    assert len(fl.keys()) == 3

    facts2 = {1: {'color': 'orange', 'name': 'orange', '__class__': Fruit},
              2: {'color': 'yellow', 'name': 'banana', '__class__': Fruit},
              3: {'color': 'purple', 'name': 'plum', '__class__': Fruit}}

    diff = jsondiff.diff(facts1, facts2)

    wm.update(diff)
    assert len(wm.facts.keys()) == 4



