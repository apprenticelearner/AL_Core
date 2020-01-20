from apprentice.explain.explanation import Explanation
from apprentice.working_memory.skills import AdditionEngine, EmptyAdditionEngine
from apprentice.working_memory.representation import Sai
from apprentice.working_memory import ExpertaWorkingMemory
import inspect

from experta import Fact

def old_test_compile_addition():
    engine = AdditionEngine()

    engine.reset()

    f1 = Fact(id='JCommTable.R0C0', value='1', contentEditable=False)
    f2 = Fact(id='JCommTable.R1C0', value='2', contentEditable=False)
    f3 = Fact(id='JCommTable.R1C1', contentEditable=True, value='')

    engine.declare(f1)
    engine.declare(f2)
    engine.declare(f3)
    engine.run(10)

    ex = Explanation(engine.sais[0])
    nr = ex.new_rule

    new_wm = ExpertaWorkingMemory(EmptyAdditionEngine())
    new_wm.add_rule(nr)

    f1b = Fact(id='JCommTable.R0C0', value='1', contentEditable=False)
    f2b = Fact(id='JCommTable.R1C0', value='2', contentEditable=False)
    f3b = Fact(id='JCommTable.R1C1', contentEditable=True, value='')

    new_wm.ke.declare(f1b)
    new_wm.ke.declare(f2b)
    new_wm.ke.declare(f3b)
    # test that the new rule fires correctly
    new_wm.ke.run(10)
    s = new_wm.ke.sais[0]
    assert s == Sai(selection='JCommTable.R1C1',
                                    action='UpdateTextField',
                                    input={'value': '3'})


def test_compile_addition_sai():
    engine = AdditionEngine()

    engine.reset()

    f1 = Fact(id='JCommTable.R0C0', value='1', contentEditable=False)
    f2 = Fact(id='JCommTable.R1C0', value='2', contentEditable=False)
    f3 = Fact(id='JCommTable.R1C1', contentEditable=True, value='')

    engine.declare(f1)
    engine.declare(f2)
    engine.declare(f3)
    engine.run(10)

    ex = Explanation(engine.sais[0])
    nr = ex.new_rule

    new_wm = ExpertaWorkingMemory(EmptyAdditionEngine())
    new_wm.add_rule(nr)

    f1b = Fact(id='JCommTable.R0C0', value='1', contentEditable=False)
    f2b = Fact(id='JCommTable.R1C0', value='2', contentEditable=False)
    f3b = Fact(id='JCommTable.R1C1', contentEditable=True, value='')

    new_wm.ke.declare(f1b)
    new_wm.ke.declare(f2b)
    new_wm.ke.declare(f3b)
    # test that the new rule fires correctly
    new_wm.ke.run(10)
    s = new_wm.ke.sais[0]
    assert s == Sai(selection='JCommTable.R1C1',
                                    action='UpdateTextField',
                                    input={'value': '3'})