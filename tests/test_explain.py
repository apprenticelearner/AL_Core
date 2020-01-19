from apprentice.explain.explanation import Explanation
from apprentice.working_memory import ExpertaWorkingMemory
from apprentice.working_memory.skills import AdditionEngine

from kill_engine import KillEngine, KillEngineEmpty
from experta import Fact, KnowledgeEngine

def test_explain():
    cf = KillEngine()
    cf.reset()
    cf.run(10)
    kill_fact = cf.facts[7]

    new_wm = ExpertaWorkingMemory(KillEngineEmpty())

    # generate a new rule and assign it to the blank working memory
    x = Explanation(kill_fact)
    r = x.new_rule
    new_wm.add_rule(r)

    # test that the new rule fires correctly
    new_wm.ke.run(10)
    assert new_wm.ke.fired == cf.fired

# def test_explain_2():
#     from apprentice.explain.explanation import Explanation
#
#     engine = AdditionEngine()
#
#     engine.reset()
#
#     f1 = Fact(id='JCommTable.R0C0', value='1', contentEditable=False)
#     f2 = Fact(id='JCommTable.R1C0', value='2', contentEditable=False)
#     f3 = Fact(id='JCommTable.R1C1', contentEditable=True, value='')
#
#     engine.declare(f1)
#     engine.declare(f2)
#     engine.declare(f3)
#     engine.run(10)
#     sais = engine.sais
#
#     ex = Explanation(engine.sais[0])
#     nr = ex.new_rule
#     new_wm = ExpertaWorkingMemory(KnowledgeEngine())
#
#     # generate a new rule and assign it to the blank working memory
#
#     new_wm.add_rule(nr)
#     new_wm.ke.declare(f1)
#     new_wm.ke.declare(f2)
#     new_wm.ke.declare(f3)
#     # test that the new rule fires correctly
#     new_wm.ke.run(10)
#

