from apprentice.explain.explanation import Explanation
from apprentice.working_memory import ExpertaWorkingMemory

from kill_engine import KillEngine, KillEngineEmpty

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
    new_wm.ke.reset()
    new_wm.ke.run(10)
    assert new_wm.ke.fired == cf.fired


test_explain()
