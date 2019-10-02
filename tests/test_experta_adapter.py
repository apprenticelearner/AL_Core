from experta import Fact
from experta import KnowledgeEngine
from experta import MATCH
from experta import Field
from experta import TEST
from experta import Rule
from experta import DefFacts
from experta import EXISTS
import pytest

from apprentice.working_memory import ExpertaWorkingMemory
from apprentice.working_memory.representation import ExpertaSkill, ExpertaCondition

def test_change_rule():
    executed = list()
    first = object()

    class KE(KnowledgeEngine):
        @Rule(EXISTS(Fact(color='green')))
        def _(self):
            executed.append(True)

    ke = KE()
    ke.reset()

    # print(ke.green_light)
    rule = ke._
    ke.declare(Fact(color='green'))

    # print(ke.get_rules())
    ke.run()
    assert executed == [True]

    skill = ExpertaSkill(rule._wrapped.__name__, ExpertaCondition(rule),
                         rule._wrapped)
    # print(skill)
    converted_skill = skill.to_experta()

    setattr(ke, '_', None)
    setattr(ke, '_', skill.to_experta())
    ke.matcher.build_network()
    ke.reset()

    ke.declare(Fact(color='green'))
    ke.run()

    assert executed == [True, True]
