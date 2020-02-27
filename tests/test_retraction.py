from apprentice.explain.explanation import Explanation
from apprentice.working_memory.skills_test import AdditionEngine, EmptyAdditionEngine
from apprentice.working_memory.representation import Sai
from apprentice.working_memory import ExpertaWorkingMemory
import inspect

from experta import Fact, KnowledgeEngine, Rule


class RetractionEngine(KnowledgeEngine):
    @Rule(
        Fact('A_parent')
    )
    def rule_A(self):
        self.declare(Fact('A_child'))

    @Rule(
        Fact('B_parent')
    )
    def rule_B(self):
        self.declare(Fact('B_child'))

    @Rule(
        Fact('A_child'),
        Fact('B_child')
    )
    def rule_AB(self):
        self.declare(Fact('AB_child'))

    @Rule(
        Fact('A_child'),
        Fact('AB_child')
    )
    def rule_AAB(self):
        self.declare(Fact('AAB_child'))

    @Rule(
        Fact('B_child'),
        Fact('AB_child')
    )
    def rule_ABB(self):
        self.declare(Fact('ABB_child'))



def test_retraction():
    engine = RetractionEngine()
    engine.reset()

    a = Fact('A_parent')
    b = Fact('B_parent')
    engine.declare(a)
    engine.declare(b)

    engine.run(5)
    assert len(engine.facts) == 8

    engine.retract(b)
    assert len(engine.facts) == 3
