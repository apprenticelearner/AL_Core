from typing import Iterable

import experta as pk

from ..base import WorkingMemory
from ..representation import Fact, Skill

# register referenced type as abstract base class for type checking
Fact.register(pk.fact.Fact)


class PyknowWorkingMemory(WorkingMemory):
    def __init__(self):
        self.ke = pk.engine.KnowledgeEngine()

    def get_facts(self) -> Iterable[Fact]:
        return self.ke.facts

    def get_skills(self) -> Iterable[Skill]:
        return self.ke.get_rules()

    def add_fact(self, fact: Fact):
        self.ke.declare(fact)

    def add_skill(self, skill: Skill):
        self.ke.setattr(skill._wrapped.__name__, skill)

    def update_fact(self, fact: Fact):
        self.ke.modify(fact)

    def update_skill(self, skill: Skill):
        pass

    #####
