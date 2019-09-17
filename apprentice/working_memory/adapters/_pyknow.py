import pyknow as pk
from typing import Iterable
from ..base import WorkingMemory
from ..representation import Fact, Skill

Fact.register(pk.fact.Fact)


class PyknowWorkingMemory(WorkingMemory):
    def __init__(self):
        self.ke = pk.engine.KnowledgeEngine()

    def get_facts(self) -> Iterable[Fact]:
        pass

    def get_skills(self) -> Iterable[Skill]:
        pass

    def add_fact(self, fact: Fact):
        pass

    def add_skill(self, skill: Skill):
        pass

    def update_fact(self, fact: Fact):
        pass

    def update_skill(self, skill: Skill):
        pass
