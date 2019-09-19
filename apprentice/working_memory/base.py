from typing import Iterable
from abc import ABCMeta, abstractmethod

from .representation import Fact
from .representation import Skill


class WorkingMemory(metaclass=ABCMeta):
    @abstractmethod
    def get_facts(self) -> Iterable[Fact]:
        pass

    @abstractmethod
    def get_skills(self) -> Iterable[Skill]:
        pass

    @abstractmethod
    def add_fact(self, fact: Fact):
        pass

    @abstractmethod
    def add_skill(self, skill: Skill):
        pass

    @abstractmethod
    def update_fact(self, fact: Fact):
        self.ke.modify(fact)

    @abstractmethod
    def update_skill(self, skill: Skill):
        pass
