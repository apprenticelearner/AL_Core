from typing import Collection
from abc import ABC, abstractmethod

from apprentice.agents.working_memory import AbstractSkill, AbstractFact


class AbstractWorkingMemory:
    """
    Abstract base class for working memory
    """
    @property
    @abstractmethod
    def facts(self) -> Collection[AbstractFact]:
        """
        :return: collection of facts in working memory
        """
        pass

    @property
    @abstractmethod
    def skills(self) -> Collection[AbstractSkill]:
        """
        :return: collection of skills currently in working memory
        """
        pass

    @abstractmethod
    def add_skill(self, skill: AbstractSkill) -> None:
        """
        Add a skill to working memory
        :param skill: skill to be added
        :return: None
        """
        pass

    @property
    @abstractmethod
    def where_component(self):
        pass

    @property
    @abstractmethod
    def when_component(self):
        pass
