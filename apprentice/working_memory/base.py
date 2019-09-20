from typing import Iterable
from abc import ABCMeta, abstractmethod
from typing import Collection

from apprentice.working_memory.representation import Fact, Skill

from .representation import Fact
from .representation import Skill


class WorkingMemory(metaclass=ABCMeta):
    """
    Abstract base class for working memory
    """

    @property
    @abstractmethod
    def facts(self) -> Collection[Fact]:
        """
        :return: collection of facts in working memory
        """
        pass

    @property
    @abstractmethod
    def skills(self) -> Collection[Skill]:
        """
        :return: collection of skills currently in working memory
        """
        pass

    @abstractmethod
    def add_fact(self, fact: Fact) -> None:
        """
        Add a fact to working memory
        :param fact: the fact to be added
        :return: None
        """
        pass

    @abstractmethod
    def add_skill(self, skill: Skill) -> None:
        """
        Add a skill to working memory
        :param skill: skill to be added
        :return: None
        """
        pass

    @abstractmethod
    def update_fact(self, fact: Fact) -> None:
        """
        Update a fact in working memory
        :param fact: the updated fact
        :return: None
        """
        pass

    @abstractmethod
    def update_skill(self, skill: Skill) -> None:
        """
        Update a skill in working memory
        :param skill: the updated skill
        :return: None
        """
        pass
