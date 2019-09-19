from abc import ABCMeta, abstractmethod
from typing import Collection, Callable, Dict


# Placeholder for now. What do these actually need to implement?
class Fact(metaclass=ABCMeta):
    pass


class Condition(metaclass=ABCMeta):
    pass


class Skill(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def conditions(self) -> Collection[Condition]: ...

    @abstractmethod
    def function(self, state: Dict) -> Callable: ...

