import uuid
from dataclasses import dataclass
from typing import Collection, Callable, Any

from experta.conditionalelement import ConditionalElement as Condition


# class Condition(tuple):
#     def __new__(cls, *args):
#         return tuple.__new__(Condition, args)


# class Fact(dict):
#     def __new__(cls, *args):
#         return dict.__new__(Fact, args)

@dataclass(frozen=True)
class Sai:
    selection: Any
    action: Any
    input: Any


@dataclass(frozen=True)
class Skill:
    conditions: Collection[Condition]
    function_: Callable
    #name: str = "skill_" + str(uuid.uuid1())


@dataclass(frozen=True)
class Activation:
    skill: Skill
    context: dict

    @property
    def fire(self) -> Any:
        raise NotImplementedError

    def __hash__(self):
        return hash((self.skill, tuple(sorted(self.context))))
