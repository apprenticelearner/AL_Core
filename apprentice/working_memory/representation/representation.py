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

    def _hash__(self):
        return hash(self.as_hash_repr())
   #     from experta import Fact
   #     c = {}
   #     for k,v in self.context.items():
   #         if isinstance(v, Fact):
   #             c[k] = v.as_frozenset()
   #         else:
   #             c[k] = v

   #     return hash((self.skill, frozenset(c)))

    def as_hash_repr(self):
        from experta import Fact
        c = {}
        for k,v in self.context.items():
            if isinstance(v, Fact):
                c[k] = v.as_frozenset()
            else:
                c[k] = v

        return self.skill, frozenset(c.items())
