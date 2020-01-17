import uuid
from dataclasses import dataclass
from typing import Collection, Callable, Any
import inspect
import experta

from experta.conditionalelement import ConditionalElement as Condition
from experta import Fact


# class Condition(tuple):
#     def __new__(cls, *args):
#         return tuple.__new__(Condition, args)


# class Fact(dict):
#     def __new__(cls, *args):
#         return dict.__new__(Fact, args)


@dataclass#(frozen=True) #
class Sai:
    selection: Any
    action: Any
    input: Any

    def __post_init__(self):
        try:
            activation_frame = inspect.currentframe().f_back.f_back.f_back.f_back
            assert type(activation_frame.f_locals[
                            'self']) == experta.activation.Activation

            self.__source__ = activation_frame.f_locals['self']
            print(activation_frame.f_locals['self'])
        except AssertionError:
            pass
        except AttributeError:
            pass


@dataclass(frozen=True)
class Skill:
    conditions: Collection[Condition]
    function_: Callable
    # name: str = "skill_" + str(uuid.uuid1())


@dataclass(frozen=True)
class Activation:
    skill: Skill
    context: dict

    @property
    def fire(self) -> Any:
        raise NotImplementedError

    def __hash__(self):
        return hash(self.as_hash_repr())

    #     from experta import Fact
    #     c = {}
    #     for k,v in self.context.items():
    #         if isinstance(v, Fact):
    #             c[k] = v.as_frozenset()
    #         else:
    #             c[k] = v

    #     return hash((self.skill, frozenset(c)))

    def get_rule_name(self):
        return self.skill.function_.__name__

    def get_rule_bindings(self):
        bindings = {}

        facts = sorted([(k, v) for k, v in self.context.items() if
                        isinstance(v, Fact)])
        facts = [v for k, v in facts]

        for i, v in enumerate(facts):
            for fk, fv in v.items():
                if Fact.is_special(fk):
                    continue
                bindings['fact-%i: %s' % (i, fk)] = fv

        # print(bindings)
        return bindings

    def as_hash_repr(self):
        c = {}
        for k, v in self.context.items():
            if isinstance(v, Fact):
                c[k] = frozenset([(fk, fv) for fk, fv in v.items() if not
                                  Fact.is_special(fk)])
            else:
                c[k] = v

        return self.skill, frozenset(c.items())
