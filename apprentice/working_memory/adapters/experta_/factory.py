import uuid
from typing import Any
from typing import Callable

import experta as ex

from apprentice.working_memory.representation import Skill, Activation

class Factory:
    pass


class ExpertaFactFactory(Factory):
    def build(self, _dict: dict) -> ex.Fact:
        return ex.Fact(**dict)

    def from_ex_fact(self, _fact: ex.Fact) -> dict:
        return _fact.as_dict()

    def to_ex_fact(self, _dict: dict) -> ex.Fact:
        return self.build(_dict)


class ExpertaConditionFactory(Factory):
    def build(self, _tuple: tuple) -> Any:
        return _tuple

    def validate(self):
        raise NotImplementedError

    def from_ex_condition(self,
                          ex_condition:
                          tuple) -> \
            Any:

        def c2r(c):
            if type(c) is tuple:
                return tuple(c2r(_) for _ in c)
            if isinstance(c, tuple):
                r = tuple(c2r(_) for _ in c)
                # print('==>', c.__class__, ' with args ', r, flush=True)
                return c.__class__(*r)
            if isinstance(c, ex.Fact):
                return c.as_dict()
            if callable(c):
                return c
            assert False

        return c2r(ex_condition)

    def to_ex_condition(self,
                        ex_rule:
                        ex.Rule) -> \
            Any:
        def r2c(c):
            if type(c) is tuple:
                return tuple(r2c(_) for _ in c)
            if isinstance(c, tuple):
                r = tuple(r2c(_) for _ in c)
                # print('==>', c.__class__, ' with args ', r, flush=True)
                return c.__class__(*r)
            if isinstance(c, dict):
                return ex.Fact(**c)
            if callable(c):
                return c
            assert False

        return r2c(ex_rule)


class ExpertaSkillFactory(Factory):
    def __init__(self, _ke: ex.KnowledgeEngine):
        self._ke = _ke
        self.condition_factory = ExpertaConditionFactory()

    def build(self, _condition: Any,
              _function: Callable,
              _name: str = None) -> Skill:
        #if _name is None:
            #_name = 'skill_' + str(uuid.uuid1())

        s = Skill(_condition, _function)#, _name)
        s._ke = self._ke
        return s

    def from_ex_rule(self, _rule: ex.Rule) -> Skill:
        return self.build(_rule._args,
                          _rule._wrapped,
                          _name=_rule._wrapped.__name__)

    def to_ex_rule(self, _skill: Skill) -> ex.Rule():
        cond = self.condition_factory.to_ex_condition(_skill.conditions)
        rule = ex.Rule.__new__(ex.Rule, *cond)(_skill.function_)
        rule.ke = self._ke
        rule._wrapped_self = self._ke
        return rule


class ExpertaActivationFactory(Factory):
    def __init__(self, _ke: ex.KnowledgeEngine):
        self._ke = _ke
        self.skill_factory = ExpertaSkillFactory(_ke)

    def build(self, _skill: Skill, _context: dict):
        return Activation(_skill, _context)

    def from_ex_activation(self,
                           ex_activation: ex.activation.Activation) -> \
            Activation:
        return self.build(
            _skill=self.skill_factory.from_ex_rule(
                ex_activation.rule),
            _context=ex_activation.context)

    def to_ex_activation(self,
                         _activation: Activation) -> ex.activation.Activation:
        return ex.activation.Activation(
            self.skill_factory.to_ex_rule(_activation.skill),
            set(_activation.context.values()), _activation.context)

    def fire(self):
        self.base.fire()