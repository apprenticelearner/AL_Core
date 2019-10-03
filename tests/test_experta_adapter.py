# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:23:40 2019

@author: robert.sheline
"""

import experta as ex
from apprentice.working_memory.adapters.experta_ import \
    ExpertaConditionFactory, ExpertaSkillFactory, ExpertaActivationFactory, \
    ExpertaWorkingMemory
from apprentice.working_memory.representation import Skill, Activation


def get_KE_fixture():
    class KE(ex.KnowledgeEngine):
        activated = False

        def declare_fact(self, _a=1, _b=2):
            self.declare(ex.Fact(a=_a))
            self.declare(ex.Fact(b=_b))
            return self

        @ex.Rule(
            ex.Fact(a=1),
            ex.Fact(b=2)
        )
        def _(self):
            self.activated = True

    return KE()


def test_experta_skill_factory_transforms():
    ke = get_KE_fixture()
    a = ke._
    b = ExpertaSkillFactory(ke).from_ex_rule(a)
    assert isinstance(b, Skill)
    c = ExpertaSkillFactory(ke).to_ex_rule(b)
    assert isinstance(c, ex.Rule)
    assert a == c


def test_experta_activation_factory_transforms():
    ke = get_KE_fixture()
    a = ke.declare_fact().step()
    b = ExpertaActivationFactory(ke).from_ex_activation(a)
    assert isinstance(b, Activation)
    c = ExpertaActivationFactory(ke).to_ex_activation(b)
    assert isinstance(c, ex.activation.Activation)
    assert a == c


def test_experta_condition_factory_transforms():
    ke = get_KE_fixture()
    a = ke._._args
    b = ExpertaConditionFactory().from_ex_condition(a)
    c = ExpertaConditionFactory().to_ex_condition(b)
    assert a == c


def test_experta_add_fact():
    """
    .. todo::
        - Add more complicated test of nested dicts/facts
    """
    wm = ExpertaWorkingMemory(get_KE_fixture())
    ke2 = get_KE_fixture()
    ke2.declare(ex.Fact(a=1))
    ke2.declare(ex.Fact(b=2))
    wm.add_facts([{'a': 1}, {'b': 2}])
    assert list(wm.facts) == list(ke2.facts.values())


def test_experta_skill_constructor():
    ke = get_KE_fixture()
    factory = ExpertaSkillFactory(ke)
    skill = factory.build(_condition=(ex.Fact(a=1), ex.Fact(b=2)),
                          _function=ke._._wrapped, _name="_")
    assert isinstance(skill, Skill)
    r = factory.to_ex_rule(skill)
    assert r == ke._


def test_experta_add_skills():
    wm = ExpertaWorkingMemory(get_KE_fixture())
    ke2 = get_KE_fixture()
    wm.add_facts([{'a': 1}, {'b': 2}])
    s = wm.skill_factory.from_ex_rule(ke2._)
    wm.add_skill(s)
    assert isinstance(next(wm.skills), Skill)


def test_experta_run():
    wm = ExpertaWorkingMemory(get_KE_fixture())
    ke2 = get_KE_fixture()
    s = wm.skill_factory.from_ex_rule(ke2._)
    wm.add_skill(s)
    assert not wm.ke.activated
    wm.add_facts([{'a': 1}, {'b': 2}])
    wm.run()
    assert wm.ke.activated


def test_experta_run_2():
    wm = ExpertaWorkingMemory()
    wm.ke.activated = False
    s = wm.skill_factory.build(({'a': 1}, {'b': 2}),
                               lambda self: setattr(self, 'activated', True),
                               "_")
    wm.add_skill(s)
    assert not wm.ke.activated
    wm.add_facts([{'a': 1}, {'b': 2}])
    wm.run()
    assert wm.ke.activated


if __name__ == "__main__":
    pass
