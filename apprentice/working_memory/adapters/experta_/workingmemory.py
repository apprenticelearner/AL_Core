import experta as ex
import jsondiff
from apprentice.working_memory.base import WorkingMemory
from apprentice.working_memory.representation import Skill

from .factory import ExpertaSkillFactory, ExpertaConditionFactory, \
    ExpertaActivationFactory


class ExpertaWorkingMemory(WorkingMemory):
    def __init__(self, ke=None, reset=True):
        if ke is None:
            ke = ex.engine.KnowledgeEngine()
        self.ke = ke
        if reset:
            self.ke.reset()

        self.skill_factory = ExpertaSkillFactory(ke)
        self.activation_factory = ExpertaActivationFactory(ke)
        self.condition_factory = ExpertaConditionFactory()

    def update(self, diff):
        for k, v in diff.items():
            if k is jsondiff.symbols.delete:
                self.retract(ex.Fact(**v))
            if k is jsondiff.symbols.add:
                self.declare(ex.Fact(**v))

    def output(self):
        raise NotImplementedError

    @property
    def facts(self):
        return [f.as_dict() for f in self.ke.facts.values()]

    def add_fact(self, fact: dict):
        f = ex.Fact(**fact)
        self.ke.declare(f)

    def update_fact(self, fact: dict):
        self.ke.modify(ex.Fact(**fact))

    @property
    def skills(self):
        for rule in self.ke.get_rules():
            yield self.skill_factory.from_ex_rule(rule)

    def add_skill(self, skill: Skill):
        rule = self.skill_factory.to_ex_rule(skill)
        setattr(self.ke, rule._wrapped.__name__, rule)
        rule.ke = self.ke
        rule._wrapped_self = self.ke
        self.ke.matcher.__init__(self.ke)

    def update_skill(self, skill: Skill):
        self.add_skill(skill)

    @property
    def activations(self):
        for a in self.activations:
            yield self.activation_factory.from_ex_activation(a)

    def run(self):
        self.ke.run()
