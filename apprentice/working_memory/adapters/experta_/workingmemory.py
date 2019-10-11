import experta as ex
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
        #self.lookup = {}
        self.skill_factory = ExpertaSkillFactory(ke)
        self.activation_factory = ExpertaActivationFactory(ke)
        self.condition_factory = ExpertaConditionFactory()

    def step(self):
        self.ke.step()

    def output(self):
        raise NotImplementedError

    @property
    def facts(self):
        d= {k: v.as_dict() for k, v in self.ke.facts.items()}
        return d
        # f in self.ke.facts.values()]

    def add_fact(self, fact: dict):
        f = ex.Fact(**fact)
        #key = hash(f)
        self.ke.declare(f)
        # todo: integrate lookup into experta factlist
        #self.lookup[key] = f

    def remove_fact(self, fact: dict = None, key: str = None):
        #if key is not None:
            #fact = self.lookup[key]

        f = ex.Fact(**fact)

        self.ke.retract(f)
        #del self.lookup[key]

    def update_fact(self, fact: dict):
        raise NotImplementedError
        # todo: what is this for
        f = ex.Fact(**fact)
        self.ke.modify()

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
        for a in self.ke.agenda.activations:
        #for a in self.ke.get_activations()[0]:
            yield self.activation_factory.from_ex_activation(a)

    def run(self):
        self.ke.run()
