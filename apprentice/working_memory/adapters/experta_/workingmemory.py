import experta as ex
from apprentice.working_memory.base import WorkingMemory
from apprentice.working_memory.representation import Skill

from .factory import (
    ExpertaSkillFactory,
    ExpertaConditionFactory,
    ExpertaActivationFactory,
)


class ExpertaWorkingMemory(WorkingMemory):
    def __init__(self, ke=None, reset=True):
        if ke is None:
            ke = ex.engine.KnowledgeEngine()
        self.ke = ke
        if reset:
            self.ke.reset()
        # self.lookup = {}
        self.skill_factory = ExpertaSkillFactory(ke)
        self.activation_factory = ExpertaActivationFactory(ke)
        self.condition_factory = ExpertaConditionFactory()

    def step(self):
        self.ke.step()

    def output(self):
        raise NotImplementedError

    @property
    def facts(self):
        labeled_facts = {k: v.as_dict() for k, v in self.ke.facts.items()}
        return labeled_facts

        # f in self.ke.facts.values()]

    @property
    def state(self):
        from experta import Fact

        # return frozenset(self.get_hashable_facts())

        # state = {}
        # for i, fact in enumerate(self.ke.facts.values()):
        #    for feature_key, feature_value in fact.as_dict().items():
        #        if Fact.is_special(feature_key):
        #            continue
        #        state['{0}_{1}'.format(str(feature_key), str(i))] = feature_value
        factlist = []
        for fact in self.ke.facts.values():
            f = {}
            for k, v in fact.as_dict().items():
                if not Fact.is_special(k):
                    f[k] = v
            factlist.append(f)

        # state = {'<f-{}>'.format(i): f for i,f in enumerate(sorted(factlist))}
        state = {}
        for i, fact in enumerate(sorted(factlist, key=lambda d: sorted(d.items()))):
            for k, v in fact.items():
                state["{0}_{1}".format(str(k), str(i))] = v

        return state

    def add_fact(self, fact: dict):
        f = ex.Fact(**fact)
        # key = hash(f)
        self.ke.declare(f)
        # todo: integrate lookup into experta factlist
        # self.lookup[key] = f

    def remove_fact(self, fact: dict = None, key: str = None):
        # if key is not None:
        # fact = self.lookup[key]

        f = ex.Fact(**fact)

        self.ke.retract(f)
        # del self.lookup[key]

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
            # for a in self.ke.get_activations()[0]:
            yield self.activation_factory.from_ex_activation(a)

    def run(self):
        self.ke.run()
