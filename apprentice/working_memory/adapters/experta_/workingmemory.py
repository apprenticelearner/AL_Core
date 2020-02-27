import jsondiff
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
        self.skill_factory = ExpertaSkillFactory(ke)
        self.activation_factory = ExpertaActivationFactory(ke)
        self.condition_factory = ExpertaConditionFactory()
        super().__init__()

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
        factlist = []
        for fact in self.ke.facts.values():
            f = {}
            for k, v in fact.as_dict().items():
                if ex.Fact.is_special(k):
                    continue
                if isinstance(v, bool):
                    f[k] = str(v)
                else:
                    f[k] = v
            factlist.append(f)

        # from pprint import pprint
        # pprint(factlist)

        state = {}
        # for fact in factlist:
        #     state[tuple(sorted("%s=%s" % (k, v)
        #                        for k, v in fact.items()))] = True

        for fact in factlist:
            if 'id' not in fact:
                continue
            for k, v in fact.items():
                if k == 'id':
                    continue
                state[(k, fact['id'])] = v

        # for i, fact in enumerate(sorted(factlist, key=lambda d:
        # sorted(d.items()))):
        #     for k, v in fact.items():
        #         state["{0}_{1}".format(str(k), str(i))] = v

        return state

    def add_fact(self, key: object, fact: dict) -> None:
        f = ex.Fact(**fact)
        self.ke.declare(f)
        self.lookup[key] = f

    def remove_fact(self, key: object) -> bool:
        if key not in self.lookup:
            return False
        f = self.lookup[key]
        self.ke.retract(f)
        del self.lookup[key]

    def update_fact(self, key: object, diff: dict) -> None:
        old_fact = self.lookup[key]
        new_fact = apply_diff_to_fact(old_fact, diff)
        self.ke.retract(old_fact)
        self.ke.declare(new_fact)
        self.lookup[key] = new_fact

    @property
    def skills(self):
        for rule in self.ke.get_rules():
            yield self.skill_factory.from_ex_rule(rule)

    def add_skill(self, skill: Skill):
        rule = self.skill_factory.to_ex_rule(skill)
        self.add_rule(rule)

    def add_rule(self, rule: ex.Rule):
        setattr(self.ke, rule._wrapped.__name__, rule)
        rule.ke = self.ke
        rule._wrapped_self = self.ke
        self.ke.matcher.__init__(self.ke)
        self.ke.reset() #todo: not sure if this is necessary

    def update_skill(self, skill: Skill):
        self.add_skill(skill)

    @property
    def activations(self):
        for a in self.ke.agenda.activations:
            # for a in self.ke.get_activations()[0]:
            yield self.activation_factory.from_ex_activation(a)

    def run(self):
        self.ke.run()


def apply_diff_to_fact(fact: ex.Fact, diff: dict) -> ex.Fact:
    if jsondiff.symbols.replace in diff:
        return ex.Fact(**diff[jsondiff.symbols.replace])

    new_fact = {}
    for k in fact:
        if (jsondiff.symbols.delete in diff and
                k in diff[jsondiff.symbols.delete]):
            continue
        new_fact[k] = fact[k]

    for k in diff:
        if k is not jsondiff.symbols.delete:
            new_fact[k] = diff[k]

    return ex.Fact(**new_fact)
