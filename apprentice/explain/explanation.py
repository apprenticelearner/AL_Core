import apprentice.explain.inspect_patch as inspect
from experta import Rule, W, TEST
from experta.unification import unify

from .util import join, rename, parse, rename_lambda


class Explanation:
    def __init__(self, fact, abstract=False):
        if abstract:
            raise NotImplementedError

        self.rules = []
        self.conditions = []
        self.tests = []  # lambdas
        self.general = self.explain_fact(fact)

        # extract variable names from W() objects
        self.mapping = {k.__bind__: v.__bind__ for k, v in
                        self.general.items()}
        self.conditions = [f.copy(sub=self.general) for f in self.conditions]
        self.tests = [TEST(rename_lambda(t[0], self.mapping)) for t in
                      self.tests]
        self.rules = self.rules[::-1]  # order rules logically
        self.new_rule = self.compose()


    def compose(self):
        asts = parse(*self.rules)

        if len(asts) == 0:
            return False  # none of the functions had an effect

        asts = [rename(self.mapping, tree) for tree in asts]

        # construct signature from bound condition values
        sig = []
        for c in self.conditions:
            for att in c.values():
                if type(att) is W:
                    sig.append(att.__bind__)

        # compose new function body and signature
        new_name = '_'.join([r._wrapped.__name__ for r in self.rules])
        func = join(new_name, ['self'] + list(set(sig)), *asts)

        # construct new function with
        self.conditions.extend(self.tests)
        r = Rule(*self.conditions)

        # programmatically decorate func with Rule r
        r.__call__(func)
        return r

    def get_rule_binding(self, rule):
        """ given an experta rule, returns """
        s = inspect.getsource(rule._wrapped)

        for l in s.split('\n'):
            if 'self.declare' in l:
                fs = l[l.find('(') + 1:-1]
        binding = {}

        sig = fs[fs.find('(') + 1:-1]
        for i, arg in enumerate(sig.split(',')):
            binding[i] = W(arg.strip())
            # todo: kwargs
        binding['__class__'] = fs[:fs.find('(')]
        return binding  # tuple(binding.values())

    def get_condition_binding(self, condition):
        binding = dict(condition)
        binding['__class__'] = condition.__class__.__name__
        return binding  # [condition.__class__.__name__] + list(
        # binding.values())

    def explain_fact_abstract(self, fact):
        raise NotImplementedError

    def explain_fact(self, fact, general={}, root=True):
        # because we are only building general substitution, only consider
        # facts who came from an activation i.e. rule
        if fact.__source__ is not None:

            # keep track of rules that need to be fired for composition
            self.rules.append(fact.__source__.rule)
            e1s = []
            for antecedent_fact in fact.__source__.facts:

                # antecedent fact came from another rule, so the backchain
                # must continue
                if antecedent_fact.__source__ is not None:
                    e1s.append((antecedent_fact,
                                tuple(self.get_rule_binding(
                                    antecedent_fact.__source__.rule).values(

                                ))))
                # antecedent fact is a terminal, so the condition of this
                # fact is the boundary condition
                else:
                    for conj in fact.__source__.rule._args:
                        if type(conj) is not TEST:
                            self.conditions.append(conj)
                    # self.conditions.extend(fact.__source__.rule._args)

            # LHS of current rule, i.e. RHS of explanation tuple

            for conj in fact.__source__.rule._args:
                if type(conj) is TEST:
                    self.tests.append(conj)
                    continue

                e2 = tuple(self.get_condition_binding(conj).values())

                for e1 in e1s:
                    # precheck that fact type matches
                    if e1[1][-1] == e2[-1]:
                        # ensure they unify, i.e. they are corresponding
                        # facts and conditions

                        u = unify(e1[1], e2, general)
                        if u is not None:
                            new_g = self.explain_fact(fact=e1[0],
                                                      general=general.update(
                                                          u),
                                                      root=False)
                            if new_g is not None:
                                general.update(new_g)

                            e1s.remove(e1)
                            break

        return general
