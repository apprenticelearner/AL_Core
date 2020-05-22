import apprentice.explain.inspect_patch as inspect
from experta import Rule, W, TEST, Fact, NOT
from experta.unification import unify

from .util import join, parse, rename_lambda, rename


class Explanation:
    def __init__(self, fact, abstract=False):
        if abstract:
            raise NotImplementedError

        self.rules = []
        self.conditions = []
        self.tests = []  # lambdas

        self.general = self.explain_fact(fact)
        # exclude empty variable bindings, i.e. W()
        self.general = {k: v for k, v in self.general.items() if
                        v.__bind__ is not None}
        # extract variable names from W() objects
        self.mapping = {k.__bind__: v.__bind__ for k, v in
                        self.general.items()}

        self.conditions = [f.copy(sub=self.general, bind=True) for f in
                           self.conditions]
        self.tests = [TEST(rename_lambda(t[0], self.mapping)) for t in
                      self.tests]
        self.rules = self.rules[::-1]  # order rules logically

        print("!!!EXPLANATION RULES ", self.rules)
        self.new_rule = self.compose()

    def compose(self):
        """ compose a new rule based on explanation"""

        rules = list(filter(lambda x: type(x) is Rule, self.rules))
        if len(rules) == 1:
            return None  # exlanation comes from one rule

        asts, globs = [], {}
        for r in rules:
            p = parse(r)
            if p is not None:
                asts.append(p[0])
                globs.update(p[1])

        if len(asts) == 0:
            return None  # none of the functions had an effect, i.e. empty
            # bodies

        # rename asts with the general mapping created by the explanation
        # process
        asts = [rename(self.mapping, tree) for tree in asts]

        # construct signature from bound condition values
        sig = []
        for c in self.conditions:
            if c.__bind__ is not None:
                x = c.__bind__
                sig.append(c.__bind__)
            for att in c.values():
                if type(att) is W:
                    if att.__bind__ is not None:
                        sig.append(att.__bind__)

        # compose new function body and signature
        new_name = '_'.join([r._wrapped.__name__ for r in self.rules])
        func = join(new_name, ['self'] + list(set(sig)), globs, *asts)

        # add lambdas to conditions and generate a new rule object
        self.conditions.extend(self.tests)
        r = Rule(*self.conditions)

        # programmatically decorate composed function with Rule
        r.__call__(func)
        return r

    def get_identifier_form(self, s):
        """
        :param s: string corresponding to source declaration of function
        argument
        :return: parsed value if it is primitive (numeric or boolean),
        W(..) if identifier
        ex: arg1 => W('arg1'), 'False' => False, '"sss"' => 'sss'
        """
        if s == 'False':
            return False
        if s == 'True':
            return True
        try:
            r = float(s)
            return r
        except:
            pass
        try:
            r = int(s)
            return r
        except:
            pass
        try:
            if s[0] == s[-1] == '\'':
                return s[1:-1]
            if s[0] == s[-1] == '\"':
                return s[1:-1]
        except:
            pass
        return W(s)

    def get_rule_binding(self, rule):
        """ given an experta rule, returns  """
        s = inspect.getsource(rule._wrapped)
        fs = ''
        parens = 0
        for l in s.split('\n'):
            self_declare = 'self.declare'
            if self_declare in l:
                parens = 1
                l = l[l.find(self_declare) + len(self_declare):]
            if parens > 0:
                parens += l.count('(')
                parens -= l.count(')')
                fs += l

        binding = {}
        fs = fs[1:-1]  # strip off self.declare parens
        sig = fs[fs.find('(') + 1:-1]
        for i, arg in enumerate(sig.split(',')):
            if '=' in arg:
                i = arg.split("=")[0].strip()
                arg = arg.split("=")[1]

            binding[i] = self.get_identifier_form(arg.strip())

            # todo: kwargs
        binding['__class__'] = fs[:fs.find('(')]
        return binding

    def get_condition_binding(self, condition):
        if type(condition) is NOT:
            condition = NOT[0]
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
                if isinstance(antecedent_fact, Fact):
                    # antecedent fact came from another rule, so collect
                    # e1s for unifying variables in backchain

                    if antecedent_fact.__source__ is not None:
                        e1s.append((antecedent_fact, self.get_rule_binding(
                            antecedent_fact.__source__.rule)))
                    # antecedent fact is a terminal, so the condition of this
                    # fact is a boundary condition

            # LHS of current rule, i.e. RHS of explanation tuple
            r = fact.__source__.rule

            for conj in fact.__source__.rule._args:
                if type(conj) is NOT:
                    continue  # ignore for now. TODO: improve this

                if type(conj) is TEST:
                    self.tests.append(conj)
                    continue

                matched = False
                e2 = self.get_condition_binding(conj)

                for e1 in e1s:
                    # precheck that fact type matches
                    if e1[1]['__class__'] == e2['__class__']:
                        # ensure they unify, i.e. they are corresponding
                        # facts and conditions
                        u = unify(e1[1], e2, general)
                        if u is not None:
                            matched = True
                            new_g = self.explain_fact(fact=e1[0],
                                                      general=general.update(
                                                          u),
                                                      root=False)
                            if new_g is not None:
                                general.update(new_g)

                            e1s.remove(e1)
                            break

                if not matched:
                    # boundary condition
                    self.conditions.append(conj)

        return general
