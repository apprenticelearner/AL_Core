from experta import Fact, KnowledgeEngine, Rule, MATCH, TEST, DefFacts


class Depressed(Fact):
    pass


class Hate(Fact):
    pass


class Buy(Fact):
    pass


class Possess(Fact):
    pass


class Gun(Fact):
    pass


class Weapon(Fact):
    pass


class Kill(Fact):
    pass


class KillEngine(KnowledgeEngine):
    fired = []

    @DefFacts()
    def first(self):
        yield Depressed("JOHN")
        yield Buy("JOHN", "OBJ1")
        yield Gun("OBJ1")

    @Rule(
        Hate(MATCH.a, MATCH.b), Possess(MATCH.a, MATCH.c), Weapon(MATCH.c),
        TEST(lambda a, b, c: True))
    def kill_rule(self, a, b):
        self.fired.append(('kill_rule', a, b))
        self.declare(Kill(a, b))

    @Rule(
        Depressed(MATCH.w),
        TEST(lambda w: w == "JOHN"))
    def hate_rule(self, w):
        self.fired.append(('hate_rule', w))
        self.declare(Hate(w, w))

    @Rule(
        Buy(MATCH.u, MATCH.v),
        TEST(lambda u: u == "JOHN"),
        TEST(lambda v: v == "OBJ1"))
    def possess_rule(self, u, v):
        self.fired.append(('possess_rule', u, v))
        self.declare(Possess(u, v))

    @Rule(
        Gun(MATCH.z),
        TEST(lambda z: True))
    def weapon_rule(self, z):
        self.fired.append(('weapon_rule', z))
        self.declare(Weapon(z))


class KillEngineEmpty(KnowledgeEngine):
    fired = []

    @DefFacts()
    def first(self):
        yield Depressed("JOHN")
        yield Buy("JOHN", "OBJ1")
        yield Gun("OBJ1")
