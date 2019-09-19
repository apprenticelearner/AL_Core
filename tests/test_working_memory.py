from pyknow import Fact, KnowledgeEngine, MATCH, Field, TEST, Rule, DefFacts


class FibonacciDigit(Fact):
    position = Field(int, mandatory=True)
    value = Field(int, mandatory=True)


class FibonacciCalculator(KnowledgeEngine):

    @DefFacts()
    def set_target_position(self, target):
        yield Fact(target_position=target)

    @DefFacts()
    def init_sequence(self):
        yield FibonacciDigit(position=1, value=1)
        yield FibonacciDigit(position=2, value=1)

    @Rule(
        FibonacciDigit(
            position=MATCH.p1,
            value=MATCH.v1),
        FibonacciDigit(
            position=MATCH.p2,
            value=MATCH.v2),
        TEST(
            lambda p1, p2: p2 == p1 + 1),
        Fact(
            target_position=MATCH.t),
        TEST(
            lambda p2, t: p2 < t))
    def compute_next(self, p2, v1, v2):
        next_digit = FibonacciDigit(
            position=p2 + 1,
            value=v1 + v2)

        self.declare(next_digit)

    @Rule(
        Fact(
            target_position=MATCH.t),
        FibonacciDigit(
            position=MATCH.t,
            value=MATCH.v))
    def print_last(self, t, v):
        print("Fibonnaci digit in position {position} is {value}".format(
            position=t, value=v))

def test_ke_1(self):


    pkke = FibonacciCalculator()
    pkke.reset(target=42)
    pkke.run()
    #pkke.declare(FibonacciDigit(position=1, value=1))
    ke.run()
print(ke.facts[0].as_dict())

rules = ke.get_rules()
print(rules[0]._wrapped.__name__)