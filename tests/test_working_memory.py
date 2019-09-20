from experta import Fact
from experta import KnowledgeEngine
from experta import MATCH
from experta import Field
from experta import TEST
from experta import Rule
from experta import DefFacts
import pytest

from apprentice.working_memory.adapters._pyknow import PyknowWorkingMemory


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

@pytest.mark.xfail
def test_ke_1():

    pkke = FibonacciCalculator()
    pkke.reset(target=42)
    pkke.run()

    apprentice_ke = PyknowWorkingMemory()

    for f in pkke.facts:
        apprentice_ke.add_fact(f)
    for r in pkke.get_rules():
        apprentice_ke.add_skill(r)

    apprentice_ke.run()


if __name__ == "__main__":
    test_ke_1()
