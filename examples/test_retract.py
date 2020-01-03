from experta import KnowledgeEngine
from experta import MATCH
from experta import Field
from experta import Rule
from experta import Fact


class Number(Fact):
    """ Holds a number """
    value = Field(int, mandatory=True)


class KB(KnowledgeEngine):

    @Rule(Number(value=MATCH.x))
    def increment(self, x):
        if x >= 3:
            return
        y = x + 1
        self.declare(Number(value=y))


if __name__ == "__main__":

    engine = KB()
    engine.reset()
    initial_f = Number(value=0)
    engine.declare(initial_f)
    engine.run()
    print(engine.facts)

    engine.retract(initial_f)
    engine.step()
    print(engine.facts)
