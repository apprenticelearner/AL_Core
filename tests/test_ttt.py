from pyknow import Fact, KnowledgeEngine, MATCH, Field, TEST, Rule, DefFacts

class Mark(Fact):
    row = Field(int, mandatory=True)
    col = Field(int, mandatory=True)
    player = Field(str, mandatory=True)

class ActivePlayer(Fact):
    player = Field(str, mandatory=True)

class TicTacToe(KnowledgeEngine):

    @DefFacts()
    def set_initial_player(self, player="X"):
        yield ActivePlayer(player)

    @DefFacts()
    def init_sequence(self):
        pass

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

    def end_state(self, t, v):
        print("Winner detected"