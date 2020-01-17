import schema
from apprentice.working_memory.representation.representation import Sai
from experta import Fact, Field, KnowledgeEngine, AS, Rule, MATCH, NOT
from tabulate import tabulate


class Square(Fact):
    row = Field(str, mandatory=True)
    col = Field(str, mandatory=True)
    player = Field(str, mandatory=True)


class PossibleMove(Square):
    row = Field(str, mandatory=True)
    col = Field(str, mandatory=True)
    player = Field(str, mandatory=True)


class CurrentPlayer(Fact):
    name = Field(schema.Or("X", "O"), mandatory=True)


class ttt_engine(KnowledgeEngine):
    # @DefFacts()
    # def init_board(self, x=3, players=['X', 'O']):
    #     return
    #     yield CurrentPlayer(name=players[0])
    #     for row in range(3):
    #         for col in range(3):
    #             yield Square(row=row, col=col, player='')

    # @Rule(
    #     AS.square1 << Square(row=MATCH.row, col=MATCH.square1col),
    #     AS.square2 << Square(row=MATCH.row, col=MATCH.square2col),
    #     TEST(lambda square1col, square2col: square2col == square1col + 1),
    # )
    # def horizontally_adj(self, row, square1col, square2col):
    #     relation = Fact(
    #         relation="horizontally_adjacent",
    #         row=row,
    #         square1col=square1col,
    #         square2col=square2col,
    #     )
    # self.declare(relation)

    # ... other relations

    # @Rule(
    #     Fact(type="CurrentPlayer", player=MATCH.player),
    #     AS.square << Fact(type="Square", row=MATCH.row, col=MATCH.col,
    #                       player=""),
    #     NOT(Fact(type="PossibleMove", row=MATCH.row, col=MATCH.col,
    #              player=MATCH.player))
    # )
    # def suggest_move(self, row, col, player):
    #     self.declare(Fact(type="PossibleMove", row=row, col=col,
    #                       player=player))

    # @Rule(
    #     Fact(type="CurrentPlayer", player=MATCH.player),
    #     AS.square
    #     << Fact(type="PossibleMove", row=MATCH.row, col=MATCH.col,
    #             player=MATCH.player),
    # )
    # def make_move(self, row, col, player):
    #     return Sai(None, "move", {"row": row, "col": col, "player": player})

    @Rule(
           Fact(type='CurrentPlayer', player=MATCH.player),
           Fact(type='Square', row=MATCH.row, col=MATCH.col, player="")
    )
    def make_move(self, row, col, player):
        print("moving", row, col, player)
        return Sai(None, 'move', {'row': row, 'col': col, 'player': player})


class ttt_oracle:
    """
    Enviornment oracle for ttt:
    """

    def __init__(self, players=["X", "O"]):
        self.players = players
        self.current_player = "X"
        self.board = [["" for _ in range(3)] for _ in range(3)]

    def move2(self, row, col, player):
        assert self.board[row][col] == ""
        self.board[row][col] = player
        return (
            [{"__class__": Square, "row": row, "col": col, "val": player}],
            [{"__class__": Square, "row": row, "col": col, "val": ""}],
        )

    def set_state(self, state):
        for fact in state:
            if fact["__class__"].__name__ == "Square":
                self.board[fact["row"]][fact["col"]] = fact["val"]

    def as_dict(self):
        def ids():
            i = 0
            while True:
                i += 1
                yield i

        idg = ids()
        d = {next(idg): {"type": "CurrentPlayer", "player":
                         self.current_player}}
        for row in range(3):
            for col in range(3):
                d[next(idg)] = {
                    "type": "Square",
                    "row": str(row),
                    "col": str(col),
                    "player": self.board[row][col],
                }
        return d

    def __str__(self):
        table = []
        table.append(["", "Col 0", "Col 1", "Col 2"])
        for i in range(3):
            table.append(["Row %i" % i] + [s for s in self.board[i]])

        return tabulate(table, tablefmt="fancy_grid", stralign="center")

    def move(self, row, col, player):
        """
        Row -> 0-2 range inclusive
        Col -> 0-2 range inclusive
        """
        row = int(row)
        col = int(col)
        if row < 0 or row > 2:
            raise ValueError("Move not on board")
        if col < 0 or col > 2:
            raise ValueError("Move not on board")
        if self.board[row][col] != "":
            raise ValueError("Move already played")

        self.board[row][col] = player

        if self.current_player == "X":
            self.current_player = "O"
        else:
            self.current_player = "X"

    def check_winner(self):
        moves = 0
        for row in range(3):
            for col in range(3):
                if self.board[row][col] != "":
                    moves += 1

            if self.board[row][0] != "" and (
                self.board[row][0] == self.board[row][1] == self.board[row][2]
            ):
                return self.board[row][0]

        for col in range(3):
            if self.board[0][col] != "" and (
                self.board[0][col] == self.board[1][col] == self.board[2][col]
            ):
                return self.board[0][col]

        if self.board[0][0] != "" and (
            self.board[0][0] == self.board[1][1] == self.board[2][2]
        ):
            return self.board[0][0]

        if self.board[2][0] != "" and (
            self.board[2][0] == self.board[1][1] == self.board[0][2]
        ):
            return self.board[2][0]

        if moves == 9:
            return "draw"

        return False


if __name__ == "__main__":
    t = ttt_oracle()
