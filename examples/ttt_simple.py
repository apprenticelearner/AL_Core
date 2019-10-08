import schema
from experta import Fact, Field, KnowledgeEngine, MATCH, AS, TEST, DefFacts, \
    Rule
from tabulate import tabulate


class Square(Fact):
    row = Field(int, mandatory=True)
    col = Field(int, mandatory=True)
    val = Field(str, mandatory=True)


class Move(Square):
    pass


class CurrentPlayer(Fact):
    name = Field(schema.Or("X", "O"), mandatory=True)


class ttt_engine(KnowledgeEngine):
    @DefFacts()
    def init_board(self, x=3, players=['X', 'O']):
        yield CurrentPlayer(name=players[0])
        for row in range(3):
            for col in range(3):
                yield Square(row=row, col=col, val='')

    @Rule(
        AS.square1 << Square(row=MATCH.row, col=MATCH.square1col),
        AS.square2 << Square(row=MATCH.row, col=MATCH.square2col),
        TEST(lambda square1col, square2col: square2col == square1col + 1))
    def horizontally_adj(self, row, square1col, square2col):
        relation = Fact(relation="horizontally_adjacent", row=row,
                        square1col=square1col, square2col=square2col)
        # self.declare(relation)

    # ... other relations

    @Rule(
        CurrentPlayer(name=MATCH.player),
        AS.square << Square(row=MATCH.row, col=MATCH.col,
                            val=""))
    def move(self, row, col, player):
        pass
        # return Move(row=row, col=col, val=player).as_dict()


class ttt_oracle:
    """
    Enviornment oracle for ttt:
    """

    def __init__(self, players=['X', 'O']):
        self.players = players
        self.board = [[None for _ in range(3)] for _ in range(3)]

    def move(self, row, col, player):
        assert self.board[row][col] == ""
        self.board[row][col] = player
        return [{'__class__': Square, 'row': row, 'col': col,
                 'val': player}], [
                   {'__class__': Square, 'row': row, 'col': col, 'val': ''}]


def set_state(self, state):
    for fact in state:
        if fact['__class__'].__name__ == 'Square':
            self.board[fact['row']][fact['col']] = fact['val']


def dict_rep(self):
    return {'type': 'TicTacToe', 'current player': 'X',
            'board': [s.dict_rep() for s in self.board]}


def machine_rep(self):
    return self.dict_rep()


def __str__(self):
    table = []
    table.append(['', 'Col 1', 'Col 2', 'Col 3'])
    for i in range(3):
        table.append(['Row %i' % (i + 1)] +
                     [s.value for s in self.board[i * 3: i * 3 + 3]])

    return tabulate(table, tablefmt="fancy_grid", stralign="center")


def mark(self, row, col, player):
    """
    Row -> 1-3 range inclusive
    Col -> 1-3 range inclusive
    """
    if row < 1 or row > 3:
        raise ValueError("Move not on board")
    if col < 1 or col > 3:
        raise ValueError("Move not on board")

    idx = (row - 1) * 3 + (col - 1)
    if idx < 0 or idx > len(self.board):
        raise ValueError("Move not on board")
    if self.board[idx].value != "":
        raise ValueError("Cannot play already marked spot")
    if player != self.current_player:
        raise ValueError("Wrong player")

    self.board[idx].value = player

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
                self.board[row][0] == self.board[row][1] == self.board[row][
            2]):
            return self.board[row][0]

    if moves == 9:
        return 'draw'

    for col in range(3):
        if self.board[0][col] != "" and (
                self.board[0][col] == self.board[1][col] == self.board[2][
            col]):
            return self.board[row][0]

    if self.board[0][0] != "" and (
            self.board[0][0] == self.board[1][1] == self.board[2][2]):
        return self.board[0][0]

    if self.board[2][0] != "" and (
            self.board[2][0] == self.board[1][1] == self.board[0][2]):
        return self.board[2][0]

    return None


if __name__ == "__main__":
    t = ttt_oracle()
