import random

from apprentice.working_memory.representation import Sai
from experta import AS
from experta import Fact
from experta import KnowledgeEngine
from experta import MATCH
from experta import Rule
from experta import TEST
from tabulate import tabulate


# from experta import DefFacts
# from apprentice.working_memory.representation import Fact
# from apprentice.working_memory.representation import Skill


class TTT_Engine(KnowledgeEngine):

    @Rule(
        Fact(type="Square", row=MATCH.row, col=MATCH.square1col),
        Fact(type="Square", row=MATCH.row, col=MATCH.square2col),
        TEST(lambda square1col, square2col: square2col == square1col + 1))
    def horizontally_adj(self, square1, square2):
        relation = Fact(relation="horizontally_adjacent", left=square1,
                        right=square2)
        self.declare(relation)

    @Rule(
        Fact(type="Square", row=MATCH.square1row, col=MATCH.col),
        Fact(type="Square", row=MATCH.square2row, col=MATCH.col),
        TEST(lambda square1row, square2row: square2row == square1row + 1))
    def vertically_adj(self, square1, square2):
        relation = Fact(relation="vertically_adjacent",
                        above=square1, below=square2)
        self.declare(relation)

    @Rule(
        Fact(type="Square", row=MATCH.square1row, col=MATCH.square1col),
        Fact(type="Square", row=MATCH.square2row, col=MATCH.square2col),
        TEST(lambda square1row, square2row: square2row == square1row + 1),
        TEST(lambda square1col, square2col: square2col == square2col + 1))
    def diagionally_adj(self, square1, square2):
        relation = Fact(relation="diagionally_adjacent",
                        upper_left=square1, lower_right=square2)
        self.declare(relation)

    @Rule(
        Fact(type="TicTacToe", current_player=MATCH.player),
        AS.square << Fact(type="Square", row=MATCH.row, col=MATCH.col,
                          value=""))
    def move(self, square, player):
        sai = Fact(type="sai", selection=square, action="mark", value=player)
        return sai
        # TODO need to do something like this to halt behavior and serve up the
        # output
        # self.output = sai
        # self.halt()


class CurrentPlayer(Fact):
    @property
    def value(self):
        return self['value']


class Square(Fact):
    @property
    def row(self):
        return self['row']

    @property
    def col(self):
        return self['col']

    @property
    def value(self):
        return self['value']


class TTTEngineBasic(KnowledgeEngine):
    @Rule(
        Fact(type="Square", row=MATCH.row, col=MATCH.col, value=""),
        CurrentPlayer(value=MATCH.player))
    def suggest_move(self, row, col, player):
        return Sai(selection=(row, col), action='Mark',
                   input={'value': player})


class TicTacToe(Fact):
    """
    Just a basic object to represent game state.
    """

    def __init__(self, board=None, current_player=None):
        if board is not None and current_player is not None:
            self.board = board
            self.current_player = current_player
        else:
            self.reset()
        self.opposite_player = {'X': 'O', 'O': 'X'}

    def reset(self):
        self.current_player = CurrentPlayer(value='X')

        self.board = [Square(row=i, col=j, value='')
                      for i in range(3) for j in
                      range(3)]

    def dict_rep(self):
        l = [self.current_player.as_dict()] + [sq.as_dict() for sq in
                                                  self.board]
        return {i: d for i, d in enumerate(l)}

    def list_rep(self):
        return [[s.row, s.col, s.value] for s in self.board]

    def __eq__(self, other):
        if isinstance(other, TicTacToe):
            return self.key() == other.key()
        return NotImplemented

    def __hash__(self):
        return hash(self.key())

    def machine_rep(self):
        return self.dict_rep()

    def next_player(self):
        return self.opposite_player[self.current_player['value']]

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
        if player != self.current_player['value']:
            raise ValueError("Wrong player")

        self.board[idx]['value'] = player

        # sel = [x for x in self.board if x['row'] == row and x['col'] ==
        # col][0]

        ret = Sai(selection=(row, col),
                  action='Mark',
                  input={'value': self.current_player['value']})

        if self.current_player['value'] == "X":
            self.current_player['value'] = "O"
        else:
            self.current_player['value'] = "X"

        return ret

    def get_random_move(self):
        moves = [sq for sq in self.board if sq.value == ""]
        sq = random.choice(moves)
        return sq.row + 1, sq.col + 1, self.current_player.value

    def winner(self):
        """
        Returns the winner if there is one ('X' or 'O'). If the game is a draw,
        then it returns 'DRAW'. If the game is still unfinised it returns None.
        """

        # rows
        if (len(set([s.value for s in self.board[0:3]])) == 1 and
                self.board[0].value != ''):
            return self.board[0].value
        if (len(set([s.value for s in self.board[3:6]])) == 1 and
                self.board[3].value != ''):
            return self.board[3].value
        if (len(set([s.value for s in self.board[6:9]])) == 1 and
                self.board[6].value != ''):
            return self.board[6].value

        # cols
        if (len(set([self.board[0].value, self.board[3].value,
                     self.board[6].value])) == 1 and self.board[
            0].value != ''):
            return self.board[0].value
        if (len(set([self.board[1].value, self.board[4].value,
                     self.board[7].value])) == 1 and
                self.board[1].value != ''):
            return self.board[1].value
        if (len(set([self.board[2].value, self.board[5].value,
                     self.board[8].value])) == 1 and
                self.board[2].value != ''):
            return self.board[2].value

        # diags
        if (len(set([self.board[0].value, self.board[4].value,
                     self.board[8].value])) == 1 and
                self.board[0].value != ''):
            return self.board[0].value
        if (len(set([self.board[6].value, self.board[4].value,
                     self.board[2].value])) == 1 and
                self.board[6].value != ''):
            return self.board[6].value

        if '' not in set([s.value for s in self.board]):
            return 'DRAW'

        return None
