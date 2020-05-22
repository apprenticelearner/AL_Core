from pprint import pprint
from tabulate import tabulate
import argparse

from apprentice.planners.fo_planner import Operator
from apprentice.agents.RLAgent import RLAgent
from apprentice.agents.ModularAgent import ModularAgent
from apprentice.agents.Memo import Memo

from experta import Fact
from experta import KnowledgeEngine
from experta import MATCH
from experta import AS

from experta import TEST
from experta import Rule
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

        # TODO need to do something like this to halt behavior and serve up the
        # output
        self.output = sai
        self.halt()


"""
Here is an example of a tic tac tow board state in JSON for the above
skills/rules

game = {
    'type': 'TicTacToe',
    'current_player': 'X',
    'board': [
        {'type': 'Square',
         'row', 1,
         'col', 1
         'value': ''}
        {'type': 'Square',
         'row', 1,
         'col', 2
         'value': ''}
        {'type': 'Square',
         'row', 1,
         'col', 3
         'value': ''}
        {'type': 'Square',
         'row', 2,
         'col', 1
         'value': ''}
        {'type': 'Square',
         'row', 2,
         'col', 2
         'value': ''}
        {'type': 'Square',
         'row', 2,
         'col', 3
         'value': ''}
        {'type': 'Square',
         'row', 3,
         'col', 1
         'value': ''}
        {'type': 'Square',
         'row', 3,
         'col', 2
         'value': ''}
        {'type': 'Square',
         'row', 3,
         'col', 3
         'value': ''}]
}

"""


class Square(Fact):
    def __init__(self, row, col, value):
        self.row = row
        self.col = col
        self.value = value

    def dict_rep(self):
        return {'type': 'Square',
                'row': self.row,
                'col': self.col,
                'value': self.value}


class TicTacToe(Fact):
    """
    Just a basic object to represent game state.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.current_player = "X"
        self.board = [Square(i, j, '') for i in range(3) for j in range(3)]

    def dict_rep(self):
        return {'type': 'TicTacToe', 'current player': 'X',
                'board': [s.dict_rep() for s in self.board]}

    def machine_rep(self):
        return self.dict_rep()

    def __str__(self):
        table = []
        table.append(['', 'Col 1', 'Col 2', 'Col 3'])
        for i in range(3):
            table.append(['Row %i' % (i+1)] +
                         [s.value for s in self.board[i*3: i*3+3]])

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
                     self.board[6].value])) == 1 and self.board[0].value != ''):
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


def play_game_manually():
    game = TicTacToe()

    new_game = True
    while new_game:
        game.reset()

        while game.winner() is None:
            print()
            print("Current Player: " + game.current_player)
            print(game)

            try:
                loc = input("Enter move as row and column integers (e.g."
                            ", 1,2):")
                loc = loc.split(',')

                row = int(loc[0])
                col = int(loc[1])
                game.mark(row, col, game.current_player)
            except ValueError:
                print("############################")
                print("# Invalid move, try again. #")
                print("############################")

        if game.winner() == "DRAW":
            print("DRAW")
        else:
            print(game.winner() + " WINS!")

        new = input("Play again? Press enter to continue or type 'no' to"
                    " stop.")
        new_game = new == ""


def train_agent(agent_class):
    game = TicTacToe()

    agent = agent_class([ttt_horizontal_adj,
                         ttt_vertical_adj,
                         # ttt_available
                         ttt_diag_adj
                         ], [ttt_move])

    new_game = True
    while new_game:
        game.reset()

        while game.winner() is None:
            print()
            print("Current Player: " + game.current_player)
            print(game)
            original_state = game.machine_rep()
            pprint(original_state)

            try:
                agent_action = agent.request(original_state)
                if agent_action == {}:
                    print("Agent unsure what to do, enter move to demonstrate"
                          " correct behavior")

                    loc = input("Enter move as row and column integers (e.g."
                                ", 1,2):")
                    loc = loc.split(',')

                    row = int(loc[0])
                    col = int(loc[1])
                    player = game.current_player

                    game.mark(row, col, player)

                    correctness = True

                else:
                    sel = '?ele-' + agent_action['selection']
                    row = original_state[sel]['row']
                    col = original_state[sel]['col']
                    player = agent_action['inputs']['value']

                    correctness = input('Would putting an %s in row %i and col'
                                        ' %i be correct? (enter for yes or'
                                        ' "no" for no)? ' % (player, row, col))
                    correctness = correctness == ""

                    if correctness:
                        game.mark(row, col, game.current_player)
                        correctness = 1
                    else:
                        correctness = -1

                agent.train(original_state, 'Cell-%i-%i' % (row, col), 'mark',
                            {'value': player},
                            # {'row': row, 'col': col, 'player': player},
                            correctness, 'mark', [])

            except ValueError:
                print("############################")
                print("# Invalid move, try again. #")
                print("############################")
                agent.train(original_state, 'Cell-%i-%i' % (row, col), 'mark',
                            {'value': player}, -1, 'mark', [])

        if game.winner() == "DRAW":
            print("DRAW")
        else:
            print(game.winner() + " WINS!")

        new = input("Play again? Press enter to continue or type 'no' to"
                    " stop.")
        new_game = new == ""


if __name__ == "__main__":
    play_game_manually()
    parser = argparse.ArgumentParser(description='An interactive training demo'
                                     'for apprentice agents.')
    parser.add_argument('-agent', choices=['Modular', 'RLAgent', 'Memo'],
                        default='Modular', help='The agent type to use')

    args = parser.parse_args()

    if args.agent == 'Memo':
        train_agent(Memo)
    elif args.agent == 'RLAgent':
        train_agent(RLAgent)
    else:
        train_agent(ModularAgent)
