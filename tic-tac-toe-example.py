from pprint import pprint
from tabulate import tabulate
import argparse

from planners.fo_planner import Operator
from agents.RLAgent import RLAgent
from agents.ModularAgent import ModularAgent
from agents.Memo import Memo

ttt_horizontal_adj = Operator(('horizontal_adj', '?s1', '?s2'),
                              [(('row', '?s1'), '?s1r'),
                               (('row', '?s2'), '?s1r'),
                               (('col', '?s1'), '?s1c'),
                               (('col', '?s2'), '?s2c'),
                               (lambda x, y: abs(x-y) == 1, '?s1c', '?s2c')],
                              [(('horizontal_adj', '?s1', '?s2'), True)])

ttt_vertical_adj = Operator(('vertical_adj', '?s1', '?s2'),
                            [(('row', '?s1'), '?s1r'),
                             (('row', '?s2'), '?s2r'),
                             (('col', '?s1'), '?s1c'),
                             (('col', '?s2'), '?s1c'),
                             (lambda x, y: abs(x-y) == 1, '?s1r', '?s2r')],
                            [(('vertical_adj', '?s1', '?s2'), True)])

ttt_diag_adj = Operator(('diag_adj', '?s1', '?s2'),
                        [(('row', '?s1'), '?s1r'),
                         (('row', '?s2'), '?s2r'),
                         (('col', '?s1'), '?s1c'),
                         (('col', '?s2'), '?s2c'),
                         (lambda x, y: abs(x-y) == 1, '?s1r', '?s2r'),
                         (lambda x, y: abs(x-y) == 1, '?s1c', '?s2c')],
                        [(('diag_adj', '?s1', '?s2'), True)])

ttt_move = Operator(('Move', '?r', '?c'),
                    [(('value', '?s'), '?p'),
                     (('id', '?s'), 'CurrentPlayer'),
                     (('id', '?cell'), '?selection'),
                     (('row', '?cell'), '?r'),
                     (('col', '?cell'), '?c'),
                     (('contentEditable', '?cell'), True)],
                    [(('sai', '?selection', 'mark', (('value', '?p'),)),
                      True)])


class TicTacToe(object):
    """
    Just a basic object to represent game state.
    """

    def __init__(self):
        self.state = ['' for i in range(9)]

    def reset(self):
        self.state = ['' for i in range(9)]

    def current_player(self):
        if self.state.count('X') <= self.state.count('O'):
            return 'X'
        return 'O'

    def machine_rep(self):
        table = []
        table.append(['', 'Col 1', 'Col 2', 'Col 3'])
        for i in range(3):
            table.append(['Row %i' % (i+1)] + self.state[i*3:i*3+3])
        state = {}
        for row, row_data in enumerate(table):
            for col, value in enumerate(row_data):
                if row > 0 and col > 0 and value == "":
                    element = {'value': value, 'row': row, 'col': col,
                               'id': 'Cell-%i-%i' % (row, col),
                               'contentEditable': True}
                else:
                    element = {'value': value, 'row': row, 'col': col,
                               'id': 'Cell-%i-%i' % (row, col)}

                state['?ele-Cell-%i-%i' % (row, col)] = element
        state['?player'] = {'value': self.current_player(), 'id':
                            'CurrentPlayer'}

        return state

    def __str__(self):
        table = []
        table.append(['', 'Col 1', 'Col 2', 'Col 3'])
        for i in range(3):
            table.append(['Row %i' % (i+1)] + self.state[i*3:i*3+3])

        return tabulate(table, tablefmt="fancy_grid", stralign="center")

    def mark(self, row, col, player):
        """
        Row -> 1-3 range inclusive
        Col -> 1-3 range inclusive
        """
        idx = (row-1) * 3 + (col-1)
        if idx < 0 or idx > len(self.state):
            raise ValueError("Move not on board")
        if self.state[idx] != "":
            raise ValueError("Cannot play already marked spot")
        if player != self.current_player():
            raise ValueError("Wrong player")

        self.state[idx] = player

    def winner(self):
        """
        Returns the winner if there is one ('X' or 'O'). If the game is a draw,
        then it returns 'DRAW'. If the game is still unfinised it returns None.
        """

        # rows
        if len(set(self.state[0:3])) == 1 and self.state[0] != '':
            return self.state[0]
        if len(set(self.state[3:6])) == 1 and self.state[3] != '':
            return self.state[3]
        if len(set(self.state[6:9])) == 1 and self.state[6] != '':
            return self.state[6]

        # cols
        if (len(set([self.state[0], self.state[3], self.state[6]])) == 1 and
                self.state[0] != ''):
            return self.state[0]
        if (len(set([self.state[1], self.state[4], self.state[7]])) == 1 and
                self.state[1] != ''):
            return self.state[1]
        if (len(set([self.state[2], self.state[5], self.state[8]])) == 1 and
                self.state[2] != ''):
            return self.state[2]

        # diags
        if (len(set([self.state[0], self.state[4], self.state[8]])) == 1 and
                self.state[0] != ''):
            return self.state[0]
        if (len(set([self.state[6], self.state[4], self.state[2]])) == 1 and
                self.state[6] != ''):
            return self.state[6]

        if '' not in set(self.state):
            return 'DRAW'

        return None


def play_game_manually():
    game = TicTacToe()

    new_game = True
    while new_game:
        game.reset()

        while game.winner() is None:
            print()
            print("Current Player: " + game.current_player())
            print(game)

            try:
                loc = input("Enter move as row and column integers (e.g."
                            ", 1,2):")
                loc = loc.split(',')

                row = int(loc[0])
                col = int(loc[1])
                game.mark(row, col, game.current_player())
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
            print("Current Player: " + game.current_player())
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
                    player = game.current_player()

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
                        game.mark(row, col, game.current_player())
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
