import atexit
import copy
import json
import pickle

# from ttt_ke import TicTacToe


def persist_to_file(file_name):
    try:
        cache = pickle.load(open(file_name, 'rb'))
    except (IOError, ValueError) as E:
        print(E)
        cache = {}

    atexit.register(lambda: pickle.dump(cache, open(file_name, 'wb')))

    def decorator(func):
        def new_func(param):
            frozen_param = str(param)
            if frozen_param not in cache:
                cache[frozen_param] = func(param)
            return cache[frozen_param]

        return new_func

    return decorator


def winner(board):
    """
    Returns the winner if there is one ('X' or 'O'). If the game is a draw,
    then it returns 'DRAW'. If the game is still unfinised it returns None.
    """

    # rows
    if (len(set([s[2] for s in board[0:3]])) == 1 and
            board[0][2] != ''):
        return board[0][2]
    if (len(set([s[2] for s in board[3:6]])) == 1 and
            board[3][2] != ''):
        return board[3][2]
    if (len(set([s[2] for s in board[6:9]])) == 1 and
            board[6][2] != ''):
        return board[6][2]

    # cols
    if (len(set([board[0][2], board[3][2],
                 board[6][2]])) == 1 and board[
        0][2] != ''):
        return board[0][2]
    if (len(set([board[1][2], board[4][2],
                 board[7][2]])) == 1 and
            board[1][2] != ''):
        return board[1][2]
    if (len(set([board[2][2], board[5][2],
                 board[8][2]])) == 1 and
            board[2][2] != ''):
        return board[2][2]

    # diags
    if (len(set([board[0][2], board[4][2],
                 board[8][2]])) == 1 and
            board[0][2] != ''):
        return board[0][2]
    if (len(set([board[6][2], board[4][2],
                 board[2][2]])) == 1 and
            board[6][2] != ''):
        return board[6][2]

    if '' not in set([s[2] for s in board]):
        return 'DRAW'

    return None


def opposite_player(player):
    return {'X': 'O', 'O': 'X'}[player]


def generate_children(node, current_player):
    for i, square in enumerate(node):
        if square[2] == '':
            child = copy.deepcopy(node)
            child[i][2] = current_player
            yield child


@persist_to_file('cache.dat')
def minimax(node):
    win = winner(node)

    moves = sum([len(space[2]) > 0 for space in node])
    current_player = 'X' if moves % 2 == 0 else 'O'

    assert 0 <= moves <= 9

    if win == 'X':
        return 1, None
    if win == 'O':
        return -1, None
    if win == 'DRAW':
        return 0, None

    best_child = None
    if current_player == 'X':
        value = -2
        for child in generate_children(node, current_player):
            new_value, _ = minimax(child)
            if new_value > value:
                value = new_value
                best_child = child

    else:
        value = 2
        for child in generate_children(node, current_player):
            new_value, _ = minimax(child)
            if new_value < value:
                value = new_value
                best_child = child

    return value, best_child


def best_move(node):
    """ returns the best move (row, col, player) given a TicTacToe board """
    _, successor = minimax(node)
    for i, square in enumerate(node):
        if successor[i][2] != square[2]:
            return successor[i][0] + 1, successor[i][1] + 1, successor[i][2]


if __name__ == "__main__":
    board = [[i, j, ''] for i in range(3) for j in range(3)]
    z = minimax(board)

    print(best_move(board))
