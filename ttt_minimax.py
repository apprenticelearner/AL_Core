from math import inf as infinity
from random import choice
import time
from os import system
import functools

def rlist(tup):
  return list(list(t) for t in tup)

def rtup(lis):
  return tuple(tuple(t) for t in lis)

def get_new_board():
    return (
        (0, 0, 0),
        (0, 0, 0), 
        (0, 0, 0),
    )

def flip(p):
    return 'O' if p=='X' else 'X'

def evaluate(state, player):
    """
    Function to heuristic evaluation of state.
    :param state: the state of the current board
    :return: +1 if the computer wins; -1 if the human wins; 0 draw
    """
    if wins(state, player):
        score = +1
    elif wins(state, flip(player)):
        score = -1
    else:
        score = 0

    return score


def wins(state, player):
    """
    This function tests if a specific player wins. Possibilities:
    * Three rows    [X X X] or [O O O]
    * Three cols    [X X X] or [O O O]
    * Two diagonals [X X X] or [O O O]
    :param state: the state of the current board
    :param player: a human or a computer
    :return: True if the player wins
    """
    win_state = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        [state[2][0], state[1][1], state[0][2]],
    ]
    if [player, player, player] in win_state:
        return True
    else:
        return False


def game_over(state):
    """
    This function test if the human or computer wins
    :param state: the state of the current board
    :return: True if the human or computer wins
    """
    return wins(state, 'X') or wins(state, 'O')


def empty_cells(state):
    """
    Each empty cell will be added into cells' list
    :param state: the state of the current board
    :return: a list of empty cells
    """
    cells = []
    #print(state)
    for x, row in enumerate(state):
        for y, cell in enumerate(row):
            if cell == 0:
                cells.append([x, y])

    return cells


def valid_move(state, x, y):
    """
    A move is valid if the chosen cell is empty
    :param x: X coordinate
    :param y: Y coordinate
    :return: True if the board[x][y] is empty
    """
    if [x, y] in empty_cells(state):
        return True
    else:
        return False


def set_move(state, x, y, player):
    """
    Set the move on board, if the coordinates are valid
    :param x: X coordinate
    :param y: Y coordinate
    :param player: the current player
    """
    if valid_move(state, x, y):
        state = rlist(state)
        state[x][y] = player
        return rtup(state)
    else:
        return ()
    

def minimax_wrapper(state, player):
    depth = state.count('') 
    state = [x if x != '' else 0 for x in state]
    state = tuple(tuple(state[i] for i in range(j*3, (j*3)+3)) for j in range(3))
    
    return minimax_entry(state, depth, player)

@functools.lru_cache(maxsize=None)
def minimax_entry(state, depth, player):
    original_player=player
    return minimax(state, depth, player, original_player)


def minimax(state, depth, player, original_player):
    """
    AI function that choice the best move
    :param state: current state of the board
    :param depth: node index in the tree (0 <= depth <= 9),
    but never nine in this case (see iaturn() function)
    :param player: an human or a computer
    :return: a list with [the best row, best col, best score]
    """
    if player == original_player:
        best = [-1, -1, -infinity]
    else:
        best = [-1, -1, +infinity]

    if depth == 0 or game_over(state):
        score = evaluate(state, original_player)
        return [-1, -1, score]

    for cell in empty_cells(state):
        x, y = cell[0], cell[1]
        new_state = set_move(state, x, y, player)
        if len(new_state)  == 0:
            continue
        score = minimax(new_state, depth - 1, flip(player), original_player)
        #state[x][y] = 0
        score[0], score[1] = x, y
        if player == original_player:
            if score[2] > best[2]:
                best = score  # max value
        else:
            if score[2] < best[2]:
                best = score  # min value
    return best


def render(state):
    print('\n' )
    for row in state:
        for cell in row:
            print (cell)
        print('\n')

def ai_turn(state, ai_id):
    """
    It calls the minimax function if the depth < 9,
    else it choices a random coordinate.
    :param c_choice: computer's choice X or O
    :param h_choice: human's choice X or O
    :return:
    """
    depth = len(empty_cells(state))
    if depth == 0 or game_over(state):
        return
    
    if depth == 9:
        x = choice([0, 1, 2])
        y = choice([0, 1, 2])
    else:
        move = minimax_entry(state, depth, ai_id)
        x, y = move[0], move[1]

    return x, y
    

def play():
    start = time.time()
    board = get_new_board()
    curr_player = 'X'
    while len(empty_cells(board)) > 0 and not game_over(board):
        x, y = ai_turn(board, curr_player)
        board = set_move(board, x, y, curr_player)
        curr_player = flip(curr_player)
        #render(board)
      
    game_time = str(time.time() - start)
    
    if wins(board, 'X'):
        print('X wins')
    elif wins(board, 'O'):
        print('O wins')
    else:
        print('draw in ' + game_time)


if __name__ == '__main__':
    while(True):
        play()