import sys
sys.path.insert(0, '/Users/nicholasbrady/Documents/School/Academic/West Research/Projects/')
from PythonHelperFunctions import *
plot_parameters()
csfont = {'fontname':'Serif'}
# latfont = {'Computer Modern Roman'}
# matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
# matplotlib.rc('text', usetex=False)
# matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
# matplotlib.rc('text', usetex=True)
# https://matplotlib.org/users/customizing.html
matplotlib.rc('xtick', top=True, bottom=True, direction='in')
matplotlib.rc('ytick', left=True, right=True, direction='in')
plt.rc("axes.spines", top=True, right=True)
matplotlib.rc('axes', edgecolor='k')

import random
import math



# import numpy as np


class Transform:
    def __init__(self, *operations):
        self.operations = operations

    def transform(self, target):
        for op in self.operations:
            target = op.transform(target)
        return target

    def reverse(self, target):
        for op in reverse(self.operations):
            target = op.reverse(target)
        return target


class Identity:
    @staticmethod
    def transform(matrix2d):
        return matrix2d

    @staticmethod
    def reverse(matrix2d):
        return matrix2d


class Rotate90:
    def __init__(self, number_of_rotations):
        self.number_of_rotations = number_of_rotations
        self.op = np.rot90

    def transform(self, matrix2d):
        return self.op(matrix2d, self.number_of_rotations)

    def reverse(self, transformed_matrix2d):
        return self.op(transformed_matrix2d, -self.number_of_rotations)


class Flip:
    def __init__(self, op):
        self.op = op

    def transform(self, matrix2d):
        return self.op(matrix2d)

    def reverse(self, transformed_matrix2d):
        return self.transform(transformed_matrix2d)


# used to get back the original board ?
def reverse(items):
    return items[::-1]





BOARD_SIZE = 3
BOARD_DIMENSIONS = (BOARD_SIZE, BOARD_SIZE)

CELL_X = 1
CELL_O = -1
CELL_EMPTY = 0

RESULT_X_WINS = 1
RESULT_O_WINS = -1
RESULT_DRAW = 0
RESULT_NOT_OVER = 2

new_board = np.array([CELL_EMPTY] * BOARD_SIZE ** 2)


class Board:
    def __init__(self, board=None, illegal_move=None):  # initialize the board
        if board is None:
            self.board = np.copy(new_board)
        else:
            self.board = board

        self.illegal_move = illegal_move

        self.board_2d = self.board.reshape(BOARD_DIMENSIONS)

    def get_game_result(self):
        if self.illegal_move is not None:
            return RESULT_O_WINS if self.get_turn() == CELL_X else RESULT_X_WINS

        rows_cols_and_diagonals = self.get_rows_cols_and_diagonals()

        sums = list(map(sum, rows_cols_and_diagonals))
        max_value = max(sums)
        min_value = min(sums)

        if max_value == BOARD_SIZE:         # if any(sums) == 3, then x wins
            return RESULT_X_WINS

        if min_value == -BOARD_SIZE:        # if any(sums) == -3, then o wins
            return RESULT_O_WINS

        if CELL_EMPTY not in self.board_2d: # if the board is full (and no win condition met) - game is a draw
            return RESULT_DRAW

        return RESULT_NOT_OVER              # otherwise - continue playing

    def is_gameover(self):
        return self.get_game_result() != RESULT_NOT_OVER

    def is_in_illegal_state(self):
        return self.illegal_move is not None

    def play_move(self, move_index):
        board_copy = np.copy(self.board)

        if move_index not in self.get_valid_move_indexes():
            return Board(board_copy, illegal_move=move_index)

        board_copy[move_index] = self.get_turn()
        return Board(board_copy)

    def get_turn(self):
        return CELL_X if sum(self.board) == 0 else CELL_O

    def get_valid_move_indexes(self):
        return ([i for i in range(self.board.size)
                 if self.board[i] == CELL_EMPTY])

    def get_illegal_move_indexes(self):
        return ([i for i in range(self.board.size)
                if self.board[i] != CELL_EMPTY])

    # def get_random_valid_move_index(self):                    # don't really want this; prefer play
    #     return random.choice(self.get_valid_move_indexes())   # strategies to be done outside of board

    def print_board(self):
        print(self.get_board_as_string())

    def get_board_as_string(self):
        rows, cols = self.board_2d.shape
        board_as_string = "-------\n"
        for r in range(rows):
            for c in range(cols):
                move = self.get_symbol(self.board_2d[r, c])
                if c == 0:
                    board_as_string += f"|{move}|"
                elif c == 1:
                    board_as_string += f"{move}|"
                else:
                    board_as_string += f"{move}|\n"
        board_as_string += "-------\n"

        return board_as_string

    # def get_rows_cols_and_diagonals(board_2d):
    def get_rows_cols_and_diagonals(self):
        rows_and_diagonal = self.get_rows_and_diagonal(self.board_2d)
        cols_and_antidiagonal = self.get_rows_and_diagonal(np.rot90(self.board_2d))
        return rows_and_diagonal + cols_and_antidiagonal

    @staticmethod
    def get_rows_and_diagonal(board_2d):
        num_rows = board_2d.shape[0]
        return ([row for row in board_2d[range(num_rows), :]]
                + [board_2d.diagonal()])

    @staticmethod
    def get_symbol(cell):
        if cell == CELL_X:
            return 'X'
        if cell == CELL_O:
            return 'O'
        return '-'


# In[1]:
class BoardCache:
    def __init__(self):
        self.cache = {} # initialize a dictionary, key=board_2d.tobytes(), value = baord position value

    def set_for_position(self, board, value):   # set the board position value
        self.cache[board.board_2d.tobytes()] = value

    def get_for_position(self, board):
        board_2d = board.board_2d

        orientations = get_symmetrical_board_orientations(board_2d)

        for b, t in orientations:
            result = self.cache.get(b.tobytes())
            if result is not None:
                return (result, t), True

        return None, False

    def reset(self):
        self.cache = {}


    # TRANSFORMATIONS = [Identity(), Rotate90(1), Rotate90(2), Rotate90(3),
    #                    Flip(np.flipud), Flip(np.fliplr),
    #                    Transform(Rotate90(1), Flip(np.flipud)),
    #                    Transform(Rotate90(1), Flip(np.fliplr))]
    #
    # @staticmethod
    # def get_symmetrical_board_orientations(board_2d):
    #     return [(t.transform(board_2d), t) for t in TRANSFORMATIONS]




TRANSFORMATIONS = [ Identity(),
                    Rotate90(1),
                    Rotate90(2),
                    Rotate90(3),
                    Flip(np.flipud),
                    Flip(np.fliplr),
                    Transform(Rotate90(1), Flip(np.flipud)),
                    Transform(Rotate90(1), Flip(np.fliplr))]

def get_symmetrical_board_orientations(board_2d):
    return [(t.transform(board_2d), t) for t in TRANSFORMATIONS]

cache = BoardCache()

board = Board()

# cache.set_for_position(board, 0)

# print(cache.get_for_position(board))

def get_position_value(board):
    # check if the board position is already cached
    cached_position_value, found = get_position_value_from_cache(board)
    if found:
        return cached_position_value

    position_value = calculate_position_value(board)

    # cache the board position (if it isn't already cached)
    put_position_value_in_cache(board, position_value)

    return position_value

b = get_symmetrical_board_orientations(board.board_2d)

for _ in b:
    print(_[0].tobytes())

# In[20]:
'''
    pseudo code: get_position_value(board)
        Input: board
        Output: board value
    1. Get all possible board orientations (there are up to 8 unique orientations of each board.board_2d)
    2. Look if any of these board orientations are cached
        if True:
            a. if it is cached return the board value
        if False:
            a. Do a full tree search to get the board value (uses the minimax search)
            b. Add the board and its value to the cache
'''

''' Min/Max game strategy '''

def choose_min_or_max_for_comparison(board):
    # returns function min() or function max() depending on whose turn it is
    turn = board.get_turn()
    return min if turn == CELL_O else max

# recursive functions calculate_position_value, get_position_value used to evaluate the value of each subsequent move
def calculate_position_value(board):
    # board is a class object
    if board.is_gameover():
        return board.get_game_result()

    valid_move_indexes = board.get_valid_move_indexes()

    values = [get_position_value(board.play_move(m))
              for m in valid_move_indexes]

    min_or_max = choose_min_or_max_for_comparison(board)
    position_value = min_or_max(values)

    return position_value

def get_position_value(board):
    # cached_position_value, found = get_position_value_from_cache(board)
    # if found:
    #     return cached_position_value

    position_value = calculate_position_value(board)

    # put_position_value_in_cache(board, position_value)

    return position_value

def get_move_value_pairs(board):
    valid_move_indexes = board.get_valid_move_indexes()

    # assertion error if valid_move_indexes is empty
    assert valid_move_indexes, "never call with an end-position"

    # (index, value)
    move_value_pairs = [(m, get_position_value(board.play_move(m)))
                        for m in valid_move_indexes]

    return move_value_pairs

def mini_max_strategy(board): # mini_max_strategy
    min_or_max = choose_min_or_max_for_comparison(board)
    move_value_pairs = get_move_value_pairs(board)
    move, best_value = min_or_max(move_value_pairs, key=lambda m_v_p: m_v_p[1])

    best_move_value_pairs = [m_v_p for m_v_p in move_value_pairs if m_v_p[1] == best_value]
    chosen_move, _ = random.choice(best_move_value_pairs)

    return chosen_move

board = Board()
board = board.play_move(1)
board = board.play_move(2)
board = board.play_move(5)
# board = board.play_move(6)
print(board.board_2d)

# print(mini_max_strategy(board))

# In[7]:
_cache_ = {}

# In[8]:
board_orientations = get_symmetrical_board_orientations(board.board_2d)

_ = [b_[0].tobytes() in _cache_ for b_ in board_orientations]
print(_)
if not any(_):
    _cache_[board.board_2d.tobytes()] = calculate_position_value(board)

print(_cache_)
    # add the first one board_orientations[0][0].tobytes() to the dictionary, with its value pair

# In[10]:
def get_position_value_from_cache(board):
    board_orientations = get_symmetrical_board_orientations(board.board_2d)

    _ = [b_[0].tobytes() in _cache_ for b_ in board_orientations]
    if not any(_):
        _cache_[board.board_2d.tobytes()] = calculate_position_value(board)

    return board_value, True

def get_position_value(board):
    # cached_position_value, found = get_position_value_from_cache(board)
    # if found:
    #     return cached_position_value

    position_value = calculate_position_value(board)

    # put_position_value_in_cache(board, position_value)

    return position_value
