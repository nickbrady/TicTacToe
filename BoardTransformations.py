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

import itertools
from copy import copy


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



TRANSFORMATIONS = [ Identity(),
                    Rotate90(1),
                    Rotate90(2),
                    Rotate90(3),
                    Flip(np.flipud),
                    Flip(np.fliplr),
                    Transform(Rotate90(1), Flip(np.flipud)),
                    Transform(Rotate90(1), Flip(np.fliplr))]

# Example usage of Transform.transform and Transform.reverse
# board = Board()
# board = board.play_move(1)
# board = board.play_move(2)
# board = board.play_move(3)
#
# print(board.board_2d)
#
# (b, t), found = cache.get_for_position(board)
#
# _ = t.transform(board.board_2d)
# print(_)
#
# __ = t.reverse(_)
# print(__)


# In[2]:
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

class BoardCache:

    def __init__(self):
        self.cache = {} # initialize a dictionary, key=board_2d.tobytes(),
                        #                          value= board position value (1, 0, -1)

    def get_for_position(self, board):
            board_2d = board.board_2d

            orientations = self.get_symmetrical_board_orientations(board_2d)

            for b, t in orientations:   # return the value as well as the transformation (useful for qtable)
                result = self.cache.get(b.tobytes())
                if result is not None:
                    return (result, t), True

            return None, False

    # def get_for_position(self, board):
    #     board_orientations = self.get_symmetrical_board_orientations(board.board_2d)
    #
    #     # boolean list of which board orientations are in the cache
    #     _ = [b_[0].tobytes() in self.cache for b_ in board_orientations]
    #
    #     # if not in cache - return False
    #     if not any(_):
    #         return None, False
    #
    #     # else
    #     board_orientations = list(itertools.compress(board_orientations, _))
    #     board_value = self.cache[board_orientations[0][0].tobytes()]
    #
    #     return board_value, True

    def set_for_position(self, board, position_value):
        self.cache[board.board_2d.tobytes()] = position_value

    def get_symmetrical_board_orientations(self, board_2d):
        return [(t.transform(board_2d), t) for t in TRANSFORMATIONS]

    def reset(self):
        self.cache = {}

def get_position_value(board):
    # check if the board position is already cached
    cached_position_value, found = cache.get_for_position(board)
    if found:
        return cached_position_value[0]

    position_value = calculate_position_value(board)

    # cache the board position (if it isn't already cached)
    cache.set_for_position(board, position_value)

    # No cached board detected, so no transformations linked to position_value
    # return (position_value, None)
    return position_value

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


def choose_min_or_max_for_comparison(board):
    # returns function min() or function max() depending on whose turn it is
    turn = board.get_turn()
    return min if turn == CELL_O else max

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

def random_move(board):
    move_ind = random.choice(board.get_valid_move_indexes())
    return move_ind


def play_tic_tac_toe(player_1_strategy=None, player_2_strategy=None, number_of_games=1):
    game_results = np.zeros(number_of_games)
    game_moves_list = []

    if player_1_strategy is None:
        player_1_strategy = random_move

    if player_2_strategy is None:
        player_2_strategy = random_move

    for i in range(number_of_games):
        board = Board()
        move_count = 0
        game_moves = np.zeros(9)        # game_moves = -np.ones(9) # move history
        while not board.is_gameover():
            if board.get_turn() == 1: # player 1
                move_ind = player_1_strategy(board)
            else:                     # player 2
                move_ind = player_2_strategy(board)

            board = board.play_move(move_ind)
            game_moves[move_count] = move_ind+1     # if zero index is possible hard to tell how many zeros
            move_count += 1

        game_results[i] = board.get_game_result()
        game_moves_list.append(game_moves)

    return game_results, game_moves_list

# In[3]:
'''
    Play some test games
'''

game_results, game_moves_list = play_tic_tac_toe(player_1_strategy=None, player_2_strategy=None,
                                            number_of_games=100)

print(len([r for r in game_results if r == 1]))
print(len([r for r in game_results if r == 0]))
print(len([r for r in game_results if r == -1]))

# In[4]:

cache = BoardCache()
board = Board()

print(board.board_2d)

print(get_position_value(board))

print("Number or symmetry unique board positions: \n", len(cache.cache))
cache_keys = list(cache.cache.keys())

# In[5]:

class q_table:

    def __init__(self):
        self.q_table = {}

    def get_state_actions(self, board):
        board_bytes = board.board_2d.tobytes()

        if board_bytes not in q_table.keys():
            self.q_table[board_bytes] = np.zeros(9)

        return self.q_table[board_bytes]

    # def update_q_table(self):
    #

# q-table no cache
EPSILON = 0.9
GAMMA = 1.0 #0.9
ALPHA = 0.4 #0.1

q_table = {}

def get_state_actions(board):
    board_bytes = board.board_2d.tobytes()
    if board_bytes not in q_table.keys():
        q_table[board_bytes] = np.zeros(9)

    return q_table[board_bytes]

print(q_table)

# In[6]:
EPSILON = 0.7

results = []
for i in range(10000):
    board = Board()

    while not board.is_gameover():
        board_bytes = board.board_2d.tobytes()
        if board_bytes not in q_table.keys():       # if this board is not in q-table, add it
            q_table[board_bytes] = np.zeros(9)      # initialize all actions to be neutral

        _state_actions_ = q_table[board_bytes]

        max_val   = max(q_table[board_bytes])
        max_index = np.where(q_table[board_bytes] == max_val)[0]

        if (random.uniform(0, 1) > EPSILON): # choose a random move
            move_ind = random.randrange(9)
        else:
            move_ind = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # q_old = q_table[board_bytes]
        board_ = board.play_move(move_ind)  # board at next move


        R = 0
        if board_.is_gameover():
            R = board_.get_game_result()
            Q_prime = np.zeros(9)
        else:
            board_ = board_.play_move(random.choice(board_.get_valid_move_indexes()))
            Q_prime = get_state_actions(board_)
        # update q-table
        # Q(s,a) = Q(s,a) + α[r + γ max(Q') - Q]
        q_values = q_table[board_bytes]
        q_values[move_ind] += ALPHA * (R + GAMMA*max(Q_prime) - q_values[move_ind])

        # print(board_.board_2d)
        # print(Q_prime)
        # print(q_values)

        board = board.play_move(move_ind)   # X move

        # O move
        if board.is_gameover():
            continue
        else:
            # move_choice = random.choice(board.get_valid_move_indexes())
            move_choice = mini_max_strategy(board)

            board = board.play_move(move_choice)

    results.append(board.get_game_result())

# In[7]:
print(board.board_2d)
print(board.get_game_result())
print(board.board_2d)
# print(board.board_2d.tobytes())
print(len(q_table))

# In[8]:
print(len([r for r in results if r == 1]))
print(len([r for r in results if r == 0]))
print(len([r for r in results if r == -1]))

# In[9]:
board = Board()
q_table[board.board_2d.tobytes()]

# In[10]:

board1 = Board()

board1 = board1.play_move(0)
board1 = board1.play_move(2)
board1 = board1.play_move(6)

print(board1.board_2d)

board2 = copy(board1)
board2.board_2d = np.rot90(board1.board_2d)
print(board2.board_2d)


# In[11]:
def get_position_value_qtable(board, cache):
    # check if the board position is already cached
    cached_position_value, found = cache.get_for_position(board)
    if found:
        return cached_position_value[0]

    position_value = calculate_position_value(board, cache)
    # print(position_value)

    # cache the board position (if it isn't already cached)
    cache.set_for_position(board, position_value)

    # No cached board detected, so no transformations linked to position_value
    # return (position_value, None)
    return position_value


def calculate_position_value(board, cache):
    # board is a class object
    if board.is_gameover():
        return board.get_game_result()

    valid_move_indexes = board.get_valid_move_indexes()

    values = [get_position_value_qtable(board.play_move(m), cache)
              for m in valid_move_indexes]

    min_or_max = choose_min_or_max_for_comparison(board)
    position_value = min_or_max(values)

    return position_value

# initialize qtable with all possible unique (by symmetry) board positions
qtable1 = BoardCache()
board = Board()

get_position_value_qtable(board, qtable1)

for k in qtable1.cache.keys():
    qtable1.cache[k] = np.zeros(9)

print(len(qtable1.cache))

# In[12]:
EPSILON = 1.
GAMMA = 1.0 #0.9
ALPHA = 0.2 #0.1
number_games = 10

# (q_values, trans_f), found = qtable1.get_for_position(board)
results = []
for i in range(number_games):
    board = Board()
    board = board.play_move(0)
    board = board.play_move(4)
    board = board.play_move(1)
    board = board.play_move(2)
    # against a minimax player, this is always a losing board position, so the Q-learning strategy should learn to avoid this board position

    boards_ = []
    boards_.append(board.board_2d)
    while not board.is_gameover():
        (q_values, trans_f), found = qtable1.get_for_position(board)

        # choose move
        if (random.uniform(0, 1) > EPSILON): # choose a random move
            move_ind = random.randrange(9)
        else:
            max_val   = max(q_values)
            max_index = np.where(q_values == max_val)[0]
            move_ind = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # play move after transforming board
        # transform board
        board = Board(trans_f.transform(board.board_2d).flatten())
        # play move
        board = board.play_move(move_ind)
        # reverse transform board
        board = Board(trans_f.reverse(board.board_2d).flatten())

        boards_.append(board.board_2d)
        # Update QTable
        R = 0
        if board.is_gameover():
            R = board.get_game_result()
            if R == -1:
                R = -2
            Q_prime = np.zeros(9)
        else:
            board_prime = board.play_move(random.choice(board.get_valid_move_indexes()))
            (Q_prime, T_prime), found = qtable1.get_for_position(board_prime)

        # Q(s,a) = Q(s,a) + α[r + γ max(Q') - Q]
        q_values[move_ind] += ALPHA * (R + GAMMA*max(Q_prime) - q_values[move_ind])

        # O move
        if board.is_gameover():
            continue
        else:
            rand_move = random.choice(board.get_valid_move_indexes())
            board = board.play_move(rand_move)
            boards_.append(board.board_2d)

    if number_games <= 10000:
        results.append(board.get_game_result())

    # print(board.get_game_result())
    # if board.get_game_result() == -1:
    #     for b in boards_:
    #         print(b)
    #     print()

    # if i % 1000 == 0:
    #     EPSILON += (1 - EPSILON) * 0.1

# In[13]:
print(len([r for r in results if r == 1]) / len(results)*100)
print(len([r for r in results if r == 0]) / len(results)*100)
print(len([r for r in results if r == -1]) / len(results)*100)


# In[14]:
board = Board()

print(qtable1.cache[board.board_2d.tobytes()])

# In[15]:
board = Board()
board = board.play_move(0)
board = board.play_move(4)
board = board.play_move(1)
board = board.play_move(2)

print(board.board_2d)

key = board.board_2d.tobytes()
print(qtable1.cache[key])

# In[16]:
def play_q_learn(board, q_tables_list):

    if len(q_tables_list) == 2:
        qtableA, qtableB = q_tables_list
        (q_values_A, trans_f), found = qtableA.get_for_position(board)
        (q_values_B, trans_f), found = qtableB.get_for_position(board)

        q_values = (q_values_A + q_values_B)/2
    else:
        qtable = q_tables_list
        (q_values, trans_f), found = qtable.get_for_position(board)





    pass

def train_q_learn:
    pass

# In[17]:
# initialize qtable with all possible unique (by symmetry) board positions
qtableA = BoardCache()
board = Board()

get_position_value_qtable(board, qtableA)

for k in qtableA.cache.keys():
    qtableA.cache[k] = np.zeros(9)

qtableB = copy(qtableA)

# In[18]:
''' Double Q-Learning '''
EPSILON = 1.0
GAMMA = 0.9 #0.9
ALPHA = 0.1 #0.1
number_games = 10000

# (q_values, trans_f), found = qtable1.get_for_position(board)
results = []
for i in range(number_games):
    board = Board()
    # board = board.play_move(0)
    # board = board.play_move(4)
    # board = board.play_move(1)
    # board = board.play_move(2)

    # boards_ = []
    while not board.is_gameover():
        (q_values_A, trans_f), found = qtableA.get_for_position(board)
        (q_values_B, trans_f), found = qtableB.get_for_position(board)

        q_values = (q_values_A + q_values_B)/2

        # choose move
        if (random.uniform(0, 1) > EPSILON): # choose a random move
            move_ind = random.randrange(9)
        else:
            max_val   = max(q_values)
            max_index = np.where(q_values == max_val)[0]
            move_ind  = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # play move after transforming board
        # transform board
        board = Board(trans_f.transform(board.board_2d).flatten())
        # play move
        board = board.play_move(move_ind)
        # reverse transform board
        board = Board(trans_f.reverse(board.board_2d).flatten())

        # boards_.append(board.board_2d)
        # Update QTable
        if random.uniform(0,1) < 0.5:   # randomly choose which q-table gets updated
            q_vals_update = q_values_A
            q_table_update = qtableA
            q_table_target = qtableB
        else:
            q_vals_update = q_values_B
            q_table_update = qtableB
            q_table_target = qtableA

        R = 0
        if board.is_gameover():
            R = board.get_game_result()
            Q_prime = np.zeros(9)
            q_vals_update[move_ind] += ALPHA * (R - q_vals_update[move_ind])
        else:
            board_prime = board.play_move(random.choice(board.get_valid_move_indexes()))
            (Q_prime_up, T_prime), found = q_table_update.get_for_position(board_prime)
            (Q_prime_tar, T_prime), found = q_table_target.get_for_position(board_prime)

            # max_val   = max(Q_prime)
            # max_index = np.where(Q_prime == max_val)[0]
            max_val   = max(Q_prime_up)
            max_index = np.where(Q_prime_up == max_val)[0]
            move_prime  = random.choice(max_index)

            q_vals_update[move_ind] += ALPHA * (R + GAMMA * Q_prime_tar[move_prime] - q_vals_update[move_ind])

        # O move
        if board.is_gameover():
            continue
        else:
            rand_move = random.choice(board.get_valid_move_indexes())
            board = board.play_move(rand_move)
            # boards_.append(board.board_2d)

    if number_games <= 10000:
        results.append(board.get_game_result())

# In[19]:
print(len([r for r in results if r == 1]) / len(results)*100)
print(len([r for r in results if r == 0]) / len(results)*100)
print(len([r for r in results if r == -1]) / len(results)*100)

# In[20]:
'''
    Q-Learning vs Random Moves  - done
    Q-Learning vs Minimax       - not done

    Double Q-Learning vs Random Moves
    Double Q-Learning vs Minimax
'''


# In[21]:
'''
    train q-table against random player (ε ~ 0.9)
        document results    (ε = 1)
                            against random player
                            against minimax player

    train q-table against minimax player (ε ~ 0.9)
        document results    (ε = 1)
                            against random player
                            against minimax player

    How many games is enough?
    write script to train and test?
        train for XXX games (updating table)
        test for ZZZ games (no updates)
        repeat cycle and plot results to visually detect the presence of a plateau, at which point training can stop
'''

qtable1 = BoardCache()
board = Board()

get_position_value_qtable(board, qtable1)

for k in qtable1.cache.keys():
    qtable1.cache[k] = np.zeros(9)

print(len(qtable1.cache))

# In[22]:
def q_table_train():


def q_table_strategy(board, q_table):
    (q_values, trans_f), found = q_table.get_for_position(board)

    q_trans = trans_f.reverse(q_values.reshape((3,3)))
    max_val   = max(q_trans.flatten())
    max_index = np.where(q_trans.flatten() == max_val)[0]
    move_ind  = random.choice(max_index)

    # board = Board(trans_f.transform(board.board_2d).flatten())
    board = board.play_move(move_ind)
    # board = Board(trans_f.reverse(board.board_2d).flatten())

# In[23]:

qtable1 = BoardCache()
board = Board()

get_position_value_qtable(board, qtable1)

for k in qtable1.cache.keys():
    qtable1.cache[k] = np.zeros(9)

print(len(qtable1.cache))

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 10001
num_test_games = 500

results = []

for i in range(number_games):
    board = Board()

    while not board.is_gameover():
        (q_values, trans_f), found = qtable1.get_for_position(board)

        # choose move
        if (random.uniform(0, 1) > EPSILON): # choose a random move
            move_ind = random.randrange(9)
        else:
            max_index = np.where(q_values == max(q_values))[0]
            move_ind  = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # play move
        board = Board(trans_f.transform(board.board_2d).flatten())
        board = board.play_move(move_ind)
        board = Board(trans_f.reverse(board.board_2d).flatten())

        # Update QTable
        R = 0
        if board.is_gameover():
            R = board.get_game_result()
            Q_prime = np.zeros(9)
        else:
            board_prime = board.play_move(random.choice(board.get_valid_move_indexes()))
            (Q_prime, T_prime), found = qtable1.get_for_position(board_prime)

        # Q(s,a) = Q(s,a) + α[r + γ max(Q') - Q]
        q_values[move_ind] += ALPHA * (R + GAMMA*max(Q_prime) - q_values[move_ind])

        # O move
        if board.is_gameover():
            continue
        else:
            rand_move = random.choice(board.get_valid_move_indexes())
            board = board.play_move(rand_move)

    # every 100 training games do a set of test games
    if i % 100 == 0:

        for j in range(num_test_games):
            board = Board()
            while not board.is_gameover():
                (q_values, trans_f), found = qtable1.get_for_position(board)

                max_index = np.where(q_values == max(q_values))[0]
                move_ind  = random.choice(max_index)

                # play X move
                board = Board(trans_f.transform(board.board_2d).flatten())
                board = board.play_move(move_ind)
                board = Board(trans_f.reverse(board.board_2d).flatten())

                # No updates
                # O move
                if board.is_gameover():
                    continue
                else:
                    rand_move = random.choice(board.get_valid_move_indexes())
                    board = board.play_move(rand_move)

            results.append(board.get_game_result())

# In[24]:

ax, fig = axes(fig_number=1, rows=1, columns=1)

for i in range(int(len(results) / num_test_games)):
    begin = i*num_test_games
    end = begin + num_test_games
    subresults = results[begin:end]

    ax[1].plot(i*100, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go')
    ax[1].plot(i*100, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks')
    ax[1].plot(i*100, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^')

ax[1].set_xlabel('Number of Training Games')
ax[1].set_ylabel('Percentage of Test Game Results')

# In[25]:

qtable1 = BoardCache()
board = Board()

get_position_value_qtable(board, qtable1)

for k in qtable1.cache.keys():
    qtable1.cache[k] = np.zeros(9)

print(len(qtable1.cache))

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 10001
num_test_games = 500

results = []

for i in range(number_games):
    board = Board()

    while not board.is_gameover():
        (q_values, trans_f), found = qtable1.get_for_position(board)

        # choose move
        if (random.uniform(0, 1) > EPSILON): # choose a random move
            move_ind = random.randrange(9)
        else:
            max_index = np.where(q_values == max(q_values))[0]
            move_ind  = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # play move
        board = Board(trans_f.transform(board.board_2d).flatten())
        board = board.play_move(move_ind)
        board = Board(trans_f.reverse(board.board_2d).flatten())

        # Update QTable
        R = 0
        if board.is_gameover():
            R = board.get_game_result()
            Q_prime = np.zeros(9)
        else:
            # board_prime = board.play_move(random.choice(board.get_valid_move_indexes()))
            board_prime = board.play_move(mini_max_strategy(board))
            (Q_prime, T_prime), found = qtable1.get_for_position(board_prime)

        # Q(s,a) = Q(s,a) + α[r + γ max(Q') - Q]
        q_values[move_ind] += ALPHA * (R + GAMMA*max(Q_prime) - q_values[move_ind])

        # O move
        if board.is_gameover():
            continue
        else:
            minimax_move = mini_max_strategy(board)
            board = board.play_move(minimax_move)

    # every 100 training games do a set of test games
    if i % 100 == 0:

        for j in range(num_test_games):
            board = Board()
            while not board.is_gameover():
                (q_values, trans_f), found = qtable1.get_for_position(board)

                max_index = np.where(q_values == max(q_values))[0]
                move_ind  = random.choice(max_index)

                # play X move
                board = Board(trans_f.transform(board.board_2d).flatten())
                board = board.play_move(move_ind)
                board = Board(trans_f.reverse(board.board_2d).flatten())

                # No updates
                # O move
                if board.is_gameover():
                    continue
                else:
                    rand_move = random.choice(board.get_valid_move_indexes())
                    board = board.play_move(rand_move)

            results.append(board.get_game_result())

# In[26]:

ax, fig = axes(fig_number=1, rows=1, columns=1)

for i in range(int(len(results) / num_test_games)):
    begin = i*num_test_games
    end = begin + num_test_games
    subresults = results[begin:end]

    ax[1].plot(i*100, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go')
    ax[1].plot(i*100, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks')
    ax[1].plot(i*100, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^')

ax[1].set_xlabel('Number of Training Games')
ax[1].set_ylabel('Percentage of Test Game Results')


# In[60]:

qtableA = BoardCache()
board = Board()

get_position_value_qtable(board, qtableA)

for k in qtableA.cache.keys():
    qtableA.cache[k] = np.zeros(9)

qtableB = copy(qtableA)

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 10001
num_test_games = 500

results = []

for i in range(number_games):
    board = Board()

    while not board.is_gameover():
        (q_values_A, trans_f), found = qtableA.get_for_position(board)
        (q_values_B, trans_f), found = qtableB.get_for_position(board)

        q_values = (q_values_A + q_values_B)/2

        # choose move
        if (random.uniform(0, 1) > EPSILON): # choose a random move
            move_ind = random.randrange(9)
        else:
            max_val   = max(q_values)
            max_index = np.where(q_values == max_val)[0]
            move_ind  = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # play move after transforming board - then transform back
        board = Board(trans_f.transform(board.board_2d).flatten())
        board = board.play_move(move_ind)
        board = Board(trans_f.reverse(board.board_2d).flatten())

        # Update QTable
        if random.uniform(0,1) < 0.5:   # randomly choose which q-table gets updated
            q_vals_update = q_values_A
            q_table_update = qtableA
            q_table_target = qtableB
        else:
            q_vals_update = q_values_B
            q_table_update = qtableB
            q_table_target = qtableA

        R = 0
        if board.is_gameover():
            R = board.get_game_result()
            Q_prime = np.zeros(9)
            q_vals_update[move_ind] += ALPHA * (R - q_vals_update[move_ind])
        else:
            board_prime = board.play_move(random.choice(board.get_valid_move_indexes()))
            (Q_prime_up, T_prime), found = q_table_update.get_for_position(board_prime)
            (Q_prime_tar, T_prime), found = q_table_target.get_for_position(board_prime)

            max_val   = max(Q_prime_up)
            max_index = np.where(Q_prime_up == max_val)[0]
            move_prime  = random.choice(max_index)

            q_vals_update[move_ind] += ALPHA * (R + GAMMA * Q_prime_tar[move_prime] - q_vals_update[move_ind])

        # O move
        if board.is_gameover():
            continue
        else:
            rand_move = random.choice(board.get_valid_move_indexes())
            board = board.play_move(rand_move)

    # every 100 training games do a set of test games
    if i % 100 == 0:

        for j in range(num_test_games):
            board = Board()
            while not board.is_gameover():
                (q_values_A, trans_f), found = qtableA.get_for_position(board)
                (q_values_B, trans_f), found = qtableB.get_for_position(board)

                q_values = (q_values_A + q_values_B)/2

                max_index = np.where(q_values == max(q_values))[0]
                move_ind  = random.choice(max_index)

                # play X move
                board = Board(trans_f.transform(board.board_2d).flatten())
                board = board.play_move(move_ind)
                board = Board(trans_f.reverse(board.board_2d).flatten())

                # No updates
                # O move
                if board.is_gameover():
                    continue
                else:
                    rand_move = random.choice(board.get_valid_move_indexes())
                    board = board.play_move(rand_move)

            results.append(board.get_game_result())

# In[100]:

ax, fig = axes(fig_number=1, rows=1, columns=1)

for i in range(int(len(results) / num_test_games)):
    begin = i*num_test_games
    end = begin + num_test_games
    subresults = results[begin:end]

    ax[1].plot(i*100, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go')
    ax[1].plot(i*100, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks')
    ax[1].plot(i*100, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^')

ax[1].set_xlabel('Number of Training Games')
ax[1].set_ylabel('Percentage of Test Game Results')


# In[60]:

qtableA = BoardCache()
board = Board()

get_position_value_qtable(board, qtableA)

for k in qtableA.cache.keys():
    qtableA.cache[k] = np.zeros(9)

qtableB = copy(qtableA)

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 10001
num_test_games = 500

results = []

for i in range(number_games):
    board = Board()

    while not board.is_gameover():
        (q_values_A, trans_f), found = qtableA.get_for_position(board)
        (q_values_B, trans_f), found = qtableB.get_for_position(board)

        q_values = (q_values_A + q_values_B)/2

        # choose move
        if (random.uniform(0, 1) > EPSILON): # choose a random move
            move_ind = random.randrange(9)
        else:
            max_val   = max(q_values)
            max_index = np.where(q_values == max_val)[0]
            move_ind  = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # play move after transforming board - then transform back
        board = Board(trans_f.transform(board.board_2d).flatten())
        board = board.play_move(move_ind)
        board = Board(trans_f.reverse(board.board_2d).flatten())

        # Update QTable
        if random.uniform(0,1) < 0.5:   # randomly choose which q-table gets updated
            q_vals_update = q_values_A
            q_table_update = qtableA
            q_table_target = qtableB
        else:
            q_vals_update = q_values_B
            q_table_update = qtableB
            q_table_target = qtableA

        R = 0
        if board.is_gameover():
            R = board.get_game_result()
            Q_prime = np.zeros(9)
            q_vals_update[move_ind] += ALPHA * (R - q_vals_update[move_ind])
        else:
            board_prime = board.play_move(random.choice(board.get_valid_move_indexes()))
            (Q_prime_up, T_prime), found = q_table_update.get_for_position(board_prime)
            (Q_prime_tar, T_prime), found = q_table_target.get_for_position(board_prime)

            max_val   = max(Q_prime_up)
            max_index = np.where(Q_prime_up == max_val)[0]
            move_prime  = random.choice(max_index)

            q_vals_update[move_ind] += ALPHA * (R + GAMMA * Q_prime_tar[move_prime] - q_vals_update[move_ind])

        # O move
        if board.is_gameover():
            continue
        else:
            opponent_move = mini_max_strategy(board)
            board = board.play_move(opponent_move)

    # every 100 training games do a set of test games
    if i % 100 == 0:

        for j in range(num_test_games):
            board = Board()
            while not board.is_gameover():
                (q_values_A, trans_f), found = qtableA.get_for_position(board)
                (q_values_B, trans_f), found = qtableB.get_for_position(board)

                q_values = (q_values_A + q_values_B)/2

                max_index = np.where(q_values == max(q_values))[0]
                move_ind  = random.choice(max_index)

                # play X move
                board = Board(trans_f.transform(board.board_2d).flatten())
                board = board.play_move(move_ind)
                board = Board(trans_f.reverse(board.board_2d).flatten())

                # No updates
                # O move
                if board.is_gameover():
                    continue
                else:
                    rand_move = random.choice(board.get_valid_move_indexes())
                    board = board.play_move(rand_move)

            results.append(board.get_game_result())

# In[100]:

ax, fig = axes(fig_number=1, rows=1, columns=1)

for i in range(int(len(results) / num_test_games)):
    begin = i*num_test_games
    end = begin + num_test_games
    subresults = results[begin:end]

    ax[1].plot(i*100, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go')
    ax[1].plot(i*100, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks')
    ax[1].plot(i*100, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^')

ax[1].set_xlabel('Number of Training Games')
ax[1].set_ylabel('Percentage of Test Game Results')


# In[200]:
board = Board()

# while not board.is_gameover():
(q_values_A, trans_f), found = qtableA.get_for_position(board)
(q_values_B, trans_f), found = qtableB.get_for_position(board)

q_values = (q_values_A + q_values_B)/2

for i in [0, 1, 2, 3, 5, 6, 7, 8]:
    q_values[i] /= 4
print(q_values)
