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
<<<<<<< HEAD
from copy import copy, deepcopy

import pickle

from transform import Transform, Identity, Rotate90, Flip
from board import Board, BoardCache, CELL_O

# In[1]:
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

def get_position_value(board):
    # check if the board position is already cached
    cached_position_value, found = minimax_cache.get_for_position(board)
=======
from copy import copy

from board import Board, BoardCache
from board import CELL_O


# In[2]:

def get_position_value(board):
    # check if the board position is already cached
    cached_position_value, found = cache.get_for_position(board)
>>>>>>> NewBranch
    if found:
        return cached_position_value[0]

    position_value = calculate_position_value(board)

    # cache the board position (if it isn't already cached)
<<<<<<< HEAD
    minimax_cache.set_for_position(board, position_value)
=======
    cache.set_for_position(board, position_value)
>>>>>>> NewBranch

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

<<<<<<< HEAD
# In[2]:
'''
    Tabular Q-Training
'''
def Q_Training(board, R_tie=0, R_win=1, R_loss=-1, record_q_vals=False):
    while not board.is_gameover():
        (q_values, trans_f), found = qtable1.get_for_position(board)
        if record_q_vals:
            if board == start_board:
                df_q_values.iloc[i] = q_values.flatten()

        # choose move
        if (random.uniform(0, 1) > EPSILON): # choose a random move
            move_ind = random.randrange(9)
        else:
            max_index = np.where(q_values == max(q_values))[0]
            move_ind  = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # play move
        board = Board(trans_f.transform(board.board_2d).flatten())
        board = board.play_move(move_ind)

        # Update QTable
        R = 0
        if board.is_gameover():
            R = board.get_game_result()
            Q_prime = np.zeros(9)

        else:
            board = Board(trans_f.reverse(board.board_2d).flatten())
            board_prime = board.play_move(random.choice(board.get_valid_move_indexes()))

            if board_prime.is_gameover():           # need to account for fact that the opponent's move could
                R = board_prime.get_game_result()   # result in a loss
                Q_prime = np.zeros(9)

            else:
                (Q_prime, T_prime), found = qtable1.get_for_position(board_prime)

        if board.is_gameover() or board_prime.is_gameover():
            if R == 0:
                R = R_tie
            elif R == 1:
                R = R_win
            elif R == -1:
                R = R_loss


        # Q(s,a) = Q(s,a) + α[r + γ max(Q') - Q]
        q_values[move_ind] += ALPHA * (R + GAMMA*max(Q_prime) - q_values[move_ind])

        # O move
        if board.is_gameover():
            continue
        else:
            rand_move = random.choice(board.get_valid_move_indexes())
            board = board.play_move(rand_move)



def Q_Testing(start_board):
    for j in range(num_test_games):
        board = start_board
        while not board.is_gameover():
            (q_values, trans_f), found = qtable1.get_for_position(board)

            max_index = np.where(q_values == max(q_values))[0]
            move_ind  = random.choice(max_index)

            # play X move
            board = Board(trans_f.transform(board.board_2d).flatten())
            board = board.play_move(move_ind)

            # No updates
            # O move
            if board.is_gameover():
                continue
            else:
                board = Board(trans_f.reverse(board.board_2d).flatten())
                rand_move = random.choice(board.get_valid_move_indexes())
                board = board.play_move(rand_move)

        results.append(board.get_game_result())

# In[2]:
'''
    Double Q-Learning "Sub"-Routines
'''
def DoubleQ_Training(board, R_tie=0, R_win=1, R_loss=-1, record_q_vals=False):
    while not board.is_gameover():
        (q_values_A, trans_f), found = qtableA.get_for_position(board)
        (q_values_B, trans_f), found = qtableB.get_for_position(board)
        if record_q_vals:
            if board == start_board:
                df_q_values_A.iloc[i] = q_values_A.flatten()
                df_q_values_B.iloc[i] = q_values_B.flatten()

        q_values = (q_values_A + q_values_B)/2

        # choose move
        # this should be modified to only be moves that produce unique boards
        if (random.uniform(0, 1) > EPSILON): # choose a random move
            move_ind = random.randrange(9)
        else:
            max_val   = max(q_values)
            max_index = np.where(q_values == max_val)[0]
            move_ind  = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # play move after transforming board - then transform back
        board = Board(trans_f.transform(board.board_2d).flatten())
        board = board.play_move(move_ind)

        # Update QTable
        if random.uniform(0,1) < 0.5:   # randomly choose which q-table gets updated
            q_vals_update  = q_values_A
            q_table_update = qtableA
            q_table_target = qtableB
        else:
            q_vals_update  = q_values_B
            q_table_update = qtableB
            q_table_target = qtableA

        R = 0
        if board.is_gameover():
            R = board.get_game_result()
            Q_prime_target = np.zeros(9)
            move_prime = 0
            # q_vals_update[move_ind] += ALPHA * (R - q_vals_update[move_ind])
        else:
            board = Board(trans_f.reverse(board.board_2d).flatten())
            board_prime = board.play_move(random.choice(board.get_valid_move_indexes()))

            if board_prime.is_gameover():
                R = board_prime.get_game_result()
                Q_prime_target = np.zeros(9)
                move_prime = 0 # just a dummy index
            else:
                (Q_prime_update, T_prime), found = q_table_update.get_for_position(board_prime)
                (Q_prime_target, T_prime), found = q_table_target.get_for_position(board_prime)

                max_val   = max(Q_prime_update)
                max_index = np.where(Q_prime_update == max_val)[0]
                move_prime  = random.choice(max_index)

        if board.is_gameover() or board_prime.is_gameover():
            if R == 0:
                R = R_tie
            elif R == 1:
                R = R_win
            elif R == -1:
                R = R_loss

        q_vals_update[move_ind] += ALPHA * (R + GAMMA * Q_prime_target[move_prime] -
                                        q_vals_update[move_ind])
=======
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

print("Number of symmetry unique board positions: \n", len(cache.cache))
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
>>>>>>> NewBranch

        # O move
        if board.is_gameover():
            continue
        else:
<<<<<<< HEAD
            rand_move = random.choice(board.get_valid_move_indexes())
            board = board.play_move(rand_move)

def DoubleQ_Testing(start_board):
    for j in range(num_test_games):
        board = start_board
        while not board.is_gameover():
            (q_values_A, trans_f), found = qtableA.get_for_position(board)
            (q_values_B, trans_f), found = qtableB.get_for_position(board)

            q_values = (q_values_A + q_values_B)/2

            max_index = np.where(q_values == max(q_values))[0]
            move_ind  = random.choice(max_index)

            # play X move
            board = Board(trans_f.transform(board.board_2d).flatten())
            board = board.play_move(move_ind)

            # No updates
            # O move
            if board.is_gameover():
                continue
            else:
                board = Board(trans_f.reverse(board.board_2d).flatten())
                rand_move = random.choice(board.get_valid_move_indexes())
                board = board.play_move(rand_move)

        results_doubleQ.append(board.get_game_result())

def Plot_DoubleQ_Results_Values():
    for i in range(int(len(results_doubleQ) / num_test_games)):
        begin = i*num_test_games
        end = begin + num_test_games
        subresults = results_doubleQ[begin:end]
        x = i*test_interval
        if i == 0:
            ax[1].plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = 'win')
            ax[1].plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = 'tie')
            ax[1].plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = 'loss')
        else:
            ax[1].plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = '_nolegend_')
            ax[1].plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = '_nolegend_')
            ax[1].plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = '_nolegend_')

    # ax[1].set_ylim(-3, 80)
    ax[1].set_xlabel('Number of Training Games')
    ax[1].set_ylabel('Percentage of Test Game Results')
    ax[1].legend(ncol=3, loc='upper right', bbox_to_anchor=(0.95,0.95), fontsize=13)


    colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'black', 'grey']
    markers = ['o', 's', 'd', '^', '<', '>', 'x', '.', 'h']
    for key in df_q_values_A.keys():
        ax[2].plot(df_q_values_A[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))
        ax[3].plot(df_q_values_B[key], linestyle='None', marker=markers[key], color=colors[key])

    handles, labels = ax[2].get_legend_handles_labels()
    order = [0,3,6,1,4,7,2,5,8]
    ax[2].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                    ncol=3, loc="upper right", bbox_to_anchor=(0.95,0.5), fontsize=13)
    #
    # ax[2].set_ylim([-1.1, 0.9])
    # ax[3].set_ylim([-1.1, 0.9])

    ax[2].set_ylabel('Q-Values A')

    ax[3].set_ylabel('Q-Values B')
    ax[2].set_xlabel('Number of Training Games')
    ax[3].set_xlabel('Number of Training Games')

    fig.tight_layout()

# In[3]:
minimax_cache = BoardCache()
print(minimax_cache.cache)
board = Board()
get_position_value(board)

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

=======
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
>>>>>>> NewBranch


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

<<<<<<< HEAD
initial_qtable = deepcopy(qtable1)

print(len(qtable1.cache))

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

# In[22]:
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

qtable1 = deepcopy(initial_qtable)

EPSILON = 0.7
GAMMA = 0.9
ALPHA = 0.1
number_games = 20001
num_test_games = 100

df_q_values = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
start_board = Board()

results = []

R_win, R_tie, R_loss = [1, 0, -1]

for i in range(number_games):
    board = start_board
    Q_Training(board, R_win=R_win, R_tie=R_tie, R_loss=R_loss, record_q_vals=True)

    # every 100 training games do a set of test games
    if i % 100 == 0:
        Q_Testing( Board() )

# In[24]:

ax, fig = axes(fig_number=1, rows=1, columns=2)

for i in range(int(len(results) / num_test_games)):
    begin = i*num_test_games
    end = begin + num_test_games
    subresults = results[begin:end]

    ax[1].plot(i*100, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go')
    ax[1].plot(i*100, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks')
    ax[1].plot(i*100, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^')

ax[1].set_xlabel('Number of Training Games')
ax[1].set_ylabel('Percentage of Test Game Results')
ax[1].legend(['Win', 'Tie', 'Loss'], ncol=3, loc="right", bbox_to_anchor=(0.99,0.3), fontsize=13)

for key in df_q_values.keys():
    ax[2].plot(df_q_values[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))

ax[2].set_xlabel('Number of Training Games')
ax[2].set_ylabel('Q-Values')

handles, labels = ax[2].get_legend_handles_labels()
order = [0,3,6,1,4,7,2,5,8]
ax[2].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                ncol=3, loc="lower right", bbox_to_anchor=(0.99,0.08), fontsize=13)

fig.tight_layout()
fig.text(x=0.5, y=0.98, s='R_win = {}; R_tie = {}; R_loss = {}'.format(R_win, R_tie, R_loss), ha = 'center', fontsize=20, fontweight='bold')
fig.savefig('Images/Q_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}.png'.format(R_win, R_tie, abs(R_loss)))



# In[25]
qtable1 = deepcopy(initial_qtable)

EPSILON = 0.7
GAMMA = 0.9
ALPHA = 0.1
number_games = 20001
num_test_games = 100
df_q_values = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])

results = []
start_board = Board()

R_win, R_tie, R_loss = [1, 0, -2]

for i in range(number_games):
    board = start_board
    Q_Training(board, R_win=R_win, R_tie=R_tie, R_loss=R_loss, record_q_vals=True)

    # every 100 training games do a set of test games
    if i % 100 == 0:
        Q_Testing( Board() )

# In[24]:

ax, fig = axes(fig_number=1, rows=1, columns=2)

for i in range(int(len(results) / num_test_games)):
    begin = i*num_test_games
    end = begin + num_test_games
    subresults = results[begin:end]

    ax[1].plot(i*100, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go')
    ax[1].plot(i*100, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks')
    ax[1].plot(i*100, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^')

ax[1].set_xlabel('Number of Training Games')
ax[1].set_ylabel('Percentage of Test Game Results')
ax[1].legend(['Win', 'Tie', 'Loss'], ncol=3, loc="right", bbox_to_anchor=(0.99,0.2), fontsize=13)

for key in df_q_values.keys():
    ax[2].plot(df_q_values[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))

ax[2].set_xlabel('Number of Training Games')
ax[2].set_ylabel('Q-Values')

handles, labels = ax[2].get_legend_handles_labels()
order = [0,3,6,1,4,7,2,5,8]
ax[2].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                ncol=3, loc="lower right", bbox_to_anchor=(0.99,0.07), fontsize=13)

fig.tight_layout()
fig.text(x=0.5, y=0.98, s='R_win = {}; R_tie = {}; R_loss = {}'.format(R_win, R_tie, R_loss), ha = 'center', fontsize=20, fontweight='bold')
fig.savefig('Images/Q_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}.png'.format(R_win, R_tie, abs(R_loss)))

# In[28]:
'''
    Board Positions of Interest:
    [[ 1 -1  1]
     [ 0 -1  0]
     [ 0  0  0]]

    [[ 0  0  1]
     [ 0 -1  1]
     [ 1 -1 -1]]

    [[-1  0  1]
     [ 0 -1  0]
     [ 1  0  0]]

    [[ 0  0  1]
     [-1  0 -1]
     [ 1  0  0]]

'''

# In[27]:
'''
[[ 1 -1  1]
 [ 0 -1  0]
 [ 0  0  0]]
[-0.3439             -0.271              -0.271
  0.                 -0.19                0.
  0.5048772746704796  0.                  0.0373670463519   ]

Obviously this is problematic because not block the O's (-1's) will frequently lead to a loss, but let's see if by playing more games from this position, the q-table can be updated (or is it already saturated).

Actually, it should be obviously that this table is underdeveloped because playing invalid positions should produce the result of -1, but it is not, which means this table is not saturated

It could be that against a random player, it is better not to block
'''

# In[27]:
qtable1 = deepcopy(initial_qtable)

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 10001
num_test_games = 100
test_interval = 100

df_q_values = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])

results = []

board = Board()
board = board.play_move(0)
board = board.play_move(4)
board = board.play_move(2)
board = board.play_move(1)
start_board = board

R_win, R_tie, R_loss = [1, 0, -2]

for i in range(number_games):
    board = start_board
    Q_Training(board, R_win=R_win, R_tie=R_tie, R_loss=R_loss, record_q_vals=True)

    # every 100 training games do a set of test games
    if i % test_interval == 0:
        Q_Testing(start_board)

# In[28]:
ax, fig = axes(fig_number=1, rows=1, columns=2)

for i in range(int(len(results) / num_test_games)):
    begin = i*num_test_games
    end = begin + num_test_games
    subresults = results[begin:end]
    x = i*test_interval

    if i == 0:
        ax[1].plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = 'win')
        ax[1].plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = 'tie')
        ax[1].plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = 'loss')
    else:
        ax[1].plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = '_nolegend_')
        ax[1].plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = '_nolegend_')
        ax[1].plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = '_nolegend_')

ax[1].set_xlabel('Number of Training Games')
ax[1].set_ylabel('Percentage of Test Game Results')
ax[1].legend(ncol=3, loc='upper right', bbox_to_anchor=(0.95,0.95), fontsize=13)



colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'black', 'grey']
markers = ['o', 's', 'd', '^', '<', '>', 'x', '.', 'h']
for key in df_q_values.keys():
    ax[2].plot(df_q_values[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))


ax[2].set_xlabel('Number of Training Games')
ax[2].set_ylabel('Q-Values')
# ax[2].set_ylim([-1.05, 1.05])

handles, labels = ax[2].get_legend_handles_labels()
order = [0,3,6,1,4,7,2,5,8]
ax[2].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                ncol=3, loc="upper right", bbox_to_anchor=(0.95,0.5), fontsize=13)

fig.tight_layout()
fig.text(x=0.5, y=0.98, s='R_win = {}; R_tie = {}; R_loss = {}'.format(R_win, R_tie, R_loss), ha = 'center', fontsize=20, fontweight='bold')
fig.savefig('Images/Q_Learning_Simple_Board_R_win_{}__R_tie_{}__R_loss_{}.png'.format(R_win, R_tie, abs(R_loss)), format='png', dpi=300, bbox_inches = "tight")


# In[100]:
'''
    Now let's see what happens is we use Double Q-Learning...
'''
qtableA = deepcopy(initial_qtable)
qtableB = deepcopy(initial_qtable)

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 15001
test_interval = 100 # every 200 training games, run a test
num_test_games = 100

results_doubleQ = []

df_q_values_A = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
df_q_values_B = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])

board = Board()
board = board.play_move(0)
board = board.play_move(4)
board = board.play_move(2)
board = board.play_move(1)
start_board = board

for i in range(number_games):
    board = start_board
    DoubleQ_Training(board, R_tie=0, R_loss=-2, record_q_vals=True)

    # every xxx training games do a set of test games
    if i % test_interval == 0:
        DoubleQ_Testing(start_board)

# In[101]:
ax, fig = axes(fig_number=1, rows=1, columns=3)
Plot_DoubleQ_Results_Values()
# fig.savefig('Images/DoubleQ_Results_Values.png', format='png', dpi=300, bbox_inches = "tight")

# In[100]:
'''
    Now let's see what happens is we use Double Q-Learning...
'''
qtableA = deepcopy(initial_qtable)
qtableB = deepcopy(initial_qtable)

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 10001
test_interval = 100 # every 200 training games, run a test
num_test_games = 100

results_doubleQ = []

df_q_values_A = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
df_q_values_B = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])

board = Board()
board = board.play_move(0)
board = board.play_move(1)
board = board.play_move(2)
board = board.play_move(4)
board = board.play_move(7)
board = board.play_move(6)
start_board = board

for i in range(number_games):
    board = start_board
    DoubleQ_Training(board, R_tie=0, R_loss=-1, record_q_vals=True)

    # every xxx training games do a set of test games
    if i % test_interval == 0:
        DoubleQ_Testing(start_board)

# In[101]:
print(start_board.board_2d)
ax, fig = axes(fig_number=1, rows=1, columns=3)
Plot_DoubleQ_Results_Values()
# fig.savefig('Images/DoubleQ_Results_Values.png', format='png', dpi=300, bbox_inches = "tight")

# In[28]:
'''
    At steady-state will produce 0 losses, but can swicht between 80% ties and 20% wins to 50-50
'''
ax, fig = axes(fig_number=1, rows=2, columns=3)

for i in range(int(len(results) / num_test_games)):
    begin = i*num_test_games
    end = begin + num_test_games
    subresults = results[begin:end]
    x = i * test_interval

    if i == 0:
        ax[1].plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = 'win')
        ax[1].plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = 'tie')
        ax[1].plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = 'loss')
    else:
        ax[1].plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = '_nolegend_')
        ax[1].plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = '_nolegend_')
        ax[1].plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = '_nolegend_')

ax[1].set_xlabel('Number of Training Games')
ax[1].set_ylabel('Percentage of Test Game Results')
ax[1].legend(ncol=3, loc='upper right', bbox_to_anchor=(0.95,0.95), fontsize=13)



colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'black', 'grey']
markers = ['o', 's', 'd', '^', '<', '>', 'x', '.', 'h']
for key in df_q_values.keys():
    ax[2].plot(df_q_values[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))
    ax[3].plot(df_q_values[key], linestyle='None', marker=markers[key], color=colors[key])

    ax[5].plot(df_q_values_1[key], linestyle='None', marker=markers[key], color=colors[key])
    ax[6].plot(df_q_values_2[key], linestyle='None', marker=markers[key], color=colors[key])


ax[2].set_xlabel('Number of Training Games')
ax[2].set_ylabel('Q-Values')
ax[2].set_ylim([-1.05, 1.05])

handles, labels = ax[2].get_legend_handles_labels()
order = [0,3,6,1,4,7,2,5,8]
ax[2].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                ncol=3, loc="upper right", bbox_to_anchor=(0.95,0.5), fontsize=13)

ax[3].set_ylabel('Q-Values')
# ax[3].set_xlim([4500, 10000])
ax[3].set_ylim([0, 1])
ax[3].set_xlabel('Number of Training Games')

print(trans_f_1)
print(trans_f_2)
print(trans_f_1.transform(board_1.board_2d))
print(trans_f_2.transform(board_2.board_2d))

fig.tight_layout()
fig.savefig('Images/Q_Learning_Results_Values_TiesWins_1_Delete.png', format='png', dpi=300, bbox_inches = "tight")

# In[40]:
qtable1 = deepcopy(initial_qtable)

EPSILON = 0.7
GAMMA = 0.9
ALPHA = 0.4
number_games = 2001
num_test_games = 100
test_interval = 20

df_q_values = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])

results = []

board = Board()
board = board.play_move(0)
board = board.play_move(4)
board = board.play_move(2)
board = board.play_move(1)
board = board.play_move(7)
board = board.play_move(6)
start_board = board

(q_values, trans_f), found = qtable1.get_for_position(board)
# q_values[3] = 1.0

for i in range(number_games):
    board = start_board

    while not board.is_gameover():
        (q_values, trans_f), found = qtable1.get_for_position(board)
        if board == start_board:
            df_q_values.iloc[i] = q_values.flatten()
=======
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

def train_q_learn():
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
    pass

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
>>>>>>> NewBranch

        # choose move
        if (random.uniform(0, 1) > EPSILON): # choose a random move
            move_ind = random.randrange(9)
        else:
            max_index = np.where(q_values == max(q_values))[0]
            move_ind  = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # play move
        board = Board(trans_f.transform(board.board_2d).flatten())
        board = board.play_move(move_ind)
<<<<<<< HEAD
=======
        board = Board(trans_f.reverse(board.board_2d).flatten())
>>>>>>> NewBranch

        # Update QTable
        R = 0
        if board.is_gameover():
            R = board.get_game_result()
<<<<<<< HEAD
            # if R == 0:
            #     R = 1
            Q_prime = np.zeros(9)
        else:
            board = Board(trans_f.reverse(board.board_2d).flatten())
=======
            Q_prime = np.zeros(9)
        else:
>>>>>>> NewBranch
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
<<<<<<< HEAD
    if i % test_interval == 0:

        for j in range(num_test_games):
            board = start_board
=======
    if i % 100 == 0:

        for j in range(num_test_games):
            board = Board()
>>>>>>> NewBranch
            while not board.is_gameover():
                (q_values, trans_f), found = qtable1.get_for_position(board)

                max_index = np.where(q_values == max(q_values))[0]
                move_ind  = random.choice(max_index)

                # play X move
                board = Board(trans_f.transform(board.board_2d).flatten())
                board = board.play_move(move_ind)
<<<<<<< HEAD
=======
                board = Board(trans_f.reverse(board.board_2d).flatten())
>>>>>>> NewBranch

                # No updates
                # O move
                if board.is_gameover():
                    continue
                else:
<<<<<<< HEAD
                    board = Board(trans_f.reverse(board.board_2d).flatten())
=======
>>>>>>> NewBranch
                    rand_move = random.choice(board.get_valid_move_indexes())
                    board = board.play_move(rand_move)

            results.append(board.get_game_result())

<<<<<<< HEAD
# In[41]:
print(start_board.board_2d)

ax, fig = axes(fig_number=1, rows=1, columns=2)
=======
# In[24]:

ax, fig = axes(fig_number=1, rows=1, columns=1)
>>>>>>> NewBranch

for i in range(int(len(results) / num_test_games)):
    begin = i*num_test_games
    end = begin + num_test_games
    subresults = results[begin:end]
<<<<<<< HEAD
    x = i * test_interval

    if i == 0:
        ax[1].plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = 'win')
        ax[1].plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = 'tie')
        ax[1].plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = 'loss')
    else:
        ax[1].plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = '_nolegend_')
        ax[1].plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = '_nolegend_')
        ax[1].plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = '_nolegend_')

ax[1].set_xlabel('Number of Training Games')
ax[1].set_ylabel('Percentage of Test Game Results')
ax[1].legend(ncol=3, loc='upper right', bbox_to_anchor=(0.95,0.95), fontsize=13)


colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'black', 'grey']
markers = ['o', 's', 'd', '^', '<', '>', 'x', '.', 'h']
for key in df_q_values.keys():
    ax[2].plot(df_q_values[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))

ax[2].set_xlabel('Number of Training Games')
ax[2].set_ylabel('Q-Values')
ax[2].set_ylim([-1.05, 1.05])

handles, labels = ax[2].get_legend_handles_labels()
order = [0,3,6,1,4,7,2,5,8]
ax[2].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                ncol=3, loc="upper right", bbox_to_anchor=(0.95,0.5), fontsize=13)


fig.tight_layout()
fig.savefig('Images/QLearning_R_Zero.png', format='png', dpi=300, bbox_inches = "tight")
# fig.savefig('Images/QLearning_50_50.png', format='png', dpi=300, bbox_inches = "tight")


# In[40]:
'''
    [[ X  O  X]
     [ -  O  -]
     [ O  X  -]]

    Double Q-Learning
'''
qtableA = deepcopy(initial_qtable)
qtableB = deepcopy(initial_qtable)

EPSILON = 0.7
GAMMA = 0.9
ALPHA = 0.4
number_games = 4001
num_test_games = 100
test_interval = 40

df_q_values_A = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
df_q_values_B = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])

results = []

board = Board()
board = board.play_move(0)
board = board.play_move(4)
board = board.play_move(2)
board = board.play_move(1)
board = board.play_move(7)
board = board.play_move(6)
start_board = board

(q_values_A, trans_f), found = qtableA.get_for_position(board)
(q_values_B, trans_f), found = qtableB.get_for_position(board)
# q_values_A[3] = 1.0
# q_values_B[3] = 1.0

for i in range(number_games):
    board = start_board

    while not board.is_gameover():
        (q_values_A, trans_f), found = qtableA.get_for_position(board)
        (q_values_B, trans_f), found = qtableB.get_for_position(board)
        if board == start_board:
            df_q_values_A.iloc[i] = q_values_A.flatten()
            df_q_values_B.iloc[i] = q_values_B.flatten()

        q_values = (q_values_A + q_values_B)/2
=======

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
>>>>>>> NewBranch

        # choose move
        if (random.uniform(0, 1) > EPSILON): # choose a random move
            move_ind = random.randrange(9)
        else:
            max_index = np.where(q_values == max(q_values))[0]
            move_ind  = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # play move
        board = Board(trans_f.transform(board.board_2d).flatten())
        board = board.play_move(move_ind)
<<<<<<< HEAD

        # Update QTable
        if random.uniform(0,1) < 0.5:   # randomly choose which q-table gets updated
            q_vals_update  = q_values_A
            q_table_update = qtableA
            q_table_target = qtableB
            # print('Update A')
        else:
            q_vals_update  = q_values_B
            q_table_update = qtableB
            q_table_target = qtableA

        R = 0
        if board.is_gameover():
            R = board.get_game_result()
            if R == 0:
                R = 1
            Q_prime = np.zeros(9)
            q_vals_update[move_ind] += ALPHA * (R - q_vals_update[move_ind])
        else:
            board = Board(trans_f.reverse(board.board_2d).flatten())
            board_prime = board.play_move(random.choice(board.get_valid_move_indexes()))
            (Q_prime_update, T_prime), found = q_table_update.get_for_position(board_prime)
            (Q_prime_target, T_prime), found = q_table_target.get_for_position(board_prime)

            max_val     = max(Q_prime_update)
            max_index   = np.where(Q_prime_update == max_val)[0]
            move_prime  = random.choice(max_index)

            q_vals_update[move_ind] += ALPHA * (R + GAMMA * Q_prime_target[move_prime] - q_vals_update[move_ind])
=======
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
>>>>>>> NewBranch

        # O move
        if board.is_gameover():
            continue
        else:
<<<<<<< HEAD
            rand_move = random.choice(board.get_valid_move_indexes())
            board = board.play_move(rand_move)

    # every 100 training games do a set of test games
    if i % test_interval == 0:

        for j in range(num_test_games):
            board = start_board
            while not board.is_gameover():
                (q_values_A, trans_f), found = qtableA.get_for_position(board)
                (q_values_B, trans_f), found = qtableB.get_for_position(board)

                q_values = (q_values_A + q_values_B)/2
=======
            minimax_move = mini_max_strategy(board)
            board = board.play_move(minimax_move)

    # every 100 training games do a set of test games
    if i % 100 == 0:

        for j in range(num_test_games):
            board = Board()
            while not board.is_gameover():
                (q_values, trans_f), found = qtable1.get_for_position(board)
>>>>>>> NewBranch

                max_index = np.where(q_values == max(q_values))[0]
                move_ind  = random.choice(max_index)

                # play X move
                board = Board(trans_f.transform(board.board_2d).flatten())
                board = board.play_move(move_ind)
<<<<<<< HEAD
=======
                board = Board(trans_f.reverse(board.board_2d).flatten())
>>>>>>> NewBranch

                # No updates
                # O move
                if board.is_gameover():
                    continue
                else:
<<<<<<< HEAD
                    board = Board(trans_f.reverse(board.board_2d).flatten())
=======
>>>>>>> NewBranch
                    rand_move = random.choice(board.get_valid_move_indexes())
                    board = board.play_move(rand_move)

            results.append(board.get_game_result())

<<<<<<< HEAD

print(start_board.board_2d)

ax, fig = axes(fig_number=1, rows=1, columns=3)
=======
# In[26]:

ax, fig = axes(fig_number=1, rows=1, columns=1)
>>>>>>> NewBranch

for i in range(int(len(results) / num_test_games)):
    begin = i*num_test_games
    end = begin + num_test_games
    subresults = results[begin:end]
<<<<<<< HEAD
    x = i * test_interval

    if i == 0:
        ax[1].plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = 'win')
        ax[1].plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = 'tie')
        ax[1].plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = 'loss')
    else:
        ax[1].plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = '_nolegend_')
        ax[1].plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = '_nolegend_')
        ax[1].plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = '_nolegend_')

ax[1].set_xlabel('Number of Training Games')
ax[1].set_ylabel('Percentage of Test Game Results')
ax[1].legend(ncol=3, loc='upper right', bbox_to_anchor=(0.95,0.95), fontsize=13)


colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'black', 'grey']
markers = ['o', 's', 'd', '^', '<', '>', 'x', '.', 'h']
for key in df_q_values_A.keys():
    ax[2].plot(df_q_values_A[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))
    ax[3].plot(df_q_values_B[key], linestyle='None', marker=markers[key], color=colors[key])

ax[2].set_xlabel('Number of Training Games')
ax[2].set_ylabel('Q-Values')
ax[2].set_ylim([-1.05, 1.05])

handles, labels = ax[2].get_legend_handles_labels()
order = [0,3,6,1,4,7,2,5,8]
ax[2].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                ncol=3, loc="upper right", bbox_to_anchor=(0.99,0.4), fontsize=13)


fig.tight_layout()
# fig.savefig('Images/QLearning_R_Zero.png', format='png', dpi=300, bbox_inches = "tight")
# fig.savefig('Images/QLearning_50_50.png', format='png', dpi=300, bbox_inches = "tight")

# In[401]:
(q_values_A, trans_f), found = qtableA.get_for_position(start_board)
(q_values_B, trans_f), found = qtableB.get_for_position(start_board)

print(q_values_A)
print(q_values_B)


# In[50]:
'''
    Now let's see what happens is we use Double Q-Learning...

    Sometimes it can take a long time to reach steady-state, but it seems that 20,000 training games is sufficient

    Double Q-Learning seems to always converge to the more optimal solution: 0% Losses, 50% Ties, 50% Wins
'''
qtableA = deepcopy(initial_qtable)
qtableB = deepcopy(initial_qtable)

EPSILON = 0.7
GAMMA = 0.9
ALPHA = 0.4
number_games = 5001
test_interval = 50 # every 200 training games, run a test
num_test_games = 100

results_doubleQ = []

df_q_values_A = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
df_q_values_B = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])

board = Board()
board = board.play_move(2)
board = board.play_move(4)
board = board.play_move(8)
board = board.play_move(5)
start_board = board

for i in range(number_games):
    board = start_board

    DoubleQ_Training(board, R_tie=1, record_q_vals=True)


    # every 100 training games do a set of test games
    if i % test_interval == 0:

        DoubleQ_Testing()

# In[31]:
ax, fig = axes(fig_number=1, rows=1, columns=3)
Plot_DoubleQ_Results_Values()

# fig.savefig('Images/DoubleQ_Results_Values_TiesWins_1.png', format='png', dpi=300, bbox_inches = "tight")



# In[50]:
'''
    Now let's see what happens is we use Double Q-Learning...
'''

qtableA = deepcopy(initial_qtable)
qtableB = deepcopy(initial_qtable)
=======

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
>>>>>>> NewBranch

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
<<<<<<< HEAD
number_games = 10000

df_q_values_A = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
df_q_values_B = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])

board = Board()
board = board.play_move(2)
board = board.play_move(4)
board = board.play_move(8)
board = board.play_move(5)
start_board = board

for i in range(number_games):
    board = start_board
=======
number_games = 10001
num_test_games = 500

results = []

for i in range(number_games):
    board = Board()
>>>>>>> NewBranch

    while not board.is_gameover():
        (q_values_A, trans_f), found = qtableA.get_for_position(board)
        (q_values_B, trans_f), found = qtableB.get_for_position(board)
<<<<<<< HEAD
        if board == start_board:
            df_q_values_A.iloc[i] = q_values_A.flatten()
            df_q_values_B.iloc[i] = q_values_B.flatten()
=======
>>>>>>> NewBranch

        q_values = (q_values_A + q_values_B)/2

        # choose move
<<<<<<< HEAD
        # this should be modified to only be moves that produce unique boards
=======
>>>>>>> NewBranch
        if (random.uniform(0, 1) > EPSILON): # choose a random move
            move_ind = random.randrange(9)
        else:
            max_val   = max(q_values)
            max_index = np.where(q_values == max_val)[0]
            move_ind  = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # play move after transforming board - then transform back
        board = Board(trans_f.transform(board.board_2d).flatten())
        board = board.play_move(move_ind)
<<<<<<< HEAD

        # Update QTable
        if random.uniform(0,1) < 0.5:   # randomly choose which q-table gets updated
            q_vals_update  = q_values_A
            q_table_update = qtableA
            q_table_target = qtableB
        else:
            q_vals_update  = q_values_B
=======
        board = Board(trans_f.reverse(board.board_2d).flatten())

        # Update QTable
        if random.uniform(0,1) < 0.5:   # randomly choose which q-table gets updated
            q_vals_update = q_values_A
            q_table_update = qtableA
            q_table_target = qtableB
        else:
            q_vals_update = q_values_B
>>>>>>> NewBranch
            q_table_update = qtableB
            q_table_target = qtableA

        R = 0
        if board.is_gameover():
            R = board.get_game_result()
<<<<<<< HEAD
            if R == 0:
                R = 1
            Q_prime = np.zeros(9)
            q_vals_update[move_ind] += ALPHA * (R - q_vals_update[move_ind])
        else:
            board = Board(trans_f.reverse(board.board_2d).flatten())
            board_prime = board.play_move(random.choice(board.get_valid_move_indexes()))
            (Q_prime_update, T_prime), found = q_table_update.get_for_position(board_prime)
            (Q_prime_target, T_prime), found = q_table_target.get_for_position(board_prime)

            max_val   = max(Q_prime_update)
            max_index = np.where(Q_prime_update == max_val)[0]
            move_prime  = random.choice(max_index)

            q_vals_update[move_ind] += ALPHA * (R + GAMMA * Q_prime_target[move_prime] - q_vals_update[move_ind])
=======
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
>>>>>>> NewBranch

        # O move
        if board.is_gameover():
            continue
        else:
            rand_move = random.choice(board.get_valid_move_indexes())
            board = board.play_move(rand_move)

<<<<<<< HEAD
# In[31]:
(q_values_A, trans_f), found = qtableA.get_for_position(start_board)
(q_values_B, trans_f), found = qtableB.get_for_position(start_board)
print(trans_f.transform(start_board.board_2d))
print(q_values_A)
print(q_values_B)

ax, fig = axes(fig_number=1, rows=1, columns=2)

colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'black', 'grey']
markers = ['o', 's', 'd', '^', '<', '>', 'x', '.', 'h']
for key in df_q_values_A.keys():
    ax[1].plot(df_q_values_A[key], marker=markers[key], color=colors[key])
    ax[2].plot(df_q_values_B[key], marker=markers[key], color=colors[key])

ax[1].set_ylim([-1.1, 0.9])
ax[2].set_ylim([-1.1, 0.9])

ax[1].set_ylabel('Q-Values A')

ax[2].set_ylabel('Q-Values B')
ax[1].set_xlabel('Number of Training Games')
ax[2].set_xlabel('Number of Training Games')

fig.tight_layout()


# In[25]:
'''
    Train against minimax player
    Test against Random player
'''

qtable1 = deepcopy(initial_qtable)
print(len(qtable1.cache))
=======
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
>>>>>>> NewBranch

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 10001
num_test_games = 500

results = []

for i in range(number_games):
    board = Board()

    while not board.is_gameover():
<<<<<<< HEAD
        (q_values, trans_f), found = qtable1.get_for_position(board)
=======
        (q_values_A, trans_f), found = qtableA.get_for_position(board)
        (q_values_B, trans_f), found = qtableB.get_for_position(board)

        q_values = (q_values_A + q_values_B)/2
>>>>>>> NewBranch

        # choose move
        if (random.uniform(0, 1) > EPSILON): # choose a random move
            move_ind = random.randrange(9)
        else:
<<<<<<< HEAD
            max_index = np.where(q_values == max(q_values))[0]
            move_ind  = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # play move
=======
            max_val   = max(q_values)
            max_index = np.where(q_values == max_val)[0]
            move_ind  = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # play move after transforming board - then transform back
>>>>>>> NewBranch
        board = Board(trans_f.transform(board.board_2d).flatten())
        board = board.play_move(move_ind)
        board = Board(trans_f.reverse(board.board_2d).flatten())

        # Update QTable
<<<<<<< HEAD
=======
        if random.uniform(0,1) < 0.5:   # randomly choose which q-table gets updated
            q_vals_update = q_values_A
            q_table_update = qtableA
            q_table_target = qtableB
        else:
            q_vals_update = q_values_B
            q_table_update = qtableB
            q_table_target = qtableA

>>>>>>> NewBranch
        R = 0
        if board.is_gameover():
            R = board.get_game_result()
            Q_prime = np.zeros(9)
<<<<<<< HEAD
        else:
            # board_prime = board.play_move(random.choice(board.get_valid_move_indexes()))
            board_prime = board.play_move(mini_max_strategy(board))
            (Q_prime, T_prime), found = qtable1.get_for_position(board_prime)

        # Q(s,a) = Q(s,a) + α[r + γ max(Q') - Q]
        q_values[move_ind] += ALPHA * (R + GAMMA*max(Q_prime) - q_values[move_ind])
=======
            q_vals_update[move_ind] += ALPHA * (R - q_vals_update[move_ind])
        else:
            board_prime = board.play_move(random.choice(board.get_valid_move_indexes()))
            (Q_prime_up, T_prime), found = q_table_update.get_for_position(board_prime)
            (Q_prime_tar, T_prime), found = q_table_target.get_for_position(board_prime)

            max_val   = max(Q_prime_up)
            max_index = np.where(Q_prime_up == max_val)[0]
            move_prime  = random.choice(max_index)

            q_vals_update[move_ind] += ALPHA * (R + GAMMA * Q_prime_tar[move_prime] - q_vals_update[move_ind])
>>>>>>> NewBranch

        # O move
        if board.is_gameover():
            continue
        else:
<<<<<<< HEAD
            minimax_move = mini_max_strategy(board)
            board = board.play_move(minimax_move)
=======
            opponent_move = mini_max_strategy(board)
            board = board.play_move(opponent_move)
>>>>>>> NewBranch

    # every 100 training games do a set of test games
    if i % 100 == 0:

        for j in range(num_test_games):
            board = Board()
            while not board.is_gameover():
<<<<<<< HEAD
                (q_values, trans_f), found = qtable1.get_for_position(board)
=======
                (q_values_A, trans_f), found = qtableA.get_for_position(board)
                (q_values_B, trans_f), found = qtableB.get_for_position(board)

                q_values = (q_values_A + q_values_B)/2
>>>>>>> NewBranch

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
<<<<<<< HEAD
=======

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
>>>>>>> NewBranch
