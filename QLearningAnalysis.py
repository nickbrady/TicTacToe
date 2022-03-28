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
colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'brown', 'black', 'grey']
markers = ['o', 's', 'd', '^', '<', '>', 'x', '.', 'h']

import random
import math

import itertools
from copy import copy, deepcopy

import pickle

from transform import Transform, Identity, Rotate90, Flip
from board import Board, BoardCache, CELL_O

def pickle_it(filename, object):
    outfile = open(filename,'wb')
    pickle.dump(object, outfile)
    outfile.close()

def un_pickle_it(filename):
    _file = open(filename, 'rb')
    object = pickle.load(_file)
    _file.close()

    return object

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
    if found:
        return cached_position_value[0]

    position_value = calculate_position_value(board)

    # cache the board position (if it isn't already cached)
    minimax_cache.set_for_position(board, position_value)

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
    '''
        No exploration only exploitation during testing
    '''
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

        # O move
        if board.is_gameover():
            continue
        else:
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
print(board.board_2d)
get_position_value(board)
print(len(minimax_cache.cache))

# In[3]:
'''
    Play some test games
'''

game_results, game_moves_list = play_tic_tac_toe(player_1_strategy=None, player_2_strategy=None,
                                            number_of_games=100)

print(len([r for r in game_results if r == 1]))
print(len([r for r in game_results if r == 0]))
print(len([r for r in game_results if r == -1]))

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


# In[23]:

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 20001
num_test_games = 100

Rewards_Array = [(1,0,-1), (1,0,-2), (1,0,-10)]

for _ in Rewards_Array:
    R_win, R_tie, R_loss = _
    print(R_win, R_tie, R_loss)

    qtable1 = deepcopy(initial_qtable)
    df_q_values = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
    start_board = Board()
    results = []

    print(qtable1.get_for_position(start_board))

    for i in range(number_games):
        board = start_board
        Q_Training(board, R_win=R_win, R_tie=R_tie, R_loss=R_loss, record_q_vals=True)

        # every 100 training games do a set of test games
        if i % 100 == 0:
            Q_Testing( Board() )


    filename = 'QLearningData/Q_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}_Results'.format(R_win, R_tie, abs(R_loss))
    pickle_it(filename, results)

    filename = 'QLearningData/Q_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}_Q_values'.format(R_win, R_tie, abs(R_loss))
    pickle_it(filename, df_q_values)

    filename = 'QLearningData/Q_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}_Full_QTable'.format(R_win, R_tie, abs(R_loss))
    pickle_it(filename, qtable1)

# In[24]:
# Rewards_Array = [(1,0,-1), (1,0,-2), (1,0,-10)]
ax, fig = axes(fig_number=1, rows=len(Rewards_Array), columns=2)

for o, _ in enumerate(Rewards_Array):
    R_win, R_tie, R_loss = _
    print(R_win, R_tie, R_loss)

    filename = 'QLearningData/Q_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}_Results'.format(R_win, R_tie, abs(R_loss))
    results = un_pickle_it(filename)

    filename = 'QLearningData/Q_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}_Q_values'.format(R_win, R_tie, abs(R_loss))
    df_q_values = un_pickle_it(filename)

    ax_ = ax[1 + 2*o]
    for i in range(int(len(results) / num_test_games)):
        begin = i*num_test_games
        end = begin + num_test_games
        subresults = results[begin:end]

        ax_.plot(i*100, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go')
        ax_.plot(i*100, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks')
        ax_.plot(i*100, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^')

    ax_.set_xlabel('Number of Training Games')
    ax_.set_ylabel('Percentage of Test Game Results')
    ax_.legend(['Win', 'Tie', 'Loss'], ncol=3, loc="right", bbox_to_anchor=(0.99,0.3), fontsize=13)
    ax_.set_title('R_win = {}; R_tie = {}; R_loss = {}'.format(R_win, R_tie, R_loss), fontsize=20, fontweight='bold')

    ax_ = ax[2 + 2*o]
    for key in df_q_values.keys():
        ax_.plot(df_q_values[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))

    ax_.set_xlabel('Number of Training Games')
    ax_.set_ylabel('Q-Values')

    handles, labels = ax[2].get_legend_handles_labels()
    order = [0,3,6,1,4,7,2,5,8]
    ax_.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                    ncol=3, loc="lower right", bbox_to_anchor=(0.99,0.08), fontsize=13)


fig.tight_layout()
fig.savefig('Images/Q_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}.png'.format(R_win, R_tie, abs(R_loss)), format='png', dpi=300, bbox_inches = "tight")



# In[27]:
cache_keys = list(qtable1.cache.keys())
X_turn_boards = []
O_turn_boards = []

for k in cache_keys:
    b_ = np.frombuffer(k, dtype=Board().board_2d.dtype)
    c = Board(b_)
    if c.get_game_result() != 2:
        continue
    if c.get_turn() == 1:
        X_turn_boards.append(k)
    else:
        O_turn_boards.append(k)

print(len(X_turn_boards))
print(len(O_turn_boards))



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
[-0.9999 -0.9999 -0.9999
 0.10512 -0.9999 0.22451
 0.08954  0.3935 0.2923]

Obviously this is problematic because not blocking the O's (-1's) will frequently lead to a loss, but let's see if by playing more games from this position, the q-table can be updated (or is it already saturated).

Actually, it should be obvious that this table is underdeveloped because playing invalid positions should produce the result of -1, but it is not, which means this table is not saturated

It could be that against a random player, it is better not to block
'''

# In[27]:
'''
    Simple Board (Not the typical starting board)

    [[ X  O  X]
     [ -  O  -]
     [ -  -  -]]

X to play...
'''

qtable1 = deepcopy(initial_qtable)

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 10001
num_test_games = 100
test_interval = 100


board = Board()
board = board.play_move(0)
board = board.play_move(4)
board = board.play_move(2)
board = board.play_move(1)
start_board = board

Rewards_Array = [(1,0,-1), (1,0,-10)]

for _ in Rewards_Array:
    R_win, R_tie, R_loss = _

    df_q_values = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])

    results = []

    for i in range(number_games):
        board = start_board
        Q_Training(board, R_win=R_win, R_tie=R_tie, R_loss=R_loss, record_q_vals=True)

        # every 100 training games do a set of test games
        if i % test_interval == 0:
            Q_Testing(start_board)

    filename = 'QLearningData/Q_Learning_Simple_Board_R_win_{}__R_tie_{}__R_loss_{}_Results'.format(R_win, R_tie, abs(R_loss))
    pickle_it(filename, results)

    filename = 'QLearningData/Q_Learning_Simple_Board_R_win_{}__R_tie_{}__R_loss_{}_Q_values'.format(R_win, R_tie, abs(R_loss))
    pickle_it(filename, df_q_values)

# In[28]:

ax, fig = axes(fig_number=1, rows=len(Rewards_Array), columns=2)

for o, _ in enumerate(Rewards_Array):
    R_win, R_tie, R_loss = _

    filename = 'QLearningData/Q_Learning_Simple_Board_R_win_{}__R_tie_{}__R_loss_{}_Results'.format(R_win, R_tie, abs(R_loss))
    results = un_pickle_it(filename)

    filename = 'QLearningData/Q_Learning_Simple_Board_R_win_{}__R_tie_{}__R_loss_{}_Q_values'.format(R_win, R_tie, abs(R_loss))
    df_q_values = un_pickle_it(filename)

    for i in range(int(len(results) / num_test_games)):
        begin = i*num_test_games
        end = begin + num_test_games
        subresults = results[begin:end]
        x = i*test_interval

        ax_ = ax[1 + 2*o]
        if i == 0:
            ax_.plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = 'win')
            ax_.plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = 'tie')
            ax_.plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = 'loss')
        else:
            ax_.plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = '_nolegend_')
            ax_.plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = '_nolegend_')
            ax_.plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = '_nolegend_')

    ax_.set_ylim([-5, 105])
    ax_.set_xlabel('Number of Training Games')
    ax_.set_ylabel('Percentage of Test Game Results')
    ax_.legend(ncol=3, loc='upper right', bbox_to_anchor=(0.95,0.95), fontsize=13)
    ax_.set_title('R_win = {}; R_tie = {}; R_loss = {}'.format(R_win, R_tie, R_loss), fontsize=20, fontweight='bold')


    ax_ = ax[2 + 2*o]
    print(df_q_values.iloc[-1].values)
    for key in df_q_values.keys():
        ax_.plot(df_q_values[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))


    ax_.set_xlabel('Number of Training Games')
    ax_.set_ylabel('Q-Values')
    # ax[2].set_ylim([-1.05, 1.05])

    handles, labels = ax[2].get_legend_handles_labels()
    order = [0,3,6,1,4,7,2,5,8]
    ax_.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                    ncol=3, loc="upper right", bbox_to_anchor=(0.95,0.5), fontsize=13)

fig.tight_layout()
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
number_games = 100#15001
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

# R_win, R_tie, R_loss = [1, 0, -2]

for i in range(number_games):
    board = start_board
    DoubleQ_Training(board, R_win=R_win, R_tie=R_tie, R_loss=R_loss, record_q_vals=True)

    # every xxx training games do a set of test games
    if i % test_interval == 0:
        DoubleQ_Testing(start_board)

filename = 'QLearningData/DoubleQ_Learning_Simple_Board_R_win_{}__R_tie_{}__R_loss_{}_Results'.format(R_win, R_tie, abs(R_loss))
pickle_it(filename, results_doubleQ)

filename = 'QLearningData/DoubleQ_Learning_Simple_Board_R_win_{}__R_tie_{}__R_loss_{}_Q_values_A'.format(R_win, R_tie, abs(R_loss))
pickle_it(filename, df_q_values_A)

filename = 'QLearningData/DoubleQ_Learning_Simple_Board_R_win_{}__R_tie_{}__R_loss_{}_Q_values_B'.format(R_win, R_tie, abs(R_loss))
pickle_it(filename, df_q_values_B)

# In[101]:
filename = 'QLearningData/DoubleQ_Learning_Simple_Board_R_win_{}__R_tie_{}__R_loss_{}_Results'.format(R_win, R_tie, abs(R_loss))
results_doubleQ = un_pickle_it(filename)

filename = 'QLearningData/DoubleQ_Learning_Simple_Board_R_win_{}__R_tie_{}__R_loss_{}_Q_values_A'.format(R_win, R_tie, abs(R_loss))
df_q_values_A = un_pickle_it(filename)

filename = 'QLearningData/DoubleQ_Learning_Simple_Board_R_win_{}__R_tie_{}__R_loss_{}_Q_values_B'.format(R_win, R_tie, abs(R_loss))
df_q_values_B = un_pickle_it(filename)

ax, fig = axes(fig_number=1, rows=1, columns=3)
Plot_DoubleQ_Results_Values()

fig.tight_layout()
fig.text(x=0.5, y=0.98, s='R_win = {}; R_tie = {}; R_loss = {}'.format(R_win, R_tie, R_loss), ha = 'center', fontsize=20, fontweight='bold')
fig.savefig('Images/DoubleQ_Learning_Simple_Board_R_win_{}__R_tie_{}__R_loss_{}.png'.format(R_win, R_tie, abs(R_loss)), format='png', dpi=300, bbox_inches = "tight")

# In[100]:
'''
    Now let's see what happens is we use Double Q-Learning...
'''
qtableA = deepcopy(initial_qtable)
qtableB = deepcopy(initial_qtable)

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 100#10001
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


# In[100]:
'''
    Now let's see what happens is we use Double Q-Learning...
'''
qtableA = deepcopy(initial_qtable)
qtableB = deepcopy(initial_qtable)

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 100#20001
test_interval = 100 # every 200 training games, run a test
num_test_games = 100

results_doubleQ = []

df_q_values_A = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
df_q_values_B = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])

board = Board()
start_board = board

# R_win, R_tie, R_loss = [1, 0, -1]

for _ in [[1, 0, -1]]:
    R_win, R_tie, R_loss = _
    qtableA = deepcopy(initial_qtable)
    qtableB = deepcopy(initial_qtable)

    for i in range(number_games):
        board = start_board
        DoubleQ_Training(board, R_win=R_win, R_tie=R_tie, R_loss=R_loss, record_q_vals=True)

        # every xxx training games do a set of test games
        if i % test_interval == 0:
            DoubleQ_Testing(start_board)

    filename = 'QLearningData/DoubleQ_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}_Results'.format(R_win, R_tie, abs(R_loss))
    pickle_it(filename, results_doubleQ)

    filename = 'QLearningData/DoubleQ_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}_Q_values_A'.format(R_win, R_tie, abs(R_loss))
    pickle_it(filename, df_q_values_A)

    filename = 'QLearningData/DoubleQ_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}_Q_values_B'.format(R_win, R_tie, abs(R_loss))
    pickle_it(filename, df_q_values_B)

# In[101]:
# R_win, R_tie, R_loss = [1, 0, -10]
filename = 'QLearningData/DoubleQ_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}_Results'.format(R_win, R_tie, abs(R_loss))
results_doubleQ = un_pickle_it(filename)

filename = 'QLearningData/DoubleQ_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}_Q_values_A'.format(R_win, R_tie, abs(R_loss))
df_q_values_A = un_pickle_it(filename)

filename = 'QLearningData/DoubleQ_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}_Q_values_B'.format(R_win, R_tie, abs(R_loss))
df_q_values_B = un_pickle_it(filename)

ax, fig = axes(fig_number=1, rows=1, columns=3)
Plot_DoubleQ_Results_Values()

handles, labels = ax[2].get_legend_handles_labels()
order = [0,3,6,1,4,7,2,5,8]
ax[2].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                ncol=3, loc="lower right", bbox_to_anchor=(0.95,0.05), fontsize=13)

fig.text(x=0.5, y=0.98, s='R_win = {}; R_tie = {}; R_loss = {}'.format(R_win, R_tie, R_loss), ha = 'center', fontsize=20, fontweight='bold')
fig.savefig('Images/DoubleQ_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}.png'.format(R_win, R_tie, abs(R_loss)), format='png', dpi=300, bbox_inches = "tight")



# In[25]:
'''
    Vary the training process
        1. Train against random player
        2. Train against minimax player
        3. Train 2 Q-Learning Agents simultaneously (most interesting)
'''


'''
    Train against minimax player
    Test against Random player
'''

qtable1 = deepcopy(initial_qtable)
print(len(qtable1.cache))

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 100#10001
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




# In[25]:
'''
    3. Train 2 Q-Learning Agents simultaneously (most interesting)


    Play X Move
        board_X = copy(board)   # save this board for updating q-table
        (q_values_X, trans_f), found = qtable_X.get_for_position(board)

        max_index = np.where(q_values_X == max(q_values_X))[0]
        X_move  = random.choice(max_index)

        board = Board(trans_f.transform(board.board_2d).flatten())
        board = board.play_move(X_move)

        if board.is_gameover():
            R = board.get_game_result()
            Q_prime = np.zeros(9)
            Update_Qtable(qtable_X, board_X, X_move, board)
            Update_Qtable(qtable_O, board_O, O_move, board)
        else:
            board = Board(trans_f.reverse(board.board_2d).flatten())
            if board == Board():
                pass    # there is no position before the starting board
            else:
                # Update O Q-Table
                Update_Qtable(qtable_O, board_O, O_move, board)

    Play O Move
        board_O = copy(board)   # save this board for updating q-table
        (q_values_O, trans_f), found = qtable_O.get_for_position(board)

        max_index = np.where(q_values_O == max(q_values_O))[0]
        O_move  = random.choice(max_index)

        board = Board(trans_f.transform(board.board_2d).flatten())
        board = board.play_move(O_move)

        if board.is_gameover():
            R = board.get_game_result()
            Q_prime = np.zeros(9)
            Update_Qtable(qtable_X, board_X, X_move, board)
            Update_Qtable(qtable_O, board_O, O_move, board)
        else:
            board = Board(trans_f.reverse(board.board_2d).flatten())
            # Update X Q-Table
            Update_Qtable(qtable_X, board_X, X_move, board)



def play_Q_move(board, qtable, EPSILON):

    (q_values, trans_f), found = qtable.get_for_position(board)

    if (random.uniform(0, 1) > EPSILON): # choose a random move
        move_ind = random.randrange(9)
    else:
        max_index = np.where(q_values == max(q_values))[0]
        move_ind  = random.choice(max_index)

    board = Board(trans_f.transform(board.board_2d).flatten())
    board = board.play_move(move_ind)

    return board, move_ind, trans_f



def Update_Qtable(qtable, board, move_ind, board_prime):
    (q_values, trans_f), found = qtable.get_for_position(board)
    (Q_prime, T_prime), found = qtable.get_for_position(board_prime)

    if board_prime.is_gameover():
        R = board_prime.get_game_result()
        Q_prime = np.zeros(9)

    # Q(s,a) = Q(s,a) + α[r + γ max(Q') - Q]
    q_values[move_ind] += ALPHA * (R + GAMMA*max(Q_prime) - q_values[move_ind])

'''


qtable_X = deepcopy(initial_qtable)
qtable_O = deepcopy(initial_qtable) # probably only need one

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 1001
num_test_games = 100
test_interval = 200

results = []

for i in range(number_games):
    board = Board()

    while not board.is_gameover():
        board_X = copy(board)
        (q_values_X, trans_f), found = qtable_X.get_for_position(board)

        # choose move
        if (random.uniform(0, 1) > EPSILON): # choose a random move
            move_ind = random.randrange(9)
        else:
            max_index = np.where(q_values_X == max(q_values_X))[0]
            move_ind  = random.choice(max_index) # if multiple value are the max, choose randomly between them

        # play move
        board = Board(trans_f.transform(board.board_2d).flatten())
        board = board.play_move(move_ind)


        R = 0
        if board.is_gameover():
            R = board.get_game_result()
            Q_prime = np.zeros(9)
        else:
            board = Board(trans_f.reverse(board.board_2d).flatten())

            # Play O move
            board_O = copy(board)
            (q_values_O, trans_f), found = qtable_O.get_for_position(board)

            # choose move
            if (random.uniform(0, 1) > EPSILON): # choose a random move
                move_ind = random.randrange(9)
            else:
                max_index = np.where(q_values_O == max(q_values_O))[0]
                move_ind  = random.choice(max_index) # if multiple value are the max, choose randomly between them

            board = Board(trans_f.transform(board.board_2d).flatten())
            board = board.play_move(move_ind)


            # board_X < board_O < board

            (Q_prime_X, T_prime), found = qtable_X.get_for_position(board)

        # Update QTable - X
        # Q(s,a) = Q(s,a) + α[r + γ max(Q') - Q]
        q_values[move_ind] += ALPHA * (R + GAMMA*max(Q_prime) - q_values[move_ind])

        # O move
        if board.is_gameover():
            continue
        else:
            (q_values, trans_f), found = qtable_O.get_for_position(board)

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
                rand_move = random.choice(board.get_valid_move_indexes())
                board_prime = board.play_move(rand_move)
                (Q_prime, T_prime), found = qtable_O.get_for_position(board_prime)

            # Q(s,a) = Q(s,a) + α[r + γ max(Q') - Q]
            q_values[move_ind] += ALPHA * (R + GAMMA*max(Q_prime) - q_values[move_ind])







    # every 100 training games do a set of test games
    if i % test_interval == 0:

        for j in range(num_test_games):
            board = Board()
            while not board.is_gameover():
                (q_values, trans_f), found = qtable_X.get_for_position(board)

                max_index = np.where(q_values == max(q_values))[0]
                move_ind  = random.choice(max_index)

                # play X move
                board = Board(trans_f.transform(board.board_2d).flatten())
                board = board.play_move(move_ind)

                if board.is_gameover():
                    continue
                else:                   # O move
                    board = Board(trans_f.reverse(board.board_2d).flatten())

                    (q_values, trans_f), found = qtable_O.get_for_position(board)
                    max_index = np.where(q_values == max(q_values))[0]
                    move_ind  = random.choice(max_index)
                    board = Board(trans_f.transform(board.board_2d).flatten())
                    board = board.play_move(move_ind)

                    if not board.is_gameover():
                        board = Board(trans_f.reverse(board.board_2d).flatten())


            results.append(board.get_game_result())


# In[50]:


'''
# Play X Move
    board_X = copy(board)   # save this board for updating q-table
    (q_values_X, trans_f), found = qtable_X.get_for_position(board)

    max_index = np.where(q_values_X == max(q_values_X))[0]
    X_move  = random.choice(max_index)

    board = Board(trans_f.transform(board.board_2d).flatten())
    board = board.play_move(X_move)

    if board.is_gameover():
        R = board.get_game_result()
        Q_prime = np.zeros(9)
        Update_Qtable(qtable_X, board_X, X_move, board)
        Update_Qtable(qtable_O, board_O, O_move, board)
    else:
        board = Board(trans_f.reverse(board.board_2d).flatten())
        if board == Board():
            pass    # there is no position before the starting board
        else:
            # Update O Q-Table
            Update_Qtable(qtable_O, board_O, O_move, board)

# Play O Move
    board_O = copy(board)   # save this board for updating q-table
    (q_values_O, trans_f), found = qtable_O.get_for_position(board)

    max_index = np.where(q_values_O == max(q_values_O))[0]
    O_move  = random.choice(max_index)

    board = Board(trans_f.transform(board.board_2d).flatten())
    board = board.play_move(O_move)

    if board.is_gameover():
        R = board.get_game_result()
        Q_prime = np.zeros(9)
        Update_Qtable(qtable_X, board_X, X_move, board)
        Update_Qtable(qtable_O, board_O, O_move, board)
    else:
        board = Board(trans_f.reverse(board.board_2d).flatten())
        # Update X Q-Table
        Update_Qtable(qtable_X, board_X, X_move, board)
'''


def play_Q_move(board, qtable, EPSILON=0.9):

    (q_values, trans_f), found = qtable.get_for_position(board)

    if (random.uniform(0, 1) > EPSILON): # choose a random move
        move_ind = random.randrange(9)
    else:
        max_index = np.where(q_values == max(q_values))[0]
        move_ind  = random.choice(max_index)

    board = Board(trans_f.transform(board.board_2d).flatten())
    board = board.play_move(move_ind)

    return board, move_ind, trans_f



def Update_Qtable(qtable, board, move_ind, board_prime, R_win=1, R_tie=0, R_loss=-1):
    (q_values, trans_f), found = qtable.get_for_position(board)

    if board_prime.is_gameover():
        R = board_prime.get_game_result()
        Q_prime = np.zeros(9)

        if board.get_turn() == -1:
            R = -R
            if R == 1:
                R = R_win
            elif R == 0:
                R = R_tie
            elif R == -1:
                R = R_loss

    else:
        (Q_prime, T_prime), found = qtable.get_for_position(board_prime)
        R = 0

    # Q(s,a) = Q(s,a) + α[r + γ max(Q') - Q]
    q_values[move_ind] += ALPHA * (R + GAMMA*max(Q_prime) - q_values[move_ind])

# In[20]:

qtable_X = deepcopy(initial_qtable)
qtable_O = deepcopy(initial_qtable) # probably only need one

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 100#30001
num_test_games = 100
test_interval = 200

update_X = True
update_O = True

results_X_RANDOM = []
results_O_RANDOM = []
results_X_MINIMAX = []
results_O_MINIMAX = []

df_q_values_X = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
df_q_values_O_corner = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
df_q_values_O_edge = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
df_q_values_O_mid = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
start_board = Board()

for i in range(number_games):
    print('\r{}'.format(i), end='')
    board = start_board

    while not board.is_gameover():
        board_X = copy(board)   # save this board for updating q-table
        (q_values_X, trans_f), found = qtable_X.get_for_position(board)

        if board == start_board:    # record q-values
            df_q_values_X.iloc[i] = q_values_X.flatten()

            board_ = start_board.play_move(0)
            (q_values_O, trans_f_O), found = qtable_O.get_for_position(board_)
            df_q_values_O_corner.iloc[i] = q_values_O.flatten()

            board_ = start_board.play_move(1)
            (q_values_O, trans_f_O), found = qtable_O.get_for_position(board_)
            df_q_values_O_edge.iloc[i] = q_values_O.flatten()

            board_ = start_board.play_move(4)
            (q_values_O, trans_f_O), found = qtable_O.get_for_position(board_)
            df_q_values_O_mid.iloc[i] = q_values_O.flatten()

        board, X_move, trans_f = play_Q_move(board, qtable_X, EPSILON=0.9)

        if board.is_gameover():
            if update_X:
                Update_Qtable(qtable_X, board_X, X_move, board)
            if update_O:
                Update_Qtable(qtable_O, board_O, O_move, board)
            continue

        else:
            board = Board(trans_f.reverse(board.board_2d).flatten())
            if all(board_X.board_2d.flatten() == start_board.board_2d.flatten()):
                pass    # there is no position before the starting board
            else:
                if update_O:
                    Update_Qtable(qtable_O, board_O, O_move, board)

        board_O = copy(board)   # save this board for updating q-table
        board, O_move, trans_f = play_Q_move(board, qtable_O, EPSILON=0.9)

        if board.is_gameover():
            if update_X:
                Update_Qtable(qtable_X, board_X, X_move, board)
            if update_O:
                Update_Qtable(qtable_O, board_O, O_move, board)
            continue

        else:
            board = Board(trans_f.reverse(board.board_2d).flatten())
            if update_X:
                Update_Qtable(qtable_X, board_X, X_move, board)

    # if i % 1000 == 0:
    #     update_X = not(update_X)
    #     update_O = not(update_O)


##############################################################################################################
# Testing
##############################################################################################################
    if i % test_interval == 0:

        for j in range(num_test_games):
            board = start_board

            while not board.is_gameover():
                board, move_ind, trans_f = play_Q_move(board, qtable_X, EPSILON=1.0)

                if board.is_gameover():
                    continue
                else:                   # O move
                    rand_move = random.choice(board.get_valid_move_indexes())
                    board = board.play_move(rand_move)

            results_X_RANDOM.append(board.get_game_result())

        for j in range(num_test_games):
            board = start_board

            while not board.is_gameover():
                X_move = random.choice(board.get_valid_move_indexes())
                board = board.play_move(X_move)

                if not board.is_gameover():
                    board, O_move, trans_f = play_Q_move(board, qtable_O, EPSILON=1.0)

            results_O_RANDOM.append(board.get_game_result())
# In[51]:
ax, fig = axes(fig_number=1, rows=1, columns=2)

ax_i = 1
results = results_X_RANDOM
for i in range(int(len(results) / num_test_games)):
    begin = i*num_test_games
    end = begin + num_test_games
    subresults = results[begin:end]
    x = i*test_interval
    if i == 0:
        ax[ax_i].plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = 'win')
        ax[ax_i].plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = 'tie')
        ax[ax_i].plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = 'loss')
    else:
        ax[ax_i].plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = '_nolegend_')
        ax[ax_i].plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = '_nolegend_')
        ax[ax_i].plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = '_nolegend_')

ax[ax_i].set_xlabel('Number of Training Games')
ax[ax_i].set_ylabel('Percentage of Test Game Results')
ax[ax_i].legend(ncol=3, loc='right', bbox_to_anchor=(0.95,0.5), fontsize=13)
ax[ax_i].set_title('Player X vs Random')

ax_i = 2
results = results_O_RANDOM
for i in range(int(len(results) / num_test_games)):
    begin = i*num_test_games
    end = begin + num_test_games
    subresults = results[begin:end]
    x = i*test_interval
    if i == 0:
        ax[ax_i].plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = 'win')
        ax[ax_i].plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = 'tie')
        ax[ax_i].plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = 'loss')
    else:
        ax[ax_i].plot(x, len([r for r in subresults if r == 1]) / len(subresults)*100, 'go', label = '_nolegend_')
        ax[ax_i].plot(x, len([r for r in subresults if r == 0]) / len(subresults)*100, 'ks', label = '_nolegend_')
        ax[ax_i].plot(x, len([r for r in subresults if r == -1]) / len(subresults)*100, 'r^', label = '_nolegend_')

ax[ax_i].set_ylim(ax[1].get_ylim())
ax[ax_i].set_xlabel('Number of Training Games')
# ax[ax_i].set_ylabel('Percentage of Test Game Results')
ax[ax_i].set_yticklabels([])
ax[ax_i].legend(ncol=3, loc='right', bbox_to_anchor=(0.98,0.5), fontsize=13)
ax[ax_i].set_title('Random vs Player O')

fig.tight_layout()

fig.savefig('Images/Q_vs_Q_Learning.png', format='png', dpi=300, bbox_inches = "tight")



# In[100]:

# print(len(ax))
#
# for key in df_q_values_X.keys():
#     ax[2].plot(df_q_values_X[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))
#
# handles, labels = ax[2].get_legend_handles_labels()
# order = [0,3,6,1,4,7,2,5,8]
# ax[2].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
#                 ncol=3, loc="lower right", bbox_to_anchor=(0.98,0.02), fontsize=13)
#
# ax[2].set_ylabel('Q-Values X')
# ax[2].set_xlabel('Number of Training Games')
#
#
# ax_i = 3
# for key in df_q_values_X.keys():
#     ax[ax_i].plot(df_q_values_O_corner[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))
#
# handles, labels = ax[ax_i].get_legend_handles_labels()
# order = [0,3,6,1,4,7,2,5,8]
# ax[ax_i].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
#                 ncol=3, loc="lower right", bbox_to_anchor=(0.98,0.02), fontsize=13)
#
# ax[ax_i].set_ylabel('Q-Values X')
# ax[ax_i].set_xlabel('Number of Training Games')
#
# ax_i = 4
# for key in df_q_values_X.keys():
#     ax[ax_i].plot(df_q_values_O_edge[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))
#
# handles, labels = ax[ax_i].get_legend_handles_labels()
# order = [0,3,6,1,4,7,2,5,8]
# ax[ax_i].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
#                 ncol=3, loc="lower right", bbox_to_anchor=(0.98,0.02), fontsize=13)
#
# ax[ax_i].set_ylabel('Q-Values X')
# ax[ax_i].set_xlabel('Number of Training Games')
#
# fig.tight_layout()

# In[55]:
qtable1 = deepcopy(initial_qtable)

EPSILON = 0.9
GAMMA = 0.9
ALPHA = 0.1
number_games = 100#30001
num_test_games = 100

df_q_values_X = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
df_q_values_O_corner = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
df_q_values_O_edge = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])
df_q_values_O_mid = pd.DataFrame(columns=[x for x in range(9)], index=[x for x in range(number_games)])

start_board = Board()

# R_win, R_tie, R_loss = [1, 0, -5]

results = []

for i in range(number_games):
    board = start_board
    board_O = None

    while not board.is_gameover():
        if board == start_board:    # record q-values
            board_ = board.play_move(0)
            (q_values_O, trans_f_O), found = qtable1.get_for_position(board_)
            df_q_values_O_corner.iloc[i] = q_values_O.flatten()

            board_ = board.play_move(1)
            (q_values_O, trans_f_O), found = qtable1.get_for_position(board_)
            df_q_values_O_edge.iloc[i] = q_values_O.flatten()

            board_ = board.play_move(4)
            (q_values_O, trans_f_O), found = qtable1.get_for_position(board_)
            df_q_values_O_mid.iloc[i] = q_values_O.flatten()

        # X Move
        X_move = random.choice(board.get_valid_move_indexes())
        board = board.play_move(X_move)

        if not board.is_gameover():
            # O Move
            if board_O:
                Update_Qtable(qtable1, board_O, O_move, board, R_loss=R_loss)

            board_O = copy(board)
            board, O_move, trans_f = play_Q_move(board, qtable1, EPSILON=0.9)

        if board.is_gameover():
            Update_Qtable(qtable1, board_O, O_move, board, R_loss=R_loss)
        else:
            board = Board(trans_f.reverse(board.board_2d).flatten())


    if i % test_interval == 0:

        for j in range(num_test_games):
            board = start_board

            while not board.is_gameover():
                X_move = random.choice(board.get_valid_move_indexes())
                board = board.play_move(X_move)

                if not board.is_gameover():
                    board, O_move, trans_f = play_Q_move(board, qtable1, EPSILON=1.0)

                # if board.is_gameover():
                #     continue
                # else:
                #     board = Board(trans_f.reverse(board.board_2d).flatten())

            results.append(board.get_game_result())

filename = 'QLearningData/Q_Learning_Player2_R_win_{}__R_tie_{}__R_loss_{}_Results'.format(R_win, R_tie, abs(R_loss))
pickle_it(filename, results)

filename = 'QLearningData/Q_Learning_Player2_R_win_{}__R_tie_{}__R_loss_{}_Q_values_Corner'.format(R_win, R_tie, abs(R_loss))
pickle_it(filename, df_q_values_O_corner)
filename = 'QLearningData/Q_Learning_Player2_R_win_{}__R_tie_{}__R_loss_{}_Q_values_Edge'.format(R_win, R_tie, abs(R_loss))
pickle_it(filename, df_q_values_O_edge)
filename = 'QLearningData/Q_Learning_Player2_R_win_{}__R_tie_{}__R_loss_{}_Q_values_Center'.format(R_win, R_tie, abs(R_loss))
pickle_it(filename, df_q_values_O_mid)

filename = 'QLearningData/Q_Learning_Player2_R_win_{}__R_tie_{}__R_loss_{}_Full_QTable'.format(R_win, R_tie, abs(R_loss))
pickle_it(filename, qtable1)

# In[100]:
ax, fig = axes(fig_number=1, rows=2, columns=2)

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

# ax[1].set_ylim(-3, 80)
ax[1].set_xlabel('Number of Training Games')
ax[1].set_ylabel('Percentage of Test Game Results')
ax[1].legend(ncol=3, loc='right', bbox_to_anchor=(0.95,0.5), fontsize=13)


ax_i = 2
for key in df_q_values_O_mid.keys():
    ax[ax_i].plot(df_q_values_O_mid[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))

handles, labels = ax[ax_i].get_legend_handles_labels()
order = [0,3,6,1,4,7,2,5,8]
ax[ax_i].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                ncol=3, loc="lower right", bbox_to_anchor=(0.98,0.02), fontsize=13)

ax[ax_i].set_ylabel('Q-Values X')
ax[ax_i].set_xlabel('Number of Training Games')
ax[ax_i].set_title('X Moves to Center')


ax_i = 3
for key in df_q_values_O_corner.keys():
    ax[ax_i].plot(df_q_values_O_corner[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))

handles, labels = ax[ax_i].get_legend_handles_labels()
order = [0,3,6,1,4,7,2,5,8]
ax[ax_i].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                ncol=3, loc="lower right", bbox_to_anchor=(0.98,0.02), fontsize=13)

ax[ax_i].set_ylabel('Q-Values X')
ax[ax_i].set_xlabel('Number of Training Games')
ax[ax_i].set_title('X Moves to Corner')

ax_i = 4
for key in df_q_values_X.keys():
    ax[ax_i].plot(df_q_values_O_edge[key], linestyle='None', marker=markers[key], color=colors[key], label=str(key))

handles, labels = ax[ax_i].get_legend_handles_labels()
order = [0,3,6,1,4,7,2,5,8]
ax[ax_i].legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                ncol=3, loc="lower right", bbox_to_anchor=(0.98,0.02), fontsize=13)

ax[ax_i].set_ylabel('Q-Values X')
ax[ax_i].set_xlabel('Number of Training Games')
ax[ax_i].set_title('X Moves to Edge')

fig.tight_layout()
# In[100]:
# R_win, R_tie, R_loss = [1, 0, -10]

filename = 'QLearningData/Q_Learning_Player2_R_win_{}__R_tie_{}__R_loss_{}_Full_QTable'.format(R_win, R_tie, abs(R_loss))
qtable_O = un_pickle_it(filename)

filename = 'QLearningData/Q_Learning_Start_Board_R_win_{}__R_tie_{}__R_loss_{}_Full_QTable'.format(R_win, R_tie, abs(R_loss))
qtable_X = un_pickle_it(filename)

# In[53]:
'''
    Test Q-Agent Strategy against different opponent strategies
'''

def q_agent_play_games(num_test_games=50, opponent = 'O', opponent_strategy=None):
    '''
        Available Opponent Strategies:
        Random
        Minimax
        Q-Agent
    '''
    if opponent_strategy == None:
        opponent_strategy = 'Random'

    results = []

    for j in range(num_test_games):
        board = start_board

        while not board.is_gameover():
            # X Move
            if opponent == 'O':
                board, move_ind, trans_f = play_Q_move(board, qtable_X, EPSILON=1.0)
            else:
                if opponent_strategy == 'Random':
                    opponent_move = random.choice(board.get_valid_move_indexes())
                    board = board.play_move(opponent_move)
                elif opponent_strategy == 'Minimax':
                    opponent_move = mini_max_strategy(board)
                    board = board.play_move(opponent_move)
                elif opponent_strategy == 'Q-Agent':
                    board, move_ind, trans_f = play_Q_move(board, qtable_X, EPSILON=1.0)


            if board.is_gameover():
                continue

            # O Move
            else:
                if opponent == 'O':
                    if opponent_strategy == 'Random':
                        opponent_move = random.choice(board.get_valid_move_indexes())
                        board = board.play_move(opponent_move)
                    elif opponent_strategy == 'Minimax':
                        opponent_move = mini_max_strategy(board)
                        board = board.play_move(opponent_move)
                    elif opponent_strategy == 'Q-Agent':
                        board, move_ind, trans_f = play_Q_move(board, qtable_X, EPSILON=1.0)
                else:
                    board, move_ind, trans_f = play_Q_move(board, qtable_O, EPSILON=1.0)


        results.append(board.get_game_result())

    return results

test = q_agent_play_games(num_test_games=50, opponent='X', opponent_strategy='Q-Agent')
print(test)

# In[54]:
opponent_player = ['O', 'X']
opponent_strategy = ['Random', 'Minimax', 'Q-Agent']
number_games = 50

for opp_player in opponent_player:
    for opp_strat in opponent_strategy:
        results = q_agent_play_games(num_test_games=number_games, opponent=opp_player, opponent_strategy=opp_strat)

        if opp_player == 'O':
            Q_agent = 'X'
            r_win = 1
        elif opp_player == 'X':
            Q_agent = 'O'
            r_win = -1


        print('Q_Agent ({}) versus {} ({}):'.format(Q_agent, opp_strat, opp_player))
        print('Win Rate: \t {:.1f}'.format(len([r for r in results if r == r_win]) / len(results)*100))
        print('Loss Rate: \t {:.1f}'.format(len([r for r in results if r == -r_win]) / len(results)*100))
        print('Tie Rate: \t {:.1f}'.format(len([r for r in results if r == 0]) / len(results)*100))
        print()
