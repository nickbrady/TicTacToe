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

'''
    There are 9 game over conditions: 8 game won conditions, 1 board full (tie-game)
    x|x|x       | |         | |
    -+-+-      -+-+-       -+-+-
     | |       x|x|x        | |
    -+-+-      -+-+-       -+-+-
     | |        | |        x|x|x

    x| |        |x|         | |x
    -+-+-      -+-+-       -+-+-
    x| |        |x|         | |x
    -+-+-      -+-+-       -+-+-
    x| |        |x|         | |x

    x| |        | |x
    -+-+-      -+-+-
     |x|        |x|
    -+-+-      -+-+-
     | |x      x| |
'''
# In[1]:

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

        rows_cols_and_diagonals = get_rows_cols_and_diagonals(self.board_2d)

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
                move = get_symbol(self.board_2d[r, c])
                if c == 0:
                    board_as_string += f"|{move}|"
                elif c == 1:
                    board_as_string += f"{move}|"
                else:
                    board_as_string += f"{move}|\n"
        board_as_string += "-------\n"

        return board_as_string

def get_rows_cols_and_diagonals(board_2d):
    rows_and_diagonal = get_rows_and_diagonal(board_2d)
    cols_and_antidiagonal = get_rows_and_diagonal(np.rot90(board_2d))
    return rows_and_diagonal + cols_and_antidiagonal

def get_rows_and_diagonal(board_2d):
    num_rows = board_2d.shape[0]
    return ([row for row in board_2d[range(num_rows), :]]
            + [board_2d.diagonal()])

def get_symbol(cell):
    if cell == CELL_X:
        return 'X'
    if cell == CELL_O:
        return 'O'
    return '-'

# In[2]:
# investigation of the code and game function
board = Board()

print(board.board_2d)
print(board.illegal_move)
print(board.get_valid_move_indexes())
print('Game Result', board.get_game_result())
print(board.is_gameover())

print(get_rows_and_diagonal(board.board_2d))
print(get_rows_and_diagonal(np.rot90(board.board_2d)))

board = board.play_move(3)
board = board.play_move(4)
print(board.get_illegal_move_indexes())
print(board.board_2d)
print()

board = board.play_move(1)
board = board.play_move(2)
board = board.play_move(5)
board = board.play_move(6)
print(board.board_2d)
print(board.is_gameover())
print(board.get_game_result())

board.print_board()



# In[3]:
# play xxx number of random games
# investigate break down of random games

number_games = 10000
game_results = np.zeros(number_games)
game_moves   = np.zeros(number_games)
for i in range (number_games):
    board = Board()
    while not board.is_gameover():
        move_ind = random.choice(board.get_valid_move_indexes())
        board = board.play_move(move_ind)

    game_results[i] = board.get_game_result()
    game_moves[i]   = np.count_nonzero(board.board)

# In[4]:
print('\t \t \t (%)')
print('Ties: \t \t \t', np.count_nonzero(game_results == 0) / len(game_results) * 100)
print('X Wins: \t \t', np.count_nonzero(game_results == 1) / len(game_results) * 100)
print('O Wins: \t \t', np.count_nonzero(game_results == -1) / len(game_results) * 100)
print()

print('5 Move Games: \t \t', np.count_nonzero(game_moves == 5) / len(game_results) * 100)
print('6 Move Games: \t \t', np.count_nonzero(game_moves == 6) / len(game_results) * 100)
print('7 Move Games: \t \t', np.count_nonzero(game_moves == 7) / len(game_results) * 100)
print('8 Move Games: \t \t', np.count_nonzero(game_moves == 8) / len(game_results) * 100)
print('9 Move Games (Wins): \t', (np.count_nonzero(game_moves == 9) - np.count_nonzero(game_results == 0)) / len(game_results) * 100)
print('9 Move Games (Ties): \t', np.count_nonzero(game_results == 0) / len(game_results) * 100)

# In[5]:
# Use sampling with replacement to get mean and standard deviations
number_resamples = 1000
ties = np.zeros(number_resamples)
X_win = np.zeros(number_resamples)
O_win = np.zeros(number_resamples)

for i in range(number_resamples):
    sub_sample = np.array(random.choices(game_results, k=10000))
    ties[i] = np.count_nonzero(sub_sample == 0)
    X_win[i] = np.count_nonzero(sub_sample == 1)
    O_win[i] = np.count_nonzero(sub_sample == -1)

ties = ties/len(sub_sample)*100
X_win = X_win/len(sub_sample)*100
O_win = O_win/len(sub_sample)*100

# In[6]:
print('\t \t mean \t \t std. dev')
print('Ties: \t \t', '{:.3f} \t'.format(np.mean(ties)),  '{:.3f}'.format(np.std(ties)))
print('X Wins: \t',  '{:.3f} \t'.format(np.mean(X_win)), '{:.3f}'.format(np.std(X_win)))
print('O Wins: \t',  '{:.3f} \t'.format(np.mean(O_win)), '{:.3f}'.format(np.std(O_win)))


# In[7]:
# Use a recursive function to build a list of all end of game board orientations (win or draw)
# and this can be used to double check that our "random" results make sense
def board_state_moves_and_result(board_state, board_list=None, number_moves=None, game_result=None):
    if board_list is None:
        board_list = []

    if number_moves is None:
        number_moves = []

    if game_result is None:
        game_result = []

    if board_state.is_gameover():
        board_list.append(board_state.board_2d)
        number_moves.append(np.count_nonzero(board_state.board_2d))
        game_result.append(board_state.get_game_result())

    else:
        legal_moves = board_state.get_valid_move_indexes()

        for l_m in legal_moves:
            board_state_moves_and_result(board_state.play_move(l_m), board_list, number_moves, game_result)

    return board_list, number_moves, game_result

# In[8]:
board = Board()

boards, moves, results = board_state_moves_and_result(board)

# In[9]:
df_results = pd.DataFrame()

df_results['Boards'] = boards
df_results['# of Moves'] = moves
df_results['Game Results'] = results

# In[10]:
df_results_summary = pd.DataFrame()
print('Total Number of End States: \t', len(df_results))
print()

total_check = 0
for m in range(5,9):
    games = np.count_nonzero(df_results['# of Moves'] == m)
    df_results_summary['{} Moves'.format(m)] = [games]

m = 9
nine_move_wins = np.count_nonzero((df_results['# of Moves'] == 9) & (df_results['Game Results'] == +1))
nine_move_ties = np.count_nonzero((df_results['# of Moves'] == 9) & (df_results['Game Results'] == 0))

df_results_summary['{} Moves'.format(m)] = nine_move_wins
df_results_summary['Ties'] = nine_move_ties

total_check += nine_move_wins + nine_move_ties


print(df_results_summary.to_markdown(index=False))
print(sum(sum(df_results_summary.values)))

# In[11]:
X_win_total = 0
O_win_total = 0
advance_percent = 1
games_to_move_k = math.factorial(9)/math.factorial(9-5)
for k in range(5,10):
    games_won = df_results_summary['{} Moves'.format(k)].values[0]

    win_percent_state = games_won / games_to_move_k
    win_percent_cum = win_percent_state * advance_percent
    advance_percent *= 1 - win_percent_state

    print('Game states at move {}: \t \t \t'.format(k), int(games_to_move_k))
    print('Game states won on move {}: \t \t'.format(k),games_won)
    games_to_advance = games_to_move_k - games_won

    if k == 9:
        print('Game states that tie: \t \t \t', int(games_to_advance))
    else:
        print('Game states to advance: \t \t', int(games_to_advance))
        games_to_move_k = games_to_advance * (9-k)

    if k % 2 == 1:
        X_win_total += win_percent_cum
        color = [0.4, 0.4, 1]
    else:
        O_win_total += win_percent_cum
        color = [1, 0.4, 0.4]

    print('Fraction of Game States Won: \t \t', win_percent_state)
    print('Cummulative Win Fraction: \t \t', win_percent_cum)
    print('Fraction of games to advance: \t \t', advance_percent)

    print()

    plt.bar('{}'.format(k), win_percent_cum*100, color = color)

plt.bar('Tie', advance_percent*100, color='gray')
plt.bar('X Win', X_win_total*100, color='blue')
plt.bar('O Win', O_win_total*100, color='red')

print('Fraction of Games Won by X: \t \t', X_win_total)
print('Fraction of Games Won by O: \t \t', O_win_total)
print('Fraction of Games That Tie: \t \t', advance_percent)

# In[12]:
'''
Agenda
        Write up summary of results
        Plot the most frequent win positions
'''

ax, fig = axes(rows=8, columns=2)

games_to_move_k = math.factorial(9)/math.factorial(9-5)
advance_percent = 1

_x_win_pos_ = np.zeros((3,3))
_o_win_pos_ = np.zeros((3,3))

_x_loss_pos_ = np.zeros((3,3))
_o_loss_pos_ = np.zeros((3,3))

for m in range(5,11):
    board_ = df_results[df_results['# of Moves'] == m]['Boards']

    if m == 9:
        board_ = df_results[(df_results['# of Moves'] == m) & (df_results['Game Results'] == +1)]['Boards']

    elif m == 10:
        board_ = df_results[(df_results['# of Moves'] == 9) & (df_results['Game Results'] == 0)]['Boards']

    _x_pos_ = np.zeros((3,3))
    _o_pos_ = np.zeros((3,3))
    ___pos_ = np.zeros((3,3))
    for b in board_:
        _x_pos_ += np.where(b == 1, b, 0)
        _o_pos_ += np.where(b == -1, b, 0)
        ___pos_ += np.where(b == 0, 1, 0)

    # normalize
    _x_pos_ *= 100/sum(sum(_x_pos_))
    _o_pos_ *= 100/sum(sum(_o_pos_))

    if m < 9:
        ___pos_ /= sum(sum(___pos_))
        ___pos_ *= 100

    if m <= 9:
        if m % 2 == 1:
            _x_win_pos_ += _x_pos_
            _o_loss_pos_ += _o_pos_
        else:
            _o_win_pos_ += _o_pos_
            _x_loss_pos_ += _x_pos_

        games_won = df_results_summary['{} Moves'.format(m)].values[0]
        win_percent_state = games_won / games_to_move_k
        win_percent_cum = win_percent_state * advance_percent
        advance_percent *= 1 - win_percent_state
        games_to_move_k = games_to_advance * (9-m)

    ax1 = ax[2*(m-5)+1]
    ax2 = ax[2*(m-5)+2]
    _1 = ax1.pcolormesh(_x_pos_, cmap='Blues', vmin=0, vmax=20)
    _2 = ax2.pcolormesh(_o_pos_, cmap='Reds',  vmin=0, vmax=20)
    # fig.colorbar(_)

    cbar1 = fig.colorbar(_1, ax=ax1)
    cbar2 = fig.colorbar(_2, ax=ax2)

    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.set_xticks([])
    ax2.set_yticks([])


# normalize
_x_win_pos_  *= 100/sum(sum(_x_win_pos_))
_x_loss_pos_ *= 100/sum(sum(_x_loss_pos_))
_o_win_pos_  *= 100/sum(sum(_o_win_pos_))
_o_loss_pos_ *= 100/sum(sum(_o_loss_pos_))

ax1 = ax[13]
ax2 = ax[14]
_1 = ax1.pcolormesh(_x_win_pos_, cmap='Blues', vmin=0)
_2 = ax2.pcolormesh(_o_win_pos_, cmap='Reds',  vmin=0)

cbar1 = fig.colorbar(_1, ax=ax1)
cbar2 = fig.colorbar(_2, ax=ax2)

ax1.set_xticks([])
ax1.set_yticks([])

ax2.set_xticks([])
ax2.set_yticks([])


ax1 = ax[15]
ax2 = ax[16]
_1 = ax1.pcolormesh(_x_loss_pos_, cmap='Blues', vmin=0)
_2 = ax2.pcolormesh(_o_loss_pos_, cmap='Reds',  vmin=0)

cbar1 = fig.colorbar(_1, ax=ax1)
cbar2 = fig.colorbar(_2, ax=ax2)

ax1.set_xticks([])
ax1.set_yticks([])

ax2.set_xticks([])
ax2.set_yticks([])

ax[1].set_title('X Position Frequency')
ax[2].set_title('O Position Frequency')

ax[1].set_ylabel('5 Move Games \n(X Wins)')
ax[3].set_ylabel('6 Move Games \n(O Wins)')
ax[5].set_ylabel('7 Move Games \n(X Wins)')
ax[7].set_ylabel('8 Move Games \n(O Wins)')
ax[9].set_ylabel('9 Move Games \n(X Wins)')
ax[11].set_ylabel('9 Move Games \n(Draw)')
ax[13].set_ylabel('Overall Winning Positions')
ax[15].set_ylabel('Overall Losing Positions')

fig.tight_layout()

# In[20]:
'''
Summary of Results

1. The programmed game works as expected
2. The results of both players choosing random moves are in agreement with the closed form statistics of the possible board orientations
3. A recursive function was written and used to identify all possible board states
4. The results from the recursive investigation are again in agreement with the closed form mathematics
5. A bar chart illustrates the statistics of all possible game results
6. 2-dimensional colormaps illustrate the most frequently played positions for games won and lost
    a. These results suggest that there is a hierarchy of the positional value
        1. The most valuable board position is the middle square
        2. The second most valuable positions are the corner squares
        3. The least valuable squares are the middle edge squares
    b. This is consistent with what we know about the game and from a superficial glance at combinatorics
        1. There are 8 possible winning orientations (see top)
        2. 4 of them involve the middle square
        3. each corner square can be used in 3 winning orientations
        4. each middle edge square can only be used in 2 winning orientations
    c. These results seem to suggest a possible game strategy that can be outlined in 3 lines
        1. choose the middle square if available
        2. choose a corner square if available
        3. else choose randomly (all that remain are the middle edge squares)
'''

# In[21]:
def tier_strategy(board):
    # Simple, non-random, tiered square strategy
    available_moves = board.get_valid_move_indexes()
    middle_square = 4
    corner_squares = [0, 2, 6, 8]
    avail_corners = [x for x in corner_squares if x in available_moves]

    if 4 in available_moves:
        move_ind = 4
    elif avail_corners:
        move_ind = random.choice(avail_corners)
    else:
        move_ind = random.choice(board.get_valid_move_indexes())

    return move_ind

# In[22]:
number_games = 1000
game_results = np.zeros(number_games)
game_moves   = np.zeros(number_games)
for i in range (number_games):
    board = Board()
    while not board.is_gameover():
        if board.get_turn() == 1: # player one
            move_ind = tier_strategy(board)
        else:
            move_ind = random.choice(board.get_valid_move_indexes())

        board = board.play_move(move_ind)

    game_results[i] = board.get_game_result()
    game_moves[i]   = np.count_nonzero(board.board)

# In[23]:
print(np.count_nonzero(game_results == 1) / len(game_results))
print(np.count_nonzero(game_results == 0) / len(game_results))
print(np.count_nonzero(game_results == -1) / len(game_results))


# In[24]:
number_games = 1000
game_results = np.zeros(number_games)
game_moves   = np.zeros(number_games)
for i in range (number_games):
    board = Board()
    while not board.is_gameover():
        if board.get_turn() == -1: # player two
            move_ind = tier_strategy(board)
        else:
            move_ind = random.choice(board.get_valid_move_indexes())

        board = board.play_move(move_ind)

    game_results[i] = board.get_game_result()
    game_moves[i]   = np.count_nonzero(board.board)

print(np.count_nonzero(game_results == -1) / len(game_results))
print(np.count_nonzero(game_results == 0) / len(game_results))
print(np.count_nonzero(game_results == 1) / len(game_results))
