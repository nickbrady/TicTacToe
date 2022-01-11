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
                move = get_symbol(self.board_2d[r, c])
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
        rows_and_diagonal = get_rows_and_diagonal(self.board_2d)
        cols_and_antidiagonal = get_rows_and_diagonal(np.rot90(self.board_2d))
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

# In[2]:
# investigation of the code and game function
board = Board()

print(board.board_2d)
print(board.illegal_move)
print(board.get_valid_move_indexes())
print('Game Result', board.get_game_result())
print(board.is_gameover())

print(board.get_rows_cols_and_diagonals())
print(board.get_rows_and_diagonal(board.board_2d))
print(board.get_rows_and_diagonal(np.rot90(board.board_2d)))

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
print('Ties: \t \t \t{:.3f}'.format( np.count_nonzero(game_results == 0) / len(game_results) * 100))
print('X Wins: \t \t{:.3f}'.format( np.count_nonzero(game_results == 1) / len(game_results) * 100))
print('O Wins: \t \t{:.3f}'.format( np.count_nonzero(game_results == -1) / len(game_results) * 100))
print()

print('5 Move Games: \t \t{:.3f}'.format( np.count_nonzero(game_moves == 5) / len(game_results) * 100))
print('6 Move Games: \t \t{:.3f}'.format( np.count_nonzero(game_moves == 6) / len(game_results) * 100))
print('7 Move Games: \t \t{:.3f}'.format( np.count_nonzero(game_moves == 7) / len(game_results) * 100))
print('8 Move Games: \t \t{:.3f}'.format( np.count_nonzero(game_moves == 8) / len(game_results) * 100))
print('9 Move Games (Wins): \t{:.3f}'.format(
        (np.count_nonzero(game_moves == 9) - np.count_nonzero(game_results == 0)) / len(game_results) * 100))
print('9 Move Games (Ties): \t{:.3f}'.format( np.count_nonzero(game_results == 0) / len(game_results) * 100))

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
print('\t \t (mean ± std. dev)')
print('Ties: \t \t', '{:.3f} ± {:.3f}'.format(np.mean(ties),  np.std(ties)))
print('X Wins: \t',  '{:.3f} ± {:.3f}'.format(np.mean(X_win), np.std(X_win)))
print('O Wins: \t',  '{:.3f} ± {:.3f}'.format(np.mean(O_win), np.std(O_win)))


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

ax, fig = axes(rows=8, columns=2, row_height=3, column_width=6*3./5)

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

# In[13]:
'''
Summary of Results (Part 1)

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
    c. These results seem to possibly suggest a more optimal game strategy than random position selection.
        The strategy can be summarized in 3 lines:
        1. choose the middle square if available
        2. choose a corner square if available
        3. else choose randomly (all that remain are the middle edge squares)
'''

# In[20]:
'''
    Define different game strategies:
        a strategy is a function that takes board information as input and returns a move (move_ind)
'''

def strategy_center_corners_else(board):
    # optimal game strategy
    # 1. pick the center square
    # 2. pick the corner squares
    # 3. pick whatever else is available
    available_moves = board.get_valid_move_indexes()
    middle_square = 4
    corner_squares = [0, 2, 6, 8]
    avail_corners = [x for x in corner_squares if x in available_moves]

    if middle_square in available_moves:
        move_ind = middle_square
    elif avail_corners:
        move_ind = random.choice(avail_corners)
    else:
        move_ind = random.choice(board.get_valid_move_indexes())

    return move_ind

def strategy_corners_center_else(board):
    # 1. pick the corner squares
    # 2. pick the center square
    # 3. pick whatever else is available
    available_moves = board.get_valid_move_indexes()
    middle_square = 4
    corner_squares = [0, 2, 6, 8]
    avail_corners = [x for x in corner_squares if x in available_moves]

    if avail_corners:
        move_ind = random.choice(avail_corners)

    elif middle_square in available_moves:
        move_ind = middle_square

    else:
        move_ind = random.choice(board.get_valid_move_indexes())

    return move_ind

def strategy_midedge_corners_center(board):
    # 1. pick the edge middle squares (least favorable squares)
    # 2. pick the corner squares
    # 3. pick the center square
    # this is the least optimal strategy - worse than just picking randomly
    available_moves = board.get_valid_move_indexes()
    middle_square = 4
    edge_middle_sq = [1, 3, 5, 7]
    corner_squares = [0, 2, 6, 8]
    avail_corners = [x for x in corner_squares if x in available_moves]
    avail_edge_mid = [x for x in edge_middle_sq if x in available_moves]

    if avail_edge_mid:
        move_ind = random.choice(avail_edge_mid)

    elif avail_corners:
        move_ind = random.choice(avail_corners)

    elif middle_square in available_moves:
        move_ind = middle_square

    return move_ind

# In[21]:
# define a function to play many games with defined playing strategies
def play_tic_tac_toe(player_1_strategy=None, player_2_strategy=None, number_of_games=1):
    game_results = np.zeros(number_of_games)
    game_moves   = np.zeros(number_of_games)
    # game_moves   = np.zeros(9)
    # game_moves_list = []

    def random_move(board):
        move_ind = random.choice(board.get_valid_move_indexes())
        return move_ind

    if player_1_strategy is None:
        player_1_strategy = random_move

    if player_2_strategy is None:
        player_2_strategy = random_move

    for i in range(number_of_games):
        board = Board()
        while not board.is_gameover():
            if board.get_turn() == 1: # player 1
                move_ind = player_1_strategy(board)
            else:                     # player 2
                move_ind = player_2_strategy(board)

            board = board.play_move(move_ind)

        game_results[i] = board.get_game_result()
        game_moves[i]   = np.count_nonzero(board.board)
        # game_moves_list.append(game_moves)

    return game_results, game_moves

# In[22]:
df_p1 = pd.DataFrame()

strategy_names = ['Favor_Edge_Middle', 'Random', 'Favor_Corners_Center', 'Favor_Center_Corners']
strategies = [strategy_midedge_corners_center, None, strategy_corners_center_else,
                strategy_center_corners_else]

for strat, strat_name in zip(strategies, strategy_names):
    # play the games with different strategies
    game_res, game_moves = play_tic_tac_toe(player_1_strategy=strat, player_2_strategy=None, number_of_games=4000)
    #
    # make a list of the game results (+1 = X Wins, -1 = O Wins, 0 = Tie)
    res_ = [+1, -1, 0]
    _ = [np.count_nonzero(game_res == r) / len(game_res) for r in res_]

    # add this list to a dataframe
    df_p1[strat_name] = _

df_p1.index = ['X Wins', 'O Wins', 'Ties']


# In[23]:
df_p2 = pd.DataFrame()

strategy_names = ['Favor_Edge_Middle', 'Random', 'Favor_Corners_Center', 'Favor_Center_Corners']
strategies = [strategy_midedge_corners_center, None, strategy_corners_center_else,
                strategy_center_corners_else]

for strat, strat_name in zip(strategies, strategy_names):
    # play the games with different strategies
    game_res, game_moves = play_tic_tac_toe(player_1_strategy=None, player_2_strategy=strat, number_of_games=4000)
    #
    # make a list of the game results (+1 = X Wins, -1 = O Wins, 0 = Tie)
    res_ = [+1, -1, 0]
    _ = [np.count_nonzero(game_res == r) / len(game_res) for r in res_]

    # add this list to a dataframe
    df_p2[strat_name] = _

df_p2.index = ['X Wins', 'O Wins', 'Ties']

# In[24]:
ax, fig = axes(fig_number=1, rows=1, columns=2)

df_p1.plot.bar(ax=ax[1], rot=0)
df_p2.plot.bar(ax=ax[2], rot=0)

ax[1].set_ylim(0, 1.0)
ax[1].set_ylabel('Results Fraction')
ax[2].set_ylim(0, 1.0)
ax[2].set_yticklabels([])

ax[1].set_title('Player X Strategy vs Random Player O')
ax[2].set_title('Player O Strategy vs Random Player X')

fig.tight_layout()

'''
    Summary of Results (Part 2)
    The bar graph illustrates the results achieved using different playing strategies vs an opponent who randomly selects their playing positions

    The first plot illustrates different player 1 strategies vs a player 2 who uses a random selection process.

    The results from the mesh plot above showed the most frequently selected positions for won, lost, and tied games. Those results seemed to suggest that the most valuable playing position is the center square, followed by the corner squares, with the edge middle squares being the least valuable. To test this hypothesis various game strategies where created:
        1. first select edge middle squares, then corner squares, then the center square (worse than random)
        2. random position selection                                                     (control)
        3. first select corner squares, then the center square, then edge middle squares (better than random
                                                                                        but still sub-optimal)
        4. first select center square, then the corner squares, then edge middle squares (optimal)

    The results illustrated in the bar graph show that the hypotheses about the differing values of the squares are indeed correct.
    Player 1 peforms worse by preferably selecting the edge middle squares, performs better when preferably selecting corner squares, and performs the best when preferably selecting the center square (followed by the corner squares).
    The same performance trend exists for Player 2.

    Hypothesis: are more optimal playing strategies better at training a neural-network than sub-optimal playing strategies?
        For instance, if a neural network is trained against an optimal player, will the neural net learn the optimal playing strategy faster?
        - proposed answer: probably it will learn faster, but is it useful?
'''

# In[30]:
''' Write a Min/Max game strategy '''

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

def mini_max_strategy_pref_center(board):
    center_square = 4
    if center_square in board.get_valid_move_indexes():
        return center_square

    return mini_max_strategy(board)




board = Board()
board = board.play_move(1)
# board = board.play_move(2)
# board = board.play_move(5)
# board = board.play_move(6)
print(board.board_2d)

print(mini_max_strategy(board))
print(mini_max_strategy_pref_center(board))

# In[6]:

def get_symmetrical_board_orientations(board_2d):
    orientations = [board_2d]

    current_board_2d = board_2d
    for i in range(3): # rotate board 3 times by 90 degrees
        current_board_2d = np.rot90(current_board_2d)
        orientations.append(current_board_2d)

    orientations.append(np.flipud(board_2d))
    orientations.append(np.fliplr(board_2d))

    orientations.append(np.flipud(np.rot90(board_2d)))
    orientations.append(np.fliplr(np.rot90(board_2d)))

    # there are 8 equivalent board orientations
    return orientations


class BoardCache:
    # cache is a dictionary, with keys "board.board_2d.tobytes()"
    # and value: board_val - which is the best possible outcome if both players play optimally
    def __init__(self):
        self.cache = {}

    def set_for_position(self, board, board_val):
        self.cache[board.board_2d.tobytes()] = board_val

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

cache = BoardCache()


# In[5]:
'''
    From the board position, we can see that placing an X (1) in the center is the best possible move because it prevents O from going there (and winning). In addition, and X in the center gives two paths to victory and O cannot block both in one turn

    board:
    [[ 0  1 -1]
     [ 0  0  1]
     [-1  0  0]]

    [(0, -1), (3, -1), (4, 1), (7, -1), (8, -1)]

    Unsatisfactory minimax strategy: (does not pick center or corner square when available)
    [[ 0  0  0]
     [ 1  0  0]
     [ 0  0  0]]
    [[ 0  0  0]
     [ 1  0  0]
     [-1  0  0]]
    [[ 0  0  1]
     [ 1  0  0]
     [-1  0  0]]
    [[ 0  0  1]
     [ 1  0  0]
     [-1  0 -1]]
    [[ 0  0  1]
     [ 1  0  0]
     [-1  1 -1]]
    [[ 0 -1  1]
     [ 1  0  0]
     [-1  1 -1]]
    [[ 0 -1  1]
     [ 1  0  1]
     [-1  1 -1]]
    [[-1 -1  1]
     [ 1  0  1]
     [-1  1 -1]]
    [[-1 -1  1]
     [ 1  1  1]
     [-1  1 -1]]
    (array([1.]), array([9.]))
'''
def play_tic_tac_toe(player_1_strategy=None, player_2_strategy=None, number_of_games=1):
    game_results = np.zeros(number_of_games)
    game_moves_list = []

    def random_move(board):
        move_ind = random.choice(board.get_valid_move_indexes())
        return move_ind

    if player_1_strategy is None:
        player_1_strategy = random_move

    if player_2_strategy is None:
        player_2_strategy = random_move

    for i in range(number_of_games):
        board = Board()
        move_count = 0
        game_moves = np.zeros(9)        # move history
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

    print(game_moves_list)
    return game_results, game_moves_list


play_tic_tac_toe(player_1_strategy=mini_max_strategy, player_2_strategy=None, number_of_games=2)
