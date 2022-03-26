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

from board import Board, BoardCache, CELL_O

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

# In[2]:
def random_move(board):
    move_ind = random.choice(board.get_valid_move_indexes())
    return move_ind

def play_tic_tac_toe(player_1_strategy=None, player_2_strategy=None, number_of_games=1):
    game_results = np.zeros(number_of_games)
    game_moves_list = []

    # def random_move(board):
    #     move_ind = random.choice(board.get_valid_move_indexes())
    #     return move_ind

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

    # print(game_moves_list)
    return game_results, game_moves_list

# In[3]:
# play xxx number of random games
# investigate break down of random games
game_results, game_moves_list = play_tic_tac_toe(player_1_strategy=None, player_2_strategy=None,
                                            number_of_games=10000)

number_of_moves = np.array([np.count_nonzero(x) for x in game_moves_list])

# make dataframe of these results
tictactoe_df = pd.DataFrame()
tictactoe_df['Results'] = game_results
tictactoe_df['Number of Moves'] = number_of_moves
tictactoe_df['Moves'] = game_moves_list

tictactoe_df.head(6)

# In[4]:
print('\t \t \t (%)')

for i in range(5,9):
    print('{} Move Games: \t \t{:.3f}'.format(i, np.count_nonzero(number_of_moves == i) / len(game_results) * 100))

print('9 Move Games (Wins): \t{:.3f}'.format(
        (np.count_nonzero(number_of_moves == 9) - np.count_nonzero(game_results == 0)) / len(game_results) * 100))
print('9 Move Games (Ties): \t{:.3f}'.format( np.count_nonzero(game_results == 0) / len(game_results) * 100))

print()

print('Ties: \t \t \t{:.3f}'.format( np.count_nonzero(game_results == 0) / len(game_results) * 100))
print('X Wins: \t \t{:.3f}'.format( np.count_nonzero(game_results == 1) / len(game_results) * 100))
print('O Wins: \t \t{:.3f}'.format( np.count_nonzero(game_results == -1) / len(game_results) * 100))


# In[5]:
# Use sampling with replacement to get mean and standard deviations
number_resamples = 1000
ties = np.zeros(number_resamples)
X_win = np.zeros(number_resamples)
O_win = np.zeros(number_resamples)

five_move_games = np.zeros(number_resamples)
six_move_games = np.zeros(number_resamples)
seven_move_games = np.zeros(number_resamples)
eight_move_games = np.zeros(number_resamples)
nine_move_wins = np.zeros(number_resamples)

for i in range(number_resamples):
    subsample_df = tictactoe_df.sample(1000, replace=True, ignore_index=True)
    ties[i] = np.count_nonzero(subsample_df['Results'] == 0)
    X_win[i] = np.count_nonzero(subsample_df['Results'] == 1)
    O_win[i] = np.count_nonzero(subsample_df['Results'] == -1)

    five_move_games[i]  = np.count_nonzero(subsample_df['Number of Moves'] == 5)
    six_move_games[i]   = np.count_nonzero(subsample_df['Number of Moves'] == 6)
    seven_move_games[i] = np.count_nonzero(subsample_df['Number of Moves'] == 7)
    eight_move_games[i] = np.count_nonzero(subsample_df['Number of Moves'] == 8)
    nine_move_wins[i]   = np.count_nonzero(subsample_df['Number of Moves'] == 9) - ties[i]


ties = ties/len(subsample_df)*100
X_win = X_win/len(subsample_df)*100
O_win = O_win/len(subsample_df)*100

five_move_games = five_move_games / len(subsample_df)*100
six_move_games  = six_move_games / len(subsample_df)*100
seven_move_games    = seven_move_games / len(subsample_df)*100
eight_move_games    = eight_move_games / len(subsample_df)*100
nine_move_wins  = nine_move_wins / len(subsample_df)*100

# In[6]:
print('\t \t (mean ± std. dev)')
print('Ties: \t \t', '{:.3f} ± {:.3f}'.format(np.mean(ties),  np.std(ties)))
print('X Wins: \t',  '{:.3f} ± {:.3f}'.format(np.mean(X_win), np.std(X_win)))
print('O Wins: \t',  '{:.3f} ± {:.3f}'.format(np.mean(O_win), np.std(O_win)))

print()
print('5 Move Games: \t',  '{:.3f} ± {:.3f}'.format(np.mean(five_move_games), np.std(five_move_games)))
print('6 Move Games: \t',  '{:.3f} ± {:.3f}'.format(np.mean(six_move_games), np.std(six_move_games)))
print('7 Move Games: \t',  '{:.3f} ± {:.3f}'.format(np.mean(seven_move_games), np.std(seven_move_games)))
print('8 Move Games: \t',  '{:.3f} ± {:.3f}'.format(np.mean(eight_move_games), np.std(eight_move_games)))
print('9 Move Wins: \t',  '{:.3f} ± {:.3f}'.format(np.mean(nine_move_wins), np.std(nine_move_wins)))


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
    print('Game states won on move {}: \t \t'.format(k), games_won)
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

    print('Fraction of Game States Won: \t \t {:.3f}'.format(win_percent_state))
    print('Cummulative Win Fraction: \t \t {:.3f}'.format(win_percent_cum * 100))
    print('Fraction of games to advance: \t \t {:.3f}'.format(advance_percent))

    print()

    plt.bar('{}'.format(k), win_percent_cum*100, color = color)

plt.bar('Tie', advance_percent*100, color='gray')
plt.bar('X', X_win_total*100, color='blue')
plt.bar('O', O_win_total*100, color='red')
plt.ylabel('Percentage of Games')

print('Fraction of Games Won by X: \t \t {:.3f}'.format(X_win_total))
print('Fraction of Games Won by O: \t \t {:.3f}'.format(O_win_total))
print('Fraction of Games That Tie: \t \t {:.3f}'.format(advance_percent))


# In[12]:
rows, columns = [2, 8]
ax, fig = axes(rows=rows, columns=columns, row_height=3, column_width=6*3./5)

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

    ax1 = ax[(m-5)+1]
    ax2 = ax[columns+(m-5)+1]
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

# overall win positions
ax1 = ax[columns-1]
ax2 = ax[columns*2-1]
_1 = ax1.pcolormesh(_x_win_pos_, cmap='Blues', vmin=0)
_2 = ax2.pcolormesh(_o_win_pos_, cmap='Reds',  vmin=0)

cbar1 = fig.colorbar(_1, ax=ax1)
cbar2 = fig.colorbar(_2, ax=ax2)

ax1.set_xticks([])
ax1.set_yticks([])

ax2.set_xticks([])
ax2.set_yticks([])

# overall loss positions
ax1 = ax[columns]
ax2 = ax[columns*2]
_1 = ax1.pcolormesh(_x_loss_pos_, cmap='Blues', vmin=0)
_2 = ax2.pcolormesh(_o_loss_pos_, cmap='Reds',  vmin=0)

cbar1 = fig.colorbar(_1, ax=ax1)
cbar2 = fig.colorbar(_2, ax=ax2)

ax1.set_xticks([])
ax1.set_yticks([])

ax2.set_xticks([])
ax2.set_yticks([])

ax[1].set_ylabel('X Position Frequency', fontweight = 'bold', fontsize=18)
ax[columns+1].set_ylabel('O Position Frequency', fontweight = 'bold', fontsize=18)

ax[1].set_title('5 Move Games (X Wins)')
ax[2].set_title('6 Move Games (O Wins)')
ax[3].set_title('7 Move Games (X Wins)')
ax[4].set_title('8 Move Games (O Wins)')
ax[5].set_title('9 Move Games (X Wins)')
ax[6].set_title('9 Move Games (Draw)')
ax[7].set_title('Overall Winning Positions')
ax[8].set_title('Overall Losing Positions')

fig.tight_layout()
fig.savefig('Tic-Tac-Toe_PlayFrequency.png', format='png', dpi=300, bbox_inches = "tight")

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

def strategy_center(board):
    # optimal game strategy
    # 1. pick the center square
    available_moves = board.get_valid_move_indexes()
    middle_square = 4

    if middle_square in available_moves:
        return middle_square

    move_ind = random.choice(available_moves)

    return move_ind

def strategy_corners(board):
    # optimal game strategy
    # 1. pick the center square
    available_moves = board.get_valid_move_indexes()
    corner_squares = [0, 2, 6, 8]
    avail_corners = [x for x in corner_squares if x in available_moves]

    if avail_corners:
        move_ind = random.choice(avail_corners)
    else:
        move_ind = random.choice(available_moves)

    return move_ind

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
strategy_names = [  'Favor Edge-Middle',
                    'Random',
                    'Favor Corners',
                    'Favor Center',
                    'Favor Corners > Center',
                    'Favor Center > Corners']

strategies = [  strategy_midedge_corners_center,
                None,
                strategy_corners,
                strategy_center,
                strategy_corners_center_else,
                strategy_center_corners_else]

# In[22]:
# player 1 vs random player
df_p1 = pd.DataFrame()

for strat, strat_name in zip(strategies, strategy_names):
    # play the games with different strategies
    game_res, game_moves_list = play_tic_tac_toe(player_1_strategy=strat, player_2_strategy=None, number_of_games=4000)
    #
    # make a list of the game results (+1 = X Wins, -1 = O Wins, 0 = Tie)
    res_ = [+1, -1, 0]
    _ = [np.count_nonzero(game_res == r) / len(game_res) for r in res_]

    # add this list to a dataframe
    df_p1[strat_name] = _

df_p1.index = ['X Wins', 'O Wins', 'Ties']


# In[23]:
# player 2 vs random player
df_p2 = pd.DataFrame()

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
ax[2].set_title('Random Player X vs Player O Strategy')

fig.tight_layout()
fig.savefig('VaryingStrategyResults.png', format='png', dpi=300, bbox_inches = "tight")

# In[25]:

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

# def choose_min_or_max_for_comparison(board):
#     # returns function min() or function max() depending on whose turn it is
#     turn = board.get_turn()
#     return min if turn == CELL_O else max
#
# # recursive functions calculate_position_value, get_position_value used to evaluate the value of each subsequent move
# def calculate_position_value(board):
#     # board is a class object
#     if board.is_gameover():
#         return board.get_game_result()
#
#     valid_move_indexes = board.get_valid_move_indexes()
#
#     values = [get_position_value(board.play_move(m))
#               for m in valid_move_indexes]
#
#     min_or_max = choose_min_or_max_for_comparison(board)
#     position_value = min_or_max(values)
#
#     return position_value
#
# def get_position_value(board):
#     # cached_position_value, found = get_position_value_from_cache(board)
#     # if found:
#     #     return cached_position_value
#
#     position_value = calculate_position_value(board)
#
#     # put_position_value_in_cache(board, position_value)
#
#     return position_value
#
# def get_move_value_pairs(board):
#     valid_move_indexes = board.get_valid_move_indexes()
#
#     # assertion error if valid_move_indexes is empty
#     assert valid_move_indexes, "never call with an end-position"
#
#     # (index, value)
#     move_value_pairs = [(m, get_position_value(board.play_move(m)))
#                         for m in valid_move_indexes]
#
#     return move_value_pairs
#
# def mini_max_strategy(board): # mini_max_strategy
#     min_or_max = choose_min_or_max_for_comparison(board)
#     move_value_pairs = get_move_value_pairs(board)
#     move, best_value = min_or_max(move_value_pairs, key=lambda m_v_p: m_v_p[1])
#
#     best_move_value_pairs = [m_v_p for m_v_p in move_value_pairs if m_v_p[1] == best_value]
#     chosen_move, _ = random.choice(best_move_value_pairs)
#
#     return chosen_move
#
# def mini_max_strategy_pref_center(board):
#     center_square = 4
#     if center_square in board.get_valid_move_indexes():
#         return center_square
#
#     return mini_max_strategy(board)


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
# In[100]:
''' Min/Max game strategy '''
def get_position_value(board):
    # check if the board position is already cached
    cached_position_value, found = cache.get_for_position(board)
    if found:
        return cached_position_value[0]

    position_value = calculate_position_value(board)

    cache.set_for_position(board, position_value)

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


def mini_max_strategy_pref_center(board):
    center_square = 4
    if center_square in board.get_valid_move_indexes():
        return center_square

    return mini_max_strategy(board)


def mini_max_strategy_center_corners(board): # mini_max_strategy
    center_square = 4
    if center_square in board.get_valid_move_indexes():
        return center_square

    min_or_max = choose_min_or_max_for_comparison(board)
    move_value_pairs = get_move_value_pairs(board)
    move, best_value = min_or_max(move_value_pairs, key=lambda m_v_p: m_v_p[1])

    best_move_value_pairs = [m_v_p for m_v_p in move_value_pairs if m_v_p[1] == best_value]

    # choose preferentially corner squares
    corners = [0, 2, 6, 8]
    available_corners = [b[0] for b in best_move_value_pairs if b[0] in corners]
    if available_corners:
        return random.choice(available_corners)

    chosen_move, _ = random.choice(best_move_value_pairs)

    return chosen_move


# In[99]:
cache = BoardCache()

board = Board()
print(board.board_2d)

get_position_value(board)
print(len(cache.cache))

# In[100]:
game_results, game_moves = play_tic_tac_toe(player_1_strategy=mini_max_strategy, player_2_strategy=None, number_of_games=1)

print(len(cache.cache))

# In[101]:
print(np.count_nonzero(game_results == 1) / len(game_results))

# In[200]:
game_results, game_moves = play_tic_tac_toe(player_1_strategy=mini_max_strategy_pref_center, player_2_strategy=None, number_of_games=10000)

# In[201]:
print(np.count_nonzero(game_results == 1) / len(game_results))

# In[100]:
game_results, game_moves = play_tic_tac_toe(player_1_strategy=None, player_2_strategy=mini_max_strategy, number_of_games=10000)

# In[101]:
print(np.count_nonzero(game_results == -1) / len(game_results))

# In[200]:
game_results, game_moves = play_tic_tac_toe(player_1_strategy=None, player_2_strategy=mini_max_strategy_pref_center, number_of_games=10000)

# In[201]:
print(np.count_nonzero(game_results == -1) / len(game_results))

# In[400]:
game_results, game_moves = play_tic_tac_toe(player_1_strategy=mini_max_strategy_pref_center, player_2_strategy=mini_max_strategy, number_of_games=100)


# In[1000]:
# player 1 vs random player


#           |   random2 |  Minimax2 |
#           |----------:|----------:|
# random1   |       res |       res |
# Minimax1  |       res |       res |

strategy_names = ['Random', 'Minimax', 'Minimax Center', 'Minimax Center > Corners']
strategies = [None, mini_max_strategy, mini_max_strategy_pref_center, mini_max_strategy_center_corners]

df_ = pd.DataFrame(index=strategy_names, columns=strategy_names)

display(df_)
print()
for strat1, strat_name1 in zip(strategies, strategy_names):
    for strat2, strat_name2 in zip(strategies, strategy_names):
        if None not in [strat1, strat2]:
            continue
        # play the games with different strategies
        game_res, game_moves_list = play_tic_tac_toe(player_1_strategy=strat1, player_2_strategy=strat2, number_of_games=1000)

        # make a list of the game results (+1 = X Wins, -1 = O Wins, 0 = Tie)
        res_ = [+1, -1, 0]
        _ = [np.count_nonzero(game_res == r) / len(game_res) for r in res_]

        # add this list to a dataframe
        df_.at[strat_name1, strat_name2] = [_]

display(df_)

# In[1000]:
strategy_names = ['Random', 'Minimax', 'Minimax Center', 'Minimax Center > Corners']
strategies = [None, mini_max_strategy, mini_max_strategy_pref_center, mini_max_strategy_center_corners]

df_ = pd.DataFrame(index=strategy_names, columns=strategy_names)

display(df_)
print()
for strat1, strat_name1 in zip(strategies, strategy_names):
    for strat2, strat_name2 in zip(strategies, strategy_names):
        if None not in [strat1, strat2]: # only look at minimax vs random
            continue
        # play the games with different strategies
        game_res, game_moves_list = play_tic_tac_toe(player_1_strategy=strat1, player_2_strategy=strat2, number_of_games=4000)

        # make a list of the game results (+1 = X Wins, -1 = O Wins, 0 = Tie)
        res_ = [+1, -1, 0]
        _ = [np.count_nonzero(game_res == r) / len(game_res) for r in res_]

        # add this list to a dataframe
        df_.at[strat_name1, strat_name2] = [_]

display(df_)

# In[1000]:
# player X
end_game_result = +1

while end_game_result == 1:

    board = Board()
    board = board.play_move(4) # both strategies play the center square first

    while not board.is_gameover():
        # play random move
        moves = board.get_valid_move_indexes()
        mov_ind = random.choice(moves)
        board = board.play_move(mov_ind)
        print(board.board_2d)

        mov_ind = mini_max_strategy_center_corners(board)
        print(mov_ind)
        print(mini_max_strategy_pref_center(board))

        board = board.play_move(mov_ind)
        print()

    end_game_result = board.get_game_result()

print(end_game_result)
print(board.board_2d)

# In[1000]:
# Player 0

end_game_result = -1

while end_game_result == -1:
    board = Board()

    while not board.is_gameover():
        # play random move
        moves = board.get_valid_move_indexes()
        mov_ind = random.choice(moves)
        board = board.play_move(random_move(board))
        print(board.board_2d)

        if len(moves) > 1:
            mov_ind = mini_max_strategy_center_corners(board)
            print(mov_ind)
            print(mini_max_strategy_pref_center(board))

            board = board.play_move(mov_ind)
        print()

    end_game_result = board.get_game_result()

print(end_game_result)
print(board.board_2d)

# In[2000]:
'''
    Can mini_max_strategy_pref_center be improved?
    Let's look at the games that end in ties

    Results:
    [[ 0 -1  0]
     [ 1  1 -1]
     [-1  1  0]]

    [[-1 -1  1]
     [ 1  1 -1]
     [-1  1  0]]

     To maximize wins the best moves to choose are upper-left and lower-right, then there is a 50% chance of winning, instead of tying.

     So there is a possibility still to improve the winning percentage.
'''
end_game_result = +1

while end_game_result == 1:

    board = Board()
    board = board.play_move(4) # both strategies play the center square first

    while not board.is_gameover():
        # play random move
        moves = board.get_valid_move_indexes()
        mov_ind = random.choice(moves)
        board = board.play_move(mov_ind)
        print(board.board_2d)

        mov_ind = mini_max_strategy_pref_center(board)

        board = board.play_move(mov_ind)
        print()

    end_game_result = board.get_game_result()

print(end_game_result)
print(board.board_2d)

# In[3000]:
'''
    Q-Learning
'''
def get_position_value_qtable(board, cache):
    # check if the board position is already cached
    cached_position_value, found = cache.get_for_position(board)
    # print(cached_position_value, found)

    if found:
        return cached_position_value#[0]

    position_value = calculate_position_value(board, cache)
    # print(position_value)

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


# In[200]:
keys = list(qtable1.cache.keys())

qtable1.cache[keys[0]]


def mini_max_strategy(board): # mini_max_strategy
    min_or_max = choose_min_or_max_for_comparison(board)
    move_value_pairs = get_move_value_pairs(board)
    move, best_value = min_or_max(move_value_pairs, key=lambda m_v_p: m_v_p[1])

    best_move_value_pairs = [m_v_p for m_v_p in move_value_pairs if m_v_p[1] == best_value]
    chosen_move, _ = random.choice(best_move_value_pairs)

    return chosen_move



def Q_table_Strat(board):



    return choose_move


# In[400]:
board = Board()

board = board.play_move(6)
print(board.board_2d)
opponent_move = mini_max_strategy(board)
board = board.play_move(opponent_move)
print(board.board_2d)


board = board.play_move(2)
print(board.board_2d)
opponent_move = mini_max_strategy(board)
board = board.play_move(opponent_move)
print(board.board_2d)

board = board.play_move(0)
print(board.board_2d)
opponent_move = mini_max_strategy(board)
board = board.play_move(opponent_move)

print(board.board_2d)
