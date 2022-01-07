import numpy as np
import random
import math

theBoard = {'7': ' ' , '8': ' ' , '9': ' ' ,
            '4': ' ' , '5': ' ' , '6': ' ' ,
            '1': ' ' , '2': ' ' , '3': ' ' }

def printBoard(board):
    print(board['7'] + '|' + board['8'] + '|' + board['9'])
    print('-+-+-')
    print(board['4'] + '|' + board['5'] + '|' + board['6'])
    print('-+-+-')
    print(board['1'] + '|' + board['2'] + '|' + board['3'])

printBoard(theBoard)

# In[1]:
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

##############################################################################################################
def game():

    turn = 'X'
    count = 0
    game_over = False

    while not game_over:

        printBoard(theBoard)
        print("It's your turn," + turn + ".Move to which place?")

        move = input()

        if theBoard[move] == ' ':
            theBoard[move] = turn
            count += 1
        else:
            print("That place is already filled.\nMove to which place?")
            continue

        if count >= 5:
            # check for win conditions
            game_won = any((
                            theBoard['7'] == theBoard['8'] == theBoard['9'] != ' ',     # across top row
                            theBoard['4'] == theBoard['5'] == theBoard['6'] != ' ',     # across middle row
                            theBoard['1'] == theBoard['2'] == theBoard['3'] != ' ',     # across bottom row

                            theBoard['1'] == theBoard['4'] == theBoard['7'] != ' ',     # down left column
                            theBoard['2'] == theBoard['5'] == theBoard['8'] != ' ',     # down middle column
                            theBoard['3'] == theBoard['6'] == theBoard['9'] != ' ',     # down right column

                            theBoard['7'] == theBoard['5'] == theBoard['3'] != ' ',     # diagonal \
                            theBoard['1'] == theBoard['5'] == theBoard['9'] != ' '      # diagonal /
                           ))

            if game_won:
                print('Game Over', '{} Won!'.format(turn))
                printBoard(theBoard)
                game_over = True
                break

            if count == 9:
                print('Game Over', 'It\'s a tie')
                printBoard(theBoard)
                game_over = True
                break

        # Now we have to change the player after every move.
        if turn =='X':
            turn = 'O'
        else:
            turn = 'X'


##############################################################################################################
def game_vs_computer():

    count = 0
    game_over = False

    while not game_over:

        turn = 'X'

        printBoard(theBoard)
        print("It's your turn," + turn + ".Move to which place?")

        move = input()

        if theBoard[move] == ' ':
            theBoard[move] = turn
            count += 1
        else:
            print("That place is already filled.\nMove to which place?")
            continue

        if count >= 5:
            # check for win conditions
            game_won = any((
                            theBoard['7'] == theBoard['8'] == theBoard['9'] != ' ',     # across top row
                            theBoard['4'] == theBoard['5'] == theBoard['6'] != ' ',     # across middle row
                            theBoard['1'] == theBoard['2'] == theBoard['3'] != ' ',     # across bottom row

                            theBoard['1'] == theBoard['4'] == theBoard['7'] != ' ',     # down left column
                            theBoard['2'] == theBoard['5'] == theBoard['8'] != ' ',     # down middle column
                            theBoard['3'] == theBoard['6'] == theBoard['9'] != ' ',     # down right column

                            theBoard['7'] == theBoard['5'] == theBoard['3'] != ' ',     # diagonal \
                            theBoard['1'] == theBoard['5'] == theBoard['9'] != ' '      # diagonal /
                           ))

            if game_won:
                print('Game Over', '{} Won!'.format(turn))
                printBoard(theBoard)
                game_over = True
                break

            if count == 9:
                print('Game Over', 'It\'s a tie')
                printBoard(theBoard)
                game_over = True
                break

        # Computer Turn
        turn = 'O'

        computer_valid_move = False
        while not computer_valid_move:
            move = str(random.randint(1, 9))
            if theBoard[move] == ' ':
                computer_valid_move = True

        theBoard[move] = turn
        count += 1


##############################################################################################################
def game_computer_vs_computer():

    turn = 'X'
    count = 0
    game_over = False

    while not game_over:

        printBoard(theBoard)
        print("It's your turn," + turn + ".Move to which place?")

        computer_valid_move = False
        while not computer_valid_move:
            move = str(random.randint(1, 9))
            if theBoard[move] == ' ':
                computer_valid_move = True

        theBoard[move] = turn
        count += 1

        if count >= 5:
            # check for win conditions
            game_won = any((
                            theBoard['7'] == theBoard['8'] == theBoard['9'] != ' ',     # across top row
                            theBoard['4'] == theBoard['5'] == theBoard['6'] != ' ',     # across middle row
                            theBoard['1'] == theBoard['2'] == theBoard['3'] != ' ',     # across bottom row

                            theBoard['1'] == theBoard['4'] == theBoard['7'] != ' ',     # down left column
                            theBoard['2'] == theBoard['5'] == theBoard['8'] != ' ',     # down middle column
                            theBoard['3'] == theBoard['6'] == theBoard['9'] != ' ',     # down right column

                            theBoard['7'] == theBoard['5'] == theBoard['3'] != ' ',     # diagonal \
                            theBoard['1'] == theBoard['5'] == theBoard['9'] != ' '      # diagonal /
                           ))

            if game_won:
                print('Game Over', '{} Won!'.format(turn))
                printBoard(theBoard)
                game_over = True
                break

            if count == 9:
                print('Game Over', 'It\'s a tie')
                printBoard(theBoard)
                game_over = True
                break

        # Now we have to change the player after every move.
        if turn =='X':
            turn = 'O'
        else:
            turn = 'X'

# In[10]:
game_computer_vs_computer()

# In[100]:

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

    # def get_random_valid_move_index(self):                    # don't really want this; prefer random play
    #     return random.choice(self.get_valid_move_indexes())   # to be done outside of the board

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



# In[200]:
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

# In[200]:
print(np.count_nonzero(game_moves == 5))
print(np.count_nonzero(game_moves == 6))
print(np.count_nonzero(game_moves == 7))
print(np.count_nonzero(game_moves == 8))
print(np.count_nonzero(game_moves == 9) - np.count_nonzero(game_results == 0))
print(np.count_nonzero(game_results == 0))
# np.count_nonzero(board.board)

# In[201]:
print(game_results[:10])

print('Ties: \t', np.count_nonzero(game_results == 0))
print('X Wins: ', np.count_nonzero(game_results == 1))
print('O Wins: ', np.count_nonzero(game_results == -1))

# In[202]:
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
print(np.mean(ties), np.std(ties))
print(np.mean(X_win), np.std(X_win))
print(np.mean(O_win), np.std(O_win))

# In[203]:
# use game board to check all possible games
board = Board()

legal_moves_1 = board.get_valid_move_indexes()

board_list = []

five_move_games = 0
six_move_games = 0
seven_move_games = 0
eight_move_games = 0
nine_move_wins = 0
nine_move_ties = 0

for l1 in legal_moves_1:
    board = Board()
    board = board.play_move(l1)

    legal_moves_2 = board.get_valid_move_indexes()
    for l2 in legal_moves_2:
        board_state = board.play_move(l2)
        legal_moves_3 = board_state.get_valid_move_indexes()

        for l3 in legal_moves_3:
            board_state = board.play_move(l2).play_move(l3)
            legal_moves_4 = board_state.get_valid_move_indexes()

            for l4 in legal_moves_4:
                board_state = board.play_move(l2).play_move(l3).play_move(l4)
                legal_moves_5 = board_state.get_valid_move_indexes()

                for l5 in legal_moves_5:
                    board_state = board.play_move(l2).play_move(l3).play_move(l4).play_move(l5)

                    if board_state.is_gameover():
                        board_list.append(board_state.board_2d)
                        legal_moves_6 = []
                        five_move_games += 1

                    else:
                        legal_moves_6 = board_state.get_valid_move_indexes()

                    for l6 in legal_moves_6:
                        board_state = board.play_move(l2).play_move(l3).play_move(l4).play_move(l5).play_move(l6)

                        if board_state.is_gameover():
                            board_list.append(board_state.board_2d)
                            legal_moves_7 = []
                            six_move_games += 1

                        else:
                            legal_moves_7 = board_state.get_valid_move_indexes()

                        for l7 in legal_moves_7:
                            board_state = board.play_move(l2).play_move(l3).play_move(l4).play_move(l5).play_move(l6).play_move(l7)

                            if board_state.is_gameover():
                                board_list.append(board_state.board_2d)
                                legal_moves_8 = []
                                seven_move_games += 1

                            else:
                                legal_moves_8 = board_state.get_valid_move_indexes()

                            for l8 in legal_moves_8:
                                board_state = board.play_move(l2).play_move(l3).play_move(l4).play_move(l5).play_move(l6).play_move(l7).play_move(l8)

                                if board_state.is_gameover():
                                    board_list.append(board_state.board_2d)
                                    legal_moves_9 = []
                                    eight_move_games += 1

                                else:
                                    legal_moves_9 = board_state.get_valid_move_indexes()

                                for l9 in legal_moves_9:
                                    board_state = board.play_move(l2).play_move(l3).play_move(l4).play_move(l5).play_move(l6).play_move(l7).play_move(l8).play_move(l9)

                                    board_list.append(board_state.board_2d)

                                    if board_state.get_game_result() == 1: # only player one can win
                                        nine_move_wins += 1

                                    else:
                                        nine_move_ties += 1





# In[205]:
print(len(board_list))

print(five_move_games)
print(six_move_games)
print(seven_move_games)
print(eight_move_games)
print(nine_move_wins)
print(nine_move_ties)
