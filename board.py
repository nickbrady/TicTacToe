import numpy as np
import itertools
from transform import Transform, Identity, Rotate90, Flip

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





TRANSFORMATIONS = [ Identity(),
                    Rotate90(1),
                    Rotate90(2),
                    Rotate90(3),
                    Flip(np.flipud),
                    Flip(np.fliplr),
                    Transform(Rotate90(1), Flip(np.flipud)),
                    Transform(Rotate90(1), Flip(np.fliplr))]

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







# class BoardCache:
#
#     def __init__(self):
#         self.cache = {} # initialize a dictionary, key=board_2d.tobytes(),
#                         #                          value= board position value (1, 0, -1)
#
#     def get_for_position(self, board):
#         board_orientations = self.get_symmetrical_board_orientations(board.board_2d)
#
#         # boolean list of which board orientations are in the cache
#         _ = [b_[0].tobytes() in self.cache for b_ in board_orientations]
#
#         # if not in cache - return False
#         if not any(_):
#             return None, False
#
#         # else
#         board_orientations = list(itertools.compress(board_orientations, _))
#         board_value = self.cache[board_orientations[0][0].tobytes()]
#
#         return board_value, True
#
#     def set_for_position(self, board, position_value):
#         self.cache[board.board_2d.tobytes()] = position_value
#
#     def get_symmetrical_board_orientations(self, board_2d):
#         return [(t.transform(board_2d), t) for t in TRANSFORMATIONS]
#
#     def reset(self):
#         self.cache = {}