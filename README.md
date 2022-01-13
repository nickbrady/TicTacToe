# Tic-Tac-Toe

Applying Machine Learning to simple game: Tic-Tac-Toe

Following this blog post
https://nestedsoftware.com/2019/06/15/tic-tac-toe-with-the-minimax-algorithm-5988.123625.html

Code based on this repository:
https://github.com/nestedsoftware/tictac/blob/220bbdc6103ff012ec60b5b424e1566205349588/tictac/board.py#L182

Tic-Tac-Toe Math:
http://www.se16.info/hgb/tictactoe.htm

## Basic Usage
```
from board import Board, BoardCache
from board import CELL_O
```

Creating a board and playing a move:
```
board = Board()
board = board.play_move(1)
```

Get a 2D display of the board
```
board.board_2d
```

Get a list of the indices already occupied (illegal_move) and a list of the valid moves
```
board.illegal_move
board.get_valid_move_indexes()
```

Play a winning game
```
board = Board()
board = board.play_move(3)
board = board.play_move(4)
board = board.play_move(1)
board = board.play_move(2)
board = board.play_move(5)
board = board.play_move(6)
```

Visualize the board:
```
board.board_2d
board.print_board()
```

Is the game over?
```
board.is_gameover()
```

Who won?
```
board.get_game_result()
```
