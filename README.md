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
print(board.board_2d)
```

Get a list of the indices already occupied (illegal_move)
And a list of the valid moves
```
print(board.illegal_move)
print(board.get_valid_move_indexes())
```



