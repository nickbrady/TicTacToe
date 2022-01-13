# Tic-Tac-Toe

Apply Machine Learning to simple game: Tic-Tac-Toe

## Introduction
Not intended to be visually appealing or for human game-play. Main objective is to quantitatively explore different (computer) game-play strategies.

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

Play a game
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

## Available Strategies
1. Random
2. Random with Tiers: center, corner squares, edge middle squares (add picture of these squares)
3. Minimax
4. Minimax with preference for center square

## References
Following this blog post
https://nestedsoftware.com/2019/06/15/tic-tac-toe-with-the-minimax-algorithm-5988.123625.html

Code based on this repository:
https://github.com/nestedsoftware/tictac/blob/220bbdc6103ff012ec60b5b424e1566205349588/tictac/board.py#L182

Tic-Tac-Toe Math:
http://www.se16.info/hgb/tictactoe.htm

## Summary of Results
The source code for analyzing and visualizing the results from different game strategies can be found in ```GameAnalysis.py```

Using ```play_tic_tac_toe``` a high number of random games (each player chooses moves random from a list of possible moves) can be played. From these games statistics can be generated to get a break down of the results. 

|              | (mean ± std. dev) |
|--------------|:--------------:|
| Ties  	     | 13.739 ± 1.097 |
| X Wins       | 57.362 ± 1.525 |
| O Wins       | 28.899 ± 1.431 |
| -            | -              |
| 5 Move Games | 9.484 ± 0.913  |
| 6 Move Games | 8.677 ± 0.899  |
| 7 Move Games | 26.583 ± 1.412 |
| 8 Move Games | 20.223 ± 1.283 |
| 9 Move Wins  | 21.296 ± 1.318 |

The break down of ties, X wins, and O wins are in good agreement with the result from nestedsoftware.com. But I was more curious if these results made practical sense.

Because the game is very simple it is possible to analytically solve the number of possible board permutations. Which can be found at http://www.se16.info/hgb/tictactoe.htm. Because the percentage of board orientation did not seem to align with the observed winning percentages from my random games, I decided to investigate further to see where the discrepency lay (if at all). 

I wrote a recursive function ```board_state_moves_and_result``` to systematically record every possible board position. The results are tabulated below:
|   5 Moves |   6 Moves |   7 Moves |   8 Moves |   9 Moves |   Ties |  Total Games |
|----------:|----------:|----------:|----------:|----------:|-------:|-------------:|
|      1440 |      5328 |     47952 |     72576 |     81792 |  46080 |        255168|

Unsurprisingly, these results are identical to the results obtained from www.se16.info. Then using these results and some logical thinking, one can determine the expected number of games that will end at each move. 

1. 9! / (9-5)! = 15120 is the number of game states after the 5th move
2. 1440 game states are won on move 5 and therefore *DO NOT* proceed to the sixth move
3. There are therefore (9! / (9-5)! - 1440) * (9-5) game states after the 6th move
4. 5328 games states are won on move 6
5. ((9! / (9-5)! - 1440) * (9-5) - 5328) * (9-6) = game states after the 7th move
6. (((9! / (9-5)! - 1440) * (9-5) - 5328) * (9-6) - 47952) * (9-7) = game states after 8th move
7. ((((9! / (9-5)! - 1440) * (9-5) - 5328) * (9-6) - 47952) * (9-7) - 72576) * (9-8) after the 9th move

The percentage of games that are won after each nth move can then calculated.  

|              |        5 Moves |        6 Moves |        7 Moves |        8 Moves |        9 Moves |           Ties |         X Wins |        O Wins |
|--------------|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|--------------:|
| Analytical   |          9.524 |          8.810 |         26.429 |         20.000 |         22.540 |           12.7 |           58.5 |          28.8 |
| Estimaged    |  9.484 ± 0.913 |  8.677 ± 0.899 | 26.583 ± 1.412 | 20.223 ± 1.283 | 21.296 ± 1.318 | 13.739 ± 1.097 | 57.362 ± 1.525 | 28.899 ± 1.431|
