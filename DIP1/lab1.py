import numpy as np

def chessboard_indices(size=8):
    board = np.indices((size, size)).sum(axis=0) % 2
    return board.astype(bool)

chess_board = chessboard_indices(8)
print("Шахматная доска 8x8:")
print(chess_board.astype(int))