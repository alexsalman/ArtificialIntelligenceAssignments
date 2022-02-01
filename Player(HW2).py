# Alex Salman - aalsalma@ucsc.edu
# CSE240 Artificial Intelligence
# Winter 2022
# Assignment 2
# Due Date: Jan 31 at 11:59 pm

import numpy as np


# get the next available row in a specific column
def get_next_open_row(board, col):
    for r in range(len(board)):
        if board[r][col] == 0:
            return r


# add 1 or 2 to the board
def drop_piece(board, row, col, player):
    board[row][col] = player


# check all possible ways (combinations) to win
def winning_move(board, player):
    # horizontal wins
    for c in range(len(board[0])-3):
        for r in range(len(board)):
            if board[r][c] == player and board[r][c+1] == player and board[r][c+2] == player and board[r][c+3] == player:
                return True
    # vertical wins
    for c in range(len(board[0])):
        for r in range(len(board)-3):
            if board[r][c] == player and board[r+1][c] == player and board[r+2][c] == player and board[r+3][c] == player:
                return True
    # positive diagonals wins
    for c in range(len(board[0])-3):
        for r in range(len(board)-3):
            if board[r][c] == player and board[r+1][c+1] == player and board[r+2][c+2] == player and board[r+3][c+3] == player:
                return True
    # negative diagonals wins
    for c in range(len(board[0])-3):
        for r in range(3, len(board)):
            if board[r][c] == player and board[r-1][c+1] == player and board[r-2][c+2] == player and board[r-3][c+3] == player:
                return True


# check if the location is valid, equals to zero
def valid_location(board, col):
    return board[len(board)-1][col] == 0


# iterate through the board columns to make a list
def valid_columns(board):
    valid_columns_list = []
    for col in range(len(board[0])):
        if valid_location(board, col):
            valid_columns_list.append(col)
    return valid_columns_list


# check if player 1 is winning, player 2 is winning, or the board is all used
def terminal_node(board):
    return winning_move(board, 1) or winning_move(board, 2) or len(valid_columns(board)) == 0


# ai bot playing
class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    # evaluate function that gives heuristic value for combination
    def evaluation_function(self, window):
        score = 0
        opponent = 1
        if self.player_number == 1:
            opponent = 2

        if window.count(self.player_number) == 4:
            score += 100
        elif window.count(self.player_number) == 3 and window.count(0) == 1:
            score += 5
        elif window.count(self.player_number) == 2 and window.count(0) == 2:
            score += 2

        if window.count(opponent) == 3 and window.count(0) == 1:
            score -= 4

        return score

        """
        Given the current stat of the board, return the scalar value that
        represents the evaluation function for the current player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """

    # score the position of the board
    def score_position(self, board):
        score = 0
        center_array = [int(i) for i in list(board[:, len(board[0])//2])]
        center_count = center_array.count(self.player_number)
        score += center_count * 3
        # horizontal
        for r in range(len(board)):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(len(board[0])-3):
                window = row_array[c:c+4]
                score += self.evaluation_function(window)
        # vertical
        for c in range(len(board[0])):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(len(board)-3):
                window = col_array[r:r+4]
                score += self.evaluation_function(window)
        # positive diagonal
        for r in range(len(board)-3):
            for c in range(len(board[0])-3):
                window = [board[r+i][c+i] for i in range(4)]
                score += self.evaluation_function(window)
        # negative diagonal
        for r in range(len(board)-3):
            for c in range(len(board[0])-3):
                window = [board[r+3-i][c+i] for i in range(4)]
                score += self.evaluation_function(window)

        return score

    # alpha-beta pruning algorithm
    def minimax(self, board, depth, alpha, beta, maximizer):
        valid_locations = valid_columns(board)
        is_terminal = terminal_node(board)
        if depth == 0 or is_terminal:
            if is_terminal:
                if winning_move(board, 1):
                    return None, 1000
                elif winning_move(board, 2):
                    return None, -1000
                else:
                    return None, 0
            else:
                return None, self.score_position(board)

        if maximizer:
            value = float('-inf')
            column = np.random.choice(valid_locations)
            for col in valid_locations:
                row = get_next_open_row(board, col)
                b_copy = board.copy()
                drop_piece(b_copy, row, col, self.player_number)
                new_score = self.minimax(b_copy, depth-1, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value

        else:
            value = float('inf')
            column = np.random.choice(valid_locations)
            for col in valid_locations:
                row = get_next_open_row(board, col)
                b_copy = board.copy()
                if self.player_number == 1:
                    drop_piece(b_copy, row, col, 2)
                else:
                    drop_piece(b_copy, row, col, 1)
                new_score = self.minimax(b_copy, depth-1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

    # alpha beta pruning algorithm along with depth-limited search for efficiency
    def get_alpha_beta_move(self, board):
        depth = 3
        alpha = float('-inf')
        beta = float('inf')
        board = np.flip(board)
        column, minimax_score = self.minimax(board, depth, alpha, beta, True)
        return column


        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        raise NotImplementedError('Whoops I don\'t know what to do')

    def expectimax(self, board, depth, alpha, beta, maximizing):
        valid_locations = valid_columns(board)
        is_terminal = terminal_node(board)

        if depth == 0 or is_terminal:
            if is_terminal:
                if winning_move(board, 1):
                    return None, 1000
                elif winning_move(board, 2):
                    return None, -1000
                else:
                    return None, 0
            else:
                return None, self.score_position(board)

        if maximizing:
            value = float('-inf')
            column = np.random.choice(valid_locations)
            for col in valid_locations:
                row = get_next_open_row(board, col)
                b_copy = board.copy()
                drop_piece(b_copy, row, col, self.player_number)
                new_score = self.expectimax(b_copy, depth-1, alpha, beta, False)[1]

                if new_score > value:
                    value = new_score
                    column = col

                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value

        else:
            value = 0
            column = np.random.choice(valid_locations)
            for col in valid_locations:
                row = get_next_open_row(board, col)
                b_copy = board.copy()
                if self.player_number == 1:
                    drop_piece(b_copy, row, col, 2)
                else:
                    drop_piece(b_copy, row, col, 1)

                new_score = self.expectimax(b_copy, depth-1, alpha, beta, True)[1]

                if new_score <= value:
                    value = new_score
                    column = col

                beta = np.floor(value/len(valid_locations))
                if alpha >= beta:
                    break
            return column, value

    # expectimax search algorithm
    def get_expectimax_move(self, board):
        depth = 5
        alpha = float('-inf')

        col, expectimax_score = self.expectimax(board, depth, alpha, 0, True)
        return col
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        raise NotImplementedError('Whoops I don\'t know what to do')


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                valid_cols.append(col)

        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move

# references
# https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
# https://en.wikipedia.org/wiki/Expectiminimax
# https://www.scirp.org/journal/paperinformation.aspx?paperid=90972
# https://www.youtube.com/watch?v=UYgyRArKDEs&list=PLFCB5Dp81iNV_inzM-R9AKkZZlePCZdtV
