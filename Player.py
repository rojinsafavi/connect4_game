import numpy as np
from operator import itemgetter
from random import shuffle


inf = float('inf')

class Helper:
    '''Helper function containing usefull methods.'''

    def __init__(self, player_number):
        self.player_number = player_number


    def other_player(self):
        '''given the current player, return the next player'''
        return 2 if self.player_number == 1 else 1

    def valid_moves(self, board):
        '''Legal moves that the user can make.
        Input: board
        Return: legal moves'''

        rows, cols = np.nonzero(board == 0)
        empty_sqrs = [(rows[i],cols[i]) for i in range(len(rows))]

        legal_moves = []
        for i in set(cols):
            new = []
            for m in empty_sqrs:
                if m[1] == i:
                    new.append(m)
            legal_moves.append(max(new))
        shuffle(legal_moves)
        return legal_moves

    def game_completed(self, board, k_in_row, player_num):

        '''adapted from ConnectFour.py with minor modificationsself.
        Input:
            board
            k_in_row: the number of k connects
            player_num: the player
        output: the number of wins with this move'''

        total_wins = 0
        player_win_str = ('{0}' * k_in_row).format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            counter = 0
            for row in b:
                if player_win_str in to_str(row):
                    counter += to_str(row).count(player_win_str)
            return counter

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            counter = 0
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    counter += to_str(root_diag).count(player_win_str)

                for i in range(1, b.shape[1]-3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            counter += diag.count(player_win_str)

            return counter
        total_wins = check_horizontal(board) + check_verticle(board) + check_diagonal(board)
        return total_wins


class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)
        self.helper = Helper(self.player_number)


    def evaluation_function(self, board):
        '''Given the current stat of the board, return the scalar value that
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
            The utility value for the current board'''

        board_eval = 0
        opponent = self.helper.other_player()
        weight = {4:1000,3:80, 2: 10}

        for k_in_row in weight.keys():
            board_eval += self.helper.game_completed(board, k_in_row, self.player_number) * weight[k_in_row]
            board_eval -= self.helper.game_completed(board, k_in_row, opponent) * weight[k_in_row]


        return board_eval

    def get_alpha_beta_move(self, board):
        '''Finding the alpha_beta move
        Input = board
        output = best action to make'''

        def min_value(board, alpha, beta, depth, player):

            valid_moves = self.helper.valid_moves(board)
            opponent = self.helper.other_player()

            if depth == 4 or len(valid_moves) == 0:
                return (self.evaluation_function(board))

            for row, col in valid_moves:
                board[row][col] = opponent
                beta = min (beta, max_value(board, alpha, beta, depth+1, player))
                board[row][col] = 0
                if beta<= alpha:
                    return beta

            return beta

        def max_value(board, alpha, beta, depth, player):

            valid_moves = self.helper.valid_moves(board)

            if depth == 4 or len(valid_moves) == 0:
                return (self.evaluation_function(board))

            for row, col in valid_moves:
                board[row][col] = player
                alpha = max(alpha, min_value(board,alpha,beta,depth+1, player))
                board[row][col] = 0
                if alpha >= beta:
                    return alpha

            return alpha


        bestScore = -inf
        beta = inf
        bestMove = None
        depth = 0

        for row , col in self.helper.valid_moves(board):
            board[row][col] = self.player_number
            v = min_value(board, bestScore, beta, depth + 1, self.player_number)
            if v > bestScore:
                bestScore, bestMove = (v, col)
            board[row][col] = 0

        return bestMove



    def get_expectimax_move(self, board):

        def max_val(board, depth, player):

            valid_moves = self.helper.valid_moves(board)
            if depth == 0 or len(valid_moves) == 0:
                return self.evaluation_function(board)

            v = -inf
            for row, col in valid_moves:
                board[row][col] = player
                val = exp_val(board, depth - 1, player)
                v = max(v, val);
            return v

        def exp_val(board, depth, player):

            valid_moves = self.helper.valid_moves(board)
            opponent = self.helper.other_player()

            if depth == 0 or len(valid_moves)==0:
                return (self.evaluation_function(board))

            v = 0
            for row,col in valid_moves:
                board[row][col] = opponent
                val = max_val(board , depth-1, player)
                v += (1.0/len(valid_moves)) * val


            return v


        player = self.player_number

        depth = 10
        all_values = []
        alpha = -inf

        for row, col in self.helper.valid_moves(board):
            board[row][col] = player
            alpha = max(alpha, exp_val(board, depth - 1 , player))
            all_values.append((alpha, col))
            board[row][col] = 0

        maxval = (max(all_values,key=itemgetter(1))[0])
        for v in all_values:
            if maxval in v:
                col_index = v[1]

        return col_index


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
            if 0 in board[:,col]:
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
