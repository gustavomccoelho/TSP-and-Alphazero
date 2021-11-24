import numpy as np
import matplotlib.pyplot as plt


class Board():

    def __init__(self, n, coordinates):
        
        self.n = n
        #self.coordinates = np.array([list((np.random.random_sample(2)-0.5)) for j in range(self.n)])
        self.coordinates = coordinates
        
        valid = False
        while valid == False:
            
            self.pieces = [[0 for j in range(n)] for i in range(n)]
        
            # Player 1
            edgesToVisit = [i for i in range(self.n)]        
            edges = [np.random.choice(edgesToVisit)]
            startEdge = edges[0]
            edgesToVisit.remove(startEdge)

            while len(edgesToVisit) > 0: 
                previousEdge = edges[-1]
                currentEdge = np.random.choice(edgesToVisit)
                edges.append(currentEdge)
                edgesToVisit.remove(currentEdge)
                self.pieces[previousEdge][currentEdge] = 1 

            self.pieces[currentEdge][startEdge] = 1

            # Player 2
            edgesToVisit = [i for i in range(self.n)]        
            edges = [np.random.choice(edgesToVisit)]
            startEdge = edges[0]
            edgesToVisit.remove(startEdge)

            while len(edgesToVisit) > 0: 
                previousEdge = edges[-1]
                currentEdge = np.random.choice(edgesToVisit)
                edges.append(currentEdge)
                edgesToVisit.remove(currentEdge)
                self.pieces[previousEdge][currentEdge] = -1 

            self.pieces[currentEdge][startEdge] = -1  
            
            # Validating
            valid = True
            for piece in self.pieces:
                if piece.count(1) != 1: valid = False
                if piece.count(-1) != 1: valid = False
                           
            if valid:
                currentEdge = 0
                nextEdge = self.pieces[currentEdge].index(1)
                for i in range(self.n):
                    if self.pieces[nextEdge][currentEdge] != 0: valid = False
                    currentEdge = nextEdge
                    nextEdge = self.pieces[currentEdge].index(1)
                    
            
        self.round = 0
        
    def plot_board(self, board, iteration, gameResult, random):
        board = [list(piece[:]) for piece in board]
        counter = 0
        i = 0
        x1 = [self.coordinates[board[0].index(1)][0]]
        y1 = [self.coordinates[board[0].index(1)][1]]
        
        while counter < self.n:
            x1.append(self.coordinates[board[i].index(1)][0])
            y1.append(self.coordinates[board[i].index(1)][1])
            i = board[i].index(1)
            counter += 1
            
        x1.append(self.coordinates[board[0].index(1)][0])
        y1.append(self.coordinates[board[0].index(1)][1])
        
        counter = 0
        i = 0
        x2 = [self.coordinates[board[0].index(-1)][0]]
        y2 = [self.coordinates[board[0].index(-1)][1]]
        
        while counter < self.n:
            x2.append(self.coordinates[board[i].index(-1)][0])
            y2.append(self.coordinates[board[i].index(-1)][1])
            i = board[i].index(-1)
            counter += 1
            
        x2.append(self.coordinates[board[0].index(-1)][0])
        y2.append(self.coordinates[board[0].index(-1)][1])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))         
        ax1.plot(x1, y1, color='blue', linestyle='dashed', linewidth = 3,
                 marker='o', markerfacecolor='blue', markersize=12)
        ax2.plot(x2, y2, color='red', linestyle='dashed', linewidth = 3,
                 marker='o', markerfacecolor='red', markersize=12)
        ax1.title.set_text('PLAYER 1')
        ax2.title.set_text('PLAYER 2')
        plt.savefig('TSP/output/game_' + str(iteration) + '_secounds_result_' + str(gameResult) + '_random_' + str(random) + '.jpg')
        
    def get_score(self, board, player):
        count = 0
        for x in range(self.n):
            for y in range(self.n):
                if board[x][y] == player:
                    count += np.sqrt((self.coordinates[x][0] - self.coordinates[y][0])**2+(self.coordinates[x][1] - self.coordinates[y][1])**2)
        return -count
        
    def get_valid_moves(self, board, player):
        validMoves = np.zeros([self.n, self.n, self.n, self.n])
        for i in range(self.n):
            for j in range(self.n):
                if board[i][j] == player:
                    for k in range(self.n):
                        for l in range(self.n):
                            if board[k][l] == player:
                                if k != i and l != j and j != k and l != i:                                    
                                    if board[i][k] == 0 and board[j][l] == 0:                                         
                                        if board[k][i] == 0 and board[l][j] == 0:                                           
                                            validMoves[i, j, k, l] = 1

            return validMoves.flatten()
            
    def play_move(self, board, player, move):
        zeros = np.zeros(self.n**4)
        zeros[move] = 1
        move = zeros[:]
        moveIndexes = np.where(np.array(move).reshape([self.n, self.n, self.n, self.n]) == 1)
        i = moveIndexes[0][0]
        j = moveIndexes[1][0]
        k = moveIndexes[2][0]
        l = moveIndexes[3][0]
        boardCopy = [piece.copy() for piece in board]
        boardCopy[i][j] = 0
        boardCopy[k][l] = 0
        boardCopy[j][l] = player
        boardCopy[i][k] = player
        currentEdge = j
        nextEdge = list(board[currentEdge]).index(player)

        while currentEdge != k:
            boardCopy[currentEdge][nextEdge] = 0
            boardCopy[nextEdge][currentEdge] = player
            currentEdge = nextEdge
            nextEdge = list(board[currentEdge]).index(player)
        self.round += 1
        return boardCopy
            
    def get_random_action(self, player):
        validMoves = self.get_valid_moves(player)
        randomIndex = np.random.choice([i for i, move in enumerate(validMoves) if move == 1])
        move = np.zeros(self.n**4)
        move[randomIndex] = 1
        return move
    
    def is_done(self, round):
        if round > self.n*5: return True
        return False
