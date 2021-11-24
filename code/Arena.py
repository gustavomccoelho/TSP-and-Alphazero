import logging
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, iteration, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.iteration = iteration

    def playGame(self, randomPlayer, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer, it) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            if randomPlayer:
                if curPlayer == 1:
                    action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))
                if curPlayer == -1:
                    valid_moves = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)
                    action = np.random.choice([i for i, move in enumerate(valid_moves) if move == 1])
            else:
                action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1, it)))
            self.display(board)

        return curPlayer * self.game.getGameEnded(board, curPlayer, it)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(randomPlayer=False, verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(randomPlayer=False, verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        self.game.board.plot_board(self.game.board.pieces, self.iteration, gameResult, random = False)
        np.save('TSP/output/board_Random_False_' + str(self.iteration), np.array(self.game.board.pieces))
        plt.close()

        start = time.time()
        gameResult = self.playGame(randomPlayer=True, verbose=verbose)
        end = time.time()
        if gameResult == -1:
            oneWon += 1
        elif gameResult == 1:
            twoWon += 1
        else:
            draws += 1

        interval = end-start
        self.game.board.plot_board(self.game.board.pieces, self.iteration, gameResult, random = True)
        np.save('TSP/output/board_Random_True_' + str(self.iteration), np.array(self.game.board.pieces))

        #cost = round(-self.game.board.get_score(self.game.board.pieces, 1), 2)

        #try:
        #    with open('TSP/output/cost_list.txt', 'r') as f:
        #        cost_list = f.read().splitlines()
        #        cost_list.append(cost)
        #except:
        #    cost_list = [cost]

        #with open('TSP/output/cost_list.txt', 'w') as f:
        #    for item in cost_list:
        #        f.write("%s\n" % item)

        #cost_list = [float(i) for i in cost_list]
        #plt.plot(cost_list)
        #plt.savefig('TSP/output/costs.jpg')

        return oneWon, twoWon, draws
