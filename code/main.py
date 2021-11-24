import logging

import coloredlogs

from Coach import Coach
from TSP.TSPGame import Game
from TSP.NeuralNet import NeuralNet as nn
from utils import *
import numpy as np

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'n': 10,
    'numIters': 100,
    'numEps': 30,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'load_coordinates': False,
    'checkpoint': 'TSP/temp',
    'load_model': False,
    'load_folder_file': ('TSP/models','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

def main():
    log.info('Loading %s...', Game.__name__)
    n = args.n
    if args.load_coordinates:
        coordinates = np.load('TSP/output/coordinates.npy')
    else:
        coordinates = np.array([list((np.random.random_sample(2)-0.5)) for j in range(n)])
        np.save('TSP/output/coordinates.npy', coordinates)
    g = Game(n, coordinates)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
