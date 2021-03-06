import logging
import coloredlogs
import numpy as np

from Coach import Coach
from TSPGame import Game
from NeuralNet import NeuralNet as nn
from utils import *

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'n': 15,
    'numIters': 10,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'load_coordinates': False,
    'checkpoint': '../data/temp',
    'load_model': False,
    'load_folder_file': ('../data/models','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

def main():
    log.info('Loading %s...', Game.__name__)
    n = args.n
    if args.load_coordinates:
        coordinates = np.load('../data/output/coordinates.npy')
    else:
        coordinates = np.array([list((np.random.random_sample(2)-0.5)) for j in range(n)])
        np.save('../data/output/coordinates.npy', coordinates)
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
