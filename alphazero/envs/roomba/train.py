import pyximport; pyximport.install()

from torch import multiprocessing as mp

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.roomba.roomba import Game
from alphazero.GenericPlayers import RawMCTSPlayer
from alphazero.utils import dotdict

args = get_args(dotdict({
    'run_name': 'roomba_fpu_2moves',
    'workers': 7,
    'startIter': 1,
    'numIters': 300,
    'numWarmupIters': 1,
    'process_batch_size': 256,
    'train_batch_size': 256,
    'gamesPerIteration': 256 * 7,
    'symmetricSamples': True,
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    'numMCTSSims': 200,  # Reduced for lower complexity
    'numFastSims': 30,
    'probFastSim': 0.5,
    'compareWithBaseline': True,
    'arenaCompare': 16 * 7,
    'arena_batch_size': 16,
    'arenaTemp': 1,
    'arenaMCTS': True,
    'baselineCompareFreq': 10,
    'compareWithPast': False,
    'pastCompareFreq': 10,
    'cpuct': 2,  # Adjusted for balanced exploration
    'fpu_reduction': 0.2,
    'load_model': True,
    'root_policy_temp': 2,  # Encouraging exploration
    'root_noise_frac': 0.6,  # Encouraging exploration
    '_num_players': 2,
    'eloMCTS': 25,
    'eloGames': 10,
    'eloMatches': 10,
    'calculateElo': True
}),
    model_gating=False,
    max_gating_iters=None,
    max_moves=50,

    lr=0.01,
    num_channels=128,
    depth=8,
    value_head_channels=32,
    policy_head_channels=32,
    value_dense_layers=[512, 256],  # Try smaller sizes
    policy_dense_layers=[512, 256]  # Try smaller sizes
)
args.scheduler_args.milestones = [75, 150]

if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
