import numpy, pyximport

pyximport.install(setup_args={'include_dirs': numpy.get_include()})

from alphazero.Arena import Arena
from alphazero.GenericPlayers import *
from alphazero.NNetWrapper import NNetWrapper as NNet

# from alphazero.envs.miniBoop.miniBoop import Game as Game
# from alphazero.envs.miniBoop.train import args

from alphazero.envs.roomba.roomba import Game
from alphazero.envs.roomba.roomba import display as displayGame

from alphazero.envs.roomba.train import args

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

"""
def calculateSingleMove():
    pieces = np.array([[ 8, 8, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0],
                       [ 0, 0, 0, 0, 0, 0]])

    grey = 0
    orange = 1

    g = Game()
    g.set_board(pieces, grey)

    displayGame(g)
    args.numMCTSSims = 6400
    nn1 = NNet(Game, args)
    nn1.load_checkpoint('', 'AlphaBoop-109.pkl')

    alphaBoop = MCTSPlayer(nn1, args=args, print_policy=True)
    print("Action: " + str(alphaBoop.play(g)))
"""

if __name__ == '__main__':
    # args.numMCTSSims = 1600
    # args.arena_batch_size = 64
    # calculateSingleMove()

    # # nnet players
    nn1 = NNet(Game, args)
    nn1.load_checkpoint('', 'D-iteration-0077.pkl')

    nn2 = NNet(Game, args)
    nn2.load_checkpoint('', 'D-iteration-0077.pkl')

    alphaRoomba = MCTSPlayer(nn1, args=args)  # , print_policy=True)
    alphaRoomba2 = MCTSPlayer(nn2, args=args)  # , print_policy=True)

    #player1 = nn1.process
    player1 = RawMCTSPlayer(Game, args)
    # # policy, value = nn1.predict(np.array([player1, player2, colour, turn], dtype=np.float32))
    # # print(policy)
    #
    #player2 = MCTSPlayer(nn1, args=args)  # , print_policy=True)
    player2 = RawMCTSPlayer(Game, args)
    stupidRoomba = RandomPlayer()
    stupidRoomba2 = RandomPlayer()
    #
    human1 = HumanMiniBoopPlayer()
    human2 = HumanMiniBoopPlayer()
    #
    # # Human goes first
    # players = [alphaBoop, human]
    #
    # # Bot goes first
    #players = [human1, human2]
    players = [alphaRoomba, alphaRoomba2 ]
    #players = [stupidRoomba, stupidRoomba2]
    #
    arena = Arena(players, Game, use_batched_mcts=False, args=args, display=print)
    wins, draws, winrates = arena.play_games(100, verbose=True)
    
    for i in range(len(wins)):
        print(f'player{i+1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    print('draws: ', draws)
    
    #
    # nn1 = NNet(Game, args)
    # nn1.load_checkpoint('', 'AlphaBoop-109.pkl')
    #
    # nn2 = NNet(Game, args)
    # nn2.load_checkpoint('', 'AlphaBoop-109.pkl')
    #
    # alphaBoop = MCTSPlayer(nn1, args=args)  # , print_policy=True)
    # alphaBoop2 = MCTSPlayer(nn2, args=args)  # , print_policy=True)
    #
    # human = HumanMiniBoopPlayer()
    #
    # # # Human goes first
    # players = [alphaBoop2, human]
    #
    # # Bot goes first
    # # players = [human, alphaBoop2]
    #
    # arena = Arena(players, Game, use_batched_mcts=False, args=args, display=displayGame)
    # wins, draws, winrates = arena.play_games(1, verbose=True)
    # for i in range(len(wins)):
    #     print(f'player{i + 1}:\n\twins: {wins[i]}\n\twin rate: {winrates[i]}')
    # print('draws: ', draws)
