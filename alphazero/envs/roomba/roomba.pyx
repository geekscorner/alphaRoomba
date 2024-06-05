# cython: language_level=3
# cython: auto_pickle=True
# cython: profile=True
from typing import List, Tuple, Any

from alphazero.Game import GameState
from alphazero.envs.roomba.RoombaLogic import Board

import numpy as np

NUM_PLAYERS = 2
MAX_TURNS = 50
MULTI_PLANE_OBSERVATION = True
NUM_CHANNELS = 2 if MULTI_PLANE_OBSERVATION else 1

class Game(GameState):
    def __init__(self):
        super().__init__(self._get_board())
        self._moves = 0
        self._invalid_move = -1

    @staticmethod
    def _get_board():
        return Board()

    def set_board(self, newBoardPieces, newPlayer):
        self._board.pieces = newBoardPieces
        self._player = newPlayer

    def __hash__(self) -> int:
        return hash(self._board.pieces.tobytes() + bytes([self.turns]) + bytes([self._player]))

    def __eq__(self, other: 'Game') -> bool:
        return self._board.pieces == other._board.pieces and self._player == other._player and self.turns == other.turns and self.turns == other.turns

    def clone(self) -> 'Game':
        game = Game()
        game._board.pieces = np.copy(np.asarray(self._board.pieces))
        game._player = self._player
        game._turns = self.turns
        game._moves = self._moves
        game.last_action = self.last_action
        return game

    @staticmethod
    def max_turns() -> int:
        return MAX_TURNS

    @staticmethod
    def has_draw() -> bool:
        return True

    @staticmethod
    def num_players() -> int:
        return NUM_PLAYERS

    @staticmethod
    def action_size() -> int:
        return 128

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        #TODO: What is the correct size?
        return NUM_CHANNELS, 4, 4

    def valid_moves(self):
        return np.asarray(self._board.get_valid_moves((1, -1)[self._player]))

    def play_action(self, action: int) -> None:
        super().play_action(action)
        self._board.makeMove(action, (1, -1)[self._player])
        self._moves += 1
        if self._moves == 2:
            self._moves = 0
            self._update_turn()

    def win_state(self) -> np.ndarray:
        result = [False] * 3
        game_over, player = self._board.get_win_state()

        if game_over:
            index = -1
            self._moves = 0
            if player == 1:
                index = 0
            elif player == -1:
                index = 1
            result[index] = True

        if self.turns == MAX_TURNS:
            result[2] = True
            
        return np.array(result, dtype=np.uint8)

    def observation(self):
        if MULTI_PLANE_OBSERVATION:
            pieces = np.asarray(self._board.pieces[0:, :])
            player1 = np.where((pieces > 0), pieces, 0)
            player2 = np.where((pieces < 0), abs(pieces), 0)
            return np.array([player1, player2], dtype=np.float32)
            #TODO: Is this enough dimensions?
        else:
            return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    def symmetries(self, pi: np.ndarray, winstate) -> List[Tuple[Any, int]]:
        assert (len(pi) == 128)

        pi_moves = np.reshape(pi[:64], (8, 8))
        pi_rotations = np.reshape(pi[64:128], (8, 8))  
        result = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(np.asarray(self._board.pieces[0:, :]), i) 
                new_pi_moves = np.rot90(pi_moves, i)
                new_pi_rotations = np.rot90(pi_rotations, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi_moves = np.fliplr(new_pi_moves)
                    new_pi_rotations = np.fliplr(new_pi_rotations)

                gs = self.clone()
                gs._board.pieces = new_b
                
                result.append((gs, np.concatenate((new_pi_moves.ravel(), new_pi_rotations.ravel())), winstate))

        return result

def display(board, action=None):
    #TODO: output a better state representation
    print(" -----------------------")
    
    