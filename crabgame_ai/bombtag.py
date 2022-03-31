"""A class that defines the 2-player BombTag game.
NOTE: Assume the map is fixed right now so we don't have to
handle the image embedding of the map.
"""
from enum import IntEnum
from typing import Sequence

import gym


class BombTagActions(IntEnum):
    """A class that defines the available actions that can be taken in the game."""
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STAY_STILL = 3
    JUMP = 4
    HAND_BOMB = 5  # only has affect if you is carrying a bomb and the cell that you're facing
    # has the opponent.


class BombTagPlayerTypes(IntEnum):
    """ A class that defines the available player types in the game."""
    SEEKER = 0
    HIDER = 1


class Direction(IntEnum):
    """A class that defines the available directions in the game."""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Player:
    """A class that defines a player in the game.
    TODO: Implement the player class. Necessary attributes might
    include my current position, the player speed ect...

    NOTE: Position is  (x, y, direction) where direction is one of the Direction IntEnum

    """

    def __init__(self, player_type: BombTagPlayerTypes, player_id: int, init_location: tuple):
        self.player_type = player_type
        self.player_id = player_id
        if player_type == BombTagPlayerTypes.SEEKER:
            self.has_bomb = True
        else:
            self.has_bomb = False
        self.x = init_location[0]
        self.y = init_location[1]
        self.direction = init_location[2]
        self.init_loc = init_location # save initial location for reset

    def reset_player(self):
        """
        resets player's position
        gives bomb back to original seeker??
        """
        self.x = self.init_loc[0]
        self.y = self.init_loc[1]
        self.direction = self.init_loc[2]

    def move_forward(self):
        match self.direction:
            case 0:
                if self.y > 0:
                    self.y -= 1
            case 1:
                if self.x < width: # how are we defining map width?
                    self.x += 1
            case 2:
                if self.y < height:
                    self.y += 1
            case 3:
                if self.x > 0:
                    self.x -= 1

    def turn_left(self):
        self.direction = (self.direction - 1) % 4

    def turn_right(self):
        self.direction = (self.direction + 1) % 4 # int or direction enum? does it matter?

    def hand_bomb(self):
        if self.has_bomb: # also check if cell facing contains opponent
            self.has_bomb = 0
            return True # let step know bomb successfully handed over
        else:
            return False



class BombTagObservation:
    """A class that defines the observation of the game.
    TODO: Implement the observation class. Necessary attributes might include
    things discussed:
    - The current position of the seeker and hider
    - The remaining explode time of the bomb
    """
    pass


class BombTagEnv(gym.Env):
    """A class that defines the 2-player BombTag game that follows
    the OpenAI Gym interface."""

    def __init__(self, init_locations: list, explode_time: int = 100):
        self.init_explode_time = explode_time
        self.explode_time = explode_time
        # self.players = []
        # for i in range(2):
        #     player = Player(BombTagPlayerTypes(i), i, init_locations[i])
        #     self.players.append(player)
        self.player1 = Player(BombTagPlayerTypes.SEEKER, 0, init_locations[0])
        self.player2 = Player(BombTagPlayerTypes.HIDER, 1, init_locations[1])
        self.players = [self.player1, self.player2]

    def reset(self):
        """Reset the environment to its initial state.
        This involves resetting the location of the players and the
        explode_time of the bomb.
        TODO: Implement this method.
        """
        self.explode_time = self.init_explode_time
        for p in self.players:
            p.reset_player()
        # self.player1.reset_player()
        # self.player2.reset_player()

    def step(self, actions: Sequence[BombTagActions]):
        """Run one timestep of the environment's dynamics.
        Args:
            actions: a list of action taken, each per player.
        Returns:
            observation: agent's observation of the current environment.
            rewards: a list of reward returned after previous action, each per player.
            done: whether the episode has ended, in which case further step() calls will return undefined results.
            info: contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).

        TODO: implement this method.
        """
        for action in actions:
            match action:
                case BombTagActions.MOVE_FORWARD:
                    # calling step() per player, or all players execute same action?
                case BombTagActions.TURN_LEFT:
                case BombTagActions.TURN_RIGHT:
                case BombTagActions.STAY_STILL:
                case BombTagActions.JUMP:
                case BombTagActions.HAND_BOMB:
                case _:
                    print('unknown action')


    def render(self, mode='human'):
        """Render the environment. (use matplotlib.pyplot or cv2)
        TODO: implement this method.
        """
        pass
