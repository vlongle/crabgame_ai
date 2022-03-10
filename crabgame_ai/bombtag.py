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

    def __init__(self, player_type: BombTagPlayerTypes, player_id: int):
        self.player_type = player_type
        self.player_id = player_id
        if player_type == BombTagPlayerTypes.HIDER:
            self.position = (0, 0, Direction.EAST)
            self.has_bomb = False
        else:
            self.position = (7, 7, Direction.NORTH)     # gym.width, gym.height for initial x, y? -EL
            self.has_bomb = True

    def reset_player(self):
        """
        resets player's position
        gives bomb back to original seeker
        """
        if self.player_type == BombTagPlayerTypes.HIDER:
            self.position = (0, 0, Direction.EAST)
            self.has_bomb = False
        else:
            self.position = (7, 7, Direction.NORTH)
            self.has_bomb = True

        # will hider/seeker roles change during the game depending on who has the bomb? -EL



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

    def __init__(self, explode_time: int = 100):
        self.init_explode_time = explode_time
        self.explode_time = explode_time
        self.seeker = Player(BombTagPlayerTypes.SEEKER, 0)
        self.hider = Player(BombTagPlayerTypes.HIDER, 1)

    def reset(self):
        """Reset the environment to its initial state.
        This involves resetting the location of the players and the
        explode_time of the bomb.
        TODO: Implement this method.
        """
        self.explode_time = self.init_explode_time
        self.seeker.reset_player()
        self.hider.reset_player()

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
        pass

    def render(self, mode='human'):
        """Render the environment. (use matplotlib.pyplot or cv2)
        TODO: implement this method.
        """
        pass
