"""A class that defines the 2-player BombTag game.
NOTE: Assume the map is fixed right now so we don't have to
handle the image embedding of the map.
"""
import gym
from enum import Enum
from typing import Sequence


class BombTagActions(Enum):
    """A class that defines the available actions that can be taken in the game."""
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
    JUMP = 4
    HAND_BOMB = 5


class BombTagPlayerTypes(Enum):
    """ A class that defines the available player types in the game."""
    SEEKER = 0
    HIDER = 1


class Player:
    """A class that defines a player in the game.
    TODO: Implement the player class. Necessary attributes might
    include my current position ect...
    """

    def __init__(self, player_type: BombTagPlayerTypes, player_id: int):
        self.player_type = player_type
        self.player_id = player_id


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
        self.explode_time = explode_time

    def reset(self):
        """Reset the environment to its initial state.
        This involves resetting the location of the players and the
        explode_time of the bomb.
        TODO: Implement this method.
        """
        pass

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
