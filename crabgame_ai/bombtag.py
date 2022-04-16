"""A class that defines the 2-player BombTag game.
NOTE: Assume the map is fixed right now so we don't have to
handle the image embedding of the map.
"""
from enum import IntEnum
from typing import Sequence

import gym
import numpy as np
import matplotlib as plt

# example map
game_map = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                     [0, 0, 0, 0, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 2, 1],
                     [0, 0, 0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0]])
# seeker, hider
start_locs = [(7, 7, 3), (0, 0, 1)]


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


class CellType(IntEnum):
    """A class that defines the possible cell types in the game."""
    GROUND = 0
    PLATFORM = 1
    STAIRCASE = 2
    SEEKER = 3
    HIDER = 4


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
        self.init_loc = init_location  # save initial location for reset

    def reset_player(self):
        """
        resets player's position
        gives bomb back to original seeker??
        """
        self.x = self.init_loc[0]
        self.y = self.init_loc[1]
        self.direction = self.init_loc[2]

    def turn_left(self):
        self.direction = (self.direction - 1) % 4

    def turn_right(self):
        self.direction = (self.direction + 1) % 4


class BombTagObservation:
    """A class that defines the observation of the game.
    TODO: Implement the observation class. Necessary attributes might include
    things discussed:
    - The current position of the seeker and hider
    - The remaining explode time of the bomb
    """

    def __init__(self, cur_env: np.ndarray, time: int):
        self.remaining_time = time
        self.seeker_pos = np.where(cur_env == BombTagPlayerTypes.SEEKER)
        self.hider_pos = np.where(cur_env == BombTagPlayerTypes.HIDER)


def forward_cell(player: Player):
    x = player.x
    y = player.y
    match player.direction:
        case 0:
            y -= 1
        case 1:
            x += 1
        case 2:
            y += 1
        case 3:
            x -= 1
    return x, y


class BombTagEnv(gym.Env):
    """A class that defines the 2-player BombTag game that follows
    the OpenAI Gym interface."""

    def __init__(self, env_map: np.ndarray, init_locations: list, explode_time: int = 100):
        self.init_explode_time = explode_time  # save for reset
        self.explode_time = explode_time
        self.init_env = env_map.copy()  # basic environment map w/o player locations
        self.cur_env = env_map.copy()  # game map w/ player locations
        self.map_height, self.map_width = env_map.shape
        # self.players = []
        # for i in range(2):
        #     player = Player(BombTagPlayerTypes(i), i, init_locations[i])
        #     self.players.append(player)

        # player 1: id=1, type=Seeker
        self.player1 = Player(BombTagPlayerTypes.SEEKER, 0, init_locations[0])
        # add player 1's initial location to the current game map
        self.cur_env[init_locations[0][0], init_locations[0][1]] = CellType.SEEKER

        # player 2: id=1, type=Hider
        self.player2 = Player(BombTagPlayerTypes.HIDER, 1, init_locations[1])
        # add player 2's initial location to the current game map
        self.cur_env[init_locations[1][0], init_locations[1][1]] = CellType.HIDER

        # create list of players
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
        # [actp1, actp2]
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
        rewards = np.zeros(len(self.players))
        done = self.explode_time <= 0

        if done:
            for p in self.players:
                if not p.has_bomb:
                    # award player for not exploding
                    rewards[p.player_id] = 1
                elif p.has_bomb:
                    # penalize player for exploding
                    rewards[p.player_id] = -1
            return BombTagObservation(self.cur_env, self.explode_time), rewards, done

        # action index in array corresponds to each player id
        for i in range(len(actions)):
            player = self.players[i]
            x = player.x
            y = player.y
            # coordinates of forward cell
            f_x, f_y = forward_cell(player)

            match actions[i]:
                case BombTagActions.MOVE_FORWARD:
                    # cant if facing wall
                    if 0 <= f_x < self.map_width and 0 <= f_y < self.map_height:
                        # other than if facing stairs, must be same tile type
                        if self.init_env[x, y] == self.cur_env[f_x, f_y] or self.cur_env[f_x, f_y] == CellType.STAIRCASE:
                            player.x = f_x
                            player.y = f_y
                            self.cur_env[x, y] = self.init_env[x, y]
                            self.cur_env[f_x, f_y] = CellType(player.player_type + 3)
                        # player dies if fall off the platform, game end?
                        if self.init_env[x, y] == CellType.PLATFORM and self.init_env[f_x, f_y] == CellType.GROUND:
                            rewards[player.player_id] = -1
                            done = True
                            return BombTagObservation(self.cur_env, self.explode_time), rewards, done
                case BombTagActions.TURN_LEFT:
                    player.turn_left()
                case BombTagActions.TURN_RIGHT:
                    player.turn_right()
                case BombTagActions.STAY_STILL:
                    # do nothing
                    continue
                case BombTagActions.JUMP:
                    # if facing opposite level, move forward
                    if 0 <= f_x < self.map_width and 0 <= f_y < self.map_height:
                        if (self.init_env[x, y] == CellType.PLATFORM and self.init_env[f_x, f_y] == CellType.GROUND) \
                                or (self.init_env[x, y] == CellType.GROUND and self.init_env[f_x, f_y] == CellType.PLATFORM):
                            player.x = f_x
                            player.y = f_y
                            self.cur_env[x, y] = self.init_env[x, y]
                            self.cur_env[f_x, f_y] = CellType(player.player_type + 3)
                case BombTagActions.HAND_BOMB:
                    if player.has_bomb and \
                            (0 <= f_x < self.map_width and 0 <= f_y < self.map_height) \
                            and self.cur_env[f_x, f_y] == CellType.HIDER:
                        player.has_bomb = False
                        player.player_type = BombTagPlayerTypes.HIDER
                        # set receiving player has_bomb = True, assume other player is hider, obtains bomb
                        self.players[1 - player.player_id].has_bomb = True
                        self.players[1 - player.player_id].player_type = BombTagPlayerTypes.SEEKER
                case _:
                    print('unknown action')

        # decrease bomb time
        self.explode_time -= 1

        return BombTagObservation(self.cur_env, self.explode_time), rewards, done

    def render(self, mode='human'):
        """Render the environment. (use matplotlib.pyplot or cv2)
        TODO: implement this method.
        """
        N = 8
        fig, ax = plt.pyplot.subplots(1, 1, tight_layout=True)
        # make color map
        my_cmap = plt.colors.ListedColormap(['w', 'r', 'g', 'b', 'y'])
        # set the 'bad' values (nan) to be white and transparent
        my_cmap.set_bad(color='w', alpha=0)
        # draw the grid
        for x in range(N + 1):
            ax.axhline(x, lw=2, color='k', zorder=5)
            ax.axvline(x, lw=2, color='k', zorder=5)
        # draw the boxes
        ax.imshow(self.cur_env, interpolation='none', cmap=my_cmap, extent=[0, N, 0, N], zorder=0)
        # turn off the axis labels
        ax.axis('off')
