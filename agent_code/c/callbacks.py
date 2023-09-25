import os
import pickle
import random
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from .model import DQN
from random import shuffle

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
COIN_COUNT = 9

from collections import deque

def find_closest_target(free_space, current_position, targets, logger=None):
    if len(targets) == 0:
        return None, None
        # Returns None to indicate that there are no targets to reach

    #  Initializing Data Structures
    frontier = deque([current_position])
    parent_dict = {current_position: current_position}
    dist_so_far = {current_position: 0}
    best = current_position
    best_dist = np.sum(np.abs(np.subtract(targets, current_position)), axis=1).min()
    best_index = np.argmin(np.sum(np.abs(np.subtract(targets, current_position)), axis=1))

    while frontier:
        current = frontier.popleft()
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        current_index = np.argmin(np.sum(np.abs(np.subtract(targets, current)), axis=1))

        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
            best_index = current_index
        if d == 0:
            best = current
            best_index = current_index
            break

        x, y = current
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        neighbors = [(xx, yy) for (xx, yy) in neighbors if free_space[xx, yy]]

        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    if logger:
        logger.debug(f'Suitable target found at {best} (Index: {best_index})')
    current = best

    while True:
        if parent_dict[current] == current_position:
            return current, best_index
        current = parent_dict[current]




def setup(self):
    """Called once before a set of games to initialize data structures etc.

        The 'self' object passed to this method will be the same in all other
        callback methods. You can assign new properties (like bomb_history below)
        here or later on and they will be persistent even across multiple games.
        You can also use the self.logger object at any time to write to the log
        file for debugging (see https://docs.python.org/3.7/library/logging.html).
        """
    # np.random.seed()
    # self.me = 2
    self.epsilon_start = 0.9  # Initial value of epsilon
    self.epsilon_min = 0.02  # Minimum value of epsilon
    self.epsilon_decay = 1 / 1000000  # Decay rate of epsilon
    self.epsilon = self.epsilon_start  # Initialize epsilon to initial value
    # EPS_START = 1
    # EPS_END = 0.02
    # EPS_DECAY = 1000000
    self.round = 0
    self.current_round = 0



    if self.train and not os.path.isfile("my-saved-model.pt"):
        print("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.q_network = weights / weights.sum()

    elif self.train and os.path.isfile("my-saved-model.pt"):
        print("Building on existing model.")

    else:
        print("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_network = pickle.load(file)




def act(self, game_state: dict) -> str:
    """
        Called each game step to determine the agent's next action.

        You can find out about the state of the game environment via game_state,
        which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
        what it contains.
        """
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]

    _, _, bombs_left, (x, y) = game_state['self']
    # If agent has been in the same location three times recently, it's a loop
    if self.coordinate_history.count((x, y)) > 2:
        self.ignore_others_timer = 5
    else:
        self.ignore_others_timer -= 1

    self.coordinate_history.append((x, y))

    self.round = self.round + 1
    print('round:',self.round)

    if self.train:
        print("######################################")
        print("self.epsilon:", self.epsilon)

        sample = random.random()

        if sample > self.epsilon:
            with torch.no_grad():
                print("######################################")
                print("Action no random.")
                print("######################################")
                features = state_to_features(game_state)
                q_values = self.q_network(torch.tensor(features, dtype=torch.float32))
                ac = ACTIONS[torch.argmax(q_values)]
                if ac == 'BOMB': self.bomb_history.append((x, y))
                return ac
        else:
            print("######################################")
            print("Choosing action purely at random.")
            print("######################################")
            acts = get_valid_action(game_state)
            count_simple = 0
            count_other = 0
            # if 'BOMB' in acts:
            #     acts.remove('BOMB')
            # if 'WAIT' in acts:
            #     acts.remove('WAIT')
            # for a in acts:
            #     if a == 'RIGHT' or a == 'LEFT' or a == 'UP' or a == 'DOWN':
            #         count_simple = count_simple + 1
            #     else:
            #         count_other = count_other + 1
            # count_other_p = 1 / (14 * count_simple + count_other)
            # count_simple_p = 14 * count_other_p
            # p = [count_simple_p] * count_simple
            # p.extend([count_other_p] * count_other)
            # print("sum:",p)
            ac = np.random.choice(acts)
            # ac = np.random.choice(acts, p=p)
            # ac = acts[torch.tensor([[np.random.choice([i for i in range(0, len(acts))])]],
            #                        dtype=torch.long)]
            if ac == 'BOMB': self.bomb_history.append((x, y))
            return ac
    else:
        features = state_to_features(game_state)
        q_values = self.q_network(torch.tensor(features, dtype=torch.float32))
        ac = ACTIONS[torch.argmax(q_values)]
        if ac == 'BOMB': self.bomb_history.append((x, y))
        return ACTIONS[torch.argmax(q_values)]

        print("######################################")
        print("Action no random.")
        print("######################################")
        features = state_to_features(game_state)
        q_values = self.q_network(torch.tensor(features, dtype=torch.float32))
        ac = ACTIONS[torch.argmax(q_values)]
        if ac == 'BOMB': self.bomb_history.append((x, y))
        return ac


def check_bomb_crates_amount(matrix, x, y):
    count = 0
    rows, cols = len(matrix), len(matrix[0])

    for i in range(max(0, x - 3), min(rows, x + 4)):
        for j in range(max(0, y - 3), min(cols, y + 4)):
            if matrix[i][j] == 1:
                count += 1

    return count


def decimal_to_fixed_width_binary(decimal_number, fixed_widthß):
    # Use the bin() function to convert a decimal number to a binary string and remove the prefix "0b".
    binary_string = bin(decimal_number)[2:]

    # If the binary string is less than a fixed number of bits long, add leading zeros
    if len(binary_string) < fixed_width:
        binary_string = "0" * (fixed_width - len(binary_string)) + binary_string

    # Create an array of integers containing 0 and 1
    binary_array = [int(bit) for bit in binary_string]

    return binary_array


def shortest_distance(board, start, end):
    # Define four directions of movement. Right, left, down, up.
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    direction_names = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0]]
    rows, cols = len(board), len(board[0])

    # Create a queue for BFS
    # Each queue element includes the current position and path
    queue = deque([(start, [])])

    # Create a collection for logging visited locations
    visited = set()
    visited.add(start)

    while queue:
        current, path = queue.popleft()
        if current == end:
            first_step = path[0] if path else []
            return len(path), first_step

        for direction, direction_name in zip(directions, direction_names):
            dx, dy = direction
            new_x, new_y = current[0] + dx, current[1] + dy
            new_position = (new_x, new_y)

            if 0 <= new_x < rows and 0 <= new_y < cols and board[new_x][new_y] == 0 and new_position not in visited:
                new_path = path + [direction_name]
                queue.append((new_position, new_path))
                visited.add(new_position)

    return 99999, []  # If the shortest path cannot be found, returns -1 and the empty path


def check_safe(d, bomb_map, game_state, others, bomb_xys, arena):
    if ((arena[d] == 0) and
            (game_state['explosion_map'][d] < 1) and
            (bomb_map[d] > 0) and
            (not d in others) and
            (not d in bomb_xys)):
        return True
    return False


def stuck_in_dead_ends(x, y, dead_ends):
    # Add proposal to drop a bomb if at a dead end
    if (x, y) in dead_ends:
        return 1
    return 0

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

    """
        Feature selection: situational awareness (what's front and back), nearest enemy position, 
        nearest gold position, which way to move to get to the enemy, which way to move to get to 
        the gold, whether there is a way to get to the enemy position, whether there is a way to 
        get to the gold position.
        Maximum number of crates that will be destroyed by placing bombs here, whether it is in 
        the safe zone, whether the front and back are safe or not (the front and back need to be 
        equal to 1 to ensure that they are safe)
    """



def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends

    if game_state is None:
        return None
    features = []
    # Gather information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # Check which moves make sense at all
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)


    # order: top right, bottom left wait Positions 1 to 5
    # features[0:5]
    if (x, y - 1) in valid_tiles:
        valid_actions.append('UP')
        features.append(1)
    else:
        features.append(0)

    if (x + 1, y) in valid_tiles:
        valid_actions.append('RIGHT')
        features.append(1)
    else:
        features.append(0)

    if (x, y + 1) in valid_tiles:
        valid_actions.append('DOWN')
        features.append(1)
    else:
        features.append(0)

    if (x - 1, y) in valid_tiles:
        valid_actions.append('LEFT')
        features.append(1)
    else:
        features.append(0)

    if (x, y) in valid_tiles:
        valid_actions.append('WAIT')
        features.append(1)
    else:
        features.append(0)

    # Disallow the BOMB action if agent dropped a bomb in the same spot recently
    # Position 6, can you put a bomb in there?
    # features[5]
    # if (bomb_history != None):
    #     if (bombs_left > 0) and ((x, y) not in bomb_history):
    #         valid_actions.append('BOMB')
    #         features.append(1)
    #     else:
    #         features.append(0)
    # else:
    #     features.append(0)

    if bombs_left > 0:
        valid_actions.append('BOMB')
        features.append(1)
    else:
        features.append(0)


    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]

    # Position 7, whether or not you're in dead ends
    # features[6]
    features.append(stuck_in_dead_ends(x, y, where_is_dead_end(arena)))

    # Position 9, up and down, left and right boxes for boxes or not
    # features[8]
    if amount_of_crates(arena, x, y) > 0:
        features.append(1)
    else:
        features.append(0)

    free_space = arena == 0
    for o in others:
        free_space[o] = False

    others = [other for other in others if other not in bomb_xys]
    target_other, target_other_index = find_closest_target(free_space, (x, y), others)
    coins = [coin for coin in coins if coin not in bomb_xys]
    target_coin, target_coin_index = find_closest_target(free_space, (x, y), coins)
    crates = [crate for crate in crates if crate not in bomb_xys]
    target_crate, target_crate_index = find_closest_target(free_space, (x, y), crates)

    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)]
    directions_binary = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]]

    if target_other is not None:
        idx = directions.index(target_other)
        if idx != None:
            features.extend(directions_binary[idx])
        else:
            features.extend([1, 1, 1])
    else:
        features.extend([1, 1, 1])

    if target_coin is not None:
        idx = directions.index(target_coin)
        if idx != None:
            features.extend(directions_binary[idx])
        else:
            features.extend([1, 1, 1])
    else:
        features.extend([1, 1, 1])

    if target_crate is not None:
        idx = directions.index(target_crate)
        if idx != None:
            features.extend(directions_binary[idx])
        else:
            features.extend([1, 1, 1])
    else:
        features.extend([1, 1, 1])

    # Position 14, in the path of the bomb?
    # features[14]
    is_in_bomb_path = 0

    bomb_list = []
    for (xb, yb), t in bombs:
        if ((xb == x) and (0 < abs(yb - y) < 4)) or ((yb == y) and (0 < abs(xb - x) < 4)):
            is_in_bomb_path = 1
            bomb_list.append(((xb, yb), t))

    not_escape_direction = not_escape_bomb(bombs, x, y)


    if not_escape_direction is None:
        features.extend([-1, -1, -1])
    else:
        if not_escape_direction == 'UP':
            features.extend([0, 0, 1])
        if not_escape_direction == 'RIGHT':
            features.extend([0, 1, 0])
        if not_escape_direction == 'DOWN':
            features.extend([0, 1, 1])
        if not_escape_direction == 'LEFT':
            features.extend([1, 0, 0])
        if not_escape_direction == 'WAIT':
            features.extend([1, 0, 1])

    features.append(is_in_bomb_path)

    # Position 15. Currently on the bomb?
    # features[15]
    features.append(in_bomb_position(bombs, x, y))

    # print("Feautures:", features)
    stacked_channels = np.stack(features)
    # and return them as a vector
    return stacked_channels.reshape(-1)

# The return value of this function is a coordinate (x, y) that represents the next target position
# that the agent should move towards. This coordinate represents the position that the agent will move
# to next. The agent can take action to move himself from his current position to this coordinate, thus
# moving towards the nearest target or the nearest target position.

# If the new coordinates are equal to this
def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of the closest target that can be reached via free tiles.

        Performs a breadth-first search of the reachable free tiles until a target is encountered.
        If no target can be reached, the path that takes the agent closest to any target is chosen.

        Args:
            free_space: Boolean numpy array. True for free tiles and False for obstacles.
            start: the coordinate from which to begin the search.
            targets: list or array holding the coordinates of all target tiles.
            logger: optional logger object for debugging.
        Returns:
            coordinate of first step towards the closest target or towards tile closest to any target.
        """
    if len(targets) == 0:
        return None, None  # Returns None to indicate that there are no targets to reach

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
    best_index = np.argmin(np.sum(np.abs(np.subtract(targets, start)), axis=1))

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        current_index = np.argmin(np.sum(np.abs(np.subtract(targets, current)), axis=1))

        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
            best_index = current_index
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            best_index = current_index
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)

        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    # print(f'Suitable target found at {best} (Index: {best_index})')
    if logger:
        # Determine the first step towards the best found target tile
        logger.debug(f'Suitable target found at {best} (Index: {best_index})')
    current = best

    while True:
        if parent_dict[current] == start:
            return current, best_index
        current = parent_dict[current]

def amount_of_crates(arena, x, y):
    return [arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1)
def is_closest_opponent(others, x, y):
    # Add proposal to drop a bomb if touching an opponent
    if len(others) > 0:
        if (min(abs(xy[0] - x) + abs(xy[1] - y) for xy in others)) <= 1:
            return 1
    return 0

def in_bomb_position(bombs, x, y):
    for (xb, yb), t in bombs:
        if xb == x and yb == y:
            return 1
    return 0
def get_valid_action(state):
    results = ['WAIT']
    _, _, is_bombs, (x, y) = state['self']
    print('I am first in [', x, ', ', y, ']')
    field = state['field']
    if y + 1 <= len(field) - 1:
        if field[x][y + 1] == 0:
            results.append('DOWN')
    if x + 1 <= len(field) - 1:
        if field[x + 1][y] == 0:
            results.append('RIGHT')
    if y - 1 >= 0:
        if field[x][y - 1] == 0:
            results.append('UP')
    if x - 1 >= 0:
        if field[x - 1][y] == 0:
            results.append('LEFT')
    if is_bombs == True:
        results.append('BOMB')

    return results

def where_is_dead_end(arena):
    # Compile a list of 'targets' the agent should head towards
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[0] - 1)
    dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    return dead_ends


def escape_bomb(bomb_list, x_b, y_b):
    # Initialize the minimum countdown and the corresponding bomb coordinates
    min_time = float('inf')
    best_bomb = None
    distance = -1
    best_distance = -1
    for bomb_coord, time in bomb_list:
        x, y = bomb_coord
        if x == x_b:
            distance = abs(y - y_b)
        elif y == y_b:
            distance = abs(x - x_b)
        # Calculate horizontal and vertical distances

        # If the current bomb countdown is less than the minimum countdown, or equal to the minimum countdown
        # but at a shorter distance, update the minimum countdown and the optimal bomb coordinates
        if time < min_time or (time == min_time and distance < best_distance):
            min_time = time
            best_bomb = bomb_coord
            best_distance = distance
            return best_bomb

    return None


def not_escape_bomb(bomb_list, x_b, y_b):
    # Initialize the minimum countdown and the corresponding bomb coordinates
    min_time = float('inf')
    best_bomb = None
    distance = -1
    best_distance = -1
    for bomb_coord, time in bomb_list:
        x, y = bomb_coord
        if x == x_b:
            distance = abs(y - y_b)
        elif y == y_b:
            distance = abs(x - x_b)
        # Calculate horizontal and vertical distances

        # If the current bomb countdown is less than the minimum countdown, or equal to the minimum countdown
        # but at a shorter distance, update the minimum countdown and the optimal bomb coordinates
        if time < min_time or (time == min_time and distance < best_distance):
            min_time = time
            best_bomb = bomb_coord
            best_distance = distance

    if best_bomb is not None:
        if best_bomb[0] > x_b:
            return 'RIGHT'
        if best_bomb[0] < x_b:
            return 'LEFT'
        if best_bomb[1] > y_b:
            return 'DOWN'
        if best_bomb[1] < y_b:
            return 'UP'

    return None