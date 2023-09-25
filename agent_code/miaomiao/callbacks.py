import os
import pickle
import random
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math


from collections import deque
from .model import DQN
from random import shuffle

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
COIN_COUNT = 9


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    np.random.seed()

    self.epsilon_start = 0.95  # epsilon的初始值
    self.epsilon_min = 0.02  # epsilon的最小值
    self.epsilon_decay = 1 / 400000  # epsilon的衰减速率
    self.epsilon = self.epsilon_start  # 初始化epsilon为初始值
    self.round = 0
    self.current_round = 0

    if self.train and not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.q_network = weights / weights.sum()

    elif self.train and os.path.isfile("my-saved-model.pt"):
        self.logger.info("Continue training...")

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_network = pickle.load(file)




def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    print("epsilon:", self.epsilon)
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]

    _, _, bombs_left, (x, y) = game_state['self']

    self.coordinate_history.append((x, y))

    self.round = self.round + 1
    print('round:',self.round)

    if self.train:

        sample = random.random()

        if sample > self.epsilon:
            with torch.no_grad():
                print("****************Action no random.****************")
                self.logger.info("Action no random.")
                features = state_to_features(game_state, self.coordinate_history, self.bomb_history)
                q_values = self.q_network(torch.tensor(features, dtype=torch.float32))
                ac = ACTIONS[torch.argmax(q_values)]
                if ac == 'BOMB': self.bomb_history.append((x, y))

                return ac
        else:
            print("****************Choosing action purely at random.****************")
            self.logger.info("Choosing action purely at random.")
            # acts = get_valid_action(game_state)
            # ac = np.random.choice(acts)
            ac = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

            if ac == 'BOMB': self.bomb_history.append((x, y))
            return ac
    else:
        self.logger.info("Action no random.")
        features = state_to_features(game_state, self.coordinate_history, self.bomb_history)
        q_values = self.q_network(torch.tensor(features, dtype=torch.float32))
        ac = ACTIONS[torch.argmax(q_values)]
        if ac == 'BOMB': self.bomb_history.append((x, y))
        return ac


def state_to_features(game_state: dict, coordinate_history: deque, bomb_history: deque) -> np.array:
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

    if game_state is None:
        return None
    features = []

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

    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)

    # Valid actions in the set of left, right, up, down, wait
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

    # Able to put a bomb in it?
    if (bomb_history != None):
        if (bombs_left > 0) and ((x, y) not in bomb_history):
            valid_actions.append('BOMB')
            features.append(1)
        else:
            features.append(0)
    else:
        features.append(0)

    dead_ends = [(x, y) for x in range(1, arena.shape[0] - 1) for y in range(1, arena.shape[0] - 1) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
    crates = [(x, y) for x in range(1, arena.shape[0] - 1) for y in range(1, arena.shape[0] - 1) if (arena[x, y] == 1)]
    targets = coins + dead_ends + crates

    # stuck in dead ends?
    features.append(stuck_in_dead_ends(x, y, where_is_dead_end(arena)))

    # have an opponent with a distance of less than one?
    features.append(is_closest_opponent(others, x, y))
    if len(coins) <= 2:
        # if len(crates) + len(coins) == 0:
        targets.extend(others)

    # How many crates at the top, bottom, left and right?
    features.append(amount_of_crates(arena, x, y))

    targets = [targets[i] for i in range(len(targets)) if targets[i] not in bomb_xys]
    free_space = arena == 0
    for o in others:
        free_space[o] = False
    pos, index = look_for_targets(free_space, (x, y), targets)

    type_of_target = [1, 1, 1]


    if index!= None:
        if targets[index] in coins:
            type_of_target = [0, 0, 1]
        if targets[index] in crates:
            type_of_target = [0, 1, 0]
        if targets[index] in dead_ends:
            type_of_target = [0, 1, 1]
        if targets[index] in others:
            type_of_target = [1, 0, 0]

    # The type of Target
    features.extend(type_of_target)

    # Location of the target
    directions = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)]
    directions_binary = [[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]]
    if pos != None:
        idx = directions.index(pos)
        if idx != None:
            features.extend(directions_binary[idx])
        else:
            features.extend([1, 1, 1])
    else:
        features.extend([1, 1, 1])

    # Location of the bomb to be escaped
    bomb_list = []
    for (xb, yb), t in bombs:
        if ((xb == x) and (0 < abs(yb - y) < 4)) or ((yb == y) and 0 < (abs(xb - x) < 4)):
            bomb_list.append(((xb, yb), t))
    best_bomb = escape_bomb(bomb_list, x, y)
    if best_bomb is not None:
        features.extend([best_bomb[0],best_bomb[1]])
    else:
        features.extend([-1, -1])

    # In the path of the bomb?
    is_in_bomb_path = 0
    for (xb, yb), t in bombs:
        if ((xb == x) and (0 < abs(yb - y) < 4)) or ((yb == y) and (0 < abs(xb - x) < 4)):
            is_in_bomb_path = 1
    features.append(is_in_bomb_path)

    # Currently on the bomb?
    features.append(in_bomb_position(bombs, x, y))

    # Determine if a loop has been entered
    if coordinate_history.count((x, y)) > 2:
        features.append(1)
    else:
        features.append(0)

    stacked_channels = np.stack(features)
    return stacked_channels.reshape(-1)


# from rule_based_agent
def look_for_targets(free_space, start, targets, logger=None):
    if len(targets) == 0:
        return None, None  # 返回 None 表示没有目标可以到达

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
    best_index = np.argmin(np.sum(np.abs(np.subtract(targets, start)), axis=1))

    while len(frontier) > 0:
        current = frontier.pop(0)
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
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)

        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1

    # print(f'Suitable target found at {best} (Index: {best_index})')
    if logger:
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

def where_is_dead_end(arena):
    # Compile a list of 'targets' the agent should head towards
    return [(x, y) for x in range(1, arena.shape[0] - 1) for y in range(1, arena.shape[0] - 1) if (arena[x, y] == 0)
                 and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]


def escape_bomb(bomb_list, x_b, y_b):
    # Initialise the minimum countdown and the corresponding bomb coordinates
    min_time = float('inf')
    best_bomb = None
    distance = -1
    best_distance = -1
    # Calculate horizontal and vertical distances
    for bomb_coord, time in bomb_list:
        x, y = bomb_coord
        if x == x_b:
            distance = abs(y - y_b)
        elif y == y_b:
            distance = abs(x - x_b)

        # If the current bomb countdown is less than the minimum countdown, or equal to the minimum countdown but at a shorter distance, update the minimum countdown and the optimal bomb coordinates
        if time < min_time or (time == min_time and distance < best_distance):
            min_time = time
            best_bomb = bomb_coord
            best_distance = distance
            return best_bomb

    return None

def stuck_in_dead_ends(x, y, dead_ends):
    if (x, y) in dead_ends:
        return 1
    return 0

def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0

def check_bomb_crates_amount(matrix, x, y):
    count = 0
    rows, cols = len(matrix), len(matrix[0])
    for i in range(max(0, x - 3), min(rows, x + 4)):
        for j in range(max(0, y - 3), min(cols, y + 4)):
            if matrix[i][j] == 1:
                count += 1

    return count

# Get valid actions
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
