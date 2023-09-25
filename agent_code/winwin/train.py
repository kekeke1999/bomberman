from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from .model import DQN
import pickle
from typing import List
from random import shuffle
import copy
import math
import matplotlib.pyplot as plt

import os

import events as e
from .callbacks import state_to_features

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

ESCAPE_FROM_BOMB = "ESCAPE_FROM_BOMB"
NOT_ESCAPE_FROM_BOMB = "NOT_ESCAPE_FROM_BOMB"


WAITED_OK = "WAITED_OK"
WAITED_NOT_OK = "WAITED_NOT_OK"

ESCAPE_FROM_SELF_BOMB = "ESCAPE_FROM_SELF_BOMB"
NOT_ESCAPE_FROM_SELF_BOMB = "NOT_ESCAPE_FROM_SELF_BOMB"

TOWARDS_TARGET = "TOWARDS_TARGET"
TOWARDS_NOTHING = "TOWARDS_NOTHING"

BOMB_CRATES = "BOMB_CRATES"
BOMB_OPPONENT = "BOMB_OPPONENT"
BOMB_DEAD_END = "BOMB_DEAD_END"
UNLESS_BOMB = "UNLESS_BOMB"

STUCK_IN_LOOP = "STUCK_IN_LOOP"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')

    if self.train and os.path.isfile("my-saved-model.pt"):
        with open("my-saved-model.pt", "rb") as file:
            self.q_network = pickle.load(file)
        if os.path.isfile("my-replay-buffer.pkl"):
            with open("my-replay-buffer.pkl", "rb") as f:
                self.replay_buffer = pickle.load(f)
        else:
            self.replay_buffer = []
    else:
        self.q_network = DQN(20, 6)
        self.replay_buffer = []

    self.target_network = DQN(20, 6)
    self.target_network.load_state_dict(self.q_network.state_dict())
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
    # self.loss_fn = nn.MSELoss()

    self.loss_fn = nn.SmoothL1Loss()

    self.buffer_size = 100000
    self.batch_size = 32
    self.gamma = 0.9

    self.tau = 0.005

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)

    self.current_round = 0

    self.one_game = {
        'rewards': [],
        'coins': [],
        'crates': [],
        'opponents': [],
        'number': [],
        'epsilon':[],
        'points': []
    }

    self.rewards = 0
    self.coins = 0
    self.crates = 0
    self.opponents = 0
    self.games = 0
    self.points = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    # Gather information about the game state
    new_arena = new_game_state['field']
    _, new_score, new_bombs_left, (new_x, new_y) = new_game_state['self']
    new_bombs = new_game_state['bombs']
    new_bomb_map = np.ones(new_arena.shape) * 5
    for (xb, yb), t in new_bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < new_bomb_map.shape[0]) and (0 < j < new_bomb_map.shape[1]):
                new_bomb_map[i, j] = min(new_bomb_map[i, j], t)

    # Gather information about the game state
    old_arena = old_game_state['field']
    _, old_score, old_bombs_left, (old_x, old_y) = old_game_state['self']
    old_bombs = old_game_state['bombs']
    old_bomb_xys = [xy for (xy, t) in old_bombs]
    old_others = [xy for (n, s, b, xy) in old_game_state['others']]
    old_coins = old_game_state['coins']
    old_bomb_map = np.ones(old_arena.shape) * 5
    for (xb, yb), t in old_bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < old_bomb_map.shape[0]) and (0 < j < old_bomb_map.shape[1]):
                old_bomb_map[i, j] = min(old_bomb_map[i, j], t)

    _, _, _, old_self_pos = old_game_state['self']
    _, _, _, new_self_pos = new_game_state['self']

    if len(self.bomb_history) != 0:
        self_bomb_history = self.bomb_history.pop()
        sbh = copy.deepcopy(self.bomb_history)
    else:
        self_bomb_history = None
        sbh = None

    sc = None
    self_coordinate = None
    if self_action is not None:
        self_coordinate = self.coordinate_history.pop()
        sc = copy.deepcopy(self.coordinate_history)

    old_features = state_to_features(old_game_state, sc, sbh)

    if self_bomb_history is not None:
        self.bomb_history.append(self_bomb_history)

    if self_coordinate is not None:
        self.coordinate_history.append(self_coordinate)

    directions = [(old_x, old_y), (old_x + 1, old_y), (old_x - 1, old_y), (old_x, old_y + 1), (old_x, old_y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((old_arena[d] == 0) and (old_game_state['explosion_map'][d] < 1) and (old_bomb_map[d] > 0) and (not d in old_others) and (not d in old_bomb_xys)):
            valid_tiles.append(d)

    # Compile a list of 'targets' the agent should head towards
    old_dead_ends = [(x, y) for x in range(1, old_arena.shape[0] - 1) for y in range(1, old_arena.shape[0] - 1) if (old_arena[x, y] == 0)
                     and ([old_arena[x + 1, y], old_arena[x - 1, y], old_arena[x, y + 1], old_arena[x, y - 1]].count(
        0) == 1)]
    old_crates = [(x, y) for x in range(1, old_arena.shape[0] - 1) for y in range(1, old_arena.shape[0] - 1) if (old_arena[x, y] == 1)]
    old_targets = old_coins + old_dead_ends + old_crates

    # order: left right up down wait
    if old_features[0] == 1:
        valid_actions.append('UP')
    if old_features[1] == 1:
        valid_actions.append('RIGHT')
    if old_features[2] == 1:
        valid_actions.append('DOWN')
    if old_features[3] == 1:
        valid_actions.append('LEFT')
    if old_features[4] == 1:
        valid_actions.append('WAIT')
    if old_features[5] == 1:
        valid_actions.append('BOMB')

    if sc.count((old_x, old_y)) > 2:
        events.append(STUCK_IN_LOOP)

    if old_features[len(old_features) - 3] == 1:
        bomb_list = []
        for (xb, yb), t in old_bombs:
            if ((xb == old_x) and (0 < abs(yb - old_y) < 4)) or ((yb == old_y) and 0 < (abs(xb - old_x) < 4)):
                bomb_list.append(((xb, yb), t))

        not_escape_direction = not_escape_bomb(old_bombs, old_x, old_y)
        if not_escape_direction != None:
            if self_action != not_escape_direction:
                if self_action != 'WAIT' and self_action != 'BOMB':
                    # if STUCK_IN_LOOP not in events:
                    events.append(ESCAPE_FROM_BOMB)
                else:
                    events.append(NOT_ESCAPE_FROM_BOMB)
            else:
                events.append(NOT_ESCAPE_FROM_BOMB)
        else:
            events.append(ESCAPE_FROM_BOMB)
            if self_action == 'WAIT':
                events.append(WAITED_OK)

    elif old_features[len(old_features) - 2] == 1:
        if self_action != 'WAIT' and self_action in valid_actions and self_action != 'BOMB':
            # if STUCK_IN_LOOP not in events:
            events.append(ESCAPE_FROM_SELF_BOMB)
        else:
            events.append(NOT_ESCAPE_FROM_SELF_BOMB)
    else:
        # Add other agents as targets if in hunting mode or no crates/coins left
        # if len(old_coins) <= 2:
            # if self.ignore_others_timer <= 0 or len(old_crates) + len(old_coins) == 0:
        old_targets.extend(old_others)

        # Exclude targets that are currently occupied by a bomb
        old_targets = [old_targets[i] for i in range(len(old_targets)) if old_targets[i] not in old_bomb_xys]
        old_free_space = old_arena == 0

        for o in old_others:
            old_free_space[o] = False

        d = find_closest_target(old_free_space, (old_x, old_y), old_targets, self.logger)


        if self_action == 'BOMB':
            # if STUCK_IN_LOOP not in events:
            if d == (old_x, old_y) and ([old_arena[old_x + 1, old_y], old_arena[old_x - 1, old_y], old_arena[old_x, old_y + 1], old_arena[old_x, old_y - 1]].count(1) > 0):
                events.append(BOMB_CRATES)

            # if 'BOMB_DROPPED' not in events:
            if (old_x, old_y) in old_dead_ends:
                events.append(BOMB_DEAD_END)

            if len(old_others) > 0:
                if (min(abs(xy[0] - old_x) + abs(xy[1] - old_y) for xy in old_others)) <= 1:
                    events.append(BOMB_OPPONENT)

            if BOMB_CRATES not in events and BOMB_OPPONENT not in events and BOMB_DEAD_END not in events and ESCAPE_FROM_BOMB not in events:
                events.append(UNLESS_BOMB)

        else:
            if self_action == 'WAIT':
                if d is None:
                    # if STUCK_IN_LOOP not in events:
                    events.append(WAITED_OK)
                else:
                    # events.append(WAITED_NOT_OK)
                    events.append(TOWARDS_NOTHING)
            else:
                if d is not None:
                    if new_self_pos[0] == d[0] and new_self_pos[1] == d[1]:
                        if self_action != 'WAIT':
                            # if STUCK_IN_LOOP not in events:
                            events.append(TOWARDS_TARGET)
                    else:
                        events.append(TOWARDS_NOTHING)

    if e.INVALID_ACTION in events:
        events.clear()
        events.append(e.INVALID_ACTION)

    if e.WAITED in events and WAITED_OK not in events and WAITED_NOT_OK not in events and ESCAPE_FROM_BOMB not in events and NOT_ESCAPE_FROM_SELF_BOMB not in events and NOT_ESCAPE_FROM_BOMB not in events and NOT_ESCAPE_FROM_SELF_BOMB not in events:
        events.append(WAITED_NOT_OK)

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # print("all events:", events)

    # Create an experience and add it to the replay buffer
    experience = Experience(state_to_features(old_game_state, sc, sbh), torch.tensor([[ACTIONS.index(self_action)]]),
                            state_to_features(new_game_state, self.coordinate_history, self.bomb_history),
                            reward_from_events(self, events), 0)
    if len(self.replay_buffer) < self.buffer_size:
        self.replay_buffer.append(experience)
    else:
        self.replay_buffer.pop(0)
        self.replay_buffer.append(experience)

    self.transitions.append(Transition(state_to_features(old_game_state, sc, sbh), self_action,
                                       state_to_features(new_game_state, self.coordinate_history, self.bomb_history),
                                       reward_from_events(self, events)))
    self.last_state = new_game_state

    if self.epsilon <= 0.2:
        self.epsilon_decay == 1 / 1000000
    self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    # self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
    #                math.exp(-1. * self.current_round / self.epsilon_decay)

    if len(self.replay_buffer) < self.buffer_size:
        self.replay_buffer.append(experience)
    else:
        self.replay_buffer.pop(0)
        self.replay_buffer.append(experience)

    if len(self.replay_buffer) < self.batch_size:
        return

    batch = random.sample(self.replay_buffer, self.batch_size)
    state_batch, action_batch, next_state_batch, reward_batch, done_batch = zip(*batch)
    state_batch = torch.tensor(state_batch, dtype=torch.float32)
    action_batch = torch.tensor(action_batch, dtype=torch.int64)
    next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
    done_batch = torch.tensor(done_batch, dtype=torch.float32)
    q_values = self.q_network(state_batch)
    next_q_values = self.target_network(next_state_batch)

    max_actions = torch.argmax(q_values, dim=1)
    next_q_values_target = self.target_network(next_state_batch).gather(1, max_actions.unsqueeze(1))

    target_q_values = q_values.clone()
    for i in range(self.batch_size):
        target_q_values[i][action_batch[i]] = reward_batch[i] + self.gamma * next_q_values_target[i] * (
                1 - done_batch[i])

    loss = self.loss_fn(q_values, target_q_values)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    target_net_state_dict = self.target_network.state_dict()
    policy_net_state_dict = self.q_network.state_dict()

    # Update the target network periodically
    if self.round % 500 == 0:
        # print("round:", self.round)
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                        1 - self.tau)
        self.target_network.load_state_dict(target_net_state_dict)

    # evaluate_round(self, events, False)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    if e.GOT_KILLED in events:
        events.append(NOT_ESCAPE_FROM_BOMB)

    print("final events:", events)
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Evaluate the game
    # evaluate_round(self, events, True)

    experience = Experience(state_to_features(last_game_state, self.coordinate_history, self.bomb_history), torch.tensor([[ACTIONS.index(last_action)]]),
                            [1] * 20,
                            reward_from_events(self, events), 1)


    if len(self.replay_buffer) < self.buffer_size:
        self.replay_buffer.append(experience)
    else:
        self.replay_buffer.pop(0)
        self.replay_buffer.append(experience)

    if len(self.replay_buffer) < self.batch_size:
        return

    target_net_state_dict = self.target_network.state_dict()
    policy_net_state_dict = self.q_network.state_dict()

    if self.round % 500 == 0:
        print("round:",self.round)
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_network.load_state_dict(target_net_state_dict)


    if self.games % 10000 == 0:
        with open("my-replay-buffer.pkl", "wb") as file:
            pickle.dump(self.replay_buffer, file)

    self.games = self.games + 1
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_network, file)



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 60,
        e.INVALID_ACTION: -100,
        e.KILLED_SELF: -50,
        e.GOT_KILLED: -60,
        e.KILLED_OPPONENT: 180,
        e.SURVIVED_ROUND: 20,
        e.CRATE_DESTROYED: 13,
        e.COIN_FOUND: 12,
        e.WAITED: -3,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,

        ESCAPE_FROM_SELF_BOMB: 25,
        NOT_ESCAPE_FROM_SELF_BOMB: -70,

        WAITED_OK: 5,
        # WAITED_NOT_OK: -40,

        BOMB_OPPONENT: 25,
        BOMB_DEAD_END: 15,
        BOMB_CRATES: 20,

        UNLESS_BOMB: -50,

        TOWARDS_TARGET: 35,
        TOWARDS_NOTHING: -90,

        ESCAPE_FROM_BOMB: 45,
        NOT_ESCAPE_FROM_BOMB: -90,

        STUCK_IN_LOOP: -15,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # print("reward_sum:", reward_sum)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

# the same as this in callbacks.py
def not_escape_bomb(bomb_list, x_b, y_b):
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

def generate_coordinates(matrix, x, y, steps):
    coordinates = []
    rows, cols = len(matrix), len(matrix[0])

    for step in range(1, steps + 1):
        # up
        if x - step >= 0:
            coordinates.append((x - step, y))
        # down
        if x + step < rows:
            coordinates.append((x + step, y))
        # left
        if y - step >= 0:
            coordinates.append((x, y - step))
        # right
        if y + step < cols:
            coordinates.append((x, y + step))

    return coordinates

def check_if_valid_action(features):
    valid_actions = []
    if features[0] == 1:
        valid_actions.append('UP')
    if features[1] == 1:
        valid_actions.append('RIGHT')
    if features[2] == 1:
        valid_actions.append('DOWN')
    if features[3] == 1:
        valid_actions.append('LEFT')
    if features[4] == 1:
        valid_actions.append('WAIT')
    if features[5] == 1:
        valid_actions.append('BOMB')

    return valid_actions

# from rule_based_agent
def find_closest_target(free_space, start, targets, logger=None):
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
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
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]
