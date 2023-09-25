from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from .model import DQN
from .callbacks import look_for_targets
import pickle
from typing import List
from random import shuffle
import copy
import math

import os

import events as e
from .callbacks import state_to_features, check_bomb_crates_amount

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
TOWARDS_TO_CLOSEST_COIN = "TOWARDS_TO_CLOSEST_COIN"
AWAY_FROM_CLOSEST_COIN = "AWAY_FROM_CLOSEST_COIN"
UNLESS_EVENTS = "UNLESS_EVENTS"
TOWARDS_TO_CLOSEST_OPPONENT = "TOWARDS_TO_CLOSEST_OPPONENT"
AWAY_FROM_CLOSEST_OPPONENT = "AWAY_FROM_CLOSEST_OPPONENT"

ESCAPE_FROM_BOMB = "ESCAPE_FROM_BOMB"
NOT_ESCAPE_FROM_BOMB = "NOT_ESCAPE_FROM_BOMB"

BOMB_CRATES = "BOMB_CRATES"
WAITED_OK = "WAITED_OK"
WAITED_NOT_OK = "WAITED_NOT_OK"
WILL_BOMB_CRATES_0 = "BOMB_CRATES_0"
WILL_BOMB_CRATES_1_2 = "WILL_BOMB_CRATES_1_2"
WILL_BOMB_CRATES_3_4 = "WILL_BOMB_CRATES_3_4"
WILL_BOMB_CRATES_5_MORE = "WILL_BOMB_CRATES_5_MORE"
INVALID = "INVALID"

TOWARDS_DANGER = "TOWARDS_DANGER"
TOWARDS_SAFETY = "TOWARDS_SAFETY"

TOWARDS_NOTHING = "TOWARDS_NOTHING"
TOWARDS_TO_DEATH = "TOWARDS_TO_DEATH"
TOWARDS_TO_DEATH_SELF = "TOWARDS_TO_DEATH_SELF"

ESCAPE_FROM_SELF_BOMB = "ESCAPE_FROM_SELF_BOMB"
NOT_ESCAPE_FROM_SELF_BOMB = "NOT_ESCAPE_FROM_SELF_BOMB"
TOWARDS_TO_CLOSEST_CRATES = "TOWARDS_TO_CRATES"
# TOWARDS_TO_DEAD_ENDS = "TOWARDS_TO_DEAD_ENDS"
TOWARDS_TO_TARGET = "TOWARDS_TO_TARGET"
AWAY_FROM_TARGET = "AWAY_FROM_TARGET"

BOMB_OPPONENT = "BOMB_OPPONENT"
BOMB_DEAD_END = "BOMB_DEAD_END"
UNLESS_BOMB = "UNLESS_BOMB"

AWAY_FROM_CLOSEST_CRATES = "AWAY_FROM_CLOSEST_CRATES"
TOWARDS_TO_DEAD_ENDS = "TOWARDS_TO_DEAD_ENDS"
STUCK_IN_LOOP = "STUCK_IN_LOOP"

WAITED_TO_DEATH = "WAITED_TO_DEATH"

from collections import deque

def find_closest_target(free_space, current_position, targets, logger=None):
    if len(targets) == 0:
        return None, None  # # Returns None to indicate that there are no targets to reach

    # Initializing Data Structures
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
    else:
        self.q_network = DQN(22, 6)

    self.target_network = DQN(22, 6)
    self.target_network.load_state_dict(self.q_network.state_dict())


    self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
    # self.loss_fn = nn.MSELoss()
    self.loss_fn = nn.SmoothL1Loss()

    self.replay_buffer = []
    self.buffer_size = 100000
    self.batch_size = 32
    self.gamma = 0.9

    self.tau = 0.005

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round = 0





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
    new_bomb_xys = [xy for (xy, t) in new_bombs]
    new_others = [xy for (n, s, b, xy) in new_game_state['others']]
    new_coins = new_game_state['coins']
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

    old_features = state_to_features(old_game_state)

    if self_bomb_history is not None:
        self.bomb_history.append(self_bomb_history)

    if self_coordinate is not None:
        self.coordinate_history.append(self_coordinate)

    valid_actions = []

    # Compile a list of 'targets' the agent should head towards
    cols = range(1, old_arena.shape[0] - 1)
    rows = range(1, old_arena.shape[0] - 1)

    old_crates = [(x, y) for x in cols for y in rows if (old_arena[x, y] == 1)]

    # order: left right up down wait Positions 1 to 5
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

    # Did escape the bomb?
    if old_features[len(old_features) - 3] == 1:
        bomb_list = []
        for (xb, yb), t in old_bombs:
            if ((xb == old_x) and (0 < abs(yb - old_y) < 4)) or ((yb == old_y) and 0 < (abs(xb - old_x) < 4)):
                bomb_list.append(((xb, yb), t))

        not_escape_direction = not_escape_bomb(old_bombs, old_x, old_y)
        if not_escape_direction != None:
            if self_action != not_escape_direction and self_action in valid_actions:
                if self_action != 'WAIT' and self_action != 'BOMB':
                    events.append(ESCAPE_FROM_BOMB)
            else:
                events.append(NOT_ESCAPE_FROM_BOMB)
    elif old_features[len(old_features) - 2] == 1:
        if self_action != 'WAIT' and self_action in valid_actions and self_action != 'BOMB':
            events.append(ESCAPE_FROM_SELF_BOMB)
        else:
            events.append(NOT_ESCAPE_FROM_SELF_BOMB)
    else:
        old_free_space = old_arena == 0
        for o in old_others:
            old_free_space[o] = False

        old_others = [other for other in old_others if other not in old_bomb_xys]
        target_other, target_other_index = find_closest_target(old_free_space, (old_x, old_y), old_others)
        old_coins = [coin for coin in old_coins if coin not in old_bomb_xys]
        target_coin, target_coin_index = find_closest_target(old_free_space, (old_x, old_y), old_coins)
        old_crates = [crate for crate in old_crates if crate not in old_bomb_xys]
        target_crate, target_crate_index = find_closest_target(old_free_space, (old_x, old_y), old_crates)

        if self_action == 'BOMB':
            min_distance = float('inf')
            for xy in old_others:
                distance = abs(xy[0] - old_x) + abs(xy[1] - old_y)
                min_distance = min(min_distance, distance)
            if min_distance <= 2:
                events.append(BOMB_OPPONENT)

            if [old_arena[old_x + 1, old_y], old_arena[old_x - 1, old_y], old_arena[old_x, old_y + 1], old_arena[old_x, old_y - 1]].count(1) > 0:
                if new_x == target_crate[0] and new_y == target_crate[1]:
                    events.append(BOMB_CRATES)

        elif self_action == 'WAIT':
            if target_crate is None and target_coin is None and target_other is None:
                events.append(WAITED_OK)
            else:
                events.append(WAITED_NOT_OK)
        else:
            if self_action != 'WAIT' and self_action != 'BOMB':
                if target_coin is not None:
                    if new_y == target_coin[1] and new_x == target_coin[0]:
                        events.append(TOWARDS_TO_CLOSEST_COIN)
                if target_other is not None:
                    if new_x == target_other[0] and new_y == target_other[1]:
                        events.append(TOWARDS_TO_CLOSEST_OPPONENT)
                if target_crate is not None:
                    if new_x == target_crate[0] and new_y == target_crate[1]:
                        events.append(TOWARDS_TO_CLOSEST_CRATES)
                if TOWARDS_TO_CLOSEST_COIN not in events and TOWARDS_TO_CLOSEST_OPPONENT not in events and TOWARDS_TO_CLOSEST_CRATES not in events:
                    events.append(TOWARDS_NOTHING)

    if e.INVALID_ACTION in events:
        events.clear()
        events.append(e.INVALID_ACTION)
        # events.append(INVALID)

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    print("all events:", events)

    experience = Experience(state_to_features(old_game_state), torch.tensor([[ACTIONS.index(self_action)]]),
                            state_to_features(new_game_state),
                            reward_from_events(self, events), 0)
    if len(self.replay_buffer) < self.buffer_size:
        self.replay_buffer.append(experience)
    else:
        self.replay_buffer.pop(0)
        self.replay_buffer.append(experience)

    self.transitions.append(Transition(state_to_features(old_game_state), self_action,
                                       state_to_features(new_game_state),
                                       reward_from_events(self, events)))
    self.last_state = new_game_state

    self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

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

    if self.round % 500 == 0:
        print("round:", self.round)
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                        1 - self.tau)
        self.target_network.load_state_dict(target_net_state_dict)


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
    self.current_round += 1
    old_arena = last_game_state['field']
    _, old_score, old_bombs_left, (old_x, old_y) = last_game_state['self']
    old_bombs = last_game_state['bombs']
    old_bomb_xys = [xy for (xy, t) in old_bombs]
    old_others = [xy for (n, s, b, xy) in last_game_state['others']]
    old_coins = last_game_state['coins']
    old_bomb_map = np.ones(old_arena.shape) * 5
    for (xb, yb), t in old_bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < old_bomb_map.shape[0]) and (0 < j < old_bomb_map.shape[1]):
                old_bomb_map[i, j] = min(old_bomb_map[i, j], t)

    _, _, _, old_self_pos = last_game_state['self']

    if len(self.bomb_history) != 0:
        self_bomb_history = self.bomb_history.pop()
        sbh = copy.deepcopy(self.bomb_history)
    else:
        self_bomb_history = None
        sbh = None

    sc = None
    self_coordinate = None
    if last_action is not None:
        self_coordinate = self.coordinate_history.pop()
        sc = copy.deepcopy(self.coordinate_history)

    if self_bomb_history is not None:
        self.bomb_history.append(self_bomb_history)

    if self_coordinate is not None:
        self.coordinate_history.append(self_coordinate)

    old_features = state_to_features(last_game_state)



    print("events final round:", events)

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')


    experience = Experience(state_to_features(last_game_state), torch.tensor([[ACTIONS.index(last_action)]]),
                            [1] * 22,
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
        e.KILLED_SELF: -15,
        e.GOT_KILLED: -30,
        e.KILLED_OPPONENT: 180,
        e.SURVIVED_ROUND: 0,
        e.CRATE_DESTROYED: 13,
        e.COIN_FOUND: 12,
        e.WAITED: -3,
        e.MOVED_UP: 1,
        e.MOVED_DOWN: 1,
        e.MOVED_LEFT: 1,
        e.MOVED_RIGHT: 1,


        ESCAPE_FROM_SELF_BOMB: 30,
        NOT_ESCAPE_FROM_SELF_BOMB: -50,
        ESCAPE_FROM_BOMB: 40,
        NOT_ESCAPE_FROM_BOMB: -60,
        # INVALID: -50,

        # TOWARDS_TO_DEAD_ENDS: 0,
        BOMB_OPPONENT: 50,
        BOMB_CRATES: 20,

        WAITED_OK: 5,
        WAITED_NOT_OK: -30,

        TOWARDS_TO_DEATH: -50,
        # TOWARDS_TO_DEATH_SELF: -55,
        WAITED_TO_DEATH: -60,

        TOWARDS_NOTHING: -60,
        TOWARDS_TO_CLOSEST_CRATES: 5,
        TOWARDS_TO_CLOSEST_COIN: 20,
        TOWARDS_TO_CLOSEST_OPPONENT: 20,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    print("reward_sum:", reward_sum)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

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

        # If the current bomb countdown is less than the minimum countdown, or equal to the minimum countdown but at a shorter distance, update the minimum countdown and the optimal bomb coordinates
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
