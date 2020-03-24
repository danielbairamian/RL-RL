import math
import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState, BoostState
import copy

class ExperienceReplay():

    def __init__(self, current_state, action, next_state, reward, done):
        self.current_state = current_state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.done = done

class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.replay_buffer = []

    def push(self, experience):
        self.replay_buffer.append(experience)
        if len(self.replay_buffer) > self.max_size:
            self.replay_buffer.pop(0)

    def clear(self):
        self.replay_buffer = []

