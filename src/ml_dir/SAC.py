import numpy as np
import tensorflow as tf
from spinningup.spinup.algos.tf1.sac import sac, core

'''
Based on OpenAI Spinning up SAC
I need to close this and make it my own
because the openai main logic is based on Gym envs
and I need to make this fit into Rocket League RLBot

Some of the variables won't be used / or deleted
'''

'''
RL Agent architecture:

Action space: Controller state:

steer: float = 0.0,
throttle: float = 0.0,
pitch: float = 0.0,
yaw: float = 0.0,
roll: float = 0.0,
jump: bool = False,
boost: bool = False,
handbrake: bool = False,
use_item: bool = False) <-- Not using this one, no items

Action Space Dimensions --> 8

State space: Car state:

self.physics =                              physics
    Physics
    self.location = location                 Vec3
    self.rotation = rotation                 Rotator (roll, pitch, yaw)
    self.velocity = velocity                 Vec3
    self.angular_velocity = angular_velocity Vec3
self.boost_amount = boost_amount             Float [0, 100]
self.jumped = jumped                         Bool
self.double_jumped = double_jumped           Bool

+ Normalized distance to ball                Float

State Space dim --> 4 + Physics (Subject to change on what physics uses)
Currently using everything except location (info encoded in distance)

==> State Space dim = 4 + 3 + 3 + 3 = 13
'''
class MockGymEnv:
    def __init__(self, high):
        self.high = [high]

class SoftActorCritic():
    def __init__(self):
        self.seed = 0
        self.epochs = 100
        self.gamma = 0.99
        self.polyak = 0.995
        self.lr = 1e-3
        self.alpha = 0.2
        self.batch_size = 100
        self.start_steps = 1000
        self.update_after = 1000
        self.update_every = 50
        self.save_freq = 1000

        self.obs_dim = 13
        self.act_dim = 8
        # all actions are clamped between -1 and 1
        # Discrete boolean actions will be treated as continuous
        # Then clamped to either -1 (False) or 1 (True)
        self.act_high = 1
        self.act_low  = -1

        # ML Inits
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

        # mock gym environment to use core.actor_critic
        self.MGE = MockGymEnv(self.act_high)






