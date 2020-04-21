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
        self.buffer_size = 500

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
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(self.obs_dim, self.act_dim, self.obs_dim, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.mu, self.pi, self.logp_pi, self.q1, self.q2 \
                = core.mlp_actor_critic(self.x_ph, self.a_ph, action_space=self.MGE)

        with tf.variable_scope('main', reuse=True):
            # compose q with pi, for pi-learning
            _, _, _, self.q1_pi, self.q2_pi = \
                core.mlp_actor_critic(self.x_ph, self.pi, action_space=self.MGE)

            # get actions and log probs of actions for next states, for Q-learning
            _, self.pi_next, self.logp_pi_next, _, _ = \
                core.mlp_actor_critic(self.x2_ph, self.a_ph, action_space=self.MGE)

        # Target value network
        with tf.variable_scope('target'):
            # target q values, using actions from *current* policy
            _, _, _, self.q1_targ, self.q2_targ = \
                core.mlp_actor_critic(self.x2_ph, self.pi_next, action_space=self.MGE)

        # Min Double-Q:
        self.min_q_pi = tf.minimum(self.q1_pi, self.q2_pi)
        self.min_q_targ = tf.minimum(self.q1_targ, self.q2_targ)

        # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
        self.q_backup = \
            tf.stop_gradient(self.r_ph +
                             self.gamma * (1 - self.d_ph) *
                             (self.min_q_targ - self.alpha * self.logp_pi_next))

        # Soft actor-critic losses
        self.pi_loss = tf.reduce_mean(self.alpha * self.logp_pi - self.min_q_pi)
        self.q1_loss = 0.5 * tf.reduce_mean((self.q_backup - self.q1) ** 2)
        self.q2_loss = 0.5 * tf.reduce_mean((self.q_backup - self.q2) ** 2)
        self.value_loss = self.q1_loss + self.q2_loss

        # Policy train op
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        self.pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_pi_op = self.pi_optimizer.minimize(self.pi_loss, var_list=core.get_vars('main/pi'))

        # Value train op
        # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
        self.value_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.value_params = core.get_vars('main/q')
        with tf.control_dependencies([self.train_pi_op]):
            self.train_value_op = self.value_optimizer.minimize(self.value_loss, var_list=self.value_params)

        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([self.train_value_op]):
            self.target_update = tf.group([tf.assign(self.v_targ, self.polyak * self.v_targ + (1 - self.polyak) * self.v_main)
                                      for self.v_main, self.v_targ in zip(core.get_vars('main'), core.get_vars('target'))])

        # All ops to call during one training step
        self.step_ops = [self.pi_loss, self.q1_loss, self.q2_loss, self.q1, self.q2, self.logp_pi,
                    self.train_pi_op, self.train_value_op, self.target_update]

        # Initializing targets to match main variables
        self.target_init = tf.group([tf.assign(self.v_targ, self.v_main)
                                for self.v_main, self.v_targ in zip(core.get_vars('main'), core.get_vars('target'))])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init)

    def test(self):
        print(self.x_ph)
        print(self.mu)
        print(self.pi_loss)
        print(self.logp_pi_next)
