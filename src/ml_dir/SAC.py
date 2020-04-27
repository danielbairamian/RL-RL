import numpy as np
import tensorflow as tf
from spinningup.spinup.algos.tf1.sac import sac, core
from rlbot.agents.base_agent import SimpleControllerState
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

Update #1 , added back location

==> State Space dim = 4 + 3 + 3 + 3 = 13
        + locaiton (3) = 16
'''


def action_mask(action, state, is_grounded, is_timedout):
    """
    state[13] = boost amount
    state[14] = jumped
    state[15] = double jump
    """

    # car is on the ground
    if is_grounded:
        action.roll = 0.0
        action.pitch = 0.0
        action.yaw = 0.0

    # car is in the air
    else:
        action.steer = 0.0
        action.throttle = 0.0
        action.handbrake = 0.0

    # car is out of boost
    if state[13] == 0:
        action.boost = False

    # car has jumped
    if state[14] == 1:
        # car has double jumped
        if state[15] == 1:
            action.jump = False
        # car hasn't double jumped
        # need to check for air timer timeout
        # In rocket league, once a car jumped
        # it has a limited time to perform a double jump
        else:
            if is_timedout:
                action.jump = False


    return action


def rand_to_bool(rand_num, thresh=0):
    if rand_num < thresh:
        return False
    else:
        return True


class MockGymEnv:
    def __init__(self, high):
        self.high = [high]


class SoftActorCritic():
    def __init__(self):
        self.steps_counter = 0
        self.update_counter = 0

        self.reward_exp_factor = 0.3

        self.seed = 0
        self.epochs = 100
        self.gamma = 0.99
        self.polyak = 0.995
        self.lr = 1e-3
        self.alpha = 0.2
        self.batch_size = 512
        self.start_steps = 100000
        self.update_after = 1000
        self.update_every = 200
        self.save_freq = 1000
        self.buffer_size = 5000

        self.NN_Size = (64, 64)

        self.obs_dim = 16
        self.act_dim = 8
        # all actions are clamped between -1 and 1
        # Discrete boolean actions will be treated as continuous
        # Then clamped to either -1 (False) or 1 (True)
        self.act_high = 1
        self.act_low  = -1

        self.stop_random = False
        self.start_train = False
        self.update_net = False

        # ML Inits
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

        # mock gym environment to use core.actor_critic
        self.MGE = MockGymEnv(self.act_high)
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(self.obs_dim, self.act_dim, self.obs_dim, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.mu, self.pi, self.logp_pi, self.q1, self.q2 \
                = core.mlp_actor_critic(self.x_ph, self.a_ph, action_space=self.MGE, hidden_sizes=self.NN_Size)

        with tf.variable_scope('main', reuse=True):
            # compose q with pi, for pi-learning
            _, _, _, self.q1_pi, self.q2_pi = \
                core.mlp_actor_critic(self.x_ph, self.pi, action_space=self.MGE, hidden_sizes=self.NN_Size)

            # get actions and log probs of actions for next states, for Q-learning
            _, self.pi_next, self.logp_pi_next, _, _ = \
                core.mlp_actor_critic(self.x2_ph, self.a_ph, action_space=self.MGE, hidden_sizes=self.NN_Size)

        # Target value network
        with tf.variable_scope('target'):
            # target q values, using actions from *current* policy
            _, _, _, self.q1_targ, self.q2_targ = \
                core.mlp_actor_critic(self.x2_ph, self.pi_next, action_space=self.MGE, hidden_sizes=self.NN_Size)

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

    def NN_To_Controller_State(self, nn_output):
        controller_state = SimpleControllerState()

        controller_state.steer    = nn_output[0]
        controller_state.throttle = nn_output[1]

        controller_state.pitch = nn_output[2]
        controller_state.yaw   = nn_output[3]
        controller_state.roll  = nn_output[4]

        controller_state.jump      = rand_to_bool(nn_output[5])
        controller_state.boost     = rand_to_bool(nn_output[6])
        controller_state.handbrake = rand_to_bool(nn_output[7])

        return controller_state

    def Sample_Random_Controller_State(self, has_jumped):

        controller_state = SimpleControllerState()

        controller_state.steer    = (np.random.rand()*2) -1
        controller_state.throttle = 1.0 # (np.random.rand()*2) -1

        controller_state.pitch = (np.random.rand()*2) -1
        controller_state.yaw   = (np.random.rand()*2) -1
        controller_state.roll  = (np.random.rand()*2) -1

        jump_roll = (np.random.rand()*2) -1
        if has_jumped == 1.0:
            controller_state.jump = rand_to_bool(jump_roll)
        else:
            controller_state.jump = rand_to_bool(jump_roll, thresh=0.99)

        controller_state.boost     = rand_to_bool((np.random.rand()*2) -1)
        controller_state.handbrake = rand_to_bool((np.random.rand()*2) -1)

        return controller_state

    def update_flags(self):

        if not (self.stop_random and self.start_train):
            self.steps_counter += 1
        if (self.steps_counter > self.start_steps) and not self.stop_random:
            print("Stopped random sampling")
            self.stop_random = True
        if (self.steps_counter > self.update_after) and  not self.start_train:
            print("Started training")
            self.start_train = True

        if self.start_train:
            self.update_counter += 1
            if self.update_counter % self.update_every == 0:
                self.update_net = True
                self.update_counter = 0

    def get_action(self, state, is_grounded, is_timedout):

        self.update_flags()
        if self.stop_random:
            action = self.NN_To_Controller_State(
                self.sess.run(self.pi, feed_dict={self.x_ph: state.reshape(1,-1)})[0])
            return action_mask(action, state, is_grounded, is_timedout)
        else:
            action = self.Sample_Random_Controller_State(has_jumped=state[14])
            return action_mask(action, state, is_grounded, is_timedout)

    def train_batch(self, rbuffer):
        if self.start_train:
            if self.update_net:
                batch = rbuffer.sample_batch(batch_size=self.batch_size)
                feed_dict = {self.x_ph: batch['obs1'],
                     self.x2_ph: batch['obs2'],
                     self.a_ph: batch['acts'],
                     self.r_ph: batch['rews'],
                     self.d_ph: batch['done']}

                outs = self.sess.run(self.step_ops, feed_dict)
                self.update_net = False
