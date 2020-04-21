import math
import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState, BoostState
import copy
from util.vec import Vec3
from src.ml_dir.experience_replay_buffer import ExperienceReplay, ReplayBuffer
from spinup.algos.tf1.sac.sac import ReplayBuffer as RBuffer
from src.ml_dir.SAC import SoftActorCritic
from src.ControllerVisualizer import Controller
import pickle
import json
from run import CONTROLLER_VIZ

class RLKickoffAgent(BaseAgent):

    def default_ball_state(self):
        return  BallState(physics=Physics(location=Vector3(0, 0, 92.75),
                                               velocity=Vector3(0, 0, 0),
                                               rotation=Rotator(0, 0, 0),
                                               angular_velocity=Vector3(0, 0, 0)))
    def default_octane_state(self):
        return  CarState(boost_amount=45,
                             # default rest position of Octane car at left diagonal boost for Mannfield
                             physics=Physics(location=Vector3(2048, -2560, 17.01),
                                             velocity=Vector3(0, 0, 0),
                                             rotation=Rotator(0, 0.75 * math.pi, 0),
                                             angular_velocity=Vector3(0, 0, 0)))

    def reset_states(self):
        self.controller_state = SimpleControllerState()
        self.car_state = self.default_octane_state()
        self.next_state = self.default_ball_state()
        self.Skip_First_Call = True
        self.current_frames_skipped = 0


    def get_distance_to_ball(self, ball_state):
        ball_location = Vec3(ball_state.physics.location)
        dist_to_ball = ball_location - self.car_state.physics.location
        dist_to_ball = dist_to_ball.length()
        return dist_to_ball


    def reset_episode(self, is_hit=False):
        # boost amout = 45 because boosts are turned off
        # and initial boost pad is given to the car
        # as it spawns with 33 boost and the small boost gives +12
        # this shouldn't affect the result of the training since
        # the amount of boost doesn't change the effect of it
        # and the agent would never use its 33 boost before getting to the pad
        # which is guaranteed to be part of the optimal solution
        # and also inevitable, which is why this design decision is made

        # Ideally I wanted to spawn with 33 and reset boosts on episode resets
        # but ever since RLBot switched to the Psyonnix API, resetting boosts is no longer supported

        # reset the car state
        car_state = self.default_octane_state()
        # reset ball state
        ball_state = self.default_ball_state()
        # reset game state (ball + car state)
        game_state = GameState(ball=ball_state, cars={self.index: car_state})

        self.set_game_state(game_state)
        self.reset_states()

        '''
        Only set the done signal if the bot actually hits the ball
        Don't do it if he times out, as this could throw off learning
        (Ignore done signal when hitting time horizon)
        
        I.E: Ignore termination conditions that do not depend on the agen't state
        
        This is also something that OpenAI SAC does, so added this for consistency and performance
        '''

        if self.rbuffer.size != 0:
            if is_hit:
                self.rbuffer.done_buf[self.last_obs_ptr] = True

    def initialize_agent(self):
        if CONTROLLER_VIZ:
            self.controller_viz = Controller(1)
        # This runs once before the bot starts up
        self.SAC_Agent = SoftActorCritic()
        self.rbuffer = RBuffer(self.SAC_Agent.obs_dim,
                               self.SAC_Agent.act_dim,
                               self.SAC_Agent.buffer_size)

        self.last_obs_ptr = self.rbuffer.ptr
        self.reset_episode()
        self.EPISODE_LAST_TIME_HIT = 0
        self.InitialReset = False
        self.ResetBoosts = False
        self.EPISODE_TIMER = 0
        # episodes last at most 3 seconds
        self.EPISODE_MAX_TIME = 3.0
        # To make training consistent, skip the first 5 frames of each episodes
        # This is to make sure that the packets have had time to catch up to the current episode
        # as there's no way to control packets explicitly
        self.SKIP_FIRST_X_FRAMES = 5
        # initial ball state, it won't move, no need to update it
        self.ball_state = self.default_ball_state()
        self.dist = self.get_distance_to_ball(self.ball_state)
        self.original_dist = copy.deepcopy(self.dist)

    def get_car_state(self, packet):
        car = packet.game_cars[self.index]
        car_state = CarState(physics=Physics(location=Vector3(
            x=car.physics.location.x,
            y=car.physics.location.y,
            z=car.physics.location.z
        ), rotation=Rotator(
            pitch=car.physics.rotation.pitch,
            yaw=car.physics.rotation.yaw,
            roll=car.physics.rotation.roll
        ), velocity=Vector3(
            x=car.physics.velocity.x,
            y=car.physics.velocity.y,
            z=car.physics.velocity.z
        ), angular_velocity=Vector3(
            x=car.physics.angular_velocity.x,
            y=car.physics.angular_velocity.y,
            z=car.physics.angular_velocity.z
        )), boost_amount=car.boost, jumped=car.jumped, double_jumped=car.double_jumped)

        return car_state

    def Skip_Helper(self, current_time, packet):
        if packet.game_info.is_round_active and not self.InitialReset:
            self.reset_episode()
            self.InitialReset = True
            self.current_frames_skipped+= 1
            self.EPISODE_TIMER = current_time

        if self.InitialReset and self.Skip_First_Call:
            self.current_frames_skipped += 1

        if self.current_frames_skipped == self.SKIP_FIRST_X_FRAMES:
            self.current_frames_skipped = 0
            self.Skip_First_Call = False

    # take action
    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # initial reset
        # after 3..2..1.. go, reset for the first time
        # only run this once, and start RL algo after (initial spawn is random)
        current_time = packet.game_info.seconds_elapsed
        # helper to skip frames for consistency
        self.Skip_Helper(current_time, packet)

        self.car_state = self.get_car_state(packet)
        self.dist = self.get_distance_to_ball(self.ball_state)

        '''
        Below is where the agent will set the controller state
        which is the action to take in this case
        '''
        # random turning for now
        turn = np.random.random()*2.0 - 1.0

        self.controller_state.throttle = 1.0
        self.controller_state.steer = turn
        self.controller_state.boost = True


        last_hit = packet.game_ball.latest_touch.time_seconds
        # if we hit the ball OR the episode timer ran out, reset
        hit_check = last_hit != self.EPISODE_LAST_TIME_HIT
        timeout_check = (current_time - self.EPISODE_TIMER) > self.EPISODE_MAX_TIME
        if (hit_check) or (timeout_check):
            self.EPISODE_LAST_TIME_HIT = packet.game_ball.latest_touch.time_seconds
            self.EPISODE_TIMER = current_time
            self.reset_episode(is_hit=hit_check)
        if CONTROLLER_VIZ:
            self.controller_viz.report(self.controller_state)
        return self.controller_state

    def reward_function(self, dist, exp_factor):
        reward = -1*(math.pow(dist, exp_factor))
        return reward

    def ObserveState(self, packet: GameTickPacket):
        if self.Skip_First_Call:
            return
        dist = self.dist / self.original_dist
        reward = self.reward_function(dist, exp_factor=1.0)

        self.next_state = self.get_car_state(packet)
        experience_data = ExperienceReplay(
            self.car_state,
            self.controller_state,
            self.next_state,
            reward,
            dist,
            done=False)

        obs = experience_data.process_state(current=True)
        act = experience_data.process_action()
        rew = reward
        obs2 = experience_data.process_state(current=False)
        done = False

        self.last_obs_ptr = self.rbuffer.ptr
        self.rbuffer.store(obs, act, rew, obs2, done)

    def SAC_Update(self):
        if self.rbuffer.size == 0:
            return

        # batch_eps = self.rbuffer.sample_batch()
        # print(batch_eps)

        obs1 = self.rbuffer.obs1_buf[self.last_obs_ptr]
        act = self.rbuffer.acts_buf[self.last_obs_ptr]
        rew = self.rbuffer.rews_buf[self.last_obs_ptr]
        obs2 = self.rbuffer.obs2_buf[self.last_obs_ptr]
        done = self.rbuffer.done_buf[self.last_obs_ptr]

        self.SAC_Agent.test()
        # print("==================================")
        # print("State: ", obs1)
        # print("Action: ", act)
        # print("Reward: ", rew)
        # print("State': ", obs2)
        # print("Done: ", done)

