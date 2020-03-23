import math
import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState, BoostState
import copy
import flatbuffers

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

    def reset_episode(self):
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


    def initialize_agent(self):
        # This runs once before the bot starts up
        self.reset_episode()
        self.EPISODE_LAST_TIME_HIT = 0
        self.InitialReset = False
        self.ResetBoosts = False
        self.EPISODE_TIMER = 0
        # episodes last at most 3 seconds
        self.EPISODE_MAX_TIME = 3.0

        # initial ball state, it won't move, no need to update it
        self.ball_state = self.default_ball_state()


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

    # take action
    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # initial reset
        # after 3..2..1.. go, reset for the first time
        # only run this once, and start RL algo after (initial spawn is random)
        if packet.game_info.is_round_active and not self.InitialReset:
            self.reset_episode()
            self.InitialReset = True

        self.car_state = self.get_car_state(packet)

        # random turning for now
        turn = np.random.random()*2.0 - 1.0

        self.controller_state.throttle = 1.0
        self.controller_state.steer = turn
        self.controller_state.boost = True

        last_hit = packet.game_ball.latest_touch.time_seconds
        current_time = packet.game_info.seconds_elapsed
        # if we hit the ball OR the episode timer ran out, reset
        if (last_hit != self.EPISODE_LAST_TIME_HIT) or ((current_time - self.EPISODE_TIMER) > self.EPISODE_MAX_TIME):
            self.EPISODE_LAST_TIME_HIT = packet.game_ball.latest_touch.time_seconds
            self.EPISODE_TIMER = current_time
            self.reset_episode()

        return self.controller_state

    def ObserveState(self, packet: GameTickPacket):
        if not packet.game_info.is_round_active:
            return

        self.next_state = self.get_car_state(packet)

        print("===============================")

        print(self.car_state.physics.location.x)
        print(self.car_state.physics.location.y)

        print(self.next_state.physics.location.x)
        print(self.next_state.physics.location.y)
