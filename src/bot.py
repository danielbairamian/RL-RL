import math

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState, BoostState


from util.orientation import Orientation
from util.vec import Vec3

class MyBot(BaseAgent):
    def reset_episode(self):
        car_state = CarState(boost_amount=45,
                             physics=Physics(location=Vector3(2048, -2560), rotation=Rotator(0, 0.75 * math.pi, 0),
                                             angular_velocity=Vector3(0, 0, 0)))

        ball_state = BallState(Physics(location=Vector3(0, 0, 92.75), velocity=Vector3(0, 0, 0),
                                       angular_velocity=Vector3(0, 0, 0), rotation=Rotator(0, 0, 0)))
        game_state = GameState(ball=ball_state, cars={self.index: car_state})
        self.set_game_state(game_state)


    def initialize_agent(self):
        # This runs once before the bot starts up
        self.controller_state = SimpleControllerState()
        self.reset_episode()
        self.EPISODE_LAST_TIME_HIT = 0
        self.InitialReset = False
        self.ResetBoosts = False

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # initial reset
        # after 3..2..1.. go, reset for the first time
        # only run this once, and start RL algo after (initial spawn is random)
        if packet.game_info.is_round_active and not self.InitialReset:
            self.reset_episode()
            self.InitialReset = True

        # starter code given to drive directly towards the ball
        # keep it for now.
        ball_location = Vec3(packet.game_ball.physics.location)
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_to_ball = ball_location - car_location

        # Find the direction of our car using the Orientation class
        car_orientation = Orientation(my_car.physics.rotation)
        car_direction = car_orientation.forward

        steer_correction_radians = find_correction(car_direction, car_to_ball)

        if steer_correction_radians > 0:
            # Positive radians in the unit circle is a turn to the left.
            turn = -1.0  # Negative value for a turn to the left.
        else:
            turn = 1.0

        self.controller_state.throttle = 1.0
        self.controller_state.steer = turn
        self.controller_state.boost = True

        # debugger if needed (keeping for now)
        # draw_debug(self.renderer, my_car, packet.game_ball, action_display)

        last_hit = packet.game_ball.latest_touch.time_seconds
        if last_hit != self.EPISODE_LAST_TIME_HIT:
            self.EPISODE_LAST_TIME_HIT = packet.game_ball.latest_touch.time_seconds
            self.reset_episode()

        return self.controller_state


def find_correction(current: Vec3, ideal: Vec3) -> float:
    # Finds the angle from current to ideal vector in the xy-plane. Angle will be between -pi and +pi.

    # The in-game axes are left handed, so use -x
    current_in_radians = math.atan2(current.y, -current.x)
    ideal_in_radians = math.atan2(ideal.y, -ideal.x)

    diff = ideal_in_radians - current_in_radians

    # Make sure that diff is between -pi and +pi.
    if abs(diff) > math.pi:
        if diff < 0:
            diff += 2 * math.pi
        else:
            diff -= 2 * math.pi

    return diff


def draw_debug(renderer, car, ball, action_display):
    renderer.begin_rendering()
    # draw a line from the car to the ball
    renderer.draw_line_3d(car.physics.location, ball.physics.location, renderer.white())
    # print the action that the bot is taking
    renderer.draw_string_3d(car.physics.location, 2, 2, action_display, renderer.white())
    renderer.end_rendering()
