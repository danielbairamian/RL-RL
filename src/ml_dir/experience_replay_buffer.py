import numpy as np

def process_state_static(state, dist_norm):
    location = state.physics.location
    rotation = state.physics.rotation
    velocity = state.physics.velocity
    ang_vel = state.physics.angular_velocity
    boost = state.boost_amount
    jumped = 1 if state.jumped else -1
    double_j = 1 if state.double_jumped else -1
    dist = dist_norm

    return location.x, location.y, location.z, \
           rotation.roll, rotation.pitch, rotation.yaw, \
           velocity.x, velocity.y, velocity.z, \
           ang_vel.x, ang_vel.y, ang_vel.z, \
           dist, boost, jumped, double_j


class ExperienceReplay():

    def __init__(self, current_state, action, next_state, reward, norm_dist, done):
        self.current_state = current_state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.done = done
        self.norm_dist = norm_dist


    def process_state(self, current=True):
        if current:
            state = self.current_state
        else:
            state = self.next_state
        location = state.physics.location
        rotation = state.physics.rotation
        velocity = state.physics.velocity
        ang_vel  = state.physics.angular_velocity
        boost = state.boost_amount
        jumped = 1 if state.jumped else -1
        double_j = 1 if state.double_jumped else -1
        dist = self.norm_dist

        return location.x, location.y, location.z, \
               rotation.roll, rotation.pitch, rotation.yaw, \
               velocity.x, velocity.y, velocity.z, \
               ang_vel.x, ang_vel.y, ang_vel.z, \
               dist, boost, jumped, double_j

    def process_action(self):
        steer = self.action.steer
        throttle = self.action.throttle
        pitch = self.action.pitch
        yaw = self.action.yaw
        roll = self.action.roll
        jump = 1 if self.action.jump else -1
        boost = 1 if self.action.boost else -1
        handbrake = 1 if self.action.handbrake else -1

        return steer, throttle, \
               pitch, yaw, roll, \
               jump, boost, handbrake




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

    def sample_batch(self, batch_size=32):
        buff_size = len(self.replay_buffer)
        sample_size = min(batch_size, buff_size)

        idxs = np.random.randint(0, buff_size, size=sample_size)
        batch = []
        for i in idxs:
            batch.append(self.replay_buffer[i])
        return batch

