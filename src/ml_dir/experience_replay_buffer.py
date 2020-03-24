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

