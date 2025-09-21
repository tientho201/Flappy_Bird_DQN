import random 
import numpy as np 

# Replay Buffer
class ReplayBuffer:
    def __init__(self,capacity = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done) # tuple
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
            self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # return value of batch_size elements from self.buffer
        states, actions, rewards, next_states, done = map(lambda x: np.array(x), zip(*batch))
        return states, actions, rewards, next_states, done
    
    def can_sample(self, batch_size):
        return len(self.buffer) >= batch_size
    
    def __len__(self):
        return len(self.buffer)

