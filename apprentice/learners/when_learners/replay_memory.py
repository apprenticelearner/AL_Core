import random
from collections import namedtuple

# Transition = namedtuple(
#     'Transition', ('state_action', 'reward', 'next_state_actions'))

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'next_actions'))


class ReplayMemory(object):

    def __init__(self, capacity):
        """
        Constructor.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Push an experience into the replay memory.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def set_capacity(self, capacity):
        """
        Update the memory (either increase or decrease) the memory to the
        specified capacity.
        """
        self.capacity = capacity
        self.memory = self.memory[:capacity]
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        if self.position >= capacity:
            self.position = 0

    def sample(self, batch_size):
        """
        Retreive a set of samples for training.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Size of memory
        """
        return len(self.memory)
