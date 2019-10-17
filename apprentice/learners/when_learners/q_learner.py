from typing import Dict, Collection, FrozenSet
from abc import ABCMeta, abstractmethod

from experta import Fact

from apprentice.learners.WhenLearner import WhenLearner
from apprentice.working_memory.representation import Activation


class Tabular:
    def __init__(self, q_init=0, learning_rate=0.1):
        self.row = {}
        self.q_init = q_init
        self.alpha = learning_rate

    def update(self, state, learned_reward):
        x = frozenset(state.items())
        if x not in self.row:
            self.row[x] = self.q_init

        self.row[x] = (1 - self.alpha) * self.row[x] + self.alpha * learned_reward

    def get_q(self, state):
        x = frozenset(state.items())
        if x not in self.row:
            self.row[x] = self.q_init
        return self.row[x]


class QLearner(WhenLearner):
    def __init__(self, q_init=0, discount=1, learning_rate=0.1, func=None):
        self.func = func
        if self.func is None:
            self.func = Tabular

        self.Q = {}
        self.q_init = q_init
        self.discount = discount
        self.learning_rate = learning_rate

    def evaluate(self, state: dict, action: Activation) -> float:
        if state is None:
            return 0
        if action not in self.Q:
            return self.q_init
        return self.Q[action].get_q(state)

    def update(
        self,
        state: dict,
        action: Activation,
        reward: float,
        next_state: dict,
        next_actions: Collection[Activation],
    ) -> None:
        q_next_est = max((self.evaluate(next_state, a) for a in next_actions))
        learned_reward = reward + self.discount * q_next_est

        if action not in self.Q:
            self.Q[action] = self.func(
                q_init=self.q_init, learning_rate=self.learning_rate
            )

        self.Q[action].update(state, learned_reward)
