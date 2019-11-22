from typing import Collection

from apprentice.learners.WhenLearner import WhenLearner
from apprentice.working_memory.representation import Activation


class Tabular:
    def __init__(self, q_init=0.6, learning_rate=0.9):
        self.row = {}
        self.q_init = q_init
        self.alpha = learning_rate

    def update(self, state, learned_reward):
        s = state
        if s not in self.row:
            self.row[s] = self.q_init

        self.row[s] = (1 - self.alpha) * self.row[s] + self.alpha * learned_reward

    def get_q(self, state):
        s = frozenset(state)
        if s not in self.row:
            self.row[s] = self.q_init
        return self.row[s]

    def __str__(self):
        return str(self.row)

class QLearner(WhenLearner):
    def __init__(self, q_init=0.6, discount=0.8, learning_rate=0.9, func=None):
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
        a = action.as_hash_repr()
        if a not in self.Q:
            return self.q_init
        return self.Q[a].get_q(state)

    def update(
        self,
        state: dict,
        action: Activation,
        reward: float,
        next_state: dict,
        next_actions: Collection[Activation],
    ) -> None:

        q_next_est = 0
        if len(next_actions) != 0:
            q_next_est = max((self.evaluate(next_state, a) for a in next_actions))

        learned_reward = reward + self.discount * q_next_est
        a = action.as_hash_repr()
        if a not in self.Q:
            self.Q[a] = self.func(
                q_init=self.q_init, learning_rate=self.learning_rate
            )

        self.Q[a].update(state, learned_reward)

