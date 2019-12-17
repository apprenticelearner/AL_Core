from copy import deepcopy
from typing import Collection

from apprentice.learners.WhenLearner import WhenLearner
from apprentice.working_memory.representation import Activation

# from concept_formation.trestle import TrestleTree
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from concept_formation.cobweb3 import Cobweb3Tree


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
            q_next_est = max((self.evaluate(next_state, a)
                              for a in next_actions))

        from pprint import pprint
        print('LEN A in Q', len(self.Q))
        for a in self.Q:
            print(self.Q[a].tree.root.av_counts['_q'])
        print("state")
        pprint(state)
        print()
        print("action")
        pprint(action)
        print()
        print("next_state")
        pprint(next_state)
        print()
        print("immediate reward", reward)
        print("discounted future reward", self.discount * q_next_est)

        learned_reward = reward + self.discount * q_next_est
        a = action.as_hash_repr()
        if a not in self.Q:
            self.Q[a] = self.func(q_init=self.q_init,
                                  learning_rate=self.learning_rate)

        self.Q[a].update(state, learned_reward)


class Tabular:
    def __init__(self, q_init=0.6, learning_rate=0.9):
        self.row = {}
        self.q_init = q_init
        self.alpha = learning_rate

    def update(self, state, learned_reward):
        s = frozenset(state)
        if s not in self.row:
            self.row[s] = self.q_init

        self.row[s] = (1 - self.alpha) * self.row[s] + \
            self.alpha * learned_reward

    def get_q(self, state):
        s = frozenset(state)
        if s not in self.row:
            self.row[s] = self.q_init
        return self.row[s]

    def __str__(self):
        return str(self.row)


class LinearFunc:
    def __init__(self, q_init=0, learning_rate=0):
        self.clf = LinearRegression()
        self.dv = DictVectorizer()
        self.X = []
        self.Y = []
        self.q_init = q_init

    def update(self, state, learned_reward):
        # from pprint import pprint
        # pprint(state)
        # print()
        self.X.append(state)
        self.Y.append(learned_reward)
        self.clf.fit(self.dv.fit_transform(self.X), self.Y)

    def get_q(self, state):
        if len(self.X) == 0:
            return self.q_init
        x = self.dv.transform([state])
        return self.clf.predict(x)[0]


class Cobweb:
    def __init__(self, q_init=0, learning_rate=0):
        self.tree = Cobweb3Tree()

    def update(self, state, learned_reward):
        x = deepcopy(state)
        x['_q'] = float(learned_reward)
        self.tree.ifit(x)

    def get_q(self, state):
        return self.tree.categorize(state).predict('_q')
