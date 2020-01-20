from copy import deepcopy
from typing import Collection

from apprentice.learners.WhenLearner import WhenLearner
from apprentice.working_memory.representation import Activation

# from concept_formation.trestle import TrestleTree
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from concept_formation.cobweb3 import Cobweb3Tree


class QLearner(WhenLearner):
    def __init__(self, q_init=0.0, discount=0.99, learning_rate=0.9,
                 func=None):
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
        name = action.get_rule_name()
        bindings = action.get_rule_bindings()
        if name not in self.Q:
            return self.q_init
        state = deepcopy(state)
        for a, v in bindings.items():
            if isinstance(v, bool):
                state[('ACTION_FEATURE', a)] = str(v)
            else:
                state[('ACTION_FEATURE', a)] = v
        return self.Q[name].get_q(state)

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

        # print('q_next_est', q_next_est)
        # print('updating %s' % action.get_rule_name(),
        #       action.get_rule_bindings())

        learned_reward = reward + self.discount * q_next_est
        name = action.get_rule_name()
        bindings = action.get_rule_bindings()
        state = deepcopy(state)
        for a, v in bindings.items():
            if isinstance(v, bool):
                state[('ACTION_FEATURE', a)] = str(v)
            else:
                state[('ACTION_FEATURE', a)] = v
        if name not in self.Q:
            self.Q[name] = self.func(q_init=self.q_init,
                                     learning_rate=self.learning_rate)

        #from pprint import pprint
        #pprint(state)
        # if name == "update_field":
        #     pprint(state)
        #     print(reward)
        #     print(learned_reward)

        self.Q[name].update(state, learned_reward)


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
    def __init__(self, q_init=0, learning_rate=0.9):
        # self.clf = LinearRegression()
        self.clf = SGDRegressor(shuffle=False, max_iter=1,
                                learning_rate="constant", eta0=learning_rate)
        self.dv = DictVectorizer(sort=False)
        self.X = []
        self.Y = []
        self.q_init = q_init

    def update(self, state, learned_reward):
        # from pprint import pprint
        # pprint(state)
        # print()
        self.X.append(state)
        self.Y.append(learned_reward)
        # print('training on', self.Y)
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
