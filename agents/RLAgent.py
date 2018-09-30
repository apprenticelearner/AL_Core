from pprint import pprint
from random import uniform
from random import random
from random import shuffle

from sklearn.feature_extraction import DictVectorizer
# from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
from concept_formation.cobweb3 import Cobweb3Tree

from planners.fo_planner import FoPlanner
from planners.fo_planner import subst
from planners.rulesets import functionsets
from planners.rulesets import featuresets
# from planners.fo_planner import arith_rules
from agents.BaseAgent import BaseAgent

# rules = arith_rules
epsilon = .9
search_depth = 2


def weighted_choice(choices):
    choices = [(w, c) for w, c in choices]
    shuffle(choices)
    total = sum(w for w, _ in choices)
    r = uniform(0, total)
    upto = 0
    for w, c in choices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"


def max_choice(choices):
    choices = [(w, random(), c) for w, c in choices]
    choices.sort(reverse=True)
    return choices[0][2]


def get_action_key(action):
    operator, mapping, effects = action
    return (operator.name, frozenset(mapping.items()))


class Stub:
    def __init__(self, v):
        self.v = v

    def predict(self, k):
        return self.v


class LinearFunc:

    def __init__(self):
        self.clf = LinearRegression()
        self.dv = DictVectorizer()
        self.X = []
        self.y = []

    def ifit(self, state):
        y = state['_q']
        del state['_q']
        self.X.append(state)
        self.y.append(y)
        self.clf.fit(self.dv.fit_transform(self.X), self.y)

    def categorize(self, state):
        X = self.dv.transform([state])
        resp = Stub(self.clf.predict(X)[0])
        return resp


class Tabular:

    def __init__(self, q_init=0, learning_rate=0.1):
        self.Q = {}
        self.q_init = q_init
        self.alpha = learning_rate

    def ifit(self, state):
        y = state['_q']
        del state['_q']

        x = frozenset(state.items())
        if x not in self.Q:
            self.Q[x] = self.q_init

        self.Q[x] = ((1 - self.alpha) * self.Q[x] + self.alpha * y)

    def categorize(self, state):
        x = frozenset(state.items())
        if x not in self.Q:
            self.Q[x] = self.q_init
        return Stub(self.Q[x])


class QLearner:

    def __init__(self, q_init=0, discount=1, func=None):
        self.func = func
        if self.func is None:
            self.func = Tabular

        self.Q = {}
        self.q_init = q_init
        self.g = discount

    def __len__(self):
        return sum([self.Q[a].root.num_concepts() for a in self.Q])

    def get_features(self, state):
        return {str(a): state[a] for a in state}

    def evaluate(self, state, action):
        if state is None:
            return 0
        if action not in self.Q:
            return self.q_init
        state = self.get_features(state)
        return self.Q[action].categorize(state).predict("_q")

    def update(self, state, action, reward, next_state, next_actions):
        q_max = 0
        if len(next_actions) > 0 and next_state is not None:
            q_max = max([self.evaluate(next_state, get_action_key(a))
                         for a in next_actions])
        y = reward + self.g * q_max
        print("Updating with", y)

        if action not in self.Q:
            self.Q[action] = self.func()

        state = self.get_features(state)
        state['_q'] = y
        self.Q[action].ifit(state)


class RLAgent(BaseAgent):
    """
    A new agent I'm developing while at SoarTech.
    """
    def __init__(self, feature_set, function_set):
        self.feature_set = feature_set
        self.function_set = function_set
        self.Q = QLearner()
        self.last_state = None
        self.last_action = None
        self.reward = None
        self.search_depth = 1
        self.epsilon = 0

    def request(self, state):
        tup = Tuplizer()
        flt = Flattener()
        state = flt.transform(tup.transform(state))

        knowledge_base = FoPlanner([(self.ground(a),
                                     state[a].replace('?', 'QM') if
                                     isinstance(state[a], str) else state[a])
                                    for a in state], self.feature_set)
        knowledge_base.fc_infer(depth=1, epsilon=self.epsilon)
        ostate = {self.unground(a): v.replace("QM", "?") if isinstance(v, str)
                  else v for a, v in knowledge_base.facts}

        knowledge_base = FoPlanner([(self.ground(a),
                                     state[a].replace('?', 'QM') if
                                     isinstance(state[a], str) else state[a])
                                    for a in state], self.function_set)
        knowledge_base.fc_infer(depth=self.search_depth, epsilon=self.epsilon)

        state = {self.unground(a): v.replace("QM", "?") if isinstance(v, str)
                 else v for a, v in knowledge_base.facts}

        actions = [{'skill_label': 'NO_LABEL', 'foci_of_attention': [],
                    'selection': vm['?selection'],
                    'action': vm['?action'],
                    'inputs': {e[0]: e[1] for e in vm['?inputs']}}
                   for vm in knowledge_base.fc_query([(('sai', '?selection',
                                                        '?action', '?inputs'),
                                                       True)], max_depth=0,
                                                     epsilon=0)]

        actions = [(self.Q.evaluate(ostate, self.get_action_key(a)), random(),
                    a) for a in actions]

        actions.sort(reverse=True)
        print(actions)

        self.last_state = ostate
        self.last_action = self.get_action_key(actions[0][2])
        self.reward = 0

        return actions[0][2]

    def train(self, state, label, foas, selection, action, inputs, correct):
        if ((self.last_state is None or self.last_action is None or
             self.reward is None)):
            return

        print("CORRECTNESS", correct)

        # add reward based on feedback.
        if correct:
            self.reward += 1
        else:
            self.reward -= 1

        print('REWARD', self.reward)

        # terminal state, no next_actions
        next_actions = []

        print("LAST ACTION", self.last_action, self.reward)
        self.Q.update(self.last_state, self.last_action, self.reward, None,
                      next_actions)
        print(selection, action, inputs)

    def check(self, state, selection, action, inputs):
        return False

    def ground(self, arg):
        if isinstance(arg, tuple):
            return tuple(self.ground(e) for e in arg)
        elif isinstance(arg, str):
            return arg.replace('?', 'QM')
        else:
            return arg

    def unground(self, arg):
        if isinstance(arg, tuple):
            return tuple(self.unground(e) for e in arg)
        elif isinstance(arg, str):
            return arg.replace('QM', '?')
        else:
            return arg

    def get_action_key(self, a):
        return frozenset(a['inputs'].items())
