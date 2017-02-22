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
from planners.fo_planner import arith_rules
from agents.BaseAgent import BaseAgent

rules = arith_rules
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
            self.func = Cobweb3Tree

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

    def __init__(self, action_set):
        self.Q = QLearner(func=Cobweb3Tree)
        self.last_state = None
        self.last_action = None
        self.reward = None
        self.max_episodes = 5

    def request(self, state):
        tup = Tuplizer()
        flt = Flattener()
        state = flt.transform(tup.transform(state))

        new = {}
        for attr in state:
            if (isinstance(attr, tuple) and attr[0] == 'value'):
                new[('editable', attr[1])] = state[attr] == ''
                for attr2 in state:
                    if (isinstance(attr2, tuple) and attr2[0] == 'value'):
                        if (attr2 == attr or attr < attr2 or (state[attr] == ""
                                                              or state[attr2]
                                                              == "")):
                            continue
                        if (state[attr] == state[attr2]):
                            new[('eq', attr, attr2)] = True
        state.update(new)
        # pprint(state)

        # for episode in range(self.max_episodes):
        while True:

            print("#########")
            print("NEW TRACE")
            print("#########")
            kb = FoPlanner([(self.ground(a),
                             state[a].replace('?', 'QM') if
                             isinstance(state[a], str) else
                             state[a])
                            for a in state], rules)

            curr_state = {x[0]: x[1] for x in kb.facts}
            next_actions = [a for a in kb.fc_get_actions(epsilon=epsilon)]
            trace_actions = []
            depth = 0

            while depth < search_depth:
                actions = [(self.Q.evaluate(curr_state, get_action_key(a)), a)
                           for a in next_actions]

                print("NEXT ACTION WEIGHTS")
                print(sorted([(w, a[0].name[0]) for w, a in actions],
                             reverse=True))

                # operator, mapping, effects = weighted_choice(actions)
                operator, mapping, effects = max_choice(actions)
                # operator, mapping, effects = choice(action_space)

                self.last_state = curr_state
                self.last_action = get_action_key((operator, mapping, effects))
                trace_actions.append(subst(mapping, operator.name))

                for f in effects:
                    kb.add_fact(f)

                # if not termainal, then decrease reward
                # self.reward = -1
                self.reward = 0
                curr_state = {x[0]: x[1] for x in kb.facts}
                depth += 1

                # check if we're in a terminal state
                # if so, query oracle
                for f in effects:
                    f = self.unground(f)
                    if f[0] == 'sai':
                        response = {}
                        response['label'] = str(trace_actions)
                        response['selection'] = f[1]
                        response['action'] = f[2]
                        response['inputs'] = {'value': f[3]}
                        # {a: rg_exp[3+i] for i, a in
                        #                       enumerate(input_args)}
                        # response['inputs'] = list(rg_exp[3:])
                        response['foas'] = []
                        # pprint(response)
                        print("EXECUTING ACTION", self.last_action)
                        print("Requesting oracle feedback")

                        return response

                # punish for failed search
                if depth >= search_depth:
                    # self.reward -= 3 * search_depth
                    curr_state = None
                    next_actions = []
                else:
                    # because we're not terminal we can compute next_actions
                    next_actions = [a for a in
                                    kb.fc_get_actions(epsilon=epsilon,
                                                      must_match=effects)]

                self.Q.update(self.last_state, self.last_action, self.reward,
                              curr_state, next_actions)

        # return {}

    def train(self, state, label, foas, selection, action, inputs, correct):
        # tup = Tuplizer()
        # flt = Flattener()
        # state = flt.transform(tup.transform(state))
        # pprint(state)
        if ((self.last_state is None or self.last_action is None or
             self.reward is None)):
            return

        print("CORRECTNESS", correct)

        # add reward based on feedback.
        if correct:
            self.reward += 1  # * search_depth
        # else:
        #     self.reward -= 1  # 2 * search_depth

        # terminal state, no next_actions
        next_actions = []

        print("LAST ACTION", self.last_action, self.reward)
        self.Q.update(self.last_state, self.last_action, self.reward, None,
                      next_actions)

        # print('searching for:', correct)
        # for i in range(30):
        #     kb = FoPlanner([(self.ground(a),
        #                      state[a].replace('?', 'QM') if
        #                      isinstance(state[a], str) else
        #                      state[a])
        #                     for a in state], rules)

        #     cum_reward = 0
        #     trace = [{str(a): v for a, v in kb.facts}]
        #     rewards = [0]
        #     while True:
        #         action_space = [a for a in
        #         kb.fc_get_actions(epsilon=epsilon)]

        #         actions = []
        #         for some_action in action_space:
        #             new_state = {}
        #             for f in kb.facts.union(some_action[2]):
        #                 if isinstance(f, tuple) and len(f) == 2:
        #                     new_state[str(f[0])] = f[1]
        #                 else:
        #                     new_state[str(f)]: True
        #             utility = 0
        #             if self.fit:
        #                 utility =
        #                 self.UtilityFun.predict(self.dv.transform([new_state]))[0]
        #             actions.append((some_action, exp(utility)))

        #         operator, mapping, effects = weighted_choice(actions)

        #         # operator, mapping, effects = choice(action_space)

        #         for f in effects:
        #             kb.add_fact(f)

        #         new_state = {}
        #         for f in kb.facts.union(action[2]):
        #             if isinstance(f, tuple) and len(f) == 2:
        #                 new_state[str(f[0])] = f[1]
        #             else:
        #                 new_state[str(f)]: True
        #         trace.append(new_state)

        #         reward = 0

        #         done = False
        #         for f in effects:
        #             if f[0] == 'sai':
        #                 done = True
        #                 gen_sai = f[:4]
        #                 goal = ('sai', selection, action, '-1' if 'value' not
        #                         in inputs else inputs['value'])
        #                 # print(gen_sai, 'vs', goal)
        #                 if gen_sai == goal:
        #                     if correct:
        #                         reward += 2 * search_depth
        #                     else:
        #                         reward -= 2 * search_depth
        #                 else:
        #                     if correct:
        #                         reward -= 2 * search_depth
        #                     # else:
        #                     #     reward += search_depth * 2

        #         rewards.append(reward)
        #         cum_reward += reward

        #         if done or cum_reward < -1 * search_depth:
        #             break

        #     self.X += trace
        #     self.y += [sum(rewards[i:]) for i in range(len(rewards))]
        #     print("reward obtained from sample %i: %i" % (i, cum_reward))

        # self.UtilityFun.fit(self.dv.fit_transform(self.X), self.y)
        # self.fit = True

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
