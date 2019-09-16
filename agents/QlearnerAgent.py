from pprint import pprint
# from random import random
from random import choice

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
from concept_formation.structure_mapper import rename_flat
from concept_formation.utils import isNumber
from concept_formation.trestle import TrestleTree
from concept_formation.visualize import visualize

from agents.BaseAgent import BaseAgent
from learners.WhenLearner import get_when_learner
from learners.WhereLearner import get_where_learner
from learners.WhichLearner import get_which_learner
from planners.base_planner import get_planner_class
from planners.VectorizedPlanner import VectorizedPlanner
# from learners.HowLearner import get_planner
# from planners.fo_planner import FoPlanner, execute_functions, unify, subst
import itertools
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

    def __init__(self, q_init=0, learning_rate=0.8):
        self.Q = {}
        self.A = {}
        self.q_init = q_init
        self.alpha = learning_rate

    def ifit(self, action, weight):
        #y = state['_q']
        #del state['_q']
        y = weight

        #encode the action, i.e. Explanation ...
        x = str(action)
        if x not in self.Q:
            self.Q[x] = self.q_init
            self.A[x] = action

        self.Q[x] = ((1 - self.alpha) * self.Q[x] + self.alpha * y)

    def categorize(self, action):
        x = str(action)
        if x not in self.Q:
            self.Q[x] = self.q_init
        return Stub(self.Q[x])

class TrestleLearner:
    def __init__(self):
        self.T = TrestleTree()
        self.action_list = {}
        self.alpha = 1
        self.g = 1

    def eval_state(self, state):
        '''
        vals = [self.evaluate(state, action) for action in self.action_list]
        if vals == []:
            return 0
        else:
            return max(vals)
        '''
        res = self.T.infer_missing(state, choice_fn='sampled', allow_none=False)
        try:
            return float(res['_q'])
        except Exception as e:
            return 0


    def evaluate(self, state, action):
        state['action'] = str(action)
        # state['_q'] = None
        # predict val of this field
        res = self.T.infer_missing(state, choice_fn='sampled', allow_none=False)
        #pprint('Eval:')
        #pprint(res)
        try:
            return float(res['_q'])
        except Exception as e:
            return 0

    def update(self, state, action, reward, next_state):
        exp_rew = self.evaluate(state, action) * (1 - self.alpha) + \
            self.alpha * (self.g * reward + self.eval_state(next_state))
        pprint(reward)
        pprint(exp_rew)
        pprint('')

        state['action'] = str(action)
        if str(action) not in self.action_list:
            self.action_list[str(action)] = action
        state['_q'] = str(reward)
        self.T.ifit(state)
        #pprint(state)


class QLearner:

    def __init__(self, q_init=0, discount=1, func=None):
        self.func = func
        if self.func is None:
            self.func = Tabular

        self.Q = {}
        self.q_init = q_init
        self.g = discount
        self.T = TrestleTree()

    def __len__(self):
        return sum([self.Q[a].root.num_concepts() for a in self.Q])
    '''
    def get_features(self, state):
        return {str(a): state[a] for a in state}
    '''
    def evaluate(self, state, action):
        if action is None:
            return 0
        if state not in self.Q:
            return self.q_init
        # just gets the value
        return self.Q[state].categorize(action).predict("_q")

    def update(self, state, action, reward, next_state):
        state = state_to_key(state)
        next_state = state_to_key(next_state)

        # for TrestleTree
        #state = self.T.ifit(flatten_state(state)).concept_id
        #next_state = self.T.ifit(flatten_state(next_state)).concept_id

        q_max = 0
        if next_state in self.Q:
            q_max = max([v for v in self.Q[next_state].Q.values()])
        y = reward + self.g * max(q_max, 0)
        #print("Updating with", y)

        if state not in self.Q:
            self.Q[state] = self.func()

        #state = self.get_features(state)
        #state['_q'] = y
        self.Q[state].ifit(action, y)



def state_to_key(state):
    return trim(tuple(sorted(state.get_view("key_vals_grounded"))))


# temporary
def isvalue(state):
    try:
        float(state)
        return True
    except ValueError:
        return False

def trim(state):
    if isinstance(state, tuple):
        return tuple(trim(ele) for ele in state)

    elif not isinstance(state, bool) and isvalue(state):
        return '#NUM'
    else:
        return state

def compute_exp_depth(exp):
    """
    Doc String
    """
    if isinstance(exp, tuple):
        return 1 + max([compute_exp_depth(sub) for sub in exp])
    return 0


# def replace_vars(arg, i=0):
#     """
#     Doc String
#     """
#     if isinstance(arg, tuple):
#         ret = []
#         for elem in arg:
#             replaced, i = replace_vars(elem, i)
#             ret.append(replaced)
#         return tuple(ret), i
#     elif isinstance(arg, str) and len(arg) > 0 and arg[0] == '?':
#         return '?foa%s' % (str(i)), i+1
#     else:
#         return arg, i

def variablize_by_where(state, match):
    mapping = {'arg' + str(i-1) if i > 0 else 'sel':
               ele for i, ele in enumerate(match)}
    r_state = rename_flat(state, {mapping[a]: a for a in mapping})
    return r_state


def unvariablize_by_where(state, match):
    mapping = {ele: 'arg' + str(i-1) if i > 0 else 'sel'
               for i, ele in enumerate(match)}
    r_state = rename_flat(state, {mapping[a]: a for a in mapping})
    return r_state


def expr_comparitor(fact, expr, mapping={}):
    if(isinstance(expr, dict)):
        if(isinstance(fact, dict)):
            # Compare keys
            if(not expr_comparitor(list(fact.keys())[0],
               list(expr.keys())[0], mapping)):
                return False
            # Compare values
            if(not expr_comparitor(list(fact.values())[0],
               list(expr.values())[0], mapping)):
                return False
            return True
        else:
            return False
    if(isinstance(expr, tuple)):
        if(isinstance(fact, tuple) and len(fact) == len(expr)):
            for x, y in zip(fact, expr):
                if(not expr_comparitor(x, y, mapping)):
                    return False
            return True
        else:
            return False
    elif expr[0] == "?" and mapping.get(expr, None) != fact:
        mapping[expr] = fact
        return True
    elif(expr == fact):
        return True
    else:
        return False


def expression_matches(expression, state):
    state = state.get_view("flat_ungrounded")
    #print(expression)
    #print(state)
    for fact_expr, value in state.items():
        if(isinstance(expression, dict)):
            fact_expr = {fact_expr: value}

        mapping = {}
        if(expr_comparitor(fact_expr, expression, mapping)):
            yield mapping


EMPTY_RESPONSE = {}


class QlearnerAgent(BaseAgent):

    ## working
    def __init__(self, feature_set, function_set,
                 when_learner='decisiontree', where_learner='MostSpecific',
                 heuristic_learner='proportion_correct', how_cull_rule='all',
                 planner='fo_planner', search_depth=1, numerical_epsilon=0.0):
        print(planner)


        self.q_learner = QLearner()
        self.t_learner = TrestleLearner()

        self.where_learner = get_where_learner(where_learner)
        self.when_learner = get_when_learner(when_learner)
        self.which_learner = get_which_learner(heuristic_learner,
                                               how_cull_rule)
        self.planner = get_planner(planner, search_depth=search_depth,
                                   function_set=function_set,
                                   feature_set=feature_set)
        self.rhs_list = []
        self.rhs_by_label = {}
        self.rhs_by_how = {}
        self.feature_set = feature_set
        self.function_set = function_set
        self.search_depth = search_depth
        self.epsilon = numerical_epsilon
        self.rhs_counter = 0
        self.prev_state = None
        self.prev_explanations = None
        self.prev_reward = None
        self.past_states = []

         # list of past state, explanation pairs

    # -----------------------------REQUEST------------------------------------

    def applicable_explanations(self, state, rhs_list=None,
                                add_skill_info=False
                                ):  # -> returns Iterator<Explanation>
        if(rhs_list is None):
            rhs_list = self.rhs_list


        # categorize the state using trestle
        #s = self.q_learner.T.categorize(flatten_state(state)).concept_id

        s = state_to_key(state)

        if s in self.q_learner.Q:
            actions = [(self.q_learner.Q[s].A[action], val) for action, val in self.q_learner.Q[s].Q.items()]
        else:
            actions = []
        # sort by q value
        actions = sorted(actions, key=lambda x:x[1], reverse=True)
        for action in actions:
            if action[1] > 0:
                yield action[0], None
        '''

        vals = [(action, self.t_learner.evaluate(state, action)) for action in self.t_learner.action_list.values()]
        vals = sorted(vals, key=lambda x:x[1], reverse=True)
        for item in vals:
            if item[1] > 0:
                yield item[0], None
        '''

        '''
        for rhs in rhs_list:
            for match in self.where_learner.get_matches(rhs, state):
                if(len(match) != len(set(match))):
                    continue

                if(self.when_learner.state_format == "variablized_state"):
                    pred_state = variablize_by_where(
                                     state.get_view("flat_ungrounded"), match)
                else:
                    pred_state = state

                if(self.when_learner.predict(rhs, pred_state) <= 0):
                    continue

                mapping = {v: m for v, m in zip(rhs.all_vars, match)}
                explanation = Explanation(rhs, mapping)

                if(add_skill_info):
                    when_info = self.when_learner.skill_info(rhs, pred_state)
                    where_info = [x.replace("?ele-", "") for x in match]
                    skill_info = {"when": tuple(when_info),
                                  "where": tuple(where_info),
                                  "how": str(rhs.input_rule),
                                  "which": 0.0}
                else:
                    skill_info = None

                yield explanation, skill_info
        '''
    def request(self, state, add_skill_info=False):  # -> Returns sai
        #state = StateMultiView("object", state)
        state_featurized = self.planner.apply_featureset(StateMultiView("object", state))
        state_featurized.compute_from('key_vals_grounded','flat_ungrounded')
        # rhs_list = self.which_learner.sort_by_heuristic(self.rhs_list, state)

        '''
        explanations = self.applicable_explanations(
                            state_featurized, rhs_list=self.rhs_list,
                            add_skill_info=add_skill_info)

        '''
        explanations = self.applicable_explanations(
                            state_featurized, rhs_list=self.rhs_list,
                            add_skill_info=add_skill_info)



        explanation, skill_info = next(iter(explanations), (None, None))
        #pprint(str(explanation))

        if(explanation is not None):
            response = explanation.to_response(state_featurized, self)
            if(add_skill_info):
                response["skill_info"] = skill_info

        else:
            response = EMPTY_RESPONSE

        #pprint(response)
        return response

    # ------------------------------TRAIN----------------------------------------

    def where_matches(self, explanations, state):  # -> list<Explanation>, list<Explanation>
        matching_explanations, nonmatching_explanations = [], []
        for exp in explanations:
            # use different?
            if(self.where_learner.check_match(
                    exp.rhs, list(exp.mapping.values()), state)):
                matching_explanations.append(exp)
            else:
                nonmatching_explanations.append(exp)
        return matching_explanations, nonmatching_explanations

    def _matches_from_foas(self, rhs, sai, foci_of_attention):
        iter_func = itertools.permutations
        for combo in iter_func(foci_of_attention):
            d = {k: v for k, v in zip(rhs.input_vars, combo)}
            d[rhs.selection_var] = sai.selection
            yield d

    def explanations_from_skills(self, state, sai, rhs_list,
                                 foci_of_attention=None):  # -> return Iterator<skill>
        for rhs in rhs_list:
            if(isinstance(rhs.input_rule, (int, float, str))):
                # TODO: Hard attr assumption fix this.
                if(rhs.input_rule in sai.inputs.values()):
                    itr = [(rhs.input_rule, {})]
                else:
                    itr = []
            else:
                itr = self.planner.how_search(state, sai,
                                              operators=[rhs.input_rule],
                                              foci_of_attention=foci_of_attention,
                                              search_depth=1,
                                              allow_bottomout=False,
                                              allow_copy=False)

            for input_rule, mapping in itr:
                m = {"?sel": "?ele-" + sai.selection}
                m.update(mapping)
                yield Explanation(rhs, m)

    def explanations_from_how_search(self, state, sai, foci_of_attention):  # -> return Iterator<Explanation>
        sel_match = next(expression_matches(
                         {('?sel_attr', '?sel'): sai.selection}, state), None)

        if(sel_match is not None):
            selection_rule = (sel_match['?sel_attr'], '?sel')
        else:
            selection_rule = sai.selection

        itr = self.planner.how_search(state, sai,
                                      foci_of_attention=foci_of_attention)
        for input_rule, mapping in itr:
            inp_vars = list(mapping.keys())
            varz = list(mapping.values())

            rhs = RHS(selection_expr=selection_rule, action=sai.action,
                      input_rule=input_rule, selection_var="?sel",
                      input_vars=inp_vars, input_attrs=list(sai.inputs.keys()))
            if sel_match is not None:
                literals = [sel_match['?sel']] + varz
            else:
                literals = [sai.selection] + varz
            ordered_mapping = {k: v for k, v in zip(rhs.all_vars, literals)}
            yield Explanation(rhs, ordered_mapping)

    def add_rhs(self, rhs, skill_label="DEFAULT_SKILL"):  # -> return None
        rhs._id_num = self.rhs_counter
        self.rhs_counter += 1
        self.rhs_list.append(rhs)
        self.rhs_by_label[skill_label] = rhs

        constraints = generate_html_tutor_constraints(rhs)
        self.where_learner.add_rhs(rhs, constraints)
        self.when_learner.add_rhs(rhs)
        self.which_learner.add_rhs(rhs)


    def fit(self, explanations, state, reward, next_state):  # -> return None
        for exp in explanations:
            #fit_state = variablize_by_where(state.get_view('flat_ungrounded'), exp.mapping.values())
            #next_fit_state = variablize_by_where(next_state.get_view('flat_ungrounded'), exp.mapping.values())
            #print('here:')
            #pprint(fit_state)
            #print(next_fit_state)
            #fit_state = self.planner.apply_featureset(StateMultiView('object', fit_state))
            #next_fit_state = self.planner.apply_featureset(StateMultiView('object', next_fit_state))

            '''
            if(self.when_learner.state_format == 'variablized_state'):
                fit_state = variablize_by_where(
                            state.get_view("flat_ungrounded"),
                            exp.mapping.values())

                self.when_learner.ifit(exp.rhs, fit_state, reward)
            else:
                self.when_learner.ifit(exp.rhs, state, reward)
            self.which_learner.ifit(exp.rhs, state, reward)
            self.where_learner.ifit(exp.rhs,
                                    list(exp.mapping.values()),
                                    state, reward)
            '''
            self.q_learner.update(state, exp, reward, next_state)
            #self.t_learner.update(state, exp, reward, next_state)



    def train(self, state, selection, action, inputs, reward,
              skill_label, foci_of_attention):  # -> return None

        #if self.prev_state is not None:
        #    pprint(self.prev_state.views)

        sai = SAIS(selection, action, inputs)
        #pprint(state.views)
        state_featurized = self.planner.apply_featureset(StateMultiView("object", state))
        state_featurized.compute_from('key_vals_grounded','flat_ungrounded')

        #next_state = StateMultiView("object", next_state)
        #pprint(state_featurized.views)

        #next_state_featurized = self.planner.apply_featureset(next_state)
        explanations = list(self.explanations_from_skills(state_featurized, sai,
                                                     self.rhs_list,
                                                     foci_of_attention))
        '''
        if (len(explanations) == 0):
            explanations = self.explanations_from_how_search(
                               state_featurized, sai, foci_of_attention)
        '''
            #print([str(next(explanations, None)) for i in range(10)])

        #explanations, nonmatching_explanations = self.where_matches(
        #                                         explanations,
        #                                         state_featurized)

        if(len(explanations) == 0):

            explanations = self.explanations_from_how_search(
                            state_featurized, sai, foci_of_attention)

            rhs_by_how = self.rhs_by_how.get(skill_label, {})
            for exp in explanations:
                if(exp.rhs.as_tuple in rhs_by_how):
                    exp.rhs = rhs_by_how[exp.rhs.as_tuple]
                else:
                    rhs_by_how[exp.rhs.as_tuple] = exp.rhs
                    self.rhs_by_how[skill_label] = rhs_by_how
                    self.add_rhs(exp.rhs)

        explanations = list(explanations)
        #pprint([str(exp.rhs.input_rule) for exp in explanations])

        # VERSION FOR INCREMENTAL UPDATES

        # ***** Note for using trestle: state vs state_featurized, should/can we apply features and then use trestle? *****
        if self.prev_state is not None:
            #self.fit(self.prev_explanations, self.prev_state, self.prev_reward, state)
            self.fit(self.prev_explanations, self.prev_state, self.prev_reward, state_featurized)

        #self.prev_state = state
        self.prev_state = state_featurized
        self.prev_reward = reward
        self.prev_explanations = explanations

        # TODO hardcoded
        if selection == 'done':
            #self.fit(explanations, state, reward, {'a':'a'})
            self.prev_state = None
            self.fit(explanations, state_featurized, reward, StateMultiView('key_vals_grounded', {}))

            #visualize(self.q_learner.T)

            #for key, val in self.q_learner.Q.items():
            #    pprint(key)
            #    pprint(val)

        '''

        # VERSION FOR DELAYED FEEDBACK UPDATE
        self.past_states.append((state, explanations, reward))

        if selection == 'done':
            # update policy in this case
            self.fit(explanations, state, reward, {'a':'a'})
            curr, _, _ = self.past_states.pop()
            while self.past_states != []:
                s, exp, r = self.past_states.pop()
                self.fit(exp, s, r, curr)
                curr = s
        '''


    # ------------------------------CHECK--------------------------------------

    def check(self, state, sai):
        state_featurized, knowledge_base = self.planner.apply_featureset(state)
        explanations = self.explanations_from_skills(state, sai, self.rhs_list)
        explanations, _ = self.where_matches(explanations)
        return len(explanations) > 0

    def get_skills(self, states=None):
        out = []
        for state in states:
            req = self.request(state,
                               add_skill_info=True).get('skill_info', None)

            if(req is not None):
                out.append(frozenset([(k, v) for k, v in req.items()]))

        uniq_lst = list(dict.fromkeys(out).keys())
        unique = [{k: v for k, v in x} for x in uniq_lst]  # set(out)]
        return unique


# ---------------------------CLASS DEFINITIONS---------------------------------

def ground(arg):
    """
    Doc String
    """
    if isinstance(arg, tuple):
        return tuple(ground(e) for e in arg)
    elif isinstance(arg, str):
        return arg.replace('?', 'QM')
    else:
        return arg


def unground(arg):
    """
    Doc String
    """
    if isinstance(arg, tuple):
        return tuple(unground(e) for e in arg)
    elif isinstance(arg, str):
        return arg.replace('QM', '?')
    else:
        return arg


def flatten_state(state):
    tup = Tuplizer()
    flt = Flattener()
    state = flt.transform(tup.transform(state))
    return state


def grounded_key_vals_state(state):
    return [(ground(a), state[a].replace('?', 'QM')
            if isinstance(state[a], str)
            else state[a])
            for a in state]


def kb_to_flat_ungrounded(knowledge_base):
    state = {unground(a): v.replace("QM", "?")
             if isinstance(v, str)
             else v
             for a, v in knowledge_base.facts}
    return state


class StateMultiView(object):
    def __init__(self, view, state):
        self.views = {}
        self.set_view(view, state)
        self.transform_dict = {}
        self.register_transform("object", "flat_ungrounded", flatten_state)
        self.register_transform("flat_ungrounded", "key_vals_grounded",
                                grounded_key_vals_state)
        self.register_transform("feat_knowledge_base", "flat_ungrounded",
                                kb_to_flat_ungrounded)

    def set_view(self, view, state):
        self.views[view] = state

    def get_view(self, view):
        out = self.views.get(view, None)
        if(out is None):
            return self.compute(view)
        else:
            return out

    def contains_view(self, view):
        return view in self.views

    def compute(self, view):
        for key in self.transform_dict[view]:
            # for key in transforms:
            #print(key)
            if(key in self.views):
                out = self.transform_dict[view][key](self.views[key])
                self.set_view(view, out)
                return out
        pprint(self.transform_dict)
        raise Exception("No transform possible from %s to %r" %
                        (list(self.views.keys()), view))

    def compute_from(self, to, frm):
        assert to in self.transform_dict
        assert frm in self.transform_dict[to]
        out = self.transform_dict[to][frm](self.views[frm])
        self.set_view(to, out)
        return out

    def register_transform(self, frm, to, function):
        transforms = self.transform_dict.get(to, {})
        transforms[frm] = function
        self.transform_dict[to] = transforms


class SAIS(object):
    def __init__(self, selection, action, inputs, state=None):
        self.selection = selection
        self.action = action
        self.inputs = inputs
        self.state = state

    def __repr__(self):
        return "S:%r, A:%r, I:%r" % (self.selection, self.action, self.inputs)


class RHS(object):
    def __init__(self, selection_expr, action, input_rule, selection_var,
                 input_vars, input_attrs, conditions=[], label=None):
        self.selection_expr = selection_expr
        self.action = action
        self.input_rule = input_rule
        self.selection_var = selection_var
        self.input_vars = input_vars
        self.input_attrs = input_attrs
        self.all_vars = tuple([self.selection_var] + self.input_vars)
        self.as_tuple = (self.selection_expr, self.action, self.input_rule)

        self.conditions = conditions
        self.label = label
        self._how_depth = None
        self._id_num = None

        self.where = None
        self.when = None
        self.which = None

    def to_xml(self, agent=None):  # -> needs some way of representing itself including its when/where/how parts
        raise NotImplementedError()

    def get_how_depth(self):
        if(self._how_depth == None):
            self._how_depth = compute_exp_depth(self.input_rule)
        return self._how_depth

    def __hash__(self):
        return self._id_num

    def __eq__(self, other):
        a = self._id_num == other._id_num
        b = self._id_num is not None
        c = other._id_num is not None
        return a and b and c


class Explanation(object):
    def __init__(self, rhs, mapping):
        assert isinstance(mapping, dict), \
               "Mapping must be type dict got type %r" % type(mapping)
        self.rhs = rhs
        self.mapping = mapping
        self.selection_literal = mapping[rhs.selection_var]
        self.input_literals = [mapping[s] for s in rhs.input_vars]

    def compute(self, state, agent):
        v = agent.planner.eval_expression([self.rhs.input_rule],
                                          self.mapping, state)[0]

        return {self.rhs.input_attrs[0]: v}

    def conditions_apply(self):
        return True

    def to_response(self, state, agent):
        response = {}
        response['skill_label'] = self.rhs.label
        response['selection'] = self.selection_literal.replace("?ele-", "")
        response['action'] = self.rhs.action
        response['inputs'] = self.compute(state, agent)
        return response

    def to_xml(self, agent=None):  # -> needs some way of representing itself including its when/where/how parts
        pass

    def get_how_depth(self):
        return self.rhs.get_how_depth()

    def __str__(self):
        r = str(self.rhs.input_rule)
        args = ",".join([x.replace("?ele-", "")
                        for x in self.input_literals])
        sel = self.selection_literal.replace("?ele-", "")
        return r + ":(" + args + ")->" + sel

    #def __hash__(self):
    #    return str(self)


def generate_html_tutor_constraints(rhs):
    """
    Given an skill, this finds a set of constraints for the SAI, so it don't
    fire in nonsensical situations.
    """
    constraints = set()

    # get action
    if rhs.action == "ButtonPressed":
        constraints.add(('id', rhs.selection_var, 'done'))
    else:
        constraints.add(('contentEditable', rhs.selection_var, True))

    # value constraints, don't select empty values
    for i, arg in enumerate(rhs.input_vars):
        constraints.add(('value', arg, '?arg%ival' % (i+1)))
        constraints.add((is_not_empty_string, '?arg%ival' % (i+1)))

    return frozenset(constraints)


def is_not_empty_string(sting):
    return sting != ''
