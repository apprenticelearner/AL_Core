import logging
# from random import random
import pprint
from random import choice

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
from concept_formation.structure_mapper import rename_flat


from agents.BaseAgent import BaseAgent
from learners.WhenLearner import get_when_learner
from learners.WhereLearner import get_where_learner
from learners.WhichLearner import get_which_learner
from planners.base_planner import get_planner_class
# from planners.VectorizedPlanner import VectorizedPlanner
# from learners.HowLearner import get_planner
# from planners.fo_planner import FoPlanner, execute_functions, unify, subst
import itertools


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
    for fact_expr, value in state.items():
        if(isinstance(expression, dict)):
            fact_expr = {fact_expr: value}

        mapping = {}
        if(expr_comparitor(fact_expr, expression, mapping)):
            yield mapping


EMPTY_RESPONSE = {}


class ModularAgent(BaseAgent):

    def __init__(self, feature_set, function_set,
                 when_learner='decisiontree', where_learner='MostSpecific',
                 heuristic_learner='proportion_correct', how_cull_rule='all',
                 planner='fo_planner', search_depth=1, numerical_epsilon=0.0):
        
        
        self.where_learner = get_where_learner(where_learner)
        self.when_learner = get_when_learner(when_learner)
        self.which_learner = get_which_learner(heuristic_learner,
                                               how_cull_rule)

        
        planner_class = get_planner_class(planner)
        self.feature_set = planner_class.resolve_operators(feature_set)
        self.function_set = planner_class.resolve_operators(function_set)
        self.planner = planner_class(search_depth=search_depth,
                                   function_set=self.function_set,
                                   feature_set=self.feature_set)
       
        self.rhs_list = []
        self.rhs_by_label = {}
        self.rhs_by_how = {}
        
       
       
        self.search_depth = search_depth
        self.epsilon = numerical_epsilon
        self.rhs_counter = 0

    # -----------------------------REQUEST------------------------------------

    def applicable_explanations(self, state, rhs_list=None,
                                add_skill_info=False
                                ):  # -> returns Iterator<Explanation>
        if(rhs_list is None):
            rhs_list = self.rhs_list

        
        
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
                

    def request(self, state, add_skill_info=False):  # -> Returns sai
        
        state = StateMultiView("object", state)
        state = self.planner.apply_featureset(state)
        rhs_list = self.which_learner.sort_by_heuristic(self.rhs_list, state)
        explanations = self.applicable_explanations(
                            state, rhs_list=rhs_list,
                            add_skill_info=add_skill_info)
        
        
        explanation, skill_info = next(iter(explanations), (None, None))

        if(explanation is not None):
            response = explanation.to_response(state, self)
            if(add_skill_info):
                response["skill_info"] = skill_info

        else:
            response = EMPTY_RESPONSE

        return response

    # ------------------------------TRAIN----------------------------------------

    def where_matches(self, explanations, state):  # -> list<Explanation>, list<Explanation>
        matching_explanations, nonmatching_explanations = [], []
        for exp in explanations:
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
                if(sai.inputs["value"] == rhs.input_rule):
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

            literals = [sel_match['?sel']] + varz
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

    def fit(self, explanations, state, reward):  # -> return None
        for exp in explanations:
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

    def train(self, state, selection, action, inputs, reward,
              skill_label, foci_of_attention):  # -> return None
        
        state = StateMultiView("object", state)
        
       
        sai = SAIS(selection, action, inputs)
        state_featurized = self.planner.apply_featureset(state)
        explanations = self.explanations_from_skills(state_featurized, sai,
                                                     self.rhs_list,
                                                     foci_of_attention)

        explanations, nonmatching_explanations = self.where_matches(
                                                 explanations,
                                                 state_featurized)

        if(len(explanations) == 0):

            if(len(nonmatching_explanations) > 0):
                explanations = [choice(nonmatching_explanations)]

            else:
                explanations = self.explanations_from_how_search(
                               state_featurized, sai, foci_of_attention)

                explanations = self.which_learner.cull_how(explanations)

                rhs_by_how = self.rhs_by_how.get(skill_label, {})
                for exp in explanations:
                    if(exp.rhs.as_tuple in rhs_by_how):
                        exp.rhs = rhs_by_how[exp.rhs.as_tuple]
                    else:
                        rhs_by_how[exp.rhs.as_tuple] = exp.rhs
                        self.rhs_by_how[skill_label] = rhs_by_how
                        self.add_rhs(exp.rhs)
        
        
        self.fit(explanations, state_featurized, reward)

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

    def __repr__(self):
        return '\n\n StateMultiView.views: ' + pprint.pformat(self.views) + '\n \n StateMultiView.transform_dict: ' + pprint.pformat(self.transform_dict)
    
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
            #logging.debug(key)
            if(key in self.views):
                out = self.transform_dict[view][key](self.views[key])
                self.set_view(view, out)
                return out
        #logging.debug(self.transform_dict)
        
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

    def __repr__(self):
        return "RHS" + str(self.selection_expr) + str(self.action)

class Explanation(object):
    def __init__(self, rhs, mapping):
        assert isinstance(mapping, dict), \
               "Mapping must be type dict got type %r" % type(mapping)
        self.rhs = rhs
        self.mapping = mapping
        self.selection_literal = mapping[rhs.selection_var]
        self.input_literals = [mapping[s] for s in rhs.input_vars]
        
    def __repr__(self):
        return "Explanation=> RHS: " + str(self.rhs) + " / mapping: " + str(self.mapping) 
        + " / selection_literal: " + str(self.selection_literal) + " / input_literals: " + str(self.input_literals)

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

if __name__=="__main__":
    from planners.fo_planner import Operator
    
    ttt_horizontal_adj = Operator(('horizontal_adj', '?s1', '?s2'),
                              [(('row', '?s1'), '?s1r'),
                               (('row', '?s2'), '?s1r'),
                               (('col', '?s1'), '?s1c'),
                               (('col', '?s2'), '?s2c'),
                               (lambda x, y: abs(x-y) == 1, '?s1c', '?s2c')],
                              [(('horizontal_adj', '?s1', '?s2'), True)])

    ttt_vertical_adj = Operator(('vertical_adj', '?s1', '?s2'),
                                [(('row', '?s1'), '?s1r'),
                                 (('row', '?s2'), '?s2r'),
                                 (('col', '?s1'), '?s1c'),
                                 (('col', '?s2'), '?s1c'),
                                 (lambda x, y: abs(x-y) == 1, '?s1r', '?s2r')],
                                [(('vertical_adj', '?s1', '?s2'), True)])
    
    ttt_diag_adj = Operator(('diag_adj', '?s1', '?s2'),
                            [(('row', '?s1'), '?s1r'),
                             (('row', '?s2'), '?s2r'),
                             (('col', '?s1'), '?s1c'),
                             (('col', '?s2'), '?s2c'),
                             (lambda x, y: abs(x-y) == 1, '?s1r', '?s2r'),
                             (lambda x, y: abs(x-y) == 1, '?s1c', '?s2c')],
                            [(('diag_adj', '?s1', '?s2'), True)])
    
    ttt_move = Operator(('Move', '?r', '?c'),
                        [(('value', '?s'), '?p'),
                         (('id', '?s'), 'CurrentPlayer'),
                         (('id', '?cell'), '?selection'),
                         (('row', '?cell'), '?r'),
                         (('col', '?cell'), '?c'),
                         (('contentEditable', '?cell'), True)],
                        [(('sai', '?selection', 'mark', (('value', '?p'),)),
                          True)])

    a = True
    if a:
        from planners import rulesets #register rulesets 
        feature_set, function_set = ['equals'], ['add', 'subtract', 'multiply', 'divide'] #refer to them by name
        learner = ModularAgent(feature_set, function_set) #pass operators
        
        #print locals() under train 
        t1 = {'foci_of_attention': None, 'skill_label': 'NO_LABEL', 'reward': 1, 'inputs': {'value': -1}, 'action': 'ButtonPressed', 'selection': 'done', 'state': {'?ele-JCommTable.R0C0': {'id': 'JCommTable.R0C0', 'value': '1', 'contentEditable': False}, '?ele-JCommTable.R1C0': {'id': 'JCommTable.R1C0', 'value': '2', 'contentEditable': False}, '?ele-JCommTable3.R0C0': {'id': 'JCommTable3.R0C0', 'value': '2', 'contentEditable': False}, '?ele-JCommTable3.R1C0': {'id': 'JCommTable3.R1C0', 'value': '3', 'contentEditable': False}, '?ele-JCommTable4.R0C0': {'id': 'JCommTable4.R0C0', 'value': '', 'contentEditable': True}, '?ele-JCommTable4.R1C0': {'id': 'JCommTable4.R1C0', 'value': '', 'contentEditable': True}, '?ele-JCommTable5.R0C0': {'id': 'JCommTable5.R0C0', 'value': '', 'contentEditable': True}, '?ele-JCommTable5.R1C0': {'id': 'JCommTable5.R1C0', 'value': '', 'contentEditable': True}, '?ele-JCommTable7.R0C0': {'id': 'JCommTable7.R0C0', 'value': '*', 'contentEditable': False}, '?ele-JCommTable2.R0C0': {'id': 'JCommTable2.R0C0', 'value': '*', 'contentEditable': False}, '?ele-JCommTable6.R0C0': {'id': 'JCommTable6.R0C0', 'value': '2', 'contentEditable': False}, '?ele-JCommTable6.R1C0': {'id': 'JCommTable6.R1C0', 'value': '6', 'contentEditable': False}, '?ele-ctatdiv68': {'id': 'ctatdiv68'}, '?ele-ctatdiv74': {'id': 'ctatdiv74'}, '?ele-done': {'id': 'done'}, '?ele-hint': {'id': 'hint'}, '?ele-ctatdiv87': {'id': 'ctatdiv87'}, '?ele-ctatdiv69': {'id': 'ctatdiv69'}, '?ele-JCommTable8.R0C0': {'id': 'JCommTable8.R0C0', 'value': '', 'contentEditable': True}}}
        x = learner.train(**t1)
    print ("============")
    if a:
        learner = ModularAgent([ttt_horizontal_adj,
                         ttt_vertical_adj,
                         # ttt_available
                         ttt_diag_adj
                         ], [ttt_move])
        t1 = {'foci_of_attention': [], 'skill_label': 'mark', 'reward': True, 'inputs': {'value': 'X'}, 'action': 'mark', 'selection': 'Cell-0-0', 'state': {'?ele-Cell-0-0': {'value': '', 'row': 0, 'col': 0, 'id': 'Cell-0-0'}, '?ele-Cell-0-1': {'value': 'Col 1', 'row': 0, 'col': 1, 'id': 'Cell-0-1'}, '?ele-Cell-0-2': {'value': 'Col 2', 'row': 0, 'col': 2, 'id': 'Cell-0-2'}, '?ele-Cell-0-3': {'value': 'Col 3', 'row': 0, 'col': 3, 'id': 'Cell-0-3'}, '?ele-Cell-1-0': {'value': 'Row 1', 'row': 1, 'col': 0, 'id': 'Cell-1-0'}, '?ele-Cell-1-1': {'value': '', 'row': 1, 'col': 1, 'id': 'Cell-1-1', 'contentEditable': True}, '?ele-Cell-1-2': {'value': '', 'row': 1, 'col': 2, 'id': 'Cell-1-2', 'contentEditable': True}, '?ele-Cell-1-3': {'value': '', 'row': 1, 'col': 3, 'id': 'Cell-1-3', 'contentEditable': True}, '?ele-Cell-2-0': {'value': 'Row 2', 'row': 2, 'col': 0, 'id': 'Cell-2-0'}, '?ele-Cell-2-1': {'value': '', 'row': 2, 'col': 1, 'id': 'Cell-2-1', 'contentEditable': True}, '?ele-Cell-2-2': {'value': '', 'row': 2, 'col': 2, 'id': 'Cell-2-2', 'contentEditable': True}, '?ele-Cell-2-3': {'value': '', 'row': 2, 'col': 3, 'id': 'Cell-2-3', 'contentEditable': True}, '?ele-Cell-3-0': {'value': 'Row 3', 'row': 3, 'col': 0, 'id': 'Cell-3-0'}, '?ele-Cell-3-1': {'value': '', 'row': 3, 'col': 1, 'id': 'Cell-3-1', 'contentEditable': True}, '?ele-Cell-3-2': {'value': '', 'row': 3, 'col': 2, 'id': 'Cell-3-2', 'contentEditable': True}, '?ele-Cell-3-3': {'value': '', 'row': 3, 'col': 3, 'id': 'Cell-3-3', 'contentEditable': True}, '?player': {'value': 'X', 'id': 'CurrentPlayer'}}}
        #print locals() under train 
        x = learner.train(**t1)