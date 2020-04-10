from pprint import pprint
# from random import random
from random import choice
from typing import Dict


from concept_formation.structure_mapper import rename_flat

from apprentice.agents.base import BaseAgent
from apprentice.learners.WhenLearner import get_when_learner
from apprentice.learners.WhereLearner import get_where_learner
from apprentice.learners.WhichLearner import get_which_learner
from apprentice.planners.base_planner import get_planner_class

from apprentice.working_memory.representation import Sai
from apprentice.working_memory.representation import RHS
from apprentice.working_memory.representation import StateMultiView
from apprentice.working_memory.representation import Explanation
# from planners.VectorizedPlanner import VectorizedPlanner
# from learners.HowLearner import get_planner
# from planners.fo_planner import FoPlanner, execute_functions, unify, subst
import itertools




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
                 planner='fo_planner', search_depth=1, numerical_epsilon=0.0, **kwargs):
        # print(planner)
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
        # print("BLOOP",self.planner.__class__.registered_operators)
        self.rhs_list = []
        self.rhs_by_label = {}
        self.rhs_by_how = {}
        
        # print()
        # print(self.feature_set,self.function_set)
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

    def request(self, state: Dict, add_skill_info=False, **kwargs):  # -> Returns sai
        state = StateMultiView("object", state)
        state = self.planner.apply_featureset(state)
        rhs_list = self.which_learner.sort_by_heuristic(self.rhs_list, state)

        explanations = self.applicable_explanations(
                            state, rhs_list=rhs_list,
                            add_skill_info=add_skill_info)

        response = EMPTY_RESPONSE
        for explanation, skill_info in explanations:
            tmp_resp = explanation.to_response(state, self)
            if tmp_resp['inputs']['value'] is None:
                continue
            response = tmp_resp
            if(add_skill_info):
                response["skill_info"] = skill_info
            break

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

    def train(self, state:Dict , sai: Sai, reward: float,
              skill_label, foci_of_attention, **kwargs):  # -> return None
        state = StateMultiView("object", state)
        # sai = Sai(selection, action, inputs)
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

    def check(self, state, sai, **kwargs):
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
