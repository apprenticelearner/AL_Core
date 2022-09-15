from pprint import pprint
from random import random
from random import choice
from typing import Dict
from apprentice.working_memory.representation.representation import Skill
from cv2 import exp
import numpy as np


from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
from concept_formation.structure_mapper import rename_flat


from apprentice.agents.base import BaseAgent
from apprentice.learners.WhenLearner import get_when_learner
from apprentice.learners.WhereLearner import get_where_learner
from apprentice.learners.WhichLearner import get_which_learner
from apprentice.planners.base_planner import get_planner_class
from apprentice.planners.VectorizedPlanner import VectorizedPlanner
from apprentice.planners.NumbaPlanner import NumbaPlanner
from types import MethodType


from apprentice.working_memory.representation import Sai
from apprentice.working_memory.representation import RHS
from apprentice.working_memory.representation import StateMultiView
from apprentice.working_memory.representation import Explanation

# from learners.HowLearner import get_planner
# from planners.fo_planner import FoPlanner, execute_functions, unify, subst
import itertools
import json
import math

import cProfile

# pr = cProfile.Profile()
# pr.enable()

import atexit
import time
import logging
from datetime import datetime

from os import path

from matplotlib import pyplot as plt
from sklearn import tree

performance_logger = logging.getLogger('al-performance')
agent_logger = logging.getLogger('al-agent')
agent_logger.setLevel("ERROR")
performance_logger.setLevel("ERROR")


# ------------------------UTILITY FUNCTIONS--------------------------------

def add_QMele_to_state(state):
    ''' A function which adds ?ele- to state keys... this is necessary in order to use
        the fo_planner pending its deprecation'''
    obj_names = state.keys()
    out = {}
    for k,v in state.items():
        k = "?ele-" + k if k[0] != "?" else k
        v_new = {}
        for _k,_v in v.items():
            if(_k != "id" and _v in obj_names):
                _v = "?ele-" + _v
            v_new[_k] = _v
        out[k] = v_new
    return out

# def cleanup(*args):
#     print("DUMP STATS")
#     pr.disable()
#     pr.dump_stats("AL_tres_fo.prof")
# atexit.register(cleanup)


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

def safe_cast(val, to_type, default=None):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default


def _inputs_equal(inputsA, inputsB):
        keys1 = set(inputsA.keys())
        keys2 = set(inputsB.keys())
        if(keys1 == keys2):
            ok = True
            for k in keys1:
                eq = inputsA[k] == inputsB[k]
                floatA = safe_cast(inputsA[k],float)
                floatB = safe_cast(inputsB[k],float)
                float_eq = (floatA != None and floatB != None and floatA == floatB)
                if(not (eq or float_eq)):
                    ok = False
            if(ok):
                return True
        return False

def variablize_by_where_swap(self,state,rhs,  match):
    if(isinstance(state, StateMultiView)):
        state = state.get_view("flat_ungrounded")
    # print(state)
    # print(type(state))
    mapping = {'arg' + str(i-1) if i > 0 else 'sel':
               ele for i, ele in enumerate(match)}
    # for i,x in enumerate(state):
    #     print("attr%i"%i,x)
    #     print("val%i"%i,state[x])

    r_state = rename_flat(state, {mapping[a]: a for a in mapping})
    # r_state = state
    #TODO: Do this better...

    # r_state = {key:val for key,val in r_state.items() if "contentEditable" in key or "value" in key}
    if(self.strip_attrs and len(self.strip_attrs) > 0):
        r_state = {key:val for key,val in r_state.items() if key[0] not in self.strip_attrs}

    # for k,v in r_state.items():
    #     print(k,v)
    #     try:
    #         v = float(v)
    #         r_state[k] = v
    #     except Exception as e:
    #         pass
    # pprint("r_state")
    # pprint(r_state)
    return r_state

def variablize_by_where_append(self,state,rhs,match):
    if(isinstance(state, StateMultiView)):
        state = state.get_view("flat_ungrounded")
    # pprint(state)
    r_state = rename_flat(state, {})
    if(len(match)>1):
        r_state[("args",tuple(list(match)[1:]))] = True
    r_state[("sel",match[0])] = True
    if(self.strip_attrs and len(self.strip_attrs) > 0):
        r_state = {key:val for key,val in r_state.items() if key[0] not in self.strip_attrs}

    # del_list = []
    # for k,v in r_state.items():
    #     try:
    #         if(not isinstance(v,bool)):
    #             v = float(v)
    #             del_list.append(k)
    #     except Exception as e:
    #         pass
    # for k in del_list:
    #     del r_state[k]
    # pprint("r_state")
    # pprint([v for k,v in r_state.items() if k[0] =='value'])
    return r_state



def unvariablize_by_where_swap(state, match):
    mapping = {ele: 'arg' + str(i-1) if i > 0 else 'sel'
               for i, ele in enumerate(match)}
    r_state = rename_flat(state, {mapping[a]: a for a in mapping})
    return r_state

dir_map = {"to_left": "l", "to_right": "r", "above": "a", "below":"b", "offsetParent":"p"}
dirs = list(dir_map.keys())

def _relative_rename_recursive(state,center,center_name="sel",mapping=None,dist_map=None):
    if(mapping is None):
        mapping = {center:center_name}
        dist_map = {center:0}
    # print(state)
    center_obj = state[center]

    stack = []
    for d in dirs:
        ele = center_obj.get(d,None)
        # print("ele")
        # print(ele)
        if(ele is None or ele == "" or
          (ele in dist_map and dist_map[ele] <= dist_map[center] + 1) or
           ele not in state):
            continue
        mapping[ele] = center_name + "." + dir_map[d]
        dist_map[ele] = dist_map[center] + 1
        stack.append(ele)
    # pprint(mapping)
    for ele in stack:
        _relative_rename_recursive(state,ele,mapping[ele],mapping,dist_map)

    return mapping

def variablize_state_relative(self,state,rhs, where_match,center_name="sel"):
    if(isinstance(state, StateMultiView)):
        state = state.get_view("object").copy()
    center = list(where_match)[0]
    mapping = _relative_rename_recursive(state,center,center_name=center_name)
    floating_elems = [x for x in state.keys() if x not in mapping and isinstance(x,str)]
    tup_elems = [x for x in state.keys() if x not in mapping and isinstance(x,tuple)]

    for f_ele in floating_elems:
        for d in dirs:
            ele = state[f_ele].get(d,None)
            if(ele is not None and ele in mapping):
                float_name = "float." + dir_map[d] + "==" + mapping[ele]
                if(float_name not in mapping):
                    mapping[f_ele] = float_name
                    break
    floating_elems = [x for x in state.keys() if x not in mapping and isinstance(x,str)]
    assert len(floating_elems) == 0, "Floating elements %s \
           could not be assigned relative to the rest of the state" % \
           floating_elems


    for tup_ele in tup_elems:
        mapping[tup_ele] = tuple([mapping.get(x,x) for x in tup_ele])

    new_state = {}
    for key,vals in state.items():

        if(isinstance(vals,dict)):
            new_vals = {}
            for k,v in vals.items():
                if(k == "contentEditable" or isinstance(key,tuple)):
                    new_vals[k] = mapping.get(v,v)
            new_state[mapping[key]] = new_vals
        else:
            new_state[key] = mapping.get(vals,vals)

    new_state = flatten_state(new_state)
    # StateMultiView.transforms(("object"))


    return new_state

def variablize_state_metaskill(self,state,rhs, where_match):
    # if(isinstance(state, StateMultiView) and second_pass):
    # try:
    #     state = state.get_view("object_skills_appended")

    # except:
        # state_obj = state.get_view("object").copy()
        # print("variablize_state_metaskill", second_pass,where_match)

        # all_expls = self.applicable_explanations(state, add_skill_info=True,second_pass=False,skip_when=True)
        # print("-------START THIS---------")
    to_append = {}
    for rhs, match in self.all_where_parts(state):
        mapping = {v: m for v, m in zip(rhs.all_vars, match)}
        exp = Explanation(rhs,mapping)
        resp = exp.to_response(state,self)
        # pprint(skill_info)
        key = ("skill-%s"%resp["rhs_id"], *mapping.values())
        to_append[key] = resp["inputs"]
        to_append[("skill-%s"%resp["rhs_id"],"count")] = to_append.get(("skill-%s"%resp["rhs_id"],"count"),0) + 1
        to_append[("all-skills","count")] = to_append.get(("all-skills","count"),0) + 1
        # for attr,val in resp["inputs"].items():
        #     key = (attr,("skill-%s"%resp["rhs_id"], *skill_info['mapping'].values()))
        #     # print(key, ":", val)
        #     flat_ungrounded[key] = val
    # print("--------END THIS---------")

    state_obj = {**state.get_view("object"),**to_append}
    # print(state_obj)
    # state.set_view("object_skills_appended",state_obj)
    state = state_obj
    state = variablize_state_relative(self,state,rhs, where_match)
    k_list = list(state.keys())


    l_core = len(state)-len(to_append)
    # pprint({k:state[k] for k in k_list[:l_core]})
    # pprint({k:state[k] for k in k_list[l_core:]})
    state = FlatState({k:state[k] for k in k_list[:l_core]},
                      {k:state[k] for k in k_list[l_core:]})
                # pprint()
    # print(state)

    # pprint("r_state")
    # pprint(r_state)
    return state



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
        # print(fact_expr, expression, mapping)
        if(expr_comparitor(fact_expr, expression, mapping)):
            yield mapping

EMPTY_RESPONSE = {}

STATE_VARIABLIZATIONS = {"whereappend": variablize_by_where_append,
                         "whereswap": variablize_by_where_swap,
                         "relative" : variablize_state_relative,
                         "metaskill" : variablize_state_metaskill}

RETRIEVAL_TYPE_PRATICE = "RETRIEVAL_TYPE_PRATICE"
RETRIEVAL_TYPE_STUDY = "RETRIEVAL_TYPE_STUDY"

class MemoryAgent(BaseAgent):

    def __init__(self, feature_set, function_set,
                 when_learner='decisiontree', where_learner='version_space',
                 heuristic_learner='proportion_correct', explanation_choice='random',
                 planner='fo_planner', state_variablization="whereswap", search_depth=1,
                 numerical_epsilon=0.0, ret_train_expl=True, strip_attrs=[],
                 constraint_set='ctat', use_memory=True,
                 s=1, c=0.277, alpha=0.177, tau=-0.7, beta=4, b_study=4, b_practice=0, agent_name=None,
                 **kwargs):


        self.where_learner = get_where_learner(where_learner,
                                            **kwargs.get("where_args",{}))
        self.when_learner = get_when_learner(when_learner,
                                            # **kwargs.get("when_args",{}))
                                            **kwargs.get("when_args",{ "cross_rhs_inference": "implicit_negatives"}))
        self.which_learner = get_which_learner(heuristic_learner,
                                               explanation_choice, **kwargs.get("which_args",{}))

        planner_class = get_planner_class(planner)

        self.feature_set = planner_class.resolve_operators(feature_set)
        self.function_set = planner_class.resolve_operators(function_set)
        self.planner = planner_class(search_depth=search_depth,
                                   function_set=self.function_set,
                                   feature_set=self.feature_set,
                                   **kwargs.get("planner_args",{}))
        sv = STATE_VARIABLIZATIONS[state_variablization.lower().replace("_","")]
        self.strip_attrs = strip_attrs
        self.state_variablizer = MethodType(sv, self)
        self.rhs_list = []
        self.rhs_by_label = {}
        self.rhs_by_how = {}
        self.feature_set = feature_set
        self.function_set = function_set
        self.search_depth = search_depth
        self.epsilon = numerical_epsilon
        self.rhs_counter = 0
        self.ret_train_expl = ret_train_expl
        self.last_state = None

        self.explanations_list = np.empty(shape=[0, 1])

        self.activations = {}
        self.exp_times = {}
        self.decays = {}
        self.bs = {}
        self.use_memory = use_memory
        self.t = 0

        self.beta = beta # 4
        self.tau = tau # 0.7
        self.c = c # 0.277
        self.alpha = alpha # 0.177
        self.s = s # 1
        self.b_study = b_study
        self.b_practice = b_practice # 0

        self.print_log = kwargs['print_log']
        self.agent_name = agent_name
        self.log = []
        self.fails = [0, 0]

        # self.activation_path = "memory-activations/memory_{}_{}.txt".format(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), self.agent_name)
        # self.decay_path = "memory-activations/decay_{}_{}.txt".format(datetime.now().strftime("%Y-%m-%d-%H:%M:%S"), self.agent_name)

        assert constraint_set in CONSTRAINT_SETS, "constraint_set %s not recognized. Choose from: %s" % (constraint_set,CONSTRAINT_SETS.keys())
        self.constraint_generator = CONSTRAINT_SETS[constraint_set]

        # if self.activation_path and not path.exists(self.activation_path):
        #     with open(self.activation_path, "a") as outfile:
        #         outfile.write("id\tquestion\tskill\ttime\tactivation\n")

        # if self.decay_path and not path.exists(self.decay_path):
        #     with open(self.decay_path, "a") as outfile:
        #         outfile.write("skill\ttime\tdecay\n")

            # with open(self.activation_path[:-4] + "_responses.txt", "a") as outfile:
            #     outfile.write("id\tquestion\tskill\tselection\taction\tinput\ttime\n")

    def update_activation_for_post_test(self, wait_time):
        # for _ in range(self.t, 2 * self.t):
        #     self._update_activation([], RETRIEVAL_TYPE_STUDY)
        #     self.t += 1
        self.t = self.t + (self.t * wait_time)

    def get_activations(self):
        return self.activations

    def get_log(self):
        return self.log

    def log_step(self, problem_name, action):
        data = {
            "problem_name": problem_name,
            "action": action
        }
        return self.log.append(data)

    def _compute_activation_recursive(self, exp, add_t=False):
        bs = self.bs[exp]
        times = self.t - self.exp_times[exp] + (1 if add_t else 0)
        decays = self.decays[exp]
        return self.beta + np.log(np.sum(bs * np.power(times, -decays)))

    def _compute_decay(self, str_exp):
        decay = self.c * math.exp(self.activations[str_exp][-1]) + self.alpha
        return decay

    def _compute_retrieval(self, exp):
        exp = str(exp)
        m = self._compute_activation_recursive(exp)
        v = (1 / (1 + math.exp((self.tau - m) / self.s)))

        # if self.print_log: print(f"{str(exp)}: {v}, {m}")
        return v, m, random() < v

    def _update_activation(self, explanations, retrieval_type):
        for exp in explanations:
            exp = str(exp)
            b = self.b_practice if retrieval_type == RETRIEVAL_TYPE_PRATICE else self.b_study

            if exp not in self.bs:
                self.bs[exp] = np.array([])
            self.bs[exp] = np.append(self.bs[exp], b)

            if exp not in self.activations:
                self.activations[exp] = np.array([-np.inf])

            if exp not in self.exp_times:
                self.exp_times[exp] = np.array([], dtype=np.int)
            self.exp_times[exp] = np.append(self.exp_times[exp], self.t)

            decay = self._compute_decay(exp)
            if exp not in self.decays:
                self.decays[exp] = np.array([])
            self.decays[exp] = np.append(self.decays[exp], decay)

            m = self._compute_activation_recursive(exp, True)
            self.activations[exp] = np.append(self.activations[exp], m)


    # -----------------------------REQUEST------------------------------------

    def all_where_parts(self,state, rhs_list=None):
        if(rhs_list is None):
            rhs_list = self.rhs_list

        for rhs in rhs_list:
            for match in self.where_learner.get_matches(rhs, state):
                # TODO: this doesn't work with concept 4
                # if(len(match) != len(set(match))):
                #     continue
                yield rhs,match


    def applicable_explanations(self, state, rhs_list=None,
                                add_skill_info=False,
                                skip_when = False,
                                ):  # -> returns Iterator<Explanation>
        self.fails = [0, 0]
        print('rhs list')
        print(rhs_list)
        for rhs,match in self.all_where_parts(state,rhs_list):
            print('in')
            print(rhs)
            self.fails[0] += 1
            if(self.when_learner.state_format == "variablized_state"):
                pred_state = state.get_view(("variablize", rhs, tuple(match)))
                # pred_state = state
            else:
                pred_state = state
            # print("MATCH", rhs,match)
            if(not skip_when):
                # print('before predict')
                # pprint(pred_state)
                p = self.when_learner.predict(rhs, pred_state)

                if(p <= 0):
                    self.fails[1] += 1
                    continue

            mapping = {v: m for v, m in zip(rhs.all_vars, match)}
            explanation = Explanation(rhs, mapping)

            if(add_skill_info):
                skill_info = explanation.get_skill_info(self,pred_state)
            else:
                skill_info = None
            print('yield')
            yield explanation, skill_info

    def request(self, state: dict, add_skill_info=False, n=1, problem_info=None, **kwargs):  # -> Returns sai
        if(type(self.planner).__name__ == "FoPlannerModule"): state = add_QMele_to_state(state)

        if(not isinstance(state,StateMultiView)):
            state = StateMultiView("object", state)
        state.register_transform("*","variablize",self.state_variablizer)
        state.set_view("flat_ungrounded", self.planner.apply_featureset(state))
        # pprint(state.get_view("flat_ungrounded"))
        # state = self.planner.apply_featureset(state)
        rhs_list = self.which_learner.sort_by_heuristic(self.rhs_list, state)

        fail_reason = "DEFAULT"

        sw = False
        explanations = self.applicable_explanations(
                            state, rhs_list=rhs_list,
                            # add_skill_info=add_skill_info,
                            add_skill_info=True,
                            skip_when=sw)
        retrieved_explanations = []
        responses = []
        itr = itertools.islice(explanations, n) if n > 0 else iter(explanations)
        selected_v, selected_m = None, None

        applicable_explanations_count = 0
        skill_infos = []
        for explanation, skill_info in itr:
            agent_logger.debug("Skill Application: {} {}".format(explanation,explanation.rhs._id_num))
            if(explanation is not None):
                applicable_explanations_count += 1
                # is there a reason for this?
                if self.use_memory and explanation.selection_literal != "done":
                    if str(explanation) in self.activations:
                        v, m, success = self._compute_retrieval(explanation)
                        if not success:
                            fail_reason = "RETRIEVAL"
                            continue

                        selected_v, selected_m = v, m

                        response = explanation.to_response(state, self)
                        if(add_skill_info):
                            response.update(skill_info)
                            response["mapping"] = explanation.mapping

                        retrieved_explanations.append((v, explanation))
                        responses.append((v, response))
                        skill_infos.append(skill_info)
                else:
                    response = explanation.to_response(state, self)
                    if(add_skill_info):
                        response.update(skill_info)
                        response["mapping"] = explanation.mapping
                    retrieved_explanations.append(explanation)
                    responses.append(response)

        if self.use_memory:
            retrieved_explanations.sort(reverse=True)
            responses.sort(reverse=True)
            retrieved_explanations = [r[1] for r in retrieved_explanations]
            responses = [r[1] for r in responses]

        if(len(responses) == 0):
            response = EMPTY_RESPONSE
        else:
            if retrieved_explanations and retrieved_explanations[0].selection_literal != "done":
                str_exp = str(retrieved_explanations[0])
                self._update_activation([str_exp], RETRIEVAL_TYPE_PRATICE)
            response = responses[0].copy()
            if(n != 1):
                response['responses'] = responses

        info = {}
        selected_skill = str(retrieved_explanations[0]) if len(retrieved_explanations) > 0 else "NO EXP"
        selected_skill_where_part = str(self.where_learner.learners[retrieved_explanations[0].rhs]) if len(retrieved_explanations) > 0 else "N/A"
        # selected_skill_when_part = str(skill_infos[0]['when']) if len(retrieved_explanations) > 0 else "N/A"

        if self.use_memory:
            self.t += 1
            info = {
                # 'applicable_explanations_count': applicable_explanations_count,
                'v': selected_v,
                'm': selected_m,
                'selected_skill': selected_skill,
                'applicable_explanations_count': applicable_explanations_count,
                'where': selected_skill_where_part,
                # 'when': selected_skill_when_part
                'when': "...",
            }

        if response == EMPTY_RESPONSE:
            # info['where'] = info['applicable_explanations_count']
            info['where'] = self.fails
            # for _ in self.all_where_parts(state,rhs_list):
            #     info['where'] += 1


        selected_v =f"{selected_v:.3f}" if selected_v else None
        selected_m =f"{selected_m:.3f}" if selected_m else None
        if self.print_log: print(f"selected_skill: {selected_skill} (v: {selected_v}, m: {selected_m})")
        if self.print_log: print(f"where-part: {selected_skill_where_part}")
        info['skill'] = f"{selected_skill} (v: {selected_v})"
        # self.log_step(problem_info['problem_name'], "request")
        return response, info


    # ------------------------------TRAIN----------------------------------------

    def where_matches(self, explanations, state):  # -> list<Explanation>, list<Explanation>
        matching_explanations, nonmatching_explanations = [], []
        partial_scores = []
        for exp in explanations:
            if(self.where_learner.check_match(
                    exp.rhs, list(exp.mapping.values()), state)):
                matching_explanations.append(exp)
            else:
                partial_scores.append(
                    self.where_learner.score_match(
                    exp.rhs, list(exp.mapping.values()), state)
                )
                nonmatching_explanations.append(exp)
        if(len(nonmatching_explanations) > 0):
            non_m_inds = np.where(partial_scores == np.max(partial_scores))[0]
            nonmatching_explanations = [nonmatching_explanations[i] for i in non_m_inds]
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
                mappings = [{}] if sai.inputs["value"] == rhs.input_rule else []
                # if(sai.inputs["value"] == rhs.input_rule):
                #     itr = [(rhs.input_rule, {})]
                # else:
                #     itr = []
            else:
                # print("Trying:", rhs)
                # print(self.planner.unify_op.__code__.co_varnames)
                mappings = self.planner.unify_op(state,rhs.input_rule, sai,
                    foci_of_attention=foci_of_attention)

                # print( "Worked" if len(mappings) > 0 else "Nope" )
                # itr = self.planner.how_search(state, sai,
                #                               operators=[rhs.input_rule],
                #                               foci_of_attention=foci_of_attention,
                #                               search_depth=1,
                #                               allow_bottomout=False,
                #                               allow_copy=False)
            for mapping in mappings:
                if(type(self.planner).__name__ == "FoPlannerModule"):
                    m = {"?sel": "?ele-" + sai.selection if sai.selection[0] != "?" else sai.selection}
                else:
                    m = {"?sel": sai.selection}
                m.update(mapping)
                if(len(m)==len(set(m.values()))):
                    yield Explanation(rhs, m)

    def explanations_from_solution(self, sai, solution):  # -> return Iterator<Explanation>
        inp_vars = list(solution['mapping'].keys())
        varz = list(solution['mapping'].values())
        rhs = RHS(selection_expr=sai.selection, action=sai.action,
                    input_rule=solution['input_rule'], selection_var="?sel",
                    input_vars=inp_vars, input_attrs=list(sai.inputs.keys()))

        literals = [sai.selection] + varz
        ordered_mapping = {k: v for k, v in zip(rhs.all_vars, literals)}
        yield Explanation(rhs, ordered_mapping)

    def explanations_from_how_search(self, state, sai, foci_of_attention):  # -> return Iterator<Explanation>
        # sel_match = next(expression_matches(
        #                  {('?sel_attr', '?sel'): sai.selection}, state), None)
        # print(sel_match, sai.selection)
        # if(sel_match is not None):
        #     selection_rule = (sel_match['?sel_attr'], '?sel')
        # else:
        # sel_match = {"?sel" : sai.selection}
        # selection_rule = sai.selection
        # print(state)
        # print("@" * 20)
        itr = self.planner.how_search(state, sai,
                                      foci_of_attention=foci_of_attention)
        for input_rule, mapping in itr:
            # print('---')
            inp_vars = list(mapping.keys())
            varz = list(mapping.values())

            # print(mapping)
            # print(type(inp_vars[0]))
            # print(type(varz[0]))
            # print('..')
            # print(input_rule)
            # print(type(input_rule[0]))

            rhs = RHS(selection_expr=sai.selection, action=sai.action,
                      input_rule=input_rule, selection_var="?sel",
                      input_vars=inp_vars, input_attrs=list(sai.inputs.keys()))

            literals = [sai.selection] + varz
            ordered_mapping = {k: v for k, v in zip(rhs.all_vars, literals)}
            yield Explanation(rhs, ordered_mapping)

        # print("DONE" + "@" * 20)

    def add_rhs(self, rhs, skill_label="DEFAULT_SKILL"):  # -> return None
        rhs._id_num = self.rhs_counter
        self.rhs_counter += 1
        self.rhs_list.append(rhs)
        self.rhs_by_label[skill_label] = rhs

        if(self.where_learner.get_strategy() == "first_order"):
            constraints = gen_html_constraints_fo(rhs)
        else:
            constraints = self.constraint_generator(rhs)
            # constraints = gen_stylus_constraints_functional(rhs)

        self.where_learner.add_rhs(rhs, constraints)
        self.when_learner.add_rhs(rhs)
        self.which_learner.add_rhs(rhs)

    def fit(self, explanations, state, reward):  # -> return None
        if(not isinstance(reward,list)): reward = [reward]*len(explanations)
        for exp,_reward in zip(explanations,reward):
            mapping = list(exp.mapping.values())
            # print(exp, mapping, 'rew:', _reward)
            self.when_learner.ifit(exp.rhs, state, mapping, _reward)
            self.which_learner.ifit(exp.rhs, state, _reward)
            self.where_learner.ifit(exp.rhs, mapping, state, _reward)

    def train(self, state:Dict, sai:Sai=None, reward:float=None,
              skill_label=None, foci_of_attention=None, rhs_id=None, mapping=None,
              ret_train_expl=False, add_skill_info=False, problem_info=None, 
              solution=None, **kwargs):  # -> return None

        fstate = None

        # Hacky FOA.
        # print('******' * 10)
        # print(foci_of_attention)
        # if foci_of_attention is not None and len(foci_of_attention) > 0:
        #     fstate = {k: state[k] for k in foci_of_attention}
        #     print('fstate')
        #     print(fstate)
        #     fstate = add_QMele_to_state(fstate)
        #     fstate = StateMultiView("object", fstate)
        #     fstate.register_transform("*","variablize",self.state_variablizer)
        #     fstate.set_view("flat_ungrounded", self.planner.apply_featureset(fstate))

        if(type(self.planner).__name__ == "FoPlannerModule"):
            state = add_QMele_to_state(state)
            sai.selection = "?ele-" + sai.selection if sai.selection[0] != "?" else sai.selection
        state = StateMultiView("object", state)
        state.register_transform("*","variablize",self.state_variablizer)

        state.set_view("flat_ungrounded", self.planner.apply_featureset(state))
        # state_featurized = state.get_view("flat_ungrounded")
        # print(sai, foci_of_attention)

        ###########ONLY NECESSARY FOR IMPLICIT NEGATIVES#############
        _ = [x for x in self.applicable_explanations(state)]
        ############################################################

        #Either the explanation (i.e. prev application of skill) is provided
        #   or we must infer it from the skills that would have fired
        if(rhs_id is not None and mapping is not None):
            explanations = [Explanation(self.rhs_list[rhs_id], mapping)]
        elif(sai is not None):
            t_s = time.time_ns()
            explanations = self.explanations_from_skills(state if fstate is None else fstate, sai,
                                                         self.rhs_list,
                                                         foci_of_attention)
            explanations = list(explanations)
            performance_logger.info("explanations_from_skills {} ms".format((time.time_ns()-t_s)/(1e6)))

            explanations, nonmatching_explanations = self.where_matches(
                                                 explanations,
                                                 state)
            if(len(explanations) == 0):
                if(len(nonmatching_explanations) > 0):
                    explanations = [choice(nonmatching_explanations)]
                else:
                    t_s = time.time_ns()

                    print(solution)
                    if solution != None:
                        print('use solution')
                        explanations = self.explanations_from_solution(sai, solution)
                    else:
                        explanations = self.explanations_from_how_search(
                                        state if fstate is None else fstate, sai, foci_of_attention)
                    performance_logger.info("explanations_from_how_search {} ms".format((time.time_ns()-t_s)/(1e6)))

                    explanations = self.which_learner.select_how(explanations)

                    rhs_by_how = self.rhs_by_how.get(skill_label, {})
                    for exp in explanations:
                        print("FOUND EX:", str(exp))
                        if(exp.rhs.as_tuple in rhs_by_how):
                            exp.rhs = rhs_by_how[exp.rhs.as_tuple]
                        else:
                            rhs_by_how[exp.rhs.as_tuple] = exp.rhs
                            self.rhs_by_how[skill_label] = rhs_by_how
                            self.add_rhs(exp.rhs)
        else:
            raise ValueError("Call to train missing SAI, or unique identifiers")

        explanations = list(explanations)

        # QUESTION: should activation be updated when AL receive correctness feedback?
        # Currently, no since the activation should already be updated in request().
        if self.use_memory and rhs_id is None:
            skip = True # skip activation of done skill
            for exp in explanations:
                str_exp = str(exp)
                if exp.selection_literal != "done":
                    skip = False
                    self.explanations_list = np.append(self.explanations_list, str_exp)

            if len(explanations) > 1:
                print("*** MORE THAN ONE EXP ***")
            if not skip:
                self._update_activation(explanations, RETRIEVAL_TYPE_STUDY)
                self.t += 1

        self.fit(explanations, state, reward)
        # self.log_step(problem_info['problem_name'], "train")
        if(self.ret_train_expl):
            out = []
            for exp in explanations:
                resp = exp.to_response(state,self)
                if(add_skill_info): resp.update(exp.get_skill_info(self))
                out.append(resp)

            selected_skill = str(explanations[0]) if len(explanations) > 0 else "NO EXP"
            selected_skill_where_part = str(self.where_learner.learners[explanations[0].rhs]) if len(explanations) > 0 else "N/A"
            # selected_skill_when_part = str(self.when_learner.skill_info(explanations[0].rhs, state.get_view('object'))) if len(explanations) > 0 else "N/A"
            selected_skill_when_part = str(explanations[0].get_skill_info(self)['when']) if len(explanations) > 0 else "N/A"
            return selected_skill_when_part, selected_skill_where_part, selected_skill
            # return out

    # ------------------------------CHECK--------------------------------------


    def check(self, state, selection, action, inputs):
        resp = self.request(state,n=-1)
        if("responses" in resp):
            responses = resp['responses']
            for resp in responses:
                # print(resp['selection'],resp['action'],resp['inputs'], _inputs_equal(resp['inputs'],inputs))
                if(resp['selection'] == selection and
                   resp['action'] == action and
                   _inputs_equal(resp['inputs'],inputs)):
                    return 1

        # state_featurized, knowledge_base = self.planner.apply_featureset(state)
        # explanations = self.explanations_from_skills(state, sai, self.rhs_list)
        # explanations, _ = self.where_matches(explanations)
        return -1

    def get_skills(self, states=None):
        out = []
        print("GET_SKILLS")
        print(states)
        for state in states:
            req = self.request(state,
                               add_skill_info=True)

            pprint(req)
            if(len(req) != 0):
                req["when"] = json.dumps(req["when"])
                req["where"] = json.dumps(req["where"])#tuple(len(list(req["where"].keys())) * ["?"])
                del req["inputs"]
                del req["mapping"]
                del req["selection"]


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


class FlatState(dict):
    def __init__(self, core_features,secondary_features):
        self.core_len = len(core_features)
        self.core_features = {**core_features}
        super(FlatState,self).__init__({**core_features,**secondary_features})

    def __setitem__(self, x,y):
        raise NotImplementedError("FlatState is not a mutable type. Cannot set key %r to %r" % (x,y))

    def __eq__(self, x):
        return self.core_features == (x.core_features if isinstance(x,FlatState) else x)
    def __str__(self):
        out = ""
        k_list = list(self.keys())
        for i,k in enumerate(k_list[:self.core_len]):
            if(i == 0):
                out += "--core features--\n"
            out += "%s : %s\n" % (k,self[k])
        for i,k in enumerate(k_list[self.core_len:]):
            if(i == 0):
                out += "--secondary features--\n"
            out += "%s : %s\n" % (k,self[k])
        return out


def is_not_empty_string(sting):
    return sting != ''


def gen_html_constraints_fo(rhs):
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

def gen_html_constraints_functional(rhs):
    # print("FUNCTIONAL")
    def selection_constraints(x):
        # print("SELc:", x)
        if(rhs.action == "ButtonPressed"):
            if(x["id"] != 'done'):
                # print("C!")
                return False
        else:
            if("contentEditable" not in x or x["contentEditable"] != True):
            # if("contentEditable" in x and x["contentEditable"] != True):
                # print("A!")
                return False
        return True

    def arg_constraints(x):
        if("value" not in x or x["value"] == ""):
            # print("B!")
            return False
        return True

    return selection_constraints, arg_constraints

def gen_stylus_constraints_functional(rhs):
    # print("FUNCTIONAL")
    def selection_constraints(x):
        # print("SELc:", x)
        if(rhs.action == "ButtonPressed"):
            if(x["id"] != 'done'):
                # print("C!")
                return False
        else:
            # if("contentEditable" not in x or x["contentEditable"] != True):
            if("contentEditable" in x and x["contentEditable"] != True):
                # print("A!")
                return False
        return True

    def arg_constraints(x):
        # print("Xc:", x)
        if("value" not in x or x["value"] == ""):
            # print("B!")
            return False
        return True

    return selection_constraints, arg_constraints


CONSTRAINT_SETS = {"stylus" : gen_stylus_constraints_functional,
                   "ctat" : gen_html_constraints_functional}
