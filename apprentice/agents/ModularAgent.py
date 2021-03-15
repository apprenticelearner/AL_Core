from pprint import pprint
# from random import random
from random import choice
from typing import Dict
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

import cProfile
 
# pr = cProfile.Profile()
# pr.enable()

import atexit
import time
import logging

performance_logger = logging.getLogger('al-performance')
agent_logger = logging.getLogger('al-agent')
agent_logger.setLevel("ERROR")
performance_logger.setLevel("ERROR")




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


    
    # pprint(mapping)
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
        sk_str = f"{str(self.rhs_list[resp['rhs_id']])}(id:{resp['rhs_id']})"
        key = (sk_str, *mapping.values())
        to_append[key] = resp["inputs"]
        # to_append[(sk_str,"count")] = to_append.get((sk_str,"count"),0) + 1
        # to_append[("all-skills","count")] = to_append.get(("all-skills","count"),0) + 1

    #Also append where_match
    if(len(where_match)>1):
        for i,x in enumerate(list(where_match)[1:]):
            to_append[("arg%d"%i,)] = x
    # to_append[("sel",)] = where_match[0]

    # print(to_append)
        # for attr,val in resp["inputs"].items():
        #     key = (attr,("skill-%s"%resp["rhs_id"], *skill_info['mapping'].values()))
        #     # print(key, ":", val)
        #     flat_ungrounded[key] = val 
    # print("--------END THIS---------")
    
    state_obj = {**state.get_view("object"),**to_append}
    # pprint(state_obj)
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



class ModularAgent(BaseAgent):

    def __init__(self, feature_set, function_set,
                 when_learner='decisiontree', where_learner='version_space',
                 which_learner='weighted_proportion_correct', explanation_choice='least_operations',
                 planner='fo_planner', state_variablization="whereswap", search_depth=1,
                 numerical_epsilon=0.0, ret_train_expl=True, strip_attrs=[],
                 constraint_set='ctat', **kwargs):
                
                
        self.where_learner = get_where_learner(where_learner,
                                            **kwargs.get("where_args",{}))
        self.when_learner = get_when_learner(self, when_learner,
                                            **kwargs.get("when_args",{}))
        self.which_learner = get_which_learner(self, which_learner,
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
        self.remove_low_utility = True
        self.one_pos_which_per_sel = True

        assert constraint_set in CONSTRAINT_SETS, "constraint_set %s not recognized. Choose from: %s" % (constraint_set,CONSTRAINT_SETS.keys())
        self.constraint_generator = CONSTRAINT_SETS[constraint_set]

    # def copy_skill(rhs_id):
    #     self.rhs_by_label[rhs_id] =



    # -----------------------------REQUEST------------------------------------

    def all_where_parts(self,state, rhs_list=None):
        if(rhs_list is None):
            rhs_list = self.rhs_list

        for rhs in rhs_list:
            for match in self.where_learner.get_matches(rhs, state):
                if(len(match) != len(set(match))):
                    continue
                yield rhs,match 




    def applicable_explanations(self, state, rhs_list=None,
                                add_skill_info=False,
                                skip_when = False,
                                ):  # -> returns Iterator<Explanation>
        for rhs,match in self.all_where_parts(state,rhs_list):
            if(self.when_learner.state_format == "variablized_state"):
                pred_state = state.get_view(("variablize", rhs, tuple(match)))
            else:
                pred_state = state
            # print("MATCH", rhs,match)
            if(not skip_when):
                p = self.when_learner.predict(rhs, pred_state)
                
                if(p <= 0):
                    continue

            mapping = {v: m for v, m in zip(rhs.all_vars, match)}
            explanation = Explanation(rhs, mapping)

            if(add_skill_info):
                skill_info = explanation.get_skill_info(self,pred_state)
            else:
                skill_info = None
            yield explanation, skill_info

    def request(self, state: dict, add_skill_info=False,n=1,**kwargs):  # -> Returns sai

        if(type(self.planner).__name__ == "FoPlannerModule"): state = add_QMele_to_state(state)
        if(not isinstance(state,StateMultiView)):
            state = StateMultiView("object", state) 
        state.register_transform("*","variablize",self.state_variablizer)
        state.set_view("flat_ungrounded", self.planner.apply_featureset(state))
        # print("REQUEST")
        # pprint({(k,v['value'],v['contentEditable']) for k,v in state.get_view("object").items() if 'value' in v and 'contentEditable' in v})
        # state = self.planner.apply_featureset(state)
        rhs_list = self.which_learner.sort_by_utility(self.rhs_list, state)

        rhs_list = [x for x in rhs_list if not getattr(x,"is_bad",False)]        


        explanations = self.applicable_explanations(
                            state, rhs_list=rhs_list,
                            add_skill_info=add_skill_info)



        responses = []
        itr = itertools.islice(explanations, n) if n > 0 else iter(explanations)
        for explanation,skill_info in itr:
            print("Skill Application: {} {}".format(explanation,explanation.rhs._id_num))
            agent_logger.debug("Skill Application: {} {}".format(explanation,explanation.rhs._id_num))
            if(explanation is not None):
                ne_garbage = self.which_learner.learners[explanation.rhs].num_incorrect
                rem_ne_garbage = self.which_learner.removal_learners[explanation.rhs].num_incorrect
                pe_garbage = self.which_learner.learners[explanation.rhs].num_correct
                rem_pe_garbage = self.which_learner.removal_learners[explanation.rhs].num_correct
                ut = self.which_learner.get_utility(explanation.rhs,state)
                rem_ut = self.which_learner.get_removal_utility(explanation.rhs,state)
                # print(ut, rem_ut)
                # ut = ut if isinstance(ut,(float,int)) else 0.0
                # rem_ut = rem_ut if isinstance(rem_ut,(float,int)) else 0.0
                print(f"UTILITY: U:({ut[0]:.3f}, {pe_garbage}:{ne_garbage}) R:({rem_ut[0]:.3f}, {rem_pe_garbage}:{rem_ne_garbage})")
                
                response = explanation.to_response(state, self)
                if(add_skill_info):
                    response.update(skill_info)
                    response["mapping"] = explanation.mapping
                responses.append(response)

        # pprint(responses)
        # print("---------------")
        if(len(responses) == 0):
            return EMPTY_RESPONSE
        else:
            response = responses[0].copy()
            if(n != 1):
                response['responses'] = responses
            return response
            

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
            if(len(matching_explanations) == 0):
                print(partial_scores)
                print(non_m_inds)
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
                # print("MAAAP", mapping)
                if(type(self.planner).__name__ == "FoPlannerModule"):
                    m = {"?sel": "?ele-" + sai.selection if sai.selection[0] != "?" else sai.selection}
                else:
                    m = {"?sel": sai.selection}
                m.update(mapping)
                if(len(m)==len(set(m.values()))):
                    yield Explanation(rhs, m)

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
        itr = self.planner.how_search(state, sai,
                                      foci_of_attention=foci_of_attention)
        for input_rule, mapping in itr:
            inp_vars = list(mapping.keys())
            varz = list(mapping.values())

            rhs = RHS(selection_expr=sai.selection, action=sai.action,
                      input_rule=input_rule, selection_var="?sel",
                      input_vars=inp_vars, input_attrs=list(sai.inputs.keys()))

            literals = [sai.selection] + varz
            ordered_mapping = {k: v for k, v in zip(rhs.all_vars, literals)}
            yield Explanation(rhs, ordered_mapping)

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

    def fit(self, explanations, state, rewards):  # -> return None
        pprint(state.get_view("object"))
        if(not isinstance(rewards,list)): rewards = [rewards]*len(explanations)

        used_selections = set()
        for exp,reward in zip(explanations,rewards):
            mapping = list(exp.mapping.values())
            # print(exp, mapping, 'rew:', _reward)
            self.when_learner.ifit(exp.rhs, state, mapping, reward)
            self.where_learner.ifit(exp.rhs, mapping, state, reward)

            #TODO: Reject adding to which if when would no longer match for this state + mapping
            # If the reward is positive only increment which for the top explanation, but
            #   if it is wrong increment neg for all
            if(not self.one_pos_which_per_sel or 
                reward <= 0 or mapping[0] not in used_selections):
                self.which_learner.ifit(exp.rhs, state, mapping, reward)    
                used_selections.add(mapping[0])

            if(self.remove_low_utility):
                rem_utility = self.which_learner.get_removal_utility(exp.rhs,state)#learners[exp.rhs].utility(state)
                exp.rhs.is_bad = rem_utility[0] < .2
                if(exp.rhs.is_bad):
                    print("MARKED BAD:"+str(exp.rhs))
            
            # used_rhss.add(exp.rhs)

        # for rhs in used_rhss:
        #     self.which_learner.ifit(rhs, state, _reward)
        #     new_utility = self.which_learner.learners[rhs].utility(state)
        #     rhs.is_bad = new_utility[0] < .2
                

    def train(self, state:Dict, sai:Sai=None, reward:float=None,
              skill_label=None, foci_of_attention=None, rhs_id=None, mapping=None,
              ret_train_expl=False, add_skill_info=False,**kwargs):  # -> return None
        # pprint(state)


        if(type(self.planner).__name__ == "FoPlannerModule"): 
            state = add_QMele_to_state(state)
            sai.selection = "?ele-" + sai.selection if sai.selection[0] != "?" else sai.selection
        state = StateMultiView("object", state)
        state.register_transform("*","variablize",self.state_variablizer)
        state.set_view("flat_ungrounded", self.planner.apply_featureset(state))
        # state_featurized = state.get_view("flat_ungrounded")
        # state_featurized =

        
        # print(sai, foci_of_attention)
        ###########ONLY NECESSARY FOR IMPLICIT NEGATIVES#############
        _ = [x for x in self.applicable_explanations(state)]
        ############################################################

        #Either the explanation (i.e. prev application of skill) is provided
        #   or we must infer it from the skills that would have fired
        if(rhs_id is not None and mapping is not None):
            # print("Reward: ", reward)
            explanations = [Explanation(self.rhs_list[rhs_id], mapping)]
            # print("EX: ",str(explanations[0]))
        elif(sai is not None):
            # pprint(state.get_view("object"))
            # print("TO HOW")
            t_s = time.time_ns()
            explanations = self.explanations_from_skills(state, sai,
                                                         self.rhs_list,
                                                         foci_of_attention)

            explanations = list([x for x in explanations if not getattr(x.rhs,"is_bad",False)])

            performance_logger.info("explanations_from_skills {} ms".format((time.time_ns()-t_s)/(1e6)))

            explanations, nonmatching_explanations = self.where_matches(
                                                 explanations,
                                                 state)
            

            if(len(explanations) == 0):

                if(len(nonmatching_explanations) > 0):
                    explanations = [choice(nonmatching_explanations)]

                else:
                    # print(state_featurized)
                    t_s = time.time_ns()
                    explanations = self.explanations_from_how_search(
                                   state, sai, foci_of_attention)
                    performance_logger.info("explanations_from_how_search {} ms".format((time.time_ns()-t_s)/(1e6)))
                    explanations = [x for x in explanations]
                    for exp in explanations:
                        print(str(exp))

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
        for exp in explanations:
            print("Updating skill ->:", str(exp))
        # print("FIT_A")
        self.fit(explanations, state, reward)
        if(self.ret_train_expl):
            out = []
            for exp in explanations:
                resp = exp.to_response(state,self)
                if(add_skill_info): resp.update(exp.get_skill_info(self))
                out.append(resp)

            return out

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
