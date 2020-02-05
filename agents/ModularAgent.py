from pprint import pprint
# from random import random
from random import choice

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
from concept_formation.structure_mapper import rename_flat


from agents.BaseAgent import BaseAgent
from learners.WhenLearner import get_when_learner
from learners.WhereLearner import get_where_learner
from learners.WhichLearner import get_which_learner
from planners.base_planner import get_planner_class
from planners.VectorizedPlanner import VectorizedPlanner
from types import MethodType

# from learners.HowLearner import get_planner
# from planners.fo_planner import FoPlanner, execute_functions, unify, subst
import itertools
import json

import cProfile
 
# pr = cProfile.Profile()
# pr.enable()

import atexit

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
    #TODO: Do this better...
    r_state = {key:val for key,val in r_state.items() if "contentEditable" in key or "value" in key}

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
        if(ele is None or ele is "" or
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
        if(expr_comparitor(fact_expr, expression, mapping)):
            yield mapping


EMPTY_RESPONSE = {}

STATE_VARIABLIZATIONS = {"whereswap": variablize_by_where_swap,
                         "relative" : variablize_state_relative,
                         "metaskill" : variablize_state_metaskill}


class ModularAgent(BaseAgent):

    def __init__(self, feature_set, function_set,
                 when_learner='decisiontree', where_learner='version_space',
                 heuristic_learner='proportion_correct', how_cull_rule='all',
                 planner='fo_planner', state_variablization="metaskill", search_depth=1,
                numerical_epsilon=0.0, ret_train_expl=True, **kwargs):
                
                
        # print(planner)
        self.where_learner = get_where_learner(where_learner,
                                            **kwargs.get("where_args",{}))
        self.when_learner = get_when_learner(when_learner,
                                            **kwargs.get("when_args",{}))
        self.which_learner = get_which_learner(heuristic_learner,
                                               how_cull_rule, **kwargs.get("which_args",{}))

        planner_class = get_planner_class(planner)
        self.feature_set = planner_class.resolve_operators(feature_set)
        self.function_set = planner_class.resolve_operators(function_set)
        self.planner = planner_class(search_depth=search_depth,
                                   function_set=self.function_set,
                                   feature_set=self.feature_set,
                                   **kwargs.get("planner_args",{}))

        sv = STATE_VARIABLIZATIONS[state_variablization.lower().replace("_","")]
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

    # -----------------------------REQUEST------------------------------------

    def all_where_parts(self,state, rhs_list=None):
        if(rhs_list is None):
            rhs_list = self.rhs_list

        for rhs in rhs_list:
            # print(rhs.input_rule)
            for match in self.where_learner.get_matches(rhs, state):
                if(len(match) != len(set(match))):
                    continue
                yield rhs,match 




    def applicable_explanations(self, state, rhs_list=None,
                                add_skill_info=False,
                                skip_when = False,
                                ):  # -> returns Iterator<Explanation>
        # print(state.get_view("object"))
        # if(rhs_list is None):
        #     rhs_list = self.rhs_list

        # for rhs in rhs_list:
        #     # print(rhs.input_rule)
        #     for match in self.where_learner.get_matches(rhs, state):
        #         if(len(match) != len(set(match))):
        #             continue
        for rhs,match in self.all_where_parts(state,rhs_list):
            if(self.when_learner.state_format == "variablized_state"):
                pred_state = state.get_view(("variablize", rhs, tuple(match)))
                # print("pred_state")
                # print(rhs,match)
                # print(pred_state)
                # print(state.contains_view(("variablize", rhs, tuple(match))))
                # print(list(state.views.keys()))
                # print()
            else:
                pred_state = state



            # print("--------------")
            # print(str(rhs),"--->",self.when_learner.predict(rhs, pred_state))
            # pprint([int(x) for x in pred_state.values()])
            # print("--------------")
            
            if(not skip_when):
                p = self.when_learner.predict(rhs, pred_state)
                
                if(p <= 0):
                    continue

            mapping = {v: m for v, m in zip(rhs.all_vars, match)}
            explanation = Explanation(rhs, mapping)

            if(add_skill_info):
                skill_info = explanation.get_skill_info(self,pred_state)
                # when_info = self.when_learner.skill_info(rhs, pred_state)
                # where_info = [x.replace("?ele-", "") for x in match]
                # skill_info = {"when": tuple(when_info),
                #               "where": tuple(where_info),
                #               "how": str(rhs.input_rule),
                #               "which": 0.0}
            else:
                skill_info = None

            yield explanation, skill_info

    def request(self, state, add_skill_info=False,n=1):  # -> Returns sai
        if(not isinstance(state,StateMultiView)):
            state = StateMultiView("object", state) 
        state.register_transform("*","variablize",self.state_variablizer)
        state = self.planner.apply_featureset(state)
        rhs_list = self.which_learner.sort_by_heuristic(self.rhs_list, state)

        explanations = self.applicable_explanations(
                            state, rhs_list=rhs_list,
                            add_skill_info=add_skill_info)

        responses = []
        itr = itertools.islice(explanations, n) if n > 0 else iter(explanations)
        for explanation,skill_info in itr:
            if(explanation is not None):
                # print(explanation.rhs.input_rule)
                response = explanation.to_response(state, self)
                if(add_skill_info):
                    # print("skill_info",skill_info)
                    response.update(skill_info)
                    response["mapping"] = explanation.mapping
                responses.append(response)


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
            # assert 
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

        if(self.where_learner.get_strategy() == "first_order"):
            constraints = gen_html_constraints_fo(rhs)
        else:
            constraints = gen_html_constraints_functional(rhs)

        self.where_learner.add_rhs(rhs, constraints)
        self.when_learner.add_rhs(rhs)
        self.which_learner.add_rhs(rhs)

    def fit(self, explanations, state, reward):  # -> return None
        if(not isinstance(reward,list)): reward = [reward]*len(explanations)
        # print("LEN!!! ",len(explanations),reward)
        # print("^^^^^^^^^^^^^^")
        # print(explanations,reward)
        # print("^^^^^^^^^^^^^^")
        for exp,_reward in zip(explanations,reward):
            mapping = list(exp.mapping.values())
            # if(self.when_learner.state_format == 'variablized_state'):
                # fit_state = self.state_variablizer(
                #             state,
                #             exp.mapping.values())
                

                # print("--------------")
                # print(exp.rhs,"<----",reward)
                # pprint([int(x) for x in fit_state.values()])
                # print("--------------")
            # print(str(exp.rhs))
            self.when_learner.ifit(exp.rhs, state, mapping, _reward)
            # else:
                #TODO: Missing
                # self.when_learner.ifit(exp.rhs, state, _reward)

            # print(_reward)
            self.which_learner.ifit(exp.rhs, state, _reward)
            # print(list(exp.mapping.values()))
            self.where_learner.ifit(exp.rhs, mapping, state, _reward)

    # def train_explicit(self,state,explanations, rewards,add_skill_info=False):
    #     print("TRAIN EXPLICIT",explanations)
    #     state = StateMultiView("object", state)
    #     state.register_transform("*","variablize",self.state_variablizer)
    #     state_featurized = self.planner.apply_featureset(state)

    #     ###########ONLY NECESSARY FOR IMPLICIT NEGATIVES#############
    #     _ = [x for x in self.applicable_explanations(state_featurized)]
    #     ############################################################


    #     expl_objs = []
    #     for expl in explanations:
    #         expl_objs.append(Explanation(self.rhs_list[expl["rhs_id"]],expl["mapping"]))
        
    #     self.fit(expl_objs, state_featurized, rewards)
        

    def train(self, state, selection=None, action=None, inputs=None, reward=None,
              skill_label=None, foci_of_attention=None, rhs_id=None, mapping=None,
              ret_train_expl=False, add_skill_info=False,**kwargs):  # -> return None
        # print("TRAIN ",inputs)
        state = StateMultiView("object", state)
        state.register_transform("*","variablize",self.state_variablizer)
        state_featurized = self.planner.apply_featureset(state)


        ###########ONLY NECESSARY FOR IMPLICIT NEGATIVES#############
        _ = [x for x in self.applicable_explanations(state_featurized)]
        ############################################################

        #Either the explanation (i.e. prev application of skill) is provided
        #   or we must infer it from the skills that would have fired
        if(rhs_id is not None and mapping is not None):
            print("Reward: ", reward)
            explanations = [Explanation(self.rhs_list[rhs_id], mapping)]
            print("EX: ",str(explanations[0]))
        elif(selection is not None and action is not None and inputs is not None):
            sai = SAIS(selection, action, inputs)
            explanations = self.explanations_from_skills(state_featurized, sai,
                                                         self.rhs_list,
                                                         foci_of_attention)

            explanations, nonmatching_explanations = self.where_matches(
                                                 explanations,
                                                 state_featurized)

            #If no skills match then we can try partial matches. If there are
            #   none then we must induce a new skill.  
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
        else:
            raise ValueError("Call to train missing SAI, or unique identifiers")

        explanations = list(explanations)
        # print("EXPLANS")
        # print("\n".join([str(x) for x in explanations]))
        # print("^^^^^^^^^^^")
        self.fit(explanations, state_featurized, reward)
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
        # pprint("OUT!!!!")
        # pprint(resp)
        # pprint("DONE!!!!")
        print("----CHECK----")
        print(selection, action, inputs)
        print("-------------")
        if("responses" in resp):
            responses = resp['responses']
            # ok = False
            for resp in responses:
                print(resp['selection'],resp['action'],resp['inputs'], _inputs_equal(resp['inputs'],inputs))
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
        # from datadiff import diff
        # print(diff(dict(self),dict(x)))
        # return super(FlatState,self).__eq__(x)
        # keys1 = set(self.keys())
        # keys2 = set(self.keys())
        # inter = keys1.intersection(set(x.keys()))
        # # if(len(inter) != 
        # for i,k in enumerate(self.keys()):
        #     if(i >= self.core_len):
        #         break
        #     if(k not in x or x[k] != self[k]):
        #         return False
        # return True
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



class StateMultiView(object):
    def __init__(self, view, state):
        self.views = {"*":self}
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
        if(isinstance(view,tuple)):
            view_key = view[0]
            view_args = view[1:]
        else:
            view_key = view
            view_args = []

        for key in self.transform_dict[view_key]:
            # print(self.views,view_key,view,key)
            # print(self.transform_dict[view_key])
            # print(list(self.views.keys()))
            # print(self.views[view_key])
            # for key in transforms:
            # print(view_args)
            # print(self.views[key])
            # print(key)
            if(key in self.views):
                out = self.transform_dict[view_key][key](self.views[key],*view_args)
                self.set_view(view, out)
                return out
        # pprint(self.transform_dict)
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
    def __str__(self):
        return str(self.input_rule)
    def __repr__(self):
        return self.__str__()


class Explanation(object):
    def __init__(self, rhs, mapping):
        assert isinstance(mapping, dict), \
               "Mapping must be type dict got type %r" % type(mapping)
        self.rhs = rhs
        self.mapping = mapping
        self.selection_literal = mapping[rhs.selection_var]
        # print(rhs.input_rule,rhs.input_vars, mapping)
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
        response['rhs_id'] = self.rhs._id_num
        return response

    def get_skill_info(self,agent,when_state=None):
        # print("GET SKILL INFO!!!")
        if(when_state is None):
            when_info = None
        else:    
            when_info = tuple(agent.when_learner.skill_info(self.rhs, when_state))
        # where_match = [x.replace("?ele-", "") for x in self.mapping.values()]
        skill_info = {"when": when_info,
                      "where": agent.where_learner.skill_info(self.rhs),
                      "how": str(self.rhs.input_rule),
                      "which": 0.0,
                      "mapping" : self.mapping}
        return skill_info

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
    def selection_constraints(x):
        if(rhs.action == "ButtonPressed"):
            if(x["id"] != 'done'):
                return False
        else:
            if("contentEditable" not in x or x["contentEditable"] != True):
                return False
        return True

    def arg_constraints(x):
        if("value" not in x or x["value"] == ""):
            return False
        return True

    return selection_constraints, arg_constraints


