from pprint import pprint
from random import random
from random import choice

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
from concept_formation.structure_mapper import rename_flat


from agents.BaseAgent import BaseAgent
from learners.WhenLearner import get_when_learner
from learners.WhereLearner import get_where_learner
from learners.WhichLearner import get_which_learner
from planners.base_planner import get_planner
from planners.VectorizedPlanner import VectorizedPlanner
# from learners.HowLearner import get_planner
from planners.fo_planner import FoPlanner, execute_functions, unify, subst
import itertools
# Chris Questions:
# Why does the explanation have 'value' deference instead of there being a whole path to the value in the resolved thing?
# How can we make where an actual thing?
# Calling all inputs and outputs foas is a little confusing?
# In training why does HowSearch run at the beginning and then only get used when there are no explanations?
# Secondary explanations... what to do....

# Nitty Gritty Questions:
# Does the when learner need to include foas & selection in search?
#   -How do we support the possibility that it does/doesn't
#   -are there accuracy/opp differences if only operating on the state without expanding / querying over all the foas/selections?


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
    mapping = {'foa' + str(i): ele for i, ele in enumerate(match)}
    r_state = rename_flat(state, {mapping[a]: a for a in mapping})
    return r_state

def unvariablize_by_where(state, match):
    mapping = {ele: 'foa' + str(i) for i, ele in enumerate(match)}
    r_state = rename_flat(state, {mapping[a]: a for a in mapping})
    return r_state

def expr_comparitor(fact, expr, mapping={}):
    if(isinstance(expr, dict)):
        if(isinstance(fact,dict)):
            if(not expr_comparitor(list(fact.keys())[0], list(expr.keys())[0], mapping)):
                return False
            if(not expr_comparitor(list(fact.values())[0], list(expr.values())[0], mapping)):
                return False
            return True
        else:
            return False
    if(isinstance(expr, tuple)):
        if(isinstance(fact, tuple) and len(fact) == len(expr)):
            for x, y in zip(fact, expr):
                # print(x,y)
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

# class HashableDict(dict):
#     def __hash__(self):
#         return hash(frozenset(self))



EMPTY_RESPONSE = {}

class ModularAgent(BaseAgent):

    def __init__(self, feature_set, function_set,
                                     when_learner='decisiontree', where_learner='MostSpecific',
                                     heuristic_learner='proportion_correct', how_cull_rule='all',
                                     planner='fo_planner', search_depth=1, numerical_epsilon=0.0):
        print(planner)
        self.where_learner = get_where_learner(where_learner)
        self.when_learner = get_when_learner(when_learner)
        self.which_learner = get_which_learner(heuristic_learner, how_cull_rule)
        self.planner = get_planner(planner, search_depth=search_depth, function_set=function_set, feature_set=feature_set)
        self.rhs_list = []
        self.rhs_by_label = {}
        self.rhs_by_how = {}
        self.feature_set = feature_set
        self.function_set = function_set
        self.search_depth = search_depth
        self.epsilon = numerical_epsilon
        self.rhs_counter = 0


    #####-----------------------REQUEST--------------------------###########

    def applicable_explanations(self, state, rhs_list=None, add_skill_info=False):# -> returns Iterator<Explanation>
        if(rhs_list == None): rhs_list = self.rhs_list
        for rhs in rhs_list:
            for match in self.where_learner.get_matches(rhs, state):
                if len(match) != len(set(match)):
                    continue

                pred_state = variablize_by_where(state.get_view("flat_ungrounded"), match) if(self.when_learner.state_format == "variablized_state") else state
                if(self.when_learner.predict(rhs, pred_state) <= 0):
                    continue
                explanation = Explanation(rhs, {v: m for v, m in zip(rhs.all_vars, match)})

                if(add_skill_info):
                    skill_info = {"when": tuple(self.when_learner.skill_info(rhs, pred_state)),
                                  "where": tuple([x.replace("?ele-", "") for x in match]),
                                  "how": str(rhs.input_rule),
                                  "which": 0.0}
                else: 
                    skill_info = None

                yield explanation, skill_info
        ######## -------------------------------------------------------#######
        

    def request(self, state, add_skill_info=False):#-> Returns sai
        state = StateMultiView("object",state)
        state = self.planner.apply_featureset(state)
        rhs_list = self.which_learner.sort_by_heuristic(self.rhs_list, state)
        explanations = [x for x in self.applicable_explanations(state, rhs_list=rhs_list, add_skill_info=add_skill_info)]
        explanation, skill_info = next(iter(explanations), (None, None))
        if explanation != None:
            response = explanation.to_response(state, self)
            if(add_skill_info):
                response["skill_info"] = skill_info

        else:   
            response = EMPTY_RESPONSE
        
        return response


    #####-----------------------TRAIN --------------------------###########

    def where_matches(self, explanations, state): #-> list<Explanation>, list<Explanation>
        matching_explanations, nonmatching_explanations = [], []
        for exp in explanations:
            if(self.where_learner.check_match(exp.rhs, list(exp.mapping.values()), state)):
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

    def explanations_from_skills(self, state, sai, rhs_list, foci_of_attention=None): # -> return Iterator<skill>
        for rhs in rhs_list:
            if(isinstance(rhs.input_rule, (int, float, str))):

                search_itr = [(rhs.input_rule, {})] if sai.inputs["value"] == rhs.input_rule else []
            else:
                search_itr = self.planner.how_search(state, sai, operators=[rhs.input_rule],
                    foci_of_attention=foci_of_attention, search_depth=1,
                    allow_bottomout=False, allow_copy=False)

            for input_rule, mapping in search_itr:
                m = {"?selection": "?ele-" + sai.selection}
                m.update(mapping)
                yield Explanation(rhs, m)

    def explanations_from_how_search(self, state, sai, foci_of_attention):# -> return Iterator<Explanation>
        sel_match = next(expression_matches({('?sel_prop', '?selection'): sai.selection}, state), None)
        selection_rule = (sel_match['?sel_prop'], '?selection') if sel_match != None else sai.selection
        for input_rule, mapping in self.planner.how_search(state, sai, foci_of_attention=foci_of_attention):
            inp_vars = list(mapping.keys())
            varz = list(mapping.values())
            rhs = RHS(selection_rule, sai.action, input_rule, "?selection", inp_vars, list(sai.inputs.keys()))
            ordered_mapping = {k: v for k, v in zip(rhs.all_vars, [sel_match['?selection']] + varz)}
            yield Explanation(rhs, ordered_mapping)
                

    def add_rhs(self, rhs, skill_label="DEFAULT_SKILL"):#-> return None
        rhs._id_num = self.rhs_counter
        self.rhs_counter += 1
        self.rhs_list.append(rhs)
        self.rhs_by_label[skill_label] = rhs

        constraints = generate_html_tutor_constraints(rhs)
        self.where_learner.add_rhs(rhs, constraints)
        self.when_learner.add_rhs(rhs)
        self.which_learner.add_rhs(rhs)


    def fit(self, explanations, state, reward):#-> return None
        for exp in explanations:
            if(self.when_learner.state_format == 'variablized_state'):
                self.when_learner.ifit(exp.rhs, variablize_by_where(state.get_view("flat_ungrounded"), exp.mapping.values()), reward)
            else:
                self.when_learner.ifit(exp.rhs, state, reward)
            self.which_learner.ifit(exp.rhs, state, reward)
            self.where_learner.ifit(exp.rhs, list(exp.mapping.values()), state, reward)


    def train(self, state, selection, action, inputs, reward, skill_label, foci_of_attention):# -> return None
        state = StateMultiView("object",state)
        sai = SAIS(selection, action, inputs)
        state_featurized = self.planner.apply_featureset(state)
        explanations = self.explanations_from_skills(state_featurized, sai, self.rhs_list, foci_of_attention)
        explanations, nonmatching_explanations = self.where_matches(explanations, state_featurized)

        if(len(explanations) == 0):

            if(len(nonmatching_explanations) > 0):
                explanations = [choice(nonmatching_explanations)]                

            else:
                explanations = self.explanations_from_how_search(state_featurized, sai, foci_of_attention)                
                explanations = self.which_learner.cull_how(explanations)#Choose all or just the most parsimonious 

                rhs_by_how = self.rhs_by_how.get(skill_label, {})
                for exp in explanations:
                    if(exp.rhs.as_tuple in rhs_by_how):
                        exp.rhs = rhs_by_how[exp.rhs.as_tuple]
                    else:
                        rhs_by_how[exp.rhs.as_tuple] = exp.rhs
                        self.rhs_by_how[skill_label] = rhs_by_how
                        self.add_rhs(exp.rhs)
                                                
        self.fit(explanations,state_featurized,reward)


    #####--------------------------CHECK-----------------------------###########

    def check(self, state, sai):
        state_featurized,knowledge_base = self.planner.apply_featureset(state)
        explanations = self.explanations_from_skills(state, sai, self.rhs_list)
        explanations, nonmatching_explanations = self.where_matches(explanations)
        return len(explanations) > 0



    def get_skills(self, states=None):
        out = []
        for state in states:
            req = self.request(state, add_skill_info=True).get('skill_info', None)

            if(req != None):
                out.append(frozenset([(k, v) for k, v in req.items()]))

        unique = [{k: v for k, v in x} for x in list(dict.fromkeys(out).keys())]# set(out)]
        return unique


#####--------------------------CLASS DEFINITIONS-----------------------------###########

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
    def __init__(self,view,state):
        self.views = {}
        self.set_view(view, state)
        self.transform_dict = {}
        self.register_transform("object", "flat_ungrounded", flatten_state)
        self.register_transform("flat_ungrounded", "key_vals_grounded", grounded_key_vals_state)
        self.register_transform("feat_knowledge_base", "flat_ungrounded", kb_to_flat_ungrounded)
    
    def set_view(self,view,state):
        self.views[view] = state

    def get_view(self,view):
        out = self.views.get(view,None)
        if(out is None):
            return self.compute(view)
            
        else:
            return out

    def contains_view(self,view):
        return view in self.views

    def compute(self,view):
        for key in self.transform_dict[view]:
            # for key in transforms:
            print(key)
            if(key in self.views):
                out = self.transform_dict[view][key](self.views[key])
                self.set_view(view,out)
                return out
        pprint(self.transform_dict)
        raise Exception("No transform possible from %s to %r" % (list(self.views.keys()),view))


    def compute_from(self,to, frm):
        assert to in self.transform_dict
        assert frm in self.transform_dict[to]
        out = self.transform_dict[to][frm](self.views[frm])
        self.set_view(to,out)
        return out

    def register_transform(self,frm, to, function):
        transforms = self.transform_dict.get(to,{})
        transforms[frm] = function
        self.transform_dict[to] = transforms


class SAIS(object):
    def __init__(self, selection, action, inputs, state=None):
        self.selection = selection
        self.action = action
        self.inputs = inputs# functions that return boolean 
        self.state = state
    def __repr__(self):
        return "S:%r, A:%r, I:%r" % (self.selection, self.action, self.inputs)


class RHS(object):
    def __init__(self, selection_expression, action, input_rule, selection_var, input_vars, input_args, conditions=[], label=None):
        self.selection_expression = selection_expression
        self.action = action
        self.input_rule = input_rule
        self.selection_var = selection_var
        self.input_vars = input_vars
        self.input_args = input_args
        self.all_vars = tuple([self.selection_var] + self.input_vars)
        self.as_tuple = (self.selection_expression,self.action,self.input_rule)

        self.conditions = conditions
        self.label = label
        self._how_depth = None
        self._id_num = None

        self.where = None
        self.when = None
        self.which = None
    
    def to_xml(self, agent=None):#-> needs some way of representing itself including its when/where/how parts
        raise NotImplementedError()
    def get_how_depth(self):
        if(self._how_depth == None):
            self._how_depth = compute_exp_depth(self.input_rule)
        return self._how_depth

    def __hash__(self):
        return self._id_num

    def __eq__(self, other):
        return self._id_num == other._id_num and self._id_num != None and other._id_num != None



class Explanation(object):
    
    def __init__(self, rhs, mapping):
        assert isinstance(mapping, dict), "Mapping must be type dict got type %r" % type(mapping)
        self.rhs = rhs
        self.mapping = mapping
        self.selection_literal = mapping[rhs.selection_var]
        self.input_literals = [mapping[s] for s in rhs.input_vars]#The Literal 

    def compute(self, state, agent):
        v = agent.planner.eval_expression([self.rhs.input_rule], self.mapping, state)[0]

        return {self.rhs.input_args[0]:v}
    def conditions_apply(self):
        return True

    def to_response(self, state, agent):
        response = {}
        response['skill_label'] = self.rhs.label
        response['selection'] = self.selection_literal.replace("?ele-", "")
        response['action'] = self.rhs.action
        response['inputs'] = self.compute(state, agent)#{a: rg_exp[3 + i] for i, a in
        return response


    def to_xml(self, agent=None):#-> needs some way of representing itself including its when/where/how parts
        pass

    def get_how_depth(self):
        return self.rhs.get_how_depth()

    def __str__(self):
        return str(self.rhs.input_rule) + ":(" + ",".join([x.replace("?ele-", "") for x in self.input_literals]) + ")->" + self.selection_literal.replace("?ele-", "")


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
        constraints.add(('value', arg, '?foa%ival' % (i+1)))
        constraints.add((is_not_empty_string, '?foa%ival' % (i+1)))

    return frozenset(constraints)

def is_not_empty_string(sting):
    return sting != ''
