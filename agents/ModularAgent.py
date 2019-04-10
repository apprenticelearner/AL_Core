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
from planners.fo_planner import FoPlanner, execute_functions, unify, subst
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

def get_vars(arg):
	"""
	Doc String
	"""
	if isinstance(arg, tuple):
		lis = []
		for elem in arg:
			for val in get_vars(elem):
				if val not in lis:
					lis.append(val)
		return lis
		# return [v for e in arg for v in self.get_vars(e)]
	elif isinstance(arg, str) and len(arg) > 0 and arg[0] == '?':
		return [arg]
	else:
		return []

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

def replace_vars(arg, i=0):
	"""
	Doc String
	"""
	if isinstance(arg, tuple):
		ret = []
		for elem in arg:
			replaced, i = replace_vars(elem, i)
			ret.append(replaced)
		return tuple(ret), i
	elif isinstance(arg, str) and len(arg) > 0 and arg[0] == '?':
		return '?foa%s' % (str(i)), i+1
	else:
		return arg, i

def variablize_by_where(state,match):
	mapping = {'foa' + str(i): ele for i, ele in enumerate(match)}
	r_state = rename_flat(state, {mapping[a]: a for a in mapping})
	return r_state


EMPTY_RESPONSE = {}

class ModularAgent(BaseAgent):

	def __init__(self, feature_set, function_set, 
									 when_learner='trestle', where_learner='MostSpecific',
									 heuristic_learner='proportion_correct', how_cull_rule='most_parsimonious',
									 search_depth=1, numerical_epsilon=0.0):
		self.where_learner = get_where_learner(where_learner)
		self.when_learner = get_when_learner(when_learner)
		self.which_learner = get_which_learner(heuristic_learner,how_cull_rule)
		self.rhs_list = []
		self.rhs_by_label = {}
		self.rhs_by_how = {}
		self.feature_set = feature_set
		self.function_set = function_set
		self.search_depth = search_depth
		self.epsilon = numerical_epsilon
		self.rhs_counter = 0


	#######-----------------------MISC----------------------------##########
	def apply_featureset(self, state):
		tup = Tuplizer()
		flt = Flattener()
		state = flt.transform(tup.transform(state))

		knowledge_base = FoPlanner([(ground(a), state[a].replace('?', 'QM')
									 if isinstance(state[a], str)
									 else state[a])
									for a in state],
								   self.feature_set)
		knowledge_base.fc_infer(depth=1, epsilon=self.epsilon)
		state = {unground(a): v.replace("QM", "?")
				 if isinstance(v, str)
				 else v
				 for a, v in knowledge_base.facts}
		return state,knowledge_base


	#####-----------------------REQUEST--------------------------###########

	def applicable_explanations(self,state,rhs_list=None):# -> returns Iterator<Explanation>
		if(rhs_list == None): rhs_list = self.rhs_list

		for rhs in rhs_list:

			for match in self.where_learner.get_matches(rhs,state):
				#TODO: Should this even ever be produced?
				if len(match) != len(set(match)): continue
				
				if(self.when_learner.state_format == "variablized_state" and 
					self.when_learner.predict(rhs, variablize_by_where(state, match)) <= 0) :
					continue
				explanation = Explanation(rhs,{v:m for v,m in zip(rhs.all_vars,match)})

				yield explanation
		######## -------------------------------------------------------#######
		

	def request(self,state): #-> Returns sai
		state_featurized,knowledge_base = self.apply_featureset(state)
		rhs_list = self.which_learner.sort_by_heuristic(self.rhs_list,state_featurized)
		explanations = self.applicable_explanations(state_featurized, rhs_list=rhs_list)
		explanation = next(explanations,None)

		return explanation.to_response(knowledge_base,self.function_set,self.epsilon) if explanation != None else EMPTY_RESPONSE


	#####-----------------------TRAIN --------------------------###########

	def where_matches(self,explanations,state): #-> list<Explanation>, list<Explanation>
		matching_explanations, nonmatching_explanations = [], []
		for exp in explanations:
			if(self.where_learner.check_match(exp.rhs, list(exp.mapping.values()), state)):
				matching_explanations.append(exp)
			else:
				nonmatching_explanations.append(exp)
		return matching_explanations, nonmatching_explanations

	def _explain_sai(self, rhs, sai, knowledge_base,state):
		
		if(rhs.action == sai.action):
			for match in self.where_learner.get_matches(rhs,state):
				mapping = {var:val for var,val in zip(rhs.all_vars,match)}
				grounded_sel = subst(mapping, rhs.selection_expression)
				grounded_inp = subst(mapping, rhs.input_rule)
				rg_exp = tuple(eval_expression([grounded_sel,grounded_inp], knowledge_base, self.function_set, self.epsilon))
				if(rg_exp[0] == sai.selection and {rhs.input_args[0]:rg_exp[1]} == sai.inputs):
					yield Explanation(rhs,mapping)

	def explanations_from_skills(self,state,knowledge_base, sai,rhs_list): # -> return Iterator<skill>
		for rhs in rhs_list:
			for explanation in self._explain_sai(rhs, sai, knowledge_base, state):
				yield explanation

	def explanations_from_how_search(self,state, sai):# -> return Iterator<Explanation>
		#TODO: Make this a method of sai ... or does it need to be sorted?
		input_args = tuple(sorted([arg for arg in sai.inputs.keys()]))

		explanations = []
		print("SEARCH DEPTH", self.search_depth)
		knowledge_base = FoPlanner([(ground(a),
									 state[a].replace('?', 'QM')
									 if isinstance(state[a], str)
									 else state[a])
									for a in state],
								   self.function_set)
		knowledge_base.fc_infer(depth=self.search_depth, epsilon=self.epsilon)

		#Danny: these lines figure out what property holds the selection
		selection_exp = sai.selection
		for sel_match in knowledge_base.fc_query([('?selection', sai.selection)],
												 max_depth=0,
												 epsilon=self.epsilon):

			selection_exp = sel_match['?selection']
			 
			sel_mapping = {ground(get_vars(unground(selection_exp))[0]) : '?selection'}
			selection_exp = list(rename_flat({selection_exp: True}, sel_mapping).keys())[0]
			break

		input_exps = []
		
		for arg in input_args:
			input_exp = input_val = sai.inputs[arg]
			
			#Danny: This populates a list of explanations found earlier in How search that work"
			possible = []
			exp_exists = False
			for iv_m in knowledge_base.fc_query([((arg, '?input'), input_val)],
												max_depth=0,
												epsilon=self.epsilon):

				varz = get_vars(  unground(  (arg, iv_m['?input'])  )  )

				#Don't allow redundencies in the where
				#TODO: Does this even matter, should we just let it be? 
				#... certainly in a vectorized version we gain no speed by culling these out.
				# if(len(varz) != len(set(varz))):
				# 	continue

				foa_vmapping = {ground(field): '?foa%s' % j for j, field in enumerate(varz)}
				vmapping = {v:k for k,v in foa_vmapping.items()}
				r_exp = list(rename_flat({(arg, iv_m['?input']): True}, foa_vmapping).keys())[0]

				ordered_mapping = {v:k for k,v in sel_mapping.items()}
				ordered_mapping.update({k:unground(v) for k,v in vmapping.items()})

				rhs = RHS(selection_exp, sai.action, r_exp, "?selection", list(vmapping.keys()), list(sai.inputs.keys()) )

				exp_exists = True
				yield Explanation(rhs, ordered_mapping)
			if(not exp_exists):
				rhs = RHS(selection_exp, sai.action, input_exp, "?selection", [], list(sai.inputs.keys()) )
				ordered_mapping = ordered_mapping = {v:k for k,v in sel_mapping.items()}
				yield Explanation(rhs, ordered_mapping)
				

	def add_rhs(self,rhs,skill_label="DEFAULT_SKILL"): #-> return None
		rhs._id_num = self.rhs_counter
		self.rhs_counter += 1
		self.rhs_list.append(rhs)
		self.rhs_by_label[skill_label] = rhs

		constraints = generate_html_tutor_constraints(rhs)
		self.where_learner.add_rhs(rhs,constraints)
		self.when_learner.add_rhs(rhs)
		self.which_learner.add_rhs(rhs)


	def fit(self,explanations, state, reward): #-> return None
		for exp in explanations:
			if(self.when_learner.state_format == 'variablized_state'):
				self.when_learner.ifit(exp.rhs, variablize_by_where(state,exp.mapping.values()), reward)
			else:
				self.when_learner.ifit(exp.rhs, state, reward)
			self.which_learner.ifit(exp.rhs, state, reward)
			self.where_learner.ifit(exp.rhs, list(exp.mapping.values()), state, reward)


	def train(self,state, selection, action, inputs, reward, skill_label, foci_of_attention):# -> return None

		sai = SAIS(selection, action, inputs)
		state_featurized,knowledge_base = self.apply_featureset(state)
		explanations = self.explanations_from_skills(state_featurized,knowledge_base,sai,self.rhs_list)
		explanations, nonmatching_explanations = self.where_matches(explanations,state_featurized)

		#TODO: DOES THIS EVER HAPPEN???? MAKING IT BREAK BECAUSE I WANT TO SEE IT. (maybe will happen with a different wherelearner)
		if(len(nonmatching_explanations) > 0):
			raise ValueError()
		
		if(len(explanations) == 0 ):
			if(len(nonmatching_explanations) > 0):
				explanations = [choice(nonmatching_explanations)]
			else:
				explanations = self.explanations_from_how_search(state_featurized,sai)
				explanations = self.which_learner.cull_how(explanations) #Choose all or just the most parsimonious 

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
		state_featurized,knowledge_base = self.apply_featureset(state)
		explanations = self.explanations_from_skills(state,sai,self.rhs_list)
		explanations, nonmatching_explanations = self.where_matches(explanations)
		return len(explanations) > 0


#####--------------------------CLASS DEFINITIONS-----------------------------###########


class SAIS(object):
	def __init__(self,selection, action, inputs,state=None):
		self.selection = selection
		self.action = action
		self.inputs = inputs# functions that return boolean 
		self.state = state
	def __repr__(self):
		return "S:%r, A:%r, I:%r" %(self.selection,self.action, self.inputs)


class RHS(object):
	def __init__(self,selection_expression, action, input_rule, selection_var, input_vars, input_args,conditions=[],label=None):
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
	
	def to_xml(self,agent=None): #-> needs some way of representing itself including its when/where/how parts
		raise NotImplementedError();
	def get_how_depth(self):
		if(self._how_depth == None):
			self._how_depth = compute_exp_depth(self.input_rule)
		return self._how_depth

	def __hash__(self):
		return self._id_num

	def __eq__(self,other):
		return self._id_num == other._id_num



class Explanation(object):
	
	def __init__(self,rhs, mapping):
		assert isinstance(mapping,dict), "Mapping must be type dict got type %r" % type(mapping)
		self.rhs = rhs
		self.mapping = mapping
		self.selection_literal = mapping[rhs.selection_var]
		self.input_literals = [mapping[s] for s in rhs.input_vars]#The Literal 

	def compute(self, knowledge_base,operators,epsilon):
		v = eval_expression([subst(self.mapping,self.rhs.input_rule)], knowledge_base,operators,epsilon)[0]

		return {self.rhs.input_args[0]:v}
	def conditions_apply(self):
		return True

	def to_response(self, knowledge_base,operators,epsilon):

		response = {}
		response['skill_label'] = self.rhs.label
		response['selection'] = self.selection_literal.replace("?ele-","")
		response['action'] = self.rhs.action
		response['inputs'] = self.compute(knowledge_base,operators,epsilon)#{a: rg_exp[3 + i] for i, a in
		response['foci_of_attention'] = []
		return response


	def to_xml(self,agent=None): #-> needs some way of representing itself including its when/where/how parts
		pass

	def get_how_depth(self):
		return self.rhs.get_how_depth()



def apply_operators(ele, knowledge_base,operators,epsilon):
    operator_output = None
    for operator in operators:
        effect = list(operator.effects)[0]
        pattern = effect[0]

        u_mapping = unify(pattern, ground(ele), {}) #Returns a mapping to name the values in the expression
        if(u_mapping):
            condition_sub = [subst(u_mapping,x) for x in operator.conditions]
            value_map = next(knowledge_base.fc_query(condition_sub, max_depth=0, epsilon=epsilon))
            try:
                operator_output = execute_functions(subst(value_map,effect[1]))
            except:
                continue

            return operator_output
		
	
def eval_expression(r_exp,knowledge_base,function_set,epsilon):
    rg_exp = []
    for ele in r_exp:
        if isinstance(ele, tuple):
            ok = False
            for var_match in knowledge_base.fc_query([(ground(ele), '?v')],
                                                     max_depth=0,
                                                      epsilon=epsilon):
                if var_match['?v'] != '':
                    rg_exp.append(var_match['?v'])
                    ok = True
                break

            if(not ok):
                operator_output = apply_operators(ele,knowledge_base,function_set,epsilon)
                if(operator_output != None and operator_output != ""):
                    rg_exp.append(operator_output)
                    ok = True

            if(not ok):
                rg_exp.append(None)

        else:
            rg_exp.append(ele)
    return rg_exp

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