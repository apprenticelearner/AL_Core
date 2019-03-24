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



# Opinions/Notes
# -Each Learner Should own its own parts. Should not be kept in skills -> makes things more modular.


#TODO: Translate
# def explain_sai(knowledge_base, exp, sai, epsilon):
#     """
#     Doc String
#     """
#     # print('trying', exp, 'with', sai)
#     if len(exp) != len(sai):
#         return

#     goals = []
#     for i, elem in enumerate(exp):
#         if elem == sai[i]:
#             continue
#         else:
#             goals.append((elem, sai[i]))

#     # print("GOALS" ,goals)

#     for match in knowledge_base.fc_query(goals, max_depth=0, epsilon=epsilon):
#         yield match
# def flatten_state(state):
# 	tup = Tuplizer()
#     flt = Flattener()
#     return flt.transform(tup.transform(state))

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
	# print("MATCH FOUND", skill_label, exp, match)
	
	# vmapping = {'?foa' + str(i): ele for i, ele in enumerate(match)}
	# r_exp = list(rename_flat({exp: True}, vmapping))[0]

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
		self.skills = []
		self.skills_by_label = {}
		# self.examples = {}
		self.feature_set = feature_set
		self.function_set = function_set
		self.search_depth = search_depth
		self.epsilon = numerical_epsilon
		self.skill_counter = 0


	#######-----------------------MISC----------------------------##########
	def apply_featureset(self, state):
		####----------Definitely Vectorizable -------------######
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
		return state


	#######---------------Needs to be implemented elsewhere--------------------##########
	# def when_learner.applicable_skills(state,skills=None):
	# 	pass
	# def which_learner.most_relevant(explanations):
	# 	pass
	# def which_learner.cull_how(explanations):
	# 	pass
	# def where_learner.check_match(explanation):
	# 	pass
	# def how_learner.how_search(state,sai):
	# 	pass

	#####-----------------------REQUEST--------------------------###########

	def applicable_explanations(self,state,skills=None):# -> returns Iterator<Explanation>
		if(skills == None): skills = self.skills

		#order here might need to be switched depending on when learner implementation (see nitty gritty questions below)
		if(self.when_learner.state_format == "state_only"):
			skills = when_learner.applicable_skills(state, skills=skills)
		else:
			skills = self.skills

		for skill in skills:
		######## ------------------  Vectorizable-----------------------#######
			for match in self.where_learner.get_matches(skill,state):
				#TODO: Should this even ever be produced?
				if len(match) != len(set(match)): continue
				print("MATCH",match)
				explanation = Explanation(skill,match)
				if(explanation.conditions_apply()):
					if(when_learner.state_format == "variablized_state" and when_learner.predict(skill, variablize_by_where(state, match))) <= 0:
						continue


					yield explanation
		######## -------------------------------------------------------#######
		

	def request(self,state): #-> Returns sai
		state_featurized = self.apply_featureset(state)
		skills = self.which_learner.sort_by_heuristic(self.skills,state_featurized)
		explanations = self.applicable_explanations(state_featurized, skills=skills)
		explanation = next(explanations,None)
		return to_response.to_sai() if explanation != None else EMPTY_RESPONSE


	#####-----------------------TRAIN --------------------------###########

	def where_matches(self,explanations,state): #-> list<Explanation>, list<Explanation>
		matching_explanations, nonmatching_explanations = [], []
		for exp in explanations:
			if(where_learner.check_match(list(exp.mapping.keys()), state)):
				matching_explanations.append(exp)
			else:
				nonmatching_explanations.append(exp)
		return matching_explanations, nonmatching_explanations

	def _explain_sai(self, skill, sai, knowledge_base,state):
		exp, iargs = skill
		# skill_where = learner_dict['where']

		for match in self.where_learner.get_matches(skill,state):
			mapping = {var:val for var,val in zip(get_vars(exp),match)}
			grounded_exp = subst(mapping, exp)
			rg_exp = tuple(eval_expression(grounded_exp,knowledge_base,self.function_set,self.epsilon))

			if(rg_exp == sai):
				yield Explanation(skill,match)

	def explanations_from_skills(self,state, sai,skills): # -> return Iterator<skill>
		for skill in skills:
			######## ------------------  Vectorizable-----------------------#######
			for explanation in self._explain_sai(state, skill,sai):
				# explanation = Explanation(skill,match)
				# if(explanation.output_selection == sai.selection && explanation.compute(state) == sai.input):
				yield explanation
			######## -------------------------------------------------------#######

	def explanations_from_how_search(self,state, sai):# -> return Iterator<Explanation>
		# def explanations_from_how_search(self,state,sai,input_args):
		#TODO: Make this a method of sai ... or does it need to be sorted?
		input_args = tuple(sorted([arg for arg in sai.inputs.keys()]))

		explanations = []
		print(state)
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
			# print("selection_exp", selection_exp)
			#selection_exp: ('id', 'QMele-done')
			break

		input_exps = []
		
		for arg in input_args:
			input_exp = input_val = sai.inputs[arg]
			# print('trying to explain', [((a, '?input'), iv)])
			#Danny: This populates a list of explanations found earlier in How search that work"
			
			possible = []
			for iv_m in knowledge_base.fc_query([((arg, '?input'), input_val)],
												max_depth=0,
												epsilon=self.epsilon):

				varz = get_vars(  unground(  (arg, iv_m['?input'])  )  )
				foa_vmapping = {field: '?foa%s' % j for j, field in enumerate(varz)}
				vmapping = {v:k for k,v in foa_vmapping.items()}
				r_exp = list(rename_flat({(arg, iv_m['?input']): True}, vmapping).keys())[0]

				ordered_mapping = {"?selection": }
				ordered_mapping.update({k:ground(v) for k,v in vmapping.items()})


				skill = Skill(selection_exp, sai.action, r_exp, "?selection", list(vmapping.keys()) )
				# vmapping["?selection"] = 
				yield Explanation(skill, ordered_mapping)
				# possible.append((arg, iv_m['?input']))

			# possible = [(compute_exp_depth(p), random(), p) for p in possible]
			# possible.sort()

			# if len(possible) > 0:
			#     _, _, input_exp = possible[0]

			# input_exps.append(input_exp)
			# print("input_exp:", input_exp)
				
		# explanations.append(unground(('sai', selection_exp, action, *input_exps)))
		# print("EXPLANATIONS:", explanations)
		# return explanations

	def add_skill(self,skill,skill_label="DEFAULT_SKILL"): #-> return None
		skill._id_num = self.skill_counter
		self.skill_counter += 1
		self.skills.append(skill)
		self.skills_by_label[skill_label] = skill
		self.where_learner.add_skill(skill)
		self.when_learner.add_skill(skill)
		self.which_learner.add_skill(skill)


	def fit(self,explanations, state, reward): #-> return None
		for exp in explanations:
			self.when_learner.ifit(exp.skill, state, reward)
			self.which_learner.ifit(exp.skill, state, reward)
			print("TRAIN ON", list(exp.mapping.values()))
			self.where_learner.ifit(exp.skill, list(exp.mapping.values()), state, reward)


	def train(self,state, selection, action, inputs, reward, skill_label, foci_of_attention):# -> return None
		sai = SAIS(selection, action, inputs)
		state_featurized = self.apply_featureset(state)
		explanations = self.explanations_from_skills(state_featurized,sai,self.skills)
		explanations, nonmatching_explanations = self.where_matches(explanations,state_featurized)
		if(len(explanations) == 0 ):
			if(len(nonmatching_explanations) > 0):
				explanations = [choice(nonmatching_explanations)]
			else:
				explanations = self.explanations_from_how_search(state_featurized,sai)
				explanations = self.which_learner.cull_how(explanations) #Choose all or just the most parsimonious 
				for exp in explanations:
					self.add_skill(exp.skill)
		# skill = which_learner.most_relevant(skills)
		self.fit(explanations,state_featurized,reward)


	#####--------------------------CHECK-----------------------------###########

	def check(self, state, sai):
		state_featurized = self.apply_featureset(state)
		explanations = self.explanations_from_skills(state,sai,self.skills)
		explanations, nonmatching_explanations = self.where_matches(explanations)
		return len(explanations) > 0


#####--------------------------CLASS DEFINITIONS-----------------------------###########


class SAIS(object):
	def __init__(self,selection, action, inputs,state=None):
		self.selection = selection
		self.action = action
		self.inputs = inputs# functions that return boolean 
		self.state = state


# class Operator(object):
# 	def __init__(self,args, function, conditions):
# 		self.num_inputs = len(args)
# 		self.function = function
# 		self.conditions = conditions# functions that return boolean 

# 	def compute(self, args):
# 		return self.function(**args)

# 	def conditions_met(self, args):
# 		for c in conditions:
# 			if(not c(...args)):
# 				return False
# 		return True


class Skill(object):
	def __init__(self,selection_rule, action, input_rule, selection_var, input_vars, conditions=[],label=None):
		self.selection_rule = selection_rule
		self.action = action
		self.input_rule = input_rule
		self.selection_var = selection_var
		self.input_vars = input_vars
		self.all_vars = tuple([self.selection_var] + self.input_vars)

		self.conditions = conditions
		self.label = label
		self._how_depth = None
		self._id_num = None
		
	def to_xml(self,agent=None): #-> needs some way of representing itself including its when/where/how parts
		pass
	def get_how_depth(self):
		if(self._how_depth == None):
			self._how_depth = compute_exp_depth(self.input_rule)
		return self._how_depth

	def __hash__(self):
		return self._id_num

	def __eq__(self,other):
		return self._id_num == other._id_num



class Explanation(object):
	
	def __init__(self,skill, mapping):
		self.skill = skill
		self.mapping = mapping
		self.selection_literal = mapping[skill.selection_var]
		self.input_literals = [mapping[s] for s in skill.input_vars]#The Literal 

	def compute(self, state):
		return operator_graph(input_selections)
	def conditions_apply():
		return True
		# for c in self.skill.conditions:
		# 	c.applies(...)
	# def to_sai():
	# 	return SAI(self.input_selections,self.skill.action, self.compute)
	def to_response(self):
		response = {}
		response['skill_label'] = self.skill.label
		response['selection'] = self.selection_literal
		response['action'] = self.skill.action
		# computed_inputs = self.compute()
		response['inputs'] = self.compute()#{a: rg_exp[3 + i] for i, a in
							  #enumerate(input_args)}
		# response['inputs'] = list(rg_exp[3:])
		response['foci_of_attention'] = []
		return response


	def to_xml(self,agent=None): #-> needs some way of representing itself including its when/where/how parts
		pass

	def get_how_depth(self):
		return self.skill.get_how_depth()


	#????? Should this be an object?, Should there be ways of translating between different types of states ???????
# class State(object):
	#???




		
	
		

	