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
		self.skills_by_how = {}
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
		return state,knowledge_base


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
			skills = self.when_learner.applicable_skills(state, skills=skills)
		# else:
		# 	skills = self.skills

		for skill in skills:

			# print("SKILL TRY:", skill._id_num)
		######## ------------------  Vectorizable-----------------------#######
			for match in self.where_learner.get_matches(skill,state):
				#TODO: Should this even ever be produced?
				if len(match) != len(set(match)): continue
				
				# if(explanation.conditions_apply()):
				# print("MATHCH", match)
				# print("PRED:",self.when_learner.predict(skill, variablize_by_where(state, match)))
				if(self.when_learner.state_format == "variablized_state" and 
					self.when_learner.predict(skill, variablize_by_where(state, match)) <= 0) :
					continue
				explanation = Explanation(skill,{v:m for v,m in zip(skill.all_vars,match)})


				yield explanation
		######## -------------------------------------------------------#######
		

	def request(self,state): #-> Returns sai
		state_featurized,knowledge_base = self.apply_featureset(state)
		skills = self.which_learner.sort_by_heuristic(self.skills,state_featurized)
		explanations = self.applicable_explanations(state_featurized, skills=skills)
		explanation = next(explanations,None)
		# exp = explanation
		# if(exp != None):
		# 	print("REQUEST:", exp.skill._id_num, list(exp.mapping.values()), exp.skill.input_rule if isinstance(exp.skill.input_rule,int) else exp.skill.input_rule[1][0])
		return explanation.to_response(knowledge_base,self.function_set,self.epsilon) if explanation != None else EMPTY_RESPONSE


	#####-----------------------TRAIN --------------------------###########

	def where_matches(self,explanations,state): #-> list<Explanation>, list<Explanation>
		matching_explanations, nonmatching_explanations = [], []
		for exp in explanations:
			if(self.where_learner.check_match(exp.skill, list(exp.mapping.values()), state)):
				matching_explanations.append(exp)
			else:
				nonmatching_explanations.append(exp)
		return matching_explanations, nonmatching_explanations

	def _explain_sai(self, skill, sai, knowledge_base,state):
		# exp, iargs = skill
		# skill_where = learner_dict['where']
		if(skill.action == sai.action):
			for match in self.where_learner.get_matches(skill,state):
				mapping = {var:val for var,val in zip(skill.all_vars,match)}
				grounded_sel = subst(mapping, skill.selection_rule)
				grounded_inp = subst(mapping, skill.input_rule)
				rg_exp = tuple(eval_expression([grounded_sel,grounded_inp], knowledge_base, self.function_set, self.epsilon))
				# print("COMPARE", rg_exp,sai)
				# print(mapping)
				if(rg_exp[0] == sai.selection and {skill.input_args[0]:rg_exp[1]} == sai.inputs):
					yield Explanation(skill,mapping)

	def explanations_from_skills(self,state,knowledge_base, sai,skills): # -> return Iterator<skill>
		for skill in skills:
			######## ------------------  Vectorizable-----------------------#######
			for explanation in self._explain_sai(skill, sai, knowledge_base, state):
				# explanation = Explanation(skill,match)
				# if(explanation.output_selection == sai.selection && explanation.compute(state) == sai.input):
				yield explanation
			######## -------------------------------------------------------#######

	def explanations_from_how_search(self,state, sai):# -> return Iterator<Explanation>
		# def explanations_from_how_search(self,state,sai,input_args):
		#TODO: Make this a method of sai ... or does it need to be sorted?
		input_args = tuple(sorted([arg for arg in sai.inputs.keys()]))

		explanations = []
		# print(state)
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
			# print("match", sel_match)
			 
			sel_mapping = {ground(get_vars(unground(selection_exp))[0]) : '?selection'}
			selection_exp = list(rename_flat({selection_exp: True}, sel_mapping).keys())[0]
			# print("selection_exp", selection_exp)
			#selection_exp: ('id', 'QMele-done')
			break

		input_exps = []
		
		for arg in input_args:
			input_exp = input_val = sai.inputs[arg]
			# print('trying to explain', [((a, '?input'), iv)])
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
				# print()
				# if(len(varz) != len(set(varz))):
				# 	continue

				# print(var)
				foa_vmapping = {ground(field): '?foa%s' % j for j, field in enumerate(varz)}
				vmapping = {v:k for k,v in foa_vmapping.items()}
				r_exp = list(rename_flat({(arg, iv_m['?input']): True}, foa_vmapping).keys())[0]

				# print("iv_m", iv_m)
				# print(foa_vmapping)
				# print(rename_flat({(arg, iv_m['?input']): True}, foa_vmapping))

				ordered_mapping = {v:k for k,v in sel_mapping.items()}
				ordered_mapping.update({k:unground(v) for k,v in vmapping.items()})

				# print(ordered_mapping)


				skill = Skill(selection_exp, sai.action, r_exp, "?selection", list(vmapping.keys()), list(sai.inputs.keys()) )
				# print("BLEEP",skill.selection_rule, skill.input_rule)
				# vmapping["?selection"] = 

				exp_exists = True
				yield Explanation(skill, ordered_mapping)
			if(not exp_exists):
				skill = Skill(selection_exp, sai.action, input_exp, "?selection", [], list(sai.inputs.keys()) )
				ordered_mapping = ordered_mapping = {v:k for k,v in sel_mapping.items()}
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

		# print("ADD SKILL")
		skill._id_num = self.skill_counter
		self.skill_counter += 1
		self.skills.append(skill)
		self.skills_by_label[skill_label] = skill

		constraints = generate_html_tutor_constraints(skill)
		self.where_learner.add_skill(skill,constraints)
		self.when_learner.add_skill(skill)
		self.which_learner.add_skill(skill)


	def fit(self,explanations, state, reward): #-> return None
		for exp in explanations:
			if(self.when_learner.state_format == 'variablized_state'):
				# print("FIT_WHEN:",exp.skill._id_num, list(exp.mapping.values()), exp.skill.input_rule if isinstance(exp.skill.input_rule,int) else exp.skill.input_rule[1][0],  reward)
				self.when_learner.ifit(exp.skill, variablize_by_where(state,exp.mapping.values()), reward)
			else:
				self.when_learner.ifit(exp.skill, state, reward)
			self.which_learner.ifit(exp.skill, state, reward)
			# print("TRAIN ON", list(exp.mapping.values()))
			self.where_learner.ifit(exp.skill, list(exp.mapping.values()), state, reward)


	def train(self,state, selection, action, inputs, reward, skill_label, foci_of_attention):# -> return None
		sai = SAIS(selection, action, inputs)
		state_featurized,knowledge_base = self.apply_featureset(state)
		# print("TOTAL SKILLS:", len(self.skills))
		explanations = self.explanations_from_skills(state_featurized,knowledge_base,sai,self.skills)
		# explanations = [x for x in explanations]
		# print("SKILL EXPS:", len(explanations))
		explanations, nonmatching_explanations = self.where_matches(explanations,state_featurized)

		#TODO: DOES THIS EVER HAPPEN???? MAKING IT BREAK BECAUSE I WANT TO SEE IT. (maybe will happen with a different wherelearner)
		if(len(nonmatching_explanations) > 0):
			raise ValueError()
		print("MATCHING:", len(explanations), "NON-MATCHING:", len(nonmatching_explanations), "TOTAL:", len(self.skills))
		for x in self.skills:
			print(x.input_rule)
		# print("WHERE EXPS:", len(explanations))

		if(len(explanations) == 0 ):
			if(len(nonmatching_explanations) > 0):
				explanations = [choice(nonmatching_explanations)]
			else:
				explanations = self.explanations_from_how_search(state_featurized,sai)
				explanations = self.which_learner.cull_how(explanations) #Choose all or just the most parsimonious 

				skills_by_how = self.skills_by_how.get(skill_label, {})
				for exp in explanations:
					if(exp.skill.as_tuple in skills_by_how):
						exp.skill = skills_by_how[exp.skill.as_tuple] 
					else:
						skills_by_how[exp.skill.as_tuple] = exp.skill
						self.skills_by_how[skill_label] = skills_by_how
						self.add_skill(exp.skill)
							
					
		# skill = which_learner.most_relevant(skills)
		self.fit(explanations,state_featurized,reward)


	#####--------------------------CHECK-----------------------------###########

	def check(self, state, sai):
		state_featurized,knowledge_base = self.apply_featureset(state)
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
	def __repr__(self):
		return "S:%r, A:%r, I:%r" %(self.selection,self.action, self.inputs)


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
	def __init__(self,selection_rule, action, input_rule, selection_var, input_vars, input_args,conditions=[],label=None):
		self.selection_rule = selection_rule
		self.action = action
		self.input_rule = input_rule
		self.selection_var = selection_var
		self.input_vars = input_vars
		self.input_args = input_args
		self.all_vars = tuple([self.selection_var] + self.input_vars)
		self.as_tuple = (self.selection_rule,self.action,self.input_rule)

		self.conditions = conditions
		self.label = label
		self._how_depth = None
		self._id_num = None
	
	# def to_tuple():
	# 	return (self.selection_rule,self.action,self.input_rule)
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
		assert isinstance(mapping,dict), "Mapping must be type dict got type %r" % type(mapping)
		self.skill = skill
		self.mapping = mapping
		self.selection_literal = mapping[skill.selection_var]
		self.input_literals = [mapping[s] for s in skill.input_vars]#The Literal 

	def compute(self, knowledge_base,operators,epsilon):

		# print("input_rule:", self.skill.input_rule)
		# print("selection_rule:", self.skill.selection_rule)
		v = eval_expression([subst(self.mapping,self.skill.input_rule)], knowledge_base,operators,epsilon)[0]

		return {self.skill.input_args[0]:v}
	def conditions_apply(self):
		return True
		# for c in self.skill.conditions:
		# 	c.applies(...)
	# def to_sai():
	# 	return SAI(self.input_selections,self.skill.action, self.compute)
	def to_response(self, knowledge_base,operators,epsilon):
		# print("SHOULD BE", eval_expression([subst(self.mapping,self.skill.selection_rule)], knowledge_base,operators,epsilon))

		response = {}
		response['skill_label'] = self.skill.label
		response['selection'] = self.selection_literal.replace("?ele-","")
		# print('selection',response['selection'])
		response['action'] = self.skill.action
		# computed_inputs = self.compute()
		response['inputs'] = self.compute(knowledge_base,operators,epsilon)#{a: rg_exp[3 + i] for i, a in
		# print('inputs',response['inputs'])
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



def apply_operators(ele, knowledge_base,operators,epsilon):
    operator_output = None
    for operator in operators:
        effect = list(operator.effects)[0]
        pattern = effect[0]

        # print("pattern:",pattern)
        # print("ground(ele):",ground(ele))
        u_mapping = unify(pattern, ground(ele), {}) #Returns a mapping to name the values in the expression
        # print("u_mapping:",u_mapping)
        if(u_mapping):
            # print(operator.conditions,"\n")
            # print(u_mapping,"\n")
            # print(effect[1],"\n")
            # print("BEEP", [subst(u_mapping,x) for x in operator.conditions],"\n")
            condition_sub = [subst(u_mapping,x) for x in operator.conditions]
            # print("CS", condition_sub)


            # print("HERE1")
            value_map = next(knowledge_base.fc_query(condition_sub, max_depth=0, epsilon=epsilon))
            # print("HERE2")
            # print(value_map)
            # print()
            try:
                operator_output = execute_functions(subst(value_map,effect[1]))
            except:
                continue

            # print("HERE3")
            return operator_output
		
	
def eval_expression(r_exp,knowledge_base,function_set,epsilon):
    rg_exp = []
    # print("E", r_ele)
    for ele in r_exp:
        if isinstance(ele, tuple):
            # print("THIS HAPPENED", ele, ground(ele), execute_functions(ground(ele)))

            
                    
                    # execute_functions(subt(u))
                    # rg_exp.append()

                # print("BLEHH:", operator.effects) 
            ok = False
            for var_match in knowledge_base.fc_query([(ground(ele), '?v')],
                                                     max_depth=0,
                                                      epsilon=epsilon):
                # print("VARM:",var_match, ground(ele))
                if var_match['?v'] != '':
                    # print("V", var_match['?v'])
                    rg_exp.append(var_match['?v'])
                    # print("HERE_A",rg_exp[-1])
                    ok = True
                break

            if(not ok):
                operator_output = apply_operators(ele,knowledge_base,function_set,epsilon)
                # print("OPERATOR_OUTPUT", ele, "->", operator_output)
                if(operator_output != None and operator_output != ""):
                    # print("O", operator_output)
                    rg_exp.append(operator_output)
                    ok = True

            if(not ok):
                print("BLAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARG")
                rg_exp.append(None)
                # print("HERE_B",operator_output)
            


            # if(operator_output != None):
            #     rg_exp.append(operator_output)


        else:
            rg_exp.append(ele)
    return rg_exp

def generate_html_tutor_constraints(skill):
    """
    Given an skill, this finds a set of constraints for the SAI, so it don't
    fire in nonsensical situations.
    """
    constraints = set()

    # get action
    if skill.action == "ButtonPressed":
        constraints.add(('id', skill.selection_var, 'done'))
    else:
        constraints.add(('contentEditable', skill.selection_var, True))

    # value constraints, don't select empty values
    for i, arg in enumerate(skill.input_vars):
        constraints.add(('value', arg, '?foa%ival' % (i+1)))
        constraints.add((is_not_empty_string, '?foa%ival' % (i+1)))
        # constraints.add(('type', a, 'MAIN::cell'))

    return frozenset(constraints)

def is_not_empty_string(sting):
    return sting != ''