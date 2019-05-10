# from agents.ModularAgent import get_vars
from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
from planners.fo_planner import FoPlanner, execute_functions, unify, subst
from concept_formation.structure_mapper import rename_flat
from planners.rulesets import custom_function_set
from learners.VectorizedHowLearner import VectorizedHowLearner



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

class FO_State(dict):
	pass
	# def __init__(self,state,knowledge_base):
	# 	dict.__init__(state)
	# 	self.knowledge_base = knowledge_base
	# def __equal__



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


class FOL_HowLearner(object):
	def __init__(self,search_depth,function_set,feature_set):
		self.function_set = function_set
		self.feature_set = feature_set
		self.epsilon = 0.0
		self.search_depth = search_depth


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
		out = FO_State(state)
		out.knowledge_base = knowledge_base
		return out


	def how_search(self,state,sai,operators=None,foci_of_attention=None,search_depth=1,epsilon=0.0):
		# pprint(state)
		# print("SEARCH DEPTH", self.search_depth)
		if(operators == None): operators = self.function_set
		knowledge_base = FoPlanner([(ground(a),
									 state[a].replace('?', 'QM')
									 if isinstance(state[a], str)
									 else state[a])
									for a in state],
								   operators)
		knowledge_base.fc_infer(depth=search_depth, epsilon=epsilon)

		input_args = tuple(sorted([arg for arg in sai.inputs.keys()]))
		for arg in input_args:
			input_exp = input_val = sai.inputs[arg]
			
			#Danny: This populates a list of explanations found earlier in How search that work"
			exp_exists = False
			for iv_m in knowledge_base.fc_query([((arg, '?input'), input_val)],
												max_depth=0,
												epsilon=epsilon):
				# print("iv_m:",iv_m)
				varz = get_vars(  unground(  (arg, iv_m['?input'])  )  )

				if(foci_of_attention != None):
					if(not all([v.replace("?ele-","") in foci_of_attention for v in varz])):
						continue

				#Don't allow redundencies in the where
				#TODO: Does this even matter, should we just let it be? 
				#... certainly in a vectorized version we gain no speed by culling these out.
				if(len(varz) != len(set(varz))):
					continue

				exp_exists = True

				foa_vmapping = {ground(field): '?foa%s' % j for j, field in enumerate(varz)}
				vmapping = {v:k for k,v in foa_vmapping.items()}
				expression = list(rename_flat({(arg, iv_m['?input']): True}, foa_vmapping).keys())[0]
				yield expression,vmapping
			if(not exp_exists):
				yield input_exp,{}


	def eval_expression(self,r_exp,mapping,state,function_set=None,epsilon=0.0):
		# r_exp = subst(mapping,r_exp)
		if(function_set == None): function_set = self.function_set
		rg_exp = []
		for ele in r_exp:
			ele = subst(mapping,ele)
			if isinstance(ele, tuple):
				ok = False
				for var_match in state.knowledge_base.fc_query([(ground(ele), '?v')],
														 max_depth=0,
														  epsilon=epsilon):
					if var_match['?v'] != '':
						rg_exp.append(var_match['?v'])
						ok = True
					break

				if(not ok):
					operator_output = apply_operators(ele,state.knowledge_base,function_set,epsilon)
					if(operator_output != None and operator_output != ""):
						rg_exp.append(operator_output)
						ok = True

				if(not ok):
					rg_exp.append(None)

			else:
				rg_exp.append(ele)
		return rg_exp



def get_how_learner(name,**learner_kwargs):
    return HOW_LEARNERS[name.lower().replace(' ', '').replace('_', '')](**learner_kwargs)

# def get_how_learner(name,):
#     return WhichLearner(heuristic_learner,how_cull_rule,learner_kwargs)

HOW_LEARNERS = {
    'fol':FOL_HowLearner,
    'vectorized': VectorizedHowLearner
    # 'simstudent':SimStudentHow
}
