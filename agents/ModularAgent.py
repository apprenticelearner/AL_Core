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


class ModularAgent(BaseAgent):

	def __init__(self, feature_set, function_set, 
									 when_learner='trestle', where_learner='MostSpecific',
									 search_depth=1, numerical_epsilon=0.0):
		self.where_learner = get_where_learner(where_learner)
		self.when_learner = get_when_learner(when_learner)
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

				explanation = Explanation(skill,match)
				if(explanation.conditions_apply()):
					if(when_learner.state_format == "var_foas_state" and when_learner.predict(skill, variablize_by_where(state, match))) <= 0:
						continue


					yield explanation
		######## -------------------------------------------------------#######
		

	def request(state): #-> Returns sai
		state_featurized = apply_featureset(state)
		skills = which_learner.sort_by_relevance(self.skills)
		explanations = applicable_explanations(state_featurized, skills=skills)
		return next(explanations).to_sai()


	#####-----------------------TRAIN --------------------------###########

	def where_matches(self,explanations): #-> list<Explanation>, list<Explanation>
		matching_explanations, nonmatching_explanations = [], []
		for exp in explanations:
			if(where_learner.check_match(exp)):
				matching_explanations.append(exp)
			else:
				nonmatching_explanations.append(exp)
		return matching_explanations, nonmatching_explanations

	def explain_sai(self, skill, learner_dict, sai, knowledge_base,state):
        exp, iargs = skill
        skill_where = learner_dict['where']

        for match in skill_where.get_matches(state):
            mapping = {var:val for var,val in zip(get_vars(exp),match)}
            grounded_exp = subst(mapping, exp)
            rg_exp = tuple(eval_expression(grounded_exp,knowledge_base,self.function_set,self.epsilon))

            if(rg_exp == sai):
                yield Explanation(skill,match)

	def explanations_from_skills(self,state, sai,skills): # -> return Iterator<skill>
		for skill in skills:
			######## ------------------  Vectorizable-----------------------#######
			for explanation in explain_sai(state, skill,sai):
				# explanation = Explanation(skill,match)
				# if(explanation.output_selection == sai.selection && explanation.compute(state) == sai.input):
				yield explanation
			######## -------------------------------------------------------#######

	def explanations_from_how_search(self,state, sai):# -> return Iterator<Explanation>
		# def explanations_from_how_search(self,state,sai,input_args):
        explanations = []

        _, selection, action, inputs = sai
        knowledge_base = FoPlanner([(ground(a),
                                     state[a].replace('?', 'QM')
                                     if isinstance(state[a], str)
                                     else state[a])
                                    for a in state],
                                   self.function_set)
        knowledge_base.fc_infer(depth=self.search_depth, epsilon=self.epsilon)

        #Danny: these lines figure out what property holds the selection
        selection_exp = selection
        for sel_match in knowledge_base.fc_query([('?selection', selection)],
                                                 max_depth=0,
                                                 epsilon=self.epsilon):

            selection_exp = sel_match['?selection']
            # print("selection_exp", selection_exp)
            #selection_exp: ('id', 'QMele-done')
            break

        input_exps = []
        
        for arg in input_args:
            input_exp = input_val = inputs[arg]
            # print('trying to explain', [((a, '?input'), iv)])
            #Danny: This populates a list of explanations found earlier in How search that work"
            skill = Skill(input_selections, output_selection, action, (arg, iv_m['?input']))
            possible = []
            for iv_m in knowledge_base.fc_query([((arg, '?input'), input_val)],
                                                max_depth=0,
                                                epsilon=self.epsilon):
            	yield Explanation(skill, iv_m)
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
		self.skill_dict[skill_label] = skill
		where_learner.add_skill(skill)
		when_learner.add_skill(skill)


	def fit(self,explanations, state, reward): #-> return None
		for exp in explanations:
			when_learner.ifit(exp.skill, state, reward)
			where_learner.ifit(exp.skill, reward)


	def train(self,state, sai, reward, skill_label, foci_of_attention):# -> return None
		state_featurized = apply_featureset(state)
		explanations = explanations_from_skills(state,sai,self.skills)
		explanations, nonmatching_explanations = where_matches(explanations)
		if(len(explanations) == 0 ):
			if(len(nonmatching_explanations) > 0):
				explanations = [choice(nonmatching_explanations)]
			else:
				explanations = explanations_from_how_search(state,sai)
				explanations = which_learner.cull_how(explanations) #Choose all or just the most parsimonious 
				for exp in explanations:
					self.add_skill(exp.skill)
		# skill = which_learner.most_relevant(skills)
		self.fit(explanations,state_featurized,reward)


	#####--------------------------CHECK-----------------------------###########

	def check(self, state, sai):
		state_featurized = apply_featureset(state)
		explanations = explanations_from_skills(state,sai,self.skills)
		explanations, nonmatching_explanations = where_matches(explanations)
		return len(explanations) > 0


#####--------------------------CLASS DEFINITIONS-----------------------------###########


class SAIS(object):
	def __init__(self,selection, action, inp,state=None):
		self.selection = selection
		self.action = action
		self.input = inp# functions that return boolean 
		self.state = state


class Operator(object):
	def __init__(self,args, function, conditions):
		self.num_inputs = len(args)
		self.function = function
		self.conditions = conditions# functions that return boolean 

	def compute(self, args):
		return function(**args)

	def conditions_met(self, args):
		for c in conditions:
			if(not c(...args)):
				return False
		return True


class Skill(object):
	def __init__(input_selections, output_selection, action, operator_graph, conditions=[],label=None):
		self.label = label
		self._id_num = None
		self.conditions = conditions
		self.action = action
		self.operator_graph = operator_graph
		self.input_selections = input_selections
		self.output_selection = output_selection
		self._how_depth = None
		
	def to_xml(self,agent=None): #-> needs some way of representing itself including its when/where/how parts
		pass
	def get_how_depth(self):
		if(self._how_depth == None):
			self._how_depth = compute_exp_depth(self.operator_graph)
		return self._how_depth

	def __hash__(self):
		return self._id_num

	def __eq__(self,other):
		return self._id_num == other._id_num



class Explanation(object):
	
	def __init__(self,skill, mapping):
		self.skill = skill
		self.selections_mapping = mapping
		self.input_selections = [mapping[s] for s in skill.input_selections]#The Literal 
		self.output_selection = mapping[skill.output_selection] 
	def compute(self, state):
		return operator_graph(input_selections)
	def conditions_apply():
		for c in self.skill.conditions:
			c.applies(...)
	def to_sai():
		return SAI(self.input_selections,self.skill.action, self.compute)

	def to_xml(self,agent=None): #-> needs some way of representing itself including its when/where/how parts
		pass

	def get_how_depth(self):
		return self.skill.get_how_depth()


 	#????? Should this be an object?, Should there be ways of translating between different types of states ???????
class State(object):
	#???




		
	
		

	