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


def explain_sai(knowledge_base, exp, sai, epsilon):
    """
    Doc String
    """
    # print('trying', exp, 'with', sai)
    if len(exp) != len(sai):
        return

    goals = []
    for i, elem in enumerate(exp):
        if elem == sai[i]:
            continue
        else:
            goals.append((elem, sai[i]))

    # print("GOALS" ,goals)

    for match in knowledge_base.fc_query(goals, max_depth=0, epsilon=epsilon):
        yield match

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
		self.where = get_where_learner(where_learner)
		self.when = get_when_learner(when_learner)
		self.skills = {}
		self.examples = {}
		self.feature_set = feature_set
		self.function_set = function_set
		self.search_depth = search_depth
		self.epsilon = numerical_epsilon


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
	def which_learner.most_relevant(explanations):
		pass
	def which_learner.cull_how(explanations):
		pass


	#####-----------------------REQUEST--------------------------###########

	def applicable_explanations(self,state,skills=None):# -> returns Iterator<Explanation>
		if(skills == None): skills = self.skills

		#order here might need to be switched depending on when learner implementation (see nitty gritty questions below)
		if(when_learner.state_format == "state_only"):
			skills = when_learner.applicable_skills(state, skills=skills)
		else:
			skills = self.skills

		for skill in skills:
		######## ------------------  Vectorizable-----------------------#######
			for match in where_learner.get_matches(skill,state):
				#TODO: Should this even ever be produced?
				if len(match) != len(set(match)): continue

				explanation = Explanation(skill,match)
				if(explanation.conditions_apply()):
					if(when_learner.type == "var_foas_state" and when_learner.predict(skill, variablize_by_where(state, match))) <= 0:
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

	def explanations_from_skills(self,state, sai,skills): # -> return Iterator<skill>
		for skill in skills:
			######## ------------------  Vectorizable-----------------------#######
			for match in explain_sai(knowledge_base, exp, sai, self.epsilon):
				explanation = Explanation(skill,match)
				if(explanation.output_selection == sai.selection && explanation.compute(state) == sai.input):
					yield explanation
			######## -------------------------------------------------------#######

	def explanations_from_how_search(self,state, sai):# -> return Iterator<skill>
		return how_learner.how_search(state,sai)

	def add_skill(self,skill): #-> return None
		self.skills.append(skill)
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
				explanations = which_learner.cull_how(skills) #Choose all or just the most parsimonious 
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


class SAI(object):
	def __init__(self,selection, action, inp):
		self.selection = selection
		self.action = action
		self.input = inp# functions that return boolean 


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
	def __init__(conditions, action, operator_chain, input_selections, output_selection):
		self.conditions = conditions
		self.action = action
		self.operator_chain = operator_chain
		self.input_selections = input_selections
		self.output_selection = output_selection
		
	def to_xml(self,agent=None): #-> needs some way of representing itself including its when/where/how parts


class Explanation(object):
	
	def __init__(skill, mapping):
		self.skill = skill
		self.selections_mapping = mapping
		self.input_selections = [mapping[s] for s in skill.input_selections]#The Literal 
		self.output_selection = mapping[skill.output_selection] 
	def compute(self, state):
		return operator_chain(input_selections)
	def conditions_apply():
		for c in self.skill.conditions:
			c.applies(...)
	def to_sai():
		return SAI(self.input_selections,self.skill.action, self.compute)

	def to_xml(self,agent=None): #-> needs some way of representing itself including its when/where/how parts


 	#????? Should this be an object?, Should there be ways of translating between different types of states ???????
class State(object):
	#???




		
	
		

	