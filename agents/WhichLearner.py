

class WhichLearner(object):

    def __init__(self, relevance_heuristic_learner, how_cull_rule):
        
        # self.learner_name = learner_name
        self.relevance_heuristic_name = relevance_heuristic_learner
        self.how_selection_name = how_selection_rule
        self.skill_dict = {}
        self.learners = {}
        self.how_cull_rule = get_how_cull_rule(how_cull_rule)

    def add_skill(self,skill):
        self.learners[skill] = get_heuristic_agent(self.learner_name)
        skills = self.skill_dict.get(skill_label,[])
        skills.append(skill)
        self.skill_dict[skill_label] = skills

    def ifit(self,skill, state, reward):
        return self.learners[skill].ifit(state, reward)

    def sort_by_relevance(self,skills):
        return skills.sorted(reverse=True, key=lambda x:self.learners[x].heuristic())

    def cull_how(self,skills):
        return self.how_cull_rule(skills)


####---------------HEURISTIC------------########

class BaseHeuristicAgent(object):
    def __init__(self):
        pass
    def ifit(self,state,reward):
        pass
    def heuristic(self,state):
        pass

class TotalCorrect(BaseHeuristicAgent):
    def __init__(self):
        self.num_correct = 0
        self.num_incorrect = 0
    def ifit(self,state,reward):
        if(reward > 0):
            self.num_correct += 1
        else:
            self.num_incorrect += 1
    def heuristic(self,state):
        return self.num_correct


class ProportionCorrect(TotalCorrect):
    def heuristic(self,state):
        p,n = self.num_correct, self.num_incorrect
        return (p / (p + n),  p + n)

####---------------HOW CULL RULE------------########

def first(skills):
    return [next(iter(skills))]

def most_parsimonious(skills):
    return [skills.sorted(key=lambda x:x.get_how_depth())[0]]


#####---------------UTILITIES------------------#####

def get_how_cull_rule(name):
    return CULL_HOW_RULES[name.lower().replace(' ', '').replace('_', '')]

def get_heuristic_agent(name,learner_kwargs={}):
    return WHICH_HEURISTIC_AGENTS[name.lower().replace(' ', '').replace('_', '')](**learner_kwargs)

def get_which_learner(name,learner_kwargs={}):
    return WhichLearner(name,learner_kwargs)



WHICH_HEURISTIC_AGENTS = {
    'proportion_correct': ProportionCorrect,
    'total_correct': TotalCorrect,   
}

CULL_HOW_RULES = {
    'first': first,
    'most_parsimonious': most_parsimonious,   
}
