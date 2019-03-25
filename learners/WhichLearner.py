

class WhichLearner(object):

    def __init__(self, heuristic_learner, how_cull_rule,learner_kwargs={}):
        
        # self.learner_name = learner_name
        self.heuristic_name = heuristic_learner
        self.how_cull_name = how_cull_rule
        self.learner_kwargs = learner_kwargs
        self.skills_by_label = {}
        self.learners = {}
        self.how_cull_rule = get_how_cull_rule(how_cull_rule)


    def add_skill(self,skill):
        self.learners[skill] = get_heuristic_agent(self.heuristic_name,**self.learner_kwargs)
        skills = self.skills_by_label.get(skill.label,[])
        skills.append(skill)
        self.skills_by_label[skill.label] = skills

    def ifit(self,skill, state, reward):
        return self.learners[skill].ifit(state, reward)

    def sort_by_heuristic(self,skills,state):
        # print([(x._id_num,self.learners[x].heuristic(state)) for x in skills])
        # out = sorted(skills,reverse=True, key=lambda x:self.learners[x].heuristic(state))
        # print([(x._id_num,self.learners[x].heuristic(state)) for x in out])
        return sorted(skills,reverse=True, key=lambda x:self.learners[x].heuristic(state))

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
    return [sorted(skills,key=lambda x:x.get_how_depth())[0]]


#####---------------UTILITIES------------------#####

def get_how_cull_rule(name):
    return CULL_HOW_RULES[name.lower().replace(' ', '').replace('_', '')]

def get_heuristic_agent(name,**learner_kwargs):
    return WHICH_HEURISTIC_AGENTS[name.lower().replace(' ', '').replace('_', '')](**learner_kwargs)

def get_which_learner(heuristic_learner,how_cull_rule,learner_kwargs={}):
    return WhichLearner(heuristic_learner,how_cull_rule,learner_kwargs)



WHICH_HEURISTIC_AGENTS = {
    'proportioncorrect': ProportionCorrect,
    'totalcorrect': TotalCorrect,   
}

CULL_HOW_RULES = {
    'first': first,
    'mostparsimonious': most_parsimonious,   
}
