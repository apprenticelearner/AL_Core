import numpy as np
import itertools
from random import shuffle


class WhichLearner(object):

    def __init__(self, agent, utility_type, explanation_choice, remove_utility_type=None, **learner_kwargs):
        
        # self.learner_name = learner_name
        self.agent = agent
        rem_util_same = (utility_type == remove_utility_type) or remove_utility_type is None
        self.utility_type = utility_type
        self.remove_utility_type = remove_utility_type if not rem_util_same else utility_type
        self.explanation_choice = explanation_choice
        self.learner_kwargs = learner_kwargs
        self.rhs_by_label = {}
        self.learners = {}
        self.removal_learners = {} if not rem_util_same else self.learners
        self.explanation_choice = get_explanation_choice(explanation_choice)



    def add_rhs(self,rhs):
        self.learners[rhs] = get_utility_sublearner(self, rhs,self.utility_type,**self.learner_kwargs)
        rhs_list = self.rhs_by_label.get(rhs.label,[])
        rhs_list.append(rhs)
        self.rhs_by_label[rhs.label] = rhs_list
        if(self.utility_type != self.remove_utility_type):
            self.removal_learners[rhs] = get_utility_sublearner(self, rhs,self.remove_utility_type,**self.learner_kwargs)


    def ifit(self,rhs, state, mapping, reward):
        self.learners[rhs].ifit(state, mapping, reward)
        if(self.utility_type != self.remove_utility_type):
            self.removal_learners[rhs].ifit(state, mapping, reward)

    def sort_by_utility(self,rhs_list,state):
        # print([(x._id_num,self.learners[x].utility(state)) for x in skills])
        # out = sorted(skills,reverse=True, key=lambda x:self.learners[x].utility(state))
        # print([(x._id_num,self.learners[x].utility(state)) for x in out])
        return sorted(rhs_list,reverse=True, key=lambda x:self.get_utility(x,state))

    def select_how(self,expl_iter):
        return self.explanation_choice(expl_iter)

    def get_utility(self, rhs, state):
        return self.learners[rhs].utility(state)

    def get_removal_utility(self, rhs, state):
        return self.removal_learners[rhs].utility(state)



####---------------utility------------########

class BaseutilityAgent(object):
    def __init__(self):
        pass
    def ifit(self,state,reward):
        pass
    def utility(self,state):
        pass





class TotalCorrect(BaseutilityAgent):
    def __init__(self):
        self.num_correct = 0
        self.num_incorrect = 0
    def ifit(self,state, mapping, reward):
        if(reward > 0):
            self.num_correct += 1
        else:
            self.num_incorrect += 1
    def utility(self,state):
        return self.num_correct

class IncrementalWhenAccuracy(TotalCorrect):
    def ifit(self, state, mapping, reward):
        v_state = state.get_view(("variablize",self.rhs,tuple(mapping)))
        pred = self.agent.when_learner.predict(self.rhs, v_state)
        if(pred == reward):
            self.num_correct += 1
        else:
            self.num_incorrect += 1

    def utility(self, state):
        p,n = self.num_correct, self.num_incorrect
        s = p + n
        return (p / s if s > 0 else 0,  s)


class ProportionCorrect(TotalCorrect):
    def utility(self,state):
        p,n = self.num_correct, self.num_incorrect
        s = p + n
        return (p / s if s > 0 else 0,  s)

class WeightedProportionCorrect(TotalCorrect):
    def utility(self,state,w=2.0):
        p,n = self.num_correct, w*self.num_incorrect
        s = p + n
        return (p / s if s > 0 else 0,  s)

class NonLinearProportionCorrect(TotalCorrect):
    def utility(self,state,a=1.0,b=.25):
        p,n = self.num_correct, self.num_incorrect
        n = a*n + b*(n*n)
        s = p + n
        return (p / s if s > 0 else 0,  s)

####---------------HOW CULL RULE------------########

def first(expl_iter):
    return [next(iter(expl_iter))]

def most_parsimonious(expl_iter):
    l = sorted(expl_iter,key=lambda x:x.get_how_depth())
    return l[:1]

def least_depth(expl_iter):
    expl_iter = list(expl_iter)
    shuffle(expl_iter)
    l = sorted(expl_iter,key=lambda x: getattr(x.rhs.input_rule,'depth',0))
    return l[:1]

def least_operations(expl_iter):
    expl_iter = list(expl_iter)
    shuffle(expl_iter)
    l = sorted(expl_iter,key=lambda x: getattr(x.rhs.input_rule,'num_ops',0))
    return l[:1]

def return_all(expl_iter):
    return [x for x in expl_iter]

def random(expl_iter):
    arr = [x for x in expl_iter]
    shuffle(arr)
    # print("RANDOM",str(arr[:1][0]))
    return arr[:1]

# import itertools
# def closest(expl_iter,knowledge_base):

#     expl_iter = sorted(expl_iter,key=lambda x:x.get_how_depth())
#     closest = None
#     min_dist = float("inf")
#     for exp in expl_iter:
#         coords = []
#         for v in exp.mapping.values():
#             b = [x for x in knowledge_base.fc_query([(("offsetTop",v),"?top"),(("offsetLeft",v),"?left")],max_depth=0)]
#             coords.append((b[0]["?left"],b[0]["?top"]))

#         l1_dist = 0
#         pairs = list(itertools.combinations(coords, 2))
#         for pair in pairs:
#             l1_dist += np.abs(pair[0][0] - pair[1][0]) + np.abs(pair[0][1] - pair[1][1])
#         l1_dist = float(l1_dist) / max(float(len(pairs)), 1)
#         if(l1_dist < min_dist):
#             closest = exp
#             min_dist = l1_dist
#         # print(l1_dist)
#     if(closest != None):
#         print("HERE",list(closest.mapping.values()),min_dist)
#     return [closest] if closest != None else []
    


#####---------------UTILITIES------------------#####

def get_explanation_choice(name):
    return CULL_HOW_RULES[name.lower().replace(' ', '').replace('_', '')]

def get_utility_sublearner(parent, rhs, name,**learner_kwargs):
    sl = WHICH_utility_AGENTS[name.lower().replace(' ', '').replace('_', '')](**learner_kwargs)
    sl.rhs = rhs
    sl.agent = parent.agent
    return sl

def get_which_learner(agent, utility_learner, explanation_choice,**kwargs):
    return WhichLearner(agent, utility_learner, explanation_choice,**kwargs)



WHICH_utility_AGENTS = {
    'incrementalwhenaccuracy': IncrementalWhenAccuracy,
    'proportioncorrect': ProportionCorrect,
    'totalcorrect': TotalCorrect,   
    'weightedproportioncorrect': WeightedProportionCorrect,   
    'nonlinearproportioncorrect': NonLinearProportionCorrect
}

CULL_HOW_RULES = {
    'first': first,
    'mostparsimonious': most_parsimonious, #probably need to depricate
    'leastdepth': least_depth,   
    'leastoperations': least_operations,   
    'all': return_all,  
    'random' : random,
    # 'closest': closest,   
}
