import numpy as np
import itertools


class WhichLearner(object):

    def __init__(self, heuristic_learner, how_cull_rule,learner_kwargs={}):
        
        # self.learner_name = learner_name
        self.heuristic_name = heuristic_learner
        self.how_cull_name = how_cull_rule
        self.learner_kwargs = learner_kwargs
        self.rhs_by_label = {}
        self.learners = {}
        self.how_cull_rule = get_how_cull_rule(how_cull_rule)


    def add_rhs(self,rhs):
        self.learners[rhs] = get_heuristic_sublearner(self.heuristic_name,**self.learner_kwargs)
        rhs_list = self.rhs_by_label.get(rhs.label,[])
        rhs_list.append(rhs)
        self.rhs_by_label[rhs.label] = rhs_list

    def ifit(self,rhs, state, reward):
        return self.learners[rhs].ifit(state, reward)

    def sort_by_heuristic(self,rhs_list,state):
        # print([(x._id_num,self.learners[x].heuristic(state)) for x in skills])
        # out = sorted(skills,reverse=True, key=lambda x:self.learners[x].heuristic(state))
        # print([(x._id_num,self.learners[x].heuristic(state)) for x in out])
        return sorted(rhs_list,reverse=True, key=lambda x:self.learners[x].heuristic(state))

    def cull_how(self,expl_iter):
        return self.how_cull_rule(expl_iter)


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

def first(expl_iter):
    return [next(iter(expl_iter))]

def most_parsimonious(expl_iter):
    l = sorted(expl_iter,key=lambda x:x.get_how_depth())
    return l[:1]

def return_all(expl_iter):
    return [x for x in expl_iter]

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

def get_how_cull_rule(name):
    return CULL_HOW_RULES[name.lower().replace(' ', '').replace('_', '')]

def get_heuristic_sublearner(name,**learner_kwargs):
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
    'all': return_all,   
    # 'closest': closest,   
}
