"""
The contents of this module are currently experimental and under active
development. More thorough documentation will be done when its development has
settled.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from itertools import product
import inspect
from random import choice
from numbers import Number
from time import time

import numpy as np

from py_search.search import Node
from py_search.search import Problem
from py_search.search import best_first_search
from py_search.search import compare_searches


def levenshtein(source, target):
    """ 
    The levenshtein edit distance, code take from here: 
        http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """
    if len(source) < len(target):
        return levenshtein(target, source)
 
    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)
 
    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))
 
    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1
 
        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))
 
        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)
 
        previous_row = current_row
 
    return previous_row[-1]

class ActionPlannerProblem(Problem):

    def successor(self, node):
        state, chosen, goal = node.state
        actions = node.extra["actions"]
        tested = node.extra['tested']

        if chosen is None:
            for ele in state:
                if ele not in tested:
                    attr, val = ele
                    tested.add(ele)
                    yield Node((tuple(state), ele, goal), node, attr,
                                   node.cost(), node.extra)

        for action in actions:
            num_args = len(inspect.getargspec(actions[action]).args)

            # TODO pass in some function that filters out the arguments that
            # are worth considering otherwise you waste your time searching too
            # much.
            #possible_args = [ele for ele in state if len(ele[0]) > 0 and
            #                 ele[0][0] == 'value']
            possible_args = [ele for ele in state if len(ele[0]) > 0]

            #print(len([ele for ele in product(possible_args, repeat=num_args)]))
            #for tupled_args in product(state, repeat=num_args):
            for tupled_args in product(possible_args, repeat=num_args):
                names = [a for a, v in tupled_args]
                values = [v for a, v in tupled_args]
                new_state = list(state)
                action_name = tuple([action] + names)
                try:
                    new_state.append((action_name, actions[action](*values)))
                    path_cost = node.cost() + 1
                    yield Node((tuple(new_state), None, goal), node, action_name,
                               path_cost, node.extra)
                except Exception as e:
                    pass

    def goal_test(self, node):
        s, chosen, goal = node.state
        epsilon = node.extra["epsilon"]

        if chosen is None:
            return False

        attr, v = chosen

        if ((not isinstance(goal, bool) and isinstance(goal, Number)) and 
            (not isinstance(v, bool) and isinstance(v, Number))):

            if abs(goal - v) <= epsilon:
                return True

        elif v == goal:
            return True

        return False

    def heuristic(self, node):
        state, chosen, goal = node.state
        tested = node.extra['tested']

        if chosen is not None:
            return 0

        #h = float('inf')
        h = 1

        is_number_goal = (isinstance(goal, Number) and 
                          not isinstance(goal, bool))

        if is_number_goal:
            for a,v in state:
                if (a, v) in tested:
                    continue 
                try:
                    v = float(v)
                    #vmin = -1000
                    vmax = 2000
                    diff = min((goal - v) * (goal - v), vmax)
                    dist = (diff) / 2000
                    if dist < h:
                        h = dist
                except:
                    pass
        else:
            for a,v in state:
                if (a, v) in tested:
                    continue 
                if isinstance(v, str):
                    vmin = -1000
                    vmax = 1000
                    diff = max(min(levenshtein(v, goal), vmax), vmin)
                    dist = (diff + 1000) / 2000
                    if dist < h:
                        h = dist

        return h

    def node_value(self, node):
        return node.cost() + self.heuristic(node)

class NoHeuristic(ActionPlannerProblem):

    def node_value(self, node):
        return node.cost()

class ActionPlanner:

    def __init__(self, actions, epsilon=0.0, depth_limit=2):

        # Added these checks because I was getting weird behavior from the web
        # API with things being strings when they shouldn't be        
        if not isinstance(epsilon,float) or epsilon < 0.0:
            raise ValueError("epsilon must be a float >= 0")
        if not isinstance(depth_limit,int):
            raise ValueError("depth_limit must be an integer")
        self.actions = actions
        self.epsilon = epsilon
        self.depth_limit = depth_limit

    def explain_sai(self, state, sai, num_expl=1, time_limit=float('inf')):
        """
        This function generates a number of explainations for a given observed SAI.
        """
        # TODO I want to add similar checks to the parameters here but  I could
        # see arguments for negative or 0 values be code for unlimited.
        exps = [self.explain_value(state, ele, num_expl, time_limit) for ele in
                sai[2:]]
        return [sai[0:2] + inp for inp in product(*exps)]

    def explain_value(self, state, value, num_expl=1, time_limit=float('inf')):
        """ 
        This function uses a planner compute the given value from the current
        state. The function returns a plan.
        """
        
        extra = {}
        extra["actions"] = self.actions
        extra["epsilon"] = self.epsilon
        extra['tested'] = set()

        state = {k:state[k] for k in state if k[0] != '_'}

        problem = ActionPlannerProblem((tuple(state.items()), None, value),extra)

        explanations = []
        
        #print ("EXPLAINING ", value)
        s_time = time()
        try:
            for solution in best_first_search(problem, cost_limit=self.depth_limit):
                #print(solution)
                if len(solution.path()) > 0:
                    state, chosen, goal = solution.state
                    #print(chosen, solution.cost())
                    explanations.append(chosen[0])
                if len(explanations) == num_expl:
                    break
                if time() - s_time > time_limit: 
                    break
        except StopIteration:
            #print("EXPLAIN FAILED")
            pass

        if len(explanations) == 0:
            return [str(value)]
        else:
            return explanations

    def execute_plan(self, plan, state):
        
        actions = self.actions

        if plan in state:
            return state[plan]

        if not isinstance(plan, tuple):
            return plan

        #print("PLAN!!! ", plan)
        args = tuple(self.execute_plan(ele, state) for ele in plan[1:])
        action = plan[0]

        return actions[action](*args)

#def car(x):
#    if isinstance(x, str) and len(x) > 1:
#        return x[0]
#
#def cdr(x):
#    if isinstance(x, str) and len(x) > 2:
#        return x[1:]
#
#def append(x, y):
#    if isinstance(x, str) and isinstance(y, str):
#        return x + y
#
#def tostring(x):
#    return str(x)

def add(x,y):
    if isinstance(x, str) and isinstance(y,str):
        x = float(x)
        y = float(y)
        return "%i" % (x+y)
    elif ((not isinstance(x, bool) and isinstance(x,Number)) and 
          (not isinstance(y, bool) and isinstance(y,Number))):
        return x+y
    else:
        raise TypeError("Arguments must both be strings or both be Numbers")


def subtract(x,y):
    if isinstance(x, str) and isinstance(y,str):
        x = float(x)
        y = float(y)
        return "%i" % (x-y)
    elif ((not isinstance(x, bool) and isinstance(x,Number)) and 
          (not isinstance(y, bool) and isinstance(y,Number))):
        return x-y
    else:
        raise TypeError("Arguments must both be strings or both be Numbers")
    

def multiply(x,y):
    if isinstance(x, str) and isinstance(y,str):
        x = float(x)
        y = float(y)
        return "%i" % (x*y)
    elif ((not isinstance(x, bool) and isinstance(x,Number)) and 
          (not isinstance(y, bool) and isinstance(y,Number))):
        return x*y
    else:
        raise TypeError("Arguments must both be strings or both be Numbers")
    

def divide(x,y):
    if isinstance(x, str) and isinstance(y,str):
        x = float(x)
        y = float(y)
        return "%i" % (x/y)
    elif ((not isinstance(x, bool) and isinstance(x,Number)) and 
          (not isinstance(y, bool) and isinstance(y,Number))):
        return x/y
    else:
        raise TypeError("Arguments must both be strings or both be Numbers")

 
math_actions = {
    "add":add,
    "subtract":subtract,
    "multiply":multiply,
    "divide":divide
}

#updating math function using pyfunction
#import sys, os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'apprentice_learner'))
#print("HELLO ARE YO thewU")
#import models
#print("HELLOHOW ARE YOU")
#new_function = models.PyFunction.objects.get(id=1)
#math_actions_temp = {
#    "subtract":"subtracting",
#    "multiply":"multiplying",
#    "divide":"dividing"
#}
#print("HEY! I AM back")
#if new_function.name not in list(math_actions_temp.keys()):
#    math_actions_temp[new_function.name] = new_function.fun_def
#    print("HURRAY!I AM HAPPY")
#    print(math_actions_temp)
#    print("I AM MORE HAPPY")
#sys.path = os.path.dirnamr(__file__)
#updating ends

if __name__ == "__main__":
    actions = {'add': add,
               'subtract': subtract, 
               'multiply': multiply, 
               'divide': divide }
    epsilon = 0.85
    ap = ActionPlanner(actions,epsilon)

    s = {('value', 'v1'): -1.03}
    explain = -2.05
    
    #plan = ap.explain_value(s, explain)

    #print(plan)
    #print(execute_plan(plan, s, actions))

    extra = {}
    extra['actions'] = actions
    extra['epsilon'] = epsilon
    extra['tested'] = set()

    problem = ActionPlannerProblem((tuple(s.items()), None, explain), extra=extra)
    problem2 = NoHeuristic((tuple(s.items()), None, explain), extra=extra)

    #print(s)
    def cost_limited(problem):
        return best_first_search(problem, cost_limit=4)

    #count = 0
    #for sol in cost_limited(problem):
    #    state, chosen, goal = sol.state
    #    print(chosen, sol.cost())
    #    count += 1
    #    #if count > 10:
    #    #    break

    compare_searches([problem, problem2], [cost_limited])
