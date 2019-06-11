from pprint import pprint
from random import random
from random import choice

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
from concept_formation.structure_mapper import rename_flat

from agents.BaseAgent import BaseAgent
from learners.WhenLearner import get_when_learner
from learners.WhereLearner import get_where_learner
from planners.fo_planner import FoPlanner, execute_functions, unify, subst
# from ilp.fo_planner import Operator

# from planners.rulesets import function_sets
# from planners.rulesets import feature_sets

# search_depth = 1
# epsilon = .9




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
                operator_output = apply_operators(ele,function_set,knowledge_base,epsilon)
                # print("OPERATOR_OUTPUT", ele, "->", operator_output)
                if(operator_output != None and operator_output != ""):
                    # print("O", operator_output)
                    rg_exp.append(operator_output)
                    ok = True

            if(not ok):
                rg_exp.append(None)
                # print("HERE_B",operator_output)
            


            # if(operator_output != None):
            #     rg_exp.append(operator_output)


        else:
            rg_exp.append(ele)
    return rg_exp


#TODO MAKE RECURSIVE SO THAT CAN USE HIGHER DEPTHS
def apply_operators(ele, operators, knowledge_base,epsilon):
    operator_output = None
    for operator in operators:
        effect = list(operator.effects)[0]
        pattern = effect[0]
        u_mapping = unify(pattern, ground(ele), {}) #Returns a mapping to name the values in the expression
        if(u_mapping):
            # print(operator.conditions,"\n")
            # print(u_mapping,"\n")
            # print(effect[1],"\n")
            # print("BEEP", [subst(u_mapping,x) for x in operator.conditions],"\n")
            condition_sub = [subst(u_mapping,x) for x in operator.conditions]
            # print("CS", condition_sub)



            value_map = next(knowledge_base.fc_query(condition_sub, max_depth=0, epsilon=epsilon))
            # print(value_map)
            # print()
            try:
                operator_output = execute_functions(subst(value_map,effect[1]))
            except:
                continue

            return operator_output

class WhereWhenHowNoFoa(BaseAgent):
    """
    This is the basis for the 2 mechanism model.
    """
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

    def request(self, state):
        """
        Doc String
        TODO - several Linter problems with this one
        """
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

        skillset = []

        # pprint(self.skills)

        for skill_label in self.skills:
            for exp in self.skills[skill_label]:
                pos = self.skills[skill_label][exp]['where'].num_pos()
                neg = self.skills[skill_label][exp]['where'].num_neg()

                skillset.append((pos / (pos + neg), pos + neg,
                                 random(), skill_label, exp,
                                 self.skills[skill_label][exp]))
        skillset.sort(reverse=True)

        # used for grounding out plans, don't need to build up each time.
        # knowledge_base = FoPlanner([(ground(a), state[a].replace('?', 'QM')
        #                              if isinstance(state[a], str)
        #                              else state[a])
        #                             for a in state],
        #                            self.function_set)
        # knowledge_base.fc_infer(depth=self.search_depth, epsilon=self.epsilon)

        # TODO - would it be too expensive to make skillset contain some kind of Skill object?
        # because this for loop is ridiculous
        for _, _, _, skill_label, (exp, input_args), skill in skillset:


            # print("STATE")
            # pprint(state)
            # print("--------")
            # print(exp)
            # print("contentEditable: ",state.get( ('contentEditable',"?ele-" + exp[1]) ,True) )
            # if(state.get( ('contentEditable',"?ele-" + exp[1]) ,True) == False):
            #     continue
            # print("")
            # print("REQUEST_EXPLAINS", exp)

            # Continue until we either do something with the rule. For example,
            # generate an SAI or determine that the rule doesn't match. If we
            # get some kind of failure, such as being unable to execute an
            # action sequence, then we want to learn from that and try again.
            failed = True

            # print("SKILL: ",exp, input_args)

            while failed:

                failed = False

                # print("STATE")
                # pprint(state)
                # print("--------")

                for match in skill['where'].get_matches(state, epsilon=self.epsilon):
                    # print("REE1")
                    if len(match) != len(set(match)):
                        continue

                    # print("MATCH FOUND", skill_label, exp, match)
                    vmapping = {'?foa' + str(i): ele for i, ele in enumerate(match)}
                    mapping = {'foa' + str(i): ele for i, ele in enumerate(match)}

                    # print("VMAP")
                    # pprint(vmapping)
                    # print("MAP")
                    # pprint(mapping)

                    r_exp = list(rename_flat({exp: True}, vmapping))[0]
                    r_state = rename_flat(state, {mapping[a]: a for a in mapping})


                    # print("KB", knowledge_base)
                    rg_exp = eval_expression(r_exp,knowledge_base,self.function_set,self.epsilon)
                    # for ele in r_exp:
                    #     if isinstance(ele, tuple):
                    #         # print("THIS HAPPENED", ele, ground(ele), execute_functions(ground(ele)))

                            
                                    
                    #                 # execute_functions(subt(u))
                    #                 # rg_exp.append()

                    #             # print("BLEHH:", operator.effects) 
                            
                    #         for var_match in knowledge_base.fc_query([(ground(ele), '?v')],
                    #                                                  max_depth=0,
                    #                                                   epsilon=self.epsilon):
                    #             # print("VARM:",var_match, ground(ele))
                    #             if var_match['?v'] != '':
                    #                 rg_exp.append(var_match['?v'])
                    #                 # print("HERE_A",rg_exp[-1])
                    #             break

                    #         operator_output = apply_operators(ele,self.function_set,knowledge_base,self.epsilon)
                    #         # print(operator_output)
                    #         if(operator_output != None and operator_output != ""):
                    #             rg_exp.append(operator_output)
                    #             # print("HERE_B",operator_output)
                            


                    #         # if(operator_output != None):
                    #         #     rg_exp.append(operator_output)


                    #     else:
                    #         rg_exp.append(ele)

                    # print("REE2")


                    # print("rg_exp:", rg_exp)
                    # print("r_exp:", r_exp)
                    
                    if len(rg_exp) != len(r_exp):
                        continue

                    # print("EXP:", r_exp)
                    # print("RSTATE ---------------")
                    # pprint(r_state)
                    # print("---------------")

                    # print("REE3")
                    prediction = skill['when'].predict([r_state])[0]

                    # print("when", skill['when'])

                    # print("PREDICTION:", type(prediction), prediction)

                    if prediction <= 0:
                        continue
                    # print("REE4")
                    response = {}
                    response['skill_label'] = skill_label
                    response['selection'] = rg_exp[1]
                    response['action'] = rg_exp[2]
                    response['inputs'] = {a: rg_exp[3 + i] for i, a in
                                          enumerate(input_args)}
                    # response['inputs'] = list(rg_exp[3:])
                    response['foci_of_attention'] = []
                    # pprint(response)
                    return response

        return {}

    def explain_sai(self, skill, learner_dict, sai, knowledge_base,state):
        """
        Doc String
        """
        # print(skill)
        # print(skill['where'])
        exp, iargs = skill
        skill_where = learner_dict['where']

#
        # print("SAI", sai)
        for match in skill_where.get_matches(state):
            # print(exp)
            # print()
            # print(m)
            # print("CLOOOL",get_vars(exp), exp)
            mapping = {var:val for var,val in zip(get_vars(exp),match)}
            grounded_exp = subst(mapping, exp)
            rg_exp = tuple(eval_expression(grounded_exp,knowledge_base,self.function_set,self.epsilon))

            # print("RG_EXP", rg_exp)

            if(rg_exp == sai):
                yield mapping
            # if()
            

            # apply_operators(grounded_exp,self.function_set, knowledge_base, self.epsilon)

        # if not skill_where.check_match(tup, example['flat_state']):
            # continue
        # print('trying', exp, 'with', sai)
        # # print(skill_where)

        # if len(exp) != len(sai):
        #     return

        # goals = []
        # for i, elem in enumerate(exp):
        #     if elem == sai[i]:
        #         continue
        #     else:
        #         goals.append((elem, sai[i]))

        # # print("GOALS" ,goals)

        # for match in knowledge_base.fc_query(goals, max_depth=0, epsilon=self.epsilon):
        #     yield match
    def explanations_from_how_search(self,state,sai,input_args):
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

            # TODO not sure what the best approach is for choosing among
            # the possible explanations. Perhaps we should choose more than
            # one. Maybe the shortest (less deep).

            # f = False

            #Danny: This populates a list of explanations found earlier in How search that work"
            possible = []
            for iv_m in knowledge_base.fc_query([((arg, '?input'), input_val)],
                                                max_depth=0,
                                                epsilon=self.epsilon):

                # print("iv_m:", iv_m)

                possible.append((arg, iv_m['?input']))

            possible = [(compute_exp_depth(p), random(), p) for p in possible]
            possible.sort()

            if len(possible) > 0:
                _, _, input_exp = possible[0]

            input_exps.append(input_exp)
            print("input_exp:", input_exp)

        explanations.append(unground(('sai', selection_exp, action, *input_exps)))
        print("EXPLANATIONS:", explanations)
        return explanations


    def train(self, state, selection, action, inputs, reward, skill_label,
              foci_of_attention):
        """
        Doc String
        """

        # print('\n'*5)
        # print('state', skill_label)
        # print('skill_label', skill_label)
        # print('selection', selection)
        # print('action', action)
        # print('inputs', inputs)
        # print('reward', reward)
        # print('state')
        # pprint(state)

        # label = 'math'

        # create example dict
        example = {}
        example['state'] = state
        example['skill_label'] = skill_label
        example['selection'] = selection
        example['action'] = action
        example['inputs'] = inputs
        example['reward'] = float(reward)

        tup = Tuplizer()
        flt = Flattener()
        example['flat_state'] = flt.transform(tup.transform(state))

        knowledge_base = FoPlanner([(ground(a), example['flat_state'][a].replace('?', 'QM')
                                     if isinstance(example['flat_state'][a], str)
                                     else example['flat_state'][a])
                                    for a in example['flat_state']],
                                   self.feature_set)

        knowledge_base.fc_infer(depth=1, epsilon=self.epsilon)

        example['flat_state'] = {unground(a): v.replace("QM", "?")
                                 if isinstance(v, str)
                                 else v
                                 for a, v in knowledge_base.facts}

        if skill_label not in self.skills:
            self.skills[skill_label] = {}

        explanations = []
        secondary_explanations = []

        # the base explaination (the constants)
        input_args = tuple(sorted([arg for arg in inputs]))
        sai = ('sai', selection, action, *[inputs[a] for a in input_args])

        # Need to do stuff with features here too.

        # used for grounding out plans, don't need to build up each time.
        # print(function_sets[self.action_set])

        # knowledge_base = FoPlanner([(ground(a),
        #                              example['flat_state'][a].replace('?', 'QM')
        #                              if isinstance(example['flat_state'][a], str)
        #                              else example['flat_state'][a])
        #                             for a in example['flat_state']],
        #                            self.function_set)
        # knowledge_base.fc_infer(depth=self.search_depth, epsilon=self.epsilon)
        
        # FACTS AFTER USING FUNCTIONS.
        # pprint(knowledge_base.facts)


        #DANNY: The gist of this is to find_ applicable_skills (explanations because the inputs/outputs are literals)
        for skill,learner_dict in self.skills[skill_label].items():
            exp, iargs = skill
            for match in self.explain_sai(skill, learner_dict, sai, knowledge_base, example['flat_state']):

                # print("MATCH", match)
                # print("COVERED", exp, m)

                #DANNY NOTES:
                #sai = ('sai', 'JCommTable5.R0C0', 'UpdateTextArea', '16')
                #exp  =  ('sai', ('id', '?foa0'), 'UpdateTextArea', ('value', ('Multiply', ('value', '?foa1'), ('value', '?foa2'))))
                #r_exp  =  the explanation renamed with the match literals 
                #ARGS = A list of input arguements to the explanations
                #IARGS: = A tuple of outputs? Or maybe a set of property names (i.e. 'value').
                #Match = A dictionary mapping from ?foas to element strings (the element names have QM instead of ?)

                
                r_exp = unground(list(rename_flat({exp: True}, match))[0])
                args = get_vars(exp)

                # print("sai:", sai)
                # print("exp:", exp)
                # print("r_exp:", r_exp)
                # print("ARGS:", args)
                # print("IARGS:", iargs, type(iargs))
                # print("match", match)


                # Need to check if it would have been actully generated
                # under where and when.
                # Danny: Makes sure that there is a match for every arguement 
                if len(args) != len(match):
                    continue

                # print("HERE1")

                #DANNY: Checks that the match resolves to string elements
                grounded = True
                for ele in match:
                    if not isinstance(match[ele], str):
                        grounded = False
                        break
                if not grounded:
                    #DANNY: Doesn't really happen... not sure in what circumstances this would happen
                    # print("BAD MATCH: ", match)
                    continue

                # print("HERE2")

                tup = tuple([match["?foa%s" % i].replace("QM", "?") for i in range(len(match))])

                # print("tup:", tup)

                #tup = a tuple of the matches 
                #Danny: Makes sure that the explanation hasn't used an foa twice 
                if len(tup) != len(set(tup)):
                    continue

                # print("HERE3")

                secondary_explanations.append(r_exp)

                #Danny: Check that the where learner approves of the match
                #       It seems like the where learner should have just generated this if it was going to check it anyway
                #       This is only at the end it seems to allow for secondary explanations which the where learner is not yet aware of
                # print("A", self.skills[skill_label])
                # print("B", self.skills[skill_label][(exp, iargs)])
                skill_where = self.skills[skill_label][(exp, iargs)]['where']
                if not skill_where.check_match(tup, example['flat_state']):
                    continue


                # print("ADDING", r_exp)
                explanations.append(r_exp)

        if len(explanations) == 0 and len(secondary_explanations) > 0:
            explanations.append(choice(secondary_explanations))

        elif len(explanations) == 0:

            explanations = self.explanations_from_how_search(example['flat_state'], ('sai', selection, action, inputs) ,input_args)

        #Danny: Do the training for all the applicable explanations
        # print("EXPLAINS", explanations)
        for exp in explanations:
            args = get_vars(exp)
            foa_vmapping = {field: '?foa%s' % j for j, field in enumerate(args)}
            foa_mapping = {field: 'foa%s' % j for j, field in enumerate(args)}
            r_exp = (list(rename_flat({exp: True}, foa_vmapping))[0], input_args)

            if r_exp not in self.skills[skill_label]:

                # TODO - Hack for specific action set
                # if self.action_set == "tutor knowledge":
                #     constraints = self.generate_tutor_constraints(r_exp[0])
                # else:
                #     constraints = self.extract_mg_h(r_exp[0])
                # constraints = extract_mg_h(r_exp[0])
                constraints = generate_html_tutor_constraints(r_exp[0])

                # print("CONSTRAINTS")
                # print(constraints)

                w_args = tuple(['?foa%s' % j for j, _ in enumerate(args)])

                self.skills[skill_label][r_exp] = {}
                where_inst = self.where(args=w_args, constraints=constraints)
                self.skills[skill_label][r_exp]['where'] = where_inst
                self.skills[skill_label][r_exp]['when'] = self.when()

            self.skills[skill_label][r_exp]['where'].ifit(args, example['flat_state'], example['reward'])
            # print('done where learning')

            # TODO
            # Need to add computed features.
            # need to rename example with foa's that are not variables
            r_flat = rename_flat(example['flat_state'], foa_mapping)

            self.skills[skill_label][r_exp]['when'].ifit(r_flat, example['reward'])

        # check for subsuming explanations (alternatively we could probably
        # just order the explanations by how many examples they cover


    def check(self, state, selection, action, inputs):
        """
        Doc String.
        """
        #TODO - implement check
        return False

#TODO - all stuff below is hacks for specific actionsets

def generate_tutor_constraints(sai):
    """
    Given an SAI, this finds a set of constraints for the SAI, so it don't
    fire in nonsensical situations.
    """
    constraints = set()
    args = get_vars(sai)

    # selection constraints, you can only select something that has an
    # empty string value.

    if len(args) == 0:
        return frozenset()

    # print("SAI", sai)
    # print("ARGS", args)
    selection = args[0]
    constraints.add(('value', selection, '?selection-value'))
    constraints.add((is_empty_string, '?selection-value'))

    # get action
    action = sai[2]
    if action == "ButtonPressed":
        # Constrain the selection to be of type button
        constraints.add(('type', selection, 'MAIN::button'))
        constraints.add(('name', selection, 'done'))
    else:
        # constraints.add(('not', ('type', selection, 'MAIN::button')))
        constraints.add(('not', ('type', selection, 'MAIN::button')))
        constraints.add(('not', ('type', selection, 'MAIN::label')))
        # constraints.add(('type', selection, 'MAIN::cell'))

    # value constraints, don't select empty values
    for i, arg in enumerate(args[1:]):
        constraints.add(('value', arg, '?foa%ival' % (i+1)))
        constraints.add((is_not_empty_string, '?foa%ival' % (i+1)))
        # constraints.add(('type', a, 'MAIN::cell'))

    return frozenset(constraints)


def generate_html_tutor_constraints(sai):
    """
    Given an SAI, this finds a set of constraints for the SAI, so it don't
    fire in nonsensical situations.
    """
    constraints = set()
    args = get_vars(sai)

    # selection constraints, you can only select something that has an
    # empty string value.

    if len(args) == 0:
        return frozenset()

    # get action
    action = sai[2]
    if action == "ButtonPressed":
        # Constrain the selection to be of type button
        # constraints.add(('type', selection, 'MAIN::button'))
        selection = args[0]
        constraints.add(('id', selection, 'done'))
    else:
        # print("SAI", sai)
        # print("ARGS", args)
        selection = args[0]
        constraints.add(('contentEditable', selection, True))
        # constraints.add(('value', selection, '?selection-value'))
        # constraints.add((is_empty_string, '?selection-value'))

    # value constraints, don't select empty values
    for i, arg in enumerate(args[1:]):
        constraints.add(('value', arg, '?foa%ival' % (i+1)))
        constraints.add((is_not_empty_string, '?foa%ival' % (i+1)))
        # constraints.add(('type', a, 'MAIN::cell'))

    return frozenset(constraints)

def extract_mg_h(sai):
    """
    Given an SAI, this find the most general pattern that will generate a
    match.

    E.g., ('sai', ('name', '?foa0'), 'UpdateTable', ('value', '?foa1'))
    will yield: {('name', '?foa0'), ('value', '?foa1')}
    """
    return frozenset({tuple(list(elem) + ['?constraint-val%i' % i])
                      for i, elem in enumerate(sai)
                      if isinstance(elem, tuple)})

def is_empty_string(sting):
    return sting == ''

def is_not_empty_string(sting):
    return sting != ''
