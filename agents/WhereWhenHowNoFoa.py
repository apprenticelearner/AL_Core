from pprint import pprint
from random import random
from random import choice

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
from concept_formation.structure_mapper import rename_flat

from agents.BaseAgent import BaseAgent
from learners.WhenLearner import get_when_learner
from learners.WhereLearner import get_where_learner
from planners.fo_planner import FoPlanner
# from ilp.fo_planner import Operator

# from planners.rulesets import function_sets
# from planners.rulesets import feature_sets

# search_depth = 1
# epsilon = .9


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

class WhereWhenHowNoFoa(BaseAgent):
    """
    This is the basis for the 2 mechanism model.
    """
    def __init__(self, feature_set, function_set, 
                 when_learner='cobweb', where_learner='MostSpecific',
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
        for skill_label in self.skills:
            for exp in self.skills[skill_label]:
                pos = self.skills[skill_label][exp]['where'].num_pos()
                neg = self.skills[skill_label][exp]['where'].num_neg()

                skillset.append((pos / (pos + neg), pos + neg,
                                 random(), skill_label, exp,
                                 self.skills[skill_label][exp]))
        skillset.sort(reverse=True)

        # used for grounding out plans, don't need to build up each time.
        knowledge_base = FoPlanner([(ground(a), state[a].replace('?', 'QM')
                                     if isinstance(state[a], str)
                                     else state[a])
                                    for a in state],
                                   self.function_set)
        knowledge_base.fc_infer(depth=self.search_depth, epsilon=self.epsilon)

        # TODO - would it be too expensive to make skillset contain some kind of Skill object?
        # because this for loop is ridiculous
        for _, _, _, skill_label, (exp, input_args), skill in skillset:

            # Continue until we either do something with the rule. For example,
            # generate an SAI or determine that the rule doesn't match. If we
            # get some kind of failure, such as being unable to execute an
            # action sequence, then we want to learn from that and try again.
            failed = True

            while failed:

                failed = False
                for match in skill['where'].get_matches(state, epsilon=self.epsilon):
                    if len(match) != len(set(match)):
                        continue

                    # print("MATCH FOUND", label, exp, m)
                    vmapping = {'?foa' + str(i): ele for i, ele in enumerate(match)}
                    mapping = {'foa' + str(i): ele for i, ele in enumerate(match)}

                    r_exp = list(rename_flat({exp: True}, vmapping))[0]
                    r_state = rename_flat(state, {mapping[a]: a for a in mapping})

                    rg_exp = []
                    for ele in r_exp:
                        if isinstance(ele, tuple):
                            for var_match in knowledge_base.fc_query([(ground(ele), '?v')],
                                                                     max_depth=0,
                                                                     epsilon=self.epsilon):
                                if var_match['?v'] != '':
                                    rg_exp.append(var_match['?v'])
                                break
                        else:
                            rg_exp.append(ele)

                    if len(rg_exp) != len(r_exp):
                        continue

                    prediction = skill['when'].predict([r_state])[0]

                    if prediction <= 0:
                        continue

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

    def train(self, state, selection, action, inputs, reward, skill_label,
              foci_of_attention):
        """
        Doc String
        """
        print('skill_label', skill_label)
        print('selection', selection)
        print('action', action)
        print('input', inputs)
        print('reward', reward)

        # label = 'math'

        # create example dict
        example = {}
        example['state'] = state
        example['skill_label'] = skill_label
        example['selection'] = selection
        example['action'] = action
        example['inputs'] = inputs
        example['reward'] = reward

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

        explainations = []
        secondary_explainations = []

        # the base explaination (the constants)
        input_args = tuple(sorted([arg for arg in inputs]))
        sai = ('sai', selection, action, *[inputs[a] for a in input_args])

        # Need to do stuff with features here too.

        # used for grounding out plans, don't need to build up each time.
        # print(function_sets[self.action_set])
        knowledge_base = FoPlanner([(ground(a),
                                     example['flat_state'][a].replace('?', 'QM')
                                     if isinstance(example['flat_state'][a], str)
                                     else example['flat_state'][a])
                                    for a in example['flat_state']],
                                   self.function_set)
        knowledge_base.fc_infer(depth=self.search_depth, epsilon=self.epsilon)
        # FACTS AFTER USING FUNCTIONS.
        # pprint(knowledge_base.facts)

        for exp, iargs in self.skills[skill_label]:
            for match in explain_sai(knowledge_base, exp, sai, self.epsilon):
                # print("COVERED", exp, m)

                # Need to check if it would have been actully generated
                # under where and when.
                r_exp = unground(list(rename_flat({exp: True}, match))[0])
                args = get_vars(exp)

                if len(args) != len(match):
                    continue

                grounded = True
                for ele in match:
                    if not isinstance(match[ele], str):
                        grounded = False
                        break
                if not grounded:
                    continue

                tup = tuple([match["?foa%s" % i].replace("QM", "?") for i in range(len(match))])

                if len(tup) != len(set(tup)):
                    continue

                secondary_explainations.append(r_exp)

                skill_where = self.skills[skill_label][(exp, iargs)]['where']
                if not skill_where.check_match(tup, example['flat_state']):
                    continue

                # print("ADDING", r_exp)
                explainations.append(r_exp)

        if len(explainations) == 0 and len(secondary_explainations) > 0:
            explainations.append(choice(secondary_explainations))

        elif len(explainations) == 0:
            selection_exp = selection
            for sel_match in knowledge_base.fc_query([('?selection', selection)],
                                                     max_depth=0,
                                                     epsilon=self.epsilon):
                selection_exp = sel_match['?selection']
                break

            input_exps = []

            for arg in input_args:
                input_exp = input_val = inputs[arg]
                # print('trying to explain', [((a, '?input'), iv)])

                # TODO not sure what the best approach is for choosing among
                # the possible explanations. Perhaps we should choose more than
                # one. Maybe the shortest (less deep).

                # f = False
                possible = []
                for iv_m in knowledge_base.fc_query([((arg, '?input'), input_val)],
                                                    max_depth=0,
                                                    epsilon=self.epsilon):

                    possible.append((arg, iv_m['?input']))

                possible = [(compute_exp_depth(p), random(), p) for p in possible]
                possible.sort()

                if len(possible) > 0:
                    _, _, input_exp = possible[0]

                input_exps.append(input_exp)

            explainations.append(unground(('sai', selection_exp, action, *input_exps)))

        for exp in explainations:
            args = get_vars(exp)
            foa_vmapping = {field: '?foa%s' % j for j, field in enumerate(args)}
            foa_mapping = {field: 'foa%s' % j for j, field in enumerate(args)}
            r_exp = (list(rename_flat({exp: True}, foa_vmapping))[0], input_args)

            if r_exp not in self.skills[skill_label]:

                #TODO - Hack for specific action set
                # if self.action_set == "tutor knowledge":
                #     constraints = self.generate_tutor_constraints(r_exp[0])
                # else:
                #     constraints = self.extract_mg_h(r_exp[0])
                constraints = extract_mg_h(r_exp[0])

                print("CONSTRAINTS")
                print(constraints)

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

        # check for subsuming explainations (alternatively we could probably
        # just order the explainations by how many examples they cover


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
