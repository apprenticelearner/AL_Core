"""
This module tests dill based serialization and deserialization of all common
agent configurations. It tries to confirm that a deserialized agent will
exhibit equivalent behavior to its origin.
"""

import dill
import copy
import json
import logging
import pytest
from os.path import dirname
from os.path import join
from pprint import pprint

from apprentice.agents.ModularAgent import ModularAgent
from apprentice.agents.WhereWhenHowNoFoa import WhereWhenHowNoFoa
from apprentice.working_memory.representation import Sai
from apprentice.working_memory.fo_planner_operators import add_rule, sub_rule, mult_rule, div_rule, equal_rule

logging.disable()
module_path = dirname(__file__)


def areAgentsSame(agent1, agent2):
    """Check that agents' underlying structures are the same"""
    agent1dict = agent1.__dict__
    agent2dict = agent2.__dict__

    to_find_common = {'constraint_generator', 'epsilon', 'feature_set', 'function_set', 'last_state',
                      'ret_train_expl', 'rhs_by_label', 'rhs_counter', 'search_depth', 'rhs_list'}
    for common in to_find_common:
        assert agent1dict[common] == agent2dict[common], "Structural equivalence failed on: {}".format(common)

    # planner is not equal ,'planner'
    assert(agent1dict['state_variablizer'].__dict__ ==
           agent2dict['state_variablizer'].__dict__)

    # which learner
    which_learner = agent1dict['which_learner']
    for key in which_learner.__dict__:
        assert(True)

    # when learner:
    when_learner = agent1dict['when_learner']
    for key in when_learner.__dict__:
        assert(True)

    # where learner
    where_learner = agent1dict['where_learner']
    for key in where_learner.__dict__:
        assert(True)

    return True


def train_A_req_A(agent):
    add_set = json.load(open(join(module_path, 'data_for_test.json'), 'r'))[
        'addition']

    for data in add_set[:-1]:
        sai = Sai(
            selection=data["selection"],
            action=data["action"],
            inputs=data["inputs"],
        )

        data['sai'] = sai
        del data['selection']
        del data['action']
        del data['inputs']

        agent.train(**data)

    return agent.request(add_set[-1]['state']), add_set[-1]


def req_A(agent):
    add_set = json.load(open(join(module_path, 'data_for_test.json'), 'r'))[
        'addition']
    return agent.request(add_set[-1]['state'])


def run_save_test(agent, verify_structure=True):
    resp, ans = train_A_req_A(agent)

    # check that deserialized version is the same
    agent_stream = dill.dumps(agent)
    agent_deserialized = dill.loads(agent_stream)

    if verify_structure:
        assert(areAgentsSame(agent, agent_deserialized))

    # make another request
    resp = req_A(agent)
    resp2 = req_A(agent_deserialized)
    assert(resp == resp2)

    # train the agents again
    resp, ans = train_A_req_A(agent)
    resp2, ans2 = train_A_req_A(agent_deserialized)

    # where the state would be wrong
    wrongAnswerState = {"arg1": {"above": "", "below": "hintWindow", "contentEditable": False, "dom_class": "CTATTextInput", "id": "arg1", "to_left": "", "to_right": "operator", "type": "TextField", "value": "2"},
                        "arg2": {"above": "", "below": "hintWindow", "contentEditable": False, "dom_class": "CTATTextInput", "id": "arg2", "to_left": "operator", "to_right": "answer", "type": "TextField", "value": "3"},
                        "answer": {"above": "", "below": "hint", "contentEditable": True, "dom_class": "CTATTextInput", "id": "answer", "to_left": "arg2", "to_right": "", "type": "TextField", "value": ""},
                        "hintWindow": {"above": "operator", "below": "", "dom_class": "CTATHintWindow", "id": "hintWindow", "to_left": "", "to_right": "hint", "type": "Component"},
                        "done": {"above": "hint", "below": "", "dom_class": "CTATDoneButton", "id": "done", "to_left": "hintWindow", "to_right": "", "type": "Component"},
                        "hint": {"above": "answer", "below": "done", "dom_class": "CTATHintButton", "id": "hint", "to_left": "hintWindow", "to_right": "", "type": "Component"},
                        "operator": {"above": "", "below": "hintWindow", "contentEditable": False, "dom_class": "CTATTextInput", "id": "operator", "to_left": "arg1", "to_right": "arg2", "type": "TextField", "value": "+"}}

    # request with the wrongAnswerState and check sai
    resp = agent.request(wrongAnswerState)
    resp2 = agent_deserialized.request(wrongAnswerState)
    assert(resp == resp2)


@pytest.mark.parametrize("when", ['decision_tree', 'cobweb', 'trestle'])
@pytest.mark.parametrize("where", ['version_space',
                                   'fastmostspecific',
                                   'mostspecific', ])
@pytest.mark.parametrize("which", ['proportioncorrect', 'totalcorrect'])
@pytest.mark.parametrize("choice", ['first', 'mostparsimonious', 'all', 'random'])
def test_fo_planner_saves(when, where, which, choice):
    function_set = [add_rule, sub_rule, mult_rule, div_rule]
    feature_set = [equal_rule]
    agent = ModularAgent(feature_set, function_set,
                         when_learner=when,
                         where_learner=where,
                         planner='fo_planner',
                         heuristic_learner=which,
                         explanation_choice=choice)
    run_save_test(agent, verify_structure=False)


@pytest.mark.parametrize("when", ['decision_tree', 'cobweb', 'trestle'])
@pytest.mark.parametrize("where", ['version_space',
                                   'fastmostspecific',
                                   'mostspecific'])
@pytest.mark.parametrize("which", ['proportioncorrect', 'totalcorrect'])
@pytest.mark.parametrize("choice", ['first', 'mostparsimonious', 'all', 'random'])
def test_numbert_saves(when, where, which, choice):
    function_set = ["RipFloatValue", "Add", "Subtract", "Multiply", "Divide"]
    feature_set = ["Equals"]
    agent = ModularAgent(feature_set, function_set,
                         when_learner=when,
                         where_learner=where,
                         planner='numbert',
                         heuristic_learner=which,
                         explanation_choice=choice,
                         search_depth=2)
    run_save_test(agent)
