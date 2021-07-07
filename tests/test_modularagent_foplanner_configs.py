"""
This module will test initial creation and do simple train/request tests on
all possible configurations of agent and learner types. The goal is mainly to
be a break glass sanity check, note to test behavior.
"""
import json
import logging
import pytest
from os.path import dirname
from os.path import join
from apprentice.agents.ModularAgent import ModularAgent
from apprentice.working_memory.representation import Sai
from apprentice.working_memory.fo_planner_operators import add_rule, sub_rule, mult_rule, div_rule, equal_rule

logging.disable()
module_path = dirname(__file__)


def train_A_req_A(agent):
    add_set = json.load(open(join(module_path, 'data_for_test.json'), 'r'))['addition']

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
    add_set = json.load(open(join(module_path, 'data_for_test.json'), 'r'))['addition']
    return agent.request(add_set[-1]['state'])


@pytest.mark.parametrize("when", ['decision_tree', 'cobweb', 'trestle'])
@pytest.mark.parametrize("where", ['version_space',
                                   'fastmostspecific',
                                   'mostspecific',])
                                   # 'stateresponselearner',
                                   # 'relationallearner',
                                   # 'specifictogeneral'])
@pytest.mark.parametrize("which", ['proportioncorrect', 'totalcorrect'])
@pytest.mark.parametrize("choice", ['first', 'mostparsimonious', 'all', 'random'])
def test_fo_planner_configs(when, where, which, choice):
    function_set = [add_rule, sub_rule, mult_rule, div_rule]
    feature_set = [equal_rule]
    agent = ModularAgent(feature_set, function_set,
                         when_learner=when,
                         where_learner=where,
                         planner='fo_planner',
                         heuristic_learner=which,
                         explanation_choice=choice)
    resp = req_A(agent)
    assert resp == {}

    resp, ans = train_A_req_A(agent)
    assert resp['selection'] == ans['selection']
    assert resp['action'] == ans['action']
    assert resp['inputs']['value'] == ans['inputs']['value']
