"""
This module will test initial creation and do simple train/request tests on
all possible configurations of agent and learner types. The goal is mainly to
be a break glass sanity check, note to test behavior.
"""
import json
import logging
import pytest
from apprentice.agents.ModularAgent import ModularAgent
from apprentice.working_memory.representation import Sai

logging.disable()


def train_A_req_A(agent):
    add_set = json.load(open('data_for_test.json', 'r'))['addition']

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
    add_set = json.load(open('data_for_test.json', 'r'))['addition']
    return agent.request(add_set[-1]['state'])


@pytest.mark.parametrize("when", ['decision_tree', 'cobweb', 'trestle'])
@pytest.mark.parametrize("where", ['version_space',
                                   'fastmostspecific',
                                   'mostspecific',
                                   # 'stateresponselearner',
                                   # 'relationallearner',
                                   'specifictogeneral'])
@pytest.mark.parametrize("which", ['proportioncorrect', 'totalcorrect'])
@pytest.mark.parametrize("choice", ['first', 'mostparsimonious', 'all', 'random'])
def test_numbert_configs(when, where, which, choice):
    function_set = ["RipFloatValue", "Add", "Subtract", "Multiply", "Divide"]
    feature_set = ["Equals"]
    agent = ModularAgent(feature_set, function_set,
                         when_learner=when,
                         where_learner=where,
                         planner='numbert',
                         heuristic_learner=which,
                         explanation_choice=choice,
                         search_depth=2)
    resp = req_A(agent)
    assert resp == {}

    resp, ans = train_A_req_A(agent)
    assert resp['selection'] == ans['selection']
    assert resp['action'] == ans['action']
    assert str(int(resp['inputs']['value'])) == ans['inputs']['value']
