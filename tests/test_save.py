# try:  # python 3
#     from apprentice_learner.models import Agent
# except ImportError:  # python 2
#     import apprentice_learner.models as Agent

from apprentice.agents.ModularAgent import ModularAgent
from apprentice.agents.WhereWhenHowNoFoa import WhereWhenHowNoFoa
from apprentice.working_memory.representation import Sai
import dill
import copy
import json
import logging
import pytest
from os.path import dirname
from os.path import join


#check if agents contain same stuff
def areAgentsSame(agent1, agent2):
    agent1dict = agent1.__dict__
    agent2dict = agent2.__dict__

    to_find_common = {'constraint_generator', 'epsilon', 'feature_set', 'function_set', 'last_state', 
    'ret_train_expl', 'rhs_by_label', 'rhs_counter', 'search_depth', 'rhs_list'}
    for common in to_find_common:
        assert(agent1dict[common] == agent2dict[common])
    
    #checking rhs_by_how
    # for key in agent1dict['rhs_by_how']:
    #     assert(agent1dict['rhs_by_how'][key] == agent2dict['rhs_by_how'][key])
# E       AssertionError: assert {None: {('answer', 'UpdateTextField', Add(?.v,?.v)): Add(?.v,?.v)}} == {None: {('answer', 'UpdateTextField', Add(?.v,?.v)): Add(?.v,?.v)}}
# E         Differing items:
# E         {None: {('answer', 'UpdateTextField', Add(?.v,?.v)): Add(?.v,?.v)}} != {None: {('answer', 'UpdateTextField', Add(?.v,?.v)): Add(?.v,?.v)}}
# E         Full diff:
# E           {None: {('answer', 'UpdateTextField', Add(?.v,?.v)): Add(?.v,?.v)}}

    #planner is not equal ,'planner'
    assert(agent1dict['state_variablizer'].__dict__ == agent2dict['state_variablizer'].__dict__)

    #which learner
    which_learner = agent1dict['which_learner']
    for key in which_learner.__dict__:
        assert(True)
        # if isinstance(agent1dict['which_learner'].__dict__[key], WhichLearner.TotalCorrect):
            # assert(agent1dict['which_learner'][key].__dict__ == agent2dict['when_learner'][key].__dict__)
    #     else:
    #         assert(agent1dict['which_learner'][key] == agent2dict['which_learner'][key])
        # if key != 'learners':
        #     assert(agent1dict['which_learner'].__dict__[key] == agent2dict['which_learner'][key])

    #'learners': {Add(?.v,?.v): <apprentice.learners.WhichLearner.TotalCorrect object at 0x7f8d70e26dc0>}} != {'learners': {Add(?.v,?.v): <apprentice.learners.WhichLearner.TotalCorrect object at 0x7f8d70ecc820>}}


    #when learner: 
    when_learner = agent1dict['when_learner']
    for key in when_learner.__dict__:
        # if isinstance(when_learner[key], WhenLeaner.ScikitCobweb):
        #     assert(True)
        # else:
        assert(True)
    #{'sub_learners': {1: <apprentice.learners.WhenLearner.ScikitCobweb object at 0x7ff052a10b80>}} != {'sub_learners': {1: <apprentice.learners.WhenLearner.ScikitCobweb object at 0x7ff052a7ab80>}}

    #where learner
    where_learner = agent1dict['where_learner']
    for key in where_learner.__dict__:
        assert(True)
    #E{'learner_kwargs': {},\n 'learner_name': 'MostSpecific',\n 'learners': {1: {('QMele-JCommTable2.R0C0',)}},\n 'rhs_by_label': {None: [1]}}
    # {'learner_kwargs': {},\n 'learner_name': 'MostSpecific',\n 'learners': {1: {('QMele-JCommTable2.R0C0',)}},\n 'rhs_by_label': {None: [1]}}

    return True


#ModularAgent()-- then request, train, give answers when dont know what to do
#when guesses and correct -- look same as when agent asks for hint and gets hint back-- reward should be 1
#needs value + cntentEditable
def test_FO_ModularAgent():
    #create agent:
    #    def __init__(self, feature_set, function_set, when_learner='decisiontree', where_learner='version_space'...
    feature_set = ["equals"]
    function_set = ["add", "subtract", "multiply", "divide"]
    agent = ModularAgent(feature_set, function_set, when_learner='trestle', where_learner='MostSpecific')

    #check class is made correctly
    assert (agent.__dict__['when_learner'].__dict__['learner_name'] == 'cobweb')

    state = {'JCommTable.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544661867957-2', 'id': 'JCommTable.R0C0', 'type': 'TextField', 'value': '15', 'contentEditable': False, 'below': '', 'above': '', 'to_right': 'JCommTable1.R0C0', 'to_left': ''}, 
            'JCommTable1.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544661867957-2', 'id': 'JCommTable1.R0C0', 'type': 'TextField', 'value': '=', 'contentEditable': False, 'below': '', 'above': '', 'to_right': 'JCommTable2.R0C0', 'to_left': 'JCommTable.R0C0'}, 
            'JCommTable2.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662032496-5', 'id': 'JCommTable2.R0C0', 'type': 'TextField', 'value': '', 'contentEditable': True, 'below': '', 'above': '', 'to_right': 'JCommTable3.R0C0', 'to_left': 'JCommTable1.R0C0'}, 
            'JCommTable3.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662032496-5', 'id': 'JCommTable3.R0C0', 'type': 'TextField', 'value': 'x', 'contentEditable': False, 'below': '', 'above': '', 'to_right': 'JCommTable4.R0C0', 'to_left': 'JCommTable2.R0C0'}, 
            'JCommTable4.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662076475-6', 'id': 'JCommTable4.R0C0', 'type': 'TextField', 'value': '', 'contentEditable': True, 'below': '', 'above': '', 'to_right': 'JCommTable5.R0C0', 'to_left': 'JCommTable3.R0C0'}, 
            'JCommTable5.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662076475-6', 'id': 'JCommTable5.R0C0', 'type': 'TextField', 'value': '+', 'contentEditable': False, 'below': '', 'above': '', 'to_right': 'JCommTable6.R0C0', 'to_left': 'JCommTable4.R0C0'}, 
            'JCommTable6.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662076475-6', 'id': 'JCommTable6.R0C0', 'type': 'TextField', 'value': '', 'contentEditable': True, 'below': '', 'above': '', 'to_right': 'JCommTable7.R0C0', 'to_left': 'JCommTable5.R0C0'}, 
            'JCommTable7.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662076475-6', 'id': 'JCommTable7.R0C0', 'type': 'TextField', 'value': "=", 'contentEditable': False, 'below': '', 'above': '', 'to_right': 'JCommTable8.R0C0', 'to_left': 'JCommTable6.R0C0'}, 
            'JCommTable8.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662076475-6', 'id': 'JCommTable8.R0C0', 'type': 'TextField', 'value': '', 'contentEditable': True, 'below': '', 'above': '', 'to_right': '', 'to_left': 'JCommTable7.R0C0'}, 
            'done': {'dom_class': 'CTATDoneButton', 'offsetParent': 'background-initial', 'id': 'done', 'type': 'Component', 'below': 'hint', 'above': '', 'to_right': '', 'to_left': 'ctatdiv87'}, 
            'hint': {'dom_class': 'CTATHintButton', 'offsetParent': 'background-initial', 'id': 'hint', 'type': 'Component', 'below': '', 'above': 'done', 'to_right': '', 'to_left': 'ctatdiv87'}, 
            'ctatdiv87': {'dom_class': 'CTATHintWindow', 'offsetParent': 'background-initial', 'id': 'ctatdiv87', 'type': 'Component', 'below': '', 'above': '', 'to_right': 'done', 'to_left': ''}}
    
    SAI = agent.request(state, reward = 1)
    assert(len(SAI) == 0)
    
    #train the agent
    #def train(self, state:Dict, sai:Sai=None, reward:float=None,...
    SAI = Sai(selection='JCommTable2.R0C0', action='UpdateTextArea', inputs={'value': '1'})
    #SAI = Sai(selection='JCommTable4.R0C0', action='UpdateTextArea', inputs={'value': '2'})
    #SAI = Sai(selection='JCommTable6.R0C0', action='UpdateTextArea', inputs={'value': '25'})
    #SAI = Sai(selection='JCommTable8.R0C0', action='UpdateTextArea', inputs={'value': '225'})


    #before and after: 
    #response: {'skill_label': None, 'selection': 'done', 'action': 'ButtonPressed', 'inputs': {'value': -1}, 'rhs_id': 3}
    #sai: Sai(selection='done', action='ButtonPressed', inputs={'value': -1})

    dc = copy.deepcopy(state)
    agent.train(dc, SAI, reward = 1)
    assert(dc == state)

    #Save the agent and deserialize
    agent_saved = dill.dumps(agent)
    agent_deserialized = dill.loads(agent_saved)

    #check that the agent is same before and after serializing
    areAgentsSame(agent, agent_deserialized)

    #planner is different right now bc of lambda
    # assert(agent.__dict__['planner'].__dict__ == agent_deserialized.__dict__['planner'].__dict__)

    #request both (train changes state?)
    SAI1 = agent.request(state, reward = 1)
    SAI2 = agent_deserialized.request(state, reward = 1)

    assert(SAI1 == SAI2)

"""
This module will test initial creation and do simple train/request tests on
all possible configurations of agent and learner types. The goal is mainly to
be a break glass sanity check, note to test behavior.
"""

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
                                   'mostspecific'
                                   # ,'stateresponselearner',
                                   # 'relationallearner',
                                   #'specifictogeneral'
                                   ])
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

    #check that copy is the same
    dc = copy.deepcopy(resp)
    assert(dc == resp)

    #check that deserialized version is the same
    agent_stream = dill.dumps(agent)
    agent_deserialized = dill.loads(agent_stream)
    assert(areAgentsSame(agent, agent_deserialized))

    #make another request
    resp = req_A(agent)
    resp2 = req_A(agent_deserialized)
    assert(resp == resp2)

    #train the agents again
    resp, ans = train_A_req_A(agent)
    resp2, ans2 = train_A_req_A(agent_deserialized)

    #where the state would be wrong
    wrongAnswerState = {"arg1": {"above": "", "below": "hintWindow", "contentEditable": False, "dom_class": "CTATTextInput", "id": "arg1", "to_left": "", "to_right": "operator", "type": "TextField", "value": "2"},
               "arg2": {"above": "", "below": "hintWindow", "contentEditable": False, "dom_class": "CTATTextInput", "id": "arg2", "to_left": "operator", "to_right": "answer", "type": "TextField", "value": "3"},
               "answer": {"above": "", "below": "hint", "contentEditable": True, "dom_class": "CTATTextInput", "id": "answer", "to_left": "arg2", "to_right": "", "type": "TextField", "value": ""},
               "hintWindow": {"above": "operator", "below": "", "dom_class": "CTATHintWindow", "id": "hintWindow", "to_left": "", "to_right": "hint", "type": "Component"},
               "done": {"above": "hint", "below": "", "dom_class": "CTATDoneButton", "id": "done", "to_left": "hintWindow", "to_right": "", "type": "Component"},
               "hint": {"above": "answer", "below": "done", "dom_class": "CTATHintButton", "id": "hint", "to_left": "hintWindow", "to_right": "", "type": "Component"},
               "operator": {"above": "", "below": "hintWindow", "contentEditable": False, "dom_class": "CTATTextInput", "id": "operator", "to_left": "arg1", "to_right": "arg2", "type": "TextField", "value": "+"}}

    #request with the wrongAnswerState and check sai
    resp = agent.request(wrongAnswerState)
    resp2 = agent_deserialized.request(wrongAnswerState)
    assert(resp == resp2)


    #check that agent does not get reinstantiated when deserialized
    agent.__dict__['epsilon'] = 100
    assert(agent.__dict__['epsilon'] == 100)
    agent_stream = dill.dumps(agent)
    agent_deserialized = dill.loads(agent_stream)
    assert(agent_deserialized.__dict__['epsilon'] == 100)


if __name__ == "__main__":
    test_FO_ModularAgent()
    

#Intstructions for Sq5:
# Find first part (i.e. 355*355 -> 35)
# 2. addOne -> 36
# 3. Multiply 35*36
# 4. Append 25 -> 35*36 + 25 =126025 (edited) 
# SAI = Selection Action Input: 
# Selection = Name of interface element
# Action = "UpdateTextField"
# Input = {"value": 7}