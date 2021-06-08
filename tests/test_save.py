# try:  # python 3
#     from apprentice_learner.models import Agent
# except ImportError:  # python 2
#     import apprentice_learner.models as Agent

from apprentice.agents.ModularAgent import ModularAgent
from apprentice.agents.WhereWhenHowNoFoa import WhereWhenHowNoFoa
from apprentice.working_memory.representation import Sai
import dill
import copy

#check if agents contain same stuff
def areAgentsSame(agent1, agent2):
    agent1dict = agent1.__dict__
    agent2dict = agent2.__dict__

    to_find_common = {'constraint_generator', 'epsilon', 'feature_set', 'function_set', 'last_state', 'rhs_by_how', 
    'ret_train_expl', 'rhs_by_label', 'rhs_counter', 'search_depth', 'rhs_list'}
    for common in to_find_common:
        assert(agent1dict[common] == agent2dict[common])
        # if (agent1dict[common] != agent2dict[common]):
        #     return False
    
    #planner is not equal ,'planner'
    learners = {'state_variablizer'}#, 'which_learner', 'when_learner', 'where_learner'}
    for learner in learners: 
        assert(agent1dict[learner].__dict__ == agent2dict[learner].__dict__)
        # if agent1dict[learner].__dict__ != agent2dict[learner].__dict__:
        #     return False

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

    #request-- state used in integration testing
    # state = {'JCommTable.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544661867957-2', 'id': 'JCommTable.R0C0', 'type': 'TextField', 'value': '1', 'contentEditable': False, 'below': 'JCommTable4.R1C0', 'above': '', 'to_right': '', 'to_left': ''}, 
    #         'JCommTable.R1C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544661867957-2', 'id': 'JCommTable.R1C0', 'type': 'TextField', 'value': '2', 'contentEditable': False, 'below': '', 'above': 'JCommTable4.R1C0', 'to_right': '', 'to_left': ''}, 
    #         'JCommTable3.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662032496-5', 'id': 'JCommTable3.R0C0', 'type': 'TextField', 'value': '2', 'contentEditable': False, 'below': 'JCommTable4.R1C0', 'above': '', 'to_right': '', 'to_left': ''}, 
    #         'JCommTable3.R1C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662032496-5', 'id': 'JCommTable3.R1C0', 'type': 'TextField', 'value': '3', 'contentEditable': False, 'below': '', 'above': 'JCommTable4.R1C0', 'to_right': '', 'to_left': ''}, 
    #         'JCommTable4.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662076475-6', 'id': 'JCommTable4.R0C0', 'type': 'TextField', 'value': '', 'contentEditable': True, 'below': 'JCommTable4.R1C0', 'above': '', 'to_right': '', 'to_left': ''}, 
    #         'JCommTable4.R1C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662076475-6', 'id': 'JCommTable4.R1C0', 'type': 'TextField', 'value': '', 'contentEditable': True, 'below': 'JCommTable.R1C0', 'above': 'JCommTable.R0C0', 'to_right': '', 'to_left': ''}, 
    #         'JCommTable5.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662076487-8', 'id': 'JCommTable5.R0C0', 'type': 'TextField', 'value': '', 'contentEditable': True, 'below': 'JCommTable4.R1C0', 'above': '', 'to_right': '', 'to_left': ''}, 
    #         'JCommTable5.R1C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662076487-8', 'id': 'JCommTable5.R1C0', 'type': 'TextField', 'value': '', 'contentEditable': True, 'below': '', 'above': 'JCommTable4.R1C0', 'to_right': '', 'to_left': ''}, 
    #         'JCommTable7.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662248828-11', 'id': 'JCommTable7.R0C0', 'type': 'TextField', 'value': '+', 'contentEditable': False, 'below': 'JCommTable4.R1C0', 'above': '', 'to_right': '', 'to_left': ''}, 
    #         'JCommTable2.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662283254-14', 'id': 'JCommTable2.R0C0', 'type': 'TextField', 'value': '+', 'contentEditable': False, 'below': 'JCommTable4.R1C0', 'above': '', 'to_right': '', 'to_left': ''}, 
    #         'JCommTable6.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662568348-15', 'id': 'JCommTable6.R0C0', 'type': 'TextField', 'value': '', 'contentEditable': True, 'below': 'JCommTable4.R1C0', 'above': '', 'to_right': '', 'to_left': ''}, 
    #         'JCommTable6.R1C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1544662568348-15', 'id': 'JCommTable6.R1C0', 'type': 'TextField', 'value': '', 'contentEditable': True, 'below': '', 'above': 'JCommTable4.R1C0', 'to_right': '', 'to_left': ''}, 
    #         'done': {'dom_class': 'CTATDoneButton', 'offsetParent': 'background-initial', 'id': 'done', 'type': 'Component', 'below': 'hint', 'above': '', 'to_right': '', 'to_left': 'ctatdiv87'}, 'hint': {'dom_class': 'CTATHintButton', 'offsetParent': 'background-initial', 'id': 'hint', 'type': 'Component', 'below': '', 'above': 'done', 'to_right': '', 'to_left': 'ctatdiv87'}, 'ctatdiv87': {'dom_class': 'CTATHintWindow', 'offsetParent': 'background-initial', 'id': 'ctatdiv87', 'type': 'Component', 'below': '', 'above': '', 'to_right': 'done', 'to_left': ''}, 'JCommTable8.R0C0': {'dom_class': 'CTATTable--cell', 'offsetParent': 'silex-id-1547530347773-0', 'id': 'JCommTable8.R0C0', 'type': 'TextField', 'value': '', 'contentEditable': True, 'below': 'JCommTable4.R1C0', 'above': '', 'to_right': '', 'to_left': ''}}

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
    # assert(agent.__dict__['num_request'] == 1)
    # assert(agent.__dict__['num_train'] == 1)
    assert(dc == state)

    #Save the agent and deserialize
    agent_saved = dill.dumps(agent)
    agent_deserialized = dill.loads(agent_saved)

    #check that the agent is same before and after serializing
    assert(areAgentsSame(agent, agent_deserialized))

    #planner is different right now bc of lambda
    assert(agent.__dict__['planner'].__dict__ == agent_deserialized.__dict__['planner'].__dict__)

    #request both (train changes state?)
    SAI1 = agent.request(state, reward = 1)
    SAI2 = agent_deserialized.request(state, reward = 1)

    assert(SAI1 == SAI2)

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
