from apprentice.agents.cre_agents.environment import Button, TextField, Container
from apprentice.agents.cre_agents.state import encode_neighbors, State
from apprentice.agents.cre_agents.dipl_base import BaseDIPLAgent
from apprentice.agents.cre_agents.cre_agent import CREAgent, SAI
from apprentice.agents.cre_agents.tests.test_state import new_mc_addition_state

from numba.types import unicode_type
from cre import MemSet
from cre.transform import MemSetBuilder, Flattener, FeatureApplier, RelativeEncoder, Vectorizer
from cre.default_funcs import Equals



def test_init_base_dipl():
    agent = BaseDIPLAgent()    

def test_cre_agent():
    function_set = ["Add3", "Mod10", "Add", "Div10", "Copy"]
    # Test no FOCI
    agent = CREAgent(feature_set=[], function_set=function_set,
                     where="antiunify")

    py_dict = new_mc_addition_state(567,491) 
    print(agent.act(py_dict))

    agent.train(py_dict, ( "0_answer", "UpdateTextField", {"value": 8} ), 1)
    py_dict['0_answer'].update({"value" : '8', "locked" : True})

    agent.train(py_dict, ( "1_answer", "UpdateTextField", {"value": 5} ), 1)
    py_dict['1_answer'].update({"value" : '5', "locked" : True})

    agent.train(py_dict, ( "2_carry", "UpdateTextField", {"value": 1} ), 1)
    py_dict['2_carry'].update({"value" : '1', "locked" : True})

    agent.train(py_dict, ( "2_answer", "UpdateTextField", {"value": 0} ), 1)
    py_dict['2_answer'].update({"value" : '0', "locked" : True})

    agent.train(py_dict, ( "3_carry", "UpdateTextField", {"value": 1} ), 1)
    py_dict['3_carry'].update({"value" : '1', "locked" : True})

    agent.train(py_dict, ( "3_answer", "UpdateTextField", {"value": 1} ), 1)
    py_dict['3_answer'].update({"value" : '1', "locked" : True})

    print(agent.act(py_dict))

    py_dict = new_mc_addition_state(456,582) 

    print(agent.act(py_dict))

    print("---------------------------")

    # Test w/ foci
    agent = CREAgent(feature_set=[], function_set=function_set,
                     where="antiunify")

    py_dict = new_mc_addition_state(567,491) 
    print(agent.act(py_dict))

    agent.train(py_dict, ( "0_answer", "UpdateTextField", {"value": 8} ), 1,
        arg_foci=["0_upper", '0_lower'])
    py_dict['0_answer'].update({"value" : '8', "locked" : True})

    agent.train(py_dict, ( "1_answer", "UpdateTextField", {"value": 5} ), 1,
        arg_foci=["1_upper", '1_lower'])
    py_dict['1_answer'].update({"value" : '5', "locked" : True},)

    agent.train(py_dict, ( "2_carry", "UpdateTextField", {"value": 1} ), 1,
        arg_foci=["1_upper", '1_lower'])
    py_dict['2_carry'].update({"value" : '1', "locked" : True})

    agent.train(py_dict, ( "2_answer", "UpdateTextField", {"value": 0} ), 1,
        arg_foci=['2_carry', "2_upper", '2_lower'])
    py_dict['2_answer'].update({"value" : '0', "locked" : True})

    agent.train(py_dict, ( "3_carry", "UpdateTextField", {"value": 1} ), 1,
        arg_foci=['2_carry', "2_upper", '2_lower'])
    py_dict['3_carry'].update({"value" : '1', "locked" : True})

    agent.train(py_dict, ( "3_answer", "UpdateTextField", {"value": 1} ), 1,
        arg_foci=['3_carry'])
    py_dict['3_answer'].update({"value" : '1', "locked" : True})


    py_dict = new_mc_addition_state(456,582) 

    print(agent.act(py_dict))


if __name__ == "__main__":
    import faulthandler; faulthandler.enable()
    # test_init_base_dipl()
    test_cre_agent()
