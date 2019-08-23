import requests
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planners.rulesets import grammar_parser_rule
from planners.rulesets import *

input_for_create = {
  "agent_type": "ModularAgent",
  "no_ops_parse": False,
  "feature_set":[],
  "function_set":[
        "add",
        "subtract",
        "multiply",
        "divide",
      ]
}

input_for_create = {"stay_active":True,"dont_save":True,"no_ops_parse":True,"args":{"when_learner":"decisiontree","where_learner":"MostSpecific","planner":"fo_planner"},"feature_set":["equals"],"function_set":
        ["add",
            "subtract",
            "multiply",
            "divide",
            ],"name":"Control_Stu_01266dfb27cc2e1a087884753dbe4f67","agent_type":"ModularAgent","project_id":1}

url = "http://127.0.0.1:8000/"
response = requests.post(url + "create/", json=input_for_create)

print(response.status_code, response.reason)

state = {"num1":{"id":"num1","value":"1","contentEditable":False},
    "num2":{"id":"num2","value":"1","contentEditable":False},
    "denom1":{"id":"denom1","value":"1","contentEditable":False},
    "denom2":{"id":"denom2","value":"1","contentEditable":False},
    "op":{"id":"op","value":"Mult","contentEditable":False},
    "num3":{"id":"num3","value":"","contentEditable":True},
    "denom3":{"id":"denom3","value":"","contentEditable":False},
}

obj = {
  "selection": "num3",
  "action": "UpdateTextField",
  "inputs": {
      "value": "1",
  },
  "reward": 1,
  "state": state
}
agentID =1
response = requests.post(url+"train/"+str(agentID)+"/", json=obj)
print(response.status_code, response.reason)

state = {"num1":{"id":"num1","value":"1","contentEditable":False},
    "num2":{"id":"num2","value":"1","contentEditable":False},
    "denom1":{"id":"denom1","value":"1","contentEditable":False},
    "denom2":{"id":"denom2","value":"1","contentEditable":False},
    "op":{"id":"op","value":"Mult","contentEditable":False},
    "num3":{"id":"num3","value":"1","contentEditable":False},
    "denom3":{"id":"denom3","value":"","contentEditable":True},
}

obj = {
  "selection": "denom3",
  "action": "UpdateTextField",
  "inputs": {
      "value": "1",
  },
  "reward": 1,
  "state": state
}
response = requests.post(url+"train/"+str(agentID)+"/", json=obj)

print(response.status_code, response.reason)
