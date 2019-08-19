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
        "lcm",
      ]
}

input_for_create = {"stay_active":True,"dont_save":True,"no_ops_parse":True,"args":{"when_learner":"decisiontree","where_learner":"MostSpecific","planner":"fo_planner"},"feature_set":["equals"],"function_set":
        ["add",
            "subtract",
            "multiply",
            "divide",
            #"lcm"
            ],"name":"Control_Stu_01266dfb27cc2e1a087884753dbe4f67","agent_type":"ModularAgent","project_id":1}

url = "http://127.0.0.1:8000/"
response = requests.post(url + "create/", json=input_for_create)

print(response.status_code, response.reason)

