import requests
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planners.rulesets import grammar_parser_rule
from planners.rulesets import custom_function_set

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

url = "http://127.0.0.1:8000/"
response = requests.post(url + "create/", json=input_for_create)

print(response.status_code, response.reason)

