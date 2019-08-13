import requests
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../db.py
from nltk import ViterbiParser
from nltk.grammar import PCFG

agentID = 1

url = "http://127.0.0.1:8000/"

fdir = "/Users/gabriel/Desktop/"
fname = "fractionInfo.csv"

opsdict = {"Mult":"*","Add":"+","Sub":"-","Div":":",}

with open(fdir + fname) as fin:
    mod = 0
    for eq in fin:
        #print(eq)
        prob, op, result = [x[y] for x in (eq.split(","),) for y in (4,5,9)]
        if not (op == "Mult"): continue
        lhs, rhs = prob.split(opsdict[op])
        num1, denom1 = lhs.split("/")
        num2, denom2 = rhs.split("/")
        if not (result.count("/") == 1): continue
        resultnum, resultdenom = result.split("/")
        try:
            int(resultnum)
            int(resultdenom)
        except ValueError:
            continue
        state = {"num1":{"id":"num1","value":num1,"contentEditable":False},
            "num2":{"id":"num2","value":num2,"contentEditable":False},
            "denom1":{"id":"denom1","value":denom1,"contentEditable":False},
            "denom2":{"id":"denom2","value":denom2,"contentEditable":False},
            "op":{"id":"op","value":op,"contentEditable":False},
            "num3":{"id":"num3","value":"","contentEditable":True},
            "denom3":{"id":"denom3","value":"","contentEditable":False},
        }
        if mod%100 == 99:
            print(eq)
            print("new test below")
            reqReq = requests.post(url+"request/"+str(agentID)+"/", json={"state": state})
            print(reqReq.json())
        obj = {
          "selection": "num3",
          "action": "UpdateTextField",
          "inputs": {
              "value": resultnum
          },
          "reward": 1,
          "state": state
        }
        trainReq = requests.post(url+"train/"+str(agentID)+"/", json=obj)
        #print(trainReq.status_code, trainReq.reason)
        #print(state)
        state["num3"] = {"id":"num3","value":resultnum,"contentEditable":False}
        state["denom3"] = {"id":"denom3","value":"","contentEditable":True}
        #print(state)
        obj = {
          "selection": "denom3",
          "action": "UpdateTextField",
          "inputs": {
              "value": resultdenom
          },
          "reward": 1,
          "state": state
        }
        trainReq = requests.post(url+"train/"+str(agentID)+"/", json=obj)
        if mod%100 == 99:
            print("new test below")
            reqReq = requests.post(url+"request/"+str(agentID)+"/", json={"state": state})
            print(reqReq.json())
        state["denom3"] = {"id":"num3","value":resultdenom,"contentEditable":False}
        #print(state)
        #print(trainReq.status_code, trainReq.reason)
        mod += 1
        mod %= 100
