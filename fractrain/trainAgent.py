import requests
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../db.py
from nltk import ViterbiParser
from nltk.grammar import PCFG
from random import randrange

agentID = 1

url = "http://127.0.0.1:8000/"

fdir = "/Users/gabriel/Desktop/"
fname = "fractionInfo.csv"

opsdict = {"Mult":"*","Add":"+","Sub":"-","Div":":",}

def log_accuracy(prob,resultnum,resultdenom,numeratorComputed,denominatorComputed):
    fi = open("log","a+")
    fi.write("Problem: "+prob+" Answer Given: "+numeratorComputed+"/"+denominatorComputed+". The correct answer was: "+resultnum+"/"+resultdenom+"\n")
    fi.close()

def logString(stringToLog):
    fi = open("log","a+")
    fi.write(stringToLog)
    fi.close()

def log_rules(state):
    obj = {"states":[state,],}
    rule = requests.post(url+"get_skills/"+str(agentID)+"/", json=obj)
    print(rule)
    print(rule.status_code, rule.reason)
    rule = rule.json()
    print("rule learned:")
    print(rule)
    fi = open("log","a+")
    fi.write(str(rule)+"\n")
    fi.close()

bignums = []

for i in range(4*300):
    xn = randrange(1,10)
    yn = randrange(1,10)
    xd = randrange(1,10)
    yd = randrange(1,10)
    bignums.append([str(xn)+"/"+str(xd)+"*"+str(yn)+"/"+str(yd),"Mult",str(xn*yn)+"/"+str(xd*yd)])
    '''xn = randrange(1,10)
    yn = randrange(1,10)
    xd = randrange(1,10)
    bignums.append([str(xn)+"/"+str(xd)+"+"+str(yn)+"/"+str(xd),"Add",str(xn+yn)+"/"+str(xd)])
    xn = randrange(1,10)
    yn = randrange(1,10)
    xd = randrange(1,10)
    yd = randrange(1,10)
    bignums.append([str(xn)+"/"+str(xd)+"-"+str(yn)+"/"+str(xd),"Sub",str(xn-yn)+"/"+str(xd)])
    xn = randrange(1,10)
    yn = randrange(1,10)
    xd = randrange(1,10)
    yd = randrange(1,10)
    bignums.append([str(xn)+"/"+str(xd)+":"+str(yn)+"/"+str(yd),"Div",str(xn*yd)+"/"+str(xd*yn)])
    '''

for eq in bignums:
    prob, op, result = eq
    result = result.replace(" ","")
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
    #numerator stuff
    state = {"num1":{"id":"num1","value":num1,"contentEditable":False},
        "num2":{"id":"num2","value":num2,"contentEditable":False},
        "denom1":{"id":"denom1","value":denom1,"contentEditable":False},
        "denom2":{"id":"denom2","value":denom2,"contentEditable":False},
        "op":{"id":"op","value":op,"contentEditable":False},
        "num3":{"id":"num3","value":"","contentEditable":True},
        "denom3":{"id":"denom3","value":"","contentEditable":False},
    }

    input_for_get = {
        "states":[
               state,
            ],
    }
    url = "http://127.0.0.1:8000/"
    reqReq = requests.post(url+"request/"+str(agentID)+"/", json={"state": state})
    print(reqReq)
    numeratorComputed = reqReq.json()["inputs"]["value"]

    logString("logging for numerator\n")
    log_accuracy(prob, resultnum, "x", numeratorComputed, "x")
    log_rules(state)

    # negative feedback:
    #'''
    if numeratorComputed != resultnum:
        print("got num wrong, correcting")
        logString("got numerator wrong. corrected version:\n")
        obj = {
          "selection": "num3",
          "action": "UpdateTextField",
          "inputs": {
              "value": numeratorComputed,
          },
          "reward": 0,
          "state": state
        }
        trainReq = requests.post(url+"train/"+str(agentID)+"/", json=obj)
        reqReq = requests.post(url+"request/"+str(agentID)+"/", json={"state": state})
        numeratorComputed = reqReq.json()["inputs"]["value"]
        log_accuracy(prob, resultnum, "x", numeratorComputed, "x")
        log_rules(state)

    #'''

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
    logString("post-training results:\n")
    log_accuracy(prob, resultnum, "x", numeratorComputed, "x")
    log_rules(state)

    #switch to denominator

    state["num3"] = {"id":"num3","value":resultnum,"contentEditable":False}
    state["denom3"] = {"id":"denom3","value":"","contentEditable":True}
    #denominator stuff

    reqReq = requests.post(url+"request/"+str(agentID)+"/", json={"state": state})
    print(reqReq.json())
    denominatorComputed = reqReq.json()["inputs"]["value"]

    logString("logging for denominator\n")
    log_accuracy(prob, "x", resultDenom, "x", denominatorComputed)
    log_rules(state)

    # negative feedback:
    #'''
    if denominatorComputed != resultDenom:
        print("got denom wrong, correcting")
        logString("got denom wrong. corrected version:\n")
        obj = {
          "selection": "denom3",
          "action": "UpdateTextField",
          "inputs": {
              "value": denominatorComputed,
          },
          "reward": 0,
          "state": state
        }
        trainReq = requests.post(url+"train/"+str(agentID)+"/", json=obj)
        reqReq = requests.post(url+"request/"+str(agentID)+"/", json={"state": state})
        denominatorComputed = reqReq.json()["inputs"]["value"]
        log_accuracy(prob, "x", resultDenom, "x", denominatorComputed)
        log_rules(state)

    #'''

    obj = {
      "selection": "denom3",
      "action": "UpdateTextField",
      "inputs": {
          "value": resultnum
      },
      "reward": 1,
      "state": state
    }
    trainReq = requests.post(url+"train/"+str(agentID)+"/", json=obj)

    logString("post-training results:\n")
    log_accuracy(prob, "x", resultDenom, "x", denominatorComputed)
    log_rules(state)
    logString("\n")
