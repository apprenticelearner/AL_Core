import requests
import os, sys
from random import randrange

agentID = 1

fdir = "/Users/gabriel/Desktop/"
fname = "fractionInfo.csv"

bignums = []

opsdict = {"Mult":"*"}

for i in range(1000):
    xn = randrange(1,10)
    yn = randrange(1,10)
    xd = randrange(1,10)
    yd = randrange(1,10)
    bignums.append([str(xn)+"/"+str(xd)+"*"+str(yn)+"/"+str(yd),"Mult",str(xn*yn)+"/"+str(xd*yd)])

with open(fdir + fname) as fin:
    for eq in bignums:
        prob, op, result = eq
        result = result.replace(" ","")
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

        input_for_get = {
            "states":[
                   state,
                ],
        }

        url = "http://127.0.0.1:8000/"

        response = requests.post(url+"request/"+str(agentID)+"/", json={"state":state})
        print(response)
        print(response.status_code, response.reason)
        print(response.json())

        response = requests.post(url + "get_skills/"+str(agentID)+"/", json=input_for_get)

        print(response)
        print(response.status_code, response.reason)
        print(response.json())

