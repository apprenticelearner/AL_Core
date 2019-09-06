import requests
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../db.py
# from nltk import ViterbiParser
# from nltk.grammar import PCFG
import random
import csv


# content editable true when answer is false or empty; otherwise content editable false
# numba


url = "http://127.0.0.1:8000/"

fdir = "/Users/gabriel/Desktop/"
fname = "fractionInfo.csv"
logfilename = "MultOnly.txt"

createNewAgent = True
includeNegativeFeedback = True

opsdict = {"Mult":"*","Add":"+","Sub":"-","Div":":",}

def log_accuracy(prob,resultnum,resultdenom,numeratorComputed,denominatorComputed):
    fi = open(logfilename,"a+")
    correctness = str(resultnum) == str(numeratorComputed)
    if resultnum == 'x':
        correctness = str(resultdenom) == str(denominatorComputed)
    fi.write("Problem: "+prob+" Answer Given: "+numeratorComputed+"/"+denominatorComputed+". The correct answer was: "+resultnum+"/"+resultdenom+"\n")
    fi.close()

def logString(stringToLog):
    fi = open(logfilename,"a+")
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
    fi = open(logfilename,"a+")
    fi.write(str(rule)+"\n")
    fi.close()
    
def create_agent():
    input_for_create = {"stay_active":True,"dont_save":True,"no_ops_parse":True,"args":{"when_learner":"decisiontree","where_learner":"MostSpecific","planner":"fo_planner"},"feature_set":["equals"],"function_set":
            ["add",
                "subtract",
                "multiply",
                "divide",
                ],"name":"Control_Stu_01266dfb27cc2e1a087884753dbe4f67","agent_type":"ModularAgent","project_id":1}
    url = "http://127.0.0.1:8000/"
    response = requests.post(url + "create/", json=input_for_create)

    print(response.status_code, response.reason)
    agentID = response.json()["agent_id"]
    print("agent id:", agentID)
    return agentID
    
def request_and_log(state, eq, part, trainingPart, url, agentID,  csvWriter):
    reqReq = requests.post(url+"request/"+str(agentID)+"/", json={"state": state})
    try:
        computedResponse = reqReq.json()["inputs"]["value"]
    except:
        computedResponse = ""
    problem, operator, result = eq
    correctResponse = get_desired_response(eq, part)
    row = {'Problem': problem, 'Part' : part, 'Operator' : operator, 'TrainingPart' : trainingPart, 'ComputedAnswer':computedResponse,
     'CorrectAnswer':correctResponse,
    'Correct' : correctResponse==computedResponse}
    csvWriter.writerow(row)
    return computedResponse
    # log_accuracy(prob, resultnum, "x", numeratorComputed, "x")

def get_desired_response(eq, part):
    '''
    eq is a tuple of (problem, operator, response), and part is 'num' or
    'denom'. This gets the numerator or the denominator of the response,
    according to what is specified in part.
    '''
    resultnum, resultdenom = eq[2].split("/")
    return resultnum if part == 'num' else resultdenom
    
    
def generate_problems(lower_bound, upper_bound, operators, num_problems, shuffle = True):
    problems = []
    for i in range(num_problems):
        for operatorWord in operators:
            operator = opsdict[operatorWord]
            
            xn = random.randrange(lower_bound,upper_bound)
            yn = random.randrange(lower_bound,upper_bound)
            xd = random.randrange(lower_bound,upper_bound)
            if operator == '+' or operator == '-':
                yd = xd
            else:
                yd = random.randrange(lower_bound,upper_bound)
            
            if operator == ':':
                resultNum = xn*yd
            else:
                resultNum = eval(str(xn) + operator + str(yn))
                
            if operator == '*':
                resultDenom = eval(str(xd) + operator + str(yd))
            elif operator == ':':
                resultDenom = xd * yn
            else:# operator == '+' or operator == '-':
                resultDenom = xd
            problems.append([str(xn)+"/"+str(xd)+operator+str(yn)+"/"+str(yd), operatorWord, str(resultNum) +"/"+ str(resultDenom)])
    print(problems)
    if shuffle:
        random.shuffle(problems)
    return problems

addQEle = True;

def train(agentID):
    bignums = generate_problems(1, 100, ['Mult','Add','Div','Sub'],10)
    
    logHeader = ['Problem','Operator','Part','TrainingPart','ComputedAnswer','CorrectAnswer','Correct']
    trainingParts = ['before','afterNegativeFeedback','afterTraining']
    with open(logfilename,'w') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=logHeader)
        csvwriter.writeheader()
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
                
            parts = ["num","denom"]
            for fractionPart in parts:
                curLogRow = {}
                state = {"?ele-num1":{"id":"num1","value":num1,"contentEditable":False},
                    "?ele-num2":{"id":"num2","value":num2,"contentEditable":False},
                    "?ele-denom1":{"id":"denom1","value":denom1,"contentEditable":False},
                    "?ele-denom2":{"id":"denom2","value":denom2,"contentEditable":False},
                    "?ele-op":{"id":"op","value":op,"contentEditable":False}

                }
                if fractionPart == "num":
                    #numerator stuff
                    state["?ele-num3"] = {"id":"num3","value":"","contentEditable":True}
                    state["?ele-denom3"] = {"id":"denom3","value":"","contentEditable":False}
                    correctResponse = resultnum
                    selection = "num3"
                else:
                    state["?ele-num3"] = {"id":"num3","value":resultnum,"contentEditable":False}
                    state["?ele-denom3"] = {"id":"denom3","value":"","contentEditable":True}
                    correctResponse = resultdenom
                    selection = "denom3"
                    
                input_for_get = {
                    "states":[
                           state,
                        ],
                }
                computedResponse = request_and_log(state, eq, fractionPart, trainingParts[0], url, agentID,  csvwriter)
                if includeNegativeFeedback and computedResponse != correctResponse:
                    obj = {
                      "selection": selection,
                      "action": "UpdateTextField",
                      "inputs": {
                          "value": computedResponse,
                      },
                      "reward": 0,
                      "state": state
                    }
                    trainReq = requests.post(url+"train/"+str(agentID)+"/", json=obj)
                    computedResponse = request_and_log(state, eq, fractionPart, trainingParts[1], url, agentID,  csvwriter)
                
        
                obj = {
                  "selection": selection,
                  "action": "UpdateTextField",
                  "inputs": {
                      "value": correctResponse
                  },
                  "reward": 1,
                  "state": state
                }
                trainReq = requests.post(url+"train/"+str(agentID)+"/", json=obj)
                computedResponse = request_and_log(state, eq, fractionPart, trainingParts[2], url, agentID,  csvwriter)

def main():
    if createNewAgent:
        agentID = create_agent()
    else:
        agentID = 1
    train(agentID)
    
if __name__ == "__main__":
    main()   
