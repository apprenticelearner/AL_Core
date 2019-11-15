import requests
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import ../db.py
# from nltk import ViterbiParser
# from nltk.grammar import PCFG
import random
import csv
from fractions import Fraction

# content editable true when answer is false or empty; otherwise content editable false
# numba


url = "http://127.0.0.1:8000/"

fdir = "/Users/gabriel/Desktop/"
fname = "fractionInfo.csv"
logfilename = "MultOnly.txt"

createNewAgent = True
includeNegativeFeedback = True

opsdict = {"Mult": "*", "Add": "+", "Sub": "-", "Div": ":", }


def log_accuracy(prob, resultnum, resultdenom, numeratorComputed, denominatorComputed):
    fi = open(logfilename, "a+")
    correctness = str(resultnum) == str(numeratorComputed)
    if resultnum == 'x':
        correctness = str(resultdenom) == str(denominatorComputed)
    fi.write(
        "Problem: " + prob + " Answer Given: " + numeratorComputed + "/" + denominatorComputed + ". The correct answer was: " + resultnum + "/" + resultdenom + "\n")
    fi.close()


def logString(stringToLog):
    fi = open(logfilename, "a+")
    fi.write(stringToLog)
    fi.close()


def log_rules(state):
    obj = {"states": [state, ], }
    rule = requests.post(url + "get_skills/" + str(agentID) + "/", json=obj)
    print(rule)
    print(rule.status_code, rule.reason)
    rule = rule.json()
    print("rule learned:")
    print(rule)
    fi = open(logfilename, "a+")
    fi.write(str(rule) + "\n")
    fi.close()


def create_agent():
    input_for_create = {"stay_active": True, "dont_save": True, "no_ops_parse": True,
                        "args": {"when_learner": "decisiontree", "where_learner": "MostSpecific",
                                 "planner": "fo_planner"}, "feature_set": ["equals", "lcm", "gcd"], "function_set":
                            ["add",
                             "subtract",
                             "multiply",
                             "divide",
                             "lcm",
                             "gcd",
                             ], "name": "Control_Stu_01266dfb27cc2e1a087884753dbe4f67", "agent_type": "ModularAgent",
                        "project_id": 1}
    url = "http://127.0.0.1:8000/"
    response = requests.post(url + "create/", json=input_for_create)

    print(response.status_code, response.reason)
    agentID = response.json()["agent_id"]
    print("agent id:", agentID)
    return agentID


def request_and_log(state, eq, part, trainingPart, url, agentID, csvWriter):
    reqReq = requests.post(url + "request/" + str(agentID) + "/", json={"state": state})
    rule = ""

    try:
        computedResponse = reqReq.json()["inputs"]["value"]
        obj = {"states": [state, ], }
        rule = requests.post(url + "get_skills/" + str(agentID) + "/", json=obj)
        rule = rule.json()
    except:
        computedResponse = ""
    problem, operator, result = eq
    correctResponse = get_desired_response(eq, part)
    row = {'Problem': problem, 'Part': part, 'Operator': operator, 'TrainingPart': trainingPart,
           'ComputedAnswer': computedResponse,
           'CorrectAnswer': correctResponse,
           'Correct': correctResponse == computedResponse, 'Rule': rule}
    csvWriter.writerow(row)
    return computedResponse
    # log_accuracy(prob, resultnum, "x", numeratorComputed, "x")


def getRules(states, eq, url, agentID, problemNumber):
    # Function that writes to a file with problem number, problem, rule # used to solve problem, rule used to solve
    # problem, all the rules it has used, and the amount of different rules it has used.

    obj = {"states": [states[problemNumber], ], }
    rule = requests.post(url + "get_skills/" + str(agentID) + "/", json=obj)
    rule = rule.json()
    problem, operator, result = eq
    rules = []
    for state in states:
        obj = {"states": [state, ], }
        rule = requests.post(url + "get_skills/" + str(agentID) + "/", json=obj)
        rule = rule.json()
        if not rule:
            rule = ["Don't know"]
        if str(rule[0]) not in rules:
            rules.append(str(rule[0]))

    if not rule:
        rule = ["Don't know"]
    if str(rule[0]) not in rules:
        rules.append(str(rule[0]))

    row = [str(problemNumber), problem, rule, rules, len(rules)]
    return row


def get_desired_response(eq, part):
    '''
    eq is a tuple of (problem, operator, response), and part is 'num' or
    'denom'. This gets the numerator or the denominator of the response,
    according to what is specified in part.
    '''
    resultnum, resultdenom = eq[2].split("/")
    return resultnum if part == 'num' else resultdenom


def generate_problems(lower_bound, upper_bound, operators, num_problems, shuffle=True):
    problems = []
    for i in range(num_problems):
        for operatorWord in operators:
            operator = opsdict[operatorWord]

            xn = random.randrange(lower_bound, upper_bound)
            yn = random.randrange(lower_bound, upper_bound)
            xd = random.randrange(lower_bound, upper_bound)
            if operator == '+' or operator == '-':
                yd = xd
            else:
                yd = random.randrange(lower_bound, upper_bound)

            if operator == ':':
                resultNum = xn * yd
            else:
                resultNum = eval(str(xn) + operator + str(yn))

            if operator == '*':
                resultDenom = eval(str(xd) + operator + str(yd))
            elif operator == ':':
                resultDenom = xd * yn
            else:  # operator == '+' or operator == '-':
                resultDenom = xd
            problems.append([str(xn) + "/" + str(xd) + operator + str(yn) + "/" + str(yd), operatorWord,
                             str(resultNum) + "/" + str(resultDenom)])
    print(problems)
    if shuffle:
        random.shuffle(problems)
    return problems

def generateMulti_problems(lower_bound, upper_bound, operators, num_problems, shuffle=False):
    problems = []
    operators =['Mult']
    for i in range(num_problems):
        for operatorWord in operators:
            operator = opsdict[operatorWord]

            xn = random.randrange(lower_bound, upper_bound)
            # xn = 1
            randomV = random.randrange(lower_bound, upper_bound)
            yn = randomV * random.randrange(1, 5)
            xd = random.randrange(lower_bound, upper_bound)
            # xd = 1

            if operator == '+' or operator == '-':
                yd = xd
            else:
                # yd = random.randrange(lower_bound, upper_bound)
                yd = randomV * random.randrange(1, 5)

            if operator == ':':
                resultNum = xn * yd
            else:
                resultNum = eval(str(xn) + operator + str(yn))

            if operator == '*':
                resultDenom = eval(str(xd) + operator + str(yd))
            elif operator == ':':
                resultDenom = xd * yn
            else:  # operator == '+' or operator == '-':
                resultDenom = xd

            # Simplifies fraction
            resultFractionParts = str(Fraction(resultNum, resultDenom)).split("/")
            resultNum = resultFractionParts[0]
            if len(resultFractionParts) == 1:
                resultDenom = 1
            else:
                resultDenom = resultFractionParts[1]

            problems.append([str(xn) + "/" + str(xd) + operator + str(yn) + "/" + str(yd), operatorWord,
                             str(resultNum) + "/" + str(resultDenom)])
    print(problems)
    if shuffle:
        random.shuffle(problems)
    return problems


def makeAllStates(bignums):
    allStates = []
    for eq in bignums:
        prob, op, result = eq
        result = result.replace(" ", "")
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

        parts = ["num", "denom"]
        for fractionPart in parts:
            curLogRow = {}
            state = {"?ele-num1": {"id": "num1", "value": num1, "contentEditable": False},
                     "?ele-num2": {"id": "num2", "value": num2, "contentEditable": False},
                     "?ele-denom1": {"id": "denom1", "value": denom1, "contentEditable": False},
                     "?ele-denom2": {"id": "denom2", "value": denom2, "contentEditable": False},
                     "?ele-op": {"id": "op", "value": op, "contentEditable": False}

                     }
            if fractionPart == "num":
                # numerator stuff
                state["?ele-num3"] = {"id": "num3", "value": "", "contentEditable": True}
                state["?ele-denom3"] = {"id": "denom3", "value": "", "contentEditable": False}
                correctResponse = resultnum
                selection = "num3"
            else:
                # demonator stuff
                state["?ele-num3"] = {"id": "num3", "value": resultnum, "contentEditable": False}
                state["?ele-denom3"] = {"id": "denom3", "value": "", "contentEditable": True}
                correctResponse = resultdenom
                selection = "denom3"

            input_for_get = {
                "states": [
                    state,
                ],
            }
            allStates.append(state)
    return allStates


def trainOneState(eq, trainingParts, agentID, csvwriter):
    fracStates = []
    prob, op, result = eq
    result = result.replace(" ", "")
    lhs, rhs = prob.split(opsdict[op])
    num1, denom1 = lhs.split("/")
    num2, denom2 = rhs.split("/")
    if not (result.count("/") == 1): return
    resultnum, resultdenom = result.split("/")
    try:
        int(resultnum)
        int(resultdenom)
    except ValueError:
        return

    parts = ["num", "denom"]
    for fractionPart in parts:
        curLogRow = {}
        state = {"?ele-num1": {"id": "num1", "value": num1, "contentEditable": False},
                 "?ele-num2": {"id": "num2", "value": num2, "contentEditable": False},
                 "?ele-denom1": {"id": "denom1", "value": denom1, "contentEditable": False},
                 "?ele-denom2": {"id": "denom2", "value": denom2, "contentEditable": False},
                 "?ele-op": {"id": "op", "value": op, "contentEditable": False}

                 }
        if fractionPart == "num":
            # numerator stuff
            state["?ele-num3"] = {"id": "num3", "value": "", "contentEditable": True}
            state["?ele-denom3"] = {"id": "denom3", "value": "", "contentEditable": False}
            correctResponse = resultnum
            selection = "num3"
        else:
            state["?ele-num3"] = {"id": "num3", "value": resultnum, "contentEditable": False}
            state["?ele-denom3"] = {"id": "denom3", "value": "", "contentEditable": True}
            correctResponse = resultdenom
            selection = "denom3"

        input_for_get = {
            "states": [
                state,
            ],
        }

        computedResponse = request_and_log(state, eq, fractionPart, trainingParts[0], url, agentID, csvwriter)
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
            trainReq = requests.post(url + "train/" + str(agentID) + "/", json=obj)
            computedResponse = request_and_log(state, eq, fractionPart, trainingParts[1], url, agentID, csvwriter)

        obj = {
            "selection": selection,
            "action": "UpdateTextField",
            "inputs": {
                "value": correctResponse
            },
            "reward": 1,
            "state": state
        }
        trainReq = requests.post(url + "train/" + str(agentID) + "/", json=obj)

        computedResponse = request_and_log(state, eq, fractionPart, trainingParts[2], url, agentID, csvwriter)

def writeGetRulesFile(statesRow, statesHeader):
    for index in range(1, len(statesRow)):
        statesRow[index].append(list(set(statesRow[index][3]) - set(statesRow[index - 1][3])))

    with open('getRules.csv', 'w') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        wr.writerows([statesHeader])
        wr.writerows(statesRow)

def writeRuleReportFile(statesRow, ruleReportHeader):
        ruleReportArray = []
        for index in range(len(statesRow)):
            ruleReportRow = []
            ruleReportRow.append(statesRow[index][2][0])
            if 'when' not in statesRow[index][2][0] and 'where' not in statesRow[index][2][0]:
                ruleReportRow.append('N/A')
                ruleReportRow.append('N/A')
                ruleReportRow.append('N/A')
                ruleReportArray.append(ruleReportRow)
                continue

            ruleReportRow.append(len(str(statesRow[index][2][0]['when'])))
            ruleReportRow.append(len(str(statesRow[index][2][0]['where'])))
            indexesOfEqual = find(str(statesRow[index][2][0]['when']), '=')
            containsDigit = False
            for indexOfEqual in indexesOfEqual:
                if str(statesRow[index][2][0]['when'])[indexOfEqual+1].isdigit():
                    containsDigit = True
            ruleReportRow.append(containsDigit)
            ruleReportArray.append(ruleReportRow)

        with open('ruleReport.csv', 'w') as result_file:
            wr = csv.writer(result_file, dialect='excel')
            wr.writerows([ruleReportHeader])
            wr.writerows(ruleReportArray)

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def train(agentID):
    bignums = generateMulti_problems(1, 50, ['Mult', 'Add', 'Sub', 'Div'], 20)

    logHeader = ['Problem', 'Operator', 'Part', 'TrainingPart', 'ComputedAnswer', 'CorrectAnswer', 'Correct', 'Rule']
    trainingParts = ['before', 'afterNegativeFeedback', 'afterTraining']
    with open(logfilename, 'w') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=logHeader)
        csvwriter.writeheader()
        problemNumber = 0

        allStates = makeAllStates(bignums)

        statesHeader = ['ProblemNumber', 'Problem', 'RuleUsedtoSolve', 'AllUniqueRules', 'TotalNumberofUniqueRules', 'ChangeInRuleset']
        ruleReportHeader = ['rule', 'whenSize', 'whereSize', 'containConstant']
        statesRow = []
        for eq in bignums:
            statesRow.append(getRules(allStates, eq, url, agentID, problemNumber))
            trainOneState(eq, trainingParts, agentID, csvwriter)
            problemNumber += 1

        #writeGetRulesFile(statesRow, statesHeader)
        writeRuleReportFile(statesRow, ruleReportHeader)

def main():
    random.seed(9001)
    if createNewAgent:
        agentID = create_agent()
    else:
        agentID = 1
    train(agentID)


if __name__ == "__main__":
    main()
