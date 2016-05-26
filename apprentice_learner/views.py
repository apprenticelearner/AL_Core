import json
import traceback

from django.views.decorators.csrf import csrf_exempt
#from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from django.http import HttpResponseServerError
from django.http import HttpResponseNotAllowed


from apprentice_learner.models import ActionSet
from apprentice_learner.models import Agent
from agents.Dummy import Dummy
from agents.WhereWhenHow import WhereWhenHow
from agents.LogicalWhenHow import LogicalWhenHow
from agents.LogicalWhereWhenHow import LogicalWhereWhenHow

agents = {'Dummy': Dummy,
          'WhereWhenHow': WhereWhenHow,
          'LogicalWhenHow': LogicalWhenHow,
          'LogicalWhereWhenHow': LogicalWhereWhenHow}




#update math_actions
#import sys
#sys.stdout = open('/home/anant/Documents/output2.txt', 'w')
#print("HELLO ARE YO thewU")
#from .models import PyFunction
#print("HELLOHOW ARE YOU")
#new_function = PyFunction.objects.get(id=1)
#math_actions_temp = {
#    "subtract":"subtracting",
#    "multiply":"multiplying",
#    "divide":"dividing"
#}
#print("HEY! I AM back")
#if new_function.name not in list(math_actions_temp.keys()):
#    math_actions_temp[new_function.name] = new_function.fun_def
#    print("HURRAY!I AM HAPPY")
#    print(math_actions_temp)
#    print("I AM MORE HAPPY")
#print('Hello World')
#

#number of events
#num_request =
#num_train =
#num_check =




@csrf_exempt
def create(request):
    """
    This is used to create a new agent with the provided configuration.

    .. todo:: TODO Ideally there should be a way to create agents both using
    the browser and via a POST object. 
    """
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    data = json.loads(request.body.decode('utf-8'))

    if 'agent_type' not in data:
        print("request body missing 'agent_type'")
        return HttpResponseBadRequest("request body missing 'agent_type'")

    if data['agent_type'] not in agents:
        print("Specified agent not supported")
        return HttpResponseBadRequest("Specified agent not supported")

    if 'args' not in data:
        args = {}
    else:
        args = data['args']

    try:
        instance = agents[data['agent_type']](*args)
        #updates
        agent = Agent(instance=instance, action_set=ActionSet.objects.all()[0], name="agent", num_request=0, num_train=0, num_check=0 )
        agent.save()
        agent.name="agent"+str(agent.id)
        #updates
        #agent.time_creation = datetime.now()
        #update actionset 
        updated_function_set = {}
        for j in agent.action_set.function.all():
            #for j in i.function.all():
            print(j.name)
            print(j.fun_def)
            temp_dict={}
            exec(j.fun_def,temp_dict)
            updated_function_set[j.name] = temp_dict[j.name]
            print(updated_function_set[j.name](2,4))
        print(updated_function_set)
        
        ##
        agent.save()
        ret_data = {'agent_id': str(agent.id)}
    except Exception as  e:
        print("Failed to create agent"+str(e))
        return HttpResponseServerError("Failed to create agent, ensure provided args are correct.")

    return HttpResponse(json.dumps(ret_data))

@csrf_exempt
def request(request, agent_id):
    """
    Returns an SAI description for a given a problem state according to the
    current knoweldge base.

    Expects an HTTP POST with a json object stored as a utf-8 btye string in the
    request body. 
    That object should have the following fields:
    """
    try:
        if request.method != "POST":
            return HttpResponseNotAllowed(["POST"])
        data = json.loads(request.body.decode('utf-8'))
        
        if 'state' not in data:
            print("request body missing 'state'")
            return HttpResponseBadRequest("request body missing 'state'")

        agent = Agent.objects.get(id=agent_id)
        #update actions
        #agent.time_modified = datetime.now()
        #====
        #updated
        agent.num_request = agent.num_request + 1
        agent.save()
        #updated
        response = agent.instance.request(data['state'])
        return HttpResponse(json.dumps(response))

    except Exception as e:
        traceback.print_exc()
        return HttpResponseServerError(str(e))

@csrf_exempt
def train(request, agent_id):
    """
    Trains the Agent with an state annotated with the SAI used / with
    feedback.
    """
    try:    
        if request.method != "POST":
            return HttpResponseNotAllowed(["POST"])
        data = json.loads(request.body.decode('utf-8'))

        if 'state' not in data:
            print("request body missing 'state'")
            return HttpResponseBadRequest("request body missing 'state'")
        if 'label' not in data:
            print("request body missing 'label'")
            return HttpResponseBadRequest("request body missing 'label'")
        if 'foas' not in data:
            print("request body missing 'foas'")
            return HttpResponseBadRequest("request body missing 'foas'")
        if 'selection' not in data:
            print("request body missing 'selection'")
            return HttpResponseBadRequest("request body missing 'selection'")
        if 'action' not in data:
            print("request body missing 'action'")
            return HttpResponseBadRequest("request body missing 'action'")
        if 'inputs' not in data:
            print("request body missing 'inputs'")
            return HttpResponseBadRequest("request body missing 'inputs'")
        if 'correct' not in data:
            print("request body missing 'correct'")
            return HttpResponseBadRequest("request body missing 'correct'")
        
        agent = Agent.objects.get(id=agent_id)
        #updated
        agent.num_train = agent.num_train+1
        #agent.time_modified = datetime.now()
        agent.save()
        #updated
        agent.instance.train(data['state'], data['label'], data['foas'],
                             data['selection'], data['action'], data['inputs'],
                             data['correct'])
        agent.save()
        return HttpResponse("OK")

    except Exception as e:
        traceback.print_exc()
        return HttpResponseServerError(str(e))

@csrf_exempt
def check(request, agent_id):
    
    """
    Uses the knoweldge base to check the correctness of an SAI in provided
    state.
    """
    try:    
        if request.method != "POST":
            return HttpResponseNotAllowed(["POST"])
        data = json.loads(request.body.decode('utf-8'))

        if 'state' not in data:
            return HttpResponseBadRequest("request body missing 'state'")
        if 'selection' not in data:
            return HttpResponseBadRequest("request body missing 'selection'")
        if 'action' not in data:
            return HttpResponseBadRequest("request body missing 'action'")
        if 'inputs' not in data:
            return HttpResponseBadRequest("request body missing 'inputs'")
        
        agent = Agent.objects.get(id=agent_id)
        #updated
        agent.num_check = agent.num_check+1
        agent.save()
        #updated
        response = {}
        response['correct'] = agent.instance.check(data['state'],
                                                   data['selection'],
                                                   data['action'],
                                                   data['inputs'])
        return HttpResponse(json.dumps(response))

    except Exception as e:
        traceback.print_exc()
        return HttpResponseServerError(str(e))
