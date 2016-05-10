import json
import traceback

from django.views.decorators.csrf import csrf_exempt
#from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from django.http import HttpResponseServerError
from django.http import HttpResponseNotAllowed

from apprentice_learner.models import Agent
from agents.Dummy import Dummy

agents = {'Dummy': Dummy}

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
        agent = Agent(instance=instance)
        agent.save()
        ret_data = {'agent_id': str(agent.id)}
    except:
        print("Failed to create agent")
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
        agent.instance.train(data['state'], data['selection'], data['action'],
                             data['inputs'], data['correct'])
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
        response = {}
        response['correct'] = agent.instance.check(data['state'],
                                                   data['selection'],
                                                   data['action'],
                                                   data['inputs'])
        return HttpResponse(json.dumps(response))

    except Exception as e:
        traceback.print_exc()
        return HttpResponseServerError(str(e))
