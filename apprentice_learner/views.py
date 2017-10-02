"""
Module Doc String
"""
import json
import traceback

from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import get_list_or_404
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.core.exceptions import ObjectDoesNotExist
from django.core.exceptions import MultipleObjectsReturned
from django.http import HttpResponse
from django.http import HttpResponseBadRequest
from django.http import HttpResponseServerError
from django.http import HttpResponseNotAllowed

# from apprentice_learner.models import ActionSet
from apprentice_learner.models import Agent
from apprentice_learner.models import Project
from apprentice_learner.models import Operator
from agents.Dummy import Dummy
from agents.WhereWhenHow import WhereWhenHow
from agents.WhereWhenHowNoFoa import WhereWhenHowNoFoa
from agents.RLAgent import RLAgent

AGENTS = {'Dummy': Dummy,
          'WhereWhenHowNoFoa': WhereWhenHowNoFoa,
          'WhereWhenHow': WhereWhenHow,
          'RLAgent': RLAgent}


def parse_operator_set(data, set_name, errs=None):
    """
    Given a data dictionary from a request looks up and compiles a set of
    Operators and throws appropriate exceptions when they have problems. I am
    allowing either a list of ints, taken as primary key's of Operators, or
    strs, which are taken as Operator names.
    """
    if errs is None:
        errs = []
    op_set = []
    for val in data.get(set_name, []):
        if isinstance(val, str):
            try:
                opr = Operator.objects.get(name=val)
                op_set.append(opr.compile())
            except ObjectDoesNotExist:
                errs.append("no operator with name {} exists".format(val))
            #This case should be impossible but I'm going to leave the error catch in
            except MultipleObjectsReturned:
                errs.append("multiple operators with name {} exist".format(val))
        elif isinstance(val, int):
            try:
                opr = Operator.objects.get(pk=val)
                op_set.append(opr.compile())
            except ObjectDoesNotExist:
                errs.append("no operator with name {} exists".format(val))
    return op_set, errs


@csrf_exempt
def create(http_request):
    """
    This is used to create a new agent with the provided configuration.

    .. todo:: TODO Ideally there should be a way to create agents both using
    the browser and via a POST object.
    """
    if http_request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    data = json.loads(http_request.body.decode('utf-8'))

    errs = []

    if 'agent_type' not in data or data['agent_type'] is None:
        errs.append("request body missing 'agent_type'")

    if not errs and data['agent_type'] not in AGENTS:
        errs.append("Specified agent not supported")

    project_id = data.get('project_id', 1)

    if project_id == 1:
        project = Project.objects.get_or_create(id=1)
    else:
        try:
            project = Project.objects.get(id=project_id)
        except ObjectDoesNotExist:
            errs.append(str.format("project: {} does not exist", project_id))
            project = None

    feature_set, errs = parse_operator_set(data, 'feature_set', errs)
    function_set, errs = parse_operator_set(data, 'function_set', errs)

    if len(errs) > 0:
        print('errors: {}'.format(','.join(errs)))
        return HttpResponseBadRequest('errors: {}'.format(','.join(errs)))

    if 'args' not in data:
        args = {}
    else:
        args = data['args']

    args['feature_set'] = feature_set
    args['function_set'] = function_set

    if project is not None:
        args['feature_set'] += project.compile_features()
        args['function_set'] += project.compile_functions()

    try:
        # args['action_set'] = action_set
        instance = AGENTS[data['agent_type']](**args)
        agent_name = data.get('name', '')
        agent = Agent(instance=instance,
                      name=agent_name)
        agent.save()
        ret_data = {'agent_id': str(agent.id)}

    except Exception as exp:
        print("Failed to create agent", exp)
        return HttpResponseServerError("Failed to create agent, "
                                       "ensure provided args are "
                                       "correct.")

    return HttpResponse(json.dumps(ret_data))


@csrf_exempt
def request(http_request, agent_id):
    """
    Returns an SAI description for a given a problem state according to the
    current knoweldge base.  Expects an HTTP POST with a json object stored as
    a utf-8 btye string in the request body.
    That object should have the following fields:
    """

    try:
        if http_request.method != "POST":
            return HttpResponseNotAllowed(["POST"])
        data = json.loads(http_request.body.decode('utf-8'))

        if 'state' not in data or data['state'] is None:
            print("request body missing 'state'")
            return HttpResponseBadRequest("request body missing 'state'")

        agent = Agent.objects.get(id=agent_id)
        agent.inc_request()
        response = agent.instance.request(data['state'])
        agent.save()
        return HttpResponse(json.dumps(response))

    except Exception as exp:
        traceback.print_exc()
        return HttpResponseServerError(str(exp))


@csrf_exempt
def request_by_name(http_request, agent_name):
    """
    A version of request that can look up an agent by its name. This will
    generally be slower but it also doesn't expose how the data is stored and
    might be easier in some cases.
    """
    agent = get_list_or_404(Agent, name=agent_name)[0]
    return request(http_request, agent.id)


@csrf_exempt
def train(http_request, agent_id):
    """
    Trains the Agent with an state annotated with the SAI used / with
    feedback.
    """
    try:
        if http_request.method != "POST":
            return HttpResponseNotAllowed(["POST"])
        data = json.loads(http_request.body.decode('utf-8'))

        errs = []

        if 'state' not in data or data['state'] is None:
            errs.append("request body missing 'state'")
        if 'label' not in data or data['label'] is None:
            data['label'] = 'NO_LABEL'
        if 'foas' not in data or data['foas'] is None:
            data['foas'] = {}
        if 'selection' not in data or data['selection'] is None:
            errs.append("request body missing 'selection'")
        if 'action' not in data or data['action'] is None:
            errs.append("request body missing 'action'")
        if 'inputs' not in data or data['inputs'] is None:
            errs.append("request body missing 'inputs'")
        if 'correct' not in data or data['correct'] is None:
            errs.append("request body missing 'correct'")

        # Linter was complaining about too many returns so I consolidated all of the errors above
        if len(errs) > 0:
            print('errors: {}'.format(','.join(errs)))
            return HttpResponseBadRequest('errors: {}'.format(','.join(errs)))

        agent = Agent.objects.get(id=agent_id)
        agent.inc_train()

        agent.instance.train(data['state'], data['label'], data['foas'],
                             data['selection'], data['action'], data['inputs'],
                             data['correct'])
        agent.save()
        return HttpResponse("OK")

    except Exception as exp:
        traceback.print_exc()
        return HttpResponseServerError(str(exp))


@csrf_exempt
def train_by_name(http_request, agent_name):
    """
    A version of train that can look up an agent by its name. This will
    generally be slower but it also doesn't expose how the data is stored and
    might be easier in some cases.
    """
    agent = get_list_or_404(Agent, name=agent_name)[0]
    return train(http_request, agent.id)


@csrf_exempt
def check(http_request, agent_id):
    """
    Uses the knoweldge base to check the correctness of an SAI in provided
    state.
    """
    try:
        if http_request.method != "POST":
            return HttpResponseNotAllowed(["POST"])
        data = json.loads(http_request.body.decode('utf-8'))

        errs = []

        if 'state' not in data:
            errs.append("request body missing 'state'")
        if 'selection' not in data:
            errs.append("request body missing 'selection'")
        if 'action' not in data:
            errs.append("request body missing 'action'")
        if 'inputs' not in data:
            errs.append("request body missing 'inputs'")

        if len(errs) > 0:
            print('errors: {}'.format(','.join(errs)))
            return HttpResponseBadRequest('errors: {}'.format(','.join(errs)))

        agent = Agent.objects.get(id=agent_id)
        agent.inc_check()
        agent.save()

        response = {}

        response['correct'] = agent.instance.check(data['state'],
                                                   data['selection'],
                                                   data['action'],
                                                   data['inputs'])
        return HttpResponse(json.dumps(response))

    except Exception as exp:
        traceback.print_exc()
        return HttpResponseServerError(str(exp))


@csrf_exempt
def check_by_name(http_request, agent_name):
    """
    A version of check that can look up an agent by its name. This will
    generally be slower but it also doesn't expose how the data is stored and
    might be easier in some cases.
    """
    agent = get_list_or_404(Agent, name=agent_name)[0]
    return check(http_request, agent.id)


def report(http_request, agent_id):
    """
    A simple method for looking up an agent's stats. I've found it to be
    really helpful for looking an agent's idea from its name to speed up
    processing. Also nice for stat counting.
    """

    if http_request.method != "GET":
        return HttpResponseNotAllowed(["GET"])

    agent = get_object_or_404(Agent, id=agent_id)

    response = {
        'id': agent.id,
        'name': agent.name,
        'num_request': agent.num_request,
        'num_train': agent.num_train,
        'num_check': agent.num_check,
        'created': agent.created,
        'updated': agent.updated
    }

    response = {k: str(response[k]) for k in response}
    return HttpResponse(json.dumps(response))


def report_by_name(http_request, agent_name):
    """
    A version of report that can look up an agent by its name. This will
    generally be slower but it also doesn't expose how the data is stored and
    might be easier in some cases.
    """
    agent = get_list_or_404(Agent, name=agent_name)[0]
    return report(http_request, agent.id)

@csrf_exempt
def test_view(http_request) :
    return render(http_request, 'apprentice_learner/tester.html')