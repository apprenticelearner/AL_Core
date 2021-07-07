"""
Module Doc String
"""
import json
import traceback
import logging
from pprint import pprint
import time

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

from apprentice_learner.models import Agent

from apprentice.agents.experta_agent import ExpertaAgent
from apprentice.agents.Stub import Stub
from apprentice.agents.Memo import Memo
from apprentice.agents.WhereWhenHowNoFoa import WhereWhenHowNoFoa
from apprentice.agents.ModularAgent import ModularAgent
from apprentice.agents.RLAgent import RLAgent
from apprentice.working_memory.representation import Sai

# import cProfile
# pr = cProfile.Profile()

log = logging.getLogger('al-django')
performance_logger = logging.getLogger('al-performance')

DEFAULT_SAVE_AGENT = False
DEFAULT_STAY_ACTIVE = True


active_agent = None
active_agent_id = None
save_agent = None


AGENTS = {
    "Stub": Stub,
    "Memo": Memo,
    "RLAgent": RLAgent,
    "WhereWhenHowNoFoa": WhereWhenHowNoFoa,
    "ModularAgent": ModularAgent,
    "ExpertaAgent": ExpertaAgent,
}

last_call_time = time.time_ns()


def get_agent_by_id(id):
    global active_agent, active_agent_id
    if id == active_agent_id:
        agent = active_agent
    else:
        agent = Agent.objects.get(id=id)
    return agent


@csrf_exempt
def create(http_request):
    """
    This is used to create a new agent with the provided configuration.

    .. todo:: TODO Ideally there should be a way to create agents both using
    the browser and via a POST object.
    """
    if http_request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    data = json.loads(http_request.body.decode("utf-8"))

    errs = []
    warns = []

    if "agent_type" not in data or data["agent_type"] is None:
        errs.append("request body missing 'agent_type'")

    if not errs and data["agent_type"] not in AGENTS:
        errs.append("Specified agent not supported")

    if "args" not in data:
        args = {}
    else:
        args = data["args"]

    if "no_ops_parse" in data:
        warns.append(
            "Deprecation Warning: no_ops_parse is provided. This field is no"
            " longer used and is always assumed to be True. Operators must be"
            " registered in the system ahead of time and referenced by name.")

    if "feature_set" in data:
        warns.append(
            "Deprecation Warning: feature_set provided at top level."
            " feature_set is no longer used as a top level parameter."
            " It should now be included in the args object of a ModularAgent.")
        if "feature_set" not in args:
            args['feature_set'] = data.pop('feature_set', [])

    if "function_set" in data:
        warns.append(
            "Deprecation Warning: function_set provided at top level."
            " function_set is no longer used as a top level parameter."
            " It should now be included in the args object of a ModularAgent.")
        if "function_set" not in args:
            args['function_set'] = data.pop('function_set', [])

    if len(errs) > 0:
        for err in errs:
            log.error(err)
        # print("errors:\n {}".format("\n".join(errs)))
        return HttpResponseBadRequest("errors: {}".format(",".join(errs)))

    try:
        instance = AGENTS[data["agent_type"]](**args)
        agent_name = data.get("name", "")
        agent = Agent(instance=instance, name=agent_name)
        agent.save()
        ret_data = {"agent_id": str(agent.id)}

    except Exception as exp:
        traceback.print_exc()
        print("Failed to create agent", exp)
        return HttpResponseServerError(
            "Failed to create agent, ensure provided args are correct."
        )

    global active_agent, active_agent_id, save_agent
    if active_agent is not None:
        active_agent = None
        active_agent_id = None

    if str(data.get("stay_active", DEFAULT_STAY_ACTIVE)).lower() == "true":
        active_agent = agent
        active_agent_id = str(agent.id)
    else:
        # warns.append(
        #     "Stability Warning: stay_active is set to false. Serialization and"
        #     " deserialization of agents is currently not working for most"
        #     " agent types. Expect Errors.")
        pass

    if "save_agent" in data:
        save_agent = str(data["save_agent"]).lower() == "true"
    elif "dont_save" in data:
        save_agent = not str(data["dont_save"]).lower() == "true"
    else:
        save_agent = DEFAULT_SAVE_AGENT

    if len(warns) > 0:
        for warn in warns:
            log.warning(warn)
        ret_data["warnings"] = warns

    last_call_time = time.time_ns()

    return HttpResponse(json.dumps(ret_data))


@csrf_exempt
def request(http_request, agent_id):
    """
    Returns an SAI description for a given a problem state according to the
    current knoweldge base.  Expects an HTTP POST with a json object stored as
    a utf-8 btye string in the request body.
    That object should have the following fields:
    """
    global last_call_time
    performance_logger.info("Interface Feedback Time: {} ms".format((time.time_ns()-last_call_time)/1e6))

    # pr.enable()
    try:
        if http_request.method != "POST":
            return HttpResponseNotAllowed(["POST"])
        data = json.loads(http_request.body.decode("utf-8"))

        if "state" not in data or data["state"] is None:
            log.error("request body missing 'state'")
            return HttpResponseBadRequest("request body missing 'state'")

        agent = get_agent_by_id(agent_id)
        agent.inc_request()

        start_t = time.time_ns()
        response = agent.instance.request(data["state"], **data.get('kwargs', {}))
        performance_logger.info("Request Elapse Time: {} ms".format((time.time_ns()-start_t)/(1e6)))

        global save_agent
        if save_agent:
            # log.warning('Agent is being saved! This is probably not working.')
            agent.save()

        # pr.disable()
        # pr.dump_stats("al.cprof")
        last_call_time = time.time_ns()

        if isinstance(response, Sai):
            temp = {'selection': response.selection,
                    'action': response.action,
                    'inputs': response.inputs,
                    'mapping': {"?sel": response.selection},
                    'how': "n/a",
                    'skill_label': 'n/a'}
           
            return HttpResponse(
                    json.dumps({'selection': response.selection,
                                'action': response.action,
                                'inputs': response.inputs,
                                'mapping': {"?sel": response.selection},
                                'responses': [temp]
                                }))

        return HttpResponse(json.dumps(response))

    except Exception as exp:
        log.exception('ERROR IN REQUEST')
        log.error('POST data:')
        log.error(data)

        # pr.disable()
        # pr.dump_stats("al.cprof")
        return HttpResponseServerError(str(exp))


@csrf_exempt
def get_skills(http_request, agent_id):
    """
    """

    # pr.enable()
    try:
        if http_request.method != "POST":
            return HttpResponseNotAllowed(["POST"])
        data = json.loads(http_request.body.decode("utf-8"))

        if "states" not in data or data["states"] is None:
            log.error("request body missing 'states'")
            return HttpResponseBadRequest("request body missing 'states'")

        agent = get_agent_by_id(agent_id)
        agent.inc_request()
        response = agent.instance.get_skills(data["states"])

        global save_agent
        if save_agent:
            # log.warning('Agent is being saved! This is probably not working.')
            agent.save()

        # pr.disable()
        # pr.dump_stats("al.cprof")

        return HttpResponse(json.dumps(response))

    except Exception as exp:
        traceback.print_exc()

        # pr.disable()
        # pr.dump_stats("al.cprof")
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
    global last_call_time
    performance_logger.info("Interface Feedback Time: {} ms".format((time.time_ns()-last_call_time)/1e6))
    # pr.enable()
    try:
        if http_request.method != "POST":
            return HttpResponseNotAllowed(["POST"])
        data = json.loads(http_request.body.decode("utf-8"))

        # print(data)

        errs = []

        if "state" not in data or data["state"] is None:
            errs.append("request body missing 'state'")
        if "skill_label" not in data or data["skill_label"] is None:
            data["skill_label"] = "NO_LABEL"
        if "foci_of_attention" not in data or data["foci_of_attention"] is None:
            data["foci_of_attention"] = None
        if "selection" not in data or data["selection"] is None:
            errs.append("request body missing 'selection'")
        if "action" not in data or data["action"] is None:
            errs.append("request body missing 'action'")
        if "inputs" not in data or data["inputs"] is None:
            errs.append("request body missing 'inputs'")
        if "reward" not in data or data["reward"] is None:
            if "correct" in data:
                data["reward"] = 2 * int(data["correct"] == True) - 1
            else:
                errs.append("request body missing 'reward'")

        # Linter was complaining about too many returns so I consolidated all
        # of the errors above
        if len(errs) > 0:
            for err in errs:
                log.error(err)
            # print("errors: {}".format(",".join(errs)))
            return HttpResponseBadRequest("errors: {}".format(",".join(errs)))

        agent = get_agent_by_id(agent_id)
        agent.inc_train()

        sai = Sai(
            selection=data["selection"],
            action=data["action"],
            inputs=data["inputs"],
        )

        data['sai'] = sai
        del data['selection']
        del data['action']
        del data['inputs']

        start_t = time.time_ns()
        response = agent.instance.train(**data)
        performance_logger.info("Train Elapse Time: {} ms".format((time.time_ns()-start_t)/(1e6)))

        global save_agent
        if save_agent:
            # log.warning('Agent is being saved! This is probably not working.')
            agent.save()

        # pr.disable()
        # pr.dump_stats("al.cprof")
        last_call_time = time.time_ns()

        if(response is not None):
            return HttpResponse(json.dumps(response))
        else:
            return HttpResponse("OK")

    except Exception as exp:
        log.exception('ERROR IN TRAIN')
        log.error('POST data:')
        log.error(data)
        # log.error(data)
        # traceback.print_exc()

        # pr.disable()
        # pr.dump_stats("al.cprof")
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
        data = json.loads(http_request.body.decode("utf-8"))

        errs = []

        if "state" not in data:
            errs.append("request body missing 'state'")
        if "selection" not in data:
            errs.append("request body missing 'selection'")
        if "action" not in data:
            errs.append("request body missing 'action'")
        if "inputs" not in data:
            errs.append("request body missing 'inputs'")

        if len(errs) > 0:
            for err in errs:
                log.error(err)
            # print("errors: {}".format(",".join(errs)))
            return HttpResponseBadRequest("errors: {}".format(",".join(errs)))

        agent = Agent.objects.get(id=agent_id)
        agent.inc_check()
        agent.save()

        response = {}

        sai = Sai(
            selection=data["selection"],
            action=data["action"],
            inputs=data["inputs"],
        )

        data['sai'] = sai
        del data['selection']
        del data['action']
        del data['inputs']

        response["reward"] = agent.instance.check(**data)
        return HttpResponse(json.dumps(response))

    except Exception as exp:
        log.exception('ERROR IN TRAIN')
        log.error('POST data:')
        log.error(data)
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
        "id": agent.id,
        "name": agent.name,
        "num_request": agent.num_request,
        "num_train": agent.num_train,
        "num_check": agent.num_check,
        "created": agent.created,
        "updated": agent.updated,
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
def test_view(http_request):
    return render(
        http_request, "apprentice_learner/tester.html", {"agents": Agent.objects.all()}
    )
