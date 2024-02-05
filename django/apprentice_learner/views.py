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
from apprentice.shared import rand_agent_uid
import importlib


log = logging.getLogger('al-django')
performance_logger = logging.getLogger('al-performance')

class LogElapse():
    def __init__(self, logger, message):
        self.logger = logger
        self.message = message
    def __enter__(self):
        self.t0 = time.time_ns()/float(1e6)
    def __exit__(self,*args):
        self.t1 = time.time_ns()/float(1e6)
        self.logger.info(f'{self.message}: {self.t1-self.t0:.6f} ms')

active_agent = None
active_agent_uid = None
dont_save = True

EMPTY_RESPONSE = {}


AGENT_PATHS = {
    # Format <Name : (module_path, ClassName)>
    "Stub": ('apprentice.agents.Stub', 'Stub'),
    "Memo": ('apprentice.agents.Memo', 'Memo'),
    "WhereWhenHowNoFoa": ('apprentice.agents.WhereWhenHowNoFoa', 'WhereWhenHowNoFoa'),
    "ModularAgent": ('apprentice.agents.ModularAgent', 'ModularAgent'),
    "RLAgent": ('apprentice.agents.RLAgent', 'RLAgent'),
    "RHS_LHS_Agent" : ('apprentice.agents.RHS_LHS_Agent', 'RHS_LHS_Agent'),
    "CREAgent" : ('apprentice.agents.cre_agents.cre_agent', 'CREAgent'),
}

last_call_time = time.time_ns()


def get_agent_by_uid(uid):
    global active_agent, active_agent_uid, dont_save
    # print("active_agent_uid", active_agent_uid)
    agent_model = None
    # if uid == active_agent_uid:
    agent = active_agent
    # else:
    #     agent_model = Agent.objects.get(uid=uid)
    #     agent = agent_model.instance
    return agent, agent_model

def ensure_field(data, fields, errs=[], default=None):
    fields = fields if(isinstance(fields, tuple)) else (fields,)    
    for field in fields:
        if(field in data):
            return data[field]
    if(default is None):
        missing_str = ' | '.join([repr(f) for f in fields])
        plural = {'s' if len(fields) else ''}
        errs.append(f"Request body missing field{plural} {missing_str}.")
    return default

# --------------------------------------------------------------
# : List Agents

@csrf_exempt
def list_agents(http_request):
    if http_request.method != "GET":
        return HttpResponseNotAllowed(["GET"])

    agent_descrs = {}
    for model in Agent.objects.all():
        descr  = {
            "name": model.name,
            "num_request": model.num_request,
            "num_train": model.num_train,
            "num_check": model.num_check,
            "created": model.created,
            "updated": model.updated,
        }
        agent_descrs[model.uid] = descr

    if(dont_save and active_agent_uid): 
        agent_descrs[active_agent_uid] = {}

    return agent_descrs


# --------------------------------------------------------------
# : Create

def _standardize_create_data(http_request, errs=[], warns=[]):
    if http_request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    data = json.loads(http_request.body.decode("utf-8"))

    agent_type = ensure_field(data, ('agent_type', 'type'), errs)
    agent_name = ensure_field(data, ('agent_name', 'name'), errs, default='Unnamed Agent')    

    if agent_type and agent_type not in AGENT_PATHS:
        errs.append(f'Specified agent {agent_type!r} not supported')

    agent_args = ensure_field(data, ('agent_args', 'args'), errs)

    if "no_ops_parse" in data:
        warns.append(
            "Deprecation Warning: no_ops_parse is provided. This field is no"
            " longer used and is always assumed to be True. Operators must be"
            " registered in the system ahead of time and referenced by name.")

    if "feature_set" in data:
        warns.append(
            "Deprecation Warning: feature_set provided at top level."
            " feature_set is no longer used as a top level parameter."
            " It should now be included in the args object of the agent.")
        if "feature_set" not in args:
            args['feature_set'] = data.pop('feature_set', [])

    if "function_set" in data:
        warns.append(
            "Deprecation Warning: function_set provided at top level."
            " function_set is no longer used as a top level parameter."
            " It should now be included in the args object of the agent.")
        if "function_set" not in args:
            args['function_set'] = data.pop('function_set', [])

    if len(errs) > 0:
        for err in errs:
            log.error(err)
        return HttpResponseBadRequest(json.dumps({"errors": errs}))

    return {"type": agent_type, "name": agent_name, "args": agent_args}

def _make_agent(data, errs=[], warns=[]):
    global active_agent, active_agent_uid, dont_save

    # Get the path and class_name from AGENT_PATHS
    path, class_name = AGENT_PATHS.get(data['type'], (None, None))
    if(path is None):
        return HttpResponseServerError(
            f"No agent type registered with name {data['type']!r}.")

    # Import the agent class, make an agent instance, and give it a new uid.
    try:
        agent_class = getattr(importlib.import_module(path), class_name)
        agent = agent_class(**data['args'])
        # agent_model = Agent(instance=agent, name=data['name'])
        agent_uid = getattr(agent, 'uid', rand_agent_uid())
        # agent_model.uid = agent_uid

    # If any of that fails print traceback and send it to the client.
    except Exception as exp:
        tb_str = traceback.format_exc()
        message = f"Agent creation failed with exception:\n{tb_str}"
        log.error(message)
        return HttpResponseServerError(message)
    
    # Garbage collect any previous 'active_agent'.
    active_agent, active_agent_uid, dont_save = None, None, False

    if str(data.get("stay_active", True)).lower() == "true":
        active_agent = agent
        active_agent_uid = agent_uid
        dont_save = str(data.get("dont_save", True)).lower() == "true"
    else:
        warns.append(
            "Stability Warning: stay_active is set to false. Serialization and"
            " deserialization of agents is currently not working for most"
            " agent types. Expect Errors.")

    return {"agent_uid": str(agent_uid)}
    
# ** END POINT ** 
@csrf_exempt
def create(http_request):
    """
    Creates a new agent with the provided 'type', 'name', and 'args'.
    """
    errs, warns = [], []

    # Ensure request data is valid and in consitent format
    data = _standardize_create_data(http_request, errs, warns)
    if(isinstance(data, HttpResponse)): return data

    # Try to instantiate the agent 
    resp_data = _make_agent(data, errs, warns)
    if(isinstance(resp_data, HttpResponse)): return resp_data

    # Emit any warnings
    for w in warns:
        log.warn(w)

    return HttpResponse(json.dumps(resp_data))

@csrf_exempt
def get_active_agent(http_request):
    """
    Returns the uid of the active agent
    """
    global active_agent_uid
    return HttpResponse(json.dumps(active_agent_uid))


# --------------------------------------------------------------
# : Act, Act_All, Act_Rollout

def _standardize_act_data(http_request, errs=[], warns=[]):
    if http_request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    data = json.loads(http_request.body.decode("utf-8"))

    state = ensure_field(data, "state", errs)
    agent_uid = ensure_field(data, "agent_uid", errs)

    if len(errs) > 0:
        for err in errs:
            log.error(err)
        return HttpResponseBadRequest(json.dumps({"errors": errs}))

    # Only require state and agent_uid, but pass data as is since 
    #  agent+client can conceivably agree on additional fields.
    return data

def _standardize_act_response(response, return_all, errs=[], warns=[]):
    # Attempt to convert any agent specifc SAI implementation 
    #  to a json friendly format in case the agent implementation 
    #  of act() does not respect json_friendly=True.
    if(not response): return response
    actions = response if isinstance(response, list) else [response]
    if(not all([isinstance(a, dict) for a in actions])):
        warns.append(
            "Agent implementation does not respect json_friendly=True. "
            "Encountered an action which is not an instance of dict. "
            "Attempting translation..."
        )

        new_actions = []
        for a in actions:
            selection = getattr(a, 'selection')
            action_type = getattr(a, 'action_type', getattr(a, 'action'))
            inputs = getattr(a, 'inputs', getattr(a, 'input'))

            if(selection is None or action_type is None):
                #
                message = f"Agent response {a} missing selection or action_type fields."
                return HttpResponseServerError(str(exp))

            new_action = {
                "selection" : selection,
                "action_type" : action_type,
                "inputs" : inputs,
            }
            new_actions.append(new_action)
        actions = new_actions


    if(return_all):
        return actions
    else:
        return actions[0]

# ** END POINT ** 
@csrf_exempt
def act(http_request):
    """
    Takes an 'agent_uid' and 'state', and returns the highest priority next action that 
    the agent believes it should take next. If the request has return_all=True, then 
    a list of all next actions are returned instead. Each action should have at least  
    the fields 'selection', 'action_type', and 'inputs' (i.e. an SAI). The agent implementation
    is free to add additional action fields. For instance, an extended set of fields might 
    describe describing how an underlying skill within the agent applied each action
    (i.e. a SkillApplication).
    """
    global dont_save
    errs, warns = [], []

    # Ensure request data is valid and in consitent format
    data = _standardize_act_data(http_request, errs, warns)
    if(isinstance(data, HttpResponse)): return data

    try:
        agent, model = get_agent_by_uid(data['agent_uid'])
        if(model): model.inc_act()    

        # Call Act to get the action from the agent 
        with LogElapse(performance_logger, "act() elapse"):
            response = agent.act(**data, json_friendly=True)

        # Ensure response is indeed json_friendly
        response = _standardize_act_response(response, False, errs, warns)

        if not dont_save:
            log.warning('Agent is being saved! This is probably not working.')
            if(model): model.save()

        # Emit any warnings
        for w in warns:
            log.warn(w)

        return HttpResponse(json.dumps(response))

    except Exception as exp:
        tb_str = traceback.format_exc()
        message = f"act() failed with an exception:\n{tb_str}"
        log.error(message)
        return HttpResponseServerError(message)

# ** END POINT ** 
@csrf_exempt
def act_all(http_request):
    """
    Takes an 'agent_uid' and 'state', and returns the highest priority next action that 
    the agent believes it should take next. If the request has return_all=True, then 
    a list of all next actions are returned instead. Each action should have at least  
    the fields 'selection', 'action_type', and 'inputs' (i.e. an SAI). The agent implementation
    is free to add additional action fields. For instance, an extended set of fields might 
    describe describing how an underlying skill within the agent applied each action
    (i.e. a SkillApplication).
    """
    global dont_save
    errs, warns = [], []

    # Ensure request data is valid and in consitent format
    data = _standardize_act_data(http_request, errs, warns)
    if(isinstance(data, HttpResponse)): return data

    try:
        agent, model = get_agent_by_uid(data['agent_uid'])
        if(model): model.inc_act()    

        # Call Act to get the action from the agent 
        with LogElapse(performance_logger, "act_all() elapse"):
            response = agent.act_all(**data, json_friendly=True)

        # Ensure response is indeed json_friendly
        response = _standardize_act_response(response, True, errs, warns)

        if not dont_save:
            log.warning('Agent is being saved! This is probably not working.')
            if(model): model.save()

        # Emit any warnings
        for w in warns:
            log.warn(w)

        return HttpResponse(json.dumps(response))

    except Exception as exp:
        tb_str = traceback.format_exc()
        message = f"act_all() failed with an exception:\n{tb_str}"
        log.error(message)
        return HttpResponseServerError(message)


# ** END POINT ** 
@csrf_exempt
def act_rollout(http_request):
    """
    Applies act_all() repeatedly starting from 'state', and fanning out to create at 
    tree of all action rollouts up to some depth. At each step in this process the agent's 
    actions produce subsequent states based on the default state change defined by each 
    action's ActionType object. A list of 'halt_policies' specifies a set of functions that 
    when evaluated to false prevent further actions. Returns a tuple (states, action_infos).
    """
    global dont_save
    errs, warns = [], []

    # Ensure request data is valid and in consitent format
    data = _standardize_act_data(http_request, errs, warns)
    if(isinstance(data, HttpResponse)): return data

    try:
        agent, model = get_agent_by_uid(data['agent_uid'])
        if(model): model.inc_act()    

        # Call Act to get the action from the agent 
        with LogElapse(performance_logger, "act_rollout() elapse"):
            response = agent.act_rollout(**data, json_friendly=True)

        # print(response)

        if not dont_save:
            log.warning('Agent is being saved! This is probably not working.')
            if(model): model.save()

        # Emit any warnings
        for w in warns:
            log.warn(w)

        return HttpResponse(json.dumps(response))

    except Exception as exp:
        tb_str = traceback.format_exc()
        message = f"act_rollout() failed with an exception:\n{tb_str}"
        log.error(message)
        return HttpResponseServerError(message)


# ---------------------------------------------------------------------
# : Train, Train_All

def _del_keys(data,keys):
    for key in keys:
        if(key in data):
            del data[key]

def _standardize_train_data(http_request, errs=[], warns=[]):
    if http_request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    data = json.loads(http_request.body.decode("utf-8"))

    ensure_field(data, "agent_uid", errs)
    ensure_field(data, "state", errs)
    selection = ensure_field(data, "selection", errs)
    action_type = ensure_field(data, ('action_type', 'action'), errs)
    inputs = ensure_field(data, ('inputs', 'input'), errs)
    ensure_field(data, 'reward', errs)

    data['sai'] = (selection, action_type, inputs)
    _del_keys(data, ['selection', 'action_type', 'action', 'inputs', 'input'])

    if len(errs) > 0:
        for err in errs:
            log.error(err)
        return HttpResponseBadRequest(json.dumps({"errors": errs}))
    return data

# ** END POINT ** 
@csrf_exempt
def train(http_request):
    """
    Trains the Agent with an state annotated with the SAI used / with
    feedback.
    """
    global dont_save
    errs, warns = [], []

    data = _standardize_train_data(http_request, errs, warns)

    try:
        agent, model = get_agent_by_uid(data['agent_uid'])
        if(model): model.inc_train()

        with LogElapse(performance_logger, "train() elapse"):
            response = agent.train(**data)

        if not dont_save:
            log.warning('Agent is being saved! This is probably not working.')
            if(model): model.save()

        return HttpResponse(json.dumps(response))

    except Exception as exp:
        tb_str = traceback.format_exc()
        message = f"train() failed with an exception:\n{tb_str}"
        log.error(message)
        return HttpResponseServerError(message)

def _standardize_train_all_data(http_request, errs=[], warns=[]):
    if http_request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    data = json.loads(http_request.body.decode("utf-8"))

    # print("STANDARD TRAIN ALL", data)
    ensure_field(data, "training_set", errs)
    ensure_field(data, "agent_uid", errs)

    for item in data['training_set']:
        ensure_field(item, "state", errs)
        selection = ensure_field(item, "selection", errs)
        action_type = ensure_field(item, ('action_type', 'action'), errs)
        inputs = ensure_field(item, ('inputs', 'input'), errs)
        ensure_field(item, 'reward', errs)

        item['sai'] = (selection, action_type, inputs)
        _del_keys(item, ['selection', 'action_type', 'action', 'inputs', 'input'])

    if len(errs) > 0:
        for err in errs:
            log.error(err)
        return HttpResponseBadRequest(json.dumps({"errors": errs}))
    return data

@csrf_exempt
def train_all(http_request):
    """
    As train but trains examples multiple at once 
    """
    global dont_save
    errs, warns = [], []

    data = _standardize_train_all_data(http_request, errs, warns)
    try:
        agent, model = get_agent_by_uid(data['agent_uid'])
        if(model): model.inc_train()


        with LogElapse(performance_logger, "train_all() elapse"):
            response = agent.train_all(**data)

        if not dont_save:
            log.warning('Agent is being saved! This is probably not working.')
            if(model): model.save()

        return HttpResponse(json.dumps(response))

    except Exception as exp:
        tb_str = traceback.format_exc()
        message = f"train_all() failed with an exception:\n{tb_str}"
        log.error(message)
        return HttpResponseServerError(message)



# ---------------------------------------------------------------------
# : Explain Demo

def _standardize_explain_demo_data(http_request, errs=[], warns=[]):
    if http_request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    data = json.loads(http_request.body.decode("utf-8"))

    ensure_field(data, "agent_uid", errs)
    ensure_field(data, "state", errs)
    selection = ensure_field(data, "selection", errs)
    action_type = ensure_field(data, ('action_type', 'action'), errs)
    inputs = ensure_field(data, ('inputs', 'input'), errs)

    data['sai'] = (selection, action_type, inputs)
    _del_keys(data, ['selection', 'action_type', 'action', 'inputs', 'input'])

    if len(errs) > 0:
        for err in errs:
            log.error(err)
        return HttpResponseBadRequest(json.dumps({"errors": errs}))
    return data

@csrf_exempt
def explain_demo(http_request):
    errs, warns = [], []
    data = _standardize_explain_demo_data(http_request, errs, warns)

    try:
        agent, model = get_agent_by_uid(data['agent_uid'])
        if(model): model.inc_train()

        with LogElapse(performance_logger, "explain_demo() elapse"):
            response = agent.explain_demo(**data, json_friendly=True)

        return HttpResponse(json.dumps(response))

    except Exception as exp:
        tb_str = traceback.format_exc()
        message = f"explain_demo() failed with an exception:\n{tb_str}"
        log.error(message)
        return HttpResponseServerError(message)

def _standardize_get_state_uid_data(http_request, errs=[], warns=[]):
    if http_request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    data = json.loads(http_request.body.decode("utf-8"))

    ensure_field(data, "agent_uid", errs)
    ensure_field(data, "state", errs)

    if len(errs) > 0:
        for err in errs:
            log.error(err)
        return HttpResponseBadRequest(json.dumps({"errors": errs}))
    return data


@csrf_exempt
def get_state_uid(http_request):
    errs, warns = [], []
    data = _standardize_get_state_uid_data(http_request, errs, warns)
    
    try:
        agent, model = get_agent_by_uid(data['agent_uid'])
        if(model): model.inc_train()

        with LogElapse(performance_logger, "get_state_uid() elapse"):
            response = agent.get_state_uid(**data, json_friendly=True)

        return HttpResponse(json.dumps(response))

    except Exception as exp:
        tb_str = traceback.format_exc()
        message = f"get_state_uid() failed with an exception:\n{tb_str}"
        log.error(message)
        return HttpResponseServerError(message)


# ---------------------------------------------------------------------
# : Predict Next State

@csrf_exempt
def predict_next_state(http_request):
    errs, warns = [], []
    data = _standardize_explain_demo_data(http_request, errs, warns)
    
    try:
        agent, model = get_agent_by_uid(data['agent_uid'])
        if(model): model.inc_train()

        with LogElapse(performance_logger, "predict_next_state() elapse"):
            print(agent)

            response = agent.predict_next_state(**data, json_friendly=True)

        return HttpResponse(json.dumps(response))

    except Exception as exp:
        tb_str = traceback.format_exc()
        message = f"predict_next_state() failed with an exception:\n{tb_str}"
        log.error(message)
        return HttpResponseServerError(message)


# ---------------------------------------------------------------------
# : Check

def _standardize_check_data(http_request, errs=[], warns=[]):
    if http_request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    data = json.loads(http_request.body.decode("utf-8"))

    ensure_field(data, "agent_uid", errs)
    ensure_field(data, "state", errs)
    selection = ensure_field(data, "selection", errs)
    action_type = ensure_field(data, ('action_type', 'action'), errs)
    inputs = ensure_field(data, ('inputs', 'input'), errs)

    data['sai'] = (selection, action_type, inputs)
    _del_keys(data, ['selection', 'action_type', 'action', 'inputs', 'input'])

    if len(errs) > 0:
        for err in errs:
            log.error(err)
        return HttpResponseBadRequest(json.dumps({"errors": errs}))
    return data

# ** END POINT ** 
@csrf_exempt
def check(http_request):
    """
    Uses the knoweldge base to check the correctness of an SAI in provided
    state.
    """
    data = _standardize_check_data(http_request, errs, warns)

    try:
        agent, model = get_agent_by_uid(data['agent_uid'])
        if(model): model.inc_check()

        with LogElapse(performance_logger, "check() elapse"):
            reward = agent.check(**data)

        return HttpResponse(json.dumps({'reward' : reward}))

    except Exception as exp:
        tb_str = traceback.format_exc()
        message = f"Check failed with an exception:\n{tb_str}"
        log.error(message)
        return HttpResponseServerError(message)


# -------------------------------------------------
# : Get Skills

def _standardize_get_skills_data(http_request, errs=[], warns=[]):
    if http_request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    data = json.loads(http_request.body.decode("utf-8"))

    agent_uid = ensure_field(data, "agent_uid", errs)

    if len(errs) > 0:
        for err in errs:
            log.error(err)
        return HttpResponseBadRequest(json.dumps({"errors": errs}))

    return data

# ** END POINT ** 
@csrf_exempt
def get_skills(http_request):
    """
    """
    try:
        global dont_save
        errs, warns = [], []

        data = _standardize_get_skills_data(http_request, errs, warns)
        if(isinstance(data, HttpResponse)): return data

        agent, model = get_agent_by_uid(data['agent_uid'])

        with LogElapse(performance_logger, "get_skills elapse"):
            response = agent.get_skills(**data, json_friendly=True)

        return HttpResponse(json.dumps(response))

    except Exception as exp:
        tb_str = traceback.format_exc()
        message = f"Get skills failed with exception:\n{tb_str}"
        log.error(message)
        return HttpResponseServerError(message)

def _standardize_gen_completeness_profile_data(http_request, errs=[], warns=[]):
    if http_request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    data = json.loads(http_request.body.decode("utf-8"))

    agent_uid = ensure_field(data, "agent_uid", errs)
    ensure_field(data, "start_states", errs)
    ensure_field(data, "output_file", errs)

    if len(errs) > 0:
        for err in errs:
            log.error(err)
        return HttpResponseBadRequest(json.dumps({"errors": errs}))

    return data

# -------------------------------------------------------
# : Generating and Evaluating Completeness Profiles

# ** END POINT ** 
@csrf_exempt
def gen_completeness_profile(http_request):
    """
    """
    try:
        global dont_save
        errs, warns = [], []

        data = _standardize_gen_completeness_profile_data(http_request, errs, warns)
        if(isinstance(data, HttpResponse)): return data

        agent, model = get_agent_by_uid(data['agent_uid'])

        with LogElapse(performance_logger, "gen_completeness_profile elapse"):
            response = agent.gen_completeness_profile(**data)

        return HttpResponse(json.dumps(response))

    except Exception as exp:
        tb_str = traceback.format_exc()
        message = f"Gen Completeness Profile failed with exception:\n{tb_str}"
        log.error(message)
        return HttpResponseServerError(message)

def _standardize_eval_completeness_data(http_request, errs=[], warns=[]):
    if http_request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    data = json.loads(http_request.body.decode("utf-8"))

    agent_uid = ensure_field(data, "agent_uid", errs)
    ensure_field(data, "profile", errs)

    if len(errs) > 0:
        for err in errs:
            log.error(err)
        return HttpResponseBadRequest(json.dumps({"errors": errs}))

    return data

# ** END POINT ** 
@csrf_exempt
def eval_completeness(http_request):
    """
    """
    try:
        global dont_save
        errs, warns = [], []

        data = _standardize_eval_completeness_data(http_request, errs, warns)
        if(isinstance(data, HttpResponse)): return data

        agent, model = get_agent_by_uid(data['agent_uid'])

        with LogElapse(performance_logger, "eval_completeness elapse"):
            response = agent.eval_completeness(**data)

        return HttpResponse(json.dumps(response))

    except Exception as exp:
        tb_str = traceback.format_exc()
        message = f"Eval Completeness failed with exception:\n{tb_str}"
        log.error(message)
        return HttpResponseServerError(message)


# NOTE DEPRICATED
def report(http_request):
    """
    A simple method for looking up an agent's stats. I've found it to be
    really helpful for looking an agent's idea from its name to speed up
    processing. Also nice for stat counting.
    """

    if http_request.method != "GET":
        return HttpResponseNotAllowed(["GET"])

    model = get_object_or_404(Agent, id=agent_uid)

    response = {
        "id": model.id,
        "name": model.name,
        "num_request": model.num_request,
        "num_train": model.num_train,
        "num_check": model.num_check,
        "created": model.created,
        "updated": model.updated,
    }

    response = {k: str(response[k]) for k in response}
    return HttpResponse(json.dumps(response))


# def report_by_name(http_request, agent_name):
#     """
#     A version of report that can look up an agent by its name. This will
#     generally be slower but it also doesn't expose how the data is stored and
#     might be easier in some cases.
#     """
#     agent = get_list_or_404(Agent, name=agent_name)[0]
#     return report(http_request, agent.id)


# @csrf_exempt
# def test_view(http_request):
#     return render(
#         http_request, "apprentice_learner/tester.html", {"agents": Agent.objects.all()}
#     )
