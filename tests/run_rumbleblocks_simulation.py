from time import sleep
import requests

from concept_formation.datasets import load_rb_s_07
from concept_formation.datasets import load_rb_s_07_human_predictions
from concept_formation.preprocessor import ObjectVariablizer

ov = ObjectVariablizer()
towers = {t['_guid']: ov.transform(t) for t in load_rb_s_07()}
human_predictions = load_rb_s_07_human_predictions()

key = None
sequences = {}

for line in human_predictions:
    line = line.rstrip().split(",")
    if key is None:
        key = {v: i for i, v in enumerate(line)}
        continue

    user = line[key['user_id']]
    guid = line[key['instance_guid']]
    if user not in sequences:
        sequences[user] = []
    sequences[user].append(guid)

hostname = 'http://localhost:8000'

for user in sequences:

    # create agent
    print("CREATING AGENT")
    sleep(0.01)
    response = requests.post(hostname + "/create/",
                             json={"agent_type": "WhereWhenHowNoFoa",
                                   "action_set":
                                   "fraction arithmetic prior knowledge"},
                             timeout=10000)
    print("RESP", response.json())
    agent_id = response.json()['agent_id']

    # assume buttion 1 = success
    # assume button 2 = failure

    for guid in sequences[user]:

        tower = {a: towers[guid][a] for a in towers[guid] if a != 'success' and
                a != '_guid'}
        correctness = bool(int(towers[guid]['success']))
        print("CORERCTNESS", correctness, type(correctness))

        print("Requesting action for %s" % guid)
        response = requests.post(hostname + "/request/" + agent_id + "/",
                                 json={'state': tower})
        print("RESP", response.json())
        if response.json() == {}:
            print("Noop, requesting hint")
        else:
            print("PREDICTED: %s - %s" % (response.json()['selection'],
                                          response.json()['selection'] ==
                                          "Category%s" % correctness))

        print("TRAINING CORRECT", "Category%i" % int(correctness))
        response = requests.post(hostname + "/train/" + agent_id + "/",
                                 json={'state': tower,
                                       'label': 'choose',
                                       'correct': True,
                                       'foas': {},
                                       'selection': "Category%i" % int(correctness),
                                       'action': 'PressButton',
                                       'inputs': {}},
                                 timeout=10000)
        print('RESP', response)
        print("TRAINING INCORRECT", "Category%i" % int(not correctness))
        response = requests.post(hostname + "/train/" + agent_id + "/",
                                 json={'state': tower,
                                       'label': 'choose',
                                       'correct': False,
                                       'foas': {},
                                       'selection': "Category%i" % int(not correctness),
                                       'action': 'PressButton',
                                       'inputs': {}},
                                 timeout=10000)

        print('RESP', response)

        # request prediction

        # train on prediction

        # log performance
