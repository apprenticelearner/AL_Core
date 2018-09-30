from agents.BaseAgent import BaseAgent


class Memo(BaseAgent):
    """
    Memorizes the state actions pairs and responds with the highest reward,
    demonstrated action for a given request.

    Made for testing the API.
    """
    def __init__(self, action_set):
        self.lookup = {}

    def request(self, state):
        print(state)

        action, _ = self.lookup.get(state, ({}, 0))
        return action

    def train(self, state, action, reward, label=None):
        prior_action, prior_reward = self.lookup.get(state, ({}, 0))
        if reward >= prior_reward:
            self.lookup[state] = (action, reward)
