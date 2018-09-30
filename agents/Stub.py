from agents.BaseAgent import BaseAgent


class Stub(BaseAgent):
    """
    Just a dummy agent that requests no actions, doesn't learn, and returns
    false for all checks. Made for testing the API.
    """
    def request(self, state):
        return {}

    def train(self, state, label, foas, selection, action, inputs, correct):
        pass

    def check(self, state, selection, action, inputs):
        return False
