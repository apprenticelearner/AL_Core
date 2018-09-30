from agents.BaseAgent import BaseAgent


class Stub(BaseAgent):
    """
    Just a dummy agent that requests no actions, doesn't learn, and returns
    false for all checks. Made for testing the API.
    """
    def request(self, state):
        return {}

    def train(self, state, selection, action, inputs, reward, skill_label,
              foci_of_attention):
        pass

    def check(self, state, selection, action, inputs):
        return False
