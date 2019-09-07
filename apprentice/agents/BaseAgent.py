class BaseAgent(object):
    """
    This is essentially an abstract class for apprentice learner agents.
    """

    def __init__(self, feature_set, function_set):
        pass

    def request(self, state):
        """
        Accepts a JSON object representing the state.

        Returns a dictionary containing selection, action, and inputs.
        """
        raise NotImplementedError("request function not implemented")

    def train(self, state, selection, action, inputs, reward, skill_label,
              foci_of_attention):
        """
        Accepts a JSON object representing the state, a string representing the
        skill label, a list of strings representing the foas, a string
        representing the selection, a string representing the action, list of
        strings representing the inputs, and a boolean correctness.
        """

        raise NotImplementedError("train function not implemented")

    def check(self, state, selection, action, inputs):
        """
        Accepts a JSON object representing the state, a string representing the
        selection, a string representing the action, list of strings
        representing the inputs.

        Uses the learned model to determine the correctness of the provided sai
        in the provided state. Returns a boolean.
        """
        raise NotImplementedError("check function not implemented")
