from agents.BaseAgent import BaseAgent


class SoarTechAgent(BaseAgent):
    """
    A SoarTech version of an Apprentice Agent.
    """

    def __init__(self, feature_set, function_set):
        # Just track the state as a set of Facts?
        self.last_state = None

        # Need a working memory class
        self.working_memory = None

        for skill in feature_set:
            self.working_memory.declare_skill(skill)

    def request(self, state):
        """
        Accepts a JSON object representing the state.

        Returns a dictionary containing selection, action, and inputs.
        """
        if self.last_state is None:
            return self.request_diff(state, [])

        pos_diff = state - self.last_state
        neg_diff = self.last_state - state

        return self.request_diff(pos_diff, neg_diff)

    def request_diff(self, state_pos_diff, state_neg_diff):
        """
        Accepts a JSON object representing diffs from the previously requested
        state. Useful for more efficiently making requests when the state
        changes only a little bit.

        Returns a dictionary containing selection, action, and inputs.
        """
        # relational inference step?
        # there could be a repeated process of updating, but maybe we wanna
        # chose the skills that get applied, so do the loop below.
        self.working_memory.retract(state_neg_diff)
        self.working_memory.declare(state_pos_diff)

        while True:
            candidate_skills = [skill for skill in self.working_memory.agenda]
            if len(candidate_skills) == 0:
                break
            best_skill = select_skill(candidate_skills, state)
            best_skill.activate()
            # what if the skills produces external action? Need to return it?
            # Does this happen within the skill?
            # Maybe activate returns something?
            # Maybe we need something special for recognizing SAI facts?

        # return empty action
        return {}

    def train(self, state, selection, action, inputs, reward, skill_label,
              foci_of_attention):
        """
        Accepts a JSON object representing the state, a string representing the
        skill label, a list of strings representing the foas, a string
        representing the selection, a string representing the action, list of
        strings representing the inputs, and a boolean correctness.
        """
        if self.last_state is None:
            return self.request_diff(state, [])

        pos_diff = state - self.last_state
        neg_diff = self.last_state - state

        return self.train_diff(pos_diff, neg_diff, selection, action, inputs,
                               reward, skill_lable, foci_of_attention)

    def train_diff(self, state_pos_diff, state_neg_diff, selection, action,
                   inputs, reward, skill_label, foci_of_attention):
        """
        Accepts a JSON object representing the state, a string representing the
        skill label, a list of strings representing the foas, a string
        representing the selection, a string representing the action, list of
        strings representing the inputs, and a boolean correctness.
        """
        # relational inference step?
        self.working_memory.retract(state_neg_diff)
        self.working_memory.declare(state_pos_diff)

        skill_sequence = self.explain(selection, action, inputs)

        # might need to distinguish here between a skill and a skill
        # instantiation; could return skill instantiation that has a pointer to
        # a skill
        if len(skill_sequence) == 1:
            skill = skill_sequence[0]
        else:
            skill = self.how_learning(skill_sequence)

        skill.update_where(self.working_memory, reward)
        skill.update_when(self.working_memory, reward)
