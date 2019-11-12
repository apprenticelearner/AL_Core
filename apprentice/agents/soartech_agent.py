import random
from abc import ABCMeta
from typing import Any
from typing import Collection
from typing import Dict

from apprentice.agents.base import BaseAgent
from apprentice.learners import WhenLearner
from apprentice.learners.when_learners.q_learner import QLearner
from apprentice.working_memory import ExpertaWorkingMemory
from apprentice.working_memory.base import WorkingMemory
from apprentice.working_memory.representation import Skill, Activation, Sai


class SoarTechAgent(BaseAgent):
    """
    A SoarTech version of an Apprentice Agent.
    """

    def __init__(self, prior_skills: Collection[Skill] = None, wm: WorkingMemory = ExpertaWorkingMemory(),
                 when: WhenLearner = QLearner):
        # Just track the state as a set of Facts?
        # initialize to None, so gets replaced on first state.
        super().__init__(prior_skills)
        self.prior_state = {}
        # Need a working memory class
        if isinstance(wm, WorkingMemory):
            self.working_memory = wm
        if isinstance(wm, ABCMeta):
            self.working_memory = wm()

        if prior_skills is not None:
            self.working_memory.add_skills(prior_skills)

        # will take a activation and facts and return reward
        if when is not None:
            self.when_learning = when()
        else:
            self.when_learning = None

    def select_activation(self, candidate_activations: Collection[
        Activation]) -> Activation:
        """
        Given a list of candidate skills evaluate them and determines which one
        has the highest expected rewared in the current state.

        .. todo::

            Not sure what a state looks like here? Is it just a collection of
                facts?
            Are candidate skills a collection of skills or skill activations?
            Might also need some kind of strategy for choosing, currently just
                choosing best.

        :param candidate_activations: Skills being considered for activation
        """
        # just passing in the working memory facts to each skill, where the
        # facts is just the current state representation.
        if self.when_learning is None:
            return random.choice(candidate_activations)

        activations = [
            (self.when_learning.evaluate(self.working_memory.state, activation), random.random(), activation) for
            activation in candidate_activations]
        activations.sort(reverse=True)
        expected_reward, _, best_activation = activations[0]
        return best_activation

    def request_diff(self, state_diff: Dict):
        """
        Queries the agent to get an action to execute in the world given the
        diff from the last state.

        :param state_diff: a state diff output from JSON Diff.
        """
        # Just loads in the differences from the state diff

        self.working_memory.update(state_diff)

        # This should do essentially what `engine.run` is doing from
        # PyKnow. Pyknow currently uses salience to choose rule order, but
        # we want to essentially set salience using the when learning.
        output = None
        candidate_activations = [activation for activation in
                                 self.working_memory.activations]
        while True:
            self.working_memory.step()


            if len(candidate_activations) == 0:
                return {}
            best_activation = self.select_activation(candidate_activations)
            state = self.working_memory.state

            output = self.working_memory.activation_factory.to_ex_activation(best_activation).fire(
                self.working_memory.ke)

            if isinstance(output, Sai):
                break

            candidate_activations = [activation for activation in
                                     self.working_memory.activations]
            next_state = self.working_memory.state

            if self.when_learning:
                self.when_learning.update(state, best_activation, 0, next_state, candidate_activations)

        return output

    def train_diff(self, state_diff, next_state_diff, sai, reward, skill_label, foci_of_attention):
        """
        Need the diff for the current state as well as the state diff for
        computing the state that results from taking the action. This is
        needed for performing Q learning.

        Accepts a JSON object representing the state, a string representing the
        skill label, a list of strings representing the foas, a string
        representing the selection, a string representing the action, list of
        strings representing the inputs, and a boolean correctness.
        """
        # relational inference step?
        self.working_memory.update(state_diff)

        state = None
        sai_activation = None
        for activation in self.working_memory.activations:
            state = self.working_memory.state
            output = self.working_memory.activation_factory.to_ex_activation(activation).fire(self.working_memory.ke)
            if output == sai:
                sai_activation = activation
                break

        self.working_memory.update(next_state_diff)
        next_state = self.working_memory.state
        next_activations = list(self.working_memory.activations)

        if self.when_learning and state and sai_activation:
            self.when_learning.update(state, sai_activation, reward, next_state, next_activations)

        #activation_sequence = None
        #for act in self.working_memory.activations():
        #    output = act.fire()
        #    # from the method args.
        #    if output == sai:
        #        activation_sequence = [act]

        #if activation_squence is None:
        #    raise Exception("no explaination")

        #if len(activation_sequence) == 1:
        #    activation = activation_sequence[0]
        #else:
        #    # compile discovered activation seq into new skill and return
        #    # activation of it
        #    activation = self.how_learning(activation_sequence)

        # activation has pointers to skill, state context, and match
        # information; still working out what this interface looks like.
        # activation.update_where(self.working_memory, reward)
        #activation.update_when(self.working_memory, reward, next_state_diff)

    def train_last_state(self, *args):
        pass
