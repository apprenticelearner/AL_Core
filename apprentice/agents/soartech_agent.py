import random
from abc import ABCMeta
# from typing import Any
from typing import Collection
from typing import Dict

from experta import KnowledgeEngine

from apprentice.agents.base import BaseAgent
from apprentice.learners import WhenLearner
# from apprentice.learners.when_learners.actor_critic_learner import ActorCriticLearner
from apprentice.learners.when_learners.dqn_learner import DQNLearner
from apprentice.working_memory import ExpertaWorkingMemory
from apprentice.working_memory.base import WorkingMemory
from apprentice.working_memory.representation import Skill, Activation, Sai
from apprentice.working_memory.skills import FractionsEngine, fraction_skill_set


class SoarTechAgent(BaseAgent):
    """
    A SoarTech version of an Apprentice Agent.
    """

    def __init__(
        self,
        wm: WorkingMemory = ExpertaWorkingMemory(ke=KnowledgeEngine()),
        when: WhenLearner = DQNLearner(),
        epsilon: float = 0.3,
        action_penalty: float = -0.05,
        negative_actions: bool = False,
        skill_map: Dict[str, Skill] = fraction_skill_set,
        prior_skills: Collection[str] = frozenset(['click_done', 'check',
                                                   'update_answer',
                                                   'update_convert', 'add',
                                                   'multiply']),
        **kwargs
    ):
        # Just track the state as a set of Facts?
        # initialize to None, so gets replaced on first state.
        super().__init__()

        self.prior_state = {}
        self.last_activation = None
        self.last_sai = None

        # Need a working memory class
        if isinstance(wm, WorkingMemory):
            self.working_memory = wm
        if isinstance(wm, ABCMeta):
            self.working_memory = wm()

        print(prior_skills)
        print(epsilon)
        if prior_skills is not None:
            prior_skills = [skill_map[s]
                            for s in prior_skills if s in skill_map]
            self.working_memory.add_skills(prior_skills)

        # will take a activation and facts and return reward
        if when is not None:
            self.when_learning = when
        else:
            self.when_learning = None

        self.epsilon = epsilon
        self.action_penalty = action_penalty
        self.negative_actions = negative_actions

    def __deepcopy__(self, memo):
        print()
        print("DEEP COPY NOT IMPLEMENTED -- RETURNING NONE!")
        print()
        return None

    def select_activation(
        self, candidate_activations: Collection[Activation]
    ) -> Activation:
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
            return random.choice(candidate_activations), 0

        if random.random() < self.epsilon:
            print('random action')
            return random.choice(candidate_activations), 0

        activations = [
            (
                self.when_learning.eval(state=self.working_memory.state,
                                        action=activation),
                random.random(),
                activation,
            )
            for activation in candidate_activations
        ]
        activations.sort(reverse=True)

        # print('q values')
        # from pprint import pprint
        print([s for s, _, _ in activations])
        # print([(s, a.get_rule_name()) for s, _, a in activations])
        expected_reward, _, best_activation = activations[0]
        return best_activation, expected_reward

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
        candidate_activations = [
            activation for activation in self.working_memory.activations
        ]

        while True:
            if len(candidate_activations) == 0:
                return {}
            best_activation, expected_reward = self.select_activation(
                candidate_activations)
            state = self.working_memory.state

            if not self.negative_actions and expected_reward < 0.0:
                return {}

            output = self.working_memory.activation_factory.to_ex_activation(
                best_activation
            ).fire(self.working_memory.ke)

            if isinstance(output, Sai):
                break

            candidate_activations = [
                activation for activation in self.working_memory.activations
            ]
            next_state = self.working_memory.state

            if self.when_learning:
                self.when_learning.update(
                    state, best_activation, self.action_penalty, next_state,
                    candidate_activations
                )

        self.last_activation = best_activation
        self.last_sai = output

        return output

    def train_diff(self, state_diff, next_state_diff, sai, reward, skill_label,
                   foci_of_attention):
        """
        Need the diff for the current state as well as the state diff for
        computing the state that results from taking the action. This is
        needed for performing Q learning.

        Accepts a JSON object representing the state, a string representing the
        skill label, a list of strings representing the foas, a string
        representing the selection, a string representing the action, list of
        strings representing the inputs, and a boolean correctness.
        """
        if self.last_sai == sai and state_diff == {}:
            print("MATCHING LAST SAI")
            self.update_final(self.working_memory.state, reward,
                              next_state_diff, self.last_activation)
            return

        # print(sai)
        # print(state_diff)
        self.working_memory.update(state_diff)

        # This should do essentially what `engine.run` is doing from
        # PyKnow. Pyknow currently uses salience to choose rule order, but
        # we want to essentially set salience using the when learning.
        output = None

        candidate_activations = [
            activation for activation in self.working_memory.activations
        ]

        while True:

            # from pprint import pprint
            # pprint(self.working_memory.state)

            # outer loop checks if the sai is the one we're trying to explain
            while True:
                # inner loop is essentially request, just keep expanding until
                # you get sais
                # print("LEN CANDIDATES", len(candidate_activations))

                if len(candidate_activations) == 0:
                    # TODO add a rule that generates the "input" into working
                    # memory, so it can be explained via recall and update.
                    print("#####################")
                    print("FAILURE TO EXPLAIN!!!")
                    print("#####################")
                    return {}

                best_activation, expected_reward = self.select_activation(
                    candidate_activations)

                # from pprint import pprint
                # pprint(self.working_memory.state)
                # print('firing', best_activation.get_rule_name())
                state = self.working_memory.state

                output = self.working_memory.activation_factory.to_ex_activation(
                    best_activation
                ).fire(self.working_memory.ke)

                if isinstance(output, Sai):
                    break

                candidate_activations = [
                    activation for activation in
                    self.working_memory.activations
                ]
                next_state = self.working_memory.state

                if self.when_learning:
                    self.when_learning.update(
                        state, best_activation, self.action_penalty,
                        next_state, candidate_activations
                    )

            # print('trying', output, 'vs.', sai)
            if output != sai:
                # print('failed!')
                # print()
                # candidate_activations = [act for act in candidate_activations
                #                          if act != best_activation]

                # if the reward is positive, then we assume any other action is
                # a negative example. So we train on the explity negatives.
                if self.when_learning and reward > 0:
                    self.when_learning.update(
                        state, best_activation,
                        self.action_penalty - 1.0,
                        # state, []
                        state, candidate_activations
                    )

                continue
            # print('success!')
            # print()

            self.update_final(state, reward, next_state_diff, best_activation)

            break

    def update_final(self, state, reward, next_state_diff, best_activation):
        """
        Just assume for now that all reward states are essentially terminal.
        """
        if next_state_diff is None:
            next_state = None
            candidate_activations = []

        else:
            self.working_memory.update(next_state_diff)
            next_state = self.working_memory.state
            candidate_activations = [activation for activation in
                                     self.working_memory.activations]

        if self.when_learning:
            self.when_learning.update(
                state, best_activation,
                self.action_penalty + reward,
                # next_state, []
                next_state, candidate_activations
            )

    def train_diff_old(self, state_diff, next_state_diff, sai, reward,
                       skill_label, foci_of_attention):
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

        while sai_activation is None:
            candidate_activations = [
                activation for activation in self.working_memory.activations
            ]
            if len(candidate_activations) == 0:
                return
            for activation in candidate_activations:
                state = self.working_memory.state
                output = self.working_memory.activation_factory.to_ex_activation(
                    activation
                ).fire(self.working_memory.ke)
                if output == sai:
                    sai_activation = activation
                    break

        if next_state_diff is None:
            next_state = None
            next_activations = []
        else:
            self.working_memory.update(next_state_diff)
            next_state = self.working_memory.state
            next_activations = list(self.working_memory.activations)

        if self.when_learning and state and sai_activation:
            self.when_learning.update(
                state, sai_activation, reward, next_state, next_activations
            )

        # activation_sequence = None
        # for act in self.working_memory.activations():
        #    output = act.fire()
        #    # from the method args.
        #    if output == sai:
        #        activation_sequence = [act]

        # if activation_squence is None:
        #    raise Exception("no explaination")

        # if len(activation_sequence) == 1:
        #    activation = activation_sequence[0]
        # else:
        #    # compile discovered activation seq into new skill and return
        #    # activation of it
        #    activation = self.how_learning(activation_sequence)

        # activation has pointers to skill, state context, and match
        # information; still working out what this interface looks like.
        # activation.update_where(self.working_memory, reward)
        # activation.update_when(self.working_memory, reward, next_state_diff)

    def train_last_state(self, *args):
        pass
