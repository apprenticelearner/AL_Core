import logging
import random
from pprint import pformat
from abc import ABCMeta
# from typing import Any
from typing import Collection
from typing import Dict

from apprentice.agents.diff_base import DiffBaseAgent
from apprentice.learners import WhenLearner
from apprentice.learners.when_learners.dqn_learner import DQNLearner
from apprentice.working_memory import ExpertaWorkingMemory
from apprentice.working_memory.base import WorkingMemory
from apprentice.working_memory.representation import Skill, Activation, Sai
from apprentice.working_memory.experta_skills import experta_skill_map
from apprentice.explain.explanation import Explanation
from experta import KnowledgeEngine
import colorama

colorama.init(autoreset=True)

log = logging.getLogger(__name__)


class ExpertaAgent(DiffBaseAgent):
    """
    An Experta version of an Apprentice Agent.

    Initially developed at Soartech
    """

    def __init__(
            self,
            wm: WorkingMemory = None,
            when: WhenLearner = None,
            request_epsilon: float = 0.01,
            train_epsilon: float = 0.3,
            action_penalty: float = -0.05,
            gamma: float = 0.7,
            negative_actions: bool = False,
            skill_map: Dict[str, Skill] = experta_skill_map,
            prior_skills=None,
            chunking=False,
            **kwargs
    ):
        # Just track the state as a set of Facts?
        # initialize to None, so gets replaced on first state.
        super().__init__()
        self.last_activation = None
        self.last_sai = None
        if wm is None:
            wm = ExpertaWorkingMemory(ke=KnowledgeEngine())
        if when is None:
            when = DQNLearner(gamma=gamma)
        if prior_skills is None:
            prior_skills = {
                "click_done": True,
                "check": True,
                "equal": False,
                "update_answer": True,
                "update_convert": True,
                "add": True,
                "multiply": True,
            }

        # Need a working memory class
        if isinstance(wm, WorkingMemory):
            self.working_memory = wm
        if isinstance(wm, ABCMeta):
            self.working_memory = wm()

        log.debug(prior_skills)
        log.debug('GAMMA = %0.2f' % gamma)

        prior_skills = [
            skill_map[s]
            for s, active in prior_skills.items()
            if active
        ]

        self.working_memory.add_skills(prior_skills)

        # will take a activation and facts and return reward
        if when is not None:
            self.when_learning = when
        else:
            self.when_learning = None

        self.request_epsilon = request_epsilon
        self.train_epsilon = train_epsilon
        self.action_penalty = action_penalty
        log.debug("action_penalty" + str(action_penalty))
        self.negative_actions = negative_actions
        self.chunking = chunking

    # def __deepcopy__(self, memo):
    #     log.debug("DEEP COPY NOT IMPLEMENTED -- RETURNING NONE!")
    #     return None

    def select_activation(
            self, candidate_activations: Collection[Activation],
            is_train=False, pop=False
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

        if not is_train and random.random() < self.request_epsilon:
            return random.choice(candidate_activations), 0

        if is_train and random.random() < self.train_epsilon:
            return random.choice(candidate_activations), 0

        # activations = [
        #     (
        #         self.when_learning.eval(
        #             state=self.working_memory.state, action=activation
        #         ),
        #         random.random(),
        #         activation,
        #     )
        #     for activation in candidate_activations
        # ]

        reward = self.when_learning.eval_multiple(
            state=self.working_memory.state, actions=candidate_activations)
        activations = [(reward[i], random.random(), activation) for i,
                       activation in enumerate(candidate_activations)]

        activations.sort(reverse=True)

        log.debug(
            "LENGTH OF ACTIVATIONS " + str(len(activations)))
        log.debug(str([(a[0], a[2].get_rule_name()) for a in activations]))

        expected_reward, _, best_activation = activations[0]
        return best_activation, expected_reward

    def request_diff(self, state_diff: Dict):
        """
        Queries the agent to get an action to execute in the world given the
        diff from the last state.

        :param state_diff: a state diff output from JSON Diff.
        """
        # Just loads in the differences from the state diff

        tabu = set()
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
                candidate_activations
            )

            candidate_activations.remove(best_activation)
            state = self.working_memory.state

            if not self.negative_actions and expected_reward < 0.0:
                return {}

            output = self.working_memory.activation_factory.to_ex_activation(
                best_activation
            ).fire(self.working_memory.ke)

            tabu.add(
                (
                    best_activation.get_rule_name(),
                    frozenset(best_activation.get_rule_bindings().items()),
                )
            )

            if isinstance(output, Sai):
                break

            candidate_activations = [
                activation
                for activation in self.working_memory.activations
                if (
                    activation.get_rule_name(),
                    frozenset(activation.get_rule_bindings().items()),
                )
                not in tabu
            ]

            next_state = self.working_memory.state

            if self.when_learning:
                self.when_learning.update(
                    state,
                    best_activation,
                    self.action_penalty,
                    next_state,
                    candidate_activations,
                )

        self.last_activation = best_activation
        self.last_sai = output

        return output

    def train_diff(self, state_diff, next_state_diff, sai, reward):
        """
        Need the diff for the current state as well as the state diff for
        computing the state that results from taking the action. This is
        needed for performing Q learning.

        Accepts a JSON object representing the state, a string representing the
        skill label, a list of strings representing the foas, a string
        representing the selection, a string representing the action, list of
        strings representing the inputs, and a boolean correctness.
        """
        tabu = set()

        if self.last_sai == sai and state_diff == {}:
            self.update_final(
                self.working_memory.state, reward, next_state_diff,
                self.last_activation
            )
            return

        self.working_memory.update(state_diff)
        # log.debug(pformat(self.working_memory.facts))

        # This should do essentially what `engine.run` is doing from
        # PyKnow. Pyknow currently uses salience to choose rule order, but
        # we want to essentially set salience using the when learning.
        output = None

        candidate_activations = [
            activation for activation in self.working_memory.activations
        ]

        while True:

            # outer loop checks if the sai is the one we're trying to explain
            while True:
                # inner loop is essentially request, just keep expanding until
                # you get sais

                if len(candidate_activations) == 0:
                    # TODO add a rule that generates the "input" into working
                    # memory, so it can be explained via recall and update.
                    log.debug("#####################")
                    log.debug("FAILURE TO EXPLAIN!!!")
                    log.debug(str(sai))
                    log.debug("#####################")
                    return {}

                best_activation, expected_reward = self.select_activation(
                    candidate_activations, is_train=True, pop=True
                )
                candidate_activations.remove(best_activation)
                state = self.working_memory.state

                output = self.working_memory.activation_factory\
                    .to_ex_activation(
                        best_activation
                    ).fire(self.working_memory.ke)

                tabu.add(
                    (
                        best_activation.get_rule_name(),
                        frozenset(best_activation.get_rule_bindings().items()),
                    )
                )

                if isinstance(output, Sai):
                    if self.chunking:
                        ex = Explanation(output)
                        self.working_memory.add_rule(ex.new_rule)
                    break

                candidate_activations = [
                    activation
                    for activation in self.working_memory.activations
                    if (
                        activation.get_rule_name(),
                        frozenset(activation.get_rule_bindings().items()),
                    )
                    not in tabu
                ]

                next_state = self.working_memory.state

                if self.when_learning:
                    self.when_learning.update(
                        state,
                        best_activation,
                        self.action_penalty,
                        next_state,
                        candidate_activations,
                    )

            log.debug("trying output:" + str(output) + " vs. demo:" + str(sai))
            if output != sai:
                # log.debug('failed!')
                log.debug("failed!")

                candidate_activations = [
                    activation
                    for activation in self.working_memory.activations
                    if (
                        activation.get_rule_name(),
                        frozenset(activation.get_rule_bindings().items()),
                    )
                    not in tabu
                ]

                # if the reward is positive, then we assume any other action is
                # a negative example. So we train on the explity negatives.
                if self.when_learning and reward > 0:
                    self.when_learning.update(
                        state,
                        best_activation,
                        self.action_penalty - 1.0,
                        # state, []
                        state, candidate_activations,
                    )

                continue

            log.debug("success!")

            self.update_final(state, reward, next_state_diff, best_activation)

            break

    def update_final(self, state, reward, next_state_diff, best_activation):
        """
        Do the final update for an observable SAI producing activation.
        """
        if next_state_diff is None:
            next_state = None
            candidate_activations = []

        else:
            self.working_memory.update(next_state_diff)
            # log.debug(pformat(self.working_memory.facts))
            next_state = self.working_memory.state
            candidate_activations = [
                activation for activation in self.working_memory.activations
            ]
        log.debug("LENGTH OF ACTIVATIONS" + str(len(candidate_activations)))

        if self.when_learning:
            self.when_learning.update(
                state,
                best_activation,
                self.action_penalty + reward,
                # next_state, []
                next_state, candidate_activations,
            )

    def check(self, state: Dict, sai: Sai, **kwargs) -> float:
        """
        Checks the correctness (reward) of an SAI action in a given state.
        """
        raise NotImplementedError("Check not implemented for Experta agent.")


if __name__ == "__main__":
    import copy
    a = ExpertaAgent()
    b = copy.deepcopy(a)
