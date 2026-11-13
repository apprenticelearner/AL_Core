''' Evaluation: the agent's own judgement of how an action went.  DRAFT.

    train() takes a reward. With a tutor the reward is the tutor's verdict, and
    AL has taken it at face value. But a verdict is evidence about what is
    being asked for, not the goal itself. A tutor implies a goal through what
    it marks right and wrong; what the learner is actually trying to do is its
    own affair. A student can be working for green ticks rather than for
    understanding. In a tutor the two mostly coincide. In a game they come
    apart: the game implies goals loosely, offers several at once, and leaves
    the player free to adopt, weight or ignore them.

    An evaluator sits where the reward comes from. It is given evidence about
    an action's outcome and returns the agent's evaluation of that action:
    a reward, or None for "nothing to learn from this".
        evidence in  : whatever the environment can offer. A tutor's verdict;
                       the state before and after; a reaction from a game
                       (a win, a trophy, a loss); text; what the display
                       draws attention to.
        reward out   : what train() then passes to where, when and which.

    Use:  agent.train(state, action, evidence=...)   instead of reward=...
    With no evaluator configured, evidence is used as the reward, so a tutor's
    verdict passed as evidence behaves exactly as it does today.

    Status: the interface and the pass-through default only. The goal
    induction developed in AL_Games (inducing what a game rewards from
    verdicts, salience, priors about games and text; judging actions whose
    consequences come late) is the intended first real evaluator, and is to be
    moved here once its form has settled. See docs/game_agents.md.
'''
from abc import ABCMeta, abstractmethod

from ..dipl_base import register_evaluator


class BaseEvaluator(metaclass=ABCMeta):
    def __init__(self, agent, **kwargs):
        self.agent = agent

    @abstractmethod
    def evaluate(self, state, action, evidence, **kwargs):
        ''' The agent's evaluation of `action` taken in `state`, given
            `evidence` about how it went: a reward, or None. '''

    def goals(self):
        ''' A description of what the agent currently takes itself to be
            pursuing, for inspection. '''
        return None


@register_evaluator
class TakeAtFaceValue(BaseEvaluator):
    ''' Trust the evidence completely: it is the reward. This is how a tutor's
        feedback has always been treated, made explicit. '''
    def evaluate(self, state, action, evidence, **kwargs):
        return evidence

    def goals(self):
        return "whatever the feedback says"


@register_evaluator
class Discounting(BaseEvaluator):
    ''' A minimal example of an agent that does not simply adopt the goal it is
        given: it attends to feedback only some of the time (`attention`), and
        can weigh being told it is wrong differently from being told it is
        right (`heed_negative`). Not a model of anything in particular; it
        shows the shape of the mechanism and gives the tests something other
        than the pass-through to exercise. '''
    def __init__(self, agent, attention=1.0, heed_negative=1.0, seed=None, **kwargs):
        super().__init__(agent, **kwargs)
        import random
        self.attention = attention
        self.heed_negative = heed_negative
        self.rng = random.Random(seed)

    def evaluate(self, state, action, evidence, **kwargs):
        if(self.rng.random() >= self.attention):
            return None
        if(evidence is not None and evidence < 0 and self.rng.random() >= self.heed_negative):
            return None
        return evidence

    def goals(self):
        return f"the feedback, attended to {self.attention:.0%} of the time"
