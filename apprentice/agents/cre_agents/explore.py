''' Explorers: what an agent does when no skill applies and there is no tutor to
    show it.

    With a tutor, CREAgent.act() returns nothing when no skill applies and the
    tutor demonstrates the step. With no tutor the agent has to try something.
    An explorer chooses among the actions the environment says are available.
    It is given what the agent's existing skills expect of those actions, so
    that trying things is informed by what has been learned: in particular, an
    action a skill predicts will go badly can be avoided.

    An explorer is a function
        explorer(agent, state, available_actions, expected, **args) -> Action or None
    where available_actions is a list of apprentice.shared.Action and expected
    maps action_key(action) to the best when-prediction any skill has for that
    action in this state (absent if no skill covers it).

    Select one with CREAgent(explorer="uniform" | "avoid_failures" | ...).
'''
import random as _random

from .dipl_base import register_explorer


@register_explorer
def uniform(agent, state, available_actions, expected, rng=None):
    ''' Any available action, with no use of what has been learned. '''
    if(len(available_actions) == 0):
        return None
    return (rng or _random).choice(available_actions)


@register_explorer
def avoid_failures(agent, state, available_actions, expected,
                   rng=None, prefer_untried=True):
    ''' Any available action the agent does not expect to go badly; if every
        one is expected to go badly, any of them. With prefer_untried, actions
        that no skill covers yet are tried before ones a skill covers but is
        unsure about. '''
    if(len(available_actions) == 0):
        return None
    rng = rng or _random

    from .cre_agent import action_key

    def pred(a):
        return expected.get(action_key(a), None)

    safe = [a for a in available_actions if pred(a) is None or pred(a) >= 0]
    pool = safe if len(safe) > 0 else list(available_actions)
    if(prefer_untried):
        untried = [a for a in pool if pred(a) is None]
        if(len(untried) > 0):
            pool = untried
    return rng.choice(pool)
