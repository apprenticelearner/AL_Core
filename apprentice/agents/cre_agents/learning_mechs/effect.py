''' Effect learning: what actions do.  DRAFT.

    AL's other mechanisms learn which action to take (how), on what (where),
    in which circumstances (when) and in preference to what (which). None of
    them learns what an action changes. In a tutor that is supplied by hand:
    each ActionType carries an apply_expected_change function, used by
    CREAgent.predict_next_state. In a game the agent has to find out.

    An effect-learning mechanism is given a state, an action taken in it, and
    the state that followed, and learns to predict the change. It is the
    agent's forward model. Uses:
      * thinking ahead: picture the result of an action before taking it;
      * a learned replacement for apply_expected_change;
      * a prediction to be surprised by, which is what accounts of curiosity
        and affect need.

    States here are the py_dict form: {id : {attribute : value}}.
    An effect is a delta, a list of (object, attribute, change), where object
    is an id, or "sel" for the object acted on, so that an effect learned on
    one object can be applied to another.

    Status: the interface and one simple mechanism (DeltaTable). Not yet used
    by predict_next_state, and nothing yet generalises over what is acted on
    beyond a key supplied by the caller. See docs/game_agents.md.
'''
from abc import ABCMeta, abstractmethod

from ..dipl_base import register_effect


def observed_delta(before, after, selection):
    ''' What changed between two py_dict states, as a delta. Only numbers on
        objects present in both are compared. '''
    delta = []
    for oid, obj in before.items():
        other = after.get(oid)
        if(other is None):
            continue
        for attr, v in obj.items():
            w = other.get(attr)
            if(isinstance(v, bool) or isinstance(w, bool)):
                continue
            if(isinstance(v, (int, float)) and isinstance(w, (int, float)) and w != v):
                delta.append(("sel" if oid == selection else oid, attr, w - v))
    return delta


def apply_delta(state, delta, selection):
    ''' The py_dict state pictured after an effect, or None if the effect could
        not happen: no quantity may go below zero. The input is not modified. '''
    pictured = dict(state)
    for ref, attr, change in delta:
        oid = selection if ref == "sel" else ref
        if(oid not in pictured or attr not in pictured[oid]):
            return None
        value = pictured[oid][attr] + change
        if(value < 0):
            return None
        pictured[oid] = {**pictured[oid], attr: value}
    return pictured


class BaseEffect(metaclass=ABCMeta):
    def __init__(self, agent, **kwargs):
        self.agent = agent

    @abstractmethod
    def ifit(self, state, action, next_state, **kwargs):
        ''' Learn from one observed transition. '''

    @abstractmethod
    def predict(self, state, action, **kwargs):
        ''' The expected delta of an action in a state, or None if unknown. '''

    def predict_state(self, state, action, **kwargs):
        ''' The expected next state, or None if unknown or impossible. '''
        delta = self.predict(state, action, **kwargs)
        if(delta is None):
            return None
        return apply_delta(state, delta, action.selection)


@register_effect
class DeltaTable(BaseEffect):
    ''' The simplest effect learner: remember the last change seen for each
        kind of action. An action's kind is (key of the object acted on,
        action type, input), where the key is the value of `key_attr` on the
        object ("things with the same key behave alike"), or its type if it
        has no such attribute. An effect can also be told directly, for games
        that show what an action will do before it is taken. '''

    def __init__(self, agent, key_attr="kind", **kwargs):
        super().__init__(agent, **kwargs)
        self.key_attr = key_attr
        self.table = {}

    def _key(self, state, action):
        obj = state.get(action.selection, {})
        kind = obj.get(self.key_attr, obj.get("type", None))
        return (kind, action.action_type, action.input)

    def ifit(self, state, action, next_state, **kwargs):
        delta = observed_delta(state, next_state, action.selection)
        if(len(delta) > 0):
            self.table[self._key(state, action)] = delta

    def tell(self, state, action, delta):
        ''' Record an effect that was shown instead of observed. '''
        self.table[self._key(state, action)] = list(delta)

    def predict(self, state, action, **kwargs):
        return self.table.get(self._key(state, action), None)
