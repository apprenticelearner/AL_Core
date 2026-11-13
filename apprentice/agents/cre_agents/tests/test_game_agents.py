''' Tests for the options that let a CREAgent act without a tutor, and regression
    tests pinning down the tutor behaviour those options must not change.

    Every option tested here is off by default. The first group of tests uses a
    default agent and must keep passing unchanged.
'''
import contextlib
import io
import random

import pytest

from apprentice.agents.cre_agents.cre_agent import CREAgent
from apprentice.agents.cre_agents.tests.test_state import new_mc_addition_state
from apprentice.shared import Action

FUNCS = ["Add3", "Mod10", "Add", "Div10", "Copy"]


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def new_agent(**config):
    return CREAgent(feature_set=[], where="antiunify", function_set=FUNCS, **config)


def teach_no_carry(agent, a, b):
    state = new_mc_addition_state(a, b)
    for col in range(3):
        digit = (a // 10 ** col % 10) + (b // 10 ** col % 10)
        sel = f"{col}_answer"
        quiet(agent.train, state, (sel, "UpdateTextField", digit), 1,
              arg_foci=[f"{col}_upper", f"{col}_lower"])
        state[sel].update({"value": str(digit), "locked": True})


def proposals(agent, state, **kwargs):
    apps = quiet(agent.act_all, state, return_kind="skill_app", **kwargs)
    return {(sa.selection, float(sa.input)) for sa in apps}


# --------------------------------------------------------------------------
# : Tutor behaviour (default agent). Must not change.

@pytest.fixture(scope="module")
def trained():
    agent = new_agent()
    teach_no_carry(agent, 333, 333)
    teach_no_carry(agent, 121, 212)
    return agent


def test_untrained_agent_proposes_nothing():
    agent = new_agent()
    assert quiet(agent.act_all, new_mc_addition_state(124, 342)) == []
    assert quiet(agent.act, new_mc_addition_state(124, 342)) is None


def test_generalises_from_demonstrations(trained):
    assert proposals(trained, new_mc_addition_state(124, 342)) == {
        ("0_answer", 6.0), ("1_answer", 6.0), ("2_answer", 4.0)}


def test_locked_fields_are_not_proposed(trained):
    state = new_mc_addition_state(124, 342)
    state["0_answer"].update({"value": "6", "locked": True})
    assert proposals(trained, state) == {("1_answer", 6.0), ("2_answer", 4.0)}


def test_act_without_available_actions_is_unchanged(trained):
    state = new_mc_addition_state(124, 342)
    sa = quiet(trained.act, state, return_kind="skill_app")
    assert (sa.selection, float(sa.input)) in proposals(trained, state)


def test_defaults_are_the_tutor_settings():
    agent = new_agent()
    assert agent.strict_action_types is False and agent.induce_from_negative is False
    assert agent.explorer is None and agent.effect_lrn_mech is None and agent.evaluator is None


def test_dict_inputs_are_rejected():
    ''' Documents the scalar-input contract. '''
    agent = new_agent()
    with pytest.raises(TypeError):
        quiet(agent.train, new_mc_addition_state(333, 333),
              ("0_answer", "UpdateTextField", {"value": 6}), 1, arg_foci=["0_upper", "0_lower"])


@pytest.mark.parametrize("inp", ["6", 6, 6.0])
def test_predict_next_state_applies_action(trained, inp):
    ''' The input may be written as a number although the field holds a
        string. Before, only "6" worked. '''
    state = new_mc_addition_state(124, 342)
    out = quiet(trained.predict_next_state, state, ("0_answer", "UpdateTextField", inp), json_friendly=True)
    field = out["next_state"]["0_answer"]
    assert field["value"] == "6" and field["locked"] is True
    assert out["next_state_uid"] != out["state_uid"]


def test_all_proposed_applications_are_returned(trained):
    ''' get_skill_applications returns the applications it could track. With
        no rollout tracker (as here) that is all of them. '''
    state = new_mc_addition_state(124, 342)
    assert len(proposals(trained, state)) == 3


# --------------------------------------------------------------------------
# : A skill first seen failing

def negative_first(agent):
    state = new_mc_addition_state(333, 333)
    quiet(agent.train, state, ("0_answer", "UpdateTextField", 7), -1, arg_foci=["0_upper", "0_lower"])
    return state


def test_negative_first_example_no_longer_breaks_a_default_agent():
    ''' Before: a skill induced from a failure had no where-conditions, and the
        next explanation raised on it. Now it is inert. '''
    agent = new_agent()
    state = negative_first(agent)
    assert quiet(agent.act_all, state) == []
    # A later demonstration on the same state used to crash while scoring the
    #  pattern-less skill as a candidate explanation.
    quiet(agent.train, state, ("0_answer", "UpdateTextField", 6), 1, arg_foci=["0_upper", "0_lower"])
    assert ("0_answer", 6.0) in proposals(agent, state)


def test_induce_from_negative_keeps_the_failure_as_something_to_avoid():
    agent = new_agent(induce_from_negative=True)
    state = negative_first(agent)
    skill = list(agent.skills.values())[0]
    assert skill.where_lrn_mech.conds is not None          # where it applies: learned from the failure
    assert proposals(agent, state) == set()                # not proposed
    state_obj = agent.standardize_state(state)
    known = quiet(agent.get_skill_applications, state_obj, ignore_filter=True, include_negative=True)
    assert [(sa.selection, sa.when_pred) for sa in known] == [("0_answer", -1)]   # but known, and expected to fail


# --------------------------------------------------------------------------
# : Skills and action types

def skills_after_same_value_under_two_action_types(**config):
    ''' Enter a value no function explains (so the skill's how-part is just
        that value) with one action type, then the same value with another.
        Returns the number of skills afterwards. '''
    agent = CREAgent(feature_set=[], where="antiunify", function_set=[], **config)
    state = new_mc_addition_state(333, 333)
    quiet(agent.train, state, ("0_answer", "UpdateTextField", 77), 1)
    quiet(agent.train, state, ("1_answer", "UpdateField", 77), 1)
    return len(agent.skills)


def test_action_types_are_kept_apart_only_when_asked():
    # Tutor behaviour: a skill is matched on how its input is produced alone.
    assert skills_after_same_value_under_two_action_types() == 1
    assert skills_after_same_value_under_two_action_types(strict_action_types=True) == 2


# --------------------------------------------------------------------------
# : Filter for acting

def test_positive_only_filter():
    from apprentice.agents.cre_agents.dipl_base import positive_only, thresholds

    class App:
        def __init__(self, pred): self.when_pred = pred
    bad, doubtful, unsure, good = App(-1.0), App(-0.6), App(None), App(0.8)
    assert positive_only(None, [bad, doubtful, unsure, good]) == [unsure, good]
    assert positive_only(None, [doubtful]) == []
    # The default drops a certain failure but, with nothing better on offer, 
    #  hands back what it thinks will probably fail.
    assert thresholds(None, [bad]) == []
    assert thresholds(None, [doubtful]) == [doubtful]


# --------------------------------------------------------------------------
# : Trying things

def available(state):
    return [("0_answer", "UpdateTextField", 6), ("0_answer", "UpdateTextField", 7)]


def test_explorer_tries_an_available_action_when_no_skill_applies():
    agent = new_agent(explorer="uniform")
    state = new_mc_addition_state(333, 333)
    assert quiet(agent.act, state) is None                              # not told what is available: as a tutor
    choice = quiet(agent.act, state, available_actions=available(state))
    assert isinstance(choice, Action) and choice.as_tuple() in available(state)
    as_json = quiet(agent.act, state, available_actions=available(state), json_friendly=True)
    assert as_json["explored"] and as_json["selection"] == "0_answer"


def test_no_explorer_means_no_exploring():
    agent = new_agent()
    state = new_mc_addition_state(333, 333)
    assert quiet(agent.act, state, available_actions=available(state)) is None


def test_exploring_steers_clear_of_what_is_expected_to_fail():
    agent = new_agent(explorer="avoid_failures", induce_from_negative=True,
                      explorer_args={"rng": random.Random(0)})
    state = negative_first(agent)                                       # writing 7 went badly
    picks = {quiet(agent.act, state, available_actions=available(state)).as_tuple()[2] for _ in range(20)}
    assert picks == {6}
    uniform = new_agent(explorer="uniform", induce_from_negative=True,
                        explorer_args={"rng": random.Random(0)})
    negative_first(uniform)
    picks = {quiet(uniform.act, state, available_actions=available(state)).as_tuple()[2] for _ in range(20)}
    assert picks == {6, 7}


def test_a_skill_is_used_only_if_it_is_available(trained):
    state = new_mc_addition_state(124, 342)
    only = [("1_answer", "UpdateTextField", 6.0)]
    sa = quiet(trained.act, state, return_kind="skill_app", available_actions=only)
    assert sa.selection == "1_answer"


# --------------------------------------------------------------------------
# : The agent's own evaluation

def test_evidence_is_the_reward_when_there_is_no_evaluator():
    agent = new_agent()
    state = new_mc_addition_state(333, 333)
    quiet(agent.train, state, ("0_answer", "UpdateTextField", 6), evidence=1, arg_foci=["0_upper", "0_lower"])
    assert ("0_answer", 6.0) in proposals(agent, state)
    assert agent.evaluate(state, ("0_answer", "UpdateTextField", 6), -1) == -1


def test_an_evaluator_can_decline_to_learn():
    agent = new_agent(evaluator="discounting", evaluator_args={"attention": 0.0})
    state = new_mc_addition_state(333, 333)
    out = quiet(agent.train, state, ("0_answer", "UpdateTextField", 6), evidence=1, arg_foci=["0_upper", "0_lower"])
    assert out is None and len(agent.skills) == 0
    # A reward given outright is still a reward: the evaluator only sees evidence.
    quiet(agent.train, state, ("0_answer", "UpdateTextField", 6), 1, arg_foci=["0_upper", "0_lower"])
    assert len(agent.skills) == 1


# --------------------------------------------------------------------------
# : What actions do

def test_delta_table_learns_and_transfers_an_effect():
    from apprentice.agents.cre_agents.learning_mechs.effect import DeltaTable, apply_delta
    mech = DeltaTable(None)
    before = {"a": {"kind": "land", "n": 0}, "bank": {"money": 10}}
    after = {"a": {"kind": "land", "n": 1}, "bank": {"money": 4}}
    build = Action(("a", "Build", "home"))
    assert mech.predict(before, build) is None
    mech.ifit(before, build, after)
    assert sorted(mech.predict(before, build)) == [("bank", "money", -6), ("sel", "n", 1)]
    # The same kind of thing elsewhere: the effect carries over, relative to it.
    other = {"b": {"kind": "land", "n": 0}, "bank": {"money": 10}}
    pictured = mech.predict_state(other, Action(("b", "Build", "home")))
    assert pictured["b"]["n"] == 1 and pictured["bank"]["money"] == 4 and other["b"]["n"] == 0
    # Not enough money: the effect cannot happen.
    poor = {"b": {"kind": "land", "n": 0}, "bank": {"money": 2}}
    assert mech.predict_state(poor, Action(("b", "Build", "home"))) is None
    assert apply_delta(poor, [("sel", "n", 1)], "b")["b"]["n"] == 1


def test_agent_exposes_its_effect_mechanism():
    agent = new_agent(effect="delta_table")
    state = new_mc_addition_state(333, 333)
    assert agent.predict_effect(state, ("0_answer", "UpdateTextField", 6)) is None
    agent.observe_effect(state, ("0_answer", "UpdateTextField", 6), state)      # nothing numeric changed
    assert agent.predict_effect(state, ("0_answer", "UpdateTextField", 6)) is None
    assert new_agent().predict_effect(state, ("0_answer", "UpdateTextField", 6)) is None
