# Acting without a tutor (branch `game-agents`, DRAFT)

These changes let a `CREAgent` act in an environment that has no tutor, such as a
game. Every option is off by default, and a default agent behaves as before.
The rationale is in `AL_Games/docs/al_core_audit.md`.

## Options on `CREAgent`

| Option | Default | Effect |
|---|---|---|
| `strict_action_types` | `False` | A demonstration is explained only by skills with the same action type. Without it, two actions with the same input but different types merge into one skill. |
| `induce_from_negative` (alias `learn_from_failure`) | `False` | A skill first seen failing learns where it applies from that example, so the failure is remembered as something to avoid. Still under consideration as a long-term commitment. |
| `explorer`, `explorer_args` | `None` | What `act` does when no skill applies and it was given `available_actions`. Registered: `uniform`, `avoid_failures`. |
| `effect`, `effect_args` | `None` | A mechanism that learns what actions change (`learning_mechs/effect.py`). Registered: `delta_table`. |
| `evaluator`, `evaluator_args` | `None` | The agent's own judgement of how an action went (`learning_mechs/evaluate.py`). Registered: `take_at_face_value`, `discounting`. |

## Changed calls

- `act(state, available_actions=None)`: when given, only skill applications among the
  available actions are used, and the explorer is the fallback. Actions are compared
  with `action_key`, which tolerates `6`, `6.0` and `"6"`.
- `train(state, action, reward=None, evidence=None)`: `evidence` goes through the
  evaluator to become the reward. If the evaluator returns `None`, nothing is learned.
  With no evaluator, evidence is the reward.
- `get_skill_applications(..., include_negative=False)`: with `True`, also returns
  applications expected to fail.
- New: `observe_effect`, `predict_effect`, `evaluate`.
- New filter `positive_only`: drops every application expected to fail.

## Fixes that apply to tutors too

- `shared.py`: `Action` built from an object read `inp.input` before `inp` was set.
- `dipl_base.py`: missing `and` in `is_bottom_out_exception`.
- `where.py` / `cre_agent.py`: a skill induced from a negative first example has no
  where-conditions and crashed the next explanation. It is now inert.

## Not done yet

- `DeltaTable` is not used by `predict_next_state`, which is still broken under
  cre 0.5.1 (`MemSet.modify` from pure Python).
- Goal induction and delayed judgement still live in AL_Games. They are meant to become
  the first real evaluator, since the goal an agent pursues is its own.
- AL_Games `GameAgent` still uses its own wrapper logic, not these options.
- The older tests under `cre_agents/tests` were stale before this branch and were not run.

## Tests

`apprentice/agents/cre_agents/tests/test_game_agents.py`: 18 pass, 1 expected failure
(`predict_next_state`). Run with `PYTHONPATH` set to this checkout and `-o addopts=""`.
