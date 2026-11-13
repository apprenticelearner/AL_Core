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

Registered names are matched ignoring case and underscores, so `"delta_table"`,
`"DeltaTable"` and `"deltatable"` are the same.

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
- `environment.py`: `predict_next_state` failed when the input was a number and the field
  a string. The input is now converted to the action type's declared type.
- `cre_agent.py`: `get_skill_applications` returned the unfiltered list instead of the
  applications it could track.

## Status of the audit's problems

Numbers are those in the audit's "Problems found in AL_Core and CRE" list.

| # | Problem | Status on this branch |
|---|---|---|
| 1 | `predict_next_state` fails when given a number for a string field | Fixed: `ActionType.coerce_input` converts the input to the declared type. It was never a CRE incompatibility. |
| 2 | Skill first seen failing crashes the next explanation | Fixed: the skill is inert. `induce_from_negative` optionally makes it useful. |
| 3 | Skills matched across action types | Fixed behind `strict_action_types`; default unchanged. |
| 4 | `thresholds` filter hands back probable failures | Default unchanged. `positive_only` added as the alternative. |
| 5 | `get_skill_applications` builds `out` but returns `skill_apps` | Fixed: returns `out`. The two differ only when a rollout tracker is present and `predict_next_state` fails for an application, so this is not exercised by the tests here, which have no process learning. |
| 6 | Missing `and` in `is_bottom_out_exception` | Fixed. |
| 7 | `Action` reads `inp` before assignment | Fixed. |
| 8 | `tests/test_agent.py` is stale and does not import | Not fixed. `test_game_agents.py` carries the regression tests instead. |
| 9 | When-learners treat numbers as categories | Not addressed. |
| 10 | CRE caches fact types by name | Not addressed (CRE). Worked around in AL_Games. |
| 11 | CRE relative encoder needs a shared base and a reference | Not addressed (CRE). Worked around in AL_Games. |

## Not done yet

- `DeltaTable` is not used by `predict_next_state`, which still relies on the
  hand-written expected change of each action type.
- Goal induction and delayed judgement still live in AL_Games. They are meant to become
  the first real evaluator, since the goal an agent pursues is its own.
- AL_Games `GameAgent` still uses its own wrapper logic, not these options.
- The older tests under `cre_agents/tests` were stale before this branch and were not run.

## Tests

`apprentice/agents/cre_agents/tests/test_game_agents.py`: 22 pass. Run with `PYTHONPATH` set to this checkout and `-o addopts=""`.
