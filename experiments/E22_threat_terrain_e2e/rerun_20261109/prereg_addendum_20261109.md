# E22 re-run addendum (2026-11-09, JKSS revision 1)

Written before the re-run. The frozen scenario in `../prereg.md` is unchanged
(route, threats, margin, mission types, 6 scored stages, pass criteria).

## Why
Reviewers R3606-1/2, R3632-2 and R3631-1 asked for per-module inputs/outputs,
termination reasons and the threshold actually used. The August run did not keep
these, and no raw pair of two runs was archived.

## What is added (harness-side tracing only)
- `verify_e22_trace.py` runs the original `run_e2e_threat.py` main() unchanged and
  wraps five functions of `modules/path_computer.py` at runtime
  (`_select_sead_targets`, `_resolve_asset_objective`, `_threats_for_asset`,
  `_run_one_path_cached`, `_smooth`). Each wrapper calls the original and returns
  its result unchanged. 0 lines changed in `modules/` and in `run_e2e_threat.py`.
- Read-only probes per route call: block threshold, threat-core membership and
  risk at the start/end points, A* console message (node limit / open set exhausted).
- Outputs go to `rerun_20261109/<run>/`; the August files are not overwritten.

## Runs
1. `run1`, `run2`: codex tree, two fresh processes, compare hashes (wall-clock fields excluded).
2. `public_run`: public release tree (`imps-release`), same scenario, terrain dir
   pointed at the codex SRTM tiles (the public tree does not ship them). Only the
   harness APP constant is overridden for this run.

## Known issue found before the run (reported, not fixed)
The supplementary `cross_module_check` reads SEAD bindings from
`formation.to_dict()["assets"][i]["target_name"]`, a key that the asset dict does
not contain (August result.json shows keys asset_id, asset_type, callsign, base,
assigned_mission, assigned_target_idx, path). Its `sead_targets_bound=False`
(`[None, None]`) therefore may reflect the missing key, not the routing module.
The routing module selects SEAD targets separately in
`path_computer._select_sead_targets`. The trace records that mapping directly.
The harness check is left unchanged so the re-run stays comparable with August;
the correction is reported from the trace.
