# E22 threat + terrain end-to-end regression — pre-registration

Written before any execution of `run_e2e_threat.py`. This scenario is frozen.
Changing threat coordinates, path, margin, seed, or pass thresholds after seeing
a FAIL in order to manufacture a PASS is cherry-picking and is explicitly
disallowed by the D3 honesty protocol. Only genuine harness/config bugs
(terrain tile path errors, threat_layout typos, wiring mistakes) may be fixed,
and any such fix must be logged below with what/why, without touching the
scenario itself.

## Codebase under test

`<repo>/` (the codex tree, NOT
the original `10. IMPS` tree — this matches the reproducibility package used
by earlier experiments in this same tree, e.g. E21, V12).

App root: `mission_plan-new_plan_250519/`

## Frozen scenario

- Route: Osan (오산(Osan)) -> Pyongyang, target lat=39.03, lon=125.75
- Mission: SEAD, STRIKE (doctrine engine will insert SEAD ahead of STRIKE if
  the router ever omits it, but the router is expected to already return
  `["SEAD", "STRIKE"]` for this exact command)
- MUM-T = True, 3D = True (A* 3D), RTB = True
- Safety margin = 8 km
- Max assets = 6
- Command string (parsed by the deterministic rule router in `LLMBrain`, no
  generative LLM call):

  ```
  MUM-T mission from Osan. target lat=39.03 lon=125.75. SEAD STRIKE, 3D, RTB,
  safety margin 8 km, maximum 6 assets.
  ```

- Threats (5), `threat_layout` keyed against `modules.mission_state.THREAT_DB`
  (identical to V12 CONFIG_A layout):

  | db_name | lat | lon |
  |---|---|---|
  | S-75 (번개-1) | 38.40 | 126.35 |
  | SNR-75 (Fan Song) 화력제어 레이더 | 38.40 | 126.35 |
  | S-125 (번개-3) | 39.00 | 125.80 |
  | SNR-125 (LOW BLOW) 화력통제 레이더 | 39.00 | 125.80 |
  | SA-13 (번개-2) | 39.05 | 125.70 |

- Terrain: real SRTM tiles `mission_plan-new_plan_250519/data/terrain/srtm_62_05.tif`
  and `srtm_62_06.tif`, loaded through `modules.terrain_loader.TerrainLoader`
  with `use_fallback` required to end up `False` (verified at run start; if it
  is `True` the run is invalid and must be reported as a harness failure, not
  silently accepted).
- Determinism: no generative LLM in the loop (deterministic command router
  only). No random seed is consumed by the modules under test in this path
  (MILP + Hungarian assignment + A* are deterministic given identical input),
  so "same seed" reduces to "same input, run twice, compare".

## Six-stage pass criteria (adapted from E21 `verification_protocol.md`)

| ID | Stage | Criterion |
|---|---|---|
| E01 | Natural-language routing | Deterministic router (`_model_used == "fast-rule"`) returns `MISSION_PLAN`, base resolves to the Osan airport key, target (39.03, 125.75), mission sequence `["SEAD","STRIKE"]`, `A* 3D`, RTB=True, margin=8.0 |
| E02 | Doctrine + MILP + Hungarian assignment | BM25 doctrine engine returns >=1 reference; policy passed to optimizer; result is feasible, <=6 total assets, satisfies policy MUM-T ratio, every asset assigned a mission within {SEAD, STRIKE} |
| E03 | 3D route generation (real terrain) | Every ingress/required egress has >=2 points; fighter route reaches target within 0.25 km; every non-expendable asset's RTB route ends within 0.25 km of Osan |
| E04 | Rule checking (with real threats + real terrain) | Report is produced and includes the expected rule families (MISSION_SEQUENCE, TARGET_IN_NFZ, MUMT_RATIO, NFZ_*, THREAT_*, ALT_*, RANGE_*, ASSET_COLLISION where applicable). Whether ERROR-level violations are raised is reported as-is — this criterion checks that the rule engine *ran and produced a verdict*, not that the verdict is clean, since the whole point of running with real threats/terrain is to see what it actually finds. |
| E05 | Output generation | Folium HTML and CZML JSON nonempty, CZML contains every planned asset ID |
| E06 | Infeasible flow | An ISR-only formation request with `max_total=1` is marked infeasible and no route/output stage is entered |

A run "PASSES" only if all 6 report PASS. Anything else is reported as X/6,
with the failing stage(s) and their raw evidence, not rounded up or hidden.

## Supplementary (not part of the 6-stage score, reported separately)

- SEAD/STRIKE chain consistency: number of assets MILP assigns to SEAD vs. the
  number of threats a naive greedy corridor-opening procedure (direct
  Osan->Pyongyang line, `XAIUtils.calculate_risk_score`, tau=0.5) needs to
  suppress before the corridor opens. This is diagnostic, not a pass/fail
  gate, and reported honestly either way.
- Determinism check: run the full scenario twice in fresh processes and
  compare corridor-open-at-k / final_max_risk / route waypoint counts and a
  hash of the JSON result.

## Debugging discipline

If any stage fails on first run, the only permitted fixes are genuine harness
bugs (e.g., wrong terrain path, `threat_layout` referencing a `THREAT_DB` key
that doesn't exist, a wiring mistake passing the wrong terrain object where a
`.get()` vs `.get_elevation()` interface mismatch causes a silent 0m
fallback). Any such fix will be listed by name with a one-line reason. The
scenario table above (route, threats, margin, mission types) will not change
for any reason after this file is written.
