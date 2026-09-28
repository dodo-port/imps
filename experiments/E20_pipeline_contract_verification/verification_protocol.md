# E20 module contract and regression protocol

## Purpose

This experiment checks implementation contracts for formation sizing, assignment, rule checking, terrain-relative altitude, and route post-processing. It does not evaluate operational mission quality or validate the threat model against measured data.

## Test-history classification

The cases were not all specified at the same time.

| Classification | Cases | History |
|---|---|---|
| Initial suite | F01-F03, V01-V06, P01 | The first run passed 8/10. V02 and P01 failed. |
| Added after the initial run | V07 | V07 fixed the terrain-relative altitude defect as an explicit regression case. The first full 11-case run passed 9/11; V07 and P01 failed. |
| Strengthened after defect review | F02, P01 | F02 now requires `is_feasible=False`; P01 checks resampled segments, not only saved waypoints. |
| Added after technical review | V08-V10 | Segment-level NFZ, terrain-ridge, and threat-penetration regressions. |

After the first two implementation fixes, the 11-case suite passed 11/11. The current reported suite contains 14 cases and must pass 14/14.

## Current cases and expected results

| ID | Area | Frozen condition | Expected result |
|---|---|---|---|
| F01 | MILP sizing | ISR-SEAD-STRIKE, two targets, maximum 12 assets | Optimal and all quantity, coverage, and MUM-T constraints satisfied. |
| F02 | MILP infeasibility | ISR, maximum one asset | Not optimal and `is_feasible=False`. |
| F03 | Assignment | Normal formation and two targets | Asset count and mission/target fields remain in their allowed domains. |
| V01 | Mission sequence | Normal and reversed sequences | Only the reversed sequence emits `MISSION_SEQUENCE`. |
| V02 | NFZ waypoint | Outside and inside waypoint paths | Only the inside path emits `NFZ_VIOLATION`. |
| V03 | Target in NFZ | Outside and inside targets | Only the inside target emits `TARGET_IN_NFZ`. |
| V04 | Minimum altitude | 500 m and 100 m over flat terrain | Only 100 m emits `MIN_ALTITUDE`. |
| V05 | Asset separation | Separated and identical paths | Only identical paths emit `ASSET_COLLISION`. |
| V06 | MUM-T ratio | 1:2 and 2:1 manned-to-unmanned counts | Only the insufficient condition emits `MUMT_RATIO`. |
| V07 | Terrain-relative altitude | 500 m MSL over 450 m terrain, 300 m AGL criterion | Emits `MIN_ALTITUDE`. |
| V08 | NFZ segment | Both endpoints outside; segment crosses NFZ | Emits `NFZ_VIOLATION`. |
| V09 | Terrain segment | Both endpoints clear; segment crosses a 450 m ridge | Emits `MIN_ALTITUDE`. |
| V10 | Threat segment | Both endpoints clear; segment crosses a high-risk area | Emits `THREAT_PENETRATION`. |
| P01 | Smoothed route | Three-point route around a synthetic obstacle | No resampled segment point enters the obstacle. |

Route segments are resampled at the 90 m DEM interval. NFZ intersection uses an exact line-rectangle clipping test.
