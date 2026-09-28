# E20 decision basis for current contract tests

## Scope

This document records why each expected result in `verification_protocol.md` follows from a field contract, constraint, geometric relation, or error-handling rule. It explains the current 14-case suite used for the final reported execution. Cases added after defect review are identified as regression cases and are not represented as having existed before the corresponding defect was found.

The actual observations and pass decisions are stored in `result.json` under the matching case identifier.

## Decision basis

| ID | Expected result basis |
|---|---|
| F01 | The output must satisfy the total-asset limit, mission-specific minimum quantities, target coverage, and the MUM-T ratio defined by the formation constraints. |
| F02 | ISR requires at least two reconnaissance UAVs while the input limits the total formation to one asset. The constraints are contradictory, so downstream execution must be rejected. |
| F03 | Every generated asset must preserve the formation count, use an allowed mission value, and either use a valid target index or remain unassigned where the schema permits it. |
| V01 | The configured mission order places ISR before SEAD and SEAD before STRIKE. Only the reversed input violates this ordering contract. |
| V02 | A path with all sampled positions outside the rectangular no-fly zone must pass, while a path containing a position inside the rectangle must be reported as a violation. |
| V03 | A target outside the rectangular no-fly zone must pass, while a target inside the rectangle must be reported as a violation. |
| V04 | On flat reference terrain, MSL and AGL are equal. Therefore 500 m satisfies the 300 m criterion and 100 m does not. |
| V05 | The separated paths remain beyond the configured proximity criterion, whereas identical paths have zero separation and must be reported. |
| V06 | A 1:2 manned-to-unmanned composition satisfies the configured ratio of two unmanned assets per manned asset; a 2:1 composition does not. |
| V07 | A path altitude of 500 m MSL over 450 m terrain gives 50 m AGL, which is below the 300 m criterion. |
| V08 | Both endpoints are outside the no-fly zone, but exact line-rectangle intersection shows that the connecting segment enters it. The segment must therefore be reported. |
| V09 | Both endpoints satisfy the altitude criterion, but the resampled segment crosses 450 m terrain at 500 m MSL, giving 50 m AGL. The segment must therefore be reported. |
| V10 | The synthetic threat field is below the decision threshold at both endpoints and above it inside the connecting segment. Segment resampling must therefore report the interior penetration. |
| P01 | The obstacle predicate defines every point inside the synthetic circle as invalid. The accepted smoothed or fallback route must contain zero invalid points after 90 m segment resampling. |

## Interpretation

These decision bases support implementation-contract checks only. They do not define an operationally correct mission plan or validate the simulated threat model against measured data.
