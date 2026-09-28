# E21 end-to-end verification protocol

## Scope

This experiment checks one normal synthetic mission and one infeasible mission through the implemented module interfaces. It is implementation verification, not operational validation. The natural-language front end uses the deterministic command router in `LLMBrain`; a generative LLM call is intentionally excluded so the full test is repeatable without a local model service.

## Frozen normal scenario

- Command: `MUM-T mission from Osan. target lat=37.30 lon=127.30. ISR SEAD STRIKE, 3D, RTB, safety margin 5 km, maximum 6 assets.`
- Terrain: flat synthetic DEM, elevation 0 m MSL
- Threats: none
- Maximum assets: 6
- Route algorithm: optimized 3D A*
- Endpoint tolerance: 0.25 km

## Protocol revision history

- Revision 1 fixed the initial six-stage criteria before the first execution.
- The first execution exposed that a one-point RTB path could satisfy a weak nonempty-path condition. Revision 2 strengthened E03 to require at least two points and endpoint tolerance.
- Revision 2 also added the actual BM25 doctrine-policy stage to E02. Only results produced under Revision 2 are reported.

## Final success criteria fixed before the reported execution

| ID | Stage | Criterion |
|---|---|---|
| E01 | Natural-language routing | The parser returns `MISSION_PLAN`, Osan, target `(37.30, 127.30)`, `ISR -> SEAD -> STRIKE`, 3D A*, RTB, and 5 km margin. |
| E02 | Doctrine, MILP, and assignment | BM25 returns at least one doctrine reference; the resulting policy is passed to the optimizer; the result is feasible, has at most six assets, satisfies the policy MUM-T ratio, and assigns every asset a mission in the requested sequence. |
| E03 | 3D route generation | Every ingress and required egress has at least two points; the fighter route reaches the target within 0.25 km; every nonexpendable asset has an RTB route ending within 0.25 km of the base. |
| E04 | Rule checking | The final report contains no `ERROR`; the expected rule families were executed. |
| E05 | Output generation | Folium HTML and CZML JSON are nonempty, and CZML contains every planned asset ID. |
| E06 | Infeasible flow | An ISR mission with `max_total=1` is marked infeasible and no route/output stage is entered. |

The run passes only if all six criteria pass.
