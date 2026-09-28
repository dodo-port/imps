# E20 findings

## Execution history

- Initial suite first run: 8/10 passed; V02 and P01 failed.
- Expanded 11-case first full run: 9/11 passed; V07 and P01 failed.
- Post-fix regression: 11/11 passed.
- Technical-review regression suite: 14/14 passed.

## Defects found and changes made

1. NFZ checking sampled only selected waypoints. It was changed to inspect all stored waypoints and then strengthened to test every path segment by exact line-rectangle intersection.
2. Minimum-altitude checking compared MSL altitude directly with an AGL criterion. Terrain elevation is now subtracted, and each segment is resampled at the 90 m DEM interval.
3. Chaikin smoothing did not recheck constraints. Smoothed and fallback routes are now checked on resampled segments; an invalid smoothed route falls back to the raw route, and an invalid raw route returns empty.
4. The heuristic MILP fallback could exceed the asset limit while returning `is_feasible=True`. It now reapplies all hard constraints and marks the result infeasible when they cannot be met.
5. Threat penetration was sampled every fifth point. It now uses the same 90 m segment resampling as the terrain check.

## Interpretation

These results support implementation-contract claims only. They do not establish route optimality, operational feasibility, or measured threat-model validity. E21 separately checks the integrated synthetic workflow and endpoint criteria.
