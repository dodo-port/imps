"""Regression case I01 (added 2026-11-09, JKSS revision): terrain-cache / line-of-sight interface.
Expected: the system terrain cache (TerrainCacheFast) provides get_elevation() equal to get(),
and A* 3D near threats runs without an exception using the system cache (no harness shim).
Runs the check against the module before the fix (backup copy) and after the fix."""
import importlib.util, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; APP = HERE.parents[1] / "mission_plan-new_plan_250519"
sys.path.insert(0, str(APP)); sys.path.insert(0, str(HERE.parent / "E22_threat_terrain_e2e"))
from run_e2e_threat import build_threats, THREAT_LAYOUT
from modules.terrain_loader import TerrainLoader
import modules.pathfinder_optimized as after
spec = importlib.util.spec_from_file_location("pf_before", HERE / "pathfinder_optimized.before_20261109.py")
before = importlib.util.module_from_spec(spec); spec.loader.exec_module(before)
tl = TerrainLoader(dem_dir=str(APP / "data" / "terrain"))
threats, _ = build_threats(THREAT_LAYOUT); td = [t.to_dict() for t in threats if t.name != "SA-13 (번개-2)"]
start, end = [37.09, 127.03, 500.0], [39.05, 125.7, 631.0]
pts = [(38.4, 126.35), (39.0, 125.8), (37.5, 127.0)]
out = {"expected": "get_elevation exists and equals get; A* 3D near threats returns without exception"}
for label, mod in (("before_fix", before), ("after_fix", after)):
    cache = mod.TerrainCacheFast(tl)
    r = {"has_get_elevation": hasattr(cache, "get_elevation")}
    if r["has_get_elevation"]:
        r["equals_get"] = all(cache.get_elevation(a, b) == cache.get(a, b) for a, b in pts)
    try:
        p = mod.AStarPathfinder3DOptimized(cache).find_path_3d_fast(start, end, td, 1.2, fuel_state=1.0, refuel_count=0)
        r["astar"] = {"result": "returned", "points": len(p)}
    except Exception as e:
        r["astar"] = {"result": "exception", "type": type(e).__name__}
    r["passed"] = bool(r["has_get_elevation"] and r.get("equals_get") and r["astar"]["result"] == "returned")
    out[label] = r
(HERE / "regression_I01_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
