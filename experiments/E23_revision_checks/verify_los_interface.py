"""JKSS revision check: does the system's own terrain cache work with the LoS check inside A* 3D?
Calls find_path_3d_fast with the unpatched TerrainCacheFast (as the Streamlit app does) and with the
harness shim, for the same E22 SEAD leg. Read-only; no module changes."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; APP = HERE.parents[1] / "mission_plan-new_plan_250519"
sys.path.insert(0, str(APP)); sys.path.insert(0, str(HERE.parent / "E22_threat_terrain_e2e"))
from run_e2e_threat import TerrainCacheFastLoS, build_threats, THREAT_LAYOUT
from modules.terrain_loader import TerrainLoader
from modules.pathfinder_optimized import AStarPathfinder3DOptimized, TerrainCacheFast
tl = TerrainLoader(dem_dir=str(APP / "data" / "terrain"))
threats, _ = build_threats(THREAT_LAYOUT); td = [t.to_dict() for t in threats if t.name != "SA-13 (번개-2)"]
start, end = [37.09, 127.03, 500.0], [39.05, 125.7, 631.0]
out = {}
for name, cache in (("system_TerrainCacheFast", TerrainCacheFast(tl)), ("harness_TerrainCacheFastLoS", TerrainCacheFastLoS(tl))):
    pf = AStarPathfinder3DOptimized(cache)
    try:
        p = pf.find_path_3d_fast(start, end, td, 1.2, fuel_state=1.0, refuel_count=0)
        out[name] = {"result": "returned", "points": len(p)}
    except Exception as e:
        out[name] = {"result": "exception", "type": type(e).__name__, "msg": str(e)[:120]}
(HERE / "los_interface.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
