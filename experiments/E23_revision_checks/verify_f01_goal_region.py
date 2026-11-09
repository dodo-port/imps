"""JKSS revision check (R3632 2nd-round mock): is F-01's node-limit stop a search-budget issue or
is the goal region closed? Enumerate every grid node that satisfies the A* arrival rule
(index distance to goal < 3) and test it with the pathfinder's own node checks (terrain AGL,
threat core, risk >= block threshold) for F-01's threat set. Read-only; no module changes."""
import json, sys, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
APP = HERE.parents[1] / "mission_plan-new_plan_250519"
sys.path.insert(0, str(APP)); sys.path.insert(0, str(HERE.parent / "E22_threat_terrain_e2e"))
from run_e2e_threat import TerrainCacheFastLoS, build_threats, THREAT_LAYOUT
from modules.terrain_loader import TerrainLoader
from modules.pathfinder_optimized import AStarPathfinder3DOptimized
from modules.xai_utils import XAIUtils
from modules.config import MIN_ALTITUDE_AGL
tl = TerrainLoader(dem_dir=str(APP / "data" / "terrain")); tc = TerrainCacheFastLoS(tl); pf = AStarPathfinder3DOptimized(tc)
threats, _ = build_threats(THREAT_LAYOUT)
td = {t.name: t.to_dict() for t in threats}
F01 = [td[n] for n in ("S-75 (번개-1)", "SNR-75 (Fan Song) 화력제어 레이더", "SNR-125 (LOW BLOW) 화력통제 레이더")]
end = [39.03, 125.75, 1541.0]; eg = pf.to_grid_3d(*end); thr = pf.risk_block_threshold(8.0, 1.0, 0)
cnt = {"total": 0, "terrain": 0, "core": 0, "risk": 0, "open": 0}; open_nodes = []
for dx in range(-3, 4):
    for dy in range(-3, 4):
        for dz in range(-3, 4):
            n = (eg[0]+dx, eg[1]+dy, eg[2]+dz)
            if not (0 <= n[0] < pf.grid_size and 0 <= n[1] < pf.grid_size and 0 <= n[2] < pf.altitude_levels): continue
            if pf.distance(n, eg) >= 3: continue
            cnt["total"] += 1
            lat, lon, alt = pf.to_latlonalt(*n)
            if alt < tc.get(lat, lon) + MIN_ALTITUDE_AGL: cnt["terrain"] += 1; continue
            if pf.is_in_threat_core(lat, lon, alt, F01, 8.0): cnt["core"] += 1; continue
            r = XAIUtils.calculate_risk_score(lat, lon, F01, 8.0, terrain_loader=tc, target_alt=alt)
            if r >= thr:
                cnt["risk"] += 1; cnt["risk_046_050"] = cnt.get("risk_046_050", 0) + (1 if r < 0.5 else 0); continue
            cnt["open"] += 1; open_nodes.append([round(lat,3), round(lon,3), alt, round(r,3)])
out = {"goal_grid": eg, "block_threshold": thr, "counts": cnt, "open_nodes_sample": open_nodes[:10]}
(HERE / "f01_goal_region.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
