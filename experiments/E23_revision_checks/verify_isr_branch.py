"""JKSS revision check (R3632-M3): when does the 'remaining targets -> ISR' branch occur?
Runs FormationOptimizer unchanged on (a) the E22 mission, (b) STRIKE with more targets than max assets,
(c) a CAS mission (no target-coverage constraint) with more targets than strike-capable assets."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "mission_plan-new_plan_250519"))
from modules.formation_optimizer import FormationOptimizer
T = [{"name": f"T{i}", "lat": 39.0 + 0.1 * i, "lon": 125.8} for i in range(3)]
cases = {
  "a_E22_SEAD_STRIKE_1target_max6": dict(mission_types=["SEAD", "STRIKE"], targets=T[:1], max_total=6),
  "b_STRIKE_3targets_max2": dict(mission_types=["STRIKE"], targets=T, max_total=2),
  "c_ISR_CAS_3targets_max4": dict(mission_types=["ISR", "CAS"], targets=T, max_total=4),
}
out = {}
for k, kw in cases.items():
    f = FormationOptimizer().run(threats=[], base="오산(Osan)", **kw)
    d = f.to_dict()
    out[k] = {"feasible": d.get("is_feasible"), "status": d.get("solver_status"),
              "counts": [d.get("n_fighter"), d.get("n_recon_uav"), d.get("n_attack_uav")],
              "assets": [(a["asset_id"], a["assigned_mission"], a["assigned_target_idx"]) for a in d.get("assets", [])]}
(HERE / "isr_branch.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1, default=int))
