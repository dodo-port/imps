from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
APP = ROOT / "mission_plan-new_plan_250519"
sys.path.insert(0, str(APP))

from modules.config import AIRPORTS, MUMT_RATIO
from modules.czml_exporter import export_czml
from modules.doctrine_policy import DoctrinePolicyEngine
from modules.formation_optimizer import FormationOptimizer, formation_allows_downstream
from modules.llm_brain import LLMBrain
from modules.map_renderer import build_tactical_map
from modules.mission_state import MissionState
from modules.path_computer import clear_path_cache, compute_formation_paths
from modules.pathfinder_optimized import AStarPathfinder3DOptimized
from modules.validator import MissionValidator


COMMAND = (
    "MUM-T mission from Osan. target lat=37.30 lon=127.30. "
    "ISR SEAD STRIKE, 3D, RTB, safety margin 5 km, maximum 6 assets."
)
TARGET = {"name": "SYNTH-TGT-01", "lat": 37.30, "lon": 127.30}
MAX_TOTAL = 6
ENDPOINT_TOLERANCE_KM = 0.25


class FlatTerrain:
    @staticmethod
    def get(lat, lon):
        return 0.0

    @staticmethod
    def get_elevation(lat, lon):
        return 0.0


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def distance_km(a, b):
    mean_lat = math.radians((float(a[0]) + float(b[0])) * 0.5)
    dlat = (float(a[0]) - float(b[0])) * 110.57
    dlon = (float(a[1]) - float(b[1])) * 110.57 * math.cos(mean_lat)
    return math.hypot(dlat, dlon)


def record(case_id, stage, expected, observed, passed):
    return {
        "case_id": case_id,
        "stage": stage,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def protocol_sha256():
    data = (HERE / "verification_protocol.md").read_bytes()
    return hashlib.sha256(data).hexdigest()


def main():
    started = time.perf_counter()
    rows = []
    state_before = {
        "target_lat": 37.8,
        "target_lon": 127.8,
        "start": "Osan",
        "algorithm": "A*",
        "margin": 3.0,
        "stpt_gap": 5,
        "enable_3d": False,
        "rtb": False,
        "fuel_state": 1.0,
        "refuel_count": 0,
        "extra_targets": [],
    }

    parsed = LLMBrain().parse_tactical_command(
        COMMAND,
        state_before,
        current_threats=[],
    )
    update = parsed.get("update_params", {})
    sequence = parsed.get("mission_sequence", [])
    parser_checks = {
        "action": parsed.get("action") == "MISSION_PLAN",
        "deterministic_router": parsed.get("_model_used") == "fast-rule",
        "base": update.get("start") in AIRPORTS and "Osan" in update.get("start", ""),
        "target_lat": abs(float(update.get("target_lat", 0.0)) - TARGET["lat"]) < 1e-9,
        "target_lon": abs(float(update.get("target_lon", 0.0)) - TARGET["lon"]) < 1e-9,
        "mission_sequence": sequence == ["ISR", "SEAD", "STRIKE"],
        "algorithm": update.get("algorithm") == "A* 3D",
        "enable_3d": update.get("enable_3d") is True,
        "rtb": update.get("rtb") is True,
        "margin": abs(float(update.get("safety_margin_km", 0.0)) - 5.0) < 1e-9,
    }
    rows.append(record(
        "E01",
        "natural_language_router",
        "all frozen parser fields match",
        {"checks": parser_checks, "model_used": parsed.get("_model_used")},
        all(parser_checks.values()),
    ))

    base = update["start"]
    doctrine_engine = DoctrinePolicyEngine(
        doctrine_dir=str(APP / "data" / "doctrine"),
        fallback_doc=str(APP / "doctrine_basis.md"),
    )
    policy = doctrine_engine.build_policy(
        mission_types=sequence,
        threats=[],
        current_margin_km=float(update["safety_margin_km"]),
    )
    policy_dict = policy.to_optimizer_dict()
    formation = FormationOptimizer().run(
        mission_types=policy.mission_sequence,
        targets=[TARGET],
        threats=[],
        base=base,
        max_total=MAX_TOTAL,
        doctrine_policy=policy_dict,
    )
    n_uav = formation.n_recon_uav + formation.n_attack_uav
    formation_checks = {
        "doctrine_refs_nonempty": bool(policy.refs),
        "policy_sequence": policy.mission_sequence == sequence,
        "feasible": formation.is_feasible,
        "max_total": formation.total_assets() <= MAX_TOTAL,
        "mumt_ratio": n_uav >= policy.mumt_ratio * formation.n_fighter,
        "mission_domain": all(asset.assigned_mission in sequence for asset in formation.assets),
    }
    rows.append(record(
        "E02",
        "doctrine_formation_assignment",
        "nonempty doctrine references and feasible assignment within all fixed constraints",
        {"doctrine_policy": policy_dict, "formation": formation.to_dict(), "checks": formation_checks},
        all(formation_checks.values()),
    ))

    mission = MissionState()
    mission.params.start = base
    mission.params.target_lat = TARGET["lat"]
    mission.params.target_lon = TARGET["lon"]
    mission.params.target_name = TARGET["name"]
    mission.params.margin = float(update["safety_margin_km"])
    mission.params.algorithm = update["algorithm"]
    mission.params.enable_3d = bool(update["enable_3d"])
    mission.params.rtb = bool(update["rtb"])
    mission.threats = []
    mission.formation = formation

    terrain = FlatTerrain()
    pathfinder = AStarPathfinder3DOptimized(terrain)
    clear_path_cache()
    paths = compute_formation_paths(
        mission=mission,
        pathfinder=pathfinder,
        effective_algorithm="A* 3D",
        terrain_fast=terrain,
        threats_dict=[],
        fuel_state=1.0,
        refuel_count=0,
    )
    base_coord = AIRPORTS[base]["coords"]
    fighter_target_distances = []
    rtb_distances = {}
    all_ingress = True
    all_rtb = True
    for asset in formation.assets:
        path_data = paths.get(asset.asset_id, {})
        ingress = path_data.get("in", [])
        egress = path_data.get("out", [])
        all_ingress = all_ingress and len(ingress) >= 2
        if asset.asset_type == "fighter" and ingress:
            fighter_target_distances.append(min(distance_km(point, [TARGET["lat"], TARGET["lon"]]) for point in ingress))
        if not path_data.get("expendable", False):
            rtb_distances[asset.asset_id] = distance_km(egress[-1], base_coord) if egress else None
            all_rtb = all_rtb and len(egress) >= 2 and rtb_distances[asset.asset_id] <= ENDPOINT_TOLERANCE_KM
    route_checks = {
        "all_ingress_nonempty": all_ingress and len(paths) == formation.total_assets(),
        "fighter_target_reached": bool(fighter_target_distances) and max(fighter_target_distances) <= ENDPOINT_TOLERANCE_KM,
        "all_nonexpendable_rtb": all_rtb,
    }
    rows.append(record(
        "E03",
        "route_generation",
        "all ingress routes exist and required endpoints are reached within 0.25 km",
        {"checks": route_checks, "fighter_target_distance_km": fighter_target_distances,
         "rtb_distance_km": rtb_distances,
         "waypoint_counts": {aid: {"in": len(data.get("in", [])), "out": len(data.get("out", []))}
                             for aid, data in paths.items()}},
        all(route_checks.values()),
    ))

    report = MissionValidator().validate(
        formation_result=formation,
        formation_paths=paths,
        threats=[],
        mission_sequence=sequence,
        margin_km=mission.params.margin,
        terrain_loader=terrain,
        targets=[TARGET],
    )
    expected_rule_prefixes = ("MISSION_SEQUENCE", "TARGET_IN_NFZ", "MUMT_RATIO", "NFZ_", "THREAT_", "ALT_", "RANGE_", "ASSET_COLLISION")
    validator_checks = {
        "no_errors": report.error_count == 0,
        "expected_rule_families": all(any(rule.startswith(prefix) for rule in report.checked_rules)
                                      for prefix in expected_rule_prefixes),
    }
    rows.append(record(
        "E04",
        "rule_checking",
        "no ERROR and every expected rule family executed",
        {"checks": validator_checks, "report": report.to_dict()},
        all(validator_checks.values()),
    ))

    output_dir = HERE / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    representative = next((data for data in paths.values() if data.get("in")), {"in": [], "out": []})
    tactical_map = build_tactical_map(
        mission,
        representative.get("in", []),
        representative.get("out", []),
        paths,
        [],
        False,
        terrain,
        base_coord,
        [TARGET["lat"], TARGET["lon"]],
    )
    map_path = output_dir / "synthetic_tactical_map.html"
    tactical_map.save(str(map_path))
    czml = export_czml(paths, mission)
    czml_path = output_dir / "synthetic_mission.czml.json"
    czml_path.write_text(json.dumps(czml, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    czml_ids = {packet.get("id") for packet in czml}
    output_checks = {
        "html_nonempty": map_path.exists() and map_path.stat().st_size > 1000,
        "czml_nonempty": czml_path.exists() and czml_path.stat().st_size > 1000,
        "all_asset_ids_in_czml": all(asset.asset_id in czml_ids for asset in formation.assets),
    }
    rows.append(record(
        "E05",
        "output_generation",
        "nonempty Folium and CZML outputs containing every asset",
        {"checks": output_checks, "html_bytes": map_path.stat().st_size,
         "czml_bytes": czml_path.stat().st_size, "czml_packets": len(czml)},
        all(output_checks.values()),
    ))

    infeasible = FormationOptimizer().run(
        mission_types=["ISR"],
        targets=[TARGET],
        threats=[],
        base=base,
        max_total=1,
    )
    failure_executed_stages = ["formation"]
    downstream_allowed = formation_allows_downstream(infeasible)
    if downstream_allowed:
        failure_executed_stages.append("route")
    failure_checks = {
        "infeasible": infeasible.is_feasible is False,
        "no_assets": infeasible.total_assets() == 0 and not infeasible.assets,
        "downstream_not_entered": not downstream_allowed and failure_executed_stages == ["formation"],
    }
    rows.append(record(
        "E06",
        "infeasible_flow",
        "infeasible formation stops before route/output generation",
        {"checks": failure_checks, "executed_stages": failure_executed_stages,
         "formation": infeasible.to_dict()},
        all(failure_checks.values()),
    ))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha256(),
        "purpose": "synthetic end-to-end implementation verification; not operational validation",
        "front_end_scope": "deterministic natural-language command router; generative LLM excluded",
        "scenario": {"command": COMMAND, "target": TARGET, "max_total": MAX_TOTAL,
                     "terrain": "flat synthetic 0 m MSL", "threats": []},
        "runtime_s": round(time.perf_counter() - started, 3),
        "summary": {"total": len(rows), "passed": sum(row["passed"] for row in rows),
                    "failed": sum(not row["passed"] for row in rows)},
        "cases": rows,
    }
    (HERE / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    lines = ["E21 end-to-end verification", "", f"PASS {payload['summary']['passed']}/{payload['summary']['total']}",
             f"Runtime {payload['runtime_s']:.3f} s"]
    lines.extend(f"[{'PASS' if row['passed'] else 'FAIL'}] {row['case_id']} {row['stage']}" for row in rows)
    (HERE / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 1 if payload["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
