"""
E22 — threat + terrain end-to-end regression.

Extends E21's 6-stage validator harness (experiments/E21_end_to_end_verification/run_e2e.py)
with the real threat/terrain pipeline setup from V12 (experiments/V12_extension_demo/exp_C_extension_demo.py):
  (a) FlatTerrain -> real TerrainLoader (SRTM tiles) + TerrainCacheFast
  (b) threats=[] -> 5-threat build_threats(threat_layout) (identical to V12 CONFIG_A)
  (c) command/parser fields -> Osan -> Pyongyang (39.03, 125.75), margin 8km, SEAD/STRIKE

Scenario is frozen in prereg.md (written before this script was ever run). Do not
edit the scenario constants below to make a failing stage pass; see prereg.md
"Debugging discipline" for what may be changed and why.
"""
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
from modules.mission_state import MissionState, Threat, THREAT_DB
from modules.path_computer import clear_path_cache, compute_formation_paths
from modules.pathfinder_optimized import AStarPathfinder3DOptimized, TerrainCacheFast
from modules.terrain_loader import TerrainLoader
from modules.validator import MissionValidator
from modules.xai_utils import XAIUtils


class TerrainCacheFastLoS(TerrainCacheFast):
    """
    Driver-level shim, not a core-code change (0 lines touched in modules/).

    modules/radar_shadow.py:check_line_of_sight() calls terrain_loader.get_elevation(),
    but TerrainCacheFast (modules/pathfinder_optimized.py) only implements .get().
    Every A* node expansion that runs a risk/LOS check near an active threat raises
    AttributeError, which path_computer.py silently swallows as an empty path
    ("except Exception: path_in = []"), masquerading as "no route found."

    This is a pre-existing core bug, already discovered and documented independently
    in experiments/V12_extension_demo/exp_C_extension_demo.py (same class name, same
    workaround, same "core bug, reported separately" note). Confirmed here by direct
    reproduction (see debugging log in the returned report) before being applied —
    the frozen scenario in prereg.md was not changed to work around this.
    """

    def get_elevation(self, lat: float, lon: float) -> float:
        return self.get(lat, lon)


# ================================================================
# FROZEN SCENARIO (see prereg.md) — do not edit to force a PASS
# ================================================================
COMMAND = (
    "MUM-T mission from Osan. target lat=39.03 lon=125.75. "
    "SEAD STRIKE, 3D, RTB, safety margin 8 km, maximum 6 assets."
)
TARGET = {"name": "Pyongyang", "lat": 39.03, "lon": 125.75}
MAX_TOTAL = 6
ENDPOINT_TOLERANCE_KM = 0.25
TAU = 0.5  # corridor-open risk threshold, diagnostic only (not a pass/fail gate)

THREAT_LAYOUT = [
    {"db_name": "S-75 (번개-1)", "lat": 38.40, "lon": 126.35},
    {"db_name": "SNR-75 (Fan Song) 화력제어 레이더", "lat": 38.40, "lon": 126.35},
    {"db_name": "S-125 (번개-3)", "lat": 39.00, "lon": 125.80},
    {"db_name": "SNR-125 (LOW BLOW) 화력통제 레이더", "lat": 39.00, "lon": 125.80},
    {"db_name": "SA-13 (번개-2)", "lat": 39.05, "lon": 125.70},
]


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
    data = (HERE / "prereg.md").read_bytes()
    return hashlib.sha256(data).hexdigest()


def build_threats(threat_layout):
    """Declarative threat_layout -> Threat objects, quoting spec from THREAT_DB."""
    threats, missing = [], []
    for item in threat_layout:
        name = item["db_name"]
        if name not in THREAT_DB:
            missing.append(name)
            continue
        spec = THREAT_DB[name]
        threats.append(Threat(
            name=name, type=spec["type"], lat=float(item["lat"]), lon=float(item["lon"]),
            radius_km=float(spec["radius_km"]), alt=0.0,
            loss=float(spec.get("loss", 21.1)) or 21.1,
            rcs_m2=float(spec.get("rcs_m2", 2.5)) or 2.5,
            pd_k=float(spec.get("pd_k", 0.40)) or 0.40,
            sskp=float(spec.get("sskp", 0.75)),
            pk_peak_km=float(spec.get("peak_ratio", 0.35)),
            pk_sigma_km=float(spec.get("sigma_ratio", 0.20)),
        ))
    return threats, missing


def corridor_samples(start_coord, target_coord, n=120):
    (slat, slon), (tlat, tlon) = start_coord[:2], target_coord[:2]
    return [[slat + (tlat - slat) * i / (n - 1), slon + (tlon - slon) * i / (n - 1)] for i in range(n)]


def corridor_max_risk(samples, active_threats, margin, terrain_loader, alt):
    max_r, arg = 0.0, -1
    tdicts = [t.to_dict() for t in active_threats]
    for i, (lat, lon) in enumerate(samples):
        r = XAIUtils.calculate_risk_score(lat, lon, tdicts, margin, terrain_loader=terrain_loader, target_alt=alt)
        if r > max_r:
            max_r, arg = r, i
    return max_r, arg


def dominant_threat_at(lat, lon, active_threats, margin, terrain_loader, alt):
    best_name, best_r = None, -1.0
    for t in active_threats:
        r = XAIUtils.calculate_risk_score(lat, lon, [t.to_dict()], margin, terrain_loader=terrain_loader, target_alt=alt)
        if r > best_r:
            best_r, best_name = r, t.name
    return best_name, best_r


def sead_open_corridor(samples, threats, margin, terrain_loader, tau, alt):
    active = list(threats)
    max_r0, arg0 = corridor_max_risk(samples, active, margin, terrain_loader, alt)
    trace, suppressed = [], []
    cur_r, cur_arg, k = max_r0, arg0, 0
    while cur_r > tau and active and k < len(threats):
        lat, lon = samples[cur_arg]
        name, _ = dominant_threat_at(lat, lon, active, margin, terrain_loader, alt)
        if name is None:
            break
        active = [t for t in active if t.name != name]
        suppressed.append(name)
        k += 1
        new_r, new_arg = corridor_max_risk(samples, active, margin, terrain_loader, alt)
        trace.append({"step": k, "suppressed": name, "risk_before": round(cur_r, 4), "risk_after": round(new_r, 4)})
        cur_r, cur_arg = new_r, new_arg
    return {
        "initial_max_risk": round(max_r0, 4), "tau": tau,
        "initially_denied": bool(max_r0 > tau),
        "suppression_trace": trace, "suppressed_threats": suppressed,
        "open_reached": bool(cur_r <= tau),
        "open_at_k": k if cur_r <= tau else None,
        "final_max_risk": round(cur_r, 4),
    }


def main():
    started = time.perf_counter()
    rows = []

    # ── real terrain setup ──
    terrain_raw = TerrainLoader(dem_dir=str(APP / "data" / "terrain"))
    terrain_setup = {
        "use_fallback": bool(terrain_raw.use_fallback),
        "n_datasets": len(terrain_raw.datasets),
    }
    terrain_cache = TerrainCacheFastLoS(terrain_raw)

    # ── real threats ──
    threats, missing_threats = build_threats(THREAT_LAYOUT)
    threat_dicts = [t.to_dict() for t in threats]

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
        current_threats=[t.to_dict() for t in threats],
    )
    update = parsed.get("update_params", {})
    sequence = parsed.get("mission_sequence", [])
    parser_checks = {
        "action": parsed.get("action") == "MISSION_PLAN",
        "deterministic_router": parsed.get("_model_used") == "fast-rule",
        "base": update.get("start") in AIRPORTS and "Osan" in update.get("start", ""),
        "target_lat": abs(float(update.get("target_lat", 0.0)) - TARGET["lat"]) < 1e-9,
        "target_lon": abs(float(update.get("target_lon", 0.0)) - TARGET["lon"]) < 1e-9,
        "mission_sequence": sequence == ["SEAD", "STRIKE"],
        "algorithm": update.get("algorithm") == "A* 3D",
        "enable_3d": update.get("enable_3d") is True,
        "rtb": update.get("rtb") is True,
        "margin": abs(float(update.get("safety_margin_km", 0.0)) - 8.0) < 1e-9,
    }
    rows.append(record(
        "E01",
        "natural_language_router",
        "all frozen parser fields match",
        {"checks": parser_checks, "model_used": parsed.get("_model_used")},
        all(parser_checks.values()),
    ))

    base = update.get("start") or "오산(Osan)"
    doctrine_engine = DoctrinePolicyEngine(
        doctrine_dir=str(APP / "data" / "doctrine"),
        fallback_doc=str(APP / "doctrine_basis.md"),
    )
    policy = doctrine_engine.build_policy(
        mission_types=sequence,
        threats=[{"type": t.type} for t in threats],
        current_margin_km=float(update.get("safety_margin_km", 8.0)),
    )
    policy_dict = policy.to_optimizer_dict()
    formation = FormationOptimizer().run(
        mission_types=policy.mission_sequence,
        targets=[TARGET],
        threats=threat_dicts,
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
        {"doctrine_policy": policy_dict, "formation": formation.to_dict(), "checks": formation_checks,
         "n_threats": len(threats), "missing_threats": missing_threats},
        all(formation_checks.values()),
    ))

    mission = MissionState()
    mission.params.start = base
    mission.params.target_lat = TARGET["lat"]
    mission.params.target_lon = TARGET["lon"]
    mission.params.target_name = TARGET["name"]
    mission.params.margin = float(update.get("safety_margin_km", 8.0))
    mission.params.algorithm = update.get("algorithm", "A* 3D")
    mission.params.enable_3d = bool(update.get("enable_3d", True))
    mission.params.rtb = bool(update.get("rtb", True))
    mission.threats = threats
    mission.formation = formation

    pathfinder = AStarPathfinder3DOptimized(terrain_cache)
    clear_path_cache()
    paths = compute_formation_paths(
        mission=mission,
        pathfinder=pathfinder,
        effective_algorithm="A* 3D",
        terrain_fast=terrain_cache,
        threats_dict=threat_dicts,
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
        "route_generation_real_terrain",
        "all ingress routes exist and required endpoints are reached within 0.25 km (real SRTM terrain)",
        {"checks": route_checks, "fighter_target_distance_km": fighter_target_distances,
         "rtb_distance_km": rtb_distances,
         "waypoint_counts": {aid: {"in": len(data.get("in", [])), "out": len(data.get("out", []))}
                             for aid, data in paths.items()},
         "terrain_setup": terrain_setup},
        all(route_checks.values()),
    ))

    report = MissionValidator().validate(
        formation_result=formation,
        formation_paths=paths,
        threats=threat_dicts,
        mission_sequence=sequence,
        margin_km=mission.params.margin,
        terrain_loader=terrain_raw,
        targets=[TARGET],
    )
    expected_rule_prefixes = ("MISSION_SEQUENCE", "TARGET_IN_NFZ", "MUMT_RATIO", "NFZ_", "THREAT_", "ALT_", "RANGE_", "ASSET_COLLISION")
    validator_checks = {
        "report_produced": report is not None,
        "expected_rule_families": all(any(rule.startswith(prefix) for rule in report.checked_rules)
                                      for prefix in expected_rule_prefixes),
    }
    rows.append(record(
        "E04",
        "rule_checking_real_threats",
        "rule engine executes all expected rule families and returns a verdict "
        "(pass/fail of the verdict itself, e.g. THREAT_PENETRATION ERRORs, is reported honestly below, "
        "not treated as harness failure)",
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
        threat_dicts,
        False,
        terrain_cache,
        base_coord,
        [TARGET["lat"], TARGET["lon"]],
    )
    map_path = output_dir / "threat_terrain_tactical_map.html"
    tactical_map.save(str(map_path))
    czml = export_czml(paths, mission)
    czml_path = output_dir / "threat_terrain_mission.czml.json"
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
        threats=threat_dicts,
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

    # ── supplementary cross-module consistency check (diagnostic, not one of the 6 scored stages) ──
    # POST-REGISTRATION NOTE: the cross_module_check below was added AFTER observing this
    # scenario's E03 route failure. The frozen scenario and the 6 scored stages (E01-E06)
    # are unchanged; this block only enriches the pre-existing supplementary diagnostic.
    # The naive count check (kept for continuity) passes, so this documents an inconsistency
    # that a count-only chain check does not catch. Deterministic: re-runs reproduce it.
    sead_assigned = sum(1 for a in formation.assets if (a.assigned_mission or "").upper() == "SEAD")
    samples = corridor_samples(base_coord, [TARGET["lat"], TARGET["lon"]], n=120)
    corridor = sead_open_corridor(samples, threats, policy.safety_margin_floor_km, terrain_raw, tau=TAU, alt=750.0)

    # (1) naive count check — the coarse chain check (matches prior behaviour): counts only
    naive_count_consistent = None
    if corridor["open_at_k"] is not None:
        naive_count_consistent = sead_assigned >= corridor["open_at_k"]

    # (2) cross-module check — compares module VERDICTS, not just counts
    routing_viable = bool(route_checks["all_ingress_nonempty"])
    corridor_claims_open = bool(corridor["open_reached"])
    assets_dict = formation.to_dict().get("assets", [])
    sead_targets = [a.get("target_name") for a in assets_dict
                    if (a.get("assigned_mission") or "").upper() == "SEAD"]
    sead_targets_bound = any(bool(t) for t in sead_targets)
    colocated = {}
    for t in threats:
        colocated.setdefault((round(t.lat, 4), round(t.lon, 4)), []).append(t.name)
    colocated_pairs = [names for names in colocated.values() if len(names) > 1]
    # The inconsistency: the corridor diagnostic reports the mission opens after k
    # suppressions, but the routing module produced no viable path (and the SEAD
    # suppressions the corridor requires are not bound to any asset target). Two modules
    # give contradictory verdicts on the same situation -> cross-module inconsistency.
    inconsistency_detected = corridor_claims_open and not routing_viable
    inconsistency_reason = (
        "corridor diagnostic reports open_reached after suppressing "
        f"{corridor['suppressed_threats']} (co-located emitter pair), but the routing "
        "module produced no viable path and the required SEAD suppressions are unbound "
        f"(SEAD asset targets={sead_targets})"
    ) if inconsistency_detected else ""

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prereg_sha256": protocol_sha256(),
        "purpose": "threat + real-terrain end-to-end implementation verification; not operational validation",
        "front_end_scope": "deterministic natural-language command router; generative LLM excluded",
        "scenario": {"command": COMMAND, "target": TARGET, "max_total": MAX_TOTAL,
                     "terrain": terrain_setup, "threats": [t.to_dict() for t in threats],
                     "missing_threats": missing_threats},
        "runtime_s": round(time.perf_counter() - started, 3),
        "summary": {"total": len(rows), "passed": sum(row["passed"] for row in rows),
                    "failed": sum(not row["passed"] for row in rows)},
        "cases": rows,
        "cross_module_consistency": {
            "post_registration_note": "cross_module_check added after observing E03 failure; "
                                      "6 scored stages unchanged; naive count check kept for contrast",
            "naive_count_check": {
                "sead_assigned": sead_assigned,
                "corridor_open_at_k": corridor["open_at_k"],
                "consistent": naive_count_consistent,
            },
            "cross_module_check": {
                "corridor_claims_open": corridor_claims_open,
                "routing_viable": routing_viable,
                "sead_targets_bound": sead_targets_bound,
                "colocated_emitter_pairs": colocated_pairs,
                "inconsistency_detected": inconsistency_detected,
                "reason": inconsistency_reason,
            },
            "corridor": corridor,
        },
    }
    (HERE / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    lines = ["E22 threat + terrain end-to-end regression", "",
             f"PASS {payload['summary']['passed']}/{payload['summary']['total']}",
             f"Runtime {payload['runtime_s']:.3f} s",
             f"terrain use_fallback={terrain_setup['use_fallback']} n_datasets={terrain_setup['n_datasets']}"]
    lines.extend(f"[{'PASS' if row['passed'] else 'FAIL'}] {row['case_id']} {row['stage']}" for row in rows)
    lines.append(f"naive count check: SEAD assigned={sead_assigned} vs corridor open_at_k="
                 f"{corridor['open_at_k']} -> consistent={naive_count_consistent}")
    lines.append(f"cross-module check: corridor_open={corridor_claims_open} routing_viable={routing_viable} "
                 f"sead_targets_bound={sead_targets_bound} -> INCONSISTENCY_DETECTED={inconsistency_detected}")
    (HERE / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    terrain_raw.close()
    return 1 if payload["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
