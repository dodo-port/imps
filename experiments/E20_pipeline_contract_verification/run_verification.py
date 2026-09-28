from __future__ import annotations

import json
import hashlib
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "mission_plan-new_plan_250519"
sys.path.insert(0, str(ROOT))

from modules.config import MISSION_ASSET_REQUIREMENTS, MUMT_RATIO
from modules.formation_optimizer import FormationMILP, FormationOptimizer
from modules.pathfinder import smooth_path_3d
from modules.path_constraints import path_satisfies, resample_path
from modules.validator import (
    check_asset_collision,
    check_min_altitude,
    check_mission_sequence,
    check_mumt_ratio,
    check_nfz_violation,
    check_targets_in_nfz,
    check_threat_penetration,
)


def record(case_id, category, expected, observed, passed, details=""):
    return {
        "case_id": case_id,
        "category": category,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "details": details,
    }


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def rule_ids(issues):
    return [item.rule_id for item in issues]


def verify_formation():
    rows = []
    milp = FormationMILP()
    missions = ["ISR", "SEAD", "STRIKE"]
    result = milp.optimize(missions, n_targets=2, max_total=12)
    nf, nr, nk = (result.get("n_fighter", 0), result.get("n_recon_uav", 0), result.get("n_attack_uav", 0))
    req = {kind: max(MISSION_ASSET_REQUIREMENTS.get(m, {}).get(kind, 0) for m in missions)
           for kind in ("fighter", "recon_uav", "attack_uav")}
    checks = {
        "optimal": result.get("status") == "Optimal",
        "max_total": nf + nr + nk <= 12,
        "min_fighter": nf >= req["fighter"],
        "min_recon_uav": nr >= max(req["recon_uav"], 2),
        "min_attack_uav": nk >= req["attack_uav"],
        "target_coverage": nf + nk >= 2,
        "mumt_ratio": nr + nk >= MUMT_RATIO * nf,
    }
    rows.append(record("F01", "formation", "all MILP constraints satisfied", {"result": result, "checks": checks}, all(checks.values())))

    impossible = milp.optimize(["ISR"], n_targets=1, max_total=1)
    impossible_checks = {
        "not_optimal": impossible.get("status") != "Optimal",
        "infeasible_flag": impossible.get("is_feasible") is False,
        "required_assets_exceed_limit": (
            impossible.get("n_fighter", 0)
            + impossible.get("n_recon_uav", 0)
            + impossible.get("n_attack_uav", 0)
        ) > 1,
    }
    rows.append(record("F02", "formation", "infeasible condition is rejected",
                       {"result": impossible, "checks": impossible_checks},
                       all(impossible_checks.values()),
                       "ISR requires at least two reconnaissance UAVs while max_total is one."))

    formation = FormationOptimizer().run(
        mission_types=missions,
        targets=[{"name": "T1", "lat": 38.8, "lon": 126.0}, {"name": "T2", "lat": 39.0, "lon": 126.2}],
        threats=[], base="Osan", max_total=12,
    )
    assignment_checks = {
        "asset_count": len(formation.assets) == formation.total_assets(),
        "mission_domain": all(a.assigned_mission in missions for a in formation.assets),
        "target_index_domain": all(a.assigned_target_idx is None or 0 <= a.assigned_target_idx < 2 for a in formation.assets),
    }
    rows.append(record("F03", "assignment", "valid asset objects and assignment fields",
                       {"formation": formation.to_dict(), "checks": assignment_checks}, all(assignment_checks.values())))
    return rows


def verify_rules():
    rows = []
    good = rule_ids(check_mission_sequence(["ISR", "SEAD", "STRIKE"]))
    bad = rule_ids(check_mission_sequence(["STRIKE", "ISR"]))
    rows.append(record("V01", "validator", "normal:no MISSION_SEQUENCE; violation:MISSION_SEQUENCE",
                       {"normal": good, "violation": bad}, "MISSION_SEQUENCE" not in good and "MISSION_SEQUENCE" in bad))

    nfz = [{"type": "NFZ", "name": "N1", "lat_min": 36.9, "lat_max": 37.1, "lon_min": 126.9, "lon_max": 127.1}]
    outside = rule_ids(check_nfz_violation("A", [[36.5, 126.5], [36.6, 126.6]], nfz))
    inside = rule_ids(check_nfz_violation("A", [[36.5, 126.5], [37.0, 127.0]], nfz))
    rows.append(record("V02", "validator", "outside:none; inside:NFZ_VIOLATION",
                       {"outside": outside, "inside": inside}, not outside and "NFZ_VIOLATION" in inside))

    target_out = rule_ids(check_targets_in_nfz([{"name": "T0", "lat": 36.5, "lon": 126.5}], nfz))
    target_in = rule_ids(check_targets_in_nfz([{"name": "T1", "lat": 37.0, "lon": 127.0}], nfz))
    rows.append(record("V03", "validator", "outside:none; inside:TARGET_IN_NFZ",
                       {"outside": target_out, "inside": target_in}, not target_out and "TARGET_IN_NFZ" in target_in))

    high = rule_ids(check_min_altitude("A", [[37.0, 127.0, 500.0]]))
    low = rule_ids(check_min_altitude("A", [[37.0, 127.0, 100.0]]))
    rows.append(record("V04", "validator", "500m:none; 100m:MIN_ALTITUDE",
                       {"500m": high, "100m": low}, not high and "MIN_ALTITUDE" in low,
                       "Flat-reference test without terrain input; MSL and AGL are equal in this condition."))

    path_a = [[37.0, 127.0, 500.0], [37.1, 127.0, 500.0], [37.2, 127.0, 500.0]]
    path_far = [[37.0, 127.1, 500.0], [37.1, 127.1, 500.0], [37.2, 127.1, 500.0]]
    separated = rule_ids(check_asset_collision({"A": path_a, "B": path_far}))
    colliding = rule_ids(check_asset_collision({"A": path_a, "B": path_a}))
    rows.append(record("V05", "validator", "separated:none; identical:ASSET_COLLISION",
                       {"separated": separated, "identical": colliding}, not separated and "ASSET_COLLISION" in colliding))

    ratio_ok = rule_ids(check_mumt_ratio(1, 2))
    ratio_bad = rule_ids(check_mumt_ratio(2, 1))
    rows.append(record("V06", "validator", "1:2 none; 2:1 MUMT_RATIO",
                       {"1fighter_2uav": ratio_ok, "2fighter_1uav": ratio_bad}, not ratio_ok and "MUMT_RATIO" in ratio_bad))

    terrain_elevation_m = 450.0
    path_alt_msl = 500.0
    min_agl_m = 300.0
    class FixedTerrain:
        @staticmethod
        def get_elevation(lat, lon):
            return terrain_elevation_m

    terrain_relative = rule_ids(check_min_altitude(
        "A", [[37.0, 127.0, path_alt_msl]], min_agl=min_agl_m, terrain_loader=FixedTerrain()
    ))
    rows.append(record("V07", "validator", "MIN_ALTITUDE for 50m AGL under a 300m criterion",
                       {"terrain_elevation_m": terrain_elevation_m, "path_alt_msl": path_alt_msl,
                        "actual_agl_m": path_alt_msl - terrain_elevation_m, "rules": terrain_relative},
                       "MIN_ALTITUDE" in terrain_relative,
                       "Terrain elevation is subtracted from MSL altitude before applying the AGL criterion."))

    segment_crossing = rule_ids(check_nfz_violation(
        "A",
        [[36.8, 127.0, 500.0], [37.2, 127.0, 500.0]],
        nfz,
    ))
    rows.append(record(
        "V08",
        "validator_regression",
        "NFZ_VIOLATION when endpoints are outside but the segment crosses the NFZ",
        {"endpoint_rules": [rule_ids(check_nfz_violation("A", [point], nfz)) for point in
                            [[36.8, 127.0, 500.0], [37.2, 127.0, 500.0]]],
         "segment_rules": segment_crossing},
        "NFZ_VIOLATION" in segment_crossing,
        "Added after technical review to prevent waypoint-only NFZ checks.",
    ))

    class RidgeTerrain:
        @staticmethod
        def get_elevation(lat, lon):
            return 450.0 if 36.99 <= lat <= 37.01 else 0.0

    ridge_path = [[36.9, 127.0, 500.0], [37.1, 127.0, 500.0]]
    ridge_rules = rule_ids(check_min_altitude(
        "A", ridge_path, min_agl=min_agl_m, terrain_loader=RidgeTerrain()
    ))
    rows.append(record(
        "V09",
        "validator_regression",
        "MIN_ALTITUDE when endpoints clear the criterion but the segment crosses a ridge",
        {"endpoint_agl_m": [500.0, 500.0], "ridge_elevation_m": 450.0,
         "path_alt_msl": 500.0, "segment_rules": ridge_rules},
        "MIN_ALTITUDE" in ridge_rules,
        "Added after technical review; the segment is resampled at the 90 m DEM interval.",
    ))

    sam = [{"type": "SAM", "name": "S1", "lat": 37.0, "lon": 127.0,
            "radius_km": 5.0, "sskp": 0.75, "pk_peak_km": 0.35,
            "pk_sigma_km": 0.20}]
    threat_segment = [[36.9, 127.0, 500.0], [37.1, 127.0, 500.0]]
    endpoint_threat_rules = [rule_ids(check_threat_penetration(
        "A", "fighter", [point], sam, margin_km=0.0
    )) for point in threat_segment]
    segment_threat_rules = rule_ids(check_threat_penetration(
        "A", "fighter", threat_segment, sam, margin_km=0.0
    ))
    rows.append(record(
        "V10",
        "validator_regression",
        "THREAT_PENETRATION when endpoints are clear but the segment crosses a high-risk area",
        {"endpoint_rules": endpoint_threat_rules, "segment_rules": segment_threat_rules},
        not any(endpoint_threat_rules) and "THREAT_PENETRATION" in segment_threat_rules,
        "Added after technical review to prevent waypoint-only threat checks.",
    ))
    return rows


def verify_smoothing():
    raw = [(0.0, 0.0, 500.0), (0.0, 1.0, 500.0), (1.0, 1.0, 500.0)]
    center = (0.25, 0.75)
    radius = 0.20
    valid = lambda p: math.hypot(p[0] - center[0], p[1] - center[1]) >= radius
    path_valid = lambda candidate: path_satisfies(candidate, valid)
    smoothed = smooth_path_3d(
        raw,
        iterations=2,
        point_is_valid=valid,
        path_is_valid=path_valid,
    )
    sampled = resample_path(smoothed)
    intrusions = [p for p in sampled if math.hypot(p[0] - center[0], p[1] - center[1]) < radius]
    return [record("P01", "path_postprocess", "no resampled segment point inside test obstacle",
                   {"waypoint_count": len(smoothed), "sample_count": len(sampled),
                    "intrusion_count": len(intrusions), "intrusions": intrusions[:5]},
                   len(intrusions) == 0,
                   "Synthetic regression test for segment-level corner cutting after Chaikin smoothing.")]


def main():
    rows = verify_formation() + verify_rules() + verify_smoothing()
    protocol_path = HERE / "verification_protocol.md"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "purpose": "module contract verification; not operational validation",
        "test_history": [
            {"stage": "initial_suite_first_run", "total": 10, "passed": 8,
             "failed_cases": ["V02", "P01"]},
            {"stage": "expanded_suite_first_full_run", "total": 11, "passed": 9,
             "failed_cases": ["V07", "P01"]},
            {"stage": "post_fix_regression", "total": 11, "passed": 11,
             "failed_cases": []},
            {"stage": "technical_review_regression", "total": len(rows),
             "passed": sum(r["passed"] for r in rows),
             "failed_cases": [r["case_id"] for r in rows if not r["passed"]]},
        ],
        "case_origin": {
            "initial_suite": ["F01", "F02", "F03", "V01", "V02", "V03", "V04", "V05", "V06", "P01"],
            "added_after_initial_run": ["V07"],
            "added_after_technical_review": ["V08", "V09", "V10"],
            "strengthened_after_defect_review": ["F02", "P01"],
        },
        "summary": {
            "total": len(rows),
            "passed": sum(r["passed"] for r in rows),
            "failed": sum(not r["passed"] for r in rows),
        },
        "cases": rows,
    }
    (HERE / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    lines = ["E20 pipeline contract verification", "", f"PASS {payload['summary']['passed']}/{payload['summary']['total']}"]
    for row in rows:
        lines.append(f"[{'PASS' if row['passed'] else 'FAIL'}] {row['case_id']} {row['category']}: {row['details']}")
    (HERE / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 1 if payload["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
