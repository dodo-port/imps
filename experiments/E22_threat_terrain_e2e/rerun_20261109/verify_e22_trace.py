"""
E22 re-run with module-boundary tracing (JKSS revision 1, 2026-11-09).

Purpose: answer reviewers R3606-1/2, R3632-2, R3631-1 by re-running the FROZEN
E22 scenario (prereg.md, unchanged) and recording, per asset and per route call,
the inputs, outputs and termination reason that the original harness did not keep.

Isolation rules (no change to the system under test):
  * 0 lines changed in modules/ and in run_e2e_threat.py.
  * The original harness main() is executed as-is. Tracing is done only by
    wrapping module-level functions of modules.path_computer at runtime; every
    wrapper calls the original function with the original arguments and returns
    its result unchanged (exceptions are recorded and re-raised).
  * Extra read-only probes (threat-core membership and risk at the route end
    point, block threshold) call the pathfinder's own public methods with the
    same arguments; they hold no state that feeds back into the search.
  * Outputs go to this run directory; the original August result.json,
    summary.txt and outputs/ are not touched.

Usage:
  python verify_e22_trace.py --out run1
  python verify_e22_trace.py --out public_run --app <repo>/mission_plan-new_plan_250519 \
         --terrain-dir "<codex>/mission_plan-new_plan_250519/data/terrain"
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import types
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
E22_DIR = HERE.parent
CODEX_ROOT = E22_DIR.parents[1]
DEFAULT_APP = CODEX_ROOT / "mission_plan-new_plan_250519"


# corridor diagnostic's suppressed set in the August run and in run1/run2 (result.json)
CORRIDOR_SUPPRESSED = {"SNR-125 (LOW BLOW) 화력통제 레이더", "S-125 (번개-3)"}


def jdefault(v):
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (set, tuple)):
        return list(v)
    return str(v)


def r4(x):
    try:
        return round(float(x), 4)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--app", default=str(DEFAULT_APP))
    ap.add_argument("--terrain-dir", default=None)
    ap.add_argument("--no-shim", action="store_true",
                    help="use the system TerrainCacheFast instead of the harness TerrainCacheFastLoS shim (after the 2026-11-09 interface fix)")
    ap.add_argument("--save-paths", action="store_true",
                    help="also store raw A* and smoothed path coordinates in trace.json (for figures)")
    ap.add_argument("--cf", default=None, choices=[None, "align_strike", "pair_sead", "corridor_only_strike"],
                    help="POST-HOC counterfactual diagnostic (not the frozen scenario): "
                         "align_strike = STRIKE asset also excludes the corridor-suppressed threats; "
                         "pair_sead = SEAD asset also excludes threats co-located with its assigned threat")
    args = ap.parse_args()

    out_dir = HERE / args.out
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run dir: {out_dir}")
    out_dir.mkdir(parents=True)
    shutil.copy2(E22_DIR / "prereg.md", out_dir / "prereg.md")  # same bytes -> same prereg_sha256

    app = Path(args.app).resolve()

    # ---- load the original harness as a module, pointed at the chosen app tree ----
    # The harness inserts its own APP into sys.path and imports modules/ at import
    # time. For the public-tree run, the only textual override is the APP constant
    # (one line, asserted to match exactly once); the rest of the harness is byte-identical.
    harness_file = E22_DIR / "run_e2e_threat.py"
    src = harness_file.read_text(encoding="utf-8")
    app_line = 'APP = ROOT / "mission_plan-new_plan_250519"'
    if app != DEFAULT_APP.resolve():
        assert src.count(app_line) == 1, "APP line not found exactly once"
        src = src.replace(app_line, f'APP = Path(r"{app}")')
    harness = types.ModuleType("e22_harness")
    harness.__file__ = str(harness_file)
    sys.modules["e22_harness"] = harness
    exec(compile(src, str(harness_file), "exec"), harness.__dict__)
    import modules.path_computer as pc  # resolved from the same tree the harness imported
    module_file = Path(pc.__file__).resolve()
    if app not in module_file.parents:
        raise SystemExit(f"module tree mismatch: path_computer loaded from {module_file}, expected under {app}")

    harness.HERE = out_dir  # result.json / summary.txt / outputs/ go to the run dir
    if args.no_shim:
        from modules.pathfinder_optimized import TerrainCacheFast as _SystemCache
        assert hasattr(_SystemCache, "get_elevation"), "system cache lacks get_elevation (fix not applied)"
        harness.TerrainCacheFastLoS = _SystemCache
    if args.terrain_dir:
        orig_tl = harness.TerrainLoader

        def _tl(dem_dir=None, **kw):
            return orig_tl(dem_dir=args.terrain_dir, **kw)
        harness.TerrainLoader = _tl

    trace = {"sead_target_mapping": None, "assets": {}, "route_calls": [], "smooth_calls": []}
    state = {"asset": None}

    # ---- wrappers (observe only) ----
    orig_select = pc._select_sead_targets

    def w_select(formation, threats_dict, target_coord):
        res = orig_select(formation, threats_dict, target_coord)
        trace["sead_target_mapping"] = {
            "target_coord": list(target_coord),
            "candidate_order_rule": "SAM before RADAR, then distance to target, then larger radius",
            "mapping": {aid: (t.get("name") if t else None) for aid, t in res.items()},
        }
        return res
    pc._select_sead_targets = w_select

    orig_resolve = pc._resolve_asset_objective

    def w_resolve(asset, formation, threats_dict, target_coord, sead_targets):
        res = orig_resolve(asset, formation, threats_dict, target_coord, sead_targets)
        state["asset"] = asset.asset_id
        trace["assets"].setdefault(asset.asset_id, {}).update({
            "asset_type": asset.asset_type,
            "assigned_mission": asset.assigned_mission,
            "assigned_target_idx": asset.assigned_target_idx,
            "objective_coord": [r4(c) for c in res[0]],
            "objective_threat_name": res[1],
        })
        return res
    pc._resolve_asset_objective = w_resolve

    orig_tfa = pc._threats_for_asset

    def w_tfa(asset, threats_dict, sead_targets):
        res = orig_tfa(asset, threats_dict, sead_targets)
        mission = (asset.assigned_mission or "").upper()
        if args.cf == "align_strike" and mission == "STRIKE":
            res = [t for t in res if t.get("name") not in CORRIDOR_SUPPRESSED]
        if args.cf == "corridor_only_strike" and mission == "STRIKE":
            # POST-HOC: STRIKE asset excludes exactly the corridor diagnostic's suppressed set, nothing else
            res = [t for t in threats_dict if t.get("name") not in CORRIDOR_SUPPRESSED]
        if args.cf == "pair_sead" and mission == "SEAD":
            a_t = sead_targets.get(asset.asset_id)
            if a_t:
                res = [t for t in res if not (round(float(t["lat"]), 4) == round(float(a_t["lat"]), 4)
                                              and round(float(t["lon"]), 4) == round(float(a_t["lon"]), 4))]
        all_names = [t.get("name") for t in threats_dict]
        kept = [t.get("name") for t in res]
        trace["assets"].setdefault(asset.asset_id, {}).update({
            "threats_considered": kept,
            "threats_removed_for_this_asset": [n for n in all_names if n not in kept],
        })
        return res
    pc._threats_for_asset = w_tfa

    orig_cached = pc._run_one_path_cached

    def w_cached(pathfinder, effective_algorithm, start, end, threats, margin, fuel_state, refuel_count, enable_3d):
        key = pc._path_cache_key(effective_algorithm, start, end, threats, margin, fuel_state, refuel_count, enable_3d)
        cache_hit = key in pc._PATH_CACHE
        rec = {
            "asset": state["asset"],
            "call_index_for_asset": sum(1 for c in trace["route_calls"] if c["asset"] == state["asset"]),
            "algorithm": effective_algorithm,
            "start": [r4(c) for c in start], "end": [r4(c) for c in end],
            "threat_names": [t.get("name") for t in threats],
            "margin_km": r4(margin), "fuel_state": r4(fuel_state), "refuel_count": refuel_count,
            "cache_hit": cache_hit,
        }
        # read-only probes with the pathfinder's own methods
        try:
            rec["risk_block_threshold"] = r4(pathfinder.risk_block_threshold(margin, fuel_state, refuel_count))
        except Exception as e:
            rec["risk_block_threshold"] = f"probe_error: {type(e).__name__}"
        for label, pt in (("start", start), ("end", end)):
            try:
                if len(pt) >= 3:
                    cores = [t.get("name") for t in threats
                             if pathfinder.is_in_threat_core(pt[0], pt[1], pt[2], [t], margin)]
                    rec[f"{label}_inside_threat_core_of"] = cores
                risk = harness.XAIUtils.calculate_risk_score(
                    pt[0], pt[1], threats, margin, terrain_loader=pathfinder.terrain,
                    target_alt=(pt[2] if len(pt) >= 3 else 500.0))
                rec[f"{label}_risk"] = r4(risk)
            except Exception as e:
                rec[f"{label}_probe_error"] = f"{type(e).__name__}: {e}"
        state["pf"] = pathfinder
        state["call"] = {"threats": threats, "margin": margin, "fuel": fuel_state, "refuel": refuel_count}
        buf = io.StringIO()
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(buf):
                res = orig_cached(pathfinder, effective_algorithm, start, end, threats, margin,
                                  fuel_state, refuel_count, enable_3d)
        except Exception as e:
            rec["exception"] = f"{type(e).__name__}: {e}"
            rec["elapsed_s"] = r4(time.perf_counter() - t0)
            rec["stdout"] = buf.getvalue().strip().splitlines()
            trace["route_calls"].append(rec)
            raise
        rec["elapsed_s"] = r4(time.perf_counter() - t0)
        out = buf.getvalue().strip().splitlines()
        rec["stdout"] = out
        rec["raw_waypoints"] = len(res) if res else 0
        if args.save_paths:
            rec["raw_path"] = [[r4(c) for c in pt] for pt in (res or [])]
        if res:
            rec["termination"] = "reached_goal_within_3_cells"
        elif any("100k" in line for line in out):
            rec["termination"] = "node_limit_100k_exceeded"
        elif any("[ERROR]" in line for line in out):
            rec["termination"] = "open_set_exhausted_no_path"
        elif cache_hit:
            rec["termination"] = "cache_hit_empty"
        else:
            rec["termination"] = "empty_without_message"
        trace["route_calls"].append(rec)
        return res
    pc._run_one_path_cached = w_cached

    orig_smooth = pc._smooth

    def w_smooth(raw, point_is_valid=None, path_is_valid=None):
        res = orig_smooth(raw, point_is_valid, path_is_valid)
        rec = {
            "asset": state["asset"], "raw_len": len(raw) if raw else 0, "result_len": len(res) if res else 0,
            "discarded_after_smoothing": bool(raw) and not res,
        }
        if rec["discarded_after_smoothing"] and point_is_valid is not None and state.get("pf") is not None:
            # read-only: which resampled points of the RAW A* path fail the module's own point check, and why
            from modules.path_constraints import resample_path
            pts = resample_path(raw)
            bad = [i for i, p in enumerate(pts) if not point_is_valid(p)]
            pf, call = state["pf"], state["call"]
            reasons = []
            for i in bad[:3] + bad[-3:]:
                p = pts[i]
                r = {"idx": i, "of": len(pts), "point": [r4(c) for c in p[:3]]}
                try:
                    r["terrain_collision"] = bool(pf.is_terrain_collision(*p[:3])) if hasattr(pf, "is_terrain_collision") else None
                    r["threat_collision_with"] = [t.get("name") for t in call["threats"]
                                                  if pf.is_collision_3d(*p[:3], [t], call["margin"], call["fuel"], call["refuel"])]
                    r["threat_core_of"] = [t.get("name") for t in call["threats"]
                                           if pf.is_in_threat_core(p[0], p[1], p[2], [t], call["margin"])]
                except Exception as e:
                    r["probe_error"] = f"{type(e).__name__}: {e}"
                reasons.append(r)
            rec["raw_resampled_points"] = len(pts)
            rec["raw_failing_points"] = len(bad)
            rec["failing_point_samples"] = reasons
        if args.save_paths:
            rec["result_path"] = [[r4(x) for x in pt] for pt in (res or [])]
        trace["smooth_calls"].append(rec)
        return res
    pc._smooth = w_smooth

    # ---- run the unchanged harness ----
    t0 = time.perf_counter()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = harness.main()
    wall = time.perf_counter() - t0

    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))

    # per-asset outcome summary from the trace
    for aid, a in trace["assets"].items():
        calls = [c for c in trace["route_calls"] if c["asset"] == aid]
        sm = [s for s in trace["smooth_calls"] if s["asset"] == aid]
        a["route_calls"] = len(calls)
        a["terminations"] = [c["termination"] for c in calls]
        a["smoothing_discards"] = sum(1 for s in sm if s["discarded_after_smoothing"])
        wp = result["cases"][2]["observed"]["waypoint_counts"].get(aid, {})
        a["final_waypoints"] = wp

    # deterministic content hash: drop wall-clock fields only
    def strip_volatile(obj):
        if isinstance(obj, dict):
            return {k: strip_volatile(v) for k, v in obj.items()
                    if k not in ("generated_at_utc", "runtime_s", "elapsed_s", "validate_time_ms", "solve_time_ms")}
        if isinstance(obj, list):
            return [strip_volatile(v) for v in obj]
        return obj

    result_hash = hashlib.sha256(json.dumps(strip_volatile(result), ensure_ascii=False, sort_keys=True,
                                            default=jdefault).encode("utf-8")).hexdigest()
    trace_hash = hashlib.sha256(json.dumps(strip_volatile(trace), ensure_ascii=False, sort_keys=True,
                                           default=jdefault).encode("utf-8")).hexdigest()

    payload = {
        "app_tree": str(app),
        "module_file_path_computer": str(module_file),
        "terrain_dir_override": args.terrain_dir,
        "counterfactual": args.cf,
        "no_shim": args.no_shim,
        "harness_return_code": rc,
        "wall_s": r4(wall),
        "harness_stdout": buf.getvalue().strip().splitlines(),
        "result_sha256_excluding_wallclock": result_hash,
        "trace_sha256_excluding_wallclock": trace_hash,
        "trace": trace,
    }
    (out_dir / "trace.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=jdefault),
                                        encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "summary": result["summary"], "result_hash": result_hash[:16],
                      "trace_hash": trace_hash[:16], "wall_s": r4(wall)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
