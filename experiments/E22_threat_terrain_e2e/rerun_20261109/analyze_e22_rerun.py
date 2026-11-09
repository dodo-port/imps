"""Compare E22 re-runs (run1, run2, public_run) with the August result and print a per-asset trace table."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUG = json.loads((HERE.parent / "result.json").read_text(encoding="utf-8"))


def load(name):
    d = HERE / name
    if not (d / "trace.json").exists():
        return None
    return {"result": json.loads((d / "result.json").read_text(encoding="utf-8")),
            "trace": json.loads((d / "trace.json").read_text(encoding="utf-8"))}


def core(res):
    return {
        "summary": res["summary"],
        "stage_pass": {c["case_id"]: c["passed"] for c in res["cases"]},
        "waypoints": res["cases"][2]["observed"]["waypoint_counts"],
        "policy": {k: v for k, v in res["cases"][1]["observed"]["doctrine_policy"].items() if k not in ("refs", "rationale")},
        "refs": [(r["source"], r["page"]) for r in res["cases"][1]["observed"]["doctrine_policy"]["refs"]],
        "formation": [(a["asset_id"], a["assigned_mission"], a["assigned_target_idx"])
                      for a in res["cases"][1]["observed"]["formation"]["assets"]],
        "validator": {k: res["cases"][3]["observed"]["report"][k] for k in ("is_valid", "error_count", "warning_count")},
        "corridor": {k: res["cross_module_consistency"]["corridor"][k]
                     for k in ("initial_max_risk", "suppressed_threats", "open_at_k", "final_max_risk")},
        "cross": {k: res["cross_module_consistency"]["cross_module_check"][k]
                  for k in ("corridor_claims_open", "routing_viable", "sead_targets_bound", "inconsistency_detected")},
    }


runs = {n: load(n) for n in ("run1", "run2", "public_run")}
aug = core(AUG)
print("== August core ==")
print(json.dumps(aug, ensure_ascii=False, indent=1))
for n, r in runs.items():
    if r is None:
        print(f"== {n}: missing"); continue
    c = core(r["result"])
    diffs = [k for k in aug if aug[k] != c[k]]
    print(f"== {n}: hash result={r['trace']['result_sha256_excluding_wallclock'][:16]} "
          f"trace={r['trace']['trace_sha256_excluding_wallclock'][:16]} wall={r['trace']['wall_s']} "
          f"| fields differing from August: {diffs or 'none'}")
    for k in diffs:
        print("   ", k, "AUG:", json.dumps(aug[k], ensure_ascii=False))
        print("   ", k, n.upper() + ":", json.dumps(c[k], ensure_ascii=False))

if runs["run1"] and runs["run2"]:
    same = (runs["run1"]["trace"]["result_sha256_excluding_wallclock"] == runs["run2"]["trace"]["result_sha256_excluding_wallclock"]
            and runs["run1"]["trace"]["trace_sha256_excluding_wallclock"] == runs["run2"]["trace"]["trace_sha256_excluding_wallclock"])
    print("run1 vs run2 identical (result+trace):", same)

r = runs["run1"]
if r:
    t = r["trace"]["trace"]
    print("\n== SEAD target mapping (routing module) ==")
    print(json.dumps(t["sead_target_mapping"], ensure_ascii=False, indent=1))
    print("\n== per asset ==")
    for aid, a in t["assets"].items():
        print(json.dumps({aid: a}, ensure_ascii=False))
    print("\n== route calls ==")
    for c in t["route_calls"]:
        keep = {k: c[k] for k in c if k not in ("start",)}
        print(json.dumps(keep, ensure_ascii=False))
    print("\n== smooth calls ==")
    for s in t["smooth_calls"]:
        print(json.dumps(s, ensure_ascii=False))
    print("\n== harness stdout ==")
    for line in r["trace"]["harness_stdout"][:20]:
        print(line)
