"""Figures for the JKSS revision response letter, drawn only from the E22 re-run records.

Fig R1: threat layout, the two modules' suppression choices, raw A* paths of the SEAD assets
        with the points the post-smoothing check rejected, and the fighter path that appears
        when its excluded threats are aligned with the corridor diagnostic (post-hoc).
Fig R2: altitude profile of the two SEAD raw paths vs terrain + 200 m (minimum AGL).

Violation points are recomputed with the module's own checks (terrain cache + MIN_ALTITUDE_AGL,
pathfinder.is_collision_3d) and their counts are compared with trace.json before plotting.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
APP = HERE.parents[2] / "mission_plan-new_plan_250519"
sys.path.insert(0, str(APP))
sys.path.insert(0, str(HERE.parent))

from modules.config import AIRPORTS  # noqa: E402
from modules.terrain_loader import TerrainLoader  # noqa: E402
from modules.pathfinder_optimized import AStarPathfinder3DOptimized  # noqa: E402
from modules.path_constraints import resample_path  # noqa: E402
import modules.path_computer as pc  # noqa: E402
from run_e2e_threat import TerrainCacheFastLoS  # noqa: E402

NAVY, INK, GREY = "#1E2761", "#23283B", "#6A7183"
RED, AMBER, GREEN = "#B23A48", "#C88A2E", "#3E7A5A"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                     "font.size": 9, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})

base = json.loads((HERE / "fig_baseline" / "trace.json").read_text(encoding="utf-8"))
cf = json.loads((HERE / "fig_cf_align_strike" / "trace.json").read_text(encoding="utf-8"))
res = json.loads((HERE / "fig_baseline" / "result.json").read_text(encoding="utf-8"))
threats = res["scenario"]["threats"]
tmap = {t["name"]: t for t in threats}
TARGET = (39.03, 125.75)
OSAN = tuple(AIRPORTS["오산(Osan)"]["coords"][:2])

SHORT = {"S-75 (번개-1)": "S-75", "SNR-75 (Fan Song) 화력제어 레이더": "SNR-75",
         "S-125 (번개-3)": "S-125", "SNR-125 (LOW BLOW) 화력통제 레이더": "SNR-125",
         "SA-13 (번개-2)": "SA-13"}

terrain_raw = TerrainLoader(dem_dir=str(APP / "data" / "terrain"))
tc = TerrainCacheFastLoS(terrain_raw)
pf = AStarPathfinder3DOptimized(tc)


def km_to_deg(lat, km):
    return km / 110.57, km / (110.57 * math.cos(math.radians(lat)))


def core_radius(t, margin):
    ratio = 0.45 if t["type"] == "SAM" else 0.30
    r = float(t["radius_km"])
    return min(r + margin, max(5.0, r * ratio + 0.4 * margin))


def dist_km(a, b):
    m = math.radians((a[0] + b[0]) / 2)
    return math.hypot((a[0] - b[0]) * 110.57, (a[1] - b[1]) * 110.57 * math.cos(m))


def raw_path(trace, asset):
    for c in trace["trace"]["route_calls"]:
        if c["asset"] == asset and c["call_index_for_asset"] == 0:
            return c, c.get("raw_path", [])
    return None, []


def classify(asset):
    call, raw = raw_path(base, asset)
    names = call["threat_names"]
    th = [tmap[n] for n in names]
    margin = call["margin_km"]
    pts = resample_path(raw)
    out = []
    for p in pts:
        ground = float(tc.get(p[0], p[1]))
        low = p[2] < ground + pc.MIN_ALTITUDE_AGL
        threat = bool(pf.is_collision_3d(p[0], p[1], p[2], th, margin, 1.0, 0))
        out.append({"p": p, "ground": ground, "low": low, "threat": threat})
    expected = next(s for s in base["trace"]["smooth_calls"] if s["asset"] == asset)["raw_failing_points"]
    got = sum(1 for o in out if o["low"] or o["threat"])
    print(f"{asset}: failing points recomputed={got}, trace={expected}")
    assert got == expected, "recomputed violation count does not match the run record"
    return raw, out


ruav_raw, ruav = classify("RUAV-01")
auav_raw, auav = classify("AUAV-01")
f01 = next(s for s in cf["trace"]["smooth_calls"] if s["asset"] == "F-01")["result_path"]

out_dir = HERE / "figures"
out_dir.mkdir(exist_ok=True)

# ---------------- Fig R1 ----------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.9), gridspec_kw={"width_ratios": [0.8, 1.2]})
route_sup = {"SA-13 (번개-2)", "S-125 (번개-3)"}
corr_sup = {"SNR-125 (LOW BLOW) 화력통제 레이더", "S-125 (번개-3)"}


def draw_threats(ax, label=True, zoom=False):
    seen = set()
    for t in threats:
        dlat, dlon = km_to_deg(t["lat"], core_radius(t, 8.0))
        e = matplotlib.patches.Ellipse((t["lon"], t["lat"]), 2 * dlon, 2 * dlat, fc=RED, ec="none", alpha=0.10)
        ax.add_patch(e)
        dlat, dlon = km_to_deg(t["lat"], float(t["radius_km"]))
        ax.add_patch(matplotlib.patches.Ellipse((t["lon"], t["lat"]), 2 * dlon, 2 * dlat, fc="none",
                                                ec=GREY, lw=0.6, ls=(0, (3, 2))))
        key = (t["lat"], t["lon"])
        ax.plot(t["lon"], t["lat"], marker="^", ms=5, color=INK, zorder=5)
        if label and zoom:
            same = [SHORT[x["name"]] for x in threats if (x["lat"], x["lon"]) == key]
            if key not in seen and 125.45 < t["lon"] < 126.0 and 38.75 < t["lat"] < 39.2:
                txt = "·".join(same)
                dx, dy = (-0.075, 0.018) if "SA-13" in txt else (0.015, -0.028)
                ax.text(t["lon"] + dx, t["lat"] + dy, txt, fontsize=7.5, color=INK, zorder=6,
                        clip_on=True, bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))
            seen.add(key)


def plot_path(ax, pts, color, lw, ls="-", z=3):
    ax.plot([p[1] for p in pts], [p[0] for p in pts], color=color, lw=lw, ls=ls, zorder=z)


for ax, zoom in ((axes[0], False), (axes[1], True)):
    draw_threats(ax, zoom=zoom)
    ax.plot([OSAN[1], TARGET[1]], [OSAN[0], TARGET[0]], color=GREY, lw=0.8, ls=":", zorder=2)
    plot_path(ax, f01, GREEN, 1.4, "--")
    for raw, pts in ((ruav_raw, ruav), (auav_raw, auav)):
        plot_path(ax, raw, NAVY, 1.1)
        bad = [o["p"] for o in pts if o["threat"]]
        low = [o["p"] for o in pts if o["low"] and not o["threat"]]
        ax.scatter([p[1] for p in bad], [p[0] for p in bad], s=4, color=RED, zorder=4, lw=0)
        ax.scatter([p[1] for p in low], [p[0] for p in low], s=4, color=AMBER, zorder=4, lw=0)
    ax.plot(TARGET[1], TARGET[0], marker="*", ms=11, color=RED, mec="white", mew=0.6, zorder=7)
    ax.set_aspect(1 / math.cos(math.radians(38.3)))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("경도 (°E)", fontsize=8)

axes[0].plot(OSAN[1], OSAN[0], marker="s", ms=6, color=NAVY, zorder=7)
axes[0].text(OSAN[1] + 0.06, OSAN[0] - 0.02, "오산", fontsize=8, color=INK)
axes[0].text(TARGET[1] + 0.07, TARGET[0] + 0.05, "표적", fontsize=8, color=RED)
axes[0].text(126.40, 38.30, "S-75·SNR-75", fontsize=7.5, color=INK)
axes[0].set_xlim(125.3, 127.4)
axes[0].set_ylim(36.9, 39.45)
axes[0].set_ylabel("위도 (°N)", fontsize=8)
axes[0].set_title("(a) 전체 경로 구간", fontsize=9, color=INK)
axes[0].add_patch(matplotlib.patches.Rectangle((125.45, 38.75), 0.55, 0.45, fc="none", ec=INK, lw=0.6))

axes[1].set_xlim(125.45, 126.0)
axes[1].set_ylim(38.75, 39.2)
axes[1].set_title("(b) 표적 주변 확대", fontsize=9, color=INK)
axes[1].text(TARGET[1] - 0.03, TARGET[0] + 0.03, "표적", fontsize=7.5, color=RED, zorder=8)
tbl = ("경로 모듈 제압 대상: SA-13, S-125\n"
       "회랑 진단 제외 대상: SNR-125, S-125")
axes[1].text(0.02, 0.02, tbl, transform=axes[1].transAxes, fontsize=7.3, color=INK, va="bottom",
             bbox=dict(fc="white", ec=GREY, lw=0.5, pad=2.5))

handles = [
    Line2D([], [], color=NAVY, lw=1.1, label="SEAD 자산 A* 경로(경로 산출 모듈의 선분 검사에서 제외)"),
    Line2D([], [], color=RED, marker="o", ls="none", ms=3, label="선분 검사 위반: 위협 핵심영역"),
    Line2D([], [], color=AMBER, marker="o", ls="none", ms=3, label="선분 검사 위반: 최저고도(지면+200 m) 미달"),
    Line2D([], [], color=GREEN, lw=1.4, ls="--", label="전투기 진입 경로(표적을 덮는 위협 3기를 모두 제외한 사후 진단)"),
    Line2D([], [], color=GREY, lw=0.8, ls=":", label="회랑 진단 직선"),
    matplotlib.patches.Patch(fc=RED, alpha=0.15, label="위협 핵심영역(안전마진 8 km 기준)"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.2, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.15, 1, 1))
fig.savefig(out_dir / "figR1_layout_paths.png", dpi=220)
plt.close(fig)

# ---------------- Fig R2 ----------------
fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=False)
for ax, pts, title in ((axes[0], ruav, "(a) 정찰 UAV(SEAD, 목표 S-125)"),
                       (axes[1], auav, "(b) 공격 UAV(SEAD, 목표 SA-13)")):
    d = [0.0]
    for a, b in zip(pts[:-1], pts[1:]):
        d.append(d[-1] + dist_km(a["p"], b["p"]))
    ax.fill_between(d, [0] * len(d), [o["ground"] for o in pts], color="#D9D4C7", lw=0)
    ax.plot(d, [o["ground"] + pc.MIN_ALTITUDE_AGL for o in pts], color=GREY, lw=0.8, ls="--")
    ax.plot(d, [o["p"][2] for o in pts], color=NAVY, lw=1.1)
    low = [(x, o["p"][2]) for x, o in zip(d, pts) if o["low"] and not o["threat"]]
    thr = [(x, o["p"][2]) for x, o in zip(d, pts) if o["threat"]]
    ax.scatter([x for x, _ in low], [y for _, y in low], s=5, color=AMBER, lw=0, zorder=4)
    ax.scatter([x for x, _ in thr], [y for _, y in thr], s=5, color=RED, lw=0, zorder=4)
    n_low = sum(1 for o in pts if o["low"])
    n_thr = sum(1 for o in pts if o["threat"])
    ax.set_title(f"{title}   점검 지점 {len(pts)}개 중 최저고도 미달 {n_low}개, 위협 핵심영역 {n_thr}개",
                 fontsize=8.5, color=INK, loc="left")
    ax.set_ylabel("고도 (m)", fontsize=8)
    ax.set_ylim(0, max(o["p"][2] for o in pts) * 1.35)
    ax.tick_params(labelsize=7)
axes[1].set_xlabel("출발점부터 경로 거리 (km)", fontsize=8)
handles = [Line2D([], [], color=NAVY, lw=1.1, label="A* 경로 고도"),
           Line2D([], [], color=GREY, lw=0.8, ls="--", label="지면 + 200 m(최저고도)"),
           matplotlib.patches.Patch(fc="#D9D4C7", label="지면(SRTM)"),
           Line2D([], [], color=AMBER, marker="o", ls="none", ms=3, label="최저고도 미달"),
           Line2D([], [], color=RED, marker="o", ls="none", ms=3, label="위협 핵심영역")]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(out_dir / "figR2_altitude_profile.png", dpi=220)
plt.close(fig)
terrain_raw.close()
print("saved", out_dir)
