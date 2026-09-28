"""
시나리오 비교 페이지 — 전술 시나리오 N종 동시 계산·비교
동일 조건(위협·출발지·목표)에서 연료/마진 조건만 바꿔 경로·위험도·거리를 자동 비교합니다.
"""
from __future__ import annotations

import hashlib
import json

import streamlit as st
import folium
import pandas as pd
from streamlit_folium import st_folium

from modules.config import (
    AIRPORTS, MAP_CENTER, MAP_ZOOM,
    RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM,
)
from modules.pathfinder_optimized import AStarPathfinder3DOptimized
from modules.scenario_compare import (
    run_all_scenarios, build_comparison_table,
    SCENARIO_PRESETS, ScenarioConfig,
)


# ── 군사 다크 테마 (메인과 동일) ──────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], .stApp {
    background-color: #0D1117 !important;
    color: #C9D1D9 !important;
    font-family: 'Segoe UI', 'Noto Sans KR', sans-serif;
}
[data-testid="stSidebar"] { background-color: #161B22 !important; border-right: 1px solid #30363D; }
h1, h2, h3, h4 { color: #58A6FF !important; }
h1 { font-size: 1.6rem !important; border-bottom: 2px solid #1F6FEB; padding-bottom: 6px; }
[data-testid="stMetric"] { background: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 10px 14px; }
[data-testid="stMetricValue"] { color: #79C0FF !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #8B949E !important; font-size: 11px; }
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1F6FEB 0%, #0D419D 100%);
    color: #fff !important; border: none; border-radius: 6px; font-weight: 600;
}
[data-testid="stButton"] > button:hover { filter: brightness(1.15); }
[data-testid="stDataFrame"] { border: 1px solid #30363D; border-radius: 8px; }
.stDataFrame thead tr th { background: #161B22 !important; color: #58A6FF !important; }
[data-testid="stExpander"] summary { background: #161B22 !important; border: 1px solid #30363D !important; border-radius: 6px; color: #58A6FF !important; font-weight: 600; }
hr { border-color: #30363D !important; }
[data-testid="stCaptionContainer"] { color: #8B949E !important; }
input, textarea, select, [data-baseweb="input"] input {
    background-color: #0D1117 !important; color: #C9D1D9 !important;
    border: 1px solid #30363D !important; border-radius: 6px;
}
[data-baseweb="select"] > div:first-child { background-color: #161B22 !important; border-color: #30363D !important; color: #C9D1D9 !important; }
[data-baseweb="popover"] { background: #161B22 !important; border: 1px solid #30363D !important; }
</style>
""", unsafe_allow_html=True)

# ── 세션 상태 확인 ──────────────────────────────────────────────────
if "mission" not in st.session_state:
    st.warning("⚠️ 메인 페이지에서 먼저 임무를 설정하세요.")
    st.page_link("streamlit_app.py", label="← 메인으로 돌아가기", icon="🏠")
    st.stop()

if "terrain" not in st.session_state:
    st.warning("⚠️ 지형 데이터가 초기화되지 않았습니다. 메인 페이지를 먼저 실행하세요.")
    st.page_link("streamlit_app.py", label="← 메인으로 돌아가기", icon="🏠")
    st.stop()

mission      = st.session_state.mission
terrain_fast = st.session_state.terrain

# ── 뒤로가기 링크 ─────────────────────────────────────────────────
st.page_link("streamlit_app.py", label="← 메인 임무계획으로 돌아가기", icon="🏠")
st.title("📊 전술 시나리오 비교")

n_presets = len(SCENARIO_PRESETS)
st.caption(f"동일 조건(위협·출발지·목표)에서 연료·마진 조건만 바꿔 {n_presets}종 경로·위험도·거리를 자동 비교합니다.")

# ── 현재 임무 상태 요약 ────────────────────────────────────────────
with st.expander("📌 현재 임무 상태", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("출발지", mission.params.start)
    c2.metric("목표 위도", f"{mission.params.target_lat:.3f}")
    c3.metric("목표 경도", f"{mission.params.target_lon:.3f}")
    c4.metric("위협 수", f"{len(mission.threats)}개")

st.divider()

# ── 프리셋 카드 요약 ───────────────────────────────────────────────
with st.expander("📋 프리셋 목록", expanded=True):
    card_cols = st.columns(n_presets)
    for idx, (col, preset) in enumerate(zip(card_cols, SCENARIO_PRESETS)):
        with col:
            swatch = (
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"background:{preset.color};border-radius:3px;vertical-align:middle;"
                f"margin-right:4px;'></span>"
            )
            st.markdown(
                f"{swatch}**{preset.label}**<br>"
                f"<span style='font-size:11px;color:#8B949E;'>{preset.description}</span>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"연료 {preset.fuel_state:.0%} | 급유 {preset.refuel_count}회 | "
                f"마진 {preset.safety_margin_km:.0f}km"
            )

# ── 사용자 정의 시나리오 설정 (옵션) ──────────────────────────────
with st.expander("⚙️ 시나리오 파라미터 직접 설정 (선택)", expanded=False):
    st.markdown("**기본 프리셋을 쓰거나 아래에서 직접 조정하세요.**")
    sc_cols = st.columns(n_presets)
    custom_configs = []
    for idx, (col, preset) in enumerate(zip(sc_cols, SCENARIO_PRESETS)):
        with col:
            st.markdown(f"**{preset.label}**")
            fs = st.slider("연료 상태", 0.20, 1.00, preset.fuel_state,
                           step=0.05, key=f"_sc_fuel_{idx}")
            rc = st.selectbox("급유 횟수", [0, 1, 2],
                              index=min(preset.refuel_count, 2), key=f"_sc_refuel_{idx}")
            mg = st.slider("안전마진(km)", 1.0, 30.0, float(preset.safety_margin_km),
                           step=1.0, key=f"_sc_margin_{idx}")
            custom_configs.append(ScenarioConfig(
                label=preset.label,
                description=preset.description,
                fuel_state=fs,
                refuel_count=rc,
                safety_margin_km=mg,
                color=preset.color,
                dash=preset.dash,
            ))

use_custom = st.checkbox("직접 설정한 파라미터 사용", value=False, key="_sc_use_custom")
_configs_for_sig = custom_configs if use_custom else SCENARIO_PRESETS
_scenario_sig_payload = {
    "params": mission.params.to_dict(),
    "threats": [t.to_dict() for t in mission.threats],
    "use_custom": bool(use_custom),
    "configs": [cfg.__dict__ for cfg in _configs_for_sig],
}
_current_scenario_sig = hashlib.md5(
    json.dumps(_scenario_sig_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
).hexdigest()
if st.session_state.get("_scenario_results_sig") != _current_scenario_sig:
    st.session_state.pop("_scenario_results", None)
    st.session_state["_scenario_results_sig"] = _current_scenario_sig

run_btn = st.button(
    f"🚀 시나리오 {n_presets}종 동시 계산",
    type="primary",
    use_container_width=True,
)

if run_btn:
    sc_start  = list(AIRPORTS[mission.params.start]["coords"])
    sc_target = [mission.params.target_lat, mission.params.target_lon]
    sc_threats = [t.to_dict() for t in mission.threats]

    # 다중 목표 체인 추출 (활성화된 extra_targets만)
    sc_extra = []
    for et in sorted(getattr(mission.params, "extra_targets", []), key=lambda t: t.get("priority", 2)):
        if et.get("enabled", True) and "lat" in et and "lon" in et:
            sc_extra.append([float(et["lat"]), float(et["lon"])])

    if mission.params.enable_3d:
        get_elev = terrain_fast.get if hasattr(terrain_fast, "get") else terrain_fast.get_elevation
        s_elev = get_elev(sc_start[0], sc_start[1])
        t_elev = get_elev(sc_target[0], sc_target[1])
        sc_start  = [*sc_start,  s_elev + 800]
        sc_target = [*sc_target, t_elev + 800]
        sc_extra  = [[e[0], e[1], get_elev(e[0], e[1]) + 800] for e in sc_extra]

    sc_pathfinder  = AStarPathfinder3DOptimized(terrain_fast)
    configs_to_use = custom_configs if use_custom else None

    with st.spinner(f"{n_presets}개 시나리오 경로 계산 중..." + (f" (목표 {1 + len(sc_extra)}개 체인)" if sc_extra else "")):
        sc_results = run_all_scenarios(
            sc_pathfinder,
            sc_start,
            sc_target,
            sc_threats,
            mission.params.rtb,
            terrain_fast,
            custom_configs=configs_to_use,
            extra_targets=sc_extra or None,
        )
    st.session_state["_scenario_results"] = sc_results
    st.session_state["_scenario_results_sig"] = _current_scenario_sig
    st.rerun()

# ── 결과 표시 ──────────────────────────────────────────────────────
sc_results = st.session_state.get("_scenario_results", [])
if not sc_results:
    st.info("위 버튼을 눌러 시나리오를 계산하세요.")
    st.stop()

st.divider()
st.markdown("### 📋 정량 비교표")
rows  = build_comparison_table(sc_results)
sc_df = pd.DataFrame(rows)

def _sc_highlight(row):
    mr = row.get("최대 위험도", 0)
    if mr >= RISK_THRESHOLD_HIGH:
        return ["background-color: #c0392b; color: #ffffff; font-weight: bold"] * len(row)
    if mr >= RISK_THRESHOLD_MEDIUM:
        return ["background-color: #e67e22; color: #ffffff; font-weight: bold"] * len(row)
    return ["background-color: #27ae60; color: #ffffff; font-weight: bold"] * len(row)

st.dataframe(
    sc_df.style.apply(_sc_highlight, axis=1),
    hide_index=True,
    use_container_width=True,
)

# ── 핵심 비교 메트릭 ──
st.divider()
st.markdown("### 📈 핵심 지표 비교")
metric_cols = st.columns(len(sc_results))
for col, r in zip(metric_cols, sc_results):
    with col:
        ok_icon    = "✅" if r.success else "❌"
        risk_color = (
            "🔴" if r.max_risk >= RISK_THRESHOLD_HIGH
            else "🟡" if r.max_risk >= RISK_THRESHOLD_MEDIUM
            else "🟢"
        )
        st.markdown(f"**{r.config.label}**")
        st.metric("총 비행거리",  f"{r.total_length_km:.1f} km")
        st.metric("최대 위험도",  f"{risk_color} {r.max_risk:.3f}")
        st.metric("고위험 구간",  f"{r.high_risk_segments}개")
        st.metric("계산 시간",    f"{r.calc_time_s:.2f}s")
        st.caption(f"{ok_icon} {r.config.description}")

# ── 시나리오 지도 (N경로 동시 표시) ──
st.divider()
st.markdown(f"### 🗺️ {n_presets}종 경로 비교 지도")
sc_map = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="OpenStreetMap")  # 키 불필요

t_color_map = {
    "SAM":   ("#D32F2F", "red"),
    "RADAR": ("#6A1B9A", "purple"),
    "NFZ":   ("#E65100", "orange"),
}
for t in mission.threats:
    hex_c, _ = t_color_map.get(t.type, ("#607D8B", "gray"))
    if t.type == "NFZ":
        if all(v is not None for v in [t.lat_min, t.lon_min, t.lat_max, t.lon_max]):
            folium.Rectangle(
                [[t.lat_min, t.lon_min], [t.lat_max, t.lon_max]],
                color=hex_c, weight=1.5, fill=True, fill_color=hex_c, fill_opacity=0.07,
            ).add_to(sc_map)
    elif t.lat is not None and t.lon is not None:
        folium.Circle(
            [t.lat, t.lon], radius=t.radius_km * 1000,
            color=hex_c, weight=1.5, fill=True, fill_color=hex_c, fill_opacity=0.06,
        ).add_to(sc_map)
        folium.Circle(
            [t.lat, t.lon], radius=t.radius_km * 1000,
            color=hex_c, weight=2.0, fill=False,
        ).add_to(sc_map)

sc_start_coord  = AIRPORTS[mission.params.start]["coords"]
sc_target_coord = [mission.params.target_lat, mission.params.target_lon]

for r in sc_results:
    if not r.path_in:
        continue
    latlon_in  = [(p[0], p[1]) for p in r.path_in]
    latlon_out = [(p[0], p[1]) for p in r.path_out] if r.path_out else []
    clr  = r.config.color
    dash = r.config.dash or None

    folium.PolyLine(latlon_in, color="white", weight=8, opacity=0.8).add_to(sc_map)
    kw = dict(
        color=clr, weight=5, opacity=0.95,
        tooltip=f"{r.config.label} ▶ Ingress | dist={r.total_length_km:.1f}km | risk={r.max_risk:.3f}",
    )
    if dash:
        kw["dash_array"] = dash
    folium.PolyLine(latlon_in, **kw).add_to(sc_map)

    if latlon_out:
        folium.PolyLine(latlon_out, color="white", weight=6, opacity=0.6).add_to(sc_map)
        folium.PolyLine(
            latlon_out, color=clr, weight=3, opacity=0.7,
            dash_array="6 4", tooltip=f"{r.config.label} ◀ Egress",
        ).add_to(sc_map)

folium.Marker(
    sc_start_coord,
    icon=folium.Icon(color="blue", icon="plane", prefix="fa"),
    tooltip=f"🛫 {mission.params.start}",
).add_to(sc_map)
folium.Marker(
    sc_target_coord,
    icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
    tooltip=f"🎯 [P1] {mission.params.target_name}",
).add_to(sc_map)

# 추가 목표 마커
_extra_prio_colors = {1: "red", 2: "orange", 3: "green"}
for _ei, _et in enumerate(sorted(getattr(mission.params, "extra_targets", []), key=lambda t: t.get("priority", 2))):
    if "lat" not in _et or "lon" not in _et:
        continue
    _prio = int(_et.get("priority", 2))
    folium.Marker(
        [float(_et["lat"]), float(_et["lon"])],
        icon=folium.Icon(color=_extra_prio_colors.get(_prio, "orange"), icon="flag", prefix="fa"),
        tooltip=f"\U0001f3af [P{_ei+2}] {_et.get('name', f'Target-{_ei+2}')}",
    ).add_to(sc_map)

legend_rows = "".join(
    f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0;'>"
    f"<div style='width:30px;height:4px;background:{r.config.color};border-radius:2px;'></div>"
    f"<span style='font-size:11px;'>{r.config.label}</span></div>"
    for r in sc_results if r.success
)
sc_map.get_root().html.add_child(folium.Element(
    f"<div style='position:fixed;bottom:40px;left:40px;z-index:9999;"
    f"background:rgba(10,10,20,0.88);color:white;padding:10px 14px;"
    f"border-radius:8px;font-size:12px;border:1px solid rgba(255,255,255,0.15);'>"
    f"<b>📍 시나리오 범례</b><br>{legend_rows}</div>"
))

st_folium(sc_map, width="100%", height=580, returned_objects=[])

# ── CSV 내보내기 ───────────────────────────────────────────────────
st.divider()
csv_sc = sc_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 비교표 CSV 다운로드",
    csv_sc,
    "scenario_comparison.csv",
    "text/csv",
    use_container_width=True,
)
