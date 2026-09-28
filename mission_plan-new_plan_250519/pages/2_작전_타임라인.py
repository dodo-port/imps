"""
작전 타임라인 페이지 — 전체화면 작전 시뮬레이터

자산 위치 / 위협 파괴 상태 / 이벤트 로그를 시간 축으로 시각화.
Plotly Scattermapbox 사용 → Folium 대비 슬라이더 반응 10배↑
"""
from __future__ import annotations

import math
import json
import hashlib

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from modules.config import AIRPORTS, ASSET_PERFORMANCE
from modules.mission_timeline import build_timeline, get_frame_at
from modules.path_computer import compute_formation_paths
from modules.pathfinder import AStarPathfinder
from modules.pathfinder_optimized import AStarPathfinder3DOptimized


# ── 세션 상태 확인 ──────────────────────────────────────────────────
if "mission" not in st.session_state:
    st.warning("⚠️ 메인 페이지에서 먼저 임무를 설정하세요.")
    st.page_link("streamlit_app.py", label="← 메인으로 돌아가기", icon="🏠")
    st.stop()

mission   = st.session_state.mission
_fp       = st.session_state.get("_formation_paths", {})
_threats  = [t.to_dict() for t in mission.threats]
_terrain  = st.session_state.get("terrain")

# ── Formation 없으면 경로 자동 계산 ────────────────────────────────
if not _fp and mission.formation and getattr(mission.formation, "is_feasible", False):
    with st.spinner("⏳ 타임라인용 경로 계산 중..."):
        _algo = mission.params.algorithm
        if _algo in ("RRT", "RRT*"):
            _algo = "A* 3D" if mission.params.enable_3d else "A*"
        try:
            _pf = AStarPathfinder3DOptimized(_terrain) if "3D" in _algo else AStarPathfinder()
            _fp_raw = compute_formation_paths(
                mission, _pf, _algo, _terrain, _threats,
                float(getattr(mission.params, "fuel_state", 1.0)),
                int(getattr(mission.params, "refuel_count", 0)),
            )
            if _fp_raw:
                _fp = {
                    aid: {
                        "in": d["in"], "out": d["out"],
                        "threats":  d.get("threats", _threats),
                        "color":    d.get("color", "#888888"),
                        "callsign": d.get("callsign", aid),
                        "mission":  d.get("mission", "?"),
                        "type":     d.get("type", "?"),
                        "objective":d.get("objective", ""),
                        "target_idx": d.get("target_idx"),
                        "expendable": d.get("expendable", False),
                        "asset_margin": d.get("asset_margin"),
                        "chain_target_coords": d.get("chain_target_coords"),
                    }
                    for aid, d in _fp_raw.items()
                }
                st.session_state["_formation_paths"] = _fp
        except Exception as e:
            st.error(f"경로 계산 실패: {e}")

if not _fp:
    st.info("💡 **편대 구성 탭** → 최적화 실행 후 다시 방문하세요.")
    st.page_link("streamlit_app.py", label="← 메인으로 돌아가기", icon="🏠")
    st.stop()

# ── 타임라인 빌드 (캐시) ────────────────────────────────────────────
def _path_signature(path: list) -> list:
    sig = []
    for p in path or []:
        item = [round(float(p[0]), 5), round(float(p[1]), 5)]
        if len(p) >= 3:
            item.append(round(float(p[2]), 1))
        sig.append(item)
    return sig


_sig_payload = {
    "paths": {
        aid: {
            "in": _path_signature(v.get("in", [])),
            "out": _path_signature(v.get("out", [])),
            "callsign": v.get("callsign"),
            "mission": v.get("mission"),
            "type": v.get("type"),
            "objective": v.get("objective"),
            "target_idx": v.get("target_idx"),
            "expendable": v.get("expendable"),
            "asset_margin": v.get("asset_margin"),
            "chain_target_coords": v.get("chain_target_coords"),
        }
        for aid, v in _fp.items()
    },
    "threats": _threats,
    "formation": {
        "sequence": getattr(mission.formation, "mission_sequence", []),
        "assets": [
            {
                "id": a.asset_id,
                "type": a.asset_type,
                "mission": a.assigned_mission,
                "target_idx": a.assigned_target_idx,
                "callsign": a.callsign,
            }
            for a in getattr(mission.formation, "assets", [])
        ] if mission.formation else [],
    },
}
_sig = hashlib.md5(
    json.dumps(_sig_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
).hexdigest()

if st.session_state.get("_tl_sig2") != _sig or st.session_state.get("_tl2") is None:
    with st.spinner("⏳ 타임라인 생성 중..."):
        _tl = build_timeline(_fp, mission.formation, _threats, time_resolution_min=0.5)
    st.session_state["_tl2"]    = _tl
    st.session_state["_tl_sig2"] = _sig

_tl = st.session_state["_tl2"]
if _tl is None:
    st.error("타임라인 데이터 생성 실패")
    st.stop()

total_min = _tl.total_duration_min


# ================================================================
# Helpers
# ================================================================

def _circle_coords(lat: float, lon: float, radius_km: float, n: int = 60):
    # 다른 모듈(xai_utils, validator, pathfinder, czml_exporter 등)이 모두
    # LAT_TO_KM=110.57로 통일되어 있어 동일 값으로 맞춤. 이전 111.0은 약 0.4%
    # 외곽으로 그려져 위협 반경이 실제보다 약간 크게 보이는 일관성 결함이 있었음.
    LAT_TO_KM = 110.57
    lats, lons = [], []
    cos_lat = math.cos(math.radians(lat))
    for i in range(n + 1):
        a = 2 * math.pi * i / n
        lats.append(lat + radius_km / LAT_TO_KM * math.cos(a))
        lons.append(lon + radius_km / (LAT_TO_KM * max(cos_lat, 0.01)) * math.sin(a))
    return lats, lons


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


_STATUS_FILL   = {"active": 0.18, "suppressed": 0.10, "destroyed": 0.03}
_STATUS_BORDER = {"active": "#EF5350", "suppressed": "#FFA726", "destroyed": "#555555"}
_STATUS_KO     = {"active": "🔴 활성", "suppressed": "🟡 억압됨", "destroyed": "💀 파괴됨"}
_PHASE_KO      = {
    "ingress":   "🔵 침투중",
    "on_target": "🔴 목표상공",
    "egress":    "🟡 이탈중",
    "rtb":       "🟢 귀환완료",
    "standby":   "⚪ 대기",
}
_PHASE_COLORS  = {"standby": "#546E7A", "ingress": "#4FC3F7", "on_target": "#EF5350", "egress": "#66BB6A", "rtb": "#A5D6A7"}


def _path_split_idx(cum_km: list, dist_km: float) -> int:
    """경로에서 dist_km에 해당하는 인덱스 반환."""
    for i, d in enumerate(cum_km):
        if d >= dist_km:
            return i
    return len(cum_km)


def _add_path_segment(fig, seg, col, width, opacity):
    if len(seg) >= 2:
        fig.add_trace(go.Scattermapbox(
            lat=[p[0] for p in seg],
            lon=[p[1] for p in seg],
            mode="lines",
            line=dict(color=col, width=width),
            opacity=opacity,
            hoverinfo="skip",
            showlegend=False,
        ))


def _build_map(frame, t_now: float) -> go.Figure:
    """현재 프레임의 Plotly 지도 생성.
    - 이미 지나간 경로: 굵고 선명
    - 아직 안 간 경로: 얇고 희미 (점선 효과)
    """
    fig = go.Figure()

    # ── 위협 반경 원 ──
    for thr in _threats:
        tname  = thr.get("name", "")
        status = frame.threat_statuses.get(tname, "active")
        tlat, tlon = thr.get("lat"), thr.get("lon")
        radius = float(thr.get("radius_km", 0))
        if tlat is None or tlon is None or radius <= 0:
            continue
        clats, clons = _circle_coords(tlat, tlon, radius)
        border_col = _STATUS_BORDER[status]
        fill_col   = _hex_to_rgba(border_col, _STATUS_FILL[status])
        fig.add_trace(go.Scattermapbox(
            lat=clats, lon=clons, mode="lines",
            fill="toself", fillcolor=fill_col,
            line=dict(color=border_col, width=2),
            hovertemplate=f"<b>{tname}</b><br>{_STATUS_KO[status]}<br>반경 {radius:.0f}km<extra></extra>",
            showlegend=False,
        ))
        st_icon = {"active": "⚠️", "suppressed": "🔕", "destroyed": "💀"}[status]
        fig.add_trace(go.Scattermapbox(
            lat=[tlat], lon=[tlon], mode="text",
            text=[f"{st_icon} {tname}"],
            textfont=dict(size=12, color=border_col),
            hoverinfo="skip", showlegend=False,
        ))

    # ── 경로: 지나간 부분(굵고 선명) / 남은 부분(얇고 희미) ──
    for aid, pdata in _fp.items():
        col    = _tl.asset_info.get(aid, {}).get("color", "#888")
        timing = _tl.asset_timing.get(aid, {})
        speed  = timing.get("speed_kmh", 300.0)
        t_arr  = timing.get("t_arrive_min", 0.0)
        t_eg   = timing.get("t_egress_start_min", t_arr + 3.0)
        t_done = timing.get("t_done_min", t_eg)

        # Ingress
        path_in = pdata.get("in", [])
        cum_in  = timing.get("cum_in", [0.0] * len(path_in))
        delay   = timing.get("launch_delay_min", 0.0)
        if len(path_in) >= 2:
            if t_now < delay:
                # 아직 이륙 전 — 전체 경로를 희미하게 표시
                _add_path_segment(fig, path_in, col, 1, 0.10)
            elif t_now <= t_arr:
                dist = max(0.0, (t_now - delay) / 60.0) * speed
                si   = _path_split_idx(cum_in, dist)
                _add_path_segment(fig, path_in[:max(si, 2)],       col, 3, 0.90)   # 지나간
                _add_path_segment(fig, path_in[max(si - 1, 0):],   col, 1, 0.15)   # 남은
            else:
                _add_path_segment(fig, path_in, col, 2, 0.50)   # 완료

        # Egress
        path_out = pdata.get("out", [])
        cum_out  = timing.get("cum_out", [0.0] * len(path_out))
        if len(path_out) >= 2:
            if t_now < t_eg:
                _add_path_segment(fig, path_out, col, 1, 0.10)  # 아직 시작 안 함
            elif t_now <= t_done:
                dist = ((t_now - t_eg) / 60.0) * speed
                si   = _path_split_idx(cum_out, dist)
                _add_path_segment(fig, path_out[:max(si, 2)],     col, 3, 0.90)
                _add_path_segment(fig, path_out[max(si - 1, 0):], col, 1, 0.15)
            else:
                _add_path_segment(fig, path_out, col, 2, 0.40)  # 완료

    # ── 자산 현재 위치 (halo + 마커 + 레이블) ──
    _label_pos = ["top right", "top left", "bottom right", "bottom left", "top center"]
    _asset_list = list(frame.asset_states.items())
    for idx, (aid, astate) in enumerate(_asset_list):
        ainfo        = _tl.asset_info.get(aid, {})
        callsign     = ainfo.get("callsign", aid)
        mission_type = ainfo.get("mission", "?")
        atype        = ainfo.get("type", "fighter")
        col          = ainfo.get("color", "#E53935")
        phase        = astate.phase
        lpos         = _label_pos[idx % len(_label_pos)]

        size   = 22 if atype == "fighter" else 16
        symbol = "airport" if atype == "fighter" else ("circle" if "recon" in atype else "square")

        # 후광(halo) — 현재 위치 강조
        fig.add_trace(go.Scattermapbox(
            lat=[astate.lat], lon=[astate.lon],
            mode="markers",
            marker=dict(size=size + 14, color=col, opacity=0.25),
            hoverinfo="skip", showlegend=False,
        ))
        # 메인 마커 + 레이블
        fig.add_trace(go.Scattermapbox(
            lat=[astate.lat], lon=[astate.lon],
            mode="markers+text",
            marker=dict(size=size, color=col, symbol=symbol),
            text=[callsign],
            textfont=dict(size=11, color="#ffffff"),
            textposition=lpos,
            hovertemplate=(
                f"<b>{callsign}</b><br>"
                f"임무: {mission_type}<br>"
                f"상태: {_PHASE_KO.get(phase, phase)}<br>"
                f"고도: {astate.alt:.0f}m<extra></extra>"
            ),
            showlegend=False,
        ))

    # ── 현재 시각 이벤트 플래시 ──
    for ev in frame.events_at:
        fig.add_trace(go.Scattermapbox(
            lat=[ev.lat], lon=[ev.lon],
            mode="markers+text",
            marker=dict(size=28, color="rgba(255,235,59,0.9)"),
            text=[ev.icon],
            textfont=dict(size=18),
            textposition="middle center",
            hovertemplate=f"<b>{ev.description}</b><extra></extra>",
            showlegend=False,
        ))

    # ── 출발 기지 마커 ──
    start_info = AIRPORTS.get(mission.params.start, {})
    start_coord = start_info.get("coords", [37.5, 127.0])
    fig.add_trace(go.Scattermapbox(
        lat=[start_coord[0]], lon=[start_coord[1]],
        mode="markers+text",
        marker=dict(size=12, color="#42A5F5", symbol="star"),
        text=[f"🛫 {mission.params.start}"],
        textfont=dict(size=10, color="#42A5F5"),
        textposition="top right",
        hoverinfo="text",
        showlegend=False,
    ))

    # ── 주 목표 마커 ──
    fig.add_trace(go.Scattermapbox(
        lat=[mission.params.target_lat], lon=[mission.params.target_lon],
        mode="markers+text",
        marker=dict(size=14, color="#EF5350", symbol="marker"),
        text=[f"🎯 {mission.params.target_name}"],
        textfont=dict(size=10, color="#EF5350"),
        textposition="top right",
        hoverinfo="text",
        showlegend=False,
    ))

    # ── 추가 목표 마커 (Secondary Targets) ──
    _extra_colors_tl = ["#FB8C00", "#AB47BC", "#00897B"]
    for _ei, _et in enumerate(getattr(mission.params, "extra_targets", [])):
        if "lat" not in _et or "lon" not in _et:
            continue
        _ec = _extra_colors_tl[_ei % len(_extra_colors_tl)]
        fig.add_trace(go.Scattermapbox(
            lat=[float(_et["lat"])], lon=[float(_et["lon"])],
            mode="markers+text",
            marker=dict(size=12, color=_ec, symbol="marker"),
            text=[f"\U0001f3af {_et.get('name', f'Target-{_ei+2}')}"],
            textfont=dict(size=9, color=_ec),
            textposition="top right",
            hoverinfo="text",
            showlegend=False,
        ))

    # ── 레이아웃 ──
    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=38.5, lon=127.0),
            zoom=6,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0e1117",
        showlegend=False,
        height=560,
        uirevision="timeline_map",   # ← 지도 줌/이동 상태 유지
    )
    return fig


def _build_gantt(t_now: float) -> go.Figure:
    fig = go.Figure()
    y_labels = []

    for aid, ainfo in _tl.asset_info.items():
        cs      = ainfo["callsign"]
        col     = ainfo["color"]
        t_delay = ainfo.get("launch_delay_min", 0.0)
        t_arr   = ainfo["t_arrive_min"]
        t_eg    = ainfo["t_egress_min"]
        t_dn    = ainfo["t_done_min"]
        y_labels.append(cs)

        def _bar(x0, x1, phase_col, label, _cs=cs):
            if x1 <= x0:
                return
            fig.add_trace(go.Bar(
                name=label, x=[x1 - x0], y=[_cs],
                base=x0, orientation="h",
                marker_color=phase_col,
                showlegend=False,
                hovertemplate=f"{_cs} {label}: T+{x0:.1f}→T+{x1:.1f}분<extra></extra>",
            ))

        _bar(0.0, t_delay, "#546E7A",             "대기")      # standby (회색)
        _bar(t_delay, t_arr, _PHASE_COLORS["ingress"],   "침투")
        _bar(t_arr, t_eg,   _PHASE_COLORS["on_target"], "목표상공")
        _bar(t_eg, t_dn,    _PHASE_COLORS["egress"],    "귀환")

    # 현재 시각 선
    fig.add_vline(
        x=t_now, line_color="yellow", line_width=2, line_dash="dash",
        annotation_text=f"T+{t_now:.1f}분",
        annotation_font=dict(color="yellow", size=12),
    )
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1a2e",
        font_color="#ffffff",
        xaxis=dict(title="경과 시간 (분)", gridcolor="#333", range=[0, total_min + 1]),
        yaxis=dict(title=""),
        height=max(140, len(y_labels) * 40 + 60),
        margin=dict(l=10, r=10, t=10, b=30),
    )
    return fig


# ================================================================
# UI
# ================================================================

# 헤더
_hc1, _hc2, _hc3 = st.columns([2, 1, 1])
with _hc1:
    st.title("⏱️ 작전 타임라인 시뮬레이터")
    st.caption(f"총 작전 시간: **{total_min:.1f}분** | 자산: {len(_fp)}개 | 위협: {len(_threats)}개 | 이벤트: {len(_tl.events)}건")
with _hc2:
    st.page_link("streamlit_app.py", label="← 메인 (임무계획)", icon="🏠")
with _hc3:
    if st.button("🔄 타임라인 재계산", use_container_width=True):
        st.session_state.pop("_tl2", None)
        st.session_state.pop("_tl_sig2", None)
        st.rerun()

st.divider()

# ── 시간 슬라이더 ──────────────────────────────────────────────────
# Streamlit 규칙: 슬라이더 뒤에 오는 버튼은 slider key를 수정할 수 없음.
# → key= 없이 value=로만 제어하고, _tl2_t 별도 상태 변수 사용.
if "_tl2_t" not in st.session_state:
    st.session_state["_tl2_t"] = 0.0

_sc1, _sc2, _sc3, _sc4 = st.columns([1, 6, 1, 1])

if _sc1.button("◀ -0.5분", use_container_width=True):
    st.session_state["_tl2_t"] = max(0.0, st.session_state["_tl2_t"] - 0.5)
    st.rerun()

_t_now = _sc2.slider(
    "작전 경과 시간 (분)",
    min_value=0.0, max_value=float(total_min),
    value=float(st.session_state["_tl2_t"]),
    step=0.5, format="T+%.1f분",
)
# 슬라이더 직접 조작 시 상태 동기화
st.session_state["_tl2_t"] = _t_now

if _sc3.button("+0.5분 ▶", use_container_width=True):
    st.session_state["_tl2_t"] = min(float(total_min), st.session_state["_tl2_t"] + 0.5)
    st.rerun()

if _sc4.button("⏮ 처음", use_container_width=True):
    st.session_state["_tl2_t"] = 0.0
    st.rerun()

# 현재 프레임 가져오기
_frame = get_frame_at(_tl, _t_now)

st.divider()

# ── 본문: 지도 + 상태 패널 ─────────────────────────────────────────
_map_col, _stat_col = st.columns([3, 2])

with _map_col:
    st.markdown(f"**🗺️ T+{_t_now:.1f}분 전력 배치도**")
    st.plotly_chart(_build_map(_frame, _t_now), use_container_width=True,
                    config={"scrollZoom": True, "displayModeBar": False})

with _stat_col:
    # ── 자산 현황 ──
    st.markdown("**✈️ 자산 현황**")
    _arows = []
    for _aid, _astate in _frame.asset_states.items():
        _ai = _tl.asset_info.get(_aid, {})
        _arows.append({
            "콜사인": _ai.get("callsign", _aid),
            "임무":   _ai.get("mission", "?"),
            "상태":   _PHASE_KO.get(_astate.phase, _astate.phase),
            "고도(m)": f"{_astate.alt:.0f}",
            "속도(km/h)": f"{_ai.get('speed_kmh', 0):.0f}",
        })
    st.dataframe(pd.DataFrame(_arows), hide_index=True, use_container_width=True)

    st.divider()

    # ── 위협 상태 ──
    st.markdown("**⚠️ 위협 상태**")
    _trows = []
    for _tname, _tstat in _frame.threat_statuses.items():
        _tobj = next((t for t in _threats if t.get("name") == _tname), {})
        _trows.append({
            "위협명":   _tname,
            "유형":     _tobj.get("type", "?"),
            "반경(km)": f"{_tobj.get('radius_km', 0):.0f}",
            "상태":     _STATUS_KO[_tstat],
        })
    if _trows:
        def _thr_style(row):
            s = row["상태"]
            if "파괴" in s:
                return ["color: #888888"] * len(row)
            if "억압" in s:
                return ["color: #FFA726; font-weight: bold"] * len(row)
            return ["color: #EF5350; font-weight: bold"] * len(row)
        _tdf = pd.DataFrame(_trows)
        st.dataframe(_tdf.style.apply(_thr_style, axis=1),
                     hide_index=True, use_container_width=True)

    st.divider()

    # ── 이벤트 로그 ──
    st.markdown("**📋 이벤트 로그**")
    _past = [e for e in _tl.events if e.time_min <= _t_now]
    if _past:
        _erows = [
            {"T+": f"{e.time_min:.1f}분", "": e.icon, "내용": e.description}
            for e in reversed(_past[-15:])
        ]
        st.dataframe(pd.DataFrame(_erows), hide_index=True, use_container_width=True)
    else:
        st.caption("아직 이벤트 없음 (T+0.0분)")

st.divider()

# ── Gantt 차트 ─────────────────────────────────────────────────────
st.markdown("**📊 자산별 임무 타임라인**")
_gc1, _gc2, _gc3, _gc4 = st.columns(4)
_gc1.markdown("⚫ **대기** (Standby)")
_gc2.markdown("🔵 **침투** (Ingress)")
_gc3.markdown("🔴 **목표상공** (On-Target)")
_gc4.markdown("🟢 **귀환** (Egress/RTB)")
st.plotly_chart(_build_gantt(_t_now), use_container_width=True,
                config={"displayModeBar": False})

# ── 전체 이벤트 목록 (접힘) ────────────────────────────────────────
with st.expander("📑 전체 작전 이벤트 목록"):
    _all_erows = [
        {
            "T+(분)":  f"{e.time_min:.1f}",
            "":        e.icon,
            "이벤트":  e.event_type,
            "자산":    e.callsign,
            "내용":    e.description,
        }
        for e in _tl.events
    ]
    st.dataframe(pd.DataFrame(_all_erows), hide_index=True, use_container_width=True)
