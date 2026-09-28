"""
통합 임무계획 시스템 v13.0 (MUM-T + Rule-based Validator)
- LLM Brain v2.1: UI 실시간 동기화 + 응답 캐싱 + 빠른 모델 자동 선택
- 경로탐색 v2.0: 위협 존재 시 저고도 우회 (RADAR 400m/SAM 200m AGL)
- 자산 색상 고대비 8색 팔레트 (색맹 친화적)
"""
# Windows 콘솔(cp949)에서 모듈의 이모지 print()가 UnicodeEncodeError로 streamlit
# 페이지 자체를 깨뜨리는 회귀 방지. 다른 import보다 먼저 stdout/stderr를 UTF-8로 설정.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        if _stream and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import time
import json
import hashlib

from modules.config import (
    AIRPORTS, CHAT_CONTAINER_HEIGHT,
    AVAILABLE_ALGORITHMS, FORMATION_MAX_TOTAL,
    RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM,
    THREAT_ALT_ENVELOPE, ASSET_PERFORMANCE
)
from modules.mission_state import MissionState, Threat, THREAT_DB
from modules.llm_brain import LLMBrain
from modules.formation_optimizer import FormationOptimizer, formation_allows_downstream
from modules.validator import MissionValidator
from modules.pathfinder import AStarPathfinder
from modules.pathfinder_optimized import AStarPathfinder3DOptimized, TerrainCacheFast
from modules.pathfinder_rrt import RRTPathfinder, RRTStarPathfinder
from modules.terrain_loader import TerrainLoader
from modules.xai_utils import XAIUtils
from modules.doctrine_policy import DoctrinePolicyEngine
from modules.fuel_model import estimate_effective_range_km, fuel_endurance_factor
from modules.path_computer import compute_formation_paths, compute_single_path, analyze_mission_risk, clear_path_cache
from modules.map_renderer import build_tactical_map
from modules.czml_exporter import export_czml
from modules.cesium_launcher import launch_simulation


# Page config is centralized in run.py because this page is executed via st.navigation.


LLM_GUIDE_QUESTIONS = [
    "현재 자폭UAV 수가 과한가?",
    "SAM 하나를 타격하는 데 필요한 SEAD 자산 수가 적절한가?",
    "현재 편대 구성에서 낭비되는 자산이 있는가?",
    "전투기 위험도를 낮추려면 편대 구성을 어떻게 바꿔야 하나?",
    "현재 임무 순서 ISR → SEAD → STRIKE가 타당한가?",
    "현재 경로에서 가장 위험한 구간은 어디인가?",
]


def _round_or_none(value, ndigits=3):
    try:
        if value is None:
            return None
        return round(float(value), ndigits)
    except Exception:
        return None


def _build_llm_context(mission, path_analysis=None, asset_risk_reports=None,
                       validation_report=None, formation_paths=None):
    """Create compact mission context for judgment-style LLM questions."""
    formation_paths = formation_paths or {}
    context = {
        "threat_count": len(mission.threats),
        "threats": [
            {
                "name": t.name,
                "type": t.type,
                "lat": _round_or_none(t.lat),
                "lon": _round_or_none(t.lon),
                "radius_km": _round_or_none(t.radius_km, 1),
            }
            for t in mission.threats[:8]
        ],
    }

    if mission.formation:
        fr = mission.formation
        context["formation"] = {
            "is_feasible": bool(fr.is_feasible),
            "solver_status": fr.solver_status,
            "mission_sequence": list(fr.mission_sequence or []),
            "n_fighter": int(fr.n_fighter),
            "n_recon_uav": int(fr.n_recon_uav),
            "n_attack_uav": int(fr.n_attack_uav),
            "total_assets": int(fr.total_assets()),
            "utilization_pct": int(getattr(fr, "utilization_pct", 0) or 0),
            "total_cost": _round_or_none(getattr(fr, "total_cost", 0), 1),
        }
        assets = []
        for a in fr.assets[:12]:
            fp_info = formation_paths.get(a.asset_id, {}) if isinstance(formation_paths, dict) else {}
            assets.append({
                "asset_id": a.asset_id,
                "callsign": a.callsign,
                "type": a.asset_type,
                "mission": a.assigned_mission,
                "target_idx": a.assigned_target_idx,
                "objective": fp_info.get("objective"),
                "expendable": fp_info.get("expendable"),
                "asset_margin_km": _round_or_none(fp_info.get("asset_margin"), 1),
            })
        context["assets"] = assets
        context["sead_attack_uav_count"] = sum(
            1 for a in fr.assets
            if a.asset_type == "attack_uav" and (a.assigned_mission or "").upper() == "SEAD"
        )

    if isinstance(path_analysis, dict):
        context["path_analysis"] = {
            "max_risk": _round_or_none(path_analysis.get("max_risk"), 3),
            "avg_risk": _round_or_none(path_analysis.get("avg_risk"), 3),
            "total_distance_km": _round_or_none(path_analysis.get("total_distance_km"), 1),
            "waypoint_count": path_analysis.get("waypoint_count"),
        }

    if asset_risk_reports:
        risk_rows = []
        for row in asset_risk_reports[:12]:
            if not isinstance(row, dict):
                continue
            risk_rows.append({
                "asset_id": row.get("asset_id"),
                "type": row.get("asset_type"),
                "mission": row.get("mission"),
                "objective": row.get("objective"),
                "max_risk": _round_or_none(row.get("max_risk"), 3),
                "avg_risk": _round_or_none(row.get("avg_risk"), 3),
                "distance_km": _round_or_none(row.get("distance_km"), 1),
            })
        context["asset_risk_reports"] = risk_rows

    if validation_report:
        try:
            report_dict = validation_report.to_dict()
            context["validation"] = {
                "is_valid": report_dict.get("is_valid"),
                "error_count": report_dict.get("error_count"),
                "warning_count": report_dict.get("warning_count"),
                "top_issues": report_dict.get("issues", [])[:5],
            }
        except Exception:
            context["validation"] = str(validation_report)

    return context


def _render_llm_question_guide():
    guide = st.popover("❔ LLM 질문 가이드") if hasattr(st, "popover") else st.expander(
        "❔ LLM 질문 가이드", expanded=False
    )
    with guide:
        st.caption("분석 질문은 상태를 바꾸지 않고, 명령형 문장은 설정을 변경합니다.")
        st.markdown("**바로 묻기**")
        for idx, question in enumerate(LLM_GUIDE_QUESTIONS):
            if st.button(question, key=f"_llm_guide_question_{idx}", use_container_width=True):
                st.session_state["_queued_llm_input"] = question
                st.rerun()
        st.markdown("**명령 예시**")
        st.code(
            "안전마진을 15km로 늘려줘\n"
            "A* 3D로 저고도 침투 경로를 다시 계산해줘\n"
            "RTB 해제해줘",
            language="text",
        )
        st.markdown("**시나리오 작성 템플릿**")
        st.code(
            "부산에서 북한 북부로 들어가는 MUM-T 임무 하나 짜줘.\n"
            "목표는 강계 인근 40.97/126.59랑 혜산 인근 41.40/128.17 두 곳으로 잡아줘.\n"
            "정찰 → 방공망 제압 → 타격 순서로 가고, 경로는 A* 3D 저고도 침투로 잡아줘.\n"
            "안전마진은 10km, STPT는 5km 간격, 타격 후 복귀하는 걸로 해줘.\n"
            "방금 기본 시나리오로 깔아둔 SAM/RADAR 방공망을 기준으로,\n"
            "자폭UAV는 SEAD, 정찰UAV는 ISR, 전투기는 STRIKE 중심으로 편대까지 구성해줘.",
            language="text",
        )


def _run_formation_optimization(mission, mission_types, max_assets=8):
    mission_types = [m for m in (mission_types or []) if m in ("ISR", "SEAD", "STRIKE", "CAS")]
    if not mission_types:
        mission_types = ["ISR", "SEAD", "STRIKE"]
    max_assets = max(3, min(FORMATION_MAX_TOTAL, int(max_assets or 8)))

    optimizer = FormationOptimizer()
    primary_target = {
        "lat": mission.params.target_lat,
        "lon": mission.params.target_lon,
        "name": mission.params.target_name,
        "priority": 1,
    }
    all_targets = [primary_target] + sorted(
        mission.params.extra_targets,
        key=lambda t: t.get("priority", 2),
    )
    targets = [
        {"lat": t["lat"], "lon": t["lon"], "name": t.get("name", f"Target-{idx+1}")}
        for idx, t in enumerate(all_targets)
    ]
    threats_dict = [t.to_dict() for t in mission.threats]
    doctrine_policy = st.session_state._doctrine_engine.build_policy(
        mission_types=mission_types,
        threats=threats_dict,
        current_margin_km=mission.params.margin,
    ).to_optimizer_dict()
    effective_sequence = doctrine_policy.get("mission_sequence") or mission_types
    margin_floor = float(doctrine_policy.get("safety_margin_floor_km", 0.0) or 0.0)
    if margin_floor > float(mission.params.margin):
        mission.params.margin = margin_floor
        st.session_state.setdefault("_pending_widget_updates", {})["_mp_margin"] = margin_floor

    f_result = optimizer.run(
        mission_types=effective_sequence,
        targets=targets,
        threats=threats_dict,
        base=mission.params.start,
        max_total=max_assets,
        doctrine_policy=doctrine_policy,
    )
    mission.set_formation(f_result)
    st.session_state["_doctrine_policy"] = doctrine_policy
    st.session_state.mission_sequence = effective_sequence
    st.session_state.pop("_formation_paths", None)
    return f_result, effective_sequence, doctrine_policy


# ===== 군사 다크 테마 CSS =====
st.markdown("""
<style>
/* ── 기본 배경 / 텍스트 ── */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
.stApp {
    background-color: #0D1117 !important;
    color: #C9D1D9 !important;
    font-family: 'Segoe UI', 'Noto Sans KR', sans-serif;
}

/* ── 사이드바 ── */
[data-testid="stSidebar"] {
    background-color: #161B22 !important;
    border-right: 1px solid #30363D;
}

/* ── 메인 컨텐츠 패딩 ── */
[data-testid="stMain"] > div { padding-top: 0.5rem; }

/* ── 헤더 / 제목 ── */
h1, h2, h3, h4 { color: #58A6FF !important; letter-spacing: 0.5px; }
h1 { font-size: 1.6rem !important; border-bottom: 2px solid #1F6FEB; padding-bottom: 6px; }

/* ── 메트릭 카드 ── */
[data-testid="stMetric"] {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 10px 14px;
}
[data-testid="stMetricValue"] { color: #79C0FF !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #8B949E !important; font-size: 11px; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── 버튼 ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1F6FEB 0%, #0D419D 100%);
    color: #fff !important;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: filter 0.15s;
}
[data-testid="stButton"] > button:hover { filter: brightness(1.15); }
[data-testid="stButton"] > button[kind="secondary"] {
    background: #21262D !important;
    border: 1px solid #30363D !important;
    color: #C9D1D9 !important;
}

/* ── 기본 (Primary) 버튼 ── */
button[data-baseweb="button"][kind="primary"],
[data-testid="stButton"] > button[data-testid*="primary"] {
    background: linear-gradient(135deg, #238636 0%, #196127 100%) !important;
    box-shadow: 0 0 8px rgba(35,134,54,0.4);
}

/* ── 입력 필드 ── */
input, textarea, select,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    background-color: #0D1117 !important;
    color: #C9D1D9 !important;
    border: 1px solid #30363D !important;
    border-radius: 6px;
}
input:focus, textarea:focus {
    border-color: #1F6FEB !important;
    box-shadow: 0 0 0 3px rgba(31,111,235,0.2) !important;
}

/* ── 슬라이더 ── */
[data-baseweb="slider"] [role="slider"] { background-color: #1F6FEB !important; }
[data-baseweb="slider"] [data-testid="stSlider"] div[class*="track"] { background: #1F6FEB !important; }

/* ── 셀렉트박스 / 멀티셀렉트 ── */
[data-baseweb="select"] > div:first-child {
    background-color: #161B22 !important;
    border-color: #30363D !important;
    color: #C9D1D9 !important;
}
[data-baseweb="popover"] { background: #161B22 !important; border: 1px solid #30363D !important; }
[data-baseweb="menu"] { background: #161B22 !important; }
[data-baseweb="option"]:hover { background: #1F6FEB22 !important; }

/* ── 탭 ── */
[data-baseweb="tab-list"] { border-bottom: 2px solid #30363D; }
[data-baseweb="tab"] {
    color: #8B949E !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 10px 16px !important;
    min-width: 80px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #58A6FF !important;
    border-bottom: 2px solid #58A6FF !important;
    font-size: 14px !important;
}
/* 탭 내 버튼 텍스트 크기 */
[data-baseweb="tab"] p, [data-baseweb="tab"] span {
    font-size: 14px !important;
    font-weight: 700 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] summary {
    background: #161B22 !important;
    border: 1px solid #30363D !important;
    border-radius: 6px;
    color: #58A6FF !important;
    font-weight: 600;
}
[data-testid="stExpander"] > div > div {
    background: #0D1117 !important;
    border: 1px solid #21262D;
    border-top: none;
    border-radius: 0 0 6px 6px;
}

/* ── DataFrame / 테이블 ── */
[data-testid="stDataFrame"] { border: 1px solid #30363D; border-radius: 8px; }
.stDataFrame thead tr th { background: #161B22 !important; color: #58A6FF !important; }
.stDataFrame tbody tr:nth-child(even) td { background: #0D1117 !important; }
.stDataFrame tbody tr:nth-child(odd)  td { background: #161B22 !important; }
.stDataFrame tbody tr:hover td { background: #1F6FEB22 !important; }

/* ── Info / Warning / Error 박스 ── */
[data-testid="stAlert"] {
    border-radius: 8px;
    border-left-width: 4px;
}

/* ── Divider ── */
hr { border-color: #30363D !important; }

/* ── Caption / Small text ── */
[data-testid="stCaptionContainer"] { color: #8B949E !important; }

/* ── 상단 타이틀 배너 ── */
[data-testid="stHeader"] { background: transparent !important; }

/* ── 채팅 메시지 ── */
[data-testid="stChatMessage"] {
    background: #161B22 !important;
    border: 1px solid #21262D;
    border-radius: 10px;
    margin-bottom: 6px;
}

/* ── 체크박스 ── */
[data-baseweb="checkbox"] svg { fill: #1F6FEB !important; }

/* ── 라디오 ── */
[data-baseweb="radio"] svg { fill: #1F6FEB !important; }

/* ── 스피너 ── */
[data-testid="stSpinner"] { color: #58A6FF !important; }

/* ── 컬럼 구분선 ── */
[data-testid="column"] { padding: 0 6px; }

/* ── 스크롤바 ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #58A6FF; }
</style>
""", unsafe_allow_html=True)

st.title("🚁 통합 임무계획 시스템 v13.0 (MUM-T + Validator)")

# ===== 상태 초기화 =====
if "mission" not in st.session_state:
    st.session_state.mission = MissionState()

if "terrain" not in st.session_state:
    with st.spinner("🌍 지형 데이터 메모리 적재 중... (최초 1회)"):
        base_loader = TerrainLoader()
        st.session_state.terrain = TerrainCacheFast(base_loader)

if "mission_sequence" not in st.session_state:
    st.session_state.mission_sequence = []

if "_doctrine_engine" not in st.session_state:
    st.session_state._doctrine_engine = DoctrinePolicyEngine()

mission = st.session_state.mission
terrain_fast = st.session_state.terrain 

if "_llm_cache_enabled" not in st.session_state:
    st.session_state["_llm_cache_enabled"] = True
LLMBrain.set_cache_enabled(st.session_state["_llm_cache_enabled"])

# The current UI selects one departure base for the whole package.
# Keep existing formation assets aligned with that global base so route recalculation
# actually changes when the operator updates the departure field.
if mission.formation and getattr(mission.formation, "assets", None):
    for _asset in mission.formation.assets:
        _asset.base = mission.params.start

# 호환성 매핑
if not hasattr(terrain_fast, 'get_elevation'):
    terrain_fast.get_elevation = terrain_fast.get


def _queue_profile_widget_sync():
    p = mission.params
    pending = {
        "_mp_start": p.start,
        "_mp_target_lat": float(p.target_lat),
        "_mp_target_lon": float(p.target_lon),
        "_mp_rtb": bool(p.rtb),
        "_mp_algorithm": p.algorithm,
        "_mp_enable_3d": bool(p.enable_3d),
        "_mp_margin": float(p.margin),
        "_mp_stpt_gap": int(p.stpt_gap),
        "_mp_fuel_state": float(getattr(p, "fuel_state", 1.0)),
        "_mp_refuel_count": int(getattr(p, "refuel_count", 0)),
        "_tgt1_name": p.target_name,
    }
    for i in range(2):
        idx = i + 2
        target = p.extra_targets[i] if i < len(p.extra_targets) else None
        pending[f"_tgt{idx}_en"] = bool(target)
        if target:
            pending[f"_tgt{idx}_lat"] = float(target.get("lat", 38.5))
            pending[f"_tgt{idx}_lon"] = float(target.get("lon", 126.0))
            pending[f"_tgt{idx}_name"] = str(target.get("name", f"Target-{idx}"))
            pending[f"_tgt{idx}_prio"] = int(target.get("priority", 2))
    st.session_state["_pending_widget_updates"] = pending


def _build_demo_threat(name: str, threat_type: str, lat: float, lon: float, radius_km: float) -> Threat:
    env = THREAT_ALT_ENVELOPE.get(threat_type, {})
    try:
        ground_elev = terrain_fast.get_elevation(lat, lon)
    except Exception:
        ground_elev = 0.0
    sam_peak = radius_km * 0.35 if threat_type == "SAM" else 0.35
    sam_sigma = radius_km * 0.20 if threat_type == "SAM" else 0.20
    return Threat(
        name=name,
        type=threat_type,
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        alt=ground_elev + 10.0,
        min_alt_m=env.get("min_alt_m"),
        max_alt_m=env.get("max_alt_m"),
        pk_peak_km=sam_peak,
        pk_sigma_km=sam_sigma,
    )


def _find_base_name(keyword: str, fallback: str = None) -> str:
    keyword = (keyword or "").lower()
    for base_name in AIRPORTS.keys():
        if keyword in base_name.lower():
            return base_name
    return fallback or list(AIRPORTS.keys())[0]




def _apply_demo_preset(preset_name: str):
    p = mission.params
    p.fuel_state = 1.0
    p.refuel_count = 0
    busan_base = _find_base_name("busan")
    suwon_base = _find_base_name("suwon", fallback=busan_base)
    osan_base = _find_base_name("osan", fallback=busan_base)
    st.session_state.pop("_threats_only_preview", None)

    if preset_name == "baseline":
        p.start = busan_base
        p.target_lat, p.target_lon = 40.9700, 126.5900
        p.target_name = "TGT-A Kanggye C2"
        p.extra_targets = [
            {
                "lat": 41.4000,
                "lon": 128.1700,
                "name": "TGT-B Hyesan Logistics",
                "priority": 2,
            }
        ]
        p.margin = 10.0
        p.algorithm = "A* 3D"
        p.enable_3d = True
        p.rtb = True
        p.stpt_gap = 5
        p.fuel_state = 0.80
        p.refuel_count = 0
        mission.threats = [
            _build_demo_threat("NK-RADAR-West-Screen", "RADAR", 39.5000, 126.0500, 58.0),
            _build_demo_threat("NK-RADAR-East-Screen", "RADAR", 39.9500, 127.9500, 60.0),
            _build_demo_threat("NK-SAM-Central-Belt", "SAM", 39.3500, 126.8000, 22.0),
            _build_demo_threat("NK-SAM-North-Guard", "SAM", 40.6500, 127.5000, 22.0),
        ]
        st.session_state["_threats_only_preview"] = True
    elif preset_name == "high_risk":
        p.start = suwon_base
        p.target_lat, p.target_lon = 40.5, 125.1
        p.target_name = "High-Risk-Test"
        p.extra_targets = []
        p.margin = 8.0
        p.algorithm = "A* 3D"
        p.enable_3d = True
        p.rtb = True
        p.stpt_gap = 4
        p.fuel_state = 0.45
        p.refuel_count = 0
        mission.threats = [
            _build_demo_threat("Threat-RADAR-A", "RADAR", 37.9, 126.8, 120.0),
            _build_demo_threat("Threat-RADAR-B", "RADAR", 38.6, 126.9, 110.0),
            _build_demo_threat("Threat-SAM-A", "SAM", 37.6, 126.9, 35.0),
            _build_demo_threat("Threat-SAM-B", "SAM", 38.2, 126.7, 35.0),
        ]
    elif preset_name == "detour":
        p.start = osan_base
        p.target_lat, p.target_lon = 39.2, 125.8
        p.target_name = "Detour-Test"
        p.extra_targets = []
        p.margin = 12.0
        p.algorithm = "A* 3D"
        p.enable_3d = True
        p.rtb = True
        p.stpt_gap = 5
        p.fuel_state = 0.95
        p.refuel_count = 1
        mission.threats = [
            _build_demo_threat("Threat-RADAR-Center", "RADAR", 37.9, 126.7, 140.0),
            _build_demo_threat("Threat-SAM-Center", "SAM", 37.8, 126.8, 45.0),
        ]

    mission.formation = None
    mission.chat_history = [{"role": "assistant", "content": "작전관님, 명령을 대기 중입니다.", "reasoning": ""}]
    st.session_state["mission_sequence"] = []
    st.session_state.pop("_formation_paths", None)
    st.session_state.pop("final_in", None)
    st.session_state.pop("final_out", None)
    st.session_state.pop("_validation_report", None)
    st.session_state.pop("_doctrine_policy", None)
    _queue_profile_widget_sync()

# ===== 레이아웃 =====
col_left, col_right = st.columns([1, 2])

with col_left:
    tab_ops, tab_intel, tab_formation, tab_validator, tab_xai, tab_debug = st.tabs([
        "💬 작전 통제", "⚠️ 위협 관리", "✈️ 편대 구성", "✅ 임무 검증",
        "🤔 AI 판단 근거", "🔧 디버그"
    ])

    # 1. 작전 통제
    with tab_ops:
        # ── 빠른 임무 프리셋 ──────────────────────────────────────────
        with st.expander("⚡ 빠른 임무 프리셋", expanded=False):
            st.caption("선택하면 마진·연료·알고리즘 등 주요 파라미터를 일괄 설정합니다.")
            _MISSION_PRESETS = {
                "🔴 고위험 돌파": {
                    "desc": "적 방공망 직선 돌파 — 속도 우선, 마진 최소",
                    "_mp_margin": 3.0, "_mp_fuel_state": 1.0, "_mp_refuel_count": 0,
                    "_mp_algorithm": "A* 3D", "_mp_enable_3d": True, "_mp_rtb": True,
                },
                "🟡 저위험 우회": {
                    "desc": "위협 외곽 대우회 — 안전 마진 최대, 연료 여유 필요",
                    "_mp_margin": 25.0, "_mp_fuel_state": 1.0, "_mp_refuel_count": 1,
                    "_mp_algorithm": "A*", "_mp_enable_3d": False, "_mp_rtb": True,
                },
                "🟢 스텔스 침투": {
                    "desc": "지형 밀착 저고도 침투 — 3D 지형 고려, 레이더 음영 활용",
                    "_mp_margin": 15.0, "_mp_fuel_state": 0.85, "_mp_refuel_count": 0,
                    "_mp_algorithm": "A* 3D", "_mp_enable_3d": True, "_mp_rtb": False,
                },
            }
            _pr_cols = st.columns(3)
            for _pi, (_pname, _pdata) in enumerate(_MISSION_PRESETS.items()):
                with _pr_cols[_pi]:
                    st.markdown(f"**{_pname}**")
                    st.caption(_pdata["desc"])
                    if st.button("적용", key=f"_mp_preset_{_pi}", use_container_width=True):
                        _upd = {k: v for k, v in _pdata.items() if k != "desc"}
                        st.session_state.update(_upd)
                        clear_path_cache()
                        st.rerun()
        st.divider()

        with st.expander("⚙️ 미션 프로파일 설정", expanded=True):
            p = mission.params

            # ── session_state 키 강제 동기화 (LLM 업데이트 반영) ──
            # LLM이 mission.params를 바꾼 뒤 st.rerun() 하면
            # 아래 session_state 키들이 이미 새 값으로 설정되어 위젯에 즉시 반영됨
            # ── session_state 초기값 설정 (최초 1회만, 이미 있으면 LLM 값 유지) ──
            # key= 위젯은 session_state 값을 그대로 표시하므로
            # 처음 렌더링 전에만 params 값으로 초기화하고, 이후는 건드리지 않음
            # LLM 업데이트 시에는 _set() 함수로 session_state를 직접 덮어씀
            _defaults = {
                "_mp_start":      p.start,
                "_mp_target_lat": p.target_lat,
                "_mp_target_lon": p.target_lon,
                "_mp_rtb":        p.rtb,
                "_mp_algorithm":  p.algorithm,
                "_mp_enable_3d":  p.enable_3d,
                "_mp_margin":     float(p.margin),
                "_mp_stpt_gap":   int(p.stpt_gap),
                "_mp_fuel_state": float(getattr(p, "fuel_state", 1.0)),
                "_mp_refuel_count": int(getattr(p, "refuel_count", 0)),
            }
            for _k, _v in _defaults.items():
                if _k not in st.session_state:
                    st.session_state[_k] = _v  # 최초 1회만 설정

            # 위젯 키는 생성 이후 직접 수정할 수 없으므로,
            # 이전 run에서 예약한 업데이트를 위젯 생성 전에 반영한다.
            _pending_widget_updates = st.session_state.pop("_pending_widget_updates", {})
            for _k, _v in _pending_widget_updates.items():
                st.session_state[_k] = _v

            # ── 위젯: key= 만 사용, index=/value= 제거 → session_state 값이 곧 표시값 ──
            p.start = st.selectbox("출발 기지", list(AIRPORTS.keys()), key="_mp_start")
            c1, c2 = st.columns(2)
            p.target_lat = c1.number_input("🎯 목표1 위도(Lat)", 33.0, 43.0, step=0.0001, format="%.4f", key="_mp_target_lat")
            p.target_lon = c2.number_input("🎯 목표1 경도(Lon)", 124.0, 132.0, step=0.0001, format="%.4f", key="_mp_target_lon")
            c_rtb, c_algo = st.columns([1, 1])
            p.rtb       = c_rtb.checkbox("Strike & RTB", key="_mp_rtb")
            p.algorithm = c_algo.selectbox("알고리즘", AVAILABLE_ALGORITHMS, key="_mp_algorithm")
            p.enable_3d = st.checkbox("3D 모드 (지형 고려)", key="_mp_enable_3d")
            p.margin    = st.slider("안전 마진(km)", 0.0, 50.0, key="_mp_margin")
            p.stpt_gap  = st.slider("STPT 표시 간격 (km)", 1, 100, key="_mp_stpt_gap",
                                     help="지도·테이블에 표시할 Steer Point 간격 (km 단위)")
            f1, f2 = st.columns([2, 1])
            p.fuel_state = f1.slider(
                "F-16 baseline fuel state",
                min_value=0.20,
                max_value=1.00,
                step=0.05,
                key="_mp_fuel_state",
                help="1.00=full planned fuel, 0.20=fuel-critical state",
            )
            p.refuel_count = f2.selectbox("AR count", [0, 1, 2], key="_mp_refuel_count")
            f16_base_range = float(ASSET_PERFORMANCE.get("fighter", {}).get("range_km", 3000.0))
            est_range = estimate_effective_range_km(f16_base_range, p.fuel_state, p.refuel_count)
            endu_factor = fuel_endurance_factor(p.fuel_state, p.refuel_count)
            st.caption(f"F-16 effective range budget: {est_range:.0f} km (endurance x{endu_factor:.2f})")
            if p.enable_3d and p.algorithm in ("RRT", "RRT*"):
                st.caption("⚠️ RRT/RRT*는 현재 3D 완전지원이 아니어서 계산 시 A* 3D로 자동 대체됩니다.")

            if st.button("🔄 설정 적용 및 경로 계산", type="primary", use_container_width=True):
                st.session_state.pop("_threats_only_preview", None)
                clear_path_cache()
                st.rerun()

        # ── 목표 관리 (다중 목표 + 우선순위) ──────────────────────────
        with st.expander("🎯 목표 관리 (최대 3개)", expanded=False):
            _PRIO = {1: "🔴 High", 2: "🟡 Medium", 3: "🟢 Low"}

            # ── 주 목표 (Primary) ── 좌표는 위 미션 프로파일에서 설정
            st.markdown("**🎯 목표 1 (Primary)**")
            _nc1, _nc2, _nc3 = st.columns([2, 2, 1])
            _nc1.metric("위도", f"{p.target_lat:.4f}")
            _nc2.metric("경도", f"{p.target_lon:.4f}")
            _nc3.metric("우선순위", "🔴 High")
            if "_tgt1_name" not in st.session_state:
                st.session_state["_tgt1_name"] = p.target_name
            p.target_name = st.text_input("목표명", key="_tgt1_name")
            st.caption("↑ 위도/경도는 '미션 프로파일 설정'의 Lat/Lon과 동일합니다.")
            st.divider()

            # ── 추가 목표 2, 3 ──
            extra_ui: list = []
            for _i in range(2):
                _idx = _i + 2
                _prev = p.extra_targets[_i] if _i < len(p.extra_targets) else {}
                st.markdown(f"**목표 {_idx} (Secondary {_i + 1})**")
                _en_key = f"_tgt{_idx}_en"
                _en = st.checkbox(
                    f"목표 {_idx} 활성화",
                    value=bool(_prev),
                    key=_en_key,
                )
                if _en:
                    _ec1, _ec2 = st.columns(2)
                    if f"_tgt{_idx}_lat" not in st.session_state:
                        st.session_state[f"_tgt{_idx}_lat"] = float(_prev.get("lat", 38.5))
                    if f"_tgt{_idx}_lon" not in st.session_state:
                        st.session_state[f"_tgt{_idx}_lon"] = float(_prev.get("lon", 126.0))
                    if f"_tgt{_idx}_name" not in st.session_state:
                        st.session_state[f"_tgt{_idx}_name"] = str(_prev.get("name", f"Target-{_idx}"))
                    _lat = _ec1.number_input(
                        "위도", 33.0, 43.0, step=0.0001, format="%.4f",
                        key=f"_tgt{_idx}_lat",
                    )
                    _lon = _ec2.number_input(
                        "경도", 124.0, 132.0, step=0.0001, format="%.4f",
                        key=f"_tgt{_idx}_lon",
                    )
                    _name = st.text_input("목표명", key=f"_tgt{_idx}_name")
                    _prio = st.selectbox(
                        "우선순위",
                        options=[1, 2, 3],
                        index=int(_prev.get("priority", 2)) - 1,
                        format_func=lambda x: _PRIO[x],
                        key=f"_tgt{_idx}_prio",
                    )
                    extra_ui.append({"lat": _lat, "lon": _lon, "name": _name, "priority": _prio})
                if _i == 0:
                    st.divider()
            p.extra_targets = extra_ui

        st.divider()
        chat_container = st.container(height=CHAT_CONTAINER_HEIGHT)
        for msg in mission.chat_history:
            with chat_container.chat_message(msg["role"]):
                st.write(msg["content"])

        _render_llm_question_guide()
        queued_input = st.session_state.pop("_queued_llm_input", None)
        typed_input = st.chat_input("명령/질문 입력 (예: 현재 자폭UAV 수가 과한가?)")
        user_input = queued_input or typed_input

        if user_input:
            st.session_state.pop("_threats_only_preview", None)
            mission.add_chat_message("user", user_input)
            with st.spinner("🧠 AI 분석 중..."):
                brain = LLMBrain()
                path_analysis = st.session_state.get("_last_path_analysis")
                llm_state = mission.params.to_dict()
                llm_state["_llm_context"] = _build_llm_context(
                    mission,
                    path_analysis=path_analysis,
                    asset_risk_reports=st.session_state.get("_last_asset_risk_reports"),
                    validation_report=st.session_state.get("_validation_report"),
                    formation_paths=st.session_state.get("_formation_paths"),
                )
                threat_sig = hashlib.md5(
                    json.dumps(
                        [t.to_dict() for t in mission.threats],
                        ensure_ascii=False,
                        sort_keys=True
                    ).encode("utf-8")
                ).hexdigest()
                result = brain.parse_tactical_command(
                    user_input,
                    llm_state,
                    path_analysis,
                    chat_history=mission.chat_history,
                    threat_signature=threat_sig,
                    current_threats=[t.to_dict() for t in mission.threats],
                )

                action = result.get("action", "CHAT")
                u = result.get("update_params", {})
                formation_change_note = None
                before_params = mission.params.to_dict().copy()
                before_threat_count = len(mission.threats)
                before_threat_signature = hashlib.md5(
                    json.dumps(
                        [t.to_dict() for t in mission.threats],
                        ensure_ascii=False,
                        sort_keys=True
                    ).encode("utf-8")
                ).hexdigest()

                # ── UPDATE / THREAT_ADD 공통: 파라미터 변경 ──
                # mission.params 와 session_state 위젯 키를 동시에 갱신해야
                # st.rerun() 후 사이드바 위젯에 즉시 반영됨
                def _sanitize_widget_value(ss_key, val):
                    if val is None:
                        return None
                    try:
                        if ss_key == "_mp_margin":
                            return max(0.0, min(50.0, float(val)))
                        if ss_key == "_mp_stpt_gap":
                            return max(1, min(50, int(float(val))))
                        if ss_key == "_mp_fuel_state":
                            return max(0.20, min(1.00, float(val)))
                        if ss_key == "_mp_refuel_count":
                            return max(0, min(2, int(float(val))))
                        if ss_key == "_mp_target_lat":
                            return max(33.0, min(43.0, float(val)))
                        if ss_key == "_mp_target_lon":
                            return max(124.0, min(132.0, float(val)))
                        if ss_key in ("_mp_rtb", "_mp_enable_3d"):
                            if isinstance(val, str):
                                norm = val.strip().lower()
                                if norm in ("true", "1", "yes", "y", "on"):
                                    return True
                                if norm in ("false", "0", "no", "n", "off"):
                                    return False
                                return None
                            return bool(val)
                        if ss_key == "_mp_algorithm":
                            return val if val in AVAILABLE_ALGORITHMS else None
                        if ss_key == "_mp_start":
                            return val if val in AIRPORTS else None
                        return val
                    except Exception:
                        return None

                # Rebind with validation so malformed LLM values do not break widgets on rerun.
                def _set(param_attr, ss_key, val):
                    safe_val = _sanitize_widget_value(ss_key, val)
                    if safe_val is None:
                        return
                    setattr(mission.params, param_attr, safe_val)
                    pending = st.session_state.setdefault("_pending_widget_updates", {})
                    pending[ss_key] = safe_val

                if action in ("UPDATE", "THREAT_ADD", "MISSION_PLAN"):
                    if u.get("safety_margin_km") is not None:
                        _set("margin",     "_mp_margin",      u["safety_margin_km"])
                    if u.get("rtb") is not None:
                        _set("rtb",        "_mp_rtb",         u["rtb"])
                    if u.get("stpt_gap") is not None:
                        _set("stpt_gap",   "_mp_stpt_gap",    u["stpt_gap"])
                    if u.get("fuel_state") is not None:
                        _set("fuel_state", "_mp_fuel_state", u["fuel_state"])
                    if u.get("refuel_count") is not None:
                        _set("refuel_count", "_mp_refuel_count", u["refuel_count"])
                    if u.get("algorithm"):
                        _set("algorithm",  "_mp_algorithm",   u["algorithm"])
                    if u.get("enable_3d") is not None:
                        _set("enable_3d",  "_mp_enable_3d",   u["enable_3d"])
                    if u.get("target_lat") is not None:
                        _set("target_lat", "_mp_target_lat",  u["target_lat"])
                    if u.get("target_lon") is not None:
                        _set("target_lon", "_mp_target_lon",  u["target_lon"])
                    if u.get("target_name"):
                        mission.params.target_name = u["target_name"]
                        st.session_state.setdefault("_pending_widget_updates", {})["_tgt1_name"] = u["target_name"]
                    if u.get("extra_targets") is not None:
                        safe_targets = []
                        for idx, et in enumerate(u.get("extra_targets") or []):
                            try:
                                lat = max(33.0, min(43.0, float(et.get("lat"))))
                                lon = max(124.0, min(132.0, float(et.get("lon"))))
                                name = str(et.get("name") or f"Target-{idx + 2}").strip()[:80]
                                priority = max(1, min(3, int(et.get("priority", idx + 2))))
                                safe_targets.append({
                                    "lat": lat,
                                    "lon": lon,
                                    "name": name,
                                    "priority": priority,
                                })
                            except Exception:
                                continue
                            if len(safe_targets) >= 2:
                                break
                        mission.params.extra_targets = safe_targets
                        pending = st.session_state.setdefault("_pending_widget_updates", {})
                        for i in range(2):
                            t_idx = i + 2
                            target = safe_targets[i] if i < len(safe_targets) else None
                            pending[f"_tgt{t_idx}_en"] = bool(target)
                            if target:
                                pending[f"_tgt{t_idx}_lat"] = target["lat"]
                                pending[f"_tgt{t_idx}_lon"] = target["lon"]
                                pending[f"_tgt{t_idx}_name"] = target["name"]
                                pending[f"_tgt{t_idx}_prio"] = target["priority"]
                    if u.get("start") and u["start"] in AIRPORTS:
                        _set("start",      "_mp_start",       u["start"])
                    if u.get("waypoint_name"):
                        mission.params.waypoint = u["waypoint_name"]

                # ── THREAT_ADD: 위협 자동 추가 ──
                if action == "THREAT_ADD":
                    if result.get("clear_existing_threats"):
                        mission.threats = []
                    for t_info in result.get("threats_to_add", []):
                        try:
                            t_lat = t_info.get("lat")
                            t_lon = t_info.get("lon")
                            if t_lat is None or t_lon is None:
                                continue  # 좌표 없는 위협 무시
                            new_threat = Threat(
                                name=t_info.get("name", f"Threat-{len(mission.threats)+1:02d}"),
                                type=t_info.get("type", "SAM"),
                                lat=t_lat,
                                lon=t_lon,
                                radius_km=t_info.get("radius_km", 30.0),
                                alt=0.0
                            )
                            if new_threat.type in ("SAM", "RADAR"):
                                env = THREAT_ALT_ENVELOPE.get(new_threat.type, {})
                                new_threat.min_alt_m = env.get("min_alt_m")
                                new_threat.max_alt_m = env.get("max_alt_m")
                            # 지형 고도 자동 보정
                            if new_threat.lat is not None and new_threat.lon is not None:
                                ground_elev = terrain_fast.get(new_threat.lat, new_threat.lon)
                                new_threat.alt = ground_elev + 10.0
                            mission.add_threat(new_threat)
                        except Exception:
                            pass

                # ── 임무 순서 표시 / 자연어 시나리오 편대 자동 구성 ──
                seq = result.get("mission_sequence", [])
                if seq:
                    st.session_state["mission_sequence"] = seq

                if result.get("_auto_run_formation"):
                    try:
                        max_assets_auto = result.get("_auto_formation_max_assets", 8)
                        f_result, effective_sequence, _ = _run_formation_optimization(
                            mission,
                            seq or st.session_state.get("mission_sequence", []),
                            max_assets=max_assets_auto,
                        )
                        if f_result.is_feasible:
                            formation_change_note = (
                                f"- 편대 구성: {f_result.summary()} "
                                f"({f_result.solver_status}, 순서 {' → '.join(effective_sequence)})"
                            )
                        else:
                            formation_change_note = f"- 편대 구성 실패: {f_result.solver_status or 'Infeasible'}"
                    except Exception as exc:
                        formation_change_note = f"- 편대 구성 실패: {exc}"

                ai_msg = result["response_text"]

                # 적용된 실제 변경값(diff) 표시
                after_params = mission.params.to_dict().copy()
                field_labels = {
                    "start": "출발기지",
                    "target_lat": "목표위도",
                    "target_lon": "목표경도",
                    "margin": "안전마진(km)",
                    "rtb": "RTB",
                    "stpt_gap": "STPT 간격",
                    "algorithm": "알고리즘",
                    "enable_3d": "3D모드",
                    "target_name": "목표명",
                    "extra_targets": "추가목표",
                    "fuel_state": "연료 상태",
                    "refuel_count": "공중급유 횟수",
                }
                applied_changes = []
                for key, label in field_labels.items():
                    b = before_params.get(key)
                    a = after_params.get(key)
                    if b != a:
                        applied_changes.append(f"- {label}: {b} → {a}")

                if len(mission.threats) != before_threat_count:
                    delta = len(mission.threats) - before_threat_count
                    applied_changes.append(f"- 위협 수: {before_threat_count} → {len(mission.threats)} (Δ {delta:+d})")
                else:
                    after_threat_signature = hashlib.md5(
                        json.dumps(
                            [t.to_dict() for t in mission.threats],
                            ensure_ascii=False,
                            sort_keys=True
                        ).encode("utf-8")
                    ).hexdigest()
                    if before_threat_signature != after_threat_signature:
                        applied_changes.append("- 위협 배치/속성 업데이트됨")
                if formation_change_note:
                    applied_changes.append(formation_change_note)

                if applied_changes:
                    ai_msg += "\n\n**적용 변경값**\n" + "\n".join(applied_changes)
                elif action in ("UPDATE", "THREAT_ADD", "MISSION_PLAN"):
                    ai_msg += "\n\n**적용 변경값**\n- 변경 없음 (동일값 또는 범위 보정으로 무시됨)"

                # RAG 출처 표시
                doctrine_refs = result.get("_doctrine_refs", []) or []
                if doctrine_refs:
                    ref_lines = []
                    for r in doctrine_refs[:4]:
                        src = r.get("source", "?")
                        page = r.get("page", 0)
                        score = r.get("score", 0.0)
                        ref_lines.append(f"- {src} p.{page} (score={score:.3f})")
                    ai_msg += "\n\n**교리 근거(RAG)**\n" + "\n".join(ref_lines)

                # 모델 정보 및 신뢰도 표시
                model_used = result.get("_model_used", "unknown")
                confidence = result.get("confidence", 0.0)
                cache_hit = result.get("_cache_hit", False)
                cache_tag = "cache-hit" if cache_hit else "live"
                ai_msg += f"\n\n`[{model_used} | {cache_tag} | 신뢰도: {confidence:.0%} | 액션: {action}]`"

                mission.add_chat_message("assistant", ai_msg, result.get("reasoning", ""))
                if action in ("UPDATE", "THREAT_ADD", "MISSION_PLAN") or result.get("_auto_run_formation"):
                    clear_path_cache()
                st.rerun()

    # 2. 위협 관리 (v2.1: 군사 DB 선택 + 수동 설정 통합)
    with tab_intel:
        st.subheader("위협 추가")

        # ── 라디오: form 바깥에 위치 (클릭 즉시 반응)
        add_type = st.radio(
            '위협 입력 방식',
            ["군사 DB (SAM/RADAR)", "수동 설정 (SAM/RADAR)", "수동 설정 (NFZ)"],
            horizontal=True
        )

        with st.form("threat_form"):
            # ── 방식 1: 군사 DB 선택 ───────────────────────────────
            if add_type == "군사 DB (SAM/RADAR)":
                selected_db = st.selectbox(
                    "적성 체계 식별명",
                    list(THREAT_DB.keys()),
                    help="F-16 전투기 대응 기준 현실화 파라미터 자동 적용"
                )
                db_preview = THREAT_DB[selected_db]
                st.caption(
                    f"▸ 유형: {db_preview['type']} "
                    f"| 반경: {db_preview['radius_km']} km "
                    f"| SSKP: {db_preview['sskp']:.0%}"
                )
                t_name = st.text_input(
                    "전술 지도 표시 명칭",
                    value=f"{selected_db.split()[0]}-{len(mission.threats)+1:02d}"
                )

            # ── 방식 2: 수동 설정 (SAM/RADAR) ─────────────────────
            elif add_type == "수동 설정 (SAM/RADAR)":
                manual_type = st.radio("위협 유형", ["SAM", "RADAR"], horizontal=True)
                t_name = st.text_input(
                    "전술 지도 표시 명칭",
                    value=f"Threat-{len(mission.threats)+1:02d}"
                )

            # ── 방식 3: NFZ ────────────────────────────────────────
            else:
                t_name = st.text_input(
                    "비행금지구역 명칭",
                    value=f"NFZ-{len(mission.threats)+1:02d}"
                )

            # ── 공통: 좌표 입력 ────────────────────────────────────
            c1, c2 = st.columns(2)
            t_lat = c1.number_input("Lat (위도)",  33.0, 43.0,  38.0, format="%.4f", key="t_lat")
            t_lon = c2.number_input("Lon (경도)", 124.0, 132.0, 127.0, format="%.4f", key="t_lon")

            # ── 수동 설정 반경 슬라이더 ────────────────────────────
            if add_type == "수동 설정 (SAM/RADAR)":
                t_rad = st.slider("반경(km)", 5, 400, 30, key="t_rad")

            # ── NFZ 범위 입력 ──────────────────────────────────────
            if add_type == "수동 설정 (NFZ)":
                l_min  = c1.number_input("Min Lat", 33.0, 43.0,  37.5, format="%.4f")
                l_max  = c2.number_input("Max Lat", 33.0, 43.0,  37.8, format="%.4f")
                ln_min = c1.number_input("Min Lon", 124.0, 132.0, 127.5, format="%.4f")
                ln_max = c2.number_input("Max Lon", 124.0, 132.0, 127.8, format="%.4f")

            if st.form_submit_button("➕ 위협 추가", type="primary"):
                ground_elev = 0.0
                try:
                    ground_elev = terrain_fast.get_elevation(t_lat, t_lon)
                except Exception:
                    pass

                # ── NFZ 추가 ──────────────────────────────────────
                if add_type == "수동 설정 (NFZ)":
                    mission.add_threat(Threat(
                        name=t_name, type="NFZ",
                        lat_min=l_min, lat_max=l_max,
                        lon_min=ln_min, lon_max=ln_max
                    ))

                # ── 군사 DB 추가 (정밀 XAI 파라미터 자동 적용) ────
                elif add_type == "군사 DB (SAM/RADAR)":
                    db_info = THREAT_DB[selected_db]
                    r_km = db_info["radius_km"]
                    env = THREAT_ALT_ENVELOPE.get(db_info["type"], {})
                    env_min = db_info.get("min_alt_m", env.get("min_alt_m"))
                    env_max = db_info.get("max_alt_m", env.get("max_alt_m"))
                    mission.add_threat(Threat(
                        name=t_name,
                        type=db_info["type"],
                        lat=t_lat,
                        lon=t_lon,
                        radius_km=r_km,
                        alt=ground_elev + 10.0,
                        min_alt_m=env_min,
                        max_alt_m=env_max,
                        loss=db_info["loss"],
                        rcs_m2=db_info["rcs_m2"],
                        pd_k=db_info["pd_k"],
                        sskp=db_info["sskp"],
                        pk_peak_km=r_km * db_info["peak_ratio"],
                        pk_sigma_km=r_km * db_info["sigma_ratio"],
                    ))

                # ── 수동 SAM/RADAR 추가 ────────────────────────────
                else:
                    if manual_type == "RADAR":
                        peak, sigma, sskp_val = 0.0, 0.0, 0.0
                    else:
                        peak      = t_rad * 0.35
                        sigma     = t_rad * 0.20
                        sskp_val  = 0.75
                    env = THREAT_ALT_ENVELOPE.get(manual_type, {})
                    mission.add_threat(Threat(
                        name=t_name,
                        type=manual_type,
                        lat=t_lat,
                        lon=t_lon,
                        radius_km=t_rad,
                        alt=ground_elev + 10.0,
                        min_alt_m=env.get("min_alt_m"),
                        max_alt_m=env.get("max_alt_m"),
                        loss=8.0,
                        rcs_m2=2.5,
                        pd_k=0.4,
                        sskp=sskp_val,
                        pk_peak_km=peak,
                        pk_sigma_km=sigma,
                    ))
                clear_path_cache()
                st.rerun()

        st.divider()
        if mission.threats:
            threat_df = pd.DataFrame([t.to_dict() for t in mission.threats])
            cols = [c for c in ['name', 'type', 'radius_km', 'lat', 'lon', 'sskp'] if c in threat_df.columns]
            st.dataframe(threat_df[cols], hide_index=True, use_container_width=True)

            c_del, c_btn = st.columns([3, 1])
            del_name = c_del.selectbox("삭제할 위협", [t.name for t in mission.threats])
            if c_btn.button("🗑️ 삭제"):
                mission.remove_threat(del_name)
                clear_path_cache()
                st.rerun()

    # 3. 편대 구성 (Formation Optimizer)
    with tab_formation:
        st.subheader("✈️ MUM-T 편대 구성 최적화")
        st.caption("MILP + 헝가리안 | 근거: JP3-30, AFDP3-03, FMI3-04.155")

        # ── 현재 설정된 목표 요약 (작전통제 탭에서 설정) ──
        _PRIO_ICON = {1: "🔴", 2: "🟡", 3: "🟢"}
        _tgt_summary = [f"🎯 #1 **{mission.params.target_name}** "
                        f"({mission.params.target_lat:.3f}, {mission.params.target_lon:.3f}) 🔴High"]
        for _si, _st_data in enumerate(mission.params.extra_targets):
            _p = _st_data.get("priority", 2)
            _tgt_summary.append(
                f"🎯 #{_si+2} **{_st_data.get('name','')}** "
                f"({_st_data.get('lat',0):.3f}, {_st_data.get('lon',0):.3f}) "
                f"{_PRIO_ICON.get(_p,'?')}{'High' if _p==1 else 'Medium' if _p==2 else 'Low'}"
            )
        st.info("**적용 목표** (변경은 '작전 통제' 탭 → 🎯 목표 관리)\n\n" +
                "\n\n".join(_tgt_summary))

        with st.form("formation_form"):
            st.markdown("**임무 유형 선택**")
            c1, c2, c3, c4 = st.columns(4)
            use_isr    = c1.checkbox("🔵 ISR",    value=True)
            use_sead   = c2.checkbox("🟡 SEAD",   value=True)
            use_strike = c3.checkbox("🔴 STRIKE", value=True)
            use_cas    = c4.checkbox("🟢 CAS",    value=False)

            max_assets = st.slider("최대 자산 수 (N_max)", 3, FORMATION_MAX_TOTAL, 8)

            run_btn = st.form_submit_button("🚀 편대 구성 최적화 실행", type="primary")

        if run_btn:
            mission_types = []
            if use_isr:    mission_types.append("ISR")
            if use_sead:   mission_types.append("SEAD")
            if use_strike: mission_types.append("STRIKE")
            if use_cas:    mission_types.append("CAS")

            if not mission_types:
                st.warning("임무 유형을 하나 이상 선택하세요.")
            else:
                with st.spinner("🧮 MILP + 헝가리안 최적화 중..."):
                    _run_formation_optimization(
                        mission,
                        mission_types,
                        max_assets=max_assets,
                    )
                st.rerun()

        # 결과 표시
        if mission.formation and mission.formation.is_feasible:
            fr = mission.formation
            utilization = getattr(fr, 'utilization_pct', 0)
            st.success(f"✅ {fr.solver_status} | 계산시간: {fr.solve_time_ms:.1f}ms | 전력활용률: {utilization}%")

            # 편대 구성 요약
            col_f, col_r, col_k, col_t = st.columns(4)
            col_f.metric("✈️ 전투기",   f"{fr.n_fighter}대",    delta=None)
            col_r.metric("🔍 정찰UAV",  f"{fr.n_recon_uav}대",  delta=None)
            col_k.metric("💥 자폭UAV",  f"{fr.n_attack_uav}대", delta=None)
            col_t.metric("📊 총 비용",  f"{fr.total_cost:.0f}", delta=None)

            # 자산별 배정 테이블
            st.divider()
            st.markdown("**자산 배정 결과 (헝가리안)**")
            # 목표 이름 조회용 리스트 (우선순위 순 정렬 유지)
            _all_tgts = [{"name": mission.params.target_name, "priority": 1}] + sorted(
                [{"name": t.get("name", f"Target-{i+2}"), "priority": t.get("priority", 2)}
                 for i, t in enumerate(mission.params.extra_targets)],
                key=lambda x: x["priority"],
            )
            _prio_icon = {1: "🔴", 2: "🟡", 3: "🟢"}
            # 이미 계산된 formation_paths 에서 실제 목적지(objective) 참조
            _fp_cached = st.session_state.get("_formation_paths", {})
            asset_data = []
            type_icon = {"fighter": "✈️", "recon_uav": "🔍", "attack_uav": "💥"}
            mission_icon = {"STRIKE": "🎯", "SEAD": "💥", "ISR": "📡", "CAS": "🔥"}
            for a in fr.assets:
                assigned_m = a.assigned_mission or "-"
                # ── 실제 목적지: formation_paths 캐시 우선, 없으면 idx 기반 ──
                fp_info = _fp_cached.get(a.asset_id, {})
                if fp_info.get("objective"):
                    t_label = fp_info["objective"]
                    # ISR 자산은 정찰 위치 명시
                    if assigned_m == "ISR" or a.asset_type == "recon_uav":
                        t_label = f"📡 정찰 [{t_label} 주변]"
                    elif assigned_m == "SEAD":
                        t_label = f"💥 SEAD → {t_label}"
                    else:
                        t_idx = a.assigned_target_idx if a.assigned_target_idx is not None else 0
                        t_info = _all_tgts[min(t_idx, len(_all_tgts)-1)]
                        t_label = f"{_prio_icon.get(t_info['priority'],'?')} {t_label}"
                else:
                    # 경로 미계산 상태 — idx 기반 fallback
                    t_idx = a.assigned_target_idx if a.assigned_target_idx is not None else 0
                    t_info = _all_tgts[min(t_idx, len(_all_tgts)-1)]
                    if assigned_m == "SEAD":
                        t_label = f"💥 SEAD → (SAM 위치 자동 배정)"
                    elif assigned_m == "ISR" or a.asset_type == "recon_uav":
                        t_label = f"📡 정찰 [{t_info['name']} 주변]"
                    else:
                        t_label = f"{_prio_icon.get(t_info['priority'],'?')} #{t_idx+1} {t_info['name']}"
                asset_data.append({
                    "자산": f"{type_icon.get(a.asset_type,'?')} {a.callsign}",
                    "유형": a.asset_type,
                    "배정임무": f"{mission_icon.get(assigned_m,'')} {assigned_m}",
                    "실제 목적지": t_label,
                    "출발기지": a.base,
                })
            if asset_data:
                st.dataframe(pd.DataFrame(asset_data), hide_index=True, use_container_width=True)

            # MUM-T 비율 표시
            st.divider()
            total_uav = fr.n_recon_uav + fr.n_attack_uav
            if fr.n_fighter > 0:
                mumt_ratio = total_uav / fr.n_fighter
                st.markdown(f"**MUM-T 비율**: 전투기 1 : UAV {mumt_ratio:.1f} "
                            f"{'✅ 교리 충족' if mumt_ratio >= 2.0 else '⚠️ 권장 비율(1:2) 미달'}")
            else:
                st.markdown("**MUM-T 비율**: 전투기 미배정 — UAV 전용 편대 (MUM-T 비율 미적용)")

    # 4. 임무 검증 (Validator)
    doctrine_policy = st.session_state.get("_doctrine_policy")
    if doctrine_policy and mission.formation and mission.formation.is_feasible:
        with tab_formation:
            st.divider()
            st.markdown("**교리 기반 자동 적용 정책**")
            for line in doctrine_policy.get("rationale", [])[:5]:
                st.markdown(f"- {line}")
            refs = doctrine_policy.get("refs", []) or []
            if refs:
                st.caption(
                    "참조 문서: " + ", ".join(
                        f"{r.get('source', '?')} p.{r.get('page', 0)}"
                        for r in refs[:4]
                    )
                )

    with tab_validator:
        st.subheader("✅ 규칙 기반 임무 검증")
        st.caption("근거: AFDP5-0 COA Analysis & Wargaming, JP3-30 ACO, FMI3-04.155, DAFMAN11-260")

        # 검증 실행 버튼
        col_vbtn, col_vinfo = st.columns([1, 2])
        run_validation = col_vbtn.button("🔍 임무 검증 실행", type="primary", use_container_width=True)
        col_vinfo.info("편대 구성 후 검증을 실행하면 교리 기반 7개 항목을 자동 검사합니다.")

        if run_validation:
            with st.spinner("📋 임무 계획 검증 중..."):
                validator = MissionValidator()
                threats_dict = [t.to_dict() for t in mission.threats]
                seq = st.session_state.get("mission_sequence", [])

                # formation_paths가 있으면 경로 포함 검증
                fp = st.session_state.get("_formation_paths", {})

                _v_targets = [{"lat": mission.params.target_lat,
                               "lon": mission.params.target_lon,
                               "name": mission.params.target_name}] + [
                    {"lat": t["lat"], "lon": t["lon"], "name": t.get("name", "")}
                    for t in mission.params.extra_targets if "lat" in t and "lon" in t
                ]
                report = validator.validate(
                    formation_result=mission.formation,
                    formation_paths=fp,
                    threats=threats_dict,
                    mission_sequence=seq,
                    margin_km=mission.params.margin,
                    terrain_loader=terrain_fast,
                    targets=_v_targets,
                )
                st.session_state["_validation_report"] = report

        # 검증 결과 표시
        report = st.session_state.get("_validation_report", None)
        if report:
            # 요약 배너
            if report.is_valid:
                st.success(f"🟢 {report.summary()} | 검증 시간: {report.validate_time_ms:.1f}ms")
            else:
                st.error(f"🔴 {report.summary()} | 검증 시간: {report.validate_time_ms:.1f}ms")

            st.divider()

            # 검증 항목별 결과
            if not report.issues:
                st.balloons()
                st.success("모든 검증 항목 통과! 임무 계획이 교리에 부합합니다.")
            else:
                # 심각도별로 분류
                errors   = [i for i in report.issues if i.severity == "ERROR"]
                warnings = [i for i in report.issues if i.severity == "WARNING"]
                infos    = [i for i in report.issues if i.severity == "INFO"]

                if errors:
                    st.markdown("### 🔴 오류 (ERROR) - 즉시 수정 필요")
                    for issue in errors:
                        with st.expander(f"{issue.icon} [{issue.rule_id}] {issue.message}", expanded=True):
                            col_a, col_b = st.columns(2)
                            col_a.markdown(f"**교리 근거:** {issue.doctrine_ref}")
                            if issue.asset_id:
                                col_b.markdown(f"**관련 자산:** `{issue.asset_id}`")
                            if issue.suggestion:
                                st.warning(f"💡 권고사항: {issue.suggestion}")

                if warnings:
                    st.markdown("### 🟡 경고 (WARNING) - 검토 권장")
                    for issue in warnings:
                        with st.expander(f"{issue.icon} [{issue.rule_id}] {issue.message}"):
                            col_a, col_b = st.columns(2)
                            col_a.markdown(f"**교리 근거:** {issue.doctrine_ref}")
                            if issue.asset_id:
                                col_b.markdown(f"**관련 자산:** `{issue.asset_id}`")
                            if issue.suggestion:
                                st.info(f"💡 권고사항: {issue.suggestion}")

                if infos:
                    st.markdown("### 🔵 참고 (INFO)")
                    for issue in infos:
                        st.info(f"{issue.icon} {issue.message}")

            st.divider()

            # 검증 규칙 목록
            with st.expander("📋 검사된 규칙 목록"):
                rules = {
                    "MISSION_SEQUENCE": "Rule 5 - 임무 순서 (JP3-30 MAAP)",
                    "MUMT_RATIO":       "Rule 7 - MUM-T 비율 (FMI3-04.155)",
                    "NFZ":              "Rule 1 - 비행금지구역 침범 (JP3-30 ACO)",
                    "THREAT":           "Rule 2 - 위협 반경 침범 (JP3-30 ROE)",
                    "ALT":              "Rule 3 - 최소 고도 (FMI3-04.155)",
                    "RANGE":            "Rule 6 - 항속거리 (DAFMAN11-260)",
                    "ASSET_COLLISION":  "Rule 4 - 자산 충돌 (FMI3-04.155)",
                }
                checked_summary = []
                for rule_id in report.checked_rules:
                    prefix = rule_id.split("_")[0]
                    label = rules.get(rule_id) or rules.get(prefix, f"Rule - {rule_id}")
                    checked_summary.append(label)

                # 중복 제거 후 표시
                for label in sorted(set(checked_summary)):
                    st.markdown(f"- ✓ {label}")

            # JSON 내보내기
            with st.expander("📄 검증 보고서 JSON"):
                st.json(report.to_dict())

        else:
            # 미실행 안내
            st.info("편대 구성 탭에서 최적화를 먼저 실행한 후 검증을 수행하세요.")

            # 검증 항목 미리보기
            st.markdown("### 📋 검증 항목 (7개 규칙)")
            rules_preview = [
                ("Rule 1", "🟥", "비행금지구역(NFZ) 침범 검사",   "JP3-30 ACO"),
                ("Rule 2", "🟧", "위협 반경(SAM/RADAR) 침범 검사", "JP3-30 ROE"),
                ("Rule 3", "🟨", "최소 비행고도 검사 (AGL 200m)",  "FMI3-04.155"),
                ("Rule 4", "🟩", "자산 간 충돌 위험 검사",         "FMI3-04.155"),
                ("Rule 5", "🟦", "임무 수행 순서 검사 (ISR→SEAD→STRIKE)", "JP3-30 MAAP"),
                ("Rule 6", "🟪", "항속거리 초과 검사",             "DAFMAN11-260"),
                ("Rule 7", "⬜", "MUM-T 비율 검사 (1:2 이상)",    "FMI3-04.155"),
            ]
            for rule_id, icon, desc, ref in rules_preview:
                st.markdown(f"{icon} **{rule_id}** {desc} _{ref}_")

    # 5. XAI & Debug
    with tab_xai:
        st.subheader("🤔 AI 판단 근거")
        if mission.chat_history:
            last = [m for m in mission.chat_history if m["role"] == "assistant"]
            if last and last[-1].get('reasoning'):
                reasoning = last[-1].get('reasoning', '')
                st.markdown("#### 📋 판단 근거")
                st.info(reasoning)

        st.markdown("#### 🧾 적용된 임무 상태")
        p = mission.params
        target_rows = [
            {
                "구분": "주목표",
                "이름": p.target_name,
                "Lat": round(float(p.target_lat), 4),
                "Lon": round(float(p.target_lon), 4),
                "우선순위": 1,
            }
        ]
        for idx, tgt in enumerate(getattr(p, "extra_targets", []) or [], start=2):
            try:
                target_rows.append({
                    "구분": f"추가목표 {idx - 1}",
                    "이름": tgt.get("name", f"Target-{idx}"),
                    "Lat": round(float(tgt.get("lat")), 4),
                    "Lon": round(float(tgt.get("lon")), 4),
                    "우선순위": int(tgt.get("priority", idx)),
                })
            except Exception:
                continue

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("출발 기지", p.start)
        c2.metric("경로 알고리즘", p.algorithm)
        c3.metric("안전 마진", f"{float(p.margin):.1f}km")
        c4.metric("STPT 간격", f"{int(p.stpt_gap)}km")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("RTB", "ON" if p.rtb else "OFF")
        c6.metric("3D 지형", "ON" if p.enable_3d else "OFF")
        c7.metric("연료 상태", f"{float(p.fuel_state):.2f}")
        c8.metric("공중급유", f"{int(p.refuel_count)}회")

        with st.expander("🎯 목표 / 위협 / 검증 요약", expanded=True):
            st.markdown("**목표 목록**")
            st.dataframe(pd.DataFrame(target_rows), hide_index=True, use_container_width=True)

            threat_rows = [
                {
                    "이름": t.name,
                    "유형": t.type,
                    "Lat": round(float(t.lat), 4) if t.lat is not None else None,
                    "Lon": round(float(t.lon), 4) if t.lon is not None else None,
                    "반경(km)": round(float(t.radius_km), 1) if t.radius_km is not None else None,
                }
                for t in mission.threats
                if t.type in ("SAM", "RADAR")
            ]
            if threat_rows:
                st.markdown("**현재 SAM/RADAR 배치**")
                st.dataframe(pd.DataFrame(threat_rows), hide_index=True, use_container_width=True)
            else:
                st.caption("현재 SAM/RADAR 위협은 배치되어 있지 않습니다.")

            path_analysis = st.session_state.get("_last_path_analysis") or {}
            validation_report = st.session_state.get("_validation_report")
            if path_analysis or validation_report:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("전투기 위험도", f"{float(path_analysis.get('fighter_max_risk', 0.0)):.3f}" if path_analysis else "-")
                m2.metric("패키지 최대 위험", f"{float(path_analysis.get('package_max_risk', 0.0)):.3f}" if path_analysis else "-")
                m3.metric("총 경로거리", f"{float(path_analysis.get('total_distance_km', 0.0)):.0f}km" if path_analysis else "-")
                if validation_report:
                    m4.metric("검증 결과", f"E {validation_report.error_count} / W {validation_report.warning_count}")
                else:
                    m4.metric("검증 결과", "미실행")

        # 임무 순서 표시
        if st.session_state.get("mission_sequence"):
            st.divider()
            st.subheader("📌 임무 수행 순서")
            seq = st.session_state["mission_sequence"]
            cols = st.columns(len(seq))
            colors = {"ISR": "🔵", "SEAD": "🟡", "STRIKE": "🔴", "CAS": "🟢"}
            for i, (col, mission_type) in enumerate(zip(cols, seq)):
                with col:
                    icon = colors.get(mission_type, "⚪")
                    st.metric(f"Step {i+1}", f"{icon} {mission_type}")

        st.divider()
        if "_show_heatmap" not in st.session_state:
            st.session_state["_show_heatmap"] = True
        st.checkbox("위험도 히트맵 (음영 반영)", key="_show_heatmap")

        # ── 위협 기여도 분해 패널 ──────────────────────────────────
        st.divider()
        st.subheader("🎯 위협별 기여도 분석")
        st.caption("각 위협이 현재 경로에 얼마나 영향을 미치는지 분석합니다.")

        _xai_path = st.session_state.get("final_in", [])
        _xai_threats = [t.to_dict() for t in mission.threats]

        if _xai_path and _xai_threats:
            with st.spinner("위협 기여도 계산 중..."):
                contributions = XAIUtils.analyze_threat_contributions(
                    _xai_path,
                    _xai_threats,
                    mission.params.margin,
                    terrain_loader=terrain_fast,
                )

            if contributions:
                import pandas as pd
                contrib_df = pd.DataFrame(contributions).rename(columns={
                    "name": "위협명",
                    "type": "유형",
                    "avg_risk": "평균 기여도",
                    "max_risk": "최대 기여도",
                    "radius_km": "반경(km)",
                })
                # 시각적 강조: 최대 기여도 기준 색 구분
                def _highlight(row):
                    mr = row["최대 기여도"]
                    if mr >= RISK_THRESHOLD_HIGH:
                        return ["background-color: #c0392b; color: #ffffff; font-weight: bold"] * len(row)
                    if mr >= RISK_THRESHOLD_MEDIUM:
                        return ["background-color: #e67e22; color: #ffffff; font-weight: bold"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    contrib_df.style.apply(_highlight, axis=1),
                    hide_index=True,
                    use_container_width=True,
                )

                # 가장 위험한 위협 하이라이트
                top = contributions[0]
                sev_icon = "🔴" if top["max_risk"] >= RISK_THRESHOLD_HIGH else "🟡"
                st.markdown(
                    f"**{sev_icon} 주요 위협:** `{top['name']}` ({top['type']}) — "
                    f"최대 기여도 **{top['max_risk']:.3f}**, 평균 **{top['avg_risk']:.3f}**"
                )
            else:
                st.success("✅ 현재 경로에서 유의미한 위협 기여도가 없습니다.")
        else:
            st.info("경로 계산 후 분석 가능합니다.")


    with tab_debug:
        st.subheader("발표/디버그 도구")
        if st.session_state.get("_demo_preset_error"):
            st.warning(st.session_state.pop("_demo_preset_error"))
        st.json(mission.params.to_dict())

        st.divider()
        st.markdown("**LLM 캐시 제어**")
        st.checkbox("응답 캐시 사용", key="_llm_cache_enabled")
        LLMBrain.set_cache_enabled(st.session_state.get("_llm_cache_enabled", True))
        c_cache1, c_cache2 = st.columns([1, 1])
        c_cache1.metric("캐시 엔트리", f"{LLMBrain.cache_size()}개")
        if c_cache2.button("캐시 비우기"):
            LLMBrain.clear_cache()
            st.rerun()

        st.divider()
        st.markdown("**발표용 시나리오 프리셋**")
        p1, p2, p3, p4 = st.columns(4)
        if p1.button("기본 시나리오"):
            _apply_demo_preset("baseline")
            clear_path_cache()
            st.rerun()
        if p2.button("고위험 시나리오"):
            _apply_demo_preset("high_risk")
            clear_path_cache()
            st.rerun()
        if p3.button("우회 검증"):
            _apply_demo_preset("detour")
            clear_path_cache()
            st.rerun()
        if p4.button("완전 리셋"):
            new_mission = MissionState()
            st.session_state.mission = new_mission
            st.session_state.mission_sequence = []
            st.session_state.pop("_threats_only_preview", None)
            st.session_state.pop("_formation_paths", None)
            st.session_state.pop("final_in", None)
            st.session_state.pop("_validation_report", None)
            st.session_state.pop("_doctrine_policy", None)
            clear_path_cache()
            p_new = new_mission.params
            st.session_state["_pending_widget_updates"] = {
                "_mp_start": p_new.start,
                "_mp_target_lat": float(p_new.target_lat),
                "_mp_target_lon": float(p_new.target_lon),
                "_mp_rtb": bool(p_new.rtb),
                "_mp_algorithm": p_new.algorithm,
                "_mp_enable_3d": bool(p_new.enable_3d),
                "_mp_margin": float(p_new.margin),
                "_mp_stpt_gap": int(p_new.stpt_gap),
                "_mp_fuel_state": float(getattr(p_new, "fuel_state", 1.0)),
                "_mp_refuel_count": int(getattr(p_new, "refuel_count", 0)),
            }
            st.rerun()


# ===== 경로 계산 =====
with col_right:
    effective_algorithm = mission.params.algorithm
    if mission.params.enable_3d and mission.params.algorithm in ("RRT", "RRT*"):
        effective_algorithm = "A* 3D"
        st.warning("3D 모드에서 RRT/RRT*는 완전 지원되지 않아 A* 3D로 대체 계산합니다.")
    st.subheader(f"🗺️ 전술 지도 ({effective_algorithm})")

    # 알고리즘 설정
    pathfinder = None
    if effective_algorithm == "A* 3D":
        pathfinder = AStarPathfinder3DOptimized(terrain_fast)
    elif effective_algorithm == "A*":
        pathfinder = AStarPathfinder()
    elif effective_algorithm == "RRT":
        pathfinder = RRTPathfinder(max_iterations=2000)
    elif effective_algorithm == "RRT*":
        pathfinder = RRTStarPathfinder(max_iterations=2000)

    # 좌표 설정
    start_coord = AIRPORTS[mission.params.start]["coords"]
    target_coord = [mission.params.target_lat, mission.params.target_lon]

    if mission.params.enable_3d:
        s_elev = terrain_fast.get_elevation(*start_coord)
        t_elev = terrain_fast.get_elevation(*target_coord)
        start_coord = [*start_coord, s_elev + 800]
        target_coord = [*target_coord, t_elev + 800]

    threats_dict = [t.to_dict() for t in mission.threats]
    fuel_state = float(getattr(mission.params, "fuel_state", 1.0))
    refuel_count = int(getattr(mission.params, "refuel_count", 0))

    # ── 경로 계산 ──
    start_time = time.time()
    final_in: list = []
    final_out: list = []
    formation_paths: dict = {}
    threats_only_preview = bool(st.session_state.get("_threats_only_preview", False))
    formation_blocked = bool(mission.formation and not formation_allows_downstream(mission.formation))

    if threats_only_preview:
        st.info("ℹ️ 기본 시나리오 미리보기 — SAM/RADAR 배치만 표시합니다. 프롬프트를 입력하거나 '설정 적용 및 경로 계산'을 누르면 경로가 생성됩니다.")
    elif formation_allows_downstream(mission.formation):
        formation_paths = compute_formation_paths(
            mission, pathfinder, effective_algorithm, terrain_fast,
            threats_dict, fuel_state, refuel_count,
        )
        if formation_paths:
            # STRIKE 자산 우선 → 없으면 fighter → 없으면 첫 자산
            _prio_order = ["STRIKE", "SEAD", "ISR", "CAS"]
            _primary_data = None
            for _mission_type in _prio_order:
                for _, _d in formation_paths.items():
                    if _d.get("mission", "").upper() == _mission_type and _d.get("in"):
                        _primary_data = _d
                        break
                if _primary_data:
                    break
            if _primary_data is None:
                _primary_data = next(iter(formation_paths.values()))
            final_in = _primary_data.get("in", [])
            final_out = _primary_data.get("out", [])
            st.session_state["_formation_paths"] = {
                aid: {
                    "in":       d["in"],
                    "out":      d["out"],
                    "threats":  d.get("threats", threats_dict),
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
                for aid, d in formation_paths.items()
            }
    elif formation_blocked:
        st.error("편대 제약을 충족할 수 없어 경로 계산과 출력 생성을 중단했습니다.")
    else:
        st.info("ℹ️ 편대 미구성 — 단일 경로로 계산합니다. MUM-T 패키지 운용을 위해 '편대 계획' 탭에서 자산을 구성하세요.")
        final_in, final_out = compute_single_path(
            mission, pathfinder, effective_algorithm,
            start_coord, target_coord, threats_dict, fuel_state, refuel_count,
        )

    calc_time = time.time() - start_time
    st.caption(f"⏱️ 계산 시간: {calc_time:.3f}초")
    st.session_state['final_in'] = final_in
    st.session_state.current_path = final_in
    
    # ── 실시간 경로 위험도 분석 ──
    if final_in:
        risk_report, per_asset_reports = analyze_mission_risk(
            formation_paths, final_in, final_out,
            threats_dict, mission.params.margin, terrain_fast,
        )
        st.session_state["_last_path_analysis"] = {
            "max_risk": risk_report.get("max_risk", 0.0),
            "package_max_risk": risk_report.get("package_max_risk", risk_report.get("max_risk", 0.0)),
            "fighter_max_risk": risk_report.get("fighter_max_risk", risk_report.get("max_risk", 0.0)),
            "fighter_avg_risk": risk_report.get("fighter_avg_risk", risk_report.get("avg_risk", 0.0)),
            "fighter_distance_km": risk_report.get("fighter_distance_km", risk_report.get("total_length_km", 0.0)),
            "waypoint_count": (
                sum(len((pdata.get("in", []) or []) + (pdata.get("out", []) or [])) for pdata in formation_paths.values())
                if formation_paths else len(final_in) + len(final_out)
            ),
            "total_distance_km": risk_report.get("total_length_km", 0.0),
        }
        st.session_state["_last_asset_risk_reports"] = per_asset_reports
        
        st.divider()
        st.markdown("### 📊 실시간 경로 위험도 분석 (3D Dominant Risk)")
        if formation_paths:
            m1, m2, m3, m4, m5 = st.columns(5)
        else:
            m1, m2, m3, m4 = st.columns(4)
        
        # ── 메트릭 카드: 전투기 생존성 중심으로 표시 ─────────────
        max_risk         = float(risk_report.get("max_risk", 0.0))
        package_max_risk = float(risk_report.get("package_max_risk", max_risk))
        # 전투기 전용 메트릭 (formation_paths에서 분리된 값)
        fighter_max_risk    = float(risk_report.get("fighter_max_risk", package_max_risk))
        fighter_avg_risk    = float(risk_report.get("fighter_avg_risk", risk_report.get("avg_risk", 0.0)))
        fighter_distance_km = float(risk_report.get("fighter_distance_km", risk_report.get("total_length_km", 0.0)))

        if fighter_max_risk >= RISK_THRESHOLD_HIGH:
            risk_delta = "🔴 고위험"
        elif fighter_max_risk >= RISK_THRESHOLD_MEDIUM:
            risk_delta = "🟡 경고"
        else:
            risk_delta = None

        if formation_paths:
            # 전투기 생존성 → 편대 패키지 건전성 순서로 표시
            m1.metric("🎖️ 전투기 위험도", f"{fighter_max_risk:.3f}", delta=risk_delta, delta_color="inverse",
                      help="조종사 경로의 최대 위험도 (생존성 기준 주요 지표)")
            m2.metric("📦 패키지 최대 위험", f"{package_max_risk:.3f}",
                      help="편대 전 자산 중 최대 위험도 (UAV 포함)")
            m3.metric("✈️ 전투기 비행거리", f"{fighter_distance_km:.0f}km",
                      help="전투기(유인기) 왕복 비행거리")
            m4.metric("치명적 구간 수", f"{risk_report['high_risk_segments']}개",
                      help="전 자산 경로 중 위험도 0.7 이상 구간 합산")
            m5.metric("📊 패키지 총거리", f"{risk_report['total_length_km']:.0f}km",
                      help="편대 전 자산 비행거리 합산")
        else:
            m1.metric("🎖️ 전투기 위험도", f"{fighter_max_risk:.3f}", delta=risk_delta, delta_color="inverse")
            m2.metric("평균 노출 위험", f"{fighter_avg_risk:.3f}")
            m3.metric("치명적 구간 수", f"{risk_report['high_risk_segments']}개")
            m4.metric("✈️ 비행거리", f"{fighter_distance_km:.0f}km")

        if formation_paths and per_asset_reports:
            st.caption(
                f"전투기 위험도 {fighter_max_risk:.3f} | 패키지 최대 {package_max_risk:.3f} | "
                f"역할 가중 {max_risk:.3f}"
            )
            if package_max_risk >= RISK_THRESHOLD_HIGH and fighter_max_risk < RISK_THRESHOLD_MEDIUM:
                st.info(
                    "패키지 최대 위험도는 UAV까지 포함한 최악값입니다. "
                    "전투기 위험도가 낮아도 정찰/SEAD UAV가 위협권에 접근하면 높게 표시될 수 있습니다."
                )
            asset_risk_df = pd.DataFrame(per_asset_reports)
            asset_risk_df = asset_risk_df.rename(columns={
                "callsign": "자산",
                "mission": "임무",
                "type": "유형",
                "objective": "세부목적",
                "distance_km": "이동거리(km)",
                "avg_risk": "평균위험",
                "max_risk": "원시최대위험",
                "adjusted_max_risk": "가중최대위험",
                "risk_weight": "가중치",
                "high_risk_segments": "고위험구간",
            })
            with st.expander("편대 자산별 위험/거리 상세", expanded=False):
                # 자산 유형 표시 개선
                def _type_label(row):
                    t = row.get("유형", "")
                    if t == "fighter":     return "🎖️ 전투기"
                    if t == "attack_uav":  return "💥 자폭UAV"
                    if t == "recon_uav":   return "👁️ 정찰UAV"
                    return t
                asset_risk_df["유형"] = asset_risk_df.apply(_type_label, axis=1)
                st.dataframe(
                    asset_risk_df[[
                        "자산", "임무", "유형", "세부목적",
                        "이동거리(km)", "평균위험", "원시최대위험",
                        "가중최대위험", "가중치", "고위험구간"
                    ]],
                    hide_index=True,
                    use_container_width=True,
                )

        # 분석 결과에 따른 전술 권고 (전투기 생존성 기준)
        if fighter_max_risk >= 1.0:
            st.error("🚨 **작전 불가**: 전투기 경로에 NFZ 침범 또는 치명적 위협이 감지되었습니다. SEAD 자산 추가 또는 경로 재설계 필요.")
        elif fighter_max_risk >= RISK_THRESHOLD_HIGH:
            st.error(f"🚨 **조종사 위험**: 전투기 경로 위험도 {fighter_max_risk:.3f} — UAV SEAD 지원 또는 안전 마진 증가 권고.")
        elif fighter_max_risk >= RISK_THRESHOLD_MEDIUM:
            st.warning(f"⚠️ **주의**: 전투기 경로 위험도 {fighter_max_risk:.3f} — 통제 가능 수준이나 SEAD 지원 권장.")
        else:
            st.success(f"✅ **생존성 양호**: 전투기 경로 위험도 {fighter_max_risk:.3f} — UAV SEAD 지원 후 안전 진입 가능.")
    else:
        st.session_state.pop("_last_path_analysis", None)
        st.session_state.pop("_last_asset_risk_reports", None)
    # ── 여기까지 추가 ──

    # ===== 지도 시각화 v3.0 (이후 기존 코드 동일) =====

    # 경로 생성 실패 진단
    if formation_paths:
        failed = [aid for aid, d in formation_paths.items() if not d["in"]]
        ok     = [aid for aid, d in formation_paths.items() if d["in"]]
        if failed:
            st.warning(
                f"⚠️ 경로 탐색 부분 실패: {', '.join(failed)}\n\n"
                "**원인 및 해결책:**\n"
                "- 위협(RADAR 80km) 반경이 경로를 모두 차단할 수 있음\n"
                "- **안전 마진을 줄이거나(예: 2km)** A* 3D 알고리즘 선택 시 저고도 우회 자동 적용\n"
                "- 위협 반경을 줄이거나 위협을 삭제한 후 재시도"
            )
        if ok:
            st.success(f"✅ 경로 탐색 성공: {', '.join(ok)} ({len(ok)}/{len(formation_paths)} 자산)")
    elif not formation_blocked and not final_in and mission.threats and not threats_only_preview:
        st.warning(
            "⚠️ 경로 탐색 실패\n\n"
            "위협 반경이 모든 경로를 차단 중입니다. "
            "안전 마진을 줄이거나(사이드바 슬라이더), "
            "위협을 일부 삭제한 후 재시도하세요.\n"
            "A* 3D 알고리즘 선택 시 저고도 우회를 자동 시도합니다."
        )

    # ===== 지도 시각화 =====
    if not formation_blocked:
        show_heatmap = st.session_state.get('_show_heatmap', True)
        m = build_tactical_map(
            mission, final_in, final_out, formation_paths,
            threats_dict, show_heatmap, terrain_fast,
            start_coord, target_coord,
        )
        st_folium(m, width="100%", height=720, returned_objects=[])

    # ===== 3D 시뮬레이션 시작 버튼 =====
    _fp_for_sim = st.session_state.get("_formation_paths", {})
    if not formation_blocked and (_fp_for_sim or final_in):
        st.divider()
        _sim_col1, _sim_col2, _sim_col3 = st.columns([1, 2, 1])
        with _sim_col2:
            st.markdown(
                "<div style='text-align:center;color:#58A6FF;font-size:13px;"
                "font-weight:600;letter-spacing:1px;margin-bottom:8px;'>"
                "3D 브라우저 시뮬레이션 (Cesium)</div>",
                unsafe_allow_html=True,
            )
            # 클라우드(예: Streamlit Community Cloud)에서는 3D 실행이 서버 쪽
            # 로컬 브라우저/서버에 의존해 방문자 화면에 뜨지 않는다. 무반응으로
            # "고장난 것처럼" 보이지 않도록 안내하고 버튼을 비활성화한다.
            # 로컬 실행 시에는 기존 동작 그대로 유지된다.
            import os as _os
            _is_cloud = _os.path.exists("/mount/src") or bool(
                _os.environ.get("STREAMLIT_RUNTIME_ENV") == "cloud"
            )
            if _is_cloud:
                st.info(
                    "클라우드 데모에서는 3D 시뮬레이션 실행이 지원되지 않습니다. "
                    "3D는 저장소를 내려받아 로컬에서 `streamlit run run.py`로 "
                    "실행할 때 브라우저로 열립니다. 아래 Steer Point List 섹션의 "
                    "'CZML 다운로드'를 받아 Cesium Sandcastle에서도 볼 수 있습니다."
                )
            if st.button(
                "\U0001f310  3D 시뮬레이션 시작",
                type="primary",
                use_container_width=True,
                disabled=_is_cloud,
                help="Cesium 기반 3D 뷰어가 브라우저에서 열립니다. 인터넷 연결 필요. (로컬 실행 전용)",
                key="_btn_launch_sim",
            ):
                try:
                    _sim_fp = _fp_for_sim if _fp_for_sim else {
                        "primary": {
                            "in": final_in, "out": final_out,
                            "color": "#64C8FF", "callsign": "ALPHA-1",
                            "mission": "STRIKE", "type": "fighter",
                            "objective": mission.params.target_name,
                        }
                    }
                    with st.spinner("3D 시뮬레이션 파일 생성 중..."):
                        _czml = export_czml(_sim_fp, mission)
                        _html_path = launch_simulation(_sim_fp, mission, _czml)
                    st.success(
                        f"\u2705 시뮬레이션 시작! 브라우저가 열리지 않으면: "
                        f"`{_html_path}`"
                    )
                    st.caption(
                        "팁: 브라우저에서 ▶/⏸ 버튼으로 재생 제어 | "
                        "10×·30×·60× 배속 지원 | '추적' 버튼으로 자산 시점 고정"
                    )
                except Exception as _sim_err:
                    st.error(f"시뮬레이션 오류: {_sim_err}")

    # ===== [복구 완료] STPT 테이블 및 다운로드 =====
    if not formation_blocked and final_in:
        st.divider()
        st.subheader("📋 Steer Point List")
        
        # km 단위 gap → 배열 인덱스 변환
        # 경로 알고리즘(A*, RRT 등)마다 점 간격이 다르므로 경로에서 직접 계산
        def _stpt_gap_cells(path: list, gap_km: float) -> int:
            """경로 앞부분 샘플로 평균 점 간격(km)을 구해 skip index 반환"""
            import math as _math
            if len(path) < 2:
                return 1
            n_sample = min(20, len(path) - 1)
            total = 0.0
            for k in range(n_sample):
                a, b = path[k], path[k + 1]
                dlat = (b[0] - a[0]) * 110.57
                dlon = (b[1] - a[1]) * 110.57 * _math.cos(_math.radians(a[0]))
                total += _math.sqrt(dlat ** 2 + dlon ** 2)
            spacing = total / n_sample
            return max(1, round(gap_km / spacing)) if spacing > 1e-9 else 1

        gap_km = mission.params.stpt_gap
        def _stpt_rows(path: list, path_type: str, gap_km: float) -> list:
            if not path:
                return []
            gap_cells = _stpt_gap_cells(path, gap_km)
            idxs = list(range(0, len(path), gap_cells))
            if idxs[-1] != len(path) - 1:
                idxs.append(len(path) - 1)
            rows = []
            for seq, idx in enumerate(idxs, start=1):
                p = path[idx]
                pt = {"Type": path_type, "Seq": seq, "Lat": f"{p[0]:.4f}", "Lon": f"{p[1]:.4f}"}
                if len(p) >= 3:
                    pt["Alt(m)"] = f"{p[2]:.0f}"
                rows.append(pt)
            return rows

        data_in = _stpt_rows(final_in, "Ingress", gap_km)
        data_out = _stpt_rows(final_out, "Egress", gap_km)

        stpt_df = pd.DataFrame(data_in + data_out)
        st.dataframe(stpt_df, use_container_width=True, hide_index=True)

        csv = stpt_df.to_csv(index=False).encode('utf-8')
        _dl_col1, _dl_col2 = st.columns(2)
        with _dl_col1:
            st.download_button("📥 STPT CSV 다운로드", csv, "mission_stpt.csv", "text/csv",
                               use_container_width=True)
        with _dl_col2:
            try:
                _dl_fp = st.session_state.get("_formation_paths", {}) or {
                    "primary": {"in": final_in, "out": final_out, "color": "#64C8FF",
                                "callsign": "ALPHA-1", "mission": "STRIKE",
                                "type": "fighter", "objective": mission.params.target_name}
                }
                import json as _json
                _czml_dl = export_czml(_dl_fp, mission)
                _czml_bytes = _json.dumps(_czml_dl, ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button(
                    "\U0001f310 CZML 다운로드 (Cesium)",
                    _czml_bytes, "mission.czml", "application/json",
                    use_container_width=True,
                    help="Cesium Sandcastle(cesium.com/sandcastle)에서 직접 열 수 있습니다.",
                )
            except Exception:
                pass

        st.success(f"\u2705 경로 생성 완료: 총 {len(final_in) + len(final_out)}개 웨이포인트")
        # AirSim 통합 섹션은 사용자 요청으로 제거 (2026-04-25)
