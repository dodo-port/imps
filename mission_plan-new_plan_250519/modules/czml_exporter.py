"""
CZML Exporter — IMPS 임무 데이터를 Cesium CZML 형식으로 변환

CZML(Cesium Language)은 시간 기반 3D 지리 데이터 포맷으로,
편대 경로 애니메이션·위협 시각화·타임라인 재생을 지원합니다.

좌표 주의: CZML cartographicDegrees 순서는 [lon, lat, alt] (Python 내부는 [lat, lon])
"""
from __future__ import annotations

import base64
import json
import math
import struct
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from modules.config import AIRPORTS, ASSET_PERFORMANCE, LAT_TO_KM

# ================================================================
# 3D 모델 GLB 생성 (외부 파일 없이 Python으로 직접 생성)
# ================================================================

def _pack_glb(vertices: list, triangles: list, color_rgb: tuple) -> bytes:
    """
    vertices: [(x,y,z), ...], triangles: [i0,i1,i2,...] flat list
    Generates minimal valid GLB (binary glTF 2.0).
    """
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]

    v_data = struct.pack(f"<{len(vertices)*3}f", *[c for v in vertices for c in v])
    v_data += b"\x00" * ((-len(v_data)) % 4)

    i_data = struct.pack(f"<{len(triangles)}H", *triangles)
    i_data += b"\x00" * ((-len(i_data)) % 4)

    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0},
                                    "indices": 1, "material": 0, "mode": 4}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(vertices),
             "type": "VEC3", "max": [max(xs), max(ys), max(zs)],
             "min": [min(xs), min(ys), min(zs)]},
            {"bufferView": 1, "componentType": 5123, "count": len(triangles),
             "type": "SCALAR", "max": [max(triangles)], "min": [0]},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(v_data), "target": 34962},
            {"buffer": 0, "byteOffset": len(v_data), "byteLength": len(i_data), "target": 34963},
        ],
        "buffers": [{"byteLength": len(v_data) + len(i_data)}],
        "materials": [{"pbrMetallicRoughness": {
            "baseColorFactor": [color_rgb[0], color_rgb[1], color_rgb[2], 1.0],
            "metallicFactor": 0.2, "roughnessFactor": 0.6,
        }, "doubleSided": True}],
    }

    j = json.dumps(gltf, separators=(",", ":")).encode()
    j += b" " * ((-len(j)) % 4)
    bin_data = v_data + i_data
    total = 12 + 8 + len(j) + 8 + len(bin_data)

    return (struct.pack("<III", 0x46546C67, 2, total)
            + struct.pack("<II", len(j), 0x4E4F534A) + j
            + struct.pack("<II", len(bin_data), 0x004E4942) + bin_data)


def _glb_uri(glb_bytes: bytes) -> str:
    return "data:model/gltf-binary;base64," + base64.b64encode(glb_bytes).decode()


def _fighter_model_uri(color_rgb: tuple) -> str:
    """
    전투기 실루엣 (탑뷰, +Y 방향이 기수)
    동체 + 후퇴익 + 수직미익 2개
    """
    verts = [
        # 동체 중심선
        ( 0.00,  1.00, 0.00),   # 0  기수
        (-0.07,  0.30, 0.00),   # 1  동체 앞-좌
        ( 0.07,  0.30, 0.00),   # 2  동체 앞-우
        (-0.07, -0.40, 0.00),   # 3  동체 뒤-좌
        ( 0.07, -0.40, 0.00),   # 4  동체 뒤-우
        # 주익 (후퇴각 35°)
        (-0.75,  0.10, 0.00),   # 5  좌익 끝
        ( 0.75,  0.10, 0.00),   # 6  우익 끝
        # 수직미익 2개
        (-0.22, -0.70, 0.00),   # 7  좌 미익
        ( 0.22, -0.70, 0.00),   # 8  우 미익
        ( 0.00, -1.00, 0.00),   # 9  꼬리 끝
    ]
    tris = [
        # 기수
        0, 2, 1,
        # 동체
        1, 2, 3,   2, 4, 3,
        # 좌익
        1, 5, 3,
        # 우익
        2, 4, 6,
        # 좌 미익
        3, 7, 9,
        # 우 미익
        4, 9, 8,
        # 꼬리 중심
        3, 9, 4,
    ]
    return _glb_uri(_pack_glb(verts, tris, color_rgb))


def _recon_uav_model_uri(color_rgb: tuple) -> str:
    """
    정찰 UAV (쿼드콥터 탑뷰: 십자형 + 4개 로터 원판)
    """
    a = 0.14  # 동체 반폭
    r = 0.18  # 로터 반경
    arm = 0.65  # 암 길이
    verts = [
        # 동체
        (-a, -a, 0.00), ( a, -a, 0.00), ( a,  a, 0.00), (-a,  a, 0.00),  # 0-3
        # 좌암
        (-arm,  a*0.5, 0.00), (-arm, -a*0.5, 0.00),  # 4,5
        # 우암
        ( arm,  a*0.5, 0.00), ( arm, -a*0.5, 0.00),  # 6,7
        # 전암
        (-a*0.5,  arm, 0.00), ( a*0.5,  arm, 0.00),  # 8,9
        # 후암
        (-a*0.5, -arm, 0.00), ( a*0.5, -arm, 0.00),  # 10,11
    ]
    tris = [
        # 동체
        0, 1, 2,   0, 2, 3,
        # 좌암
        3, 4, 5,   3, 5, 0,
        # 우암
        1, 7, 6,   1, 2, 6,
        # 전암
        3, 2, 9,   3, 9, 8,
        # 후암
        0, 11, 1,  0, 10, 11,
    ]
    return _glb_uri(_pack_glb(verts, tris, color_rgb))


def _missile_model_uri() -> str:
    """미사일 GLB: 탄두(원추) + 동체 + 안정날개"""
    verts = [
        ( 0.00,  1.00, 0.00),  # 0 탄두
        (-0.07,  0.65, 0.00),  # 1 동체 상단 좌
        ( 0.07,  0.65, 0.00),  # 2 동체 상단 우
        (-0.07, -1.00, 0.00),  # 3 동체 하단 좌
        ( 0.07, -1.00, 0.00),  # 4 동체 하단 우
        (-0.30, -0.85, 0.00),  # 5 좌 안정날개
        ( 0.30, -0.85, 0.00),  # 6 우 안정날개
        (-0.07, -0.60, 0.05),  # 7 동체 측면 (두께감)
        ( 0.07, -0.60, 0.05),  # 8
    ]
    tris = [
        0, 2, 1,
        1, 2, 4,  1, 4, 3,
        3, 5, 1,
        4, 2, 6,
        1, 3, 7,  2, 8, 4,
    ]
    return _glb_uri(_pack_glb(verts, tris, (1.0, 0.65, 0.05)))


def _attack_uav_model_uri(color_rgb: tuple) -> str:
    """
    공격 UAV (자폭 UAV: 소형 고정익, 전투기보다 날개 비율 큼)
    """
    verts = [
        ( 0.00,  0.90, 0.00),   # 0  기수
        (-0.05,  0.20, 0.00),   # 1  동체 앞-좌
        ( 0.05,  0.20, 0.00),   # 2  동체 앞-우
        (-0.05, -0.30, 0.00),   # 3  동체 뒤-좌
        ( 0.05, -0.30, 0.00),   # 4  동체 뒤-우
        # 넓은 글라이더형 주익
        (-0.95,  0.00, 0.00),   # 5  좌익 끝
        ( 0.95,  0.00, 0.00),   # 6  우익 끝
        # V-tail
        (-0.18, -0.65, 0.00),   # 7
        ( 0.18, -0.65, 0.00),   # 8
        ( 0.00, -0.90, 0.00),   # 9  꼬리
    ]
    tris = [
        0, 2, 1,
        1, 2, 3,  2, 4, 3,
        1, 5, 3,
        2, 4, 6,
        3, 7, 9,
        4, 9, 8,
        3, 9, 4,
    ]
    return _glb_uri(_pack_glb(verts, tris, color_rgb))

# ================================================================
# 사용자 제공 GLB 모델 경로 (data/models/*.glb)
# ================================================================
_MODELS_DIR = None  # 런타임에 설정됨

def _resolve_models_dir() -> "Path":
    from pathlib import Path
    global _MODELS_DIR
    if _MODELS_DIR is None:
        for candidate in [
            Path("data/models"),
            Path(__file__).parent.parent / "data" / "models",
        ]:
            if candidate.exists():
                _MODELS_DIR = candidate
                break
        else:
            _MODELS_DIR = Path("data/models")
    return _MODELS_DIR


def _glb_exists(filename: str) -> bool:
    """data/models/{filename} 파일 존재 여부 확인"""
    try:
        return (_resolve_models_dir() / filename).exists()
    except Exception:
        return False


# GLB 파일명 매핑 (자산 타입 → GLB 파일명)
_GLB_MAP = {
    "fighter":    "fighter.glb",
    "recon_uav":  "recon_uav.glb",
    "attack_uav": "attack_uav.glb",
}

# GLB 자산별 스케일 설정
# minimumPixelSize: 멀리서도 최소 이 픽셀 크기 유지 (항공기 3종 동일하게 맞춤)
# maximumScale: 가까이 봤을 때 실제 크기 상한
_GLB_SCALE = {
    "fighter":    {"scale": 1.0, "minimumPixelSize": 40, "maximumScale": 6000},
    "recon_uav":  {"scale": 1.0, "minimumPixelSize": 40, "maximumScale": 6000},
    "attack_uav": {"scale": 1.0, "minimumPixelSize": 40, "maximumScale": 6000},
}


# ================================================================
# 사용자 제공 이미지 로더 (data/icons/*.png → base64 data URI)
# ================================================================
_ICONS_DIR = None   # 런타임에 설정됨

def _resolve_icons_dir() -> "Path":
    from pathlib import Path
    global _ICONS_DIR
    if _ICONS_DIR is None:
        # streamlit_app.py 기준 경로 탐색
        for candidate in [
            Path("data/icons"),
            Path(__file__).parent.parent / "data" / "icons",
        ]:
            if candidate.exists():
                _ICONS_DIR = candidate
                break
        else:
            _ICONS_DIR = Path("data/icons")  # 없어도 경로 보관
    return _ICONS_DIR


def _load_icon(filename: str) -> Optional[str]:
    """data/icons/{filename} → 'data:image/png;base64,...' URI. 없으면 None."""
    path = _resolve_icons_dir() / filename
    try:
        if path.exists():
            raw = path.read_bytes()
            ext = path.suffix.lower().lstrip(".")
            mime = "image/png" if ext == "png" else f"image/{ext}"
            return f"data:{mime};base64," + base64.b64encode(raw).decode()
    except Exception:
        pass
    return None


# 아이콘 캐시 (첫 호출에만 파일 I/O)
_ICON_CACHE: dict = {}

def _get_icon(key: str, fallback_fn) -> str:
    """캐시된 사용자 이미지 또는 SVG 폴백 반환."""
    if key not in _ICON_CACHE:
        uri = _load_icon(f"{key}.png") or _load_icon(f"{key}.jpg")
        _ICON_CACHE[key] = uri if uri else fallback_fn()
    return _ICON_CACHE[key]


# ================================================================
# 임무 단계 출발 지연 — 동적 계산 (교리 기반)
# JP3-30: 정찰 완료 확인 → SEAD 출발 → SAM 파괴 확인 → STRIKE 출발
# ================================================================

def _offset_positions(positions: list, delay: float) -> list:
    """positions 배열의 모든 타임스탬프에 delay(초) 추가"""
    result = list(positions)
    for i in range(len(result) // 4):
        result[i * 4] += delay
    return result


def _compute_dynamic_delays(raw_timing: dict) -> dict:
    """
    각 자산의 실제 비행 시간 기반으로 임무 단계 지연 계산.

    [ISR] T+0 출발
      → 위협 지역 도달 시간 + 선회 완료 시간 = ISR 완료 시점

    [SEAD] ISR 완료 후 출발
      → 출발 지연 + 비행 시간 + 교전 시간 = SEAD 완료 시점

    [STRIKE] SEAD 완료 후 출발 (모든 SAM 파괴 확인 후 진입)
    """
    delays: dict = {}

    # ── 1단계: ISR 완료 시간 ────────────────────────────────────
    # path_computer에서 ISR path_in에 이미 선회 경로를 포함시켰으므로
    # 마지막 waypoint 시각 = 선회 완료 시각 (중복 계산 방지)
    isr_done_t = 0.0
    for asset_id, rd in raw_timing.items():
        mission_t = (rd["pdata"].get("mission") or "").upper()
        atype     = rd["pdata"].get("type", "")
        if mission_t == "ISR" or atype == "recon_uav":
            delays[asset_id] = 0.0   # ISR: 즉시 출발
            pos_in = rd["pos_in"]
            in_pts = rd["in_pts"]
            if in_pts > 1 and pos_in:
                # 마지막 waypoint = 선회 완료 시각 (circle 경로 포함)
                isr_done_t = max(isr_done_t, pos_in[(in_pts - 1) * 4])

    if isr_done_t < 60.0:
        isr_done_t = 600.0   # ISR 자산 없으면 기본 10분

    # ── 2단계: SEAD 완료 시간 ────────────────────────────────────
    sead_done_t = isr_done_t
    has_sead = False
    for asset_id, rd in raw_timing.items():
        mission_t = (rd["pdata"].get("mission") or "").upper()
        if mission_t == "SEAD":
            has_sead = True
            delays[asset_id] = isr_done_t   # ISR 완료 후 출발
            pos_in = rd["pos_in"]
            in_pts = rd["in_pts"]
            if in_pts > 1 and pos_in:
                # SEAD 비행 시간 (딜레이 제외한 순수 비행)
                flight_t        = pos_in[(in_pts - 1) * 4]
                engagement_time = 240.0   # SAM 교전·확인 4분
                sead_done_t = max(sead_done_t, isr_done_t + flight_t + engagement_time)

    if not has_sead:
        sead_done_t = isr_done_t + 600.0   # SEAD 없으면 +10분 여유

    # ── 3단계: STRIKE/CAS 및 미할당 자산 지연 ───────────────────────
    # 이전엔 if 분기 직후 같은 루프에서 `if asset_id not in delays: delays = 0.0`
    # 가드를 두어, mission이 None/빈 자산이 항상 0.0으로 떨어져 ISR보다 먼저
    # 출발하는 시각화 결함이 있었음. 또한 STRIKE를 무조건 sead_done_t로 출발
    # 시키면 mission_timeline의 "최대한 일찍 출발 + 도착이 SEAD 완료 이후가 되도록
    # 지연만 보정" 정책과 어긋나 두 시각화가 다른 결과를 보였음. 비행시간을 빼서
    # 정확히 SEAD 완료 시각에 도착하도록 통일.
    for asset_id, rd in raw_timing.items():
        if asset_id in delays:
            continue
        mission_t = (rd["pdata"].get("mission") or "").upper()
        # 자산의 path_in 비행 시간 (딜레이 제외한 순수 비행)
        flight_in_t = 0.0
        pos_in = rd.get("pos_in") or []
        in_pts = rd.get("in_pts", 0)
        if in_pts > 1 and pos_in:
            flight_in_t = pos_in[(in_pts - 1) * 4]

        if mission_t in ("STRIKE", "CAS"):
            # SEAD 완료 시각에 도착하도록 출발 지연. 비행이 길어 즉시 출발해도
            # SEAD 완료 후 도착하면 delay=0.
            delays[asset_id] = max(0.0, sead_done_t - flight_in_t)
        elif mission_t == "SEAD":
            # 안전망 — 1단계에서 처리되어 여기 들어올 일은 거의 없음.
            delays[asset_id] = max(0.0, isr_done_t - flight_in_t)
        elif mission_t == "ISR":
            delays[asset_id] = 0.0
        else:
            # 미할당 자산은 SEAD 완료 후 합류.
            delays[asset_id] = max(0.0, sead_done_t - flight_in_t)

    return delays


# ================================================================
# 자산 속도 (m/s) — ASSET_PERFORMANCE km/h 기반
# ================================================================
def _kmh_to_ms(kmh: float) -> float:
    return kmh / 3.6

ASSET_SPEEDS_MS = {
    k: _kmh_to_ms(v["speed_kmh"])
    for k, v in ASSET_PERFORMANCE.items()
}

# ================================================================
# 색상 팔레트
# ================================================================
THREAT_FILL_RGBA = {
    "SAM":   [220,  50,  50,  70],
    "RADAR": [160,  50, 230,  60],
    "NFZ":   [255, 140,   0,  60],
}
THREAT_LINE_RGBA = {
    "SAM":   [255,  80,  80, 220],
    "RADAR": [200,  80, 255, 220],
    "NFZ":   [255, 160,  30, 200],
}


# ================================================================
# SVG 아이콘 (Base64 Data URI)
# ================================================================
def _b64svg(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode()


def _icon_fighter(color: str = "#64C8FF") -> str:
    return _b64svg(f"""<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40">
  <polygon points="20,3 25,22 20,18 15,22" fill="{color}" stroke="#000" stroke-width="1.2"/>
  <polygon points="4,27 20,18 36,27 20,23" fill="{color}" stroke="#000" stroke-width="1.2" opacity="0.9"/>
  <polygon points="12,36 20,28 28,36" fill="{color}" stroke="#000" stroke-width="1"/>
  <circle cx="20" cy="17" r="2" fill="#fff" opacity="0.6"/>
</svg>""")


def _icon_uav_recon(color: str = "#64FF96") -> str:
    return _b64svg(f"""<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">
  <circle cx="16" cy="16" r="5" fill="{color}" stroke="#000" stroke-width="1"/>
  <line x1="2" y1="16" x2="11" y2="16" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="21" y1="16" x2="30" y2="16" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="16" y1="2" x2="16" y2="11" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>
  <line x1="16" y1="21" x2="16" y2="30" stroke="{color}" stroke-width="2.5" stroke-linecap="round"/>
  <circle cx="16" cy="16" r="2" fill="#fff" opacity="0.7"/>
</svg>""")


def _icon_uav_attack(color: str = "#FF6432") -> str:
    return _b64svg(f"""<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">
  <polygon points="16,2 19,20 16,17 13,20" fill="{color}" stroke="#000" stroke-width="1"/>
  <polygon points="6,22 16,17 26,22 16,20" fill="{color}" stroke="#000" stroke-width="1" opacity="0.8"/>
  <polygon points="13,30 16,23 19,30" fill="{color}" stroke="#000" stroke-width="1"/>
</svg>""")


def _icon_target() -> str:
    return _b64svg("""<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">
  <circle cx="18" cy="18" r="15" fill="none" stroke="#FF3333" stroke-width="2.5"/>
  <circle cx="18" cy="18" r="9"  fill="none" stroke="#FF3333" stroke-width="2"/>
  <circle cx="18" cy="18" r="3"  fill="#FF3333"/>
  <line x1="18" y1="3"  x2="18" y2="33" stroke="#FF3333" stroke-width="1.5"/>
  <line x1="3"  y1="18" x2="33" y2="18" stroke="#FF3333" stroke-width="1.5"/>
</svg>""")


def _icon_destroyed() -> str:
    """타격 후 파괴 마커 (빨간 X)"""
    return _b64svg("""<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 36 36">
  <circle cx="18" cy="18" r="16" fill="rgba(0,0,0,0.55)" stroke="#CC3333" stroke-width="2"/>
  <line x1="9" y1="9" x2="27" y2="27" stroke="#FF4444" stroke-width="4" stroke-linecap="round"/>
  <line x1="27" y1="9" x2="9" y2="27" stroke="#FF4444" stroke-width="4" stroke-linecap="round"/>
</svg>""")


def _icon_base() -> str:
    return _b64svg("""<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">
  <polygon points="16,3 30,26 2,26" fill="none" stroke="#4488FF" stroke-width="2.5"/>
  <polygon points="16,9 25,24 7,24"  fill="#4488FF" opacity="0.4"/>
  <rect x="13" y="19" width="6" height="7" fill="#4488FF" opacity="0.7"/>
</svg>""")


# ================================================================
# 거리 계산
# ================================================================
def _dist_km(p1: list, p2: list) -> float:
    dlat = (p1[0] - p2[0]) * LAT_TO_KM
    dlon = (p1[1] - p2[1]) * LAT_TO_KM * math.cos(math.radians((p1[0] + p2[0]) * 0.5))
    return math.sqrt(dlat ** 2 + dlon ** 2)


def _bearing_deg(p1: list, p2: list) -> float:
    """Approximate bearing from p1 to p2 in degrees clockwise from north."""
    lat1 = math.radians(float(p1[0]))
    lat2 = math.radians(float(p2[0]))
    dlon = math.radians(float(p2[1]) - float(p1[1]))
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _offset_coord(lat: float, lon: float, bearing_deg: float, distance_km: float) -> List[float]:
    theta = math.radians(bearing_deg)
    dlat = (distance_km / LAT_TO_KM) * math.cos(theta)
    lon_scale = max(0.1, LAT_TO_KM * math.cos(math.radians(lat)))
    dlon = (distance_km / lon_scale) * math.sin(theta)
    return [lat + dlat, lon + dlon]


def _base_ground_alt(mission, fallback_alt: float = 0.0) -> float:
    base_info = AIRPORTS.get(getattr(mission.params, "start", ""), {})
    if "elevation" in base_info:
        return max(0.0, float(base_info.get("elevation", 0.0))) + 8.0
    return max(0.0, float(fallback_alt) - 800.0)


def _visualize_takeoff(path: list, atype: str, mission) -> list:
    """Add ground roll and climb-out points for Cesium-only visualization."""
    if not path:
        return path

    pts = [list(p) for p in path]
    first = pts[0]
    cruise_alt = float(first[2]) if len(first) > 2 else ASSET_PERFORMANCE.get(atype, {}).get("altitude_m", 1000.0)
    ground_alt = _base_ground_alt(mission, cruise_alt)
    climb_alt = max(cruise_alt, ground_alt + 250.0)

    if len(pts) > 1:
        bearing = _bearing_deg(first, pts[1])
        leg_km = max(_dist_km(first, pts[1]), 3.0)
    else:
        bearing = 0.0
        leg_km = 6.0

    runway_back_km = 1.2 if atype == "fighter" else 0.45
    mid_km = min(max(leg_km * 0.40, 2.0), 8.0)
    join_km = min(max(leg_km * 0.80, 4.0), 14.0)

    runway_start = _offset_coord(first[0], first[1], (bearing + 180.0) % 360.0, runway_back_km)
    climb_mid = _offset_coord(first[0], first[1], bearing, mid_km)
    climb_join = _offset_coord(first[0], first[1], bearing, join_km)

    visual = [
        [runway_start[0], runway_start[1], ground_alt],
        [first[0], first[1], ground_alt + 12.0],
        [climb_mid[0], climb_mid[1], ground_alt + (climb_alt - ground_alt) * 0.40],
        [climb_join[0], climb_join[1], ground_alt + (climb_alt - ground_alt) * 0.78],
    ]

    return visual + (pts[1:] if len(pts) > 1 else pts)


def _visualize_landing(path: list, atype: str, mission) -> list:
    """Add approach, flare, touchdown, and rollout points for Cesium-only visualization."""
    if not path:
        return path

    pts = [list(p) for p in path]
    if len(pts) < 2:
        return pts

    base = pts[-1]
    prev = pts[-2]
    cruise_alt = float(base[2]) if len(base) > 2 else ASSET_PERFORMANCE.get(atype, {}).get("altitude_m", 1000.0)
    ground_alt = _base_ground_alt(mission, cruise_alt)
    approach_alt = max(cruise_alt, ground_alt + 250.0)

    inbound_bearing = _bearing_deg(prev, base)
    from_base_to_prev = (inbound_bearing + 180.0) % 360.0
    leg_km = max(_dist_km(prev, base), 3.0)
    far_km = min(max(leg_km * 0.70, 4.0), 14.0)
    near_km = min(max(leg_km * 0.30, 1.8), 7.0)

    approach_far = _offset_coord(base[0], base[1], from_base_to_prev, far_km)
    approach_near = _offset_coord(base[0], base[1], from_base_to_prev, near_km)
    rollout = _offset_coord(base[0], base[1], inbound_bearing, 0.9 if atype == "fighter" else 0.35)

    visual_tail = [
        [approach_far[0], approach_far[1], ground_alt + (approach_alt - ground_alt) * 0.70],
        [approach_near[0], approach_near[1], ground_alt + (approach_alt - ground_alt) * 0.28],
        [base[0], base[1], ground_alt + 10.0],
        [rollout[0], rollout[1], ground_alt],
    ]

    return pts[:-1] + visual_tail


# ================================================================
# 경로 → CZML 시간-좌표 배열 변환
# ================================================================
def _path_to_timed_positions(
    path: list,
    speed_ms: float,
    start_offset_s: float = 0.0,
) -> Tuple[List[float], float]:
    """
    path [[lat, lon] or [lat, lon, alt], ...] → CZML cartographicDegrees 배열
    반환: (flat_array, total_elapsed_seconds)
    CZML cartographicDegrees: [t0, lon0, lat0, alt0, t1, lon1, lat1, alt1, ...]
    """
    if not path:
        return [], 0.0

    result: List[float] = []
    t = start_offset_s

    def _alt(p: list) -> float:
        return float(p[2]) if len(p) > 2 else 500.0

    # 첫 점
    result.extend([t, float(path[0][1]), float(path[0][0]), _alt(path[0])])

    for i in range(1, len(path)):
        d_km = _dist_km(path[i - 1], path[i])
        dt = max(1.0, (d_km * 1000.0) / speed_ms)
        t += dt
        result.extend([t, float(path[i][1]), float(path[i][0]), _alt(path[i])])

    return result, t - start_offset_s


# ================================================================
# 색상 헬퍼
# ================================================================
def _hex_to_rgba(hex_color: str, alpha: int = 220) -> List[int]:
    h = hex_color.lstrip("#")
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except (ValueError, IndexError):
        r, g, b = 100, 200, 255
    return [r, g, b, alpha]


def _rgba_to_css(rgba: List[int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(rgba[0], rgba[1], rgba[2])


# ================================================================
# 메인 CZML 생성 함수
# ================================================================
def export_czml(
    formation_paths: Dict,
    mission,
    base_time: Optional[datetime] = None,
    time_multiplier: int = 30,
) -> List[dict]:
    """
    IMPS formation_paths + MissionState → CZML 패킷 리스트

    Args:
        formation_paths: path_computer.compute_formation_paths() 반환값
        mission:         MissionState 객체
        base_time:       시뮬레이션 시작 UTC 시각 (None이면 오늘 06:00 KST)
        time_multiplier: 기본 재생 속도 배율 (30x 권장)

    Returns:
        Cesium CZML 패킷 list (JSON 직렬화 가능)
    """
    if base_time is None:
        base_time = datetime.now(timezone.utc).replace(
            hour=21, minute=0, second=0, microsecond=0  # KST 06:00 = UTC 21:00 전날
        )

    epoch_str = base_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 아이콘 캐시는 색상이 키에 포함되지 않아 export 간 색상 변경 시 stale 위험이
    # 있다. 함수 시작 시 한 번만 비우면 자산 루프 내에서는 재사용되어 base64
    # 인코딩 비용이 누적되지 않는다(이전엔 자산마다 clear되어 N번 재생성).
    _ICON_CACHE.clear()

    # ── 전체 임무 지속 시간 계산 ──────────────────────────────
    max_duration_s = 0.0
    asset_data: Dict[str, dict] = {}

    # ── Pass 1: 딜레이 없이 순수 비행 시간 계산 ─────────────────────
    raw_timing: dict = {}
    for asset_id, pdata in formation_paths.items():
        atype = pdata.get("type", "fighter")
        speed = ASSET_SPEEDS_MS.get(atype, 100.0)
        path_in = _visualize_takeoff(pdata.get("in", []) or [], atype, mission)
        pos_in_raw, dur_in_raw = _path_to_timed_positions(path_in, speed, 0.0)
        raw_timing[asset_id] = {
            "pos_in": pos_in_raw,
            "dur_in": dur_in_raw,
            "in_pts": len(pos_in_raw) // 4,
            "pdata":  pdata,
            "speed":  speed,
            "path_in": path_in,
        }

    # ── 동적 임무 단계 지연 계산 (ISR 완료 → SEAD 출발 → SEAD 완료 → STRIKE 출발) ──
    phase_delays = _compute_dynamic_delays(raw_timing)

    # ── Pass 2: 지연 적용 후 최종 positions 생성 ─────────────────────
    for asset_id, rd in raw_timing.items():
        pdata      = rd["pdata"]
        speed      = rd["speed"]
        in_pts     = rd["in_pts"]
        start_delay = phase_delays.get(asset_id, 0.0)

        path_in = rd.get("path_in") or _visualize_takeoff(pdata.get("in", []) or [], pdata.get("type", "fighter"), mission)
        path_out = _visualize_landing(pdata.get("out", []) or [], pdata.get("type", "fighter"), mission)

        pos_in, dur_in = _path_to_timed_positions(path_in, speed, 0.0)
        # 타임스탬프에 출발 지연 추가
        pos_in = _offset_positions(pos_in, start_delay)

        pos_out, dur_out = _path_to_timed_positions(path_out, speed, start_delay + dur_in)

        combined  = pos_in + (pos_out[4:] if pos_out else [])
        total_dur = start_delay + dur_in + dur_out
        max_duration_s = max(max_duration_s, total_dur)

        asset_data[asset_id] = {
            "positions":   combined,
            "duration":    total_dur,
            "pdata":       pdata,
            "speed":       speed,
            "in_pts":      in_pts,
            "start_delay": start_delay,
        }

    if max_duration_s < 300:
        max_duration_s = 3600.0

    end_time = base_time + timedelta(seconds=max_duration_s * 1.15)
    end_str  = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    interval = f"{epoch_str}/{end_str}"

    packets: List[dict] = []

    # ── Document ────────────────────────────────────────────────
    packets.append({
        "id": "document",
        "name": "IMPS 임무 시뮬레이션",
        "version": "1.0",
        "clock": {
            "interval":    interval,
            "currentTime": epoch_str,
            "multiplier":  time_multiplier,
            "range":       "LOOP_STOP",
            "step":        "SYSTEM_CLOCK_MULTIPLIER",
        },
    })

    # ── 출발 기지 ────────────────────────────────────────────────
    base_info   = AIRPORTS.get(mission.params.start, {})
    base_coords = base_info.get("coords", [37.0, 127.5])
    packets.append({
        "id":   "marker_base",
        "name": f"기지: {mission.params.start}",
        "position": {
            "cartographicDegrees": [base_coords[1], base_coords[0], 30.0]
        },
        "billboard": {
            "image":           _icon_base(),
            "scale":           1.5,
            "verticalOrigin":  "BOTTOM",
            "heightReference": "CLAMP_TO_GROUND",
        },
        "label": {
            "text":         f"\u2708 {mission.params.start}",
            "font":         "bold 13px 'Segoe UI', sans-serif",
            "fillColor":    {"rgba": [100, 180, 255, 255]},
            "outlineColor": {"rgba": [0, 0, 0, 200]},
            "outlineWidth": 2,
            "style":        "FILL_AND_OUTLINE",
            "verticalOrigin": "TOP",
            "pixelOffset":  {"cartesian2": [0, 10]},
            "heightReference": "CLAMP_TO_GROUND",
        },
    })

    # ── 주 목표 ──────────────────────────────────────────────────
    packets.append({
        "id":   "marker_target_primary",
        "name": f"목표: {mission.params.target_name}",
        "position": {
            "cartographicDegrees": [
                mission.params.target_lon,
                mission.params.target_lat,
                60.0,
            ]
        },
        "billboard": {
            "image":           _icon_target(),
            "scale":           1.5,
            "heightReference": "CLAMP_TO_GROUND",
        },
        "label": {
            "text":         f"\U0001f3af {mission.params.target_name}",
            "font":         "bold 13px 'Segoe UI', sans-serif",
            "fillColor":    {"rgba": [255, 80, 80, 255]},
            "outlineColor": {"rgba": [0, 0, 0, 200]},
            "outlineWidth": 2,
            "style":        "FILL_AND_OUTLINE",
            "verticalOrigin": "TOP",
            "pixelOffset":  {"cartesian2": [0, 12]},
            "heightReference": "CLAMP_TO_GROUND",
        },
    })

    # ── 추가 목표 ─────────────────────────────────────────────────
    _extra_rgba = [
        [255, 160, 0, 255],
        [180, 80, 255, 255],
        [0, 210, 180, 255],
    ]
    for ei, et in enumerate(
        sorted(getattr(mission.params, "extra_targets", []), key=lambda t: t.get("priority", 2))
    ):
        if "lat" not in et or "lon" not in et:
            continue
        ec = _extra_rgba[ei % len(_extra_rgba)]
        packets.append({
            "id":   f"marker_target_extra_{ei}",
            "name": et.get("name", f"Target-{ei + 2}"),
            "position": {
                "cartographicDegrees": [float(et["lon"]), float(et["lat"]), 60.0]
            },
            "billboard": {
                "image":           _icon_target(),
                "scale":           1.3,
                "color":           {"rgba": ec},
                "heightReference": "CLAMP_TO_GROUND",
            },
            "label": {
                "text":         f"\U0001f3af {et.get('name', f'Target-{ei+2}')}",
                "font":         "bold 12px 'Segoe UI', sans-serif",
                "fillColor":    {"rgba": ec},
                "outlineColor": {"rgba": [0, 0, 0, 200]},
                "outlineWidth": 2,
                "style":        "FILL_AND_OUTLINE",
                "verticalOrigin": "TOP",
                "pixelOffset":  {"cartesian2": [0, 12]},
                "heightReference": "CLAMP_TO_GROUND",
            },
        })

    # ── 위협 시각화 ───────────────────────────────────────────────
    for ti, threat in enumerate(mission.threats):
        if threat.type == "NFZ":
            if all(v is not None for v in [threat.lat_min, threat.lon_min, threat.lat_max, threat.lon_max]):
                fc = THREAT_FILL_RGBA["NFZ"]
                oc = THREAT_LINE_RGBA["NFZ"]
                packets.append({
                    "id":   f"threat_nfz_{ti}",
                    "name": f"NFZ: {threat.name}",
                    "rectangle": {
                        "coordinates": {
                            "wsenDegrees": [
                                float(threat.lon_min), float(threat.lat_min),
                                float(threat.lon_max), float(threat.lat_max),
                            ]
                        },
                        "fill":         True,
                        "material":     {"solidColor": {"color": {"rgba": fc}}},
                        "outline":      True,
                        "outlineColor": {"rgba": oc},
                        "outlineWidth": 2.0,
                        "heightReference": "CLAMP_TO_GROUND",
                    },
                    "label": {
                        "text":         f"NFZ: {threat.name}",
                        "font":         "11px 'Segoe UI'",
                        "fillColor":    {"rgba": oc},
                        "outlineColor": {"rgba": [0, 0, 0, 180]},
                        "outlineWidth": 2,
                        "style":        "FILL_AND_OUTLINE",
                        "heightReference": "CLAMP_TO_GROUND",
                    },
                    "position": {
                        "cartographicDegrees": [
                            (float(threat.lon_min) + float(threat.lon_max)) / 2,
                            (float(threat.lat_min) + float(threat.lat_max)) / 2,
                            10.0,
                        ]
                    },
                })
        elif threat.lat is not None and threat.lon is not None:
            fc = THREAT_FILL_RGBA.get(threat.type, [100, 100, 255, 60])
            oc = THREAT_LINE_RGBA.get(threat.type, [130, 130, 255, 220])
            label_icon = "\U0001f534" if threat.type == "SAM" else "\U0001f7e3"

            # 위협 GLB 모델 선택 (sam.glb / radar.glb)
            threat_glb = "sam.glb" if threat.type == "SAM" else "radar.glb"
            use_threat_glb = _glb_exists(threat_glb)

            threat_packet: dict = {
                "id":   f"threat_{threat.type}_{ti}",
                "name": f"{threat.type}: {threat.name}",
                "position": {
                    "cartographicDegrees": [float(threat.lon), float(threat.lat), 0.0]
                },
                "ellipse": {
                    "semiMajorAxis":       threat.radius_km * 1000.0,
                    "semiMinorAxis":       threat.radius_km * 1000.0,
                    "fill":                True,
                    "material":            {"solidColor": {"color": {"rgba": fc}}},
                    "outline":             True,
                    "outlineColor":        {"rgba": oc},
                    "outlineWidth":        2.0,
                    "numberOfVerticalLines": 0,
                    "heightReference":     "CLAMP_TO_GROUND",
                },
                "label": {
                    "text":            f"{label_icon} {threat.name}",
                    "font":            "bold 12px 'Segoe UI'",
                    "fillColor":       {"rgba": oc},
                    "outlineColor":    {"rgba": [0, 0, 0, 200]},
                    "outlineWidth":    2,
                    "style":           "FILL_AND_OUTLINE",
                    "verticalOrigin":  "BOTTOM",
                    "horizontalOrigin":"CENTER",
                    "heightReference": "CLAMP_TO_GROUND",
                    "pixelOffset":     {"cartesian2": [0, -20]},   # 고정 20px (반경 무관)
                    "distanceDisplayCondition": {"distanceDisplayCondition": [0, 2.5e6]},
                },
            }

            if use_threat_glb:
                # GLB 모델 크기: SAM은 발사대, RADAR는 안테나 → 동일 기준 픽셀 크기
                threat_packet["model"] = {
                    "gltf":             f"__MODEL_BASE__/{threat_glb}",
                    "scale":            1.0,
                    "minimumPixelSize": 32,   # 항공기와 동일 기준
                    "maximumScale":     2000, # 너무 커지지 않게 제한
                    "runAnimations":    True,
                    "heightReference":  "CLAMP_TO_GROUND",
                }

            packets.append(threat_packet)

    # ── 자산 애니메이션 ───────────────────────────────────────────
    for asset_id, info in asset_data.items():
        positions = info["positions"]
        duration  = info["duration"]
        pdata     = info["pdata"]

        if not positions:
            continue

        atype     = pdata.get("type", "fighter")
        callsign  = pdata.get("callsign", asset_id)
        color_hex = pdata.get("color", "#64C8FF")
        mission_t = pdata.get("mission", "")
        obj_label = pdata.get("objective", "")

        rgba      = _hex_to_rgba(color_hex, 240)
        path_rgba = _hex_to_rgba(color_hex, 180)

        # ── 아이콘/모델 선택 ────────────────────────────────────────
        # GLB 파일(data/models/)이 있으면 3D 모델, 없으면 PNG/SVG 빌보드
        css_color = _rgba_to_css(rgba)
        # 캐시는 (atype, css_color) 키로 분리되어 있어 색상 변경에도 자동 무효화됨.
        # 매 export마다 clear()하면 자산 N개의 base64 SVG를 매번 재생성하는 비용
        # 누적이 발생하므로 제거.

        glb_filename = _GLB_MAP.get(atype)
        use_glb = glb_filename and _glb_exists(glb_filename)

        if not use_glb:
            # 빌보드 크기 통일 (전 자산 동일 scale = 1.8)
            if atype == "fighter":
                icon  = _load_icon("fighter.png") or _icon_fighter(css_color)
            elif atype == "recon_uav":
                icon  = _load_icon("recon_uav.png") or _icon_uav_recon(css_color)
            else:
                icon  = _load_icon("attack_uav.png") or _icon_uav_attack(css_color)
            scale = 1.8

        start_delay = info.get("start_delay", 0.0)
        asset_start = base_time + timedelta(seconds=start_delay)
        asset_end   = base_time + timedelta(seconds=max(duration, start_delay + 60.0))
        avail_str   = f"{asset_start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{asset_end.strftime('%Y-%m-%dT%H:%M:%SZ')}"

        label_text = callsign
        if mission_t:
            label_text += f" [{mission_t}]"

        asset_packet: dict = {
            "id":           asset_id,
            "name":         f"{callsign} — {obj_label}",
            "availability": avail_str,
            "position": {
                "epoch":                    epoch_str,
                "cartographicDegrees":      positions,
                "interpolationAlgorithm":   "LAGRANGE",
                "interpolationDegree":      2,
                "forwardExtrapolationType": "HOLD",
                "backwardExtrapolationType":"HOLD",
            },
            "orientation": {
                "velocityReference": f"{asset_id}#position"
            },
            "label": {
                "text":         label_text,
                "font":         "bold 12px 'Segoe UI', monospace",
                "fillColor":    {"rgba": rgba},
                "outlineColor": {"rgba": [0, 0, 0, 230]},
                "outlineWidth": 2,
                "style":        "FILL_AND_OUTLINE",
                "pixelOffset":  {"cartesian2": [0, -28]},
                "distanceDisplayCondition": {"distanceDisplayCondition": [0, 1.5e6]},
            },
            "path": {
                "show":      [{"interval": avail_str, "boolean": True}],
                "width":     3.5 if atype == "fighter" else 2.5,
                "material":  {"solidColor": {"color": {"rgba": path_rgba}}},
                "leadTime":  0,
                "trailTime": 86400,
                "resolution": 5,
            },
        }

        if use_glb:
            # 3D GLB 모델 사용 (HTTP 서버를 통해 제공)
            glb_cfg = _GLB_SCALE.get(atype, {"scale": 1.0, "minimumPixelSize": 32, "maximumScale": 5000})
            asset_packet["model"] = {
                "gltf":             f"__MODEL_BASE__/{glb_filename}",
                "scale":            glb_cfg["scale"],
                "minimumPixelSize": glb_cfg["minimumPixelSize"],
                "maximumScale":     glb_cfg["maximumScale"],
                "runAnimations":    True,
                "silhouetteColor":  {"rgba": rgba},
                "silhouetteSize":   1.5,
                "color":            {"rgba": rgba},
                "colorBlendMode":   "HIGHLIGHT",
                "colorBlendAmount": 0.3,
            }
        else:
            # 빌보드 아이콘 폴백
            asset_packet["billboard"] = {
                "image":            icon,
                "scale":            scale,
                "verticalOrigin":   "CENTER",
                "horizontalOrigin": "CENTER",
                "eyeOffset":        {"cartesian": [0.0, 0.0, -500.0]},
                "pixelOffsetScaleByDistance": {
                    "nearFarScalar": [1.0e3, 1.0, 1.0e6, 0.5]
                },
            }

        packets.append(asset_packet)

        # ── 공격 효과 (미사일/자폭/AGM) ──────────────────────────────
        if atype == "attack_uav" or mission_t in ("STRIKE", "SEAD"):
            chain_targets = pdata.get("chain_target_coords") or []

            if chain_targets and atype == "fighter":
                # 전투기 체인 타격: 각 목표마다 독립 AGM 효과
                for ci, tc in enumerate(chain_targets):
                    try:
                        _add_attack_effects(
                            packets, f"{asset_id}_t{ci}", positions, atype, mission_t,
                            mission, epoch_str, base_time,
                            in_pts=info.get("in_pts", len(positions) // 8),
                            tgt_lat=float(tc[0]), tgt_lon=float(tc[1]),
                        )
                    except Exception as e:
                        print(f"[CZML] chain attack effects 오류 (무시): {e}")
            else:
                try:
                    _add_attack_effects(
                        packets, asset_id, positions, atype, mission_t,
                        mission, epoch_str, base_time,
                        in_pts=info.get("in_pts", len(positions) // 8),
                    )
                except Exception as e:
                    print(f"[CZML] _add_attack_effects 오류 (무시): {e}")

    # ── 위협 원 숨김: 파괴된 위협의 ellipse/label/model 제거 ───────
    _hide_destroyed_threats(packets, end_str)

    return packets


# ================================================================
# 미사일 발사 + 폭발 CZML 헬퍼
# ================================================================
def _find_wpt_near(positions: list, lat: float, lon: float) -> int:
    """positions 배열에서 (lat, lon)에 가장 가까운 waypoint 인덱스 반환"""
    n = len(positions) // 4
    best_i, best_d = 0, float("inf")
    for i in range(n):
        wlat = positions[i * 4 + 2]
        wlon = positions[i * 4 + 1]
        d = (wlat - lat) ** 2 + (wlon - lon) ** 2
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _add_attack_effects(
    packets: list,
    asset_id: str,
    positions: list,
    atype: str,
    mission_t: str,
    mission,
    epoch_str: str,
    base_time,
    in_pts: int = 0,
    tgt_lat: Optional[float] = None,
    tgt_lon: Optional[float] = None,
) -> None:
    """
    공격 자산별 타격 + 폭발 효과 CZML 패킷을 packets에 추가.

    [attack_uav] 자폭 돌진 (kamikaze):
      - UAV 본체가 SAM 지점으로 수직 급강하 → 지면 충돌 폭발

    [fighter] AGM 공대지 미사일:
      - 전투기가 목표 상공 직전에 AGM 투하 → 비스듬히 낙하 (사선)
      - tgt_lat/tgt_lon 지정 시 해당 목표 타격 (다중 목표 지원)

    시점: positions 배열의 timestamp 자체에 start_delay가 포함되어 있으므로
          별도 오프셋 불필요 — base_time + positions[t] 로 절대 시각 계산
    """
    if len(positions) < 8:
        return

    target_name = getattr(mission.params, "target_name", "TARGET")
    n_total = len(positions) // 4
    if n_total < 2:
        return
    if in_pts < 2:
        in_pts = max(2, n_total // 2)

    # ── 발사/타격 waypoint 결정 ──────────────────────────────────
    if tgt_lat is not None and tgt_lon is not None:
        # 전투기 체인: 지정된 목표와 가장 가까운 waypoint 탐색
        lnch_idx = _find_wpt_near(positions, tgt_lat, tgt_lon)
        impact_lat = tgt_lat
        impact_lon = tgt_lon
    else:
        lnch_idx   = min(in_pts - 1, n_total - 1)
        impact_lat = positions[lnch_idx * 4 + 2]
        impact_lon = positions[lnch_idx * 4 + 1]

    launch_t   = positions[lnch_idx * 4]
    launch_lon = positions[lnch_idx * 4 + 1]
    launch_lat = positions[lnch_idx * 4 + 2]
    launch_alt = max(float(positions[lnch_idx * 4 + 3]), 500.0)

    launch_time = base_time + timedelta(seconds=launch_t)
    launch_iso  = launch_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    if atype == "attack_uav":
        # ── 자폭 UAV: 수직 급강하 → 지면 충돌 ──────────────────────
        dive_spd = 280.0   # m/s
        dive_s   = max(launch_alt / dive_spd, 3.0)
        impact_t = launch_t + dive_s

        impact_time  = base_time + timedelta(seconds=impact_t)
        exp_end_time = impact_time + timedelta(seconds=14)
        impact_iso   = impact_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        exp_end_iso  = exp_end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        packets.append({
            "id":           f"missile_{asset_id}",
            "name":         "자폭 돌진",
            "availability": f"{launch_iso}/{impact_iso}",
            "position": {
                "epoch":               launch_iso,
                "cartographicDegrees": [
                    0.0,    launch_lon, launch_lat, launch_alt,
                    dive_s, impact_lon, impact_lat, 0.0,
                ],
                "interpolationAlgorithm": "LINEAR",
            },
            "orientation": {"velocityReference": f"missile_{asset_id}#position"},
            "model": {
                "gltf":             _missile_model_uri(),
                "scale":            1.0,
                "minimumPixelSize": 8,
                "maximumScale":     2000,
                "runAnimations":    False,
            },
            "path": {
                "show":      True,
                "width":     3.0,
                "material":  {"solidColor": {"color": {"rgba": [255, 60, 20, 220]}}},
                "leadTime":  0,
                "trailTime": 300,
                "resolution": 1,
            },
        })

    else:
        # ── 전투기 AGM: 목표 상공 직전에서 사선 투하 ────────────────
        agm_spd  = 420.0   # m/s
        drop_alt = launch_alt

        # 전투기 접근 방향 계산 → AGM 발사점을 목표 2~4km 전방으로 설정
        if lnch_idx > 0:
            prev_lon = positions[(lnch_idx - 1) * 4 + 1]
            prev_lat = positions[(lnch_idx - 1) * 4 + 2]
            d_lat    = impact_lat - prev_lat
            d_lon    = impact_lon - prev_lon
            norm     = max(math.sqrt(d_lat ** 2 + d_lon ** 2), 1e-7)
            offset   = 0.030   # ~3.3km 앞서 투하
            rel_lat  = impact_lat - d_lat / norm * offset
            rel_lon  = impact_lon - d_lon / norm * offset
        else:
            rel_lat = launch_lat
            rel_lon = launch_lon

        # 수평 거리 기반 비행 시간 (사선 궤적)
        horiz_km = _dist_km([rel_lat, rel_lon], [impact_lat, impact_lon])
        drop_s   = max(math.sqrt(horiz_km ** 2 * 1e6 + drop_alt ** 2) / agm_spd, 5.0)
        impact_t = launch_t + drop_s

        impact_time  = base_time + timedelta(seconds=impact_t)
        exp_end_time = impact_time + timedelta(seconds=18)
        impact_iso   = impact_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        exp_end_iso  = exp_end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        packets.append({
            "id":           f"missile_{asset_id}",
            "name":         "AGM",
            "availability": f"{launch_iso}/{impact_iso}",
            "position": {
                "epoch":               launch_iso,
                "cartographicDegrees": [
                    0.0,    rel_lon,    rel_lat,    drop_alt,  # 전투기 투하 지점
                    drop_s, impact_lon, impact_lat, 0.0,       # 사선 낙하 → 목표 충돌
                ],
                "interpolationAlgorithm": "LINEAR",
            },
            "orientation": {"velocityReference": f"missile_{asset_id}#position"},
            "model": {
                "gltf":             _missile_model_uri(),
                "scale":            1.0,
                "minimumPixelSize": 6,
                "maximumScale":     3000,
                "runAnimations":    False,
            },
            "path": {
                "show":      True,
                "width":     2.5,
                "material":  {"solidColor": {"color": {"rgba": [255, 200, 40, 200]}}},
                "leadTime":  0,
                "trailTime": 600,
                "resolution": 1,
            },
        })

    # ── 폭발 효과 1: 팽창 타원체 (주 폭발구) ─────────────────────────
    packets.append({
        "id":           f"explosion_main_{asset_id}",
        "name":         f"폭발: {target_name}",
        "availability": f"{impact_iso}/{exp_end_iso}",
        "position":     {"cartographicDegrees": [impact_lon, impact_lat, 80.0]},
        "ellipsoid": {
            "radii": {
                "epoch": impact_iso,
                "cartesian": [
                    0,    0,    0,    0,
                    1,    800,  800,  300,
                    3,    3500, 3500, 1200,
                    6,    6000, 6000, 2500,
                    10,   4000, 4000, 1500,
                    18,   200,  200,   80,
                ],
                "interpolationAlgorithm": "LINEAR",
            },
            "material": {
                "solidColor": {
                    "color": {
                        "epoch": impact_iso,
                        "rgba": [
                            0,   255, 255, 100,   0,
                            0.3, 255, 200,   0, 240,
                            1.5, 255,  80,   0, 220,
                            4,   200,  60,  20, 180,
                            8,    90,  90,  90, 120,
                            18,    0,   0,   0,   0,
                        ],
                    }
                }
            },
            "fill": True,
        },
    })

    # ── 폭발 효과 2: 섬광 (충격파 링) ───────────────────────────────
    flash_end_iso = (impact_time + timedelta(seconds=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
    packets.append({
        "id":           f"explosion_flash_{asset_id}",
        "name":         "충격파",
        "availability": f"{impact_iso}/{flash_end_iso}",
        "position":     {"cartographicDegrees": [impact_lon, impact_lat, 500.0]},
        "ellipsoid": {
            "radii": {
                "epoch": impact_iso,
                "cartesian": [
                    0,   0,      0,      50,
                    0.5, 8000,   8000,   200,
                    2,   15000,  15000,  300,
                    4,   18000,  18000,  100,
                ],
                "interpolationAlgorithm": "LINEAR",
            },
            "material": {
                "solidColor": {
                    "color": {
                        "epoch": impact_iso,
                        "rgba": [
                            0,   255, 255, 255, 240,
                            0.5, 255, 200,  50, 180,
                            2,   255, 120,   0,  80,
                            4,     0,   0,   0,   0,
                        ],
                    }
                }
            },
            "fill":         True,
            "outline":      True,
            "outlineColor": {"rgba": [255, 200, 50, 160]},
            "outlineWidth": 3.0,
        },
    })

    # ── 파괴 마커: 타격 후 X 아이콘 + 회색 라벨 (시뮬레이션 종료까지 유지) ──
    destroyed_iso = (impact_time + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    destroy_label = "파괴" if atype == "attack_uav" else "타격 완료"
    packets.append({
        "id":           f"destroyed_{asset_id}",
        "name":         destroy_label,
        "availability": f"{impact_iso}/{destroyed_iso}",
        "position":     {"cartographicDegrees": [impact_lon, impact_lat, 5.0]},
        "billboard": {
            "image":           _icon_destroyed(),
            "scale":           1.6,
            "verticalOrigin":  "CENTER",
            "heightReference": "CLAMP_TO_GROUND",
            "color":           {"rgba": [220, 80, 80, 230]},
        },
        "label": {
            "text":            f"[{destroy_label}]",
            "font":            "bold 11px 'Segoe UI'",
            "fillColor":       {"rgba": [180, 60, 60, 220]},
            "outlineColor":    {"rgba": [0, 0, 0, 200]},
            "outlineWidth":    2,
            "style":           "FILL_AND_OUTLINE",
            "verticalOrigin":  "BOTTOM",
            "heightReference": "CLAMP_TO_GROUND",
            "pixelOffset":     {"cartesian2": [0, -18]},
            "distanceDisplayCondition": {"distanceDisplayCondition": [0, 1.5e6]},
        },
    })


def _hide_destroyed_threats(packets: list, end_str: str) -> None:
    """
    파괴 마커(destroyed_*)가 있는 위치와 가장 가까운 위협(threat_*)을 매칭하여
    타격 시점 이후 위협 원(ellipse)·라벨·모델을 숨깁니다.

    CZML 패킷 머지 동작:
      동일 id의 패킷은 시간 인터벌 단위로 속성이 병합됩니다.
      hide_interval 동안 show=false 를 덮어쓰면 원(circle)이 사라집니다.
    """
    # ── 1. 파괴 마커 수집: impact 위치 + 시각 ──────────────────────
    destroyed_markers = []
    for pkt in packets:
        pid = pkt.get("id", "")
        if not pid.startswith("destroyed_"):
            continue
        avail = pkt.get("availability", "")
        if "/" not in avail:
            continue
        impact_iso = avail.split("/")[0]
        pos = pkt.get("position", {}).get("cartographicDegrees", [])
        if len(pos) >= 3:
            destroyed_markers.append({
                "lon":        pos[0],
                "lat":        pos[1],
                "impact_iso": impact_iso,
            })

    if not destroyed_markers:
        return

    # ── 2. 위협 엔티티 수집 (NFZ 제외, ellipse 있는 것만) ──────────
    # ellipse.semiMajorAxis는 위협 반경(meter). 이 값을 함께 보존해야 아래 동적
    # 매칭 임계(반경의 50%)가 의미를 갖는다. 이전엔 lon/lat만 저장하여 1420줄의
    # `threat_entities[best_tid].get("radius_km", 30.0)`이 항상 30km로 폴백되어
    # 동적 임계 로직이 사실상 무효였음.
    threat_entities = {}
    for pkt in packets:
        pid = pkt.get("id", "")
        if not pid.startswith("threat_") or "nfz" in pid.lower():
            continue
        ellipse = pkt.get("ellipse")
        if not ellipse:
            continue
        pos = pkt.get("position", {}).get("cartographicDegrees", [])
        if len(pos) >= 3:
            # semiMajorAxis는 m 단위. km로 변환해 보존.
            semi_m = float(ellipse.get("semiMajorAxis", 30000.0) or 30000.0)
            threat_entities[pid] = {
                "lon": pos[0],
                "lat": pos[1],
                "radius_km": semi_m / 1000.0,
            }

    if not threat_entities:
        return

    # ── 3. 매칭 → 숨김 패킷 추가 (55 km 이내 가장 가까운 위협) ────
    used_threats: set = set()
    for dm in destroyed_markers:
        best_tid  = None
        best_dist = float("inf")
        for tid, thr in threat_entities.items():
            if tid in used_threats:
                continue
            dlat = (dm["lat"] - thr["lat"]) * LAT_TO_KM
            dlon = (dm["lon"] - thr["lon"]) * LAT_TO_KM * math.cos(
                math.radians((dm["lat"] + thr["lat"]) * 0.5)
            )
            dist = math.sqrt(dlat ** 2 + dlon ** 2)
            if dist < best_dist:
                best_dist = dist
                best_tid  = tid

        # Long-range SAMs (예: SA-5 250km, KN-06 150km)는 55km 임계로는 매칭이
        # 어려워 파괴 후에도 지도에 영구 표시되는 결함이 있었음. 위협 반경에
        # 비례한 동적 임계(min 55km, 위협 반경의 50%)로 확장.
        if best_tid is not None:
            best_radius = float(threat_entities[best_tid].get("radius_km", 30.0) or 30.0)
            match_threshold = max(55.0, 0.5 * best_radius)
        else:
            match_threshold = 55.0
        if best_tid is not None and best_dist <= match_threshold:
            used_threats.add(best_tid)
            hide_iv = f"{dm['impact_iso']}/{end_str}"
            # 동일 id 패킷 → CZML이 시간 인터벌별로 병합함
            packets.append({
                "id": best_tid,
                "ellipse": {
                    "show": [{"interval": hide_iv, "boolean": False}]
                },
                "label": {
                    "show": [{"interval": hide_iv, "boolean": False}]
                },
                "model": {
                    "show": [{"interval": hide_iv, "boolean": False}]
                },
            })


def _sanitize_value(obj):
    """NaN/Infinity를 0으로 치환하여 JSON 호환성 보장"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_value(v) for v in obj]
    return obj


def czml_to_json_str(packets: List[dict]) -> str:
    """CZML 패킷 리스트 → JSON 문자열"""
    return json.dumps(_sanitize_value(packets), ensure_ascii=False, indent=2)
