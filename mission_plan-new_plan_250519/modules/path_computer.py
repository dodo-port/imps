"""
Path computation helpers extracted from streamlit_app.py (Phase 3a refactor).

Covers:
- Asset objective resolution (SEAD, ISR, STRIKE)
- Formation path computation (per-asset A*/RRT pathfinding)
- Single path computation
- Mission risk analysis aggregation
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Dict, List, Optional, Tuple

from modules.config import AIRPORTS, MAP_BOUNDS, ASSET_PERFORMANCE, LAT_TO_KM, MIN_ALTITUDE_AGL
from modules.path_constraints import path_satisfies
from modules.pathfinder import smooth_path, smooth_path_3d
from modules.xai_utils import XAIUtils

# ================================================================
# 경로 계산 캐시 (세션 내 동일 구간 재계산 방지)
# ================================================================
_PATH_CACHE: Dict[str, list] = {}
_PATH_CACHE_MAX = 64  # 최대 64개 구간 캐시


def _path_cache_key(
    algo: str, start: list, end: list,
    threats: list, margin: float,
    fuel_state: float, refuel_count: int, enable_3d: bool,
) -> str:
    """경로 요청의 해시 키 생성 (소수점 3자리로 정규화)"""
    sig = json.dumps({
        "a": algo,
        "s": [round(x, 3) for x in start],
        "e": [round(x, 3) for x in end],
        "m": round(margin, 2),
        "f": round(fuel_state, 2),
        "r": refuel_count,
        "3d": enable_3d,
        "t": sorted(
            [(round(t.get("lat", 0), 3), round(t.get("lon", 0), 3),
              t.get("type", ""), round(t.get("radius_km", 0), 1),
              t.get("name", ""))
             for t in threats],
            key=lambda x: x[0],
        ),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(sig.encode()).hexdigest()


def clear_path_cache() -> None:
    """경로 캐시 초기화 (임무 파라미터 변경 시 호출)"""
    _PATH_CACHE.clear()


# ================================================================
# Geo utilities
# ================================================================

def _dist_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = (lat1 - lat2) * LAT_TO_KM
    dlon = (lon1 - lon2) * LAT_TO_KM * math.cos(math.radians((lat1 + lat2) * 0.5))
    return math.sqrt(dlat * dlat + dlon * dlon)


def _clamp_coord(lat: float, lon: float) -> List[float]:
    return [
        max(MAP_BOUNDS["min_lat"], min(MAP_BOUNDS["max_lat"], lat)),
        max(MAP_BOUNDS["min_lon"], min(MAP_BOUNDS["max_lon"], lon)),
    ]


def _offset_coord(lat: float, lon: float, bearing_deg: float, distance_km: float) -> List[float]:
    theta = math.radians(bearing_deg)
    dlat = (distance_km / LAT_TO_KM) * math.cos(theta)
    lon_scale = max(0.1, LAT_TO_KM * math.cos(math.radians(lat)))
    dlon = (distance_km / lon_scale) * math.sin(theta)
    return _clamp_coord(lat + dlat, lon + dlon)


def _asset_serial(asset_id: str) -> int:
    digits = "".join(ch for ch in (asset_id or "") if ch.isdigit())
    return max(1, int(digits or "1"))


# ================================================================
# Asset objective resolution
# ================================================================

def _select_sead_targets(formation, threats_dict: List[dict], target_coord: List[float]) -> dict:
    if not formation or not getattr(formation, "assets", None):
        return {}

    sead_assets = [
        a for a in formation.assets
        if (a.assigned_mission or "").upper() == "SEAD"
    ]
    sead_assets.sort(key=lambda a: (0 if a.asset_type == "attack_uav" else 1, _asset_serial(a.asset_id)))

    candidates = [
        t for t in threats_dict
        if t.get("type") in ("SAM", "RADAR") and t.get("lat") is not None and t.get("lon") is not None
    ]
    candidates.sort(
        key=lambda t: (
            0 if t.get("type") == "SAM" else 1,
            _dist_km(t["lat"], t["lon"], target_coord[0], target_coord[1]),
            -float(t.get("radius_km", 0.0)),
        )
    )
    if not sead_assets or not candidates:
        return {}

    mapping = {}
    for idx, asset in enumerate(sead_assets):
        mapping[asset.asset_id] = candidates[idx % len(candidates)]
    return mapping


def _build_recon_objective(asset, formation, threats_dict: List[dict], target_coord: List[float]) -> List[float]:
    recon_assets = []
    if formation and getattr(formation, "assets", None):
        recon_assets = [a for a in formation.assets if a.asset_type == "recon_uav"]
    recon_assets.sort(key=lambda a: _asset_serial(a.asset_id))

    recon_idx = 0
    for idx, recon_asset in enumerate(recon_assets):
        if recon_asset.asset_id == asset.asset_id:
            recon_idx = idx
            break

    centers = [(t["lat"], t["lon"]) for t in threats_dict if t.get("lat") is not None and t.get("lon") is not None]
    if centers:
        center_lat = sum(p[0] for p in centers) / len(centers)
        center_lon = sum(p[1] for p in centers) / len(centers)
        max_radius = max(float(t.get("radius_km", 30.0)) for t in threats_dict if t.get("lat") is not None and t.get("lon") is not None)
        scan_radius = max(35.0, min(110.0, max_radius * 0.75 + 10.0))
    else:
        center_lat, center_lon = target_coord[:2]
        scan_radius = 60.0

    bearings = [210.0, 270.0, 330.0, 30.0, 90.0, 150.0]
    return _offset_coord(center_lat, center_lon, bearings[recon_idx % len(bearings)], scan_radius)


def _resolve_asset_objective(
    asset,
    formation,
    threats_dict: List[dict],
    target_coord: List[float],
    sead_targets: dict,
) -> Tuple[List[float], Optional[str]]:
    mission_name = (asset.assigned_mission or "").upper()
    if mission_name == "SEAD":
        assigned_threat = sead_targets.get(asset.asset_id)
        if assigned_threat:
            return [assigned_threat["lat"], assigned_threat["lon"]], assigned_threat.get("name")
    if mission_name == "ISR" or asset.asset_type == "recon_uav":
        return _build_recon_objective(asset, formation, threats_dict, target_coord), None
    return list(target_coord[:2]), None


def _threats_for_asset(asset, threats_dict: List[dict], sead_targets: dict) -> List[dict]:
    mission_name = (asset.assigned_mission or "").upper()
    assigned_threat = sead_targets.get(asset.asset_id)
    if mission_name == "SEAD" and assigned_threat:
        return [t for t in threats_dict if t.get("name") != assigned_threat.get("name")]

    if mission_name == "STRIKE" and sead_targets:
        suppressed = {t.get("name") for t in sead_targets.values() if t.get("name")}
        return [t for t in threats_dict if t.get("name") not in suppressed]
    return threats_dict


def _asset_risk_weight(asset_type: str, mission_name: str) -> float:
    mission_name = (mission_name or "").upper()
    if mission_name == "ISR":
        return 0.55 if asset_type == "recon_uav" else 0.70
    if mission_name == "SEAD":
        return 0.75 if asset_type == "attack_uav" else 0.90
    if mission_name == "CAS":
        return 0.95
    return 1.0


# ================================================================
# Low-level path runner
# ================================================================

def _run_one_path(
    pathfinder,
    effective_algorithm: str,
    start: list,
    end: list,
    threats: list,
    margin: float,
    fuel_state: float,
    refuel_count: int,
    enable_3d: bool,
) -> list:
    """Call the appropriate pathfinder method and return a raw waypoint list."""
    if effective_algorithm == "A* 3D":
        return pathfinder.find_path_3d_fast(
            start, end, threats, margin,
            fuel_state=fuel_state, refuel_count=refuel_count,
        )
    if hasattr(pathfinder, "find_path_3d") and enable_3d:
        return pathfinder.find_path_3d(
            start, end, threats, margin,
            fuel_state=fuel_state, refuel_count=refuel_count,
        )
    if effective_algorithm == "A*":
        return pathfinder.find_path(
            start[:2], end[:2], threats, margin,
            fuel_state=fuel_state, refuel_count=refuel_count,
        )
    try:
        return pathfinder.find_path(
            start[:2], end[:2], threats, margin,
            fuel_state=fuel_state, refuel_count=refuel_count,
        )
    except TypeError:
        return pathfinder.find_path(start[:2], end[:2], threats, margin)


def _run_one_path_cached(
    pathfinder,
    effective_algorithm: str,
    start: list,
    end: list,
    threats: list,
    margin: float,
    fuel_state: float,
    refuel_count: int,
    enable_3d: bool,
) -> list:
    """캐시 래퍼 ? 동일 구간 재계산 스킵"""
    key = _path_cache_key(
        effective_algorithm, start, end, threats,
        margin, fuel_state, refuel_count, enable_3d,
    )
    if key in _PATH_CACHE:
        return list(_PATH_CACHE[key])   # 복사본 반환 (부작용 방지)

    result = _run_one_path(
        pathfinder, effective_algorithm, start, end,
        threats, margin, fuel_state, refuel_count, enable_3d,
    )
    # LRU-style: 최대치 초과 시 가장 오래된 항목 제거
    if len(_PATH_CACHE) >= _PATH_CACHE_MAX:
        del _PATH_CACHE[next(iter(_PATH_CACHE))]
    _PATH_CACHE[key] = result
    return result


def _smooth(raw: list, point_is_valid=None, path_is_valid=None) -> list:
    if not raw:
        return []
    if len(raw[0]) >= 3:
        candidate = smooth_path_3d(
            raw,
            point_is_valid=point_is_valid,
            path_is_valid=path_is_valid,
        )
    else:
        candidate = smooth_path(raw)
    if path_is_valid is None or path_is_valid(candidate):
        return candidate
    if path_is_valid(raw):
        return raw
    return []


RECON_LOITER_RADIUS_KM = 12.0


def _build_recon_circle(
    center: list,
    radius_km: float = RECON_LOITER_RADIUS_KM,
    n_pts: int = 8,
    enable_3d: bool = False,
    terrain_fast=None,
) -> list:
    """
    ISR 정찰 선회 경로 (center 주변 원형 waypoints).
    실제 정찰기는 위협 지역 외곽을 반복 선회하며 센서로 탐지.
    """
    circle = []
    for i in range(n_pts + 1):  # +1: 첫 점 복귀 → 완전한 원
        angle = 360.0 * i / n_pts
        pt = _offset_coord(center[0], center[1], angle, radius_km)
        if enable_3d and terrain_fast is not None:
            try:
                elev = terrain_fast.get(*pt[:2]) if hasattr(terrain_fast, "get") else 0.0
            except Exception:
                elev = 0.0
            alt = ASSET_PERFORMANCE.get("recon_uav", {}).get("altitude_m", 3000)
            circle.append([pt[0], pt[1], float(elev) + alt * 0.45])
        else:
            circle.append(list(pt))
    return circle


# ================================================================
# Public: formation path computation
# ================================================================

# High-contrast 8-color palette (color-blind friendly)
ASSET_PALETTE = [
    "#E53935",  # 1: bright red    - Eagle-1
    "#43A047",  # 2: bright green  - Scout-1
    "#FB8C00",  # 3: bright orange - Scout-2
    "#8E24AA",  # 4: bright purple - Scout-3
    "#F9A825",  # 5: golden yellow - Scout-4
    "#00897B",  # 6: deep teal     - Viper-1
    "#D81B60",  # 7: crimson pink  - Viper-2
    "#1565C0",  # 8: deep blue     - extra
]


def compute_formation_paths(
    mission,
    pathfinder,
    effective_algorithm: str,
    terrain_fast,
    threats_dict: List[dict],
    fuel_state: float,
    refuel_count: int,
) -> Dict[str, dict]:
    """
    Compute per-asset ingress/egress paths for a formation.

    Returns:
        {asset_id: {"in": path, "out": path, "color": str, "callsign": str,
                    "mission": str, "type": str, "objective": str, "threats": list}}
    """
    formation = mission.formation
    # 기본 목표 (우선순위 1)
    primary_coord = [mission.params.target_lat, mission.params.target_lon]
    primary_name  = mission.params.target_name
    # 추가 목표 리스트 (우선순위 순 정렬)
    extra = sorted(getattr(mission.params, "extra_targets", []), key=lambda t: t.get("priority", 2))
    extra = [t for t in extra if "lat" in t and "lon" in t]   # KeyError 방어
    all_target_coords = [primary_coord] + [[float(t["lat"]), float(t["lon"])] for t in extra]
    all_target_names  = [primary_name]  + [t.get("name", f"Target-{i+2}") for i, t in enumerate(extra)]

    margin = mission.params.margin
    rtb = mission.params.rtb
    enable_3d = mission.params.enable_3d

    sead_targets = _select_sead_targets(formation, threats_dict, primary_coord)
    formation_paths: Dict[str, dict] = {}

    for asset_idx, asset in enumerate(formation.assets):
        # 자산에 배정된 목표 인덱스 (헝가리안 결과, 없으면 0=기본목표)
        t_idx = asset.assigned_target_idx if asset.assigned_target_idx is not None else 0
        t_idx = min(t_idx, len(all_target_coords) - 1)
        target_coord = all_target_coords[t_idx]

        _base_info = AIRPORTS.get(asset.base) or AIRPORTS.get(mission.params.start) or {}
        asset_base_coords = _base_info.get("coords", [35.179, 129.075])
        a_start = list(asset_base_coords)
        mission_name = (asset.assigned_mission or "").upper()
        objective_coord, objective_threat_name = _resolve_asset_objective(
            asset, formation, threats_dict, target_coord, sead_targets,
        )
        asset_threats = _threats_for_asset(asset, threats_dict, sead_targets)

        def _point_is_valid(point) -> bool:
            try:
                if enable_3d and len(point) >= 3:
                    if hasattr(pathfinder, "is_terrain_collision") and pathfinder.is_terrain_collision(*point[:3]):
                        return False
                    if not hasattr(pathfinder, "is_terrain_collision") and terrain_fast is not None:
                        if hasattr(terrain_fast, "get"):
                            ground = terrain_fast.get(*point[:2])
                        else:
                            ground = terrain_fast.get_elevation(*point[:2])
                        if float(point[2]) < float(ground) + MIN_ALTITUDE_AGL:
                            return False
                    if hasattr(pathfinder, "is_collision_3d"):
                        return not pathfinder.is_collision_3d(
                            *point[:3], asset_threats, asset_margin, fuel_state, refuel_count
                        )
                elif hasattr(pathfinder, "is_collision"):
                    return not pathfinder.is_collision(
                        *point[:2], asset_threats, asset_margin, fuel_state, refuel_count
                    )
            except Exception:
                return False
            return True

        def _path_is_valid(candidate) -> bool:
            return bool(candidate) and path_satisfies(candidate, _point_is_valid)

        # ── 자산별 안전 마진 결정 ─────────────────────────────────
        # 공격 UAV (자폭): 소모 가능 → 마진 최소화, 직선 침투
        # 정찰 UAV: 생존 필요 → 일반 마진 유지
        # 전투기: 조종사 생존 최우선 → 일반 마진 유지
        if asset.asset_type == "attack_uav":
            asset_margin = max(1.0, margin * 0.15)   # 원래 마진의 15% (최소 1km)
        else:
            asset_margin = margin

        # ── 전투기: 표적 분담 (헝가리안 할당 우선) ─────────────
        # 이전엔 모든 fighter가 동일한 [t1, t2, ..., tN] chain을 따라가서
        # N대 fighter가 시각화상 같은 경로/거리/위험도로 겹쳐 보이는 결함이 있었음.
        # 정책:
        #   - fighter 1대 + 표적 N개: 모든 표적 chain (단일 자산이 모두 타격)
        #   - fighter N대 + 표적 M개:
        #       · 각 fighter는 자기 assigned_target_idx 1개 우선
        #       · 미할당 표적은 첫 fighter(asset_id 사전순)에 chain 추가
        is_fighter = (asset.asset_type == "fighter")
        is_expendable = (asset.asset_type == "attack_uav")

        _all_fighters = [
            a for a in formation.assets if a.asset_type == "fighter"
        ]
        _first_fighter_id = (
            sorted(_all_fighters, key=lambda f: f.asset_id)[0].asset_id
            if _all_fighters else None
        )

        # 이 fighter가 chain으로 처리할 표적 좌표 리스트.
        my_chain_target_coords: List[List[float]] = []
        if is_fighter and len(all_target_coords) > 1:
            if len(_all_fighters) == 1:
                # 단일 fighter — 모든 표적 chain (기존 동작)
                my_chain_target_coords = list(all_target_coords)
            else:
                # 다중 fighter — 자기 assigned_target_idx 우선
                my_idx = asset.assigned_target_idx
                if my_idx is not None and 0 <= my_idx < len(all_target_coords):
                    my_chain_target_coords = [all_target_coords[my_idx]]
                else:
                    my_chain_target_coords = [list(target_coord)]
                # 미할당 표적: 첫 fighter가 추가 chain
                if asset.asset_id == _first_fighter_id:
                    assigned_set = {
                        f.assigned_target_idx for f in _all_fighters
                        if f.assigned_target_idx is not None
                    }
                    for ti in range(len(all_target_coords)):
                        if ti not in assigned_set:
                            my_chain_target_coords.append(list(all_target_coords[ti]))

        def _with_alt_3d(coord2d: list) -> list:
            if not enable_3d:
                return list(coord2d[:2])
            # terrain_fast 조회는 좌표가 한반도 밖이거나 loader가 None이면
            # 예외/None을 반환할 수 있다. silent fallback으로 0m를 사용해
            # AttributeError로 자산 경로가 통째로 누락되는 것을 방지.
            try:
                elev = terrain_fast.get(*coord2d[:2]) if terrain_fast else 0.0
                if elev is None:
                    elev = 0.0
            except Exception:
                elev = 0.0
            alt_offset = ASSET_PERFORMANCE.get(asset.asset_type, {}).get("altitude_m", 800)
            return [coord2d[0], coord2d[1], float(elev) + alt_offset * 0.3]

        if is_fighter and len(my_chain_target_coords) >= 1 and len(all_target_coords) > 1:
            # 체인 경로: 기지 → my_chain_target_coords[0] → ... → my_chain_target_coords[N]
            # my_chain_target_coords는 위에서 fighter 수에 따라 분담 결정됨.
            chain_waypoints = (
                [_with_alt_3d(a_start)]
                + [_with_alt_3d(tc) for tc in my_chain_target_coords]
            )
            path_in: list = []
            for i in range(len(chain_waypoints) - 1):
                seg_s = list(path_in[-1]) if path_in else chain_waypoints[i]
                seg_e = chain_waypoints[i + 1]
                try:
                    raw_seg = _run_one_path_cached(
                        pathfinder, effective_algorithm, seg_s, seg_e,
                        asset_threats, asset_margin, fuel_state, refuel_count, enable_3d,
                    )
                    smoothed = _smooth(raw_seg, _point_is_valid, _path_is_valid)
                    if smoothed:
                        path_in.extend(smoothed[1:] if path_in else smoothed)
                except Exception:
                    pass
        else:
            a_target = _with_alt_3d(list(objective_coord[:2]))
            try:
                raw = _run_one_path_cached(
                    pathfinder, effective_algorithm, _with_alt_3d(a_start), a_target,
                    asset_threats, asset_margin, fuel_state, refuel_count, enable_3d,
                )
                path_in = _smooth(raw, _point_is_valid, _path_is_valid)
            except Exception:
                path_in = []

            # ISR: 목표 지역 도달 후 정찰 선회 패턴 추가
            # 실제 정찰기는 위협 외곽을 원형 비행하며 SAM/RADAR 위치 확인
            if (mission_name == "ISR" or asset.asset_type == "recon_uav") and path_in:
                circle = _build_recon_circle(
                    path_in[-1],
                    radius_km=RECON_LOITER_RADIUS_KM,
                    n_pts=8,
                    enable_3d=enable_3d,
                    terrain_fast=terrain_fast,
                )
                candidate = path_in + circle
                if _path_is_valid(candidate):
                    path_in = candidate

        # 정찰 UAV는 RTB 필수 (생존 필요), 공격 UAV는 편도
        path_out: list = []
        if rtb and path_in and not is_expendable:
            try:
                raw_o = _run_one_path_cached(
                    pathfinder, effective_algorithm, path_in[-1], _with_alt_3d(a_start),
                    asset_threats, asset_margin, fuel_state, refuel_count, enable_3d,
                )
                path_out = _smooth(raw_o, _point_is_valid, _path_is_valid)
            except Exception:
                path_out = []

        color = ASSET_PALETTE[asset_idx % len(ASSET_PALETTE)]
        # ── objective 라벨 결정 ──────────────────────────────────
        if objective_threat_name:
            # SEAD: SAM/RADAR 이름을 직접 표시
            obj_label = objective_threat_name
        elif mission_name == "ISR" or asset.asset_type == "recon_uav":
            # ISR: 정찰 위치를 명시 (실제 좌표는 _build_recon_objective로 오프셋됨)
            ref_name = all_target_names[min(t_idx, len(all_target_names) - 1)]
            obj_label = f"{ref_name} 주변 정찰"
        elif is_fighter and len(all_target_coords) > 1:
            # 전투기 체인 타격: 모든 목표 표시
            obj_label = " → ".join(all_target_names)
        else:
            obj_label = all_target_names[min(t_idx, len(all_target_names) - 1)]

        # 전투기 체인 타격: 자기에게 할당된 표적 좌표만 CZML 폭발 효과에 전달.
        # 이전엔 모든 fighter가 전체 표적 리스트를 받아 다중 fighter 시나리오에서
        # 같은 표적에 중복 폭발 효과가 표시되는 결함이 있었음.
        chain_target_coords = (
            my_chain_target_coords
            if (is_fighter and len(my_chain_target_coords) >= 1 and len(all_target_coords) > 1)
            else None
        )

        formation_paths[asset.asset_id] = {
            "in": path_in,
            "out": path_out,
            "color": color,
            "callsign": asset.callsign,
            "mission": asset.assigned_mission or "?",
            "type": asset.asset_type,
            "objective": obj_label,
            "target_idx": t_idx,
            "threats": asset_threats,
            "expendable": is_expendable,          # 자폭 UAV 여부 (편도·소모 가능)
            "asset_margin": asset_margin,          # 실제 적용 마진 (자산별 상이)
            "chain_target_coords": chain_target_coords,  # 전투기 다중 목표 좌표
        }

    return formation_paths


# ================================================================
# Public: single path computation
# ================================================================

def compute_single_path(
    mission,
    pathfinder,
    effective_algorithm: str,
    start_coord: list,
    target_coord: list,
    threats_dict: List[dict],
    fuel_state: float,
    refuel_count: int,
    terrain_loader=None,
) -> Tuple[list, list]:
    """
    Compute ingress (and optional egress) path for a solo mission.
    다중 목표(extra_targets)가 있으면 start → t1 → t2 → t3 → RTB 체인으로 연결.

    Args:
        terrain_loader: 3D 모드에서 추가 목표 좌표의 지형 고도 조회용. 없거나
            조회 실패 시 0m 폴백. 이전 구현은 `mission._terrain`만 시도했지만
            streamlit_app은 terrain을 mission 객체에 부착하지 않고
            `st.session_state.terrain`에 보관하므로 항상 0m로 떨어지는
            결함이 있었음. 호출자가 명시적으로 terrain_fast를 넘기도록 한다.

    Returns:
        (final_in, final_out)
    """
    margin = mission.params.margin
    rtb = mission.params.rtb
    enable_3d = mission.params.enable_3d

    # ── 다중 목표 체인 구성 ───────────────────────────────────────
    extra = sorted(
        [t for t in getattr(mission.params, "extra_targets", []) if "lat" in t and "lon" in t],
        key=lambda t: t.get("priority", 2),
    )
    extra_coords = [[float(t["lat"]), float(t["lon"])] for t in extra]

    # terrain 폴백 우선순위: terrain_loader 인자 → mission._terrain → 0m
    _terrain_src = terrain_loader if terrain_loader is not None else getattr(mission, "_terrain", None)

    def _with_alt(coord: list) -> list:
        if not enable_3d or len(coord) >= 3:
            return coord
        elev = 0.0
        if _terrain_src is not None:
            try:
                if hasattr(_terrain_src, "get_elevation"):
                    elev = _terrain_src.get_elevation(coord[0], coord[1])
                elif hasattr(_terrain_src, "get"):
                    elev = _terrain_src.get(coord[0], coord[1])
            except Exception:
                elev = 0.0
            if elev is None:
                elev = 0.0
        return [coord[0], coord[1], float(elev) + 500]

    # 전체 경유 순서: start → primary → extra[0] → extra[1] → ... (3D 고도 레이어 적용)
    waypoints = [_with_alt(start_coord), _with_alt(target_coord)] + [_with_alt(c) for c in extra_coords]

    # 구간별 경로 계산 후 이어 붙이기
    final_in: list = []
    for i in range(len(waypoints) - 1):
        seg_start = waypoints[i]
        # 이전 구간 끝점을 시작점으로 사용 (누적 오차 방지)
        if final_in:
            seg_start = list(final_in[-1])
        seg_end = waypoints[i + 1]
        try:
            raw_seg = _run_one_path(
                pathfinder, effective_algorithm, seg_start, seg_end,
                threats_dict, margin, fuel_state, refuel_count, enable_3d,
            )
            smoothed = _smooth(raw_seg)
            if smoothed:
                # 중복 접합점 제거: 이전 끝점과 새 시작점이 같으면 첫 점 제거
                if final_in and smoothed:
                    final_in.extend(smoothed[1:])
                else:
                    final_in.extend(smoothed)
        except Exception:
            pass

    final_out: list = []
    if rtb and final_in:
        try:
            raw_out = _run_one_path(
                pathfinder, effective_algorithm, list(final_in[-1]), _with_alt(start_coord),
                threats_dict, margin, fuel_state, refuel_count, enable_3d,
            )
            final_out = _smooth(raw_out)
        except Exception:
            final_out = []

    return final_in, final_out


# ================================================================
# Public: mission risk analysis
# ================================================================

def analyze_mission_risk(
    formation_paths: Dict[str, dict],
    final_in: list,
    final_out: list,
    threats_dict: List[dict],
    margin: float,
    terrain_fast,
) -> Tuple[dict, List[dict]]:
    """
    Aggregate risk analysis across all assets (or a single path).

    Returns:
        (risk_report, per_asset_reports)
        risk_report keys: avg_risk, max_risk, package_max_risk, high_risk_segments, total_length_km
        per_asset_reports: list of per-asset dicts (empty if no formation)
    """
    per_asset_reports: List[dict] = []

    if formation_paths:
        weighted_avg_sum = 0.0
        weighted_count = 0.0
        total_length_km = 0.0
        total_high_risk = 0
        max_risk = 0.0
        package_max_risk = 0.0

        for asset_id, pdata in formation_paths.items():
            combined_path = (pdata.get("in") or []) + (pdata.get("out") or [])
            if not combined_path:
                continue
            report_threats = pdata.get("threats", threats_dict)
            rr = XAIUtils.analyze_path_risk(combined_path, report_threats, margin, terrain_loader=terrain_fast)
            mission_name = pdata.get("mission", "?")
            asset_type = pdata.get("type", "?")
            risk_weight = _asset_risk_weight(asset_type, mission_name)
            adjusted_max_risk = float(rr.get("max_risk", 0.0)) * risk_weight
            adjusted_avg_risk = float(rr.get("avg_risk", 0.0)) * risk_weight
            per_asset_reports.append({
                "asset_id": asset_id,
                "callsign": pdata.get("callsign", asset_id),
                "mission": mission_name,
                "type": asset_type,
                "objective": pdata.get("objective", "-"),
                "distance_km": float(rr.get("total_length_km", 0.0)),
                "avg_risk": float(rr.get("avg_risk", 0.0)),
                "max_risk": float(rr.get("max_risk", 0.0)),
                "adjusted_max_risk": adjusted_max_risk,
                "risk_weight": risk_weight,
                "high_risk_segments": int(rr.get("high_risk_segments", 0)),
            })
            w = max(float(rr.get("total_length_km", 0.0)), 0.1)
            weighted_avg_sum += adjusted_avg_risk * w
            weighted_count += w
            total_length_km += rr.get("total_length_km", 0.0)
            total_high_risk += int(rr.get("high_risk_segments", 0))
            max_risk = max(max_risk, adjusted_max_risk)
            package_max_risk = max(package_max_risk, float(rr.get("max_risk", 0.0)))

        if per_asset_reports:
            # ── 전투기 단독 메트릭 (조종사 생존성 기준) ───────────
            fighter_reports = [r for r in per_asset_reports if r.get("type") == "fighter"]
            if fighter_reports:
                fighter_max_risk    = max(r["max_risk"]    for r in fighter_reports)
                fighter_avg_risk    = sum(r["avg_risk"]    for r in fighter_reports) / len(fighter_reports)
                fighter_distance_km = sum(r["distance_km"] for r in fighter_reports)
            else:
                # 단독 임무 시 전체 경로를 전투기 기준으로 사용
                fighter_max_risk    = package_max_risk
                fighter_avg_risk    = (weighted_avg_sum / weighted_count) if weighted_count > 0 else 0.0
                fighter_distance_km = total_length_km

            risk_report = {
                "avg_risk": (weighted_avg_sum / weighted_count) if weighted_count > 0 else 0.0,
                "max_risk": max_risk,
                "package_max_risk": package_max_risk,
                "high_risk_segments": total_high_risk,
                "total_length_km": total_length_km,
                # 전투기 전용 메트릭
                "fighter_max_risk": fighter_max_risk,
                "fighter_avg_risk": fighter_avg_risk,
                "fighter_distance_km": fighter_distance_km,
            }
        else:
            risk_report = XAIUtils.analyze_path_risk(
                final_in + final_out, threats_dict, margin, terrain_loader=terrain_fast
            )
    else:
        risk_report = XAIUtils.analyze_path_risk(
            final_in + final_out, threats_dict, margin, terrain_loader=terrain_fast
        )

    return risk_report, per_asset_reports
