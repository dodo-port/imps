"""
Mission Timeline - 작전 시간대별 전력 위치 및 위협 상태 추적.

경로 + 자산 속도를 이용해 시간축 상의 위치를 보간하고,
SEAD → 억압/파괴, STRIKE 도달 등의 이벤트를 자동 생성한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from modules.config import ASSET_PERFORMANCE, LAT_TO_KM


# ================================================================
# Data structures
# ================================================================

@dataclass
class MissionEvent:
    time_min: float
    event_type: str     # LAUNCH / SEAD_ARRIVE / SUPPRESSED / DESTROYED / STRIKE_HIT / ISR_ON_STATION / EGRESS / RTB
    asset_id: str
    callsign: str
    description: str
    lat: float
    lon: float
    icon: str


@dataclass
class AssetState:
    lat: float
    lon: float
    alt: float
    phase: str          # "standby" / "ingress" / "on_target" / "egress" / "rtb"
    heading_deg: float


@dataclass
class TimeFrame:
    time_min: float
    asset_states: Dict[str, AssetState]
    threat_statuses: Dict[str, str]     # name -> "active" / "suppressed" / "destroyed"
    events_at: List[MissionEvent] = field(default_factory=list)


@dataclass
class MissionTimeline:
    total_duration_min: float
    frames: List[TimeFrame]
    events: List[MissionEvent]
    asset_info: Dict[str, dict]         # asset_id -> {callsign, mission, type, color, t_arrive_min, t_done_min}
    asset_timing: Dict[str, dict]       # internal timing data for Gantt


# ================================================================
# Geo helpers
# ================================================================

def _dist_km(p1, p2) -> float:
    lat1, lon1 = float(p1[0]), float(p1[1])
    lat2, lon2 = float(p2[0]), float(p2[1])
    dlat = (lat2 - lat1) * LAT_TO_KM
    dlon = (lon2 - lon1) * LAT_TO_KM * math.cos(math.radians((lat1 + lat2) * 0.5))
    dalt = ((float(p2[2]) - float(p1[2])) / 1000.0) if len(p1) >= 3 and len(p2) >= 3 else 0.0
    return math.sqrt(dlat ** 2 + dlon ** 2 + dalt ** 2)


def _cumulative_km(path: list) -> List[float]:
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + _dist_km(path[i - 1], path[i]))
    return cum


def _interpolate_pos(
    path: list,
    cum_km: List[float],
    target_km: float,
) -> Tuple[float, float, float]:
    """경로 상에서 target_km 지점의 (lat, lon, alt) 보간."""
    if not path:
        return 0.0, 0.0, 0.0
    if target_km <= 0:
        p = path[0]
        return float(p[0]), float(p[1]), float(p[2]) if len(p) >= 3 else 0.0
    if target_km >= cum_km[-1]:
        p = path[-1]
        return float(p[0]), float(p[1]), float(p[2]) if len(p) >= 3 else 0.0
    for i in range(1, len(cum_km)):
        if cum_km[i] >= target_km:
            seg = max(cum_km[i] - cum_km[i - 1], 1e-9)
            frac = (target_km - cum_km[i - 1]) / seg
            pa, pb = path[i - 1], path[i]
            lat = float(pa[0]) + frac * (float(pb[0]) - float(pa[0]))
            lon = float(pa[1]) + frac * (float(pb[1]) - float(pa[1]))
            alt_a = float(pa[2]) if len(pa) >= 3 else 500.0
            alt_b = float(pb[2]) if len(pb) >= 3 else 500.0
            return lat, lon, alt_a + frac * (alt_b - alt_a)
    p = path[-1]
    return float(p[0]), float(p[1]), float(p[2]) if len(p) >= 3 else 0.0


def _bearing(p1, p2) -> float:
    """북쪽 기준 방위각 (도)."""
    dlat = float(p2[0]) - float(p1[0])
    dlon = (float(p2[1]) - float(p1[1])) * math.cos(math.radians((float(p1[0]) + float(p2[0])) * 0.5))
    return math.degrees(math.atan2(dlon, dlat)) % 360.0


# ================================================================
# Main builder
# ================================================================

def build_timeline(
    formation_paths: Dict[str, dict],
    formation,
    threats: List[dict],
    time_resolution_min: float = 0.5,
) -> Optional[MissionTimeline]:
    """
    Args:
        formation_paths:  compute_formation_paths() 결과
        formation:        FormationResult (.assets)
        threats:          위협 dict 리스트 (name, lat, lon, type 포함)
        time_resolution_min: 타임프레임 간격 (분)
    Returns:
        MissionTimeline or None (데이터 없으면)
    """
    if not formation_paths or not formation or not getattr(formation, "assets", None):
        return None

    asset_map: Dict[str, object] = {a.asset_id: a for a in formation.assets}

    def _spd(asset_type: str) -> float:
        return float(ASSET_PERFORMANCE.get(asset_type, {}).get("speed_kmh", 300.0))

    # ── 1차 패스: 순수 비행시간(지연 없이) 계산 ──────────────────────
    _raw: Dict[str, dict] = {}
    for aid, pdata in formation_paths.items():
        asset_obj = asset_map.get(aid)
        if asset_obj is None:
            continue
        spd = max(_spd(asset_obj.asset_type), 1.0)  # ZeroDivisionError 방어
        path_in  = pdata.get("in",  []) or []
        path_out = pdata.get("out", []) or []
        cum_in   = _cumulative_km(path_in)  if path_in  else [0.0]
        cum_out  = _cumulative_km(path_out) if path_out else [0.0]
        dist_in  = cum_in[-1]
        dist_out = cum_out[-1]
        mission  = (getattr(asset_obj, "assigned_mission", None) or "").upper()
        _raw[aid] = {
            "mission":   mission,
            "spd":       spd,
            "dist_in":   dist_in,
            "dist_out":  dist_out,
            "t_fly_in":  dist_in  / spd * 60.0,
            "t_fly_out": dist_out / spd * 60.0 if path_out else 0.0,
            "path_in":   path_in,
            "path_out":  path_out,
            "cum_in":    cum_in,
            "cum_out":   cum_out,
        }

    # ── 작전 순서에 따른 출발 지연 계산 (ISR→SEAD→STRIKE) ─────────
    # ▶ ISR: 지연 없음 (즉시 출발)
    isr_arrive_times = [v["t_fly_in"] for v in _raw.values() if v["mission"] == "ISR"]
    max_isr_arrive = max(isr_arrive_times) if isr_arrive_times else 0.0

    # ▶ SEAD 실제 도착 시각 계산 (ISR 완료 후 출발 지연 포함)
    # 핵심 버그 수정: SEAD 자신의 launch_delay를 반영한 "실제 도착 시각" 기준으로
    # sead_suppress_done을 계산해야 STRIKE가 SEAD 완료 후 진입 가능
    sead_actual_arrivals = []
    for raw in _raw.values():
        if raw["mission"] == "SEAD":
            # SEAD는 ISR 도착 확인 후 5분 뒤 출발
            sead_launch_delay = max(0.0, max_isr_arrive - raw["t_fly_in"] + 5.0)
            sead_actual_arrivals.append(raw["t_fly_in"] + sead_launch_delay)

    # SEAD 억압 완료 = 가장 늦은 SEAD 실제 도착 + 교전 시간 5분
    # (SEAD 없으면 0 → STRIKE 지연 없음)
    sead_suppress_done = (max(sead_actual_arrivals) + 5.0) if sead_actual_arrivals else 0.0

    def _launch_delay(mission: str, t_fly_in: float) -> float:
        """작전 교리 순서에 따른 이륙 지연 (분).
        ISR: 즉시 출발 / SEAD: ISR 도착 후 출발 / STRIKE: SEAD 억압 완료 후 도착 보장.
        """
        if mission == "ISR":
            return 0.0
        if mission == "SEAD":
            # ISR 도착 확인 후 5분 뒤 출발 (동시 도착 방지)
            return max(0.0, max_isr_arrive - t_fly_in + 5.0)
        if mission in ("STRIKE", "CAS"):
            # SEAD 실제 억압 완료 시각에 맞춰 도착하도록 출발 지연
            # = sead_suppress_done - 내 비행시간 (이때 sead_suppress_done은 지연 포함 계산)
            return max(0.0, sead_suppress_done - t_fly_in)
        return 0.0

    # ── 2차 패스: 지연 적용 타이밍 DB 구성 ──────────────────────────
    timing_db: Dict[str, dict] = {}
    max_dur = 0.0

    for aid, raw in _raw.items():
        mission  = raw["mission"]
        spd      = raw["spd"]
        delay    = _launch_delay(mission, raw["t_fly_in"])

        t_arrive        = raw["t_fly_in"]  + delay
        t_on_target     = 3.0
        t_egress_start  = t_arrive + t_on_target
        t_done          = t_egress_start + raw["t_fly_out"]

        timing_db[aid] = {
            "speed_kmh":          spd,
            "path_in":            raw["path_in"],
            "path_out":           raw["path_out"],
            "cum_in":             raw["cum_in"],
            "cum_out":            raw["cum_out"],
            "dist_in_km":         raw["dist_in"],
            "dist_out_km":        raw["dist_out"],
            "launch_delay_min":   delay,
            "t_arrive_min":       t_arrive,
            "t_on_target_min":    t_on_target,
            "t_egress_start_min": t_egress_start,
            "t_done_min":         t_done,
        }
        max_dur = max(max_dur, t_done)

    if not timing_db:
        return None
    max_dur = max(max_dur, 1.0)

    # ── 이벤트 생성 ────────────────────────────────────────────────
    events: List[MissionEvent] = []

    for aid, tim in timing_db.items():
        asset_obj = asset_map.get(aid)
        if not asset_obj:
            continue
        callsign = asset_obj.callsign or aid
        mission  = (asset_obj.assigned_mission or "?").upper()
        pdata    = formation_paths[aid]
        path_in  = tim["path_in"]
        path_out = tim["path_out"]
        t_arrive = tim["t_arrive_min"]
        t_egress = tim["t_egress_start_min"]
        t_done   = tim["t_done_min"]

        # 이륙 (지연 적용)
        delay = tim["launch_delay_min"]
        if path_in:
            p = path_in[0]
            events.append(MissionEvent(delay, "LAUNCH", aid, callsign,
                f"{callsign} 이륙 ({mission})",
                float(p[0]), float(p[1]), "🛫"))

        # 목표 도착
        if path_in:
            ep = path_in[-1]
            _icon, _etype, _desc = {
                "SEAD":   ("💥", "SEAD_ARRIVE",    f"{callsign} SEAD 공격 개시"),
                "STRIKE": ("🎯", "STRIKE_HIT",     f"{callsign} 표적 타격"),
                "ISR":    ("📡", "ISR_ON_STATION", f"{callsign} 정찰 임무 개시"),
                "CAS":    ("🔥", "CAS_ON_STATION", f"{callsign} CAS 지원 개시"),
            }.get(mission, ("✅", "ON_TARGET", f"{callsign} 목표 도착"))

            # 다중 표적 STRIKE: chain_target_coords가 있으면 각 표적별로 별도
            # STRIKE_HIT 이벤트를 생성. 도착 시각은 path_in 누적거리 기준으로
            # 표적 위치에 가장 가까운 waypoint의 ratio로 보간.
            chain_targets = pdata.get("chain_target_coords") or []
            if mission == "STRIKE" and chain_targets and len(chain_targets) > 1:
                cum_in = tim.get("cum_in") or [0.0]
                total_dist = cum_in[-1] if cum_in else 1.0
                t_total_in = t_arrive - delay  # 순수 비행 시간 (분)
                for idx, tc in enumerate(chain_targets):
                    if not tc or len(tc) < 2:
                        continue
                    # 표적 좌표에 가장 가까운 path_in waypoint 인덱스
                    best_i, best_d = 0, float("inf")
                    for i, p in enumerate(path_in):
                        d = (float(p[0]) - float(tc[0])) ** 2 + (float(p[1]) - float(tc[1])) ** 2
                        if d < best_d:
                            best_d = d
                            best_i = i
                    ratio = (cum_in[best_i] / total_dist) if total_dist > 0 else 1.0
                    t_hit = delay + ratio * t_total_in
                    events.append(MissionEvent(
                        t_hit, "STRIKE_HIT", aid, callsign,
                        f"{callsign} 표적 {idx+1} 타격",
                        float(tc[0]), float(tc[1]), "🎯",
                    ))
            else:
                events.append(MissionEvent(t_arrive, _etype, aid, callsign, _desc,
                    float(ep[0]), float(ep[1]), _icon))

        # 이탈 시작
        if path_out and path_in:
            ep = path_in[-1]
            events.append(MissionEvent(t_egress, "EGRESS", aid, callsign,
                f"{callsign} 귀환 기동 시작",
                float(ep[0]), float(ep[1]), "\u21a9\ufe0f"))

        # 귀환
        if path_out:
            lp = path_out[-1]
            events.append(MissionEvent(t_done, "RTB", aid, callsign,
                f"{callsign} 기지 귀환",
                float(lp[0]), float(lp[1]), "\U0001f3e0"))

    # ── SEAD 위협 억압/파괴 이벤트 ────────────────────────────────
    threat_suppression: Dict[str, float] = {}   # threat_name -> 억압 시각(분)
    threat_destruction: Dict[str, float] = {}   # threat_name -> 파괴 시각(분)

    for aid, tim in timing_db.items():
        asset_obj = asset_map.get(aid)
        if not asset_obj:
            continue
        mission = (asset_obj.assigned_mission or "").upper()
        if mission != "SEAD":
            continue

        pdata    = formation_paths[aid]
        obj_name = pdata.get("objective")
        callsign = asset_obj.callsign or aid
        t_arrive = tim["t_arrive_min"]

        if obj_name:
            threat_suppression[obj_name] = t_arrive
            threat_destruction[obj_name] = t_arrive + 2.0

            # 위협 좌표
            thr_coord = None
            for thr in threats:
                if thr.get("name") == obj_name:
                    if thr.get("lat") is not None:
                        thr_coord = (float(thr["lat"]), float(thr["lon"]))
                    break
            if thr_coord is None and tim["path_in"]:
                ep = tim["path_in"][-1]
                thr_coord = (float(ep[0]), float(ep[1]))

            if thr_coord:
                events.append(MissionEvent(t_arrive, "SUPPRESSED", aid, callsign,
                    f"위협 억압: {obj_name}",
                    thr_coord[0], thr_coord[1], "\U0001f515"))
                events.append(MissionEvent(t_arrive + 2.0, "DESTROYED", aid, callsign,
                    f"위협 파괴: {obj_name}",
                    thr_coord[0], thr_coord[1], "\U0001f4a5"))

    # ── STRIKE / CAS BDA (Battle Damage Assessment) 이벤트 ────────
    # STRIKE 자산이 목표에 도달했을 때 교전 결과(피해 평가)를 확률 모델로 산출
    for aid, tim in timing_db.items():
        asset_obj = asset_map.get(aid)
        if not asset_obj:
            continue
        mission = (asset_obj.assigned_mission or "").upper()
        if mission not in ("STRIKE", "CAS"):
            continue

        pdata    = formation_paths[aid]
        callsign = asset_obj.callsign or aid
        obj_name = pdata.get("objective", "목표")
        t_arrive = tim["t_arrive_min"]

        # 타격 시점에 억압된 위협 계산
        suppressed_at = {
            name for name, t_sup in threat_suppression.items()
            if t_sup <= t_arrive
        }
        # 활성 SAM/RADAR 위협 수 → 타격 성공률 감쇄
        n_active = sum(
            1 for thr in threats
            if thr.get("name") not in suppressed_at
            and thr.get("type") in ("SAM", "RADAR")
        )

        # 피해 평가 확률 모델:
        # 기본 타격 성공률 0.88, 활성 위협 1개당 -0.10 (최저 0.25)
        base_p_hit  = 0.88
        p_hit       = max(0.25, base_p_hit - n_active * 0.10)
        # 부분 피해 vs 완전 파괴 구분
        p_destroy   = p_hit * 0.70
        p_damage    = p_hit - p_destroy

        if p_hit >= 0.70:
            bda_icon  = "\U0001f525"
            bda_label = f"파괴 추정 (P_destroy={p_destroy:.0%})"
            bda_etype = "TARGET_DESTROYED"
        elif p_hit >= 0.45:
            bda_icon  = "\u26a0\ufe0f"
            bda_label = f"부분 피해 (P_hit={p_hit:.0%})"
            bda_etype = "TARGET_DAMAGED"
        else:
            bda_icon  = "\u2753"
            bda_label = f"타격 결과 불확실 (P_hit={p_hit:.0%}) — SEAD 지원 필요"
            bda_etype = "TARGET_UNCERTAIN"

        threat_ctx = f", 잔여 위협 {n_active}개 활성" if n_active > 0 else ", 위협 없음"
        path_in = tim["path_in"]
        if path_in:
            ep = path_in[-1]
            # BDA 이벤트: 타격 1분 후 (평가 시간 반영)
            events.append(MissionEvent(
                t_arrive + 1.0, bda_etype, aid, callsign,
                f"[BDA] {callsign} → {obj_name} | {bda_label}{threat_ctx}",
                float(ep[0]), float(ep[1]), bda_icon,
            ))

    events.sort(key=lambda e: e.time_min)

    # ── 타임프레임 생성 ────────────────────────────────────────────
    frames: List[TimeFrame] = []
    t = 0.0

    while t <= max_dur + time_resolution_min * 0.5:
        asset_states: Dict[str, AssetState] = {}

        for aid, tim in timing_db.items():
            path_in  = tim["path_in"]
            path_out = tim["path_out"]
            cum_in   = tim["cum_in"]
            cum_out  = tim["cum_out"]
            spd      = tim["speed_kmh"]
            t_arrive = tim["t_arrive_min"]
            t_egress = tim["t_egress_start_min"]
            t_done   = tim["t_done_min"]

            delay = tim.get("launch_delay_min", 0.0)
            if t < delay:
                # 아직 이륙 전 - 기지 대기
                phase = "standby"
                p = path_in[0] if path_in else [0, 0, 0]
                lat, lon, alt = float(p[0]), float(p[1]), float(p[2]) if len(p) >= 3 else 500.0
                heading = 0.0

            elif t <= t_arrive and path_in:
                phase    = "ingress"
                dist_km  = ((t - delay) / 60.0) * spd
                lat, lon, alt = _interpolate_pos(path_in, cum_in, dist_km)
                dist_next = dist_km + spd / 60.0 * time_resolution_min
                lat2, lon2, _ = _interpolate_pos(path_in, cum_in, dist_next)
                heading  = _bearing((lat, lon), (lat2, lon2))

            elif t <= t_egress and path_in:
                phase   = "on_target"
                ep      = path_in[-1]
                lat, lon, alt = float(ep[0]), float(ep[1]), float(ep[2]) if len(ep) >= 3 else 500.0
                heading = 180.0

            elif path_out and t <= t_done:
                phase       = "egress"
                elapsed     = t - t_egress
                dist_km     = (elapsed / 60.0) * spd
                lat, lon, alt = _interpolate_pos(path_out, cum_out, dist_km)
                dist_next   = dist_km + spd / 60.0 * time_resolution_min
                lat2, lon2, _ = _interpolate_pos(path_out, cum_out, dist_next)
                heading     = _bearing((lat, lon), (lat2, lon2))

            else:
                phase = "rtb"
                src   = path_out[-1] if path_out else (path_in[-1] if path_in else [0, 0, 0])
                lat, lon, alt = float(src[0]), float(src[1]), float(src[2]) if len(src) >= 3 else 0.0
                heading = 0.0

            asset_states[aid] = AssetState(
                lat=lat, lon=lon, alt=alt, phase=phase, heading_deg=heading
            )

        # 위협 상태
        threat_statuses: Dict[str, str] = {}
        for thr in threats:
            tname = thr.get("name", "")
            if not tname:
                continue
            if tname in threat_destruction and t >= threat_destruction[tname]:
                threat_statuses[tname] = "destroyed"
            elif tname in threat_suppression and t >= threat_suppression[tname]:
                threat_statuses[tname] = "suppressed"
            else:
                threat_statuses[tname] = "active"

        # 현재 타임프레임의 이벤트
        half = time_resolution_min * 0.5
        events_at = [e for e in events if t - half <= e.time_min < t + half]

        frames.append(TimeFrame(
            time_min=round(t, 2),
            asset_states=asset_states,
            threat_statuses=threat_statuses,
            events_at=events_at,
        ))
        t = round(t + time_resolution_min, 6)

    # ── asset_info (UI 표시용) ──────────────────────────────────────
    asset_info: Dict[str, dict] = {}
    for aid, pdata in formation_paths.items():
        tim = timing_db.get(aid, {})
        asset_info[aid] = {
            "callsign":           pdata.get("callsign", aid),
            "mission":            pdata.get("mission", "?"),
            "type":               pdata.get("type", "?"),
            "color":              pdata.get("color", "#888888"),
            "launch_delay_min":   tim.get("launch_delay_min", 0.0),
            "t_arrive_min":       tim.get("t_arrive_min", 0.0),
            "t_egress_min":       tim.get("t_egress_start_min", 0.0),
            "t_done_min":         tim.get("t_done_min", 0.0),
            "dist_in_km":         tim.get("dist_in_km", 0.0),
            "dist_out_km":        tim.get("dist_out_km", 0.0),
            "speed_kmh":          tim.get("speed_kmh", 300.0),
        }

    return MissionTimeline(
        total_duration_min=round(max_dur, 1),
        frames=frames,
        events=events,
        asset_info=asset_info,
        asset_timing=timing_db,
    )


def get_frame_at(timeline: MissionTimeline, t_min: float) -> TimeFrame:
    """시간 t_min에 가장 가까운 TimeFrame 반환."""
    if not timeline.frames:
        return TimeFrame(t_min, {}, {})
    best = min(timeline.frames, key=lambda f: abs(f.time_min - t_min))
    return best
