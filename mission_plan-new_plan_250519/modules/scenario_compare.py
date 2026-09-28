"""
시나리오 비교 모듈 - 연료 상태별 3종 경로 비교

저연료(위험감수) / 표준 / 급유지원(안전우선) 시나리오를 동일 조건에서
병렬 계산하여 정량 비교표를 생성합니다.

발표용 핵심 결과물: 3개 경로를 지도에 동시 표시 + 성능 비교표
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from modules.xai_utils import XAIUtils


# ================================================================
# 시나리오 정의
# ================================================================

@dataclass
class ScenarioConfig:
    """단일 시나리오 설정"""
    label: str                   # 표시 이름 (예: "A: 저연료")
    description: str             # 설명
    fuel_state: float            # 0.2~1.0
    refuel_count: int            # 0~2
    safety_margin_km: float      # km
    color: str                   # 지도 경로 색상 (hex)
    dash: str = ""               # 경로 선 스타일 (빈 문자열=실선, "10 5"=점선)


# 발표용 3종 프리셋 — 모두 MUM-T 패키지 전제 (전투기 조종사 시각)
# 동일 위협·출발지·목표 조건에서 연료/마진만 달리하여 전투기 경로·위험도 비교
SCENARIO_PRESETS: List[ScenarioConfig] = [
    ScenarioConfig(
        label="A: 위험감수 (저연료)",
        description="UAV SEAD 최소 지원, 연료 30% · 안전마진 3km → 직선 돌파 경로",
        fuel_state=0.30,
        refuel_count=0,
        safety_margin_km=3.0,
        color="#E53935",   # 빨강
        dash="",
    ),
    ScenarioConfig(
        label="B: 표준 MUM-T",
        description="UAV ISR+SEAD 선행 지원, 연료 70% · 안전마진 8km → 기본 우회 경로",
        fuel_state=0.70,
        refuel_count=0,
        safety_margin_km=8.0,
        color="#FB8C00",   # 주황
        dash="8 4",
    ),
    ScenarioConfig(
        label="C: 안전우선 (급유지원)",
        description="UAV 완전 억압 + 공중급유 2회, 연료 100% · 안전마진 15km → 최대 우회 경로",
        fuel_state=1.00,
        refuel_count=2,
        safety_margin_km=15.0,
        color="#43A047",   # 초록
        dash="4 4",
    ),
]


# ================================================================
# 시나리오 결과
# ================================================================

@dataclass
class ScenarioResult:
    """단일 시나리오 실행 결과"""
    config: ScenarioConfig
    path_in: List = field(default_factory=list)
    path_out: List = field(default_factory=list)
    calc_time_s: float = 0.0
    metrics: Dict = field(default_factory=dict)
    success: bool = False
    error_msg: str = ""

    @property
    def total_length_km(self) -> float:
        return float(self.metrics.get("total_length_km", 0.0))

    @property
    def avg_risk(self) -> float:
        return float(self.metrics.get("avg_risk", 0.0))

    @property
    def max_risk(self) -> float:
        return float(self.metrics.get("max_risk", 0.0))

    @property
    def high_risk_segments(self) -> int:
        return int(self.metrics.get("high_risk_segments", 0))

    @property
    def waypoint_count(self) -> int:
        return len(self.path_in) + len(self.path_out)


# ================================================================
# 단일 시나리오 실행
# ================================================================

def _run_path(
    pathfinder,
    start: list,
    end: list,
    threats: list,
    margin: float,
    fuel_state: float,
    refuel_count: int,
) -> List:
    """경로탐색기 종류에 따라 적절한 메서드 호출"""
    if hasattr(pathfinder, "find_path_3d_fast"):
        return pathfinder.find_path_3d_fast(
            start, end, threats, margin,
            fuel_state=fuel_state, refuel_count=refuel_count,
        )
    if hasattr(pathfinder, "find_path_3d"):
        return pathfinder.find_path_3d(
            start, end, threats, margin,
            fuel_state=fuel_state, refuel_count=refuel_count,
        )
    # 2D fallback
    try:
        return pathfinder.find_path(
            start[:2], end[:2], threats, margin,
            fuel_state=fuel_state, refuel_count=refuel_count,
        )
    except TypeError:
        return pathfinder.find_path(start[:2], end[:2], threats, margin)


def run_scenario(
    config: ScenarioConfig,
    pathfinder,
    start_coord: list,
    target_coord: list,
    threats_dict: list,
    rtb: bool,
    terrain_loader=None,
    extra_targets: Optional[List[list]] = None,
) -> ScenarioResult:
    """단일 시나리오 실행 → 경로 + 위험도 메트릭 반환.

    extra_targets: 추가 목표 좌표 리스트 [[lat, lon], ...] (다중 목표 체인)
    경로: start → target → extra[0] → extra[1] → ... (→ start if rtb)
    """
    from modules.pathfinder import smooth_path, smooth_path_3d

    def _smooth(raw):
        if not raw:
            return []
        return smooth_path_3d(raw) if (raw[0] and len(raw[0]) >= 3) else smooth_path(raw)

    t0 = time.time()

    # 전체 경유 순서: start → primary → extra targets
    waypoints = [start_coord, target_coord] + (extra_targets or [])

    # 구간별 경로 체인으로 연결
    # 이전 구현은 첫 구간 실패만 ScenarioResult(success=False)로 반환하고,
    # 두 번째 이후 구간 실패는 silent로 통과해 사용자에겐 "성공"으로 보였음.
    # 다중 목표 시나리오에서 일부만 도달했는데 비교표가 success=True로 표시되는
    # 결함을 방지하기 위해, 어떤 구간이라도 실패하면 즉시 실패로 보고.
    path_in: List = []
    for i in range(len(waypoints) - 1):
        seg_start = list(path_in[-1]) if path_in else waypoints[i]
        seg_end = waypoints[i + 1]
        try:
            raw_seg = _run_path(
                pathfinder, seg_start, seg_end,
                threats_dict, config.safety_margin_km,
                config.fuel_state, config.refuel_count,
            )
            seg = _smooth(raw_seg)
            if not seg:
                return ScenarioResult(
                    config=config,
                    calc_time_s=time.time() - t0,
                    success=False,
                    error_msg=f"구간 {i+1}/{len(waypoints)-1} 경로 계산 실패 (빈 결과)",
                )
            path_in.extend(seg[1:] if path_in else seg)
        except Exception as e:
            return ScenarioResult(
                config=config,
                calc_time_s=time.time() - t0,
                success=False,
                error_msg=f"구간 {i+1}/{len(waypoints)-1} 실패: {e}",
            )

    if not path_in:
        return ScenarioResult(config=config, calc_time_s=time.time() - t0,
                              success=False, error_msg="경로 계산 실패")

    path_out: List = []
    if rtb and path_in:
        try:
            raw_out = _run_path(
                pathfinder, path_in[-1], start_coord,
                threats_dict, config.safety_margin_km,
                config.fuel_state, config.refuel_count,
            )
            path_out = _smooth(raw_out)
        except Exception:
            path_out = []

    calc_time = time.time() - t0
    combined = path_in + path_out
    metrics = XAIUtils.analyze_path_risk(
        combined, threats_dict, config.safety_margin_km, terrain_loader
    ) if combined else {
        "avg_risk": 0.0, "max_risk": 0.0,
        "high_risk_segments": 0, "total_length_km": 0.0,
    }

    return ScenarioResult(
        config=config,
        path_in=path_in,
        path_out=path_out,
        calc_time_s=round(calc_time, 3),
        metrics=metrics,
        success=bool(path_in),
    )


# ================================================================
# 3종 시나리오 일괄 실행
# ================================================================

def run_all_scenarios(
    pathfinder,
    start_coord: list,
    target_coord: list,
    threats_dict: list,
    rtb: bool,
    terrain_loader=None,
    custom_configs: Optional[List[ScenarioConfig]] = None,
    extra_targets: Optional[List[list]] = None,
) -> List[ScenarioResult]:
    """
    3종(또는 custom) 시나리오를 순서대로 실행하여 결과 목록 반환.

    Args:
        pathfinder:    경로탐색 객체 (AStarPathfinder3DOptimized 등)
        start_coord:   출발 좌표 [lat, lon] 또는 [lat, lon, alt]
        target_coord:  1번 목표 좌표
        threats_dict:  위협 dict 목록
        rtb:           복귀 여부
        terrain_loader: 지형 데이터 (선택)
        custom_configs: 사용자 정의 시나리오 설정 (None이면 SCENARIO_PRESETS)
        extra_targets: 추가 목표 좌표 리스트 [[lat,lon], ...] (다중 목표 체인)
    """
    configs = custom_configs or SCENARIO_PRESETS
    results: List[ScenarioResult] = []
    for cfg in configs:
        result = run_scenario(
            cfg, pathfinder, start_coord, target_coord,
            threats_dict, rtb, terrain_loader,
            extra_targets=extra_targets,
        )
        results.append(result)
    return results


# ================================================================
# 비교표 데이터 생성 (pandas DataFrame용)
# ================================================================

def build_comparison_table(results: List[ScenarioResult]) -> List[Dict]:
    """
    시나리오 비교표 데이터 생성.
    streamlit st.dataframe()에 바로 전달 가능.
    """
    rows = []
    for r in results:
        rows.append({
            "시나리오": r.config.label,
            "연료 상태": f"{r.config.fuel_state:.0%}",
            "급유 횟수": f"{r.config.refuel_count}회",
            "안전마진(km)": r.config.safety_margin_km,
            "총 비행거리(km)": round(r.total_length_km, 1),
            "평균 위험도": round(r.avg_risk, 3),
            "최대 위험도": round(r.max_risk, 3),
            "고위험 구간 수": r.high_risk_segments,
            "웨이포인트 수": r.waypoint_count,
            "계산 시간(s)": r.calc_time_s,
            "경로 생성": "✅" if r.success else f"❌ {r.error_msg[:30]}",
        })
    return rows
