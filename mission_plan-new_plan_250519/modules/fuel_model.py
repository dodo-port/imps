"""
Fuel-aware planning helpers.

F-16 기준 연료 모델 (데모/연구용):
- fuel_state + refuel_count → endurance factor → 경로 위험비용 편향
- 속도/고도 프로파일별 연료소모율 추정 (F-16 기준 근사값)

근거:
- USAF F-16 Fact Sheet (ferry range ~3,220 km, internal fuel ~7,000 lb)
  https://www.af.mil/About-Us/Fact-Sheets/Display/Article/104505/f-16-fighting-falcon/
- 공개 F-16 성능 데이터: 고고도 순항(9km+) vs 저고도 침투(150m) 연료소모 비율
  저고도 고속(해면 900km/h)은 고고도 순항 대비 약 2~3배 소모 (Jane's All the World's Aircraft 기준 추정)
- Boeing KC-46 급유 운용 (데모용 gain=0.45는 가정값)
  https://www.boeing.com/defense/tankers-and-transports/kc-46-pegasus
"""

from modules.config import FUEL_POLICY


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


# ================================================================
# F-16 속도/고도별 연료소모율 근사 테이블 (데모용)
# 기준: 고고도 순항(9,000m, 800km/h) = 1.0 (정규화)
# 출처: 공개 F-16 운용 데이터 기반 추정 (Jane's, USAF Fact Sheet)
# ================================================================
_F16_CONSUMPTION_PROFILE = {
    # (고도구간_m, 속도구간_kmh): 상대 소모율
    "high_cruise":   {"alt_min": 7000, "alt_max": 15000, "spd_min": 700,  "spd_max": 900,  "rate": 1.00},
    "medium_cruise": {"alt_min": 3000, "alt_max": 7000,  "spd_min": 600,  "spd_max": 800,  "rate": 1.20},
    "low_fast":      {"alt_min": 0,    "alt_max": 1000,  "spd_min": 800,  "spd_max": 1100, "rate": 2.50},
    "low_slow":      {"alt_min": 0,    "alt_max": 1000,  "spd_min": 400,  "spd_max": 800,  "rate": 1.80},
    "combat":        {"alt_min": 1000, "alt_max": 5000,  "spd_min": 900,  "spd_max": 1200, "rate": 2.80},
}


def estimate_consumption_rate(altitude_m: float, speed_kmh: float) -> float:
    """
    고도·속도 기반 F-16 상대 연료소모율 반환.
    1.0 = 고고도 순항 기준.

    Args:
        altitude_m: 비행 고도 (m MSL)
        speed_kmh: 비행 속도 (km/h)

    Returns:
        소모율 배율 (1.0~3.0)
    """
    best_rate = None
    for profile in _F16_CONSUMPTION_PROFILE.values():
        if (profile["alt_min"] <= altitude_m <= profile["alt_max"] and
                profile["spd_min"] <= speed_kmh <= profile["spd_max"]):
            best_rate = profile["rate"]
            break

    if best_rate is None:
        # 매칭 안 되면 고도·속도 선형 보간
        alt_norm = _clamp(altitude_m / 9000.0, 0.0, 1.0)
        spd_norm = _clamp(speed_kmh / 900.0, 0.0, 1.5)
        # 저고도 고속일수록 소모 증가
        best_rate = 1.0 + (1.0 - alt_norm) * 0.8 + max(0.0, spd_norm - 1.0) * 1.5
        best_rate = _clamp(best_rate, 1.0, 3.0)

    return round(best_rate, 3)


def estimate_fuel_consumed_fraction(
    path_length_km: float,
    altitude_m: float = 5000.0,
    speed_kmh: float = 800.0,
    fuel_state: float = 1.0,
) -> float:
    """
    경로 길이·고도·속도 기반 연료 소모 비율 추정.

    Args:
        path_length_km: 경로 총 길이 (km)
        altitude_m: 평균 비행 고도 (m)
        speed_kmh: 평균 속도 (km/h)
        fuel_state: 초기 연료 상태 (0.2~1.0)

    Returns:
        소모 비율 (0.0~1.0), 초기 연료 대비
    """
    f16_ferry_range = float(FUEL_POLICY["f16_reference"]["ferry_range_km"])
    consumption_rate = estimate_consumption_rate(altitude_m, speed_kmh)
    # 고고도 순항 기준 소비: path_length_km / ferry_range
    base_fraction = (path_length_km / f16_ferry_range) * consumption_rate
    # 초기 연료 상태 반영 (연료 부족하면 가용 버퍼가 적음)
    if fuel_state <= 0.0:
        return 1.0  # 연료 완전 소진 → 소모 비율 100%
    return _clamp(base_fraction / max(fuel_state, 0.2), 0.0, 1.0)


def fuel_endurance_factor(fuel_state: float, refuel_count: int) -> float:
    """
    Convert mission fuel settings into a normalized endurance factor.

    fuel_state: 0.2~1.0 (UI slider; 1.0 means full planned fuel at takeoff)
    refuel_count: number of AR events (0..N)
    """
    fs = _clamp(fuel_state, 0.05, 1.2)
    max_refuel = int(FUEL_POLICY.get("max_refuel_events", 2))
    rc = max(0, min(int(refuel_count), max_refuel))
    gain = float(FUEL_POLICY.get("refuel_gain_per_event", 0.45))
    return _clamp(fs * (1.0 + rc * gain), 0.10, 2.50)


def fuel_risk_modifiers(fuel_state: float, refuel_count: int) -> tuple[float, float]:
    """
    Return (risk_penalty_scale, threshold_bias).

    - risk_penalty_scale multiplies continuous threat penalty.
      low fuel -> smaller penalty -> planner accepts shorter/riskier path.
      high fuel -> larger penalty -> planner prefers safer detour.
    - threshold_bias shifts the hard risk block threshold.
      low fuel -> positive bias (less strict).
      high fuel -> negative bias (more strict).
    """
    endu = fuel_endurance_factor(fuel_state, refuel_count)
    norm = _clamp((endu - 0.20) / 1.00, 0.0, 1.0)

    # 0.02~1.32 (low fuel strongly prioritizes short path)
    risk_penalty_scale = 0.02 + 1.30 * norm

    # Aggressive bias for visible behavior:
    # low fuel -> much less strict blocking, high fuel -> stricter blocking.
    threshold_bias = 0.45 * (1.0 - norm) - 0.12 * max(0.0, endu - 1.0)
    threshold_bias = _clamp(threshold_bias, -0.14, 0.48)
    return risk_penalty_scale, threshold_bias


def estimate_effective_range_km(base_range_km: float, fuel_state: float, refuel_count: int) -> float:
    return float(base_range_km) * fuel_endurance_factor(fuel_state, refuel_count)
