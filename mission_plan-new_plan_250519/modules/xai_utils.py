"""
XAI utilities for threat risk scoring and path-level analysis.
"""

import math
from typing import Dict, List, Optional

import numpy as np

from modules.config import (
    HEATMAP_RESOLUTION,
    LAT_TO_KM,
    MAP_BOUNDS,
    RISK_THRESHOLD_HIGH,
    THREAT_ALT_ENVELOPE,
)

try:
    from modules.radar_shadow import check_line_of_sight
except ImportError:
    def check_line_of_sight(*args, **kwargs):
        return True


# LAT_TO_KM은 modules.config에서 import (위)


class XAIUtils:
    """XAI helper methods used by planner, UI heatmap, and validator."""

    @staticmethod
    def calculate_risk_score(
        lat: float,
        lon: float,
        threats: List[dict],
        margin: float,
        terrain_loader=None,
        target_alt: Optional[float] = None,
    ) -> float:
        """
        Calculate risk score at a point.

        - NFZ risk is strict (inside zone => 1.0).
        - Weapon risk combines RADAR detection and SAM kill factors.
        """
        if not threats:
            return 0.0

        if target_alt is None:
            try:
                ground_elev = terrain_loader.get_elevation(lat, lon) if terrain_loader else 0.0
                target_alt = ground_elev + 500.0
            except Exception:
                target_alt = 500.0

        mul_not_p_d = 1.0
        mul_not_p_k = 1.0
        nfz_risk = 0.0

        for t in threats:
            t_type = t.get("type")

            if t_type == "NFZ":
                margin_deg = margin / LAT_TO_KM
                if (
                    t.get("lat_min", 0.0) - margin_deg <= lat <= t.get("lat_max", 0.0) + margin_deg
                    and t.get("lon_min", 0.0) - margin_deg <= lon <= t.get("lon_max", 0.0) + margin_deg
                ):
                    nfz_risk = 1.0
                continue

            t_lat = t.get("lat")
            t_lon = t.get("lon")
            if t_lat is None or t_lon is None:
                continue

            dist_2d_km = math.sqrt(
                ((lat - t_lat) * LAT_TO_KM) ** 2
                + ((lon - t_lon) * LAT_TO_KM * math.cos(math.radians(lat))) ** 2
            )
            threat_alt_msl = float(t.get("alt", 0.0))
            rel_alt_m = target_alt - threat_alt_msl

            env = THREAT_ALT_ENVELOPE.get(t_type, {})
            min_alt_m = float(t.get("min_alt_m", env.get("min_alt_m", -1e9)))
            max_alt_m = float(t.get("max_alt_m", env.get("max_alt_m", 1e9)))
            if rel_alt_m > max_alt_m:
                continue

            # 최저 교전고도 이하: 지수감쇠 (0.30 고정 하한 제거)
            # e-folding 거리 1000m → gap=1000m 시 factor?0.37, gap=3000m 시 ?0.05
            below_min_factor = 1.0
            if rel_alt_m < min_alt_m:
                gap = min_alt_m - rel_alt_m
                below_min_factor = math.exp(-gap / 1000.0)

            alt_diff_km = rel_alt_m / 1000.0
            dist_3d_km = math.sqrt(dist_2d_km ** 2 + alt_diff_km ** 2)

            threat_radius = float(t.get("radius_km", 0.0))
            if threat_radius <= 0.0:
                continue
            if dist_3d_km >= threat_radius + margin:
                continue

            los_factor = 1.0
            if terrain_loader:
                is_visible = check_line_of_sight(
                    (t_lat, t_lon, threat_alt_msl),
                    (lat, lon, target_alt),
                    terrain_loader,
                )
                if not is_visible:
                    # Terrain masking — 음영 깊이 강화 (zip 0.20/0.35 → 0.10/0.20).
                    # 산악 너머 격자가 더 명확히 안전(파랑/투명)으로 보이도록.
                    los_factor = 0.10 if t_type == "RADAR" else 0.20

            proximity = max(0.0, 1.0 - (dist_3d_km / max(threat_radius + margin, 1e-6)))

            if t_type == "RADAR":
                loss = float(t.get("loss", 21.1))
                rcs_m2 = float(t.get("rcs_m2", 2.5))
                pd_k = max(float(t.get("pd_k", 0.4)), 0.01)  # 물리적 최솟값 방어
                if dist_3d_km <= 1e-9:
                    p_d = 1.0
                else:
                    r_d = threat_radius + margin
                    logit_pd = math.log(0.1 / 0.9)
                    snr_req = 10.0 ** (logit_pd / pd_k / 10.0)
                    snr0 = snr_req * ((r_d * 1000.0) ** 4) * (loss / max(rcs_m2, 1e-12))
                    snr = snr0 * (rcs_m2 / loss) / ((dist_3d_km * 1000.0) ** 4)
                    snr_log = max(-60.0, min(60.0, 10.0 * math.log10(max(snr, 1e-30))))
                    p_d = 1.0 / (1.0 + math.exp(-pd_k * snr_log))

                p_d = p_d * los_factor * below_min_factor
                p_d = max(p_d, 0.12 * proximity * los_factor)
                p_d = float(max(0.0, min(1.0, p_d)))
                if p_d > 1e-6:
                    mul_not_p_d *= (1.0 - p_d)

            elif t_type == "SAM":
                sskp = float(t.get("sskp", 0.75))
                raw_d0 = float(t.get("pk_peak_km", 0.35 * threat_radius))
                raw_sigma = float(t.get("pk_sigma_km", 0.20 * threat_radius))

                # Backward compatibility:
                # if 0~1 is passed, treat as radius ratio (0.35 -> 35% of radius).
                d0 = raw_d0 * threat_radius if 0.0 < raw_d0 <= 1.0 else raw_d0
                sigma = raw_sigma * threat_radius if 0.0 < raw_sigma <= 1.0 else raw_sigma
                if sigma <= 0.0:
                    sigma = max(0.20 * threat_radius, 1e-6)

                p_k = sskp * math.exp(-((dist_3d_km - d0) ** 2) / (2.0 * sigma ** 2))
                p_k = p_k * los_factor * below_min_factor
                p_k = max(p_k, 0.28 * proximity * los_factor)
                p_k = float(max(0.0, min(1.0, p_k)))
                if p_k > 1e-6:
                    mul_not_p_k *= (1.0 - p_k)

        total_pd = 1.0 - mul_not_p_d
        total_pk = 1.0 - mul_not_p_k

        # Both terms are probabilities in [0, 1], so their product cannot
        # exceed either component and is redundant inside a maximum.
        weapon_risk = max(total_pd, total_pk)

        total_risk = max(weapon_risk, nfz_risk)
        return float(min(1.0, total_risk))

    @staticmethod
    def generate_heatmap_data(
        threats: List[Dict],
        margin: float,
        terrain_loader=None,
        operational_alt_m: Optional[float] = 750.0,
    ) -> List[List[float]]:
        """
        작전고도 기반 위험도 히트맵 생성.

        Args:
            operational_alt_m: 비행기가 비행한다고 가정하는 고정 작전고도(MSL, m).
                None이면 격자별 자동 추정(ground+500m, 격자 alt가 들쑥날쑥해 시각화
                직관과 어긋남). 기본 750m는 F-16 전형적 cruise 고도.

        의미: "내 비행기가 작전고도로 비행한다고 할 때, 각 지점에서의 위험도".
        산악 음영은 비행기 고도 < 산 정상 + 위협이 산 너머일 때 자연스럽게 0이 됨.
        """
        min_lat, max_lat = MAP_BOUNDS["min_lat"], MAP_BOUNDS["max_lat"]
        min_lon, max_lon = MAP_BOUNDS["min_lon"], MAP_BOUNDS["max_lon"]
        heatmap_data: List[List[float]] = []
        step_lat = (max_lat - min_lat) / HEATMAP_RESOLUTION
        step_lon = (max_lon - min_lon) / HEATMAP_RESOLUTION

        for i in range(HEATMAP_RESOLUTION):
            for j in range(HEATMAP_RESOLUTION):
                lat = min_lat + i * step_lat
                lon = min_lon + j * step_lon
                risk = XAIUtils.calculate_risk_score(
                    lat, lon, threats, margin, terrain_loader,
                    target_alt=operational_alt_m,
                )
                if risk > 0.01:
                    heatmap_data.append([lat, lon, risk])
        return heatmap_data

    @staticmethod
    def analyze_threat_contributions(
        path: List,
        threats: List[dict],
        margin: float,
        terrain_loader=None,
        sample_step: int = 3,
    ) -> List[Dict]:
        """
        경로 전체에서 위협별 기여도 분석.

        Returns:
            List of dicts sorted by avg_risk descending:
            [{"name", "type", "avg_risk", "max_risk", "radius_km"}, ...]
        """
        if not path or not threats:
            return []

        sampled = path[::sample_step] if len(path) > sample_step else path
        results = []

        for t in threats:
            t_name = t.get("name", "Unknown")
            t_type = t.get("type", "?")
            scores = []
            for p in sampled:
                lat, lon = p[0], p[1]
                target_alt = p[2] if len(p) >= 3 else None
                risk = XAIUtils.calculate_risk_score(
                    lat, lon, [t], margin, terrain_loader, target_alt
                )
                scores.append(risk)

            avg_r = float(np.mean(scores)) if scores else 0.0
            max_r = float(max(scores)) if scores else 0.0
            if avg_r > 0.0 or max_r > 0.0:
                results.append({
                    "name": t_name,
                    "type": t_type,
                    "avg_risk": round(avg_r, 4),
                    "max_risk": round(max_r, 4),
                    "radius_km": float(t.get("radius_km", 0.0)),
                })

        results.sort(key=lambda x: x["avg_risk"], reverse=True)
        return results

    @staticmethod
    def analyze_path_risk(path, threats, margin, terrain_loader=None):
        if not path:
            return {"avg_risk": 0, "max_risk": 0, "high_risk_segments": 0, "total_length_km": 0}

        risks = []
        total_length = 0.0
        for i, p in enumerate(path):
            lat, lon = p[0], p[1]
            target_alt = p[2] if len(p) >= 3 else None

            risk = XAIUtils.calculate_risk_score(lat, lon, threats, margin, terrain_loader, target_alt)
            risks.append(risk)

            if i > 0:
                d2d = math.sqrt(
                    ((lat - path[i - 1][0]) * LAT_TO_KM) ** 2
                    + ((lon - path[i - 1][1]) * LAT_TO_KM * math.cos(math.radians(lat))) ** 2
                )
                if len(p) >= 3:
                    alt_cur = p[2]
                else:
                    alt_cur = terrain_loader.get_elevation(lat, lon) + 500 if terrain_loader else 500
                if len(path[i - 1]) >= 3:
                    alt_prev = path[i - 1][2]
                else:
                    alt_prev = (
                        terrain_loader.get_elevation(path[i - 1][0], path[i - 1][1]) + 500
                        if terrain_loader
                        else 500
                    )
                total_length += math.sqrt(d2d ** 2 + ((alt_cur - alt_prev) / 1000.0) ** 2)

        return {
            "avg_risk": float(np.mean(risks)) if risks else 0.0,
            "max_risk": float(max(risks)) if risks else 0.0,
            "high_risk_segments": sum(1 for r in risks if r >= RISK_THRESHOLD_HIGH),
            "total_length_km": total_length,
        }
