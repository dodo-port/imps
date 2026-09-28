"""
Tactical map renderer extracted from streamlit_app.py (Phase 3a refactor).

Builds a Folium map with:
- Risk heatmap (Layer 1)
- Threat area fills (Layer 2)
- Route lines with outline effect (Layer 3)
- Threat borders + icons (Layer 4)
- Asset departure markers (Layer 5)
- Base / target markers (Layer 6)
- Formation legend (Layer 7)
"""

from __future__ import annotations

from typing import Dict, List

import folium
from folium.plugins import HeatMap

from modules.config import MAP_CENTER, MAP_ZOOM
from modules.xai_utils import XAIUtils


# Threat type → (fill color, icon color)
_T_COLOR_MAP = {
    "SAM":   ("#D32F2F", "red"),
    "RADAR": ("#6A1B9A", "purple"),
    "NFZ":   ("#E65100", "orange"),
}
_DEFAULT_T_COLOR = ("#607D8B", "gray")

# Asset type → short emoji label
_TYPE_ICON  = {"fighter": "\u2708\ufe0f", "recon_uav": "\U0001f441\ufe0f", "attack_uav": "\U0001f4a5"}
_TYPE_ICON2 = {"fighter": "\u2708\ufe0f", "recon_uav": "\U0001f50d",       "attack_uav": "\U0001f4a5"}

# 히트맵 데이터 모듈 레벨 캐시 (동일 위협 구성 반복 계산 방지)
_heatmap_cache: dict = {}


def _draw_path_with_outline(
    m: folium.Map,
    path_coords: list,
    color: str,
    weight: int = 6,
    opacity: float = 0.97,
    dash: str = "",
    tooltip_text: str = "",
) -> None:
    """Draw a route as shadow + white outline + colored line for maximum visibility."""
    if not path_coords:
        return
    latlon = [(p[0], p[1]) for p in path_coords]
    # Shadow
    folium.PolyLine(latlon, color="#111111", weight=weight + 5,
                    opacity=0.4, tooltip=tooltip_text).add_to(m)
    # White outline
    folium.PolyLine(latlon, color="white", weight=weight + 2,
                    opacity=0.9, tooltip=tooltip_text).add_to(m)
    # Colored line
    kwargs: dict = dict(color=color, weight=weight, opacity=opacity, tooltip=tooltip_text)
    if dash:
        kwargs["dash_array"] = dash
    folium.PolyLine(latlon, **kwargs).add_to(m)


def build_tactical_map(
    mission,
    final_in: list,
    final_out: list,
    formation_paths: Dict[str, dict],
    threats_dict: List[dict],
    show_heatmap: bool,
    terrain_fast,
    start_coord: list,
    target_coord: list,
) -> folium.Map:
    """
    Build and return a Folium tactical map.

    Args:
        mission:          MissionState
        final_in:         Representative ingress waypoints (first-asset or solo)
        final_out:        Representative egress waypoints
        formation_paths:  Per-asset path dict from compute_formation_paths()
        threats_dict:     Raw threat dicts
        show_heatmap:     Whether to draw the XAI heatmap layer
        terrain_fast:     TerrainCacheFast instance
        start_coord:      [lat, lon, (alt)]
        target_coord:     [lat, lon, (alt)]

    Returns:
        Configured folium.Map ready for st_folium()
    """
    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=MAP_ZOOM,
        tiles="OpenStreetMap",  # 키 불필요. CartoDB는 2025년부터 API 키를 요구함.
    )

    # ── [Layer 1] Heatmap ──
    if show_heatmap and mission.threats:
        import hashlib, json as _json
        # 작전고도 결정 — fighter 기본 altitude_m을 ASSET_PERFORMANCE에서 가져오고,
        # 없으면 750m(F-16 전형 cruise) 사용. 히트맵은 "이 고도로 비행한다면 어디가
        # 위험한가"를 의미.
        try:
            from modules.config import ASSET_PERFORMANCE
            _op_alt = float(ASSET_PERFORMANCE.get("fighter", {}).get("altitude_m", 750.0) or 750.0)
        except Exception:
            _op_alt = 750.0
        _hm_payload = {
            "margin": round(float(getattr(mission.params, "margin", 0.0)), 3),
            "operational_alt_m": round(float(_op_alt), 1),
            "threats": sorted(
                [
                    {
                        "name": t.get("name", ""),
                        "type": t.get("type", ""),
                        "lat": round(float(t.get("lat", 0.0) or 0.0), 5),
                        "lon": round(float(t.get("lon", 0.0) or 0.0), 5),
                        "radius_km": round(float(t.get("radius_km", 0.0) or 0.0), 3),
                        "loss": round(float(t.get("loss", 0.0) or 0.0), 3),
                        "rcs_m2": round(float(t.get("rcs_m2", 0.0) or 0.0), 6),
                        "pd_k": round(float(t.get("pd_k", 0.0) or 0.0), 6),
                        "sskp": round(float(t.get("sskp", 0.0) or 0.0), 6),
                        "pk_peak_km": round(float(t.get("pk_peak_km", 0.0) or 0.0), 3),
                        "pk_sigma_km": round(float(t.get("pk_sigma_km", 0.0) or 0.0), 3),
                        "min_alt_m": t.get("min_alt_m"),
                        "max_alt_m": t.get("max_alt_m"),
                        "lat_min": t.get("lat_min"),
                        "lat_max": t.get("lat_max"),
                        "lon_min": t.get("lon_min"),
                        "lon_max": t.get("lon_max"),
                    }
                    for t in threats_dict
                ],
                key=lambda x: (x["type"], x["name"], x["lat"], x["lon"]),
            ),
        }
        _hm_key = hashlib.md5(
            _json.dumps(_hm_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        if _hm_key not in _heatmap_cache:
            _heatmap_cache[_hm_key] = XAIUtils.generate_heatmap_data(
                threats_dict, mission.params.margin,
                terrain_loader=terrain_fast,
                operational_alt_m=_op_alt,
            )
        h_data = _heatmap_cache[_hm_key]
        if h_data:
            # 격자별 위험도 차이(특히 산악 음영 패치)를 시각적으로 분명히 드러내려면
            # radius/blur를 줄여야 함. 이전 값(radius=15, blur=25)은 가우시안 합성이
            # 너무 부드러워 동/서 risk 0.9 vs 0.09 같은 큰 차이도 동심원으로 합쳐 보였음.
            # radius=10, blur=12로 조이면 격자 cell이 약간 보이지만 비대칭 패치가 명확.
            HeatMap(h_data, radius=10, blur=12, min_opacity=0.15, max_opacity=0.55).add_to(m)

    # ── [Layer 2] Threat area fills ──
    for t in mission.threats:
        hex_color, _ = _T_COLOR_MAP.get(t.type, _DEFAULT_T_COLOR)
        if t.type == "NFZ":
            if all(v is not None for v in [t.lat_min, t.lon_min, t.lat_max, t.lon_max]):
                folium.Rectangle(
                    [[t.lat_min, t.lon_min], [t.lat_max, t.lon_max]],
                    color=hex_color, weight=1.5,
                    fill=True, fill_color=hex_color, fill_opacity=0.07,
                ).add_to(m)
        elif t.lat is not None and t.lon is not None:
            folium.Circle(
                [t.lat, t.lon], radius=t.radius_km * 1000,
                color=hex_color, weight=1.5, opacity=0.5,
                fill=True, fill_color=hex_color, fill_opacity=0.06,
            ).add_to(m)

    # ── [Layer 3] Route lines ──
    if formation_paths:
        for asset_id, pdata in formation_paths.items():
            clr = pdata["color"]
            label = f"{_TYPE_ICON.get(pdata['type'], '?')} {pdata['callsign']} [{pdata['mission']}]"
            if pdata["in"]:
                _draw_path_with_outline(m, pdata["in"], clr, weight=6, opacity=0.97,
                                        tooltip_text=f"{label} ▶ Ingress")
            if pdata["out"]:
                _draw_path_with_outline(m, pdata["out"], clr, weight=4, opacity=0.80,
                                        dash="10 6", tooltip_text=f"{label} ◀ Egress")
    else:
        if final_in:
            _draw_path_with_outline(m, final_in, "#E53935", weight=6, opacity=0.97,
                                    tooltip_text="Ingress")
            # ── 다중 목표 체인: 각 경유지 도달 지점에 웨이포인트 마커 표시
            extra_targets = sorted(
                [t for t in getattr(mission.params, "extra_targets", []) if "lat" in t and "lon" in t],
                key=lambda t: t.get("priority", 2),
            )
            if extra_targets:
                _wp_colors = ["#FB8C00", "#8E24AA", "#00897B"]  # 주황, 보라, 청록
                for wp_idx, et in enumerate(extra_targets):
                    et_lat, et_lon = float(et["lat"]), float(et["lon"])
                    # final_in에서 해당 목표에 가장 가까운 점을 경유 마커로 표시
                    closest = min(final_in, key=lambda p: (p[0]-et_lat)**2 + (p[1]-et_lon)**2)
                    wp_color = _wp_colors[wp_idx % len(_wp_colors)]
                    folium.CircleMarker(
                        closest[:2], radius=10,
                        color="white", weight=2,
                        fill=True, fill_color=wp_color, fill_opacity=0.9,
                        tooltip=f"📍 경유 P{wp_idx+2}: {et.get('name', f'Target-{wp_idx+2}')}",
                    ).add_to(m)
        if final_out:
            _draw_path_with_outline(m, final_out, "#1E88E5", weight=5, opacity=0.85,
                                    dash="10 6", tooltip_text="Egress")

    # ── [Layer 4] Threat borders + icons ──
    for t in mission.threats:
        hex_color, icon_color = _T_COLOR_MAP.get(t.type, _DEFAULT_T_COLOR)
        if t.type != "NFZ" and t.lat is not None and t.lon is not None:
            folium.Circle(
                [t.lat, t.lon], radius=t.radius_km * 1000,
                color=hex_color, weight=2.5, opacity=0.8,
                fill=False, dash_array="10 5",
            ).add_to(m)
            icon_name = "rocket" if t.type == "SAM" else "wifi"
            alt_txt = ""
            if t.type == "SAM" and (
                getattr(t, "min_alt_m", None) is not None
                or getattr(t, "max_alt_m", None) is not None
            ):
                alt_txt = f", 고도 {float(t.min_alt_m or 0):.0f}~{float(t.max_alt_m or 0):.0f}m"
            folium.Marker(
                [t.lat, t.lon],
                icon=folium.Icon(color=icon_color, icon=icon_name, prefix="fa"),
                tooltip=f"{t.type}: {t.name} (반경 {t.radius_km:.0f}km{alt_txt})",
            ).add_to(m)

    # ── [Layer 5] Asset departure circles ──
    if formation_paths:
        for asset_id, pdata in formation_paths.items():
            if pdata["in"]:
                label = f"{_TYPE_ICON.get(pdata['type'], '?')} {pdata['callsign']} [{pdata['mission']}]"
                folium.CircleMarker(
                    pdata["in"][0][:2], radius=8,
                    color="white", weight=2.5,
                    fill=True, fill_color=pdata["color"], fill_opacity=1.0,
                    tooltip=label,
                ).add_to(m)

    # ── [Layer 6] Base and target markers ──
    folium.Marker(
        start_coord[:2],
        icon=folium.Icon(color="blue", icon="plane", prefix="fa"),
        tooltip=f"🛫 Base: {mission.params.start}",
    ).add_to(m)

    # 주 목표 (Priority 1 ? 빨간 crosshairs)
    folium.Marker(
        target_coord[:2],
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
        tooltip=f"🎯 [P1·High] {mission.params.target_name}",
    ).add_to(m)

    # 추가 목표 (Priority 2/3 ? 주황/초록 flag)
    _extra_colors = {1: "red", 2: "orange", 3: "green"}
    _extra_prio_label = {1: "\U0001f534 High", 2: "\U0001f7e1 Medium", 3: "\U0001f7e2 Low"}
    for i, et in enumerate(getattr(mission.params, "extra_targets", [])):
        prio = int(et.get("priority", 2))
        folium.Marker(
            [et["lat"], et["lon"]],
            icon=folium.Icon(color=_extra_colors.get(prio, "orange"), icon="flag", prefix="fa"),
            tooltip=f"🎯 [P{i+2}·{_extra_prio_label.get(prio,'?')}] {et.get('name', f'Target-{i+2}')}",
        ).add_to(m)

    # ── [Layer 7] Formation legend ──
    if formation_paths:
        legend_rows = ""
        for asset_id, pdata in formation_paths.items():
            icon_c = _TYPE_ICON2.get(pdata["type"], "?")
            legend_rows += (
                f"<div style='display:flex;align-items:center;margin:4px 0;gap:8px;'>"
                f"<div style='width:32px;height:5px;background:{pdata['color']};"
                f"border:1px solid white;border-radius:3px;flex-shrink:0;'></div>"
                f"<span>{icon_c} <b>{pdata['callsign']}</b> "
                f"<span style='color:#ccc;font-size:11px;'>[{pdata['mission']}]</span></span>"
                f"</div>"
            )
        rtb_note = ""
        if mission.params.rtb:
            rtb_note = (
                "<div style='margin-top:8px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.2);"
                "color:#aaa;font-size:10px;'>실선 \u2500 Ingress &nbsp;\u2502&nbsp; 점선 \u2508 Egress</div>"
            )
        legend_html = (
            f"<div style='background:rgba(10,10,20,0.90);color:white;"
            f"padding:12px 16px;border-radius:10px;font-size:12px;"
            f"border:1px solid rgba(255,255,255,0.15);min-width:190px;"
            f"box-shadow:0 4px 12px rgba(0,0,0,0.5);'>"
            f"<div style='font-size:13px;font-weight:bold;margin-bottom:6px;'>📍 자산 경로 범례</div>"
            f"{legend_rows}{rtb_note}</div>"
        )
        m.get_root().html.add_child(folium.Element(
            f"<div style='position:fixed;bottom:40px;left:40px;z-index:9999;'>{legend_html}</div>"
        ))

    return m
