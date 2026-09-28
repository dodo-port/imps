"""
Cesium 시뮬레이션 런처

CZML 데이터를 포함한 군사 다크 테마 HTML 파일을 생성하고
기본 브라우저로 자동 실행합니다.

필요 없는 외부 설치: CesiumJS CDN에서 자동 로드 (인터넷 연결 필요)
Cesium Ion 토큰: config에서 CESIUM_ION_TOKEN 설정 시 위성 지형 사용,
                없으면 ArcGIS World Imagery (무료) 사용
"""
from __future__ import annotations

import html as _html
import http.server
import json
import os
import socket
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

# ── 로컬 HTTP 서버 (file:// 대신 http://localhost 로 제공) ──────
# Chrome은 file:// 에서 blob Worker URL을 차단 → Cesium Worker 불가
# daemon=True 로 Streamlit 종료 시 자동 정리
_server_instance: dict = {}  # {"httpd": HTTPServer, "port": int}


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ensure_server(serve_dir: str) -> int:
    """serve_dir 을 서빙하는 로컬 HTTP 서버를 띄우고 포트 반환.

    이전 구현은 첫 호출의 디렉토리에 핸들러가 고정되어, 두 번째 시뮬레이션
    실행 시 새 tmp_dir로 호출해도 옛 디렉토리만 서빙되어 모델/HTML이 404로
    실패하는 결함이 있었음. 디렉토리가 바뀌면 기존 서버를 종료하고 새로
    기동하도록 수정.
    """
    global _server_instance

    # 같은 디렉토리에 대한 재호출이면 포트 재사용.
    if (
        _server_instance.get("httpd")
        and _server_instance.get("dir") == serve_dir
    ):
        try:
            _server_instance["httpd"].server_address  # 살아있는지 확인
            return _server_instance["port"]
        except Exception:
            pass

    # 디렉토리가 바뀌었으면 기존 서버를 깨끗이 종료.
    if _server_instance.get("httpd"):
        try:
            _server_instance["httpd"].shutdown()
            _server_instance["httpd"].server_close()
        except Exception:
            pass
        _server_instance.clear()

    port = _find_free_port()

    # 로그 억제 핸들러 (클래스 팩토리로 serve_dir 캡처)
    def _make_handler(directory: str):
        class _H(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=directory, **kw)
            def log_message(self, *args):
                pass  # 콘솔 로그 억제
        return _H

    httpd = http.server.HTTPServer(("127.0.0.1", port), _make_handler(serve_dir))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)  # 서버 기동 대기

    _server_instance["httpd"] = httpd
    _server_instance["port"]  = port
    _server_instance["dir"]   = serve_dir
    return port

CESIUM_VERSION = "1.115"
CESIUM_CDN = f"https://cesium.com/downloads/cesiumjs/releases/{CESIUM_VERSION}/Build/Cesium"


def _get_token() -> str:
    """config에서 Cesium Ion 토큰을 읽어옴.

    이전 구현은 매 호출마다 `importlib.reload(modules.config)`를 수행했는데,
    이는 같은 streamlit 세션에서 다른 모듈이 import해 둔 config 상수의 객체
    동일성을 깨뜨려 미묘한 캐시 무효화 결함을 유발할 수 있었음. 환경변수 →
    config 상수 → 빈 문자열 순서의 정적 조회로 단순화.
    """
    env_token = os.environ.get("CESIUM_ION_TOKEN")
    if env_token:
        return env_token
    try:
        from modules.config import CESIUM_ION_TOKEN
        return CESIUM_ION_TOKEN or ""
    except Exception:
        return ""


def _build_asset_rows(formation_paths: Dict) -> str:
    """자산 상태 패널 HTML 행 생성"""
    type_colors = {
        "fighter":    "#64C8FF",
        "recon_uav":  "#64FF96",
        "attack_uav": "#FF9632",
    }
    type_icons = {
        "fighter":    "\u2708",
        "recon_uav":  "\U0001f441",
        "attack_uav": "\U0001f4a5",
    }
    rows = []
    for aid, pdata in formation_paths.items():
        atype    = pdata.get("type", "fighter")
        callsign = pdata.get("callsign", aid)
        mission  = pdata.get("mission", "-")
        obj      = pdata.get("objective", "")
        color    = type_colors.get(atype, "#888")
        icon     = type_icons.get(atype, "?")
        tooltip  = f"{obj[:30]}..." if len(obj) > 30 else obj
        rows.append(
            f'<div class="asset-row" title="{tooltip}">'
            f'<div class="asset-dot" style="background:{color}"></div>'
            f'<span>{icon} {callsign}</span>'
            f'<span class="asset-mission">{mission}</span>'
            f'</div>'
        )
    return "\n".join(rows) if rows else '<div class="asset-row">편대 없음</div>'


def _build_legend_rows(formation_paths: Dict) -> str:
    """경로 범례 HTML 행 생성"""
    rows = []
    for aid, pdata in formation_paths.items():
        color    = pdata.get("color", "#888")
        callsign = pdata.get("callsign", aid)
        mission  = pdata.get("mission", "")
        rows.append(
            f'<div class="legend-row">'
            f'<div class="legend-line" style="background:{color}"></div>'
            f'<span>{callsign} [{mission}]</span>'
            f'</div>'
        )
    return "\n".join(rows)


def _build_html(
    czml_json: str,
    formation_paths: Dict,
    mission,
    center_lat: float,
    center_lon: float,
) -> str:
    """군사 다크 테마 Cesium HTML 생성"""

    ion_token = _get_token()
    has_token  = bool(ion_token.strip())

    # 사용자/LLM 입력은 HTML/JS에 직접 보간하면 XSS·인젝션 위험.
    # mission_name·base_name은 HTML 본문(HUD)에 들어가므로 html.escape로 처리,
    # 토큰은 JS 문자열 컨텍스트에 들어가므로 json.dumps로 안전하게 직렬화.
    raw_mission_name = str(getattr(mission.params, "target_name", "Unknown Target") or "Unknown Target")
    raw_base_name    = str(getattr(mission.params, "start", "Unknown Base") or "Unknown Base")
    # 길이 클램프 — HUD 레이아웃 보호.
    raw_mission_name = raw_mission_name[:80]
    raw_base_name    = raw_base_name[:80]

    mission_name  = _html.escape(raw_mission_name, quote=True)
    base_name     = _html.escape(raw_base_name, quote=True)
    threat_count  = len(getattr(mission, "threats", []))
    asset_rows    = _build_asset_rows(formation_paths)
    legend_rows   = _build_legend_rows(formation_paths)

    # json.dumps는 따옴표·</script>·줄바꿈 등을 모두 안전하게 escape한다.
    token_js = (
        f"Cesium.Ion.defaultAccessToken = {json.dumps(ion_token)};" if ion_token else ""
    )
    has_token_js = "true" if has_token else "false"

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>IMPS 임무 시뮬레이션 — {mission_name}</title>
<!-- CESIUM_BASE_URL 반드시 Cesium.js 로드 전에 설정해야 buildModuleUrl이 CDN 경로 사용 -->
<script>window.CESIUM_BASE_URL = "{CESIUM_CDN}/";</script>
<script src="{CESIUM_CDN}/Cesium.js"></script>
<link href="{CESIUM_CDN}/Widgets/widgets.css" rel="stylesheet"/>
<style>
/* ── 기본 ── */
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#060C14;font-family:'Segoe UI','Noto Sans KR',monospace;overflow:hidden;color:#C9D1D9;}}
#cesiumContainer{{width:100vw;height:100vh;}}

/* ── HUD 상단 바 ── */
#hud{{
  position:fixed;top:0;left:0;right:0;z-index:200;
  background:linear-gradient(180deg,rgba(6,12,20,0.97)0%,rgba(6,12,20,0.6)85%,transparent100%);
  padding:10px 20px 18px;
  display:flex;align-items:center;gap:16px;
  border-bottom:1px solid rgba(88,166,255,0.15);
  pointer-events:none;
}}
#hud-logo{{
  color:#58A6FF;font-size:20px;font-weight:800;letter-spacing:3px;
  text-shadow:0 0 18px rgba(88,166,255,0.7);
  white-space:nowrap;
}}
#hud-divider{{width:1px;height:32px;background:rgba(88,166,255,0.3);}}
#hud-info{{flex:1;}}
#hud-mission-label{{color:#8B949E;font-size:10px;letter-spacing:2px;text-transform:uppercase;}}
#hud-mission-name{{color:#E6EDF3;font-size:14px;font-weight:600;}}
#hud-right{{margin-left:auto;display:flex;align-items:center;gap:20px;}}
#hud-status{{
  color:#3FB950;font-size:11px;font-weight:700;letter-spacing:2px;
  animation:blink 1.8s ease-in-out infinite;
}}
#hud-clock{{
  color:#58A6FF;font-size:22px;font-weight:700;
  font-family:'Courier New',monospace;
  text-shadow:0 0 12px rgba(88,166,255,0.6);
  min-width:110px;text-align:right;
}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0.35}}}}

/* ── 스캔라인 오버레이 (분위기용) ── */
#scanlines{{
  position:fixed;inset:0;z-index:150;pointer-events:none;
  background:repeating-linear-gradient(
    to bottom,transparent,transparent 2px,rgba(0,0,0,0.04) 2px,rgba(0,0,0,0.04) 4px
  );
}}

/* ── 자산 패널 ── */
#asset-panel{{
  position:fixed;right:12px;top:70px;z-index:200;
  background:rgba(6,12,20,0.92);
  border:1px solid rgba(88,166,255,0.2);
  border-radius:8px;padding:12px 16px;min-width:230px;
  backdrop-filter:blur(6px);
}}
#asset-panel h3{{
  color:#58A6FF;font-size:10px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;margin-bottom:10px;
  border-bottom:1px solid rgba(88,166,255,0.2);padding-bottom:6px;
}}
.asset-row{{
  display:flex;align-items:center;gap:8px;
  margin-bottom:7px;font-size:12px;
}}
.asset-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0;}}
.asset-mission{{
  margin-left:auto;
  color:#58A6FF;font-size:10px;font-weight:700;
  background:rgba(88,166,255,0.12);
  padding:1px 6px;border-radius:3px;
}}

/* ── 위협 요약 ── */
#threat-summary{{
  position:fixed;right:12px;bottom:80px;z-index:200;
  background:rgba(6,12,20,0.92);
  border:1px solid rgba(220,60,60,0.25);
  border-radius:8px;padding:10px 14px;min-width:180px;
  backdrop-filter:blur(6px);
}}
#threat-summary h3{{
  color:#F85149;font-size:10px;font-weight:700;
  letter-spacing:2px;text-transform:uppercase;
  margin-bottom:7px;
}}
.threat-item{{font-size:11px;color:#8B949E;margin-bottom:4px;}}

/* ── 범례 ── */
#legend{{
  position:fixed;left:12px;bottom:80px;z-index:200;
  background:rgba(6,12,20,0.92);
  border:1px solid rgba(255,255,255,0.1);
  border-radius:8px;padding:10px 14px;
  backdrop-filter:blur(6px);
}}
#legend h4{{
  color:#58A6FF;font-size:10px;font-weight:700;
  letter-spacing:2px;text-transform:uppercase;
  margin-bottom:8px;
}}
.legend-row{{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:11px;color:#8B949E;}}
.legend-line{{width:28px;height:3px;border-radius:2px;flex-shrink:0;}}
.legend-sep{{border-top:1px solid rgba(255,255,255,0.08);margin:6px 0;}}

/* ── 컨트롤 바 ── */
#controls{{
  position:fixed;bottom:16px;left:50%;transform:translateX(-50%);
  display:flex;gap:8px;z-index:200;
  background:rgba(6,12,20,0.88);
  border:1px solid rgba(88,166,255,0.18);
  border-radius:10px;padding:8px 12px;
  backdrop-filter:blur(6px);
}}
.ctrl-btn{{
  background:rgba(31,111,235,0.75);
  border:1px solid rgba(88,166,255,0.35);
  color:#fff;padding:6px 14px;border-radius:6px;
  cursor:pointer;font-size:12px;font-weight:600;
  transition:all 0.15s;letter-spacing:0.5px;
  white-space:nowrap;
}}
.ctrl-btn:hover{{background:rgba(31,111,235,1);transform:translateY(-1px);box-shadow:0 4px 12px rgba(31,111,235,0.5);}}
.ctrl-btn:active{{transform:translateY(0);}}
.ctrl-btn.active{{background:rgba(63,185,80,0.8);border-color:rgba(63,185,80,0.5);}}
.ctrl-btn.danger{{background:rgba(180,30,30,0.75);border-color:rgba(248,81,73,0.4);}}
.ctrl-btn.danger:hover{{background:rgba(200,40,40,0.95);}}
#speed-label{{
  color:#58A6FF;font-size:11px;align-self:center;
  min-width:40px;text-align:center;
}}

/* ── Cesium UI 커스터마이징 ── */
.cesium-widget-credits{{display:none!important;}}
.cesium-viewer-bottom{{display:none!important;}}
.cesium-viewer-toolbar{{
  background:rgba(6,12,20,0.85)!important;
  border-bottom:1px solid rgba(88,166,255,0.15)!important;
}}
.cesium-button{{
  background:rgba(31,111,235,0.7)!important;
  color:#fff!important;
  border:1px solid rgba(88,166,255,0.3)!important;
}}
.cesium-timeline-bar{{background:#0D1117!important;}}
</style>
</head>
<body>
<div id="cesiumContainer"></div>
<div id="scanlines"></div>

<!-- HUD -->
<div id="hud">
  <div id="hud-logo">⬡ IMPS</div>
  <div id="hud-divider"></div>
  <div id="hud-info">
    <div id="hud-mission-label">ACTIVE MISSION</div>
    <div id="hud-mission-name">{mission_name} &nbsp;|&nbsp; BASE: {base_name} &nbsp;|&nbsp; THREATS: {threat_count}</div>
  </div>
  <div id="hud-right">
    <div id="hud-status">● SIM RUNNING</div>
    <div id="hud-clock">T+00:00:00</div>
  </div>
</div>

<!-- 자산 패널 -->
<div id="asset-panel">
  <h3>편대 자산 구성</h3>
  {asset_rows}
</div>

<!-- 위협 요약 -->
<div id="threat-summary">
  <h3>위협 현황</h3>
  <div class="threat-item" id="threat-info">위협 {threat_count}개 활성</div>
</div>

<!-- 범례 -->
<div id="legend">
  <h4>경로 범례</h4>
  {legend_rows}
  <div class="legend-sep"></div>
  <div class="legend-row"><div class="legend-line" style="background:rgba(220,50,50,0.9)"></div>SAM 위협 반경</div>
  <div class="legend-row"><div class="legend-line" style="background:rgba(160,50,230,0.9)"></div>RADAR 탐지 반경</div>
  <div class="legend-row"><div class="legend-line" style="background:rgba(255,140,0,0.9)"></div>비행금지구역 (NFZ)</div>
</div>

<!-- 컨트롤 바 -->
<div id="controls">
  <button class="ctrl-btn active" id="btn-play" onclick="togglePlay()">&#9646;&#9646; 일시정지</button>
  <button class="ctrl-btn" onclick="setSpeed(1)">1×</button>
  <button class="ctrl-btn" onclick="setSpeed(10)">10×</button>
  <button class="ctrl-btn" onclick="setSpeed(30)">30×</button>
  <button class="ctrl-btn" onclick="setSpeed(60)">60×</button>
  <button class="ctrl-btn" onclick="setSpeed(120)">120×</button>
  <div id="speed-label">30×</div>
  <button class="ctrl-btn" onclick="resetTime()">&#8635; 처음</button>
  <button class="ctrl-btn" onclick="followFirst()">&#128247; 추적</button>
  <button class="ctrl-btn" onclick="freeCamera()">&#127758; 전체뷰</button>
  <button class="ctrl-btn danger" onclick="window.close()">&#10005; 닫기</button>
</div>

<script>
// ── Cesium 초기화 (async — Cesium 1.100+ 호환) ───────────────
{token_js}

const HAS_TOKEN = {has_token_js};

async function initCesium() {{
  // ── 영상 로딩 전략 ────────────────────────────────────────────
  // 핵심: CESIUM_BASE_URL을 위에서 CDN으로 설정했으므로
  // buildModuleUrl이 CDN 경로를 반환함.
  // 그러나 로컬 file:// 에서 가장 안정적인 방법은
  // CDN URL을 직접 문자열로 전달하는 것.

  // ── 영상 전략: ESRI World Imagery (무료 위성, zoom 19) ─────────
  // UrlTemplateImageryProvider는 동기 생성 가능 (await 불필요)
  // ESRI tile URL: /MapServer/tile/{{z}}/{{y}}/{{x}} (level/row/col)
  let baseProvider;
  try {{
    baseProvider = new Cesium.UrlTemplateImageryProvider({{
      url: 'https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
      maximumLevel: 19,
      credit: 'Esri, DigitalGlobe, GeoEye, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN',
    }});
    console.log('[IMPS] ESRI World Imagery 설정 (고해상도 위성 zoom 19)');
  }} catch(e) {{
    console.warn('[IMPS] ESRI 실패, Natural Earth 폴백:', e.message);
    baseProvider = await Cesium.TileMapServiceImageryProvider.fromUrl(
      '{CESIUM_CDN}/Assets/Textures/NaturalEarthII/',
      {{ fileExtension: 'jpg', maximumLevel: 5 }}
    );
  }}
  const baseLayer = new Cesium.ImageryLayer(baseProvider);

  const viewer = new Cesium.Viewer('cesiumContainer', {{
    animation:            true,
    timeline:             true,
    baseLayerPicker:      false,
    navigationHelpButton: false,
    sceneModePicker:      true,
    geocoder:             false,
    homeButton:           true,
    fullscreenButton:     false,
    selectionIndicator:   true,
    infoBox:              true,
    shadows:              false,
    baseLayer:            baseLayer,
  }});

  // ── 토큰 있으면 3D 지형 추가 (영상은 ESRI로 충분) ─────────────
  if (HAS_TOKEN) {{
    try {{
      viewer.terrainProvider = await Cesium.CesiumTerrainProvider.fromIonAssetId(1, {{
        requestVertexNormals: true,
      }});
      console.log('[IMPS] Cesium World Terrain 로드 성공');
    }} catch(e) {{
      console.warn('[IMPS] 3D 지형 실패 (평면 유지):', e.message);
    }}
  }}

  _startApp(viewer);
}}

initCesium().catch(err => {{
  console.error('[IMPS] 초기화 실패:', err);
  document.getElementById('cesiumContainer').innerHTML =
    '<div style="color:#FF6B6B;font-size:18px;text-align:center;padding-top:40vh;">'+
    '⚠️ Cesium 초기화 실패<br><small>'+err.message+'</small></div>';
}});

function _startApp(viewer) {{

viewer.scene.globe.enableLighting        = false;
viewer.scene.globe.showGroundAtmosphere  = true;
viewer.scene.skyAtmosphere.show          = true;
viewer.scene.fog.enabled                 = false;
viewer.scene.backgroundColor             = Cesium.Color.fromCssColorString('#060C14');

// ── CZML 로드 ─────────────────────────────────────────────────
const czmlData = {czml_json};
const ds = new Cesium.CzmlDataSource();
viewer.dataSources.add(ds.load(czmlData)).then(() => {{
  console.log('[IMPS] CZML 로드 완료. 엔티티 수:', ds.entities.values.length);
}}).catch(err => {{
  console.error('[IMPS] CZML 로드 실패:', err);
  // 에러 발생 시에도 기본 카메라 설정은 유지
}});

// ── 초기 카메라: 즉시 한국 전역으로 이동 ───────────────────────
// setView = 즉시 이동 (Globe 타일이 로드된 뒤 flyTo로 전환)
viewer.camera.setView({{
  destination: Cesium.Cartesian3.fromDegrees({center_lon}, {center_lat}, 700000),
  orientation: {{
    heading: Cesium.Math.toRadians(0),
    pitch:   Cesium.Math.toRadians(-90),
    roll:    0,
  }},
}});
setTimeout(() => {{
  viewer.camera.flyTo({{
    destination: Cesium.Cartesian3.fromDegrees({center_lon}, {center_lat} - 1.5, 480000),
    orientation: {{
      heading: Cesium.Math.toRadians(0),
      pitch:   Cesium.Math.toRadians(-55),
      roll:    0,
    }},
    duration: 2.0,
  }});
}}, 1500);

// ── 클록 HUD 업데이트 ─────────────────────────────────────────
const startJd = viewer.clock.startTime;
viewer.clock.onTick.addEventListener((clock) => {{
  const elapsed = Cesium.JulianDate.secondsDifference(clock.currentTime, startJd);
  const e = Math.max(0, elapsed);
  const h = Math.floor(e / 3600);
  const m = Math.floor((e % 3600) / 60);
  const s = Math.floor(e % 60);
  document.getElementById('hud-clock').textContent =
    'T+' + String(h).padStart(2,'0') + ':' + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
}});

// ── 재생 상태 ─────────────────────────────────────────────────
let playing = true;
viewer.clock.shouldAnimate = true;
viewer.clock.multiplier    = 30;

function togglePlay() {{
  playing = !playing;
  viewer.clock.shouldAnimate = playing;
  const btn = document.getElementById('btn-play');
  if (playing) {{
    btn.textContent = '\u23F8 일시정지';
    btn.className   = 'ctrl-btn active';
    document.getElementById('hud-status').textContent = '● SIM RUNNING';
  }} else {{
    btn.textContent = '\u25B6 재생';
    btn.className   = 'ctrl-btn';
    document.getElementById('hud-status').textContent = '⏸ PAUSED';
  }}
}}

function setSpeed(x) {{
  viewer.clock.multiplier = x;
  document.getElementById('speed-label').textContent = x + '×';
}}

function resetTime() {{
  viewer.clock.currentTime   = viewer.clock.startTime.clone();
  viewer.clock.shouldAnimate = true;
  playing = true;
  document.getElementById('btn-play').textContent = '\u23F8 일시정지';
  document.getElementById('btn-play').className   = 'ctrl-btn active';
}}

let tracking = false;
let trackedEntity = null;

function followFirst() {{
  const entities = ds.entities.values;
  // 자산 엔티티만 (marker_ prefix 제외, 위협 제외)
  const assets = entities.filter(e =>
    e.position && e.position._callback === undefined &&
    !e.id.startsWith('marker_') &&
    !e.id.startsWith('threat_')
  );
  if (assets.length === 0) return;
  if (!tracking || trackedEntity !== assets[0]) {{
    trackedEntity = assets[0];
    viewer.trackedEntity = trackedEntity;
    tracking = true;
  }} else {{
    viewer.trackedEntity = undefined;
    tracking = false;
  }}
}}

function freeCamera() {{
  viewer.trackedEntity = undefined;
  tracking = false;
  viewer.camera.flyTo({{
    destination: Cesium.Cartesian3.fromDegrees({center_lon}, {center_lat}, 600000),
    orientation: {{
      heading: Cesium.Math.toRadians(0),
      pitch:   Cesium.Math.toRadians(-70),
      roll:    0,
    }},
    duration: 1.8,
  }});
}}

// ── 파티클 연기/불꽃 효과 (폭발 후 잔연) ────────────────────────
function _createSmokeParticles(viewer, lon, lat, alt, startJd) {{
  try {{
    const pos = Cesium.Cartesian3.fromDegrees(lon, lat, alt + 200);

    let emitter;
    try {{
      emitter = new Cesium.CircleEmitter(300.0);
    }} catch(e) {{
      return;  // ParticleSystem API 없으면 건너뜀
    }}

    const ps = new Cesium.ParticleSystem({{
      image: _smokeImageUri(),
      emitter,
      modelMatrix: Cesium.Transforms.eastNorthUpToFixedFrame(pos),
      startScale: 0.5,
      endScale: 6.0,
      particleLife: 12.0,
      speed: 15.0,
      imageSize: new Cesium.Cartesian2(60, 60),
      emissionRate: 40.0,
      lifetime: 15.0,
      startColor: new Cesium.Color(1.0, 0.5, 0.1, 0.9),
      endColor: new Cesium.Color(0.3, 0.3, 0.3, 0.0),
    }});

    // 폭발 시점에 맞춰 파티클 시작
    const checkHandler = viewer.clock.onTick.addEventListener(function(clock) {{
      const diff = Cesium.JulianDate.secondsDifference(clock.currentTime, startJd);
      if (diff >= 0 && diff < 0.5) {{
        viewer.scene.primitives.add(ps);
        checkHandler();  // 리스너 해제
      }}
    }});
  }} catch(e) {{
    console.warn('[IMPS] 파티클 생성 실패 (무시):', e.message);
  }}
}}

function _smokeImageUri() {{
  // 작은 원 캔버스를 data URI로 변환
  const c = document.createElement('canvas');
  c.width = 32; c.height = 32;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(16,16,0, 16,16,16);
  g.addColorStop(0.0, 'rgba(255,200,80,1)');
  g.addColorStop(0.4, 'rgba(200,100,30,0.8)');
  g.addColorStop(1.0, 'rgba(80,80,80,0)');
  ctx.fillStyle = g;
  ctx.beginPath(); ctx.arc(16,16,16,0,Math.PI*2); ctx.fill();
  return c.toDataURL();
}}

// ── 폭발 타이밍 감지 → 파티클 트리거 ────────────────────────────
try {{
  ds.entities.collectionChanged.addEventListener(function() {{
    try {{
      const impactEntities = ds.entities.values.filter(
        e => e.id && e.id.startsWith('explosion_main_')
      );
      impactEntities.forEach(function(e) {{
        if (e._particleRegistered) return;
        e._particleRegistered = true;
        try {{
          const avail = e.availability;
          if (!avail || !avail.intervals || avail.intervals.length === 0) return;
          const impactStart = avail.intervals.get(0).start;
          const pos = e.position && e.position.getValue(impactStart);
          if (pos) {{
            const carto = Cesium.Cartographic.fromCartesian(pos);
            _createSmokeParticles(
              viewer,
              Cesium.Math.toDegrees(carto.longitude),
              Cesium.Math.toDegrees(carto.latitude),
              carto.height,
              impactStart
            );
          }}
        }} catch(inner) {{
          console.warn('[IMPS] 파티클 등록 오류 (무시):', inner.message);
        }}
      }});
    }} catch(err) {{
      console.warn('[IMPS] collectionChanged 핸들러 오류:', err.message);
    }}
  }});
}} catch(e) {{
  console.warn('[IMPS] collectionChanged 리스너 등록 실패:', e.message);
}}

// ── Cesium Timeline 스타일 조정 ───────────────────────────────
try {{
  viewer.timeline.container.style.background = 'rgba(6,12,20,0.92)';
  viewer.animation.container.style.visibility = 'hidden';
}} catch(e) {{}}

// 전역 함수로 노출 (버튼 onclick에서 호출)
window.togglePlay  = togglePlay;
window.setSpeed    = setSpeed;
window.resetTime   = resetTime;
window.followFirst = followFirst;
window.freeCamera  = freeCamera;

}} // end _startApp
</script>
</body>
</html>"""


def launch_simulation(
    formation_paths: Dict,
    mission,
    czml_packets: List[dict],
    output_dir: Optional[str] = None,
) -> str:
    """
    CZML HTML 파일 생성 → 기본 브라우저로 열기

    Args:
        formation_paths: path_computer 반환값
        mission:         MissionState
        czml_packets:    czml_exporter.export_czml() 반환값
        output_dir:      저장 경로 (None이면 시스템 임시 디렉터리)

    Returns:
        생성된 HTML 파일 경로
    """
    # 지도 중심 계산 (기지 + 목표 중간)
    base_info   = {}
    from modules.config import AIRPORTS
    base_info   = AIRPORTS.get(getattr(mission.params, "start", ""), {})
    base_coords = base_info.get("coords", [37.0, 127.5])
    t_lat = getattr(mission.params, "target_lat", 38.0)
    t_lon = getattr(mission.params, "target_lon", 127.5)
    center_lat  = (base_coords[0] + t_lat) / 2
    center_lon  = (base_coords[1] + t_lon) / 2

    # NaN/Infinity 방어: allow_nan=False가 기본이지만 명시적으로 설정
    import math

    def _sanitize(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    # 파일 저장 경로 결정
    if output_dir:
        out_path = Path(output_dir) / "imps_simulation.html"
        tmp_dir  = Path(output_dir)
    else:
        tmp_dir  = Path(tempfile.gettempdir()) / "imps_sim"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out_path = tmp_dir / "imps_simulation.html"

    # ── http://localhost 포트 먼저 확정 ────────────────────────────
    port = _ensure_server(str(tmp_dir))
    model_base_url = f"http://127.0.0.1:{port}"

    # ── data/models/*.glb → tmpdir 복사 후 CZML 모델 URL 교체 ──────
    _copy_glb_models(tmp_dir)
    # CZML 패킷 내 __MODEL_BASE__ 플레이스홀더를 실제 URL로 치환
    czml_json_str = json.dumps(_sanitize(czml_packets), ensure_ascii=False)
    czml_json_str = czml_json_str.replace("__MODEL_BASE__", model_base_url)

    html = _build_html(
        czml_json      = czml_json_str,
        formation_paths= formation_paths,
        mission        = mission,
        center_lat     = center_lat,
        center_lon     = center_lon,
    )

    out_path.write_text(html, encoding="utf-8")

    url = f"{model_base_url}/{out_path.name}"
    webbrowser.open(url)

    return str(out_path)


def _copy_glb_models(tmp_dir: Path) -> None:
    """
    data/models/*.glb 파일을 tmp_dir에 복사.
    HTTP 서버가 모델 파일을 제공할 수 있게 함.
    """
    import shutil
    candidates = [
        Path("data/models"),
        Path(__file__).parent.parent / "data" / "models",
    ]
    for models_dir in candidates:
        if models_dir.exists():
            for glb in models_dir.glob("*.glb"):
                dest = tmp_dir / glb.name
                try:
                    shutil.copy2(glb, dest)
                except Exception:
                    pass
            break
