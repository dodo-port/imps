"""
LLM Brain 모듈 v2.0 - 군사 전문 참모 AI
- 모델: qwen3:14b (ollama)
- 자연어 전술 명령 → 문맥 이해 → 파라미터 자동 조정
- 교리 근거: JP3-30, AFDP3-03, AFDP5-0, FMI3-04.155
"""
import ollama
import json
import re
import concurrent.futures
from pathlib import Path
from typing import Optional, Literal, List, Tuple, Dict
from pydantic import BaseModel, Field
from modules.doctrine_rag import DoctrineRAG
from modules.config import (
    LLM_MODEL, LLM_MODEL_FALLBACK, LLM_TEMPERATURE, LLM_TIMEOUT,
    AIRPORTS, MISSION_TYPES, MARGIN_LEVELS, THREAT_DEFAULT_RADIUS
)

try:
    from pypdf import PdfReader
    HAS_PDF_READER = True
except Exception:
    HAS_PDF_READER = False


# ================================================================
# Pydantic 스키마 - 확장된 구조
# ================================================================

class ThreatInfo(BaseModel):
    """LLM이 명령에서 추출한 위협 정보"""
    name: str = Field(..., description="위협 명칭 (예: Enemy-RADAR-01)")
    type: Literal["SAM", "RADAR", "NFZ"] = Field(..., description="위협 유형")
    lat: Optional[float] = Field(None, description="위협 위도")
    lon: Optional[float] = Field(None, description="위협 경도")
    radius_km: Optional[float] = Field(None, description="위협 반경(km)")


class MissionUpdateParams(BaseModel):
    """미션 파라미터 변경 내용"""
    safety_margin_km: Optional[float] = Field(None, ge=0.0, le=50.0, description="안전 마진(km)")
    rtb: Optional[bool] = Field(None, description="복귀(Return To Base) 여부")
    waypoint_name: Optional[str] = Field(None, description="경유할 공항 이름")
    stpt_gap: Optional[int] = Field(None, ge=1, le=100, description="STPT 표시 간격 (km 단위)")
    algorithm: Optional[Literal["A*", "A* 3D", "RRT", "RRT*"]] = Field(None, description="알고리즘")
    enable_3d: Optional[bool] = Field(None, description="3D 지형 고려 여부")
    target_lat: Optional[float] = Field(None, ge=33.0, le=43.0, description="목표 위도")
    target_lon: Optional[float] = Field(None, ge=124.0, le=132.0, description="목표 경도")
    target_name: Optional[str] = Field(None, description="목표 명칭")
    start: Optional[str] = Field(None, description="출발 기지명")
    fuel_state: Optional[float] = Field(None, ge=0.2, le=1.0, description="F-16 기준 연료 상태")
    refuel_count: Optional[int] = Field(None, ge=0, le=2, description="공중급유 횟수")
    extra_targets: Optional[List[Dict]] = Field(None, description="추가 목표 목록")


class LLMResponse(BaseModel):
    """LLM 응답 전체 구조"""
    action: Literal["UPDATE", "THREAT_ADD", "MISSION_PLAN", "EXPLAIN", "CHAT"] = Field(
        ...,
        description=(
            "UPDATE: 파라미터만 변경 | "
            "THREAT_ADD: 위협 추가 + 파라미터 변경 | "
            "MISSION_PLAN: 임무 유형 분류 및 순서 결정 | "
            "EXPLAIN: 교리/전술 설명 | "
            "CHAT: 일반 대화"
        )
    )
    update_params: MissionUpdateParams = Field(
        default_factory=MissionUpdateParams,
        description="변경할 파라미터 (없으면 모두 null)"
    )
    threats_to_add: List[ThreatInfo] = Field(
        default_factory=list,
        description="추가할 위협 목록 (THREAT_ADD 액션 시 사용)"
    )
    clear_existing_threats: bool = Field(
        default=False,
        description="true이면 기존 위협을 비우고 threats_to_add를 새 배치로 적용"
    )
    mission_sequence: List[str] = Field(
        default_factory=list,
        description="임무 수행 순서 (예: ['ISR', 'SEAD', 'STRIKE'])"
    )
    response_text: str = Field(..., description="사용자에게 보여줄 한국어 응답")
    reasoning: str = Field(..., description="판단 근거 - Why(왜), What(무엇을), How(어떻게) 형식으로 Korean으로 작성")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="판단 신뢰도 0~1")


# ================================================================
# 시스템 프롬프트 - 교리 기반 군사 전문 참모
# ================================================================

SYSTEM_PROMPT_TEMPLATE = """
You are IMPS-AI, a military mission planning expert AI assistant for Korean Air Force operations.
You act as a tactical staff officer who understands military doctrine and translates commander's intent into mission parameters.

## CRITICAL LANGUAGE RULE — HIGHEST PRIORITY
- You MUST respond ONLY in Korean (한국어). This is NON-NEGOTIABLE.
- NEVER use Chinese, Japanese, or any other language. Korean ONLY.
- Even if the user writes in another language, always reply in Korean.
- response_text 필드는 반드시 한국어로만 작성하세요. 중국어/일본어 절대 금지.

## DOCTRINE BASIS
- JP 3-30 Joint Air Operations: ROE, asset allocation, JPPA 7-step planning
- AFDP 3-03 Counterland: AI(Air Interdiction), CAS, SEAR mission types
- AFDP 5-0 Planning: COA development, constraints/restraints
- FMI 3-04.155 UAS Operations: ISR, surveillance, MUM-T teaming

## DOCTRINE EXCERPTS (loaded from local files)
{doctrine_context}

## MISSION TYPES (execute in this sequence when multiple apply)
1. ISR  - Intelligence/Surveillance/Reconnaissance (정보·감시·정찰) - ALWAYS FIRST
2. SEAD - Suppression of Enemy Air Defenses (적 방공망 제압) - BEFORE STRIKE
3. STRIKE - Air Interdiction Strike (공중 타격)
4. CAS - Close Air Support (근접항공지원)

## CURRENT MISSION STATE
{state_desc}

## AVAILABLE BASES (출발 가능 기지)
{airports}

## SAFETY MARGIN REFERENCE
- 최소(2km): 위험 감수, 속도 우선
- 표준(5km): 기본값
- 안전(15km): 안전 우선 (연료 여유 있을 때)
- 최대(30km): 최대 회피

## TACTICAL INTERPRETATION RULES
- "저고도" / "지형추종" / "레이더 회피" → enable_3d=true, algorithm="A* 3D"
- "안전하게" / "연료 여유" / "우회" → safety_margin_km 증가 (현재값 + 10~15)
- "빠르게" / "직선" → safety_margin_km 감소, algorithm="A*"
- "타격 후 복귀" / "RTB" → rtb=true
- 좌표 언급 시 → target_lat/target_lon 추출
- 위협 언급 시 (레이더, SAM, 방공망) → threats_to_add에 추가
- 위협 재배치/재분포/축소 요청 시 → clear_existing_threats=true 후 threats_to_add로 새 목록 제시
- 연료/급유 요청 시 → fuel_state / refuel_count 반영

## ACTION SELECTION RULES
- 위협 정보가 언급되면 → "THREAT_ADD"
- 파라미터만 바꾸면 되면 → "UPDATE"  
- 임무 유형/순서 질문이면 → "MISSION_PLAN"
- 타당성/필요성/위험도/편대 구성 검토 질문이면 → "EXPLAIN" 또는 "CHAT" (명시적 변경 명령이 없으면 값을 바꾸지 말 것)
- 교리/전술 설명 요청이면 → "EXPLAIN"
- 그 외 대화 → "CHAT"

## KOREAN CITY COORDINATES (지명 → 좌표 정확 매핑)
### 북한 주요 도시
- 평양(Pyongyang):   lat=39.03, lon=125.75
- 함흥(Hamhung):     lat=39.92, lon=127.54
- 청진(Chongjin):    lat=41.79, lon=129.78
- 원산(Wonsan):      lat=39.15, lon=127.44
- 개성(Kaesong):     lat=37.97, lon=126.55
- 신의주(Sinuiju):   lat=40.10, lon=124.40
- 해주(Haeju):       lat=38.01, lon=125.70
- 사리원(Sariwon):   lat=38.51, lon=125.76
- 남포(Nampo):       lat=38.74, lon=125.41
- 혜산(Hyesan):      lat=41.40, lon=128.17
- 강계(Kanggye):     lat=40.97, lon=126.59
- 나진(Rajin):       lat=42.26, lon=130.30
- 단천(Dancheon):    lat=40.99, lon=129.16
### 남한 주요 도시 (공항 외)
- 서울(Seoul):       lat=37.57, lon=127.00
- 인천(Incheon):     lat=37.46, lon=126.71
- 대전(Daejeon):     lat=36.35, lon=127.38
- 울산(Ulsan):       lat=35.54, lon=129.31
- 춘천(Chuncheon):   lat=37.88, lon=127.73
- 전주(Jeonju):      lat=35.82, lon=127.15
- 포항(Pohang):      lat=36.02, lon=129.37
- 제주(Jeju):        lat=33.51, lon=126.52

CRITICAL: When a user mentions a city name, look up the EXACT coordinates from the table above.
- "함흥" → lat=39.92, lon=127.54  (NOT lat=37.x)
- "평양" → lat=39.03, lon=125.75  (NOT lon=124.x)
- Never guess or approximate city coordinates — use the table.

## OUTPUT RULES — MANDATORY
- response_text: 반드시 한국어(Korean)로만 작성. 중국어/영어 혼용 절대 금지.
- reasoning: 반드시 한국어(Korean)로만 작성. format "Why: ... / What: ... / How: ..."
- DO NOT write any Chinese characters (漢字/中文). Korean Hangul ONLY.
- All coordinates must be within Korea bounds (lat 33~43, lon 124~132)
- When user mentions a Korean city, ALWAYS use coordinates from the table above
- If threat radius not mentioned, use defaults: SAM=30km, RADAR=80km

{path_info}
"""


# ================================================================
# LLMBrain 클래스
# ================================================================

class LLMBrain:
    # 클래스 레벨 캐시: 동일 명령어 반복 시 재사용
    _response_cache: dict = {}
    _cache_enabled: bool = True
    _available_model: str = None  # 주모델 생존 여부 캐시
    _doctrine_context_cache: Optional[str] = None
    _doctrine_rag: Optional[DoctrineRAG] = None
    _model_list_cache: Optional[list] = None  # ollama.list() 결과 세션 캐시

    def __init__(self, model_name: str = LLM_MODEL):
        self.model = model_name
        self.temperature = LLM_TEMPERATURE

    @classmethod
    def set_cache_enabled(cls, enabled: bool) -> None:
        cls._cache_enabled = bool(enabled)

    @classmethod
    def clear_cache(cls) -> None:
        cls._response_cache.clear()

    @classmethod
    def cache_size(cls) -> int:
        return len(cls._response_cache)

    def _extract_pdf_text(self, pdf_path: Path, max_pages: int = 8, max_chars: int = 1800) -> str:
        if not HAS_PDF_READER:
            return ""
        try:
            reader = PdfReader(str(pdf_path))
            pages = reader.pages[:max_pages]
            text_parts = []
            for p in pages:
                txt = p.extract_text() or ""
                if txt:
                    text_parts.append(txt.strip())
                if sum(len(t) for t in text_parts) >= max_chars:
                    break
            text = "\n".join(text_parts)
            return re.sub(r"\s+", " ", text).strip()[:max_chars]
        except Exception:
            return ""

    def _extract_text_file(self, path: Path, max_chars: int = 1800) -> str:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            return re.sub(r"\s+", " ", raw).strip()[:max_chars]
        except Exception:
            return ""

    def _load_doctrine_context(self, max_files: int = 5, max_total_chars: int = 5000) -> str:
        if LLMBrain._doctrine_context_cache is not None:
            return LLMBrain._doctrine_context_cache

        doctrine_dir = Path("data/doctrine")
        files = []
        if doctrine_dir.exists():
            for ext in ("*.md", "*.txt", "*.pdf"):
                files.extend(doctrine_dir.glob(ext))

        # fallback to project doctrine summary
        fallback_doc = Path("doctrine_basis.md")
        if fallback_doc.exists():
            files.append(fallback_doc)

        if not files:
            LLMBrain._doctrine_context_cache = "No local doctrine files found."
            return LLMBrain._doctrine_context_cache

        def _priority(p: Path) -> int:
            name = p.name.lower()
            score = 0
            for kw in ("jp3", "joint", "3-03", "afdp5", "fmi3-04-155", "dafman11-260"):
                if kw in name:
                    score += 10
            if p.suffix.lower() == ".md":
                score += 2
            return -score

        files = sorted(set(files), key=lambda p: (_priority(p), p.name.lower()))

        chunks = []
        total = 0
        for path in files:
            if len(chunks) >= max_files or total >= max_total_chars:
                break
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                text = self._extract_pdf_text(path)
            else:
                text = self._extract_text_file(path)
            if not text:
                continue

            remaining = max_total_chars - total
            snippet = text[:remaining]
            if not snippet:
                break
            chunks.append(f"[{path.name}] {snippet}")
            total += len(snippet)

        if not chunks:
            chunks = ["Doctrine files exist, but no text was extracted (PDF parser unavailable or empty text)."]

        LLMBrain._doctrine_context_cache = "\n\n".join(chunks)
        return LLMBrain._doctrine_context_cache

    def _get_doctrine_context(self, query: str, max_total_chars: int = 2200) -> str:
        ctx, _ = self._get_doctrine_context_bundle(
            query=query,
            max_total_chars=max_total_chars,
            top_k=3,
        )
        return ctx

    def _get_doctrine_context_bundle(
        self,
        query: str,
        max_total_chars: int = 2200,
        top_k: int = 3,
    ):
        refs = []
        # Primary path: query-aware RAG retrieval
        try:
            if LLMBrain._doctrine_rag is None:
                LLMBrain._doctrine_rag = DoctrineRAG(
                    doctrine_dir="data/doctrine",
                    fallback_doc="doctrine_basis.md",
                )
            hits = LLMBrain._doctrine_rag.search(query=query, top_k=top_k)
            if hits:
                refs = [
                    {
                        "source": h.get("source", ""),
                        "page": int(h.get("page", 0) or 0),
                        "score": float(h.get("score", 0.0) or 0.0),
                    }
                    for h in hits
                ]
            rag_ctx = LLMBrain._doctrine_rag.format_context(
                query=query,
                top_k=top_k,
                max_chars=max_total_chars,
            )
            if rag_ctx and "No retrieved doctrine chunks." not in rag_ctx:
                return rag_ctx, refs
        except Exception:
            pass

        # Fallback path: static context load
        return self._load_doctrine_context(max_total_chars=max_total_chars), refs

    def _build_state_desc(self, current_state: dict) -> str:
        """현재 상태를 LLM이 읽기 좋은 텍스트로 변환"""
        base_desc = (
            f"출발기지: {current_state.get('start', '?')} | "
            f"목표: lat={current_state.get('target_lat', '?')}, lon={current_state.get('target_lon', '?')} "
            f"({current_state.get('target_name', '?')}) | "
            f"안전마진: {current_state.get('margin', '?')}km | "
            f"RTB: {current_state.get('rtb', '?')} | "
            f"알고리즘: {current_state.get('algorithm', 'A*')} | "
            f"3D모드: {current_state.get('enable_3d', False)}"
        )
        context = current_state.get("_llm_context")
        if not context:
            return base_desc
        try:
            context_text = json.dumps(context, ensure_ascii=False, indent=2, default=str)
        except Exception:
            context_text = str(context)
        return f"{base_desc}\n\n## CURRENT PACKAGE / RISK CONTEXT\n{context_text}"

    def _build_airports_desc(self) -> str:
        """공항 목록 텍스트 생성"""
        return ", ".join(AIRPORTS.keys())

    def _build_recent_chat_messages(self, chat_history: Optional[List[dict]], current_user_msg: str, max_turns: int = 4) -> List[dict]:
        """
        Convert mission chat history into Ollama message format.
        Keeps only recent turns to limit token usage.
        """
        if not chat_history:
            return []

        recent = chat_history[-max_turns:]
        messages: List[dict] = []
        for msg in recent:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role not in ("user", "assistant") or not content:
                continue
            messages.append({"role": role, "content": content})

        # streamlit_app adds current user message to history before LLM call.
        # Avoid sending the exact same user turn twice.
        if messages and messages[-1]["role"] == "user" and messages[-1]["content"] == current_user_msg.strip():
            messages = messages[:-1]

        # Reinforce Korean-only at user message level (qwen2.5 Chinese override prevention)
        for msg in messages:
            if msg["role"] == "user" and not msg["content"].startswith("[한국어]"):
                msg["content"] = "[한국어로만 답변] " + msg["content"]

        return messages

    def _try_parse_response(self, raw_content: str) -> dict:
        """
        LLM 응답 파싱 - 5단계 fallback 체인
        1) <think> 제거 후 직접 Pydantic 검증
        2) ```json 블록 추출 후 검증
        3) 첫 번째 {...} 블록 추출 후 검증
        4) 부분 JSON → 필수 필드 채워 구성
        5) 원본 텍스트로 최소 CHAT 응답 구성
        """
        # Step 0: <think>...</think> 제거 (qwen3 thinking mode 대응)
        content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

        # Step 1: 직접 Pydantic 검증
        try:
            return LLMResponse.model_validate_json(content).model_dump()
        except Exception:
            pass

        # Step 2: ```json ... ``` 마크다운 블록 추출
        json_block = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_block:
            try:
                return LLMResponse.model_validate_json(json_block.group(1)).model_dump()
            except Exception:
                pass

        # Step 3: 첫 번째 { ... } 블록 추출 (가장 큰 블록 우선)
        brace_matches = list(re.finditer(r'\{', content))
        for start_m in brace_matches:                      # 앞에서부터 탐색 (첫 번째 완전한 JSON 블록 우선)
            start = start_m.start()
            depth, end = 0, -1
            for i, ch in enumerate(content[start:], start):
                if ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                candidate = content[start:end]
                try:
                    return LLMResponse.model_validate_json(candidate).model_dump()
                except Exception:
                    pass

        # Step 4: 부분 JSON → 필수 필드만 채워 구성
        raw_dict: dict = {}
        if brace_matches:
            try:
                start = brace_matches[0].start()
                raw_dict = json.loads(content[start:])
            except Exception:
                pass
        if raw_dict:
            try:
                up_raw = raw_dict.get('update_params') or {}
                if isinstance(up_raw, dict):
                    up = MissionUpdateParams(**{k: v for k, v in up_raw.items() if v is not None})
                else:
                    up = MissionUpdateParams()
                return LLMResponse(
                    action=raw_dict.get('action', 'CHAT'),
                    update_params=up,
                    threats_to_add=raw_dict.get('threats_to_add', []),
                    clear_existing_threats=bool(raw_dict.get('clear_existing_threats', False)),
                    mission_sequence=raw_dict.get('mission_sequence', []),
                    response_text=str(raw_dict.get('response_text', content[:200])),
                    reasoning=str(raw_dict.get('reasoning', '부분 파싱 성공')),
                    confidence=float(raw_dict.get('confidence', 0.5)),
                ).model_dump()
            except Exception:
                pass

        # Step 5: 최후 수단 - 원본 텍스트로 최소 CHAT 응답
        safe_text = re.sub(r'<[^>]+>', '', content).strip()[:300] or '명령을 처리했습니다.'
        return LLMResponse(
            action='CHAT',
            response_text=safe_text,
            reasoning='JSON 파싱 실패 - 원본 텍스트 사용. 다시 시도해 주세요.',
            confidence=0.2,
        ).model_dump()

    def _extract_lat_lon_from_text(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract coordinates from free-form Korean/English text.
        Supports patterns like:
        - "위도 41.0, 경도 125.7"
        - "lat=41.0, lon=125.7"
        - "41.0/125.7"
        """
        if not text:
            return None, None

        lat = lon = None
        t = text.lower()

        m_lat = re.search(r"(?:위도|lat)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", t)
        m_lon = re.search(r"(?:경도|lon|lng)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", t)
        if m_lat:
            lat = float(m_lat.group(1))
        if m_lon:
            lon = float(m_lon.group(1))

        if lat is None or lon is None:
            # 'lat/lon' 또는 'lat,lon' 형태이면서 명시적인 소수점이 모두 포함되어
            # 있을 때만 매칭. "3시 / 125번"처럼 우연히 숫자가 들어간 자연어와의
            # 충돌을 피하기 위해 양쪽 모두 소수부 필수로 변경.
            pairs = self._extract_coord_pairs_from_text(t)
            if pairs:
                lat = lat if lat is not None else pairs[0][0]
                lon = lon if lon is not None else pairs[0][1]

        if lat is not None and not (33.0 <= lat <= 43.0):
            lat = None
        if lon is not None and not (124.0 <= lon <= 132.0):
            lon = None

        return lat, lon

    def _extract_coord_pairs_from_text(self, text: str) -> List[Tuple[float, float]]:
        """Extract all decimal lat/lon pairs such as 41.79/129.78."""
        if not text:
            return []
        t = text.lower()
        pairs = []
        for m in re.finditer(
            r"(?<![0-9.])([3-4][0-9]\.[0-9]+)\s*[/,]\s*([12][0-9]{2}\.[0-9]+)(?![0-9.])",
            t,
        ):
            lat = float(m.group(1))
            lon = float(m.group(2))
            if 33.0 <= lat <= 43.0 and 124.0 <= lon <= 132.0:
                pair = (lat, lon)
                if pair not in pairs:
                    pairs.append(pair)
        return pairs

    def _extract_margin_km_from_text(self, text: str) -> Optional[float]:
        """
        Extract safety margin km from text.
        Examples:
        - "안전마진 15km"
        - "마진을 12.5 km로"
        - "safety margin 10km"
        """
        if not text:
            return None

        t = text.lower()

        # Prefer numbers near margin-related keywords.
        m = re.search(
            r"(?:안전\s*마진|안전마진|마진|safety\s*margin)[^0-9]{0,16}([0-9]+(?:\.[0-9]+)?)\s*km",
            t,
        )
        if not m:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*km[^a-zA-Z0-9가-힣]{0,8}(?:안전\s*마진|안전마진|마진)", t)
        if not m:
            return None

        val = float(m.group(1))
        if 0.0 <= val <= 50.0:
            return val
        return None

    def _extract_stpt_gap_from_text(self, text: str) -> Optional[int]:
        """Extract steer point display interval in km."""
        if not text:
            return None
        t = text.lower()
        patterns = [
            r"(?:stpt|steer\s*point|steerpoint|스티어포인트|steerpint|표시\s*간격|간격)[^0-9]{0,20}([0-9]{1,3})\s*km",
            r"([0-9]{1,3})\s*km[^a-zA-Z0-9가-힣]{0,10}(?:stpt|steer\s*point|steerpoint|스티어포인트|표시\s*간격|간격)",
        ]
        for pat in patterns:
            m = re.search(pat, t)
            if m:
                val = int(m.group(1))
                if 1 <= val <= 100:
                    return val
        return None

    def _extract_max_assets_from_text(self, text: str) -> Optional[int]:
        """Extract requested max package size for formation optimization."""
        if not text:
            return None
        t = text.lower()
        patterns = [
            r"(?:최대\s*자산\s*수|최대\s*전력\s*수|n[_\s-]*max|max\s*assets)[^0-9]{0,16}([0-9]{1,2})",
            r"([0-9]{1,2})\s*(?:대|개)?[^0-9a-z가-힣]{0,8}(?:이내|까지)[^0-9a-z가-힣]{0,8}(?:편대|자산|전력)",
        ]
        for pat in patterns:
            m = re.search(pat, t)
            if m:
                val = int(m.group(1))
                if 3 <= val <= 20:
                    return val
        return None

    def _extract_refuel_count_from_text(self, text: str) -> Optional[int]:
        if not text:
            return None
        t = text.lower()
        patterns = [
            r"(?:급유|공중급유|ar)[^0-9]{0,10}([0-2])\s*회?",
            r"([0-2])\s*회?\s*(?:급유|공중급유|ar)",
        ]
        for pat in patterns:
            m = re.search(pat, t)
            if m:
                return max(0, min(2, int(m.group(1))))
        return None

    def _extract_fuel_state_from_text(self, text: str) -> Optional[float]:
        if not text:
            return None
        t = text.lower()
        m_pct = re.search(r"(?:연료|fuel)[^0-9]{0,32}([0-9]{1,3})\s*%", t)
        if m_pct:
            return max(0.2, min(1.0, round(int(m_pct.group(1)) / 100.0, 2)))

        # ratio 형태(0.0~1.0)는 반드시 소수점이 있어야 인식 — 이렇게 해야
        # "연료 1대", "fuel 2개" 같은 기수 표현과 충돌하지 않는다.
        ratio_patterns = [
            r"(?:연료|fuel)[^0-9]{0,32}(0\.[0-9]+|1\.0+)",
            r"(0\.[0-9]+|1\.0+)\s*(?:까지|수준|정도|state|상태)?[^0-9a-z가-힣]{0,8}(?:사용|반영|설정|가능)",
        ]
        for pat in ratio_patterns:
            m_ratio = re.search(pat, t)
            if m_ratio and ("연료" in t or "fuel" in t):
                return max(0.2, min(1.0, round(float(m_ratio.group(1)), 2)))
        return None

    def _extract_radius_km_from_text(self, text: str) -> Optional[float]:
        if not text:
            return None
        t = text.lower()
        m = re.search(r"(?:반경|radius)[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)\s*km", t)
        if not m:
            return None
        val = float(m.group(1))
        if 1.0 <= val <= 400.0:
            return val
        return None

    def _extract_start_from_text(self, text: str) -> Optional[str]:
        if not text:
            return None
        t = text.lower()
        for base_name in AIRPORTS.keys():
            aliases = {base_name.lower()}
            if "(" in base_name and ")" in base_name:
                aliases.add(base_name.split("(", 1)[1].split(")", 1)[0].lower())
            aliases.add(re.sub(r"\(.*?\)", "", base_name).strip().lower())
            aliases = {a for a in aliases if a}
            if any(alias in t for alias in aliases):
                return base_name
        return None

    def _extract_mission_sequence_from_text(self, text: str) -> List[str]:
        if not text:
            return []
        t = text.lower()
        mission_map = [
            ("ISR", ["isr", "정찰", "탐색", "탐지", "감시", "recon"]),
            ("SEAD", ["sead", "방공망", "sam", "radar", "레이더", "무력화", "제압"]),
            ("STRIKE", ["strike", "타격", "공격", "폭격", "목표 타격"]),
            ("CAS", ["cas", "근접항공지원", "근접 지원"]),
        ]
        seq = []
        for mission, keywords in mission_map:
            if any(k in t for k in keywords):
                seq.append(mission)
        if not seq and any(k in t for k in ["편대", "mum-t", "맘티", "패키지"]):
            seq = ["ISR", "SEAD", "STRIKE"]
        return seq

    def _looks_like_scenario_build_request(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        build_terms = [
            "시나리오", "전체 작전", "작전 계획", "임무 계획", "임무계획",
            "계획해", "작성해", "수립해", "구성해", "반영해",
            "짜줘", "짜 줘", "잡아줘", "잡아 줘", "계획 잡아",
        ]
        if not any(k in t for k in build_terms):
            return False
        scope_terms = [
            "출발", "목표", "좌표", "임무 순서", "isr", "sead", "strike",
            "경로", "알고리즘", "안전마진", "rtb", "stpt", "편대",
        ]
        scope_hits = sum(1 for k in scope_terms if k in t)
        coord_detected = any(v is not None for v in self._extract_lat_lon_from_text(text))
        return coord_detected or scope_hits >= 2

    def _looks_like_analysis_question(self, text: str) -> bool:
        """
        True when the user is asking for judgment/explanation instead of asking
        the app to mutate mission parameters.
        """
        if not text:
            return False
        if self._looks_like_scenario_build_request(text):
            return False
        t = text.strip().lower()
        analysis_terms = [
            "불필요", "필요한가", "필요해", "필요할까", "충분", "부족", "과한",
            "과다", "낭비", "중복", "타당", "적절", "괜찮", "맞나", "맞을까",
            "위험한가", "문제", "왜", "근거", "판단", "평가", "분석", "검토",
            "어떻게 생각", "어때", "추천", "권장",
        ]
        question_markers = ["?", "인가", "한가", "할까", "될까", "나요", "습니까", "까요"]
        has_analysis_signal = any(k in t for k in analysis_terms) or any(k in t for k in question_markers)
        if not has_analysis_signal:
            return False

        explicit_commands = [
            "추가해", "넣어줘", "생성해", "삭제해", "제거해", "바꿔", "변경해",
            "설정해", "적용해", "계산해", "실행해", "줄여줘", "늘려줘", "올려줘",
            "내려줘", "해제해", "켜줘", "꺼줘", "작성해", "계획해", "수립해",
            "구성해", "반영해", "활성화", "사용해", "운용해", "배정",
            "짜줘", "잡아줘",
        ]
        judgment_terms = [
            "불필요", "필요한가", "필요할까", "충분", "부족", "과한", "낭비",
            "중복", "타당", "적절", "괜찮", "왜", "근거", "판단", "평가",
            "분석", "검토", "어때",
        ]
        if any(k in t for k in explicit_commands) and not any(k in t for k in judgment_terms):
            return False
        return True

    def _looks_like_structured_command(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        coord_detected = any(v is not None for v in self._extract_lat_lon_from_text(text))
        structured_keywords = [
            "위협", "sam", "radar", "nfz", "목표", "좌표", "경로", "마진", "rtb",
            "3d", "a*", "rrt", "편대", "정찰", "sead", "strike", "cas",
            "급유", "fuel", "연료", "북쪽", "남쪽", "올려", "내려", "재배치", "분포",
            "레이더", "방공망", "추가", "더해", "넣어", "시나리오", "작전 계획",
            "임무 계획", "임무", "계획해", "작성해", "짜줘", "잡아줘",
        ]
        return coord_detected or any(k in t for k in structured_keywords)

    def _build_replacement_threats(
        self,
        current_threats: Optional[List[Dict]],
        current_state: dict,
        user_msg: str,
    ) -> List[Dict]:
        threats = current_threats or []
        # 0~99까지 명시적 숫자 인식. 이전엔 [1-9][0-9]? 였어서 "0개"가 매칭되지
        # 않아 "위협을 0개로 줄여" 명령이 디폴트(=현재 개수 또는 3개)로 떨어졌음.
        count_match = re.search(r"([0-9]{1,2})\s*개", user_msg)
        if count_match is not None:
            count = int(count_match.group(1))
        else:
            count = max(1, len(threats) or 3)
        # 0개 명령은 진짜 0개로 처리(전부 제거 의도). 상한은 8.
        count = max(0, min(count, 8))
        if count == 0:
            return []

        region_msg = (user_msg or "").lower()
        target_lat = float(current_state.get("target_lat", 39.0))
        target_lon = float(current_state.get("target_lon", 125.7))

        south_points = [(35.6, 126.7), (36.2, 127.5), (36.8, 128.2), (35.9, 129.0), (37.1, 126.9)]
        north_points = [(37.7, 126.6), (38.2, 127.2), (38.7, 127.9), (39.1, 126.9), (39.4, 128.1)]
        nationwide_points = [(36.1, 126.8), (36.9, 127.7), (37.6, 128.5), (38.3, 126.8), (39.0, 127.8)]

        if any(k in region_msg for k in ["남한", "남부", "south"]):
            anchor_points = south_points
        elif any(k in region_msg for k in ["북한", "북부", "north"]):
            anchor_points = north_points
        elif any(k in region_msg for k in ["목표 주변", "타겟 주변", "target area"]):
            anchor_points = [
                (target_lat - 0.7, target_lon - 0.4),
                (target_lat - 0.2, target_lon - 0.7),
                (target_lat + 0.3, target_lon - 0.3),
                (target_lat + 0.4, target_lon + 0.3),
                (target_lat - 0.3, target_lon + 0.4),
            ]
        else:
            anchor_points = nationwide_points

        type_cycle = [t.get("type", "SAM") for t in threats if t.get("type") in ("SAM", "RADAR")]
        if not type_cycle:
            type_cycle = ["SAM", "RADAR", "SAM"]

        radius_by_type = {}
        for t_type in ("SAM", "RADAR"):
            vals = [float(t.get("radius_km", THREAT_DEFAULT_RADIUS[t_type])) for t in threats if t.get("type") == t_type]
            radius_by_type[t_type] = round(sum(vals) / len(vals), 1) if vals else THREAT_DEFAULT_RADIUS[t_type]

        rebuilt = []
        for idx in range(count):
            lat, lon = anchor_points[idx % len(anchor_points)]
            t_type = type_cycle[idx % len(type_cycle)]
            rebuilt.append(
                {
                    "name": f"{t_type}-AUTO-{idx+1:02d}",
                    "type": t_type,
                    "lat": round(min(43.0, max(33.0, lat)), 4),
                    "lon": round(min(132.0, max(124.0, lon)), 4),
                    "radius_km": float(radius_by_type.get(t_type, THREAT_DEFAULT_RADIUS.get(t_type, 30.0))),
                }
            )
        return rebuilt

    def _build_presentation_threat_layout(self) -> List[Dict]:
        """Default North Korea SAM/RADAR layout for demo scenario prompts."""
        return [
            {
                "name": "NK-RADAR-West-Screen",
                "type": "RADAR",
                "lat": 39.50,
                "lon": 126.05,
                "radius_km": 58.0,
            },
            {
                "name": "NK-RADAR-East-Screen",
                "type": "RADAR",
                "lat": 39.95,
                "lon": 127.95,
                "radius_km": 60.0,
            },
            {
                "name": "NK-SAM-Central-Belt",
                "type": "SAM",
                "lat": 39.35,
                "lon": 126.80,
                "radius_km": 22.0,
            },
            {
                "name": "NK-SAM-North-Guard",
                "type": "SAM",
                "lat": 40.65,
                "lon": 127.50,
                "radius_km": 22.0,
            },
        ]

    def _fast_rule_based_response(
        self,
        user_msg: str,
        current_state: dict,
        current_threats: Optional[List[Dict]] = None,
    ) -> Optional[dict]:
        if not self._looks_like_structured_command(user_msg):
            return None

        msg = (user_msg or "").strip()
        msg_lower = msg.lower()

        update: Dict[str, object] = {}
        threats_to_add: List[Dict] = []
        clear_existing_threats = False
        mission_sequence = self._extract_mission_sequence_from_text(msg)
        scenario_build = self._looks_like_scenario_build_request(msg)
        formation_requested = scenario_build or any(
            k in msg_lower
            for k in ["편대", "mum-t", "맘티", "패키지", "자폭uav", "정찰uav", "전투기", "자폭 uav", "정찰 uav"]
        )
        max_assets = self._extract_max_assets_from_text(msg)
        action = "CHAT"

        coord_pairs = self._extract_coord_pairs_from_text(msg)
        target_lat, target_lon = self._extract_lat_lon_from_text(msg)
        if target_lat is not None:
            update["target_lat"] = target_lat
        if target_lon is not None:
            update["target_lon"] = target_lon
        if scenario_build and coord_pairs:
            if "강계" in msg_lower or "kanggye" in msg_lower:
                update["target_name"] = "TGT-A Kanggye C2"
            elif "청진" in msg_lower or "chongjin" in msg_lower:
                update["target_name"] = "TGT-A Chongjin C2"
            elif "북부" in msg_lower or "북한 북부" in msg_lower:
                update["target_name"] = "TGT-A North C2"
        if scenario_build and len(coord_pairs) > 1:
            extra_targets = []
            for idx, (lat, lon) in enumerate(coord_pairs[1:3], start=2):
                if ("혜산" in msg_lower or "hyesan" in msg_lower) and idx == 2:
                    name = "TGT-B Hyesan Logistics"
                elif "북부" in msg_lower or "북한 북부" in msg_lower:
                    name = f"TGT-{chr(63 + idx)} North Objective"
                else:
                    name = f"Target-{idx}"
                extra_targets.append({
                    "lat": lat,
                    "lon": lon,
                    "name": name,
                    "priority": min(idx, 3),
                })
            update["extra_targets"] = extra_targets

        margin = self._extract_margin_km_from_text(msg)
        if margin is not None:
            update["safety_margin_km"] = margin

        stpt_gap = self._extract_stpt_gap_from_text(msg)
        if stpt_gap is not None:
            update["stpt_gap"] = stpt_gap

        refuel_count = self._extract_refuel_count_from_text(msg)
        if refuel_count is not None:
            update["refuel_count"] = refuel_count

        fuel_state = self._extract_fuel_state_from_text(msg)
        if fuel_state is not None:
            update["fuel_state"] = fuel_state

        start_base = self._extract_start_from_text(msg)
        if start_base:
            update["start"] = start_base

        if any(k in msg_lower for k in ["3d", "지형", "저고도", "레이더 회피", "terrain"]):
            update["enable_3d"] = True
            update["algorithm"] = "A* 3D"
        if any(k in msg_lower for k in ["빠르게", "직선", "최단", "신속"]):
            update["algorithm"] = "A*"
        if "rtb" in msg_lower or "복귀" in msg_lower:
            update["rtb"] = True
        if any(k in msg_lower for k in ["rtb 해제", "복귀 해제"]):
            update["rtb"] = False

        cur_lat = float(current_state.get("target_lat", 39.0))
        if any(k in msg_lower for k in ["북쪽", "위쪽", "올려", "상향", "north"]) and update.get("target_lat") is None:
            update["target_lat"] = min(43.0, round(cur_lat + 1.0, 4))
        if any(k in msg_lower for k in ["남쪽", "아래쪽", "내려", "하향", "south"]) and update.get("target_lat") is None:
            update["target_lat"] = max(33.0, round(cur_lat - 1.0, 4))

        wants_threat_relayout = "위협" in msg_lower and any(
            k in msg_lower for k in ["재배치", "재분배", "분포", "분산", "고루", "줄여", "줄여줘", "옮겨"]
        )
        if wants_threat_relayout:
            threats_to_add = self._build_replacement_threats(current_threats, current_state, msg)
            clear_existing_threats = True
            action = "THREAT_ADD"

        has_named_threat = any(k in msg_lower for k in ["sam", "radar", "nfz", "방공망", "레이더"])
        wants_presentation_threat_layout = (
            scenario_build
            and has_named_threat
            and any(k in msg_lower for k in ["배치", "깔아", "구성", "설정", "넣어", "추가"])
            and not any(k in msg_lower for k in ["있으면", "걸리면", "탐지되면"])
            and not any(k in msg_lower for k in ["깔아둔", "기준으로", "기준"])
        )
        if wants_presentation_threat_layout:
            threats_to_add = self._build_presentation_threat_layout()
            clear_existing_threats = True
            action = "THREAT_ADD"

        # 단순 추가 명령 ("위협 추가", "SAM 하나 더 넣어" 등) — 좌표 없이도 동작.
        # 기존 위협을 유지하면서 새 위협 1개를 mission target 부근에 추가한다.
        wants_threat_add = (
            ("위협" in msg_lower or "방공망" in msg_lower or "sam" in msg_lower
             or "radar" in msg_lower or "레이더" in msg_lower or "nfz" in msg_lower)
            and any(k in msg_lower for k in ["추가", "더해", "더 넣어", "넣어", "만들어", "생성", "add"])
            and not wants_threat_relayout
        )
        if wants_threat_add and not threats_to_add:
            # 위협 type 추정
            if "nfz" in msg_lower:
                t_type = "NFZ"
            elif any(k in msg_lower for k in ["radar", "레이더"]):
                t_type = "RADAR"
            else:
                t_type = "SAM"
            n_open = self._extract_radius_km_from_text(msg) or THREAT_DEFAULT_RADIUS.get(t_type, 30.0)
            tgt_lat = float(current_state.get("target_lat", 39.0))
            tgt_lon = float(current_state.get("target_lon", 125.7))
            # 좌표 명시 시 우선
            ex_lat, ex_lon = self._extract_lat_lon_from_text(msg)
            new_lat = ex_lat if ex_lat is not None else round(tgt_lat - 0.6, 4)
            new_lon = ex_lon if ex_lon is not None else round(tgt_lon - 0.3, 4)
            existing_n = len(current_threats or [])
            new_threat = {
                "name": f"{t_type}-AUTO-{existing_n + 1:02d}",
                "type": t_type,
                "lat": new_lat,
                "lon": new_lon,
                "radius_km": float(n_open),
            }
            if t_type == "NFZ":
                # NFZ는 사각형이라 좌표 boxing 필요 — radius_km을 사용해 약식 박스 생성
                d = float(n_open) / 110.57
                new_threat.update(
                    lat=None, lon=None,
                    lat_min=round(new_lat - d, 4), lat_max=round(new_lat + d, 4),
                    lon_min=round(new_lon - d, 4), lon_max=round(new_lon + d, 4),
                )
            threats_to_add = [new_threat]
            clear_existing_threats = False
            action = "THREAT_ADD"

        if not threats_to_add and has_named_threat and not scenario_build:
            lat, lon = self._extract_lat_lon_from_text(msg)
            if lat is not None and lon is not None:
                if "nfz" in msg_lower:
                    t_type = "NFZ"
                elif any(k in msg_lower for k in ["radar", "레이더"]):
                    t_type = "RADAR"
                else:
                    t_type = "SAM"
                threats_to_add = [{
                    "name": f"{t_type}-AUTO-01",
                    "type": t_type,
                    "lat": lat,
                    "lon": lon,
                    "radius_km": self._extract_radius_km_from_text(msg) or THREAT_DEFAULT_RADIUS.get(t_type, 30.0),
                }]
                action = "THREAT_ADD"

        if mission_sequence and action == "CHAT":
            action = "MISSION_PLAN"

        if any(v is not None for v in update.values()) and action == "CHAT":
            action = "UPDATE"

        if action == "CHAT":
            return None

        summary_parts = []
        if update.get("target_lat") is not None or update.get("target_lon") is not None:
            summary_parts.append(
                f"목표 좌표를 lat={update.get('target_lat', current_state.get('target_lat'))}, "
                f"lon={update.get('target_lon', current_state.get('target_lon'))}로 반영합니다."
            )
        if update.get("safety_margin_km") is not None:
            summary_parts.append(f"안전 마진을 {update['safety_margin_km']}km로 조정합니다.")
        if update.get("stpt_gap") is not None:
            summary_parts.append(f"STPT 표시 간격은 {update['stpt_gap']}km로 반영합니다.")
        if update.get("extra_targets"):
            summary_parts.append(f"추가 목표 {len(update['extra_targets'])}개를 반영합니다.")
        if update.get("algorithm"):
            summary_parts.append(f"경로 알고리즘은 {update['algorithm']}로 설정합니다.")
        if update.get("start"):
            summary_parts.append(f"출발 기지는 {update['start']}으로 변경합니다.")
        if update.get("fuel_state") is not None:
            summary_parts.append(f"연료 상태는 F-16 기준 {update['fuel_state']:.2f}로 반영합니다.")
        if update.get("refuel_count") is not None:
            summary_parts.append(f"공중급유 횟수는 {update['refuel_count']}회로 반영합니다.")
        if clear_existing_threats and threats_to_add:
            summary_parts.append(f"기존 위협을 대체하고 {len(threats_to_add)}개 위협으로 재배치합니다.")
        elif threats_to_add:
            summary_parts.append(f"위협 {len(threats_to_add)}개를 추가합니다.")
        if mission_sequence:
            summary_parts.append(f"임무 순서는 {' → '.join(mission_sequence)}로 정리합니다.")
        if formation_requested and mission_sequence:
            summary_parts.append("편대 구성 최적화를 자동 실행합니다.")

        evidence_lines = []
        applied_lines = []
        logic_lines = []
        limit_lines = []

        coord_count = 0
        if update.get("target_lat") is not None and update.get("target_lon") is not None:
            coord_count += 1
            evidence_lines.append(
                f"주목표 좌표 lat={update['target_lat']:.4f}, lon={update['target_lon']:.4f}를 입력에서 확인했습니다."
            )
            applied_lines.append("주목표 좌표를 임무 파라미터에 반영했습니다.")
        if update.get("extra_targets"):
            coord_count += len(update["extra_targets"])
            target_summaries = []
            for idx, tgt in enumerate(update["extra_targets"], start=2):
                try:
                    target_summaries.append(
                        f"T{idx} lat={float(tgt['lat']):.4f}, lon={float(tgt['lon']):.4f}"
                    )
                except Exception:
                    continue
            if target_summaries:
                evidence_lines.append(f"추가 목표 좌표 {len(target_summaries)}개를 확인했습니다: {', '.join(target_summaries)}.")
                applied_lines.append("추가 목표는 우선순위 순서로 편대/경로 계획에 넘겼습니다.")

        if update.get("start"):
            evidence_lines.append(f"출발 기지 요청을 '{update['start']}'로 해석했습니다.")
            applied_lines.append(f"출발 기지를 {update['start']}으로 설정했습니다.")
        if update.get("algorithm"):
            evidence_lines.append(f"경로 알고리즘 요청을 '{update['algorithm']}'로 해석했습니다.")
            applied_lines.append(f"경로 알고리즘을 {update['algorithm']}로 설정했습니다.")
        if update.get("enable_3d") is not None:
            applied_lines.append(f"3D 지형 고려 모드를 {'활성화' if update['enable_3d'] else '비활성화'}했습니다.")
        if update.get("safety_margin_km") is not None:
            applied_lines.append(f"안전 마진을 {update['safety_margin_km']}km로 적용했습니다.")
        if update.get("stpt_gap") is not None:
            applied_lines.append(f"STPT 표시 간격을 {update['stpt_gap']}km로 적용했습니다.")
        if update.get("rtb") is not None:
            applied_lines.append(f"타격 후 복귀(RTB)를 {'활성화' if update['rtb'] else '비활성화'}했습니다.")
        if update.get("fuel_state") is not None:
            applied_lines.append(f"연료 상태를 F-16 기준 {update['fuel_state']:.2f}로 적용했습니다.")
        if update.get("refuel_count") is not None:
            applied_lines.append(f"공중급유 횟수를 {update['refuel_count']}회로 적용했습니다.")

        if mission_sequence:
            evidence_lines.append(f"임무 순서 키워드를 {' → '.join(mission_sequence)}로 확인했습니다.")
            applied_lines.append(f"임무 순서를 {' → '.join(mission_sequence)}로 정리했습니다.")
            if "STRIKE" in mission_sequence and "SEAD" in mission_sequence:
                logic_lines.append("STRIKE 전 SEAD를 선행하도록 하여 방공망 제압 후 타격 흐름을 유지했습니다.")
            if mission_sequence and mission_sequence[0] == "ISR":
                logic_lines.append("ISR을 첫 단계로 두어 목표 주변 정보 수집을 선행하도록 했습니다.")

        if clear_existing_threats and threats_to_add:
            applied_lines.append(f"기존 위협을 대체하고 새 위협 {len(threats_to_add)}개를 반영했습니다.")
        elif threats_to_add:
            applied_lines.append(f"새 위협 {len(threats_to_add)}개를 추가했습니다.")
        else:
            current_threat_count = len(current_threats or [])
            if current_threat_count:
                limit_lines.append(f"새 위협 생성 요청이 아니므로 현재 배치된 SAM/RADAR {current_threat_count}개를 유지했습니다.")

        if formation_requested and mission_sequence:
            logic_lines.append("MUM-T/편대/UAV 임무 요청이 있어 편대 구성 최적화를 자동 실행하도록 연결했습니다.")
            logic_lines.append("정찰UAV는 ISR, 자폭UAV는 SEAD, 전투기는 STRIKE 임무에 우선 배정하는 규칙을 사용했습니다.")

        if coord_count == 0:
            limit_lines.append("새 좌표가 명시되지 않아 기존 목표 좌표를 유지했습니다.")
        if not threats_to_add:
            limit_lines.append("명시되지 않은 위협 반경, 고도, 세부 센서 파라미터는 기존 시나리오 값을 유지했습니다.")
        limit_lines.append("빠른 규칙 경로는 입력에서 명확히 추출 가능한 값만 변경하고, 불확실한 값은 임의 추정하지 않습니다.")

        def _section(title: str, lines: list) -> str:
            if not lines:
                return f"**{title}**\n- 해당 없음"
            return f"**{title}**\n" + "\n".join(f"- {line}" for line in lines)

        reasoning = "\n\n".join([
            _section("입력에서 확인한 사실", evidence_lines),
            _section("실제 반영한 값", applied_lines),
            _section("판단 논리", logic_lines),
            _section("유지한 값과 한계", limit_lines),
        ])

        response = {
            "action": action,
            "update_params": update,
            "threats_to_add": threats_to_add,
            "clear_existing_threats": clear_existing_threats,
            "mission_sequence": mission_sequence,
            "response_text": " ".join(summary_parts) or "요청하신 내용을 바로 반영합니다.",
            "reasoning": reasoning,
            "confidence": 0.95,
            "_model_used": "fast-rule",
            "_cache_hit": False,
            "_doctrine_refs": [],
        }
        if formation_requested and mission_sequence:
            response["_auto_run_formation"] = True
            if max_assets is not None:
                response["_auto_formation_max_assets"] = max_assets
        return response

    def _normalize_result(self, result: dict, user_msg: str, current_state: dict) -> dict:
        """
        Make model output safer and consistent with UI updates.
        - If text contains coordinates but update_params is empty, backfill target_lat/lon.
        - If user asked north/south shift but model omitted coordinates, apply small deterministic shift.
        - If any update field exists, coerce action to UPDATE unless THREAT_ADD/MISSION_PLAN.
        """
        update = result.get("update_params") or {}
        action = result.get("action", "CHAT")
        analysis_question = self._looks_like_analysis_question(user_msg)

        if analysis_question:
            result["action"] = "CHAT" if action == "CHAT" else "EXPLAIN"
            result["update_params"] = {}
            result["threats_to_add"] = []
            result["clear_existing_threats"] = False
            return result

        # 1) backfill coordinates from model response text
        t_lat, t_lon = self._extract_lat_lon_from_text(result.get("response_text", ""))
        if t_lat is not None and update.get("target_lat") is None:
            update["target_lat"] = t_lat
        if t_lon is not None and update.get("target_lon") is None:
            update["target_lon"] = t_lon

        # 2) deterministic directional fallback when user asks "north/up" etc.
        msg = (user_msg or "").lower()
        cur_lat = float(current_state.get("target_lat", 39.0))
        directional_requested = any(k in msg for k in ["북쪽", "위쪽", "올려", "상향", "north"])
        if directional_requested and update.get("target_lat") is None:
            update["target_lat"] = min(43.0, round(cur_lat + 1.0, 4))

        south_requested = any(k in msg for k in ["남쪽", "아래쪽", "내려", "하향", "south"])
        if south_requested and update.get("target_lat") is None:
            update["target_lat"] = max(33.0, round(cur_lat - 1.0, 4))

        # 3) if update fields exist, ensure action is UPDATE (unless stronger action already chosen)
        # margin normalization
        m_resp = self._extract_margin_km_from_text(result.get("response_text", ""))
        m_user = self._extract_margin_km_from_text(user_msg or "")
        if update.get("safety_margin_km") is None:
            if m_resp is not None:
                update["safety_margin_km"] = m_resp
            elif m_user is not None:
                update["safety_margin_km"] = m_user

        # directional request with no explicit value -> deterministic step
        fuel_related = any(k in msg for k in ["연료", "fuel", "급유", "ar"])
        if update.get("safety_margin_km") is None and not fuel_related and not analysis_question:
            if any(k in msg for k in ["안전", "우회", "회피", "위험"]):
                try:
                    cur_margin = float(current_state.get("margin", 5.0))
                except Exception:
                    cur_margin = 5.0
                update["safety_margin_km"] = min(50.0, round(cur_margin + 10.0, 2))

        fuel_from_text = self._extract_fuel_state_from_text(result.get("response_text", "")) or self._extract_fuel_state_from_text(user_msg or "")
        if update.get("fuel_state") is None and fuel_from_text is not None:
            update["fuel_state"] = fuel_from_text

        refuel_from_text = self._extract_refuel_count_from_text(result.get("response_text", "")) or self._extract_refuel_count_from_text(user_msg or "")
        if update.get("refuel_count") is None and refuel_from_text is not None:
            update["refuel_count"] = refuel_from_text

        # if update fields exist, ensure action is UPDATE (unless stronger action already chosen)
        has_update = any(v is not None for v in update.values())
        if has_update and action not in ("THREAT_ADD", "MISSION_PLAN"):
            result["action"] = "UPDATE"
        if result.get("clear_existing_threats") and result.get("threats_to_add") and result.get("action") == "UPDATE":
            result["action"] = "THREAT_ADD"

        result["update_params"] = update
        return result

    def _store_cached_result(self, cache_key: str, result: dict) -> None:
        if not LLMBrain._cache_enabled:
            return
        if len(LLMBrain._response_cache) >= 50:
            oldest = next(iter(LLMBrain._response_cache))
            del LLMBrain._response_cache[oldest]
        # 호출자(streamlit_app)가 update_params/threats_to_add 같은 내부 dict/list를
        # 변경할 수 있으므로 deep copy로 저장. 이전 shallow copy는 mutable 내부
        # 객체를 공유해 다음 캐시 hit 때 변형된 값이 반환되는 캐시 오염 위험이 있었음.
        import copy as _copy
        LLMBrain._response_cache[cache_key] = _copy.deepcopy(result)

    def parse_tactical_command(
        self,
        user_msg: str,
        current_state: dict,
        path_analysis: dict = None,
        chat_history: Optional[List[dict]] = None,
        threat_signature: Optional[str] = None,
        current_threats: Optional[List[Dict]] = None,
    ) -> dict:
        """
        자연어 전술 명령 파싱 → 구조화된 응답 반환

        Returns:
            dict with keys:
                action, update_params, threats_to_add,
                mission_sequence, response_text, reasoning, confidence
        """
        import hashlib, json as _json

        # ── 간단 응답 캐시 (동일 명령+상태 조합은 LLM 재호출 스킵) ──
        history_sig = []
        if chat_history:
            for m in chat_history[-6:]:
                history_sig.append({
                    "role": m.get("role"),
                    "content": (m.get("content") or "")[:120],
                })

        # path_analysis(현 위험도/거리)와 extra_targets(다중표적)가 키에 포함되지
        # 않으면, 사용자가 안전마진/추가목표를 바꾼 뒤 같은 명령("RTB 해제")을
        # 내렸을 때 옛 캐시가 반환되어 새 상황을 무시하는 결함이 있었음.
        # stpt_gap도 비행 경로 해상도에 영향이 커서 키에 포함.
        _path_sig = ""
        if path_analysis:
            try:
                _path_sig = _json.dumps({
                    'avg_risk': round(float(path_analysis.get('avg_risk', 0.0)), 2),
                    'max_risk': round(float(path_analysis.get('max_risk', 0.0)), 2),
                    'dist_km':  round(float(path_analysis.get('total_distance_km', 0.0)), 0),
                    'wp':       int(path_analysis.get('waypoint_count', 0)),
                }, sort_keys=True)
            except Exception:
                _path_sig = ""

        _extra_sig = []
        for et in (current_state.get('extra_targets') or []):
            if isinstance(et, dict):
                _extra_sig.append({
                    'lat': round(float(et.get('lat', 0.0) or 0.0), 3),
                    'lon': round(float(et.get('lon', 0.0) or 0.0), 3),
                    'priority': int(et.get('priority', 2) or 2),
                })

        _context_sig = ""
        if current_state.get("_llm_context"):
            try:
                _context_sig = _json.dumps(
                    current_state.get("_llm_context"),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )[:3000]
            except Exception:
                _context_sig = str(current_state.get("_llm_context"))[:3000]

        _state_key = _json.dumps({
            'router': 'scenario-auto-formation-v5',
            'msg':   user_msg.strip().lower(),
            'state': {k: current_state.get(k) for k in
                      ('target_lat','target_lon','start','algorithm','margin','stpt_gap',
                       'enable_3d','rtb','fuel_state','refuel_count')},
            'history': history_sig,
            'threat_sig': threat_signature or "",
            'path_sig':   _path_sig,
            'extra':      _extra_sig,
            'context':    _context_sig,
        }, ensure_ascii=False, sort_keys=True)
        _cache_key = hashlib.md5(_state_key.encode()).hexdigest()
        if LLMBrain._cache_enabled and _cache_key in LLMBrain._response_cache:
            import copy as _copy
            cached = _copy.deepcopy(LLMBrain._response_cache[_cache_key])
            # '? ' prefix는 ASCII 깨짐 잔재 — 사용자에게 의미 전달이 안 되어 제거.
            # 캐시 hit 표시는 _model_used/_cache_hit 메타로만 처리.
            cached['_model_used']   = 'cache'
            cached['_cache_hit'] = True
            return cached

        analysis_question = self._looks_like_analysis_question(user_msg)
        if not analysis_question:
            fast_result = self._fast_rule_based_response(
                user_msg=user_msg,
                current_state=current_state,
                current_threats=current_threats,
            )
            if fast_result:
                fast_result = self._normalize_result(fast_result, user_msg, current_state)
                self._store_cached_result(_cache_key, fast_result)
                return fast_result

        state_desc = self._build_state_desc(current_state)
        airports_desc = self._build_airports_desc()

        path_info = ""
        if path_analysis:
            path_info = (
                f"## CURRENT PATH ANALYSIS\n"
                f"Max Risk: {path_analysis.get('max_risk', 0):.2f} | "
                f"Waypoints: {path_analysis.get('waypoint_count', 0)} | "
                f"Distance: {path_analysis.get('total_distance_km', 0):.1f}km"
            )

        doctrine_context, doctrine_refs = self._get_doctrine_context_bundle(
            user_msg,
            max_total_chars=3500,
            top_k=5,
        )

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            state_desc=state_desc,
            airports=airports_desc,
            path_info=path_info,
            doctrine_context=doctrine_context
        )

        history_messages = self._build_recent_chat_messages(chat_history, user_msg, max_turns=4)

        # 모델 순서를 동적으로 결정 (ollamatags로 사용 가능한 모델 확인)
        def _get_model_list():
            """ollama에서 사용 가능한 모델 리스트 반환 (세션 내 캐시)"""
            if LLMBrain._model_list_cache is not None:
                return LLMBrain._model_list_cache
            try:
                models_resp = ollama.list()
                raw_models = models_resp.get('models', []) if isinstance(models_resp, dict) else (models_resp.models if hasattr(models_resp, 'models') else [])
                LLMBrain._model_list_cache = [
                    (m['name'] if isinstance(m, dict) else getattr(m, 'model', getattr(m, 'name', str(m))))
                    for m in raw_models
                ]
                return LLMBrain._model_list_cache
            except Exception:
                return []

        available_models = _get_model_list()
        structured_intent = self._looks_like_structured_command(user_msg) and not analysis_question
        priority_models = []
        if available_models:
            fast_models = [m for m in available_models
                          if any(tag in m for tag in ['3b','1b','7b','8b','mistral','phi3','gemma2:2b'])]
            if structured_intent and fast_models:
                priority_models.extend(fast_models[:1])
        priority_models.append(self.model)
        if available_models:
            fast_models = [m for m in available_models
                          if any(tag in m for tag in ['3b','1b','7b','8b','mistral','phi3','gemma2:2b'])]
            if fast_models:
                priority_models.extend(fast_models[:1])
        priority_models.append(LLM_MODEL_FALLBACK)
        # deduplicate while preserving order
        models_to_try = list(dict.fromkeys(priority_models))

        # 사용 가능한 모델이 없으면 즉시 폴백
        if not models_to_try:
            return self._fallback_response("사용 가능한 모델 없음")

        # 모델 순서대로 시도 (빠른 모델 우선 → qwen3:14b → fallback)
        # 각 모델당 파싱 실패 시 confidence<0.3 이면 1회 재시도
        for model in models_to_try:
            last_exc: Optional[str] = None
            for attempt in range(2):  # 최대 2회 시도 (원본 + 재시도 1회)
                try:
                    _enforced_user_msg = "[한국어로만 답변] " + user_msg
                    _messages = (
                        [{'role': 'system', 'content': system_prompt}]
                        + history_messages
                        + [{'role': 'user', 'content': _enforced_user_msg}]
                    )
                    _options = {
                        'temperature': self.temperature if attempt == 0 else 0.05,
                        # 대화형 질문은 토큰을 줄여 속도 향상
                        'num_predict': 512 if structured_intent else 640,
                        # System prompt에 doctrine excerpt(~2200자) + 8턴 히스토리 + 사용자 입력이
                        # 들어가면 2048~2560은 잘림 위험이 있어 시스템 지침이 사라지는 silent
                        # degradation이 발생함. structured는 짧은 JSON이라 4096, 대화형은 6144로
                        # 헤드룸 확보(qwen3:14b 32K 컨텍스트 지원 가정).
                        'num_ctx':    4096 if structured_intent else 6144,
                        'think': False,
                    }
                    # concurrent.futures로 타임아웃 강제 적용
                    # 이전 구현은 `with ThreadPoolExecutor(...) as _pool` 컨텍스트
                    # 종료 시점에 `shutdown(wait=True)`로 동작해 타임아웃이 떴어도
                    # 이미 실행 중인 ollama.chat가 끝날 때까지 메인 스레드가 블로킹
                    # 됐음. 즉 LLM_TIMEOUT 보호가 사실상 무효. 명시적으로 executor를
                    # 만든 뒤 timeout 시 shutdown(wait=False, cancel_futures=True)로
                    # 즉시 반환하여 폴백 모델로 빠르게 전환되도록 한다.
                    _pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    _future = _pool.submit(
                        ollama.chat,
                        model=model,
                        messages=_messages,
                        format=LLMResponse.model_json_schema(),
                        options=_options,
                    )
                    try:
                        response = _future.result(timeout=LLM_TIMEOUT)
                        _pool.shutdown(wait=False)
                    except concurrent.futures.TimeoutError:
                        # cancel_futures=True는 미실행 작업만 취소 — 이미 실행 중인
                        # ollama.chat는 백그라운드에서 마저 돌게 두고(별도 스레드)
                        # 메인 흐름은 즉시 폴백 모델로 진행.
                        try:
                            _future.cancel()
                        except Exception:
                            pass
                        try:
                            _pool.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            # Python 3.8 호환: cancel_futures kwarg 미지원.
                            _pool.shutdown(wait=False)
                        raise TimeoutError(f"LLM 응답 초과 ({LLM_TIMEOUT}s) — 모델: {model}")

                    raw_content = response['message']['content']
                    result = self._try_parse_response(raw_content)

                    # 파싱 신뢰도가 낮으면(Step 4/5 fallback) 재시도
                    if result.get('confidence', 1.0) < 0.3 and attempt == 0:
                        continue

                    # 중국어 응답 감지: response_text에 CJK 한자 비율이 높으면 재시도
                    _rt = result.get('response_text', '')
                    _cjk_count = sum(1 for c in _rt if '\u4e00' <= c <= '\u9fff')
                    if _cjk_count > 5 and attempt == 0:
                        # 중국어 응답 → 한 번 더 시도
                        continue
                    if _cjk_count > 5:
                        # 재시도도 중국어면 response_text만 교체
                        result['response_text'] = '(언어 오류) 다시 시도해 주세요. 임무 파라미터는 정상 업데이트됐습니다.'

                    result = self._normalize_result(result, user_msg, current_state)
                    result["_doctrine_refs"] = doctrine_refs

                    # 공항 유효성 검증
                    wp = result['update_params'].get('waypoint_name')
                    if wp and wp not in AIRPORTS:
                        result['update_params']['waypoint_name'] = None
                        result['response_text'] += f" (⚠️ '{wp}' 기지 없음)"

                    start_base = result['update_params'].get('start')
                    if start_base and start_base not in AIRPORTS:
                        result['update_params']['start'] = None
                        result['response_text'] += f" (⚠️ '{start_base}' 기지 없음)"

                    # 위협 기본 반경 보정
                    for threat in result.get('threats_to_add', []):
                        if threat.get('radius_km') is None:
                            threat['radius_km'] = THREAT_DEFAULT_RADIUS.get(
                                threat.get('type', 'SAM'), 30.0
                            )

                    result['_model_used'] = model
                    result['_cache_hit'] = False
                    result['_attempt'] = attempt + 1
                    self._store_cached_result(_cache_key, result)
                    return result

                except Exception as e:
                    last_exc = str(e)
                    if attempt == 0:
                        continue  # 같은 모델로 재시도
                    break  # 재시도도 실패 → 다음 모델로

            if model == models_to_try[-1]:
                # 최종 폴백 응답
                return self._fallback_response(last_exc or "알 수 없는 오류")

        # for 루프 정상 종료 후 안전망 (이론상 도달 불가)
        return self._fallback_response("모든 모델 실패")

    def _fallback_response(self, error_msg: str) -> dict:
        """모든 모델 실패 시 안전한 기본 응답"""
        return {
            "action": "CHAT",
            "update_params": {},
            "threats_to_add": [],
            "clear_existing_threats": False,
            "mission_sequence": [],
            "response_text": "⚠️ AI 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "reasoning": f"시스템 오류: {error_msg}",
            "confidence": 0.0,
            "_model_used": "fallback",
            "_cache_hit": False,
            "_doctrine_refs": [],
        }
