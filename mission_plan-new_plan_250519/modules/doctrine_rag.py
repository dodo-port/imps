"""
Lightweight local RAG for doctrine files.

- Source: data/doctrine/*.(pdf|md|txt)
- Retrieval: BM25 lexical ranking (no external vector DB dependency)
- Output: top-k snippets with source/page citations for prompt augmentation
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from pypdf import PdfReader
    HAS_PDF_READER = True
except Exception:
    HAS_PDF_READER = False


TOKEN_RE = re.compile(r"[A-Za-z0-9_\u3131-\u318E\uAC00-\uD7A3]+")

# ── 군사 용어 동의어 확장 테이블 ──────────────────────────────────────
# 쿼리에 왼쪽 단어가 있으면 오른쪽 단어들도 함께 검색
_MILITARY_SYNONYMS: Dict[str, List[str]] = {
    # 임무 유형
    "sead":     ["방공망", "제압", "sam", "radar", "레이더", "무력화", "suppression"],
    "strike":   ["타격", "공격", "폭격", "공습", "interdiction"],
    "cas":      ["근접항공지원", "근접지원", "close air support"],
    "isr":      ["정찰", "감시", "탐색", "탐지", "reconnaissance", "surveillance"],
    "방공망":    ["sead", "sam", "radar", "레이더", "제압"],
    "타격":      ["strike", "공격", "폭격"],
    "정찰":      ["isr", "감시", "탐색"],
    # 경로/전술
    "저고도":    ["terrain", "지형추종", "nap of earth", "회피"],
    "우회":      ["회피", "bypass", "안전", "detour"],
    "rtb":      ["복귀", "return to base", "귀환"],
    # 위협
    "sam":      ["지대공", "미사일", "방공", "surface to air"],
    "radar":    ["레이더", "탐지", "조기경보", "화력통제"],
    "nfz":      ["비행금지구역", "no fly zone", "금지"],
    # 편대
    "mum-t":    ["맘티", "유무인", "복합", "teaming", "uas", "무인기"],
    "편대":      ["formation", "package", "패키지"],
    # 연료
    "급유":      ["ar", "aerial refueling", "tanker", "급유기"],
    "연료":      ["fuel", "fuel state", "bingo"],
    # 교리
    "jp3":      ["joint", "합동", "doctrine"],
    "afdp":     ["air force", "교리", "doctrine"],
}


@dataclass
class DoctrineChunk:
    source: str
    page: int
    text: str
    tokens: List[str]
    term_freq: Dict[str, int]
    length: int


class DoctrineRAG:
    def __init__(
        self,
        doctrine_dir: str = "data/doctrine",
        fallback_doc: str = "doctrine_basis.md",
        max_pdf_pages: Optional[int] = None,
        chunk_chars: int = 900,
        chunk_overlap: int = 180,
    ) -> None:
        self.doctrine_dir = Path(doctrine_dir)
        self.fallback_doc = Path(fallback_doc)
        self.max_pdf_pages = max_pdf_pages
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap

        self._chunks: List[DoctrineChunk] = []
        self._idf: Dict[str, float] = {}
        self._avg_len: float = 1.0
        self._built = False

    def _tokenize(self, text: str) -> List[str]:
        return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]

    def _expand_query(self, query: str) -> str:
        """군사 용어 동의어 확장: 쿼리에 있는 키워드의 동의어를 추가해 검색 범위 확장"""
        tokens = set(self._tokenize(query))
        extras: List[str] = []
        for key, synonyms in _MILITARY_SYNONYMS.items():
            if key in tokens:
                extras.extend(synonyms)
        if extras:
            return query + " " + " ".join(extras)
        return query

    def _chunk_text(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        chunks = []
        i = 0
        n = len(text)
        step = max(1, self.chunk_chars - self.chunk_overlap)
        while i < n:
            j = min(n, i + self.chunk_chars)
            # Try to cut at sentence-ish boundary for readability.
            if j < n:
                cut = max(text.rfind(". ", i, j), text.rfind("? ", i, j), text.rfind(" ", i, j))
                if cut > i + int(self.chunk_chars * 0.65):
                    j = cut + 1
            chunk = text[i:j].strip()
            if len(chunk) >= 80:
                chunks.append(chunk)
            i += step
        return chunks

    def _read_text_file(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def _read_pdf_pages(self, path: Path) -> List[Tuple[int, str]]:
        if not HAS_PDF_READER:
            return []
        try:
            reader = PdfReader(str(path))
            out: List[Tuple[int, str]] = []
            pages = reader.pages if self.max_pdf_pages is None else reader.pages[: self.max_pdf_pages]
            for idx, page in enumerate(pages, start=1):
                txt = (page.extract_text() or "").strip()
                if txt:
                    out.append((idx, txt))
            return out
        except Exception:
            return []

    def _iter_documents(self) -> List[Tuple[str, int, str]]:
        docs: List[Tuple[str, int, str]] = []

        if self.doctrine_dir.exists():
            for ext in ("*.md", "*.txt", "*.pdf"):
                for path in sorted(self.doctrine_dir.glob(ext)):
                    if path.suffix.lower() == ".pdf":
                        for page, text in self._read_pdf_pages(path):
                            docs.append((path.name, page, text))
                    else:
                        text = self._read_text_file(path)
                        if text:
                            docs.append((path.name, 1, text))

        if self.fallback_doc.exists():
            text = self._read_text_file(self.fallback_doc)
            if text:
                docs.append((self.fallback_doc.name, 1, text))

        return docs

    def build_index(self, force: bool = False) -> None:
        if self._built and not force:
            return

        chunks: List[DoctrineChunk] = []
        for source, page, text in self._iter_documents():
            for c in self._chunk_text(text):
                tokens = self._tokenize(c)
                if not tokens:
                    continue
                tf: Dict[str, int] = {}
                for t in tokens:
                    tf[t] = tf.get(t, 0) + 1
                chunks.append(
                    DoctrineChunk(
                        source=source,
                        page=page,
                        text=c,
                        tokens=tokens,
                        term_freq=tf,
                        length=len(tokens),
                    )
                )

        self._chunks = chunks
        self._idf = {}
        self._avg_len = 1.0

        if not chunks:
            self._built = True
            return

        n_docs = len(chunks)
        self._avg_len = sum(c.length for c in chunks) / n_docs

        df: Dict[str, int] = {}
        for c in chunks:
            for t in set(c.tokens):
                df[t] = df.get(t, 0) + 1

        for term, freq in df.items():
            self._idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

        self._built = True

    def _score_bm25(self, query_tokens: List[str], chunk: DoctrineChunk, k1: float = 1.2, b: float = 0.75) -> float:
        if not query_tokens:
            return 0.0
        score = 0.0
        dl = max(1, chunk.length)
        norm = k1 * (1.0 - b + b * dl / max(1e-9, self._avg_len))
        for t in set(query_tokens):
            f = chunk.term_freq.get(t, 0)
            if f <= 0:
                continue
            idf = self._idf.get(t, 0.0)
            score += idf * (f * (k1 + 1.0)) / (f + norm)
        return score

    def search(self, query: str, top_k: int = 5, min_score: float = 0.03) -> List[Dict[str, object]]:
        self.build_index()
        if not self._chunks:
            return []

        # 원본 쿼리 + 동의어 확장 쿼리를 합산해 스코어링
        expanded = self._expand_query(query)
        q_tokens_orig = self._tokenize(query)
        q_tokens_exp  = self._tokenize(expanded)

        ranked: List[Tuple[float, DoctrineChunk]] = []
        for c in self._chunks:
            s_orig = self._score_bm25(q_tokens_orig, c)
            s_exp  = self._score_bm25(q_tokens_exp,  c)
            # 원본 쿼리 가중치 0.6, 확장 쿼리 가중치 0.4
            s = 0.6 * s_orig + 0.4 * s_exp
            if s >= min_score:
                ranked.append((s, c))

        ranked.sort(key=lambda x: x[0], reverse=True)

        # 중복 소스 억제: 같은 파일·페이지에서 최고 점수 1개만
        seen: set = set()
        deduped: List[Tuple[float, DoctrineChunk]] = []
        for score, chunk in ranked:
            key = (chunk.source, chunk.page)
            if key not in seen:
                deduped.append((score, chunk))
                seen.add(key)
            if len(deduped) >= top_k * 2:
                break

        out: List[Dict[str, object]] = []
        for score, chunk in deduped[:top_k]:
            out.append(
                {
                    "source": chunk.source,
                    "page": chunk.page,
                    "score": round(float(score), 4),
                    "text": chunk.text,
                }
            )
        return out

    def format_context(self, query: str, top_k: int = 5, max_chars: int = 6000) -> str:
        hits = self.search(query=query, top_k=top_k)
        if not hits:
            return "No retrieved doctrine chunks."

        parts: List[str] = []
        remain = max_chars
        for idx, h in enumerate(hits, 1):
            source = h['source']
            page   = h['page']
            score  = h['score']
            body   = str(h["text"])

            # LLM이 읽기 쉬운 구조화된 헤더
            head = f"[Doctrine Ref {idx} | {source} p.{page} | relevance={score:.3f}]\n"
            piece = head + body

            if len(piece) > remain:
                piece = piece[: max(0, remain - 3)].rstrip() + "..."
            if piece.strip():
                parts.append(piece)
                remain -= len(piece) + 6   # 구분자 "\n\n---\n" = 6 chars
            if remain <= 150:
                break

        if not parts:
            return "No relevant doctrine found."
        return "\n\n---\n".join(parts)
