"""
IMPS v13.0 진입점 — 네비게이션 컨트롤러
실행: streamlit run run.py
"""
import streamlit as st

st.set_page_config(
    page_title="IMPS v13.0 (MUM-T)",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 사이드바 네비게이션 텍스트 크기 CSS ──
st.markdown("""
<style>
/* 사이드바 전체 배경 */
[data-testid="stSidebar"] {
    background-color: #161B22 !important;
    border-right: 1px solid #30363D;
}
/* st.navigation 항목 텍스트 */
[data-testid="stSidebarNavItems"] a,
[data-testid="stSidebarNavItems"] span,
[data-testid="stSidebarNavItems"] p {
    font-size: 15px !important;
    font-weight: 700 !important;
    color: #C9D1D9 !important;
    letter-spacing: 0.3px;
}
[data-testid="stSidebarNavItems"] a:hover span,
[data-testid="stSidebarNavItems"] a:hover p {
    color: #58A6FF !important;
}
/* 선택된 페이지 강조 */
[data-testid="stSidebarNavItems"] [aria-selected="true"] span,
[data-testid="stSidebarNavItems"] [aria-selected="true"] p {
    color: #58A6FF !important;
}
/* 사이드바 로고/앱 이름 */
[data-testid="stSidebarNavSeparator"] { border-color: #30363D !important; }
/* 네비게이션 아이템 패딩 */
[data-testid="stSidebarNavItems"] li {
    padding: 4px 0 !important;
}
/* 앱 이름 영역 */
[data-testid="stSidebarUserContent"] { padding-top: 0.5rem; }
/* 스크롤바 */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #58A6FF; }
</style>
""", unsafe_allow_html=True)

pg = st.navigation(
    [
        st.Page("streamlit_app.py",            title="IMPS 임무계획",   icon="🚁", default=True),
        st.Page("pages/2_작전_타임라인.py",     title="작전 타임라인",   icon="⏱️"),
        st.Page("pages/3_시나리오_비교.py",     title="시나리오 비교",   icon="📊"),
    ],
    position="sidebar",
)
pg.run()
