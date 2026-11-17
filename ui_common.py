"""ui_common.py — 화면 공통: 색·스타일·작은 조각들·차트 기본 설정."""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go

INK, SUB, LINE = "#17242F", "#5D6B77", "#E6EAED"
TEAL, TEAL_BG, PAPER = "#146B5C", "#E9F3F0", "#F6F8F9"
MUTED = "#98A4AE"
WARN, WARN_BG = "#9A5B00", "#FFF4DE"
BAD, BAD_BG = "#A63D57", "#FBECEF"

# 시나리오 색. 기준선(현행 유지)은 중립 먹색 점선, 나머지는 색각 이상에서도 이웃끼리
# 구별되도록 검증된 8색을 '고정 순서'로 배정한다(순위가 아니라 시나리오를 따라간다).
BASELINE_COLOR = INK
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OVERFLOW_COLOR = "#9aa5ae"
SEQ = [[0.0, "#eef4fd"], [0.25, "#b7d3f6"], [0.5, "#6da7ec"], [0.75, "#256abf"], [1.0, "#0d366b"]]
LOW_C, HIGH_C = "#2a78d6", "#eb6834"
GROUP_C = {"young": "#9ec5f4", "mid": "#3987e5", "pre": "#eda100", "old": "#eb6834", "ce": "#6250d6"}

PLOTLY_CONFIG = {
    "displayModeBar": True, "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "hoverClosestCartesian",
                               "hoverCompareCartesian", "toggleSpikelines"],
}

CSS = f"""<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
html, body, .stApp, [class*="st-"] {{ font-family:'Pretendard',-apple-system,'Malgun Gothic',sans-serif; }}
/* 위 규칙이 아이콘 글꼴까지 덮어써서 펼침 화살표·업로드 아이콘이 글자로 깨져 보였다.
   아이콘 요소만 원래 글꼴로 되돌린다. */
span[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] {{
  font-family:'Material Symbols Rounded' !important; }}
/* 한글이 단어 중간에서 끊기지 않게 */
body, .stApp, [class*="st-"] {{ word-break:keep-all; overflow-wrap:break-word; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility:hidden; height:0; }}
.block-container {{ padding-top:0.6rem; max-width:1240px; }}

.topbar {{ display:flex; align-items:baseline; justify-content:space-between;
  padding:18px 4px 14px; border-bottom:1px solid {LINE}; margin-bottom:22px; }}
.wordmark {{ font-weight:800; font-size:1.05rem; letter-spacing:.22em; color:{INK}; }}
.wordmark span {{ color:{TEAL}; }}
.topbar .tag {{ color:{SUB}; font-size:.85rem; }}
.h-title {{ font-size:1.55rem; font-weight:700; color:{INK}; letter-spacing:-.02em; margin:0 0 4px; }}
.h-sub {{ color:{SUB}; font-size:.95rem; margin-bottom:16px; line-height:1.7; }}
.sec {{ font-weight:700; color:{INK}; font-size:1.08rem; margin:26px 0 2px; }}
.sec-sub {{ color:{SUB}; font-size:.87rem; line-height:1.75; margin-bottom:10px; }}
.note {{ color:{SUB}; font-size:.85rem; line-height:1.8; }}
.fine {{ color:{MUTED}; font-size:.78rem; line-height:1.7; }}

div[data-testid="stForm"] {{ background:{PAPER}; border:1px solid {LINE};
  border-radius:14px; padding:20px 22px 8px; }}
label p {{ font-size:.82rem !important; color:{SUB}; font-weight:600; }}
.stNumberInput input {{ font-variant-numeric:tabular-nums; }}
button[kind="primary"] {{ background:{INK}; border:none; border-radius:10px; font-weight:700; }}
button[kind="primary"]:hover {{ background:{TEAL}; }}

div[data-testid="stMetric"] {{ background:#fff; border:1px solid {LINE};
  border-radius:14px; padding:16px 18px 12px; }}
div[data-testid="stMetric"] label p {{ color:{SUB}; font-size:.8rem !important; font-weight:600; }}
div[data-testid="stMetricValue"] {{ color:{INK}; font-weight:700; font-variant-numeric:tabular-nums;
  font-size:1.65rem; }}

button[data-baseweb="tab"] {{ font-weight:600; color:{SUB}; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{INK}; }}
div[data-baseweb="tab-highlight"] {{ background:{TEAL}; }}

.kpi {{ background:#fff; border:1px solid {LINE}; border-radius:14px; padding:15px 18px 13px; height:100%; }}
.kpi-l {{ color:{SUB}; font-size:.8rem; font-weight:600; }}
.kpi-v {{ color:{INK}; font-weight:700; font-size:1.6rem; font-variant-numeric:tabular-nums; line-height:1.35; }}
.kpi-s {{ color:{MUTED}; font-size:.8rem; min-height:1.1em; }}
.chip {{ display:inline-block; padding:3px 12px; border-radius:999px; font-weight:700; font-size:.78rem; }}
.chip-in {{ background:{TEAL_BG}; color:{TEAL}; }}
.chip-out {{ background:#EEF1F3; color:{SUB}; font-weight:600; }}
.chip-warn {{ background:{WARN_BG}; color:{WARN}; }}
.chip-bad {{ background:{BAD_BG}; color:{BAD}; }}
.card {{ background:{PAPER}; border:1px solid {LINE}; border-radius:14px; padding:16px 18px;
  height:100%; }}
.card-t {{ font-weight:700; color:{INK}; margin-bottom:6px; font-size:.95rem; }}
.card-b {{ color:{SUB}; font-size:.87rem; line-height:1.75; }}
.card-b b {{ color:{INK}; }}
.issue {{ border-left:3px solid {LINE}; padding:6px 0 6px 12px; margin-bottom:8px; }}
.issue.warn {{ border-color:#E0A93B; }} .issue.error {{ border-color:{BAD}; }}
.issue.info {{ border-color:#9ec5f4; }}
.issue-t {{ font-weight:700; color:{INK}; font-size:.88rem; }}
.issue-d {{ color:{SUB}; font-size:.84rem; line-height:1.65; }}
.footnote {{ color:{MUTED}; font-size:.78rem; border-top:1px solid {LINE};
  padding-top:14px; margin-top:38px; line-height:1.7; }}
</style>"""


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def gap(px=12):
    html(f"<div style='height:{px}px'></div>")


def section(title, sub=None):
    html(f'<div class="sec">{title}</div>' + (f'<div class="sec-sub">{sub}</div>' if sub else ""))


def note(text, cls="note"):
    html(f'<div class="{cls}">{text}</div>')


def card(col, title, body):
    col.markdown(f'<div class="card"><div class="card-t">{title}</div>'
                 f'<div class="card-b">{body}</div></div>', unsafe_allow_html=True)


def chip(text, kind="in"):
    return f'<span class="chip chip-{kind}">{text}</span>'


def short(name):
    """'S2 여당 초안(2037년 65세)' → 'S2'"""
    return name.split(" ")[0]


def label(name):
    """'S2 여당 초안(2037년 65세)' → 'S2 여당 초안'"""
    return name.split("(")[0].strip()


def color_map(names, baseline):
    """시나리오 → 색. 기준선은 먹색, 나머지는 실행 순서대로 고정 배정(8개 초과분은 회색)."""
    out, i = {}, 0
    for n in names:
        if n == baseline:
            out[n] = BASELINE_COLOR
        else:
            out[n] = SERIES[i] if i < len(SERIES) else OVERFLOW_COLOR
            i += 1
    return out


def base_layout(fig, height=420, unit=None, **kw):
    """unit을 주면 세로축 제목 대신 그래프 왼쪽 위에 가로로 단위를 적는다
    (세로로 눕힌 축 제목은 읽기 어렵다는 피드백 반영)."""
    m = dict(kw.pop("margin", dict(t=24, b=10, l=10, r=10)))
    if unit:
        m["t"] = max(m.get("t", 24), 34)
        fig.add_annotation(text=unit, xref="paper", yref="paper", x=0, y=1.0, yshift=22, xanchor="left",
                           showarrow=False, font=dict(size=12, color=SUB))
    kw["margin"] = m
    fig.update_layout(height=height, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Pretendard, Malgun Gothic, sans-serif", color=INK, size=13),
                      hoverlabel=dict(bgcolor="white", font_size=12.5), **kw)
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor="#c9d1d7", linecolor=LINE)
    return fig


def show(fig, key=None, bar=True):
    """bar=False면 확대 도구 막대를 숨긴다(확대할 일이 없는 작은 차트에서 축 글자를 가리지 않도록)."""
    cfg = PLOTLY_CONFIG if bar else {"displayModeBar": False}
    st.plotly_chart(fig, width="stretch", config=cfg, key=key)


def kpi(col, label_, value, sub=""):
    """요약 숫자 타일. st.metric은 보조 문구 앞에 화살표를 붙여 '증가'로 오해하게 만들어 직접 그린다."""
    col.markdown(f'<div class="kpi"><div class="kpi-l">{label_}</div><div class="kpi-v">{value}</div>'
                 f'<div class="kpi-s">{sub}</div></div>', unsafe_allow_html=True)


def pct(x, digits=1, sign=True):
    return f"{x*100:+.{digits}f}%" if sign else f"{x*100:.{digits}f}%"
