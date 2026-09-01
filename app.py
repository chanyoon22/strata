# app.py — STRATA · 인력구조 시나리오 플래너
# 실행: streamlit run app.py   (같은 폴더의 .streamlit/config.toml 필요)
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from engine_retirement_v2 import CompanyProfile, monte_carlo
from dss_layer import preset_scenarios, pareto_frontier

st.set_page_config(page_title="STRATA — 인력구조 시나리오 플래너",
                   layout="wide", initial_sidebar_state="collapsed")

INK, SUB, LINE = "#17242F", "#5D6B77", "#E6EAED"
TEAL, TEAL_BG, PAPER = "#146B5C", "#E9F3F0", "#F6F8F9"
SERIES = ["#17242F", "#146B5C", "#B4690E", "#5B4B8A", "#A63D57", "#3B7DD8"]

st.markdown(f"""<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
html, body, .stApp, [class*="st-"] {{ font-family:'Pretendard',-apple-system,sans-serif; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility:hidden; height:0; }}
.block-container {{ padding-top:0.6rem; max-width:1240px; }}

.topbar {{ display:flex; align-items:baseline; justify-content:space-between;
  padding:18px 4px 14px; border-bottom:1px solid {LINE}; margin-bottom:26px; }}
.wordmark {{ font-weight:800; font-size:1.05rem; letter-spacing:.22em; color:{INK}; }}
.wordmark span {{ color:{TEAL}; }}
.topbar .tag {{ color:{SUB}; font-size:.85rem; letter-spacing:.02em; }}

.h-title {{ font-size:1.62rem; font-weight:700; color:{INK}; letter-spacing:-.02em;
  margin:0 0 4px; }}
.h-sub {{ color:{SUB}; font-size:.95rem; margin-bottom:22px; }}

div[data-testid="stForm"] {{ background:{PAPER}; border:1px solid {LINE};
  border-radius:14px; padding:22px 24px 8px; }}
div[data-testid="stForm"] label p {{ font-size:.82rem; color:{SUB}; font-weight:600; }}
.stNumberInput input {{ font-variant-numeric:tabular-nums; }}
div[data-testid="stFormSubmitButton"] button {{
  background:{INK}; color:#fff; border:none; border-radius:10px;
  font-weight:700; padding:.62rem 1.5rem; letter-spacing:.02em; }}
div[data-testid="stFormSubmitButton"] button:hover {{ background:{TEAL}; color:#fff; }}

div[data-testid="stMetric"] {{ background:#fff; border:1px solid {LINE};
  border-radius:14px; padding:18px 20px 14px; }}
div[data-testid="stMetric"] label p {{ color:{SUB}; font-size:.8rem; font-weight:600; }}
div[data-testid="stMetricValue"] {{ color:{INK}; font-weight:700;
  font-variant-numeric:tabular-nums; }}

button[data-baseweb="tab"] {{ font-weight:600; color:{SUB}; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:{INK}; }}
div[data-baseweb="tab-highlight"] {{ background:{TEAL}; }}

.chip-in {{ display:inline-block; padding:3px 12px; border-radius:999px;
  background:{TEAL_BG}; color:{TEAL}; font-weight:700; font-size:.78rem; }}
.chip-out {{ display:inline-block; padding:3px 12px; border-radius:999px;
  background:#EEF1F3; color:{SUB}; font-weight:600; font-size:.78rem; }}
.footnote {{ color:#98A4AE; font-size:.78rem; border-top:1px solid {LINE};
  padding-top:14px; margin-top:38px; line-height:1.7; }}
.method h4 {{ color:{INK}; font-size:1.0rem; margin:18px 0 6px; }}
.method p {{ color:{SUB}; font-size:.9rem; line-height:1.75; margin:0 0 4px; }}
</style>""", unsafe_allow_html=True)

# ── 상단 바 ──
st.markdown("""<div class="topbar">
<div class="wordmark">STRATA<span>.</span></div>
<div class="tag">인력구조 시나리오 플래너</div></div>""", unsafe_allow_html=True)

st.markdown('<p class="h-title">정년연장이 우리 회사 인력구조에 미치는 영향</p>', unsafe_allow_html=True)
st.markdown('<p class="h-sub">기업 조건을 입력하면 국회에서 경합 중인 6개 정책 시나리오의 '
            '10년 결과를 비교하고, 열위 시나리오를 자동으로 걸러냅니다.</p>', unsafe_allow_html=True)

# ── 설정 (메인 영역, 가로 폼) ──
with st.form("cfg"):
    r1 = st.columns([1, 1, 1, 1])
    n_emp = r1[0].number_input("직원 수 (명)", 100, 20000, 1000, 100)
    avg_wage = r1[1].number_input("월평균임금 (만원)", 200, 1500, 465, 5)
    push = r1[2].number_input("자발이직률 (%/년)", 0.5, 6.0, 1.5, 0.1) / 100
    other = r1[3].number_input("기타 이직률 (%/년)", 1.0, 10.0, 4.5, 0.5) / 100
    r2 = st.columns([1, 1, 1, 1])
    accept = r2[0].number_input("재고용 수락률", 0.30, 0.95, 0.62, 0.02)
    wage_ratio = r2[1].number_input("재고용 임금률", 0.35, 0.95, 0.55, 0.02)
    horizon = r2[2].selectbox("분석 기간", [10, 12, 15], 1, format_func=lambda x: f"{x}년")
    runs = r2[3].selectbox("시뮬레이션 정밀도", [50, 100, 200], 1,
                           format_func=lambda x: {50: "빠름", 100: "표준", 200: "정밀"}[x])
    go_btn = st.form_submit_button("시나리오 분석")

@st.cache_data(show_spinner="6개 시나리오를 시뮬레이션하고 있습니다")
def run_analysis(n_emp, avg_wage, push, other, accept, wage_ratio, horizon, runs):
    prof = CompanyProfile(n_employees=n_emp, avg_wage=avg_wage,
                          base_push_turnover=push, other_exit_rate=other)
    results = [monte_carlo(prof, s, runs=runs, horizon=horizon)
               for s in preset_scenarios(wage_ratio, accept)]
    fr, dom = pareto_frontier([r["summary"] for r in results])
    return results, [f["scenario"] for f in fr], [d["scenario"] for d in dom]

if go_btn:
    st.session_state["res"] = run_analysis(n_emp, avg_wage, push, other,
                                           accept, wage_ratio, horizon, runs)

if "res" in st.session_state:
    results, frontier, dominated = st.session_state["res"]
    npvs = [r["summary"]["cost_npv_mean"] for r in results]
    hires = [r["summary"]["total_hires_mean"] for r in results]
    e55 = [r["summary"]["exp55_person_years_mean"] for r in results]

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    k = st.columns(4)
    k[0].metric("남은 후보", f"{len(frontier)} / 6",
                f"{len(dominated)}개 열위 탈락" if dominated else "전원 유지",
                delta_color="off")
    k[1].metric("인건비 격차 (최대–최소)", f"{(max(npvs)-min(npvs))/1e4:,.0f}억 원",
                f"{(max(npvs)/min(npvs)-1)*100:+.1f}%", delta_color="off")
    k[2].metric("신규채용 여력 격차", f"{max(hires)-min(hires):,.0f}명", delta_color="off")
    k[3].metric("숙련 유지 격차 (55세+ 인년)", f"{max(e55)-min(e55):,.0f}", delta_color="off")

    tab1, tab2, tab3 = st.tabs(["시나리오 비교", "연도별 추이", "트레이드오프 지도"])

    with tab1:
        rows = []
        for r in results:
            s = r["summary"]
            rows.append({"시나리오": s["scenario"],
                         "판정": "후보" if s["scenario"] in frontier else "탈락",
                         "10년 인건비 NPV(억)": s["cost_npv_mean"] / 1e4,
                         "90% 신뢰구간(억)": f"{s['cost_npv_p5']/1e4:,.0f} – {s['cost_npv_p95']/1e4:,.0f}",
                         "신규채용 누계(명)": s["total_hires_mean"],
                         "승진 대기 p90(년)": s["promo_congestion_mean"],
                         "55세+ 재직 인년": s["exp55_person_years_mean"]})
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.format({"10년 인건비 NPV(억)": "{:,.0f}", "신규채용 누계(명)": "{:,.0f}",
                             "승진 대기 p90(년)": "{:.1f}", "55세+ 재직 인년": "{:,.0f}"})
              .map(lambda v: f"color:{TEAL};font-weight:700" if v == "후보"
                   else f"color:{SUB}", subset=["판정"]),
            use_container_width=True, hide_index=True)
        st.markdown('<span class="chip-in">후보</span>&nbsp; 네 가지 지표(비용·채용·승진·숙련) 어디서도 '
                    '다른 시나리오에 밀리지 않음 &nbsp;&nbsp;'
                    '<span class="chip-out">탈락</span>&nbsp; 같은 조건에서 더 나은 대안 존재',
                    unsafe_allow_html=True)

    with tab2:
        metric = st.radio("m", ["연간 인건비", "55세 이상 비중", "연간 신규채용"],
                          horizontal=True, label_visibility="collapsed")
        key, scale, unit = {"연간 인건비": ("labor_cost", 1e-4, "억 원"),
                            "55세 이상 비중": ("share_55plus", 100, "%"),
                            "연간 신규채용": ("n_hire", 1, "명")}[metric]
        fig = go.Figure()
        for r, c in zip(results, SERIES):
            p = r["path"]
            nm = r["summary"]["scenario"]
            fig.add_trace(go.Scatter(
                x=[x["year"] for x in p], y=[x[key] * scale for x in p],
                name=nm.split("(")[0].strip(),
                line=dict(color=c, width=2.4 if nm in frontier else 1.4,
                          dash=None if nm in frontier else "dot")))
        fig.update_layout(height=430, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Pretendard", color=INK),
                          margin=dict(t=24, b=10, l=10, r=10),
                          yaxis_title=unit,
                          legend=dict(orientation="h", y=-0.18, font=dict(size=12)))
        fig.update_xaxes(gridcolor=LINE, dtick=2)
        fig.update_yaxes(gridcolor=LINE)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        fig = go.Figure()
        for r, c in zip(results, SERIES):
            s = r["summary"]
            onf = s["scenario"] in frontier
            fig.add_trace(go.Scatter(
                x=[s["cost_npv_mean"] / 1e4], y=[s["exp55_person_years_mean"]],
                mode="markers+text", text=[s["scenario"].split(" ")[0]],
                textposition="top center", textfont=dict(size=12, color=INK if onf else SUB),
                marker=dict(size=17 if onf else 11, color=c if onf else "#C9D1D7",
                            line=dict(color=INK if onf else "#C9D1D7", width=1.5)),
                showlegend=False))
        fig.update_layout(height=460, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Pretendard", color=INK),
                          xaxis_title="10년 인건비 NPV (억 원) — 왼쪽일수록 유리",
                          yaxis_title="55세+ 재직 인년 — 위일수록 유리",
                          margin=dict(t=30, b=10))
        fig.update_xaxes(gridcolor=LINE)
        fig.update_yaxes(gridcolor=LINE)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('<span style="color:#5D6B77;font-size:.85rem">채워진 큰 점이 후보 시나리오입니다. '
                    '비용과 숙련유지 두 축의 트레이드오프에서 어느 방향을 우선할지는 경영 판단의 영역입니다.</span>',
                    unsafe_allow_html=True)

    with st.expander("이 분석이 계산되는 방식"):
        st.markdown(f"""<div class="method">
<h4>인력 구성</h4><p>한국노동패널(KLIPS) 300인 이상 기업 재직자 1,789명의 실제 나이·근속·임금·근로시간
분포를 통째로 표본추출해 가상의 회사를 만듭니다. 입력한 직원 수와 평균임금에 맞춰 조정됩니다.
55세 이상 비중은 사업체패널(WPS2023) 대기업 500개 사업체의 보고치와 0.2%p 이내로 일치합니다.</p>
<h4>이동 확률</h4><p>자발 이직은 나이·근속·임금·근로시간의 실측 통계모형(KLIPS 로지스틱, 사건 1,858건,
최신 차수 재검증 완료)을 따르고, 정년퇴직 후 재고용의 수락률·임금률 기본값은 정년퇴직자 추적
데이터(재취업률 62%, 임금비 중위 0.51)에서 왔습니다. 임금피크제를 켜면 감액된 임금이
이직 확률에 자동 반영됩니다. 참고로 300인 이상 기업의 임금피크제 도입률은 49.6%입니다(WPS2023).</p>
<h4>제도 효과의 검증</h4><p>2016년 60세 정년 의무화 당시 55–59세 비자발 퇴출률이 대기업에서
상대적으로 4.0%p 하락(p=0.005)한 것을 준실험 설계로 확인했습니다. 이 도구의 정년 메커니즘이
실제 제도 변화와 같은 방향으로 작동함을 뜻합니다.</p>
<h4>한계</h4><p>재고용 수락률은 사내 제안 수락이 통계로 직접 관측되지 않아 재취업률 기반
추정 구간(0.37–0.62)을 씁니다. 자사 실측치가 있다면 위 설정에서 교체하십시오.
교육투자 등 효과가 검증되지 않은 정책은 비용만 계산하고 효과를 가정하지 않습니다.</p>
</div>""", unsafe_allow_html=True)
else:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    c = st.columns(3)
    boxes = [("6개 정책 시나리오", "현행 유지부터 2032년 65세 급진 연장까지, 국회·경사노위에서 실제 경합 중인 안을 그대로 반영합니다."),
             ("자동 후보 선별", "비용·채용·승진·숙련 네 지표에서 열위인 시나리오를 알고리즘이 걸러내고, 남은 후보의 트레이드오프만 보여줍니다."),
             ("불확실성 표시", "모든 수치는 몬테카를로 반복의 90% 신뢰구간과 함께 제공됩니다. 단정 대신 범위로 판단하십시오.")]
    for col, (t, b) in zip(c, boxes):
        col.markdown(f"""<div style="background:{PAPER};border:1px solid {LINE};
        border-radius:14px;padding:20px 22px;min-height:150px">
        <div style="font-weight:700;color:{INK};margin-bottom:8px">{t}</div>
        <div style="color:{SUB};font-size:.9rem;line-height:1.7">{b}</div></div>""",
                     unsafe_allow_html=True)

st.markdown('<div class="footnote">데이터 기반: 한국노동패널조사(KLIPS) 26·27차 · 사업체패널조사(WPS) 2023 · '
            '고용형태별근로실태조사 2025(KOSIS). 계산 방법과 한계는 결과 하단의 '
            '"이 분석이 계산되는 방식"에서 확인할 수 있습니다. STRATA는 의사결정 참고용 도구이며 '
            '법률·재무 자문을 대체하지 않습니다.</div>', unsafe_allow_html=True)
