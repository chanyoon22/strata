# app.py — STRATA · 인력구조 시나리오 플래너
# 실행: streamlit run app.py   (같은 폴더의 .streamlit/config.toml 필요)
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from engine_retirement_v2 import CompanyProfile, monte_carlo, WAGE_REGIMES
from dss_layer import (preset_scenarios, pareto_frontier, HIRING_POLICIES,
                       sensitivity as run_sensitivity)
from llm_report import build_llm_input, generate_report

st.set_page_config(page_title="STRATA — 인력구조 시나리오 플래너",
                   layout="wide", initial_sidebar_state="collapsed")

INK, SUB, LINE = "#17242F", "#5D6B77", "#E6EAED"
TEAL, TEAL_BG, PAPER = "#146B5C", "#E9F3F0", "#F6F8F9"
SERIES = ["#17242F", "#146B5C", "#B4690E", "#5B4B8A", "#A63D57", "#3B7DD8", "#2E7D5B"]

# 모든 그래프에 공통으로 쓰는 설정. 확대해본 뒤 원래대로 되돌리는 버튼을 쓸 수 있도록
# 툴바를 켜두되(예전엔 displayModeBar:False로 아예 숨겨서 되돌리기가 불가능했음),
# 불필요한 버튼(선택 올가미 등)은 지워 화면을 덜 복잡하게 만든다.
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "hoverClosestCartesian",
                               "hoverCompareCartesian", "toggleSpikelines"],
}

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
</style>""", unsafe_allow_html=True)

# ── 상단 바 ──
st.markdown("""<div class="topbar">
<div class="wordmark">STRATA<span>.</span></div>
<div class="tag">인력구조 시나리오 플래너</div></div>""", unsafe_allow_html=True)

N_SCEN = len(preset_scenarios())

st.markdown('<p class="h-title">정년연장이 우리 회사 인력구조에 미치는 영향</p>', unsafe_allow_html=True)
st.markdown(f'<p class="h-sub">기업 조건을 입력하면 국회에서 경합 중인 {N_SCEN}개 정책 시나리오의 '
            '10~15년 결과를 비교하고, 더 나은 대안이 있는 시나리오는 자동으로 걸러냅니다.</p>', unsafe_allow_html=True)

# ── 설정 (메인 영역, 가로 폼) ──
with st.form("cfg"):
    r1 = st.columns([1, 1, 1, 1])
    n_emp = r1[0].number_input("직원 수 (명)", 100, 20000, 1000, 100,
                               help="분석 대상 회사의 전체 임직원 수입니다.")
    avg_wage = r1[1].number_input("월평균임금 (만원)", 200, 1500, 465, 5,
                                  help="전체 직원의 월평균 임금입니다. 인건비 계산의 기준이 됩니다.")
    push = r1[2].number_input("자발적 퇴사율 (%/년)", 0.5, 6.0, 1.5, 0.1,
                              help="정년연장과 상관없이, 직원이 스스로 회사를 그만두는 비율입니다.") / 100
    other = r1[3].number_input("기타 퇴사율 (%/년)", 1.0, 10.0, 4.5, 0.5,
                               help="정년퇴직·자발적 퇴사를 뺀 나머지 사유(권고사직 등)로 회사를 "
                                    "떠나는 비율입니다.") / 100
    r2 = st.columns([1, 1, 1, 1])
    accept = r2[0].number_input("재고용 수락률", 0.30, 0.95, 0.62, 0.02,
                                help="정년을 채운 직원에게 재고용(계약직 등으로 다시 채용)을 "
                                     "제안했을 때, 실제로 받아들이는 비율입니다.")
    wage_ratio = r2[1].number_input("재고용 임금률", 0.35, 0.95, 0.55, 0.02,
                                    help="재고용됐을 때 받는 임금이, 정년 전 임금의 몇 %인지입니다. "
                                         "예: 0.55 = 정년 전의 55% 수준.")
    horizon = r2[2].selectbox("분석 기간", [10, 12, 15], 1, format_func=lambda x: f"{x}년",
                              help="앞으로 몇 년치 결과를 계산할지 정합니다.")
    runs = r2[3].selectbox("시뮬레이션 정밀도", [50, 100, 200], 1,
                           format_func=lambda x: {50: "빠름", 100: "표준", 200: "정밀"}[x],
                           help="같은 조건을 몇 번 반복 계산해 평균 낼지입니다. 높을수록 정확하지만 "
                                "계산 시간이 더 걸립니다.")
    # 결과를 크게 가르는 두 가지 회사 정책. 예전에는 코드에 고정돼 있어 사용자가 바꿀 수 없었다.
    r3 = st.columns([1, 1, 2])
    hiring = r3[0].selectbox("정년연장 부담을 흡수하는 방식", list(HIRING_POLICIES),
                             format_func=lambda k: HIRING_POLICIES[k][0],
                             help="정년이 늘어 고령자가 남을 때 무엇을 조정할지 고릅니다. "
                                  "이 선택에 따라 권장 후보가 실제로 달라집니다.")
    regime = r3[1].selectbox("50세 이후 임금이 어떻게 변할지", list(WAGE_REGIMES),
                             format_func=lambda k: WAGE_REGIMES[k]["label"],
                             help="정년연장 비용을 좌우하는 핵심 가정입니다. 기본값(현재 추세 그대로)이 "
                                  "가장 현실적이고, 호봉제로 갈수록 비용이 크게 올라갑니다.")
    r3[2].markdown(f'<div style="color:{SUB};font-size:.8rem;padding-top:26px;line-height:1.6">'
                   f'{HIRING_POLICIES[hiring][1]}</div>', unsafe_allow_html=True)
    go_btn = st.form_submit_button("시나리오 분석")

@st.cache_data(show_spinner=f"{N_SCEN}개 시나리오를 시뮬레이션하고 있습니다")
def run_analysis(n_emp, avg_wage, push, other, accept, wage_ratio, horizon, runs,
                 hiring, regime):
    prof = CompanyProfile(n_employees=n_emp, avg_wage=avg_wage,
                          base_push_turnover=push, other_exit_rate=other,
                          wage_regime=regime)
    results = [monte_carlo(prof, s, runs=runs, horizon=horizon)
               for s in preset_scenarios(wage_ratio, accept, hiring=hiring)]
    fr, dom = pareto_frontier([r["summary"] for r in results])
    return results, [f["scenario"] for f in fr], [d["scenario"] for d in dom]

if go_btn:
    st.session_state["res"] = run_analysis(n_emp, avg_wage, push, other, accept,
                                           wage_ratio, horizon, runs, hiring, regime)
    st.session_state.pop("sens", None)   # 설정이 바뀌면 이전 민감도·해석 결과는 무효
    st.session_state.pop("report", None)

if "res" in st.session_state:
    results, frontier, dominated = st.session_state["res"]
    npvs = [r["summary"]["cost_npv_mean"] for r in results]
    hires = [r["summary"]["total_hires_mean"] for r in results]
    e55 = [r["summary"]["exp55_person_years_mean"] for r in results]
    H = results[0]["summary"]["horizon"]  # 선택한 분석 기간. "10년" 하드코딩 금지(2026-08-28 수정):
                                          # 12·15년을 선택해도 라벨이 계속 "10년"으로 남는 버그가 있었음
    npv_col = f"{H}년 인건비 NPV(억)"

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    # KPI 순서는 '시나리오 간 실제 변별력'이 큰 순. (2026-08-28 재정렬)
    # 실측(runs=200·horizon=12): 55세+ 인년 42.1% / 승진대기 28.5% / 신규채용 15.0% /
    # 인건비 1.5%. 인건비가 네 지표 중 가장 안 갈리는데도 예전 배치는 그것을 억원 단위
    # 큰 숫자로 맨 앞에 놓아 "정년연장은 곧 인건비 문제"라는 잘못된 인상을 먼저 주고 있었다.
    kpis = sorted([
        ("신규채용 규모 차이", f"{max(hires)-min(hires):,.0f}명",
         f"{(max(hires)/max(min(hires),1)-1)*100:+.1f}%", (max(hires)/max(min(hires),1)-1)),
        ("55세 이상 근속 차이", f"{max(e55)-min(e55):,.0f}",
         f"{(max(e55)/max(min(e55),1)-1)*100:+.1f}%", (max(e55)/max(min(e55),1)-1)),
        ("인건비 차이", f"{(max(npvs)-min(npvs))/1e4:,.0f}억 원",
         f"{(max(npvs)/min(npvs)-1)*100:+.1f}%", (max(npvs)/min(npvs)-1)),
    ], key=lambda x: -x[3])

    k = st.columns(4)
    k[0].metric("남은 후보", f"{len(frontier)} / {len(results)}",
                f"{len(dominated)}개 열위 탈락" if dominated else "전원 유지",
                delta_color="off")
    for col, (label, val, delta, _) in zip(k[1:], kpis):
        col.metric(label, val, delta, delta_color="off")

    top = kpis[0][0].replace(" 차이", "")
    st.markdown(f'<div style="color:{SUB};font-size:.86rem;margin:10px 2px 0">'
                f'시나리오마다 가장 크게 벌어지는 것은 <b style="color:{INK}">{top}</b>입니다. '
                f'인건비 차이는 {kpis[-1][2].lstrip("+")}로 크지 않습니다 — 정년연장의 부담은 '
                f'인건비보다 채용 규모와 인력 나이 구성에서 먼저 드러납니다.</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["시나리오 비교", "연도별 추이", "비용 vs 인력 비교", "조건이 달라지면?"])

    with tab1:
        wait_col = "승진 대기 기간(년)"
        e55_col = "55세+ 근속 총량"
        rows = []
        for r in results:
            s = r["summary"]
            rows.append({"시나리오": s["scenario"],
                         "판정": "후보" if s["scenario"] in frontier else "탈락",
                         npv_col: s["cost_npv_mean"] / 1e4,
                         "90% 변동구간(억)": f"{s['cost_npv_p5']/1e4:,.0f} – {s['cost_npv_p95']/1e4:,.0f}",
                         "신규채용 누계(명)": s["total_hires_mean"],
                         wait_col: s["promo_congestion_mean"],
                         e55_col: s["exp55_person_years_mean"]})
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.format({npv_col: "{:,.0f}", "신규채용 누계(명)": "{:,.0f}",
                             wait_col: "{:.1f}", e55_col: "{:,.0f}"})
              .map(lambda v: f"color:{TEAL};font-weight:700" if v == "후보"
                   else f"color:{SUB}", subset=["판정"]),
            width="stretch", hide_index=True,
            column_config={
                npv_col: st.column_config.NumberColumn(
                    help="이 시나리오를 택했을 때 앞으로 들어갈 인건비를, 오늘 돈 가치로 환산해 더한 금액입니다."),
                wait_col: st.column_config.NumberColumn(
                    help="직원 10명 중 9명이 승진하기까지 걸리는 기간입니다. 길수록 승진이 밀린다는 뜻입니다."),
                e55_col: st.column_config.NumberColumn(
                    help="55세 이상 직원이 회사에 남아 일한 '인원 × 연수'의 총합입니다. 클수록 고령 인력이 오래 남습니다."),
            })
        st.markdown(f'<div style="margin-top:10px;line-height:2.1">'
                    f'<div><span class="chip-in">후보</span>&nbsp; 다른 안보다 나쁜 점이 없는 시나리오</div>'
                    f'<div><span class="chip-out">탈락</span>&nbsp; 더 나은 대안이 있어 제외된 시나리오</div>'
                    f'</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:#98A4AE;font-size:.78rem;margin-top:10px;line-height:1.6">'
                    f'"90% 변동구간"은 같은 조건에서 시뮬레이션을 여러 번 돌렸을 때 생기는 계산상 오차 범위입니다. '
                    f'재고용 수락률처럼 회사마다 다른 조건은 <b>조건이 달라지면?</b> 탭에서 따로 확인하세요.</div>',
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
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

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
                          xaxis_title=f"{H}년 인건비 (억 원)",
                          yaxis_title="55세+ 근속 총량",
                          margin=dict(t=30, b=10))
        fig.update_xaxes(gridcolor=LINE)
        fig.update_yaxes(gridcolor=LINE)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
        st.markdown('<span style="color:#5D6B77;font-size:.85rem">큰 점 = 후보 시나리오. 왼쪽 아래로 '
                    '갈수록 비용이 적고, 위로 갈수록 55세 이상 인력이 많이 남습니다. 어느 쪽을 '
                    '우선할지는 회사의 선택입니다.</span>',
                    unsafe_allow_html=True)

    with tab4:
        st.markdown(f'<div style="color:{SUB};font-size:.88rem;line-height:1.6">'
                    f'위 결과는 "재고용을 얼마나 받아들일지·재고용 시 임금을 얼마나 줄지"를 회사가 정확히 '
                    f'모른 채 <b style="color:{INK}">추정값</b>으로 계산한 것입니다. 이 추정이 틀려도 추천 '
                    f'후보가 그대로인지, 아래에서 미리 확인할 수 있습니다.</div>',
                    unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        sens_go = st.button(f"조건을 바꿔가며 다시 계산 (5×5 조합 × {N_SCEN}개 시나리오, 약 1분 소요)")

        @st.cache_data(show_spinner="수락률·임금률 조합 25개를 재시뮬레이션하고 있습니다")
        def run_sensitivity_grid(n_emp, avg_wage, push, other, horizon, hiring, regime):
            prof = CompanyProfile(n_employees=n_emp, avg_wage=avg_wage,
                                  base_push_turnover=push, other_exit_rate=other,
                                  wage_regime=regime)
            # 민감도는 정밀도보다 반응성 우선 — 메인 분석보다 낮은 반복수(30) 고정.
            # 25개 조합 × 시나리오 수 × 30회 반복.
            return run_sensitivity(prof, runs=30, horizon=horizon, hiring=hiring)

        if sens_go:
            st.session_state["sens"] = run_sensitivity_grid(
                n_emp, avg_wage, push, other, horizon, hiring, regime)

        if "sens" in st.session_state:
            sens = st.session_state["sens"]
            accepts = sorted({r["accept"] for r in sens["grid"]})
            wages = sorted({r["wage_ratio"] for r in sens["grid"]})
            cell = {(r["accept"], r["wage_ratio"]): r["frontier"] for r in sens["grid"]}
            z = [[len(set(cell[(a, w)]) ^ set(frontier)) for w in wages] for a in accepts]
            hover = [[", ".join(x.split(" ")[0] for x in cell[(a, w)]) for w in wages] for a in accepts]

            fig = go.Figure(go.Heatmap(
                z=z, x=[f"{w:.2f}" for w in wages], y=[f"{a:.2f}" for a in accepts],
                customdata=hover,
                hovertemplate="수락률 %{y} · 임금률 %{x}<br>후보군: %{customdata}<extra></extra>",
                colorscale=[[0, TEAL], [0.5, "#F3EFD8"], [1, "#A63D57"]],
                zmin=0, zmax=max(1, max(v for row in z for v in row)),
                colorbar=dict(title="차이", tickfont=dict(size=10))))
            # 히트맵 격자는 입력 폼의 슬라이더 step과 어긋날 수 있어(예: 격자 0.75는 0.02 step으로
            # 정확히 도달 불가) 현재 입력값을 그대로 좌표로 찍으면 축 밖에 별도 카테고리로 붙어버림.
            # 그래서 가장 가까운 격자점에 별표를 찍고, 정확히 일치하지 않으면 그 사실을 캡션에 명시한다.
            near_a = min(accepts, key=lambda a: abs(a - accept))
            near_w = min(wages, key=lambda w: abs(w - wage_ratio))
            fig.add_trace(go.Scatter(x=[f"{near_w:.2f}"], y=[f"{near_a:.2f}"],
                                     mode="markers", marker=dict(symbol="star", size=16,
                                     color=INK, line=dict(color="#fff", width=1.5)),
                                     name="현재 입력값과 가장 가까운 격자점", showlegend=False))
            # 세로축 제목("재고용 수락률")을 plotly의 기본 yaxis_title로 두면 90도 회전된
            # 세로 글씨로 렌더링돼 읽기 어렵다(사용자 피드백). 축 제목을 아예 빼고,
            # 그래프 위에 가로로 읽는 캡션을 둬서 "세로=수락률, 가로=임금률"을 알려준다.
            fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Pretendard", color=INK),
                              margin=dict(t=20, b=10))
            st.markdown(f'<div style="color:{SUB};font-size:.82rem;margin-bottom:4px">'
                        f'<b>표 읽는 법</b> — 세로(위→아래): 재고용 수락률이 늘어남 · '
                        f'가로(왼쪽→오른쪽): 재고용 임금률이 늘어남</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
            near_note = "" if (near_a == accept and near_w == wage_ratio) else " (가장 가까운 격자점 기준)"
            st.markdown(f'<span style="color:#5D6B77;font-size:.85rem">색이 짙을수록 지금 추천 후보와 달라지는 '
                        f'조합입니다(★ = 지금 입력한 값{near_note}). 화면 전체가 한 가지 색에 가까우면, 재고용 '
                        f'조건이 예상과 달라도 추천 후보는 바뀌지 않는다는 뜻입니다. 그래프 오른쪽 위 아이콘으로 '
                        f'확대·원상복구를 할 수 있습니다.</span>',
                        unsafe_allow_html=True)

            fp = sens["flip_points"]
            if fp:
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                st.markdown(f'<div style="font-weight:700;color:{INK};margin-bottom:2px">'
                            f'조건별로 추천 후보가 어떻게 바뀌는지</div>'
                            f'<div style="color:{SUB};font-size:.85rem;margin-bottom:8px">'
                            f'재고용 수락률·임금률이 아래 범위 안에 있으면, 그때마다 추천 후보는 같은 조합으로 '
                            f'유지됩니다. 범위를 벗어나면 다른 조합으로 바뀝니다.</div>',
                            unsafe_allow_html=True)
                # 격자점 나열 대신 후보군 구성이 같은 조합을 묶어 보여준다 —
                # 거의 같은 행이 12줄 반복되면 읽는 사람이 패턴을 못 본다.
                groups = {}
                for a, w, f in fp:
                    groups.setdefault(tuple(f), []).append((a, w))
                fp_rows = []
                for f, pts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                    acc = [a for a, _ in pts]; wg = [w for _, w in pts]
                    fp_rows.append({"이때의 추천 후보": ", ".join(x.split(" ")[0] for x in f),
                                    "해당 조합 수 (25개 중)": len(pts),
                                    "재고용 수락률 범위": f"{min(acc):.2f} – {max(acc):.2f}",
                                    "재고용 임금률 범위": f"{min(wg):.2f} – {max(wg):.2f}"})
                st.dataframe(pd.DataFrame(fp_rows), width="stretch", hide_index=True)
            else:
                st.markdown(f'<div style="margin-top:16px"><span class="chip-in">견고함</span>&nbsp; '
                            f'<span style="color:{SUB};font-size:.88rem">재고용 조건을 이리저리 바꿔봐도 '
                            f'추천 후보는 전혀 바뀌지 않았습니다.</span></div>', unsafe_allow_html=True)
        else:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(f'<span style="color:#98A4AE;font-size:.85rem">위 버튼을 누르면 계산됩니다. '
                        f'메인 분석과 별도로 실행되며, 계산 결과는 이 화면을 벗어나지 않는 한 남아있습니다.</span>',
                        unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        rc1, rc2 = st.columns([3, 1])
        rc1.markdown(f'<div style="font-weight:700;color:{INK};margin-bottom:2px">해석 리포트</div>'
                     f'<div style="color:{SUB};font-size:.85rem">위 표·그래프의 결과를 짧은 글로 풀어 '
                     f'설명합니다. 표에 없는 숫자는 절대 만들어내지 않도록 문장마다 확인을 거칩니다.</div>',
                     unsafe_allow_html=True)
        report_go = rc2.button("해석 리포트 생성", width="stretch")
        if report_go:
            LIMITATIONS = [
                "재고용 수락률은 실측치가 아니라 추정값입니다.",
                "2016년 정년 의무화 효과는 참고용일 뿐, 확실한 인과관계로 보기는 어렵습니다.",
                f"채용 방식({HIRING_POLICIES[hiring][0]})·임금 가정({WAGE_REGIMES[regime]['label']})을 "
                f"다르게 고르면 결과가 달라질 수 있습니다.",
                "55세 이상 인력 수치는 규모일 뿐, 숙련도를 뜻하지 않습니다.",
                "효과가 검증되지 않은 정책(예: 교육투자)은 비용만 반영했습니다.",
            ]
            sens_for_report = st.session_state.get("sens") or dict(
                baseline_frontier=frontier, flip_points=[])
            payload = build_llm_input(dict(results=results, frontier=frontier, dominated=dominated),
                                      sens_for_report, LIMITATIONS)
            with st.spinner("해석 리포트를 생성하고 있습니다"):
                st.session_state["report"] = generate_report(payload)

        if "report" in st.session_state:
            rep = st.session_state["report"]
            badge = ('<span class="chip-in">AI가 방금 작성</span>' if rep["is_live"] else
                     '<span class="chip-out">기본 요약 (AI 미연결)</span>')
            st.markdown(f'<div style="margin:14px 0 8px">{badge}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="background:{PAPER};border:1px solid {LINE};border-radius:12px;'
                        f'padding:16px 18px;color:{INK};font-size:.92rem;line-height:1.8">'
                        f'{rep["prose"] or "표시할 문장이 없습니다 — 생성된 산문의 모든 수치가 사후 검증에서 탈락했습니다."}'
                        f'</div>', unsafe_allow_html=True)
            if rep["dropped"]:
                st.markdown(f'<div style="color:#98A4AE;font-size:.78rem;margin-top:8px">'
                            f'근거가 불확실한 문장 {len(rep["dropped"])}개는 표시하지 않았습니다.</div>',
                            unsafe_allow_html=True)
            # 한계 문구는 저자 텍스트 — 검증을 거치지 않고 원문 그대로 렌더한다.
            if rep.get("limitations"):
                items = "".join(f"<li style='margin-bottom:5px'>{x}</li>" for x in rep["limitations"])
                st.markdown(f'<div style="margin-top:14px;color:{SUB};font-size:.83rem;line-height:1.7">'
                            f'<b style="color:{INK}">참고하면 좋은 점</b>'
                            f'<ul style="margin:8px 0 0 18px;padding:0">{items}</ul></div>',
                            unsafe_allow_html=True)

else:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    c = st.columns(3)
    boxes = [(f"{N_SCEN}개 정책 시나리오", "현행 유지부터 2032년 급진 연장, 2041년 완만한 연장까지 "
                                        "국회 특위·경사노위에서 실제 검토 중인 안을 그대로 반영합니다."),
             ("자동 후보 선별", "비용·채용·승진·고령인력, 이 네 가지를 모두 따져서 더 나은 대안이 "
                            "있는 시나리오는 자동으로 제외합니다. 남은 후보끼리는 무엇을 얻고 "
                            "무엇을 포기하는지만 비교하면 됩니다."),
             ("불확실성 표시", "모든 수치는 하나로 딱 떨어지는 값이 아니라, 계산 과정에서 생길 수 "
                            "있는 오차를 반영한 범위(90% 구간)로 함께 보여드립니다. 숫자 하나로 "
                            "단정하기보다는 범위로 판단해보세요.")]
    for col, (t, b) in zip(c, boxes):
        col.markdown(f"""<div style="background:{PAPER};border:1px solid {LINE};
        border-radius:14px;padding:20px 22px;min-height:150px">
        <div style="font-weight:700;color:{INK};margin-bottom:8px">{t}</div>
        <div style="color:{SUB};font-size:.9rem;line-height:1.7">{b}</div></div>""",
                     unsafe_allow_html=True)

st.markdown('<div class="footnote">데이터 기반: 한국노동패널조사(KLIPS) 26·27차 · 사업체패널조사(WPS) 2023 · '
            '고용형태별근로실태조사 2025(KOSIS). STRATA는 의사결정 참고용 도구이며 '
            '법률·재무 자문을 대체하지 않습니다.</div>', unsafe_allow_html=True)
