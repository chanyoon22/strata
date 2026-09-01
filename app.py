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

N_SCEN = len(preset_scenarios())

st.markdown('<p class="h-title">정년연장이 우리 회사 인력구조에 미치는 영향</p>', unsafe_allow_html=True)
st.markdown(f'<p class="h-sub">기업 조건을 입력하면 국회에서 경합 중인 {N_SCEN}개 정책 시나리오의 '
            '10~15년 결과를 비교하고, 열위 시나리오를 자동으로 걸러냅니다.</p>', unsafe_allow_html=True)

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
    # 결과를 크게 가르는 두 가지 회사 정책. 예전에는 코드에 고정돼 있어 사용자가 바꿀 수 없었다.
    r3 = st.columns([1, 1, 2])
    hiring = r3[0].selectbox("정년연장 부담을 흡수하는 방식", list(HIRING_POLICIES),
                             format_func=lambda k: HIRING_POLICIES[k][0],
                             help="정년이 늘어 고령자가 남을 때 무엇을 조정할지 고릅니다. "
                                  "이 선택에 따라 권장 후보가 실제로 달라집니다.")
    regime = r3[1].selectbox("50세 이후 임금 궤적", list(WAGE_REGIMES),
                             format_func=lambda k: WAGE_REGIMES[k]["label"],
                             help="정년연장 비용을 좌우하는 핵심 가정입니다. 기본값은 현재 "
                                  "관측되는 연령-임금 곡선이며, 호봉제가 정년까지 작동한다고 "
                                  "보면 비용이 크게 올라갑니다.")
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
        ("신규채용 여력 격차", f"{max(hires)-min(hires):,.0f}명",
         f"{(max(hires)/max(min(hires),1)-1)*100:+.1f}%", (max(hires)/max(min(hires),1)-1)),
        ("고령인력 규모 격차 (55세+ 인년)", f"{max(e55)-min(e55):,.0f}",
         f"{(max(e55)/max(min(e55),1)-1)*100:+.1f}%", (max(e55)/max(min(e55),1)-1)),
        ("인건비 격차 (최대–최소)", f"{(max(npvs)-min(npvs))/1e4:,.0f}억 원",
         f"{(max(npvs)/min(npvs)-1)*100:+.1f}%", (max(npvs)/min(npvs)-1)),
    ], key=lambda x: -x[3])

    k = st.columns(4)
    k[0].metric("남은 후보", f"{len(frontier)} / {len(results)}",
                f"{len(dominated)}개 열위 탈락" if dominated else "전원 유지",
                delta_color="off")
    for col, (label, val, delta, _) in zip(k[1:], kpis):
        col.metric(label, val, delta, delta_color="off")

    top = kpis[0][0].split(" 격차")[0]
    st.markdown(f'<div style="color:{SUB};font-size:.86rem;margin:10px 2px 0">'
                f'시나리오를 가장 크게 가르는 것은 <b style="color:{INK}">{top}</b>입니다. '
                f'인건비 차이는 {kpis[-1][2].lstrip("+")}에 그칩니다 — 정년연장의 부담은 '
                f'임금 총액보다 채용 여력과 인력 연령구성에서 먼저 나타납니다.</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["시나리오 비교", "연도별 추이", "트레이드오프 지도", "민감도 분석"])

    with tab1:
        rows = []
        for r in results:
            s = r["summary"]
            rows.append({"시나리오": s["scenario"],
                         "판정": "후보" if s["scenario"] in frontier else "탈락",
                         npv_col: s["cost_npv_mean"] / 1e4,
                         "90% 변동구간(억)": f"{s['cost_npv_p5']/1e4:,.0f} – {s['cost_npv_p95']/1e4:,.0f}",
                         "신규채용 누계(명)": s["total_hires_mean"],
                         "승진 대기 p90(년)": s["promo_congestion_mean"],
                         "55세+ 재직 인년": s["exp55_person_years_mean"]})
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.format({npv_col: "{:,.0f}", "신규채용 누계(명)": "{:,.0f}",
                             "승진 대기 p90(년)": "{:.1f}", "55세+ 재직 인년": "{:,.0f}"})
              .map(lambda v: f"color:{TEAL};font-weight:700" if v == "후보"
                   else f"color:{SUB}", subset=["판정"]),
            width="stretch", hide_index=True)
        st.markdown('<span class="chip-in">후보</span>&nbsp; 네 가지 지표(비용·채용·승진·고령인력) 어디서도 '
                    '다른 시나리오에 밀리지 않음 &nbsp;&nbsp;'
                    '<span class="chip-out">탈락</span>&nbsp; 같은 조건에서 더 나은 대안 존재',
                    unsafe_allow_html=True)
        st.markdown(f'<div style="color:#98A4AE;font-size:.78rem;margin-top:8px;line-height:1.65">'
                    f'90% 변동구간은 <b>인력 구성 표본변동 + 이직모형 계수 추정오차</b> 두 가지를 '
                    f'반영합니다. 재고용 수락률·임금률처럼 값 자체가 불확실한 입력의 영향은 여기에 '
                    f'포함되지 않으며, 민감도 분석 탭에서 따로 확인하십시오. 미래 경기·산업 충격은 '
                    f'모형에 없습니다 — 이 구간을 예측 신뢰구간으로 읽지 마십시오.</div>',
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
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

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
                          xaxis_title=f"{H}년 인건비 NPV (억 원) — 왼쪽일수록 유리",
                          yaxis_title="55세+ 재직 인년 — 위일수록 유리",
                          margin=dict(t=30, b=10))
        fig.update_xaxes(gridcolor=LINE)
        fig.update_yaxes(gridcolor=LINE)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown('<span style="color:#5D6B77;font-size:.85rem">채워진 큰 점이 후보 시나리오입니다. '
                    '비용과 고령인력 규모 두 축의 트레이드오프에서 어느 방향을 우선할지는 경영 판단의 '
                    '영역입니다. 세로축은 55세 이상 재직 인년(연인원)이며, 숙련도를 직접 측정한 값이 '
                    '아니라 고령인력의 규모를 나타냅니다.</span>',
                    unsafe_allow_html=True)

    with tab4:
        st.markdown('<span style="color:#5D6B77;font-size:.88rem">재고용 수락률·재고용 임금률은 이 분석에서 '
                    '가장 불확실한 입력값입니다(사내 실측 없이 KLIPS 재취업 통계로 추정). 이 두 값이 '
                    '어느 지점에서 위 "후보" 판정 자체를 바꾸는지 격자 탐색으로 확인합니다.</span>',
                    unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        sens_go = st.button(f"민감도 분석 실행 (5×5 격자 × {N_SCEN}개 시나리오, 약 1분 소요)")

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
            fig.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Pretendard", color=INK),
                              xaxis_title="재고용 임금률", yaxis_title="재고용 수락률",
                              margin=dict(t=20, b=10))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            near_note = ("" if (near_a == accept and near_w == wage_ratio) else
                        f" (현재 입력값 수락률 {accept:.2f}·임금률 {wage_ratio:.2f}과(와) 가장 가까운 격자점 기준)")
            st.markdown(f'<span style="color:#5D6B77;font-size:.85rem">색이 짙을수록 지금 입력값 기준 '
                        f'"후보" 판정과 다른 시나리오 조합이 후보군에 들어옵니다(★ 표시{near_note}). '
                        f'테두리 없는 균일한 색이면 결론이 이 구간에서 견고하다는 뜻입니다.</span>',
                        unsafe_allow_html=True)

            fp = sens["flip_points"]
            if fp:
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                # 격자점 나열 대신 후보군 구성이 같은 조합을 묶어 보여준다 —
                # 거의 같은 행이 12줄 반복되면 읽는 사람이 패턴을 못 본다.
                groups = {}
                for a, w, f in fp:
                    groups.setdefault(tuple(f), []).append((a, w))
                fp_rows = []
                for f, pts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                    acc = [a for a, _ in pts]; wg = [w for _, w in pts]
                    fp_rows.append({"그 조건의 후보군": ", ".join(x.split(" ")[0] for x in f),
                                    "해당 조합 수": len(pts),
                                    "수락률 범위": f"{min(acc):.2f} – {max(acc):.2f}",
                                    "임금률 범위": f"{min(wg):.2f} – {max(wg):.2f}"})
                st.dataframe(pd.DataFrame(fp_rows), width="stretch", hide_index=True)
            else:
                st.markdown('<span class="chip-in">견고함</span>&nbsp; 탐색한 격자 전체에서 '
                            '후보군 구성이 바뀌지 않았습니다.', unsafe_allow_html=True)
        else:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(f'<span style="color:#98A4AE;font-size:.85rem">위 버튼을 누르면 계산됩니다. '
                        f'메인 분석과 별도로 실행되며 결과는 세션에 남습니다.</span>',
                        unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        rc1, rc2 = st.columns([3, 1])
        rc1.markdown(f'<div style="font-weight:700;color:{INK};margin-bottom:2px">해석 리포트</div>'
                     f'<div style="color:{SUB};font-size:.85rem">위 수치를 바탕으로 트레이드오프·위험요인·'
                     f'추가 검토 질문을 산문으로 정리합니다. 새 숫자를 만들지 않도록 모든 문장을 사후 검증합니다.</div>',
                     unsafe_allow_html=True)
        report_go = rc2.button("해석 리포트 생성", width="stretch")
        if report_go:
            LIMITATIONS = [
                "재고용 수락률은 사내 제안 수락이 통계로 직접 관측되지 않아 재취업률 앵커(0.37~0.62) 기반 "
                "구간 파라미터입니다 — 자사 실측이 선행 과제입니다.",
                "2016년 60세 의무화 분석은 평행추세 가정을 통과하지 못했습니다. 인과효과가 아니라 "
                "'제도 변화와 같은 방향'이라는 정합성 확인으로만 해석하십시오.",
                f"채용 정책({HIRING_POLICIES[hiring][0]})과 50세 이후 임금 궤적"
                f"({WAGE_REGIMES[regime]['label']})은 이 분석의 전제입니다. 다른 전제를 고르면 "
                f"권장 후보가 달라질 수 있습니다.",
                "55세+ 재직 인년은 고령인력의 규모이지 숙련도를 직접 측정한 값이 아닙니다.",
                "교육투자 등 효과가 검증되지 않은 채널은 비용만 반영하고 효과를 주장하지 않습니다.",
            ]
            sens_for_report = st.session_state.get("sens") or dict(
                baseline_frontier=frontier, flip_points=[])
            payload = build_llm_input(dict(results=results, frontier=frontier, dominated=dominated),
                                      sens_for_report, LIMITATIONS)
            with st.spinner("해석 리포트를 생성하고 있습니다"):
                st.session_state["report"] = generate_report(payload)

        if "report" in st.session_state:
            rep = st.session_state["report"]
            badge = ('<span class="chip-in">실시간 AI 생성</span>' if rep["is_live"] else
                     '<span class="chip-out">오프라인 렌더러(템플릿) — ANTHROPIC_API_KEY 미설정 시 기본값</span>')
            st.markdown(f'<div style="margin:14px 0 8px">{badge}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="background:{PAPER};border:1px solid {LINE};border-radius:12px;'
                        f'padding:16px 18px;color:{INK};font-size:.92rem;line-height:1.8">'
                        f'{rep["prose"] or "표시할 문장이 없습니다 — 생성된 산문의 모든 수치가 사후 검증에서 탈락했습니다."}'
                        f'</div>', unsafe_allow_html=True)
            if rep["dropped"]:
                st.markdown(f'<div style="color:#98A4AE;font-size:.78rem;margin-top:8px">'
                            f'근거 없는 수치가 포함돼 제외된 문장 {len(rep["dropped"])}개(사후 검증 로그)</div>',
                            unsafe_allow_html=True)
            # 한계 문구는 저자 텍스트 — 검증을 거치지 않고 원문 그대로 렌더한다.
            if rep.get("limitations"):
                items = "".join(f"<li style='margin-bottom:5px'>{x}</li>" for x in rep["limitations"])
                st.markdown(f'<div style="margin-top:14px;color:{SUB};font-size:.83rem;line-height:1.7">'
                            f'<b style="color:{INK}">이 해석에 함께 읽어야 할 한계</b>'
                            f'<ul style="margin:8px 0 0 18px;padding:0">{items}</ul></div>',
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
<h4>제도 효과의 검증 — 그리고 그 한계</h4><p>2016년 60세 정년 의무화 당시 55–59세 비자발
퇴출률이 대기업에서 상대적으로 4.0%p 낮아진 것을 확인했습니다. 다만 이 수치를 인과효과로
읽지는 않습니다. 제도 시행 <b>이전부터</b> 대기업과 중소기업의 격차가 이미 연 1.4%p씩 좁혀지고
있었고(사전추세 검정), 그 추세를 통제하면 효과가 사라집니다. 따라서 이 결과는 "제도 변화와
같은 방향의 움직임이 데이터에 있다"는 정합성 확인이지, "의무화가 퇴출을 4%p 줄였다"는
인과 주장이 아닙니다. 관측 표본도 연 20~160명 수준으로 작습니다.</p>
<h4>이 도구가 답하지 못하는 것</h4><p>재고용 수락률은 사내 제안 수락이 통계로 직접 관측되지
않아 재취업률 기반 추정 구간(0.37–0.62)을 씁니다. 자사 실측치가 있다면 위 설정에서 교체하십시오.
50세 이후 임금 궤적은 현재 관측되는 연령별 임금 곡선에서 가져왔는데, 이 곡선에는 이미
임금피크·재고용 전환이 섞여 있어 호봉제가 정년까지 작동하는 회사에는 맞지 않을 수 있습니다 —
설정에서 궤적을 바꿔 결과가 얼마나 달라지는지 확인하십시오. 교육투자 등 효과가 검증되지 않은
정책은 비용만 계산하고 효과를 가정하지 않습니다. 경기 변동·산업 구조 변화·임금 협상은
모형에 없습니다.</p>
</div>""", unsafe_allow_html=True)
else:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    c = st.columns(3)
    boxes = [(f"{N_SCEN}개 정책 시나리오", "현행 유지부터 2032년 급진 연장, 2041년 완만한 연장까지 "
                                        "국회 특위·경사노위에서 실제 검토 중인 안을 그대로 반영합니다."),
             ("자동 후보 선별", "비용·채용·승진·고령인력 네 지표에서 열위인 시나리오를 알고리즘이 "
                            "걸러내고, 남은 후보의 트레이드오프만 보여줍니다."),
             ("불확실성 표시", "모든 수치는 인력 구성과 이직모형 계수의 변동을 반영한 90% 구간과 "
                            "함께 제공됩니다. 단정 대신 범위로 판단하십시오.")]
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
