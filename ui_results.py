"""ui_results.py — ③ 결과: 요약, 비교표, 연도별 추이, 비용 분해, 인력 구조, 부서별."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import decision as D
import interpret as I
from ui_common import (INK, SUB, LINE, TEAL, MUTED, GROUP_C, LOW_C, HIGH_C, html, gap, section, note,
                       card, chip, short, label, color_map, base_layout, show, kpi)

STALE_MSG = "설정이나 데이터가 바뀌었습니다. 아래 결과는 이전 설정으로 계산한 것이니, ② 단계에서 다시 실행해 주세요."


def judged(res):
    """화면에 쓸 판정(④에서 기준을 바꿨으면 그 결과, 아니면 실행 시점 판정)."""
    return st.session_state.get("judge") or res["out"]


def render(res, stale):
    out, base = res["out"], res["baseline"]
    results = out["results"]
    j = judged(res)
    frontier, dominated = j["frontier"], j["dominated"]
    H = res["horizon"]
    names = [r["summary"]["scenario"] for r in results]
    colors = color_map(names, base)
    d = I.deltas(results, base)
    if stale:
        st.warning(STALE_MSG)

    html(f'<div class="h-sub" style="margin:4px 0 14px"><b style="color:{INK}">{res["profile"].name}</b> · '
         f'{results[0]["summary"]["n_initial"]:,}명 · {res["profile"].start_year}~{res["profile"].start_year+H-1}년 '
         f'({H}년) · 시나리오 {len(results)}개 · {res["runs"]}회 반복</div>')

    others = [n for n in names if n != base]
    k = st.columns(4)
    kpi(k[0], "남은 후보", f"{len(frontier)} / {len(results)}", f"{len(dominated)}개 제외" if dominated else "모두 후보")
    if others:
        h_n = min(others, key=lambda n: d[n]["hires"])
        c_n = max(others, key=lambda n: abs(d[n]["cost"]))
        w_n = max(others, key=lambda n: d[n]["wait"])
        kpi(k[1], "신규채용이 가장 많이 줄어드는 안", f"{d[h_n]['hires_abs']:+,.0f}명",
            f"{short(h_n)} · 현행 대비 {d[h_n]['hires']*100:+.1f}%")
        kpi(k[2], "인건비가 가장 크게 달라지는 안", f"{d[c_n]['cost_abs']:+,.0f}억 원",
            f"{short(c_n)} · 현행 대비 {d[c_n]['cost']*100:+.2f}%")
        kpi(k[3], "승진이 가장 많이 밀리는 안", f"{d[w_n]['wait']:+.1f}년", f"{short(w_n)} · 승진 대상자 대기 기간")
    gap(10)
    cards = I.headline(out, base)
    if cards:
        cols = st.columns(len(cards))
        for col, (t, b) in zip(cols, cards):
            card(col, t, b)
    gap(8)
    if others:
        note(I.verdict_sentence(dict(frontier=frontier, dominated=dominated)))
    else:
        st.info("지금은 현행 유지(S1)만 계산했습니다. 다른 안과 비교하려면 ② 단계에서 시나리오를 고르거나 직접 만들어 다시 실행해 주세요.")

    tabs = ["시나리오 비교", "연도별 추이", "왜 차이가 나는가", "인력 구조", "비용과 인력"]
    has_skill = others and "skill_value_npv" in results[0]["runs"]
    if has_skill:
        tabs.insert(3, "숙련 손실까지 보면")
    multi_dept = len(res["profile"].dept_names) > 1
    if multi_dept:
        tabs.insert(tabs.index("비용과 인력"), "부서별")
    T = dict(zip(tabs, st.tabs(tabs)))

    with T["시나리오 비교"]:
        _compare(res, j, d, colors)
    with T["연도별 추이"]:
        _trend(res, frontier, colors)
    with T["왜 차이가 나는가"]:
        _decompose(res, others)
    if has_skill:
        with T["숙련 손실까지 보면"]:
            _skill(res, others, colors, frontier)
    with T["인력 구조"]:
        _structure(res, others)
    if multi_dept:
        with T["부서별"]:
            _dept(res, others)
    with T["비용과 인력"]:
        _tradeoff(res, frontier, colors)


# ── 비교표 ──────────────────────────────────────────────────────────────────
def _compare(res, j, d, colors):
    out, base, H = res["out"], res["baseline"], res["horizon"]
    frontier, stab = j["frontier"], j.get("stability", {})
    tg = I.tags(out["results"], frontier)
    rows = []
    for r in out["results"]:
        s = r["summary"]
        n = s["scenario"]
        rows.append({"시나리오": n, "판정": "후보" if n in frontier else "제외",
                     "인건비(억)": s["cost_npv_mean"] / 1e4, "인건비 변화": d[n]["cost"] * 100,
                     "신규채용(명)": s["total_hires_mean"], "채용 변화": d[n]["hires"] * 100,
                     "승진 대기(년)": s["promo_congestion_mean"],
                     "숙련 공백(인년)": s.get("skill_gap_mean", np.nan),
                     "소득 공백(인년)": s["income_gap_mean"],
                     "판정 안정성": stab.get(n, np.nan) * 100})
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.map(lambda v: f"color:{TEAL};font-weight:700" if v == "후보" else f"color:{SUB}",
                     subset=["판정"]),
        width="stretch", hide_index=True,
        column_config={
            "인건비(억)": st.column_config.NumberColumn(
                "인건비(억)", format="%,.0f",
                help="이 시나리오를 택했을 때 앞으로 들어갈 인건비를 오늘 돈 가치로 환산해 더한 금액입니다(할인율 연 3%)."),
            "인건비 변화": st.column_config.NumberColumn("현행 대비", format="%+.2f%%", width="small"),
            "신규채용(명)": st.column_config.NumberColumn("신규채용(명)", format="%,.0f", help="분석 기간 동안 새로 뽑는 인원의 합계입니다."),
            "채용 변화": st.column_config.NumberColumn("현행 대비 ", format="%+.1f%%", width="small"),
            "승진 대기(년)": st.column_config.NumberColumn(
                "승진 대기", format="%.1f년", width="small", help="승진 자격을 갖춘 직원 10명 중 9명이 승진할 때까지 더 기다리는 햇수입니다. 길수록 승진이 밀립니다."),
            "숙련 공백(인년)": st.column_config.NumberColumn(
                "숙련 공백", format="%,.0f", width="small",
                help="사람이 나간 자리가 다시 제 몫을 할 때까지의 손실을 '인원 × 햇수'로 더한 값입니다. "
                     "충원 기간 + 후임이 익숙해지는 동안의 부족분. 적을수록 좋습니다."),
            "소득 공백(인년)": st.column_config.NumberColumn(
                "소득 공백", format="%,.0f", width="small", help="퇴직자가 국민연금을 받기 시작할 때까지 소득이 끊기는 햇수의 합입니다. 근로자 측 지표입니다."),
            "판정 안정성": st.column_config.ProgressColumn(
                "판정 안정성", format="%.0f%%", min_value=0, max_value=100, width="small",
                help="계산을 다시 해도 '후보'로 남을 확률입니다. 100%에 가까우면 확실한 후보, 0%에 가까우면 확실한 제외, "
                     "그 사이면 경계선에 있다는 뜻입니다."),
        })
    strong = [f'<b style="color:{INK}">{short(n)}</b> {" · ".join(t)}' for n, t in tg.items() if t and n in frontier]
    if strong and len(out["results"]) > 1:
        note(f"모든 금액·인원은 {H}년 합계입니다. 후보별 강점 — " + " &nbsp;/&nbsp; ".join(strong))
    else:
        note(f"모든 금액·인원은 {H}년 합계입니다.", "fine")
    html(f'<div style="margin-top:10px;line-height:2.1">'
         f'<div>{chip("후보")}&nbsp; 다른 어떤 시나리오에도 모든 면에서 밀리지 않는 시나리오</div>'
         f'<div>{chip("제외", "out")}&nbsp; 더 나은 대안이 있어 제외된 시나리오</div></div>')
    if j["dominated"]:
        gap(6)
        html('<div class="card-t">제외된 이유</div>')
        levers = {s.name for s in res["scenarios"]
                  if s.wage_peak or s.wage_reform_regime or s.erp_take_rate > 0}
        flag = False
        for n in j["dominated"]:
            html(f'<div class="note"><b style="color:{INK}">{label(n)}</b> — {I.why_text(n, j["why"])}</div>')
            flag = flag or j["why"].get(n, {}).get("by") in levers
        if flag:
            note("주의: 임금피크·임금체계 개편·희망퇴직을 넣은 안은 <b>줄어드는 인건비만</b> 계산에 들어갑니다. "
                 "노사 협의 비용, 사기 저하, 법적 분쟁 위험(합리적 이유 없는 임금피크제는 무효라는 2022년 대법원 판결)은 "
                 "계산에 없으므로, 이런 수단을 쓴 안이 '쓰지 않은 같은 안'을 이기는 것은 당연한 결과입니다. "
                 "실제 도입 가능성은 따로 판단해야 합니다.", "fine")
    gt = I.income_gap_text(out, base) if len(out["results"]) > 1 else None
    if gt:
        gap(6)
        note(gt)

    others = [r["summary"]["scenario"] for r in out["results"] if r["summary"]["scenario"] != base]
    if others:
        gap(4)
        html('<div class="card-t" style="margin-top:14px">현행 유지와 비교한 변화</div>')
        c = st.columns(2)
        for col, key, title, fmt in ((c[0], "hires", "신규채용 (%)", "%{x:+.1f}%"),
                                     (c[1], "cost", "인건비 (%)", "%{x:+.2f}%")):
            vals = [d[n][key] * 100 for n in others]
            fig = go.Figure(go.Bar(
                y=[label(n) for n in others], x=vals, orientation="h",
                marker_color=[colors[n] if n in frontier else "#c9d1d7" for n in others],
                marker_cornerradius=4, text=[f"{v:+.1f}%" if key == "hires" else f"{v:+.2f}%" for v in vals],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>" + title.split(" ")[0] + " " + fmt + "<extra></extra>"))
            pad = max(max(abs(v) for v in vals) * 0.35, 0.2)
            base_layout(fig, 70 + 38 * len(others), unit=title, bargap=0.35, margin=dict(t=34, b=10, l=10, r=40))
            fig.update_xaxes(range=[min(min(vals), 0) - pad, max(max(vals), 0) + pad], ticksuffix="%")
            fig.update_yaxes(autorange="reversed", showgrid=False)
            with col:
                show(fig, f"delta_{key}", bar=False)
        note("색이 있는 막대는 후보, 회색 막대는 제외된 시나리오입니다. "
             "'90% 범위' 등 오차 정보는 ④ 단계의 '두 시나리오 맞비교'에서 볼 수 있습니다.", "fine")


# ── 연도별 추이 ─────────────────────────────────────────────────────────────
METRICS = {
    "연간 인건비": ("labor_cost", 1e-4, "억 원", "한 해에 나가는 인건비(임금 + 4대보험·퇴직급여)입니다."),
    "신규채용": ("n_hire", 1, "명", "그 해에 새로 뽑는 인원입니다. 정년이 한 살 올라가는 해마다 퇴직자가 줄어 채용이 움푹 꺼집니다."),
    "55세 이상 비중": ("share_55plus", 100, "%", "전체 직원 중 55세 이상의 비율입니다."),
    "평균 나이": ("avg_age", 1, "세", "전체 직원의 평균 나이입니다."),
    "승진자 수": ("n_promoted", 1, "명", "그 해에 한 직급 올라간 인원입니다. 윗자리가 비지 않으면 줄어듭니다."),
    "재고용 인원": ("n_ce", 1, "명", "정년 후 재고용(촉탁)으로 일하고 있는 인원입니다."),
    "정년 나이": ("retirement_age", 1, "세", "그 해에 적용되는 정년입니다. 시나리오마다 올라가는 속도가 다릅니다."),
}


def _trend(res, frontier, colors):
    out, base = res["out"], res["baseline"]
    metric = st.radio("지표", list(METRICS), horizontal=True, label_visibility="collapsed", key="w_trend_metric")
    key, scale, unit, desc = METRICS[metric]
    fig = go.Figure()
    for r in out["results"]:
        n, p = r["summary"]["scenario"], r["path"]
        on = n in frontier
        fig.add_trace(go.Scatter(
            x=[x["year"] for x in p], y=[x[key] * scale for x in p], name=label(n),
            mode="lines", line_shape="hv" if key == "retirement_age" else "linear",
            line=dict(color=colors[n], width=2.6 if n == base else (2.2 if on else 1.4),
                      dash="dash" if n == base else (None if on else "dot")),
            opacity=1.0 if on or n == base else 0.6,
            hovertemplate=f"%{{y:,.1f}}{unit}<extra>{label(n)}</extra>"))
    base_layout(fig, 440, unit=unit, hovermode="x unified",
                legend=dict(orientation="h", y=-0.16, font=dict(size=12)))
    fig.update_xaxes(dtick=2 if res["horizon"] <= 15 else 3)
    show(fig, "trend")
    note(f"{desc}<br>굵은 점선은 현행 유지, 실선은 후보, 가는 점선은 제외된 시나리오입니다. "
         "그래프 위에 마우스를 올리면 그 해의 값을 볼 수 있습니다.", "fine")


# ── 비용 분해 ───────────────────────────────────────────────────────────────
PARTS = [("cost_reg", "기존 정년 이하 정규 인력", GROUP_C["mid"]),
         ("cost_ext", "정년이 늘어 더 일하는 사람", GROUP_C["old"]),
         ("cost_ce", "재고용된 사람", GROUP_C["ce"]),
         ("cost_erp", "희망퇴직 위로금", "#e34948"),
         ("cost_edu", "재교육 투자", "#1baf7a")]


def _pick(res, others, key, default=None):
    if not others:
        return None
    idx = others.index(default) if default in others else 0
    return st.selectbox("살펴볼 시나리오", others, index=idx, key=key, format_func=label)


def _decompose(res, others):
    if not others:
        note("현행 유지 외에 비교할 시나리오가 없습니다.")
        return
    out, base = res["out"], res["baseline"]
    pick = _pick(res, others, "w_dec_pick")
    P = {r["summary"]["scenario"]: r["path"] for r in out["results"]}
    a, b = P[pick], P[base]
    years = [x["year"] for x in a]
    fig = go.Figure()
    tot = {}
    for key, lab, col in PARTS:
        diff = [(x[key] - y[key]) / 1e4 for x, y in zip(a, b)]
        tot[key] = sum(diff)
        if max(abs(v) for v in diff) < 0.005:
            continue
        fig.add_bar(x=years, y=diff, name=lab, marker_color=col,
                    hovertemplate="%{x}년<br>" + lab + " %{y:+,.1f}억<extra></extra>")
    net = [(x["labor_cost"] - y["labor_cost"]) / 1e4 for x, y in zip(a, b)]
    fig.add_trace(go.Scatter(x=years, y=net, name="합계(순 변화)", mode="lines+markers",
                             line=dict(color=INK, width=2), marker=dict(size=8, color=INK, line=dict(color="white", width=2)),
                             hovertemplate="%{x}년<br>순 변화 %{y:+,.1f}억<extra></extra>"))
    base_layout(fig, 440, unit="현행 유지와의 차이 (억 원)", barmode="relative", bargap=0.3, hovermode="x unified",
                legend=dict(orientation="h", y=-0.16, font=dict(size=12)))
    fig.update_traces(marker_line=dict(color="white", width=1.5), selector=dict(type="bar"))
    fig.update_xaxes(dtick=2 if res["horizon"] <= 15 else 3)
    show(fig, "decomp")
    up = tot["cost_ext"] + tot["cost_ce"] + tot["cost_erp"] + tot["cost_edu"]
    down = tot["cost_reg"]
    total = sum(net)
    peak_saved = sum(x["peak_saving"] for x in a) / 1e4
    msg = (f"<b>{label(pick)}</b>를 택하면 {res['horizon']}년 동안 고령 인력 쪽 인건비가 "
           f"<b>{up:+,.0f}억 원</b> 늘어납니다(정년 연장 {tot['cost_ext']:+,.0f}억, 재고용 {tot['cost_ce']:+,.0f}억"
           + (f", 희망퇴직 위로금 {tot['cost_erp']:+,.0f}억" if abs(tot["cost_erp"]) >= 0.5 else "") + ").<br>")
    if down < -0.5:
        share = min(abs(down) / max(up, 1e-9), 9.99)
        msg += (f"반대로 기존 정년 이하 인력의 인건비는 <b>{down:+,.0f}억 원</b>입니다. 나가는 사람이 줄어 신입을 덜 뽑기 때문이며, "
                f"늘어난 비용의 <b>{share*100:.0f}%</b>를 상쇄합니다.<br>")
    msg += f"둘을 합친 순 변화는 <b>{total:+,.0f}억 원</b>(할인 전 단순 합계)입니다."
    if peak_saved >= 0.5:
        msg += f" 임금피크로 줄어든 금액 {peak_saved:,.0f}억 원은 이미 반영돼 있습니다."
    note(msg)
    note("막대가 0 위로 올라가면 현행보다 더 드는 돈, 아래로 내려가면 덜 드는 돈입니다. "
         "'인건비 차이가 작다'는 결과는 두 방향이 서로 지워진 결과이지, 아무 일도 없다는 뜻이 아닙니다.", "fine")


# ── 인력 구조 ───────────────────────────────────────────────────────────────
BANDS = [("n_u35", "35세 미만", GROUP_C["young"]), ("n_35_54", "35~54세", GROUP_C["mid"]),
         ("n_55_59", "55~59세", GROUP_C["pre"]), ("n_60p", "60세 이상", GROUP_C["old"])]


def _structure(res, others):
    out, base = res["out"], res["baseline"]
    P = {r["summary"]["scenario"]: r["path"] for r in out["results"]}
    pick = _pick(res, others, "w_str_pick") if others else None
    cols = st.columns(2 if pick else 1)
    for col, name in zip(cols, [base] + ([pick] if pick else [])):
        p = P[name]
        fig = go.Figure()
        for key, lab, c in BANDS:
            fig.add_bar(x=[x["year"] for x in p], y=[x[key] for x in p], name=lab, marker_color=c,
                        hovertemplate="%{x}년 " + lab + " %{y:,.0f}명<extra></extra>")
        base_layout(fig, 380, unit=f"{label(name)} (명)", barmode="stack", bargap=0.2,
                    legend=dict(orientation="h", y=-0.18, font=dict(size=12)), hovermode="x unified")
        fig.update_traces(marker_line=dict(color="white", width=1.5))
        fig.update_xaxes(dtick=3)
        with col:
            show(fig, f"struct_{name}")
    if pick:
        a, b = P[pick][-1], P[base][-1]
        n_a, n_b = max(a["headcount"], 1), max(b["headcount"], 1)
        y = a["year"]
        note(f"{y}년 기준으로 <b>{label(pick)}</b>에서는 60세 이상이 <b>{a['n_60p']:,.0f}명({a['n_60p']/n_a*100:.1f}%)</b>, "
             f"현행 유지에서는 {b['n_60p']:,.0f}명({b['n_60p']/n_b*100:.1f}%)입니다.<br>"
             f"35세 미만은 {a['n_u35']:,.0f}명({a['n_u35']/n_a*100:.1f}%) 대 {b['n_u35']:,.0f}명({b['n_u35']/n_b*100:.1f}%) — "
             f"고령층이 늘어난 만큼 청년층 자리가 <b>{b['n_u35']-a['n_u35']:,.0f}명</b> 줄어든 구조입니다. "
             f"평균 나이는 {a['avg_age']:.1f}세 대 {b['avg_age']:.1f}세입니다.")


# ── 부서별 ──────────────────────────────────────────────────────────────────
def _dept(res, others):
    out, base, prof = res["out"], res["baseline"], res["profile"]
    P = {r["summary"]["scenario"]: r["path"] for r in out["results"]}
    if "dept_hc" not in P[base][0]:
        note("부서별 기록이 없습니다.")
        return
    pick = _pick(res, others, "w_dept_pick") if others else base
    depts = prof.dept_names
    last_a, last_b = P[pick][-1], P[base][-1]
    sh = lambda row: [100 * a / max(h, 1e-9) for a, h in zip(row["dept_55"], row["dept_hc"])]
    first = sh(P[base][0])
    fig = go.Figure()
    fig.add_bar(y=depts, x=first, name=f"지금({P[base][0]['year']}년)", orientation="h", marker_color="#c9d1d7",
                hovertemplate="%{y}<br>지금 %{x:.1f}%<extra></extra>")
    fig.add_bar(y=depts, x=sh(last_b), name=f"현행 유지 {last_b['year']}년", orientation="h", marker_color=LOW_C,
                hovertemplate="%{y}<br>현행 유지 %{x:.1f}%<extra></extra>")
    if pick != base:
        fig.add_bar(y=depts, x=sh(last_a), name=f"{short(pick)} {last_a['year']}년", orientation="h",
                    marker_color=HIGH_C, hovertemplate="%{y}<br>" + short(pick) + " %{x:.1f}%<extra></extra>")
    base_layout(fig, 90 + 56 * len(depts), barmode="group", bargap=0.3, bargroupgap=0.08,
                xaxis_title="55세 이상 비중 (%)", legend=dict(orientation="h", y=1.08, x=0))
    fig.update_traces(marker_cornerradius=3)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    show(fig, "dept55", bar=False)
    if pick != base:
        diff = np.array(sh(last_a)) - np.array(sh(last_b))
        i = int(np.argmax(diff))
        cost_a = np.sum([r["dept_cost"] for r in P[pick]], axis=0) / 1e4
        cost_b = np.sum([r["dept_cost"] for r in P[base]], axis=0) / 1e4
        ci = int(np.argmax(cost_a - cost_b))
        note(f"<b>{label(pick)}</b>의 영향이 가장 크게 나타나는 부서는 <b>{depts[i]}</b>입니다. "
             f"{last_a['year']}년 55세 이상 비중이 현행 유지보다 <b>{diff[i]:+.1f}%p</b> 높아집니다.<br>"
             f"{res['horizon']}년 인건비가 가장 많이 늘어나는 부서는 <b>{depts[ci]}</b>"
             f"(<b>{cost_a[ci]-cost_b[ci]:+,.0f}억 원</b>)입니다. 제도가 바뀌면 이 부서부터 인력 계획을 다시 짜야 합니다.")
    out_a = np.sum([r["dept_out"] for r in P[pick]], axis=0)
    out_b = np.sum([r["dept_out"] for r in P[base]], axis=0)
    t = pd.DataFrame({"부서": depts, "지금 인원": np.round(P[base][0]["dept_hc"]).astype(int),
                      f"정년·재고용 종료로 나가는 인원 (현행 유지, {res['horizon']}년)": np.round(out_b).astype(int)})
    if pick != base:
        t[f"같은 기간 ({short(pick)})"] = np.round(out_a).astype(int)
        t["차이"] = (np.round(out_a) - np.round(out_b)).astype(int)
    if "dept_skill" in P[base][0]:
        sk_b = np.sum([r["dept_skill"] for r in P[base]], axis=0)
        t["숙련 공백(인년, 현행)"] = np.round(sk_b).astype(int)
        if pick != base:
            sk_a = np.sum([r["dept_skill"] for r in P[pick]], axis=0)
            t[f"숙련 공백({short(pick)})"] = np.round(sk_a).astype(int)
    st.dataframe(t, width="stretch", hide_index=True)
    note("신규 인원은 결원이 생긴 부서에 채워 부서 규모가 유지된다고 가정합니다. 부서 간 이동·조직 개편은 반영하지 않습니다.", "fine")


# ── 숙련 손실까지 보면 ──────────────────────────────────────────────────────
def _skill(res, others, colors, frontier):
    out, base, H, prof = res["out"], res["baseline"], res["horizon"], res["profile"]
    R = {r["summary"]["scenario"]: r for r in out["results"]}
    b = R[base]
    note("사람이 나가면 그 자리가 다시 제 몫을 할 때까지 손실이 생깁니다. "
         "<b>충원까지 비어 있는 기간</b>과 <b>후임이 익숙해지는 동안 모자라는 몫</b>을 더한 것이 숙련 공백입니다.<br>"
         "예: 숙련 3년짜리 자리에서 20년 일한 사람이 나가면, 충원 3개월(0.25년) + 3년 동안 평균 절반의 몫(1.5년) = "
         "<b>1.75인-년</b>의 공백이 생깁니다.")
    if prof.job_table_is_default():
        st.warning("직군별 충원·숙련 기간이 입력되지 않아 모든 직군에 같은 기본값(충원 3개월·숙련 2년)을 썼습니다. "
                   "지금 숫자는 '몇 명이, 얼마나 오래 일한 사람이 나가는가'의 차이만 담습니다. "
                   "② 단계의 직군별 특성 표를 채우면 우리 회사에 맞는 결과가 나옵니다.")

    # ① 숙련 공백 변화
    sk = {n: R[n]["summary"]["skill_gap_mean"] for n in R}
    vals = [(sk[n] / max(sk[base], 1e-9) - 1) * 100 for n in others]
    fig = go.Figure(go.Bar(
        y=[label(n) for n in others], x=vals, orientation="h",
        marker_color=[colors[n] if n in frontier else "#c9d1d7" for n in others], marker_cornerradius=4,
        text=[f"{v:+.1f}%" for v in vals], textposition="outside", cliponaxis=False,
        customdata=[sk[n] for n in others],
        hovertemplate="%{y}<br>현행 대비 %{x:+.1f}% (%{customdata:,.0f}인-년)<extra></extra>"))
    pad = max(max(abs(v) for v in vals) * 0.35, 1)
    base_layout(fig, 70 + 38 * len(others), unit="숙련 공백 (현행 대비 %)", bargap=0.35,
                margin=dict(t=34, b=10, l=10, r=40))
    fig.update_xaxes(range=[min(min(vals), 0) - pad, max(max(vals), 0) + pad], ticksuffix="%")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    show(fig, "skill_delta", bar=False)
    best = min(others, key=lambda n: sk[n])
    note(f"현행 유지의 {H}년 숙련 공백은 <b>{sk[base]:,.0f}인-년</b>입니다. "
         f"가장 많이 줄이는 {I.josa(short(best), '은', '는')} <b>{sk[base]-sk[best]:,.0f}인-년</b>"
         f"({(1-sk[best]/max(sk[base],1e-9))*100:.0f}%)을 덜 잃습니다.<br>"
         "정년이 늦춰지면 숙련자가 나가는 시점이 뒤로 밀리기 때문입니다. 단, 분석 기간이 끝난 뒤로 밀린 퇴장은 "
         "계산에 없어 그만큼은 과대평가일 수 있습니다.", "fine")

    # ② 손익분기
    gap(8)
    html('<div class="card-t">숙련 손실을 돈으로 얼마로 보면 결론이 바뀌나</div>')
    note("숙련 공백 1인-년을 얼마의 손실로 볼지는 회사마다 다릅니다. 그래서 하나로 정하지 않고 "
         "<b>평균 인건비 1년의 몇 %로 보느냐</b>에 따라 결과가 어떻게 바뀌는지 그립니다.<br>"
         "100% = '빈자리 1인-년이 그 직군 직원 한 명의 1년 인건비만큼 손실'입니다. 임금이 생산성과 같다는 표준 가정입니다.")
    lam = np.linspace(0, 2, 41)
    fig = go.Figure()
    be = {n: D.break_even(R[n], b) for n in others}
    ymax = 0.0
    for n in others:
        y = (be[n]["dc"] + lam * be[n]["dv"]) / 1e4
        ymax = max(ymax, float(np.abs(y).max()))
        on = n in frontier
        fig.add_trace(go.Scatter(x=lam * 100, y=y, name=label(n), mode="lines",
                                 line=dict(color=colors[n], width=2.4 if on else 1.4, dash=None if on else "dot"),
                                 opacity=1.0 if on else 0.6,
                                 hovertemplate=f"숙련 가치 %{{x:.0f}}%일 때 %{{y:+,.0f}}억 원<extra>{label(n)}</extra>"))
        if be[n]["x"] is not None and 0 <= be[n]["x"] <= 2:
            fig.add_trace(go.Scatter(x=[be[n]["x"] * 100], y=[0], mode="markers", showlegend=False,
                                     marker=dict(size=9, color=colors[n], line=dict(color="white", width=1.5)),
                                     hovertemplate=f"{label(n)}<br>손익분기 %{{x:.0f}}%<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#8a8f98", width=1))
    ymax = max(ymax, 1.0) * 1.1
    fig.add_annotation(x=1, xref="paper", y=-ymax * 0.92, text="현행보다 유리 ↓", showarrow=False,
                       xanchor="right", font=dict(color=SUB, size=12))
    fig.add_annotation(x=1, xref="paper", y=ymax * 0.92, text="현행보다 불리 ↑", showarrow=False,
                       xanchor="right", font=dict(color=SUB, size=12))
    base_layout(fig, 440, unit="현행 대비 (인건비 + 숙련 손실), 억 원",
                xaxis_title="숙련 공백 1인-년의 가치 (평균 인건비 1년 대비 %)",
                legend=dict(orientation="h", y=-0.2, font=dict(size=12)))
    fig.update_xaxes(ticksuffix="%", range=[0, 200])
    fig.update_yaxes(range=[-ymax, ymax])
    show(fig, "skill_breakeven", bar=False)
    note("선이 0 아래에 있으면 그 가치에서 현행 유지보다 총손실이 적다는 뜻입니다. 점은 선이 0을 지나는 지점(손익분기)입니다.<br>"
         "오른쪽으로 갈수록 숙련을 귀하게 보는 회사입니다. 오른쪽에서 내려가는 선은 숙련을 지키는 안, "
         "올라가는 선은 인건비를 아끼는 대신 숙련을 잃는 안입니다.", "fine")
    order = sorted(others, key=lambda n: (["always", "above", "below", "never"].index(be[n]["kind"]),
                                          be[n]["x"] if be[n]["x"] is not None else 0))
    lines = "".join(f'<div class="note" style="margin:4px 0"><b style="color:{INK}">{label(n)}</b> — '
                    f'{I.break_even_text(n, be[n], H)}</div>' for n in order)
    html(lines)

    # ③ 우리 회사 기준으로 보면
    gap(8)
    html('<div class="card-t">우리 회사가 보는 숙련 가치로 줄 세우기</div>')
    v = st.slider("숙련 공백 1인-년의 가치 (평균 인건비 1년 대비 %)", 0, 200, 100, 10, key="w_skill_lambda",
                  help="0%면 인건비만 보는 것과 같습니다. 숙련이 핵심인 회사일수록 높게 잡습니다.")
    rows = []
    for n in [base] + others:
        s = R[n]["summary"]
        tot = (s["cost_npv_mean"] + v / 100 * s["skill_value_npv_mean"]) / 1e4
        rows.append({"시나리오": label(n), "인건비(억)": s["cost_npv_mean"] / 1e4,
                     "숙련 손실(억)": v / 100 * s["skill_value_npv_mean"] / 1e4, "합계(억)": tot})
    t = pd.DataFrame(rows)
    t["현행 대비"] = t["합계(억)"] - t.loc[0, "합계(억)"]
    t = t.sort_values("합계(억)").reset_index(drop=True)
    t.insert(0, "순위", range(1, len(t) + 1))
    st.dataframe(t, width="stretch", hide_index=True,
                 column_config={k: st.column_config.NumberColumn(format="%,.0f") for k in
                                ("인건비(억)", "숙련 손실(억)", "합계(억)")} |
                               {"현행 대비": st.column_config.NumberColumn(format="%+,.0f"),
                                "순위": st.column_config.NumberColumn(width="small")})
    top = t.iloc[0]
    note(f"숙련 가치를 {v}%로 보면 총손실이 가장 적은 것은 <b>{top['시나리오']}</b>입니다"
         + (f"(현행 대비 {top['현행 대비']:+,.0f}억 원)." if top["시나리오"] != label(base) else "(현행 유지).")
         + "<br>금액은 모두 현재가치이며, 숙련 손실은 그 직군 정규직 평균 인건비를 기준으로 환산했습니다.", "fine")


# ── 비용과 인력 ─────────────────────────────────────────────────────────────
def _tradeoff(res, frontier, colors):
    out, base, H = res["out"], res["baseline"], res["horizon"]
    ys = {"신규채용": ("total_hires_mean", "명", "위로 갈수록 신규채용이 많습니다"),
          "고령 인력 유지": ("exp55_person_years_mean", "인년", "위로 갈수록 55세 이상 인력이 많이 남습니다"),
          "승진 대기": ("promo_congestion_mean", "년", "아래로 갈수록 승진이 빠릅니다")}
    yk = st.radio("세로축", list(ys), horizontal=True, key="w_trade_y", label_visibility="collapsed")
    key, unit, hint = ys[yk]
    fig = go.Figure()
    for r in out["results"]:
        s = r["summary"]
        n = s["scenario"]
        on = n in frontier
        fig.add_trace(go.Scatter(
            x=[s["cost_npv_mean"] / 1e4], y=[s[key]], mode="markers+text", text=[short(n)],
            textposition="top center", textfont=dict(size=12, color=INK if on else SUB),
            marker=dict(size=17 if on else 11, color=colors[n] if on else "#C9D1D7",
                        line=dict(color="white", width=2)),
            name=label(n), showlegend=False,
            error_x=dict(type="data", symmetric=False, thickness=1, width=0, color="#b5bec6",
                         array=[(s["cost_npv_p95"] - s["cost_npv_mean"]) / 1e4],
                         arrayminus=[(s["cost_npv_mean"] - s["cost_npv_p5"]) / 1e4]) if on else None,
            hovertemplate=f"{label(n)}<br>인건비 %{{x:,.0f}}억<br>{yk} %{{y:,.1f}}{unit}<extra></extra>"))
    base_layout(fig, 460, unit=f"{yk} ({unit})", xaxis_title=f"{H}년 인건비 (억 원)")
    show(fig, "tradeoff")
    note(f"큰 점은 후보, 작고 옅은 점은 제외된 시나리오입니다. 왼쪽일수록 비용이 적고, {hint}.<br>"
         "후보 점에 붙은 가로선은 인건비의 90% 범위입니다. 선이 서로 많이 겹치면 인건비로는 "
         "두 시나리오를 구별하기 어렵다는 뜻이니, 다른 축(채용·승진)을 보고 판단하는 편이 낫습니다.", "fine")
