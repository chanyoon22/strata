"""ui_decide.py — ④ 우리 회사에 맞는 선택 / ⑤ 결론이 흔들리는가."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import app_logic as A
import decision as D
import interpret as I
from dss_layer import (sensitivity, robustness, tornado, design_space, HIRING_POLICIES)
from ui_common import (INK, SUB, LINE, TEAL, MUTED, SEQ, SERIES, LOW_C, HIGH_C, html, gap, section, note,
                       chip, short, label, base_layout, show)

OBJ_NAMES = {"cost_npv": "인건비를 아끼는 것", "total_hires": "신규채용을 유지하는 것",
             "promo_congestion": "승진이 밀리지 않는 것", "exp55_person_years": "55세 이상 인력 규모를 유지하는 것",
             "income_gap": "퇴직자의 소득 공백을 줄이는 것",
             "skill_gap": "사람이 나간 자리의 숙련 공백을 줄이는 것"}
OBJ_SHORT = {k: v["label"] for k, v in D.OBJECTIVES.items()}
OBJ_COLOR = dict(zip(D.OBJECTIVES, SERIES))
LEVELS = {0: "고려 안 함", 1: "보통", 2: "중요", 3: "매우 중요"}


# ═════════════════ ④ 우리 회사에 맞는 선택 ═════════════════
def render_decide(res, cfg, stale):
    if stale:
        st.warning("설정이 바뀌었습니다. 아래는 이전 설정의 결과입니다.")
    out, base = res["out"], res["baseline"]
    results = out["results"]

    section("① 무엇을 기준으로 가릴까요",
            "'후보'는 선택한 항목 가운데 <b>어느 것 하나에서도 뚜렷이 밀리지 않는</b> 시나리오입니다.<br>"
            "우리 회사가 신경 쓰지 않는 항목은 빼고 판정할 수 있습니다. 계산을 다시 돌리지 않고 바로 바뀝니다.")
    c = st.columns([3, 1.3])
    objs = c[0].multiselect("판정에 쓸 항목", list(D.OBJECTIVES), default=cfg["objectives"],
                            format_func=lambda k: OBJ_SHORT[k], key="w_objs")
    mat = c[1].number_input("이보다 작은 차이는 '같다'고 봄 (%)", 0.1, 5.0, float(cfg["materiality"]), 0.1,
                            key="w_mat",
                            help="통계적으로 구별되더라도 이 크기보다 작은 차이는 의사결정에 의미가 없다고 봅니다. "
                                 "기본 0.5%.")
    if not objs:
        st.info("항목을 하나 이상 골라주세요.")
        return
    cfg["objectives"], cfg["materiality"] = objs, mat
    jkey = (res["sig"], tuple(objs), mat)
    if st.session_state.get("judge_key") != jkey:
        st.session_state["judge"] = A.rejudge(res, objs, mat)
        st.session_state["judge_key"] = jkey
    j = st.session_state["judge"]
    note(I.verdict_sentence(j) + " 판정은 평균만 비교하지 않고, 같은 조건으로 돌린 반복 계산의 차이가 "
         "<b>우연으로 보기 어려울 만큼 일관된지</b>(95% 신뢰구간)까지 확인합니다.")

    section("② 넘으면 안 되는 한도가 있나요",
            "예산이나 채용 계획상 받아들일 수 없는 선이 있다면 넣어주세요. 현행 유지와 비교한 값입니다. 없으면 비워두면 됩니다.")
    c = st.columns(3)
    def limit(col, key, lab, default, step, hlp):
        on = col.checkbox(lab, value=cfg[key] is not None, key=f"w_on_{key}")
        v = col.number_input(lab, 0.0, 100.0, float(cfg[key] if cfg[key] is not None else default), step,
                             key=f"w_{key}", label_visibility="collapsed", disabled=not on, help=hlp)
        cfg[key] = float(v) if on else None
        return cfg[key]
    lc = limit(c[0], "lim_cost", "인건비 증가 한도 (%)", 1.0, 0.1, "현행 유지보다 인건비가 이 비율 넘게 늘면 제외합니다.")
    lh = limit(c[1], "lim_hire", "신규채용 감소 한도 (%)", 10.0, 1.0, "현행 유지보다 신규채용이 이 비율 넘게 줄면 제외합니다.")
    lw = limit(c[2], "lim_wait", "승진 대기 증가 한도 (년)", 1.0, 0.1, "현행 유지보다 승진 대기가 이 햇수 넘게 늘면 제외합니다.")
    viol = D.apply_constraints(results, base,
                               None if lc is None else lc / 100, None if lh is None else lh / 100, lw)

    section("③ 무엇이 더 중요한가요",
            "항목마다 중요도를 정하면, 후보 가운데 우리 회사 우선순위에 가장 잘 맞는 순서로 줄을 세웁니다.")
    keys = list(D.OBJECTIVES)
    cols = [c for _ in range((len(keys) + 2) // 3) for c in st.columns(3)]
    w = {}
    for col, k in zip(cols, keys):
        cur = int(cfg["weights"].get(k, 0))
        w[k] = col.select_slider(OBJ_NAMES[k], options=list(LEVELS), value=cur,
                                 format_func=lambda v: LEVELS[v], key=f"w_wt_{k}")
    cfg["weights"] = w
    if sum(w.values()) == 0:
        st.info("중요도를 하나 이상 '보통' 이상으로 올려주세요.")
        return

    feasible = [n for n in j["frontier"] if not viol[n]]
    pr = D.priority_scores(results, w, names=feasible) if feasible else {}
    ranked = sorted(((n, v["score"]) for n, v in pr.items()), key=lambda x: x[1])
    n_fail = sum(1 for n in j["frontier"] if viol[n])
    gap(6)
    html(f'<div class="card"><div class="card-t">우리 회사 조건에 맞춘 결과</div>'
         f'<div class="card-b">{I.recommend_text(ranked, w, n_fail)}</div></div>')

    if ranked:
        gap(10)
        fig = go.Figure()
        order = [n for n, _ in ranked]
        wsum = sum(w.values())
        for k in D.OBJECTIVES:
            if w[k] <= 0:
                continue
            fig.add_bar(y=[label(n) for n in order],
                        x=[100 * w[k] / wsum * pr[n]["parts"][k] for n in order],
                        name=OBJ_SHORT[k], orientation="h", marker_color=OBJ_COLOR[k],
                        hovertemplate="%{y}<br>" + OBJ_SHORT[k] + "에서 아쉬운 정도 %{x:.1f}점<extra></extra>")
        base_layout(fig, 90 + 44 * len(order), barmode="stack", bargap=0.35,
                    xaxis_title="아쉬움 점수 (0 = 모든 항목에서 최선, 낮을수록 좋음)",
                    legend=dict(orientation="h", y=1.15, x=0))
        fig.update_traces(marker_line=dict(color="white", width=2))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(range=[0, max(max(s for _, s in ranked) * 100 * 1.15, 10)])
        show(fig, "priority", bar=False)
        note("막대가 짧을수록 우리 회사 우선순위에 잘 맞습니다. 색깔 조각은 '어느 항목에서 최선의 시나리오보다 "
             "얼마나 아쉬운지'를 뜻합니다 — 조각이 없으면 그 항목에서는 이 시나리오가 최선입니다.<br>"
             "이 점수는 <b>지금 비교한 시나리오들 사이의 상대 순위</b>입니다. 중요도를 바꾸면 순위도 바뀌므로, "
             "숫자 자체보다 '왜 이 순서인가'를 보는 용도로 쓰세요.", "fine")

    rows = []
    rank_of = {n: i + 1 for i, (n, _) in enumerate(ranked)}
    for r in results:
        n = r["summary"]["scenario"]
        if n in j["dominated"]:
            state, reason = "제외", I.why_text(n, j["why"])
        elif viol[n]:
            state, reason = "한도 초과", " / ".join(viol[n])
        else:
            state, reason = "후보", ""
        rows.append({"_r": rank_of.get(n, 999), "순위": str(rank_of[n]) if n in rank_of else "–",
                     "시나리오": n, "상태": state,
                     "아쉬움 점수": f"{pr[n]['score']*100:.1f}" if n in pr else "–",
                     "판정 안정성": j["stability"].get(n, np.nan) * 100, "사유": reason})
    df = pd.DataFrame(rows).sort_values("_r").drop(columns="_r")
    st.dataframe(df, width="stretch", hide_index=True,
                 column_config={"순위": st.column_config.TextColumn(width="small"),
                                "아쉬움 점수": st.column_config.TextColumn(width="small"),
                                "판정 안정성": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100, width="small"),
                                "사유": st.column_config.TextColumn(width="large")})

    _pairwise(results, objs, mat)


def _pairwise(results, objs, mat):
    section("두 시나리오 맞비교",
            "두 시나리오의 차이가 <b>진짜 차이인지, 계산할 때마다 달라지는 우연인지</b>를 항목별로 확인합니다.")
    names = [r["summary"]["scenario"] for r in results]
    c = st.columns(2)
    a = c[0].selectbox("시나리오 A", names, index=min(1, len(names) - 1), key="w_pair_a", format_func=label)
    b = c[1].selectbox("시나리오 B", names, index=0, key="w_pair_b", format_func=label)
    if a == b:
        note("서로 다른 두 시나리오를 골라주세요.", "fine")
        return
    ra = next(r for r in results if r["summary"]["scenario"] == a)
    rb = next(r for r in results if r["summary"]["scenario"] == b)
    cmp = D.compare(ra, rb, list(D.OBJECTIVES), mat / 100.0)
    rows = []
    for k, v in cmp.items():
        o = D.OBJECTIVES[k]
        va, vb = ra["summary"][f"{k}_mean"] * o["scale"], rb["summary"][f"{k}_mean"] * o["scale"]
        diff = (va - vb)
        lo, hi = sorted([o["sign"] * v["lo"] * o["scale"], o["sign"] * v["hi"] * o["scale"]])
        verdict = {"better": f"{short(a)}가 낫다", "worse": f"{short(b)}가 낫다", "same": "구별되지 않음"}[v["verdict"]]
        rows.append({"항목": f"{o['label']} ({o['unit']})", short(a): va, short(b): vb, "차이 (A−B)": diff,
                     "차이의 95% 범위": f"{lo:+,.1f} ~ {hi:+,.1f}", "판정": verdict,
                     "판정에 사용": "✓" if k in objs else ""})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                 column_config={short(a): st.column_config.NumberColumn(format="%,.1f"),
                                short(b): st.column_config.NumberColumn(format="%,.1f"),
                                "차이 (A−B)": st.column_config.NumberColumn(format="%+,.1f")})
    note("'차이의 95% 범위'가 0을 포함하면 그 차이는 우연과 구별되지 않습니다. 0을 포함하지 않아도 차이가 "
         f"{mat:.1f}% 미만이면 '구별되지 않음'으로 봅니다(위 ①에서 조정).", "fine")


# ═════════════════ ⑤ 결론이 흔들리는가 ═════════════════
def render_robust(res, cfg, stale):
    if stale:
        st.warning("설정이 바뀌었습니다. 아래 분석은 ② 단계에서 다시 실행한 뒤에 돌려주세요.")
    note("지금까지의 결과는 여러 <b>가정</b> 위에 서 있습니다. 가정이 틀려도 결론이 유지되는지를 네 가지 방법으로 확인합니다.<br>"
         "각 분석은 버튼을 눌러야 계산되고, 계산 시간을 줄이려고 반복 횟수를 낮춰 돌립니다.")
    t = st.tabs(["재고용 조건이 다르면", "큰 가정이 다르면", "어떤 가정이 가장 흔드나", "정책 공간 탐색"])
    with t[0]:
        _sens(res, cfg)
    with t[1]:
        _robust(res, cfg)
    with t[2]:
        _tornado(res, cfg)
    with t[3]:
        _space(res, cfg)


def _progress(label0):
    bar = st.progress(0.0, text=label0)
    return bar, (lambda f, name: bar.progress(min(float(f), 1.0), text=f"{name}"))


def _judge(res):
    return st.session_state.get("judge") or res["out"]


def _sens(res, cfg):
    note("재고용 <b>수락률</b>과 <b>임금률</b>은 회사가 미리 알기 가장 어려운 값입니다. 두 값을 5단계씩 바꿔 "
         "25가지 조합에서 후보가 그대로인지 확인합니다.")
    if st.button(f"25가지 조합 계산하기 (약 {max(int(len(res['scenarios']) * 3.5 * res['horizon'] / 12), 10)}초)", key="go_sens"):
        bar, cb = _progress("계산을 시작합니다")
        st.session_state["sens"] = sensitivity(
            res["profile"], runs=30, horizon=res["horizon"], hiring=cfg["hiring"], scenarios=res["scenarios"],
            objectives=cfg["objectives"], materiality=cfg["materiality"] / 100.0,
            base=(cfg["accept"], cfg["wage_ratio"]), progress=cb)
        bar.empty()
    sens = st.session_state.get("sens")
    if not sens:
        return
    accepts = sorted({r["accept"] for r in sens["grid"]})
    wages = sorted({r["wage_ratio"] for r in sens["grid"]})
    cell = {(r["accept"], r["wage_ratio"]): r["frontier"] for r in sens["grid"]}
    ref = set(sens["baseline_frontier"])
    z = [[len(set(cell[(a, w)]) ^ ref) for w in wages] for a in accepts]
    hover = [[", ".join(short(x) for x in cell[(a, w)]) for w in wages] for a in accepts]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{w:.2f}" for w in wages], y=[f"{a:.2f}" for a in accepts], customdata=hover,
        text=z, texttemplate="%{text}", textfont=dict(size=12),
        hovertemplate="수락률 %{y} · 임금률 %{x}<br>후보: %{customdata}<br>기준과 달라진 시나리오 %{z}개<extra></extra>",
        colorscale=SEQ, zmin=0, zmax=max(1, max(v for row in z for v in row)), xgap=3, ygap=3,
        colorbar=dict(title="달라진<br>개수", tickfont=dict(size=10), dtick=1, len=0.8)))
    na, nw = sens["baseline_point"]
    fig.add_trace(go.Scatter(x=[f"{nw:.2f}"], y=[f"{na:.2f}"], mode="markers",
                             marker=dict(symbol="star", size=20, color="#eb6834", line=dict(color="#fff", width=2)),
                             showlegend=False, hoverinfo="skip"))
    base_layout(fig, 400, xaxis_title="재고용 임금률 →", margin=dict(t=20, b=10, l=10, r=10))
    fig.update_xaxes(type="category", showgrid=False)
    fig.update_yaxes(type="category", showgrid=False)
    html(f'<div class="note" style="margin-bottom:2px"><b>읽는 법</b> — 칸 안의 숫자는 그 조건에서 '
         f'<b>후보 판정이 달라진 시나리오 수</b>입니다(0 = 지금과 같음). 세로는 재고용 수락률, 가로는 임금률, '
         f'★은 지금 입력값에 가장 가까운 칸입니다.</div>')
    show(fig, "sens", bar=False)
    n_grid, n_flip = len(sens["grid"]), len(sens["flip_points"])
    if n_flip == 0:
        note(f"<b>{n_grid}가지 조합 모두에서 후보가 바뀌지 않았습니다.</b> 재고용 조건을 정확히 몰라도 지금의 결론을 믿어도 됩니다.")
    else:
        share = n_flip / n_grid
        tone = ("재고용 조건에 따라 결론이 <b>자주 바뀝니다.</b> 실제 수락률을 사내 설문 등으로 먼저 확인해야 합니다." if share > 0.5
                else "재고용 조건이 크게 달라지면 결론도 <b>일부 바뀝니다.</b>")
        note(f"<b>{n_grid}가지 중 {n_flip}가지</b>에서 후보가 달라졌습니다. {tone}")
        groups = {}
        for a, w, f in sens["flip_points"]:
            groups.setdefault(tuple(f), []).append((a, w))
        rows = []
        for f, pts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            acc = [a for a, _ in pts]; wg = [w for _, w in pts]
            gained = sorted(set(f) - ref, key=short); lost = sorted(ref - set(f), key=short)
            rows.append({"이때의 후보": ", ".join(short(x) for x in f),
                         "새로 후보가 됨": ", ".join(short(x) for x in gained),
                         "후보에서 빠짐": ", ".join(short(x) for x in lost),
                         "해당 조합 수": len(pts),
                         "수락률 범위": f"{min(acc):.2f} ~ {max(acc):.2f}",
                         "임금률 범위": f"{min(wg):.2f} ~ {max(wg):.2f}"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _robust(res, cfg):
    note("결과에 가장 큰 영향을 주는 세 가지 가정 — <b>50세 이후 임금</b>(3가지) × <b>퇴사율</b>(낮음·보통·높음) × "
         "<b>채용 방식</b>(3가지) — 을 모두 조합한 <b>27가지 상황</b>에서 다시 판정합니다.<br>"
         "어떤 상황에서도 후보로 남는 시나리오가 '가정이 틀려도 후회가 적은 선택'입니다.")
    est = max(int(27 * len(res["scenarios"]) * 30 * res["horizon"] * max(res["out"]["results"][0]["summary"]["n_initial"], 300) / 1000 * 0.0008), 15)
    if st.button(f"27가지 상황 계산하기 (약 {est}초)", key="go_robust"):
        bar, cb = _progress("계산을 시작합니다")
        st.session_state["robust"] = robustness(
            res["profile"], res["scenarios"], hiring=cfg["hiring"], runs=30, horizon=res["horizon"],
            objectives=cfg["objectives"], materiality=cfg["materiality"] / 100.0,
            weights={k: v for k, v in cfg["weights"].items() if v > 0} or None, progress=cb)
        bar.empty()
    rb = st.session_state.get("robust")
    if not rb:
        return
    names = [s.name for s in res["scenarios"]]
    share, reg, N = rb["frontier_share"], rb["regret"], rb["n_states"]
    wins = {n: reg.get(n, {}).get("n_best", 0) for n in names}
    order = sorted(names, key=lambda n: (-wins[n], reg.get(n, {}).get("max_regret", 9)))
    fig = go.Figure(go.Bar(
        y=[label(n) for n in order], x=[wins[n] for n in order], orientation="h",
        marker_color=["#256abf" if wins[n] >= N * 0.5 else ("#6da7ec" if wins[n] > 0 else "#c9d1d7") for n in order],
        marker_cornerradius=4, text=[f"{wins[n]}번" for n in order], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>" + str(N) + "가지 상황 중 %{x}번 1위<extra></extra>"))
    base_layout(fig, 90 + 40 * len(order), unit=f"{N}가지 상황 가운데 우리 회사 우선순위로 1위를 한 횟수",
                bargap=0.35, margin=dict(t=34, b=10, l=10, r=50))
    fig.update_xaxes(range=[0, N * 1.12], dtick=3)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    show(fig, "robust", bar=False)
    top = order[0]
    msg = []
    if wins[top] == N:
        msg.append(f"<b>{label(top)}</b>는 {N}가지 상황 <b>전부</b>에서 1위였습니다. 임금·퇴사율·채용 방식 가정이 "
                   "틀리더라도 ④에서 정한 우선순위로는 같은 결론이 나온다는 뜻입니다.")
    elif wins[top] >= N * 0.6:
        msg.append(f"<b>{label(top)}</b>가 {N}가지 중 <b>{wins[top]}번</b> 1위였습니다. 대부분의 상황에서 같은 결론이지만, "
                   "아래 '전체 보기'에서 순위가 바뀌는 상황이 우리 회사에 해당하는지 확인해 보세요.")
    else:
        msg.append(f"1위가 상황마다 갈립니다(가장 많은 {label(top)}도 {N}가지 중 {wins[top]}번). <b>가정에 따라 결론이 달라지는 "
                   "상태</b>이므로, 다음 탭에서 어떤 가정이 결과를 가장 흔드는지 확인하고 그 값부터 실제 수치로 채워야 합니다.")
    never = [n for n in names if share[n] <= 0.001]
    shaky = [n for n in names if 0.001 < share[n] < 0.999]
    if shaky:
        msg.append(f"{', '.join(short(n) for n in shaky)}는 상황에 따라 후보에서 빠지기도 합니다.")
    if never:
        msg.append(f"{', '.join(short(n) for n in never)}는 어떤 상황에서도 후보가 되지 못했습니다.")
    note("<br>".join(msg))
    lever = {s.name for s in res["scenarios"] if s.wage_peak or s.wage_reform_regime or s.erp_take_rate > 0}
    if top in lever:
        note("주의: 1위인 안은 임금피크·임금체계 개편·희망퇴직 같은 수단을 쓴 안입니다. 이런 수단은 줄어드는 인건비만 "
             "계산에 들어가고 노사 협의·사기 저하·법적 분쟁 비용은 빠져 있어, 실제보다 유리하게 나옵니다. "
             "수단을 쓰지 않은 안들 사이의 순위도 함께 보세요.", "fine")
    if reg:
        best = min(reg, key=lambda n: reg[n]["max_regret"])
        t = pd.DataFrame([{"시나리오": n, "1위 횟수": f"{wins[n]} / {N}",
                           "후보로 남은 상황": f"{share[n]*N:.0f} / {N}",
                           "최악의 경우 아쉬움": reg[n]["max_regret"] * 100,
                           "평균 아쉬움": reg[n]["mean_regret"] * 100} for n in order if n in reg])
        st.dataframe(t, width="stretch", hide_index=True,
                     column_config={"최악의 경우 아쉬움": st.column_config.NumberColumn(format="%.1f", help="각 상황에서 1위와의 점수 차를 구했을 때 가장 크게 벌어진 값입니다. 작을수록 '어떤 상황이 와도 크게 후회하지 않는' 선택입니다."),
                                    "평균 아쉬움": st.column_config.NumberColumn(format="%.1f")})
        note(f"'최악의 경우 아쉬움'이 가장 작은 것은 <b>{label(best)}</b>입니다. 1위 횟수가 많은 안과 다르다면, "
             "'대체로 최선인 안'과 '최악을 피하는 안' 사이에서 회사의 위험 성향에 따라 고르면 됩니다.", "fine")
    if st.toggle("27가지 상황별 후보 전체 보기", key="w_rb_detail"):
        rows = []
        for s in rb["states"]:
            row = {"50세 이후 임금": s["regime"], "퇴사율": s["turnover"], "채용 방식": s["hiring"],
                   "지금 설정": "★" if s["is_base"] else ""}
            for n in names:
                row[short(n)] = "●" if n in s["frontier"] else ""
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=420)


def _tornado(res, cfg):
    others = [s for s in res["scenarios"][1:]]
    if not others:
        note("현행 유지 외에 비교할 시나리오가 없습니다.")
        return
    note("가정을 <b>하나씩만</b> 바꿨을 때, 고른 시나리오와 현행 유지 사이의 격차가 얼마나 움직이는지 봅니다.<br>"
         "두 점 사이가 먼 가정일수록 결과를 크게 흔드는 가정이므로, 그 값부터 우리 회사 실제 수치로 확인해야 합니다.")
    names = [s.name for s in others]
    j = _judge(res)
    default = next((n for n in names if n in j["frontier"]), names[0])
    pick = st.selectbox("살펴볼 시나리오", names, index=names.index(default), key="w_tor_pick", format_func=label)
    if st.button("가정별 영향 계산하기 (약 20초)", key="go_tor"):
        bar, cb = _progress("계산을 시작합니다")
        scen = next(s for s in others if s.name == pick)
        st.session_state["tornado"] = tornado(res["profile"], scen, res["scenarios"][0], hiring=cfg["hiring"],
                                              accept=cfg["accept"], wage_ratio=cfg["wage_ratio"], runs=30,
                                              horizon=res["horizon"], progress=cb)
        bar.empty()
    tor = st.session_state.get("tornado")
    if not tor:
        return
    if tor["scenario"] != pick:
        note(f"아래는 {label(tor['scenario'])}의 결과입니다. 다른 시나리오를 보려면 다시 계산해 주세요.", "fine")
    c = st.columns(2)
    for col, key, title in ((c[0], "cost", "인건비 격차 (현행 대비 %)"), (c[1], "hires", "신규채용 격차 (현행 대비 %)")):
        rows = sorted(tor["rows"], key=lambda r: abs(r[f"{key}_high"] - r[f"{key}_low"]))
        b0 = tor["base"][key]
        fig = go.Figure()
        ys = [r["param"] for r in rows]
        for r in rows:      # 낮게~높게 사이를 잇는 선
            fig.add_trace(go.Scatter(x=[r[f"{key}_low"], r[f"{key}_high"]], y=[r["param"]] * 2, mode="lines",
                                     line=dict(color="#c9d1d7", width=6), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[r[f"{key}_low"] for r in rows], y=ys, mode="markers", name="낮게 잡으면",
            marker=dict(size=13, color=LOW_C, line=dict(color="white", width=2)),
            customdata=[r["low_label"] for r in rows],
            hovertemplate="%{y} = %{customdata}<br>격차 %{x:+.2f}%<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[r[f"{key}_high"] for r in rows], y=ys, mode="markers", name="높게 잡으면",
            marker=dict(size=13, color=HIGH_C, line=dict(color="white", width=2)),
            customdata=[r["high_label"] for r in rows],
            hovertemplate="%{y} = %{customdata}<br>격차 %{x:+.2f}%<extra></extra>"))
        fig.add_vline(x=b0, line=dict(color=INK, width=1.5, dash="dot"))
        base_layout(fig, 330, unit=title, legend=dict(orientation="h", y=-0.18))
        fig.update_xaxes(ticksuffix="%")
        fig.update_yaxes(showgrid=False)
        with col:
            show(fig, f"tor_{key}", bar=False)
    top_c = max(tor["rows"], key=lambda r: abs(r["cost_high"] - r["cost_low"]))
    top_h = max(tor["rows"], key=lambda r: abs(r["hires_high"] - r["hires_low"]))
    note(f"파란 점은 그 가정을 낮게(임금은 '동결'), 주황 점은 높게(임금은 '호봉제') 잡았을 때입니다. "
         f"세로 점선은 지금 가정에서의 격차입니다(인건비 {tor['base']['cost']:+.2f}%, 신규채용 {tor['base']['hires']:+.1f}%).<br>"
         f"인건비 격차를 가장 크게 흔드는 가정은 <b>{top_c['param']}</b>"
         f"({top_c['cost_low']:+.2f}% ~ {top_c['cost_high']:+.2f}%), "
         f"신규채용 격차를 가장 크게 흔드는 가정은 <b>{top_h['param']}</b>"
         f"({top_h['hires_low']:+.1f}% ~ {top_h['hires_high']:+.1f}%)입니다. "
         "이 두 값을 우리 회사 실제 수치로 확인하는 것이 결과의 신뢰도를 가장 크게 높입니다.")


KIND_COLOR = {"현행 유지": INK, "정년연장만": SERIES[0], "정년연장+재고용": SERIES[1], "재고용만": SERIES[2]}


def _space(res, cfg):
    note("공개된 안에 얽매이지 않고 <b>도달 나이(62~65세) × 시작 연도(2028~2030) × 올리는 간격(1~3년) × "
         "방식(정년연장만 / 정년연장+재고용 / 재고용만)</b>을 모두 조합한 <b>108가지 설계</b>를 한꺼번에 계산합니다.<br>"
         "우리 회사에 유리한 설계가 어느 쪽에 몰려 있는지 지도처럼 볼 수 있습니다.")
    n0 = max(res["out"]["results"][0]["summary"]["n_initial"], 300)
    est = max(int(109 * 20 * res["horizon"] * n0 / 1000 * 0.0008), 20)
    if st.button(f"108가지 설계 계산하기 (약 {est}초)", key="go_space"):
        bar, cb = _progress("계산을 시작합니다")
        st.session_state["space"] = design_space(
            res["profile"], hiring=cfg["hiring"], accept=cfg["accept"], wage_ratio=cfg["wage_ratio"],
            runs=20, horizon=res["horizon"], first_years=tuple(res["profile"].start_year + k for k in (2, 3, 4)),
            objectives=cfg["objectives"], materiality=cfg["materiality"] / 100.0, progress=cb)
        bar.empty()
    sp = st.session_state.get("space")
    if not sp:
        return
    df = pd.DataFrame(sp["rows"])
    fig = go.Figure()
    for kind, color in KIND_COLOR.items():
        for on in (False, True):
            sub = df[(df["kind"] == kind) & (df["on_frontier"] == on)]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["cost_pct"], y=sub["hires_pct"], mode="markers",
                name=kind + (" · 후보" if on else " · 제외"), legendgroup=kind, showlegend=on or kind == "현행 유지",
                marker=dict(size=13 if on else 7, color=color, opacity=1.0 if on else 0.3,
                            symbol="diamond" if kind == "현행 유지" else "circle",
                            line=dict(color="white", width=1.5 if on else 0)),
                customdata=sub[["name", "wait", "income_gap"]].to_numpy(),
                hovertemplate="%{customdata[0]}<br>인건비 %{x:+.2f}% · 신규채용 %{y:+.1f}%<br>"
                              "승진 대기 %{customdata[1]:.1f}년 · 소득 공백 %{customdata[2]:,.0f}인년<extra></extra>"))
    base_layout(fig, 480, unit="신규채용 (현행 대비 %)", xaxis_title="인건비 (현행 대비 %)",
                legend=dict(orientation="h", y=-0.16))
    fig.update_xaxes(ticksuffix="%"); fig.update_yaxes(ticksuffix="%")
    show(fig, "space")
    front = df[df["on_frontier"]]
    by_kind = front["kind"].value_counts()
    tot_kind = df["kind"].value_counts()
    parts = [f"{k} {by_kind.get(k, 0)}/{tot_kind.get(k, 0)}" for k in ("정년연장만", "정년연장+재고용", "재고용만")]
    note(f"큰 점이 후보, 작고 옅은 점이 제외된 설계입니다. 왼쪽 위(비용은 적고 채용은 많은 쪽)일수록 회사에 유리합니다.<br>"
         f"109개 설계 중 <b>{len(front)}개</b>가 후보로 남았습니다 — {' · '.join(parts)}. "
         "후보가 한 방식에 몰려 있다면 그 방식이 우리 회사 구조에 구조적으로 유리하다는 신호입니다.")

    w = {k: v for k, v in cfg["weights"].items() if v > 0}
    cand = front[front["kind"] != "현행 유지"].copy()
    if cand.empty:
        return
    col_of = {"cost_npv": ("cost_eok", -1), "total_hires": ("hires", 1), "promo_congestion": ("wait", -1),
              "exp55_person_years": ("exp55", 1), "income_gap": ("income_gap", -1),
              "skill_gap": ("skill_gap", -1)}
    score = np.zeros(len(cand))
    for k, wt in w.items():
        col, sign = col_of[k]
        v = sign * cand[col].to_numpy(float)
        span = v.max() - v.min()
        score += (wt / sum(w.values())) * ((v.max() - v) / span if span > 1e-12 else 0)
    cand["아쉬움 점수"] = score * 100
    cand["_k"] = (cand["cost_pct"].round(2).astype(str) + "/" + cand["hires_pct"].round(1).astype(str)
                  + "/" + cand["wait"].round(1).astype(str))
    top = cand.sort_values("아쉬움 점수").drop_duplicates("_k").head(10)
    html('<div class="card-t" style="margin-top:14px">④에서 정한 중요도에 가장 잘 맞는 설계 10개</div>')
    st.dataframe(
        top[["name", "아쉬움 점수", "cost_pct", "hires_pct", "wait", "skill_gap", "income_gap"]].rename(
            columns={"name": "설계", "cost_pct": "인건비(현행 대비)", "hires_pct": "신규채용(현행 대비)",
                     "wait": "승진 대기(년)", "skill_gap": "숙련 공백(인년)", "income_gap": "소득 공백(인년)"}),
        width="stretch", hide_index=True,
        column_config={"아쉬움 점수": st.column_config.NumberColumn(format="%.1f"),
                       "인건비(현행 대비)": st.column_config.NumberColumn(format="%+.2f%%"),
                       "신규채용(현행 대비)": st.column_config.NumberColumn(format="%+.1f%%"),
                       "승진 대기(년)": st.column_config.NumberColumn(format="%.1f"),
                       "숙련 공백(인년)": st.column_config.NumberColumn(format="%,.0f"),
                       "소득 공백(인년)": st.column_config.NumberColumn(format="%,.0f")})
    c = st.columns([3, 1])
    pick = c[0].selectbox("내 시나리오로 가져올 설계", list(top["name"]), key="w_space_pick", label_visibility="collapsed")
    if c[1].button("이 설계를 내 시나리오에 추가", width="stretch"):
        spec = dict(next(r["spec"] for r in sp["rows"] if r["name"] == pick))
        n = 1 + max([int(x["name"].split(" ")[0][1:]) for x in cfg["custom"]] or [0])
        spec["name"] = f"C{n} {pick}"
        spec["desc"], spec["source"] = "정책 공간 탐색에서 가져온 설계", "사용자 정의"
        cfg["custom"].append(spec)
        st.success(f"'{spec['name']}'을(를) 추가했습니다. ② 단계에서 다시 실행하면 정밀 계산 결과에 포함됩니다.")
    note("탐색은 반복 20회의 빠른 계산이라 순위가 근소한 설계끼리는 뒤바뀔 수 있습니다. "
         "마음에 드는 설계는 내 시나리오로 가져가 정밀하게 다시 계산하세요.", "fine")
