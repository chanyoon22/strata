"""
dss_layer.py — DSS 층: 시나리오 정의·실행, 민감도·견고성 분석, 정책 공간 탐색
====================================================================================
원칙: 권장 후보는 결정론적 알고리즘(decision.py의 통계적 지배 필터)이 도출한다.
      LLM은 그 결과 수치의 해석 산문만 생성한다(llm_report.py).

시나리오 프리셋의 근거 (2026-11-16 웹 재검증, 출처는 design_doc.md 15.2)
  정년연장 입법은 2026년 11월 현재 **국회를 통과하지 않았다**(개정안 다수 소위 계류,
  여당은 연내 처리를 촉구 중). 따라서 "어느 안이 될지 모르는 상태에서 미리 비교한다"는
  이 도구의 전제는 그대로 유효하다. 프리셋은 실제로 공개된 안에 1:1로 대응시킨다.

  S1 현행 유지            정년 60세 + 회사의 현재 재고용 관행 그대로
  S2 여당 초안            2029년 61세, 2년마다 +1세 → 2037년 65세. 재고용 의무는 정년보다
                          1세 위까지(2028년 61세부터). 더불어민주당 정년연장특위 초안(2026.6.11 보도)
  S3 빠른 연장(1안)       2028년 61세, 2년마다 +1세 → 2036년 65세 (특위 2025.12 제시 1안)
  S4 절충안(2안)          2029·2032·2035·2037·2039년에 +1세 → 2039년 65세 (같은 자리 2안)
  S5 완만한 연장(3안)     2029년부터 3년마다 +1세 → 2041년 65세 (같은 자리 3안)
  S6 재고용 중심          정년 60세 유지 + 계속고용 의무 2028년 62세 → 2030년 63세 →
                          2032년 64세 → 2033년 65세 (경사노위 공익위원 제언, 2025.5.8)
  S7 여당 초안+임금피크   S2에 정년 전 3년간 연 10%p 감액을 병행(회사 선택 수단)
  S8 재고용-직무유지형    S6을 직무·직위 유지형으로(경사노위 제언의 기본형; 임금 85% 이상)
  S9 연금 수급연령 연동   정년을 국민연금 수급 개시 연령에 맞춤: 2027년 63세 → 2028년 64세 →
                          2033년 65세 (노동계 요구 방향)
  S10 여당 초안+임금개편  S2에 2029년부터 '50세 이후 임금 동결'을 병행(직무급 전환의 근사)

  v2의 "2036년 단계적 연장"·"2032년 급진연장(정흥준안)"은 각각 S3·S9로 대체했다.
  여당 초안이 2037년 도달로 정리되면서 가장 유력한 안이 프리셋에 없던 상태였다.
"""
from __future__ import annotations
import dataclasses
import itertools
import numpy as np
from engine_retirement_v2 import CompanyProfile, Scenario, monte_carlo, WAGE_REGIMES
import decision as D

# ── 정년 일정 ──
SCHED_RULING = {2026: 60, 2029: 61, 2031: 62, 2033: 63, 2035: 64, 2037: 65}
CE_RULING = {2026: 60, 2028: 61, 2029: 62, 2031: 63, 2033: 64, 2035: 65}
SCHED_FAST = {2026: 60, 2028: 61, 2030: 62, 2032: 63, 2034: 64, 2036: 65}
SCHED_MID = {2026: 60, 2029: 61, 2032: 62, 2035: 63, 2037: 64, 2039: 65}
SCHED_SLOW = {2026: 60, 2029: 61, 2032: 62, 2035: 63, 2038: 64, 2041: 65}
SCHED_PENSION = {2026: 60, 2027: 63, 2028: 64, 2033: 65}
CE_ESLC = {2026: 60, 2028: 62, 2030: 63, 2032: 64, 2033: 65}


def _plus_one(sched, cap=65):
    """정년보다 1세 위까지 재고용(특위 안의 공통 병행 방식). 65세가 상한."""
    return {y: min(a + 1, cap) if a > 60 else a for y, a in sched.items()}


# 채용 정책 — 정년연장의 부담을 무엇으로 흡수할 것인가.
HIRING_POLICIES = {
    "replace": ("정원 유지", "나간 만큼만 새로 뽑아 인원을 유지합니다. 고령자가 남으면 그만큼 "
                            "퇴직이 줄어 신규채용이 자동으로 감소합니다."),
    "budget":  ("인건비 총액 유지", "인건비 예산을 지금 수준으로 묶습니다. 고령자 인건비가 "
                                 "늘면 그만큼 신규채용을 더 줄여 흡수합니다."),
    "fixed":   ("채용 규모 고정", "매년 같은 인원을 뽑습니다. 정년연장의 부담이 채용이 아니라 "
                               "인원·인건비 증가로 나타납니다."),
}

DEFAULT_PRESET_CODES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]


def preset_scenarios(rehire_wage=0.55, rehire_accept=0.62, hiring="replace", hiring_fixed_n=0,
                     codes=None):
    """공개된 안에 대응하는 프리셋 시나리오. codes로 일부만 고를 수 있다."""
    def S(name, **kw):
        return Scenario(name, hiring_mode=hiring, hiring_fixed_n=hiring_fixed_n, **kw)
    ce = dict(continued_type="rehire", rehire_wage_ratio=rehire_wage, rehire_accept=rehire_accept)
    all_ = [
        S("S1 현행 유지(정년 60세)",
          desc="법이 바뀌지 않고, 회사가 지금 하던 방식 그대로 가는 경우입니다.",
          source="현행 고령자고용법 제19조"),
        S("S2 여당 초안(2037년 65세)", retirement_schedule=SCHED_RULING,
          ce_end_age_schedule=CE_RULING, **ce,
          desc="2029년 61세에서 시작해 2년마다 한 살씩, 2037년에 65세가 됩니다. "
               "정년보다 한 살 위까지는 재고용 의무가 붙습니다.",
          source="더불어민주당 정년연장특위 초안(2026.6.11 보도)"),
        S("S3 빠른 연장(2036년 65세)", retirement_schedule=SCHED_FAST,
          ce_end_age_schedule=_plus_one(SCHED_FAST), **ce,
          desc="2028년 61세에서 시작해 2년마다 한 살씩, 2036년에 65세가 됩니다.",
          source="정년연장특위 제시 1안(2025.12)"),
        S("S4 절충안(2039년 65세)", retirement_schedule=SCHED_MID,
          ce_end_age_schedule=_plus_one(SCHED_MID), **ce,
          desc="2029년 61세에서 시작해 처음엔 3년, 나중엔 2년 간격으로 올려 2039년에 65세가 됩니다.",
          source="정년연장특위 제시 2안(2025.12)"),
        S("S5 완만한 연장(2041년 65세)", retirement_schedule=SCHED_SLOW,
          ce_end_age_schedule=_plus_one(SCHED_SLOW), **ce,
          desc="2029년 61세에서 시작해 3년마다 한 살씩, 2041년에 65세가 됩니다.",
          source="정년연장특위 제시 3안(2025.12)"),
        S("S6 재고용 중심(정년 60세+계속고용 65세)", retirement_schedule={2026: 60},
          ce_end_age_schedule=CE_ESLC, **ce,
          desc="정년은 60세 그대로 두고, 퇴직자를 다시 고용할 의무만 2033년까지 65세로 넓힙니다.",
          source="경사노위 공익위원 제언(2025.5.8)"),
        S("S7 여당 초안+임금피크", retirement_schedule=SCHED_RULING,
          ce_end_age_schedule=CE_RULING, **ce, wage_peak=True, peak_years=3, peak_cut=0.10,
          desc="여당 초안에, 정년 전 3년 동안 임금을 해마다 10%p씩 줄이는 임금피크제를 더한 경우입니다.",
          source="S2 + 회사 선택 수단"),
        S("S8 재고용-직무유지형", retirement_schedule={2026: 60},
          continued_type="job_keep", ce_end_age_schedule=CE_ESLC,
          rehire_wage_ratio=0.85, rehire_accept=min(rehire_accept + 0.15, 0.95),
          desc="재고용하되 하던 일과 자리를 그대로 유지합니다. 임금은 85% 이상, 승진 자리는 비지 않습니다.",
          source="경사노위 제언의 기본형(직무유지형)"),
        S("S9 연금 연동(2033년 65세)", retirement_schedule=SCHED_PENSION,
          desc="정년을 국민연금 받는 나이에 맞춥니다. 2027년 63세, 2028년 64세, 2033년 65세.",
          source="노동계 요구 방향 · 국민연금 수급연령 일정"),
        S("S10 여당 초안+임금체계 개편", retirement_schedule=SCHED_RULING,
          ce_end_age_schedule=CE_RULING, **ce,
          wage_reform_regime="frozen", wage_reform_year=2029,
          desc="여당 초안에, 2029년부터 50세 이후 임금 인상을 멈추는 임금체계 개편을 더한 경우입니다.",
          source="S2 + 회사 선택 수단(직무급 전환의 근사)"),
    ]
    if codes is None:
        return all_
    return [s for s in all_ if s.name.split(" ")[0] in codes]


# ── 사용자 정의 시나리오 ────────────────────────────────────────────────────
def build_schedule(first_year, step_years, target_age, base_age=60, base_year=2026):
    """'first_year에 +1세, 이후 step_years마다 +1세, target_age에서 멈춤' 형태의 정년 일정."""
    target_age, base_age = int(target_age), int(base_age)
    sched = {int(base_year): base_age}
    if target_age <= base_age:
        return sched
    step_years = max(int(step_years), 1)
    for i, age in enumerate(range(base_age + 1, target_age + 1)):
        sched[int(first_year) + i * step_years] = age
    return sched


def schedule_text(sched):
    """정년 일정을 한 줄로 요약(화면 표시용)."""
    ys = sorted(sched)
    if len(ys) == 1:
        return f"{sched[ys[0]]}세 유지"
    return " → ".join(f"{y}년 {sched[y]}세" for y in ys[1:])


def scenario_to_dict(scen: Scenario) -> dict:
    d = dataclasses.asdict(scen)
    for k in ("retirement_schedule", "ce_end_age_schedule"):
        d[k] = {str(y): int(a) for y, a in d[k].items()}
    return d


def scenario_from_dict(d: dict) -> Scenario:
    d = dict(d)
    for k in ("retirement_schedule", "ce_end_age_schedule"):
        d[k] = {int(y): int(a) for y, a in (d.get(k) or {}).items()}
    fields = {f.name for f in dataclasses.fields(Scenario)}
    return Scenario(**{k: v for k, v in d.items() if k in fields})


def with_ce_params(scen: Scenario, accept, wage_ratio):
    """민감도 분석용: 재고용 수락률·임금률만 바꾼 사본.
    직무유지형은 프리셋 정의(임금 85%·수락률 +15%p)를 따라가도록 같은 규칙으로 변환한다."""
    if scen.continued_type == "none" or scen.ce_fixed:
        return scen
    if scen.continued_type == "job_keep":
        return dataclasses.replace(scen, rehire_accept=min(accept + 0.15, 0.95))
    return dataclasses.replace(scen, rehire_accept=accept, rehire_wage_ratio=wage_ratio)


# ── 실행 ───────────────────────────────────────────────────────────────────
def run_all(profile=None, runs=200, horizon=12, rehire_wage=0.55, rehire_accept=0.62,
            hiring="replace", param_uncertainty=True, scenarios=None,
            objectives=D.DEFAULT_OBJECTIVES, materiality=D.DEFAULT_MATERIALITY,
            detail=False, stability=True, progress=None):
    """시나리오 묶음을 같은 난수로 실행하고 통계적 지배 필터로 후보를 가린다."""
    profile = profile or CompanyProfile()
    if scenarios is None:
        scenarios = preset_scenarios(rehire_wage, rehire_accept, hiring=hiring,
                                     codes=DEFAULT_PRESET_CODES)
    results = []
    for i, s in enumerate(scenarios):
        results.append(monte_carlo(profile, s, runs=runs, horizon=horizon,
                                   param_uncertainty=param_uncertainty, detail=detail))
        if progress:
            progress((i + 1) / len(scenarios), s.name)
    frontier, dominated, why = D.statistical_frontier(results, objectives, materiality)
    out = dict(results=results, frontier=frontier, dominated=dominated, why=why,
               objectives=list(objectives), materiality=materiality)
    if stability:
        out["stability"] = D.frontier_stability(results, objectives, materiality)
    return out


# ── 민감도: 재고용 조건 두 개가 후보 판정을 뒤집는 지점 ─────────────────────
def sensitivity(profile=None, runs=80, horizon=12,
                accept_grid=(0.37, 0.50, 0.62, 0.75, 0.85),
                wage_grid=(0.40, 0.51, 0.55, 0.59, 0.70), hiring="replace",
                scenarios=None, objectives=D.DEFAULT_OBJECTIVES,
                materiality=D.DEFAULT_MATERIALITY, base=None, progress=None):
    """KLIPS 앵커(수락률 0.374~0.621, 임금비 0.51~0.59) 주변 격자 탐색.
    계수는 점추정에 고정한다 — 여기서 보려는 것은 '이 두 조건이 결론을 뒤집는가'다.
    base=(수락률, 임금률)을 주면 그 지점과 가장 가까운 격자점을 기준 후보군으로 삼는다."""
    profile = profile or CompanyProfile()
    if scenarios is None:
        scenarios = preset_scenarios(hiring=hiring, codes=DEFAULT_PRESET_CODES)
    records, total, done = [], len(accept_grid) * len(wage_grid), 0
    for a in accept_grid:
        for w in wage_grid:
            out = run_all(profile, runs=runs, horizon=horizon, param_uncertainty=False,
                          scenarios=[with_ce_params(s, a, w) for s in scenarios],
                          objectives=objectives, materiality=materiality, stability=False)
            records.append(dict(accept=a, wage_ratio=w, frontier=out["frontier"]))
            done += 1
            if progress:
                progress(done / total, f"수락률 {a:.2f} · 임금률 {w:.2f}")
    if base is None:
        ref = records[len(records) // 2]
    else:
        ref = min(records, key=lambda r: abs(r["accept"] - base[0]) + abs(r["wage_ratio"] - base[1]))
    base_set = set(ref["frontier"])
    flips = [r for r in records if set(r["frontier"]) != base_set]
    return dict(grid=records, baseline_frontier=sorted(base_set),
                baseline_point=(ref["accept"], ref["wage_ratio"]),
                flip_points=[(r["accept"], r["wage_ratio"], sorted(r["frontier"])) for r in flips])


# ── 견고성: 여러 가정 조합에서 후보로 남는가 ────────────────────────────────
def assumption_states(profile: CompanyProfile, hiring: str):
    """결론을 흔들 수 있는 큰 가정 3종의 조합. 각 상태는 (라벨, 프로파일, 채용정책)."""
    regimes = [k for k in ("observed", "frozen", "seniority")]
    if profile.wage_regime == "custom":
        regimes = ["custom", "frozen", "seniority"]
    turn = [("퇴사 적음", 0.7), ("퇴사 보통", 1.0), ("퇴사 많음", 1.3)]
    hires = list(HIRING_POLICIES)
    states = []
    for rg, (tl, tm), hp in itertools.product(regimes, turn, hires):
        label_r = "자사 입력 임금" if rg == "custom" else WAGE_REGIMES[rg]["label"]
        p = dataclasses.replace(profile, wage_regime=rg,
                                base_push_turnover=profile.base_push_turnover * tm,
                                other_exit_rate=profile.other_exit_rate * tm)
        states.append(dict(label=f"{label_r} · {tl} · {HIRING_POLICIES[hp][0]}",
                           regime=rg, regime_label=label_r, turnover=tl, hiring=hp,
                           hiring_label=HIRING_POLICIES[hp][0], profile=p,
                           is_base=(rg == profile.wage_regime and tm == 1.0 and hp == hiring)))
    return states


def robustness(profile, scenarios, hiring="replace", runs=30, horizon=12,
               objectives=D.DEFAULT_OBJECTIVES, materiality=D.DEFAULT_MATERIALITY,
               weights=None, progress=None):
    """가정 조합 27가지 각각에서 다시 판정해, 시나리오별로 '후보로 남은 비율'과
    '우선순위 점수의 최대 후회'를 구한다."""
    states = assumption_states(profile, hiring)
    weights = weights or {k: 1.0 for k in objectives}
    table, scores = [], {}
    for i, st in enumerate(states):
        scens = [dataclasses.replace(s, hiring_mode=st["hiring"]) for s in scenarios]
        out = run_all(st["profile"], runs=runs, horizon=horizon, param_uncertainty=False,
                      scenarios=scens, objectives=objectives, materiality=materiality,
                      stability=False)
        pr = D.priority_scores(out["results"], weights)
        scores[st["label"]] = {n: v["score"] for n, v in pr.items()}
        table.append(dict(label=st["label"], regime=st["regime_label"], turnover=st["turnover"],
                          hiring=st["hiring_label"], is_base=st["is_base"],
                          frontier=out["frontier"]))
        if progress:
            progress((i + 1) / len(states), st["label"])
    names = [s.name for s in scenarios]
    share = {n: float(np.mean([n in t["frontier"] for t in table])) for n in names}
    return dict(states=table, frontier_share=share, regret=D.minimax_regret(scores),
                scores=scores, n_states=len(states))


# ── 토네이도: 어떤 가정이 결과를 가장 흔드는가 ──────────────────────────────
def tornado(profile, scenario, baseline, hiring="replace", accept=0.62, wage_ratio=0.55,
            runs=30, horizon=12, progress=None):
    """가정을 하나씩만 낮게/높게 바꿨을 때 '시나리오 − 기준선' 격차가 얼마나 움직이는지.
    격차는 인건비(%)와 신규채용(%) 두 가지로 본다."""
    def gap(p, acc, wr):
        s = with_ce_params(scenario, acc, wr)
        a = monte_carlo(p, s, runs=runs, horizon=horizon, param_uncertainty=False)["summary"]
        b = monte_carlo(p, baseline, runs=runs, horizon=horizon, param_uncertainty=False)["summary"]
        return dict(cost=(a["cost_npv_mean"] / b["cost_npv_mean"] - 1) * 100,
                    hires=(a["total_hires_mean"] / max(b["total_hires_mean"], 1e-9) - 1) * 100)

    rep = dataclasses.replace
    cases = [
        ("재고용 수락률", f"{max(accept-0.2,0.05):.2f}", f"{min(accept+0.2,0.95):.2f}",
         lambda: gap(profile, max(accept - 0.2, 0.05), wage_ratio),
         lambda: gap(profile, min(accept + 0.2, 0.95), wage_ratio)),
        ("재고용 임금률", f"{max(wage_ratio-0.15,0.3):.2f}", f"{min(wage_ratio+0.15,0.95):.2f}",
         lambda: gap(profile, accept, max(wage_ratio - 0.15, 0.3)),
         lambda: gap(profile, accept, min(wage_ratio + 0.15, 0.95))),
        ("자발적 퇴사율", "-30%", "+30%",
         lambda: gap(rep(profile, base_push_turnover=profile.base_push_turnover * 0.7), accept, wage_ratio),
         lambda: gap(rep(profile, base_push_turnover=profile.base_push_turnover * 1.3), accept, wage_ratio)),
        ("기타 퇴사율", "-30%", "+30%",
         lambda: gap(rep(profile, other_exit_rate=profile.other_exit_rate * 0.7), accept, wage_ratio),
         lambda: gap(rep(profile, other_exit_rate=profile.other_exit_rate * 1.3), accept, wage_ratio)),
        ("50세 이후 임금", "동결", "호봉제",
         lambda: gap(rep(profile, wage_regime="frozen"), accept, wage_ratio),
         lambda: gap(rep(profile, wage_regime="seniority"), accept, wage_ratio)),
    ]
    base = gap(profile, accept, wage_ratio)
    rows = []
    for i, (label, lo_txt, hi_txt, lo_fn, hi_fn) in enumerate(cases):
        lo, hi = lo_fn(), hi_fn()
        rows.append(dict(param=label, low_label=lo_txt, high_label=hi_txt,
                         cost_low=lo["cost"], cost_high=hi["cost"],
                         hires_low=lo["hires"], hires_high=hi["hires"]))
        if progress:
            progress((i + 1) / len(cases), label)
    return dict(base=base, rows=rows, scenario=scenario.name, baseline=baseline.name)


# ── 정책 설계 공간 탐색 ─────────────────────────────────────────────────────
def design_space(profile, hiring="replace", accept=0.62, wage_ratio=0.55, runs=20, horizon=12,
                 targets=(62, 63, 64, 65), first_years=(2028, 2029, 2030), steps=(1, 2, 3),
                 objectives=D.DEFAULT_OBJECTIVES, materiality=D.DEFAULT_MATERIALITY,
                 progress=None):
    """공개된 안에 얽매이지 않고 '도달 연령 × 시작 연도 × 연장 간격 × 방식'을 격자로 훑는다.
    방식 3종: 정년연장만 / 정년연장+재고용(정년+1세) / 재고용만(정년 60세 유지, 같은 일정으로 상한 확대).
    계수는 점추정에 고정하고 반복 수를 낮춘 탐색용 실행이다."""
    specs = [Scenario("기준 현행 유지", hiring_mode=hiring)]
    meta = [dict(kind="현행 유지", target=60, first=None, step=None)]
    for tgt, fy, st in itertools.product(targets, first_years, steps):
        sched = build_schedule(fy, st, tgt, base_year=profile.start_year)
        tag = f"{fy}년 시작·{st}년 간격·{tgt}세"
        specs.append(Scenario(f"연장만 {tag}", retirement_schedule=sched, hiring_mode=hiring))
        meta.append(dict(kind="정년연장만", target=tgt, first=fy, step=st))
        specs.append(Scenario(f"연장+재고용 {tag}", retirement_schedule=sched,
                              continued_type="rehire", ce_end_age_schedule=_plus_one(sched, cap=tgt + 1),
                              rehire_accept=accept, rehire_wage_ratio=wage_ratio, hiring_mode=hiring))
        meta.append(dict(kind="정년연장+재고용", target=tgt, first=fy, step=st))
        specs.append(Scenario(f"재고용만 {tag}", retirement_schedule={profile.start_year: 60},
                              continued_type="rehire", ce_end_age_schedule=sched,
                              rehire_accept=accept, rehire_wage_ratio=wage_ratio, hiring_mode=hiring))
        meta.append(dict(kind="재고용만", target=tgt, first=fy, step=st))
    results = []
    for i, s in enumerate(specs):
        results.append(monte_carlo(profile, s, runs=runs, horizon=horizon, param_uncertainty=False))
        if progress:
            progress((i + 1) / len(specs), s.name)
    frontier, dominated, _ = D.statistical_frontier(results, objectives, materiality)
    base = results[0]["summary"]
    rows = []
    for r, m, s in zip(results, meta, specs):
        x = r["summary"]
        rows.append(dict(name=x["scenario"], **m, on_frontier=x["scenario"] in frontier,
                         cost_eok=x["cost_npv_mean"] / 1e4,
                         cost_pct=(x["cost_npv_mean"] / base["cost_npv_mean"] - 1) * 100,
                         hires=x["total_hires_mean"],
                         hires_pct=(x["total_hires_mean"] / max(base["total_hires_mean"], 1e-9) - 1) * 100,
                         wait=x["promo_congestion_mean"], exp55=x["exp55_person_years_mean"],
                         income_gap=x["income_gap_mean"], skill_gap=x["skill_gap_mean"],
                         spec=scenario_to_dict(s)))
    return dict(rows=rows, n=len(rows), frontier=frontier)


if __name__ == "__main__":
    out = run_all(runs=60, horizon=12)
    for r in out["results"]:
        s = r["summary"]
        print(f"{s['scenario']:<34} 인건비 {s['cost_npv_mean']/1e4:,.0f}억 | "
              f"채용 {s['total_hires_mean']:,.0f} | 대기 {s['promo_congestion_mean']:.1f}년 | "
              f"55+인년 {s['exp55_person_years_mean']:,.0f} | 소득공백 {s['income_gap_mean']:,.0f} | "
              f"숙련공백 {s['skill_gap_mean']:,.0f}")
    print("\n후보:", out["frontier"])
    print("탈락:", out["dominated"])
