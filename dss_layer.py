"""
dss_layer.py — DSS 층: 시나리오 프리셋, 파레토 프론티어 권장 후보 도출, 민감도 분석
====================================================================================
원칙: 권장은 결정론적 알고리즘(지배 관계 필터)이 수행. LLM은 이 결과 수치의
      해석 산문만 생성(llm_report.py). PassNavi 환각방지 아키텍처 계승.

시나리오 프리셋 근거 (2026-07 웹 검증, design_doc.md 출처 목록 참조):
  S1 현행유지: 60세 (베이스라인)
  S2 민주당 정년특위안: 2029년 61세 시작, 2년마다 +1세 → 2037년 65세.
     퇴직 후 재고용 병행(정년+1세까지, 희망자 전원 원칙)
  S3 경사노위 계속고용의무형: 정년 60세 유지 + 65세까지 재고용 의무(수락률 반영)
  S4 급진 연장안(정흥준 2026-06 국회토론회): 2028년부터 매년 +1세 → 2032년 65세
  S5 S2 + 임금피크제 병행 (정년 전 3년, 연 10% 감액)
  S6 S3 + 직무유지형 (감액 제한형 계속고용)
"""
from __future__ import annotations
import numpy as np
from engine_retirement_v2 import CompanyProfile, Scenario, monte_carlo

# 민주당안 스케줄: 2029(61) → 2031(62) → 2033(63) → 2035(64) → 2037(65)
DP_SCHEDULE = {2026: 60, 2029: 61, 2031: 62, 2033: 63, 2035: 64, 2037: 65}
DP_CE_END = {y: a + 1 for y, a in DP_SCHEDULE.items()}          # 정년+1세 재고용
FAST_SCHEDULE = {2026: 60, 2028: 61, 2029: 62, 2030: 63, 2031: 64, 2032: 65}
ESLC_CE_END = {2026: 65}                                          # 경사노위: 65세까지 계속고용


def preset_scenarios(rehire_wage=0.55, rehire_accept=0.62):
    return [
        Scenario("S1 현행유지(60세)"),
        Scenario("S2 민주당안(단계적 65세+재고용)", retirement_schedule=DP_SCHEDULE,
                 continued_type="rehire", ce_end_age_schedule=DP_CE_END,
                 rehire_wage_ratio=rehire_wage, rehire_accept=rehire_accept),
        Scenario("S3 계속고용의무(60세+재고용 65세)", retirement_schedule={2026: 60},
                 continued_type="rehire", ce_end_age_schedule=ESLC_CE_END,
                 rehire_wage_ratio=rehire_wage, rehire_accept=rehire_accept),
        Scenario("S4 급진연장(2032년 65세)", retirement_schedule=FAST_SCHEDULE),
        Scenario("S5 민주당안+임금피크", retirement_schedule=DP_SCHEDULE,
                 continued_type="rehire", ce_end_age_schedule=DP_CE_END,
                 rehire_wage_ratio=rehire_wage, rehire_accept=rehire_accept,
                 wage_peak=True, peak_years=3, peak_cut=0.10),
        Scenario("S6 계속고용-직무유지형", retirement_schedule={2026: 60},
                 continued_type="job_keep", ce_end_age_schedule=ESLC_CE_END,
                 rehire_wage_ratio=0.85, rehire_accept=min(rehire_accept + 0.15, 0.95)),
    ]


# ── 파레토 프론티어 (결정론적 권장 후보 도출) ──
# 목적함수: cost↓, hires↑, congestion↓, exp55↑
OBJECTIVES = [("cost_npv_mean", -1), ("total_hires_mean", +1),
              ("promo_congestion_mean", -1), ("exp55_person_years_mean", +1)]


EPS_REL = 0.004  # 몬테카를로 노이즈 허용폭(상대 0.4%): 이보다 작은 차이는 '동등'으로 간주


def dominates(a, b):
    """ε-지배: 모든 목적에서 유의하게 나쁘지 않고, 최소 하나에서 유의하게 좋음.
    ε는 MC 표본오차 수준의 상대 허용폭 — 노이즈로 인한 과잉 프론티어 방지."""
    better = False
    for key, sign in OBJECTIVES:
        va, vb = sign * a[key], sign * b[key]
        tol = EPS_REL * max(abs(va), abs(vb), 1e-9)
        if va < vb - tol:
            return False
        if va > vb + tol:
            better = True
    return better


def pareto_frontier(summaries):
    frontier, dominated = [], []
    for s in summaries:
        if any(dominates(o, s) for o in summaries if o is not s):
            dominated.append(s)
        else:
            frontier.append(s)
    return frontier, dominated


def run_all(profile=None, runs=200, horizon=12, rehire_wage=0.55, rehire_accept=0.62):
    profile = profile or CompanyProfile()
    results = [monte_carlo(profile, s, runs=runs, horizon=horizon)
               for s in preset_scenarios(rehire_wage, rehire_accept)]
    summaries = [r["summary"] for r in results]
    frontier, dominated = pareto_frontier(summaries)
    return dict(results=results, frontier=[f["scenario"] for f in frontier],
                dominated=[d["scenario"] for d in dominated])


# ── 민감도 분석: 불확실 파라미터가 결론(프론티어 구성)을 뒤집는 지점 탐색 ──
def sensitivity(profile=None, runs=80, horizon=12,
                accept_grid=(0.37, 0.50, 0.62, 0.75, 0.85),
                wage_grid=(0.40, 0.51, 0.55, 0.59, 0.70)):
    """KLIPS 앵커: 수락률 [0.374(1년내 재취업), 0.621(전체 재취업)],
       임금비 [0.51(중위), 0.59(평균)] 주변 격자 탐색."""
    profile = profile or CompanyProfile()
    records = []
    for a in accept_grid:
        for w in wage_grid:
            out = run_all(profile, runs=runs, horizon=horizon,
                          rehire_wage=w, rehire_accept=a)
            records.append(dict(accept=a, wage_ratio=w, frontier=out["frontier"]))
    # 프론티어 구성이 바뀌는 경계 식별
    base = set(records[len(records) // 2]["frontier"])
    flips = [r for r in records if set(r["frontier"]) != base]
    return dict(grid=records, baseline_frontier=sorted(base),
                flip_points=[(r["accept"], r["wage_ratio"], sorted(r["frontier"])) for r in flips])


if __name__ == "__main__":
    import json
    out = run_all(runs=60, horizon=12)
    for r in out["results"]:
        s = r["summary"]
        print(f"{s['scenario']:<28} NPV비용 {s['cost_npv_mean']/1e4:,.0f}억 | "
              f"채용 {s['total_hires_mean']:,.0f} | 적체p90 {s['promo_congestion_mean']:.1f}년 | "
              f"55+인년 {s['exp55_person_years_mean']:,.0f}")
    print("\n파레토 프론티어(권장 후보군):", out["frontier"])
    print("지배당해 탈락:", out["dominated"])
