"""
dss_layer.py — DSS 층: 시나리오 프리셋, 파레토 프론티어 권장 후보 도출, 민감도 분석
====================================================================================
원칙: 권장은 결정론적 알고리즘(지배 관계 필터)이 수행. LLM은 이 결과 수치의
      해석 산문만 생성(llm_report.py). PassNavi 환각방지 아키텍처 계승.

시나리오 프리셋 근거 (2026-08-28 웹 재검증 — 아래 "입법 현황 갱신" 참조):
  S1 현행유지: 60세 (베이스라인)
  S2 단계적 연장 2036: 2028년부터 2년마다 +1세 → 2036년 65세 (특위 검토안 중 가장 빠른 안)
  S3 경사노위 계속고용의무형: 정년 60세 유지 + 65세까지 재고용 의무(수락률 반영)
  S4 급진 연장안(정흥준): 2028년부터 매년 +1세 → 2032년 65세
  S5 S2 + 임금피크제 병행 (정년 전 3년, 연 10% 감액)
  S6 S3 + 직무유지형 (감액 제한형 계속고용)
  S7 완만한 연장 2041: 2029년부터 3년마다 +1세 → 2041년 65세 (특위 검토안 중 가장 느린 안)

입법 현황 갱신 (2026-08-28 웹 검증):
  - 정년연장 입법은 2026년 상반기 목표를 넘겨 **아직 국회를 통과하지 않았다**. 민주당
    정년연장특위는 4월 30일 간담회 이후 후속 일정이 잡히지 않았고, 7월 말 기준 청와대가
    청년 고용 악화·세대 간 일자리 갈등을 이유로 속도조절을 요청한 상태다.
    → 제품의 전제("경합 중")는 여전히 유효하다.
  - 다만 특위가 실제로 검토 중인 도달 시점은 2036 / 2039 / 2041년이며, 기존 코드가 쓰던
    2037년은 그중 어느 것과도 일치하지 않았다. 실제 정책 공간(2032·2036·2041)을 덮도록
    스케줄을 교체하고, 가장 느린 안(S7)을 추가했다.
  - 속도조절의 사유가 '청년 고용'이라는 점은 이 모델의 결과와 정확히 맞물린다: 시나리오 간
    변별력이 가장 큰 지표는 인건비(1.5%)가 아니라 55세+ 인년(42.1%)·승진적체(28.5%)·신규채용(15.0%)이다.
    출처: 브라보마이라이프 2026-07-03, 위키트리 2026-07-30 (design_doc.md 출처 목록).
"""
from __future__ import annotations
from engine_retirement_v2 import CompanyProfile, Scenario, monte_carlo

# 특위 검토안 A(가장 빠름): 2028년부터 2년마다 +1세 → 2036년 65세
DP_SCHEDULE = {2026: 60, 2028: 61, 2030: 62, 2032: 63, 2034: 64, 2036: 65}
DP_CE_END = {y: a + 1 for y, a in DP_SCHEDULE.items()}          # 정년+1세 재고용
# 특위 검토안 C(가장 느림): 2029년부터 3년마다 +1세 → 2041년 65세
SLOW_SCHEDULE = {2026: 60, 2029: 61, 2032: 62, 2035: 63, 2038: 64, 2041: 65}
SLOW_CE_END = {y: a + 1 for y, a in SLOW_SCHEDULE.items()}
# 정흥준안: 2028년부터 매년 +1세 → 2032년 65세
FAST_SCHEDULE = {2026: 60, 2028: 61, 2029: 62, 2030: 63, 2031: 64, 2032: 65}
ESLC_CE_END = {2026: 65}                                          # 경사노위: 65세까지 계속고용

# 채용 정책 — 정년연장의 부담을 무엇으로 흡수할 것인가. 이 선택이 결과를 크게 가르는데
# 기존 코드는 전 시나리오를 "replace"로 고정해 두어, 정작 기업이 실제로 내려야 하는
# 결정(채용을 줄일 것인가, 인건비를 늘릴 것인가)이 가정에 묻혀 있었다(2026-08-28 노출).
HIRING_POLICIES = {
    "replace": ("정원 유지", "결원을 전원 충원해 인원을 유지한다. 고령자가 남으면 그만큼 "
                            "퇴직이 줄어 신규채용이 자동으로 감소한다."),
    "budget":  ("인건비 총액 유지", "인건비 예산을 초기 수준으로 묶는다. 고령자 인건비가 "
                                 "늘면 그 폭만큼 신규채용을 더 줄여 흡수한다."),
    "fixed":   ("채용 규모 고정", "매년 같은 인원을 뽑는다. 정년연장의 부담이 채용이 아니라 "
                               "인원·인건비 증가로 나타난다."),
}


def preset_scenarios(rehire_wage=0.55, rehire_accept=0.62, hiring="replace", hiring_fixed_n=0):
    def S(name, **kw):
        return Scenario(name, hiring_mode=hiring, hiring_fixed_n=hiring_fixed_n, **kw)
    return [
        S("S1 현행유지(60세)"),
        S("S2 단계적연장(2036년 65세)", retirement_schedule=DP_SCHEDULE,
          continued_type="rehire", ce_end_age_schedule=DP_CE_END,
          rehire_wage_ratio=rehire_wage, rehire_accept=rehire_accept),
        S("S3 계속고용의무(60세+재고용 65세)", retirement_schedule={2026: 60},
          continued_type="rehire", ce_end_age_schedule=ESLC_CE_END,
          rehire_wage_ratio=rehire_wage, rehire_accept=rehire_accept),
        S("S4 급진연장(2032년 65세)", retirement_schedule=FAST_SCHEDULE),
        S("S5 단계적연장+임금피크", retirement_schedule=DP_SCHEDULE,
          continued_type="rehire", ce_end_age_schedule=DP_CE_END,
          rehire_wage_ratio=rehire_wage, rehire_accept=rehire_accept,
          wage_peak=True, peak_years=3, peak_cut=0.10),
        S("S6 계속고용-직무유지형", retirement_schedule={2026: 60},
          continued_type="job_keep", ce_end_age_schedule=ESLC_CE_END,
          rehire_wage_ratio=0.85, rehire_accept=min(rehire_accept + 0.15, 0.95)),
        S("S7 완만한연장(2041년 65세)", retirement_schedule=SLOW_SCHEDULE,
          continued_type="rehire", ce_end_age_schedule=SLOW_CE_END,
          rehire_wage_ratio=rehire_wage, rehire_accept=rehire_accept),
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


def run_all(profile=None, runs=200, horizon=12, rehire_wage=0.55, rehire_accept=0.62,
            hiring="replace", param_uncertainty=True):
    profile = profile or CompanyProfile()
    results = [monte_carlo(profile, s, runs=runs, horizon=horizon,
                           param_uncertainty=param_uncertainty)
               for s in preset_scenarios(rehire_wage, rehire_accept, hiring=hiring)]
    summaries = [r["summary"] for r in results]
    frontier, dominated = pareto_frontier(summaries)
    return dict(results=results, frontier=[f["scenario"] for f in frontier],
                dominated=[d["scenario"] for d in dominated])


# ── 민감도 분석: 불확실 파라미터가 결론(프론티어 구성)을 뒤집는 지점 탐색 ──
def sensitivity(profile=None, runs=80, horizon=12,
                accept_grid=(0.37, 0.50, 0.62, 0.75, 0.85),
                wage_grid=(0.40, 0.51, 0.55, 0.59, 0.70), hiring="replace"):
    """KLIPS 앵커: 수락률 [0.374(1년내 재취업), 0.621(전체 재취업)],
       임금비 [0.51(중위), 0.59(평균)] 주변 격자 탐색.
       격자 탐색은 계수를 점추정에 고정(param_uncertainty=False)한다 — 여기서 보고 싶은 것은
       '이 두 파라미터가 결론을 뒤집는가'이지 계수 추정 오차가 아니며, 고정해야 격자 간
       비교가 공통 난수로 깨끗해진다."""
    profile = profile or CompanyProfile()
    records = []
    for a in accept_grid:
        for w in wage_grid:
            out = run_all(profile, runs=runs, horizon=horizon,
                          rehire_wage=w, rehire_accept=a, hiring=hiring,
                          param_uncertainty=False)
            records.append(dict(accept=a, wage_ratio=w, frontier=out["frontier"]))
    # 프론티어 구성이 바뀌는 경계 식별
    base = set(records[len(records) // 2]["frontier"])
    flips = [r for r in records if set(r["frontier"]) != base]
    return dict(grid=records, baseline_frontier=sorted(base),
                flip_points=[(r["accept"], r["wage_ratio"], sorted(r["frontier"])) for r in flips])


if __name__ == "__main__":
    out = run_all(runs=60, horizon=12)
    for r in out["results"]:
        s = r["summary"]
        print(f"{s['scenario']:<28} NPV비용 {s['cost_npv_mean']/1e4:,.0f}억 | "
              f"채용 {s['total_hires_mean']:,.0f} | 적체p90 {s['promo_congestion_mean']:.1f}년 | "
              f"55+인년 {s['exp55_person_years_mean']:,.0f}")
    print("\n파레토 프론티어(권장 후보군):", out["frontier"])
    print("지배당해 탈락:", out["dominated"])
