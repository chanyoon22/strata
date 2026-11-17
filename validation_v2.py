"""
validation_v2.py — 검증 스위트 (11종)

V1 캘리브레이션 재현 | V2 인원 보존 | V3 2016년 60세 의무화 DiD |
V4 외부 벤치마크 | V5 자체 재적합 표본외 검증 | V6 정년퇴직 식별 나이 검증 |
V7 KLIPS-WPS 교차검증 | V8 사전추세(평행추세) 검정 |
V9 고용노동부 행정통계 대조 | V10 기준선 연속성 | V11 판정 규칙 자기일관성

각 결과의 method 필드가 그 검사의 성격을 말한다 — 세 종류를 뭉뚱그리지 말 것:
  computed   (8종) 이 코드가 지금 이 데이터로 직접 재계산해 통과/실패를 판정한다.
  documented (2종) 원본 KLIPS 직업력 원자료를 1회 분석한 결과를 인용한다. 이 배포본에는
                   원자료가 없어 재현할 수 없다. passed=True는 "재현됐다"가 아니라
                   "근거가 문서로 남아 있다"는 뜻이다.
  diagnostic (1종) 통과/실패가 아니라 **가정을 진단**한다. V8이 여기 속하며 passed는
                   구조적으로 항상 True다 — 실제 판단은 parallel_trends_ok 필드로 한다.

★ 표기 규칙: "검증 11종 전부 통과"라고 쓰지 말 것. 정확한 표현은
   "8종 자동 계산 통과 + 2종 근거 문서화 + 1종 가정 검정(평행추세 미성립)".
   그리고 V3의 DiD를 인과효과로 서술하지 말 것 — V8이 그 가정을 기각한다.

회사 명부를 올린 경우의 검증(과거 명부로 현재 맞히기)은 backtest.py가 따로 맡는다.
"""
import json
import numpy as np
from pathlib import Path
from engine_retirement_v2 import CompanyProfile, Scenario, generate_workforce, _hazard, simulate

_D = Path(__file__).parent

def v1_calibration():
    prof = CompanyProfile()
    wf = generate_workforce(prof)
    reg = wf["status"] == 0                      # 절편은 정규 재직자 평균에 맞춘다(v3)
    rate = float(_hazard({k: wf[k][reg] for k in ("age", "tenure", "wage", "hours")}, wf["_b0"]).mean())
    return dict(test="V1 캘리브레이션", method="computed", target=prof.base_push_turnover,
                actual=round(rate, 5), passed=bool(abs(rate - prof.base_push_turnover) < 5e-4))

def v2_conservation(runs=3):
    """기말 인원 = 기초 − (자발·기타·희망퇴직·정년퇴직·계속고용 종료) + 채용. 모든 프리셋과
    레버를 다 켠 시나리오까지 확인한다(v2는 한 가지 시나리오만 봐서, 재고용이 조용히
    무력화되는 경로를 놓친 적이 있다)."""
    from dss_layer import preset_scenarios
    prof = CompanyProfile(ce_rate_now=0.4, ce_end_age_now=64)
    scens = preset_scenarios() + [
        Scenario("검증-전 레버", retirement_schedule={2026: 60, 2028: 61, 2030: 62}, continued_type="rehire",
                 wage_peak=True, erp_start=2027, erp_end=2029, erp_take_rate=0.1,
                 wage_reform_regime="frozen", wage_reform_year=2028, hiring_mode="budget"),
        Scenario("검증-상한 미지정 재고용", retirement_schedule={2026: 60, 2029: 61}, continued_type="rehire")]
    bad = checked = 0
    for scen in scens:
        for r in range(runs):
            prev = prof.n_employees
            for row in simulate(prof, scen, seed=100 + r):
                out = (row["n_push"] + row["n_other"] + row["n_erp"]
                       + (row["n_retire"] - row["n_rehired"]) + row["n_ce_exit"])
                bad += int(prev - out + row["n_hire"] != row["headcount"])
                checked += 1
                prev = row["headcount"]
    return dict(test="V2 인원 보존", method="computed", violations=bad, checked_years=checked,
                scenarios=len(scens), passed=bad == 0)

def v3_did():
    d = json.loads((_D / "did_results.json").read_text())
    return dict(test="V3 준실험(2016년 60세 의무화 DiD)", method="computed",
                estimate_pp=round(d["did_estimate"] * 100, 2), p=round(d["p"], 4),
                design=d["design"], caveat=d["caveat"],
                warning="V8 사전추세 검정을 함께 볼 것 — 이 추정치는 평행추세 가정을 통과하지 못한다.",
                passed=bool(d["did_estimate"] < 0 and d["p"] < 0.05))


def _ols(y, X):
    """계수·표준오차·양측 p값. (statsmodels 없이 동작하도록 numpy로 구현)"""
    import numpy as _np
    b, *_ = _np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    dof = len(y) - X.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * _np.linalg.inv(X.T @ X)
    se = _np.sqrt(_np.diag(cov))
    t = b / se
    try:
        from scipy.stats import t as _t
        p = 2 * _t.sf(_np.abs(t), dof)
    except Exception:
        p = _np.full_like(t, _np.nan)
    return b, se, t, p, dof


def v8_pretrend():
    """V8 사전추세(평행추세) 검정 — DiD의 식별가정을 이 프로젝트의 자체 데이터로 직접 검증.

    2026-08-28 추가. DiD가 인과추정이 되려면 처치 전 처치군-대조군 격차가 평행해야 한다.
    event_study.csv(2010~2023, 2016 제외)의 연도별 격차로 두 가지를 계산한다:
      (1) 처치 전(2010~2015) 격차의 선형추세 — 0이어야 평행추세 가정이 성립
      (2) 그 추세를 통제한 뒤의 post 효과 — 추세 통제 후에도 효과가 남는지

    결과(실행해 보면 재현됨): 처치 전 격차가 연 -1.38%p로 이미 좁혀지고 있었고(t=-2.41),
    이 추세를 통제하면 post 계수가 +2.44%p(p=0.31)로 부호가 뒤집히며 유의성도 사라진다.
    즉 V3의 -4.03%p는 상당 부분 '이미 진행 중이던 수렴 추세의 연장'일 수 있다.
    또한 셀 크기가 처치군 연 20~163 사람-연도·사건 1~11건으로 매우 작아 검정력이 낮다.
    → 이 검정은 '통과/실패'가 아니라 **V3 주장을 어디까지 할 수 있는지의 상한**을 정한다.
       passed=True는 '평행추세가 성립한다'가 아니라 '검정을 실제로 수행하고 결과를 공개했다'는 뜻.
    """
    import numpy as _np
    import pandas as pd
    es = pd.read_csv(_D / "event_study.csv")
    es.columns = ["year", "control", "treated", "gap"]
    pre = es[es.year < 2016]

    Xp = _np.column_stack([_np.ones(len(pre)), pre.year.to_numpy(float) - 2015.0])
    bp, sep, tp, pp, _ = _ols(pre.gap.to_numpy(float), Xp)

    d = es[es.year != 2016]
    Xa = _np.column_stack([_np.ones(len(d)), d.year.to_numpy(float) - 2015.0,
                           (d.year > 2016).astype(float).to_numpy()])
    ba, sea, ta, pa, _ = _ols(d.gap.to_numpy(float), Xa)

    parallel_ok = bool(pp[1] > 0.10)
    survives = bool(ba[2] < 0 and pa[2] < 0.10)
    return dict(
        # method="diagnostic": 통과/실패를 판정하는 검증이 아니라 **가정을 진단**하는 검사다.
        # passed가 구조적으로 항상 True이므로 이걸 "N종 통과" 집계에 넣으면 통과 수가 부풀려진다
        # (자체 감사에서 지적받아 computed → diagnostic으로 분리, 2026-08-28).
        test="V8 사전추세(평행추세) 검정", method="diagnostic",
        pretrend_pp_per_year=round(float(bp[1]) * 100, 3),
        pretrend_t=round(float(tp[1]), 2), pretrend_p=round(float(pp[1]), 4),
        parallel_trends_ok=parallel_ok,
        did_trend_adjusted_pp=round(float(ba[2]) * 100, 2),
        did_trend_adjusted_p=round(float(pa[2]), 4),
        effect_survives_trend_adjustment=survives,
        interpretation=("처치 전 격차가 이미 유의하게 좁혀지고 있어 평행추세 가정이 성립하지 않는다. "
                        "추세를 통제하면 효과가 사라진다 — V3는 인과효과가 아니라 "
                        "'제도 변화와 같은 방향의 정합성 확인'으로만 해석해야 한다."
                        if not (parallel_ok and survives) else
                        "사전추세가 유의하지 않고 추세 통제 후에도 효과가 유지된다."),
        passed=True,  # 검정 수행·공개 여부. 가정 성립 여부는 parallel_trends_ok 필드로 판단할 것.
    )

def v4_external():
    """문서화 근거(재계산 불가): 재취업 임금비 중위 0.51·평균 0.59(params_v2.json)가
    남재량·김동배(2020)의 정년퇴직 재취업 임금 하락폭(최초 재취업 시 약 -33%, 이후 연 -10%대
    수렴)과 같은 방향·같은 자릿수라는 문헌 대조. 두 추정치를 같은 코드로 재현하는 것이 아니라
    "우리 수치가 선행연구와 크게 어긋나지 않는다"는 정합성 확인이다."""
    p = json.loads((_D / "params_v2.json").read_text(encoding="utf-8"))
    wr = p["retirement_reemployment"]["big300"]["wage_ratio_median"]
    plausible = 0.35 <= wr <= 0.75  # 문헌 범위 대비 그럴듯한 구간인지 최소한의 자동 체크
    return dict(test="V4 외부 벤치마크(재취업 임금비)", method="documented",
                our_value=wr, benchmark="남재량·김동배(2020) 최초 재취업 약 -33%, 이후 연 -10%대",
                note="자동 재현 아님 — 문헌 대조로 그럴듯함만 확인. 원자료 재분석 없이는 재현 불가.",
                passed=bool(plausible))

def v5_refit():
    r = json.loads((_D / "refit_coefficients.json").read_text())
    c = r["coefficients"]
    prev = {"age": -0.0350, "tenure": -0.1044, "wage": -0.0031, "hours": 0.0135, "edu": -0.0924}
    match = sum(np.sign(c[k]["coef"]) == np.sign(prev[k]) for k in prev)
    return dict(test="V5 자체 재적합 표본외 검증(26→27차, 사건 574)", method="computed",
                direction_match=f"{match}/5",
                significant=[k for k in prev if c[k]["p"] < 0.05], passed=bool(match == 5))

def v7_cross_source():
    """KLIPS 개인 실표본 vs WPS 사업체 보고 — 독립 소스 교차검증."""
    import pandas as pd
    s = pd.read_csv(_D / "big_firm_sample_2023.csv")
    klips_55p = float((s.age >= 55).mean())
    wps = json.loads((_D / "params_v2.json").read_text(encoding="utf-8"))["wps2023"]
    wps_55p = wps["55세이상_비중"]
    return dict(test="V7 교차검증(55세+ 비중: KLIPS 개인표본 vs WPS 사업체보고)", method="computed",
                klips=round(klips_55p, 3), wps=round(wps_55p, 3),
                passed=bool(abs(klips_55p - wps_55p) < 0.05))

def v6_retirement_id():
    """문서화 근거(재계산 불가): 이 배포본엔 원본 KLIPS P파일·직업력 파일이 포함되지 않아
    나이 결합 자체를 이 코드로 재현할 수 없다. n=595 결합 결과(퇴직나이 중위 60세,
    55~65세 비중 86%)를 그대로 인용한다."""
    return dict(test="V6 정년퇴직 식별(P파일 나이 결합)", method="documented",
                evidence="j602=5 퇴직나이 중위 60세, 55~65세 비중 86% (n=595)",
                note="자동 재현 아님 — 원본 P파일·직업력 파일이 이 배포본에 없어 이 코드로 재계산 불가.",
                passed=True)

# 고용노동부 고령자 고용현황: 300인 이상 사업장의 55세 이상 근로자 비중(각 연도 말).
# 출처: 고용노동부 통계(신아일보 2024-07-28 보도 인용). 모형과 독립적인 행정통계다.
MOEL_55PLUS = {2013: 0.101, 2015: 0.115, 2017: 0.132, 2019: 0.154, 2021: 0.172, 2023: 0.188}


def v9_admin_benchmark():
    """KLIPS 실표본(2023)의 55세 이상 비중이 고용노동부 행정통계(2023)와 3%p 안에서 맞는가.
    V7(사업체패널)과 달리 전수에 가까운 행정자료와의 대조다. 표본이 고령층을 조금 덜 담고
    있다는 방향성(표본 17.1% < 행정 18.8%)도 함께 기록한다."""
    import pandas as pd
    s = pd.read_csv(_D / "big_firm_sample_2023.csv")
    ours = float((s.age >= 55).mean())
    ref = MOEL_55PLUS[2023]
    yrs = sorted(MOEL_55PLUS)
    slope = (MOEL_55PLUS[yrs[-1]] - MOEL_55PLUS[yrs[0]]) / (yrs[-1] - yrs[0])
    return dict(test="V9 행정통계 대조(55세+ 비중: KLIPS 표본 vs 고용노동부 2023)", method="computed",
                klips=round(ours, 3), moel=ref, gap_pp=round((ours - ref) * 100, 1),
                moel_trend_pp_per_year=round(slope * 100, 2),
                note=f"행정통계 대비 {abs(ours-ref)*100:.1f}%p 낮음. 300인 이상 사업장의 55세 이상 비중은 "
                     f"2013~2023년 연 {slope*100:.2f}%p씩 올랐다 — 모형은 이 추세(코호트 고령화)를 따로 넣지 않고 "
                     "현재 명부의 연령 구조가 시간이 지나며 만드는 변화만 계산한다.",
                passed=bool(abs(ours - ref) < 0.03))


def v10_baseline_continuity():
    """기준선(현행 유지)이 현재를 이어가는가. v2는 초기 인구의 60세 이상을 1년차에 전원
    내보내 첫해 퇴장이 이후의 6배로 튀었다. 현재 재고용 관행을 넣은 v3에서 첫해 퇴장 인원이
    3~5년차 평균의 0.6~1.5배 안에 있는지(튀지도 꺼지지도 않는지) 확인한다."""
    import company_data as C
    prof = C.demo_profile(with_ce_practice=True)
    base = Scenario("현행 유지")
    y1, later = [], []
    for r in range(20):
        rows = simulate(prof, base, seed=200 + r, horizon=6)
        ex = [x["n_retire"] - x["n_rehired"] + x["n_ce_exit"] for x in rows]
        y1.append(ex[0]); later.append(sum(ex[2:5]) / 3)
    ratio = float(sum(y1) / max(sum(later), 1e-9))
    return dict(test="V10 기준선 연속성(첫해 정년·재고용 퇴장이 튀지 않는가)", method="computed",
                first_year=round(sum(y1) / 20, 1), later_avg=round(sum(later) / 20, 1), ratio=round(ratio, 2),
                note="0.6~1.5배면 통과. v2 방식에서는 약 6배였다.", passed=bool(0.6 <= ratio <= 1.5))


def v11_decision_rule():
    """판정 규칙의 자기일관성. (a) 같은 시나리오를 이름만 바꿔 넣으면 서로 지배하지 않는다.
    (b) 비용만 5% 더 드는 복제 시나리오는 반드시 제외된다. (c) 같은 입력이면 판정이 같다(재현성)."""
    import copy
    import decision as D
    from engine_retirement_v2 import monte_carlo
    prof = CompanyProfile(n_employees=400)
    a = monte_carlo(prof, Scenario("A"), runs=20, horizon=6)
    twin = copy.deepcopy(a); twin["summary"]["scenario"] = "A-복제"
    worse = copy.deepcopy(a); worse["summary"]["scenario"] = "A-비싼"
    worse["runs"]["cost_npv"] = worse["runs"]["cost_npv"] * 1.05
    worse["summary"]["cost_npv_mean"] *= 1.05
    f1, d1, _ = D.statistical_frontier([a, twin, worse])
    f2, d2, _ = D.statistical_frontier([a, twin, worse])
    ok = (set(f1) == {"A", "A-복제"}) and d1 == ["A-비싼"] and (f1, d1) == (f2, d2)
    a2 = monte_carlo(prof, Scenario("A"), runs=20, horizon=6)
    same = bool(abs(a2["summary"]["cost_npv_mean"] - a["summary"]["cost_npv_mean"]) < 1e-6)
    return dict(test="V11 판정 규칙 자기일관성·재현성", method="computed", frontier=f1, dominated=d1,
                reproducible=same, passed=bool(ok and same))


ALL_TESTS = (v1_calibration, v2_conservation, v3_did, v4_external, v5_refit,
             v6_retirement_id, v7_cross_source, v8_pretrend,
             v9_admin_benchmark, v10_baseline_continuity, v11_decision_rule)


def summary(results):
    computed = [r for r in results if r["method"] == "computed"]
    documented = [r for r in results if r["method"] == "documented"]
    diagnostic = [r for r in results if r["method"] == "diagnostic"]
    pt = next((r for r in results if r["test"].startswith("V8")), None)
    diag = ""
    if diagnostic:
        verdict = ("평행추세 미성립 → DiD는 인과효과가 아니라 방향 정합성으로만 해석"
                   if pt and not pt["parallel_trends_ok"] else "가정 성립")
        diag = f" + {len(diagnostic)}종 가정 검정(결과: {verdict})"
    return dict(
        label=f"{sum(r['passed'] for r in computed)}/{len(computed)}종 자동 계산 통과 "
              f"+ {len(documented)}종 근거 문서화(재계산 불가)" + diag,
        computed_all_passed=all(r["passed"] for r in computed),
        computed_n=len(computed), documented_n=len(documented),
        diagnostic_n=len(diagnostic),
        parallel_trends_ok=(pt["parallel_trends_ok"] if pt else None))


if __name__ == "__main__":
    results = [fn() for fn in ALL_TESTS]
    for r in results:
        print(json.dumps(r, ensure_ascii=False))
    print("---")
    print(json.dumps(summary(results), ensure_ascii=False))
