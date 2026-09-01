"""
engine_retirement_v2.py — 정년연장·계속고용 DSS 시뮬레이션 엔진 v2
====================================================================
v1 → v2 변경 (전부 데이터 근거 교체):
  1. 인구 생성: 가정 연령분포 → KLIPS 26차 300인+ 재직자 실표본 1,789명
     joint 부트스트랩 (나이·근속·임금·시간·학력 상관구조 보존)
  2. 임금 동학: 근속 단일 기울기 → KOSIS 고용형태별근로실태조사 2025
     연령-임금 곡선(정액급여 지수: 29세↓ 0.717 / 30대 1.050 / 40대 1.192 /
     50대 1.113 / 60세↑ 0.781) + 근속 프리미엄 결합
  3. 기타 이직률: 가정 3% → 자체 실측 (26→27차 전이: 비push 이직 — 기업
     프로파일 입력으로 조정 가능)
  4. hazard 계수: v3 대표본 실측 유지 + 자체 재적합(사건 574) 표본외 검증 통과
     (5/5 방향 일치; 근로시간·학력은 소표본 비유의 → 민감도 폭 확대 대상)
  5. 검증: 2016년 60세 의무화 DiD (-4.0%p, p=0.005) 근거 통합 (validation_v2)
파라미터 출처: params_v2.json / big_firm_sample_2023.csv
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

RNG_SEED = 42
_DIR = Path(__file__).parent

# ── 실측 계수 (v3 대표본; params_v2.json에 재적합 검증 기록) ──
B_AGE, B_TENURE, B_WAGE, B_HOURS = -0.0350, -0.1044, -0.0031, +0.0135

PARAMS = json.loads((_DIR / "params_v2.json").read_text(encoding="utf-8"))
_SAMPLE_PATH = _DIR / "big_firm_sample_2023.csv"

# KOSIS 2025 연령-임금 지수 (정액급여, 전체=1.0)
AGE_WAGE_INDEX = [(0, 29, 0.717), (30, 39, 1.050), (40, 49, 1.192),
                  (50, 59, 1.113), (60, 99, 0.781)]

GRADES = ["사원", "대리", "과장", "차장", "부장"]
GRADE_MIN_YEARS = {"사원": 4, "대리": 4, "과장": 5, "차장": 5, "부장": 99}
GRADE_TARGET_SHARE = {"사원": 0.32, "대리": 0.24, "과장": 0.20, "차장": 0.14, "부장": 0.10}
OVERHEAD = 1.19  # 퇴직급여 적립 1/12 + 4대보험 사업주부담 [문서화 가정]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _age_curve(age):
    out = np.full_like(np.asarray(age, dtype=float), 1.0)
    for lo, hi, v in AGE_WAGE_INDEX:
        out[(age >= lo) & (age <= hi)] = v
    return out


@dataclass
class CompanyProfile:
    n_employees: int = 1000
    avg_wage: float = 465.0        # 월평균임금(만원). KOSIS 300인+ 급여 513만/총액 632만,
                                   # KLIPS 실표본 429만 — 중간값 기본, 자사값 입력 권장
    base_push_turnover: float = 0.0148   # 대기업 push형 이직률 (KLIPS 실측)
    other_exit_rate: float = 0.045       # 비자발+생애주기 (자체 실측 7.5%는 전규모;
                                         # 대기업 하향 조정, 민감도 대상)
    current_retirement_age: int = 60
    entry_age_mean: float = 27.0
    hiring_mode_default: str = "replace"


@dataclass
class Scenario:
    name: str
    retirement_schedule: dict = field(default_factory=lambda: {2026: 60})
    continued_type: str = "none"     # none | rehire | job_keep | flexible
    ce_end_age_schedule: dict = field(default_factory=dict)
    rehire_wage_ratio: float = 0.55  # KLIPS 실측: 재취업 임금비 중위 0.51~평균 0.59
    rehire_accept: float = 0.62      # KLIPS 앵커: [1년내 0.374, 전체 0.621]
    wage_peak: bool = False
    peak_years: int = 3
    peak_cut: float = 0.10
    hiring_mode: str = "replace"     # replace | fixed | budget
    hiring_fixed_n: int = 0
    edu_invest_per_head: float = 0.0  # 비용만 반영 (효과 채널 미검증)

    def retirement_age(self, year):
        ys = sorted(y for y in self.retirement_schedule if y <= year)
        return self.retirement_schedule[ys[-1]] if ys else 60

    def ce_end_age(self, year):
        if not self.ce_end_age_schedule:
            return self.retirement_age(year)
        ys = sorted(y for y in self.ce_end_age_schedule if y <= year)
        return self.ce_end_age_schedule[ys[-1]] if ys else self.retirement_age(year)


# ═════════════════ 인구: KLIPS 실표본 joint 부트스트랩 ═════════════════
_BASE = pd.read_csv(_SAMPLE_PATH) if _SAMPLE_PATH.exists() else None


def generate_workforce(profile: CompanyProfile, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = profile.n_employees
    if _BASE is not None:
        idx = rng.integers(0, len(_BASE), size=n)
        s = _BASE.iloc[idx].reset_index(drop=True)
        age = s.age.to_numpy(float)
        tenure = s.tenure.to_numpy(float)
        wage = s.wage.to_numpy(float) * (profile.avg_wage / _BASE.wage.mean())
        hours = s.hours.to_numpy(float)
        edu = s.edu.to_numpy(float)
    else:  # fallback (실표본 파일 없을 때만)
        age = np.clip(rng.normal(43.4, 11, n), 22, 65)
        tenure = np.clip(rng.normal(11, 9, n), 0, 40)
        wage = np.clip(rng.normal(profile.avg_wage, profile.avg_wage * .45, n), 150, 3000)
        hours = np.clip(rng.normal(42, 6, n), 30, 70)
        edu = np.clip(rng.normal(5, 1.3, n), 1, 8)

    grade = _assign_grades(tenure, rng)
    yrs_in_grade = np.clip(tenure - _cum_min_years(grade), 0, None)
    wf = dict(age=age, tenure=tenure, grade=grade, wage=wage, hours=hours, edu=edu,
              yrs_in_grade=yrs_in_grade, status=np.zeros(n, dtype=int))
    wf["_b0"] = _calibrate(wf, profile.base_push_turnover)
    wf["_entry_wage"] = float(profile.avg_wage * AGE_WAGE_INDEX[0][2] /
                              np.average([v for _, _, v in AGE_WAGE_INDEX[:4]]))
    return wf


def _assign_grades(tenure, rng):
    cuts = [0, 4, 8, 13, 18]
    g = np.empty(len(tenure), dtype=object)
    for i, t in enumerate(tenure):
        k = sum(t >= c for c in cuts) - 1
        k = min(k + (rng.random() < 0.15) - (rng.random() < 0.15), len(GRADES) - 1)
        g[i] = GRADES[max(int(k), 0)]
    return g


def _cum_min_years(grades):
    cum = {"사원": 0, "대리": 4, "과장": 8, "차장": 13, "부장": 18}
    return np.array([cum[g] for g in grades], dtype=float)


def _hazard(wf, b0):
    logit = (b0 + B_AGE * wf["age"] + B_TENURE * wf["tenure"]
             + B_WAGE * wf["wage"] + B_HOURS * wf["hours"])
    return _sigmoid(logit)


def _calibrate(wf, target, lo=-15, hi=15):
    for _ in range(60):
        mid = (lo + hi) / 2
        if _hazard(wf, mid).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _annual_wage_growth(age):
    """KOSIS 연령-임금 곡선의 연간 상승률 근사 (곡선 기울기 → 연률)."""
    g = np.zeros_like(np.asarray(age, dtype=float))
    g[(age < 30)] = 0.055
    g[(age >= 30) & (age < 40)] = 0.040
    g[(age >= 40) & (age < 50)] = 0.012
    g[(age >= 50) & (age < 60)] = -0.007
    g[(age >= 60)] = -0.030
    return g


# ═════════════════ 연 단위 시뮬레이션 ═════════════════
def simulate(profile: CompanyProfile, scen: Scenario,
             start_year=2026, horizon=12, seed=RNG_SEED, discount=0.03):
    rng = np.random.default_rng(seed)
    wf = generate_workforce(profile, seed=seed)
    b0 = wf["_b0"]
    entry_wage = wf["_entry_wage"]
    init_cost = float(wf["wage"].sum() * 12 * OVERHEAD)
    rows = []

    for step in range(horizon):
        year = start_year + step
        ret_age, ce_age = scen.retirement_age(year), scen.ce_end_age(year)
        n = len(wf["age"])

        pay = wf["wage"].copy()
        if scen.wage_peak:
            to_ret = ret_age - wf["age"]
            for k in range(scen.peak_years):
                m = (to_ret == k + 1) & (wf["status"] == 0)
                pay[m] *= (1 - scen.peak_cut * (scen.peak_years - k))

        labor_cost = float(pay.sum() * 12 * OVERHEAD) + n * scen.edu_invest_per_head

        # push 이직 (임금피크 감액분이 임금채널로 hazard에 자동 반영)
        p_push = _hazard(dict(wf, wage=pay), b0)
        p_push[wf["status"] == 1] *= 0.3
        leave_push = rng.random(n) < p_push
        leave_other = (rng.random(n) < profile.other_exit_rate) & ~leave_push

        at_ret = (wf["age"] >= ret_age) & (wf["status"] == 0) & ~leave_push & ~leave_other
        n_retire = int(at_ret.sum())
        rehired = np.zeros(n, dtype=bool)
        if scen.continued_type != "none" and n_retire:
            rehired = at_ret & (rng.random(n) < scen.rehire_accept)
            if scen.continued_type == "rehire":
                wf["wage"][rehired] *= scen.rehire_wage_ratio
            elif scen.continued_type == "job_keep":
                wf["wage"][rehired] *= max(scen.rehire_wage_ratio, 0.85)
            else:
                wf["wage"][rehired] *= rng.uniform(scen.rehire_wage_ratio, 0.9, rehired.sum())
            wf["status"][rehired] = 1
        exit_retire = at_ret & ~rehired
        ce_out = (wf["status"] == 1) & (wf["age"] >= ce_age)

        keep = ~(leave_push | leave_other | exit_retire | ce_out)
        stats = dict(n_push=int(leave_push.sum()), n_other=int(leave_other.sum()),
                     n_retire=n_retire, n_rehired=int(rehired.sum()),
                     n_ce_exit=int(ce_out.sum()))
        exp55 = int((wf["age"] >= 55).sum())
        for k in wf:
            if isinstance(wf[k], np.ndarray):
                wf[k] = wf[k][keep]

        n_promoted, congestion = _promote(wf, rng)

        n_now = len(wf["age"])
        if scen.hiring_mode == "replace":
            n_hire = max(profile.n_employees - n_now, 0)
        elif scen.hiring_mode == "fixed":
            n_hire = scen.hiring_fixed_n
        else:
            room = init_cost - float(wf["wage"].sum() * 12 * OVERHEAD)
            n_hire = max(int(room / (entry_wage * 12 * OVERHEAD)), 0)
        _hire(wf, profile, entry_wage, n_hire, rng)

        # aging + 임금 동학 (KOSIS 연령곡선 연률, 정규직만)
        wf["age"] += 1
        wf["tenure"] += 1
        wf["yrs_in_grade"] += 1
        reg = wf["status"] == 0
        wf["wage"][reg] *= (1 + _annual_wage_growth(wf["age"][reg]))

        rows.append(dict(year=year, headcount=int(n_now + n_hire),
                         retirement_age=ret_age,
                         labor_cost=labor_cost,
                         labor_cost_pv=labor_cost / (1 + discount) ** step,
                         n_hire=int(n_hire), n_promoted=n_promoted,
                         promo_wait_p90=congestion,
                         avg_age=float(wf["age"].mean()) if len(wf["age"]) else 0,
                         share_55plus=exp55 / max(n, 1),
                         exp55_person_years=exp55, **stats))
    return rows


def _promote(wf, rng):
    n = len(wf["age"])
    if n == 0:
        return 0, 0.0
    reg = wf["status"] == 0
    promoted, waits = 0, []
    for gi in range(len(GRADES) - 1):
        g, gn = GRADES[gi], GRADES[gi + 1]
        vac = max(int(GRADE_TARGET_SHARE[gn] * n - (wf["grade"] == gn).sum()), 0)
        elig = np.where(reg & (wf["grade"] == g) &
                        (wf["yrs_in_grade"] >= GRADE_MIN_YEARS[g]))[0]
        if len(elig):
            waits.extend((wf["yrs_in_grade"][elig] - GRADE_MIN_YEARS[g]).tolist())
        if vac and len(elig):
            chosen = elig[np.argsort(-wf["yrs_in_grade"][elig])][:vac]
            wf["grade"][chosen] = gn
            wf["yrs_in_grade"][chosen] = 0
            promoted += len(chosen)
    return promoted, (float(np.percentile(waits, 90)) if waits else 0.0)


def _hire(wf, profile, entry_wage, n_hire, rng):
    if n_hire <= 0:
        return
    new = dict(age=np.clip(rng.normal(profile.entry_age_mean, 2, n_hire), 22, 35).astype(int).astype(float),
               tenure=np.zeros(n_hire),
               grade=np.array(["사원"] * n_hire, dtype=object),
               wage=np.full(n_hire, entry_wage),
               hours=np.clip(rng.normal(42, 4, n_hire), 35, 60),
               edu=np.full(n_hire, 6.0),
               yrs_in_grade=np.zeros(n_hire),
               status=np.zeros(n_hire, dtype=int))
    for k in new:
        wf[k] = np.concatenate([wf[k], new[k]])


def monte_carlo(profile, scen, runs=200, horizon=12, start_year=2026, seed=RNG_SEED):
    all_runs = [simulate(profile, scen, start_year, horizon, seed=seed + r)
                for r in range(runs)]
    years = [r["year"] for r in all_runs[0]]
    mean_path = []
    for i, y in enumerate(years):
        row = {"year": y}
        for k in all_runs[0][0]:
            if k != "year":
                row[k] = float(np.mean([run[i][k] for run in all_runs]))
        mean_path.append(row)

    def tot(key):
        return np.array([sum(r[key] for r in run) for run in all_runs])

    npv, hires = tot("labor_cost_pv"), tot("n_hire")
    exp55, push = tot("exp55_person_years"), tot("n_push")
    cong = np.array([np.mean([r["promo_wait_p90"] for r in run]) for run in all_runs])
    return dict(summary=dict(
        scenario=scen.name, runs=runs, horizon=horizon,
        cost_npv_mean=float(npv.mean()), cost_npv_p5=float(np.percentile(npv, 5)),
        cost_npv_p95=float(np.percentile(npv, 95)),
        total_hires_mean=float(hires.mean()),
        exp55_person_years_mean=float(exp55.mean()),
        promo_congestion_mean=float(cong.mean()),
        total_push_exits_mean=float(push.mean())), path=mean_path)
