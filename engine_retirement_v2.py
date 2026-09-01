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
     ※ 단 이 DiD는 평행추세 가정을 통과하지 못한다(V8) — 인과효과가 아니라
       '제도 변화와 같은 방향'이라는 정합성 확인으로만 쓸 것.
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
BASE_COEF = dict(age=B_AGE, tenure=B_TENURE, wage=B_WAGE, hours=B_HOURS)

PARAMS = json.loads((_DIR / "params_v2.json").read_text(encoding="utf-8"))
_SAMPLE_PATH = _DIR / "big_firm_sample_2023.csv"

# KOSIS 2025 연령-임금 지수 (정액급여, 전체=1.0)
AGE_WAGE_INDEX = [(0, 29, 0.717), (30, 39, 1.050), (40, 49, 1.192),
                  (50, 59, 1.113), (60, 99, 0.781)]

# 재고용(status=1) 상태의 이직위험 배수.
# 2026-08-28 측정 결과 이 값을 0.15~1.0으로 바꿔도 결과 지표가 사실상 변하지 않는다
# (재고용자는 고연령·장기근속이라 기저 hazard가 이미 0에 가까움). 즉 실질적으로 비활성
# 상수다. 그럼에도 코드에만 박혀 있으면 "출처 없는 가정"으로 보이므로 명시적으로 이름을
# 붙이고 근거 없음을 기록한다 — 이 프로젝트의 "미검증 채널은 효과 미반영" 원칙에 따라
# 값을 바꾸고 싶다면 근거를 먼저 확보할 것.
REHIRE_HAZARD_MULT = 0.3  # [근거 없음 · 영향 없음으로 측정됨]

# WPS2023 300인+ 사업체 신규채용률 중위값(연 10%). "채용 규모 고정" 모드의 기본값.
WPS_HIRE_RATE_MEDIAN = float(PARAMS["wps2023"]["신규채용률(연)"]["median"])

# ── 고령자 임금 궤적 레짐 ────────────────────────────────────────────────────
# 중요(2026-08-28 발견): 기존 코드는 KOSIS "연령별 임금" 곡선의 기울기를 개인의 연도별
# 임금상승률로 그대로 사용했다. 그러나 그 곡선은 **횡단면**(어느 한 해에 각 연령대가
# 얼마를 받는가)이지 **종단면**(한 사람의 임금이 나이 들며 어떻게 변하는가)이 아니다.
# 횡단면에서 50대 후반~60대 임금이 낮은 것은 상당 부분 (a) 임금피크제 적용자, (b) 이미
# 촉탁·재고용으로 전환된 사람, (c) 고임금자의 조기퇴직 선택(선택편의) 때문이다.
# 이를 개인 궤적으로 쓰면 "정년을 연장해도 그 사람 임금은 어차피 떨어진다"는 결론이
# 가정에서 바로 도출되어, 정년연장 비용을 체계적으로 과소평가한다. 한국 대기업의
# 호봉제는 정반대 방향(근속에 따라 상승)이고, 그것이 바로 기업이 정년연장을 우려하는 이유다.
#
# 이 배포본에는 종단면 임금 패널이 없어 어느 쪽이 옳은지 데이터로 판정할 수 없다.
# 따라서 가정을 코드에 숨기지 않고 사용자가 고르는 입력으로 노출한다(DSS 원칙).
# 측정된 영향(runs=200·horizon=12): 급진연장(S4)의 인건비가 현행유지 대비
# +0.72%(observed) / +1.31%(frozen) / +1.83%(seniority) — 약 2.5배 차이.
# 다만 파레토 프론티어 구성(S1·S3·S6·S7)은 세 레짐 모두에서 동일했다:
# 결론은 견고하고 비용 '규모'는 견고하지 않다.
WAGE_REGIMES = {
    "observed": {"label": "현재 임금 추세 그대로",
                 "rates": [(0, 29, 0.055), (30, 39, 0.040), (40, 49, 0.012),
                           (50, 59, -0.007), (60, 99, -0.030)],
                 "note": "KOSIS 연령-임금 곡선의 기울기. 임금피크·재고용 전환이 이미 섞인 값이라 "
                         "정년연장 비용을 낮게 볼 소지가 있음."},
    "frozen":   {"label": "50세 이후 임금 동결",
                 "rates": [(0, 29, 0.055), (30, 39, 0.040), (40, 49, 0.012),
                           (50, 59, 0.0), (60, 99, 0.0)],
                 "note": "50세 시점 임금이 정년까지 유지된다고 가정(임금피크 미도입 + 승급 정지)."},
    "seniority": {"label": "호봉제 유지(나이 들수록 계속 오름)",
                  "rates": [(0, 29, 0.055), (30, 39, 0.040), (40, 49, 0.025),
                            (50, 59, 0.020), (60, 99, 0.010)],
                  "note": "연공급이 정년까지 작동한다고 가정. 정년연장 비용의 상한 시나리오."},
}
DEFAULT_WAGE_REGIME = "observed"

GRADES = ["사원", "대리", "과장", "차장", "부장"]
GRADE_MIN_YEARS = {"사원": 4, "대리": 4, "과장": 5, "차장": 5, "부장": 99}
GRADE_TARGET_SHARE = {"사원": 0.32, "대리": 0.24, "과장": 0.20, "차장": 0.14, "부장": 0.10}
OVERHEAD = 1.19  # 퇴직급여 적립 1/12 + 4대보험 사업주부담 [문서화 가정]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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
    wage_regime: str = DEFAULT_WAGE_REGIME  # WAGE_REGIMES 참조 — 결론에 영향이 큰 가정이라 노출


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


def generate_workforce(profile: CompanyProfile, seed=RNG_SEED, coef=None):
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

    # 초기 재직상태: 이미 현행 정년 이상인 사람은 '정규 재직'이 아니라 '계속고용(촉탁·재고용)'
    # 상태로 시작한다. (2026-08-28 수정)
    # 이전 버전은 전원을 status=0으로 두어 두 가지 오류를 냈다:
    #  (1) 정년 60인 회사인데 초기 인구의 7.6%(1,000명 중 88명)가 이미 60세 이상이었고,
    #      이들이 1년차에 한꺼번에 정년 이벤트를 맞아 정년퇴직이 83명 → 2년차 13명으로
    #      약 6.4배 인위적으로 튀었다(신규채용 곡선도 같이 왜곡).
    #  (2) 이들의 표본 임금은 이미 재고용 수준이다(KLIPS 300인+ 실표본에서 55~59세 534만원
    #      vs 60~64세 340만원). 여기에 rehire_wage_ratio 0.55를 다시 곱하면 임금 삭감을
    #      이중으로 반영하게 된다.
    # status=1로 시작시키면 at_ret 대상에서 제외되어 두 문제가 함께 해소되고, 이들의 퇴장은
    # 각 시나리오의 계속고용 상한(ce_end_age)이 관장한다 — 현행유지(계속고용 없음)에서는
    # 곧바로 퇴장하고, 계속고용 의무 시나리오에서는 상한 연령까지 남는다.
    status = (age >= profile.current_retirement_age).astype(int)

    wf = dict(age=age, tenure=tenure, grade=grade, wage=wage, hours=hours, edu=edu,
              yrs_in_grade=yrs_in_grade, status=status)
    wf["_b0"] = _calibrate(wf, profile.base_push_turnover, coef=coef)
    wf["_entry_wage"] = float(profile.avg_wage * AGE_WAGE_INDEX[0][2] /
                              np.average([v for _, _, v in AGE_WAGE_INDEX[:4]]))
    return wf


def _assign_grades(tenure, rng):
    """근속 순위를 GRADE_TARGET_SHARE 비율에 맞춰 직급으로 배분한다.

    2026-08-28 교체. 기존에는 근속 구간(0/4/8/13/18년)만으로 직급을 정했는데, KLIPS 실표본은
    장기근속자가 많아 부장 232명·차장 155명이 나왔다(목표 비중은 각각 100명·140명).
    그 결과 상위 직급의 결원이 수년간 0이어서, 파레토 목적함수 중 하나인 승진대기 p90이
    정책 시나리오가 아니라 **초기화 방식**에 좌우되고 있었다.
    근속 순위 기반으로 바꾸면 초기 직급 구성이 목표 비중과 일치해, 승진 적체의 변화가
    시나리오 효과로만 나타난다. 순위에 약한 잡음을 섞어 근속-직급이 완전히 결정적이 되는 것은 피한다.
    """
    n = len(tenure)
    jitter = tenure + rng.normal(0, 1.5, n)          # 동일 근속자 간 순서를 흩뜨림
    order = np.argsort(jitter)                        # 근속 짧은 순
    g = np.empty(n, dtype=object)
    start = 0
    for gi, name in enumerate(GRADES):
        share = GRADE_TARGET_SHARE[name]
        end = n if gi == len(GRADES) - 1 else start + int(round(share * n))
        g[order[start:end]] = name
        start = end
    return g


def _cum_min_years(grades):
    cum = {"사원": 0, "대리": 4, "과장": 8, "차장": 13, "부장": 18}
    return np.array([cum[g] for g in grades], dtype=float)


def _hazard(wf, b0, coef=None):
    c = coef or BASE_COEF
    logit = (b0 + c["age"] * wf["age"] + c["tenure"] * wf["tenure"]
             + c["wage"] * wf["wage"] + c["hours"] * wf["hours"])
    return _sigmoid(logit)


def _calibrate(wf, target, lo=-15, hi=15, coef=None):
    for _ in range(60):
        mid = (lo + hi) / 2
        if _hazard(wf, mid, coef).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _annual_wage_growth(age, regime=DEFAULT_WAGE_REGIME):
    """개인의 연간 임금상승률. regime 선택 근거는 WAGE_REGIMES 주석 참조.
    기본값 'observed'는 KOSIS 횡단면 곡선의 기울기이며, 이를 개인 궤적으로 쓰는 것은
    알려진 편의(횡단면≠종단면)를 안고 있다 — 그래서 레짐을 바꿀 수 있게 해 두었다."""
    a = np.asarray(age, dtype=float)
    g = np.zeros_like(a)
    for lo, hi, rate in WAGE_REGIMES[regime]["rates"]:
        g[(a >= lo) & (a <= hi)] = rate
    return g


# ── 모수 불확실성: hazard 계수의 표집분포에서 추출 ──────────────────────────
# 기존 몬테카를로 구간은 '어떤 1,000명을 뽑았는가'(부트스트랩 표본변동)만 반영했다.
# 반복수를 25→400회로 늘려도 구간 폭이 4.3%→3.9%로 거의 줄지 않는 것이 그 증거다
# (단순 시뮬레이션 노이즈였다면 √n에 반비례해 좁아져야 한다).
# 여기에 '계수를 얼마나 정확히 추정했는가'(모수 불확실성)를 더한다.
# SE는 refit_coefficients.json의 p값에서 역산한다: z = Φ⁻¹(1-p/2), SE = |coef|/z.
# 재적합 표본(n=8,801·사건 574)은 주 계수의 대표본(사건 1,858)보다 작으므로 이 SE를
# 주 계수에 적용하면 실제보다 **넓은**(보수적인) 구간이 된다 — 과소평가보다 낫다고 판단.
def _coef_sd():
    try:
        from scipy.stats import norm
        r = json.loads((_DIR / "refit_coefficients.json").read_text(encoding="utf-8"))["coefficients"]
    except Exception:
        return None
    sd = {}
    for k, base in BASE_COEF.items():
        v = r.get(k)
        if not v or not (0 < v["p"] < 1) or v["coef"] == 0:
            return None
        z = float(norm.isf(v["p"] / 2))
        if z <= 0:
            return None
        sd[k] = abs(base) * (abs(v["coef"]) / z) / abs(v["coef"])  # 상대 SE를 주 계수에 적용
    return sd


_COEF_SD = _coef_sd()


def draw_coefficients(rng):
    """1회 시뮬레이션용 계수 벡터를 표집분포에서 추출. SD 산출 실패 시 점추정 그대로."""
    if _COEF_SD is None:
        return dict(BASE_COEF)
    return {k: float(rng.normal(v, _COEF_SD[k])) for k, v in BASE_COEF.items()}


# ═════════════════ 연 단위 시뮬레이션 ═════════════════
def simulate(profile: CompanyProfile, scen: Scenario,
             start_year=2026, horizon=12, seed=RNG_SEED, discount=0.03, coef=None):
    rng = np.random.default_rng(seed)
    wf = generate_workforce(profile, seed=seed, coef=coef)
    regime = getattr(profile, "wage_regime", DEFAULT_WAGE_REGIME)
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
        p_push = _hazard(dict(wf, wage=pay), b0, coef)
        p_push[wf["status"] == 1] *= REHIRE_HAZARD_MULT
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
        # ce_out도 push/기타 이직자를 제외해야 한다(2026-08-28 수정).
        # 제외하지 않으면 같은 사람이 '이직'과 '계속고용 종료'로 이중 계상되어
        # 기말인원 = 기초 − 이탈 + 채용 항등식이 깨진다. 기존에는 status=1인 사람이
        # 시뮬레이션 도중 재고용으로만 생겨 ce_age 도달까지 시차가 있었던 탓에
        # 우연히 겹치지 않았을 뿐, 조건이 조금만 달라지면 터지는 잠복 버그였다
        # (초기 인구에 계속고용자를 넣자마자 V2 인원보존 검증이 즉시 위반 5건을 잡아냈다).
        # ~rehired: 올해 막 재고용된 사람은 올해 계속고용 종료 대상이 아니다(2026-08-28 수정).
        # ce_end_age_schedule을 지정하지 않으면 ce_end_age()가 그 해의 정년으로 되돌아가는데,
        # 재고용 판정 직후 age >= ce_age가 곧바로 참이 되어 '재고용하자마자 같은 해에 내보내는'
        # 무한 무효화가 일어났다(n_ce_exit == n_rehired가 매년 성립). 인원보존 항등식은
        # 이 경우에도 성립하므로 V2가 잡지 못했고, 공교롭게 v2_conservation의 검증용
        # 시나리오 자체가 그런 시나리오였다. 프리셋 7종은 모두 스케줄을 지정해 영향이 없었으나,
        # 사용자가 직접 시나리오를 만들면 계속고용이 조용히 무력화되는 함정이었다.
        ce_out = (wf["status"] == 1) & (wf["age"] >= ce_age) & ~rehired & ~leave_push & ~leave_other

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
            # hiring_fixed_n을 지정하지 않으면 WPS2023 300인+ 신규채용률 중위값(연 10%)을 쓴다.
            # 기존에는 기본값 0이 그대로 적용돼 "채용 규모 고정"이 사실상 "채용 전면 중단"으로
            # 동작했다(2026-08-28 수정) — 결과가 극단적으로 왜곡되던 미사용 경로.
            n_hire = (scen.hiring_fixed_n if scen.hiring_fixed_n > 0
                      else int(round(profile.n_employees * WPS_HIRE_RATE_MEDIAN)))
        else:
            room = init_cost - float(wf["wage"].sum() * 12 * OVERHEAD)
            n_hire = max(int(room / (entry_wage * 12 * OVERHEAD)), 0)
        _hire(wf, profile, entry_wage, n_hire, rng)

        # aging + 임금 동학 (KOSIS 연령곡선 연률, 정규직만)
        wf["age"] += 1
        wf["tenure"] += 1
        wf["yrs_in_grade"] += 1
        reg = wf["status"] == 0
        wf["wage"][reg] *= (1 + _annual_wage_growth(wf["age"][reg], regime))

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


def monte_carlo(profile, scen, runs=200, horizon=12, start_year=2026, seed=RNG_SEED,
                param_uncertainty=True):
    """param_uncertainty=True면 반복마다 hazard 계수를 표집분포에서 새로 뽑는다.
    이때 산출 구간은 '인력 구성 표본변동 + 계수 추정 불확실성' 두 가지를 함께 반영한다.
    False면 계수를 점추정에 고정 — 시나리오 간 순수 비교(공통 난수)용."""
    all_runs = []
    for r in range(runs):
        coef = draw_coefficients(np.random.default_rng(10_000 + r)) if param_uncertainty else None
        all_runs.append(simulate(profile, scen, start_year, horizon, seed=seed + r, coef=coef))
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
        param_uncertainty=bool(param_uncertainty),
        wage_regime=getattr(profile, "wage_regime", DEFAULT_WAGE_REGIME),
        cost_npv_mean=float(npv.mean()), cost_npv_p5=float(np.percentile(npv, 5)),
        cost_npv_p95=float(np.percentile(npv, 95)),
        total_hires_mean=float(hires.mean()),
        exp55_person_years_mean=float(exp55.mean()),
        promo_congestion_mean=float(cong.mean()),
        total_push_exits_mean=float(push.mean())), path=mean_path)
