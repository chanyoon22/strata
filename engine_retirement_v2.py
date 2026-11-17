"""
engine_retirement_v2.py — 정년연장·계속고용 DSS 시뮬레이션 엔진 (엔진 v3, 2026-11)
====================================================================
파일명은 배포 호환을 위해 유지하되 내용은 v3다.

v2 → v3 변경 요약 (근거는 design_doc.md 15장)
  1. 가상 기업 전용 → **실제 인사 명부 구동**. CompanyProfile.roster에 표준화된 명부를
     넣으면 그 인원 그대로(resample=False) 시뮬레이션한다. 직급 체계·부서·고용형태·
     신입 임금도 명부에서 가져온다. 명부가 없으면 종전처럼 KLIPS 300인+ 실표본을
     부트스트랩한다(체험용).
  2. **현행 관행 모델링**. v2는 초기 인구의 60세 이상(표본의 7.6%)을 '현행유지'에서
     1년차에 전원 내보냈다 — 기준선이 현재를 재현하지 못해 모든 비교가 기준선에 유리하게
     기울었다. v3는 회사의 현재 재고용 관행(비율·상한 연령·임금률)을 프로파일에 두고,
     어떤 시나리오에서도 '법이 요구하는 수준과 현재 관행 중 높은 쪽'을 적용한다.
  3. **재고용자의 직위 반납**. v2는 재고용자가 부장 자리를 그대로 차지해 승진을 막았다.
     실제 촉탁 재고용은 보직을 내려놓는 것이 일반적이고, 그것이 '정년연장'과 '재고용'을
     가르는 핵심 차이다. v3는 재고용(rehire)이면 직위를 비우고, 직무유지형(job_keep)과
     정년연장은 직위를 유지한다.
  4. 기업이 실제로 쓰는 **조정 수단**을 시나리오 레버로 추가: 임금체계 개편(특정 연도부터
     임금 궤적 변경), 희망퇴직 프로그램(대상 연령·수락률·위로금), 교육투자비.
  5. **분해 가능한 산출**: 인건비를 정규(기존 정년 미만)/연장 재직/재고용/임금피크 절감/
     희망퇴직 위로금으로 분해하고, 연령대·부서별 인원을 매년 기록한다.
  6. **근로자 측 지표**: 퇴장 시점부터 국민연금 수급 개시까지의 '소득 공백 인년'.
  7. 몬테카를로가 **반복별 원자료**를 돌려준다 — 시나리오 간 대응표본 검정과
     후보 판정 신뢰도 계산(decision.py)에 쓴다.
파라미터 출처: params_v2.json / big_firm_sample_2023.csv
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

ENGINE_VERSION = "3.1"
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

# 재고용(status=1) 상태의 이직위험 배수. 0.15~1.0으로 바꿔도 결과 지표가 사실상 변하지
# 않는 것으로 측정된 비활성 상수다(재고용자는 고연령·장기근속이라 기저 hazard가 0에 가깝다).
REHIRE_HAZARD_MULT = 0.3  # [근거 없음 · 영향 없음으로 측정됨]

# WPS2023 300인+ 사업체 신규채용률 중위값(연 10%). "채용 규모 고정" 모드의 기본값.
WPS_HIRE_RATE_MEDIAN = float(PARAMS["wps2023"]["신규채용률(연)"]["median"])

# ── 고령자 임금 궤적 레짐 ────────────────────────────────────────────────────
# KOSIS "연령별 임금" 곡선은 횡단면이지 개인의 종단 궤적이 아니다. 50대 후반~60대 임금이
# 낮은 것은 상당 부분 임금피크 적용자·재고용 전환자·고임금자 조기퇴직(선택편의) 때문이다.
# 이를 개인 궤적으로 쓰면 정년연장 비용을 과소평가한다. 종단 임금 패널이 없어 어느 쪽이
# 옳은지 판정할 수 없으므로 가정을 숨기지 않고 사용자가 고르게 한다. 'custom'은 회사가
# 자사 승급률을 직접 넣는 경우다(CompanyProfile.custom_wage_rates).
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

# 기본 직급 체계(명부에 직급이 없을 때만 사용)
GRADES = ["사원", "대리", "과장", "차장", "부장"]
GRADE_MIN_YEARS = {"사원": 4, "대리": 4, "과장": 5, "차장": 5, "부장": 99}
GRADE_TARGET_SHARE = {"사원": 0.32, "대리": 0.24, "과장": 0.20, "차장": 0.14, "부장": 0.10}
OVERHEAD = 1.19  # 퇴직급여 적립 1/12 + 4대보험 사업주부담 [문서화 가정]

NO_GRADE = -1    # 직위가 없는 상태(재고용으로 보직을 내려놓은 경우)


def pension_age(birth_year):
    """국민연금 노령연금 수급 개시 연령(출생연도 기준, 국민연금법 부칙).
    1953~56년생 61세 → 1957~60년생 62세 → 1961~64년생 63세 → 1965~68년생 64세 → 1969년생~ 65세."""
    b = np.asarray(birth_year, dtype=float)
    out = np.full_like(b, 65.0)
    out[b <= 1968] = 64
    out[b <= 1964] = 63
    out[b <= 1960] = 62
    out[b <= 1956] = 61
    out[b <= 1952] = 60
    return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class GradeSystem:
    """직급 체계. names는 낮은 직급 → 높은 직급 순서.
    min_years는 다음 직급으로 승진하기 위한 최소 체류 연수(최상위 직급은 무시).
    target_share는 직위 보유 인원 대비 각 직급의 정원 비중."""
    names: list = field(default_factory=lambda: list(GRADES))
    min_years: list = field(default_factory=lambda: [GRADE_MIN_YEARS[g] for g in GRADES])
    target_share: list = field(default_factory=lambda: [GRADE_TARGET_SHARE[g] for g in GRADES])

    def __post_init__(self):
        if not (len(self.names) == len(self.min_years) == len(self.target_share)):
            raise ValueError("직급 체계의 이름·최소연수·정원비중 길이가 서로 다릅니다.")
        tot = float(sum(self.target_share))
        if tot <= 0:
            raise ValueError("직급 정원 비중의 합이 0입니다.")
        self.target_share = [float(s) / tot for s in self.target_share]

    @property
    def n(self):
        return len(self.names)

    def cum_min_years(self):
        cum, out = 0.0, []
        for m in self.min_years:
            out.append(cum)
            cum += float(m) if m < 90 else 0.0
        return np.array(out)


# 직군별 기본값. 근거 있는 산업·직군별 수치가 없으므로 모든 직군에 같은 '자리표시' 값을 쓴다.
# 회사가 자기 직군의 충원 기간·숙련 기간을 넣으면 그 값으로 바뀐다. 결론은 손익분기
# (숙련 공백 1인-년의 가치가 평균 인건비의 몇 % 이상이면 판단이 바뀌는가)로 함께 보여준다.
JOB_DEFAULTS = dict(vacancy_months=3.0, ttp_years=2.0, ce_accept_rel=1.0, ce_wage_rel=1.0,
                    turnover_offset=0.0)
# 후임자는 적응 기간 동안 생산성이 0에서 100%까지 직선으로 오른다고 본다 → 평균 부족분 50%.
RAMP_GAP = 0.5


@dataclass
class CompanyProfile:
    n_employees: int = 1000
    avg_wage: float = 465.0        # 월평균임금(만원). 체험용 표본을 이 수준으로 맞춘다
    base_push_turnover: float = 0.0148   # 대기업 push형 이직률 (KLIPS 실측)
    other_exit_rate: float = 0.045       # 비자발+생애주기 (자체 실측 7.5%는 전규모; 하향 조정)
    current_retirement_age: int = 60
    entry_age_mean: float = 27.0
    hiring_mode_default: str = "replace"
    wage_regime: str = DEFAULT_WAGE_REGIME
    custom_wage_rates: list | None = None    # wage_regime="custom"일 때 [(lo, hi, rate), ...]
    # ── 현재 재고용 관행(기준선이 현재를 재현하도록) ──
    ce_rate_now: float = 0.0       # 지금 정년 도달자 중 재고용되는 비율
    ce_end_age_now: int = 60       # 지금 재고용의 상한 연령
    ce_wage_ratio_now: float = 0.55
    # ── 실제 명부 구동 ──
    roster: pd.DataFrame | None = None   # 표준 열: age, tenure, wage[, hours, grade, status, dept]
    resample: bool = True          # True: 명부를 n_employees명으로 부트스트랩 / False: 명부 그대로
    grades: GradeSystem = field(default_factory=GradeSystem)
    dept_names: list = field(default_factory=lambda: ["전체"])
    entry_wage: float | None = None      # 신입 초임(월 만원). None이면 평균임금에서 유도
    entry_hours: float | None = None
    hazard_offsets: dict | None = None   # 근속 구간별 로짓 보정 {"bins":[...], "offsets":[...]}
    hire_pool: pd.DataFrame | None = None   # 최근 입사자(age, wage, hours, grade). 있으면 신규 인력을 여기서 뽑는다
    # ── 직군(부서)별 설정: dept_names와 같은 순서의 표. 없으면 모든 직군에 JOB_DEFAULTS ──
    #   vacancy_months  사람이 나간 뒤 충원까지 걸리는 개월
    #   ttp_years       새로 온 사람이 혼자 제 몫을 하기까지 걸리는 햇수(숙련 도달 기간)
    #   ce_accept_rel   재고용 비율(회사 평균 = 1)
    #   ce_wage_rel     재고용 임금 비율(회사 평균 = 1)
    #   turnover_offset 자발적 퇴사 로짓 보정(퇴사자 명부로 추정; 0 = 회사 평균)
    job_table: pd.DataFrame | None = None
    start_year: int = 2026
    name: str = "표준 대기업 표본"

    def job_arrays(self):
        """직군별 설정을 dept 인덱스 순서의 배열로 돌려준다. 빈칸·누락은 기본값."""
        n = len(self.dept_names)
        out = {k: np.full(n, v, dtype=float) for k, v in JOB_DEFAULTS.items()}
        jt = self.job_table
        if jt is not None and len(jt):
            for k in out:
                if k in jt:
                    v = pd.to_numeric(jt[k], errors="coerce").to_numpy(float)[:n]
                    out[k][:len(v)] = np.where(np.isfinite(v), v, out[k][:len(v)])
        lo_hi = dict(vacancy_months=(0, 36), ttp_years=(0, 20), ce_accept_rel=(0, 3),
                     ce_wage_rel=(0.3, 2.0), turnover_offset=(-3, 3))
        for k, (lo, hi) in lo_hi.items():
            out[k] = np.clip(out[k], lo, hi)
        return out

    def job_table_is_default(self):
        """직군별 숙련 설정(충원·숙련 기간)을 아무도 바꾸지 않았는가."""
        a = self.job_arrays()
        return bool(np.allclose(a["vacancy_months"], JOB_DEFAULTS["vacancy_months"]) and
                    np.allclose(a["ttp_years"], JOB_DEFAULTS["ttp_years"]))

    def wage_rates(self, regime=None):
        regime = regime or self.wage_regime
        if regime == "custom":
            if not self.custom_wage_rates:
                raise ValueError("wage_regime='custom'인데 custom_wage_rates가 비어 있습니다.")
            return self.custom_wage_rates
        return WAGE_REGIMES[regime]["rates"]


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
    # ── v3 레버 ──
    wage_reform_regime: str | None = None   # 지정하면 wage_reform_year부터 이 임금 궤적 적용
    wage_reform_year: int = 0
    erp_start: int = 0               # 희망퇴직 시행 첫 해(0=미시행)
    erp_end: int = 0
    erp_age_min: int = 50
    erp_age_max: int = 57
    erp_take_rate: float = 0.0       # 대상자 중 연간 수락 비율
    erp_severance_months: float = 24.0
    ce_fixed: bool = False           # True면 민감도 분석에서 재고용 수락률·임금률을 흔들지 않는다
    desc: str = ""
    source: str = ""

    def __post_init__(self):
        if self.continued_type not in ("none", "rehire", "job_keep", "flexible"):
            raise ValueError(f"알 수 없는 계속고용 유형: {self.continued_type}")
        if self.hiring_mode not in ("replace", "fixed", "budget"):
            raise ValueError(f"알 수 없는 채용 정책: {self.hiring_mode}")
        if not self.retirement_schedule:
            raise ValueError("정년 일정이 비어 있습니다.")
        if self.wage_peak and self.peak_cut * self.peak_years >= 1.0:
            raise ValueError("임금피크 감액률 × 적용 연수가 100% 이상입니다(임금이 0 이하가 됨).")
        if not (0.0 <= self.rehire_accept <= 1.0):
            raise ValueError("재고용 수락률은 0~1 사이여야 합니다.")
        if not (0.0 < self.rehire_wage_ratio <= 1.5):
            raise ValueError("재고용 임금률은 0보다 크고 1.5 이하여야 합니다.")
        if not (0.0 <= self.erp_take_rate <= 1.0):
            raise ValueError("희망퇴직 수락률은 0~1 사이여야 합니다.")

    def retirement_age(self, year):
        ys = sorted(y for y in self.retirement_schedule if y <= year)
        return self.retirement_schedule[ys[-1]] if ys else 60

    def ce_end_age(self, year):
        if not self.ce_end_age_schedule:
            return self.retirement_age(year)
        ys = sorted(y for y in self.ce_end_age_schedule if y <= year)
        return self.ce_end_age_schedule[ys[-1]] if ys else self.retirement_age(year)

    def final_retirement_age(self):
        return max(self.retirement_schedule.values())


# ═════════════════ 인구 생성 ═════════════════
_BASE = pd.read_csv(_SAMPLE_PATH) if _SAMPLE_PATH.exists() else None


def _base_frame(profile: CompanyProfile):
    if profile.roster is not None:
        return profile.roster
    return _BASE


def generate_workforce(profile: CompanyProfile, seed=RNG_SEED, coef=None):
    """초기 인력을 만든다.

    - profile.roster가 있고 resample=False → 명부의 인원을 그대로 쓴다(실제 기업 모드).
    - 그 외 → 원천 표본(명부 또는 KLIPS)을 n_employees명으로 joint 부트스트랩한다.
      나이·근속·임금·직급의 상관구조가 행 단위로 보존된다.
    """
    rng = np.random.default_rng(seed)
    base = _base_frame(profile)
    gs = profile.grades
    if base is not None:
        if profile.roster is not None and not profile.resample:
            s = base.reset_index(drop=True)
        else:
            idx = rng.integers(0, len(base), size=profile.n_employees)
            s = base.iloc[idx].reset_index(drop=True)
        n = len(s)
        age = s["age"].to_numpy(float).copy()
        tenure = s["tenure"].to_numpy(float).copy()
        wage = s["wage"].to_numpy(float).copy()
        if profile.roster is None:
            wage = wage * (profile.avg_wage / _BASE.wage.mean())
        hours = (s["hours"].to_numpy(float).copy() if "hours" in s else np.full(n, 40.0))
        dept = (s["dept"].to_numpy(int).copy() if "dept" in s else np.zeros(n, dtype=int))
        status = (s["status"].to_numpy(int).copy() if "status" in s
                  else (age > profile.current_retirement_age).astype(int))
        grade_in = s["grade"].to_numpy(int).copy() if "grade" in s else None
        yig_in = s["yrs_in_grade"].to_numpy(float).copy() if "yrs_in_grade" in s else None
    else:  # fallback (실표본 파일도 명부도 없을 때만)
        n = profile.n_employees
        age = np.clip(rng.normal(43.4, 11, n), 22, 65).round()
        tenure = np.clip(rng.normal(11, 9, n), 0, 40).round()
        wage = np.clip(rng.normal(profile.avg_wage, profile.avg_wage * .45, n), 150, 3000)
        hours = np.clip(rng.normal(42, 6, n), 30, 70)
        dept = np.zeros(n, dtype=int)
        status = (age > profile.current_retirement_age).astype(int)
        grade_in = yig_in = None

    reg = status == 0
    if grade_in is None:
        grade = np.full(n, NO_GRADE, dtype=int)
        grade[reg] = _assign_grades(tenure[reg], rng, gs)
    else:
        grade = grade_in
    # 계속고용(촉탁·재고용) 상태로 시작하는 사람은 직위를 갖지 않는다(v3 원칙 3).
    grade[~reg] = NO_GRADE

    if yig_in is None:
        yrs_in_grade = _estimate_yrs_in_grade(tenure, grade, gs, known=grade_in is not None)
    else:
        yrs_in_grade = yig_in

    # 이미 현재 관행의 상한 연령을 넘긴 계속고용자는 1~3년에 걸쳐 흩어져 퇴장한다.
    # 한 해에 몰아 내보내면 1년차 채용·비용 곡선에 인위적인 절벽이 생긴다(v2의 결함).
    ce_until = np.zeros(n)
    cap = max(profile.ce_end_age_now, profile.current_retirement_age)
    ce_until[status == 1] = cap          # 지금 있는 계속고용자는 회사 관행의 상한까지
    over = (status == 1) & (age > cap)
    if over.any():      # 올해~2년 뒤 사이에 흩어져 퇴장(상한 나이인 사람은 올해 말 정상 퇴장)
        ce_until[over] = age[over] + rng.integers(0, 3, size=int(over.sum()))

    wf = dict(age=age, tenure=tenure, grade=grade, wage=wage, hours=hours,
              yrs_in_grade=yrs_in_grade, status=status, dept=dept, ce_until=ce_until)
    jobs = profile.job_arrays()
    dept_off = jobs["turnover_offset"] if np.any(jobs["turnover_offset"] != 0) else None
    wf["_dept_off"] = dept_off
    wf["_jobs"] = jobs
    wf["_b0"] = _calibrate(wf, profile.base_push_turnover, coef=coef,
                           offsets=profile.hazard_offsets, dept_off=dept_off)
    if profile.entry_wage:
        wf["_entry_wage"] = float(profile.entry_wage)
    else:
        mean_w = float(wage[reg].mean()) if (profile.roster is not None and reg.any()) else profile.avg_wage
        wf["_entry_wage"] = float(mean_w * AGE_WAGE_INDEX[0][2] /
                                  np.average([v for _, _, v in AGE_WAGE_INDEX[:4]]))
    if profile.hire_pool is not None and len(profile.hire_pool):
        wf["_entry_wage"] = float(profile.hire_pool["wage"].mean())    # 예산형 채용의 1인당 비용 기준
    wf["_entry_hours"] = float(profile.entry_hours if profile.entry_hours
                               else (np.median(hours[reg]) if reg.any() else 40.0))
    wf["_dept_target"] = np.bincount(dept, minlength=len(profile.dept_names)).astype(float)
    return wf


def _assign_grades(tenure, rng, gs: GradeSystem):
    """근속 순위를 정원 비중에 맞춰 직급 코드로 배분한다(명부에 직급이 없을 때).
    순위에 약한 잡음을 섞어 근속-직급이 완전히 결정적이 되는 것은 피한다."""
    n = len(tenure)
    jitter = tenure + rng.normal(0, 1.5, n)
    order = np.argsort(jitter)
    g = np.empty(n, dtype=int)
    start = 0
    for gi in range(gs.n):
        end = n if gi == gs.n - 1 else start + int(round(gs.target_share[gi] * n))
        g[order[start:end]] = gi
        start = end
    return g


def _estimate_yrs_in_grade(tenure, grade, gs: GradeSystem, known: bool):
    """현 직급 체류 연수 추정.
    - 직급을 근속 순위로 배정한 경우: 근속 − 그 직급까지의 누적 최소연수.
    - 명부의 실제 직급인 경우: 근속 − (그 직급 내 근속의 하위 10% 값). 명부에 승진일이
      없을 때 쓰는 근사이며, 같은 직급 안에서 근속이 긴 사람일수록 오래 머문 것으로 본다."""
    out = np.zeros(len(tenure))
    if not known:
        cum = gs.cum_min_years()
        m = grade >= 0
        out[m] = np.clip(tenure[m] - cum[grade[m]], 0, None)
        return out
    for gi in range(gs.n):
        m = grade == gi
        if m.any():
            floor = np.percentile(tenure[m], 10) if gi > 0 else 0.0
            out[m] = np.clip(tenure[m] - floor, 0, None)
    return out


def _offset_vec(tenure, offsets):
    if not offsets:
        return 0.0
    bins = np.asarray(offsets["bins"], dtype=float)      # 구간 상한(마지막은 inf)
    idx = np.searchsorted(bins, tenure, side="left")
    return np.asarray(offsets["offsets"], dtype=float)[np.clip(idx, 0, len(bins) - 1)]


def _hazard(wf, b0, coef=None, offsets=None, dept_off=None):
    c = coef or BASE_COEF
    logit = (b0 + c["age"] * wf["age"] + c["tenure"] * wf["tenure"]
             + c["wage"] * wf["wage"] + c["hours"] * wf["hours"]
             + _offset_vec(wf["tenure"], offsets))
    if dept_off is not None and "dept" in wf:
        logit = logit + np.asarray(dept_off, dtype=float)[wf["dept"]]
    return _sigmoid(logit)


def _calibrate(wf, target, lo=-15, hi=15, coef=None, offsets=None, dept_off=None):
    """정규 재직자의 평균 자발적 퇴사 확률이 target이 되도록 절편을 맞춘다.
    직군별 보정(dept_off)은 직군 사이의 '상대' 차이만 바꾸고 회사 전체 수준은 target에 맞춘다."""
    reg = wf["status"] == 0 if "status" in wf else np.ones(len(wf["age"]), dtype=bool)
    if not reg.any():
        reg = np.ones(len(wf["age"]), dtype=bool)
    keys = ("age", "tenure", "wage", "hours") + (("dept",) if (dept_off is not None and "dept" in wf) else ())
    sub = {k: wf[k][reg] for k in keys}
    for _ in range(60):
        mid = (lo + hi) / 2
        if _hazard(sub, mid, coef, offsets, dept_off).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _annual_wage_growth(age, rates):
    a = np.asarray(age, dtype=float)
    g = np.zeros_like(a)
    for lo, hi, rate in rates:
        g[(a >= lo) & (a <= hi)] = rate
    return g


# ── 모수 불확실성: hazard 계수의 표집분포에서 추출 ──────────────────────────
# SE는 refit_coefficients.json의 p값에서 역산한다: z = Φ⁻¹(1-p/2), SE = |coef|/z.
# 재적합 표본(사건 574)은 주 계수의 대표본(사건 1,858)보다 작으므로 이 SE를 주 계수에
# 적용하면 실제보다 넓은(보수적인) 구간이 된다.
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
        sd[k] = abs(base) / z      # 상대 SE(1/z)를 주 계수에 적용
    return sd


_COEF_SD = _coef_sd()


def draw_coefficients(rng):
    """1회 시뮬레이션용 계수 벡터를 표집분포에서 추출. SD 산출 실패 시 점추정 그대로."""
    if _COEF_SD is None:
        return dict(BASE_COEF)
    return {k: float(rng.normal(v, _COEF_SD[k])) for k, v in BASE_COEF.items()}


# ═════════════════ 연 단위 시뮬레이션 ═════════════════
_ARRAY_KEYS = ("age", "tenure", "grade", "wage", "hours", "yrs_in_grade", "status", "dept", "ce_until")


def simulate(profile: CompanyProfile, scen: Scenario,
             start_year=None, horizon=12, seed=RNG_SEED, discount=0.03, coef=None,
             detail=False, return_state=False):
    """1회 시뮬레이션. 연도별 dict의 리스트를 돌려준다.
    detail=True면 부서별 배열(dept_*)도 함께 기록한다.
    return_state=True면 (연도별 기록, 마지막 해가 끝난 뒤의 인력 상태)를 돌려준다(과거 검증용)."""
    start_year = profile.start_year if start_year is None else start_year
    rng = np.random.default_rng(seed)
    wf = generate_workforce(profile, seed=seed, coef=coef)
    gs = profile.grades
    n_dept = len(profile.dept_names)
    base_ret = profile.current_retirement_age
    b0, entry_wage, entry_hours = wf["_b0"], wf["_entry_wage"], wf["_entry_hours"]
    dept_target = wf["_dept_target"]
    jobs, dept_off = wf["_jobs"], wf["_dept_off"]
    acc_rel, wage_rel = jobs["ce_accept_rel"], jobs["ce_wage_rel"]
    vac_y, ttp = jobs["vacancy_months"] / 12.0, jobs["ttp_years"]
    init_cost = float(wf["wage"].sum() * 12 * OVERHEAD)
    n_target = len(wf["age"])
    base_rates = profile.wage_rates()
    reform_rates = profile.wage_rates(scen.wage_reform_regime) if scen.wage_reform_regime else None
    scen_has_ce = scen.continued_type != "none"
    rows = []

    for step in range(horizon):
        year = start_year + step
        # 회사 정년이 이미 법정 일정보다 높으면 회사 정년을 따른다
        ret_age = max(scen.retirement_age(year), base_ret)
        # 재고용은 두 겹이다(v3 원칙 2).
        #  · 법정 의무: 시나리오가 그 해에 정년보다 높은 상한을 요구할 때만 켜진다.
        #  · 회사 관행: 지금 하던 재고용(비율·상한 연령)이 어떤 시나리오에서도 이어진다.
        # 수락 비율은 둘 중 높은 쪽이고, 관행으로 남는 사람은 그 부분집합이다. 의무로만 남은
        # 사람은 법정 상한에서, 관행으로 남은 사람은 회사 상한(ce_until)에서 나간다.
        scen_ce = scen.ce_end_age(year) if scen_has_ce else 0
        mandate_on = scen_has_ce and scen_ce > ret_age
        cur_on = profile.ce_rate_now > 0 and profile.ce_end_age_now > ret_age
        acc_cur = profile.ce_rate_now if cur_on else 0.0
        accept = max(scen.rehire_accept if mandate_on else 0.0, acc_cur)
        ce_age = max(scen_ce, profile.ce_end_age_now if cur_on else 0, ret_age)   # 기록용
        ctype = scen.continued_type if mandate_on else "rehire"
        ratio = scen.rehire_wage_ratio if mandate_on else profile.ce_wage_ratio_now
        n = len(wf["age"])
        reg = wf["status"] == 0

        pay = wf["wage"].copy()
        if scen.wage_peak:
            to_ret = ret_age - wf["age"]
            for k in range(scen.peak_years):
                m = (to_ret == k + 1) & reg
                pay[m] *= (1 - scen.peak_cut * (scen.peak_years - k))
        peak_saving = float((wf["wage"] - pay).sum() * 12 * OVERHEAD)

        # 나이 규약: age는 '그 해에 도달하는 만 나이'다(명부 기준연도 − 출생연도).
        # 정년 60세면 60세가 되는 해의 연말에 퇴직하므로, 그 해 임금은 정규 인건비에 든다.
        # 따라서 '연장 재직'은 기존 정년을 넘긴 해(age > 기존 정년)부터다.
        ext = reg & (wf["age"] > base_ret)
        cost_reg = float(pay[reg & ~ext].sum() * 12 * OVERHEAD)
        cost_ext = float(pay[ext].sum() * 12 * OVERHEAD)
        cost_ce = float(pay[~reg].sum() * 12 * OVERHEAD)
        cost_edu = n * scen.edu_invest_per_head

        # ① 자발적 퇴사(임금피크 감액분이 임금 채널로 hazard에 반영) ② 기타 퇴사
        p_push = _hazard(dict(wf, wage=pay), b0, coef, profile.hazard_offsets, dept_off)
        p_push[~reg] *= REHIRE_HAZARD_MULT
        leave_push = rng.random(n) < p_push
        leave_other = (rng.random(n) < profile.other_exit_rate) & ~leave_push
        gone = leave_push | leave_other

        # ③ 희망퇴직(회사가 여는 프로그램; 위로금은 일시 비용)
        leave_erp = np.zeros(n, dtype=bool)
        cost_erp = 0.0
        if scen.erp_take_rate > 0 and scen.erp_start <= year <= scen.erp_end:
            elig = reg & ~gone & (wf["age"] >= scen.erp_age_min) & (wf["age"] <= scen.erp_age_max)
            leave_erp = elig & (rng.random(n) < scen.erp_take_rate)
            cost_erp = float(wf["wage"][leave_erp].sum() * scen.erp_severance_months)
            gone = gone | leave_erp

        # ④ 정년 도달 → 재고용 또는 퇴직
        at_ret = (wf["age"] >= ret_age) & reg & ~gone
        n_retire = int(at_ret.sum())
        rehired = np.zeros(n, dtype=bool)
        if accept > 0 and n_retire:
            u = rng.random(n)
            # 직군별 재고용 비율·임금 비율은 회사 평균 대비 상대값으로 곱한다.
            # 관행으로 남는 사람(acc_cur)이 전체 재고용(accept)의 부분집합이라는 관계는 그대로 유지된다.
            rel = acc_rel[wf["dept"]]
            rehired = at_ret & (u < np.clip(accept * rel, 0, 1))
            wf["ce_until"][rehired & (u < np.clip(acc_cur * rel, 0, 1))] = profile.ce_end_age_now
            r_i = np.minimum(ratio * wage_rel[wf["dept"]], 1.0)
            if ctype == "rehire":
                wf["wage"][rehired] *= r_i[rehired]
                wf["grade"][rehired] = NO_GRADE          # 보직 반납
            elif ctype == "job_keep":
                wf["wage"][rehired] *= np.maximum(r_i[rehired], 0.85)  # 직무·직위 유지
            else:
                lo_r = np.minimum(r_i[rehired], 0.9)
                wf["wage"][rehired] *= lo_r + (0.9 - lo_r) * rng.random(int(rehired.sum()))
                wf["grade"][rehired] = NO_GRADE
            wf["status"][rehired] = 1
        exit_retire = at_ret & ~rehired
        # ⑤ 계속고용 종료(올해 막 재고용된 사람·이미 떠난 사람 제외)
        ce_cap = np.maximum(scen_ce, wf["ce_until"])
        ce_out = (wf["status"] == 1) & (wf["age"] >= ce_cap) & ~rehired & ~gone

        # 소득 공백: 정년퇴직·계속고용 종료로 나가는 사람이 연금 수급까지 남은 햇수
        out_old = exit_retire | ce_out
        gap = np.clip(pension_age(year - wf["age"][out_old]) - wf["age"][out_old], 0, None)
        income_gap = float(gap.sum())

        # 숙련 공백: 나가는 모든 사람(자발·기타·희망퇴직·정년·계속고용 종료)의 자리가 다시 제 몫을
        # 할 때까지의 손실을 '인-년'으로 센다. = 충원 기간 + 후임 적응기의 생산성 부족분
        # (RAMP_GAP × min(떠난 사람의 근속, 직군 숙련 기간)). 근속이 숙련 기간보다 짧은 사람은
        # 자신도 아직 다 익지 않았으므로 그만큼만 센다. 가치 환산은 그 직군 정규직의 평균 임금을
        # 생산성의 대용치로 쓴다(개인 임금은 연공 때문에 고령자일수록 부풀려져 있어 쓰지 않는다).
        exits = gone | exit_retire | ce_out
        dd = wf["dept"]
        sg_i = np.where(exits, vac_y[dd] + RAMP_GAP * np.minimum(wf["tenure"], ttp[dd]), 0.0)
        skill_gap = float(sg_i.sum())
        cnt = np.bincount(dd[reg], minlength=n_dept).astype(float)
        wsum = np.bincount(dd[reg], weights=wf["wage"][reg], minlength=n_dept)
        mean_all = float(wf["wage"][reg].mean()) if reg.any() else float(wf["wage"].mean())
        g_wage = np.where(cnt > 0, wsum / np.maximum(cnt, 1), mean_all)
        skill_value = float((sg_i * g_wage[dd]).sum() * 12 * OVERHEAD)
        skill_unit = mean_all * 12 * OVERHEAD          # 숙련 공백 1인-년 = 평균 인건비 1년(100%) 기준

        keep = ~(gone | exit_retire | ce_out)
        age0 = wf["age"]
        stats = dict(n_push=int(leave_push.sum()), n_other=int(leave_other.sum()),
                     n_erp=int(leave_erp.sum()), n_retire=n_retire,
                     n_rehired=int(rehired.sum()), n_ce_exit=int(ce_out.sum()))
        exp55 = int((age0 >= 55).sum())
        bands = dict(n_u35=int((age0 < 35).sum()), n_35_54=int(((age0 >= 35) & (age0 < 55)).sum()),
                     n_55_59=int(((age0 >= 55) & (age0 < 60)).sum()), n_60p=int((age0 >= 60).sum()),
                     n_ce=int((wf["status"] == 1).sum()) - int(rehired.sum()))
        drow = {}
        if detail:
            d = wf["dept"]
            drow = dict(dept_hc=np.bincount(d, minlength=n_dept).tolist(),
                        dept_55=np.bincount(d[age0 >= 55], minlength=n_dept).tolist(),
                        dept_out=np.bincount(d[out_old], minlength=n_dept).tolist(),
                        dept_cost=(np.bincount(d, weights=pay, minlength=n_dept) * 12 * OVERHEAD).tolist(),
                        dept_skill=np.bincount(d, weights=sg_i, minlength=n_dept).tolist())
        for k in _ARRAY_KEYS:
            wf[k] = wf[k][keep]

        n_now = len(wf["age"])
        if scen.hiring_mode == "replace":
            n_hire = max(n_target - n_now, 0)
        elif scen.hiring_mode == "fixed":
            n_hire = (scen.hiring_fixed_n if scen.hiring_fixed_n > 0
                      else int(round(n_target * WPS_HIRE_RATE_MEDIAN)))
        else:
            room = init_cost - float(wf["wage"].sum() * 12 * OVERHEAD)
            n_hire = max(int(room / (entry_wage * 12 * OVERHEAD)), 0)
        _hire(wf, profile, entry_wage, entry_hours, n_hire, rng, dept_target)
        # 승진은 충원 뒤에 한다. v2는 충원 전에 정원(인원 × 비중)을 계산해, 그해 퇴사자만큼
        # 정원이 일시적으로 줄어든 상태에서 결원을 세었다 — 상위 직급 결원이 매년 약 6%씩
        # 과소 계상되어 초기 수년간 승진이 거의 일어나지 않았다(1년차 승진 2명 수준).
        n_promoted, congestion = _promote(wf, gs)

        # 나이·근속 +1, 임금 동학(정규직만; 임금체계 개편 연도 이후엔 개편 궤적)
        wf["age"] += 1
        wf["tenure"] += 1
        wf["yrs_in_grade"] += 1
        regn = wf["status"] == 0
        rates = (reform_rates if (reform_rates is not None and year >= scen.wage_reform_year)
                 else base_rates)
        wf["wage"][regn] *= (1 + _annual_wage_growth(wf["age"][regn], rates))

        labor_cost = cost_reg + cost_ext + cost_ce + cost_edu + cost_erp
        rows.append(dict(year=year, headcount=int(n_now + n_hire), retirement_age=ret_age,
                         ce_end_age=ce_age,
                         labor_cost=labor_cost,
                         labor_cost_pv=labor_cost / (1 + discount) ** step,
                         cost_reg=cost_reg, cost_ext=cost_ext, cost_ce=cost_ce,
                         cost_erp=cost_erp, cost_edu=cost_edu, peak_saving=peak_saving,
                         n_hire=int(n_hire), n_promoted=n_promoted,
                         promo_wait_p90=congestion,
                         avg_age=float(age0.mean()) if n else 0.0,
                         share_55plus=exp55 / max(n, 1),
                         exp55_person_years=exp55, income_gap=income_gap,
                         skill_gap=skill_gap, skill_value=skill_value,
                         skill_value_pv=skill_value / (1 + discount) ** step, skill_unit=skill_unit,
                         **bands, **stats, **drow))
    if return_state:
        return rows, {k: wf[k] for k in _ARRAY_KEYS}
    return rows


def _promote(wf, gs: GradeSystem):
    """직급별 결원을 체류 연수가 긴 순으로 채운다. 정원은 '직위 보유 인원 × 정원 비중'."""
    n_pos = int((wf["grade"] >= 0).sum())
    if n_pos == 0:
        return 0, 0.0
    reg = wf["status"] == 0
    promoted, waits = 0, []
    for gi in range(gs.n - 1):
        vac = max(int(gs.target_share[gi + 1] * n_pos - (wf["grade"] == gi + 1).sum()), 0)
        elig = np.where(reg & (wf["grade"] == gi) &
                        (wf["yrs_in_grade"] >= gs.min_years[gi]))[0]
        if len(elig):
            waits.extend((wf["yrs_in_grade"][elig] - gs.min_years[gi]).tolist())
        if vac and len(elig):
            chosen = elig[np.argsort(-wf["yrs_in_grade"][elig])][:vac]
            wf["grade"][chosen] = gi + 1
            wf["yrs_in_grade"][chosen] = 0
            promoted += len(chosen)
    return promoted, (float(np.percentile(waits, 90)) if waits else 0.0)


def _hire(wf, profile, entry_wage, entry_hours, n_hire, rng, dept_target):
    if n_hire <= 0:
        return
    # 신규 인원은 결원이 큰 부서부터 채운다(부서 규모를 유지한다는 가정)
    cur = np.bincount(wf["dept"], minlength=len(dept_target)).astype(float)
    short = np.clip(dept_target - cur, 0, None)
    p = short / short.sum() if short.sum() > 0 else dept_target / max(dept_target.sum(), 1)
    pool = profile.hire_pool
    if pool is not None and len(pool):
        # 신규 인력은 '최근 입사자'의 실제 분포에서 뽑는다(경력 채용이 섞인 회사의 현실을 반영).
        # v2는 전원을 27세 안팎·최하위 직급·단일 초임으로 넣어, 시간이 갈수록 인력이 실제보다
        # 빠르게 젊어지고(15년 뒤 35세 미만 26%→41%) 신입 인건비가 과소 계상됐다.
        k = rng.integers(0, len(pool), n_hire)
        h_age = pool["age"].to_numpy(float)[k]
        h_wage = pool["wage"].to_numpy(float)[k]
        h_hours = pool["hours"].to_numpy(float)[k] if "hours" in pool else np.full(n_hire, entry_hours)
        h_grade = pool["grade"].to_numpy(int)[k] if "grade" in pool else np.zeros(n_hire, dtype=int)
    else:
        h_age = np.clip(rng.normal(profile.entry_age_mean, 2, n_hire), 20, 45).round()
        h_wage, h_hours = np.full(n_hire, entry_wage), np.full(n_hire, entry_hours)
        h_grade = np.zeros(n_hire, dtype=int)
    new = dict(age=h_age,
               tenure=np.zeros(n_hire),
               grade=h_grade,
               wage=h_wage,
               hours=h_hours,
               yrs_in_grade=np.zeros(n_hire),
               status=np.zeros(n_hire, dtype=int),
               dept=rng.choice(len(dept_target), size=n_hire, p=p),
               ce_until=np.zeros(n_hire))
    for k in new:
        wf[k] = np.concatenate([wf[k], new[k]])


# 요약 지표: (요약 키, 연도별 키, 집계 방식)
_TOTALS = [("cost_npv", "labor_cost_pv", "sum"), ("total_hires", "n_hire", "sum"),
           ("exp55_person_years", "exp55_person_years", "sum"),
           ("promo_congestion", "promo_wait_p90", "mean"),
           ("income_gap", "income_gap", "sum"), ("total_push_exits", "n_push", "sum"),
           ("total_promoted", "n_promoted", "sum"), ("cost_nominal", "labor_cost", "sum"),
           ("skill_gap", "skill_gap", "sum"), ("skill_value_npv", "skill_value_pv", "sum")]


def monte_carlo(profile, scen, runs=200, horizon=12, start_year=None, seed=RNG_SEED,
                param_uncertainty=True, detail=False):
    """param_uncertainty=True면 반복마다 hazard 계수를 표집분포에서 새로 뽑는다.
    산출 구간은 '인력 구성·개인 사건의 우연 변동 + 계수 추정 불확실성'을 함께 반영한다.
    반환값의 runs에는 반복별 지표 배열이 들어 있다(같은 seed를 쓰면 시나리오 간 대응표본)."""
    all_runs = []
    for r in range(runs):
        coef = draw_coefficients(np.random.default_rng(10_000 + r)) if param_uncertainty else None
        all_runs.append(simulate(profile, scen, start_year, horizon, seed=seed + r,
                                 coef=coef, detail=detail))
    years = [r["year"] for r in all_runs[0]]
    mean_path = []
    for i, y in enumerate(years):
        row = {"year": y}
        for k, v0 in all_runs[0][i].items():
            if k == "year":
                continue
            if isinstance(v0, list):
                row[k] = np.mean([run[i][k] for run in all_runs], axis=0).tolist()
            else:
                row[k] = float(np.mean([run[i][k] for run in all_runs]))
        mean_path.append(row)

    per_run = {}
    for name, key, how in _TOTALS:
        arr = np.array([[r[key] for r in run] for run in all_runs], dtype=float)
        per_run[name] = arr.sum(axis=1) if how == "sum" else arr.mean(axis=1)

    summary = dict(scenario=scen.name, runs=runs, horizon=horizon,
                   param_uncertainty=bool(param_uncertainty),
                   wage_regime=profile.wage_regime, n_initial=int(all_runs[0][0]["headcount"]))
    for name, arr in per_run.items():
        summary[f"{name}_mean"] = float(arr.mean())
        summary[f"{name}_p5"] = float(np.percentile(arr, 5))
        summary[f"{name}_p95"] = float(np.percentile(arr, 95))
    return dict(summary=summary, path=mean_path, runs=per_run)
