"""
company_data.py — 회사 데이터 층: 인사 명부를 읽고, 점검하고, 엔진이 쓸 수 있는 형태로 바꾼다
====================================================================
이 파일이 하는 일
  1. 읽기        CSV(UTF-8·CP949)·엑셀 명부를 읽는다.
  2. 열 맞추기   '생년', '입사일', '연봉'처럼 회사마다 다른 열 이름을 자동으로 알아보고,
                 나이·근속·월 임금(만원)으로 환산한다.
  3. 품질 점검   빠진 값·말이 안 되는 값·단위 혼용을 찾아 무엇을 어떻게 처리했는지 보고한다.
  4. 프로파일    직급 체계·부서·신입 초임·현재 재고용 관행을 명부에서 추정해 CompanyProfile로 만든다.
  5. 진단        시뮬레이션 없이도 알 수 있는 것(부서별 정년 도달 인원 달력 등)을 계산한다.
  6. 퇴사 보정   퇴사자 명부가 있으면 자사 퇴사율과 근속 구간별 패턴을 모형에 반영한다.
  7. 집계표 합성 개인별 명부를 올릴 수 없는 회사를 위해 연령대별 집계표로 가상 명부를 만든다.

개인정보 원칙: 사번·이름처럼 식별에 쓰이는 열은 읽자마자 버린다. 이 모듈은 어떤 데이터도
파일로 저장하지 않으며, 표준화된 명부는 호출자의 메모리(세션)에만 머문다.
"""
from __future__ import annotations
import io
import re
import numpy as np
import pandas as pd
from engine_retirement_v2 import (CompanyProfile, GradeSystem, BASE_COEF, _hazard, _calibrate, JOB_DEFAULTS,
                                  _BASE, AGE_WAGE_INDEX, NO_GRADE)

# ── 열 이름 사전(소문자·공백 제거 후 '포함' 여부로 판단, 위에서부터 우선) ──
COLUMN_HINTS = {
    "id":       ["사번", "사원번호", "직원번호", "성명", "이름", "주민", "employeeid", "empno", "name"],
    "birth":    ["생년월일", "출생연도", "출생년도", "생년", "birth", "dob"],
    "age":      ["만나이", "나이", "연령", "age"],
    "hire":     ["입사일", "입사연도", "입사년도", "입사", "hiredate", "joined", "joindate"],
    "tenure":   ["근속연수", "근속년수", "근속기간", "재직기간", "근속", "tenure"],
    "wage_annual": ["연봉", "연간급여", "연간임금", "annual"],
    "wage":     ["월평균임금", "월급여", "월임금", "월급", "기본급", "급여", "임금", "wage", "salary", "pay"],
    "yrs_in_grade": ["직급연차", "현직급", "승진후", "체류연수", "yearsingrade"],
    "grade":    ["직급", "직위", "호칭", "grade", "rank", "position", "level"],
    "dept":     ["부서", "본부", "사업부", "직군", "조직", "팀", "department", "dept", "division"],
    "emp_type": ["고용형태", "계약형태", "근로형태", "고용구분", "사원구분", "employment", "emptype", "contract"],
    "hours":    ["주당근로시간", "근로시간", "소정근로", "hours"],
}
FIELD_LABELS = {
    "age": "나이", "birth": "생년(월일)", "tenure": "근속연수", "hire": "입사일",
    "wage": "월 임금", "wage_annual": "연봉", "grade": "직급", "dept": "부서",
    "emp_type": "고용형태", "hours": "주당 근로시간", "yrs_in_grade": "현 직급 연차", "id": "식별자(버림)",
}
WAGE_UNITS = {"만원": 1.0, "천원": 0.1, "원": 1e-4}
CE_KEYWORDS = ["촉탁", "재고용", "계속고용", "정년후", "정년 후", "시니어", "re-hire", "rehire"]
EXCLUDE_KEYWORDS = ["파견", "도급", "인턴", "아르바이트", "알바", "일용", "용역"]
EXEC_KEYWORDS = ["임원", "이사", "상무", "전무", "부사장", "사장", "대표", "감사", "ceo", "cfo", "cto"]
MAX_DEPTS = 14
BASE_YEAR = 2026


def _norm(s):
    return re.sub(r"[\s_\-·./()\[\]]", "", str(s)).lower()


def read_table(file, filename: str | None = None) -> pd.DataFrame:
    """CSV 또는 엑셀을 읽는다. file은 경로·bytes·파일 객체 모두 가능."""
    name = (filename or getattr(file, "name", "") or str(file)).lower()
    raw = file
    if hasattr(file, "read"):
        raw = file.read()
        if hasattr(file, "seek"):
            file.seek(0)
    if isinstance(raw, (bytes, bytearray)):
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(raw))
        last = None
        for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-16"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc)
                if df.shape[1] == 1:   # 구분자가 세미콜론·탭인 경우
                    df2 = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
                    df = df2 if df2.shape[1] > 1 else df
                return df
            except Exception as e:      # 다음 인코딩 시도
                last = e
        raise ValueError(f"파일을 읽지 못했습니다. CSV(UTF-8 또는 CP949)나 엑셀(.xlsx)인지 확인해 주세요. ({last})")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(raw)
    return read_table(open(raw, "rb").read(), name)


def auto_map(df: pd.DataFrame) -> dict:
    """열 이름으로 각 항목에 해당하는 열을 추정. {항목: 열 이름 또는 None}."""
    cols = list(df.columns)
    normed = {c: _norm(c) for c in cols}
    used, mapping = set(), {}
    for fld, hints in COLUMN_HINTS.items():
        hit = None
        for h in hints:
            for c in cols:
                if c in used:
                    continue
                if _norm(h) in normed[c]:
                    hit = c
                    break
            if hit:
                break
        mapping[fld] = hit
        if hit and fld != "id":
            used.add(hit)
    return mapping


def _to_year(series: pd.Series) -> pd.Series:
    """생년·입사일 열에서 연도를 뽑는다. 1975, '1975-03-02', 19750302, '75.03.02' 모두 처리."""
    def one(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            return float(pd.Timestamp(v).year)
        s = str(v).strip()
        m = re.match(r"^(\d{4})", s)
        if m and 1900 <= int(m.group(1)) <= 2100:
            return float(m.group(1))
        m = re.match(r"^(\d{2})[.\-/]?\d{2}[.\-/]?\d{2}$", s)      # 두 자리 연도
        if m:
            yy = int(m.group(1))
            return float(1900 + yy if yy > 30 else 2000 + yy)
        try:
            f = float(s)
            if 1900 <= f <= 2100:
                return f
            if 19000101 <= f <= 21001231:
                return float(int(f) // 10000)
        except ValueError:
            pass
        d = pd.to_datetime(s, errors="coerce")
        return float(d.year) if pd.notna(d) else np.nan
    return series.map(one)


def _to_num(series: pd.Series) -> pd.Series:
    if series.dtype.kind in "biuf":
        return series.astype(float)
    return pd.to_numeric(series.astype(str).str.replace(r"[,\s원만천]", "", regex=True), errors="coerce")


def detect_wage_unit(values: pd.Series, annual: bool):
    """임금 열의 단위를 중앙값 크기로 추정. (단위, 연봉 여부, 설명)."""
    med = float(np.nanmedian(_to_num(values)))
    monthly = med / 12 if annual else med
    if monthly >= 100_000:
        return "원", annual, f"중앙값 {med:,.0f} → 원 단위로 보입니다."
    if monthly >= 20_000 and not annual:
        return "천원", annual, f"중앙값 {med:,.0f} → 천원 단위로 보입니다."
    if not annual and med >= 1800:
        return "만원", True, f"중앙값 {med:,.0f}만원은 월급으로 보기엔 커서 연봉으로 보입니다."
    return "만원", annual, f"중앙값 {med:,.0f} → 만원 단위로 보입니다."


def standardize(df: pd.DataFrame, mapping: dict, base_year: int = BASE_YEAR,
                wage_unit: str = "만원", wage_is_annual: bool | None = None,
                retirement_age: int = 60, ce_values: list | None = None,
                exclude_values: list | None = None, drop_executives: bool = True):
    """원본 명부 → (표준 명부, 품질 보고서).

    표준 명부 열: age, tenure, wage(월 만원), hours, status(0 정규/1 계속고용),
                  grade_name, dept_name, emp_type (+ yrs_in_grade가 있으면 포함)
    보고서: dict(n_raw, n_used, issues=[{level, title, detail, n}], dropped=n)
    level은 'error'(쓸 수 없음) / 'warn'(확인 필요) / 'info'(자동 처리함).
    """
    issues = []
    n_raw = len(df)
    out = pd.DataFrame(index=df.index)

    def add(level, title, detail, n=0):
        issues.append(dict(level=level, title=title, detail=detail, n=int(n)))

    # ── 나이 ──
    if mapping.get("age"):
        out["age"] = _to_num(df[mapping["age"]])
    elif mapping.get("birth"):
        out["age"] = base_year - _to_year(df[mapping["birth"]])
        add("info", "나이를 생년에서 계산했습니다", f"{base_year}년 − 출생연도로 계산했습니다(만 나이와 최대 1세 차이).")
    else:
        add("error", "나이 정보가 없습니다", "'나이' 또는 '생년' 열이 필요합니다.")
        return None, dict(n_raw=n_raw, n_used=0, issues=issues, dropped=n_raw)
    # ── 근속 ──
    if mapping.get("tenure"):
        out["tenure"] = _to_num(df[mapping["tenure"]])
    elif mapping.get("hire"):
        out["tenure"] = base_year - _to_year(df[mapping["hire"]])
        add("info", "근속연수를 입사일에서 계산했습니다", f"{base_year}년 − 입사연도로 계산했습니다.")
    else:
        add("error", "근속 정보가 없습니다", "'근속연수' 또는 '입사일' 열이 필요합니다.")
        return None, dict(n_raw=n_raw, n_used=0, issues=issues, dropped=n_raw)
    # ── 임금 ──
    wcol = mapping.get("wage") or mapping.get("wage_annual")
    if not wcol:
        add("error", "임금 정보가 없습니다", "'월 임금' 또는 '연봉' 열이 필요합니다.")
        return None, dict(n_raw=n_raw, n_used=0, issues=issues, dropped=n_raw)
    annual = (not mapping.get("wage")) if wage_is_annual is None else wage_is_annual
    w = _to_num(df[wcol]) * WAGE_UNITS[wage_unit]
    out["wage"] = w / 12 if annual else w

    out["hours"] = _to_num(df[mapping["hours"]]) if mapping.get("hours") else 40.0
    out["grade_name"] = (df[mapping["grade"]].astype(str).str.strip() if mapping.get("grade") else "")
    out["dept_name"] = (df[mapping["dept"]].astype(str).str.strip() if mapping.get("dept") else "전체")
    out["emp_type"] = (df[mapping["emp_type"]].astype(str).str.strip() if mapping.get("emp_type") else "")
    if mapping.get("yrs_in_grade"):
        out["yrs_in_grade"] = _to_num(df[mapping["yrs_in_grade"]])
    for c in ("grade_name", "dept_name", "emp_type"):
        out[c] = out[c].replace({"nan": "", "None": "", "NaN": ""})
    out.loc[out["dept_name"] == "", "dept_name"] = "미지정"

    # ── 중복 식별자 ──
    if mapping.get("id"):
        dup = int(df[mapping["id"]].duplicated().sum())
        if dup:
            add("warn", "같은 식별자가 여러 번 나옵니다",
                f"'{mapping['id']}' 값이 겹치는 행이 {dup}건 있습니다. 같은 사람이 두 번 들어갔는지 확인해 주세요.", dup)

    # ── 분석 대상에서 빼는 행 ──
    keep = pd.Series(True, index=out.index)
    excl = exclude_values
    if excl is None:
        excl = [v for v in out["emp_type"].unique() if any(k in _norm(v) for k in map(_norm, EXCLUDE_KEYWORDS))]
    if excl:
        m = out["emp_type"].isin(excl)
        if m.any():
            add("info", "정년 적용 대상이 아닌 고용형태를 제외했습니다",
                f"{', '.join(map(str, excl))} {int(m.sum())}명은 분석에서 뺐습니다.", m.sum())
            keep &= ~m
    if drop_executives and mapping.get("grade"):
        m = out["grade_name"].map(lambda v: any(k in _norm(v) for k in EXEC_KEYWORDS)) & keep
        if m.any():
            add("info", "임원을 제외했습니다",
                f"임원은 정년 규정의 적용을 받지 않으므로 {int(m.sum())}명을 분석에서 뺐습니다.", m.sum())
            keep &= ~m

    miss = out[["age", "tenure", "wage"]].isna().any(axis=1) & keep
    if miss.any():
        add("warn", "필수 값이 빠진 행을 제외했습니다",
            f"나이·근속·임금 중 하나라도 비어 있는 {int(miss.sum())}행을 뺐습니다.", miss.sum())
        keep &= ~miss
    bad_age = ((out["age"] < 15) | (out["age"] > 80)) & keep
    if bad_age.any():
        add("warn", "나이가 범위를 벗어난 행을 제외했습니다",
            f"15세 미만 또는 80세 초과 {int(bad_age.sum())}행. 생년 열 형식을 확인해 주세요.", bad_age.sum())
        keep &= ~bad_age
    bad_wage = (out["wage"] <= 0) & keep
    if bad_wage.any():
        add("warn", "임금이 0 이하인 행을 제외했습니다", f"{int(bad_wage.sum())}행.", bad_wage.sum())
        keep &= ~bad_wage

    out = out[keep].copy()
    if len(out) == 0:
        add("error", "쓸 수 있는 행이 없습니다", "열 지정과 단위를 다시 확인해 주세요.")
        return None, dict(n_raw=n_raw, n_used=0, issues=issues, dropped=n_raw)

    out["age"] = out["age"].round().astype(int)
    neg = out["tenure"] < 0
    if neg.any():
        add("warn", "근속이 음수인 값을 0으로 바꿨습니다",
            f"{int(neg.sum())}행. 기준 연도({base_year}년) 이후 입사자로 보입니다.", neg.sum())
        out.loc[neg, "tenure"] = 0
    too_long = out["tenure"] > (out["age"] - 15)
    if too_long.any():
        add("warn", "나이에 비해 근속이 너무 긴 값을 줄였습니다",
            f"{int(too_long.sum())}행의 근속을 '나이 − 15'로 맞췄습니다.", too_long.sum())
        out.loc[too_long, "tenure"] = (out.loc[too_long, "age"] - 15).clip(lower=0)
    out["tenure"] = out["tenure"].astype(float)
    h_bad = out["hours"].isna() | (out["hours"] < 5) | (out["hours"] > 90)
    if h_bad.any() and mapping.get("hours"):
        add("info", "근로시간이 비었거나 이상한 값은 40시간으로 채웠습니다", f"{int(h_bad.sum())}행.", h_bad.sum())
    out.loc[h_bad, "hours"] = 40.0

    # ── 임금 단위 점검 ──
    med = float(out["wage"].median())
    if med < 120 or med > 2500:
        add("warn", "임금 단위가 맞는지 확인해 주세요",
            f"환산한 월 임금의 중앙값이 {med:,.0f}만원입니다. 보통 200~1,500만원 사이입니다.")
    lo, hi = out["wage"].quantile([0.01, 0.99])
    if hi / max(lo, 1e-9) > 40:
        add("warn", "임금 단위가 섞여 있을 수 있습니다",
            f"하위 1%({lo:,.0f}만원)와 상위 1%({hi:,.0f}만원)의 차이가 40배를 넘습니다. "
            "월급과 연봉이 한 열에 섞였는지 확인해 주세요.")

    # ── 고용형태 → 정규/계속고용 ──
    if mapping.get("emp_type"):
        ce = ce_values
        if ce is None:
            ce = [v for v in out["emp_type"].unique() if any(_norm(k) in _norm(v) for k in CE_KEYWORDS)]
        out["status"] = out["emp_type"].isin(ce).astype(int)
        late = (out["status"] == 0) & (out["age"] > retirement_age)
        if late.any():
            add("info", "정년을 넘긴 정규직이 있습니다",
                f"{retirement_age}세를 넘겼는데 정규직으로 표시된 {int(late.sum())}명은 첫해에 정년퇴직하는 것으로 처리됩니다. "
                "촉탁·재고용이라면 고용형태 값을 확인해 주세요.", late.sum())
    else:
        out["status"] = (out["age"] > retirement_age).astype(int)
        n_ce = int(out["status"].sum())
        if n_ce:
            add("info", "고용형태 열이 없어 나이로 구분했습니다",
                f"{retirement_age}세를 넘긴 {n_ce}명을 정년 후 재고용(촉탁)으로 보았습니다.", n_ce)

    if len(out) < 100:
        add("warn", "인원이 적어 결과의 변동폭이 큽니다",
            f"분석 인원이 {len(out)}명입니다. 이직 모형은 300인 이상 사업장 자료로 추정한 것이라 "
            "작은 조직에서는 참고용으로만 보세요.")
    out = out.reset_index(drop=True)
    return out, dict(n_raw=n_raw, n_used=len(out), dropped=n_raw - len(out), issues=issues)


# ── 직급 체계 추정 ──────────────────────────────────────────────────────────
def infer_grades(std: pd.DataFrame):
    """정규직의 직급을 '근속 중앙값이 짧은 순 → 임금 중앙값이 낮은 순'으로 줄 세운다.
    반환: DataFrame(직급, 인원, 근속중앙값, 임금중앙값, 서열, 최소체류연수) — 화면에서 수정 가능."""
    reg = std[(std["status"] == 0) & (std["grade_name"] != "")]
    if reg.empty:
        return None
    g = reg.groupby("grade_name").agg(인원=("age", "size"), 근속중앙값=("tenure", "median"),
                                      임금중앙값=("wage", "median")).reset_index()
    # 임금과 근속 순위를 합쳐 서열을 추정(임금에 더 큰 비중: 직급은 보통 임금과 더 잘 맞는다)
    g["_score"] = g["임금중앙값"].rank() * 0.6 + g["근속중앙값"].rank() * 0.4
    g = g.sort_values(["_score", "임금중앙값"]).reset_index(drop=True)
    g["서열"] = np.arange(1, len(g) + 1)
    gaps = g["근속중앙값"].diff().shift(-1)
    g["최소체류연수"] = gaps.mul(0.8).round().clip(lower=2, upper=8).fillna(99).astype(int)
    g = g.rename(columns={"grade_name": "직급"}).drop(columns="_score")
    g[["근속중앙값", "임금중앙값"]] = g[["근속중앙값", "임금중앙값"]].round(1)
    return g


def infer_ce_practice(std: pd.DataFrame, retirement_age=60, annual_exit=0.06):
    """현재 재고용 관행 추정. 명부의 '정년 직전 5개 연령 인원'과 '정년 후 인원'을 비교한다.

    나이는 '그 해에 도달하는 만 나이'다. 정년이 60세면 60세인 사람까지는 정규직이고,
    재고용자는 61세부터 상한 연령 E세까지 일한다(E세가 되는 해 말에 퇴장 → E−60년 근무).
    정년 직전 5개 연령(56~60세)의 연 평균 인원이 매년 정년에 도달하고 그중 rate만큼이
    남는다고 보면, 정년 후 인원 ≈ 코호트 × rate × (머무는 햇수, 연 6% 자연 퇴장 반영).
    이 식을 rate에 대해 푼다. 표본이 작으면(코호트 2명 미만) 추정하지 않는다."""
    reg = std["status"] == 0
    pre = reg & (std["age"] > retirement_age - 5) & (std["age"] <= retirement_age)
    post = (std["status"] == 1) & (std["age"] > retirement_age)
    cohort = pre.sum() / 5.0
    if post.sum() == 0 or cohort < 2:
        return dict(rate=0.0, end_age=retirement_age, wage_ratio=0.55, n_post=int(post.sum()),
                    cohort=float(cohort), basis="정년 후 재직자가 없거나 표본이 작아 '관행 없음'으로 두었습니다.")
    ages = np.sort(std.loc[post, "age"].to_numpy())
    end = retirement_age + 1
    for a in range(retirement_age + 1, retirement_age + 11):
        end = a
        if (ages <= a).mean() >= 0.9:
            break
    end = min(end, retirement_age + 5)
    stock = int((ages <= end).sum())
    s = 1 - annual_exit
    dur = sum(s ** k for k in range(end - retirement_age))
    rate = float(np.clip(stock / (cohort * dur), 0.0, 0.95))
    w_pre = std.loc[pre, "wage"].median()
    w_post = std.loc[post & (std["age"] <= end), "wage"].median()
    ratio = float(np.clip(w_post / w_pre, 0.3, 1.0)) if w_pre > 0 and pd.notna(w_post) else 0.55
    return dict(rate=round(rate, 2), end_age=int(end), wage_ratio=round(ratio, 2),
                n_post=int(post.sum()), cohort=float(cohort),
                basis=f"정년 직전 5개 연령 인원이 연 평균 {cohort:.0f}명, "
                      f"{retirement_age+1}~{end}세 재직자가 {stock}명입니다.")


def group_depts(std: pd.DataFrame):
    """부서(직군)가 너무 많으면 큰 순서대로 남기고 나머지는 '기타'로 묶는다.
    반환: (dept 인덱스 열이 붙은 사본, 직군 이름 목록 — 인원이 많은 순)."""
    std = std.copy()
    counts = std["dept_name"].value_counts()
    if len(counts) > MAX_DEPTS:
        small = counts.index[MAX_DEPTS - 1:]
        std.loc[std["dept_name"].isin(small), "dept_name"] = "기타"
        counts = std["dept_name"].value_counts()
    dept_names = list(counts.index)
    std["dept"] = std["dept_name"].map({d: i for i, d in enumerate(dept_names)}).astype(int)
    return std, dept_names


JOB_COLS = ["vacancy_months", "ttp_years", "ce_accept_rel", "ce_wage_rel"]


def job_table_for(dept_names, user_jobs=None, dept_offsets=None) -> pd.DataFrame:
    """엔진용 직군 표(dept_names 순서). user_jobs = {직군: {열: 값}}, dept_offsets = {직군: 로짓 보정}.
    입력이 없는 칸은 엔진 기본값(JOB_DEFAULTS)."""
    user_jobs, dept_offsets = user_jobs or {}, dept_offsets or {}
    rows = []
    for d in dept_names:
        u = user_jobs.get(d, {})
        row = {"name": d}
        for k in JOB_COLS:
            v = u.get(k)
            row[k] = float(v) if v is not None and np.isfinite(float(v)) else JOB_DEFAULTS[k]
        row["turnover_offset"] = float(dept_offsets.get(d, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


def build_profile(std: pd.DataFrame, grade_table: pd.DataFrame | None = None,
                  retirement_age=60, base_push_turnover=0.0148, other_exit_rate=0.045,
                  wage_regime="observed", custom_wage_rates=None, ce_practice=None,
                  start_year=BASE_YEAR, name="우리 회사", hazard_offsets=None,
                  entry_wage=None, entry_age=None, user_jobs=None, dept_offsets=None) -> CompanyProfile:
    """표준 명부 → 엔진용 CompanyProfile(실제 인원 그대로 구동)."""
    std, dept_names = group_depts(std)

    roster = std[["age", "tenure", "wage", "hours", "status", "dept"]].copy()
    reg = std["status"] == 0
    if grade_table is not None and len(grade_table) >= 2:
        gt = grade_table.sort_values("서열").reset_index(drop=True)
        code = {g: i for i, g in enumerate(gt["직급"])}
        gcode = std["grade_name"].map(code)
        # 직급이 비어 있거나 표에 없는 정규직은 근속이 가장 가까운 직급으로 배정
        unk = gcode.isna() & reg
        if unk.any():
            med = gt["근속중앙값"].to_numpy(float)
            gcode.loc[unk] = std.loc[unk, "tenure"].map(lambda t: int(np.argmin(np.abs(med - t))))
        roster["grade"] = gcode.fillna(NO_GRADE).astype(int)
        roster.loc[~reg, "grade"] = NO_GRADE
        shares = roster.loc[reg, "grade"].value_counts(normalize=True).reindex(range(len(gt))).fillna(0.0)
        shares = shares.clip(lower=1e-4)
        mins = [int(m) for m in gt["최소체류연수"]]
        mins[-1] = 99
        grades = GradeSystem(names=list(gt["직급"]), min_years=mins, target_share=list(shares))
    else:
        grades = GradeSystem()
    if "yrs_in_grade" in std and std["yrs_in_grade"].notna().mean() > 0.9 and "grade" in roster:
        roster["yrs_in_grade"] = std["yrs_in_grade"].fillna(0).clip(lower=0)

    newbie = reg & (std["tenure"] <= 1)
    if entry_wage is None:
        if newbie.sum() >= 5:
            entry_wage = float(std.loc[newbie, "wage"].median())
        else:
            entry_wage = float(std.loc[reg, "wage"].quantile(0.10)) if reg.any() else None
    if entry_age is None:
        entry_age = float(std.loc[newbie, "age"].mean()) if newbie.sum() >= 5 else 27.0
    ce = ce_practice or infer_ce_practice(std, retirement_age)
    pool = hire_pool_from(roster, retirement_age)
    return CompanyProfile(
        hire_pool=pool,
        n_employees=len(roster), avg_wage=float(std["wage"].mean()),
        base_push_turnover=base_push_turnover, other_exit_rate=other_exit_rate,
        current_retirement_age=retirement_age, entry_age_mean=float(np.clip(entry_age, 20, 45)),
        wage_regime=wage_regime, custom_wage_rates=custom_wage_rates,
        ce_rate_now=float(ce["rate"]), ce_end_age_now=int(ce["end_age"]),
        ce_wage_ratio_now=float(ce["wage_ratio"]),
        roster=roster, resample=False, grades=grades, dept_names=dept_names,
        entry_wage=entry_wage, entry_hours=float(std.loc[reg, "hours"].median()) if reg.any() else 40.0,
        hazard_offsets=hazard_offsets, start_year=start_year, name=name,
        job_table=job_table_for(dept_names, user_jobs, dept_offsets))


MIN_HIRE_POOL = 20


def hire_pool_from(roster: pd.DataFrame, retirement_age=60):
    """최근 입사자(정규직·근속 1년 이하·정년 3년 전 미만) 표본. 20명 미만이면 None(기본 가정 사용)."""
    m = (roster["status"] == 0) & (roster["tenure"] <= 1) & (roster["age"] <= retirement_age - 2)
    if m.sum() < MIN_HIRE_POOL:
        return None
    cols = [c for c in ("age", "wage", "hours", "grade") if c in roster]
    pool = roster.loc[m, cols].copy().reset_index(drop=True)
    if "grade" in pool:
        pool["grade"] = pool["grade"].clip(lower=0)
    return pool


# ── 체험용 표준 표본 ────────────────────────────────────────────────────────
def klips_standard(include_post_retirement=False, retirement_age=60) -> pd.DataFrame:
    """KLIPS 300인+ 실표본을 표준 명부 형식으로. 기본은 60세 미만만 쓴다.

    표본은 정년·재고용 관행이 서로 다른 여러 회사의 사람을 모은 것이라, 60세 이상(7.6%)을
    한 회사의 인력으로 보면 '이미 과반을 65세까지 재고용하는 회사'가 되어버린다. 체험용 기본값은
    '정년 60세, 재고용 관행 없음'인 회사로 두고, 재고용 관행을 켜면 60세 이상도 포함한다."""
    if _BASE is None:
        raise FileNotFoundError("big_firm_sample_2023.csv가 없습니다.")
    s = _BASE.copy()
    if not include_post_retirement:
        s = s[s["age"] <= retirement_age]
    std = pd.DataFrame(dict(age=s["age"].astype(int), tenure=s["tenure"].astype(float),
                            wage=s["wage"].astype(float), hours=s["hours"].astype(float)))
    std["status"] = (std["age"] > retirement_age).astype(int)
    std["grade_name"], std["dept_name"], std["emp_type"] = "", "전체", ""
    return std.reset_index(drop=True)


def demo_profile(n_employees=1000, avg_wage=465.0, base_push_turnover=0.0148,
                 other_exit_rate=0.045, wage_regime="observed", with_ce_practice=False,
                 start_year=BASE_YEAR) -> CompanyProfile:
    """체험용 프로파일: KLIPS 표본을 n_employees명으로 부트스트랩하고 평균임금을 맞춘다."""
    std = klips_standard(include_post_retirement=with_ce_practice)
    ce = infer_ce_practice(std) if with_ce_practice else dict(rate=0.0, end_age=60, wage_ratio=0.55)
    roster = std[["age", "tenure", "wage", "hours", "status"]].copy()
    roster["wage"] = roster["wage"] * (avg_wage / roster["wage"].mean())
    return CompanyProfile(
        n_employees=n_employees, avg_wage=avg_wage, base_push_turnover=base_push_turnover,
        other_exit_rate=other_exit_rate, wage_regime=wage_regime,
        ce_rate_now=float(ce["rate"]), ce_end_age_now=int(ce["end_age"]),
        ce_wage_ratio_now=float(ce["wage_ratio"]),
        roster=roster, resample=True, start_year=start_year,
        hire_pool=hire_pool_from(roster),
        name="표준 대기업 표본(KLIPS 300인+)")


# ── 진단: 시뮬레이션 없이 명부만으로 알 수 있는 것 ──────────────────────────
def structure_summary(std: pd.DataFrame, retirement_age=60) -> dict:
    reg = std["status"] == 0
    a = std["age"]
    return dict(n=len(std), n_regular=int(reg.sum()), n_ce=int((~reg).sum()),
                avg_age=float(a.mean()), avg_tenure=float(std["tenure"].mean()),
                avg_wage=float(std["wage"].mean()),
                share_55p=float((a >= 55).mean()), share_u35=float((a < 35).mean()),
                reach_5y=float((reg & (a > retirement_age - 5)).sum() / max(reg.sum(), 1)),
                reach_10y=float((reg & (a > retirement_age - 10)).sum() / max(reg.sum(), 1)))


def age_pyramid(std: pd.DataFrame, width=5) -> pd.DataFrame:
    lo = (std["age"] // width * width).clip(lower=20, upper=65)
    lab = lo.map(lambda x: "20대 초반 이하" if x <= 20 else ("65세 이상" if x >= 65 else f"{int(x)}~{int(x)+width-1}세"))
    t = (std.assign(band=lab, lo=lo).groupby(["lo", "band", "status"]).size().unstack(fill_value=0)
         .rename(columns={0: "정규", 1: "재고용·촉탁"}).reset_index().sort_values("lo"))
    for c in ("정규", "재고용·촉탁"):
        if c not in t:
            t[c] = 0
    return t[["band", "정규", "재고용·촉탁"]].reset_index(drop=True)


def retirement_calendar(std: pd.DataFrame, by="dept_name", retirement_age=60, years=10,
                        start_year=BASE_YEAR) -> pd.DataFrame:
    """현행 정년 기준으로 '몇 년에 몇 명이 정년에 도달하는가'(중도 퇴사 미반영, 정규직만).
    행: 부서(또는 직급) / 열: 연도. 이미 정년 이상인 정규직은 첫해에 넣는다."""
    reg = std[std["status"] == 0]
    yr = (start_year + (retirement_age - reg["age"]).clip(lower=0)).astype(int)
    m = reg.assign(year=yr)[lambda d: d["year"] < start_year + years]
    cal = m.groupby([by, "year"]).size().unstack(fill_value=0)
    cal = cal.reindex(columns=range(start_year, start_year + years), fill_value=0)
    cal = cal.reindex(reg[by].value_counts().index, fill_value=0)
    return cal


def dept_risk(std: pd.DataFrame, retirement_age=60, by="dept_name") -> pd.DataFrame:
    """부서별 고령화 위험. '5년 내 정년 도달 비중'이 높은 순."""
    reg = std[std["status"] == 0]
    g = reg.groupby(by).agg(인원=("age", "size"), 평균나이=("age", "mean"),
                            비중55=("age", lambda a: (a >= 55).mean()),
                            도달5=("age", lambda a: (a > retirement_age - 5).mean()),
                            도달10=("age", lambda a: (a > retirement_age - 10).mean()),
                            청년35=("age", lambda a: (a < 35).mean())).reset_index()
    return g.sort_values("도달5", ascending=False).reset_index(drop=True)


# ── 퇴사자 명부로 자사 퇴사 패턴 반영 ──────────────────────────────────────
INVOLUNTARY_KEYS = ["권고", "해고", "구조조정", "징계", "희망퇴직", "명예퇴직", "해지", "사망", "involuntary"]
VOLUNTARY_KEYS = ["자발", "개인", "이직", "전직", "사직", "자진", "의원", "학업", "voluntary", "resign"]
RETIRE_KEYS = ["정년", "만료", "계약종료", "retire"]
TENURE_BINS = [2, 5, 10, np.inf]
TENURE_LABELS = ["2년 이하", "3~5년", "6~10년", "11년 이상"]


def classify_reason(v) -> str:
    s = _norm(v)
    if any(_norm(k) in s for k in INVOLUNTARY_KEYS):      # '권고사직'은 '사직'이 들어 있어도 비자발
        return "기타"
    if any(_norm(k) in s for k in RETIRE_KEYS):
        return "정년·계약만료"
    if any(_norm(k) in s for k in VOLUNTARY_KEYS):
        return "자발"
    return "기타"


def calibrate_from_leavers(std: pd.DataFrame, leavers: pd.DataFrame, mapping: dict,
                           n_years: float | None = None, shrink_k: float = 10.0) -> dict:
    """퇴사자 명부 → 자사 자발적 퇴사율·기타 퇴사율 + 근속 구간별 보정값.

    근속 구간별 보정: 모형이 예측한 구간별 퇴사자 수와 실제를 비교해 로짓 보정값을 구하되,
    사건 수가 적은 구간은 0 쪽으로 당긴다(수축 계수 = 사건 수 ÷ (사건 수 + 10)).
    한계: 위험집단(분모)을 '현재 재직자'로 근사한다. 인력 구성이 최근 크게 바뀐 회사는
    보정값이 왜곡될 수 있다."""
    lv = pd.DataFrame()
    lv["tenure"] = _to_num(leavers[mapping["tenure"]]) if mapping.get("tenure") else np.nan
    lv["age"] = _to_num(leavers[mapping["age"]]) if mapping.get("age") else np.nan
    lv["kind"] = (leavers[mapping["reason"]].map(classify_reason) if mapping.get("reason") else "자발")
    if n_years is None:
        if mapping.get("year"):
            n_years = max(int(_to_year(leavers[mapping["year"]]).nunique()), 1)
        else:
            n_years = 1.0
    reg = std[std["status"] == 0]
    n_reg = max(len(reg), 1)
    n_vol = int((lv["kind"] == "자발").sum())
    n_oth = int((lv["kind"] == "기타").sum())
    push = n_vol / (n_reg * n_years)
    other = n_oth / (n_reg * n_years)

    wf = dict(age=reg["age"].to_numpy(float), tenure=reg["tenure"].to_numpy(float),
              wage=reg["wage"].to_numpy(float), hours=reg["hours"].to_numpy(float),
              status=np.zeros(len(reg), dtype=int))
    table, offsets = None, None
    vol = lv[(lv["kind"] == "자발") & lv["tenure"].notna()]
    if len(vol) >= 10 and push > 0:
        b0 = _calibrate(wf, push)
        p = _hazard(wf, b0)
        band_r = np.searchsorted(TENURE_BINS, wf["tenure"], side="left")
        band_l = np.searchsorted(TENURE_BINS, vol["tenure"].to_numpy(float), side="left")
        rows, offs = [], []
        for j, lab in enumerate(TENURE_LABELS):
            pred = float(p[band_r == j].sum() * n_years)
            act = int((band_l == j).sum())
            raw = np.log((act + 0.5) / (pred + 0.5))
            lam = act / (act + shrink_k)
            offs.append(float(lam * raw))
            rows.append(dict(근속구간=lab, 재직자=int((band_r == j).sum()), 실제퇴사=act,
                             모형예측=round(pred, 1), 보정값=round(lam * raw, 2)))
        table = pd.DataFrame(rows)
        offsets = dict(bins=[2.0, 5.0, 10.0, 1e9], offsets=offs)

    # 직군(부서)별 보정: 근속 구간 보정을 넣은 모형의 직군별 예측 퇴사자 수와 실제를 비교한다.
    # 근속 구간 보정과 같은 수축 규칙을 쓴다 — 퇴사자가 몇 명뿐인 직군은 회사 평균(0)에 가깝게 남는다.
    dept_table, dept_offsets, n_unmatched = None, None, 0
    if mapping.get("dept") and "dept_name" in reg and len(vol) >= 10 and push > 0:
        lv_dept = leavers.loc[vol.index, mapping["dept"]].astype(str).str.strip()
        names = list(reg["dept_name"].value_counts().index)
        known = lv_dept.isin(names)
        n_unmatched = int((~known).sum())
        if known.sum() >= 10 and len(names) > 1:
            b1 = _calibrate(wf, push, offsets=offsets)
            p1 = _hazard(wf, b1, offsets=offsets)
            dn = reg["dept_name"].to_numpy()
            rows, dept_offsets = [], {}
            for d in names:
                pred = float(p1[dn == d].sum() * n_years)
                act = int((lv_dept[known] == d).sum())
                raw = np.log((act + 0.5) / (pred + 0.5))
                lam = act / (act + shrink_k)
                dept_offsets[d] = float(lam * raw)
                rows.append(dict(직군=d, 재직자=int((dn == d).sum()), 실제퇴사=act,
                                 모형예측=round(pred, 1), 보정값=round(lam * raw, 2)))
            dept_table = pd.DataFrame(rows)
    dev = None
    if table is not None and table["실제퇴사"].sum() > 0:
        dev = float((table["실제퇴사"] - table["모형예측"]).abs().sum() / (2 * table["실제퇴사"].sum()))
    return dict(push_rate=float(push), other_rate=float(other), n_vol=n_vol, n_other=n_oth,
                n_retire=int((lv["kind"] == "정년·계약만료").sum()), n_years=float(n_years),
                table=table, offsets=offsets, misfit=dev,
                dept_table=dept_table, dept_offsets=dept_offsets, n_dept_unmatched=n_unmatched)


LEAVER_HINTS = {"age": COLUMN_HINTS["age"], "tenure": COLUMN_HINTS["tenure"],
                "reason": ["퇴사사유", "퇴직사유", "사유", "reason"],
                "dept": COLUMN_HINTS["dept"],
                "year": ["퇴사연도", "퇴사년도", "퇴사일", "퇴직일", "퇴직연도", "year", "date"]}


def auto_map_leavers(df: pd.DataFrame) -> dict:
    out = {}
    for fld, hints in LEAVER_HINTS.items():
        out[fld] = next((c for h in hints for c in df.columns if _norm(h) in _norm(c)), None)
    return out


# ── 집계표 → 가상 명부 ──────────────────────────────────────────────────────
BAND_TEMPLATE = pd.DataFrame({
    "연령대": ["29세 이하", "30~39세", "40~49세", "50~54세", "55~59세", "60세 이상"],
    "하한": [15, 30, 40, 50, 55, 60], "상한": [29, 39, 49, 54, 59, 75],
    "인원": [120, 260, 330, 140, 110, 40],
    "평균 월임금(만원)": [300, 400, 520, 560, 540, 330],
    "평균 근속(년)": [2.0, 6.5, 13.0, 18.0, 21.0, 16.0],
})


def synthesize_from_bands(table: pd.DataFrame, seed=42, retirement_age=60) -> pd.DataFrame:
    """연령대별 인원·평균임금·평균근속 집계표로 가상 명부를 만든다.
    각 연령대에서 KLIPS 실표본의 같은 연령대 사람을 복원추출해 나이·근속·임금의 '모양'을
    빌려오고, 평균만 입력값에 맞춘다. 개인별 명부를 내놓기 어려운 회사를 위한 차선책이다."""
    if _BASE is None:
        raise FileNotFoundError("big_firm_sample_2023.csv가 없습니다.")
    rng = np.random.default_rng(seed)
    parts = []
    for _, r in table.iterrows():
        n = int(r["인원"])
        if n <= 0:
            continue
        pool = _BASE[(_BASE["age"] >= r["하한"]) & (_BASE["age"] <= r["상한"])]
        s = pool.iloc[rng.integers(0, len(pool), n)].copy()
        if s["wage"].mean() > 0:
            s["wage"] = s["wage"] * (float(r["평균 월임금(만원)"]) / s["wage"].mean())
        if s["tenure"].mean() > 0:
            s["tenure"] = (s["tenure"] * (float(r["평균 근속(년)"]) / s["tenure"].mean()))
        s["tenure"] = np.minimum(s["tenure"], s["age"] - 15).clip(lower=0).round(1)
        parts.append(s)
    if not parts:
        return None
    s = pd.concat(parts, ignore_index=True)
    std = pd.DataFrame(dict(age=s["age"].astype(int), tenure=s["tenure"].astype(float),
                            wage=s["wage"].astype(float), hours=s["hours"].astype(float)))
    std["status"] = (std["age"] > retirement_age).astype(int)
    std["grade_name"], std["dept_name"], std["emp_type"] = "", "전체", ""
    return std


# ── 예시 명부(가상 회사) ────────────────────────────────────────────────────
_DEPTS = ["생산", "품질관리", "연구개발", "영업", "구매·물류", "경영지원", "IT"]
_GRADE_NAMES = ["사원", "대리", "과장", "차장", "부장"]
_DEPT_QUIT = np.array([0.7, 1.0, 1.4, 1.1, 1.0, 1.0, 2.0])   # 예시 회사의 부서별 자발 이직 배수(_DEPTS 순서)


def make_sample_company(n=1200, seed=2026, start_year=2021, end_year=BASE_YEAR):
    """예시용 가상 회사 '한빛정밀(가상)'의 5년 전 명부·현재 명부·퇴사자 명부를 만든다.

    실제 회사 자료가 아니다. KLIPS 표본의 사람을 뽑아 부서·직급·고용형태를 입힌 뒤,
    **STRATA 엔진과는 다른 규칙**(근속 2년 이하가 더 많이 나가고, 신규 입사자는 평균 31세,
    정년 도달자의 45%가 촉탁으로 64세까지 남는 회사)으로 5년을 흘려보내 현재 명부를 만든다.
    엔진과 다른 규칙을 쓰는 이유는 '과거 명부로 검증하기' 기능이 자기 자신을 맞히는
    동어반복이 되지 않게 하기 위해서다. 열 이름·형식은 실제 인사 시스템 추출본처럼
    (생년, 입사일, 연봉) 만들었다."""
    rng = np.random.default_rng(seed)
    pool = _BASE[_BASE["age"] <= 59]
    s = pool.iloc[rng.integers(0, len(pool), n)].reset_index(drop=True)
    base_p = np.array([0.38, 0.10, 0.16, 0.14, 0.08, 0.09, 0.05])
    tilt = np.array([0.045, 0.02, -0.04, 0.0, 0.015, 0.0, -0.05])     # 나이가 많을수록 생산 쪽
    z = (s["age"].to_numpy(float) - 43)[:, None] * tilt[None, :]
    p = base_p[None, :] * np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    order = (s["tenure"] + rng.normal(0, 2.0, n)).rank(pct=True).to_numpy()
    cuts = [0.30, 0.54, 0.75, 0.90, 1.01]
    st = dict(id=np.arange(n), age=s["age"].to_numpy(float).copy(), tenure=s["tenure"].to_numpy(float).copy(),
              wage=s["wage"].to_numpy(float).copy(), hours=s["hours"].to_numpy(float).copy(),
              dept=np.array([rng.choice(7, p=row) for row in p]),
              grade=np.searchsorted(cuts, order).clip(0, 4), yig=rng.integers(0, 5, n).astype(float),
              ce=np.zeros(n, dtype=bool), hired=np.zeros(n, dtype=int))
    st["hired"] = (start_year - st["tenure"]).astype(int)
    target = np.bincount(st["dept"], minlength=7).astype(float)
    next_id = n

    def frame(state, year):
        m = len(state["age"])
        month = (state["id"] * 7) % 12 + 1
        df = pd.DataFrame({
            "사번": [f"H{100000 + int(i)}" for i in state["id"]],
            "부서": [_DEPTS[d] for d in state["dept"]],
            "직급": np.where(state["ce"], "촉탁", [_GRADE_NAMES[g] for g in state["grade"]]),
            "고용형태": np.where(state["ce"], "촉탁(정년 후 재고용)", "정규직"),
            "생년": (year - state["age"]).astype(int),
            "입사일": [f"{int(h)}-{int(mo):02d}-01" for h, mo in zip(state["hired"], month)],
            "연봉(만원)": (state["wage"] * 12 * 1.08).round(-1).astype(int),
            "주당근로시간": state["hours"].round().astype(int)})
        # 인턴 몇 명(분석에서 자동 제외되는 고용형태의 예)
        k = max(int(0.01 * m), 3)
        young = df.index[(year - df["생년"] < 28) & (df["고용형태"] == "정규직")][:k]
        df.loc[young, "고용형태"] = "인턴"
        return df

    roster_past = frame(st, start_year)
    leavers = []
    for year in range(start_year, end_year):
        m = len(st["age"])
        reg = ~st["ce"]
        wf = dict(age=st["age"], tenure=st["tenure"], wage=st["wage"], hours=st["hours"])
        b0 = _calibrate(dict(wf, status=(~reg).astype(int)), 0.028)
        ph = _hazard(wf, b0) * np.where(st["tenure"] <= 2, 1.6, np.where(st["tenure"] > 10, 0.7, 1.0))
        ph = ph * _DEPT_QUIT[st["dept"]]          # IT·연구개발은 이직이 잦고 생산은 드문 회사
        vol = reg & (rng.random(m) < ph)
        oth = reg & ~vol & (rng.random(m) < 0.03)
        ret = reg & ~vol & ~oth & (st["age"] >= 60)
        stay_ce = ret & (rng.random(m) < 0.45)
        ce_end = st["ce"] & (st["age"] >= 64)
        out = vol | oth | (ret & ~stay_ce) | ce_end
        for mask, reasons in ((vol, ["개인 사유(이직)", "개인 사유(학업)", "자발적 사직"]),
                              (oth, ["권고사직", "계약 해지", "기타"]),
                              (ret & ~stay_ce, ["정년퇴직"]), (ce_end, ["계약 만료(촉탁)"])):
            for i in np.where(mask)[0]:
                leavers.append((year, int(st["age"][i]), float(st["tenure"][i]), str(rng.choice(reasons)),
                                _DEPTS[st["dept"][i]]))
        st["wage"][stay_ce] *= 0.63
        st["ce"][stay_ce] = True
        for k in st:
            st[k] = st[k][~out]
        # 승진: 4년 이상 머문 정규직의 25%
        elig = (~st["ce"]) & (st["yig"] >= 4) & (st["grade"] < 4)
        up = elig & (rng.random(len(elig)) < 0.25)
        st["grade"][up] += 1
        st["yig"][up] = 0
        # 충원: 결원이 큰 부서부터, 평균 31세 경력 혼합
        n_hire = n - len(st["age"])
        if n_hire > 0:
            cur = np.bincount(st["dept"], minlength=7).astype(float)
            short = np.clip(target - cur, 0, None)
            pd_ = short / short.sum() if short.sum() > 0 else target / target.sum()
            a = np.clip(rng.normal(31, 5, n_hire), 22, 50).round()
            new = dict(id=np.arange(next_id, next_id + n_hire), age=a, tenure=np.zeros(n_hire),
                       wage=(250 + (a - 25) * 9 + rng.normal(0, 25, n_hire)).clip(220, 700),
                       hours=np.full(n_hire, 40.0), dept=rng.choice(7, n_hire, p=pd_),
                       grade=np.where(a >= 40, 2, np.where(a >= 33, 1, 0)), yig=np.zeros(n_hire),
                       ce=np.zeros(n_hire, dtype=bool), hired=np.full(n_hire, year + 1))
            next_id += n_hire
            for k in st:
                st[k] = np.concatenate([st[k], new[k]])
        st["age"] += 1
        st["tenure"] += 1
        st["yig"] += 1
        r = ~st["ce"]
        st["wage"][r] *= 1 + np.where(st["age"][r] < 40, 0.035, np.where(st["age"][r] < 50, 0.02, 0.005))
    roster_now = frame(st, end_year)
    lv = pd.DataFrame(leavers, columns=["퇴사연도", "나이", "근속연수", "퇴사사유", "부서"])
    lv = lv[lv["퇴사연도"] >= end_year - 2].sample(frac=1, random_state=seed).reset_index(drop=True)
    return roster_past, roster_now, lv


def make_sample_roster(n=1200, seed=2026, base_year=BASE_YEAR):
    """(현재 명부, 퇴사자 명부). make_sample_company의 간편 호출."""
    _, now, lv = make_sample_company(n=n, seed=seed, end_year=base_year, start_year=base_year - 5)
    return now, lv


TEMPLATE = pd.DataFrame({
    "연령대": ["29세 이하", "30~39세", "40~49세", "50~54세", "55~59세", "60세 이상"],
    "하한": [15, 30, 40, 50, 55, 60], "상한": [29, 39, 49, 54, 59, 75],
    "인원": [120, 260, 330, 140, 110, 40],
    "평균 월임금(만원)": [300, 400, 520, 560, 540, 330],
    "평균 근속(년)": [2.0, 6.5, 13.0, 18.0, 21.0, 16.0],
})


def synthesize_from_bands(table: pd.DataFrame, seed=42, retirement_age=60) -> pd.DataFrame:
    """연령대별 인원·평균임금·평균근속 집계표로 가상 명부를 만든다.
    각 연령대에서 KLIPS 실표본의 같은 연령대 사람을 복원추출해 나이·근속·임금의 '모양'을
    빌려오고, 평균만 입력값에 맞춘다. 개인별 명부를 내놓기 어려운 회사를 위한 차선책이다."""
    if _BASE is None:
        raise FileNotFoundError("big_firm_sample_2023.csv가 없습니다.")
    rng = np.random.default_rng(seed)
    parts = []
    for _, r in table.iterrows():
        n = int(r["인원"])
        if n <= 0:
            continue
        pool = _BASE[(_BASE["age"] >= r["하한"]) & (_BASE["age"] <= r["상한"])]
        s = pool.iloc[rng.integers(0, len(pool), n)].copy()
        if s["wage"].mean() > 0:
            s["wage"] = s["wage"] * (float(r["평균 월임금(만원)"]) / s["wage"].mean())
        if s["tenure"].mean() > 0:
            s["tenure"] = (s["tenure"] * (float(r["평균 근속(년)"]) / s["tenure"].mean()))
        s["tenure"] = np.minimum(s["tenure"], s["age"] - 15).clip(lower=0).round(1)
        parts.append(s)
    if not parts:
        return None
    s = pd.concat(parts, ignore_index=True)
    std = pd.DataFrame(dict(age=s["age"].astype(int), tenure=s["tenure"].astype(float),
                            wage=s["wage"].astype(float), hours=s["hours"].astype(float)))
    std["status"] = (std["age"] > retirement_age).astype(int)
    std["grade_name"], std["dept_name"], std["emp_type"] = "", "전체", ""
    return std


# ── 예시 명부(가상 회사) ────────────────────────────────────────────────────
def make_sample_roster(n=1200, seed=2026, base_year=BASE_YEAR):
    """예시용 가상 회사 '한빛정밀(가상)'의 명부와 퇴사자 명부를 만든다.
    실제 회사 자료가 아니다. KLIPS 표본의 사람을 뽑아 부서·직급·고용형태를 입힌 것으로,
    생산 부문이 고령화되고 연구개발·IT가 젊은 전형적인 제조업 구조를 흉내 낸다.
    열 이름과 형식은 일부러 실제 인사 시스템 추출본처럼(생년, 입사일, 연봉) 만들었다."""
    rng = np.random.default_rng(seed)
    pool = _BASE[_BASE["age"] <= 64]
    s = pool.iloc[rng.integers(0, len(pool), n)].reset_index(drop=True).copy()
    depts = ["생산", "품질관리", "연구개발", "영업", "구매·물류", "경영지원", "IT"]
    base_p = np.array([0.38, 0.10, 0.16, 0.14, 0.08, 0.09, 0.05])
    tilt = np.array([0.045, 0.02, -0.04, 0.0, 0.015, 0.0, -0.05])     # 나이가 많을수록 생산 쪽
    z = (s["age"].to_numpy(float) - 43)[:, None] * tilt[None, :]
    p = base_p[None, :] * np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    s["부서"] = [depts[rng.choice(len(depts), p=row)] for row in p]
    ce = s["age"] >= 60
    keep = ~ce | (rng.random(n) < 0.45)        # 60세 이상은 일부만 촉탁으로 남아 있다
    s, ce = s[keep].reset_index(drop=True), ce[keep].reset_index(drop=True)
    order = (s["tenure"] + rng.normal(0, 2.0, len(s))).rank(pct=True)
    cuts, names = [0.30, 0.54, 0.75, 0.90, 1.01], ["사원", "대리", "과장", "차장", "부장"]
    s["직급"] = [names[int(np.searchsorted(cuts, v))] for v in order]
    s["고용형태"] = np.where(ce, "촉탁(정년 후 재고용)", "정규직")
    s.loc[ce, "직급"] = "촉탁"
    interns = rng.random(len(s)) < 0.012
    s.loc[interns & ~ce & (s["age"] < 30), "고용형태"] = "인턴"
    s["생년"] = (base_year - s["age"]).astype(int)
    month = rng.integers(1, 13, len(s))
    s["입사일"] = [f"{int(base_year - t)}-{m:02d}-01" for t, m in zip(s["tenure"], month)]
    s["연봉(만원)"] = (s["wage"] * 12 * 1.08).round(-1).astype(int)
    s["주당근로시간"] = s["hours"].round().astype(int)
    s.insert(0, "사번", [f"H{100000 + i}" for i in rng.permutation(len(s))])
    roster = s[["사번", "부서", "직급", "고용형태", "생년", "입사일", "연봉(만원)", "주당근로시간"]]

    # 퇴사자: 근속이 짧은 층이 모형 예측보다 더 많이 나가는 회사로 설정
    reg = s[s["고용형태"] == "정규직"]
    wf = dict(age=reg["age"].to_numpy(float), tenure=reg["tenure"].to_numpy(float),
              wage=reg["wage"].to_numpy(float), hours=reg["hours"].to_numpy(float),
              status=np.zeros(len(reg), dtype=int))
    b0 = _calibrate(wf, 0.028)
    ph = _hazard(wf, b0) * np.where(wf["tenure"] <= 2, 1.6, np.where(wf["tenure"] > 10, 0.7, 1.0))
    vol = rng.random(len(reg)) < ph
    oth = (rng.random(len(reg)) < 0.03) & ~vol
    reasons_v = ["개인 사유(이직)", "개인 사유(학업)", "자발적 사직"]
    reasons_o = ["권고사직", "계약 해지", "기타"]
    rows = []
    for i in np.where(vol)[0]:
        rows.append((base_year - 1, int(wf["age"][i]), float(wf["tenure"][i]), rng.choice(reasons_v)))
    for i in np.where(oth)[0]:
        rows.append((base_year - 1, int(wf["age"][i]), float(wf["tenure"][i]), rng.choice(reasons_o)))
    for _ in range(int(0.02 * len(reg))):
        rows.append((base_year - 1, 60, float(rng.integers(15, 35)), "정년퇴직"))
    leavers = pd.DataFrame(rows, columns=["퇴사연도", "나이", "근속연수", "퇴사사유"])
    leavers = leavers.sample(frac=1, random_state=seed).reset_index(drop=True)
    return roster, leavers


TEMPLATE = pd.DataFrame({
    "부서": ["생산", "영업", "경영지원"], "직급": ["과장", "대리", "촉탁"],
    "고용형태": ["정규직", "정규직", "촉탁(정년 후 재고용)"],
    "생년": [1975, 1990, 1964], "입사일": ["2003-03-01", "2018-07-01", "1992-01-01"],
    "연봉(만원)": [7200, 4800, 4200], "주당근로시간": [40, 40, 40],
})


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).parent / "sample_data"
    out.mkdir(exist_ok=True)
    past, roster, leavers = make_sample_company()
    past.to_csv(out / "sample_roster_2021.csv", index=False, encoding="utf-8-sig")
    roster.to_csv(out / "sample_roster.csv", index=False, encoding="utf-8-sig")
    leavers.to_csv(out / "sample_leavers.csv", index=False, encoding="utf-8-sig")
    TEMPLATE.to_csv(out / "roster_template.csv", index=False, encoding="utf-8-sig")
    print("예시 명부", len(roster), "명 / 5년 전 명부", len(past), "명 / 퇴사자", len(leavers), "명 →", out)
