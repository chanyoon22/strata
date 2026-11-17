"""ui_data.py — ① 회사 데이터: 명부 올리기·점검·인력구조 진단."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import company_data as C
from ui_common import (INK, SUB, LINE, TEAL, SEQ, GROUP_C, html, gap, section, note, card, chip,
                       base_layout, show, kpi)

_DIR = Path(__file__).parent
SRC_DEMO, SRC_UPLOAD, SRC_BANDS = "표준 표본으로 체험", "우리 회사 명부 올리기", "연령대 집계표로 입력"
MOEL_55P_2023 = 0.188      # 고용노동부 고령자 고용현황: 300인 이상 사업장 55세 이상 비중(2023년 말)
NONE = "(없음)"


def _csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


@st.cache_data(show_spinner=False)
def _sample_files():
    """(현재 명부, 퇴사자 명부, 5년 전 명부) — 가상 회사 예시."""
    d = _DIR / "sample_data"
    names = ["sample_roster.csv", "sample_leavers.csv", "sample_roster_2021.csv"]
    if not all((d / n).exists() for n in names):
        past, roster, leavers = C.make_sample_company()
        return roster, leavers, past
    return tuple(pd.read_csv(d / n, encoding="utf-8-sig") for n in names)


@st.cache_data(show_spinner=False)
def _sample_zip(leavers, past):
    import io, zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("예시_퇴사자_명부.csv", _csv_bytes(leavers))
        z.writestr("예시_명부_2021년.csv", _csv_bytes(past))
    return buf.getvalue()


def _clear_downstream():
    for k in ("res", "sens", "robust", "tornado", "space", "report"):
        st.session_state.pop(k, None)


def _set_data(**kw):
    st.session_state["data"] = kw
    _clear_downstream()


def render(cfg):
    section("어떤 데이터로 분석할까요?",
            "우리 회사 인사 명부를 올리면 실제 인력 구조 그대로 계산합니다.<br>"
            "명부가 없어도 표준 표본으로 먼저 체험해볼 수 있습니다.")
    src = st.segmented_control("데이터 종류", [SRC_DEMO, SRC_UPLOAD, SRC_BANDS],
                               default=cfg["src"], key="w_src", label_visibility="collapsed")
    src = src or cfg["src"]
    if src != cfg["src"]:
        cfg["src"] = src
        st.session_state.pop("data", None)
        st.session_state.pop("raw", None)
        _clear_downstream()
    gap(6)
    if src == SRC_DEMO:
        _demo(cfg)
    elif src == SRC_UPLOAD:
        _upload(cfg)
    else:
        _bands(cfg)

    data = st.session_state.get("data")
    if data and data.get("std") is not None:
        _diagnosis(data, cfg)
        gap(8)
        if st.button("다음: 가정과 시나리오 정하기 →", type="primary"):
            st.session_state["goto"] = "② 가정·시나리오"
            st.rerun()


# ── A. 표준 표본 ────────────────────────────────────────────────────────────
def _demo(cfg):
    note("한국노동패널(KLIPS)에 실제로 응답한 <b>300인 이상 사업장 재직자</b>를 뽑아 "
         "가상의 회사를 구성합니다.<br>나이·근속·임금의 관계가 실제 대기업 근로자와 같게 유지됩니다.")
    gap(4)
    c = st.columns([1, 1, 2])
    n = c[0].number_input("직원 수 (명)", 100, 20000, int(cfg["demo_n"]), 100, key="w_demo_n")
    w = c[1].number_input("월평균임금 (만원)", 200, 1500, int(cfg["demo_wage"]), 5, key="w_demo_wage",
                          help="전체 직원의 월평균 임금입니다. 표본의 임금 수준을 이 값에 맞춥니다.")
    ce = c[2].radio("지금 정년퇴직자를 다시 고용하고 있나요?",
                    ["아니요 (정년 60세에 모두 퇴직)", "예 (표본에서 추정한 수준으로)"],
                    index=1 if cfg["demo_ce"] else 0, key="w_demo_ce", horizontal=True,
                    help="'예'를 고르면 표본의 60세 이상 재직자 비중으로 현재 재고용 수준을 추정해, "
                         "현행 유지 시나리오에도 그 관행이 이어지는 것으로 계산합니다.")
    changed = (n != cfg["demo_n"] or w != cfg["demo_wage"] or ce.startswith("예") != cfg["demo_ce"])
    cfg["demo_n"], cfg["demo_wage"], cfg["demo_ce"] = n, w, ce.startswith("예")
    data = st.session_state.get("data")
    if changed or not data or data.get("mode") != SRC_DEMO:
        std = C.klips_standard(include_post_retirement=cfg["demo_ce"])
        std = std.assign(wage=std["wage"] * (w / std["wage"].mean()))
        _set_data(mode=SRC_DEMO, std=std, quality=None, grade_table=None, name="표준 대기업 표본",
                  ce=C.infer_ce_practice(std) if cfg["demo_ce"] else dict(rate=0.0, end_age=60, wage_ratio=0.55))


# ── B. 명부 업로드 ──────────────────────────────────────────────────────────
def _upload(cfg):
    roster_s, leavers_s, past_s = _sample_files()
    c = st.columns([1.15, 1, 1, 1])
    use_sample = c[0].button("예시 명부로 바로 해보기", width="stretch",
                             help="가상의 제조업체 '한빛정밀'(약 1,150명) 명부를 불러옵니다. 실제 회사가 아닙니다.")
    c[1].download_button("빈 양식 받기", _csv_bytes(C.TEMPLATE), "명부_양식.csv", "text/csv", width="stretch")
    c[2].download_button("예시 명부 받기", _csv_bytes(roster_s), "예시_명부_한빛정밀.csv", "text/csv", width="stretch")
    c[3].download_button("예시 퇴사자·과거 명부 받기", _sample_zip(leavers_s, past_s), "예시_퇴사자_과거명부.zip",
                         "application/zip", width="stretch")
    note("필요한 정보는 세 가지뿐입니다 — <b>나이(또는 생년) · 근속(또는 입사일) · 임금(월급 또는 연봉)</b>.<br>"
         "부서·직급·고용형태가 있으면 부서별 진단과 승진 분석이 더 정확해집니다. "
         "이름·사번은 필요 없고, 있어도 읽는 즉시 버립니다.", "fine")
    up = st.file_uploader("인사 명부 파일 (CSV 또는 엑셀)", type=["csv", "xlsx", "xls"], key="w_roster")
    # 올린 파일은 이름+크기로 구분한다(같은 이름으로 고쳐 올려도 다시 읽도록). 예시 명부를 고른 뒤에는
    # 업로드 칸에 남아 있는 예전 파일이 예시를 다시 덮어쓰지 않게 건너뛴다.
    up_key = f"{up.name}:{up.size}" if up is not None else None
    if use_sample:
        _load_raw(roster_s.copy(), "예시 명부(한빛정밀·가상)")
        st.session_state["sample_leavers"] = True
        st.session_state["raw_key"], st.session_state["skip_upload"] = "sample", up_key
    elif up is not None and up_key not in (st.session_state.get("raw_key"), st.session_state.get("skip_upload")):
        try:
            _load_raw(C.read_table(up, up.name), up.name)
            st.session_state.pop("sample_leavers", None)
            st.session_state["raw_key"] = up_key
            st.session_state.pop("skip_upload", None)
        except Exception as e:
            st.error(str(e))
    raw = st.session_state.get("raw")
    if raw is None:
        note("이 화면에서 올린 파일은 분석하는 동안에만 서버 메모리에 머물고, 어디에도 저장되지 않습니다. "
             "민감한 자료라면 이 프로그램을 회사 PC에 내려받아 실행하는 방법(README 참고)을 권합니다.", "fine")
        return

    html(f'<div class="sec" style="margin-top:14px">열 맞추기 — {st.session_state["raw_name"]}</div>'
         f'<div class="sec-sub">파일의 어느 열이 무엇인지 자동으로 골라봤습니다. 틀린 곳만 고쳐주세요.</div>')
    st.dataframe(raw.head(4), width="stretch", hide_index=True)
    m0 = st.session_state["raw_map"]
    cols = [NONE] + list(raw.columns)

    def pick(col, label, guess, key, help=None):
        idx = cols.index(guess) if guess in cols else 0
        v = col.selectbox(label, cols, index=idx, key=f"w_map_{key}", help=help)
        return None if v == NONE else v

    r1 = st.columns(3)
    age_col = pick(r1[0], "나이 또는 생년", m0.get("age") or m0.get("birth"), "age")
    ten_col = pick(r1[1], "근속연수 또는 입사일", m0.get("tenure") or m0.get("hire"), "ten")
    wage_col = pick(r1[2], "임금 (월급 또는 연봉)", m0.get("wage") or m0.get("wage_annual"), "wage")
    r2 = st.columns(4)
    dept_col = pick(r2[0], "부서 (선택)", m0.get("dept"), "dept")
    grade_col = pick(r2[1], "직급 (선택)", m0.get("grade"), "grade")
    emp_col = pick(r2[2], "고용형태 (선택)", m0.get("emp_type"), "emp")
    hours_col = pick(r2[3], "주당 근로시간 (선택)", m0.get("hours"), "hours")

    def looks_year(col):
        if not col:
            return False
        v = C._to_year(raw[col].dropna().head(200))
        num = pd.to_numeric(raw[col], errors="coerce")
        return bool(v.notna().mean() > 0.8 and (num.isna().all() or num.median() > 150))

    age_is_birth, ten_is_hire = looks_year(age_col), looks_year(ten_col)
    annual_guess = bool(wage_col and (wage_col == m0.get("wage_annual")))
    unit_guess, annual_guess, unit_msg = (C.detect_wage_unit(raw[wage_col], annual_guess)
                                          if wage_col else ("만원", False, ""))
    r3 = st.columns(4)
    wage_kind = r3[0].selectbox("임금 열의 종류", ["월급", "연봉"], index=1 if annual_guess else 0, key="w_wkind")
    wage_unit = r3[1].selectbox("임금 단위", list(C.WAGE_UNITS), index=list(C.WAGE_UNITS).index(unit_guess),
                                key="w_wunit", help=unit_msg)
    base_year = r3[2].number_input("명부 기준 연도", 2020, 2035, int(cfg["start_year"]), key="w_base_year",
                                   help="생년·입사일을 나이·근속으로 바꿀 때 기준이 되는 해입니다.")
    ret_age = r3[3].number_input("현재 우리 회사 정년", 55, 70, int(cfg["ret_age"]), key="w_ret_age")
    ce_vals = excl_vals = None
    if emp_col:
        vals = [str(v) for v in raw[emp_col].dropna().astype(str).str.strip().unique()]
        r4 = st.columns(2)
        ce_vals = r4[0].multiselect("정년 후 재고용(촉탁)에 해당하는 값", vals,
                                    default=[v for v in vals if any(C._norm(k) in C._norm(v) for k in C.CE_KEYWORDS)],
                                    key="w_ce_vals")
        excl_vals = r4[1].multiselect("분석에서 뺄 값 (파견·인턴 등)", vals,
                                      default=[v for v in vals if any(C._norm(k) in C._norm(v) for k in C.EXCLUDE_KEYWORDS)],
                                      key="w_excl_vals")
    if st.button("이 설정으로 명부 읽기", type="primary"):
        mapping = dict(age=None if age_is_birth else age_col, birth=age_col if age_is_birth else None,
                       tenure=None if ten_is_hire else ten_col, hire=ten_col if ten_is_hire else None,
                       wage=wage_col, dept=dept_col, grade=grade_col, emp_type=emp_col, hours=hours_col,
                       yrs_in_grade=m0.get("yrs_in_grade"))
        read_args = dict(mapping=mapping, wage_unit=wage_unit, wage_is_annual=(wage_kind == "연봉"),
                         retirement_age=int(ret_age), ce_values=ce_vals, exclude_values=excl_vals)
        std, q = C.standardize(raw, base_year=int(base_year), **read_args)
        if st.session_state.get("raw_dup"):
            q["issues"].insert(0, dict(level="warn", title="같은 식별자가 여러 번 나옵니다",
                                       detail=f"사번·이름이 겹치는 행이 {st.session_state['raw_dup']}건 있습니다. "
                                              "같은 사람이 두 번 들어갔는지 확인해 주세요.",
                                       n=st.session_state["raw_dup"]))
        cfg["start_year"], cfg["ret_age"] = int(base_year), int(ret_age)
        if std is None:
            st.session_state["data"] = dict(mode=SRC_UPLOAD, std=None, quality=q)
        else:
            ce = C.infer_ce_practice(std, int(ret_age))
            _set_data(mode=SRC_UPLOAD, std=std, quality=q, grade_table=C.infer_grades(std),
                      name=st.session_state["raw_name"], ce=ce, leaver=None, read_args=read_args,
                      base_year=int(base_year), columns=list(raw.columns))
            st.session_state.pop("backtest", None)
            cfg["ce_rate"], cfg["ce_end"], cfg["ce_ratio"] = ce["rate"], ce["end_age"], ce["wage_ratio"]
            if ce["rate"] > 0:
                cfg["accept"] = max(cfg["accept"], ce["rate"])
                cfg["wage_ratio"] = max(cfg["wage_ratio"], ce["wage_ratio"])

    data = st.session_state.get("data")
    if not data or data.get("mode") != SRC_UPLOAD:
        return
    _quality(data["quality"])
    if data.get("std") is None:
        return
    _grades(data)
    _leavers(data, cfg, leavers_s)
    _backtest(data, cfg, past_s)


def _load_raw(df, name):
    m = C.auto_map(df)
    dup = 0
    if m.get("id"):
        dup = int(df[m["id"]].duplicated().sum())
    id_cols = [c for c in df.columns if any(C._norm(h) in C._norm(c) for h in C.COLUMN_HINTS["id"])]
    df = df.drop(columns=id_cols, errors="ignore")       # 식별자는 읽는 즉시 버린다
    st.session_state["raw"], st.session_state["raw_name"] = df, name
    st.session_state["raw_map"], st.session_state["raw_dup"] = m, dup
    st.session_state.pop("data", None)
    for k in [k for k in st.session_state if k.startswith("w_map_") or k in
              ("w_wkind", "w_wunit", "w_ce_vals", "w_excl_vals")]:
        del st.session_state[k]
    _clear_downstream()


def _quality(q):
    if not q:
        return
    section("데이터 점검 결과")
    errs = [i for i in q["issues"] if i["level"] == "error"]
    warns = [i for i in q["issues"] if i["level"] == "warn"]
    c = st.columns(4)
    kpi(c[0], "올린 행", f"{q['n_raw']:,}")
    kpi(c[1], "분석에 쓰는 인원", f"{q['n_used']:,}명")
    kpi(c[2], "제외한 행", f"{q['dropped']:,}")
    kpi(c[3], "확인이 필요한 항목", f"{len(warns) + len(errs)}건")
    gap(8)
    if errs:
        st.error("명부를 읽지 못했습니다. 아래 내용을 확인하고 열 지정을 고쳐주세요.")
    elif not warns:
        html(f'{chip("문제 없음")}&nbsp; <span class="note">빠진 값이나 이상한 값 없이 모두 읽었습니다.</span>')
    order = {"error": 0, "warn": 1, "info": 2}
    tag = {"error": "오류", "warn": "확인 필요", "info": "자동 처리"}
    for i in sorted(q["issues"], key=lambda x: order[x["level"]]):
        html(f'<div class="issue {i["level"]}"><div class="issue-t">[{tag[i["level"]]}] {i["title"]}</div>'
             f'<div class="issue-d">{i["detail"]}</div></div>')


def _grades(data):
    gt = data.get("grade_table")
    if gt is None or len(gt) < 2:
        return
    section("직급 체계 확인",
            "명부의 임금·근속으로 직급 순서를 추정했습니다. 순서가 틀렸으면 '서열' 숫자를 고쳐주세요(1이 가장 낮은 직급).<br>"
            "'최소 체류 연수'는 다음 직급으로 승진하려면 그 직급에 최소 몇 년 있어야 하는지입니다.")
    edited = st.data_editor(
        gt, key="w_grade_table", hide_index=True, width="stretch",
        disabled=["직급", "인원", "근속중앙값", "임금중앙값"],
        column_config={"서열": st.column_config.NumberColumn(min_value=1, max_value=len(gt), step=1),
                       "최소체류연수": st.column_config.NumberColumn("최소 체류 연수", min_value=1, max_value=99, step=1),
                       "근속중앙값": st.column_config.NumberColumn("근속 중앙값(년)", format="%.1f"),
                       "임금중앙값": st.column_config.NumberColumn("월 임금 중앙값(만원)", format="%.0f")})
    if not edited.equals(gt):
        if edited["서열"].nunique() < len(edited):
            st.warning("서열 숫자가 겹칩니다. 직급마다 다른 숫자를 넣어주세요.")
        else:
            data["grade_table"] = edited.sort_values("서열").reset_index(drop=True)
            for k in ("res", "sens", "robust", "tornado", "space", "report"):
                st.session_state.pop(k, None)


def _leavers(data, cfg, sample):
    section("퇴사자 명부가 있다면 (선택)",
            "최근 1~3년 퇴사자 명부(나이·근속·퇴사 사유)를 올리면 우리 회사의 실제 퇴사율을 계산해 넣고,<br>"
            "근속 구간별로 모형 예측과 실제가 얼마나 다른지 비교해 그 차이까지 반영합니다.<br>"
            "<b>부서(직군) 열</b>까지 있으면 직군마다 퇴사가 얼마나 잦은지도 따로 반영합니다.")
    up = st.file_uploader("퇴사자 명부 (CSV 또는 엑셀)", type=["csv", "xlsx", "xls"], key="w_leavers")
    lv = None
    if up is not None:
        try:
            lv = C.read_table(up, up.name)
        except Exception as e:
            st.error(str(e))
    elif st.session_state.get("sample_leavers"):
        lv = sample
        note("예시 명부를 쓰는 중이라 예시 퇴사자 명부를 함께 불러왔습니다.", "fine")
    if lv is None:
        return
    cal = C.calibrate_from_leavers(data["std"], lv, C.auto_map_leavers(lv))
    c = st.columns(4)
    kpi(c[0], "자발적 퇴사율", f"{cal['push_rate']*100:.1f}%", f"퇴사자 {cal['n_vol']}명")
    kpi(c[1], "기타 퇴사율", f"{cal['other_rate']*100:.1f}%", f"퇴사자 {cal['n_other']}명")
    kpi(c[2], "정년·계약만료", f"{cal['n_retire']}명", "퇴사율 계산에서 제외")
    kpi(c[3], "집계 기간", f"{cal['n_years']:.0f}년")
    gap(8)
    if cal["table"] is not None:
        t = cal["table"]
        fig = go.Figure()
        fig.add_bar(x=t["근속구간"], y=t["모형예측"], name="모형이 예측한 퇴사자", marker_color="#9ec5f4",
                    hovertemplate="%{x}<br>모형 예측 %{y:.1f}명<extra></extra>")
        fig.add_bar(x=t["근속구간"], y=t["실제퇴사"], name="실제 퇴사자", marker_color="#256abf",
                    hovertemplate="%{x}<br>실제 %{y}명<extra></extra>")
        base_layout(fig, 300, unit="퇴사자 수 (명)", barmode="group", bargap=0.35, bargroupgap=0.08,
                    legend=dict(orientation="h", y=-0.16, x=0))
        fig.update_traces(marker_cornerradius=4)
        show(fig, "leaver_fit", bar=False)
        worst = t.iloc[(t["실제퇴사"] - t["모형예측"]).abs().idxmax()]
        diff = worst["실제퇴사"] - worst["모형예측"]
        if cal["misfit"] < 0.1:
            msg = "모형 예측과 실제 퇴사 분포가 <b>잘 맞습니다.</b> 보정 없이 써도 무방합니다."
        else:
            msg = (f"<b>{worst['근속구간']}</b> 구간에서 실제 퇴사자가 모형 예측보다 "
                   f"<b>{abs(diff):.0f}명 {'많습니다' if diff > 0 else '적습니다'}.</b> "
                   "이 차이를 반영하면 우리 회사에 더 가까운 결과가 나옵니다.")
        note(msg + "<br>퇴사자가 적은 구간은 우연일 수 있어 보정을 약하게 넣습니다(사건이 10건이면 절반만 반영).")
    else:
        note("자발적 퇴사자가 10명 미만이라 근속 구간별 비교는 하지 않고, 퇴사율만 반영합니다.", "fine")
    _leaver_depts(cal)
    use = st.checkbox("이 퇴사율과 퇴사 패턴을 분석에 반영", value=bool(data.get("leaver")), key="w_use_leaver")
    prev = bool(data.get("leaver"))
    if use:
        data["leaver"] = cal
        if not prev:
            cfg["push"], cfg["other"] = round(cal["push_rate"] * 100, 1), round(cal["other_rate"] * 100, 1)
            for k in ("w_push", "w_other", "res", "sens", "robust", "tornado", "space", "report"):
                st.session_state.pop(k, None)
    elif prev:
        data["leaver"] = None
        for k in ("res", "sens", "robust", "tornado", "space", "report"):
            st.session_state.pop(k, None)


def _leaver_depts(cal):
    t = cal.get("dept_table")
    if t is None:
        if cal.get("n_dept_unmatched"):
            note(f"퇴사자 명부의 부서 이름 {cal['n_dept_unmatched']}건이 지금 명부의 부서 이름과 맞지 않아 "
                 "직군별 비교는 하지 않았습니다. 두 파일의 부서 이름을 같게 맞춰 주세요.", "fine")
        return
    t = t.assign(배수=np.exp(t["보정값"])).sort_values("배수")
    gap(6)
    html('<div class="card-t">직군별로 보면 — 회사 평균보다 퇴사가 잦은 곳과 드문 곳</div>')
    fig = go.Figure()
    fig.add_bar(y=t["직군"], x=t["배수"] - 1, base=1, orientation="h",
                marker_color=np.where(t["배수"] >= 1, "#256abf", "#9ec5f4"),
                customdata=t[["실제퇴사", "모형예측", "재직자"]].to_numpy(),
                hovertemplate="%{y}<br>회사 평균의 %{x:.2f}배<br>실제 %{customdata[0]}명 · "
                              "평균 수준이면 %{customdata[1]:.1f}명 (재직 %{customdata[2]}명)<extra></extra>")
    fig.add_vline(x=1, line=dict(color="#8a8f98", width=1, dash="dot"))
    base_layout(fig, 70 + 34 * len(t), xaxis_title="자발적 퇴사 (회사 평균 = 1배)")
    fig.update_xaxes(range=[min(0.4, t["배수"].min() - 0.1), max(1.6, t["배수"].max() + 0.1)], ticksuffix="배")
    fig.update_yaxes(showgrid=False)
    show(fig, "leaver_dept", bar=False)
    hi, lo = t.iloc[-1], t.iloc[0]
    from interpret import josa
    note(f"<b>{josa(hi['직군'], '은', '는')}</b> 회사 평균의 <b>{hi['배수']:.2f}배</b>, "
         f"<b>{josa(lo['직군'], '은', '는')}</b> <b>{lo['배수']:.2f}배</b> 수준으로 스스로 그만둡니다. "
         "이 차이를 직군별로 반영합니다.<br>"
         "퇴사자가 적은 직군은 우연일 수 있어 회사 평균 쪽으로 당겨서 넣습니다. 그래서 막대가 실제 비율보다 짧게 보일 수 있습니다."
         + (f" 부서 이름이 맞지 않은 퇴사자 {cal['n_dept_unmatched']}명은 직군 비교에서 뺐습니다." if cal.get("n_dept_unmatched") else ""),
         "fine")


def _backtest(data, cfg, sample_past):
    section("몇 년 전 명부가 있다면 — 모형이 우리 회사를 맞히는지 확인 (선택)",
            "과거 명부에서 출발해 '제도 변화 없음'으로 오늘까지 계산한 뒤, 모형이 예측한 오늘의 연령 구성을 "
            "<b>실제 오늘 명부와 비교</b>합니다.<br>모형이 과거를 잘 맞혔다면 앞으로의 예측도 더 믿을 수 있습니다. "
            "지금 명부와 같은 형식(같은 열 이름)의 파일이어야 합니다.")
    c = st.columns([3, 1])
    up = c[0].file_uploader("과거 명부 (CSV 또는 엑셀)", type=["csv", "xlsx", "xls"], key="w_past")
    is_sample = bool(st.session_state.get("sample_leavers")) and up is None
    past_year = c[1].number_input("과거 명부의 기준 연도", 2005, int(data["base_year"]) - 1,
                                  int(data["base_year"]) - 5, key="w_past_year")
    raw = None
    if up is not None:
        try:
            raw = C.read_table(up, up.name)
        except Exception as e:
            st.error(str(e))
    elif is_sample:
        raw = sample_past
        note("예시 명부를 쓰는 중이라 같은 가상 회사의 5년 전(2021년) 명부를 함께 불러왔습니다.", "fine")
    if raw is None:
        return
    need = [v for k, v in data["read_args"]["mapping"].items() if v]
    miss = [col for col in need if col not in raw.columns]
    if miss:
        st.error(f"과거 명부에 다음 열이 없습니다: {', '.join(miss)}. 지금 명부와 같은 열 이름으로 맞춰주세요.")
        return
    if st.button("과거 명부로 검증하기 (약 10초)"):
        import backtest as B
        past, q = C.standardize(raw, base_year=int(past_year), **data["read_args"])
        if past is None:
            st.error("과거 명부를 읽지 못했습니다. " + " / ".join(i["title"] for i in q["issues"] if i["level"] == "error"))
            return
        lv = data.get("leaver")
        prof = C.build_profile(past, C.infer_grades(past), retirement_age=int(cfg["ret_age"]),
                               base_push_turnover=cfg["push"] / 100.0, other_exit_rate=cfg["other"] / 100.0,
                               wage_regime="observed" if cfg["regime"] == "custom" else cfg["regime"],
                               ce_practice=dict(rate=float(cfg["ce_rate"]), end_age=int(cfg["ce_end"]),
                                                wage_ratio=float(cfg["ce_ratio"])),
                               start_year=int(past_year), hazard_offsets=(lv or {}).get("offsets"))
        with st.spinner("과거에서 오늘까지 계산하고 있습니다"):
            st.session_state["backtest"] = B.run_backtest(prof, data["std"], int(data["base_year"]) - int(past_year))
    bt = st.session_state.get("backtest")
    if not bt:
        return
    import backtest as B
    grade, text = B.verdict(bt)
    k = st.columns(4)
    kpi(k[0], "연령 구성 오차 (모형)", f"{bt['mae_model']:.1f}%p", "연령대별 비중이 실제와 어긋난 평균")
    kpi(k[1], "단순 예측의 오차", f"{min(bt['mae_same'], bt['mae_aging']):.1f}%p",
        f"변화 없음 {bt['mae_same']:.1f} · 나이만 먹음 {bt['mae_aging']:.1f}")
    kpi(k[2], "평균 나이", f"{bt['avg_age']['pred']:.1f}세", f"실제 {bt['avg_age']['actual']:.1f}세 · 과거 {bt['avg_age']['past']:.1f}세")
    kpi(k[3], "55세 이상 비중", f"{bt['share55']['pred']*100:.1f}%",
        f"실제 {bt['share55']['actual']*100:.1f}% · 과거 {bt['share55']['past']*100:.1f}%")
    gap(8)
    t = bt["table"]
    fig = go.Figure()
    fig.add_bar(x=t["연령대"], y=t["과거"], name=f"과거({int(data['base_year']) - bt['years']}년)", marker_color="#c9d1d7",
                hovertemplate="%{x}<br>과거 %{y:.1f}%<extra></extra>")
    fig.add_bar(x=t["연령대"], y=t["모형 예측"], name="모형이 예측한 오늘", marker_color="#6da7ec",
                error_y=dict(type="data", symmetric=False, array=t["예측 상한"] - t["모형 예측"],
                             arrayminus=t["모형 예측"] - t["예측 하한"], thickness=1.2, width=3, color="#5D6B77"),
                hovertemplate="%{x}<br>모형 예측 %{y:.1f}%<extra></extra>")
    fig.add_bar(x=t["연령대"], y=t["실제(현재)"], name="실제 오늘", marker_color="#184f95",
                hovertemplate="%{x}<br>실제 %{y:.1f}%<extra></extra>")
    base_layout(fig, 330, unit="전체 인원 중 비중 (%)", barmode="group", bargap=0.3, bargroupgap=0.08,
                legend=dict(orientation="h", y=-0.14, x=0))
    fig.update_traces(marker_cornerradius=3)
    show(fig, "backtest", bar=False)
    tag = {"good": chip("잘 맞음"), "fair": chip("대체로 맞음", "warn"), "poor": chip("잘 안 맞음", "bad")}[grade]
    worst = t.iloc[(t["모형 예측"] - t["실제(현재)"]).abs().idxmax()]
    note(f"{tag}&nbsp; {text}<br>가장 크게 어긋난 연령대는 <b>{worst['연령대']}</b>"
         f"(예측 {worst['모형 예측']:.1f}% · 실제 {worst['실제(현재)']:.1f}%)입니다. "
         f"그 사이 들어와 지금 남아 있는 인원은 예측 {bt['hires']['pred']:,.0f}명 · 실제 {bt['hires']['actual']:,}명입니다.")
    note("검증에는 ② 단계의 퇴사율·재고용 관행 가정을 그대로 씁니다. 가정을 바꾼 뒤 다시 검증하면 "
         "어떤 가정이 우리 회사에 더 잘 맞는지 비교할 수 있습니다. 막대 위 선은 예측의 90% 범위입니다.", "fine")


# ── C. 집계표 ───────────────────────────────────────────────────────────────
def _bands(cfg):
    note("개인별 명부를 올리기 어렵다면, 연령대별 <b>인원·평균 임금·평균 근속</b>만 넣어도 됩니다.<br>"
         "각 연령대 안의 세부 분포는 실제 대기업 근로자 표본(KLIPS)의 모양을 빌려 채웁니다. "
         "부서·직급별 분석은 할 수 없습니다.")
    base = st.session_state.get("bands_table", C.BAND_TEMPLATE)
    edited = st.data_editor(base[["연령대", "인원", "평균 월임금(만원)", "평균 근속(년)"]],
                            key="w_bands", hide_index=True, width="stretch", disabled=["연령대"],
                            column_config={"인원": st.column_config.NumberColumn(min_value=0, step=1),
                                           "평균 월임금(만원)": st.column_config.NumberColumn(min_value=50, step=10),
                                           "평균 근속(년)": st.column_config.NumberColumn(min_value=0.0, step=0.5, format="%.1f")})
    if st.button("이 집계표로 가상 명부 만들기", type="primary"):
        tbl = C.BAND_TEMPLATE.copy()
        for col in ("인원", "평균 월임금(만원)", "평균 근속(년)"):
            tbl[col] = pd.to_numeric(edited[col], errors="coerce").fillna(0).to_numpy()
        if tbl["인원"].sum() < 30:
            st.error("인원 합계가 30명 이상이어야 합니다.")
            return
        st.session_state["bands_table"] = tbl
        std = C.synthesize_from_bands(tbl, retirement_age=int(cfg["ret_age"]))
        ce = C.infer_ce_practice(std, int(cfg["ret_age"]))
        _set_data(mode=SRC_BANDS, std=std, quality=None, grade_table=None, name="집계표 기반 가상 명부", ce=ce)
        cfg["ce_rate"], cfg["ce_end"], cfg["ce_ratio"] = ce["rate"], ce["end_age"], ce["wage_ratio"]


# ── 인력구조 진단 ───────────────────────────────────────────────────────────
def _diagnosis(data, cfg):
    std, ret = data["std"], int(cfg["ret_age"])
    s = C.structure_summary(std, ret)
    section("지금 우리 인력 구조", "시뮬레이션을 돌리기 전에, 명부만 봐도 알 수 있는 사실입니다.")
    c = st.columns(4)
    kpi(c[0], "분석 인원", f"{s['n']:,}명", f"재고용·촉탁 {s['n_ce']:,}명 포함" if s["n_ce"] else "전원 정규직")
    kpi(c[1], "평균 나이", f"{s['avg_age']:.1f}세", f"평균 근속 {s['avg_tenure']:.1f}년")
    kpi(c[2], "55세 이상 비중", f"{s['share_55p']*100:.1f}%", f"300인 이상 사업장 평균 {MOEL_55P_2023*100:.1f}%")
    kpi(c[3], "5년 안에 정년 도달", f"{s['reach_5y']*100:.1f}%", f"정규직 {int(round(s['reach_5y']*s['n_regular'])):,}명")

    diff = s["share_55p"] - MOEL_55P_2023
    if diff > 0.03:
        a = f"55세 이상 비중이 300인 이상 사업장 평균보다 <b>{diff*100:.1f}%p 높습니다.</b> 정년 제도 변화의 영향을 평균보다 크게 받는 구조입니다."
    elif diff < -0.03:
        a = f"55세 이상 비중이 300인 이상 사업장 평균보다 <b>{abs(diff)*100:.1f}%p 낮습니다.</b> 정년 제도 변화의 영향이 평균보다 늦게, 작게 나타납니다."
    else:
        a = "55세 이상 비중이 300인 이상 사업장 평균과 <b>비슷한 수준</b>입니다."
    r5 = s["reach_5y"]
    if r5 >= 0.20:
        b = f"정규직 5명 중 1명 이상이 5년 안에 정년에 도달합니다. <b>인력 공백과 기술 전수 계획이 시급한 수준</b>입니다."
    elif r5 >= 0.10:
        b = "정규직 10명 중 1~2명이 5년 안에 정년에 도달합니다. <b>지금부터 대비하면 충분한 수준</b>입니다."
    else:
        b = "5년 안에 정년에 도달하는 인원이 10% 미만이라 <b>당장의 퇴직 물결은 크지 않습니다.</b>"
    gap(8)
    note(a + "<br>" + b)
    if data["mode"] == SRC_DEMO:
        note("표준 표본은 60세 미만만 사용해 55세 이상 비중이 실제 대기업 평균보다 낮게 시작합니다"
             "(재고용 관행을 '예'로 두면 60세 이상도 포함됩니다).", "fine")

    left, right = st.columns([1, 1.25])
    with left:
        html('<div class="card-t" style="margin-top:18px">연령 분포</div>')
        p = C.age_pyramid(std)
        fig = go.Figure()
        fig.add_bar(y=p["band"], x=p["정규"], name="정규직", orientation="h", marker_color=GROUP_C["mid"],
                    hovertemplate="%{y}<br>정규직 %{x:,}명<extra></extra>")
        if p["재고용·촉탁"].sum() > 0:
            fig.add_bar(y=p["band"], x=p["재고용·촉탁"], name="재고용·촉탁", orientation="h",
                        marker_color=GROUP_C["ce"], hovertemplate="%{y}<br>재고용·촉탁 %{x:,}명<extra></extra>")
        base_layout(fig, 340, barmode="stack", bargap=0.25, xaxis_title="명",
                    legend=dict(orientation="h", y=1.1, x=0),
                    showlegend=bool(p["재고용·촉탁"].sum() > 0))
        fig.update_traces(marker_line=dict(color="white", width=2))
        show(fig, "pyramid", bar=False)
    with right:
        multi = std["dept_name"].nunique() > 1
        by = "dept_name" if multi else None
        html(f'<div class="card-t" style="margin-top:18px">{"부서별 " if multi else ""}정년 도달 달력</div>')
        if multi:
            cal = C.retirement_calendar(std, "dept_name", ret, 10, int(cfg["start_year"]))
        else:
            reg = std[std["status"] == 0].assign(_all="전체")
            cal = C.retirement_calendar(reg, "_all", ret, 10, int(cfg["start_year"]))
        z = cal.to_numpy()
        fig = go.Figure(go.Heatmap(z=z, x=[str(y) for y in cal.columns], y=list(cal.index),
                                   colorscale=SEQ, zmin=0, text=z, texttemplate="%{text}",
                                   textfont=dict(size=11), showscale=False, xgap=2, ygap=2,
                                   hovertemplate="%{y} · %{x}년<br>정년 도달 %{z}명<extra></extra>"))
        base_layout(fig, max(180, 60 + 34 * len(cal)) if multi else 150)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=False, side="top")
        show(fig, "calendar", bar=False)
        note(f"현재 정년({ret}세)이 그대로일 때, 그 해에 정년이 되는 정규직 수입니다(중도 퇴사는 반영하지 않음).", "fine")

    if std["dept_name"].nunique() > 1:
        risk = C.dept_risk(std, ret)
        top = risk.iloc[0]
        avg5 = s["reach_5y"]
        if top["도달5"] > max(avg5 * 1.3, 0.12) and top["인원"] >= 10:
            note(f"가장 먼저 영향을 받는 곳은 <b>{top['dept_name']}</b>입니다. 정규직 {int(top['인원'])}명 중 "
                 f"<b>{top['도달5']*100:.0f}%</b>가 5년 안에 정년에 도달합니다 — 회사 전체({avg5*100:.0f}%)의 "
                 f"<b>{top['도달5']/max(avg5,1e-9):.1f}배</b>입니다. 정년이 늘어나면 이 부서의 인건비와 승진 적체가 "
                 "가장 먼저 움직이고, 늘어나지 않으면 이 부서의 숙련 인력이 가장 먼저 빠집니다.")
        show_t = risk.rename(columns={"dept_name": "부서", "평균나이": "평균 나이", "비중55": "55세 이상",
                                      "도달5": "5년 내 정년", "도달10": "10년 내 정년", "청년35": "35세 미만"})
        for col in ("55세 이상", "5년 내 정년", "10년 내 정년", "35세 미만"):
            show_t[col] = show_t[col] * 100
        pc = lambda lab: st.column_config.ProgressColumn(lab, format="%.0f%%", min_value=0, max_value=100)
        st.dataframe(show_t, width="stretch", hide_index=True,
                     column_config={"평균 나이": st.column_config.NumberColumn(format="%.1f"),
                                    "55세 이상": pc("55세 이상"), "5년 내 정년": pc("5년 내 정년"),
                                    "10년 내 정년": pc("10년 내 정년"), "35세 미만": pc("35세 미만")})

    ce = data.get("ce") or {}
    if ce.get("rate", 0) > 0:
        note(f"<b>현재 재고용 관행(추정)</b> — 정년 도달자의 약 <b>{ce['rate']*100:.0f}%</b>가 "
             f"<b>{ce['end_age']}세</b>까지, 정년 전 임금의 <b>{ce['wage_ratio']*100:.0f}%</b> 수준으로 남아 있는 것으로 보입니다. "
             f"{ce.get('basis','')} 다음 단계에서 고칠 수 있습니다.")
