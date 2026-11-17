"""ui_setup.py — ② 가정·시나리오: 회사 조건, 비교할 안 고르기, 직접 만들기, 실행."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
import streamlit as st

import app_logic as A
from dss_layer import preset_scenarios, HIRING_POLICIES, schedule_text, scenario_from_dict
from engine_retirement_v2 import WAGE_REGIMES
from ui_common import INK, SUB, html, gap, section, note, chip, short

REGIME_LABELS = {**{k: v["label"] for k, v in WAGE_REGIMES.items()}, "custom": "우리 회사 승급률 직접 입력"}
REGIME_HELP = {
    "observed": "나이대별 평균 임금 통계의 모양을 그대로 따릅니다. 50대 후반부터 임금이 조금씩 내려가는 모양이라, "
                "호봉제 회사에는 맞지 않고 정년연장 비용을 낮게 볼 수 있습니다.",
    "frozen": "50세부터는 임금이 오르지도 내리지도 않는다고 봅니다. 직무급·성과급 중심 회사에 가깝습니다.",
    "seniority": "나이가 들수록 임금이 계속 오릅니다(50대 연 2%, 60대 연 1%). 호봉제 회사에 가깝고, "
                 "정년연장 비용이 가장 크게 나옵니다.",
    "custom": "아래 표에 나이대별 연간 임금 인상률을 직접 넣습니다.",
}


def _w(cfg, key, widget, *args, **kw):
    """위젯 값을 cfg에 되돌려 저장(단계를 오가도 값이 유지되도록)."""
    v = widget(*args, key=f"w_{key}", **kw)
    cfg[key] = v
    return v


JOB_COLS = {"vacancy_months": "충원 기간(개월)", "ttp_years": "숙련 도달 기간(년)",
            "ce_accept_rel": "재고용 비율(평균=1)", "ce_wage_rel": "재고용 임금(평균=1)"}


def _jobs(cfg, data):
    """직군별 특성 표. 산업마다 다른 값을 STRATA가 정하지 않고, 회사가 자기 현실을 넣는다."""
    from engine_retirement_v2 import JOB_DEFAULTS
    section("직군별 특성 — 숙련은 직군마다 다릅니다",
            "사람이 나가면 그 자리가 다시 제 몫을 할 때까지 손실이 생깁니다. 이 손실(<b>숙련 공백</b>)은 직군마다, "
            "같은 직군이라도 산업마다 크게 다릅니다.<br>"
            "STRATA는 산업별 기본값을 정해두지 않습니다. 근거 있는 공개 통계가 없기 때문입니다. "
            "대신 현장 관리자가 이미 알고 있는 두 가지를 물어 넣습니다.<br>"
            "<b>① 사람이 나가면 충원까지 몇 달 걸리나요?</b> &nbsp; "
            "<b>② 새로 온 사람에게 혼자 일을 맡기려면 몇 년 걸리나요?</b>")
    names, counts = A.dept_names_for(cfg, data)
    jobs = cfg.get("jobs") or {}
    lv = (data or {}).get("leaver") or {}
    offs = lv.get("dept_offsets") or {}
    rows = []
    for d, n in zip(names, counts):
        u = jobs.get(d, {})
        row = {"직군": d, "정규직(명)": n}
        for k, lab in JOB_COLS.items():
            row[lab] = float(u.get(k, JOB_DEFAULTS[k]))
        if offs:
            row["퇴사 배수(명부)"] = round(float(np.exp(offs.get(d, 0.0))), 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    key = "w_jobs_" + hashlib.md5("|".join(names).encode()).hexdigest()[:8]
    cc = st.column_config
    ed = st.data_editor(
        df, key=key, hide_index=True, width="stretch",
        disabled=["직군", "정규직(명)", "퇴사 배수(명부)"],
        column_config={
            "충원 기간(개월)": cc.NumberColumn(min_value=0.0, max_value=36.0, step=0.5, format="%.1f",
                                          help="사람이 나간 뒤 후임이 자리에 앉기까지의 기간. 그 동안의 업무 손실을 셉니다."),
            "숙련 도달 기간(년)": cc.NumberColumn(min_value=0.0, max_value=20.0, step=0.5, format="%.1f",
                                            help="후임이 혼자 제 몫을 하기까지의 기간. 이 기간 동안 생산성이 0에서 100%까지 "
                                                 "오른다고 봅니다. 떠난 사람의 근속이 이보다 짧으면 그 근속만큼만 셉니다."),
            "재고용 비율(평균=1)": cc.NumberColumn(min_value=0.0, max_value=3.0, step=0.1, format="%.1f",
                                             help="이 직군에서 정년 도달자가 재고용되는 정도를 회사 평균 대비로 넣습니다. "
                                                  "예: 1.3 = 평균보다 30% 더 많이 남음(상한 100%)."),
            "재고용 임금(평균=1)": cc.NumberColumn(min_value=0.3, max_value=2.0, step=0.05, format="%.2f",
                                             help="재고용 임금 비율을 회사 평균 대비로 넣습니다. 정년 전 임금을 넘지는 않습니다."),
            "퇴사 배수(명부)": cc.NumberColumn(format="%.2f배",
                                          help="퇴사자 명부로 추정한 이 직군의 자발적 퇴사 수준(회사 평균 = 1). 자동 반영됩니다."),
        })
    new = {}
    for _, r in ed.iterrows():
        new[r["직군"]] = {k: float(pd.to_numeric(r[lab], errors="coerce")) if pd.notna(pd.to_numeric(r[lab], errors="coerce"))
                         else JOB_DEFAULTS[k] for k, lab in JOB_COLS.items()}
    cfg["jobs"] = new
    if A._jobs_entered(cfg):
        note("입력하신 직군별 값으로 숙련 공백을 계산합니다. ③ 결과의 <b>숙련 손실까지 보면</b>에서 "
             "이 값이 결론을 어떻게 바꾸는지 볼 수 있습니다.", "fine")
    else:
        note(f"지금은 모든 직군에 같은 기본값(충원 {JOB_DEFAULTS['vacancy_months']:.0f}개월 · "
             f"숙련 {JOB_DEFAULTS['ttp_years']:.0f}년)이 들어가 있습니다. 근거 있는 값이 아니라 자리표시입니다.<br>"
             "그대로 두면 숙련 공백은 <b>'몇 명이, 얼마나 오래 일한 사람이 나가는가'</b>의 차이만 담습니다. "
             "직군 간 차이를 보려면 위 표를 고쳐 주세요.", "fine")


def render(cfg, data):
    section("회사 조건",
            "결과를 좌우하는 가정들입니다. 모르는 값은 그대로 두어도 되지만, "
            "<b>퇴사율과 채용 방식</b>은 우리 회사 실제에 맞춰야 결과를 믿을 수 있습니다.")
    lv = (data or {}).get("leaver")
    r1 = st.columns(4)
    _w(cfg, "push", r1[0].number_input, "자발적 퇴사율 (%/년)", 0.1, 15.0, float(cfg["push"]), 0.1,
       help="직원이 스스로 그만두는 비율입니다(정년퇴직 제외). 대기업 평균은 약 1.5%입니다."
            + (" 퇴사자 명부에서 계산한 값이 들어가 있습니다." if lv else ""))
    _w(cfg, "other", r1[1].number_input, "기타 퇴사율 (%/년)", 0.0, 15.0, float(cfg["other"]), 0.5,
       help="권고사직·계약 종료·건강 등 그 밖의 사유로 떠나는 비율입니다.")
    _w(cfg, "horizon", r1[2].selectbox, "분석 기간", [10, 15, 20],
       index=[10, 15, 20].index(cfg["horizon"]) if cfg["horizon"] in (10, 15, 20) else 1,
       format_func=lambda x: f"{x}년 ({cfg['start_year']}~{cfg['start_year']+x-1})",
       help="정년이 65세에 도달하는 해(안에 따라 2033~2041년)까지 담기도록 15년 이상을 권합니다.")
    _w(cfg, "runs", r1[3].selectbox, "계산 정밀도", list(A.RUN_LEVELS),
       index=list(A.RUN_LEVELS).index(cfg["runs"]) if cfg["runs"] in A.RUN_LEVELS else 1,
       format_func=lambda x: f"{A.RUN_LEVELS[x]} ({x}회 반복)",
       help="같은 조건을 여러 번 계산해 평균과 오차 범위를 구합니다. 많을수록 정확하지만 오래 걸립니다.")

    r2 = st.columns([1, 1, 2])
    hiring = _w(cfg, "hiring", r2[0].selectbox, "정년연장 부담을 흡수하는 방식", list(HIRING_POLICIES),
                index=list(HIRING_POLICIES).index(cfg["hiring"]), format_func=lambda k: HIRING_POLICIES[k][0],
                help="정년이 늘어 고령자가 남을 때 무엇을 조정할지입니다. 이 선택에 따라 후보가 실제로 달라집니다.")
    regime = _w(cfg, "regime", r2[1].selectbox, "50세 이후 임금이 어떻게 변할지", list(REGIME_LABELS),
                index=list(REGIME_LABELS).index(cfg["regime"]), format_func=lambda k: REGIME_LABELS[k],
                help="정년연장 비용의 크기를 좌우하는 가정입니다.")
    r2[2].markdown(f'<div class="note" style="padding-top:26px"><b>{HIRING_POLICIES[hiring][0]}</b> — '
                   f'{HIRING_POLICIES[hiring][1]}<br><b>{REGIME_LABELS[regime]}</b> — {REGIME_HELP[regime]}</div>',
                   unsafe_allow_html=True)
    if regime == "custom":
        df = pd.DataFrame({"나이대": A.CUSTOM_BAND_LABELS, "연간 임금 인상률(%)": cfg["custom_rates"]})
        ed = st.data_editor(df, key="w_custom_rates", hide_index=True, disabled=["나이대"], width=420,
                            column_config={"연간 임금 인상률(%)": st.column_config.NumberColumn(
                                min_value=-10.0, max_value=15.0, step=0.1, format="%.1f")})
        cfg["custom_rates"] = [float(x) for x in pd.to_numeric(ed["연간 임금 인상률(%)"], errors="coerce").fillna(0)]
        note("물가 상승분(베이스업)을 뺀 실질 인상률로 넣어주세요. 모든 금액은 지금 물가 기준입니다.", "fine")

    r3 = st.columns(4)
    _w(cfg, "accept", r3[0].number_input, "재고용 수락률", 0.05, 0.95, float(cfg["accept"]), 0.02,
       help="정년퇴직자 가운데 재고용으로 남는 비율입니다. 기본값 0.62는 대기업 정년퇴직자의 재취업률(KLIPS)입니다.")
    _w(cfg, "wage_ratio", r3[1].number_input, "재고용 임금률", 0.30, 1.00, float(cfg["wage_ratio"]), 0.01,
       help="재고용됐을 때 임금이 정년 전의 몇 %인지입니다. 예: 0.55 = 정년 전의 55%.")
    if data and data["mode"] != A.SRC_DEMO:
        _w(cfg, "ce_rate", r3[2].number_input, "지금 재고용하는 비율", 0.0, 0.95, float(cfg["ce_rate"]), 0.05,
           help="현재 정년퇴직자 중 실제로 다시 고용하는 비율입니다. 명부에서 추정한 값이 들어가 있습니다. "
                "0이면 '지금은 재고용 없음'입니다.")
        _w(cfg, "ce_end", r3[3].number_input, "지금 재고용 상한 나이", int(cfg["ret_age"]), 75,
           max(int(cfg["ce_end"]), int(cfg["ret_age"])), 1,
           help="현재 재고용자가 몇 세까지 일하는지입니다.")
        if cfg["ce_rate"] > 0:
            note(f"현행 유지(S1)에도 지금의 재고용 관행(정년 도달자의 {cfg['ce_rate']*100:.0f}%를 "
                 f"{cfg['ce_end']}세까지)이 그대로 이어지는 것으로 계산합니다. 법이 더 높은 수준을 요구하는 "
                 "시나리오에서는 둘 중 높은 쪽을 적용합니다.", "fine")

    if data:
        try:
            pool = A.make_profile(cfg, data).hire_pool
        except Exception:
            pool = None
        if pool is not None:
            note(f"<b>새로 뽑는 사람</b>은 지금 명부의 최근 입사자(근속 1년 이하 {len(pool):,}명)와 같은 모습이라고 봅니다 — "
                 f"평균 {pool['age'].mean():.0f}세, 월 {pool['wage'].mean():,.0f}만원. 신입만 뽑는 회사가 아니라면 이 방식이 실제에 가깝습니다.", "fine")
        else:
            note("최근 입사자가 20명 미만이라, 새로 뽑는 사람은 모두 신입(평균 27세 안팎·초임)이라고 가정합니다.", "fine")

    _jobs(cfg, data)

    # ── 시나리오 ──
    section("비교할 시나리오",
            "정년연장 법안은 2026년 11월 현재 <b>아직 국회를 통과하지 않았습니다.</b> "
            "어느 안이 될지 모르는 지금, 공개된 안들을 미리 나란히 놓고 비교합니다.<br>"
            "S1(현행 유지)은 비교 기준이라 항상 포함됩니다.")
    presets = preset_scenarios()
    picked = []
    for s in presets:
        code = short(s.name)
        c = st.columns([0.06, 0.94])
        on = c[0].checkbox(code, value=(code in cfg["presets"] or code == "S1"), key=f"w_pre_{code}",
                           disabled=(code == "S1"), label_visibility="collapsed")
        if on or code == "S1":
            picked.append(code)
        ce = ""
        if s.continued_type != "none":
            kind = "직무유지형 재고용" if s.continued_type == "job_keep" else "재고용"
            ce = f" · {kind} {schedule_text(s.ce_end_age_schedule).split(' → ')[-1].replace('년 ', '년부터 ')}까지"
        c[1].markdown(f'<div style="line-height:1.65"><b style="color:{INK}">{s.name}</b>'
                      f'<span class="fine"> &nbsp;{s.source}</span><br>'
                      f'<span class="note">{s.desc}</span></div>', unsafe_allow_html=True)
    cfg["presets"] = picked

    _builder(cfg)

    scens = A.make_scenarios(cfg)
    end, late = A.horizon_warnings(scens, int(cfg["start_year"]), int(cfg["horizon"]))
    if late:
        names = ", ".join(f"{short(n)}({y}년 완료)" for n, y in late)
        st.warning(f"분석 기간이 {end}년에 끝나는데, {names}는 그 뒤에 일정이 끝납니다. "
                   "이 시나리오들은 효과가 일부만 반영되므로 분석 기간을 늘리는 것을 권합니다.")
    n_emp = len(data["std"]) if data and data["mode"] != A.SRC_DEMO else int(cfg["demo_n"])
    est = len(scens) * cfg["runs"] * cfg["horizon"] * max(n_emp, 300) / 1000 * 0.00075
    gap(6)
    c = st.columns([1.2, 3])
    go = c[0].button(f"시나리오 {len(scens)}개 분석 실행", type="primary", width="stretch", disabled=not data)
    c[1].markdown(f'<div class="fine" style="padding-top:10px">예상 소요 약 {max(int(est), 3)}초 · '
                  f'{n_emp:,}명 × {len(scens)}개 시나리오 × {cfg["runs"]}회 × {cfg["horizon"]}년</div>',
                  unsafe_allow_html=True)
    if go:
        bar = st.progress(0.0, text="시뮬레이션을 시작합니다")
        try:
            res = A.run_analysis(cfg, data, progress=lambda f, name: bar.progress(min(f, 1.0), text=f"{name} 계산 완료"))
        except Exception as e:
            bar.empty()
            st.error(f"실행 중 오류가 났습니다: {e}")
            return
        bar.empty()
        st.session_state["res"] = res
        for k in ("sens", "robust", "tornado", "space", "report"):
            st.session_state.pop(k, None)
        st.session_state["goto"] = "③ 결과"
        st.rerun()


def _builder(cfg):
    section("시나리오 직접 만들기",
            "공개된 안에 없는 조합을 직접 설계할 수 있습니다. 예: '2028년부터 2년마다 올려 63세까지 + 임금피크 + 희망퇴직'.<br>"
            "어떤 조합이 좋을지 감이 없다면, 분석 후 <b>⑤ 결론이 흔들리는가 → 정책 공간 탐색</b>에서 100여 개 설계를 한꺼번에 훑어볼 수 있습니다.")
    sy, ra = int(cfg["start_year"]), int(cfg["ret_age"])
    with st.form("builder", clear_on_submit=False):
        a = st.columns([2, 1, 1, 1])
        name = a[0].text_input("시나리오 이름", placeholder="예: 63세까지 + 임금피크")
        target = a[1].number_input("정년을 몇 세까지", ra, 70, max(ra, 63), help=f"{ra}로 두면 정년은 그대로입니다.")
        first = a[2].number_input("첫 인상 연도", sy, sy + 15, sy + 2)
        step = a[3].number_input("몇 년마다 +1세", 1, 5, 2)

        st.markdown(f'<div class="card-t" style="margin-top:6px">정년 후 재고용</div>', unsafe_allow_html=True)
        b = st.columns([1.3, 1.2, 0.8, 1, 1.4])
        ce_type = b[0].selectbox("방식", ["none", "rehire", "job_keep"],
                                 format_func=lambda k: {"none": "없음", "rehire": "재고용 (임금 조정·보직 반납)",
                                                        "job_keep": "직무유지형 (임금 85% 이상·보직 유지)"}[k])
        ce_mode = b[1].selectbox("상한 나이 정하는 법", ["plus", "fixed"],
                                 format_func=lambda k: {"plus": "정년보다 N세 위까지", "fixed": "N세까지 (지정 연도부터)"}[k])
        ce_n = b[2].number_input("N", 1, 75, 1, help="'정년보다 N세 위'면 1~5, 'N세까지'면 61~70 사이로 넣습니다.")
        ce_year = b[3].number_input("시작 연도", sy, sy + 15, sy + 2, help="'N세까지'를 고른 경우에만 쓰입니다.")
        own = b[4].checkbox("수락률·임금률을 따로 지정", help="끄면 위쪽 '회사 조건'의 공통 값을 씁니다.")
        b2 = st.columns(4)
        acc = b2[0].number_input("이 시나리오의 수락률", 0.05, 0.95, float(cfg["accept"]), 0.02)
        rat = b2[1].number_input("이 시나리오의 임금률", 0.30, 1.00, float(cfg["wage_ratio"]), 0.01)

        st.markdown(f'<div class="card-t" style="margin-top:6px">회사가 함께 쓸 수 있는 수단</div>', unsafe_allow_html=True)
        c = st.columns([1.2, 1, 1, 1.2, 1.3, 1])
        peak = c[0].checkbox("임금피크제", help="정년 전 몇 년 동안 임금을 단계적으로 줄입니다.")
        pk_y = c[1].number_input("적용 기간(년)", 1, 5, 3)
        pk_c = c[2].number_input("해마다 감액(%p)", 1, 30, 10)
        reform = c[3].checkbox("임금체계 개편", help="지정한 해부터 50세 이후 임금 궤적을 바꿉니다.")
        rf_r = c[4].selectbox("개편 후 궤적", ["frozen", "observed", "seniority"], format_func=lambda k: WAGE_REGIMES[k]["label"])
        rf_y = c[5].number_input("개편 연도", sy, sy + 15, sy + 2)
        d = st.columns([1.2, 1, 1, 1, 1, 1, 1])
        erp = d[0].checkbox("희망퇴직", help="정해진 기간 동안 대상 연령 직원에게 위로금을 주고 퇴직 신청을 받습니다.")
        e_s = d[1].number_input("시행 시작", sy, sy + 15, sy + 1)
        e_e = d[2].number_input("시행 종료", sy, sy + 15, sy + 2)
        e_lo = d[3].number_input("대상 나이(부터)", 40, 64, 50)
        e_hi = d[4].number_input("대상 나이(까지)", 40, 64, 57)
        e_t = d[5].number_input("연 수락률(%)", 1, 50, 8)
        e_m = d[6].number_input("위로금(개월분)", 0, 60, 24)
        edu = st.number_input("재교육 투자 (1인당 만원/년)", 0, 500, 0, 5,
                              help="비용만 반영합니다. 교육이 생산성·이직에 주는 효과는 검증된 값이 없어 넣지 않습니다.")
        add = st.form_submit_button("이 시나리오 추가")
    if add:
        try:
            if erp and e_e < e_s:
                raise ValueError("희망퇴직 종료 연도가 시작 연도보다 빠릅니다.")
            if erp and e_hi < e_lo:
                raise ValueError("희망퇴직 대상 나이의 범위가 뒤집혀 있습니다.")
            if ce_type != "none" and ce_mode == "plus" and not (1 <= ce_n <= 10):
                raise ValueError("'정년보다 N세 위'의 N은 1~10 사이로 넣어주세요.")
            if ce_type == "none" and target <= ra and not (peak or reform or erp or edu):
                raise ValueError("정년도 재고용도 다른 수단도 바꾸지 않아 현행 유지(S1)와 같습니다.")
            n = 1 + max([int(x["name"].split(" ")[0][1:]) for x in cfg["custom"]] or [0])
            spec = A.custom_spec(n, name, target, first, step, ce_type, ce_mode, ce_n, ce_year, own, acc, rat,
                                 peak, pk_y, pk_c / 100.0, reform, rf_r, rf_y, erp, e_s, e_e, e_lo, e_hi,
                                 e_t / 100.0, e_m, edu, start_year=sy, base_age=ra)
            cfg["custom"].append(spec)
            st.rerun()
        except ValueError as e:
            st.error(str(e))

    for i, spec in enumerate(list(cfg["custom"])):
        s = scenario_from_dict(spec)
        bits = [schedule_text(s.retirement_schedule)]
        if s.continued_type != "none":
            bits.append(("직무유지형 " if s.continued_type == "job_keep" else "") + "재고용 "
                        + schedule_text(s.ce_end_age_schedule).split(" → ")[-1] + "까지")
        if s.wage_peak:
            bits.append(f"임금피크 {s.peak_years}년·연 {s.peak_cut*100:.0f}%p")
        if s.wage_reform_regime:
            bits.append(f"{s.wage_reform_year}년 임금체계 개편({WAGE_REGIMES[s.wage_reform_regime]['label']})")
        if s.erp_take_rate > 0:
            bits.append(f"희망퇴직 {s.erp_start}~{s.erp_end}년 {s.erp_age_min}~{s.erp_age_max}세")
        if s.edu_invest_per_head:
            bits.append(f"재교육 1인당 {s.edu_invest_per_head:.0f}만원")
        c = st.columns([0.9, 0.1])
        c[0].markdown(f'{chip("직접 만든 안")}&nbsp; <b style="color:{INK}">{s.name}</b><br>'
                      f'<span class="note">{" · ".join(bits)}</span>', unsafe_allow_html=True)
        if c[1].button("삭제", key=f"del_{i}"):
            cfg["custom"].pop(i)
            st.rerun()
