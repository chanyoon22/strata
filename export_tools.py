"""export_tools.py — 결과를 엑셀 파일로 내보내기(회사 내부 보고·재가공용)."""
from __future__ import annotations
import io
import numpy as np
import pandas as pd

import interpret as I
from dss_layer import HIRING_POLICIES, schedule_text
from engine_retirement_v2 import WAGE_REGIMES, ENGINE_VERSION

YEAR_COLS = [("year", "연도"), ("headcount", "인원"), ("retirement_age", "정년"), ("ce_end_age", "재고용 상한"),
             ("labor_cost", "연간 인건비(만원)"), ("cost_reg", "기존 정년 이하 정규(만원)"),
             ("cost_ext", "연장 재직(만원)"), ("cost_ce", "재고용(만원)"), ("cost_erp", "희망퇴직 위로금(만원)"),
             ("peak_saving", "임금피크 절감(만원)"), ("n_hire", "신규채용"), ("n_promoted", "승진"),
             ("promo_wait_p90", "승진 대기(년)"), ("avg_age", "평균 나이"), ("share_55plus", "55세 이상 비중"),
             ("n_u35", "35세 미만"), ("n_35_54", "35~54세"), ("n_55_59", "55~59세"), ("n_60p", "60세 이상"),
             ("n_ce", "재고용 인원"), ("n_push", "자발적 퇴사"), ("n_other", "기타 퇴사"), ("n_erp", "희망퇴직"),
             ("n_retire", "정년 도달"), ("n_rehired", "재고용 전환"), ("n_ce_exit", "재고용 종료"),
             ("income_gap", "소득 공백(인년)"), ("skill_gap", "숙련 공백(인년)"),
             ("skill_value", "숙련 손실(평균 인건비 100% 환산, 만원)")]


def summary_table(res, judge=None):
    out, base = res["out"], res["baseline"]
    j = judge or out
    d = I.deltas(out["results"], base)
    rows = []
    for r in out["results"]:
        s = r["summary"]
        n = s["scenario"]
        rows.append({"시나리오": n, "판정": "후보" if n in j["frontier"] else "제외",
                     "인건비 현재가치(억)": round(s["cost_npv_mean"] / 1e4, 1),
                     "인건비 하위 5%(억)": round(s["cost_npv_p5"] / 1e4, 1),
                     "인건비 상위 5%(억)": round(s["cost_npv_p95"] / 1e4, 1),
                     "인건비 현행 대비(%)": round(d[n]["cost"] * 100, 2),
                     "신규채용 누계(명)": round(s["total_hires_mean"], 1),
                     "신규채용 현행 대비(%)": round(d[n]["hires"] * 100, 1),
                     "승진 대기(년)": round(s["promo_congestion_mean"], 2),
                     "55세 이상 재직(인년)": round(s["exp55_person_years_mean"], 0),
                     "퇴직자 소득 공백(인년)": round(s["income_gap_mean"], 0),
                     "숙련 공백(인년)": round(s.get("skill_gap_mean", float("nan")), 0),
                     "숙련 손실 현재가치(억, 100% 환산)": round(s.get("skill_value_npv_mean", float("nan")) / 1e4, 1),
                     "판정 안정성(%)": round(j.get("stability", {}).get(n, float("nan")) * 100, 0),
                     "제외 사유": I.why_text(n, j.get("why", {}))})
    return pd.DataFrame(rows)


def assumptions_table(res, cfg):
    p = res["profile"]
    rows = [("분석 대상", p.name), ("분석 인원(명)", res["out"]["results"][0]["summary"]["n_initial"]),
            ("시작 연도", p.start_year), ("분석 기간(년)", res["horizon"]), ("반복 횟수", res["runs"]),
            ("현재 정년(세)", p.current_retirement_age),
            ("자발적 퇴사율(%/년)", round(p.base_push_turnover * 100, 2)),
            ("기타 퇴사율(%/년)", round(p.other_exit_rate * 100, 2)),
            ("재고용 수락률", cfg["accept"]), ("재고용 임금률", cfg["wage_ratio"]),
            ("채용 방식", HIRING_POLICIES[cfg["hiring"]][0]),
            ("50세 이후 임금", "자사 입력" if p.wage_regime == "custom" else WAGE_REGIMES[p.wage_regime]["label"]),
            ("현재 재고용 비율", p.ce_rate_now), ("현재 재고용 상한(세)", p.ce_end_age_now),
            ("신입 초임(월 만원)", None if p.entry_wage is None else round(p.entry_wage, 0)),
            ("직급 체계", " < ".join(p.grades.names)),
            ("자사 퇴사 패턴 보정", "반영" if p.hazard_offsets else "미반영"),
            ("직군별 충원·숙련 기간", "기본값(모든 직군 충원 3개월·숙련 2년)" if p.job_table_is_default() else "회사 입력값"),
            ("할인율", "연 3%"), ("간접 인건비 배수", "1.19 (4대보험·퇴직급여)"),
            ("판정 기준", f"대응표본 95% 신뢰구간 + 실질 차이 {res['out']['materiality']*100:.1f}% 이상"),
            ("엔진 버전", ENGINE_VERSION)]
    return pd.DataFrame(rows, columns=["항목", "값"])


def scenario_table(res):
    rows = []
    for s in res["scenarios"]:
        rows.append({"시나리오": s.name, "정년 일정": schedule_text(s.retirement_schedule),
                     "재고용": {"none": "없음", "rehire": "재고용", "job_keep": "직무유지형", "flexible": "자율"}[s.continued_type],
                     "재고용 상한 일정": schedule_text(s.ce_end_age_schedule) if s.ce_end_age_schedule else "",
                     "수락률": s.rehire_accept if s.continued_type != "none" else None,
                     "임금률": s.rehire_wage_ratio if s.continued_type != "none" else None,
                     "임금피크": f"{s.peak_years}년·연 {s.peak_cut*100:.0f}%p" if s.wage_peak else "",
                     "임금체계 개편": f"{s.wage_reform_year}년~ {WAGE_REGIMES[s.wage_reform_regime]['label']}" if s.wage_reform_regime else "",
                     "희망퇴직": (f"{s.erp_start}~{s.erp_end}년 {s.erp_age_min}~{s.erp_age_max}세 "
                              f"연 {s.erp_take_rate*100:.0f}%·{s.erp_severance_months:.0f}개월분") if s.erp_take_rate > 0 else "",
                     "출처": s.source, "설명": s.desc})
    return pd.DataFrame(rows)


def job_table(res):
    """직군별 설정과 손익분기(엑셀 시트용)."""
    import decision as D
    p = res["profile"]
    a = p.job_arrays()
    jt = pd.DataFrame({"직군": p.dept_names, "충원 기간(개월)": a["vacancy_months"],
                       "숙련 도달 기간(년)": a["ttp_years"], "재고용 비율(평균=1)": a["ce_accept_rel"],
                       "재고용 임금(평균=1)": a["ce_wage_rel"],
                       "자발적 퇴사 배수(명부)": [round(float(x), 2) for x in np.exp(a["turnover_offset"])]})
    R = {r["summary"]["scenario"]: r for r in res["out"]["results"]}
    b = R[res["baseline"]]
    rows = []
    for n, r in R.items():
        if n == res["baseline"]:
            continue
        be = D.break_even(r, b)
        rows.append({"시나리오": n, "인건비 차이(억)": round(be["dc"] / 1e4, 1),
                     "숙련 손실 차이(억, 100% 환산)": round(be["dv"] / 1e4, 1),
                     "판정": {"always": "숙련 가치와 무관하게 유리", "never": "숙련 가치와 무관하게 불리",
                            "above": "숙련 가치가 손익분기 이상이면 유리", "below": "숙련 가치가 손익분기 이하일 때만 유리"}[be["kind"]],
                     "손익분기(%)": None if be["x"] is None else round(be["x"] * 100, 0),
                     "95% 구간 하한(%)": None if be["lo"] is None else round(be["lo"] * 100, 0),
                     "95% 구간 상한(%)": None if be["hi"] is None else round(be["hi"] * 100, 0)})
    return jt, pd.DataFrame(rows)


def yearly_table(res):
    frames = []
    for r in res["out"]["results"]:
        df = pd.DataFrame([{lab: row.get(k) for k, lab in YEAR_COLS} for row in r["path"]])
        df.insert(0, "시나리오", r["summary"]["scenario"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True).round(2)


def to_excel(res, cfg, judge=None, quality=None, limitations=None) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summary_table(res, judge).to_excel(xw, "요약", index=False)
        scenario_table(res).to_excel(xw, "시나리오 정의", index=False)
        yearly_table(res).to_excel(xw, "연도별", index=False)
        assumptions_table(res, cfg).to_excel(xw, "가정", index=False)
        jt, bt = job_table(res)
        jt.to_excel(xw, "직군 설정", index=False)
        if len(bt):
            bt.to_excel(xw, "숙련 손익분기", index=False)
        if quality and quality.get("issues"):
            pd.DataFrame([{"구분": {"error": "오류", "warn": "확인 필요", "info": "자동 처리"}[i["level"]],
                           "항목": i["title"], "내용": i["detail"], "해당 행": i["n"]}
                          for i in quality["issues"]]).to_excel(xw, "데이터 점검", index=False)
        if limitations:
            pd.DataFrame({"이 분석의 한계": limitations}).to_excel(xw, "한계", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width * 1.6), 60)
            ws.freeze_panes = "B2"
    return buf.getvalue()
