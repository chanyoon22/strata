"""ui_report.py — ⑥ 리포트: 해석문, 내려받기(엑셀·PDF·설정), 모형 검증, 한계."""
from __future__ import annotations
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

import app_logic as A
import export_tools as E
from dss_layer import HIRING_POLICIES
from engine_retirement_v2 import WAGE_REGIMES
from llm_report import build_llm_input, generate_report
from ui_common import INK, SUB, LINE, PAPER, html, gap, section, note, chip


def _judged_out(res):
    j = st.session_state.get("judge") or res["out"]
    return dict(res["out"], frontier=j["frontier"], dominated=j["dominated"], why=j.get("why", {}),
                stability=j.get("stability", {}))


def render(res, cfg, data, stale):
    if stale:
        st.warning("설정이 바뀌었습니다. 아래 리포트는 이전 설정의 결과로 만들어집니다.")
    lim = A.limitations(cfg, data, res)
    out = _judged_out(res)
    sens = st.session_state.get("sens") or dict(baseline_frontier=out["frontier"], flip_points=[], checked=False)
    robust = st.session_state.get("robust")

    section("해석 리포트", "위 결과를 짧은 글로 풀어 설명합니다.<br>"
            "글에 나오는 숫자가 계산 결과에 실제로 있는 값인지 문장마다 대조하고, 확인되지 않는 문장은 싣지 않습니다.")
    if st.button("해석 리포트 만들기", type="primary"):
        payload = build_llm_input(out, sens, lim, robust)
        with st.spinner("해석 리포트를 만들고 있습니다"):
            st.session_state["report"] = generate_report(payload)
    rep = st.session_state.get("report")
    if rep:
        badge = chip("AI가 방금 작성") if rep["is_live"] else chip("기본 요약 (AI 미연결)", "out")
        html(f'<div style="margin:12px 0 8px">{badge}</div>'
             f'<div style="background:{PAPER};border:1px solid {LINE};border-radius:12px;padding:16px 18px;'
             f'color:{INK};font-size:.93rem;line-height:1.85">'
             f'{rep["prose"] or "표시할 문장이 없습니다 — 만들어진 글의 숫자가 모두 대조에서 탈락했습니다."}</div>')
        if rep["dropped"]:
            note(f"근거가 확인되지 않은 문장 {len(rep['dropped'])}개는 싣지 않았습니다.", "fine")
        if not st.session_state.get("sens") and not robust:
            note("⑤ 단계의 분석을 먼저 돌리면 '결론이 얼마나 단단한지'까지 리포트에 담깁니다.", "fine")

    section("내려받기", "회사 안에서 보고하거나 다시 가공할 수 있도록 결과를 파일로 가져갑니다.")
    c = st.columns(3)
    xlsx = E.to_excel(res, cfg, st.session_state.get("judge"), (data or {}).get("quality"), lim)
    c[0].download_button("엑셀로 받기 (요약·연도별·가정)", xlsx, "STRATA_결과.xlsx",
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    if c[1].button("PDF 리포트 만들기", width="stretch"):
        try:
            st.session_state["pdf"] = _make_pdf(res, cfg, data, out, sens, robust, rep, lim)
        except Exception as e:
            st.error(f"PDF를 만들지 못했습니다: {e}")
    if st.session_state.get("pdf"):
        c[1].download_button("PDF 내려받기", st.session_state["pdf"], "STRATA_리포트.pdf", "application/pdf",
                             width="stretch")
    c[2].download_button("지금 설정 저장 (.json)", A.export_settings(cfg), "STRATA_설정.json", "application/json",
                         width="stretch", help="가정·시나리오·판정 기준을 파일로 저장합니다. 명부 데이터는 들어가지 않습니다.")
    up = st.file_uploader("저장해 둔 설정 불러오기", type=["json"], key="w_cfg_up")
    if up is not None and st.session_state.get("cfg_loaded") != up.name:
        try:
            A.import_settings(cfg, up.getvalue())
            st.session_state["cfg_loaded"] = up.name
            for k in [k for k in st.session_state if k.startswith("w_") and k != "w_cfg_up"]:
                del st.session_state[k]
            st.success("설정을 불러왔습니다. ② 단계에서 확인하고 다시 실행해 주세요.")
        except Exception as e:
            st.error(f"설정 파일을 읽지 못했습니다: {e}")

    section("이 결과를 읽을 때 알아야 할 한계")
    html("<ul style='margin:0 0 0 18px;padding:0'>" +
         "".join(f"<li class='note' style='margin-bottom:6px'>{x}</li>" for x in lim) + "</ul>")

    section("모형 점검", "계산 엔진이 스스로를 검사하는 항목들입니다. 통과 여부를 포장하지 않고 종류별로 나눠 보여줍니다.")
    if st.button("점검 실행 (약 10초)"):
        import validation_v2 as V
        with st.spinner("점검하고 있습니다"):
            rs = [fn() for fn in V.ALL_TESTS]
            st.session_state["val"] = (rs, V.summary(rs))
    if st.session_state.get("val"):
        rs, sm = st.session_state["val"]
        kind = {"computed": "지금 다시 계산", "documented": "문서 근거 인용", "diagnostic": "가정 진단"}
        rows = []
        for r in rs:
            if r["method"] == "diagnostic":
                state = "가정 성립" if r.get("parallel_trends_ok") else "가정 미성립(공개)"
            else:
                state = "통과" if r["passed"] else "실패"
            rows.append({"항목": r["test"], "종류": kind[r["method"]], "결과": state,
                         "내용": r.get("note") or r.get("interpretation") or r.get("detail") or ""})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        note(sm["label"], "fine")


def _make_pdf(res, cfg, data, out, sens, robust, rep, lim):
    from report_generator import build_pdf
    p = res["profile"]
    regime = "자사 입력 승급률" if p.wage_regime == "custom" else WAGE_REGIMES[p.wage_regime]["label"]
    desc = (f"{p.name} · {out['results'][0]['summary']['n_initial']:,}명 · 채용 방식 {HIRING_POLICIES[cfg['hiring']][0]} · "
            f"50세 이후 임금 {regime}")
    prose = (rep or {}).get("prose")
    if not prose:
        prose = generate_report(build_llm_input(out, sens, lim, robust))["prose"]
    meta = ["데이터: 회사 명부(올린 경우) + 한국노동패널(KLIPS) 26·27차 이직 모형 + 사업체패널(WPS) 2023 + "
            "고용형태별근로실태조사 2025 + 고용노동부 고령자 고용현황",
            "후보 판정은 결정론적 알고리즘(대응표본 통계적 지배 필터)이 수행하고, AI는 해석 문장만 씁니다(숫자 대조 포함)."] + lim
    with tempfile.TemporaryDirectory() as td:
        path = build_pdf(out, sens, prose, desc, meta, out_path=str(Path(td) / "report.pdf"), outdir=td,
                         robustness_output=robust, profile_job_default=p.job_table_is_default())
        return Path(path).read_bytes()
