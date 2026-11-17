# app.py — STRATA · 인력구조 시나리오 플래너
# 실행: streamlit run app.py   (같은 폴더의 .streamlit/config.toml 필요)
#
# 화면은 여섯 단계로 나뉜다. 각 단계의 내용은 ui_*.py에 있고, 이 파일은 단계 이동과
# 공통 상태만 맡는다.
#   ① 회사 데이터(ui_data)  ② 가정·시나리오(ui_setup)  ③ 결과(ui_results)
#   ④ 우리 회사에 맞는 선택 · ⑤ 결론이 흔들리는가(ui_decide)  ⑥ 리포트(ui_report)
import streamlit as st

import app_logic as A
import ui_data, ui_setup, ui_results, ui_decide, ui_report
from ui_common import CSS, INK, SUB, LINE, PAPER, html, gap, note

st.set_page_config(page_title="STRATA — 인력구조 시나리오 플래너",
                   layout="wide", initial_sidebar_state="collapsed")
html(CSS)

ss = st.session_state
if "cfg" not in ss:
    ss["cfg"] = A.new_cfg()
cfg = ss["cfg"]
STEPS = A.STEPS
if "goto" in ss:                      # 버튼으로 단계를 옮길 때(위젯이 만들어지기 전에 값 지정)
    ss["step"] = ss.pop("goto")
ss.setdefault("step", STEPS[0])
if ss["step"] not in STEPS:
    ss["step"] = ss.get("step_last", STEPS[0])

html("""<div class="topbar">
<div class="wordmark">STRATA<span>.</span></div>
<div class="tag">인력구조 시나리오 플래너</div></div>""")
html('<p class="h-title">정년연장이 우리 회사 인력구조에 미치는 영향</p>')
html('<p class="h-sub">우리 회사 인사 명부를 올리면, 논의 중인 정년연장·재고용 안들이 앞으로 10~20년 동안 '
     '채용·인건비·승진·인력 구성을 어떻게 바꾸는지 계산해 비교합니다.<br>'
     '다른 안보다 나은 점이 없는 안은 걸러내고, 남은 후보 중 우리 회사 조건에 맞는 선택을 찾도록 돕습니다.</p>')

step = st.segmented_control("단계", STEPS, key="step", label_visibility="collapsed")
if step is None:
    step = ss.get("step_last", STEPS[0])
ss["step_last"] = step

data = ss.get("data")
res = ss.get("res")
has_data = bool(data and data.get("std") is not None)
stale = bool(res and has_data and res["sig"] != A.signature(cfg, data))


def need(what, target, btn):
    gap(10)
    html(f'<div class="card" style="max-width:640px"><div class="card-t">{what}</div>'
         f'<div class="card-b">이 단계는 앞 단계를 마친 뒤에 볼 수 있습니다.</div></div>')
    gap(8)
    if st.button(btn, type="primary"):
        ss["goto"] = target
        st.rerun()


if step == STEPS[0]:
    ui_data.render(cfg)
    if not ss.get("data"):
        gap(14)
        c = st.columns(3)
        boxes = [("① 명부를 올리면", "나이·근속·임금만 있으면 됩니다.<br>빠진 값·단위 오류를 먼저 점검하고, 부서별로 "
                                  "언제 몇 명이 정년에 도달하는지부터 보여줍니다."),
                 ("② 안을 고르고 만들면", "국회·경사노위에서 실제 논의 중인 안 10가지가 들어 있습니다.<br>"
                                    "임금피크·희망퇴직을 섞은 우리 회사만의 안도 직접 만들 수 있습니다."),
                 ("③ 결과를 검증합니다", "차이가 우연인지 진짜인지 통계로 가리고,<br>가정이 틀려도 결론이 유지되는지 "
                                   "27가지 상황과 108가지 설계로 확인합니다.")]
        for col, (t, b) in zip(c, boxes):
            col.markdown(f'<div class="card"><div class="card-t">{t}</div><div class="card-b">{b}</div></div>',
                         unsafe_allow_html=True)
elif step == STEPS[1]:
    if not has_data:
        need("먼저 분석할 데이터를 정해주세요", STEPS[0], "① 회사 데이터로 가기")
    else:
        ui_setup.render(cfg, data)
else:
    if not has_data:
        need("먼저 분석할 데이터를 정해주세요", STEPS[0], "① 회사 데이터로 가기")
    elif not res:
        need("아직 분석을 실행하지 않았습니다", STEPS[1], "② 가정·시나리오로 가기")
    elif step == STEPS[2]:
        ui_results.render(res, stale)
        gap(8)
        if st.button("다음: 우리 회사에 맞는 선택 찾기 →", type="primary"):
            ss["goto"] = STEPS[3]
            st.rerun()
    elif step == STEPS[3]:
        ui_decide.render_decide(res, cfg, stale)
    elif step == STEPS[4]:
        ui_decide.render_robust(res, cfg, stale)
    else:
        ui_report.render(res, cfg, data, stale)

html('<div class="footnote">데이터: 한국노동패널조사(KLIPS) 26·27차 · 사업체패널조사(WPS) 2023 · '
     '고용형태별근로실태조사 2025(KOSIS) · 고용노동부 고령자 고용현황(2023). '
     '정년연장 법안은 2026년 11월 현재 국회 계류 중이며, 시나리오는 공개된 안을 옮긴 것입니다.<br>'
     'STRATA는 의사결정 참고용 도구이며 법률·재무 자문을 대체하지 않습니다. '
     '올린 명부는 분석하는 동안에만 메모리에 머물고 저장되지 않습니다.</div>')
