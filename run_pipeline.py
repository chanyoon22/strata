"""run_pipeline.py — 화면 없이 전체 흐름을 한 번에 실행: 시뮬레이션 → 판정 → 민감도·견고성 → 검증 → 해석 → PDF

사용법
  python run_pipeline.py                      표준 표본(KLIPS 300인+)으로 실행
  python run_pipeline.py 명부.csv             회사 명부로 실행(열 이름은 자동 인식)
  python run_pipeline.py 명부.csv 2026        명부 기준 연도를 지정

검증 요약 문구는 validation_v2.summary()에서 그때그때 계산해 넣는다(하드코딩 금지)."""
import json
import sys
import time
from pathlib import Path

import app_logic as A
import company_data as C
import validation_v2 as V
from dss_layer import sensitivity, robustness, HIRING_POLICIES
from engine_retirement_v2 import WAGE_REGIMES
from llm_report import build_llm_input, render_offline, build_prompt, validate_prose
from report_generator import build_pdf

_OUT = Path(__file__).parent          # 항상 이 스크립트가 있는 폴더에 출력
t0 = time.time()
cfg = A.new_cfg()
cfg.update(runs=200, horizon=15)

if len(sys.argv) > 1:
    raw = C.read_table(sys.argv[1])
    base_year = int(sys.argv[2]) if len(sys.argv) > 2 else C.BASE_YEAR
    m = C.auto_map(raw)
    annual = not m.get("wage")
    wcol = m.get("wage") or m.get("wage_annual")
    unit, annual, msg = C.detect_wage_unit(raw[wcol], annual)
    m["wage"] = wcol
    std, q = C.standardize(raw, m, base_year=base_year, wage_unit=unit, wage_is_annual=annual)
    for i in q["issues"]:
        print(f"  [{i['level']}] {i['title']} — {i['detail']}")
    if std is None:
        sys.exit("명부를 읽지 못했습니다.")
    ce = C.infer_ce_practice(std)
    cfg.update(src=A.SRC_UPLOAD, start_year=base_year, ce_rate=ce["rate"], ce_end=ce["end_age"], ce_ratio=ce["wage_ratio"])
    data = dict(mode=A.SRC_UPLOAD, std=std, quality=q, grade_table=C.infer_grades(std), name=Path(sys.argv[1]).name, ce=ce)
    print(f"[0/5] 명부 {q['n_used']:,}명 읽음 ({msg})")
else:
    data = dict(mode=A.SRC_DEMO, std=C.klips_standard(), name="표준 대기업 표본")

res = A.run_analysis(cfg, data)
out, prof = res["out"], res["profile"]
print(f"[1/5] 시나리오 {len(out['results'])}종 몬테카를로 ({cfg['runs']}회 × {cfg['horizon']}년) 완료")

print("[2/5] 재고용 조건 격자 탐색 (수락률 5 × 임금률 5)...")
sens = sensitivity(prof, runs=30, horizon=cfg["horizon"], hiring=cfg["hiring"], scenarios=res["scenarios"],
                   base=(cfg["accept"], cfg["wage_ratio"]))
print("[3/5] 가정 조합 27가지 견고성 분석...")
robust = robustness(prof, res["scenarios"], hiring=cfg["hiring"], runs=20, horizon=cfg["horizon"],
                    weights={k: v for k, v in cfg["weights"].items() if v > 0})

LIMITATIONS = A.limitations(cfg, data, res)
payload = build_llm_input(out, sens, LIMITATIONS, robust)
prose = render_offline(payload)
checked, dropped = validate_prose(prose, payload)
json.dump(payload, open(_OUT / "llm_payload.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
(_OUT / "llm_prompt_example.txt").write_text(build_prompt(json.loads(json.dumps(payload, default=str))), encoding="utf-8")

print("[4/5] 검증 스위트 실행...")
val_results = [fn() for fn in V.ALL_TESTS]
val_summary = V.summary(val_results)
if not val_summary["computed_all_passed"]:
    print("경고: 자동 계산 검증 중 실패 항목 있음 —", val_summary["label"])

print("[5/5] PDF 리포트 생성...")
regime = "자사 입력 승급률" if prof.wage_regime == "custom" else WAGE_REGIMES[prof.wage_regime]["label"]
meta = [
    "데이터: 회사 명부(올린 경우) + KLIPS 26·27차 이직 모형(재적합·DiD 포함) + WPS 2023 + 고용형태별근로실태조사 2025 + 고용노동부 고령자 고용현황",
    f"검증: {val_summary['label']}",
    "후보 판정은 결정론적 알고리즘(대응표본 통계적 지배 필터)이 수행하고, AI는 해석 문장만 씁니다(숫자 대조 포함).",
] + LIMITATIONS
pdf = build_pdf(out, sens, checked,
                profile_desc=(f"{prof.name} · {out['results'][0]['summary']['n_initial']:,}명 · "
                              f"채용 방식 {HIRING_POLICIES[cfg['hiring']][0]} · 50세 이후 임금 {regime}"),
                params_meta=meta, out_path=str(_OUT / "executive_report.pdf"), outdir=str(_OUT),
                robustness_output=robust, profile_job_default=prof.job_table_is_default())
print(f"완료 ({time.time()-t0:.0f}초): {pdf}")
print("후보:", [n.split(' ')[0] for n in out["frontier"]], "/ 제외:", [n.split(' ')[0] for n in out["dominated"]])
print("재고용 조건에서 후보가 바뀌는 조합:", len(sens["flip_points"]), "/ 25")
print("27가지 상황 1위 횟수:", {n.split(' ')[0]: v["n_best"] for n, v in robust["regret"].items()})
print("해석문:", checked)
if dropped:
    print("검증에서 빠진 문장:", dropped)
