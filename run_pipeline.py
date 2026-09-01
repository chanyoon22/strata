"""run_pipeline.py — 전체 파이프라인 데모 실행: 시뮬레이션 → 파레토 → 민감도 → 검증 → 해석 → PDF
검증 요약 문구는 validation_v2.summary()에서 그때그때 계산해 넣는다(하드코딩 금지) —
코드가 바뀌어도 리포트 문구가 자동으로 따라오게 하기 위함(2026-08-17 발견된 문서-코드
불일치 재발 방지)."""
import json, time
from pathlib import Path
from engine_retirement_v2 import CompanyProfile, WAGE_REGIMES
from dss_layer import run_all, sensitivity, HIRING_POLICIES
from llm_report import build_llm_input, render_offline, build_prompt, validate_prose
from report_generator import build_pdf
import validation_v2 as V

# 리포트가 어떤 전제로 만들어졌는지 문서에 남기기 위해 상수로 뽑아 둔다.
HIRING = "replace"      # 정원 유지
REGIME = "observed"     # 현행 관측 임금 궤적

# 실행 위치(사용자 PC의 임의 폴더)와 무관하게 항상 이 스크립트가 있는 폴더를 기준으로
# 출력한다. 이전 버전은 /home/claude/dss(작업 세션 전용 경로)에 하드코딩돼 있어
# 사용자 PC에서 직접 실행하면 FileNotFoundError로 즉시 죽는 버그가 있었음(2026-08-28 수정).
_OUT = Path(__file__).parent
t0 = time.time()
profile = CompanyProfile(n_employees=1000, wage_regime=REGIME)

out = run_all(profile, runs=200, horizon=12, hiring=HIRING)
print(f"[1/4] 시나리오 {len(out['results'])}종 몬테카를로 (200회 × 12년) 완료")

print("[2/4] 민감도 격자 탐색 (수락률 5 × 임금률 5)...")
sens = sensitivity(profile, runs=40, horizon=12, hiring=HIRING)

LIMITATIONS = [
    "재고용 수락률은 기업 내 제안 수락이 KLIPS에서 직접 관측되지 않아 재취업률 앵커(0.37~0.62) 기반 구간 파라미터 — 자사 실측 필요",
    "push hazard 주 계수는 KLIPS 대표본 실측(사건 1,858건), 26→27차 자체 재적합(사건 574건)에서 5/5 방향 일치·3/5 유의로 표본외 검증",
    "2016년 60세 의무화 DiD는 평행추세 가정을 통과하지 못함 — 처치 전 격차가 이미 연 -1.4%p로 좁혀지고 있었고(V8), "
    "추세 통제 시 효과가 소멸. 인과효과가 아니라 '제도 변화와 같은 방향'이라는 정합성 확인으로만 해석할 것. "
    "300인 미만도 1년 뒤 시행되어 순수 대조군도 없음",
    "50세 이후 임금 궤적은 KOSIS 연령별 임금의 횡단면 곡선에서 가져온 값 — 이 곡선엔 임금피크·재고용 전환이 이미 "
    "섞여 있어 개인 궤적으로 쓰면 정년연장 비용을 과소평가할 수 있음. 호봉제 가정 시 정년연장 비용이 약 2.5배로 커짐(레짐 선택 가능, 후보군 구성은 불변)",
    "채용 정책(정원 유지·인건비 총액 유지·채용 규모 고정)은 사용자 선택이며 이 선택에 따라 권장 후보군이 바뀜 — "
    "모형이 답을 주는 것이 아니라 회사가 먼저 정해야 하는 전제",
    "55세+ 재직 인년은 고령인력의 '규모'이지 숙련도를 직접 측정한 값이 아님 — 정년연장 시 정의상 증가하는 지표",
    "90% 구간은 인력 구성 표본변동과 이직모형 계수 추정오차만 반영 — 경기·산업 충격은 모형에 없음",
    "기타 이직률 기본값 4.5%는 자체 실측 7.5%(전규모)를 대기업 기준 하향 조정한 값 — 자사값 입력 권장",
    "교육투자 등 미검증 채널은 비용만 반영, 효과는 인과 주장하지 않음",
]

print("[3/4] 해석문 생성 (오프라인 렌더러) + 수치 검증...")
payload = build_llm_input(out, sens, LIMITATIONS)
prose = render_offline(payload)
checked, dropped = validate_prose(prose, payload)
json.dump(payload, open(_OUT / "llm_payload.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
(_OUT / "llm_prompt_example.txt").write_text(build_prompt(payload), encoding="utf-8")

print("[3.5/4] 검증 스위트 실행 (문구는 지금 이 결과에서 생성)...")
val_results = [fn() for fn in V.ALL_TESTS]
val_summary = V.summary(val_results)
if not val_summary["computed_all_passed"]:
    print("경고: 자동 계산 검증 중 실패 항목 있음 —", val_summary["label"])

print("[4/4] 경영진 PDF 리포트 생성...")
meta = [
    "데이터: KLIPS 26·27차 개인패널+직업력 자체 분석(재적합·DiD 포함) + KOSIS 고용형태별근로실태조사 2025 + KLIPS 대기업 실표본 1,789명 joint 부트스트랩",
    "정년퇴직 후 재취업률 62.1% / 1년내 37.4% / 재취업 임금비 중위 0.51 (300인+ 출신, n=190)",
    f"검증: {val_summary['label']} — 자동 계산 {val_summary['computed_n']}종은 이 파이프라인 실행 시점에 "
    f"재계산된 결과, 문서화 {val_summary['documented_n']}종(외부벤치마크·정년퇴직 나이식별)은 원자료 1회 분석을 인용",
    f"분석 전제: 채용 정책 = {HIRING_POLICIES[HIRING][0]} / 50세 이후 임금 궤적 = {WAGE_REGIMES[REGIME]['label']} "
    f"— 두 전제 모두 결과를 크게 바꾸므로 다른 전제의 회사에는 그대로 적용할 수 없음",
    "권장 후보는 파레토 지배 필터가 결정론적으로 도출, AI는 수치 해석 산문만 생성(환각 사후검증 포함)",
] + LIMITATIONS
pdf = build_pdf(out, sens, checked,
                profile_desc=(f"직원 1,000명, 300인 이상 대기업 가정 프로파일 · "
                              f"채용 정책 {HIRING_POLICIES[HIRING][0]} · "
                              f"임금 궤적 {WAGE_REGIMES[REGIME]['label']}"),
                params_meta=meta, out_path=str(_OUT / "executive_report.pdf"), outdir=str(_OUT))
print(f"완료 ({time.time()-t0:.0f}초): {pdf}")
print("프론티어:", out["frontier"])
print("플립 포인트 수:", len(sens["flip_points"]))
