"""run_pipeline.py — 전체 파이프라인 데모 실행: 시뮬레이션 → 파레토 → 민감도 → 해석 → PDF"""
import json, time
from engine_retirement_v2 import CompanyProfile
from dss_layer import run_all, sensitivity
from llm_report import build_llm_input, render_offline, build_prompt, validate_prose
from report_generator import build_pdf

t0 = time.time()
profile = CompanyProfile(n_employees=1000)

print("[1/4] 시나리오 6종 몬테카를로 (200회 × 12년)...")
out = run_all(profile, runs=200, horizon=12)

print("[2/4] 민감도 격자 탐색 (수락률 5 × 임금률 5)...")
sens = sensitivity(profile, runs=40, horizon=12)

LIMITATIONS = [
    "재고용 수락률은 기업 내 제안 수락이 KLIPS에서 직접 관측되지 않아 재취업률 앵커(0.37~0.62) 기반 구간 파라미터 — 자사 실측 필요",
    "push hazard 주 계수는 KLIPS 대표본 실측(사건 1,858건), 26→27차 자체 재적합(사건 574건)에서 5/5 방향 일치·3/5 유의로 표본외 검증",
    "2016년 60세 의무화 DiD는 300인 미만도 1년 뒤 시행되어 순수 대조군 부재 — 대기업 상대효과로 해석",
    "기타 이직률 기본값 4.5%는 자체 실측 7.5%(전규모)를 대기업 기준 하향 조정한 값 — 자사값 입력 권장",
    "교육투자 등 미검증 채널은 비용만 반영, 효과는 인과 주장하지 않음",
]

print("[3/4] 해석문 생성 (오프라인 렌더러) + 수치 검증...")
payload = build_llm_input(out, sens, LIMITATIONS)
prose = render_offline(payload)
checked, dropped = validate_prose(prose, payload)
json.dump(payload, open("/home/claude/dss/llm_payload.json", "w"), ensure_ascii=False, indent=1)
open("/home/claude/dss/llm_prompt_example.txt", "w").write(build_prompt(payload))

print("[4/4] 경영진 PDF 리포트 생성...")
meta = [
    "데이터: KLIPS 26·27차 개인패널+직업력 자체 분석(재적합·DiD 포함) + KOSIS 고용형태별근로실태조사 2025 + KLIPS 대기업 실표본 1,789명 joint 부트스트랩",
    "정년퇴직 후 재취업률 62.1% / 1년내 37.4% / 재취업 임금비 중위 0.51 (300인+ 출신, n=190)",
    "검증 6종 통과: 캘리브레이션 · 인원보존 · DiD(-4.0%p, p=0.005) · 외부벤치마크 · 재적합 표본외검증(5/5 방향일치) · 정년퇴직 나이식별(중위 60세)",
    "권장 후보는 파레토 지배 필터가 결정론적으로 도출, AI는 수치 해석 산문만 생성(환각 사후검증 포함)",
] + LIMITATIONS
pdf = build_pdf(out, sens, checked,
                profile_desc="직원 1,000명, 300인 이상 대기업 가정 프로파일 (연령·임금 분포 조정 가능)",
                params_meta=meta)
print(f"완료 ({time.time()-t0:.0f}초): {pdf}")
print("프론티어:", out["frontier"])
print("플립 포인트 수:", len(sens["flip_points"]))
