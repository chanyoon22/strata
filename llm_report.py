"""
llm_report.py — AI 해석 리포트 층 (PassNavi 환각방지 아키텍처 계승)
====================================================================
역할 경계 (설계 원칙):
  - 결정: 파레토 지배 필터(dss_layer)가 결정론적으로 수행. LLM은 후보를 추가/제거 불가.
  - 수치: 엔진이 계산한 값만 사용. LLM은 새 숫자 생성 금지 (프롬프트에 명시 + 사후 검증).
  - 해석: LLM은 제공된 수치의 의미·트레이드오프·위험요인 서술만 담당.

사후 검증(validate_prose): 산문에 등장하는 모든 수치가 입력 JSON에 존재하는지
정규식으로 대조 → 미등재 수치 발견 시 해당 문장 폐기 + 경고 로그.
API 미연결 환경을 위해 결정론적 오프라인 렌더러(render_offline)도 제공.
"""
from __future__ import annotations
import json
import re

PROMPT_TEMPLATE = """당신은 HR 의사결정 지원 시스템의 해석 담당자입니다. 아래 JSON은 \
결정론적 시뮬레이션 엔진이 계산한 결과입니다. 규칙을 엄격히 지키십시오.

[규칙]
1. 아래 JSON에 없는 숫자를 절대 만들지 마십시오. 인용하는 모든 수치는 JSON 값 그대로 써야 합니다.
2. 권장 후보군(frontier)은 이미 알고리즘이 확정했습니다. 후보를 추가하거나 빼지 마십시오.
3. 당신의 역할은 (a) 각 후보의 트레이드오프 서술, (b) 민감도 결과가 시사하는 위험요인,
   (c) 경영진이 추가로 검토할 질문 제안 — 이 세 가지뿐입니다.
4. 단정적 추천("S3을 선택하라")은 금지. "채용 여력을 우선한다면 ~" 형식의 조건부 서술만 허용.
5. 시뮬레이션의 한계(limitations 필드)를 반드시 본문에 포함하십시오.

[시뮬레이션 결과 JSON]
{payload}

위 규칙에 따라 경영진 보고용 해석문(한국어, 600자 내외)을 작성하십시오."""


def build_llm_input(run_output, sensitivity_output, limitations):
    """엔진 산출 → LLM 입력 JSON. 이 함수 밖의 데이터는 LLM에 전달되지 않는다."""
    return dict(
        scenarios=[r["summary"] for r in run_output["results"]],
        frontier=run_output["frontier"],
        dominated=run_output["dominated"],
        sensitivity=dict(baseline_frontier=sensitivity_output["baseline_frontier"],
                         flip_points=sensitivity_output["flip_points"][:8]),
        limitations=limitations,
    )


def build_prompt(payload: dict) -> str:
    return PROMPT_TEMPLATE.format(payload=json.dumps(payload, ensure_ascii=False, indent=1))


_NUM = re.compile(r"\d[\d,]*\.?\d*")


def validate_prose(prose: str, payload: dict, tol=0.5):
    """산문 속 숫자가 payload에 실재하는지 검증. 근거 없는 수치 포함 문장은 제거."""
    allowed = set()
    def collect(x):
        if isinstance(x, dict):
            for v in x.values():
                collect(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                collect(v)
        elif isinstance(x, (int, float)):
            allowed.add(round(float(x), 1))
    collect(payload)

    kept, dropped = [], []
    for sent in re.split(r"(?<=[.다요])\s+", prose):
        nums = [float(m.replace(",", "")) for m in _NUM.findall(sent)]
        ok = all(any(abs(n - a) <= tol or (a and abs(n / a - 1) < 0.001) for a in allowed)
                 or n in (2026, 2037, 2032, 2028, 2029, 60, 61, 62, 63, 64, 65)  # 제도 연도/연령
                 for n in nums)
        (kept if ok else dropped).append(sent)
    return " ".join(kept), dropped


def render_offline(payload: dict) -> str:
    """API 없이도 동작하는 결정론적 해석문(템플릿). LLM 사용 시엔 이 문장을 대체."""
    s = {x["scenario"]: x for x in payload["scenarios"]}
    fr = payload["frontier"]
    base = next(iter(s.values()))
    lines = [f"파레토 분석 결과 {len(fr)}개 시나리오가 권장 후보군으로 남았고, "
             f"{len(payload['dominated'])}개는 모든 지표에서 열위로 탈락했습니다."]
    for name in fr:
        v = s[name]
        lines.append(f"· {name}: 10년 인건비 NPV {v['cost_npv_mean']/1e4:,.0f}억원, "
                     f"신규채용 누계 {v['total_hires_mean']:,.0f}명, "
                     f"55세 이상 재직 인년 {v['exp55_person_years_mean']:,.0f}.")
    if payload["sensitivity"]["flip_points"]:
        a, w, _ = payload["sensitivity"]["flip_points"][0]
        lines.append(f"단, 재고용 수락률 {a:.2f}·임금률 {w:.2f} 부근에서 후보 구성이 "
                     f"달라지므로 이 두 파라미터의 자사 실측이 선행 과제입니다.")
    lines.append("한계: " + " / ".join(payload["limitations"]))
    return "\n".join(lines)
