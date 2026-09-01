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
import os
import re

PROMPT_TEMPLATE = """당신은 HR 의사결정 지원 시스템의 해석 담당자입니다. 아래 JSON은 \
결정론적 시뮬레이션 엔진이 계산한 결과입니다. 규칙을 엄격히 지키십시오.

[규칙]
1. 아래 JSON에 없는 숫자를 절대 만들지 마십시오. 특히 금액은 반드시 *_eok 필드(억원 단위로
   이미 변환되어 있음)를 그대로 쓰고, 직접 나누거나 곱해 새 숫자를 계산하지 마십시오.
   그 외 수치도 JSON 값 그대로 써야 합니다.
2. 권장 후보군(frontier)은 이미 알고리즘이 확정했습니다. 후보를 추가하거나 빼지 마십시오.
3. 당신의 역할은 (a) 각 후보의 트레이드오프 서술, (b) 민감도 결과가 시사하는 위험요인,
   (c) 경영진이 추가로 검토할 질문 제안 — 이 세 가지뿐입니다.
4. 단정적 추천("S3을 선택하라")은 금지. "채용 여력을 우선한다면 ~" 형식의 조건부 서술만 허용.
5. 시뮬레이션의 한계(limitations 필드)를 반드시 본문에 포함하십시오.

[시뮬레이션 결과 JSON]
{payload}

위 규칙에 따라 경영진 보고용 해석문(한국어, 600자 내외)을 작성하십시오."""


def _with_display_units(summary: dict) -> dict:
    """cost_npv_* (원 단위 NPV)에 억원 환산 필드를 나란히 추가한다.
    렌더러(오프라인/LLM)가 이 필드를 그대로 인용하게 함으로써, validate_prose가
    '원문 그대로'만 검사해도 되게 만든다 — 스케일 변환을 사후 검증기가 추측할 필요가 없어짐.
    (2026-08-28: 이 환산을 렌더러가 즉석에서(/1e4) 계산해 쓰는 이전 방식은 payload에
    그 결과가 존재하지 않아 validate_prose가 정상 문장을 전부 오탐 삭제하는 버그였음."""
    out = dict(summary)
    for k in ("cost_npv_mean", "cost_npv_p5", "cost_npv_p95"):
        if k in out:
            out[k.replace("cost_npv", "cost_npv_eok")] = round(out[k] / 1e4, 0)
    return out


def build_llm_input(run_output, sensitivity_output, limitations):
    """엔진 산출 → LLM 입력 JSON. 이 함수 밖의 데이터는 LLM에 전달되지 않는다."""
    return dict(
        scenarios=[_with_display_units(r["summary"]) for r in run_output["results"]],
        frontier=run_output["frontier"],
        dominated=run_output["dominated"],
        # 개수는 리스트 길이로 유도 가능하지만, "N개" 서술이 검증기(leaf 숫자 대조)를
        # 통과하려면 숫자 자체가 JSON에 명시적으로 있어야 함 — 그래서 명시적으로 포함.
        frontier_count=len(run_output["frontier"]),
        dominated_count=len(run_output["dominated"]),
        sensitivity=dict(baseline_frontier=sensitivity_output["baseline_frontier"],
                         flip_points=sensitivity_output["flip_points"][:8]),
        limitations=limitations,
    )


def build_prompt(payload: dict) -> str:
    return PROMPT_TEMPLATE.format(payload=json.dumps(payload, ensure_ascii=False, indent=1))


_NUM = re.compile(r"(?<![A-Za-z])\d[\d,]*\.?\d*")  # "S1", "S3" 같은 시나리오 코드의 숫자는 제외


# 제도 상수: 시뮬레이션 시작연도와 정년 연령대, "55세 이상" 지표 정의.
# 개별 법안의 도달연도(2032·2036·2041 등)는 여기 적지 않는다 — 시나리오 이름에서
# 자동으로 수집한다(아래 collect의 문자열 처리). 예전에는 이 목록에 연도를 직접 박아 두어,
# dss_layer의 스케줄을 2037→2036/2041로 교체했을 때 목록이 따라오지 않아 S2·S7 문장이
# 통째로 삭제되고 있었다(2026-08-28 발견·수정). 상수 목록과 시나리오 정의가 각각 놀면
# 반드시 어긋나므로, 데이터에서 끌어오는 쪽으로 바꾼다.
_INSTITUTIONAL = {2026, 55, 60, 61, 62, 63, 64, 65}


def validate_prose(prose: str, payload: dict, tol=0.5):
    """산문 속 숫자가 payload에 실재하는지 검증. 근거 없는 수치 포함 문장은 제거.

    payload 값을 '있는 그대로'만 인정한다(스케일 변환 추측 없음 — 변환이 필요한 값은
    build_llm_input 단계에서 이미 변환된 필드로 넣어둔다, 예: cost_npv_eok_mean).
    시나리오 이름 같은 문자열에 박힌 숫자(2036, 65 등)도 payload에 실재하는 근거이므로 함께 수집한다.
    """
    allowed = set()

    def collect(x):
        if isinstance(x, bool):
            # bool은 파이썬에서 int의 하위형이라 그냥 두면 True가 1.0으로 수집된다.
            # 그러면 [0.5, 1.5] 구간의 아무 숫자나 통과해버려(예: "1.2배 증가") 검증이 뚫린다.
            # param_uncertainty 같은 플래그가 실제로 이 구멍을 열어 두고 있었다(2026-08-28 수정).
            return
        if isinstance(x, dict):
            for v in x.values():
                collect(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                collect(v)
        elif isinstance(x, (int, float)):
            allowed.add(round(float(x), 2))
        elif isinstance(x, str):
            for m in _NUM.findall(x):          # "S2 단계적연장(2036년 65세)" → 2036, 65
                allowed.add(round(float(m.replace(",", "")), 2))

    collect(payload)
    allowed |= {float(v) for v in _INSTITUTIONAL}

    def close(n, a):
        # 허용폭은 두 값 중 '큰 쪽'의 크기로 정한다. payload 값만 보고 정하면,
        # payload에 1.0 같은 작은 값이 하나라도 있을 때 절대오차 0.5가 적용되어
        # 0.5~1.5의 임의 숫자가 전부 통과한다.
        scale = max(abs(n), abs(a))
        if scale < 10:
            return abs(n - a) <= 0.01 or (a and abs(n / a - 1) < 0.02)
        return abs(n - a) <= tol or (a and abs(n / a - 1) < 0.001)

    kept_lines, dropped = [], []
    # 줄 구조를 보존한다. 예전에는 문장 단위로 흩어 " ".join으로 되붙여, render_offline이
    # 만든 시나리오별 줄바꿈이 사라지고 PDF 1페이지에서 한 문단으로 뭉쳐 나왔다.
    for line in prose.split("\n"):
        kept_sents = []
        for sent in re.split(r"(?<=[.다요])\s+", line):
            if not sent.strip():
                continue
            nums = [float(m.replace(",", "")) for m in _NUM.findall(sent)]
            if all(any(close(n, a) for a in allowed) for n in nums):
                kept_sents.append(sent)
            else:
                dropped.append(sent)
        if kept_sents:
            kept_lines.append(" ".join(kept_sents))
    return "\n".join(kept_lines), dropped


def call_llm_api(payload: dict, api_key: str | None = None,
                  model: str = "claude-sonnet-4-5-20250929") -> tuple[str, str]:
    """Anthropic API로 해석문 생성. 실패하거나 키가 없으면 (오프라인 렌더러 결과, 사유)를 반환.
    반환값과 무관하게 이 함수를 거친 산문은 항상 validate_prose를 다시 통과해야 한다 —
    API 응답이라고 수치 창조 위험이 사라지는 게 아니다(이 프로젝트의 핵심 원칙)."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return render_offline(payload), "offline_no_key"
    try:
        import anthropic
    except ImportError:
        return render_offline(payload), "offline_sdk_missing"
    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model, max_tokens=800,
            messages=[{"role": "user", "content": build_prompt(payload)}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        if not text.strip():
            return render_offline(payload), "offline_empty_response"
        return text.strip(), "api"
    except Exception as e:  # 네트워크/키/레이트리밋 등 — 조용히 실패하지 않고 사유를 남긴 채 폴백
        return render_offline(payload), f"offline_api_error:{type(e).__name__}"


def generate_report(payload: dict, api_key: str | None = None) -> dict:
    """LLM 해석 리포트 생성 + 사후 검증을 하나로 묶은 진입점. UI는 이 함수 하나만 부르면 된다.

    반환값의 prose(검증 통과한 해석문)와 limitations(우리가 직접 쓴 한계 문구)는 분리되어 있다.
    한계 문구는 LLM 산출물이 아니라 저자 텍스트이므로 **환각 검증 대상이 아니다**.
    (2026-08-28 수정: 이전에는 한계 문구까지 산문에 이어붙여 함께 검증에 넣는 바람에,
     "사건 1,858건", "연 -1.4%p" 같이 payload에 없는 정당한 수치를 근거로 문장이 잘려나가
     PDF의 한계 서술이 문장 중간에서 끊긴 채 출력되고 있었다. 정직성을 위해 넣은 문구가
     정직성 장치에 의해 삭제되던 셈이다.)"""
    prose, source = call_llm_api(payload, api_key=api_key)
    checked, dropped = validate_prose(prose, payload)
    return dict(prose=checked, dropped=dropped, source=source,
                is_live=(source == "api"),
                limitations=list(payload.get("limitations", [])))


def render_offline(payload: dict) -> str:
    """API 없이도 동작하는 결정론적 해석문(템플릿). LLM 사용 시엔 이 문장을 대체."""
    s = {x["scenario"]: x for x in payload["scenarios"]}
    fr = payload["frontier"]
    base = next(iter(s.values()))
    h = int(base.get("horizon", 10))  # 하드코딩된 "10년"은 horizon이 12/15로 바뀌면 오기재가 됨(2026-08-28 수정)
    lines = [f"파레토 분석 결과 {len(fr)}개 시나리오가 권장 후보군으로 남았고, "
             f"{len(payload['dominated'])}개는 모든 지표에서 열위로 탈락했습니다."]
    for name in fr:
        v = s[name]
        lines.append(f"· {name}: {h}년 인건비 NPV {v['cost_npv_eok_mean']:,.0f}억원, "
                     f"신규채용 누계 {v['total_hires_mean']:,.0f}명, "
                     f"55세 이상 재직 인년 {v['exp55_person_years_mean']:,.0f}.")
    if payload["sensitivity"]["flip_points"]:
        a, w, _ = payload["sensitivity"]["flip_points"][0]
        lines.append(f"단, 재고용 수락률 {a:.2f}·임금률 {w:.2f} 부근에서 후보 구성이 "
                     f"달라지므로 이 두 파라미터의 자사 실측이 선행 과제입니다.")
    # 한계 문구는 여기서 이어붙이지 않는다 — 저자 텍스트이므로 검증을 거치지 않고
    # 소비자(PDF 6장·UI 한계 섹션)가 원문 그대로 렌더한다. generate_report()의 주석 참조.
    return "\n".join(lines)
