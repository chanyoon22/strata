"""
interpret.py — 결과를 쉬운 문장으로 풀어주는 결정론적 해석기
====================================================================
LLM을 쓰지 않는다. 모든 문장은 계산된 수치에 규칙을 적용해 만든다(같은 입력 → 같은 문장).
문장마다 '숫자가 얼마인지 → 그게 큰 건지 작은 건지 → 그래서 무엇을 봐야 하는지' 순서로 쓴다.
크기 판단 기준(작다/뚜렷하다/크다)은 아래 상수에 모아 두었다.
"""
from __future__ import annotations
import numpy as np
import decision as D

# 기준선 대비 변화율의 크기 구분(절대값 기준)
COST_SMALL, COST_BIG = 0.01, 0.03       # 인건비: 1% 미만 작음, 3% 이상 큼
HIRE_SMALL, HIRE_BIG = 0.05, 0.15       # 신규채용: 5% 미만 작음, 15% 이상 큼
WAIT_SMALL, WAIT_BIG = 0.3, 1.0         # 승진 대기(년)


def _s(name):
    return name.split(" ")[0]


_DIGIT_BATCHIM = set("013678")          # 영·일·삼·육·칠·팔은 받침이 있다


def josa(word, with_batchim, without):
    """한글 조사 고르기. josa('S6', '이', '가') → 'S6이', josa('현행 유지', '이', '가') → '현행 유지가'."""
    w = str(word).rstrip(") ")
    ch = w[-1] if w else ""
    if "가" <= ch <= "힣":
        has = (ord(ch) - 0xAC00) % 28 != 0
    elif ch.isdigit():
        has = ch in _DIGIT_BATCHIM
    else:
        has = ch.lower() in "lmnr"
    return f"{word}{with_batchim if has else without}"


def _rng_txt(lo, hi, unit="%", digits=1):
    if abs(hi - lo) < 10 ** (-digits):
        return f"{lo:+.{digits}f}{unit}"
    return f"{lo:+.{digits}f}{unit} ~ {hi:+.{digits}f}{unit}"


def deltas(results, baseline):
    """기준 시나리오 대비 변화. {이름: dict(cost, hires, wait, exp55, gap, ...)}"""
    b = next(r["summary"] for r in results if r["summary"]["scenario"] == baseline)
    out = {}
    for r in results:
        s = r["summary"]
        out[s["scenario"]] = dict(
            cost=s["cost_npv_mean"] / b["cost_npv_mean"] - 1,
            cost_abs=(s["cost_npv_mean"] - b["cost_npv_mean"]) / 1e4,
            hires=s["total_hires_mean"] / max(b["total_hires_mean"], 1e-9) - 1,
            hires_abs=s["total_hires_mean"] - b["total_hires_mean"],
            wait=s["promo_congestion_mean"] - b["promo_congestion_mean"],
            exp55=s["exp55_person_years_mean"] / max(b["exp55_person_years_mean"], 1e-9) - 1,
            gap=s["income_gap_mean"] - b["income_gap_mean"],
            gap_level=s["income_gap_mean"],
            skill=s.get("skill_gap_mean", 0) / max(b.get("skill_gap_mean", 0), 1e-9) - 1,
            skill_abs=s.get("skill_gap_mean", 0) - b.get("skill_gap_mean", 0))
    return out


def tags(results, frontier):
    """시나리오별 한 줄 특징(후보 가운데 어떤 점에서 가장 좋은지)."""
    pool = [r["summary"] for r in results if r["summary"]["scenario"] in frontier] or \
           [r["summary"] for r in results]
    best = {
        "인건비 최저": min(pool, key=lambda s: s["cost_npv_mean"])["scenario"],
        "신규채용 최다": max(pool, key=lambda s: s["total_hires_mean"])["scenario"],
        "승진 대기 최단": min(pool, key=lambda s: s["promo_congestion_mean"])["scenario"],
        "고령 인력 최다": max(pool, key=lambda s: s["exp55_person_years_mean"])["scenario"],
        "소득 공백 최소": min(pool, key=lambda s: s["income_gap_mean"])["scenario"],
        "숙련 공백 최소": min(pool, key=lambda s: s.get("skill_gap_mean", 0))["scenario"],
    }
    out = {r["summary"]["scenario"]: [] for r in results}
    for tag, name in best.items():
        out[name].append(tag)
    return out


def headline(out, baseline):
    """결과 화면 맨 위 요약 문장 3개 이내. [(제목, 본문 html)]"""
    results = out["results"]
    d = deltas(results, baseline)
    others = [n for n in d if n != baseline]
    horizon = results[0]["summary"]["horizon"]
    cards = []
    if not others:
        return cards

    # ① 채용
    h = [d[n]["hires"] for n in others]
    worst = min(others, key=lambda n: d[n]["hires"])
    per_year = abs(d[worst]["hires_abs"]) / horizon
    mag = abs(min(h))
    if mag >= HIRE_BIG:
        tone = "<b>크게 줄어듭니다.</b> 청년 채용 계획과 인력 수급에 직접 영향을 주는 수준입니다."
    elif mag >= HIRE_SMALL:
        tone = "<b>뚜렷하게 줄어듭니다.</b> 채용 계획을 다시 짜야 할 수 있는 수준입니다."
    else:
        tone = "<b>거의 변하지 않습니다.</b>"
    cards.append(("신규채용",
                  f"현행과 비교한 {horizon}년 누계 신규채용은 <b>{_rng_txt(min(h)*100, max(h)*100)}</b> 범위입니다.<br>"
                  f"가장 많이 줄어드는 {josa(_s(worst), '은', '는')} 해마다 약 <b>{per_year:,.0f}명</b>씩 덜 뽑게 됩니다. {tone}"))

    # ② 인건비
    c = [d[n]["cost"] for n in others]
    hi_n = max(others, key=lambda n: d[n]["cost"])
    mag = max(abs(min(c)), abs(max(c)))
    if mag >= COST_BIG:
        tone = "인건비 부담이 <b>의미 있게 달라집니다.</b> 예산 계획에 반영해야 합니다."
    elif mag >= COST_SMALL:
        tone = "인건비 차이가 <b>작지만 무시할 정도는 아닙니다.</b>"
    else:
        tone = ("인건비 차이는 <b>1% 미만으로 작습니다.</b> 고령자가 남는 만큼 신입을 덜 뽑아 "
                "총액이 서로 상쇄되기 때문입니다.")
    cards.append(("인건비",
                  f"{horizon}년 인건비(현재가치)는 현행 대비 <b>{_rng_txt(min(c)*100, max(c)*100, digits=2)}</b>입니다.<br>"
                  f"가장 많이 드는 {josa(_s(hi_n), '이', '가')} 현행보다 약 <b>{d[hi_n]['cost_abs']:+,.0f}억 원</b>입니다. {tone}"))

    # ③ 승진
    w = [d[n]["wait"] for n in others]
    long_n = max(others, key=lambda n: d[n]["wait"])
    mag = max(w)
    if mag >= WAIT_BIG:
        tone = "승진 적체가 <b>체감될 만큼 길어집니다.</b> 중간 직급의 불만·이탈 위험을 함께 봐야 합니다."
    elif mag >= WAIT_SMALL:
        tone = "승진이 <b>조금 늦어집니다.</b>"
    else:
        tone = "승진 속도는 <b>거의 달라지지 않습니다.</b>"
    cards.append(("승진",
                  f"승진 대상자가 기다리는 기간은 현행 대비 <b>{_rng_txt(min(w), max(w), unit='년')}</b> 달라집니다.<br>"
                  f"가장 길어지는 것은 {_s(long_n)}입니다. {tone}"))
    return cards


def verdict_sentence(out):
    n_f, n_d = len(out["frontier"]), len(out["dominated"])
    if n_d == 0:
        return (f"{n_f}개 시나리오가 모두 후보로 남았습니다. 어느 하나가 다른 것보다 모든 면에서 "
                "낫다고 할 수 없어, 회사가 무엇을 우선하는지에 따라 선택이 갈립니다.")
    return (f"{n_f + n_d}개 가운데 <b>{n_f}개가 후보</b>로 남았고, {n_d}개는 "
            "다른 시나리오보다 나은 점이 없어 제외했습니다.")


_WHY = {"cost_npv": "인건비가 {x:.1f}% 더 적", "total_hires": "신규채용이 {x:.1f}% 더 많",
        "promo_congestion": "승진 대기가 {x:.1f}% 더 짧", "exp55_person_years": "고령 인력이 {x:.1f}% 더 많이 남",
        "income_gap": "퇴직자 소득 공백이 {x:.1f}% 더 적", "skill_gap": "숙련 공백이 {x:.1f}% 더 적"}


def why_text(name, why):
    """탈락 사유를 한 문장으로. 예: 'S7 쪽이 인건비가 1.1% 더 적습니다. 나머지 항목은 비슷합니다.'"""
    w = why.get(name)
    if not w:
        return ""
    parts = [_WHY[k].format(x=abs(w["detail"][k]["rel"]) * 100) for k in w["better_on"]]
    return f"{_s(w['by'])} 쪽이 " + "고, ".join(parts) + "습니다. 나머지 항목은 비슷합니다."


def income_gap_text(out, baseline):
    d = deltas(out["results"], baseline)
    base = d[baseline]["gap_level"]
    if base <= 0:
        return None
    best = min(d, key=lambda n: d[n]["gap_level"])
    cut = 1 - d[best]["gap_level"] / base
    return (f"현행대로 가면 퇴직자들이 국민연금을 받기 전까지 소득이 끊기는 기간이 합계 <b>{base:,.0f}인년</b> "
            f"생깁니다(퇴직자 한 사람이 1년 공백이면 1인년).<br>"
            f"{josa(_s(best), '은', '는')} 이 공백을 <b>{cut*100:.0f}%</b> 줄입니다. 노사 협의에서 근로자 측이 가장 "
            f"중요하게 보는 숫자입니다.")


def recommend_text(ranked, weights, n_fail):
    """우선순위 반영 결과 문장. ranked: [(이름, 점수)] 조건 통과분만."""
    names = {"cost_npv": "인건비", "total_hires": "신규채용", "promo_congestion": "승진 속도",
             "exp55_person_years": "고령 인력 유지", "income_gap": "퇴직자 소득 공백",
             "skill_gap": "숙련 공백"}
    top_w = max(weights.values()) if weights else 0
    focus = [names[k] for k, w in weights.items() if w == top_w and w > 0]
    if not ranked:
        return ("입력하신 한도를 모두 만족하는 시나리오가 없습니다. 한도를 조금 넓히거나, "
                "아래 '정책 공간 탐색'에서 다른 설계를 찾아보세요.")
    best, score = ranked[0]
    head = f"입력하신 조건에서는 <b>{josa(best.split('(')[0].strip(), '이', '가')}</b> 가장 잘 맞습니다."
    why = f" {josa('·'.join(focus), '을', '를')} 가장 중요하게 본 결과입니다." if focus and len(focus) < len(weights) else ""
    close = ""
    if len(ranked) > 1 and ranked[1][1] - score < 0.05:
        close = (f"<br>다만 2위 {josa(ranked[1][0].split(' ')[0], '과', '와')} 점수 차이가 작아, 중요도를 조금만 바꿔도 "
                 "순위가 뒤바뀔 수 있습니다.")
    fail = f"<br>한도를 넘어 제외된 시나리오는 {n_fail}개입니다." if n_fail else ""
    return head + why + close + fail


def break_even_text(name, be, horizon):
    """손익분기 결과를 한 문장으로. be = decision.break_even(...)."""
    nm = josa(_s(name), "은", "는")
    dc = be["dc"] / 1e4
    if be["kind"] == "always":
        return f"{nm} 인건비도 적고 숙련 손실도 적어, 숙련 가치를 얼마로 보든 현행보다 낫습니다."
    if be["kind"] == "never":
        return f"{nm} 인건비도 더 들고 숙련 손실도 커서, 숙련 가치를 얼마로 보든 현행보다 불리합니다."
    x = be["x"] * 100
    band = ""
    if be.get("lo") is not None:
        band = f"(반복 계산의 흔들림을 감안하면 {be['lo']*100:,.0f}~{be['hi']*100:,.0f}%)"
    if be["kind"] == "above":
        return (f"{nm} {horizon}년 인건비가 현행보다 약 {dc:+,.0f}억 원 더 들지만 숙련 손실을 줄입니다.<br>"
                f"숙련 공백 1인-년의 가치를 평균 인건비의 <b>{x:,.0f}% 이상</b>으로 본다면 현행보다 낫습니다{band}.")
    return (f"{nm} 인건비를 약 {abs(dc):,.0f}억 원 줄이지만 숙련 손실이 커집니다.<br>"
            f"숙련 공백 1인-년의 가치가 평균 인건비의 <b>{x:,.0f}% 이하</b>일 때만 현행보다 낫습니다{band}.")
