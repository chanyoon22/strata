"""
decision.py — 결정 규칙 층: 통계적 지배, 후보 판정 안정성, 제약·우선순위 반영 순위
====================================================================
이 파일의 모든 판단은 결정론적 알고리즘이다. LLM은 여기에 관여하지 않는다.

왜 필요한가
  v2의 파레토 필터는 평균값만 비교하고 "0.4% 이내 차이는 같다"는 고정 허용폭(ε)을 썼다.
  그 0.4%는 근거가 없는 상수였고, 차이가 우연인지 실제인지도 따지지 않았다.
  v3는 두 가지를 함께 본다.
    (1) 통계적 유의성 — 같은 난수(seed)로 돌린 시나리오 쌍의 '반복별 차이'로 신뢰구간을
        만든다(대응표본). 구간이 0을 포함하면 그 차이는 우연과 구별되지 않는다.
    (2) 실질적 크기 — 통계적으로 유의해도 차이가 materiality(기본 0.5%) 미만이면
        의사결정에 의미 없는 차이로 본다. 공통 난수를 쓰면 검정력이 매우 높아 0.05% 같은
        무의미한 차이도 '유의'하게 나오기 때문에 이 기준이 꼭 필요하다.
"""
from __future__ import annotations
import numpy as np

try:
    from scipy.stats import t as _tdist
except Exception:  # scipy가 없으면 정규 근사
    _tdist = None

# 목적함수: (요약 키, 방향, 화면 표시명, 단위). 방향 +1 = 클수록 좋음.
OBJECTIVES = {
    "cost_npv":           dict(sign=-1, label="인건비", unit="억 원", scale=1e-4),
    "total_hires":        dict(sign=+1, label="신규채용", unit="명", scale=1),
    "promo_congestion":   dict(sign=-1, label="승진 대기", unit="년", scale=1),
    "exp55_person_years": dict(sign=+1, label="55세+ 인력 규모", unit="인년", scale=1),
    "income_gap":         dict(sign=-1, label="퇴직자 소득 공백", unit="인년", scale=1),
    "skill_gap":          dict(sign=-1, label="숙련 공백", unit="인년", scale=1),
}
# v3.1: 기본 판정 항목에서 '고령 인력 유지(55세+ 인년, 많을수록 좋음)'를 '숙련 공백(적을수록 좋음)'으로
# 바꿨다. 55세 이상이 많을수록 좋다는 것은 가치 판단이지 회사의 손익이 아니다. 숙련 공백은
# '사람이 나간 자리가 다시 제 몫을 할 때까지의 손실'이라 근속·직군별 숙련 기간을 반영한다.
DEFAULT_OBJECTIVES = ["cost_npv", "total_hires", "promo_congestion", "skill_gap"]
DEFAULT_MATERIALITY = 0.005
DEFAULT_ALPHA = 0.05


def _tcrit(dof, alpha):
    if _tdist is not None and dof > 0:
        return float(_tdist.ppf(1 - alpha / 2, dof))
    return 1.96


def paired_diff(a_runs, b_runs, key, sign, alpha=DEFAULT_ALPHA):
    """a − b 의 반복별 차이(방향 보정: 양수 = a가 좋음)에 대한 평균·신뢰구간·상대크기."""
    a, b = np.asarray(a_runs[key], float), np.asarray(b_runs[key], float)
    if len(a) != len(b):
        raise ValueError("대응표본 비교에는 반복 횟수가 같아야 합니다.")
    d = sign * (a - b)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    half = _tcrit(n - 1, alpha) * se
    scale = max(abs(float(a.mean())), abs(float(b.mean())), 1e-9)
    return dict(mean=mean, lo=mean - half, hi=mean + half, rel=mean / scale, se=se, n=n)


def compare(a, b, objectives=DEFAULT_OBJECTIVES, materiality=DEFAULT_MATERIALITY,
            alpha=DEFAULT_ALPHA):
    """시나리오 a를 b와 목적별로 비교. 각 목적에 판정을 붙인다.

    verdict  'better' / 'worse' : 신뢰구간이 0을 벗어나고(유의) 차이가 materiality 이상(실질적)
             'same'             : 그 외
    blocks   a의 평균이 b보다 materiality 넘게 나쁘면 True — 유의성과 무관.
             지배 판정에서는 이 값을 쓴다. 반복 수가 적어 구간이 넓을 때 '나쁜 점이 우연에 묻혀
             안 보이는' 바람에 멀쩡한 시나리오가 탈락하는 일을 막기 위해서다. 즉 '낫다'는 주장은
             엄격하게(유의해야 인정), '나쁘지 않다'는 주장도 엄격하게(평균이 나쁘면 불인정) 본다."""
    out = {}
    for k in objectives:
        r = paired_diff(a["runs"], b["runs"], k, OBJECTIVES[k]["sign"], alpha)
        if r["lo"] > 0 and r["rel"] > materiality:
            r["verdict"] = "better"
        elif r["hi"] < 0 and r["rel"] < -materiality:
            r["verdict"] = "worse"
        else:
            r["verdict"] = "same"
        r["blocks"] = bool(r["rel"] < -materiality)
        out[k] = r
    return out


def dominates(a, b, objectives=DEFAULT_OBJECTIVES, materiality=DEFAULT_MATERIALITY,
              alpha=DEFAULT_ALPHA):
    """a가 b를 지배: 어떤 목적에서도 실질적으로 나쁘지 않고, 하나 이상에서 유의·실질적으로 낫다."""
    c = compare(a, b, objectives, materiality, alpha).values()
    return (not any(x["blocks"] for x in c)) and any(x["verdict"] == "better" for x in c)


def statistical_frontier(results, objectives=DEFAULT_OBJECTIVES,
                         materiality=DEFAULT_MATERIALITY, alpha=DEFAULT_ALPHA):
    """(후보 이름 목록, 탈락 이름 목록, 탈락 사유 dict). 사유는 '누구에게 어떤 목적에서 밀렸는지'."""
    frontier, dominated, why = [], [], {}
    for b in results:
        beaten_by = None
        for a in results:
            if a is b:
                continue
            c = compare(a, b, objectives, materiality, alpha)
            if (not any(x["blocks"] for x in c.values())) and any(x["verdict"] == "better" for x in c.values()):
                beaten_by = dict(by=a["summary"]["scenario"],
                                 better_on=[k for k, x in c.items() if x["verdict"] == "better"],
                                 detail={k: dict(rel=x["rel"], lo=x["lo"], hi=x["hi"])
                                         for k, x in c.items()})
                break
        name = b["summary"]["scenario"]
        if beaten_by:
            dominated.append(name)
            why[name] = beaten_by
        else:
            frontier.append(name)
    return frontier, dominated, why


def _mean_frontier(means, signs, materiality):
    """평균 행렬(시나리오×목적)만으로 지배 여부 판정 — 부트스트랩 내부용."""
    v = means * signs                                 # 클수록 좋게 정렬
    n = len(v)
    keep = np.ones(n, dtype=bool)
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            scale = np.maximum(np.maximum(np.abs(v[i]), np.abs(v[j])), 1e-9)
            rel = (v[i] - v[j]) / scale
            if (rel >= -materiality).all() and (rel > materiality).any():
                keep[j] = False
                break
    return keep


def frontier_stability(results, objectives=DEFAULT_OBJECTIVES,
                       materiality=DEFAULT_MATERIALITY, n_boot=400, seed=7):
    """후보 판정 안정성: 시뮬레이션 반복을 복원추출해 다시 판정했을 때 후보로 남는 비율.
    1.0에 가까우면 '반복을 더 돌려도 판정이 안 바뀐다', 0.5 근처면 '경계선에 있다'는 뜻."""
    rng = np.random.default_rng(seed)
    signs = np.array([OBJECTIVES[k]["sign"] for k in objectives], float)
    data = np.stack([np.stack([np.asarray(r["runs"][k], float) for k in objectives], axis=1)
                     for r in results])               # (시나리오, 반복, 목적)
    n_runs = data.shape[1]
    hits = np.zeros(len(results))
    for _ in range(n_boot):
        idx = rng.integers(0, n_runs, n_runs)         # 모든 시나리오에 같은 인덱스(대응 유지)
        hits += _mean_frontier(data[:, idx, :].mean(axis=1), signs, materiality)
    return {r["summary"]["scenario"]: float(h / n_boot) for r, h in zip(results, hits)}


# ── 회사 조건(제약)과 우선순위 반영 ─────────────────────────────────────────
def apply_constraints(results, baseline_name, max_cost_increase=None,
                      max_hire_drop=None, max_wait_increase=None):
    """기준 시나리오 대비 허용 한도를 넘는 시나리오를 걸러낸다.
    한도는 None이면 미적용. 반환: {이름: [위반 사유, ...]} (빈 리스트 = 통과)."""
    base = next(r["summary"] for r in results if r["summary"]["scenario"] == baseline_name)
    out = {}
    for r in results:
        s, bad = r["summary"], []
        if max_cost_increase is not None:
            inc = s["cost_npv_mean"] / base["cost_npv_mean"] - 1
            if inc > max_cost_increase + 1e-12:
                bad.append(f"인건비 {inc*100:+.1f}% (한도 +{max_cost_increase*100:.1f}%)")
        if max_hire_drop is not None:
            drop = 1 - s["total_hires_mean"] / max(base["total_hires_mean"], 1e-9)
            if drop > max_hire_drop + 1e-12:
                bad.append(f"신규채용 {-drop*100:+.1f}% (한도 -{max_hire_drop*100:.0f}%)")
        if max_wait_increase is not None:
            inc = s["promo_congestion_mean"] - base["promo_congestion_mean"]
            if inc > max_wait_increase + 1e-12:
                bad.append(f"승진 대기 {inc:+.1f}년 (한도 +{max_wait_increase:.1f}년)")
        out[s["scenario"]] = bad
    return out


def priority_scores(results, weights, names=None):
    """가중 후회(regret) 점수. 낮을수록 회사 우선순위에 잘 맞는다(0 = 모든 목적에서 최선).

    각 목적에서 '가장 좋은 시나리오와의 격차'를 0~1로 정규화하고 중요도로 가중 평균한다.
    정규화 범위는 비교 대상 집합에 따라 달라지므로, 이 점수는 '이 집합 안에서의 상대
    순위'이지 절대 평가가 아니다."""
    pool = [r for r in results if names is None or r["summary"]["scenario"] in names]
    keys = [k for k, w in weights.items() if w > 0]
    if not pool or not keys:
        return {}
    wsum = float(sum(weights[k] for k in keys))
    score = {r["summary"]["scenario"]: 0.0 for r in pool}
    parts = {r["summary"]["scenario"]: {} for r in pool}
    for k in keys:
        sign = OBJECTIVES[k]["sign"]
        vals = np.array([sign * r["summary"][f"{k}_mean"] for r in pool])
        best, worst = vals.max(), vals.min()
        span = best - worst
        for r, v in zip(pool, vals):
            reg = 0.0 if span <= 1e-12 else float((best - v) / span)
            name = r["summary"]["scenario"]
            parts[name][k] = reg
            score[name] += weights[k] / wsum * reg
    return {n: dict(score=score[n], parts=parts[n]) for n in score}


def minimax_regret(score_by_state):
    """상태(가정 조합)별 점수 dict → 시나리오별 최대 후회.
    score_by_state: {state_label: {scenario: score}}. 후회 = 그 상태에서 최선과의 점수 차."""
    regret = {}
    for scores in score_by_state.values():
        if not scores:
            continue
        best = min(scores.values())
        for name, s in scores.items():
            regret.setdefault(name, []).append(s - best)
    return {n: dict(max_regret=float(max(v)), mean_regret=float(np.mean(v)),
                    n_best=int(sum(1 for x in v if x <= 1e-12)), n_states=len(v))
            for n, v in regret.items()}


# ── 숙련 손실까지 넣은 손익분기 ─────────────────────────────────────────────
def break_even(a, b, n_boot=500, seed=0):
    """시나리오 a를 기준 b와 '인건비 + λ × 숙련 손실 가치'로 비교할 때, a가 더 나아지는 λ의 경계.

    λ = 숙련 공백 1인-년의 가치를 '그 직군 평균 인건비 1년'의 몇 배로 보느냐(1.0 = 100%).
    a가 나은 조건: ΔC + λ·ΔV < 0  (ΔC = 인건비 차이, ΔV = 숙련 손실 가치 차이, 둘 다 a − b)

    반환 kind
      always  : λ와 무관하게 a가 낫다(인건비도 적고 숙련 손실도 적음)
      never   : λ와 무관하게 b가 낫다
      above   : λ ≥ x 이면 a가 낫다(인건비는 더 들지만 숙련 손실을 줄이는 안)
      below   : λ ≤ x 이면 a가 낫다(인건비는 줄지만 숙련 손실이 늘어나는 안)
    x의 95% 구간은 반복(대응표본)을 다시 뽑는 부트스트랩으로 구한다."""
    ca, cb = np.asarray(a["runs"]["cost_npv"], float), np.asarray(b["runs"]["cost_npv"], float)
    va, vb = (np.asarray(a["runs"]["skill_value_npv"], float),
              np.asarray(b["runs"]["skill_value_npv"], float))
    dc, dv = ca - cb, va - vb
    mc, mv = float(dc.mean()), float(dv.mean())
    tol = 1e-9 * max(abs(float(cb.mean())), 1.0)
    out = dict(dc=mc, dv=mv, x=None, lo=None, hi=None)
    if abs(mv) <= tol:
        out["kind"] = "always" if mc <= 0 else "never"
        return out
    if mv < 0:
        if mc <= 0:
            out["kind"] = "always"
            return out
        out["kind"] = "above"
    else:
        if mc >= 0:
            out["kind"] = "never"
            return out
        out["kind"] = "below"
    out["x"] = -mc / mv
    rng = np.random.default_rng(seed)
    n = len(dc)
    if n > 1:
        idx = rng.integers(0, n, size=(n_boot, n))
        bc, bv = dc[idx].mean(axis=1), dv[idx].mean(axis=1)
        ok = np.sign(bv) == np.sign(mv)
        xs = -bc[ok] / bv[ok]
        if len(xs) > n_boot * 0.5:
            out["lo"], out["hi"] = float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))
    return out
