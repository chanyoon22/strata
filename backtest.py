"""
backtest.py — 과거 명부로 모형 검증하기
====================================================================
"이 모형이 우리 회사에 맞는가?"에 답하는 가장 직접적인 방법은 과거에서 출발해 현재를
맞혀보는 것이다. 몇 년 전 명부를 넣고 '제도 변화 없음'으로 오늘까지 흘려보낸 뒤,
모형이 예측한 오늘의 인력 구조를 실제 오늘 명부와 비교한다.

모형이 쓸모 있으려면 아무 계산도 안 하는 단순한 예측보다는 나아야 한다. 그래서 두 가지
기준선과 함께 본다.
  · 변화 없음   과거의 연령 구성이 오늘도 그대로라고 가정
  · 나이만 먹음 과거 인원이 한 명도 안 나가고 안 들어온 채 나이만 먹었다고 가정
지표는 '연령대별 구성비가 실제와 평균 몇 %p 어긋나는가'(낮을수록 좋음)다.
"""
from __future__ import annotations
import dataclasses
import numpy as np
import pandas as pd
from engine_retirement_v2 import Scenario, simulate

BANDS = [(0, 29, "29세 이하"), (30, 39, "30대"), (40, 49, "40대"), (50, 54, "50~54세"),
         (55, 59, "55~59세"), (60, 99, "60세 이상")]


def band_share(ages):
    a = np.asarray(ages, float)
    n = max(len(a), 1)
    return np.array([((a >= lo) & (a <= hi)).sum() / n for lo, hi, _ in BANDS])


def run_backtest(profile_past, now_std: pd.DataFrame, years: int, runs=40, seed=11):
    """profile_past: 과거 명부로 만든 프로파일(start_year = 과거 연도).
    now_std: 현재 표준 명부. years: 두 명부 사이의 햇수."""
    if years < 1:
        raise ValueError("과거 명부의 기준 연도가 현재보다 앞서야 합니다.")
    base = Scenario("현행 유지(검증용)")
    shares, avg_age, n55, hires, ce = [], [], [], [], []
    for r in range(runs):
        rows, wf = simulate(profile_past, base, horizon=years, seed=seed + r, return_state=True)
        shares.append(band_share(wf["age"]))
        avg_age.append(float(wf["age"].mean()))
        n55.append(float((wf["age"] >= 55).mean()))
        hires.append(float(((wf["tenure"] < years) & (wf["status"] == 0)).sum()))   # 그 사이 들어와 아직 남은 사람
        ce.append(float((wf["status"] == 1).mean()))
    pred = np.mean(shares, axis=0)
    lo, hi = np.percentile(shares, 5, axis=0), np.percentile(shares, 95, axis=0)
    past_ages = profile_past.roster["age"].to_numpy(float)
    actual = band_share(now_std["age"])
    naive_same = band_share(past_ages)
    naive_age = band_share(past_ages + years)
    mae = lambda p: float(np.abs(p - actual).mean() * 100)      # %p
    actual_hires = int(((now_std["tenure"] < years) & (now_std["status"] == 0)).sum())
    table = pd.DataFrame({"연령대": [b[2] for b in BANDS], "과거": naive_same * 100, "모형 예측": pred * 100,
                          "예측 하한": lo * 100, "예측 상한": hi * 100, "실제(현재)": actual * 100})
    return dict(
        table=table, years=years, runs=runs,
        mae_model=mae(pred), mae_same=mae(naive_same), mae_aging=mae(naive_age),
        avg_age=dict(pred=float(np.mean(avg_age)), actual=float(now_std["age"].mean()),
                     past=float(past_ages.mean())),
        share55=dict(pred=float(np.mean(n55)), actual=float((now_std["age"] >= 55).mean()),
                     past=float((past_ages >= 55).mean())),
        hires=dict(pred=float(np.mean(hires)), actual=actual_hires),
        ce_share=dict(pred=float(np.mean(ce)), actual=float((now_std["status"] == 1).mean())),
        n_past=len(past_ages), n_now=len(now_std))


def verdict(bt) -> tuple[str, str]:
    """(등급, 설명). 등급: good / fair / poor."""
    m, s, a = bt["mae_model"], bt["mae_same"], bt["mae_aging"]
    best_naive = min(s, a)
    skill = 1 - m / best_naive if best_naive > 0 else 0.0
    if m <= 1.5 and skill > 0:
        g = "good"
        t = (f"모형이 예측한 연령 구성은 실제와 평균 <b>{m:.1f}%p</b> 차이로, <b>잘 맞습니다.</b> "
             f"단순한 예측(가장 나은 것이 {best_naive:.1f}%p)보다 오차가 <b>{skill*100:.0f}%</b> 작습니다.")
    elif skill > 0.15:
        g = "fair"
        t = (f"모형 예측은 실제와 평균 <b>{m:.1f}%p</b> 차이입니다. 단순한 예측({best_naive:.1f}%p)보다는 "
             f"<b>{skill*100:.0f}%</b> 정확하지만 오차가 작지는 않습니다. 퇴사율·신규 입사자 가정을 다시 확인해 보세요.")
    else:
        g = "poor"
        t = (f"모형 예측(오차 {m:.1f}%p)이 단순한 예측({best_naive:.1f}%p)보다 <b>낫지 않습니다.</b> "
             "이 기간에 구조조정·대규모 채용처럼 모형에 없는 일이 있었거나, 퇴사율 가정이 실제와 크게 다를 수 있습니다. "
             "이 상태로는 앞으로의 예측도 참고용으로만 보세요.")
    return g, t
