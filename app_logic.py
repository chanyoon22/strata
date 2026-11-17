"""app_logic.py — 화면과 엔진 사이: 설정값 → 프로파일·시나리오 → 실행, 설정 저장/불러오기."""
from __future__ import annotations
import hashlib
import json
import numpy as np

import company_data as C
import decision as D
from dss_layer import (preset_scenarios, scenario_from_dict, scenario_to_dict, run_all,
                       build_schedule, DEFAULT_PRESET_CODES)
from engine_retirement_v2 import ENGINE_VERSION

SRC_DEMO, SRC_UPLOAD, SRC_BANDS = "표준 표본으로 체험", "우리 회사 명부 올리기", "연령대 집계표로 입력"
STEPS = ["① 회사 데이터", "② 가정·시나리오", "③ 결과", "④ 우리 회사에 맞는 선택", "⑤ 결론이 흔들리는가", "⑥ 리포트"]
CUSTOM_BANDS = [(0, 29), (30, 39), (40, 49), (50, 54), (55, 59), (60, 99)]
CUSTOM_BAND_LABELS = ["29세 이하", "30대", "40대", "50~54세", "55~59세", "60세 이상"]
RUN_LEVELS = {50: "빠름", 100: "표준", 200: "정밀"}

DEFAULT_CFG = dict(
    src=SRC_DEMO, demo_n=1000, demo_wage=465, demo_ce=False,
    start_year=2026, ret_age=60,
    push=1.5, other=4.5,                 # %/년
    accept=0.62, wage_ratio=0.55,
    horizon=15, runs=100, hiring="replace", regime="observed",
    custom_rates=[5.5, 4.0, 1.2, 0.0, 0.0, 0.0],      # %/년, CUSTOM_BANDS 순서
    ce_rate=0.0, ce_end=60, ce_ratio=0.55,
    presets=list(DEFAULT_PRESET_CODES), custom=[],
    objectives=list(D.DEFAULT_OBJECTIVES), materiality=0.5,     # %
    weights={"cost_npv": 2, "total_hires": 2, "promo_congestion": 1, "exp55_person_years": 0,
                 "income_gap": 0, "skill_gap": 1},
    lim_cost=None, lim_hire=None, lim_wait=None,
    jobs={},          # {직군: {vacancy_months, ttp_years, ce_accept_rel, ce_wage_rel}} — 비우면 기본값
)
SETTING_KEYS = [k for k in DEFAULT_CFG if k not in ("src",)]


def new_cfg():
    return json.loads(json.dumps(DEFAULT_CFG))


def custom_rates(cfg):
    return [(lo, hi, r / 100.0) for (lo, hi), r in zip(CUSTOM_BANDS, cfg["custom_rates"])]


def make_profile(cfg, data):
    """현재 설정과 데이터로 엔진용 CompanyProfile을 만든다."""
    rates = custom_rates(cfg) if cfg["regime"] == "custom" else None
    push, other = cfg["push"] / 100.0, cfg["other"] / 100.0
    if data["mode"] == SRC_DEMO:
        p = C.demo_profile(int(cfg["demo_n"]), float(cfg["demo_wage"]), push, other, cfg["regime"],
                           with_ce_practice=bool(cfg["demo_ce"]), start_year=int(cfg["start_year"]))
        p.custom_wage_rates = rates
        p.job_table = C.job_table_for(p.dept_names, cfg.get("jobs"))
        return p
    ce = dict(rate=float(cfg["ce_rate"]), end_age=int(cfg["ce_end"]), wage_ratio=float(cfg["ce_ratio"]))
    lv = data.get("leaver")
    return C.build_profile(data["std"], data.get("grade_table"), retirement_age=int(cfg["ret_age"]),
                           base_push_turnover=push, other_exit_rate=other, wage_regime=cfg["regime"],
                           custom_wage_rates=rates, ce_practice=ce, start_year=int(cfg["start_year"]),
                           name=data.get("name", "우리 회사"),
                           hazard_offsets=(lv or {}).get("offsets"),
                           user_jobs=cfg.get("jobs"), dept_offsets=(lv or {}).get("dept_offsets"))


def dept_names_for(cfg, data):
    """직군 설정 표에 보여줄 직군 목록(엔진과 같은 묶음 규칙)과 정규직 인원."""
    if not data or data["mode"] == SRC_DEMO or data.get("std") is None:
        return ["전체"], [int(cfg["demo_n"])]
    std, names = C.group_depts(data["std"])
    reg = std[std["status"] == 0]
    return names, [int((reg["dept_name"] == d).sum()) for d in names]


def make_scenarios(cfg):
    """선택한 프리셋 + 직접 만든 시나리오. 맨 앞은 항상 기준선(S1 현행 유지)."""
    codes = ["S1"] + [c for c in cfg["presets"] if c != "S1"]
    scens = preset_scenarios(cfg["wage_ratio"], cfg["accept"], hiring=cfg["hiring"], codes=codes)
    for spec in cfg["custom"]:
        d = dict(spec)
        d["hiring_mode"] = cfg["hiring"]
        if not d.get("ce_fixed"):
            d["rehire_accept"], d["rehire_wage_ratio"] = cfg["accept"], cfg["wage_ratio"]
            if d.get("continued_type") == "job_keep":
                d["rehire_wage_ratio"] = max(cfg["wage_ratio"], 0.85)
        scens.append(scenario_from_dict(d))
    return scens


def custom_spec(n, name, target_age, first_year, step, ce_type, ce_mode, ce_n, ce_year,
                own_ce, accept, ratio, peak, peak_years, peak_cut, reform, reform_regime, reform_year,
                erp, erp_start, erp_end, erp_min, erp_max, erp_take, erp_months, edu, start_year=2026,
                base_age=60):
    """빌더 입력값 → 시나리오 사양(dict). 잘못된 조합이면 ValueError."""
    sched = build_schedule(first_year, step, target_age, base_age=base_age, base_year=start_year)
    ce_sched = {}
    if ce_type != "none":
        if ce_mode == "plus":
            ce_sched = {y: a + int(ce_n) for y, a in sched.items()}
        else:
            if int(ce_n) <= base_age:
                raise ValueError("계속고용 상한 연령은 현재 정년보다 높아야 합니다.")
            ce_sched = {start_year: base_age, int(ce_year): int(ce_n)}
    title = name.strip() or "직접 만든 안"
    spec = dict(name=f"C{n} {title}", retirement_schedule=sched, continued_type=ce_type,
                ce_end_age_schedule=ce_sched, ce_fixed=bool(own_ce and ce_type != "none"),
                rehire_accept=float(accept), rehire_wage_ratio=float(ratio),
                wage_peak=bool(peak), peak_years=int(peak_years), peak_cut=float(peak_cut),
                wage_reform_regime=reform_regime if reform else None,
                wage_reform_year=int(reform_year) if reform else 0,
                erp_start=int(erp_start) if erp else 0, erp_end=int(erp_end) if erp else 0,
                erp_age_min=int(erp_min), erp_age_max=int(erp_max),
                erp_take_rate=float(erp_take) if erp else 0.0, erp_severance_months=float(erp_months),
                edu_invest_per_head=float(edu), desc="직접 만든 시나리오", source="사용자 정의")
    spec = scenario_to_dict(scenario_from_dict(spec))      # 검증 + 직렬화 가능한 형태로
    return spec


def signature(cfg, data):
    """결과가 어떤 설정에서 나왔는지 식별(설정이 바뀌면 결과가 낡았음을 알리기 위해)."""
    keys = ["demo_n", "demo_wage", "demo_ce", "start_year", "ret_age", "push", "other", "accept",
            "wage_ratio", "horizon", "runs", "hiring", "regime", "custom_rates", "ce_rate", "ce_end",
            "ce_ratio", "presets", "custom", "jobs"]
    body = {k: cfg[k] for k in keys}
    std = data.get("std")
    body["data"] = [data["mode"], data.get("name"), 0 if std is None else len(std),
                    0 if std is None else float(std["wage"].sum()), bool(data.get("leaver"))]
    gt = data.get("grade_table")
    body["grades"] = None if gt is None else gt[["직급", "서열", "최소체류연수"]].values.tolist()
    return hashlib.md5(json.dumps(body, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def run_analysis(cfg, data, progress=None):
    profile = make_profile(cfg, data)
    scens = make_scenarios(cfg)
    out = run_all(profile, runs=int(cfg["runs"]), horizon=int(cfg["horizon"]), scenarios=scens,
                  objectives=cfg["objectives"], materiality=cfg["materiality"] / 100.0,
                  detail=len(profile.dept_names) > 1, progress=progress)
    return dict(out=out, profile=profile, scenarios=scens, baseline=scens[0].name,
                sig=signature(cfg, data), horizon=int(cfg["horizon"]), runs=int(cfg["runs"]),
                engine=ENGINE_VERSION)


def rejudge(res, objectives, materiality_pct):
    """시뮬레이션을 다시 돌리지 않고 판정 기준만 바꿔 후보를 다시 가린다."""
    results = res["out"]["results"]
    m = materiality_pct / 100.0
    frontier, dominated, why = D.statistical_frontier(results, objectives, m)
    return dict(frontier=frontier, dominated=dominated, why=why,
                stability=D.frontier_stability(results, objectives, m, n_boot=300))


def horizon_warnings(scens, start_year, horizon):
    end = start_year + horizon - 1
    out = []
    for s in scens:
        last = max(list(s.retirement_schedule) + list(s.ce_end_age_schedule or {start_year: 0}))
        if last > end:
            out.append((s.name, last))
    return end, out


# ── 설정 저장·불러오기 ──────────────────────────────────────────────────────
def _jobs_entered(cfg):
    from engine_retirement_v2 import JOB_DEFAULTS
    for v in (cfg.get("jobs") or {}).values():
        for k in ("vacancy_months", "ttp_years"):
            if v.get(k) is not None and abs(float(v[k]) - JOB_DEFAULTS[k]) > 1e-9:
                return True
    return False


def export_settings(cfg) -> bytes:
    body = dict(app="STRATA", engine=ENGINE_VERSION, settings={k: cfg[k] for k in SETTING_KEYS})
    return json.dumps(body, ensure_ascii=False, indent=1).encode("utf-8")


def import_settings(cfg, raw: bytes):
    body = json.loads(raw.decode("utf-8-sig"))
    if body.get("app") != "STRATA" or "settings" not in body:
        raise ValueError("STRATA 설정 파일이 아닙니다.")
    for spec in body["settings"].get("custom", []):
        scenario_from_dict(spec)                       # 잘못된 시나리오면 여기서 오류
    for k, v in body["settings"].items():
        if k in DEFAULT_CFG and k != "src":
            cfg[k] = v
    return cfg


# ── 한계(저자 텍스트: 화면·엑셀·PDF에 그대로 싣는다) ───────────────────────
def limitations(cfg, data, res=None):
    from dss_layer import HIRING_POLICIES
    from engine_retirement_v2 import WAGE_REGIMES
    regime = "자사 입력 승급률" if cfg["regime"] == "custom" else WAGE_REGIMES[cfg["regime"]]["label"]
    out = []
    if data and data["mode"] == SRC_DEMO:
        out.append("표준 표본은 여러 회사의 근로자를 모은 것이지 실제 한 회사가 아닙니다. 우리 회사 결과를 보려면 명부를 올려야 합니다.")
    if data and data["mode"] == SRC_BANDS:
        out.append("집계표로 만든 가상 명부는 연령대 안의 세부 분포를 대기업 표본에서 빌려온 것입니다. 부서·직급별 결과는 볼 수 없고, 개인별 명부보다 정확도가 낮습니다.")
    out += [
        "재고용 수락률은 회사 안에서 직접 관측한 값이 아니라 정년퇴직자의 재취업률(KLIPS)에서 가져온 추정값입니다. 사내 의향 조사로 확인하는 것이 가장 정확합니다.",
        "퇴사 확률 모형은 300인 이상 사업장 근로자 자료로 추정했습니다. 우리 회사만의 퇴사 패턴은 퇴사자 명부를 올린 경우에만, 근속 구간 단위로 반영됩니다."
        + (" (현재 반영됨)" if data and data.get("leaver") else " (현재 미반영)"),
        f"50세 이후 임금은 '{regime}'으로 가정했습니다. 이 가정에 따라 정년연장 비용의 크기가 2배 이상 달라질 수 있습니다.",
        f"채용 방식은 '{HIRING_POLICIES[cfg['hiring']][0]}'로 두었습니다. 회사가 먼저 정해야 하는 전제이며, 바꾸면 후보가 달라질 수 있습니다.",
        "신규 인원은 명부의 최근 입사자(근속 1년 이하)와 같은 모습으로, 최근 입사자가 20명 미만이면 신입 초임으로 들어온다고 가정합니다. 결원이 생긴 부서에 채워지며, 부서 이동·조직 개편은 반영하지 않습니다.",
        "임금피크·임금체계 개편·희망퇴직은 줄어드는 인건비(와 위로금)만 계산합니다. 노사 협의 비용, 사기 저하, 법적 분쟁 위험은 계산에 없어, 이런 수단을 넣은 안이 실제보다 유리하게 나옵니다.",
        "개인별 성과는 측정하지 않습니다. 숙련은 '사람이 나간 자리가 다시 제 몫을 할 때까지의 손실(숙련 공백)'로만 계산하며, 그 직군 평균 임금을 생산성의 대용치로 씁니다. 후임자의 생산성은 숙련 기간 동안 0에서 100%까지 직선으로 오른다고 봅니다."
        + (" 직군별 충원·숙련 기간을 입력하지 않아 모든 직군에 같은 기본값(충원 3개월·숙련 2년)을 썼습니다. 이 경우 숙련 공백은 '나가는 사람 수와 근속'의 차이만 담습니다."
           if not _jobs_entered(cfg) else " 직군별 충원·숙련 기간은 회사가 입력한 값입니다."),
        "숙련 공백을 돈으로 얼마로 볼지는 회사마다 달라 하나로 정하지 않았습니다. 대신 '숙련 공백 1인-년의 가치가 평균 인건비의 몇 % 이상이면 판단이 바뀌는가'(손익분기)를 함께 보여줍니다.",
        "오차 범위는 '누가 언제 그만두는가'의 우연과 퇴사 모형의 추정 오차만 담습니다. 경기 변동, 산업 충격, 법안의 세부 조항(임금 조정 특례 등)은 담지 않습니다.",
        "모든 금액은 지금 물가 기준이며, 할인율 연 3%, 간접 인건비(4대보험·퇴직급여) 19%를 가정했습니다.",
        "분석 기간이 끝난 뒤의 비용·효과는 계산에 없습니다. 기간 막바지에 일정이 끝나는 시나리오는 효과가 일부만 반영됩니다.",
        "2016년 정년 60세 의무화의 효과 분석은 '제도 시행 전부터 두 집단의 추세가 달랐다'는 문제가 있어, 인과 효과가 아니라 방향 참고용으로만 씁니다.",
        "재교육 투자는 비용만 반영합니다. 교육이 생산성·이직에 주는 효과는 검증된 값이 없어 넣지 않았습니다.",
    ]
    return out
