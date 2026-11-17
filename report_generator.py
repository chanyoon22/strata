"""
report_generator.py — 경영진 제출용 PDF 리포트
결정론적 수치 + 해석문(llm_report 산출)을 하나의 문서로 조립한다.
v3: 회사 데이터 요약, 현행 대비 변화, 제외 사유, 판정 안정성, 견고성 결과를 담는다.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether)
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

import interpret as I

# ── 한글 글꼴 ──────────────────────────────────────────────────────────────
# PDF에 글꼴을 '심어야' 어느 PC·뷰어에서나 글자가 나온다. 예전에는 reportlab 내장 CID 글꼴
# (HYGothic)을 썼는데, 이 방식은 글꼴을 PDF에 넣지 않고 뷰어가 한글 글꼴 팩을 갖고 있기를
# 기대한다 — 일부 뷰어·리눅스 환경에서는 본문이 통째로 빈칸으로 나왔다. 차트(matplotlib)도
# Noto Sans CJK만 찾아 윈도우 PC에서는 네모로 깨졌다.
# 이제 트루타입 글꼴을 다음 순서로 찾아 PDF와 차트에 함께 쓴다:
#   ① koreanize-matplotlib 패키지에 든 나눔고딕(requirements.txt로 설치됨, OFL 라이선스)
#   ② 윈도우의 맑은 고딕  ③ 시스템에 깔린 나눔고딕
# 하나도 없을 때만 예전 방식(CID 글꼴)으로 내려간다.
from importlib.util import find_spec
from reportlab.pdfbase.ttfonts import TTFont


def _find_ttf():
    """(일반체 경로, 굵은체 경로 또는 None)"""
    try:
        spec = find_spec("koreanize_matplotlib")
        if spec and spec.origin:
            d = Path(spec.origin).parent / "fonts"
            if (d / "NanumGothic.ttf").exists():
                bold = d / "NanumGothicBold.ttf"
                return str(d / "NanumGothic.ttf"), (str(bold) if bold.exists() else None)
    except Exception:
        pass
    win = Path("C:/Windows/Fonts")
    if (win / "malgun.ttf").exists():
        bold = win / "malgunbd.ttf"
        return str(win / "malgun.ttf"), (str(bold) if bold.exists() else None)
    for f in font_manager.findSystemFonts():
        if Path(f).name.lower() == "nanumgothic.ttf":
            bold = Path(f).with_name("NanumGothicBold.ttf")
            return f, (str(bold) if bold.exists() else None)
    return None, None


_TTF, _TTF_BOLD = _find_ttf()
EMBEDDED = False
BODY_FONT, BOLD_FONT, SMALL_FONT = "HYGothic-Medium", "HYGothic-Medium", "HYSMyeongJo-Medium"
KOREAN_FONT = None
if _TTF:
    try:
        pdfmetrics.registerFont(TTFont("STRATA-KR", _TTF))
        pdfmetrics.registerFont(TTFont("STRATA-KR-Bold", _TTF_BOLD or _TTF))
        pdfmetrics.registerFontFamily("STRATA-KR", normal="STRATA-KR", bold="STRATA-KR-Bold",
                                      italic="STRATA-KR", boldItalic="STRATA-KR-Bold")
        BODY_FONT, BOLD_FONT, SMALL_FONT = "STRATA-KR", "STRATA-KR-Bold", "STRATA-KR"
        EMBEDDED = True
        font_manager.fontManager.addfont(_TTF)
        KOREAN_FONT = font_manager.FontProperties(fname=_TTF).get_name()
    except Exception:
        EMBEDDED = False
if not EMBEDDED:
    pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
    for f in font_manager.findSystemFonts():
        if any(k in f for k in ("NotoSansCJK", "Nanum", "malgun", "Malgun", "AppleGothic", "AppleSDGothic")):
            try:
                font_manager.fontManager.addfont(f)
            except Exception:
                pass
    have = {f.name for f in font_manager.fontManager.ttflist}
    KOREAN_FONT = next((c for c in ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",
                                    "Noto Sans CJK JP", "AppleGothic", "Apple SD Gothic Neo"] if c in have), None)
if KOREAN_FONT:
    plt.rcParams["font.family"] = KOREAN_FONT
plt.rcParams["axes.unicode_minus"] = False

KO = ParagraphStyle("ko", fontName=BODY_FONT, fontSize=9.5, leading=15)
KO_TITLE = ParagraphStyle("kot", fontName=BOLD_FONT, fontSize=17, leading=24, spaceAfter=8)
KO_H = ParagraphStyle("koh", fontName=BOLD_FONT, fontSize=12.5, leading=18,
                      spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#146B5C"))
KO_SMALL = ParagraphStyle("kos", fontName=SMALL_FONT, fontSize=8, leading=12,
                          textColor=colors.HexColor("#555555"))
SERIES = ["#17242F", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _lab(name):
    # 차트 글꼴이 없으면 한글 라벨 대신 코드(S1, S2…)만 쓴다 — 네모로 깨지는 것보다 낫다
    return name.split("(")[0].strip() if KOREAN_FONT else name.split(" ")[0]


def _t(ko, en):
    return ko if KOREAN_FONT else en


def _charts(run_output, outdir="."):
    paths = {}
    res = run_output["results"]
    frontier = set(run_output["frontier"])
    cyc = [SERIES[i] if i < len(SERIES) else "#9aa5ae" for i in range(len(res))]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, (r, c) in enumerate(zip(res, cyc)):
        p = r["path"]; yrs = [x["year"] for x in p]
        ls = "--" if i == 0 else ("-" if r["summary"]["scenario"] in frontier else ":")
        axes[0].plot(yrs, [x["labor_cost"] / 1e4 for x in p], color=c, lw=1.8, ls=ls, label=_lab(r["summary"]["scenario"]))
        axes[1].plot(yrs, [x["share_55plus"] * 100 for x in p], color=c, lw=1.8, ls=ls)
        axes[2].plot(yrs, [x["n_hire"] for x in p], color=c, lw=1.8, ls=ls)
    axes[0].set_title(_t("연간 인건비 (억원)", "Annual labor cost (100M KRW)"))
    axes[1].set_title(_t("55세 이상 비중 (%)", "Share aged 55+ (%)"))
    axes[2].set_title(_t("연간 신규채용 (명)", "New hires per year"))
    axes[0].legend(fontsize=7, loc="best")
    for ax in axes:
        ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    paths["paths"] = f"{outdir}/chart_paths.png"
    fig.savefig(paths["paths"], dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for r, c in zip(res, cyc):
        s = r["summary"]
        onf = s["scenario"] in frontier
        ax.scatter(s["cost_npv_mean"] / 1e4, s["total_hires_mean"], s=120 if onf else 55,
                   color=c if onf else "#c9d1d7", edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(s["scenario"].split(" ")[0], (s["cost_npv_mean"] / 1e4, s["total_hires_mean"]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
    h = res[0]["summary"]["horizon"]
    ax.set_xlabel(_t(f"{h}년 인건비 현재가치 (억원)", f"{h}-year labor cost NPV (100M KRW)"))
    ax.set_ylabel(_t(f"{h}년 신규채용 누계 (명)", f"{h}-year total new hires"))
    ax.set_title(_t("비용과 채용 — 큰 점이 후보", "Cost vs hiring — large dots are candidates"))
    ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    paths["pareto"] = f"{outdir}/chart_pareto.png"
    fig.savefig(paths["pareto"], dpi=140); plt.close(fig)
    return paths


def _table(rows, widths, head_bg="#146B5C"):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), BODY_FONT), ("FONTNAME", (0, 0), (-1, 0), BOLD_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(head_bg)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d1d7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8F9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


def build_pdf(run_output, sensitivity_output, prose, profile_desc, params_meta,
              out_path="executive_report.pdf", outdir=None, robustness_output=None, data_summary=None,
              profile_job_default=True):
    ch = _charts(run_output, outdir=outdir or str(Path(out_path).parent))
    res = run_output["results"]
    h = res[0]["summary"]["horizon"]
    base = res[0]["summary"]["scenario"]
    d = I.deltas(res, base)
    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    story = [Paragraph("정년연장·계속고용 시나리오 인력구조 의사결정 지원 리포트", KO_TITLE),
             Paragraph("STRATA | 실제 인사 명부 기반 몬테카를로 마이크로시뮬레이션 · 통계적 지배 필터", KO_SMALL),
             Spacer(1, 6), Paragraph(f"<b>분석 대상:</b> {profile_desc}", KO)]
    if data_summary:
        story.append(Paragraph(data_summary, KO))
    story.append(Spacer(1, 4))

    story.append(Paragraph("1. 핵심 요약", KO_H))
    for line in prose.split("\n"):
        story.append(Paragraph(line, KO))
    story.append(Spacer(1, 6))

    story.append(Paragraph(f"2. 시나리오 비교 ({h}년, {res[0]['summary']['runs']}회 반복)", KO_H))
    stab = run_output.get("stability", {})
    rows = [["시나리오", "판정", "인건비(억)", "현행 대비", "신규채용", "현행 대비", "승진 대기", "숙련 공백", "안정성"]]
    for r in res:
        s = r["summary"]; n = s["scenario"]
        rows.append([n.split("(")[0].strip(), "후보" if n in run_output["frontier"] else "제외",
                     f"{s['cost_npv_mean']/1e4:,.0f}", f"{d[n]['cost']*100:+.2f}%",
                     f"{s['total_hires_mean']:,.0f}", f"{d[n]['hires']*100:+.1f}%",
                     f"{s['promo_congestion_mean']:.1f}년", f"{s.get('skill_gap_mean', float('nan')):,.0f}",
                     f"{stab[n]*100:.0f}%" if n in stab else "-"])
    story.append(_table(rows, [44 * mm, 12 * mm, 19 * mm, 17 * mm, 17 * mm, 17 * mm, 17 * mm, 17 * mm, 14 * mm]))
    story.append(Paragraph("후보 = 다른 어떤 시나리오에도 모든 항목에서 밀리지 않는 시나리오. 판정은 같은 난수로 돌린 "
                           f"반복 계산의 대응표본 95% 신뢰구간과 실질 차이 {run_output.get('materiality', 0.005)*100:.1f}% 기준을 함께 씁니다. "
                           "안정성 = 반복을 다시 뽑아 판정했을 때 후보로 남는 비율.", KO_SMALL))
    why = run_output.get("why", {})
    if why:
        story.append(Spacer(1, 4))
        for n in run_output["dominated"]:
            story.append(Paragraph(f"· <b>{n.split('(')[0].strip()}</b> 제외 사유 — {I.why_text(n, why)}", KO))

    story.append(KeepTogether([Paragraph(f"3. {h}년 경로", KO_H),
                               Image(ch["paths"], width=172 * mm, height=47 * mm),
                               Paragraph("점선 = 현행 유지, 실선 = 후보, 가는 점선 = 제외", KO_SMALL)]))
    story.append(KeepTogether([Paragraph("4. 비용과 채용", KO_H),
                               Image(ch["pareto"], width=115 * mm, height=74 * mm)]))

    if len(res) > 1 and "skill_value_npv" in res[0]["runs"]:
        import decision as D
        story.append(Paragraph("숙련 손실까지 보면", KO_H))
        story.append(Paragraph("숙련 공백 = 사람이 나간 자리가 다시 제 몫을 할 때까지의 손실(충원 기간 + 후임 적응기의 부족분, 인-년). "
                               "숙련 공백 1인-년을 평균 인건비 1년의 몇 %로 볼 때 판단이 바뀌는지(손익분기)를 함께 적습니다.", KO_SMALL))
        for r in res[1:]:
            n = r["summary"]["scenario"]
            txt = I.break_even_text(n, D.break_even(r, res[0]), h).replace("<br>", " ")
            story.append(Paragraph("· " + txt, KO))
        if profile_job_default:
            story.append(Paragraph("직군별 충원·숙련 기간을 입력하지 않아 모든 직군에 같은 기본값(충원 3개월·숙련 2년)을 썼습니다.", KO_SMALL))

    story.append(Paragraph("5. 결론이 흔들리는가", KO_H))
    if robustness_output:
        n_st = robustness_output["n_states"]
        story.append(Paragraph(f"50세 이후 임금(3) × 퇴사율(3) × 채용 방식(3) = {n_st}가지 상황에서 다시 판정한 결과입니다. "
                               "'1위 횟수'는 회사가 정한 중요도 기준으로 그 상황에서 가장 잘 맞은 횟수, '최악의 경우 아쉬움'은 "
                               "어느 상황에서든 1위와 가장 크게 벌어진 점수 차(작을수록 안전)입니다.", KO))
        story.append(Spacer(1, 4))
        rows = [["시나리오", f"1위 횟수(/{n_st})", f"후보로 남은 상황(/{n_st})", "최악의 경우 아쉬움"]]
        reg = robustness_output.get("regret", {})
        for n, v in sorted(robustness_output["frontier_share"].items(),
                           key=lambda kv: -reg.get(kv[0], {}).get("n_best", 0)):
            rows.append([n.split("(")[0].strip(), f"{reg[n]['n_best']}" if n in reg else "-", f"{v*n_st:.0f}",
                         f"{reg[n]['max_regret']*100:.1f}" if n in reg else "-"])
        story.append(_table(rows, [68 * mm, 32 * mm, 38 * mm, 34 * mm]))
        story.append(Spacer(1, 6))
    fp = sensitivity_output.get("flip_points") or []
    if sensitivity_output.get("grid"):
        story.append(Paragraph(
            "재고용 수락률·임금률은 미리 알기 가장 어려운 값입니다. KLIPS 실측 앵커(수락률 0.37~0.62, "
            "임금비 0.51~0.59) 주변 25가지 조합에서 후보 구성이 달라지는 경우:", KO))
        if fp:
            groups = {}
            for a, w, f in fp:
                groups.setdefault(tuple(f), []).append((a, w))
            rows = [["그 조건에서의 후보", "조합 수", "수락률 범위", "임금률 범위"]]
            for f, pts in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]:
                acc = [a for a, _ in pts]; wg = [w for _, w in pts]
                rows.append([", ".join(x.split(" ")[0] for x in f), str(len(pts)),
                             f"{min(acc):.2f}~{max(acc):.2f}", f"{min(wg):.2f}~{max(wg):.2f}"])
            story.append(_table(rows, [86 * mm, 18 * mm, 33 * mm, 33 * mm], head_bg="#5D6B77"))
        else:
            story.append(Paragraph("25가지 조합 모두에서 후보 구성이 바뀌지 않았습니다 — 재고용 조건에 대해 결론이 견고합니다.", KO))
    elif not robustness_output:
        story.append(Paragraph("견고성 분석을 실행하지 않았습니다. 앱의 ⑤ 단계에서 실행하면 이 장에 포함됩니다.", KO))

    story.append(Paragraph("6. 방법·데이터·한계", KO_H))
    for line in params_meta:
        story.append(Paragraph("· " + line, KO))
    doc.build(story)
    return out_path
