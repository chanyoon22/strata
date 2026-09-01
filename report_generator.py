"""
report_generator.py — 경영진 제출용 PDF 리포트
결정론적 수치 + 해석문(llm_report 산출)을 하나의 문서로 조립.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# 한글 폰트
pdfmetrics.registerFont(UnicodeCIDFont("HYGothic-Medium"))
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

KO = ParagraphStyle("ko", fontName="HYGothic-Medium", fontSize=9.5, leading=15)
KO_TITLE = ParagraphStyle("kot", fontName="HYGothic-Medium", fontSize=18, leading=24,
                          spaceAfter=8)
KO_H = ParagraphStyle("koh", fontName="HYGothic-Medium", fontSize=12.5, leading=18,
                      spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1a3d6e"))
KO_SMALL = ParagraphStyle("kos", fontName="HYSMyeongJo-Medium", fontSize=8, leading=12,
                          textColor=colors.HexColor("#555555"))


def _charts(run_output, outdir="/home/claude/dss"):
    paths = {}
    res = run_output["results"]
    frontier = set(run_output["frontier"])
    cyc = plt.cm.tab10(np.linspace(0, 1, len(res)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for r, c in zip(res, cyc):
        p = r["path"]; yrs = [x["year"] for x in p]
        axes[0].plot(yrs, [x["labor_cost"] / 1e4 for x in p], color=c, lw=1.8,
                     label=r["summary"]["scenario"].split("(")[0])
        axes[1].plot(yrs, [x["share_55plus"] * 100 for x in p], color=c, lw=1.8)
        axes[2].plot(yrs, [x["n_hire"] for x in p], color=c, lw=1.8)
    axes[0].set_title("연간 인건비 (억원)"); axes[1].set_title("55세 이상 비중 (%)")
    axes[2].set_title("연간 신규채용 (명)")
    axes[0].legend(fontsize=7, loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    fig.tight_layout()
    paths["paths"] = f"{outdir}/chart_paths.png"
    fig.savefig(paths["paths"], dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for r, c in zip(res, cyc):
        s = r["summary"]
        onf = s["scenario"] in frontier
        ax.scatter(s["cost_npv_mean"] / 1e4, s["exp55_person_years_mean"],
                   s=110 if onf else 60, color=c, edgecolor="k" if onf else "none",
                   linewidth=1.5, zorder=3)
        ax.annotate(s["scenario"].split(" ")[0], 
                    (s["cost_npv_mean"] / 1e4, s["exp55_person_years_mean"]),
                    fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("10년 인건비 NPV (억원)"); ax.set_ylabel("55세+ 재직 인년 (숙련유지)")
    ax.set_title("파레토 분석 — 테두리 표시 = 권장 후보군(프론티어)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths["pareto"] = f"{outdir}/chart_pareto.png"
    fig.savefig(paths["pareto"], dpi=140); plt.close(fig)
    return paths


def build_pdf(run_output, sensitivity_output, prose, profile_desc, params_meta,
              out_path="/home/claude/dss/executive_report.pdf"):
    ch = _charts(run_output)
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    story = []
    story.append(Paragraph("정년연장·계속고용 시나리오 인력구조 의사결정 지원 리포트", KO_TITLE))
    story.append(Paragraph("HR Workforce DSS v1 | KLIPS 27차 실측 파라미터 기반 몬테카를로 마이크로시뮬레이션", KO_SMALL))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>대상 기업 프로파일:</b> {profile_desc}", KO))
    story.append(Spacer(1, 4))

    story.append(Paragraph("1. 핵심 요약 (Executive Summary)", KO_H))
    for line in prose.split("\n"):
        story.append(Paragraph(line, KO))
    story.append(Spacer(1, 6))

    story.append(Paragraph("2. 시나리오 비교 (10년 몬테카를로, 200회 반복)", KO_H))
    head = ["시나리오", "인건비 NPV(억)", "90% CI", "채용 누계", "승진적체 p90", "55+ 인년"]
    rows = [head]
    for r in run_output["results"]:
        s = r["summary"]
        mark = " ●" if s["scenario"] in run_output["frontier"] else ""
        rows.append([s["scenario"] + mark, f"{s['cost_npv_mean']/1e4:,.0f}",
                     f"{s['cost_npv_p5']/1e4:,.0f}~{s['cost_npv_p95']/1e4:,.0f}",
                     f"{s['total_hires_mean']:,.0f}", f"{s['promo_congestion_mean']:.1f}년",
                     f"{s['exp55_person_years_mean']:,.0f}"])
    t = Table(rows, colWidths=[62 * mm, 24 * mm, 30 * mm, 18 * mm, 22 * mm, 18 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "HYGothic-Medium"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d6e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4fa")]),
    ]))
    story.append(t)
    story.append(Paragraph("● = 파레토 프론티어(어떤 시나리오에도 지배당하지 않는 권장 후보)", KO_SMALL))

    story.append(Paragraph("3. 10년 경로 전망", KO_H))
    story.append(Image(ch["paths"], width=172 * mm, height=47 * mm))
    story.append(Paragraph("4. 파레토 분석", KO_H))
    story.append(Image(ch["pareto"], width=120 * mm, height=78 * mm))
    story.append(PageBreak())

    story.append(Paragraph("5. 민감도 분석 — 결론이 뒤집히는 지점", KO_H))
    story.append(Paragraph(
        "재고용 수락률·재고용 임금률은 이 분석에서 가장 불확실한 파라미터입니다. "
        "KLIPS 실측 앵커(수락률 하한 0.37=1년내 재취업률, 상한 0.62=전체 재취업률 / "
        "임금비 0.51~0.59) 주변 격자에서 권장 후보군 구성이 달라지는 지점:", KO))
    fp = sensitivity_output["flip_points"]
    if fp:
        rows = [["수락률", "임금률", "해당 조건의 프론티어"]]
        for a, w, f in fp[:10]:
            rows.append([f"{a:.2f}", f"{w:.2f}", ", ".join(x.split(" ")[0] for x in f)])
        t2 = Table(rows, colWidths=[20 * mm, 20 * mm, 130 * mm])
        t2.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "HYGothic-Medium"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5"))]))
        story.append(t2)
    else:
        story.append(Paragraph("탐색 격자 내에서 프론티어 구성 변화 없음 — 결론이 견고함.", KO))

    story.append(Paragraph("6. 방법론·데이터·한계", KO_H))
    for line in params_meta:
        story.append(Paragraph("· " + line, KO))
    doc.build(story)
    return out_path
