"""Task 4 presentation: 4 slides (pptx) built from reports/. Edit the text, then export to PDF.

  python scripts/make_slides.py     # -> reports/presentation.pptx
"""
import json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
import pandas as pd

R = Path("reports")
s = json.load(open(R / "decision_summary.json"))
df = pd.read_csv(R / "comparison.csv")
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.33), Inches(7.5)
blank = prs.slide_layouts[5]


def slide(title):
    sl = prs.slides.add_slide(blank); sl.shapes.title.text = title
    sl.shapes.title.text_frame.paragraphs[0].font.size = Pt(30)
    return sl


def text(sl, lines, x, y, w, h, size=16):
    tf = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame; tf.word_wrap = True
    for i, l in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.text = l; p.font.size = Pt(size)


def pic(sl, path, x, y, w=None, h=None):
    if Path(path).exists():
        sl.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w) if w else None, height=Inches(h) if h else None)


# 1 context
sl = slide("Pipeline context: the quality gate is stage 1")
text(sl, ["Telegram bot: user sends a photo of a Russian document",
          "Stage 1 = quality gate: accept the photo or ask for a retake, BEFORE OCR / extraction",
          "Goal: reject blur, glare, skew, occlusion, bad exposure - fast (< 100 ms per image on CPU)",
          "Data: synthetic quality dataset (good / bad), split 70/15/15, seed = student ID",
          "Compared under identical protocol: epochs 50, patience 5, batch 32"], 0.7, 1.6, 12, 5, 22)
# 2 table + chart
sl = slide("Comparison on the test split")
rows = len(df) + 1; tbl = sl.shapes.add_table(rows, 6, Inches(0.4), Inches(1.4), Inches(7.4), Inches(0.4 * rows)).table
for j, h in enumerate(["Model", "Acc", "F1", "ms/img", "MB", "Epochs"]): tbl.cell(0, j).text = h
for i, r in enumerate(df.itertuples(), 1):
    for j, v in enumerate([r.experiment, f"{r.accuracy:.3f}", f"{r.f1_macro:.3f}", f"{r.latency_ms_per_image:.1f}", f"{r.model_size_mb:.1f}", int(r.epochs_run)]):
        tbl.cell(i, j).text = str(v)
for c in range(rows):
    for j in range(6):
        for p in tbl.cell(c, j).text_frame.paragraphs: p.font.size = Pt(11)
pic(sl, R / "acc_vs_latency.png", 8.0, 1.4, w=5.0)
# 3 failures
sl = slide(f"Failure gallery - hardest defect: {s['hardest_defect']}")
first = sorted((R / "failure_gallery").glob("*.png"))
pic(sl, first[0] if first else R / "x.png", 0.4, 1.3, h=4.6)
text(sl, [f"Hardest defect overall: {s['hardest_defect']} (thesis: glare, F1 = 0.92)", "Fix: stronger glare augmentation, real glare photos, hard negatives"], 6.6, 1.6, 6.4, 4, 18)
# 4 decision
c = s["chosen_stats"]
sl = slide("Decision and risks")
text(sl, [f"Decision: {s['chosen']} is the quality gate",
          f"Evidence: acc {c['accuracy']:.3f}, F1 {c['f1_macro']:.3f}, {c['latency_ms_per_image']:.1f} ms (budget 100 ms), {c['model_size_mb']:.1f} MB",
          "Risk 1: synthetic-to-real gap 2.8-5.8 % -> real-photo check set before release",
          "Risk 2: glare weakness -> glare augmentation, threshold tuning",
          "Mitigation: operator verification loop for borderline photos",
          "Next: CPU server now; TFLite/ONNX export for mobile later"], 0.7, 1.6, 12, 5, 22)
prs.save(R / "presentation.pptx"); print("reports/presentation.pptx written")
