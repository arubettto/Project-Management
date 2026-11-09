# Quality gate: Tasks 2, 3, 4

Run everything from the repository root (the folder that contains `data/quality_split/`).
Replace `<ID>` with your student ID (it is the seed everywhere). Copy `scripts/` into your repo.

```
pip install -r requirements.txt

# Task 2 - baseline (week 6)
python scripts/train_keras_quality.py --model mobilenetv2 --regime frozen --seed <ID> --week 6
python scripts/make_reports.py baseline          # reports/baseline.md skeleton -> write the 5-10 lines
git add -A && git commit -m "task 2" && git tag task2

# Task 3 - YOLOv8 (week 7)
python scripts/train_yolo_quality.py --seed <ID> --variants n s --week 7      # add --batch 16 on CUDA OOM
python scripts/make_reports.py week7             # reports/week7_comparison.md -> write the paragraph
git add -A && git commit -m "task 3" && git tag week7

# Task 4 - third model + comparison (week 8)
python scripts/train_keras_quality.py --model efficientnet_b0 --regime frozen --seed <ID> --week 8
python scripts/remeasure_latency.py results/*/            # optional: latency on CPU if you trained on GPU
python scripts/collect_results.py --data data/quality_split --metadata data/metadata.jsonl
python scripts/make_slides.py                              # reports/presentation.pptx -> edit, export PDF
git add -A && git commit -m "task 4" && git tag week8
```

Smoke test first with `--epochs 1` (Keras) to check the paths; delete `results/` afterwards, because the
real runs must use epochs 50 / patience 5.

Failed attempts: a crashed run writes `results/<exp>/FAILED.json`; add `--note "why"` and every run (ok or failed)
is appended to `results/experiment_log.jsonl`.

What you still write yourself: baseline interpretation, Task 3 comparison paragraph, defect/augmentation text in
`reports/error_analysis.md`, final edits to the decision memo and slides.
