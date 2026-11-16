# Decision memo: quality gate model (stage 1 of the pipeline)

## 1. Decision
**Use efficientnet_b0_frozen (efficientnet_b0, input 224x224) as the quality gate.** TODO: confirm this is your decision; if you pick another model, justify it below.

## 2. Evidence
All values are on the same TEST split (180 images), same seed and fixed training parameters; source: results/*/results.json.

| Model | Accuracy | F1 macro | Recall bad | Latency ms | Size MB | All criteria |
|---|---|---|---|---|---|---|
| efficientnet_b0_frozen | 0.994 (pass) | 0.994 (pass) | 0.989 (pass) | 96.2 (pass) | 18.1 (pass) | YES |
| mobilenetv2_frozen | 0.989 (pass) | 0.989 (pass) | 0.989 (pass) | 80.5 (pass) | 11.1 (pass) | YES |
| week7_yolov8n_cls | 0.917 (pass) | 0.917 (pass) | 0.878 (pass) | 33.3 (pass) | 2.8 (pass) | YES |
| week7_yolov8s_cls | 0.989 (pass) | 0.989 (pass) | 0.978 (pass) | 77.0 (pass) | 9.8 (pass) | YES |

Criteria (from the Task 1 memo): accuracy >= 0.91, macro F1 >= 0.9, recall of the "bad" class >= 0.85, latency < 100 ms/image (budget of the first pipeline gate), size <= 50 MB.
The chosen model: accuracy 0.994, F1 0.994, latency 96.2 ms (4 ms below the 100 ms budget), size 18.1 MB.
Latency devices: cpu (run scripts/remeasure_latency.py if the budget must be proven on the CPU server).

## 3. Rejected alternatives and why
- **mobilenetv2_frozen**: F1 0.989 vs 0.994, latency 80.5 vs 96.2 ms, size 11.1 vs 18.1 MB; meets all criteria but is not better on the deciding metrics (F1, then latency). TODO: add the real reason in your own words.
- **week7_yolov8n_cls**: F1 0.917 vs 0.994, latency 33.3 vs 96.2 ms, size 2.8 vs 18.1 MB; meets all criteria but is not better on the deciding metrics (F1, then latency). TODO: add the real reason in your own words.
- **week7_yolov8s_cls**: F1 0.989 vs 0.994, latency 77.0 vs 96.2 ms, size 9.8 vs 18.1 MB; meets all criteria but is not better on the deciding metrics (F1, then latency). TODO: add the real reason in your own words.
- Not trained (excluded in the Task 1 memo before any experiment): ResNet50 (≈98 MB FP32 weights > 50 MB limit); VGG19 (≈548 MB); ViT-B/16 (≈330 MB, data-hungry) - they violate the size/latency limits of a stage-1 CPU gate.

## 4. Risks and mitigation
- **Synthetic-to-real gap (thesis Table 7: 2.8-5.8 %).** The test split is synthetic, so real-photo accuracy will probably be lower by 3-6 points. Mitigation: keep a small real-photo check set and re-measure before release; stricter synthetic threshold (>= 0.91 for a real target of 0.85).
- **Glare weakness** (thesis F1 = 0.92; hardest defect in our failure analysis: glare). Mitigation: stronger glare augmentation (varied position, size, intensity, laminate-style reflections), add real glare photos, hard-negative mining from the failure gallery, threshold tuning for the "bad" class so that uncertain glare photos are rejected rather than accepted.
- **Operator verification loop:** photos that are borderline (probability close to 0.5) go to a human operator instead of silently passing, and operator decisions are logged to grow the training set.

## 5. Deployment note
CPU server now: run the model with the framework used for training (efficientnet_b0); latency above includes preprocessing. Mobile/edge later: export to TFLite (Keras models, with optional quantisation) or ONNX (YOLOv8 via `model.export(format="onnx")`, Keras via tf2onnx), then re-measure latency with the same protocol (20 warm-up + 100 timed) and re-check accuracy on the test split after export/quantisation.
