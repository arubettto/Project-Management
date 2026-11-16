# Error analysis (failure gallery)

### efficientnet_b0_frozen
- misclassified test images: 1 (gallery shows 1; need >= 5)
- failures by defect: {'glare': 1}
- **dominant failure defect: glare** (1 images)
- error rate per defect (errors / test images with that defect): glare: 1/16

### mobilenetv2_frozen
- misclassified test images: 2 (gallery shows 2; need >= 5)
- failures by defect: {'glare': 1, 'none': 1}
- **dominant failure defect: glare** (1 images)
- error rate per defect (errors / test images with that defect): glare: 1/16, none: 1/90

### week7_yolov8n_cls
- misclassified test images: 15 (gallery shows 6; need >= 5)
- failures by defect: {'glare': 3, 'occlusion': 5, 'exposure': 3, 'none': 4}
- **dominant failure defect: occlusion** (5 images)
- error rate per defect (errors / test images with that defect): glare: 3/16, occlusion: 5/17, exposure: 3/19, none: 4/90

### week7_yolov8s_cls
- misclassified test images: 2 (gallery shows 2; need >= 5)
- failures by defect: {'glare': 2}
- **dominant failure defect: glare** (2 images)
- error rate per defect (errors / test images with that defect): glare: 2/16

## Hardest defect overall

**glare** (most failures summed over all models). Expected from the thesis: glare (F1 = 0.92). Check that your data agrees; if not, say so.
