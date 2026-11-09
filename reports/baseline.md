# Task 2 - MobileNetV2 frozen baseline

Command: `python scripts/train_keras_quality.py --model mobilenetv2 --regime frozen --seed 123 --week 6`
Seed = 123; epochs <= 50, patience 5, batch 32, Adam lr 1e-3, input 224x224, preprocess_input mode "tf", threshold 0.5, split 70/15/15.

## Test-split results (180 images)
| accuracy | precision macro | recall macro | F1 macro | latency ms/img | size MB | params | train time s |
|---|---|---|---|---|---|---|---|
| 0.9889 | 0.9889 | 0.9889 | 0.9889 | 80.46 | 11.055 | 2,422,081 | 347.2 |

Per class: bad P/R/F1 = 0.989/0.989/0.989; good P/R/F1 = 0.989/0.989/0.989.
Confusion matrix (rows true bad/good, cols predicted bad/good): [[89, 1], [1, 89]]. Images: curves.png, confusion_matrix.png; errors in results/mobilenetv2_frozen/errors/.

## Facts from the curves
- epochs run: 32 of 50 planned (early stopping triggered); best val_loss at epoch 27
- final train accuracy 0.996 vs final val accuracy 0.994 (gap +0.002)
- final train loss 0.013 vs val loss 0.033; min val loss 0.030
- harder class by F1: **bad**

## Interpretation

The curves show neither overfitting nor underfitting: the final train accuracy (0.996) and validation accuracy (0.994) differ by only 0.002, and both are close to 1. Validation loss reached its minimum (0.030) at epoch 27 and did not improve afterwards, so early stopping ended training at epoch 32 of 50 and restored the best weights. The train loss (0.013) is lower than the validation loss (0.033), which is a small generalisation gap, but validation accuracy did not drop, so it is not harmful overfitting. With a frozen ImageNet base and only about 164k trainable parameters in the head, the model already separates good and bad photos almost perfectly on the synthetic test set: 2 errors out of 180 images (accuracy and macro F1 = 0.9889).

Neither class is clearly harder: precision, recall and F1 are 0.989 for both, and the confusion matrix has one bad photo accepted as good and one good photo rejected as bad (the "harder class: bad" line above is only a tie). The accepted bad photo is a glare case: a reflection that covers only a small part of the page looks similar to a normal good photo after downscaling to 224x224, which makes glare the weakest point of the model and is consistent with the weakness of glare reported in the thesis. The rejected good photo has no visible defect in its name, so it is most likely a borderline false alarm caused by uneven lighting or texture. The result must be treated with care: the test set is synthetic and small, so one error changes accuracy by 0.6 points, and on real photos accuracy is expected to be lower by the known synthetic-to-real gap of 2.8-5.8 %. As a next step I would fine-tune the base with a low learning rate (1e-5) and add glare augmentation (different position, size and intensity), then check the result on real photos.
