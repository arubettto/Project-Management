# Bonus 5A - Fine-tuning delta memo (frozen vs finetune_top30)

Protocol: top 30 backbone layers unfrozen (BatchNorm frozen: True), Adam lr = 1e-05
(10x lower than head training 0.001), epochs <= 50, patience 5, batch 32, seed 123,
same train/val/test split as the frozen run. Test split used once, after training.

## Delta table
| metric (test split) | frozen | fine-tuned top30 | delta |
|---|---|---|---|
| accuracy | 0.9833 | 0.9944 | +0.0111 |
| precision_macro | 0.9834 | 0.9945 | +0.0111 |
| recall_macro | 0.9833 | 0.9944 | +0.0111 |
| f1_macro | 0.9833 | 0.9944 | +0.0111 |
| latency CPU mean (ms) | 124.05 | 121.34 | -2.7107 |
| model size (MB, no optimizer) | 10.27 | 10.27 | -0.0000 |
| total params | 2,422,081 | 2,422,081 | 0 |
| trainable params | head only (see Task 2) | 1,674,817 |  |
| train-val acc gap | n/a (set BEST_HISTORY_CSV) | 0.0032 |  |
| val loss - train loss | n/a | 0.0085 |  |

Curves: `results/task5a_finetune_top30/curves.png` - Errors: `results/task5a_finetune_top30/errors/`

## Memo (5 lines)
1. **Improved:** fine-tuned test accuracy is 0.9944 (179/180), macro-F1 0.9944; precision of "bad" = 1.000,
   recall of "good" = 1.000. Versus frozen: [paste frozen accuracy / macro-F1 and the delta from the table].
2. **Got worse / not improved:** the only error is 1 bad photo accepted as good (false accept, recall of
   "bad" 0.989). [If frozen had fewer or equal errors: say so.] Training cost: 96 s, 10 epochs, vs [frozen
   time]. One image = 0.56 pp of accuracy, so a difference of 1-2 images on a 180-image test set is within
   noise and not strong evidence of improvement.
3. **Latency and size unchanged:** model size 10.27 MB and 2,422,081 parameters, the same as the frozen model.
   Unfreezing changes only the values of the weights, not the graph, number of FLOPs or file format, so
   per-image cost is identical. The CPU latency (mean 140 ms, median 165 ms, std 32 ms) is noisy, so small
   differences between runs are measurement noise; real edge numbers come from the TFLite benchmark in 5D.
4. **Train-val gap (overfitting sign):** at the best epoch (5 of 10) train acc 0.9976 vs val acc 0.9889,
   gap 0.0087; train loss 0.0106 vs val loss 0.0259. In later epochs the gap widens (train acc ~0.999 vs val
   acc ~0.978, val loss 0.05-0.11 vs train loss ~0.01) and val loss spikes at epoch 7, so val loss stops
   improving after epoch 5 and early stopping correctly triggered. Frozen gap: at its best epoch (27 of 32) train acc 0.9964 vs val acc 0.9833, gap 0.0131 (val loss 0.0303 vs train loss 0.0131, gap 0.0171); fine-tuned: 0.0087 (loss gap 0.0153). The gap did not grow, so unfreezing did not increase overfitting at the selected epoch. This is a mild
   overfitting tendency: the 1.67M trainable parameters fit the small (840-image) train set better than the
   val set.
5. **Deploy verdict:** It makes fewer test errors than the frozen one, has the same size and parameter count (so no deployment cost), and its train-val gap at the best epoch is not larger (0.0087 vs 0.0131) and its best val loss is lower (0.0259 vs 0.0303). The evidence is weak: the difference is only 1-2 images out of 180, so this is a cautious "yes", and the threshold analysis (5B) matters more than this delta.
