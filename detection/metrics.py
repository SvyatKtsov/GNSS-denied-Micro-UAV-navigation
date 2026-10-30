"""detection metrics: IoU matching, cumulative per-class P/R/F1, AP@0.5, confusion matrix."""
import json
import numpy as np


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


class DetectionMetrics:
    def __init__(self, classes, iou_thresh=0.5):
        self.classes, self.iou_thresh = list(classes), iou_thresh
        n = len(self.classes) + 1                       # index 0 = background
        self.confusion = np.zeros((n, n), int)          # rows = ground truth, cols = prediction
        self.scores = {c: [] for c in range(1, n)}      # (score, is_tp) per predicted class for AP
        self.n_gt = {c: 0 for c in range(1, n)}
        self.frames = 0

    def update(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        self.frames += 1
        for c in gt_labels:
            self.n_gt[int(c)] += 1
        order = np.argsort(-pred_scores)
        pred_boxes, pred_labels, pred_scores = pred_boxes[order], pred_labels[order], pred_scores[order]
        ious = iou_matrix(pred_boxes, gt_boxes)
        used = np.zeros(len(gt_boxes), bool)
        for i in range(len(pred_boxes)):                # greedy, highest score first
            cand = np.where(~used & (ious[i] >= self.iou_thresh))[0] if len(gt_boxes) else []
            p = int(pred_labels[i])
            if len(cand):
                j = cand[np.argmax(ious[i, cand])]
                used[j] = True
                g = int(gt_labels[j])
                self.confusion[g, p] += 1
                self.scores[p].append((float(pred_scores[i]), g == p))
            else:
                self.confusion[0, p] += 1               # false positive on background
                self.scores[p].append((float(pred_scores[i]), False))
        for j in np.where(~used)[0]:
            self.confusion[int(gt_labels[j]), 0] += 1   # missed object

    def _ap(self, c):
        s = sorted(self.scores[c], key=lambda x: -x[0])
        if self.n_gt[c] == 0:
            return float('nan')
        tp = np.cumsum([t for _, t in s]); fp = np.cumsum([not t for _, t in s])
        rec = tp / self.n_gt[c] if len(s) else np.zeros(0)
        prec = tp / np.maximum(tp + fp, 1) if len(s) else np.zeros(0)
        # all-point interpolated AP (VOC2010+/COCO style)
        mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        return float(np.sum(np.diff(mrec) * mpre[1:]))

    def summary(self):
        per = {}
        for c, name in enumerate(self.classes, start=1):
            tp = int(self.confusion[c, c])
            fp = int(self.confusion[:, c].sum() - tp)
            fn = int(self.confusion[c, :].sum() - tp)
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            per[name] = {'tp': tp, 'fp': fp, 'fn': fn, 'precision': p, 'recall': r,
                         'f1': 2 * p * r / (p + r) if p + r else 0.0, 'ap50': self._ap(c), 'n_gt': self.n_gt[c]}
        aps = [v['ap50'] for v in per.values() if not np.isnan(v['ap50'])]
        return {'frames': self.frames, 'iou_thresh': self.iou_thresh,
                'mAP50': float(np.mean(aps)) if aps else float('nan'),
                'per_class': per, 'confusion_matrix': self.confusion.tolist(),
                'confusion_labels': ['background'] + self.classes}

    def save(self, json_path, png_path=None):
        s = self.summary()
        with open(json_path, 'w') as f:
            json.dump(s, f, indent=2)
        if png_path:
            from matplotlib.figure import Figure
            cm, labels = self.confusion, s['confusion_labels']
            fig = Figure(figsize=(8, 7)); ax = fig.subplots()
            ax.imshow(cm, cmap='Greens')
            ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
                   xlabel='predicted', ylabel='ground truth',
                   title=f"confusion matrix (IoU>={self.iou_thresh}, {self.frames} frames, mAP50={s['mAP50']:.3f})")
            ax.tick_params(axis='x', labelrotation=45)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
            fig.tight_layout(); fig.savefig(png_path, dpi=150)
        return s
