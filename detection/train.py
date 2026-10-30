"""Fine-tune SSDLite on the labelled EuRoC frames, validate on the held-out (later-in-time) frames.

    python -m detection.train --epochs 12 --out models/ssd_v2.pth
    python -m detection.train --eval-only models/ssd_model.pth      # score an existing checkpoint
"""
import argparse
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .model import CLASSES, split_items, load_yolo, preprocess, build_model, load_detector, predict
from .metrics import DetectionMetrics


class EuRoCDetDataset(Dataset):
    def __init__(self, items, augment):
        self.items, self.augment, self.tf = items, augment, preprocess(False)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_path, lbl_path = self.items[i]
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        boxes, labels = load_yolo(lbl_path, w, h)
        if self.augment:
            if random.random() < 0.5:                       # horizontal flip (no vertical: gravity matters)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            # EuRoC has strong auto-exposure changes and motion blur -> photometric augmentation
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
        sx, sy = 320 / w, 320 / h                           # boxes must be in the resized image's pixels
        boxes = boxes * np.array([sx, sy, sx, sy], np.float32)
        return self.tf(img), {'boxes': torch.from_numpy(boxes), 'labels': torch.from_numpy(labels)}


def evaluate(model, tf, items, device, score_thresh=0.3, legacy=False):
    m = DetectionMetrics(CLASSES)
    model.eval()
    for img_path, lbl_path in tqdm(items, desc='val', leave=False):
        img = Image.open(img_path).convert('RGB')
        b, l, s = predict(model, tf, img, device, score_thresh, legacy)
        gb, gl = load_yolo(lbl_path, *img.size)
        m.update(b, l, s, gb, gl)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--out', default='models/ssd_v2.pth')
    ap.add_argument('--eval-only', default=None, help='checkpoint to evaluate on the validation split')
    ap.add_argument('--force-scale', action='store_true', help='(debug) rescale legacy outputs by w/320 like the old GUI')
    a = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_items, val_items = split_items()
    print(f'train {len(train_items)}  val {len(val_items)}  device {device}')

    if a.eval_only:
        model, tf, legacy = load_detector(a.eval_only, device)
        if a.force_scale:
            legacy = False
        s = evaluate(model, tf, val_items, device, legacy=legacy).save('results/detection_val_metrics.json',
                                                         'results/detection_val_confusion.png')
        print(f"mAP50 {s['mAP50']:.3f}", {k: round(v['ap50'], 3) for k, v in s['per_class'].items()})
        return

    random.seed(0); torch.manual_seed(0)
    dl = DataLoader(EuRoCDetDataset(train_items, augment=True), batch_size=a.bs, shuffle=True,
                    num_workers=4, collate_fn=lambda x: tuple(zip(*x)), drop_last=True)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl))
    tf_val, best = preprocess(False), -1.0
    for ep in range(a.epochs):
        model.train()
        pbar = tqdm(dl, desc=f'epoch {ep + 1}/{a.epochs}')
        for imgs, tgts in pbar:
            imgs = [x.to(device) for x in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            pbar.set_postfix(loss=f'{loss.item():.3f}')
        s = evaluate(model, tf_val, val_items, device).summary()
        print(f"epoch {ep + 1}: val mAP50 {s['mAP50']:.3f}")
        if s['mAP50'] > best:                               # keep the best epoch on validation
            best = s['mAP50']
            torch.save({'state_dict': model.state_dict(), 'classes': CLASSES, 'normalize_imagenet': False,
                        'val_mAP50': best, 'epoch': ep + 1}, a.out)
    print(f'best val mAP50 {best:.3f} -> {a.out}')


if __name__ == '__main__':
    main()
