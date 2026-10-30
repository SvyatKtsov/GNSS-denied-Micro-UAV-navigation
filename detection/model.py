"""ssdlite-mobilenetv3 detector for EuRoC objects + the dataset split shared by training and evaluation."""
import os
import numpy as np
import torch
from torchvision import transforms, models
from torchvision.models.detection.ssd import SSDHead

CLASSES = ['pipe', 'construction_board', 'gym_mat', 'radiator']   # label id = index + 1 (0 = background)
IMG_SIZE = 320
# torchvision detection models normalise internally, so the input is a plain [0,1] tensor
TF = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])

_ROOT = os.path.join(os.path.dirname(__file__), '..', 'object_detection_data', 'to_label')
# (image folder, yolo label folder): disjoint, time-contiguous segments of MH_02 and V2_01
DATA = [(os.path.join(_ROOT, img), os.path.join(_ROOT, lbl, 'obj_train_data')) for img, lbl in [
    ('mh_02', '1'), ('mh_02_continuation', '2'), ('mh_02_continuation_2', '4_mh_continuation_2'),
    ('v2_01', '3'), ('v2_01_continuation', '5_v2_01_continuation')]]


def split_items(val_fraction=0.2):
    """the last `val_fraction` (by time) of every segment is validation. splitting by time, not at
    random, matters: neighbouring video frames are near-duplicates, a random split leaks."""
    train, val = [], []
    for img_dir, lbl_dir in DATA:
        if not os.path.isdir(img_dir):
            continue
        items = [(os.path.join(img_dir, f), os.path.join(lbl_dir, f[:-4] + '.txt')) for f in sorted(os.listdir(img_dir))
                 if f.endswith('.png') and os.path.exists(os.path.join(lbl_dir, f[:-4] + '.txt'))]
        n_val = int(round(len(items) * val_fraction))
        train += items[:len(items) - n_val]
        val += items[len(items) - n_val:]
    return train, val


def load_yolo(label_path, w, h):
    """yolo txt (class xc yc w h, normalised) -> xyxy boxes in pixels, labels (id + 1)."""
    rows = [list(map(float, l.split()[:5])) for l in open(label_path) if len(l.split()) >= 5] \
        if os.path.exists(label_path) else []
    a = np.array(rows, np.float32).reshape(-1, 5)
    boxes = np.stack([(a[:, 1] - a[:, 3] / 2) * w, (a[:, 2] - a[:, 4] / 2) * h,
                      (a[:, 1] + a[:, 3] / 2) * w, (a[:, 2] + a[:, 4] / 2) * h], 1)
    return boxes, a[:, 0].astype(np.int64) + 1


def build_model(pretrained=True):
    # weights_backbone=None keeps the same (reduced-tail) backbone that weights='DEFAULT' builds
    model = models.detection.ssdlite320_mobilenet_v3_large(weights='DEFAULT' if pretrained else None,
                                                           weights_backbone=None)
    model.head = SSDHead([672, 480, 512, 256, 256, 128], model.anchor_generator.num_anchors_per_location(),
                         len(CLASSES) + 1)
    return model


def load_detector(path, device):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device)['state_dict'])
    return model.to(device).eval()


@torch.no_grad()
def predict(model, img_rgb, device, score_thresh=0.5):
    """boxes in original image pixels: (N,4) xyxy, labels (N,), scores (N,)."""
    w, h = img_rgb.size
    out = model(TF(img_rgb).unsqueeze(0).to(device))[0]
    keep = out['scores'] > score_thresh
    boxes = out['boxes'][keep].cpu().numpy() * np.array([w / IMG_SIZE, h / IMG_SIZE] * 2)
    return boxes, out['labels'][keep].cpu().numpy(), out['scores'][keep].cpu().numpy()
