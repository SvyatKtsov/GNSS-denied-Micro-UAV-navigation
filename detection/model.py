"""SSDLite-MobileNetV3 detector for EuRoC objects + the dataset/split shared by training and evaluation."""
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, models
from torchvision.models.detection.ssd import SSDHead

CLASSES = ["pipe", "construction_board", "gym_mat", "radiator"]   # label id = index + 1 (0 = background)
IMG_SIZE = 320

_ROOT = "/home/skrand/KNU/КНУ_4курс_1семестр/Курсова/object_detection_data/to_label"
# (image folder, YOLO label folder) - disjoint, time-contiguous segments of MH_02 and V2_01
DATA = [
    (f"{_ROOT}/mh_02", f"{_ROOT}/1/obj_train_data"),
    (f"{_ROOT}/mh_02_continuation", f"{_ROOT}/2/obj_train_data"),
    (f"{_ROOT}/mh_02_continuation_2", f"{_ROOT}/4_mh_continuation_2/obj_train_data"),
    (f"{_ROOT}/v2_01", f"{_ROOT}/3/obj_train_data"),
    (f"{_ROOT}/v2_01_continuation", f"{_ROOT}/5_v2_01_continuation/obj_train_data"),
]
VAL_FRACTION = 0.2


def split_items(val_fraction=VAL_FRACTION):
    """Last `val_fraction` (by time) of every segment is validation. Splitting by time and not at
    random matters: neighbouring video frames are near-duplicates, a random split leaks."""
    train, val = [], []
    for img_dir, lbl_dir in DATA:
        if not os.path.isdir(img_dir):
            continue
        names = sorted(f for f in os.listdir(img_dir) if f.endswith('.png')
                       and os.path.exists(os.path.join(lbl_dir, f[:-4] + '.txt')))
        n_val = int(round(len(names) * val_fraction))
        items = [(os.path.join(img_dir, n), os.path.join(lbl_dir, n[:-4] + '.txt')) for n in names]
        train += items[:len(items) - n_val]
        val += items[len(items) - n_val:]
    return train, val


def load_yolo(label_path, w, h):
    boxes, labels = [], []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                xc, yc, bw, bh = map(float, p[1:5])
                boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
                labels.append(int(p[0]) + 1)
    return np.array(boxes, np.float32).reshape(-1, 4), np.array(labels, np.int64)


def preprocess(normalize_imagenet=False):
    # torchvision detection models normalise internally (GeneralizedRCNNTransform, mean=std=0.5 for
    # SSDLite), so the input must be a plain [0,1] tensor. The first checkpoint (ssd_model.pth) was
    # trained WITH an extra ImageNet Normalize, so it needs the same input to work.
    t = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    if normalize_imagenet:
        t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(t)


def build_model(num_classes=len(CLASSES) + 1, pretrained=True):
    # weights_backbone=None keeps the same (reduced-tail) backbone that weights='DEFAULT' builds
    model = models.detection.ssdlite320_mobilenet_v3_large(weights='DEFAULT' if pretrained else None,
                                                           weights_backbone=None)
    in_channels = [672, 480, 512, 256, 256, 128]
    model.head = SSDHead(in_channels, model.anchor_generator.num_anchors_per_location(), num_classes)
    return model


def load_detector(path, device):
    """Returns (model, preprocess, legacy). Old checkpoints are bare state_dicts: trained with the extra
    ImageNet Normalize AND with box targets left in 752x480 pixels on a 320x320 tensor (so their
    outputs are already in original-image pixels, clipped at 320)."""
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state, legacy = ckpt['state_dict'], ckpt.get('normalize_imagenet', False)
    else:
        state, legacy = ckpt, True
    model = build_model(pretrained=False)
    model.load_state_dict(state)
    return model.to(device).eval(), preprocess(legacy), legacy


@torch.no_grad()
def predict(model, tf, img_rgb: Image.Image, device, score_thresh=0.5, legacy=False):
    """Boxes in ORIGINAL image pixels: (N,4) xyxy, labels (N,), scores (N,)."""
    w, h = img_rgb.size
    out = model(tf(img_rgb).unsqueeze(0).to(device))[0]
    keep = out['scores'] > score_thresh
    boxes = out['boxes'][keep].cpu().numpy()
    if not legacy:
        boxes = boxes * np.array([w / IMG_SIZE, h / IMG_SIZE] * 2)
    return boxes, out['labels'][keep].cpu().numpy(), out['scores'][keep].cpu().numpy()
