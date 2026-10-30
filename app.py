"""gui: stereo vo (separate process, live matplotlib window) + object detection on the left camera.

    python app.py          (needs torch, torchvision, PySide6)
"""
import os
import sys
import dataclasses
import multiprocessing as mp

import cv2
import numpy as np
import torch
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QFormLayout, QMessageBox, QGroupBox, QListWidget,
                               QListWidgetItem, QComboBox)

from vo import VOConfig
from vo.euroc import stereo_pairs
from detection.model import CLASSES, load_detector, predict, split_items, load_yolo
from detection.metrics import DetectionMetrics

# VOConfig fields exposed in the form
FORM_FIELDS = ['dataset_path', 'start_frame', 'max_frames', 'n_features', 'ratio', 'max_row_diff',
               'min_disparity', 'max_depth', 'pnp_reproj_err', 'min_inliers', 'kf_min_tracked']


def vo_process(cfg_dict):
    import matplotlib
    matplotlib.use('TkAgg')        # qt is owned by the parent process
    from run_vo import run
    run(VOConfig(**cfg_dict))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('стерео VO та детекція об\'єктів')
        self.resize(1000, 720)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vo_proc = None
        self.timer = QTimer(self, timeout=self.process_frame)
        self.frames, self.idx = [], 0
        self.metrics = None
        # only held-out frames are scored - training frames would give a meaningless, inflated score
        self.val_labels = {os.path.basename(i): l for i, l in split_items()[1]}
        self._build_ui()

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        self.view = QLabel('очікування...'); self.view.setFixedSize(640, 410)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet('background: black; color: white;')
        left = QGroupBox('детекція (cam0)'); QVBoxLayout(left).addWidget(self.view)
        lay.addWidget(left)

        right = QVBoxLayout(); lay.addLayout(right)
        form = QFormLayout(); right.addLayout(form)
        d = VOConfig()
        self.inputs = {}
        for name in FORM_FIELDS:
            self.inputs[name] = QLineEdit(str(getattr(d, name))); form.addRow(name, self.inputs[name])
        self.detector_box = QComboBox(); self.detector_box.addItems(['orb', 'sift', 'gftt'])
        form.addRow('detector', self.detector_box)
        self.weights = QLineEdit('models/ssd_v2.pth'); form.addRow('weights', self.weights)

        self.class_list = QListWidget()
        for c in CLASSES:
            it = QListWidgetItem(c); it.setFlags(it.flags() | Qt.ItemIsUserCheckable); it.setCheckState(Qt.Checked)
            self.class_list.addItem(it)
        right.addWidget(QLabel('класи для відображення')); right.addWidget(self.class_list)
        self.status = QLabel(''); right.addWidget(self.status)
        self.btn_start = QPushButton('запустити', clicked=self.start)
        self.btn_stop = QPushButton('зупинити', clicked=self.stop, enabled=False)
        right.addWidget(self.btn_start); right.addWidget(self.btn_stop)

    def _config(self):
        d = VOConfig()
        vals = {name: type(getattr(d, name))(w.text()) for name, w in self.inputs.items()}
        return dataclasses.replace(d, **vals, detector=self.detector_box.currentText(), plot_live=True)

    def start(self):
        try:
            cfg = self._config()
            self.frames = stereo_pairs(cfg.dataset_path)
            self.model = load_detector(self.weights.text(), self.device)
        except Exception as e:
            QMessageBox.critical(self, 'помилка конфігурації', str(e)); return
        self.idx = min(cfg.start_frame, len(self.frames) - 1)
        self.metrics = DetectionMetrics(CLASSES)
        self.vo_proc = mp.Process(target=vo_process, args=(dataclasses.asdict(cfg),), daemon=True)
        self.vo_proc.start()
        self.timer.start(50)
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)

    def stop(self):
        self.timer.stop()
        if self.vo_proc and self.vo_proc.is_alive():
            self.vo_proc.terminate(); self.vo_proc.join()
        if self.metrics and self.metrics.frames:
            os.makedirs('results', exist_ok=True)
            s = self.metrics.save('results/detection_metrics.json', 'results/confusion_matrix.png')
            print(f"detection: {s['frames']} labelled val frames, mAP50={s['mAP50']:.3f}")
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)

    def process_frame(self):
        if self.idx >= len(self.frames):
            self.stop(); return
        _, path, _ = self.frames[self.idx]
        self.idx += 1
        bgr = cv2.imread(path)
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        boxes, labels, scores = predict(self.model, img, self.device, 0.5)

        lbl = self.val_labels.get(os.path.basename(path))
        if lbl is not None:
            gb, gl = load_yolo(lbl, *img.size)
            self.metrics.update(boxes, labels, scores, gb, gl)
            s = self.metrics.summary()
            self.status.setText(f"оцінено кадрів: {s['frames']}   mAP50: {s['mAP50']:.3f}")

        shown = {self.class_list.item(i).text() for i in range(self.class_list.count())
                 if self.class_list.item(i).checkState() == Qt.Checked}
        for (x1, y1, x2, y2), l, sc in zip(boxes.astype(int), labels, scores):
            name = CLASSES[l - 1] if 0 < l <= len(CLASSES) else '?'
            if name in shown:
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bgr, f'{name} {sc:.2f}', (x1, max(0, y1 - 6)), 0, 0.5, (0, 255, 0), 1)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        self.view.setPixmap(QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888))
                            .scaled(640, 410, Qt.KeepAspectRatio))

    def closeEvent(self, e):
        self.stop(); e.accept()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())
