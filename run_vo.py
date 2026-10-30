"""stereo visual odometry on EuRoC sequences or on a synchronised stereo video pair.

    python run_vo.py --seq data/euroc/V2_01_easy/mav0                    # live window, results in results/
    python run_vo.py --seq data/euroc/V2_02_medium/mav0 --detector sift
    python run_vo.py --video data/videos/cam0_vid.mp4 data/videos/cam1_vid.mp4 --seq data/euroc/V2_01_easy/mav0
    python run_vo.py --compare                                           # every detector x every data/euroc sequence
"""
import os
import glob
import json
import time
import argparse
import itertools
import dataclasses
import cv2
import numpy as np

from vo import VOConfig, StereoVO
from vo import evaluation as ev
from vo.device import setup as setup_device
from vo.euroc import StereoCalib, GroundTruth, stereo_pairs, load_pointcloud
from vo.geometry import inv_T


def seq_name(path):
    return os.path.basename(os.path.dirname(os.path.normpath(path)))


def euroc_frames(cfg):
    gt = GroundTruth(cfg.dataset_path)
    pairs = [p for p in stereo_pairs(cfg.dataset_path) if gt.covers(p[0])][cfg.start_frame:]
    pairs = pairs[:cfg.max_frames or None]
    read = lambda f: cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    return len(pairs), gt, ((ts, read(fl), read(fr)) for ts, fl, fr in pairs)


def video_frames(cfg, left, right):
    """frame i of the left video must be frame i of the right one; timestamps come from the video fps."""
    caps = [cv2.VideoCapture(v) for v in (left, right)]
    if not all(c.isOpened() for c in caps):
        raise IOError(f'cannot open {left} / {right}')
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 20.0
    n = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps)) - cfg.start_frame
    n = min(n, cfg.max_frames or n)
    for c in caps:
        c.set(cv2.CAP_PROP_POS_FRAMES, cfg.start_frame)

    def gen():
        for k in range(n):
            imgs = [c.read()[1] for c in caps]
            if any(i is None for i in imgs):
                return
            yield (int((cfg.start_frame + k) / fps * 1e9),
                   *[cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) if i.ndim == 3 else i for i in imgs])
    return n, None, gen()


def run(cfg, video=None, verbose=True):
    """runs on an EuRoC sequence (ground truth -> metrics) or stereo video (calibration from cfg.dataset_path)."""
    n, gt, frames = video_frames(cfg, *video) if video else euroc_frames(cfg)
    name = (os.path.splitext(os.path.basename(video[0]))[0] if video else seq_name(cfg.dataset_path)) \
        + f'_{cfg.detector}'
    os.makedirs(cfg.out_dir, exist_ok=True)
    out = lambda suffix: os.path.join(cfg.out_dir, name + suffix)

    calib = StereoCalib(cfg.dataset_path)
    vo = StereoVO(cfg, calib, setup_device(cfg.use_gpu, verbose))
    first = list(itertools.islice(frames, 20))
    if cfg.self_calibrate:
        vo.rect.self_calibrate([(l, r) for _, l, r in first])
    if verbose:
        print(f'{name}: {n} stereo pairs, baseline {calib.baseline:.4f} m, rectified f={vo.rect.f:.1f}px, '
              f'residual row offset {vo.rect.row_offset:+.2f}px')
    live = None
    if cfg.plot_live:
        from vo.live import LivePlot
        live = LivePlot(name, cfg.plot_fps)

    ts_list, est_T, gt_T, samples, n_kf = [], [], [], [], 0
    xyz_est, xyz_gt = np.zeros((n, 3)), np.zeros((n, 3))     # positions in the gt world frame, for the plots
    t0 = time.time()
    for k, (ts, img_l, img_r) in enumerate(itertools.chain(first, frames)):
        info = vo.process(img_l, img_r)
        T_gt = gt.pose(ts) if gt else np.eye(4)
        if k == 0:
            T_align = T_gt @ inv_T(info['T_WB'])               # only the first pose is known (gnss lost)
        ts_list.append(ts); est_T.append(info['T_WB']); gt_T.append(T_gt)
        xyz_est[k], xyz_gt[k] = (T_align @ info['T_WB'])[:3, 3], T_gt[:3, 3]
        n_kf += info['keyframe']
        if gt and info['keyframe'] and len(info['kf_points']):
            samples.append((T_gt @ vo.T_B_C, info['kf_points']))
        if verbose and k % 100 == 0:
            print(f"  frame {k:5d}  tracked {info['tracked']:4d}  inliers {info['inliers']:4d}  lost {vo.n_lost}")
        if live and live.due():
            live.update(info, xyz_est[:k + 1], xyz_gt[:k + 1] if gt else None)
    n = k + 1
    xyz_est, xyz_gt = xyz_est[:n], xyz_gt[:n]
    res = {'name': name, 'frames': n, 'keyframes': n_kf, 'lost_frames': vo.n_lost,
           'fps': n / (time.time() - t0), 'row_offset_px': vo.rect.row_offset}

    ev.save_tum(out('_est.tum'), ts_list, est_T)
    if gt:
        ev.save_tum(out('_gt.tum'), ts_list, gt_T)
        res.update(ev.evaluate(np.array(est_T), np.array(gt_T), samples, load_pointcloud(cfg.dataset_path),
                               out('_trajectory.png'), name))
    else:
        length = float(np.linalg.norm(np.diff(xyz_est, axis=0), axis=1).sum())
        res['length_m'] = length
        ev.plot(out('_trajectory.png'), f'{name}: path length {length:.1f} m, lost {vo.n_lost} '
                f'(frame of the first pose)', {'VO': xyz_est})
    with open(out('_metrics.json'), 'w') as f:
        json.dump({**res, 'config': dataclasses.asdict(cfg)}, f, indent=2)
    if verbose:
        print(json.dumps(res, indent=2))
    if live:
        live.update(info, xyz_est, xyz_gt if gt else None)     # final state
        live.close()
    return res


def compare(base_cfg, detectors=('orb', 'sift', 'gftt')):
    head = ['sequence', 'detector', 'ATE SE3 [m]', 'drift [%]', 'RPE 1s [m]', 'RPE 1s [deg]', 'Sim3 scale',
            'lost frames', 'depth err vs laser [m]', 'fps']
    rows = []
    for seq, det in itertools.product(sorted(glob.glob('data/euroc/*/mav0')), detectors):
        try:
            r = run(dataclasses.replace(base_cfg, dataset_path=seq, detector=det), verbose=False)
        except Exception as e:
            print(f'{seq} {det}: FAILED ({e})', flush=True)
            continue
        dl = r['depth_vs_laser']
        rows.append([seq_name(seq), det, f"{r['ate_se3']:.3f}", f"{r['drift_pct']:.2f}", f"{r['rpe_1s_m']:.3f}",
                     f"{r['rpe_1s_deg']:.2f}", f"{r['sim3_scale']:.3f}", str(r['lost_frames']),
                     f"{dl['median']:.3f}" if dl else 'n/a', f"{r['fps']:.0f}"])
        print(' | '.join(rows[-1]), flush=True)
    path = os.path.join(base_cfg.out_dir, 'comparison.md')
    with open(path, 'w') as f:
        f.write('\n'.join('| ' + ' | '.join(r) + ' |' for r in [head, ['---'] * len(head)] + rows) + '\n')
    print(f'saved {path}')


if __name__ == '__main__':
    d = VOConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seq', default=d.dataset_path, help='EuRoC mav0 folder (also the calibration for --video)')
    p.add_argument('--video', nargs=2, metavar=('LEFT', 'RIGHT'), help='stereo video pair instead of images')
    p.add_argument('--detector', default=d.detector, choices=['orb', 'sift', 'gftt'])
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--frames', type=int, default=0, help='0 = whole sequence')
    p.add_argument('--cpu', type=int, choices=[0, 1], default=1, help='0 = gpu (opencl), 1 = cpu')
    p.add_argument('--no-live', action='store_true', help='no live window (faster, plots are still saved)')
    p.add_argument('--plot-fps', type=float, default=d.plot_fps, help='max redraws/s of the live window')
    p.add_argument('--out', default=d.out_dir)
    p.add_argument('--compare', action='store_true', help='every detector on every sequence in data/euroc')
    a = p.parse_args()
    cfg = VOConfig(dataset_path=a.seq, detector=a.detector, start_frame=a.start, max_frames=a.frames,
                   use_gpu=a.cpu == 0, plot_live=not (a.no_live or a.compare), plot_fps=a.plot_fps, out_dir=a.out)
    compare(cfg) if a.compare else run(cfg, a.video)
