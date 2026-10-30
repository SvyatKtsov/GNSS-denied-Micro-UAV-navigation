"""Stereo VO on EuRoC.

    python run_vo.py --seq data/euroc/V1_01_easy/mav0                   # ORB, save results
    python run_vo.py --seq data/euroc/V2_01_easy/mav0 --detector sift
    python run_vo.py --compare                                          # all detectors x all sequences
    python run_vo.py --video data/videos/cam0_vid.mp4 data/videos/cam1_vid.mp4 --seq data/euroc/V2_01_easy/mav0

A live window (camera + distances | 3D | 2D) opens by default: run with ~/py_envs/py_env1/bin/python.
"""
import os
import sys
import json
import time
import argparse
import dataclasses
import cv2
import numpy as np

from vo import VOConfig, StereoVO
from vo.euroc import StereoCalib, GroundTruth, stereo_pairs, load_pointcloud
from vo import evaluation as ev
from vo.geometry import inv_T

SEQUENCES = ['data/euroc/V1_01_easy/mav0', 'data/euroc/V2_01_easy/mav0', 'data/euroc/MH_02_easy/mav0']


def seq_name(path):
    return os.path.basename(os.path.dirname(os.path.normpath(path)))


def draw_depth(gray, pts, depth):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x, y), z in zip(pts, depth):
        c = cv2.applyColorMap(np.uint8([[min(255, int(255 * z / 8.0))]]), cv2.COLORMAP_TURBO)[0, 0].tolist()
        cv2.circle(vis, (int(x), int(y)), 3, c, -1)
    for (x, y), z in list(zip(pts, depth))[::8]:
        org = (int(x) + 5, int(y) - 5)
        cv2.putText(vis, f'{z:.1f}m', org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(vis, f'{z:.1f}m', org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return vis


class LivePlot:
    """One window, three panels: camera with distances to tracked points | 3D trajectory | 2D top view."""

    def __init__(self, title='stereo VO'):
        import matplotlib
        import matplotlib.pyplot as plt
        if matplotlib.get_backend().lower() == 'agg':
            print('WARNING: matplotlib has no window backend in this env (agg) - nothing will be shown. '
                  'Run with ~/py_envs/py_env1/bin/python (Qt backend).')
        self.plt = plt
        plt.ion()
        self.fig = plt.figure(figsize=(19, 6))
        self.fig.canvas.manager.set_window_title(title)
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])
        self.ax_img = self.fig.add_subplot(gs[0])
        self.ax_3d = self.fig.add_subplot(gs[1], projection='3d')
        self.ax_2d = self.fig.add_subplot(gs[2])
        self.fig.tight_layout()
        plt.show(block=False)

    def update(self, info, est, gt=None):
        """est, gt: (N,3) positions in the ground-truth world frame (gt may be None)."""
        est = np.asarray(est)
        gt = None if gt is None or len(gt) == 0 else np.asarray(gt)

        self.ax_img.clear(); self.ax_img.axis('off')
        self.ax_img.imshow(draw_depth(info['gray'], info['pts_2d'], info['depth'])[:, :, ::-1])
        d = info['depth']
        self.ax_img.set_title(f"tracked {info['tracked']}  inliers {info['inliers']}"
                              f"{'  KEYFRAME' if info['keyframe'] else ''}{'  LOST' if info['lost'] else ''}\n"
                              f"distance to points: {d.min():.2f}-{d.max():.2f} m (median {np.median(d):.2f})"
                              if len(d) else 'no points', fontsize=9)

        self.ax_3d.clear()
        if gt is not None:
            self.ax_3d.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'k-', lw=1.2, label='GT')
        self.ax_3d.plot(est[:, 0], est[:, 1], est[:, 2], '-', color='tab:blue', lw=1.2, label='VO')
        self.ax_3d.scatter(*est[-1], color='tab:red', s=25)
        self.ax_3d.set_xlabel('x [m]'); self.ax_3d.set_ylabel('y [m]'); self.ax_3d.set_zlabel('z [m]')
        self.ax_3d.legend(fontsize=8); self.ax_3d.set_title('3D trajectory', fontsize=10)

        self.ax_2d.clear()
        if gt is not None:
            self.ax_2d.plot(gt[:, 0], gt[:, 1], 'k-', lw=1.2, label='GT')
        self.ax_2d.plot(est[:, 0], est[:, 1], '-', color='tab:blue', lw=1.2, label='VO')
        self.ax_2d.plot(*est[-1, :2], 'o', color='tab:red', ms=5)
        title = 'top view (x-y)'
        if gt is not None:
            title += f'   error now {np.linalg.norm(est[-1] - gt[-1]):.3f} m'
        self.ax_2d.set_title(title, fontsize=10)
        self.ax_2d.axis('equal'); self.ax_2d.grid(alpha=.3); self.ax_2d.legend(fontsize=8)
        self.ax_2d.set_xlabel('x [m]'); self.ax_2d.set_ylabel('y [m]')
        self.plt.pause(0.001)

    def close(self):
        if self.plt.fignum_exists(self.fig.number):
            self.plt.ioff(); self.plt.show()        # keep the window open until it is closed

def run(cfg, verbose=True):
    calib = StereoCalib(cfg.dataset_path)
    pairs = stereo_pairs(cfg.dataset_path)
    gt = GroundTruth(cfg.dataset_path)
    pairs = [p for p in pairs if gt.covers(p[0])][cfg.start_frame:]
    if cfg.max_frames:
        pairs = pairs[:cfg.max_frames]
    name = f'{seq_name(cfg.dataset_path)}_{cfg.detector}'
    os.makedirs(cfg.out_dir, exist_ok=True)

    vo = StereoVO(cfg, calib)
    row_off = 0.0
    if cfg.self_calibrate:
        row_off = vo.rect.self_calibrate([(cv2.imread(fl, cv2.IMREAD_GRAYSCALE), cv2.imread(fr, cv2.IMREAD_GRAYSCALE))
                                          for _, fl, fr in pairs[:20]])
    if verbose:
        print(f'{name}: {len(pairs)} stereo pairs, baseline {calib.baseline:.4f} m, '
              f'rectified f={vo.rect.f:.1f}px, residual row offset {row_off:+.2f}px')
    live = LivePlot(name) if cfg.plot_live else None

    ts_list, est_T, gt_T, depth_samples, n_kf = [], [], [], [], 0
    T_align = None
    t0 = time.time()
    for k, (ts, fl, fr) in enumerate(pairs):
        info = vo.process(cv2.imread(fl, cv2.IMREAD_GRAYSCALE), cv2.imread(fr, cv2.IMREAD_GRAYSCALE))
        T_gt = gt.pose(ts)
        if T_align is None:
            T_align = T_gt @ inv_T(info['T_WB'])
        ts_list.append(ts); est_T.append(info['T_WB']); gt_T.append(T_gt)
        if info['keyframe']:
            n_kf += 1
            if len(info['kf_points']):
                depth_samples.append((T_gt @ vo.T_B_C, info['kf_points']))
        if verbose and k % 100 == 0:
            print(f"  frame {k:5d}  tracked {info['tracked']:4d}  inliers {info['inliers']:4d}  kf {n_kf}")
        if live and k % cfg.plot_every == 0:
            live.update(info, [(T_align @ T)[:3, 3] for T in est_T], [T[:3, 3] for T in gt_T])
    fps = len(pairs) / (time.time() - t0)

    est_T, gt_T = np.array(est_T), np.array(gt_T)
    est_xyz, gt_xyz = est_T[:, :3, 3], gt_T[:, :3, 3]
    res = {
        'sequence': seq_name(cfg.dataset_path), 'detector': cfg.detector, 'frames': len(pairs),
        'keyframes': n_kf, 'lost_frames': vo.n_lost, 'fps': fps, 'row_offset_px': vo.rect.row_offset,
        'ate_se3': ev.ate(est_xyz, gt_xyz, with_scale=False),
        'ate_sim3': ev.ate(est_xyz, gt_xyz, with_scale=True),
        'anchored': ev.anchored(est_T, gt_T),
        'rpe_1s': ev.rpe(est_T, gt_T, delta=20),
        'depth_vs_laser': ev.depth_vs_laser(depth_samples, load_pointcloud(cfg.dataset_path)),
    }
    ev.save_tum(os.path.join(cfg.out_dir, f'{name}_est.tum'), ts_list, est_T)
    ev.save_tum(os.path.join(cfg.out_dir, f'{name}_gt.tum'), ts_list, gt_T)
    stats = (f"ATE(SE3) {res['ate_se3']['rmse']:.3f} m | drift {res['anchored']['drift_pct']:.2f}% "
             f"| Sim3 scale {res['ate_sim3']['scale']:.3f}")
    ev.plot_results(cfg.out_dir, name, gt_xyz, res['ate_se3'], res['anchored'], stats)
    summary = summarize(res)
    with open(os.path.join(cfg.out_dir, f'{name}_metrics.json'), 'w') as f:
        json.dump({**summary, 'config': dataclasses.asdict(cfg)}, f, indent=2)
    if verbose:
        print(json.dumps(summary, indent=2))
    if live:
        live.close()
    return summary


def run_video(cfg, left_video, right_video, verbose=True):
    """Same pipeline on a synchronised stereo VIDEO pair (frame i of left == frame i of right).
    Calibration comes from cfg.dataset_path; there is no ground truth, so only the trajectory is saved."""
    calib = StereoCalib(cfg.dataset_path)
    cap_l, cap_r = cv2.VideoCapture(left_video), cv2.VideoCapture(right_video)
    if not (cap_l.isOpened() and cap_r.isOpened()):
        raise IOError(f'cannot open {left_video} / {right_video}')
    fps_video = cap_l.get(cv2.CAP_PROP_FPS) or 20.0
    for cap in (cap_l, cap_r):
        cap.set(cv2.CAP_PROP_POS_FRAMES, cfg.start_frame)

    def frames():
        k = 0
        while not cfg.max_frames or k < cfg.max_frames:
            ok_l, fl = cap_l.read(); ok_r, fr = cap_r.read()
            if not (ok_l and ok_r):
                return
            if fl.ndim == 3:
                fl, fr = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            yield fl, fr
            k += 1

    name = os.path.splitext(os.path.basename(left_video))[0] + f'_{cfg.detector}'
    os.makedirs(cfg.out_dir, exist_ok=True)
    vo = StereoVO(cfg, calib)
    gen = frames()
    first = [f for _, f in zip(range(20), gen)]
    if cfg.self_calibrate:
        off = vo.rect.self_calibrate(first)
        if verbose:
            print(f'{name}: residual row offset {off:+.2f}px')
    live = LivePlot(name) if cfg.plot_live else None
    ts_list, est_T, t0 = [], [], time.time()
    for k, (fl, fr) in enumerate(__import__('itertools').chain(first, gen)):
        info = vo.process(fl, fr)
        ts_list.append(int((cfg.start_frame + k) / fps_video * 1e9)); est_T.append(info['T_WB'])
        if verbose and k % 100 == 0:
            print(f"  frame {k:5d}  tracked {info['tracked']:4d}  inliers {info['inliers']:4d}  lost {vo.n_lost}")
        if live and k % cfg.plot_every == 0:
            T0 = inv_T(est_T[0])
            live.update(info, [(T0 @ T)[:3, 3] for T in est_T])
    est_T = np.array(est_T)
    xyz = np.array([(inv_T(est_T[0]) @ T)[:3, 3] for T in est_T])   # start at the origin
    ev.save_tum(os.path.join(cfg.out_dir, f'{name}_est.tum'), ts_list, est_T)
    from matplotlib.figure import Figure
    fig = Figure(figsize=(12, 5.5))
    ax = fig.subplots(1, 2)
    for a, (i, j) in zip(ax, [(0, 1), (0, 2)]):
        a.plot(xyz[:, i], xyz[:, j]); a.axis('equal'); a.grid(alpha=.3)
        a.set_xlabel('xyz'[i] + ' [m]'); a.set_ylabel('xyz'[j] + ' [m]')
    length = float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
    fig.suptitle(f'{name}: {len(xyz)} frames, path length {length:.1f} m, lost {vo.n_lost} (body frame of first pose)')
    fig.tight_layout(); fig.savefig(os.path.join(cfg.out_dir, f'{name}_trajectory.png'), dpi=130)
    if verbose:
        print(f'{len(xyz)} frames, {len(xyz) / (time.time() - t0):.0f} fps, path length {length:.1f} m, '
              f'lost {vo.n_lost} -> {cfg.out_dir}/{name}_est.tum, {name}_trajectory.png')
    if live:
        live.close()


def summarize(res):
    out = {}
    for k, v in res.items():
        if isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
        else:
            out[k] = v
    return out


def compare(base_cfg, seqs, detectors):
    rows = []
    for seq in seqs:
        for det in detectors:
            cfg = dataclasses.replace(base_cfg, dataset_path=seq, detector=det)
            try:
                r = run(cfg, verbose=False)
            except Exception as e:
                print(f'{seq} {det}: FAILED ({e})', flush=True)
                continue
            rows.append(r)
            dl = r['depth_vs_laser']
            print(f"{r['sequence']:12s} {det:5s} ATE {r['ate_se3']['rmse']:.3f} m  "
                  f"drift {r['anchored']['drift_pct']:5.2f}%  RPE1s {r['rpe_1s']['trans_rmse']:.3f} m "
                  f"{r['rpe_1s']['rot_rmse_deg']:.2f}deg  scale {r['ate_sim3']['scale']:.3f}  "
                  f"lost {r['lost_frames']}  laser-med {dl['median'] if dl else float('nan'):.3f} m  "
                  f"{r['fps']:.0f} fps", flush=True)
    md = ['| sequence | detector | ATE SE3 [m] | drift [%] | RPE 1s [m] | RPE 1s [deg] | Sim3 scale '
          '| lost frames | depth err vs laser (median) [m] | fps |', '|' + '---|' * 10]
    for r in rows:
        dl = r['depth_vs_laser']
        md.append(f"| {r['sequence']} | {r['detector']} | {r['ate_se3']['rmse']:.3f} | "
                  f"{r['anchored']['drift_pct']:.2f} | {r['rpe_1s']['trans_rmse']:.3f} | "
                  f"{r['rpe_1s']['rot_rmse_deg']:.2f} | {r['ate_sim3']['scale']:.3f} | {r['lost_frames']} | "
                  f"{dl['median']:.3f} | {r['fps']:.0f} |" if dl else
                  f"| {r['sequence']} | {r['detector']} | {r['ate_se3']['rmse']:.3f} | "
                  f"{r['anchored']['drift_pct']:.2f} | {r['rpe_1s']['trans_rmse']:.3f} | "
                  f"{r['rpe_1s']['rot_rmse_deg']:.2f} | {r['ate_sim3']['scale']:.3f} | {r['lost_frames']} | "
                  f"n/a | {r['fps']:.0f} |")
    path = os.path.join(base_cfg.out_dir, 'comparison.md')
    with open(path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f'\nsaved {path}')


def parse_args(argv=None):
    d = VOConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seq', default=d.dataset_path)
    p.add_argument('--detector', default=d.detector, choices=['orb', 'sift', 'gftt'])
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--frames', type=int, default=0, help='0 = whole sequence')
    p.add_argument('--no-live', action='store_true', help='no live window (faster; plots are still saved)')
    p.add_argument('--out', default=d.out_dir)
    p.add_argument('--compare', action='store_true', help='run every detector on every sequence')
    p.add_argument('--video', nargs=2, metavar=('LEFT', 'RIGHT'),
                   help='stereo video pair instead of EuRoC folders (calibration from --seq)')
    return p.parse_args(argv)


if __name__ == '__main__':
    a = parse_args()
    cfg = VOConfig(dataset_path=a.seq, detector=a.detector, start_frame=a.start,
                   max_frames=a.frames, plot_live=not (a.no_live or a.compare), out_dir=a.out)
    if a.video:
        run_video(cfg, *a.video)
    elif a.compare:
        compare(cfg, [s for s in SEQUENCES if os.path.isdir(s)], ['orb', 'sift', 'gftt'])
    else:
        run(cfg)
