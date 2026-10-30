"""Trajectory evaluation (ATE / RPE / drift) and plots."""
import os
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import umeyama, inv_T, transform, rotation_angle


def save_tum(path, ts_ns, Ts):
    """TUM format (timestamp[s] tx ty tz qx qy qz qw) - readable by `evo_ape` / `evo_rpe`."""
    with open(path, 'w') as f:
        for ts, T in zip(ts_ns, Ts):
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            f.write(f'{ts * 1e-9:.9f} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} '
                    f'{q[0]:.7f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f}\n')


def ate(est_xyz, gt_xyz, with_scale=False):
    s, R, t = umeyama(est_xyz, gt_xyz, with_scale)
    aligned = s * est_xyz @ R.T + t
    err = np.linalg.norm(aligned - gt_xyz, axis=1)
    return {'rmse': float(np.sqrt(np.mean(err ** 2))), 'mean': float(err.mean()),
            'median': float(np.median(err)), 'max': float(err.max()), 'scale': float(s),
            'errors': err, 'aligned': aligned}


def anchored(est_T, gt_T):
    """Align only with the FIRST pose (what a drone knows at the moment GNSS is lost) and report
    how far it drifts. This is the honest GNSS-denied number; ATE with a global fit is optimistic."""
    T_align = gt_T[0] @ inv_T(est_T[0])
    est = np.array([(T_align @ T)[:3, 3] for T in est_T])
    gt = np.array([T[:3, 3] for T in gt_T])
    err = np.linalg.norm(est - gt, axis=1)
    length = float(np.sum(np.linalg.norm(np.diff(gt, axis=0), axis=1)))
    return {'final_err': float(err[-1]), 'rmse': float(np.sqrt(np.mean(err ** 2))),
            'length': length, 'drift_pct': 100.0 * err[-1] / max(length, 1e-9),
            'errors': err, 'aligned': est}


def rpe(est_T, gt_T, delta):
    """Relative pose error over `delta` frames: translation RMSE [m] and rotation RMSE [deg]."""
    et, er = [], []
    for i in range(len(est_T) - delta):
        E = inv_T(inv_T(gt_T[i]) @ gt_T[i + delta]) @ (inv_T(est_T[i]) @ est_T[i + delta])
        et.append(np.linalg.norm(E[:3, 3]))
        er.append(np.degrees(rotation_angle(E[:3, :3])))
    if not et:
        return {'trans_rmse': float('nan'), 'rot_rmse_deg': float('nan')}
    return {'trans_rmse': float(np.sqrt(np.mean(np.square(et)))),
            'rot_rmse_deg': float(np.sqrt(np.mean(np.square(er))))}


def depth_vs_laser(samples, cloud, max_pts=200_000):
    """Accuracy of TRIANGULATED points, independent of VO drift: each keyframe's points are placed
    into the world with the ground-truth camera pose and compared to the Leica laser scan
    (nearest-neighbour distance). samples: list of (T_W_C_gt, pts_cam)."""
    from scipy.spatial import cKDTree
    if cloud is None or not samples:
        return None
    if len(cloud) > max_pts:
        cloud = cloud[np.random.default_rng(0).choice(len(cloud), max_pts, replace=False)]
    tree = cKDTree(cloud)
    pts = np.vstack([transform(T, P) for T, P in samples])
    depth = np.concatenate([P[:, 2] for _, P in samples])
    d, _ = tree.query(pts)
    bins = [0, 2, 4, 6, 10, np.inf]
    by_depth = {f'{bins[i]}-{bins[i + 1]}m': float(np.median(d[(depth >= bins[i]) & (depth < bins[i + 1])]))
                for i in range(len(bins) - 1) if np.any((depth >= bins[i]) & (depth < bins[i + 1]))}
    return {'n': int(len(d)), 'median': float(np.median(d)), 'p90': float(np.percentile(d, 90)),
            'by_depth_median': by_depth}


def plot_results(out_dir, name, gt_xyz, ate_res, anch_res, stats):
    from matplotlib.figure import Figure          # backend-independent: doesn't touch the live window
    fig = Figure(figsize=(18, 5.5))
    ax = fig.subplots(1, 3)
    for a, (i, j, lab) in zip(ax[:2], [(0, 1, 'top view (x-y)'), (0, 2, 'side view (x-z)')]):
        a.plot(gt_xyz[:, i], gt_xyz[:, j], 'k-', lw=1.5, label='ground truth')
        a.plot(ate_res['aligned'][:, i], ate_res['aligned'][:, j], '-', lw=1, label='VO (SE3 aligned)')
        a.plot(anch_res['aligned'][:, i], anch_res['aligned'][:, j], '--', lw=1, label='VO (first-pose anchored)')
        a.set_title(lab); a.axis('equal'); a.grid(alpha=.3); a.legend(fontsize=8)
        a.set_xlabel('xyz'[i] + ' [m]'); a.set_ylabel('xyz'[j] + ' [m]')
    t = np.arange(len(gt_xyz)) / 20.0
    ax[2].plot(t, ate_res['errors'], label=f"ATE (SE3)  rmse={ate_res['rmse']:.3f} m")
    ax[2].plot(t, anch_res['errors'], label=f"anchored  drift={anch_res['drift_pct']:.2f}%")
    ax[2].set_xlabel('time [s]'); ax[2].set_ylabel('position error [m]'); ax[2].grid(alpha=.3)
    ax[2].legend(fontsize=8); ax[2].set_title('error over time')
    fig.suptitle(f"{name} | {stats}")
    fig.tight_layout()
    path = os.path.join(out_dir, f'{name}_trajectory.png')
    fig.savefig(path, dpi=130)
    return path
