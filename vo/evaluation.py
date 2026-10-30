"""trajectory evaluation (ATE / RPE / drift / depth vs laser scan) and plots."""
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import umeyama, inv_T, transform, rotation_angle

rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))


def save_tum(path, ts_ns, Ts):
    """tum format (timestamp[s] tx ty tz qx qy qz qw), readable by `evo_ape` / `evo_rpe`."""
    with open(path, 'w') as f:
        for ts, T in zip(ts_ns, Ts):
            f.write(f'{ts * 1e-9:.9f} ' + ' '.join(f'{v:.7f}' for v in
                    [*T[:3, 3], *Rotation.from_matrix(T[:3, :3]).as_quat()]) + '\n')


def rpe(est_T, gt_T, delta):
    """relative pose error over `delta` frames: translation rmse [m], rotation rmse [deg]."""
    E = [inv_T(inv_T(gt_T[i]) @ gt_T[i + delta]) @ inv_T(est_T[i]) @ est_T[i + delta]
         for i in range(len(est_T) - delta)]
    if not E:
        return float('nan'), float('nan')
    return rmse([np.linalg.norm(e[:3, 3]) for e in E]), rmse([np.degrees(rotation_angle(e[:3, :3])) for e in E])


def depth_vs_laser(samples, cloud, max_pts=200_000):
    """accuracy of the triangulated points alone (independent of vo drift): each keyframe's points are
    put into the world with the ground-truth camera pose and compared to the leica laser scan
    (nearest-neighbour distance). samples: list of (T_W_C_gt, pts_cam)."""
    from scipy.spatial import cKDTree
    if cloud is None or not samples:
        return None
    if len(cloud) > max_pts:
        cloud = cloud[np.random.default_rng(0).choice(len(cloud), max_pts, replace=False)]
    d, _ = cKDTree(cloud).query(np.vstack([transform(T, P) for T, P in samples]))
    depth = np.concatenate([P[:, 2] for _, P in samples])
    bins = [0, 2, 4, 6, 10, np.inf]
    by_depth = {f'{a}-{b}m': float(np.median(d[(depth >= a) & (depth < b)]))
                for a, b in zip(bins, bins[1:]) if np.any((depth >= a) & (depth < b))}
    return {'n': len(d), 'median': float(np.median(d)), 'p90': float(np.percentile(d, 90)), 'by_depth': by_depth}


def evaluate(est_T, gt_T, samples, cloud, png, name):
    """ATE after se(3) umeyama alignment (stereo is metric, so no scale fit; the sim(3) scale is only
    reported as a check), drift with only the first pose aligned (the gnss-denied case), rpe over 1 s."""
    est, gt = est_T[:, :3, 3], gt_T[:, :3, 3]
    s, R, t = umeyama(est, gt, with_scale=False)
    ate_xyz = est @ R.T + t
    anch_xyz = (gt_T[0] @ inv_T(est_T[0]) @ est_T)[:, :3, 3]
    e_ate, e_anch = np.linalg.norm(ate_xyz - gt, axis=1), np.linalg.norm(anch_xyz - gt, axis=1)
    length = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    rpe_m, rpe_deg = rpe(est_T, gt_T, delta=20)                 # 20 frames = 1 s
    res = {'ate_se3': rmse(e_ate), 'sim3_scale': float(umeyama(est, gt)[0]), 'length_m': length,
           'drift_pct': 100 * float(e_anch[-1]) / max(length, 1e-9), 'anchored_rmse': rmse(e_anch),
           'rpe_1s_m': rpe_m, 'rpe_1s_deg': rpe_deg, 'depth_vs_laser': depth_vs_laser(samples, cloud)}
    plot(png, f"{name} | ATE(SE3) {res['ate_se3']:.3f} m | drift {res['drift_pct']:.2f}% | "
              f"Sim3 scale {res['sim3_scale']:.3f}",
         {'ground truth': gt, 'VO (SE3 aligned)': ate_xyz, 'VO (first-pose anchored)': anch_xyz},
         {f"ATE (SE3)  rmse={res['ate_se3']:.3f} m": e_ate, f"anchored  drift={res['drift_pct']:.2f}%": e_anch})
    return res


def plot(png, title, tracks, errors=None):
    """tracks: {label: (N,3)} -> top view | side view [| errors {label: (N,)} over time]."""
    from matplotlib.figure import Figure          # backend-independent: doesn't touch the live window
    fig = Figure(figsize=(18 if errors else 12, 5.5))
    ax = fig.subplots(1, 3 if errors else 2)
    for a, (i, j, lab) in zip(ax, [(0, 1, 'top view (x-y)'), (0, 2, 'side view (x-z)')]):
        for k, (label, xyz) in enumerate(tracks.items()):
            a.plot(xyz[:, i], xyz[:, j], 'k-' if label == 'ground truth' else ['-', '-', '--'][k % 3],
                   lw=1.5 if label == 'ground truth' else 1, label=label)
        a.set(title=lab, xlabel='xyz'[i] + ' [m]', ylabel='xyz'[j] + ' [m]')
        a.axis('equal'); a.grid(alpha=.3); a.legend(fontsize=8)
    if errors:
        for label, e in errors.items():
            ax[2].plot(np.arange(len(e)) / 20.0, e, label=label)
        ax[2].set(title='error over time', xlabel='time [s]', ylabel='position error [m]')
        ax[2].grid(alpha=.3); ax[2].legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(png, dpi=130)
