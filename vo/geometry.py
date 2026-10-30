import cv2
import numpy as np


def to_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return to_T(R.T, -R.T @ t)


def transform(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def orthonormalize(T):
    """Project the rotation block back onto SO(3); chained 4x4 products slowly lose orthogonality."""
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return to_T(R, T[:3, 3])


def rvec_tvec_to_T(rvec, tvec):
    return to_T(cv2.Rodrigues(rvec)[0], tvec)


def T_to_rvec_tvec(T):
    return cv2.Rodrigues(T[:3, :3])[0], T[:3, 3].reshape(3, 1).copy()


def rotation_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def umeyama(src, dst, with_scale=True):
    """Least-squares similarity (s, R, t) minimising sum ||dst - (s R src + t)||^2  (Umeyama, 1991).

    For STEREO VO the scale is observable (metric baseline), so the honest evaluation is
    with_scale=False (SE(3)). Sim(3) alignment hides scale errors and is meant for monocular VO.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t
