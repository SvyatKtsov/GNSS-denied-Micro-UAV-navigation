"""Stereo visual odometry: stereo triangulation on keyframes + LK tracking + PnP for every frame.

Frames:  W = rectified left camera at the first processed frame (VO world),
         C = rectified left camera at the current frame.  T_WC maps C-coordinates to W.
"""
import cv2
import numpy as np

from .features import Rectifier, StereoMatcher, track_lk
from .geometry import inv_T, transform, rvec_tvec_to_T, T_to_rvec_tvec, rotation_angle, orthonormalize, to_T


class StereoVO:
    def __init__(self, cfg, calib):
        self.cfg = cfg
        self.rect = Rectifier(calib)
        self.matcher = StereoMatcher(cfg, self.rect)
        self.K = self.rect.K
        # rectified-left camera -> body (IMU) frame, to compare with EuRoC ground truth
        self.T_C_B = self.rect.T_rect_C0 @ inv_T(calib.T_B_C0)
        self.T_B_C = inv_T(self.T_C_B)

        self.T_WC = np.eye(4)
        self.velocity = np.eye(4)        # T_{C_prev C_curr}, constant-velocity motion model
        self.prev_gray = None
        self.pts_w = np.zeros((0, 3))    # landmarks in W
        self.pts_2d = np.zeros((0, 2), np.float32)   # their pixel positions in the previous frame
        self.kf_age = 0
        self.n_lost = 0

    # ---- keyframe: stereo matching + triangulation -------------------------------------------
    def _add_keyframe(self, gray_l, gray_r):
        mask = np.full(gray_l.shape, 255, np.uint8)
        for x, y in self.pts_2d:                      # don't re-detect where we already track
            cv2.circle(mask, (int(x), int(y)), 10, 0, -1)
        pl, pr = self.matcher.match(gray_l, gray_r, mask)
        P_c, p2d, _ = self.matcher.triangulate(pl, pr)
        self.pts_w = np.vstack([self.pts_w, transform(self.T_WC, P_c)])
        self.pts_2d = np.vstack([self.pts_2d, p2d]).astype(np.float32)
        self.kf_age = 0
        return P_c

    # ---- pose -----------------------------------------------------------------------------------
    def _pnp(self, pts_w, pts_2d):
        cfg = self.cfg
        T_CW_pred = inv_T(self.T_WC @ self.velocity)
        rvec, tvec = T_to_rvec_tvec(T_CW_pred)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            pts_w, pts_2d, self.K, None, rvec, tvec, useExtrinsicGuess=True,
            iterationsCount=cfg.pnp_iters, reprojectionError=cfg.pnp_reproj_err,
            confidence=cfg.pnp_conf, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inl is None or len(inl) < cfg.min_inliers:
            return None, None
        inl = inl.ravel()
        # non-linear (Levenberg-Marquardt) refinement on the inliers, then re-classify inliers
        rvec, tvec = cv2.solvePnPRefineLM(pts_w[inl], pts_2d[inl], self.K, None, rvec, tvec)
        proj, _ = cv2.projectPoints(pts_w, rvec, tvec, self.K, None)
        err = np.linalg.norm(proj.reshape(-1, 2) - pts_2d, axis=1)
        inl = np.flatnonzero(err < cfg.pnp_reproj_err)
        if len(inl) < cfg.min_inliers:
            return None, None
        return inv_T(rvec_tvec_to_T(rvec, tvec)), inl

    def process(self, img_l, img_r):
        """img_l, img_r: raw (distorted) grayscale images. Returns info dict for this frame."""
        gray_l, gray_r = self.rect(img_l, img_r)
        info = {'keyframe': False, 'tracked': 0, 'inliers': 0, 'lost': False}

        if self.prev_gray is None:
            info['kf_points'] = self._add_keyframe(gray_l, gray_r)
            info['keyframe'] = True
        else:
            pts, good = track_lk(self.prev_gray, gray_l, self.pts_2d, self.cfg)
            self.pts_w, self.pts_2d = self.pts_w[good], pts[good].astype(np.float32)
            info['tracked'] = len(self.pts_w)

            T_WC, inl = (None, None)
            if len(self.pts_w) >= self.cfg.min_inliers:
                T_WC, inl = self._pnp(self.pts_w, self.pts_2d)
            if T_WC is not None:
                step = inv_T(self.T_WC) @ T_WC
                # (NaN-safe: a NaN comparison is False, so check finiteness explicitly)
                if not np.all(np.isfinite(step)) or not (np.linalg.norm(step[:3, 3]) <= self.cfg.max_step
                                                         and rotation_angle(step[:3, :3]) <= 0.5):
                    T_WC = None
            if T_WC is None:
                # tracking failure: coast with a decaying motion model and re-initialise the map here
                T_WC, inl = self.T_WC @ self.velocity, np.zeros(0, int)
                info['lost'] = True
                self.n_lost += 1
                rv = cv2.Rodrigues(self.velocity[:3, :3])[0] * 0.8
                self.velocity = to_T(cv2.Rodrigues(rv)[0], self.velocity[:3, 3] * 0.8)
            else:
                self.velocity = inv_T(self.T_WC) @ T_WC
            self.T_WC = orthonormalize(T_WC)
            self.pts_w, self.pts_2d = self.pts_w[inl], self.pts_2d[inl]
            info['inliers'] = len(inl)

            self.kf_age += 1
            if len(self.pts_w) < self.cfg.kf_min_tracked or self.kf_age >= self.cfg.kf_max_age:
                info['kf_points'] = self._add_keyframe(gray_l, gray_r)
                info['keyframe'] = True

        self.prev_gray = gray_l
        info['T_WC'] = self.T_WC.copy()
        info['T_WB'] = self.T_WC @ self.T_C_B           # body pose in VO world
        info['gray'] = gray_l
        info['pts_2d'] = self.pts_2d
        info['depth'] = transform(inv_T(self.T_WC), self.pts_w)[:, 2] if len(self.pts_w) else np.zeros(0)
        return info
