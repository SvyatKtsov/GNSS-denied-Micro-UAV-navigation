"""Stereo front-end: rectification -> detection -> description -> L/R matching -> triangulation."""
import cv2
import numpy as np

from .geometry import to_T


class Rectifier:
    def __init__(self, calib):
        w, h = calib.size
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            calib.K0, calib.D0, calib.K1, calib.D1, (w, h), calib.R, calib.t,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)   # alpha=0: no black invalid borders
        self.maps_l = cv2.initUndistortRectifyMap(calib.K0, calib.D0, R1, P1, (w, h), cv2.CV_32FC1)
        self.maps_r = cv2.initUndistortRectifyMap(calib.K1, calib.D1, R2, P2, (w, h), cv2.CV_32FC1)
        self.K = P1[:3, :3].copy()
        self.f, self.cx, self.cy = P1[0, 0], P1[0, 2], P1[1, 2]
        self.baseline = -P2[0, 3] / P2[0, 0]
        assert abs(P2[1, 3]) < 1e-6 and self.baseline > 0, 'stereo rig must be horizontal, left = cam0'
        # X_rect = R1 @ X_cam0  ->  pose of the original cam0 w.r.t. the rectified left camera
        self.T_rect_C0 = to_T(R1, np.zeros(3))

        self.row_offset = 0.0

    def __call__(self, img_l, img_r):
        return (cv2.remap(img_l, *self.maps_l, cv2.INTER_LINEAR),
                cv2.remap(img_r, *self.maps_r, cv2.INTER_LINEAR))

    def self_calibrate(self, raw_pairs, min_matches=200):
        """Online check of the factory calibration: in a perfectly rectified pair, y_l - y_r = 0.
        A constant residual = small relative pitch between the cameras (e.g. ~1.8 px in V2_01,
        the rig moved after calibration). Estimate it robustly and shift the right image."""
        orb, bf, dy = cv2.ORB_create(2000), cv2.BFMatcher(cv2.NORM_HAMMING), []
        for img_l, img_r in raw_pairs:
            gl, gr = self(img_l, img_r)
            kl, dl = orb.detectAndCompute(gl, None)
            kr, dr = orb.detectAndCompute(gr, None)
            if dl is None or dr is None:
                continue
            for m in bf.knnMatch(dl, dr, k=2):
                if len(m) == 2 and m[0].distance < 0.7 * m[1].distance:
                    d = kl[m[0].queryIdx].pt[1] - kr[m[0].trainIdx].pt[1]
                    if abs(d) < 8 and kl[m[0].queryIdx].pt[0] > kr[m[0].trainIdx].pt[0]:
                        dy.append(d)
        if len(dy) < min_matches:
            return 0.0
        off = float(np.median(dy))
        if abs(off) > 0.3:
            self.maps_r = (self.maps_r[0], self.maps_r[1] - off)
            self.row_offset += off
        return off


def bucket(kps, shape, grid, per_cell):
    """Keep the strongest `per_cell` keypoints in every grid cell -> uniform spatial coverage,
    which makes PnP far better conditioned than many clustered points."""
    gh, gw = grid
    h, w = shape[:2]
    cells = {}
    for kp in kps:
        key = (min(int(kp.pt[1] * gh / h), gh - 1), min(int(kp.pt[0] * gw / w), gw - 1))
        cells.setdefault(key, []).append(kp)
    out = []
    for c in cells.values():
        out.extend(sorted(c, key=lambda k: -k.response)[:per_cell])
    return out


class FeatureExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.kind = cfg.detector.lower()
        if self.kind == 'orb':
            self.det = cv2.ORB_create(nfeatures=cfg.n_features, scaleFactor=1.2, nlevels=8, fastThreshold=10)
            self.norm = cv2.NORM_HAMMING
        elif self.kind == 'sift':
            self.det = cv2.SIFT_create(nfeatures=cfg.n_features)
            self.norm = cv2.NORM_L2
        elif self.kind == 'gftt':
            self.det, self.norm = None, None
        else:
            raise ValueError(f'unknown detector {cfg.detector}')

    def detect(self, gray, mask=None, n=None):
        n = n or self.cfg.n_features
        if self.kind == 'gftt':
            pts = cv2.goodFeaturesToTrack(gray, n, 0.01, 7, mask=mask, blockSize=5)
            if pts is None:
                return []
            pts = cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            return [cv2.KeyPoint(float(x), float(y), 7, response=1.0) for x, y in pts.reshape(-1, 2)]
        return list(self.det.detect(gray, mask))

    def describe(self, gray, kps):
        if self.kind == 'gftt' or not kps:
            return kps, None
        kps, desc = self.det.compute(gray, kps)
        return list(kps), desc


def _lk(img_a, img_b, pts_a, pts_b=None, win=21, levels=3):
    flags = 0
    if pts_b is not None:
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW
        pts_b = pts_b.reshape(-1, 1, 2).astype(np.float32).copy()
    out, st, _ = cv2.calcOpticalFlowPyrLK(
        img_a, img_b, pts_a.reshape(-1, 1, 2).astype(np.float32), pts_b,
        winSize=(win, win), maxLevel=levels, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    return out.reshape(-1, 2), st.ravel().astype(bool)


def track_lk(prev, curr, pts, cfg):
    """Temporal tracking with a forward-backward consistency check."""
    if len(pts) == 0:
        return pts, np.zeros(0, bool)
    fwd, st1 = _lk(prev, curr, pts, win=cfg.lk_win, levels=cfg.lk_levels)
    bwd, st2 = _lk(curr, prev, fwd, win=cfg.lk_win, levels=cfg.lk_levels)
    h, w = curr.shape[:2]
    inside = (fwd[:, 0] >= 0) & (fwd[:, 0] < w - 1) & (fwd[:, 1] >= 0) & (fwd[:, 1] < h - 1)
    good = st1 & st2 & inside & (np.linalg.norm(pts - bwd, axis=1) < cfg.fb_thresh)
    return fwd, good


class StereoMatcher:
    """Matches features of a RECTIFIED pair. In a rectified pair the epipolar line of a left
    pixel is the same row in the right image and disparity d = x_l - x_r > 0, so we only allow
    candidates satisfying that (instead of estimating a fundamental matrix with RANSAC)."""

    def __init__(self, cfg, rect):
        self.cfg, self.rect = cfg, rect
        self.fx = FeatureExtractor(cfg)
        # disparity of the closest allowed point
        self.max_disp = rect.f * rect.baseline / cfg.min_depth

    def _geom_ok(self, pl, pr):
        d = pl[:, 0] - pr[:, 0]
        return (np.abs(pl[:, 1] - pr[:, 1]) <= self.cfg.max_row_diff) & \
               (d >= self.cfg.min_disparity) & (d <= self.max_disp)

    def match(self, gray_l, gray_r, mask_l=None):
        """Returns pts_l (N,2), pts_r (N,2) of verified correspondences."""
        cfg = self.cfg
        kps_l = bucket(self.fx.detect(gray_l, mask_l), gray_l.shape, cfg.grid, cfg.per_cell)
        if not kps_l:
            return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

        if self.fx.kind == 'gftt':
            pl = np.float32([k.pt for k in kps_l])
            pr, st = _lk(gray_l, gray_r, pl, win=cfg.lk_win, levels=cfg.lk_levels)
            back, st2 = _lk(gray_r, gray_l, pr, win=cfg.lk_win, levels=cfg.lk_levels)
            ok = st & st2 & (np.linalg.norm(back - pl, axis=1) < cfg.fb_thresh)
            pl, pr = pl[ok], pr[ok]
        else:
            kps_l, des_l = self.fx.describe(gray_l, kps_l)
            kps_r, des_r = self.fx.describe(gray_r, self.fx.detect(gray_r, n=2 * cfg.n_features))
            if des_l is None or des_r is None or len(kps_r) < 2:
                return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
            xl, xr = np.float32([k.pt for k in kps_l]), np.float32([k.pt for k in kps_r])
            # candidate mask: same row (+-tol) and positive, bounded disparity
            dy = np.abs(xl[:, None, 1] - xr[None, :, 1])
            dx = xl[:, None, 0] - xr[None, :, 0]
            mask = ((dy <= cfg.max_row_diff + 1.0) & (dx >= cfg.min_disparity - 1.0) &
                    (dx <= self.max_disp)).astype(np.uint8)
            bf = cv2.BFMatcher(self.fx.norm)
            fwd = bf.knnMatch(des_l, des_r, k=2, mask=mask)
            bwd = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(des_r, des_l, k=1, mask=mask.T) if m}
            pairs = []
            for m in fwd:
                if not m:
                    continue
                if len(m) == 2 and m[0].distance >= cfg.ratio * m[1].distance:
                    continue                              # ambiguous (repetitive texture)
                if bwd.get(m[0].trainIdx) != m[0].queryIdx:
                    continue                              # not mutual best match
                pairs.append((m[0].queryIdx, m[0].trainIdx))
            if not pairs:
                return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
            qi, ti = map(np.array, zip(*pairs))
            pl, pr = xl[qi], xr[ti]

            if cfg.subpixel_refine:
                # keypoints from coarse pyramid levels are only accurate to ~2^octave px, and the depth
                # error grows as Z^2/(f*B) * disparity_error -> refine x_r to sub-pixel precision with LK
                ref, st = _lk(gray_l, gray_r, pl, pr, win=11, levels=1)
                ok = st & (np.linalg.norm(ref - pr, axis=1) < 3.0)
                pr = np.where(ok[:, None], ref, pr)

        ok = self._geom_ok(pl, pr)
        return pl[ok].astype(np.float32), pr[ok].astype(np.float32)

    def triangulate(self, pl, pr):
        """Rectified stereo: Z = f*B/d, X = (u-cx)Z/f, Y = (v-cy)Z/f  (left rectified camera frame).
        Identical to cv2.triangulatePoints(P1, P2, ...) for a rectified pair, but exact and cheaper."""
        r = self.rect
        d = pl[:, 0] - pr[:, 0]
        Z = r.f * r.baseline / d
        X = (pl[:, 0] - r.cx) * Z / r.f
        Y = (pl[:, 1] - r.cy) * Z / r.f
        P = np.stack([X, Y, Z], axis=1)
        ok = (Z > self.cfg.min_depth) & (Z < self.cfg.max_depth)
        return P[ok], pl[ok], ok
