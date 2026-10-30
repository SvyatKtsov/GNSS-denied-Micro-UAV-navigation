"""live window: camera + depth | 3d trajectory | 2d top view."""
import time
import cv2
import numpy as np

DEPTH_RANGE = (0.5, 6.0)      # m, colour scale: near = red ... far = blue
# reversed turbo lut (rgb): index 0 (near) = red, 255 (far) = blue - same map as matplotlib 'turbo_r'
_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[::-1, 0, ::-1].copy()


def _depth_idx(z):
    lo, hi = DEPTH_RANGE
    return np.clip((np.asarray(z) - lo) * (255 / (hi - lo)), 0, 255).astype(np.uint8)


def draw_depth(gray, pts, depth, overlay=0.35, sigma=24.0):
    """frame + tracked points coloured by distance + translucent depth map.
    the sparse stereo depths are interpolated by kernel (nadaraya-watson) regression
    Z(x) = sum_i w_i(x) Z_i / sum_i w_i(x), w_i = gaussian(|x - x_i|, sigma); it fades out away from
    tracked points, so untextured regions show the plain frame."""
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if len(pts) == 0:
        return vis
    h, w = gray.shape[:2]
    xy = np.round(pts).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, w - 1); xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)

    if overlay > 0:
        s, k = 4, sigma / 4                           # 1/4 resolution: 16x cheaper, the map is smooth anyway
        num, den = np.zeros((2, (h + s - 1) // s, (w + s - 1) // s), np.float32)
        np.add.at(num, (xy[:, 1] // s, xy[:, 0] // s), depth.astype(np.float32))
        np.add.at(den, (xy[:, 1] // s, xy[:, 0] // s), 1.0)
        num, den = cv2.GaussianBlur(num, (0, 0), k), cv2.GaussianBlur(den, (0, 0), k)
        conf = np.clip(den / (0.5 / (2 * np.pi * k * k)), 0, 1)   # ~1 within ~sigma of a point, 0 far away
        col = cv2.resize(_LUT[_depth_idx(num / np.maximum(den, 1e-6))].astype(np.float32), (w, h))
        alpha = cv2.resize(overlay * conf, (w, h))[..., None]
        vis = (vis * (1 - alpha) + col * alpha).astype(np.uint8)

    for p, c in zip(xy.tolist(), _LUT[_depth_idx(depth)].tolist()):
        cv2.circle(vis, p, 6, (20, 20, 20), -1, cv2.LINE_AA)      # dark rim: readable on any background
        cv2.circle(vis, p, 5, c, -1, cv2.LINE_AA)
    for (x, y), z in list(zip(xy.tolist(), depth))[::8]:
        for col, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(vis, f'{z:.1f}m', (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th, cv2.LINE_AA)
    return vis


class LivePlot:
    """fast live view:
      - artists are created once, updates only swap their data (no axes.clear() + re-plot);
      - blitting: axes/ticks/grid/colorbar are rendered once and cached, an update restores that bitmap
        and draws only the image, lines and titles; a full redraw happens only when the trajectory
        leaves the axis box (it grows in steps) or the window is resized/rotated;
      - redraws are throttled by wall-clock time (max_fps, and <= ~1/3 of the time), the vo runs in between."""

    def __init__(self, title='stereo VO', max_fps=10.0):
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        self.plt, self.period, self.t_next = plt, 1.0 / max_fps, 0.0
        plt.ion()
        self.fig = plt.figure(figsize=(19, 6))
        self.fig.canvas.manager.set_window_title(title)
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])
        self.ax_img = self.fig.add_subplot(gs[0])
        self.ax_3d = self.fig.add_subplot(gs[1], projection='3d')
        self.ax_2d = self.fig.add_subplot(gs[2])

        self.ax_img.axis('off')
        self.im = None                                  # created on the first frame (needs the image size)
        self.img_title = self.ax_img.set_title('', fontsize=9)
        cb = self.fig.colorbar(ScalarMappable(Normalize(*DEPTH_RANGE), 'turbo_r'), ax=self.ax_img,
                               fraction=0.025, pad=0.01)
        cb.set_label('distance [m]  (near = red, far = blue)', fontsize=8); cb.ax.tick_params(labelsize=8)

        style = [dict(color='k', lw=1.2, label='GT'), dict(color='tab:blue', lw=1.2, label='VO'),
                 dict(color='tab:red', marker='o', ms=5, ls='')]
        self.gt3, self.vo3, self.cur3 = [self.ax_3d.plot([], [], [], **s)[0] for s in style]
        self.gt2, self.vo2, self.cur2 = [self.ax_2d.plot([], [], **s)[0] for s in style]
        self.ax_3d.set(xlabel='x [m]', ylabel='y [m]', zlabel='z [m]')
        self.ax_3d.set_title('3D trajectory', fontsize=10)
        self.ax_2d.set(xlabel='x [m]', ylabel='y [m]', aspect='equal', adjustable='box')
        self.ax_2d.grid(alpha=.3)
        self.title2 = self.ax_2d.set_title('top view (x-y)', fontsize=10)
        for a in (self.ax_3d, self.ax_2d):
            a.legend(fontsize=8)

        self.animated = [self.img_title, self.gt3, self.vo3, self.cur3, self.gt2, self.vo2, self.cur2, self.title2]
        for art in self.animated:
            art.set_animated(True)                      # excluded from the cached background
        self.bg, self.box = None, None
        self.fig.canvas.mpl_connect('draw_event', self._on_draw)
        self.fig.tight_layout()
        plt.show(block=False)

    def is_open(self):
        return self.plt.fignum_exists(self.fig.number)

    def due(self):
        return time.perf_counter() >= self.t_next and self.is_open()

    def _on_draw(self, _):
        # after any full draw (ours, resize, 3d rotation with the mouse): re-cache the background
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for art in self.animated:
            self.fig.draw_artist(art)

    def _fit_limits(self, xyz):
        # grow the axis box in steps only when the trajectory leaves it -> rare full redraws
        lo, hi = xyz.min(0), xyz.max(0)
        if self.box is not None and np.all(lo >= self.box[0]) and np.all(hi <= self.box[1]):
            return False
        pad = max(0.5, 0.2 * float((hi - lo).max()))
        lo, hi = self.box = (lo - pad, hi + pad)
        self.ax_3d.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]))
        c, r = (lo[:2] + hi[:2]) / 2, float((hi[:2] - lo[:2]).max()) / 2     # square x-y window
        self.ax_2d.set(xlim=(c[0] - r, c[0] + r), ylim=(c[1] - r, c[1] + r))
        return True

    def update(self, info, est, gt=None):
        """est, gt: (N,3) positions in the ground-truth world frame (gt may be None)."""
        t0 = time.perf_counter()
        rgb = draw_depth(info['gray'], info['pts_2d'], info['depth'])
        full = self.im is None
        if full:
            self.im = self.ax_img.imshow(rgb, animated=True)
            self.animated.insert(0, self.im)
        else:
            self.im.set_data(rgb)
        d = info['depth']
        self.img_title.set_text(
            f"tracked {info['tracked']}  inliers {info['inliers']}"
            f"{'  KEYFRAME' if info['keyframe'] else ''}{'  LOST' if info['lost'] else ''}\n"
            f"distance to points: {d.min():.2f}-{d.max():.2f} m (median {np.median(d):.2f})" if len(d) else 'no points')

        title, both = 'top view (x-y)', est
        for (l3, l2), xyz in (((self.vo3, self.vo2), est), ((self.cur3, self.cur2), est[-1:]),
                              ((self.gt3, self.gt2), gt)):
            if xyz is not None:
                l3.set_data_3d(xyz[:, 0], xyz[:, 1], xyz[:, 2]); l2.set_data(xyz[:, 0], xyz[:, 1])
        if gt is not None:
            title += f'   error now {np.linalg.norm(est[-1] - gt[-1]):.3f} m'
            both = np.vstack([est, gt])
        self.title2.set_text(title)

        c = self.fig.canvas
        if self._fit_limits(both) or full or self.bg is None:
            c.draw()                                    # full redraw, _on_draw re-caches the background
        else:
            c.restore_region(self.bg)
            for art in self.animated:
                self.fig.draw_artist(art)
        c.blit(self.fig.bbox)
        c.flush_events()                                # gui events: resize, rotate, close
        self.t_next = time.perf_counter() + max(self.period, 2 * (time.perf_counter() - t0))

    def close(self):
        for art in self.animated:
            art.set_animated(False)                     # final view = ordinary figure (resizable, savable)
        if self.is_open():
            self.plt.ioff(); self.plt.show()            # keep the window open until it is closed
