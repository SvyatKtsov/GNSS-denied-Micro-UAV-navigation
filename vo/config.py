from dataclasses import dataclass


@dataclass
class VOConfig:
    dataset_path: str = 'data/euroc/V2_01_easy/mav0'
    start_frame: int = 0
    max_frames: int = 0             # 0 = till the end of the sequence

    # feature detection (stereo matching on keyframes)
    detector: str = 'orb'           # orb | sift | gftt
    n_features: int = 2000
    grid: tuple = (8, 12)           # bucketing grid (rows, cols): spreads features over the image
    per_cell: int = 12              # max features kept per grid cell
    ratio: float = 0.8              # lowe's ratio test
    subpixel_refine: bool = True    # refine right-image correspondences with lk (sub-pixel disparity)
    self_calibrate: bool = True     # estimate the residual vertical misalignment from the first frames

    # rectified stereo geometry checks
    max_row_diff: float = 1.5       # |y_l - y_r| in px (after rectification epipolar lines are rows)
    min_disparity: float = 2.0      # px; smaller disparity -> depth too uncertain
    min_depth: float = 0.15         # m
    max_depth: float = 10.0         # m; EuRoC rooms are small, farther "points" are mismatches

    # temporal tracking (lucas-kanade)
    lk_win: int = 21
    lk_levels: int = 3
    fb_thresh: float = 1.0          # forward-backward consistency check, px

    # pose estimation (pnp)
    pnp_reproj_err: float = 2.0     # ransac inlier threshold, px
    pnp_iters: int = 200
    pnp_conf: float = 0.999
    min_inliers: int = 30
    max_step: float = 0.5           # m; reject frame-to-frame jumps bigger than this (20 hz drone)

    # keyframe policy: a new stereo triangulation when tracking gets weak
    kf_min_tracked: int = 150
    kf_max_age: int = 20            # frames

    use_gpu: bool = False           # image ops on the gpu through opencl (vo/device.py)
    out_dir: str = 'results'
    plot_live: bool = False
    plot_fps: float = 10.0          # max redraws/s of the live window, the vo runs in between
