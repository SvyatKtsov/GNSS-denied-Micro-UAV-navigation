"""loading of EuRoC MAV data: stereo calibration, time-synchronised image pairs, ground truth, laser scan."""
import os
import csv
import yaml
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def load_camera(sensor_yaml):
    with open(sensor_yaml) as f:
        d = yaml.safe_load(f)
    fx, fy, cx, cy = d['intrinsics']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    T_BS = np.array(d['T_BS']['data'], dtype=np.float64).reshape(4, 4)     # sensor -> body (imu)
    return K, np.array(d['distortion_coefficients'], dtype=np.float64), T_BS, tuple(d['resolution'])


class StereoCalib:
    """cam0 = left, cam1 = right, everything from the EuRoC yaml files."""

    def __init__(self, mav0):
        self.K0, self.D0, self.T_B_C0, self.size = load_camera(os.path.join(mav0, 'cam0/sensor.yaml'))
        self.K1, self.D1, self.T_B_C1, _ = load_camera(os.path.join(mav0, 'cam1/sensor.yaml'))
        # X_c1 = T_C1_C0 @ X_c0 (what cv2.stereoRectify expects as R, T)
        T_C1_C0 = np.linalg.inv(self.T_B_C1) @ self.T_B_C0
        self.R, self.t = T_C1_C0[:3, :3], T_C1_C0[:3, 3]
        self.baseline = float(np.linalg.norm(self.t))


def _read_cam_csv(mav0, cam):
    img_dir = os.path.join(mav0, cam, 'data')
    on_disk = set(os.listdir(img_dir))              # some sequences list images that are missing on disk
    with open(os.path.join(mav0, cam, 'data.csv')) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith('#')]
    return {int(r[0]): os.path.join(img_dir, r[1].strip()) for r in rows if r[1].strip() in on_disk}


def stereo_pairs(mav0):
    """left/right pairs matched by timestamp (not by sorted index: cam0/cam1 folders can differ)."""
    left, right = _read_cam_csv(mav0, 'cam0'), _read_cam_csv(mav0, 'cam1')
    return [(ts, left[ts], right[ts]) for ts in sorted(set(left) & set(right))]


class GroundTruth:
    """pose of the body (imu) frame in the world frame, interpolated at image timestamps."""

    def __init__(self, mav0):
        data = np.loadtxt(os.path.join(mav0, 'state_groundtruth_estimate0/data.csv'), delimiter=',', comments='#')
        self.ts, self.p = data[:, 0], data[:, 1:4]
        self._slerp = Slerp(self.ts, Rotation.from_quat(data[:, [5, 6, 7, 4]]))   # wxyz -> xyzw

    def covers(self, ts):
        return self.ts[0] <= ts <= self.ts[-1]

    def pose(self, ts):
        T = np.eye(4)
        T[:3, :3] = self._slerp([ts]).as_matrix()[0]
        T[:3, 3] = [np.interp(ts, self.ts, self.p[:, k]) for k in range(3)]
        return T


def load_pointcloud(mav0):
    """leica laser scan of the vicon room (V1/V2 sequences only, ascii ply), in the world frame."""
    ply = os.path.join(mav0, 'pointcloud0/data.ply')
    if not os.path.exists(ply):
        return None
    with open(ply, 'rb') as f:
        skip = next(i for i, line in enumerate(f) if line.strip() == b'end_header') + 1
    pts = np.loadtxt(ply, skiprows=skip, usecols=(0, 1, 2))
    with open(os.path.join(mav0, 'pointcloud0/sensor.yaml')) as f:
        T_WR = np.array(yaml.safe_load(f)['T_WR']['data']).reshape(4, 4)
    return pts @ T_WR[:3, :3].T + T_WR[:3, 3]
