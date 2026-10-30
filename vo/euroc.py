"""EuRoC MAV dataset loading: stereo calibration, time-synchronised image pairs, ground truth."""
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
    dist = np.array(d['distortion_coefficients'], dtype=np.float64)
    T_BS = np.array(d['T_BS']['data'], dtype=np.float64).reshape(4, 4)   # sensor -> body (IMU)
    w, h = d['resolution']
    return K, dist, T_BS, (w, h)


class StereoCalib:
    """cam0 = left, cam1 = right. All extrinsics from the EuRoC yaml files."""

    def __init__(self, mav0):
        self.K0, self.D0, self.T_B_C0, self.size = load_camera(os.path.join(mav0, 'cam0/sensor.yaml'))
        self.K1, self.D1, self.T_B_C1, _ = load_camera(os.path.join(mav0, 'cam1/sensor.yaml'))
        # X_c1 = T_C1_C0 @ X_c0  (what cv2.stereoRectify expects as R, T)
        self.T_C1_C0 = np.linalg.inv(self.T_B_C1) @ self.T_B_C0
        self.R, self.t = self.T_C1_C0[:3, :3], self.T_C1_C0[:3, 3]

    @property
    def baseline(self):
        return float(np.linalg.norm(self.t))


def _read_cam_csv(mav0, cam):
    with open(os.path.join(mav0, cam, 'data.csv')) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith('#')]
    files = {int(r[0]): r[1].strip() for r in rows}
    # some EuRoC sequences list images in data.csv that are missing on disk
    img_dir = os.path.join(mav0, cam, 'data')
    on_disk = set(os.listdir(img_dir))
    return {ts: os.path.join(img_dir, fn) for ts, fn in files.items() if fn in on_disk}


def stereo_pairs(mav0):
    """Left/right image pairs matched by timestamp (NOT by sorted index: cam0/cam1 folders differ)."""
    left, right = _read_cam_csv(mav0, 'cam0'), _read_cam_csv(mav0, 'cam1')
    common = sorted(set(left) & set(right))
    return [(ts, left[ts], right[ts]) for ts in common]


class GroundTruth:
    """state_groundtruth_estimate0: pose of the body (IMU) frame in the world frame."""

    def __init__(self, mav0):
        path = os.path.join(mav0, 'state_groundtruth_estimate0/data.csv')
        data = np.loadtxt(path, delimiter=',', comments='#')
        self.ts = data[:, 0].astype(np.int64)
        self.p = data[:, 1:4]
        q_wxyz = data[:, 4:8]
        self._rot = Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]])
        self._slerp = Slerp(self.ts.astype(np.float64), self._rot)

    def covers(self, ts):
        return self.ts[0] <= ts <= self.ts[-1]

    def pose(self, ts):
        """Interpolated 4x4 T_W_B at timestamp ts (ns)."""
        t = float(ts)
        T = np.eye(4)
        T[:3, :3] = self._slerp([t]).as_matrix()[0]
        T[:3, 3] = [np.interp(t, self.ts, self.p[:, k]) for k in range(3)]
        return T


def load_pointcloud(mav0):
    """Leica laser scan of the Vicon room (V1/V2 sequences only), in the world frame."""
    ply = os.path.join(mav0, 'pointcloud0/data.ply')
    if not os.path.exists(ply):
        return None
    with open(ply, 'rb') as f:
        header, n, props = [], 0, []
        while True:
            line = f.readline().decode('ascii', 'ignore').strip()
            header.append(line)
            if line.startswith('element vertex'):
                n = int(line.split()[-1])
            elif line.startswith('property'):
                props.append(line.split())
            elif line == 'end_header':
                break
        fmt = header[1].split()[1]
        if fmt == 'ascii':
            pts = np.loadtxt(f, max_rows=n)[:, :3]
        else:
            types = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'uchar': 'u1', 'uint8': 'u1',
                     'int': 'i4', 'int32': 'i4', 'short': 'i2', 'ushort': 'u2', 'uint': 'u4'}
            endian = '<' if 'little' in fmt else '>'
            dtype = np.dtype([(p[-1], endian + types[p[1]]) for p in props if p[1] != 'list'])
            arr = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)
            pts = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float64)
    with open(os.path.join(mav0, 'pointcloud0/sensor.yaml')) as f:
        T_WR = np.array(yaml.safe_load(f)['T_WR']['data']).reshape(4, 4)
    return pts @ T_WR[:3, :3].T + T_WR[:3, 3]
