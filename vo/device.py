"""cpu / gpu switch. the gpu path uses opencv's opencl backend (cv2.UMat): it ships with the stock pip
opencv-python on windows, linux and macos and runs on nvidia, amd, intel and apple gpus (no cuda build
needed). remap, orb/gftt detection and pyramidal lk run on the gpu; matching, pnp etc. stay on the cpu."""
import cv2


def setup(use_gpu, verbose=True):
    """returns True if the gpu (opencl) path is active."""
    cv2.ocl.setUseOpenCL(bool(use_gpu))
    gpu = bool(use_gpu and cv2.ocl.haveOpenCL() and cv2.ocl.Device.getDefault().available())
    cv2.ocl.setUseOpenCL(gpu)
    if use_gpu and not gpu:
        print('WARNING: no opencl gpu found, falling back to cpu')
    if verbose:
        print(f"device: {'gpu via opencl (' + cv2.ocl.Device.getDefault().name() + ')' if gpu else 'cpu'}")
    return gpu


def upload(img, gpu):
    return cv2.UMat(img) if gpu else img


def host(x):
    """gpu result -> numpy (no-op for numpy / None)."""
    return x.get() if isinstance(x, cv2.UMat) else x
