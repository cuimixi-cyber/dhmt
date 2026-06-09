import numpy as np
import cv2
from typing import List, Tuple, Union


def project_3d_to_2d(
    points_3d: List[Tuple[float, float, float]],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> List[Tuple[float, float]]:
    points_3d = np.array(points_3d, dtype=np.float64)
    t = t.reshape(3, 1)
    Pc = (R @ points_3d.T) + t
    p  = K @ Pc
    u  = p[0, :] / p[2, :]
    v  = p[1, :] / p[2, :]
    return list(zip(u.tolist(), v.tolist()))


def project_via_homography(
    points_world: List[Tuple[float, float]],
    H: np.ndarray
) -> List[Tuple[float, float]]:
    results = []
    for (X, Y) in points_world:
        p    = np.array([X, Y, 1.0], dtype=np.float64)
        proj = H @ p
        u    = proj[0] / proj[2]
        v    = proj[1] / proj[2]
        results.append((u, v))
    return results


def compute_reprojection_error(
    points_2d_original: List[Tuple[float, float]],
    points_2d_reprojected: List[Tuple[float, float]]
) -> Tuple[float, float, List[float]]:
    errors = []
    for (u1, v1), (u2, v2) in zip(points_2d_original, points_2d_reprojected):
        err = np.sqrt((u1 - u2) ** 2 + (v1 - v2) ** 2)
        errors.append(err)
    mean_error = float(np.mean(errors))
    max_error  = float(np.max(errors))
    return mean_error, max_error, errors


def rvec_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    return R
