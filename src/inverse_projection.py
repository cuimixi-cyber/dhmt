import numpy as np
import cv2
from typing import List, Tuple, Optional


def unproject_via_homography(
    point_2d: Tuple[float, float],
    H_inv: np.ndarray
) -> Tuple[float, float]:
    u, v    = point_2d
    p       = np.array([u, v, 1.0], dtype=np.float64)
    result  = H_inv @ p
    X = result[0] / result[2]
    Y = result[1] / result[2]
    return (X, Y)


def unproject_multiple_via_homography(
    points_2d: List[Tuple[float, float]],
    H_inv: np.ndarray
) -> List[Tuple[float, float]]:
    return [unproject_via_homography(pt, H_inv) for pt in points_2d]


def estimate_camera_pose(
    points_3d: List[Tuple[float, float, float]],
    points_2d: List[Tuple[float, float]],
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    use_ransac: bool = False
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    if len(points_3d) < 4:
        raise ValueError("PnP requires at least 4 points.")

    obj_pts = np.array(points_3d, dtype=np.float64)
    img_pts = np.array(points_2d, dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    if use_ransac:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, K, dist_coeffs
        )
        if inliers is not None:
            print(f"[PnP-RANSAC] Inliers: {len(inliers)}/{len(points_3d)}")
    else:
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    if not success:
        print("[PnP] Estimation failed!")
        return False, None, None, None

    R, _ = cv2.Rodrigues(rvec)
    return True, R, tvec, rvec


def unproject_with_known_z(
    point_2d: Tuple[float, float],
    z_world: float,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> Tuple[float, float, float]:
    u, v  = point_2d
    t     = t.reshape(3)
    K_inv = np.linalg.inv(K)
    ray_cam   = K_inv @ np.array([u, v, 1.0])
    R_inv     = R.T
    denom     = R_inv[2, :] @ ray_cam
    if abs(denom) < 1e-10:
        raise RuntimeError("Ray is parallel to Z plane. Cannot unproject.")
    s           = (z_world + R_inv[2, :] @ t) / denom
    point_cam   = s * ray_cam
    point_world = R_inv @ (point_cam - t)
    return tuple(point_world.tolist())


def get_camera_position_in_world(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t.reshape(3)
    C = -(R.T @ t)
    return C


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    sy       = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        rx = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = 0.0
    return (rx, ry, rz)


def print_camera_pose(R: np.ndarray, t: np.ndarray, rvec: np.ndarray):
    C     = get_camera_position_in_world(R, t)
    euler = rotation_matrix_to_euler(R)
    print("=" * 50)
    print("  Camera Pose Estimation Result")
    print("=" * 50)
    print(f"  Camera position (World): X={C[0]:.4f}, Y={C[1]:.4f}, Z={C[2]:.4f}")
    print(f"  Euler angles           : Rx={euler[0]:.2f}, Ry={euler[1]:.2f}, Rz={euler[2]:.2f} deg")
    print(f"  Translation vector t   : {t.flatten().round(4)}")
    print(f"  Rodrigues rvec         : {rvec.flatten().round(4)}")
    print()
    print("  Rotation matrix R:")
    print(np.round(R, 4))
    print("=" * 50)
