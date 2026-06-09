import numpy as np
import cv2
from typing import List, Tuple


def compute_homography(
    points_world: List[Tuple[float, float]],
    points_image: List[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    if len(points_world) < 4 or len(points_image) < 4:
        raise ValueError("Need at least 4 point pairs to compute Homography.")

    src = np.array(points_world, dtype=np.float64)
    dst = np.array(points_image, dtype=np.float64)

    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise RuntimeError("Cannot compute Homography. Check input points.")

    H_inv = np.linalg.inv(H)
    return H, H_inv


def compute_homography_manual(
    points_world: List[Tuple[float, float]],
    points_image: List[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(points_world)
    if n < 4:
        raise ValueError("Need at least 4 point pairs.")

    A = []
    for (X, Y), (u, v) in zip(points_world, points_image):
        A.append([-X, -Y, -1,   0,  0,  0, u*X, u*Y, u])
        A.append([  0,  0,  0, -X, -Y, -1, v*X, v*Y, v])

    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)
    H = H / H[2, 2]
    H_inv = np.linalg.inv(H)
    return H, H_inv


def print_homography_info(H: np.ndarray, H_inv: np.ndarray):
    print("=" * 50)
    print("  Homography Matrix H (World -> Image)")
    print("=" * 50)
    print(np.round(H, 4))
    print()
    print("  Inverse Homography H_inv (Image -> World)")
    print("=" * 50)
    print(np.round(H_inv, 4))
    print("=" * 50)
