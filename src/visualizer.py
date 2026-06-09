"""
MODULE 6: visualizer.py
========================
Vẽ và trực quan hóa kết quả phép chiếu ngược lên ảnh.

Bao gồm:
  - Vẽ điểm điều khiển (control points)
  - Vẽ lưới thế giới được chiếu lên ảnh
  - Vẽ trục tọa độ camera (XYZ → RGB)
  - Vẽ sai số tái chiếu (reprojection error)
  - Lưu và hiển thị kết quả
"""

import warnings
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional

# Tắt warning font không hỗ trợ ký tự Unicode/tiếng Việt
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Cấu hình font toàn cục — dùng font sans-serif hỗ trợ Unicode
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Segoe UI', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────
# Màu sắc chuẩn
# ─────────────────────────────────────────────
COLOR_CONTROL   = (0, 255, 0)     # Xanh lá  — điểm điều khiển gốc
COLOR_PROJECTED = (0, 0, 255)     # Đỏ       — điểm chiếu lại
COLOR_UNPROJECTED=(255, 165, 0)   # Cam      — điểm chiếu ngược
COLOR_GRID      = (180, 180, 180) # Xám      — lưới
COLOR_TEXT      = (255, 255, 255) # Trắng    — chữ
COLOR_X_AXIS    = (0, 0, 255)     # Đỏ       — trục X
COLOR_Y_AXIS    = (0, 255, 0)     # Xanh lá  — trục Y
COLOR_Z_AXIS    = (255, 0, 0)     # Xanh dương — trục Z


def draw_control_points(
    image: np.ndarray,
    points_2d: List[Tuple[float, float]],
    world_coords: Optional[List] = None,
    radius: int = 8,
    color=COLOR_CONTROL
) -> np.ndarray:
    """
    Vẽ các điểm điều khiển (control points) lên ảnh.

    Args:
        image      (np.ndarray): Ảnh gốc (BGR).
        points_2d  (list):       Danh sách tọa độ pixel [(u, v), ...].
        world_coords (list):     Tọa độ thực để ghi nhãn (tùy chọn).
        radius     (int):        Bán kính điểm.
        color      (tuple):      Màu BGR.

    Returns:
        np.ndarray: Ảnh đã vẽ.
    """
    img = image.copy()
    for i, (u, v) in enumerate(points_2d):
        u, v = int(round(u)), int(round(v))

        # Vẽ hình tròn + điểm tâm
        cv2.circle(img, (u, v), radius, color, 2)
        cv2.circle(img, (u, v), 2, color, -1)

        # Ghi nhãn số thứ tự
        label = f"P{i+1}"
        if world_coords and i < len(world_coords):
            wc = world_coords[i]
            if len(wc) == 2:
                label += f"\n({wc[0]:.1f},{wc[1]:.1f})"
            else:
                label += f"\n({wc[0]:.1f},{wc[1]:.1f},{wc[2]:.1f})"

        cv2.putText(img, f"P{i+1}", (u + 10, v - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def draw_reprojection_error(
    image: np.ndarray,
    points_original: List[Tuple[float, float]],
    points_reprojected: List[Tuple[float, float]],
    errors: List[float]
) -> np.ndarray:
    """
    Vẽ sai số tái chiếu: đường nối giữa điểm gốc và điểm tái chiếu.

    Args:
        image              (np.ndarray): Ảnh gốc.
        points_original    (list):       Điểm 2D gốc [(u,v),...].
        points_reprojected (list):       Điểm 2D tái chiếu [(u,v),...].
        errors             (list):       Sai số từng điểm (pixel).

    Returns:
        np.ndarray: Ảnh đã vẽ.
    """
    img = image.copy()
    for i, ((u1, v1), (u2, v2), err) in enumerate(
            zip(points_original, points_reprojected, errors)):
        u1, v1 = int(round(u1)), int(round(v1))
        u2, v2 = int(round(u2)), int(round(v2))

        # Điểm gốc (xanh lá)
        cv2.circle(img, (u1, v1), 6, COLOR_CONTROL, -1)
        # Điểm tái chiếu (đỏ)
        cv2.circle(img, (u2, v2), 6, COLOR_PROJECTED, -1)
        # Đường nối sai số
        cv2.line(img, (u1, v1), (u2, v2), (0, 165, 255), 1)
        # Nhãn sai số
        mx, my = (u1 + u2) // 2, (v1 + v2) // 2
        cv2.putText(img, f"{err:.1f}px", (mx + 3, my - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    return img


def draw_world_grid_on_image(
    image: np.ndarray,
    H: np.ndarray,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    step: float = 1.0,
    color=COLOR_GRID,
    thickness: int = 1
) -> np.ndarray:
    """
    Vẽ lưới tọa độ thế giới được chiếu lên ảnh qua Homography.
    Giúp trực quan hóa phép ánh xạ.

    Args:
        image   (np.ndarray):  Ảnh gốc.
        H       (np.ndarray):  Ma trận Homography (World → Image).
        x_range (tuple):       (x_min, x_max) của lưới thế giới.
        y_range (tuple):       (y_min, y_max) của lưới thế giới.
        step    (float):       Bước lưới.
        color   (tuple):       Màu BGR của đường lưới.
        thickness (int):       Độ dày đường.

    Returns:
        np.ndarray: Ảnh đã vẽ.
    """
    img = image.copy()
    h_img, w_img = img.shape[:2]

    def project_pt(x, y):
        p = H @ np.array([x, y, 1.0])
        return (int(p[0] / p[2]), int(p[1] / p[2]))

    xs = np.arange(x_range[0], x_range[1] + step, step)
    ys = np.arange(y_range[0], y_range[1] + step, step)

    # Vẽ đường theo chiều X (cố định x, thay đổi y)
    for x in xs:
        pts = [project_pt(x, y) for y in ys]
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i+1]
            if (0 <= p1[0] < w_img and 0 <= p1[1] < h_img and
                    0 <= p2[0] < w_img and 0 <= p2[1] < h_img):
                cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    # Vẽ đường theo chiều Y (cố định y, thay đổi x)
    for y in ys:
        pts = [project_pt(x, y) for x in xs]
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i+1]
            if (0 <= p1[0] < w_img and 0 <= p1[1] < h_img and
                    0 <= p2[0] < w_img and 0 <= p2[1] < h_img):
                cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

    return img


def draw_camera_axes(
    image: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    origin_3d: Tuple[float, float, float] = (0, 0, 0),
    axis_length: float = 1.0
) -> np.ndarray:
    """
    Vẽ trục tọa độ XYZ của camera lên ảnh (RGB = XYZ).

    Args:
        image       (np.ndarray):  Ảnh gốc.
        K, R, t     (np.ndarray):  Thông số camera.
        origin_3d   (tuple):       Gốc tọa độ 3D.
        axis_length (float):       Chiều dài trục (đơn vị thực).

    Returns:
        np.ndarray: Ảnh đã vẽ.
    """
    from src.projection import project_3d_to_2d

    img = image.copy()
    ox, oy, oz = origin_3d
    points_3d = [
        (ox, oy, oz),                          # Gốc O
        (ox + axis_length, oy, oz),            # X
        (ox, oy + axis_length, oz),            # Y
        (ox, oy, oz + axis_length),            # Z
    ]

    pts_2d = project_3d_to_2d(points_3d, K, R, t)
    O  = (int(pts_2d[0][0]), int(pts_2d[0][1]))
    Xp = (int(pts_2d[1][0]), int(pts_2d[1][1]))
    Yp = (int(pts_2d[2][0]), int(pts_2d[2][1]))
    Zp = (int(pts_2d[3][0]), int(pts_2d[3][1]))

    cv2.arrowedLine(img, O, Xp, COLOR_X_AXIS, 2, tipLength=0.2)
    cv2.arrowedLine(img, O, Yp, COLOR_Y_AXIS, 2, tipLength=0.2)
    cv2.arrowedLine(img, O, Zp, COLOR_Z_AXIS, 2, tipLength=0.2)

    cv2.putText(img, "X", (Xp[0]+5, Xp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_X_AXIS, 2)
    cv2.putText(img, "Y", (Yp[0]+5, Yp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_Y_AXIS, 2)
    cv2.putText(img, "Z", (Zp[0]+5, Zp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_Z_AXIS, 2)

    return img


def draw_unprojected_label(
    image: np.ndarray,
    point_2d: Tuple[float, float],
    world_coord,
    color=COLOR_UNPROJECTED
) -> np.ndarray:
    """Vẽ nhãn tọa độ thực lên một điểm ảnh."""
    img = image.copy()
    u, v = int(round(point_2d[0])), int(round(point_2d[1]))
    cv2.circle(img, (u, v), 7, color, -1)

    if len(world_coord) == 2:
        label = f"({world_coord[0]:.2f}, {world_coord[1]:.2f})"
    else:
        label = f"({world_coord[0]:.2f}, {world_coord[1]:.2f}, {world_coord[2]:.2f})"

    # Background cho text
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (u + 8, v - th - 6), (u + 8 + tw, v), (0, 0, 0), -1)
    cv2.putText(img, label, (u + 8, v - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
    return img


def show_result_matplotlib(
    image_original: np.ndarray,
    image_result: np.ndarray,
    title: str = "Inverse Perspective Projection Result",
    stats_text: Optional[str] = None
):
    """
    Display original and result images side-by-side using Matplotlib.

    Args:
        image_original (np.ndarray): Original image (BGR).
        image_result   (np.ndarray): Processed image (BGR).
        title          (str):        Window title.
        stats_text     (str):        Stats text shown on the right panel.
    """
    if stats_text:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Original image
    axes[0].imshow(cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image (Input)", fontsize=11)
    axes[0].axis('off')

    # Result image
    axes[1].imshow(cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Result (Output)", fontsize=11)
    axes[1].axis('off')

    # Stats panel (optional)
    if stats_text:
        axes[2].axis('off')
        axes[2].text(0.05, 0.95, stats_text,
                     transform=axes[2].transAxes,
                     fontsize=10, verticalalignment='top',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
        axes[2].set_title("Statistics", fontsize=11)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#00ff00', label='Control Points (original)'),
        mpatches.Patch(color='#0000ff', label='Reprojected Points'),
        mpatches.Patch(color='#ffa500', label='Unprojected Points'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9)

    plt.tight_layout()
    plt.show()


def save_result_image(image: np.ndarray, filepath: str):
    """Lưu ảnh kết quả ra file."""
    import os
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    cv2.imwrite(filepath, image)
    print(f"[Visualizer] Đã lưu ảnh: {filepath}")


def plot_world_coordinates(
    recovered_points: List[Tuple],
    control_points: Optional[List[Tuple]] = None,
    title: str = "Recovered World Coordinates"
):
    """
    Vẽ đồ thị 2D/3D thể hiện tọa độ phục hồi được.

    Args:
        recovered_points  (list): Tọa độ phục hồi [(X,Y) hoặc (X,Y,Z)].
        control_points    (list): Điểm điều khiển đã biết (để so sánh).
        title             (str):  Tiêu đề đồ thị.
    """
    if not recovered_points:
        return

    is_3d = len(recovered_points[0]) == 3

    if is_3d:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        xs, ys, zs = zip(*recovered_points)
        ax.scatter(xs, ys, zs, c='orange', s=80, label='Recovered points', zorder=5)

        if control_points:
            cxs, cys, czs = zip(*control_points)
            ax.scatter(cxs, cys, czs, c='green', s=100, marker='^',
                       label='Control points', zorder=6)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
    else:
        fig, ax = plt.subplots(figsize=(10, 8))

        xs, ys = zip(*recovered_points)
        ax.scatter(xs, ys, c='orange', s=80, zorder=5, label='Recovered points')

        if control_points:
            cxs, cys = zip(*control_points)
            ax.scatter(cxs, cys, c='green', s=100, marker='^',
                       zorder=6, label='Control points')

        # Nhãn từng điểm
        for i, (x, y) in enumerate(recovered_points):
            ax.annotate(f"({x:.2f},{y:.2f})", (x, y),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax.set_xlabel("X (unit)")
        ax.set_ylabel("Y (unit)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal')

    ax.set_title(title, fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()
