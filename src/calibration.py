"""
MODULE 5: calibration.py
=========================
Hiệu chỉnh camera (Camera Calibration) để lấy ma trận nội tại K.

Lý thuyết:
    Để thực hiện chiếu ngược chính xác, cần ma trận K thực.
    Calibration dùng bàn cờ (chessboard) với kích thước ô đã biết.
    OpenCV tìm góc ô cờ → tính K và hệ số méo (distortion).

    Nếu không có điều kiện calibrate, module cung cấp phương pháp
    ước lượng K từ thông tin EXIF của ảnh.
"""

import numpy as np
import cv2
import os
import json
import glob
from typing import Tuple, Optional, List
from src.camera_model import CameraModel


def calibrate_from_chessboard(
    images_dir: str,
    board_size: Tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    show_corners: bool = False
) -> Tuple[CameraModel, np.ndarray, float]:
    """
    Hiệu chỉnh camera từ nhiều ảnh bàn cờ.

    Args:
        images_dir      (str):   Thư mục chứa ảnh chessboard (*.jpg, *.png).
        board_size      (tuple): Số góc trong bàn cờ (cols, rows), mặc định (9,6).
        square_size_mm  (float): Kích thước mỗi ô (mm), mặc định 25mm.
        show_corners    (bool):  Hiển thị ảnh với góc phát hiện được.

    Returns:
        camera_model (CameraModel): Mô hình camera với K đã hiệu chỉnh.
        dist_coeffs  (np.ndarray):  Hệ số méo ống kính.
        rms_error    (float):       Sai số RMS của calibration (pixel).

    Raises:
        RuntimeError: Nếu không tìm thấy đủ ảnh hợp lệ.
    """
    # Chuẩn bị tọa độ 3D của các góc bàn cờ (Z = 0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm  # đơn vị mm

    obj_points = []  # điểm 3D trong thực
    img_points = []  # điểm 2D trên ảnh

    # Tìm tất cả ảnh
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for pat in patterns:
        image_files.extend(glob.glob(os.path.join(images_dir, pat)))

    if len(image_files) < 3:
        raise RuntimeError(
            f"Cần ít nhất 3 ảnh bàn cờ. Chỉ tìm thấy {len(image_files)} ảnh."
        )

    img_size = None
    successful = 0

    for fname in image_files:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        # Phát hiện góc bàn cờ
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            # Tinh chỉnh vị trí góc đến độ chính xác sub-pixel
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            obj_points.append(objp)
            img_points.append(corners_refined)
            successful += 1

            if show_corners:
                cv2.drawChessboardCorners(img, board_size, corners_refined, ret)
                cv2.imshow(f'Corners: {os.path.basename(fname)}', img)
                cv2.waitKey(500)
        else:
            print(f"  [WARNING] Không phát hiện bàn cờ trong: {os.path.basename(fname)}")

    if show_corners:
        cv2.destroyAllWindows()

    if successful < 3:
        raise RuntimeError(
            f"Chỉ phát hiện bàn cờ trong {successful} ảnh. Cần ít nhất 3."
        )

    print(f"[Calibration] Thành công trên {successful}/{len(image_files)} ảnh.")

    # Thực hiện calibration
    rms, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    print(f"[Calibration] RMS Error = {rms:.4f} pixel")
    camera_model = CameraModel.from_matrix(K)
    camera_model.print_info()

    return camera_model, dist_coeffs, rms


def estimate_K_from_exif(image_path: str) -> Optional[CameraModel]:
    """
    Ước lượng ma trận K từ thông tin EXIF của ảnh.
    Dùng khi không có điều kiện calibrate thực tế.

    Công thức ước lượng:
        fx = fy ≈ (focal_length_mm / sensor_width_mm) * image_width_px

    Args:
        image_path (str): Đường dẫn tới ảnh JPEG có EXIF.

    Returns:
        CameraModel hoặc None nếu không đọc được EXIF.
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        img = Image.open(image_path)
        width, height = img.size

        exif_data = img._getexif()
        if exif_data is None:
            print("[EXIF] Không tìm thấy dữ liệu EXIF.")
            return None

        exif = {TAGS.get(k, k): v for k, v in exif_data.items()}

        # Focal length (mm)
        focal_tag = exif.get('FocalLength')
        if focal_tag is None:
            print("[EXIF] Không có thông tin FocalLength.")
            return None

        focal_mm = float(focal_tag) if isinstance(focal_tag, (int, float)) else focal_tag[0] / focal_tag[1]

        # Focal length in 35mm equivalent → ước lượng sensor width
        focal_35mm = exif.get('FocalLengthIn35mmFilm')
        if focal_35mm and focal_35mm > 0:
            sensor_width_mm = 36.0 * focal_mm / focal_35mm
        else:
            sensor_width_mm = 36.0  # giả sử full-frame 35mm

        # Tính fx
        fx = (focal_mm / sensor_width_mm) * width
        fy = fx
        cx = width  / 2.0
        cy = height / 2.0

        print(f"[EXIF] FocalLength={focal_mm}mm, SensorWidth≈{sensor_width_mm:.1f}mm")
        print(f"[EXIF] Ước lượng: fx=fy={fx:.2f}px, cx={cx:.1f}, cy={cy:.1f}")

        return CameraModel(fx, fy, cx, cy)

    except Exception as e:
        print(f"[EXIF] Lỗi: {e}")
        return None


def save_calibration(
    camera_model: CameraModel,
    dist_coeffs: np.ndarray,
    filepath: str = "data/calibration.json"
):
    """Lưu kết quả calibration ra file JSON."""
    data = {
        "fx": camera_model.fx,
        "fy": camera_model.fy,
        "cx": camera_model.cx,
        "cy": camera_model.cy,
        "dist_coeffs": dist_coeffs.flatten().tolist()
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Calibration] Đã lưu ra: {filepath}")


def load_calibration(filepath: str = "data/calibration.json") -> Tuple[CameraModel, np.ndarray]:
    """Tải kết quả calibration từ file JSON."""
    with open(filepath, "r") as f:
        data = json.load(f)
    camera_model = CameraModel(data["fx"], data["fy"], data["cx"], data["cy"])
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    print(f"[Calibration] Đã tải từ: {filepath}")
    return camera_model, dist_coeffs
