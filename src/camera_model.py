import numpy as np


class CameraModel:
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)

    def get_intrinsic(self) -> np.ndarray:
        return self.K

    def pixel_to_normalized(self, u: float, v: float):
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        return (x, y)

    def normalized_to_pixel(self, x: float, y: float):
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        return (u, v)

    @classmethod
    def from_image_size(cls, width: int, height: int,
                        fov_deg: float = 60.0) -> "CameraModel":
        fov_rad = np.deg2rad(fov_deg)
        fx = (width / 2.0) / np.tan(fov_rad / 2.0)
        fy = fx
        cx = width  / 2.0
        cy = height / 2.0
        print(f"[CameraModel] Estimated K from {width}x{height}, FOV={fov_deg}deg")
        print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        return cls(fx, fy, cx, cy)

    @classmethod
    def from_matrix(cls, K: np.ndarray) -> "CameraModel":
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2])
        )

    def __repr__(self):
        return (f"CameraModel(fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f})")

    def print_info(self):
        print("=" * 40)
        print("  Camera Intrinsic Matrix K")
        print("=" * 40)
        print(f"  Focal Length : fx={self.fx:.4f}, fy={self.fy:.4f}")
        print(f"  Principal Pt : cx={self.cx:.4f}, cy={self.cy:.4f}")
        print("  Matrix K:")
        print(self.K)
        print("=" * 40)
