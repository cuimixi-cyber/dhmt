# Đề Tài 12: Tái Tạo Phép Chiếu Phối Cảnh Ngược (2D → 3D)

> Xây dựng chương trình tái tạo tọa độ camera từ các thông số trong một bức ảnh 2D  
> Tham khảo: [projection-3d-2d v2.0.8](https://www.npmjs.com/package/projection-3d-2d/v/2.0.8)

---

## Cấu Trúc Thư Mục

```
de_tai_12/
├── data/
│   ├── images/                  # Ảnh đầu vào
│   └── annotations/
│       ├── points_2d.json       # Điểm mẫu chế độ 2D
│       └── points_3d.json       # Điểm mẫu chế độ 3D
├── src/
│   ├── camera_model.py          # [MODULE 1] Ma trận nội tại K
│   ├── projection.py            # [MODULE 2] Chiều thuận 3D→2D
│   ├── inverse_projection.py    # [MODULE 3] Chiều ngược 2D→3D ⭐
│   ├── homography.py            # [MODULE 4] Tính Homography H
│   ├── calibration.py           # [MODULE 5] Hiệu chỉnh camera
│   └── visualizer.py            # [MODULE 6] Trực quan hóa kết quả
├── main.py                      # Chương trình chính
├── requirements.txt             # Thư viện Python
└── README.md                    # Tài liệu này
```

---

## Cài Đặt Môi Trường

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Cài thư viện
pip install -r requirements.txt
```

---

## Cách Chạy

### 1. Chạy Demo Tự Động (không cần ảnh thực)
```bash
python main.py --mode demo
```
Demo sẽ mô phỏng sân bóng đá, tính Homography từ 4 góc khu vực phạt đền,
chiếu ngược vị trí thủ môn về tọa độ thực (giống ví dụ trong README của thư viện JS).

### 2. Chế Độ 2D — Homography (4 điểm, mặt phẳng Z=0)
```bash
python main.py --mode 2d --image data/images/sample.jpg --points data/annotations/points_2d.json
```

### 3. Chế Độ 3D — PnP Camera Pose (6 điểm 3D)
```bash
python main.py --mode 3d --image data/images/sample.jpg --points data/annotations/points_3d.json
```

---

## Format File Điểm JSON

### Chế Độ 2D (`points_2d.json`)
```json
{
  "points_world": [[0,0], [16.5,0], [16.5,40], [0,40]],
  "points_image": [[744,303], [486,349], [223,197], [424,176]],
  "query_points": [[517, 227]]
}
```

### Chế Độ 3D (`points_3d.json`)
```json
{
  "points_3d":   [[X,Y,Z], ...],
  "points_image": [[u,v], ...],
  "query_points": [[u,v], ...],
  "query_z": 0.0,
  "camera_K": [[fx,0,cx],[0,fy,cy],[0,0,1]]
}
```

---

## Mô Tả Các Module

| Module | Chức Năng | Tương đương JS |
|--------|-----------|----------------|
| `camera_model.py` | Ma trận nội tại K | — |
| `homography.py` | Tính H và H_inv từ 4 điểm | `ProjectionCalculator2d` |
| `projection.py` | Chiếu 3D→2D | `getProjectedPoint()` |
| `inverse_projection.py` | Chiếu ngược 2D→3D ⭐ | `getUnprojectedPoint()` |
| `calibration.py` | Hiệu chỉnh camera | — |
| `visualizer.py` | Vẽ kết quả lên ảnh | — |

---

## Tiêu Chí Đánh Giá

| Tiêu chí | Ngưỡng tốt |
|----------|-----------|
| Reprojection Error (trung bình) | < 2 pixel |
| Reprojection Error (lớn nhất) | < 5 pixel |
| Độ chính xác vị trí camera | So sánh với ground truth |

---

## Lý Thuyết Tóm Tắt

**Phép chiếu thuận (Forward):**
```
[u, v, 1]^T ∝ K * [R|t] * [X, Y, Z, 1]^T
```

**Phép chiếu ngược (Inverse) qua Homography (Z=0):**
```
[X, Y, 1]^T ∝ H_inv * [u, v, 1]^T
```

**Ước lượng Camera Pose (PnP):**
```
cv2.solvePnP(points_3d, points_2d, K, dist) → rvec, tvec
```
