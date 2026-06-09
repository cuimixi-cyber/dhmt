import argparse
import json
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.camera_model import CameraModel
from src.homography import compute_homography, print_homography_info
from src.projection import project_via_homography, project_3d_to_2d, compute_reprojection_error
from src.inverse_projection import (
    unproject_multiple_via_homography,
    estimate_camera_pose,
    unproject_with_known_z,
    print_camera_pose
)
from src.visualizer import (
    draw_control_points,
    draw_reprojection_error,
    draw_world_grid_on_image,
    draw_camera_axes,
    draw_unprojected_label,
    show_result_matplotlib,
    save_result_image,
    plot_world_coordinates
)


def run_mode_2d(image_path: str, points_file: str, output_path: str = "output_2d.jpg"):
    print("\n" + "=" * 55)
    print("  MODE 2D — Homography Projection (4 points)")
    print("=" * 55)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    print(f"[INPUT] Image: {image_path} ({image.shape[1]}x{image.shape[0]}px)")

    with open(points_file) as f:
        data = json.load(f)

    points_world = data["points_world"]
    points_image = data["points_image"]
    query_points = data.get("query_points", [])

    print(f"[INPUT] Control points: {len(points_world)}")
    print(f"[INPUT] Query points  : {len(query_points)}")

    H, H_inv = compute_homography(points_world, points_image)
    print_homography_info(H, H_inv)

    recovered = unproject_multiple_via_homography(query_points, H_inv)
    print("\n[RESULT] Recovered world coordinates:")
    for i, (pt2d, pt_world) in enumerate(zip(query_points, recovered)):
        print(f"  Point {i+1}: Pixel({pt2d[0]:.0f},{pt2d[1]:.0f}) "
              f"-> World({pt_world[0]:.4f}, {pt_world[1]:.4f})")

    reprojected = project_via_homography(points_world, H)
    mean_err, max_err, errors = compute_reprojection_error(points_image, reprojected)
    print(f"\n[EVAL] Reprojection Error:")
    print(f"  Mean : {mean_err:.4f} px  ({'GOOD' if mean_err < 2 else 'CHECK POINTS'})")
    print(f"  Max  : {max_err:.4f} px")

    img_result = image.copy()
    x_vals = [p[0] for p in points_world]
    y_vals = [p[1] for p in points_world]
    img_result = draw_world_grid_on_image(
        img_result, H,
        x_range=(min(x_vals), max(x_vals)),
        y_range=(min(y_vals), max(y_vals)),
        step=max((max(x_vals) - min(x_vals)) / 10, 0.5)
    )
    img_result = draw_control_points(img_result, points_image, points_world)
    img_result = draw_reprojection_error(img_result, points_image, reprojected, errors)
    for pt2d, pt_world in zip(query_points, recovered):
        img_result = draw_unprojected_label(img_result, pt2d, pt_world)

    stats = (
        f"MODE: Homography 2D\n"
        f"Control points: {len(points_world)}\n"
        f"Query points  : {len(query_points)}\n\n"
        f"Reprojection Error:\n"
        f"  Mean: {mean_err:.4f} px\n"
        f"  Max : {max_err:.4f} px\n\n"
        f"Matrix H:\n{np.round(H, 3)}\n\n"
        f"Recovered coords:\n"
        + "\n".join([f"  P{i+1}: ({r[0]:.3f}, {r[1]:.3f})"
                     for i, r in enumerate(recovered)])
    )

    save_result_image(img_result, output_path)
    show_result_matplotlib(image, img_result, "2D Result — Homography Projection", stats)
    plot_world_coordinates(recovered, [tuple(p) for p in points_world])

    return H, H_inv, recovered


def run_mode_3d(image_path: str, points_file: str, output_path: str = "output_3d.jpg"):
    print("\n" + "=" * 55)
    print("  MODE 3D — PnP Camera Pose Estimation (6 points)")
    print("=" * 55)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h_img, w_img = image.shape[:2]
    print(f"[INPUT] Image: {image_path} ({w_img}x{h_img}px)")

    with open(points_file) as f:
        data = json.load(f)

    points_3d    = data["points_3d"]
    points_image = data["points_image"]
    query_points = data.get("query_points", [])
    query_z      = data.get("query_z", 0.0)

    if "camera_K" in data:
        K = np.array(data["camera_K"], dtype=np.float64)
        camera = CameraModel.from_matrix(K)
    else:
        camera = CameraModel.from_image_size(w_img, h_img, fov_deg=60.0)
    camera.print_info()

    success, R, t, rvec = estimate_camera_pose(
        points_3d, points_image, camera.K, use_ransac=False
    )
    if not success:
        print("[ERROR] Camera pose estimation failed!")
        return

    print_camera_pose(R, t, rvec)

    reprojected = project_3d_to_2d(points_3d, camera.K, R, t)
    mean_err, max_err, errors = compute_reprojection_error(points_image, reprojected)
    print(f"\n[EVAL] Reprojection Error:")
    print(f"  Mean : {mean_err:.4f} px  ({'GOOD' if mean_err < 2 else 'CHECK POINTS'})")
    print(f"  Max  : {max_err:.4f} px")

    recovered_3d = []
    if query_points:
        print(f"\n[RESULT] Recovered 3D coordinates (Z = {query_z}):")
        for i, pt2d in enumerate(query_points):
            pt3d = unproject_with_known_z(pt2d, query_z, camera.K, R, t)
            recovered_3d.append(pt3d)
            print(f"  Point {i+1}: Pixel({pt2d[0]:.0f},{pt2d[1]:.0f}) "
                  f"-> 3D({pt3d[0]:.4f}, {pt3d[1]:.4f}, {pt3d[2]:.4f})")

    img_result = image.copy()
    img_result = draw_control_points(img_result, points_image, points_3d)
    img_result = draw_reprojection_error(img_result, points_image, reprojected, errors)
    for pt2d, pt3d in zip(query_points, recovered_3d):
        img_result = draw_unprojected_label(img_result, pt2d, pt3d)

    try:
        img_result = draw_camera_axes(img_result, camera.K, R, t,
                                       origin_3d=tuple(points_3d[0]),
                                       axis_length=1.0)
    except Exception:
        pass

    stats = (
        f"MODE: PnP 3D\n"
        f"Control points: {len(points_3d)}\n\n"
        f"Reprojection Error:\n"
        f"  Mean: {mean_err:.4f} px\n"
        f"  Max : {max_err:.4f} px\n\n"
        f"Camera Position:\n"
        f"  (see terminal)\n\n"
        f"Recovered 3D coords:\n"
        + "\n".join([f"  P{i+1}: ({r[0]:.3f},{r[1]:.3f},{r[2]:.3f})"
                     for i, r in enumerate(recovered_3d)])
    )

    save_result_image(img_result, output_path)
    show_result_matplotlib(image, img_result, "3D Result — PnP Camera Pose Estimation", stats)

    return R, t, recovered_3d


def run_demo():
    print("\n" + "=" * 55)
    print("  DEMO — Football Field (Homography 2D)")
    print("  (Based on projection-3d-2d README example)")
    print("=" * 55)

    points_world = [
        [0,    0],
        [16.5, 0],
        [16.5, 40],
        [0,    40],
    ]
    points_image = [
        [744, 303],
        [486, 349],
        [223, 197],
        [424, 176],
    ]

    goalkeeper_pixel   = [517, 227]
    penalty_spot_world = [11, 20]

    print("[DEMO] Computing Homography from 4 penalty area corners...")
    H, H_inv = compute_homography(points_world, points_image)
    print_homography_info(H, H_inv)

    gk_world = unproject_multiple_via_homography([goalkeeper_pixel], H_inv)[0]
    print(f"\n[RESULT] Goalkeeper position:")
    print(f"  Pixel : ({goalkeeper_pixel[0]}, {goalkeeper_pixel[1]})")
    print(f"  World : ({gk_world[0]:.4f}m, {gk_world[1]:.4f}m)")
    print(f"  [JS ref]: (2.1288, 21.6137)")

    penalty_pixel = project_via_homography([penalty_spot_world], H)[0]
    print(f"\n[RESULT] Penalty spot on image:")
    print(f"  World : ({penalty_spot_world[0]}m, {penalty_spot_world[1]}m)")
    print(f"  Pixel : ({penalty_pixel[0]:.4f}, {penalty_pixel[1]:.4f})")
    print(f"  [JS ref]: (409.6322, 247.8373)")

    reprojected = project_via_homography(points_world, H)
    mean_err, max_err, errors = compute_reprojection_error(points_image, reprojected)
    print(f"\n[EVAL] Reprojection Error:")
    print(f"  Mean : {mean_err:.6f} px")
    print(f"  Max  : {max_err:.6f} px")

    demo_img = _create_football_field_image(points_image)
    img_result = demo_img.copy()
    img_result = draw_world_grid_on_image(
        img_result, H,
        x_range=(0, 16.5), y_range=(0, 40), step=2.0,
        color=(200, 200, 200)
    )
    img_result = draw_control_points(img_result, points_image, points_world)
    img_result = draw_reprojection_error(img_result, points_image, reprojected, errors)
    img_result = draw_unprojected_label(img_result, goalkeeper_pixel, gk_world)
    img_result = draw_unprojected_label(img_result, penalty_pixel, penalty_spot_world)

    save_result_image(img_result, "output_demo.jpg")
    print("\n[DEMO] Saved: output_demo.jpg")

    show_result_matplotlib(
        demo_img, img_result,
        "DEMO — Inverse Perspective Projection (Football Field)",
        stats_text=(
            "Example: Football Field\n"
            "4 corners of penalty area\n\n"
            "Goalkeeper:\n"
            f"  Pixel: {goalkeeper_pixel}\n"
            f"  World: ({gk_world[0]:.4f}m,\n"
            f"          {gk_world[1]:.4f}m)\n\n"
            "Penalty spot:\n"
            f"  World: {penalty_spot_world}m\n"
            f"  Pixel: ({penalty_pixel[0]:.1f},\n"
            f"          {penalty_pixel[1]:.1f})\n\n"
            f"Reprojection Error:\n"
            f"  Mean = {mean_err:.4f} px\n"
            f"  Max  = {max_err:.4f} px"
        )
    )


def _create_football_field_image(control_points):
    us = [p[0] for p in control_points]
    vs = [p[1] for p in control_points]
    w = max(us) + 100
    h = max(vs) + 100
    img = np.full((h, w, 3), (34, 139, 34), dtype=np.uint8)
    pts = np.array(control_points, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=3)
    cv2.fillPoly(img, [pts], color=(45, 160, 45))
    cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=3)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Inverse Perspective Projection (2D -> 3D)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--mode", choices=["2d", "3d", "demo"],
                        default="demo",
                        help="2d   - Homography (plane Z=0, 4 points)\n"
                             "3d   - PnP Pose Estimation (6 3D points)\n"
                             "demo - Auto demo with sample data")
    parser.add_argument("--image",  type=str, default="", help="Input image path")
    parser.add_argument("--points", type=str, default="", help="JSON annotation file")
    parser.add_argument("--output", type=str, default="",  help="Output image path")

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo()
    elif args.mode == "2d":
        if not args.image or not args.points:
            parser.error("--mode 2d requires --image and --points")
        output = args.output or "output_2d.jpg"
        run_mode_2d(args.image, args.points, output)
    elif args.mode == "3d":
        if not args.image or not args.points:
            parser.error("--mode 3d requires --image and --points")
        output = args.output or "output_3d.jpg"
        run_mode_3d(args.image, args.points, output)


if __name__ == "__main__":
    main()
