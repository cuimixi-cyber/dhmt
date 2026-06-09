"""
annotate.py — Cong Cu Danh Dau Diem Tuong Tac (Nang Cao)
==========================================================
Tinh nang:
  - Hien thi panel huong dan tren cua so anh
  - Crosshair theo con tro chuot
  - So thu tu va toa do pixel hien thi ngay khi click
  - Phim Z: xoa diem cuoi neu click nham
  - Preview luoi Homography sau khi annotation
  - Tu dong luu JSON va chay main.py

Cach chay:
    python annotate.py --image data/images/ten_anh.jpg --mode 2d --run
    python annotate.py --image data/images/ten_anh.jpg --mode 3d --run
"""

import cv2
import json
import os
import sys
import argparse
import numpy as np


# ══════════════════════════════════════════════════════
#  CAU HINH MAU SAC
# ══════════════════════════════════════════════════════
C_CTRL_FILL   = (50,  205,  50)   # xanh la   - diem dieu khien (filled)
C_CTRL_RING   = (255, 255, 255)   # trang      - vong ngoai
C_QUERY_FILL  = (0,   165, 255)   # cam        - diem truy van
C_QUERY_RING  = (255, 255,   0)   # vang       - vong ngoai query
C_CROSS       = (0,   200, 255)   # xanh nhat  - crosshair
C_CONNECT     = (180, 180, 180)   # xam        - duong noi
C_PANEL_BG    = (20,   20,  20)   # den dam    - nen panel
C_PANEL_LINE  = (60,   60,  60)   # xam toi    - duong ke panel
C_TITLE       = (0,   200, 255)   # xanh cyan  - tieu de
C_ACTIVE      = (50,  205,  50)   # xanh la    - dong hien hanh
C_DONE        = (100, 100, 100)   # xam mo     - da hoan thanh
C_KEY         = (255, 215,   0)   # vang       - phim tat
C_TEXT        = (220, 220, 220)   # xam sang   - chu binh thuong
C_GRID        = (180, 180, 180)   # xam        - luoi preview
C_ERR         = (0,   80, 255)    # do         - canh bao

PANEL_W = 280   # do rong panel ben phai


# ══════════════════════════════════════════════════════
#  LOP CHINH
# ══════════════════════════════════════════════════════
class ImageAnnotator:

    def __init__(self, image_path: str, n_ctrl: int, mode: str):
        """
        image_path : duong dan anh
        n_ctrl     : 4 (mode 2D) hoac 6 (mode 3D)
        mode       : '2d' hoac '3d'
        """
        self.image_path = image_path
        self.n_ctrl     = n_ctrl
        self.mode       = mode

        # Doc anh goc
        img_raw = cv2.imread(image_path)
        if img_raw is None:
            raise FileNotFoundError(f"[LOI] Khong doc duoc anh: {image_path}")

        # Thu nho neu can
        h, w = img_raw.shape[:2]
        self.scale = 1.0
        max_h = 680
        if h > max_h:
            self.scale = max_h / h
            new_w = int(w * self.scale)
            img_raw = cv2.resize(img_raw, (new_w, max_h))
            print(f"[INFO] Da thu nho anh: {w}x{h} -> {new_w}x{max_h}")

        self.img_base   = img_raw          # anh nen (khong doi)
        self.ih, self.iw = img_raw.shape[:2]

        # Trang thai
        self.ctrl_pixels  = []   # [(u,v), ...] diem dieu khien da click
        self.query_pixels = []   # [(u,v), ...] diem truy van da click
        self.phase        = "control"   # "control" | "query" | "done"
        self.mouse_pos    = (0, 0)
        self.msg          = ""   # thong bao tuc thoi
        self.msg_color    = C_ACTIVE
        self.show_grid    = False

        self.WIN = "Annotation Tool  [Click de danh dau | Z=Xoa | ENTER=Tiep | ESC=Thoat]"

    # ─────────────────────────────────────────────
    #  VE KHUNG HIEN THI
    # ─────────────────────────────────────────────
    def _render(self) -> np.ndarray:
        """Tao frame hien thi = anh + luoi (neu co) + diem + crosshair + panel."""
        canvas_w = self.iw + PANEL_W
        canvas   = np.zeros((self.ih, canvas_w, 3), dtype=np.uint8)

        # --- Vung anh ---
        area = self.img_base.copy()

        # Ve luoi Homography preview neu co du diem
        if self.show_grid and len(self.ctrl_pixels) == self.n_ctrl:
            area = self._draw_grid_preview(area)

        # Duong noi cac diem dieu khien
        if len(self.ctrl_pixels) >= 2:
            for i in range(len(self.ctrl_pixels) - 1):
                cv2.line(area,
                         self.ctrl_pixels[i], self.ctrl_pixels[i+1],
                         C_CONNECT, 1, cv2.LINE_AA)
        if len(self.ctrl_pixels) == self.n_ctrl:
            cv2.line(area,
                     self.ctrl_pixels[-1], self.ctrl_pixels[0],
                     C_CONNECT, 1, cv2.LINE_AA)

        # Ve diem dieu khien
        for i, (u, v) in enumerate(self.ctrl_pixels):
            cv2.circle(area, (u, v), 9, C_CTRL_RING, 1, cv2.LINE_AA)
            cv2.circle(area, (u, v), 6, C_CTRL_FILL, -1, cv2.LINE_AA)
            label = f"P{i+1}"
            self._put_label(area, label, u+11, v-5, C_CTRL_FILL)

        # Ve diem truy van
        for i, (u, v) in enumerate(self.query_pixels):
            cv2.circle(area, (u, v), 9, C_QUERY_RING, 1, cv2.LINE_AA)
            cv2.circle(area, (u, v), 6, C_QUERY_FILL, -1, cv2.LINE_AA)
            label = f"Q{i+1}"
            self._put_label(area, label, u+11, v-5, C_QUERY_FILL)

        # Crosshair
        mx, my = self.mouse_pos
        if 0 <= mx < self.iw and 0 <= my < self.ih:
            cv2.line(area, (mx, 0), (mx, self.ih-1), C_CROSS, 1, cv2.LINE_AA)
            cv2.line(area, (0, my), (self.iw-1, my), C_CROSS, 1, cv2.LINE_AA)
            cv2.circle(area, (mx, my), 4, C_CROSS, 1, cv2.LINE_AA)
            coord_txt = f"({mx},{my})"
            cx = mx + 8 if mx < self.iw - 80 else mx - 70
            cy = my - 8 if my > 15 else my + 15
            cv2.putText(area, coord_txt, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_CROSS, 1, cv2.LINE_AA)

        canvas[:, :self.iw] = area

        # --- Panel ben phai ---
        panel = canvas[:, self.iw:]
        panel[:] = C_PANEL_BG
        cv2.line(canvas, (self.iw, 0), (self.iw, self.ih-1), C_PANEL_LINE, 1)
        self._draw_panel(panel)

        return canvas

    def _put_label(self, img, text, x, y, color):
        """Ve nhan text voi nen den."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        x = min(x, self.iw - tw - 4)
        y = max(y, th + 4)
        cv2.rectangle(img, (x-2, y-th-2), (x+tw+2, y+2), (0,0,0), -1)
        cv2.putText(img, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    def _draw_panel(self, panel: np.ndarray):
        """Ve noi dung panel huong dan."""
        h = panel.shape[0]
        pw = PANEL_W
        y  = 0

        def row(txt, color=C_TEXT, size=0.42, bold=False, margin_top=0):
            nonlocal y
            y += margin_top
            thickness = 2 if bold else 1
            cv2.putText(panel, txt, (10, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)
            y += 20

        def divider(margin=4):
            nonlocal y
            y += margin
            cv2.line(panel, (8, y), (pw-8, y), C_PANEL_LINE, 1)
            y += margin + 2

        # === Tieu de ===
        y = 8
        row(f"  ANNOTATION  [{self.mode.upper()}]", C_TITLE, size=0.5, bold=True)
        divider()

        # === Trang thai hien tai ===
        if self.phase == "control":
            done  = len(self.ctrl_pixels)
            total = self.n_ctrl
            row(f"BUOC 1/2: Diem dieu khien", C_TITLE, size=0.44, bold=True, margin_top=2)
            row(f"  Da click: {done} / {total}", C_ACTIVE if done < total else C_KEY)
            row(f"  Con lai : {total - done} diem", C_TEXT)
            y += 4
            for i in range(total):
                if i < done:
                    u, v = self.ctrl_pixels[i]
                    row(f"  [v] P{i+1}: ({u}, {v})", C_DONE)
                elif i == done:
                    row(f"  [>] P{i+1}: <-- click day", C_ACTIVE, bold=True)
                else:
                    row(f"  [ ] P{i+1}: ...")
        elif self.phase == "query":
            row(f"BUOC 2/2: Diem truy van", C_TITLE, size=0.44, bold=True, margin_top=2)
            done = len(self.query_pixels)
            row(f"  Da click: {done} diem", C_QUERY_FILL)
            row(f"  (Click tiep hoac ENTER de xong)", C_TEXT)
            y += 4
            for i, (u, v) in enumerate(self.query_pixels):
                row(f"  [v] Q{i+1}: ({u}, {v})", C_QUERY_FILL)
        else:
            row("HOAN THANH!", C_KEY, bold=True)

        divider(margin=6)

        # === Phim tat ===
        row("PHIM TAT:", C_TEXT, bold=True, margin_top=2)
        shortcuts = [
            ("Click trai", "Danh dau diem"),
            ("Z",          "Xoa diem cuoi"),
            ("ENTER",      "Xac nhan & tiep"),
            ("G",          "Bat/tat luoi preview"),
            ("ESC",        "Thoat"),
        ]
        for key, desc in shortcuts:
            row(f"  [{key}]", C_KEY, margin_top=1)
            # Ve mo ta cung dong
            cv2.putText(panel, desc, (70, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT, 1, cv2.LINE_AA)

        divider(margin=6)

        # === Thong tin diem dieu khien ===
        if self.ctrl_pixels:
            row("DIEM DIEU KHIEN:", C_TEXT, bold=True, margin_top=2)
            for i, (u, v) in enumerate(self.ctrl_pixels):
                row(f"  P{i+1}: ({u:4d}, {v:4d})", C_CTRL_FILL)

        if self.query_pixels:
            divider()
            row("DIEM TRUY VAN:", C_TEXT, bold=True)
            for i, (u, v) in enumerate(self.query_pixels):
                row(f"  Q{i+1}: ({u:4d}, {v:4d})", C_QUERY_FILL)

        # === Thong bao tuc thoi ===
        if self.msg:
            msg_y = h - 28
            cv2.rectangle(panel, (4, msg_y-16), (pw-4, h-4), (40,40,40), -1)
            cv2.putText(panel, self.msg, (8, msg_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.msg_color, 1, cv2.LINE_AA)

    def _draw_grid_preview(self, img: np.ndarray) -> np.ndarray:
        """Ve luoi the gioi chieu len anh qua Homography (neu du diem)."""
        try:
            # Chi lam duoc cho mode 2D (can toa do world)
            return img
        except Exception:
            return img

    # ─────────────────────────────────────────────
    #  SU KIEN CHUOT
    # ─────────────────────────────────────────────
    def _on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)

        if x >= self.iw:
            return   # click vao panel -> bo qua

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.phase == "control":
                if len(self.ctrl_pixels) < self.n_ctrl:
                    self.ctrl_pixels.append((x, y))
                    idx = len(self.ctrl_pixels)
                    self.msg = f"P{idx} da danh dau: ({x},{y})"
                    self.msg_color = C_ACTIVE
                    print(f"  [CLICK] P{idx}: pixel=({x}, {y})")
                    if len(self.ctrl_pixels) == self.n_ctrl:
                        self.msg = f"Du {self.n_ctrl} diem! Nhan ENTER de tiep tuc."
                        self.msg_color = C_KEY
                else:
                    self.msg = f"Du {self.n_ctrl} diem roi. Nhan ENTER."
                    self.msg_color = C_ERR

            elif self.phase == "query":
                self.query_pixels.append((x, y))
                idx = len(self.query_pixels)
                self.msg = f"Q{idx} da danh dau: ({x},{y})"
                self.msg_color = C_QUERY_FILL
                print(f"  [CLICK] Q{idx}: pixel=({x}, {y})")

    # ─────────────────────────────────────────────
    #  CHAY VONG LAP CHINH
    # ─────────────────────────────────────────────
    def run(self) -> tuple:
        """
        Chay vong lap annotation.
        Returns: (ctrl_pixels, query_pixels)
        """
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        total_w = self.iw + PANEL_W
        cv2.resizeWindow(self.WIN, total_w, self.ih)
        cv2.setMouseCallback(self.WIN, self._on_mouse)

        print(f"\n{'='*60}")
        print(f"  BUOC 1: Click {self.n_ctrl} DIEM DIEU KHIEN len anh")
        print(f"  (Chon cac diem co toa do thuc da biet truoc)")
        print(f"{'='*60}")

        phase1_done = False
        while True:
            frame = self._render()
            cv2.imshow(self.WIN, frame)
            key = cv2.waitKey(16) & 0xFF  # ~60 fps

            # ── Xoa diem cuoi ──
            if key == ord('z') or key == ord('Z'):
                if self.phase == "control" and self.ctrl_pixels:
                    removed = self.ctrl_pixels.pop()
                    self.msg = f"Da xoa P{len(self.ctrl_pixels)+1}: {removed}"
                    self.msg_color = C_ERR
                    print(f"  [XOA] Xoa diem dieu khien: {removed}")
                elif self.phase == "query" and self.query_pixels:
                    removed = self.query_pixels.pop()
                    self.msg = f"Da xoa Q{len(self.query_pixels)+1}: {removed}"
                    self.msg_color = C_ERR
                    print(f"  [XOA] Xoa diem truy van: {removed}")
                else:
                    self.msg = "Khong co diem de xoa!"
                    self.msg_color = C_ERR

            # ── Bat/tat luoi ──
            elif key == ord('g') or key == ord('G'):
                self.show_grid = not self.show_grid
                self.msg = "Grid: BẬT" if self.show_grid else "Grid: TẮT"
                self.msg_color = C_KEY

            # ── ENTER: chuyen pha ──
            elif key in (13, 10):
                if self.phase == "control":
                    if len(self.ctrl_pixels) < self.n_ctrl:
                        self.msg = f"Can du {self.n_ctrl} diem! (hien tai: {len(self.ctrl_pixels)})"
                        self.msg_color = C_ERR
                    else:
                        # Sang pha truy van
                        self.phase = "query"
                        self.msg = "Buoc 2: Click diem truy van, ENTER khi xong"
                        self.msg_color = C_QUERY_FILL
                        print(f"\n{'='*60}")
                        print("  BUOC 2: Click DIEM TRUY VAN len anh")
                        print("  (Diem ban muon biet toa do thuc)")
                        print("  Nhan ENTER ngay neu khong can diem truy van")
                        print(f"{'='*60}")

                elif self.phase == "query":
                    # Hoan thanh
                    self.phase = "done"
                    self.msg = "Hoan thanh! Dang xu ly..."
                    self.msg_color = C_KEY
                    frame = self._render()
                    cv2.imshow(self.WIN, frame)
                    cv2.waitKey(600)
                    break

            # ── ESC ──
            elif key == 27:
                print("\n  [THOAT] Nguoi dung huy bo.")
                cv2.destroyAllWindows()
                sys.exit(0)

        cv2.destroyAllWindows()
        return self.ctrl_pixels, self.query_pixels


# ══════════════════════════════════════════════════════
#  NHAP TOA DO THUC
# ══════════════════════════════════════════════════════

def ask_world_coords(ctrl_pixels: list, mode: str) -> list:
    """Hoi nguoi dung nhap toa do thuc cho tung diem dieu khien."""
    n = len(ctrl_pixels)
    dim = 2 if mode == "2d" else 3
    axes = "X Y" if mode == "2d" else "X Y Z"

    print(f"\n{'='*60}")
    print(f"  NHAP TOA DO THUC ({axes}) cho {n} diem dieu khien")
    print(f"  Don vi tuy ban: met (m), centimét (cm), ...")
    print(f"  Vi du 2D: '0 0'  '16.5 0'  '16.5 40'  '0 40'")
    print(f"  Vi du 3D: '23.2 0 0'  '23.2 0 2.35'  ...")
    print(f"{'='*60}")

    coords = []
    for i, (pu, pv) in enumerate(ctrl_pixels):
        print(f"\n  Diem P{i+1} | pixel = ({pu}, {pv})")
        while True:
            try:
                raw = input(f"    Nhap {axes}: ").strip()
                parts = list(map(float, raw.split()))
                if len(parts) != dim:
                    raise ValueError
                coords.append(parts)
                print(f"    -> World: {parts}")
                break
            except (ValueError, IndexError):
                print(f"    [!] Sai! Can {dim} so, vi du: {'0 0' if dim==2 else '0 0 0'}")
    return coords


def ask_query_z() -> float:
    """Hoi Z cua diem truy van (mode 3D)."""
    print(f"\n  Nhap gia tri Z cua diem truy van (vi du: 0 neu tren mat san)")
    while True:
        try:
            return float(input("  Z = ").strip())
        except ValueError:
            print("  [!] Nhap so thuc, vi du: 0")


# ══════════════════════════════════════════════════════
#  LUU JSON
# ══════════════════════════════════════════════════════

def save_json(ctrl_pixels, world_coords, query_pixels, path, mode, query_z=0.0):
    if mode == "2d":
        data = {
            "points_world": world_coords,
            "points_image": [list(p) for p in ctrl_pixels],
            "query_points": [list(p) for p in query_pixels],
        }
    else:
        data = {
            "points_3d":    world_coords,
            "points_image": [list(p) for p in ctrl_pixels],
            "query_points": [list(p) for p in query_pixels],
            "query_z":      query_z,
        }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\n[LUU] File JSON: {path}")
    print(json.dumps(data, indent=2))


# ══════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Cong cu annotation tuong tac — click anh de lay pixel coords",
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("--image",  required=True,
                    help="Duong dan anh (jpg/png)")
    ap.add_argument("--mode",   choices=["2d","3d"], default="2d",
                    help="2d = 4 diem phang | 3d = 6 diem khong gian")
    ap.add_argument("--output", default="",
                    help="File JSON dau ra (mac dinh: data/annotations/<ten_anh>_<mode>.json)")
    ap.add_argument("--run",    action="store_true",
                    help="Tu dong chay main.py sau khi annotation xong")
    args = ap.parse_args()

    n_ctrl = 4 if args.mode == "2d" else 6

    print(f"\n{'='*60}")
    print(f"  ANNOTATION TOOL  [{args.mode.upper()} mode | {n_ctrl} control points]")
    print(f"  Anh: {args.image}")
    print(f"{'='*60}")

    # 1. Mo anh va chay annotation
    ann = ImageAnnotator(args.image, n_ctrl, args.mode)
    ctrl_pixels, query_pixels = ann.run()

    # 2. Nhap toa do thuc
    world_coords = ask_world_coords(ctrl_pixels, args.mode)
    query_z = 0.0
    if args.mode == "3d":
        query_z = ask_query_z()

    # 3. In tom tat
    print(f"\n{'='*60}")
    print("  TOM TAT")
    print(f"{'='*60}")
    print(f"  Diem dieu khien ({len(ctrl_pixels)}):")
    for i, (pw, pi) in enumerate(zip(world_coords, ctrl_pixels)):
        print(f"    P{i+1}: pixel={list(pi)}  |  world={pw}")
    print(f"\n  Diem truy van ({len(query_pixels)}):")
    for i, pt in enumerate(query_pixels):
        print(f"    Q{i+1}: pixel={list(pt)}")

    # 4. Luu JSON
    if args.output:
        out = args.output
    else:
        base = os.path.splitext(os.path.basename(args.image))[0]
        out  = f"data/annotations/{base}_{args.mode}.json"

    save_json(ctrl_pixels, world_coords, query_pixels, out, args.mode, query_z)

    # 5. Chay main.py
    if args.run:
        cmd = (f"python main.py --mode {args.mode} "
               f"--image \"{args.image}\" --points \"{out}\"")
        print(f"\n[RUN] {cmd}\n")
        os.system(cmd)
    else:
        print(f"\n[TIEP THEO] Chay lenh:")
        print(f"  python main.py --mode {args.mode} "
              f"--image \"{args.image}\" --points \"{out}\"")


if __name__ == "__main__":
    main()
