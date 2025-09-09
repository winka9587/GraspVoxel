#!/usr/bin/env python3
"""
demo(will try to get camera stream):
    python wilor_video_fps_inference.py --heatmap_only

"""
from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import time
import open3d as o3d

from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import Renderer, cam_crop_to_full, RendererO3D
from ultralytics import YOLO
from visualization.optim import (
    GraspVolumeHeatmap, compute_mano_rays_in_cam,
    render_heatmap_image_only, overlay_heat_on_image
)

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
DISPLAY_COLOR = (0, 255, 0)  # show green info on origin image

# ---------- Open3D 尺度（把 WiLoR 输出整体缩小到可视化更合适的量级） ----------
VIS_SCALE = 1.0            # 1m -> 1cm in O3D space
DESIRED_Z_VIS = 0.52        # 目标深度(可视化坐标下)，越小越近（0.18~0.30 均可）

def extract_prob_and_grid(heat3d):
    """从 heat3d 拿 p3/voxel_size/origin；若 p3 未缓存则由 L 过 sigmoid 计算。"""
    if getattr(heat3d, "L", None) is None:
        # 空 heatmap 时给个最小占位，避免 None
        vox = float(getattr(heat3d, "voxel_size", 0.01))
        mrg = float(getattr(heat3d, "margin", 0.06))
        n = int(max(1, np.ceil((2 * mrg) / max(vox, 1e-9))))
        p3 = np.zeros((n, n, n), dtype=np.float32)
        origin = np.array([-mrg, -mrg, -mrg], dtype=np.float64)
        return p3, vox, origin

    p3 = heat3d.probability()
    if p3 is None:
        L = np.asarray(heat3d.L, dtype=np.float32)
        p3 = 1.0 / (1.0 + np.exp(-L))  # sigmoid
    p3 = np.asarray(p3, dtype=np.float32)
    vox = float(heat3d.voxel_size)
    origin = np.asarray(heat3d.origin_xyz, dtype=np.float64)
    return p3, vox, origin


def filter_active_voxels(p3, mode="quantile", thresh=0.20, q=90.0, cap=5000, min_keep=300):
    """
    过滤要画的体素索引：
      - mode="quantile" -> 取>=百分位(q) 与 >=thresh 的较高者
      - 至少保底 min_keep 个（全局 top-k）
      - 最多 cap 个（按概率从高到低截取）
    """
    p = np.asarray(p3, dtype=np.float32)
    nz_mask = p > 0
    if not np.any(nz_mask):
        return np.empty((0, 3), np.int32)

    if mode == "quantile":
        pq = float(np.percentile(p[nz_mask], q))
        thr = max(float(thresh), pq)
    else:
        thr = float(thresh)

    idxs = np.argwhere(p >= thr)

    if idxs.shape[0] < min_keep:
        flat = p.ravel()
        k = min(min_keep, flat.size)
        topk = np.argpartition(-flat, k-1)[:k]
        idxs = np.column_stack(np.unravel_index(topk, p.shape))

    if idxs.shape[0] > cap:
        vals = p[idxs[:, 0], idxs[:, 1], idxs[:, 2]]
        order = np.argsort(-vals)[:cap]
        idxs = idxs[order]
    return idxs

# ================== Heatmap 平滑（原逻辑） ==================
def temporal_smooth_heatmap(cur_heat_img_bgr, prev_heat_img_bgr,
                            min_alpha=0.60, max_alpha=0.95, adapt_scale=0.05):
    cur = cur_heat_img_bgr.astype(np.float32)
    if prev_heat_img_bgr is None:
        return cur.astype(np.uint8), cur  # pass first frame

    prev = prev_heat_img_bgr.astype(np.float32)
    diff = np.mean(np.abs(cur - prev)) / 255.0
    alpha = min_alpha + (max_alpha - min_alpha) * np.exp(-diff / max(adapt_scale, 1e-6))
    alpha = float(np.clip(alpha, min_alpha, max_alpha))
    smoothed = alpha * prev + (1.0 - alpha) * cur
    smoothed = cv2.GaussianBlur(smoothed, (0, 0), 0.6)
    return smoothed.astype(np.uint8), smoothed


def draw_2d_bbox(base_rgba: np.ndarray, boxes: torch.Tensor, color=(1.0, 0.0, 0.0), alpha=0.5, thickness=2):
    overlay_rgba = base_rgba.copy()
    h, w = base_rgba.shape[:2]
    overlay_bgr = np.ascontiguousarray((overlay_rgba[..., :3] * 255).astype(np.uint8)[..., ::-1])
    for box in boxes.cpu().numpy():
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2),
                      (int(color[2]*255), int(color[1]*255), int(color[0]*255)), thickness)
    overlay_rgb = overlay_bgr[..., ::-1].astype(np.float32) / 255.0
    overlay_rgba = np.concatenate([overlay_rgb, np.full((h, w, 1), alpha, dtype=np.float32)], axis=-1)
    return overlay_rgba


def _rotx_deg(angle):
    a = np.deg2rad(angle)
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=np.float64)


def _project_pinhole(X_cam, fx, fy, cx, cy):
    z = X_cam[:, 2:3]
    eps = 1e-9
    uv = np.empty((X_cam.shape[0], 2), dtype=np.float64)
    denom = (z[:, 0] + eps)
    uv[:, 0] = fx * (X_cam[:, 0] / denom) + cx
    uv[:, 1] = fy * (X_cam[:, 1] / denom) + cy
    return uv


def hand_silhouette_overlay(
    verts, faces, cam_t, img_w, img_h, focal_length,
    is_right=1, line_color=(0, 255, 0), thickness=1,
    apply_x_flip=False, apply_rotx_180=False,
    invert_x=False, invert_y=False
):
    V = np.asarray(verts, dtype=np.float64).copy()
    if apply_rotx_180:
        V = (_rotx_deg(180.0) @ V.T).T

    cam_t = np.asarray(cam_t, dtype=np.float64).copy()
    if apply_x_flip:
        cam_t[0] *= -1.0

    V_cam = V + cam_t
    valid_z = V_cam[:, 2] > 1e-6

    F = np.asarray(faces, dtype=np.int32)
    v0 = V_cam[F[:, 0]]
    v1 = V_cam[F[:, 1]]
    v2 = V_cam[F[:, 2]]

    n = np.cross(v1 - v0, v2 - v0)
    centers = (v0 + v1 + v2) / 3.0
    front = (np.sum(n * (-centers), axis=1) > 0)

    from collections import defaultdict
    edge2faces = defaultdict(list)

    def _add_edge(a, b, fi):
        if a > b:
            a, b = b, a
        edge2faces[(a, b)].append(fi)

    for fi, (a, b, c) in enumerate(F):
        _add_edge(a, b, fi)
        _add_edge(b, c, fi)
        _add_edge(c, a, fi)

    silhouette_edges = []
    for (a, b), flist in edge2faces.items():
        if len(flist) == 1:
            silhouette_edges.append((a, b))
        elif len(flist) == 2 and (front[flist[0]] != front[flist[1]]):
            silhouette_edges.append((a, b))

    fx = fy = float(focal_length)
    cx, cy = img_w / 2.0, img_h / 2.0
    uv = _project_pinhole(V_cam, fx, fy, cx, cy)

    if invert_x:
        uv[:, 0] = 2 * cx - uv[:, 0]
    if invert_y:
        uv[:, 1] = 2 * cy - uv[:, 1]

    overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    draw_view = overlay[:, :, :3]
    needs_copyback = False
    if not draw_view.flags['C_CONTIGUOUS']:
        draw = np.ascontiguousarray(draw_view)
        needs_copyback = True
    else:
        draw = draw_view

    c = (int(line_color[0]), int(line_color[1]), int(line_color[2]))
    for a, b in silhouette_edges:
        if not (valid_z[a] and valid_z[b]):
            continue
        x0, y0 = uv[a]
        x1, y1 = uv[b]
        coords = np.array([x0, y0, x1, y1], dtype=np.float64)
        if not np.isfinite(coords).all():
            continue
        p0 = (int(np.round(x0)), int(np.round(y0)))
        p1 = (int(np.round(x1)), int(np.round(y1)))
        cv2.line(draw, p0, p1, c, thickness=thickness, lineType=cv2.LINE_AA)

    if needs_copyback:
        overlay[:, :, :3] = draw
    coverage = overlay[:, :, :3].max(axis=2, keepdims=True).astype(np.float32) / 255.0
    overlay[:, :, :3] = np.where(coverage > 0, 255, overlay[:, :, :3])
    overlay[:, :, 3:4] = (coverage * 255).astype(np.uint8)
    return overlay, int(coverage.sum() > 0)


def project_full_img(points, cam_trans, focal_length, img_res):
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = camera_center[0]
    K[1, 2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:]
    V_2d = (K @ points.T).T
    return V_2d[..., :-1]


def draw_perf_info(img, inference_time, fps, history_times):
    cv2.putText(img, f"Inference: {inference_time*1000:.1f}ms",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DISPLAY_COLOR, 2)
    cv2.putText(img, f"FPS: {fps:.1f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DISPLAY_COLOR, 2)

    chart_height = 80
    chart_width = 200
    chart_x = img.shape[1] - chart_width - 10
    chart_y = img.shape[0] - chart_height - 10

    cv2.rectangle(img, (chart_x, chart_y),
                  (chart_x + chart_width, chart_y + chart_height),
                  (50, 50, 50), -1)

    max_time = max(history_times + [0.05])
    for i, t in enumerate(history_times):
        x = chart_x + int((i / len(history_times)) * chart_width)
        h = int((t / max_time) * chart_height)
        cv2.line(img, (x, chart_y + chart_height),
                 (x, chart_y + chart_height - h), DISPLAY_COLOR, 2)

    ref_time = 0.033
    ref_h = int((ref_time / max_time) * chart_height)
    cv2.line(img, (chart_x, chart_y + chart_height - ref_h),
             (chart_x + chart_width, chart_y + chart_height - ref_h),
             (0, 0, 255), 1)
    cv2.putText(img, "30FPS", (chart_x + 5, chart_y + chart_height - ref_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


# ========================= Open3D 可视化辅助 =========================
def _dense_regressor(J_regressor):
    try:
        return J_regressor.coalesce().to_dense().cpu().numpy()
    except Exception:
        return np.asarray(J_regressor.cpu().numpy())


def _is_finite_arr(x):
    x = np.asarray(x)
    return np.isfinite(x).all()


class _StableOffset:
    """ 平滑 & 限速的可视化平移（XY/Z 分开限速，EMA 平滑） """
    def __init__(self, momentum=0.90,
                 max_jump_xy=0.02, max_jump_z=0.06,
                 max_abs_xy=0.50, max_abs_z=1.00,
                 z_min=-0.20, z_max=0.80):
        self.ok = False
        self.v = np.zeros(3, dtype=np.float64)
        self.m = float(momentum)
        self.max_jump_xy = float(max_jump_xy)
        self.max_jump_z = float(max_jump_z)
        self.max_abs_xy = float(max_abs_xy)
        self.max_abs_z = float(max_abs_z)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

    def _clamp_step(self, dv):
        dv = np.asarray(dv, np.float64).reshape(3)
        planar = dv.copy()
        planar[2] = 0.0
        n = np.linalg.norm(planar)
        if n > self.max_jump_xy and n > 1e-12:
            planar *= (self.max_jump_xy / n)
        dv[0], dv[1] = planar[0], planar[1]
        dv[2] = np.clip(dv[2], -self.max_jump_z, self.max_jump_z)
        return dv

    def _clamp_abs(self, v):
        v = np.asarray(v, np.float64).reshape(3)
        v[0] = np.clip(v[0], -self.max_abs_xy, self.max_abs_xy)
        v[1] = np.clip(v[1], -self.max_abs_xy, self.max_abs_xy)
        v[2] = np.clip(v[2], -self.max_abs_z, self.max_abs_z)
        v[2] = np.clip(v[2], self.z_min, self.z_max)
        return v

    def update(self, new_off):
        if new_off is None or not _is_finite_arr(new_off):
            return
        new_off = np.asarray(new_off, np.float64).reshape(3)
        if not self.ok:
            self.v = self._clamp_abs(new_off)
            self.ok = True
            return
        dv = self._clamp_step(new_off - self.v)
        cand = self.v + dv
        self.v = self.m * self.v + (1.0 - self.m) * cand
        self.v = self._clamp_abs(self.v)

    def get(self):
        return self.v if self.ok else np.zeros(3, dtype=np.float64)


class SimpleHandAndVoxelsVis:
    """ 轻量稳定的 O3D 可视化（hand mesh + filtered voxels） """
    def __init__(self, mano_faces: np.ndarray):
        self.vis = None
        self.hand = None
        self.voxels = None
        self.lines = None
        self.faces = np.asarray(mano_faces, dtype=np.int32)
        self.cam_inited = False

    def start(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Open3D Hand + Voxels", width=1024, height=768, visible=True)

        self.hand = o3d.geometry.TriangleMesh()
        self.voxels = o3d.geometry.TriangleMesh()
        self.lines = o3d.geometry.LineSet()

        self.vis.add_geometry(self.hand)
        self.vis.add_geometry(self.voxels)
        self.vis.add_geometry(self.lines)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        self.vis.add_geometry(axis)

        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.98, 0.98, 0.98])
        opt.mesh_show_back_face = True
        opt.line_width = 1.0

        vc = self.vis.get_view_control()
        vc.set_front([0.0, 0.0, -1.0])
        vc.set_up([0.0, -1.0, 0.0])
        vc.set_lookat([0.0, 0.0, 0.0])
        vc.set_zoom(5)

        self._cam_template = vc.convert_to_pinhole_camera_parameters()

        self.cam_inited = True
        self.vis.poll_events()
        self.vis.update_renderer()

    def _restore_cam(self):  
        # 锁定相机位置和朝向
        try:
            vc = self.vis.get_view_control()
            vc.convert_from_pinhole_camera_parameters(self._cam_template, allow_arbitrary=True)
        except Exception:
            pass

    def stop(self):
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None

    def update_hand(self, verts_list_cam, offset=None):
        if not verts_list_cam:             # ★ 没有手，直接跳过
            return

        merged = o3d.geometry.TriangleMesh()
        F = self.faces.astype(np.int32)
        off = np.zeros(3) if offset is None else np.asarray(offset, np.float64).reshape(3)

        any_added = False
        for V in verts_list_cam:
            V = np.asarray(V, np.float64)
            if V.ndim != 2 or V.shape[1] != 3 or V.size == 0:
                continue
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices  = o3d.utility.Vector3dVector(V + off[None, :])
            mesh.triangles = o3d.utility.Vector3iVector(F)
            if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
                continue
            mesh.paint_uniform_color([0.75, 0.70, 0.95])
            mesh.compute_vertex_normals()
            merged += mesh
            any_added = True

        if not any_added or len(merged.vertices) == 0:
            return  # ★ 不加空几何，避免 AABB 警告

        try:
            self.vis.remove_geometry(self.hand, reset_bounding_box=False)
        except Exception:
            pass
        self.hand = merged
        self.vis.add_geometry(self.hand)
        self._restore_cam()
        self.vis.poll_events()
        self.vis.update_renderer()

    def value_to_rgb01(self, v: np.ndarray) -> np.ndarray:
        """
        0..1 -> B(0,0,1) 到 R(1,0,0) 的线性插值（蓝->红），返回 float RGB in [0,1]
        也可改成更丰富的 colormap（如 TURBO/PLASMA），这里按需求保持蓝红两端。
        """
        v = np.clip(v, 0.0, 1.0).astype(np.float32)
        # 蓝(0,0,1) -> 红(1,0,0)
        r = v
        g = np.zeros_like(v)
        b = 1.0 - v
        return np.stack([r, g, b], axis=-1)

    @staticmethod
    def _cube_corners(min_corner, s):
        x, y, z = min_corner
        return np.array([
            [x,   y,   z  ],
            [x+s, y,   z  ],
            [x,   y+s, z  ],
            [x+s, y+s, z  ],
            [x,   y,   z+s],
            [x+s, y,   z+s],
            [x,   y+s, z+s],
            [x+s, y+s, z+s],
        ], dtype=np.float64)

    def update_voxels(self, p3, idxs_zyx, voxel_size, origin, offset=None):
        if idxs_zyx is None or len(idxs_zyx) == 0:
            return  # ★ 没有体素可画，直接跳过

        off = np.zeros(3) if offset is None else np.asarray(offset, np.float64).reshape(3)
        all_mesh = o3d.geometry.TriangleMesh()
        lines_pts, lines_idx = [], []
        pt_ofs = 0

        for (iz, iy, ix) in idxs_zyx:
            v = float(p3[iz, iy, ix])
            color = self.value_to_rgb01(v)

            cx = origin[0] + (ix + 0.5) * voxel_size
            cy = origin[1] + (iy + 0.5) * voxel_size
            cz = origin[2] + (iz + 0.5) * voxel_size
            center = np.array([cx, cy, cz], np.float64) + off
            min_corner = center - 0.5 * voxel_size

            box = o3d.geometry.TriangleMesh.create_box(voxel_size, voxel_size, voxel_size)
            box.translate(min_corner)
            nv = np.asarray(box.vertices).shape[0]
            if nv == 0:
                continue
            box.vertex_colors = o3d.utility.Vector3dVector(
                np.tile(color.reshape(1, 3), (nv, 1))
            )
            all_mesh += box

            # 线框
            verts = self._cube_corners(min_corner, voxel_size)
            edges = np.array([
                [0,1],[0,2],[1,3],[2,3],
                [4,5],[4,6],[5,7],[6,7],
                [0,4],[1,5],[2,6],[3,7]
            ], dtype=np.int32) + pt_ofs
            pt_ofs += 8
            lines_pts.append(verts)
            lines_idx.append(edges)

        if len(all_mesh.vertices) == 0:
            return  # ★ 不加空几何

        # 面
        try:
            self.vis.remove_geometry(self.voxels, reset_bounding_box=False)
        except Exception:
            pass
        self.voxels = all_mesh
        self.vis.add_geometry(self.voxels)

        # 线
        if lines_pts:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(np.vstack(lines_pts))
            ls.lines  = o3d.utility.Vector2iVector(np.vstack(lines_idx))
            ls.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[0,0,0]], dtype=np.float64), (len(ls.lines), 1))
            )
            try:
                self.vis.remove_geometry(self.lines, reset_bounding_box=False)
            except Exception:
                pass
            self.lines = ls
            self.vis.add_geometry(self.lines)

        self._restore_cam()
        self.vis.poll_events()
        self.vis.update_renderer()

    # 提取 heatmap 体与网格信息
    @staticmethod
    def extract_prob_and_grid(heat3d: GraspVolumeHeatmap):
        if heat3d.L is None:
            vox = float(getattr(heat3d, "voxel_size", 0.01))
            mrg = float(getattr(heat3d, "margin", 0.06))
            n = int(max(1, np.ceil((2 * mrg) / max(vox, 1e-9))))
            p3 = np.zeros((n, n, n), dtype=np.float32)
            origin = np.array([-mrg, -mrg, -mrg], dtype=np.float64)
            return p3, vox, origin
        p3 = heat3d.probability()
        if p3 is None:
            L = np.asarray(heat3d.L, dtype=np.float32)
            p3 = 1.0 / (1.0 + np.exp(-L))
        p3 = np.asarray(p3, dtype=np.float32)
        vox = float(heat3d.voxel_size)
        origin = np.asarray(heat3d.origin_xyz, dtype=np.float64)
        return p3, vox, origin

    # 体素过滤 + 保底/上限
    @staticmethod
    def filter_active_voxels(p3: np.ndarray,
                             mode: str = "quantile",
                             thresh: float = 0.20,
                             q: float = 90.0,
                             cap: int = 5000,
                             min_keep: int = 200):
        p = np.asarray(p3, dtype=np.float32)
        nz = p > 0
        if not np.any(nz):
            return np.empty((0, 3), np.int32)
        if mode == "quantile":
            pq = float(np.percentile(p[nz], q))
            thr = max(float(thresh), pq)
        else:
            thr = float(thresh)
        idxs = np.argwhere(p >= thr)
        if idxs.shape[0] < min_keep:
            flat = p.ravel()
            k = min(min_keep, flat.size)
            topk = np.argpartition(-flat, k - 1)[:k]
            idxs = np.column_stack(np.unravel_index(topk, p.shape))
        if idxs.shape[0] > cap:
            vals = p[idxs[:, 0], idxs[:, 1], idxs[:, 2]]
            order = np.argsort(-vals)[:cap]
            idxs = idxs[order]
        return idxs

# ============================ 主流程 ============================
def main():
    parser = argparse.ArgumentParser(description='WiLoR video / webcam demo')
    parser.add_argument('--source', type=str, default='0',
                        help='摄像头 id 或视频文件路径')
    parser.add_argument('--out_folder', type=str, default='out_video',
                        help='输出目录')
    parser.add_argument('--save', action='store_true',
                        help='是否保存叠加后的视频帧')
    parser.add_argument('--save_mesh', action='store_true',
                        help='是否逐帧保存 hand mesh')
    parser.add_argument('--rescale_factor', type=float, default=2.0)
    parser.add_argument('--heatmap_only', action='store_true',
                        help='只渲染heatmap，跳过网格/轮廓/2D框/mesh导出等一切可视化')
    parser.add_argument('--no_o3d', action='store_true',
                        help='禁用 Open3D 3D 可视化（仅 2D 输出）')
    parser.add_argument('--voxel_mode', type=str, default='quantile',
                        choices=['quantile', 'thresh'])
    parser.add_argument('--voxel_thresh', type=float, default=0.20)
    parser.add_argument('--voxel_q', type=float, default=90.0)
    parser.add_argument('--voxel_cap', type=int, default=5000)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_wilor(
        checkpoint_path='./pretrained_models/wilor_final.ckpt',
        cfg_path='./pretrained_models/model_config.yaml')
    detector = YOLO('./pretrained_models/detector.pt')

    renderer = None
    if not args.heatmap_only:
        renderer = RendererO3D(cfg, faces=model.mano.faces)

    model = model.to(device).eval()
    detector = detector.to(device)

    os.makedirs(args.out_folder, exist_ok=True)

    # video source
    try:
        src = int(args.source)  # camera id
    except ValueError:
        src = args.source       # video file
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f'Cannot open {src}')
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    frame_id = 0

    # performance statistics variables
    inference_times = []
    fps_values = []
    frame_count = 0
    start_time = time.time()
    perf_history = []

    heat3d = GraspVolumeHeatmap(
        voxel_size=0.01,
        margin=0.06,
        decay=0.98
    )

    def update_grasp_heatmap(all_verts, all_cam_t, all_right):
        # 更新 GraspVolumeHeatmap
        for verts, cam_t, right_flag in zip(all_verts, all_cam_t, all_right):
            fids = list(range(len(all_verts)))
            for fid in fids:
                cam_t_used = cam_t
                is_right_n = int(right_flag)
                O_cam, D_cam, V_cam, pc, pn = compute_mano_rays_in_cam(
                    out, model, fid=fid, cam_t=cam_t_used, is_right_n=is_right_n,
                    axis='y-', apply_rotx_180=False, exclude_kps={13, 14, 15}
                )
                heat3d.update_with_rays(
                    O_list=O_cam, D_list=D_cam, V_cam_for_bbox=V_cam,
                    palm_center=pc, palm_normal=pn,
                    prefer_dist=0.08, sigma_para=0.03,
                    sigma_perp=0.01,
                    step=None,
                    alpha_hit=0.8, alpha_gate=0.5
                )

    # ------------ 渲染热图和 2D 投影 ----------------

    def render_interaction_region(img_cv2, fx, fy, cx, cy, prev_heat_img_state):
        # 渲染与热图交互区域
        H, W = img_cv2.shape[:2]
        heatmap_img = render_heatmap_image_only(
            heat3d, img_w=W, img_h=H, fx=fx, fy=fy, cx=cx, cy=cy,
            gamma=0.85, block_px=12, blur_px=0,
            colormap=cv2.COLORMAP_PLASMA,
            draw_grid=True, grid_step=16, grid_color=(48, 36, 64), grid_alpha=0.28
        )

        # 对热图进行平滑
        heatmap_img, prev_heat_img_state = temporal_smooth_heatmap(
            heatmap_img, prev_heat_img_state, 0.60, 0.95, 0.05
        )
        
        # 将热图与原始图像叠加
        interaction_overlay = overlay_heat_on_image(img_cv2, heatmap_img,
                                                    min_intensity=10,
                                                    alpha_gain=0.9,
                                                    alpha_gamma=0.75,
                                                    mode='screen')
        return interaction_overlay

    # ---------- Open3D 可视化器 ----------
    vis = None
    stable_offset = _StableOffset(
        momentum=0.90,
        max_jump_xy=0.02, max_jump_z=0.06,
        max_abs_xy=0.50, max_abs_z=1.00,
        z_min=-0.20, z_max=0.80
    )
    if not args.no_o3d:
        try:
            vis = SimpleHandAndVoxelsVis(mano_faces=model.mano.faces)
            vis.start()
        except Exception as e:
            print(f"[WARN] Open3D init failed, 3D view disabled: {e}")
            vis = None

    prev_heat_img_state = None
    resize_target = 640.0
    miss_counter = 0  # 连续“无手帧”计数

    while True:
        ok, img_cv2 = cap.read()
        if not ok:
            break
        else:
            h, w = img_cv2.shape[:2]
            scale = resize_target / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_cv2 = cv2.resize(img_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame_count += 1

        overlay = img_cv2.copy()
        overlay_2dbbox = img_cv2.copy()
        mano_ray_direction = img_cv2.copy()
        heat_img = img_cv2.copy()
        mesh_rendered = False

        current_time = time.time()
        inference_start = time.time()

        # --------- YOLO 检测 ---------
        # 手部检测
        detections = detector(img_cv2, conf=0.3, verbose=False)[0]

        bboxes, is_right = [], []
        for det in detections:
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            if Bbox.ndim == 1:
                Bbox = Bbox[None, :]
            for row in Bbox:
                x1, y1, x2, y2 = row[:4]
                bboxes.append([x1, y1, x2, y2])
                is_right.append(int(row[5]) if row.shape[0] >= 6 else 1)

        has_hand = len(bboxes) > 0

        if not has_hand:
            miss_counter += 1
            # 可选：加速衰减避免幽灵热点
            # for _ in range(min(6, 2 + miss_counter // 2)):
            #     heat3d.decay_step()
        else:
            miss_counter = 0

        overlay = img_cv2
        mesh_rendered = False

        all_verts, all_cam_t, all_right = [], [], []
        scaled_focal = None

        # 更新热图
        if has_hand:
            all_verts, all_cam_t, all_right = [], [], []
            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            dataset = ViTDetDataset(cfg, img_cv2, boxes, right,
                                    rescale_factor=args.rescale_factor)
            loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=len(bboxes),
                                                 shuffle=False,
                                                 num_workers=0)

            for batch in loader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)

                # === 还原到整图相机坐标：计算 scaled_focal & pred_cam_t_full ===
                multiplier = (2 * batch['right'] - 1)                    # 左右手镜像修正
                pred_cam = out['pred_cam']
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]

                box_center = batch['box_center'].float()
                box_size   = batch['box_size'].float()
                img_size   = batch['img_size'].float()

                # 按整图分辨率缩放焦距
                scaled_focal = (cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * img_size.max())

                # 从裁剪框坐标还原到整图相机坐标的相机平移
                pred_cam_t_full = cam_crop_to_full(
                    pred_cam, box_center, box_size, img_size, scaled_focal
                ).detach().cpu().numpy()


                # 更新手部姿势和摄像机位姿
                for n in range(batch['img'].shape[0]):
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                    is_right_n = batch['right'][n].cpu().numpy()
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)
                    all_right.append(is_right_n)

            # 通过射线更新体素概率
            if all_verts:
                update_grasp_heatmap(all_verts, all_cam_t, all_right)

            # 渲染交互区域热图并叠加
            H, W = img_cv2.shape[:2]
            if 'scaled_focal' not in locals() or scaled_focal is None:
                scaled_focal = float(max(H, W))  # 一个安全兜底
            fx = fy = float(scaled_focal)
            cx, cy = W / 2.0, H / 2.0

            overlay = render_interaction_region(img_cv2, fx, fy, cx, cy, prev_heat_img_state)
            cv2.imshow('Interaction Region', overlay)

        # --------- Open3D 3D 可视化（手 + 体素）---------
        # if vis is not None:
        #     try:
        #         # 把 verts+cam_t 转相机坐标，并缩放
        #         verts_list_cam = []
        #         for verts, cam_t in zip(all_verts, all_cam_t):
        #             V_cam = (verts + np.asarray(cam_t, dtype=np.float64).reshape(1, 3)) * VIS_SCALE
        #             verts_list_cam.append(V_cam)

        #         # 1) 手腕（相机坐标）
        #         # —— 计算一次性平移，让手腕到原点 ——
        #         Jr = _dense_regressor(model.mano.J_regressor)
        #         if(len(all_verts)>0):
        #             wrist_cam = (Jr @ all_verts[0])[0] + np.asarray(all_cam_t[0], np.float64).reshape(3)
        #             # —— 计算一次性平移，让手腕到原点（或固定深度）——
        #             off = None
        #             if len(all_verts) > 0:
        #                 Jr = _dense_regressor(model.mano.J_regressor)
        #                 V0 = np.asarray(all_verts[0], np.float64)
        #                 T0 = np.asarray(all_cam_t[0], np.float64).reshape(3)
        #                 wrist_cam = (Jr @ V0)[0] + T0
        #                 # 固定在原点：off = -wrist_cam
        #                 # 或固定在某深度 z0：off = np.array([-wrist_cam[0], -wrist_cam[1], z0 - wrist_cam[2]])
        #                 off = -wrist_cam


        #             # —— 手：先变到相机坐标，再整体平移 off —— 
        #             verts_list_cam = []
        #             for V, T in zip(all_verts, all_cam_t):
        #                 V_cam = np.asarray(V, np.float64) + np.asarray(T, np.float64).reshape(1, 3)
        #                 V_vis = V_cam + off[None, :]              # 平移到 wrist=0
        #                 verts_list_cam.append(V_vis)

        #             # —— 体素：取 p3 / voxel_size / origin，并对 origin 应用同一个 off —— 
        #             p3, voxel_size, origin = extract_prob_and_grid(heat3d)
        #             idxs = filter_active_voxels(
        #                 p3, mode="quantile", thresh=0.20, q=90.0, cap=5000, min_keep=300
        #             )
        #             origin_vis = origin + off                       # 体素整体同样平移

        #             print(f"verts_list_cam: {verts_list_cam}, p3: {p3}, voxel_size: {voxel_size}, origin_vis: {origin_vis}")

        #             # —— 喂给 Open3D —— 
        #             # vis.update_hand(verts_list_cam, offset=None)    # 你已经把 off 应在数据上，这里 offset=None
        #             # vis.update_voxels(p3, idxs, voxel_size, origin_vis, offset=None)

        #     except Exception as e:
        #         print(f"[WARN] Open3D update failed: {e}")
        #         vis = None  # 出错后关闭 3D，继续 2D

                    # --------- Open3D 3D 可视化（手 + 体素）---------
            if vis is not None:
                try:
                    # 1) 如果本帧没手，直接跳过 3D 更新，避免空几何触发相机变化
                    if len(all_verts) == 0:
                        pass
                    else:
                        # 2) 计算一次性平移，把“手腕”放到原点
                        Jr = _dense_regressor(model.mano.J_regressor)
                        V0 = np.asarray(all_verts[0], np.float64)
                        T0 = np.asarray(all_cam_t[0], np.float64).reshape(3)
                        wrist_cam = (Jr @ V0)[0] + T0           # 相机坐标下的手腕
                        off = -wrist_cam                        # 让手腕到原点
                        off = off.astype(np.float64)

                        # 3) 手：统一做一次变换 -> (V + T + off) * VIS_SCALE
                        verts_list_cam = []
                        for V, T in zip(all_verts, all_cam_t):
                            V_cam = np.asarray(V, np.float64) + np.asarray(T, np.float64).reshape(1, 3)
                            V_vis = (V_cam + off[None, :]) * VIS_SCALE
                            verts_list_cam.append(V_vis)

                        # 4) 体素：同样一次性变换（只对可视化副本，绝不原地改）
                        p3, voxel_size, origin = extract_prob_and_grid(heat3d)
                        idxs = filter_active_voxels(
                            p3, mode=args.voxel_mode, thresh=args.voxel_thresh,
                            q=args.voxel_q, cap=args.voxel_cap, min_keep=300
                        )
                        if idxs is not None and len(idxs) > 0:
                            origin_vis = (np.asarray(origin, np.float64) + off) * VIS_SCALE
                            voxel_size_vis = float(voxel_size) * VIS_SCALE
                        else:
                            origin_vis = None
                            voxel_size_vis = None

                        # 5) 喂给 Open3D（空保护）
                        if verts_list_cam and len(verts_list_cam[0]) > 0:
                            vis.update_hand(verts_list_cam, offset=None)

                        if origin_vis is not None and voxel_size_vis is not None and len(idxs) > 0:
                            vis.update_voxels(p3, idxs, voxel_size_vis, origin_vis, offset=None)

                except Exception as e:
                    print(f"[WARN] Open3D update failed: {e}")
                    vis = None  # 出错后关闭 3D，继续 2D


        # --------- 性能/FPS & 窗口 ---------
        inference_time = time.time() - inference_start
        if mesh_rendered:
            inference_times.append(inference_time)

        if not args.heatmap_only:
            elapsed_time = current_time - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
                fps_values.append(fps)
            perf_history.append(inference_time)
            if len(perf_history) > 30:
                perf_history.pop(0)
            avg_inference_time = np.mean(inference_times[-10:]) if inference_times else 0
            avg_fps = np.mean(fps_values[-10:]) if fps_values else 0
            draw_perf_info(overlay, avg_inference_time, avg_fps, perf_history)
            cv2.imshow('Hand Recon', overlay)
            cv2.imshow('Detection Result', overlay_2dbbox)

        cv2.imshow('interaction region', mano_ray_direction)
        cv2.imshow('heatmap only', heat_img)

        # 写视频
        if args.save and writer is None:
            h, w = overlay.shape[:2]
            writer = cv2.VideoWriter(
                os.path.join(args.out_folder, 'result.mp4'),
                fourcc, 25, (w, h))
        if writer is not None:
            writer.write(overlay)

        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if vis is not None:
        vis.stop()


if __name__ == '__main__':
    main()
