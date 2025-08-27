#!/usr/bin/env python3
"""
wilor_video_fps.py
用法：
    # 摄像头
    python wilor_video_fps.py --source 0

    # 离线视频
    python wilor_video_fps.py --source myvideo.mp4 --save --save_mesh
"""
from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import time
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import Renderer, cam_crop_to_full, RendererO3D
from ultralytics import YOLO

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
DISPLAY_COLOR = (0, 255, 0)  # 绿色显示性能信息

import numpy as np
import cv2
from collections import defaultdict


# -------------------- 新增：从 mesh 投影生成 2D 框 --------------------
def _project_pinhole_np(X_cam, fx, fy, cx, cy, eps=1e-9):
    z = X_cam[:, 2:3]
    denom = (z[:, 0] + eps)
    uv = np.empty((X_cam.shape[0], 2), dtype=np.float64)
    uv[:, 0] = fx * (X_cam[:, 0] / denom) + cx
    uv[:, 1] = fy * (X_cam[:, 1] / denom) + cy
    return uv

def verts_to_xyxy(
    verts: np.ndarray,
    cam_t: np.ndarray,
    img_w: int, img_h: int,
    focal_length: float,
    margin: float = 0.05,      # 额外扩大 5%
    min_size: int = 10,        # 框最小边阈值（像素）
    clip: bool = True
):
    """
    根据顶点 + 相机平移 cam_t 进行投影，返回 xyxy 框（像素坐标）。
    verts: (V,3) in model coords (已根据左右手做过 x 轴镜像则直接用)
    cam_t: (3,)  相机平移（与原代码一致：直接 V + cam_t）
    """
    V = np.asarray(verts, dtype=np.float64)
    cam_t = np.asarray(cam_t, dtype=np.float64)
    V_cam = V + cam_t
    valid = V_cam[:, 2] > 1e-6
    if not np.any(valid):
        return None  # 全部不可见，视作无效

    fx = fy = float(focal_length)
    cx, cy = img_w / 2.0, img_h / 2.0
    uv = _project_pinhole_np(V_cam, fx, fy, cx, cy)

    # 仅统计有限 & 深度有效的点
    finite = np.isfinite(uv).all(axis=1)
    mask = valid & finite
    if not np.any(mask):
        return None

    u = uv[mask, 0]
    v = uv[mask, 1]
    x1, y1, x2, y2 = u.min(), v.min(), u.max(), v.max()

    # 扩边
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    x1 -= w * margin
    x2 += w * margin
    y1 -= h * margin
    y2 += h * margin

    # 裁剪到图像范围
    if clip:
        x1 = max(0.0, min(float(img_w - 1), x1))
        y1 = max(0.0, min(float(img_h - 1), y1))
        x2 = max(0.0, min(float(img_w - 1), x2))
        y2 = max(0.0, min(float(img_h - 1), y2))

    # 尺寸最小限制
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None

    return [float(x1), float(y1), float(x2), float(y2)]


# 绘制2D包围盒
def draw_2d_bbox(base_rgba: np.ndarray, boxes: torch.Tensor, color=(1.0, 0.0, 0.0), alpha=0.5, thickness=2):
    """
    在 RGBA overlay 上绘制 2D 包围盒，并返回 overlay_rgba。
    base_rgba: float32 [H,W,4] in [0,1], RGBA
    boxes: (N,4) torch.Tensor [x1,y1,x2,y2]
    """
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
    """X_cam: (N,3) in camera coords -> (N,2) pixels"""
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
    apply_x_flip=False,
    apply_rotx_180=False,
    invert_x=False,
    invert_y=False
):
    V = np.asarray(verts, dtype=np.float64).copy()

    if apply_rotx_180:
        V = (_rotx_deg(180.0) @ V.T).T

    cam_t = np.asarray(cam_t, dtype=np.float64).copy()
    if apply_x_flip:
        cam_t[0] *= -1.0

    V_cam = V + cam_t  # (N,3)
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
        if a > b: a, b = b, a
        edge2faces[(a, b)].append(fi)
    for fi, (a, b, c) in enumerate(F):
        _add_edge(a, b, fi); _add_edge(b, c, fi); _add_edge(c, a, fi)

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
    cnt = 0
    for a, b in silhouette_edges:
        if not (valid_z[a] and valid_z[b]): continue
        x0, y0 = uv[a]; x1, y1 = uv[b]
        coords = np.array([x0, y0, x1, y1], dtype=np.float64)
        if not np.isfinite(coords).all(): continue
        p0 = (int(np.round(x0)), int(np.round(y0)))
        p1 = (int(np.round(x1)), int(np.round(y1)))
        cv2.line(draw, p0, p1, c, thickness=thickness, lineType=cv2.LINE_AA)
        cnt += 1

    if needs_copyback:
        overlay[:, :, :3] = draw
    coverage = overlay[:, :, :3].max(axis=2, keepdims=True).astype(np.float32) / 255.0
    overlay[:, :, :3] = np.where(coverage > 0, 255, overlay[:, :, :3])
    alpha_value = 1.0
    overlay[:, :, 3:4] = (coverage * (alpha_value * 255)).astype(np.uint8)

    return overlay, cnt


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
    
    max_time = max(history_times + [0.05])  # 至少显示50ms的范围
    
    for i, t in enumerate(history_times):
        x = chart_x + int((i / len(history_times)) * chart_width)
        h = int((t / max_time) * chart_height)
        cv2.line(img, (x, chart_y + chart_height), 
                (x, chart_y + chart_height - h), DISPLAY_COLOR, 2)

    ref_time = 0.033  # 30FPS的参考线(33.3ms)
    ref_h = int((ref_time / max_time) * chart_height)
    cv2.line(img, (chart_x, chart_y + chart_height - ref_h),
            (chart_x + chart_width, chart_y + chart_height - ref_h),
            (0, 0, 255), 1)
    cv2.putText(img, "30FPS", (chart_x + 5, chart_y + chart_height - ref_h - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


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
    # 新增：追踪失败时回退到检测的选项
    parser.add_argument('--fallback_detect', action='store_true',
                      help='若由投影生成的框无效/丢失，则回退到检测一次')
    args = parser.parse_args()

    # ------------ 1. 模型加载 ------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_wilor(
        checkpoint_path='./pretrained_models/wilor_final.ckpt',
        cfg_path='./pretrained_models/model_config.yaml')
    detector = YOLO('./pretrained_models/detector.pt')
    renderer = RendererO3D(cfg, faces=model.mano.faces)
    model = model.to(device).eval()
    detector = detector.to(device)

    os.makedirs(args.out_folder, exist_ok=True)

    # ------------ 2. 打开视频源 ------------
    try:
        src = int(args.source)  # 摄像头 id
    except ValueError:
        src = args.source       # 视频文件
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f'无法打开 {src}')
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    frame_id = 0

    # 性能统计变量
    inference_times = []
    fps_values = []
    frame_count = 0
    start_time = time.time()
    perf_history = []

    resize_target = 640.0

    # -------------------- 新增：保留上一帧的框与左右手标签 --------------------
    prev_boxes_xyxy = None     # list[list[4]]，本帧用于喂给 ViTDetDataset 的框
    prev_right_flags = None    # list[int]  (0/1)

    while True:
        ok, img_cv2 = cap.read()
        if not ok:
            break
        else:
            # resize img max w/h to resize_target
            h, w = img_cv2.shape[:2]
            scale = resize_target / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_cv2 = cv2.resize(img_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame_count += 1
        current_time = time.time()
        inference_start = time.time()

        H, W = img_cv2.shape[0], img_cv2.shape[1]
        overlay = img_cv2
        overlay_2dbbox = img_cv2.copy()
        mesh_rendered = False

        # -------------------- 第一帧（或需要回退）才运行 detector --------------------
        run_detector_this_frame = (frame_id == 0)
        detections = None

        # 若启用回退，并且上一帧没有有效框，则本帧回退检测
        if args.fallback_detect and prev_boxes_xyxy is not None:
            if len(prev_boxes_xyxy) == 0:  # 明确无框
                run_detector_this_frame = True

        if run_detector_this_frame:
            detection_start_time = time.time()
            detections = detector(img_cv2, conf=0.3, verbose=False)[0]
            detection_end_time = time.time()
            print(f"Detector inference time: {detection_end_time - detection_start_time:.3f} seconds")

            bboxes_init, is_right_init = [], []
            for det in detections:
                Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                is_right_init.append(int(det.boxes.cls.cpu().detach().squeeze().item()))
                bboxes_init.append(Bbox[:4].tolist())

            prev_boxes_xyxy = bboxes_init
            prev_right_flags = is_right_init

        # -------------------- 用 “prev_boxes_xyxy / prev_right_flags” 作为本帧输入 --------------------
        if prev_boxes_xyxy is not None and len(prev_boxes_xyxy) > 0:
            boxes = np.stack(prev_boxes_xyxy, axis=0)
            right = np.array(prev_right_flags, dtype=np.int32)
            dataset = ViTDetDataset(cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
            loader = torch.utils.data.DataLoader(dataset, batch_size=len(prev_boxes_xyxy),
                                                 shuffle=False, num_workers=0)
        else:
            # 没框时直接显示，并准备下一帧（可回退检测）
            cv2.imshow('Hand Recon', overlay)
            cv2.imshow('Detection Result', overlay_2dbbox)
            frame_id += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # 让下一帧回退检测
            if args.fallback_detect:
                prev_boxes_xyxy = []  # 标记无框
            continue

        # -------------------- WiLoR 重建 --------------------
        all_verts, all_cam_t, all_right = [], [], []
        scaled_focal = None

        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                recon_start_time = time.time()
                out = model(batch)
                recon_end_time = time.time()
                print(f"WiLoR inference time: {recon_end_time - recon_start_time:.3f} seconds")

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch['box_center'].float()
            box_size = batch['box_size'].float()
            img_size = batch['img_size'].float()
            scaled_focal = (cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * img_size.max())
            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size,
                scaled_focal).detach().cpu().numpy()

            for n in range(batch['img'].shape[0]):
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                is_right_n = int(batch['right'][n].detach().cpu().numpy())
                # 左右镜像（与原实现一致）
                verts[:, 0] = (2 * is_right_n - 1) * verts[:, 0]
                joints[:, 0] = (2 * is_right_n - 1) * joints[:, 0]
                cam_t = pred_cam_t_full[n]

                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right_n)

                if args.save_mesh:
                    tmesh = renderer.vertices_to_trimesh(
                        verts, cam_t, LIGHT_PURPLE, is_right=is_right_n)
                    tmesh.export(
                        os.path.join(args.out_folder, f'{frame_id:06d}_{n}.obj'))

        # -------------------- 可视化 mesh 轮廓叠加 --------------------
        if len(all_verts) > 0:
            # 基图转 RGBA
            base_rgba = np.concatenate(
                [img_cv2[..., ::-1].astype(np.float32) / 255.0,
                 np.ones((H, W, 1), dtype=np.float32)], axis=2)

            overlay_rgba = np.zeros((H, W, 4), dtype=np.float32)
            for verts, cam_t, right_flag in zip(all_verts, all_cam_t, all_right):
                rgba_i, _ = hand_silhouette_overlay(
                    verts=verts,
                    faces=model.mano.faces,
                    cam_t=cam_t,
                    img_w=W, img_h=H,
                    focal_length=float(scaled_focal),
                    is_right=int(right_flag),
                    line_color=(0, 255, 0),
                    thickness=2,
                )
                overlay_rgba += rgba_i.astype(np.float32) / 255.0

            overlay_rgba = np.clip(overlay_rgba, 0.0, 1.0)
            out_rgb = base_rgba[..., :3] * (1.0 - overlay_rgba[..., 3:]) + overlay_rgba[..., :3] * overlay_rgba[..., 3:]
            out_u8 = np.clip(out_rgb * 255, 0, 255).astype(np.uint8)
            overlay = np.ascontiguousarray(out_u8[..., ::-1])  # BGR
            mesh_rendered = True

        # -------------------- 用 “当前帧重建结果” 生成新的 2D 框（供下一帧使用） --------------------
        new_boxes = []
        new_right_flags = []
        if len(all_verts) > 0:
            for verts, cam_t, right_flag in zip(all_verts, all_cam_t, all_right):
                xyxy = verts_to_xyxy(
                    verts=verts,
                    cam_t=cam_t,
                    img_w=W, img_h=H,
                    focal_length=float(scaled_focal),
                    margin=0.05, min_size=10, clip=True
                )
                if xyxy is not None:
                    new_boxes.append(xyxy)
                    new_right_flags.append(int(right_flag))

        # 显示目前的 2D 框（来自：第一帧 detector 或上一帧投影）
        base_rgba_vis = np.concatenate(
            [img_cv2[..., ::-1].astype(np.float32) / 255.0,
             np.ones((H, W, 1), dtype=np.float32)], axis=2)
        if prev_boxes_xyxy is not None and len(prev_boxes_xyxy) > 0:
            prev_boxes_tensor = torch.tensor(prev_boxes_xyxy, dtype=torch.float32)
            overlay_bbox = draw_2d_bbox(base_rgba_vis, prev_boxes_tensor, color=(0, 1, 0), alpha=0.6)
            out_rgb_bbox = base_rgba_vis[..., :3] * (1.0 - overlay_bbox[..., 3:]) + overlay_bbox[..., :3] * overlay_bbox[..., 3:]
            out_u8_bbox = np.clip(out_rgb_bbox * 255, 0, 255).astype(np.uint8)
            overlay_2dbbox = np.ascontiguousarray(out_u8_bbox[..., ::-1])

        # 计算推理时间(从开始推理到手模渲染完成)
        inference_time = time.time() - inference_start
        if mesh_rendered:
            inference_times.append(inference_time)

        # FPS
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

        cv2.imshow('Hand Recon', overlay)               # hand 重建结果
        # cv2.imshow('Detection Result', overlay_2dbbox)  # 本帧使用的“输入框”的可视化

        # 保存视频（叠加重建结果）
        if args.save and writer is None:
            hh, ww = overlay.shape[:2]
            writer = cv2.VideoWriter(
                os.path.join(args.out_folder, 'result.mp4'),
                fourcc, 25, (ww, hh))
        if writer is not None:
            writer.write(overlay)

        # -------------------- 更新用于“下一帧”的框 --------------------
        # 若本帧通过 mesh 投影得到的框为空，并且允许回退，下帧会重新检测
        if len(new_boxes) > 0:
            prev_boxes_xyxy = new_boxes
            prev_right_flags = new_right_flags
        else:
            prev_boxes_xyxy = [] if args.fallback_detect else prev_boxes_xyxy
            # 若未启用回退，则沿用上一帧的框（可能导致短暂漂移）

        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # 打印最终性能统计
    if inference_times:
        print("\n性能统计:")
        print(f"平均推理时间: {np.mean(inference_times)*1000:.1f}ms")
        print(f"最小推理时间: {np.min(inference_times)*1000:.1f}ms")
        print(f"最大推理时间: {np.max(inference_times)*1000:.1f}ms")
        print(f"平均FPS: {np.mean(fps_values):.1f}")


if __name__ == '__main__':
    main()
