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
from visualization.optim import GraspVolumeHeatmap, compute_mano_rays_in_cam

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
DISPLAY_COLOR = (0, 255, 0)  # 绿色显示性能信息



def temporal_smooth_heatmap(cur_heat_img_bgr, prev_heat_img_bgr,
                            min_alpha=0.60, max_alpha=0.95, adapt_scale=0.05):
    """
    对彩色热图做自适应 EMA:
      out = alpha * prev + (1 - alpha) * cur
    其中 alpha ∈ [min_alpha, max_alpha]，随帧间差异自适应变化。
    """
    cur = cur_heat_img_bgr.astype(np.float32)
    if prev_heat_img_bgr is None:
        return cur.astype(np.uint8), cur  # 首帧直接通过

    prev = prev_heat_img_bgr.astype(np.float32)

    # 用平均像素差估计帧间变化（0..1）
    diff = np.mean(np.abs(cur - prev)) / 255.0
    # 差异越小 -> alpha 越大（更平滑）；差异大 -> alpha 降低（减少拖影）
    alpha = min_alpha + (max_alpha - min_alpha) * np.exp(-diff / max(adapt_scale, 1e-6))
    alpha = float(np.clip(alpha, min_alpha, max_alpha))

    smoothed = alpha * prev + (1.0 - alpha) * cur

    # 可选：轻微空间去噪（极小高斯核），进一步抑制抖动
    smoothed = cv2.GaussianBlur(smoothed, (0, 0), 0.6)

    return smoothed.astype(np.uint8), smoothed  # 第二个返回 float32 作为下次的 prev


# 绘制2D包围盒
def draw_2d_bbox(base_rgba: np.ndarray, boxes: torch.Tensor, color=(1.0, 0.0, 0.0), alpha=0.5, thickness=2):
    """
    在 RGBA overlay 上绘制 2D 包围盒，并返回 overlay_rgba。

    Parameters
    ----------
    base_rgba : np.ndarray
        原始图像，float32 [H, W, 4]，值范围 [0,1]，RGBA
    boxes : torch.Tensor
        包围盒 (N,4)，格式 [x1,y1,x2,y2]
    color : tuple
        颜色 (R,G,B)，范围 [0,1]
    alpha : float
        透明度 (0=透明, 1=不透明)
    thickness : int
        边框线宽

    Returns
    -------
    overlay_rgba : np.ndarray
        绘制后的 overlay (RGBA, float32, [0,1])
    """
    overlay_rgba = base_rgba.copy()
    h, w = base_rgba.shape[:2]

    # 转换成 uint8 BGR，方便用 cv2 绘制
    # overlay_bgr = (overlay_rgba[..., :3] * 255).astype(np.uint8)[..., ::-1]
    overlay_bgr = np.ascontiguousarray((overlay_rgba[..., :3] * 255).astype(np.uint8)[..., ::-1])


    for box in boxes.cpu().numpy():
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(overlay_bgr, (x1, y1), (x2, y2), (int(color[2]*255), int(color[1]*255), int(color[0]*255)), thickness)

    # 转回 float32 RGBA
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
    apply_x_flip=False,      # 默认与 project_full_img 对齐：不对 cam_t[0] 取负
    apply_rotx_180=False,    # 默认不绕 X 轴 180°
    invert_x=False,          # 如左右颠倒则置 True
    invert_y=False           # 如上下颠倒则置 True（OpenCV 常见需要=False；若你看到上下反了就置 True）
):
    V = np.asarray(verts, dtype=np.float64).copy()

    if apply_rotx_180:
        V = (_rotx_deg(180.0) @ V.T).T

    cam_t = np.asarray(cam_t, dtype=np.float64).copy()
    if apply_x_flip:
        cam_t[0] *= -1.0

    # 与 project_full_img 一样：直接加 cam_t
    V_cam = V + cam_t  # (N,3)
    valid_z = V_cam[:, 2] > 1e-6

    F = np.asarray(faces, dtype=np.int32)
    v0 = V_cam[F[:, 0]]
    v1 = V_cam[F[:, 1]]
    v2 = V_cam[F[:, 2]]

    # 视向判定：front-facing
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

    # 投影（与 project_full_img 同式）
    fx = fy = float(focal_length)
    cx, cy = img_w / 2.0, img_h / 2.0
    uv = _project_pinhole(V_cam, fx, fy, cx, cy)

    # 可选镜像修正（在像平面绕中心镜像）
    if invert_x:
        uv[:, 0] = 2 * cx - uv[:, 0]
    if invert_y:
        uv[:, 1] = 2 * cy - uv[:, 1]

    # 叠加图，确保 OpenCV 可写
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
    args = parser.parse_args()

    # ------------ 1. 模型加载 ------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_wilor(
        checkpoint_path='./pretrained_models/wilor_final.ckpt',
        cfg_path='./pretrained_models/model_config.yaml')
    detector = YOLO('./pretrained_models/detector.pt')
    # renderer = Renderer(cfg, faces=model.mano.faces)
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

    
    heat3d = GraspVolumeHeatmap(
        voxel_size=0.01,   # 体素边长 ~1cm；若无尺度，用 bbox 尺度：把 0.01 改成 0.01*diag
        margin=0.06,
        decay=0.98
    )

    prev_heat_img_state = None
    resize_target = 640.0
    while True:
        ok, img_cv2 = cap.read()
        if not ok:
            break
        else:
            # resize img max w/h to resize_target
            h, w = img_cv2.shape[:2]
            scale = resize_target / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)

            # 缩放
            img_cv2 = cv2.resize(img_cv2, (new_w, new_h), interpolation=cv2.INTER_AREA)
        frame_count += 1
        

        # safe value
        # —— 每帧安全默认值 —— 
        overlay = img_cv2.copy()          # 手重建/叠加主图
        overlay_2dbbox = img_cv2.copy()   # 2D 检测框可视化
        mano_ray_direction = img_cv2.copy()
        heat_img = img_cv2.copy()
        mesh_rendered = False             # 是否真的完成了重建/渲染

        current_time = time.time()
        # 记录推理开始时间
        inference_start = time.time()
        
        # 执行检测和推理
        detection_start_time = time.time()
        detections = detector(img_cv2, conf=0.3, verbose=False)[0]
        detection_end_time = time.time()
        print(f"Detector inference time: {detection_end_time - detection_start_time:.3f} seconds")
        
        bboxes, is_right = [], []
        for det in detections:
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())

        overlay = img_cv2
        mesh_rendered = False
        
        if len(bboxes) > 0:
            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            dataset = ViTDetDataset(cfg, img_cv2, boxes, right,
                                  rescale_factor=args.rescale_factor)
            loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=len(bboxes),
                                               shuffle=False,
                                               num_workers=0)

            all_verts, all_cam_t, all_right = [], [], []
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
                scaled_focal = (cfg.EXTRA.FOCAL_LENGTH /
                              cfg.MODEL.IMAGE_SIZE * img_size.max())
                pred_cam_t_full = cam_crop_to_full(
                    pred_cam, box_center, box_size, img_size,
                    scaled_focal).detach().cpu().numpy()

                for n in range(batch['img'].shape[0]):
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                    is_right_n = batch['right'][n].cpu().numpy()
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
                            os.path.join(args.out_folder,
                                       f'{frame_id:06d}_{n}.obj'))

            
            if all_verts:
                H, W = img_cv2.shape[0], img_cv2.shape[1]
                # 先把原图转到 RGBA，便于叠加
                base_rgba = np.concatenate(
                    [img_cv2[..., ::-1].astype(np.float32) / 255.0,
                    np.ones((H, W, 1), dtype=np.float32)], axis=2)


                # 多只手的轮廓叠到同一张 overlay 上
                overlay_rgba = np.zeros((H, W, 4), dtype=np.float32)
                for verts, cam_t, right_flag in zip(all_verts, all_cam_t, all_right):
                    rgba_i, _ = hand_silhouette_overlay(
                        verts=verts,
                        faces=model.mano.faces,   # 使用右手 faces；如果你有 faces_left 并想严格区分，可按 right_flag 选择
                        cam_t=cam_t,
                        img_w=W, img_h=H,
                        focal_length=float(scaled_focal),
                        is_right=int(right_flag),
                        line_color=(0, 255, 0),
                        thickness=2,
                    )
                    overlay_rgba += rgba_i.astype(np.float32) / 255.0  # 多个轮廓简单叠加（线不重合基本没问题）

                # 裁剪 alpha 到 [0,1]
                overlay_rgba = np.clip(overlay_rgba, 0.0, 1.0)

                # Alpha 合成到原图 (前景=overlay)
                out_rgb = base_rgba[..., :3] * (1.0 - overlay_rgba[..., 3:]) + overlay_rgba[..., :3] * overlay_rgba[..., 3:]
                # overlay = np.clip(out_rgb * 255, 0, 255).astype(np.uint8)[..., ::-1]  # 回到 BGR 给 OpenCV 显示
                out_u8 = np.clip(out_rgb * 255, 0, 255).astype(np.uint8)   # 连续的 RGB uint8
                overlay = np.ascontiguousarray(out_u8[..., ::-1])          # 变为 BGR，并确保连续
                
                # 绘制2D包围盒
                print("detections.boxes.xyxy:", detections.boxes.xyxy)
                overlay_bbox = draw_2d_bbox(base_rgba, detections.boxes.xyxy, color=(0,1,0), alpha=0.6)
                out_rgb_bbox = base_rgba[..., :3] * (1.0 - overlay_bbox[..., 3:]) + overlay_bbox[..., :3] * overlay_bbox[..., 3:]
                out_u8_bbox = np.clip(out_rgb_bbox * 255, 0, 255).astype(np.uint8)
                overlay_2dbbox = np.ascontiguousarray(out_u8_bbox[..., ::-1])
                mesh_rendered = True

                
                num_hands = len(all_cam_t)
                if num_hands > 0:
                    fids = list(range(num_hands))
                    heat3d.decay_step()  # 每帧开始：遗忘一点

                    # 对每只手（与你多手循环一致）：
                    #   先得到 O_cam, D_cam, V_cam, palm_center, palm_normal
                    #   调用 update_with_rays 累积到热图
                    for n in range(num_hands):
                        cam_t_used = all_cam_t[n]
                        is_right_n = int(all_right[n])

                        O_cam, D_cam, V_cam, pc, pn = compute_mano_rays_in_cam(
                            out, model, fid=fids[n], cam_t=cam_t_used, is_right_n=is_right_n,
                            axis='y-', apply_rotx_180=False, exclude_kps={13,14,15}
                        )

                        # 累积更新
                        heat3d.update_with_rays(
                            O_list=O_cam, D_list=D_cam, V_cam_for_bbox=V_cam,
                            palm_center=pc, palm_normal=pn,
                            prefer_dist=0.08, sigma_para=0.03,
                            sigma_perp=0.0,                 # ★ 先关掉横向 splat
                            step=None,                      # ★ 自适应，不会比体素更密
                            alpha_hit=0.8, alpha_gate=0.5
                        )


                from visualization.optim import render_heatmap_image_only, overlay_heat_on_image
                fx = fy = float(scaled_focal); cx, cy = W/2.0, H/2.0
                heat_img = render_heatmap_image_only(
                    heat3d, img_w=W, img_h=H, fx=fx, fy=fy, cx=cx, cy=cy,
                    gamma=0.85, block_px=12, blur_px=0,
                    colormap=cv2.COLORMAP_PLASMA,  # 或 cv2.COLORMAP_TURBO
                    draw_grid=True, grid_step=16, grid_color=(48,36,64), grid_alpha=0.28
                )

                heat_img, prev_heat_img_state = temporal_smooth_heatmap(
                    heat_img, prev_heat_img_state,
                    min_alpha=0.60,   # 越接近 1 越平滑
                    max_alpha=0.95,   # 上限；可调到 0.98 更稳
                    adapt_scale=0.05  # 帧间差异的“灵敏度”，越小越敏感
                )

                
                mano_ray_direction = overlay_heat_on_image(img_cv2, heat_img,
                                                    min_intensity=10,
                                                    alpha_gain=0.9,
                                                    alpha_gamma=0.75,
                                                    mode='screen')

        # 计算推理时间(从开始推理到手模渲染完成)
        inference_time = time.time() - inference_start
        if mesh_rendered:  # 只有当手模实际渲染时才记录时间
            inference_times.append(inference_time)
        
        # 计算FPS
        elapsed_time = current_time - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            fps_values.append(fps)
        
        # 更新性能历史记录(最多保留30帧)
        perf_history.append(inference_time)
        if len(perf_history) > 30:
            perf_history.pop(0)
        
        # 计算平均性能指标
        avg_inference_time = np.mean(inference_times[-10:]) if inference_times else 0
        avg_fps = np.mean(fps_values[-10:]) if fps_values else 0
        
        # 在图像上绘制性能信息
        draw_perf_info(overlay, avg_inference_time, avg_fps, perf_history)

        cv2.imshow('Hand Recon', overlay)  # hand 重建结果
        cv2.imshow('Detection Result', overlay_2dbbox)  # detection的2D包围盒
        cv2.imshow('interaction region', mano_ray_direction)
        cv2.imshow('heatmap only', heat_img)


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

    # 打印最终性能统计
    if inference_times:
        print("\n性能统计:")
        print(f"平均推理时间: {np.mean(inference_times)*1000:.1f}ms")
        print(f"最小推理时间: {np.min(inference_times)*1000:.1f}ms")
        print(f"最大推理时间: {np.max(inference_times)*1000:.1f}ms")
        print(f"平均FPS: {np.mean(fps_values):.1f}")

if __name__ == '__main__':
    main()