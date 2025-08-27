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
    """在图像上绘制性能信息"""
    # 显示当前推理时间和FPS
    cv2.putText(img, f"Inference: {inference_time*1000:.1f}ms", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DISPLAY_COLOR, 2)
    cv2.putText(img, f"FPS: {fps:.1f}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, DISPLAY_COLOR, 2)
    
    # 绘制推理时间历史图表
    chart_height = 80
    chart_width = 200
    chart_x = img.shape[1] - chart_width - 10
    chart_y = img.shape[0] - chart_height - 10
    
    # 绘制背景
    cv2.rectangle(img, (chart_x, chart_y), 
                 (chart_x + chart_width, chart_y + chart_height), 
                 (50, 50, 50), -1)
    
    # 计算最大时间值用于缩放
    max_time = max(history_times + [0.05])  # 至少显示50ms的范围
    
    # 绘制时间线
    for i, t in enumerate(history_times):
        x = chart_x + int((i / len(history_times)) * chart_width)
        h = int((t / max_time) * chart_height)
        cv2.line(img, (x, chart_y + chart_height), 
                (x, chart_y + chart_height - h), DISPLAY_COLOR, 2)
    
    # 绘制参考线
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

    while True:
        ok, img_cv2 = cap.read()
        if not ok:
            break

        frame_count += 1
        current_time = time.time()
        
        # 记录推理开始时间
        inference_start = time.time()
        
        # 执行检测和推理
        detections = detector(img_cv2, conf=0.3, verbose=False)[0]
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
                    out = model(batch)

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
                misc_args = dict(mesh_base_color=LIGHT_PURPLE,
                               scene_bg_color=(1, 1, 1),
                               focal_length=scaled_focal)
                cam_view = renderer.render_rgba_multiple(
                    all_verts, cam_t=all_cam_t,
                    render_res=(img_cv2.shape[1], img_cv2.shape[0]),
                    is_right=all_right, **misc_args)

                input_img = img_cv2.astype(np.float32)[..., ::-1] / 255.0
                input_img = np.concatenate(
                    [input_img, np.ones_like(input_img[..., :1])], axis=2)
                overlay = input_img[..., :3] * (1 - cam_view[..., 3:]) + \
                          cam_view[..., :3] * cam_view[..., 3:]
                overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
                mesh_rendered = True

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
        
        cv2.imshow('WiLoR video', overlay)
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