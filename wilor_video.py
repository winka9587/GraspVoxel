#!/usr/bin/env python3
"""
wilor_video.py
WiLoR 实时视频/摄像头 demo
用法：
    # 摄像头
    python wilor_video.py --source 0

    # 离线视频
    python wilor_video.py --source myvideo.mp4 --save --save_mesh
"""
from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)


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
    renderer = Renderer(cfg, faces=model.mano.faces)
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

    while True:
        ok, img_cv2 = cap.read()
        if not ok:
            break

        detections = detector(img_cv2, conf=0.3, verbose=False)[0]
        bboxes, is_right = [], []
        for det in detections:
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())

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
            else:
                overlay = img_cv2
        else:
            overlay = img_cv2

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


if __name__ == '__main__':
    main()