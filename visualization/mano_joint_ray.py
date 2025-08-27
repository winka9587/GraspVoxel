import numpy as np
import torch
import cv2

# ---------- 小工具 ----------
def _rotx_deg(deg: float):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]], dtype=np.float64)


def _dense_regressor(J_regressor):
    try:
        Jr = J_regressor.coalesce().to_dense().cpu().numpy()
    except Exception:
        Jr = np.asarray(J_regressor.cpu().numpy())
    return Jr

def compute_mano_joint_rays_mano_overlay_single(
    out, model,
    img_w, img_h,
    cam_t,
    focal_length,
    axis='y-',
    is_right_n=1,
    line_color=(0, 0, 255), thickness=2, alpha_value=1.0,
    apply_x_flip=False, apply_rotx_180=False,
    invert_x=False, invert_y=False,
    ray_len=None, fid=0,
    verts_preflipped=None,   # 若外部已做左手 x 取反，就把那份 verts 传进来
):
    """单手版本（与之前一致）：返回 overlay(uint8 RGBA) 与绘制条数。"""
    is_right = (int(is_right_n) == 1)
    s = 1.0 if is_right else -1.0

    # 顶点（MANO坐标）：与 silhouette 一致的“左手 x 取反”
    if verts_preflipped is not None:
        verts = np.asarray(verts_preflipped, dtype=np.float64).copy()
    else:
        verts = out['pred_vertices'][fid].detach().cpu().numpy().astype(np.float64)
        verts[:, 0] *= s

    # 关节（回归器）：仅有 model.mano 时，直接使用它
    Jr = _dense_regressor(model.mano.J_regressor)
    joints = Jr @ verts
    if joints.shape[0] < 16:
        return np.zeros((img_h, img_w, 4), np.uint8), 0

    # 姿态旋转
    g = out['pred_mano_params']['global_orient']
    R_global = (g[fid, 0] if g.ndim == 4 else g[fid]).detach().cpu().numpy()
    R_hand   = out['pred_mano_params']['hand_pose'][fid].detach().cpu().numpy()  # (15,3,3)

    # 基轴
    if isinstance(axis, str):
        amap = {'x':[1,0,0],'y':[0,1,0],'z':[0,0,1],'x-':[-1,0,0],'y-':[0,-1,0],'z-':[0,0,-1]}
        axis_vec = np.array(amap.get(axis.lower(), [0,-1,0]), dtype=np.float64)
    else:
        axis_vec = np.asarray(axis, dtype=np.float64)
        axis_vec /= (np.linalg.norm(axis_vec) + 1e-12)

    # 左手镜像方向（与几何一致）
    S = np.diag([s, 1.0, 1.0])

    # 生成射线（MANO坐标）
    origins, directions = [], []
    for j_idx, kp_idx in enumerate(range(1, 16)):
        if kp_idx >= joints.shape[0]:
            continue
        d = R_global @ (R_hand[j_idx] @ axis_vec)
        d = (S @ d)
        nrm = np.linalg.norm(d)
        if not np.isfinite(d).all() or nrm < 1e-8:
            continue
        origins.append(joints[kp_idx])
        directions.append(d / nrm)

    if not origins:
        return np.zeros((img_h, img_w, 4), np.uint8), 0

    O = np.asarray(origins, dtype=np.float64)
    D = np.asarray(directions, dtype=np.float64)

    # rotx180 → +cam_t
    if apply_rotx_180:
        R180 = _rotx_deg(180.0)
        O = (R180 @ O.T).T
        D = (R180 @ D.T).T

    cam_t = np.asarray(cam_t, dtype=np.float64).copy()
    if apply_x_flip:
        cam_t[0] *= -1.0
    O_cam = O + cam_t

    # 自适应长度（基于 V_cam）
    V = verts.copy()
    if apply_rotx_180:
        V = (_rotx_deg(180.0) @ V.T).T
    V_cam = V + cam_t
    if ray_len is None:
        bbox = V_cam.max(axis=0) - V_cam.min(axis=0)
        L = 0.15 * float(np.max(bbox) if np.isfinite(bbox).all() else 1.0)
    else:
        L = float(ray_len)

    P0 = O_cam
    P1 = O_cam + D * L

    # 针孔投影
    fx = fy = float(focal_length)
    cx, cy = img_w / 2.0, img_h / 2.0
    def _proj(P):
        z = P[:, 2]
        valid = z > 1e-6
        z_safe = np.where(valid, z, 1.0)
        u = (P[:, 0] / z_safe) * fx + cx
        v = (P[:, 1] / z_safe) * fy + cy
        return np.stack([u, v], -1), valid

    uv0, m0 = _proj(P0)
    uv1, m1 = _proj(P1)
    mask = m0 & m1

    if invert_x:
        uv0[:, 0] = 2 * cx - uv0[:, 0]; uv1[:, 0] = 2 * cx - uv1[:, 0]
    if invert_y:
        uv0[:, 1] = 2 * cy - uv0[:, 1]; uv1[:, 1] = 2 * cy - uv1[:, 1]

    # 绘制到 overlay
    overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    draw_view = overlay[:, :, :3]
    draw = draw_view if draw_view.flags['C_CONTIGUOUS'] else np.ascontiguousarray(draw_view)

    c = (int(line_color[0]), int(line_color[1]), int(line_color[2]))
    cnt = 0
    for i in range(len(P0)):
        if not mask[i]: 
            continue
        x0, y0 = uv0[i]; x1, y1 = uv1[i]
        if not np.isfinite([x0, y0, x1, y1]).all():
            continue
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)), int(round(y1)))
        cv2.circle(draw, p0, radius=max(1, thickness+1), color=c, thickness=-1, lineType=cv2.LINE_AA)
        cv2.arrowedLine(draw, p0, p1, c, thickness=thickness, tipLength=0.25, line_type=cv2.LINE_AA)
        cnt += 1

    if draw is not draw_view:
        overlay[:, :, :3] = draw
    cov = overlay[:, :, :3].max(axis=2, keepdims=True).astype(np.float32) / 255.0
    overlay[:, :, :3] = np.where(cov > 0, 255, overlay[:, :, :3])
    overlay[:, :, 3:4] = (cov * (alpha_value * 255)).astype(np.uint8)

    return overlay, cnt


def compute_mano_joint_rays_mano_overlay_multi(
    out, model,
    img_w, img_h,
    cam_t_list,                # list[(3,)] 或 (N,3) np.ndarray
    focal_length,              # 标量或 list/ndarray 长度 N
    is_right_list,             # list/ndarray，元素为 0/1
    axis='y-',
    line_color=(0, 0, 255), thickness=2, alpha_value=1.0,
    apply_x_flip=False, apply_rotx_180=False,
    invert_x=False, invert_y=False,
    ray_len=None,
    fids=None,                 # 可选：每只手对应的 out 批次索引；默认 0..N-1
    verts_list_preflipped=None # 可选：若外部已对左手做 x 取反后的 verts 列表
):
    """多手版本：把每只手的射线叠加到同一张 overlay 上。"""
    # 归一化输入为列表
    cam_t_list = np.asarray(cam_t_list, dtype=np.float64)
    if cam_t_list.ndim == 1:
        cam_t_list = cam_t_list.reshape(1, 3)
    N = cam_t_list.shape[0]

    if isinstance(focal_length, (list, tuple, np.ndarray)):
        focal_list = list(np.asarray(focal_length).reshape(-1))
        if len(focal_list) == 1 and N > 1:
            focal_list = focal_list * N
    else:
        focal_list = [float(focal_length)] * N

    is_right_list = list(np.asarray(is_right_list).astype(int).reshape(-1))
    if len(is_right_list) != N:
        # 若给的 is_right 数量不匹配 cam_t 数量，做简单广播
        is_right_list = (is_right_list * N)[:N]

    if fids is None:
        fids = list(range(N))
    else:
        fids = list(np.asarray(fids).astype(int).reshape(-1))
        if len(fids) != N:
            raise ValueError("fids 长度必须与手的数量一致")

    if verts_list_preflipped is not None:
        verts_list_preflipped = [np.asarray(v, dtype=np.float64) for v in verts_list_preflipped]
        if len(verts_list_preflipped) != N:
            raise ValueError("verts_list_preflipped 长度必须与手的数量一致")

    # 累加 overlay（float）
    acc = np.zeros((img_h, img_w, 4), dtype=np.float32)
    total_cnt = 0

    for i in range(N):
        verts_i = None if verts_list_preflipped is None else verts_list_preflipped[i]
        ov_i, cnt_i = compute_mano_joint_rays_mano_overlay_single(
            out, model,
            img_w, img_h,
            cam_t=cam_t_list[i],
            focal_length=focal_list[i],
            axis=axis,
            is_right_n=is_right_list[i],
            line_color=line_color, thickness=thickness, alpha_value=alpha_value,
            apply_x_flip=apply_x_flip, apply_rotx_180=apply_rotx_180,
            invert_x=invert_x, invert_y=invert_y,
            ray_len=ray_len, fid=fids[i],
            verts_preflipped=verts_i
        )
        acc += ov_i.astype(np.float32) / 255.0
        total_cnt += cnt_i

    # 裁剪到 [0,1] 并转回 uint8
    acc = np.clip(acc, 0.0, 1.0)
    overlay = (acc * 255.0 + 0.5).astype(np.uint8)
    return overlay, total_cnt
