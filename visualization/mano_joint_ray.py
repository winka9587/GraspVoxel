import numpy as np
import torch
import cv2

# # ---------- 小工具 ----------
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

# def compute_mano_joint_rays_mano_overlay_single(
#     out, model,
#     img_w, img_h,
#     cam_t,
#     focal_length,
#     axis='y-',
#     is_right_n=1,
#     line_color=(0, 0, 255), thickness=2, alpha_value=1.0,
#     apply_x_flip=False, apply_rotx_180=False,
#     invert_x=False, invert_y=False,
#     ray_len=None, fid=0,
#     verts_preflipped=None,   # 若外部已做左手 x 取反，就把那份 verts 传进来
# ):
#     """单手版本（与之前一致）：返回 overlay(uint8 RGBA) 与绘制条数。"""
#     is_right = (int(is_right_n) == 1)
#     s = 1.0 if is_right else -1.0

#     # 顶点（MANO坐标）：与 silhouette 一致的“左手 x 取反”
#     if verts_preflipped is not None:
#         verts = np.asarray(verts_preflipped, dtype=np.float64).copy()
#     else:
#         verts = out['pred_vertices'][fid].detach().cpu().numpy().astype(np.float64)
#         verts[:, 0] *= s

#     # 关节（回归器）：仅有 model.mano 时，直接使用它
#     Jr = _dense_regressor(model.mano.J_regressor)
#     joints = Jr @ verts
#     if joints.shape[0] < 16:
#         return np.zeros((img_h, img_w, 4), np.uint8), 0

#     # 姿态旋转
#     g = out['pred_mano_params']['global_orient']
#     R_global = (g[fid, 0] if g.ndim == 4 else g[fid]).detach().cpu().numpy()
#     R_hand   = out['pred_mano_params']['hand_pose'][fid].detach().cpu().numpy()  # (15,3,3)

#     # 基轴
#     if isinstance(axis, str):
#         amap = {'x':[1,0,0],'y':[0,1,0],'z':[0,0,1],'x-':[-1,0,0],'y-':[0,-1,0],'z-':[0,0,-1]}
#         axis_vec = np.array(amap.get(axis.lower(), [0,-1,0]), dtype=np.float64)
#     else:
#         axis_vec = np.asarray(axis, dtype=np.float64)
#         axis_vec /= (np.linalg.norm(axis_vec) + 1e-12)

#     # 左手镜像方向（与几何一致）
#     S = np.diag([s, 1.0, 1.0])

#     # 生成射线（MANO坐标）
#     origins, directions = [], []
#     for j_idx, kp_idx in enumerate(range(1, 16)):
#         if kp_idx >= joints.shape[0]:
#             continue
#         d = R_global @ (R_hand[j_idx] @ axis_vec)
#         d = (S @ d)
#         nrm = np.linalg.norm(d)
#         if not np.isfinite(d).all() or nrm < 1e-8:
#             continue
#         origins.append(joints[kp_idx])
#         directions.append(d / nrm)

#     if not origins:
#         return np.zeros((img_h, img_w, 4), np.uint8), 0

#     O = np.asarray(origins, dtype=np.float64)
#     D = np.asarray(directions, dtype=np.float64)

#     # rotx180 → +cam_t
#     if apply_rotx_180:
#         R180 = _rotx_deg(180.0)
#         O = (R180 @ O.T).T
#         D = (R180 @ D.T).T

#     cam_t = np.asarray(cam_t, dtype=np.float64).copy()
#     if apply_x_flip:
#         cam_t[0] *= -1.0
#     O_cam = O + cam_t

#     # 自适应长度（基于 V_cam）
#     V = verts.copy()
#     if apply_rotx_180:
#         V = (_rotx_deg(180.0) @ V.T).T
#     V_cam = V + cam_t
#     if ray_len is None:
#         bbox = V_cam.max(axis=0) - V_cam.min(axis=0)
#         L = 0.15 * float(np.max(bbox) if np.isfinite(bbox).all() else 1.0)
#     else:
#         L = float(ray_len)

#     P0 = O_cam
#     P1 = O_cam + D * L

#     # 针孔投影
#     fx = fy = float(focal_length)
#     cx, cy = img_w / 2.0, img_h / 2.0
#     def _proj(P):
#         z = P[:, 2]
#         valid = z > 1e-6
#         z_safe = np.where(valid, z, 1.0)
#         u = (P[:, 0] / z_safe) * fx + cx
#         v = (P[:, 1] / z_safe) * fy + cy
#         return np.stack([u, v], -1), valid

#     uv0, m0 = _proj(P0)
#     uv1, m1 = _proj(P1)
#     mask = m0 & m1

#     if invert_x:
#         uv0[:, 0] = 2 * cx - uv0[:, 0]; uv1[:, 0] = 2 * cx - uv1[:, 0]
#     if invert_y:
#         uv0[:, 1] = 2 * cy - uv0[:, 1]; uv1[:, 1] = 2 * cy - uv1[:, 1]

#     # 绘制到 overlay
#     overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
#     draw_view = overlay[:, :, :3]
#     draw = draw_view if draw_view.flags['C_CONTIGUOUS'] else np.ascontiguousarray(draw_view)

#     c = (int(line_color[0]), int(line_color[1]), int(line_color[2]))
#     cnt = 0
#     for i in range(len(P0)):
#         if not mask[i]: 
#             continue
#         x0, y0 = uv0[i]; x1, y1 = uv1[i]
#         if not np.isfinite([x0, y0, x1, y1]).all():
#             continue
#         p0 = (int(round(x0)), int(round(y0)))
#         p1 = (int(round(x1)), int(round(y1)))
#         cv2.circle(draw, p0, radius=max(1, thickness+1), color=c, thickness=-1, lineType=cv2.LINE_AA)
#         cv2.arrowedLine(draw, p0, p1, c, thickness=thickness, tipLength=0.25, line_type=cv2.LINE_AA)
#         cnt += 1

#     if draw is not draw_view:
#         overlay[:, :, :3] = draw
#     cov = overlay[:, :, :3].max(axis=2, keepdims=True).astype(np.float32) / 255.0
#     overlay[:, :, :3] = np.where(cov > 0, 255, overlay[:, :, :3])
#     overlay[:, :, 3:4] = (cov * (alpha_value * 255)).astype(np.uint8)

#     return overlay, cnt


# def compute_mano_joint_rays_mano_overlay_multi(
#     out, model,
#     img_w, img_h,
#     cam_t_list,                # list[(3,)] 或 (N,3) np.ndarray
#     focal_length,              # 标量或 list/ndarray 长度 N
#     is_right_list,             # list/ndarray，元素为 0/1
#     axis='y-',
#     line_color=(0, 0, 255), thickness=2, alpha_value=1.0,
#     apply_x_flip=False, apply_rotx_180=False,
#     invert_x=False, invert_y=False,
#     ray_len=None,
#     fids=None,                 # 可选：每只手对应的 out 批次索引；默认 0..N-1
#     verts_list_preflipped=None # 可选：若外部已对左手做 x 取反后的 verts 列表
# ):
#     """多手版本：把每只手的射线叠加到同一张 overlay 上。"""
#     # 归一化输入为列表
#     cam_t_list = np.asarray(cam_t_list, dtype=np.float64)
#     if cam_t_list.ndim == 1:
#         cam_t_list = cam_t_list.reshape(1, 3)
#     N = cam_t_list.shape[0]

#     if isinstance(focal_length, (list, tuple, np.ndarray)):
#         focal_list = list(np.asarray(focal_length).reshape(-1))
#         if len(focal_list) == 1 and N > 1:
#             focal_list = focal_list * N
#     else:
#         focal_list = [float(focal_length)] * N

#     is_right_list = list(np.asarray(is_right_list).astype(int).reshape(-1))
#     if len(is_right_list) != N:
#         # 若给的 is_right 数量不匹配 cam_t 数量，做简单广播
#         is_right_list = (is_right_list * N)[:N]

#     if fids is None:
#         fids = list(range(N))
#     else:
#         fids = list(np.asarray(fids).astype(int).reshape(-1))
#         if len(fids) != N:
#             raise ValueError("fids 长度必须与手的数量一致")

#     if verts_list_preflipped is not None:
#         verts_list_preflipped = [np.asarray(v, dtype=np.float64) for v in verts_list_preflipped]
#         if len(verts_list_preflipped) != N:
#             raise ValueError("verts_list_preflipped 长度必须与手的数量一致")

#     # 累加 overlay（float）
#     acc = np.zeros((img_h, img_w, 4), dtype=np.float32)
#     total_cnt = 0

#     for i in range(N):
#         verts_i = None if verts_list_preflipped is None else verts_list_preflipped[i]
#         ov_i, cnt_i = compute_mano_joint_rays_mano_overlay_single(
#             out, model,
#             img_w, img_h,
#             cam_t=cam_t_list[i],
#             focal_length=focal_list[i],
#             axis=axis,
#             is_right_n=is_right_list[i],
#             line_color=line_color, thickness=thickness, alpha_value=alpha_value,
#             apply_x_flip=apply_x_flip, apply_rotx_180=apply_rotx_180,
#             invert_x=invert_x, invert_y=invert_y,
#             ray_len=ray_len, fid=fids[i],
#             verts_preflipped=verts_i
#         )
#         acc += ov_i.astype(np.float32) / 255.0
#         total_cnt += cnt_i

#     # 裁剪到 [0,1] 并转回 uint8
#     acc = np.clip(acc, 0.0, 1.0)
#     overlay = (acc * 255.0 + 0.5).astype(np.uint8)
#     return overlay, total_cnt


# # ===================

# import numpy as np, cv2

# # ====== 方向约束：把“掌心前半空间 + 指尖方向锥体”引入到区域提取 ======

# def _norm(v, eps=1e-12):
#     n = np.linalg.norm(v, axis=-1, keepdims=True)
#     return v / (n + eps)

# def _make_dir_grid(W, H, fx, fy, cx, cy):
#     xs = (np.arange(W, dtype=np.float32) - cx) / fx
#     ys = (np.arange(H, dtype=np.float32) - cy) / fy
#     X, Y = np.meshgrid(xs, ys)                       # (H,W)
#     D = np.stack([X, Y, np.ones_like(X)], axis=-1)   # (H,W,3)
#     return _norm(D.astype(np.float32))               # 相机系每像素视线方向

# def _axis_to_vec(axis):
#     if isinstance(axis, str):
#         amap = {'x':[1,0,0],'y':[0,1,0],'z':[0,0,1],
#                 'x-':[-1,0,0],'y-':[0,-1,0],'z-':[0,0,-1]}
#         return np.array(amap.get(axis.lower(), [0,-1,0]), dtype=np.float64)
#     v = np.asarray(axis, dtype=np.float64)
#     return v / (np.linalg.norm(v) + 1e-12)

# def _estimate_palm_normal_cam(joints_cam, tip_dirs_cam):
#     """
#     joints_cam: (>=16,3) 相机坐标的关节（0=腕）
#     tip_dirs_cam: (K,3)   指尖方向（相机系）
#     """
#     wrist = joints_cam[0]
#     # 经验：用食指/小指 MCP 近似掌面法向（按你的回归器关节顺序可调整索引）
#     idx_mcp, little_mcp = 4, 13
#     v1 = joints_cam[idx_mcp]   - wrist
#     v2 = joints_cam[little_mcp]- wrist
#     n  = _norm(np.cross(v1, v2)[None, :])[0]
#     if tip_dirs_cam.size > 0:
#         mean_tip = _norm(np.mean(tip_dirs_cam, axis=0, keepdims=True))[0]
#         if np.dot(n, mean_tip) < 0:
#             n = -n
#     return n

# def compute_interaction_region_from_overlay_dir(
#     overlay_ray,                      # (H,W,4) uint8——你已画好的射线RGBA
#     out, model,
#     img_w, img_h,
#     cam_t_list,                       # list[(3,)] 或 (N,3)
#     focal_length,                     # 标量或 (N,)；同你绘制用的 focal
#     is_right_list,                    # 0/1，左/右手
#     fids=None,                        # 每只手对应 out 的 batch 索引，默认 0..N-1
#     verts_list_preflipped=None,       # 若外部已做左手 x 取反后的 verts，传入可避免重复
#     axis='y-',
#     apply_rotx_180=False, apply_x_flip=False,
#     # —— 可调参数 —— 
#     # expand_px=12,                     # 在 alpha 带上再做一次小膨胀以连通
#     # theta_front_deg=70,               # 掌心前半空间开角，越小越“紧”
#     # phi_finger_deg=35,                # 指尖方向锥角，越小越“紧”
#     # min_area=300                      # 面积过滤，去掉小噪声
#     expand_px=16,                     # 在 alpha 带上再做一次小膨胀以连通
#     theta_front_deg=80,               # 掌心前半空间开角，越小越“紧”
#     phi_finger_deg=40,                # 指尖方向锥角，越小越“紧”
#     min_area=300                      # 面积过滤，去掉小噪声
# ):
#     H, W = img_h, img_w
#     min_area = 0.0003*H*W
#     alpha = overlay_ray[:, :, 3]
#     # 从 RGBA 的 alpha 得到“射线像素带”
#     ray_bin = (alpha > 8).astype(np.uint8)
#     if expand_px and expand_px > 0:
#         k = int(max(3, round(expand_px)))
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
#         band = cv2.dilate(ray_bin, kernel, iterations=1)
#     else:
#         band = ray_bin

#     # 相机内参 → 像素视线方向场
#     if isinstance(focal_length, (list, tuple, np.ndarray)):
#         fx = fy = float(np.asarray(focal_length).reshape(-1)[0])
#     else:
#         fx = fy = float(focal_length)
#     cx, cy = W/2.0, H/2.0
#     dir_grid = _make_dir_grid(W, H, fx, fy, cx, cy)     # (H,W,3)

#     cam_t_list = np.asarray(cam_t_list, dtype=np.float64)
#     if cam_t_list.ndim == 1: cam_t_list = cam_t_list.reshape(1,3)
#     N = cam_t_list.shape[0]

#     is_right_list = np.asarray(is_right_list).astype(int).reshape(-1)
#     if is_right_list.size != N:
#         is_right_list = (list(is_right_list) * N)[:N]
#         is_right_list = np.asarray(is_right_list)

#     if fids is None:
#         fids = np.arange(N, dtype=int)
#     else:
#         fids = np.asarray(fids, dtype=int)

#     cos_front = float(np.cos(np.deg2rad(theta_front_deg)))
#     cos_tip   = float(np.cos(np.deg2rad(phi_finger_deg)))

#     # 稀疏关节回归器（单 mano）
#     Jr = _dense_regressor(model.mano.J_regressor)

#     # 末节关节索引（按你的 joints 顺序可调整）
#     tip_kp_indices = [3, 6, 9, 12, 15]

#     R180 = _rotx_deg(180.0) if apply_rotx_180 else None
#     axis_vec = _axis_to_vec(axis)

#     acc_mask = np.zeros((H, W), np.uint8)

#     for i in range(N):
#         fid = int(fids[i])
#         is_right = bool(is_right_list[i])
#         s = 1.0 if is_right else -1.0
#         S = np.diag([s, 1.0, 1.0])            # 左手镜像（x 取反）

#         # 顶点（MANO坐标，左手需 x 取反）
#         if verts_list_preflipped is not None:
#             verts = np.asarray(verts_list_preflipped[i], dtype=np.float64).copy()
#         else:
#             verts = out['pred_vertices'][fid].detach().cpu().numpy().astype(np.float64)
#             verts[:, 0] *= s

#         # 关节（MANO坐标）
#         joints = Jr @ verts                    # (nJ,3)

#         # 姿态旋转
#         g  = out['pred_mano_params']['global_orient']
#         Rh = out['pred_mano_params']['hand_pose']
#         Rg = (g[fid,0] if g.ndim == 4 else g[fid]).detach().cpu().numpy()      # (3,3)
#         Rl = Rh[fid].detach().cpu().numpy()                                     # (15,3,3)

#         # 指尖方向（相机系）
#         tip_dirs_cam = []
#         for j_local, kp_idx in enumerate(range(1, 16)):  # 1..15
#             if kp_idx not in tip_kp_indices:
#                 continue
#             d = Rg @ (Rl[j_local] @ axis_vec)  # 手系方向
#             d = (S @ d)                        # 左手镜像
#             if apply_rotx_180: d = (R180 @ d)
#             d = _norm(d[None, :])[0]
#             tip_dirs_cam.append(d)
#         tip_dirs_cam = np.array(tip_dirs_cam, dtype=np.float32) if tip_dirs_cam else np.zeros((0,3), np.float32)

#         # 关节到相机系（估掌心法向 / 与 silhouette 同规则：可选 rotx180，然后 + cam_t）
#         cam_t = cam_t_list[i].astype(np.float64).copy()
#         if apply_x_flip: cam_t[0] *= -1.0
#         J_cam = joints.copy()
#         if apply_rotx_180: J_cam = (R180 @ J_cam.T).T
#         J_cam = J_cam + cam_t

#         # 掌心法向（相机系）
#         n_palm = _estimate_palm_normal_cam(J_cam, tip_dirs_cam)

#         # 掩码1：掌心前半空间
#         dot_front = (dir_grid * n_palm.reshape(1,1,3)).sum(axis=-1)  # (H,W)
#         mask_front = (dot_front > cos_front).astype(np.uint8)

#         # 掩码2：指尖方向锥体（与任一指尖方向夹角 < phi）
#         if tip_dirs_cam.shape[0] > 0:
#             max_dot = np.full((H, W), -1.0, np.float32)
#             for d in tip_dirs_cam:
#                 max_dot = np.maximum(max_dot, (dir_grid * d.reshape(1,1,3)).sum(axis=-1).astype(np.float32))
#             mask_tips = (max_dot > cos_tip).astype(np.uint8)
#         else:
#             mask_tips = np.ones((H, W), np.uint8)

#         # 合并：只在“射线带”里保留“掌心前向+指尖锥体”
#         acc_mask |= (band & mask_front & mask_tips)

#     # 后处理：闭运算+按面积过滤
#     acc_mask = cv2.morphologyEx(acc_mask, cv2.MORPH_CLOSE,
#                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)))
#     num, lab, stats, _ = cv2.connectedComponentsWithStats(acc_mask, connectivity=8)
#     final = np.zeros_like(acc_mask)
#     for i in range(1, num):
#         if stats[i, cv2.CC_STAT_AREA] >= min_area:
#             final[lab == i] = 255

#     return final.astype(np.uint8)


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
    verts_preflipped=None,
    collect_segments=False,   # ★ 新增：是否返回 2D 线段
):
    import numpy as np, cv2

    is_right = (int(is_right_n) == 1)
    s = 1.0 if is_right else -1.0

    # 顶点（与 silhouette 一致：左手对 x 取反）
    if verts_preflipped is not None:
        verts = np.asarray(verts_preflipped, dtype=np.float64).copy()
    else:
        verts = out['pred_vertices'][fid].detach().cpu().numpy().astype(np.float64)
        verts[:, 0] *= s

    # 关节（单 mano）
    Jr = _dense_regressor(model.mano.J_regressor)
    joints = Jr @ verts
    if joints.shape[0] < 16:
        overlay = np.zeros((img_h, img_w, 4), np.uint8)
        return (overlay, 0, []) if collect_segments else (overlay, 0)

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

    # 左手镜像方向
    S = np.diag([s, 1.0, 1.0])

    # 生成射线（MANO 坐标）+ 记录关节索引
    origins, directions, kp_list = [], [], []
    for j_idx, kp_idx in enumerate(range(1, 16)):  # 1..15
        if kp_idx >= joints.shape[0]: 
            continue
        d = R_global @ (R_hand[j_idx] @ axis_vec)
        d = (S @ d)
        nrm = np.linalg.norm(d)
        if not np.isfinite(d).all() or nrm < 1e-8:
            continue
        origins.append(joints[kp_idx])
        directions.append(d / nrm)
        kp_list.append(kp_idx)

    if not origins:
        overlay = np.zeros((img_h, img_w, 4), np.uint8)
        return (overlay, 0, []) if collect_segments else (overlay, 0)

    O = np.asarray(origins, dtype=np.float64)
    D = np.asarray(directions, dtype=np.float64)
    kp_list = np.asarray(kp_list, dtype=np.int32)

    # rotx180 → +cam_t
    if apply_rotx_180:
        R180 = _rotx_deg(180.0)
        O = (R180 @ O.T).T
        D = (R180 @ D.T).T

    cam_t = np.asarray(cam_t, dtype=np.float64).copy()
    if apply_x_flip:
        cam_t[0] *= -1.0
    O_cam = O + cam_t

    # 自适应长度
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

    # 绘制 & 可选收集线段
    overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    draw_view = overlay[:, :, :3]
    draw = draw_view if draw_view.flags['C_CONTIGUOUS'] else np.ascontiguousarray(draw_view)

    c = (int(line_color[0]), int(line_color[1]), int(line_color[2]))
    cnt = 0
    segments = [] if collect_segments else None

    # 仅遍历有效的 indices
    valid_idx = np.where(mask)[0]
    for idx in valid_idx:
        x0, y0 = uv0[idx]; x1, y1 = uv1[idx]
        if not np.isfinite([x0, y0, x1, y1]).all():
            continue

        if collect_segments:
            segments.append({
                'uv0': (float(x0), float(y0)),
                'uv1': (float(x1), float(y1)),
                'kp_idx': int(kp_list[idx]),  # 真实的 1..15
                'hand_id': int(fid)
            })

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

    return (overlay, cnt, segments) if collect_segments else (overlay, cnt)


def compute_mano_joint_rays_mano_overlay_multi(
    out, model,
    img_w, img_h,
    cam_t_list,
    focal_length,
    is_right_list,
    axis='y-',
    line_color=(0, 0, 255), thickness=2, alpha_value=1.0,
    apply_x_flip=False, apply_rotx_180=False,
    invert_x=False, invert_y=False,
    ray_len=None,
    fids=None,
    verts_list_preflipped=None,
    collect_segments=False      # ★ 新增
):
    import numpy as np

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

    acc = np.zeros((img_h, img_w, 4), dtype=np.float32)
    total_cnt = 0
    all_segments = [] if collect_segments else None

    for i in range(N):
        verts_i = None if verts_list_preflipped is None else verts_list_preflipped[i]
        if collect_segments:
            ov_i, cnt_i, seg_i = compute_mano_joint_rays_mano_overlay_single(
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
                verts_preflipped=verts_i,
                collect_segments=True
            )
            all_segments.extend(seg_i)
        else:
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
                verts_preflipped=verts_i,
                collect_segments=False
            )
        acc += ov_i.astype(np.float32) / 255.0
        total_cnt += cnt_i

    overlay = (np.clip(acc, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return (overlay, total_cnt, all_segments) if collect_segments else (overlay, total_cnt)

def build_forward_region_from_segments(segments, img_shape,
                                       width_px=18,        # 单侧半宽
                                       extend_ratio=1.25,  # 在 uv1 方向的额外伸长倍数
                                       tips_only=True,     # 仅末节
                                       min_area=300,
                                       close_ks=11):
    import numpy as np, cv2
    H, W = img_shape[:2]
    canvas = np.zeros((H, W), np.uint8)
    tip_kp = {3, 6, 9, 12, 15}  # 末节索引（按你的 joints 映射）

    for s in segments:
        kp = s.get('kp_idx', 1)
        if tips_only and kp not in tip_kp:
            continue

        p0 = np.array(s['uv0'], dtype=np.float32)
        p1 = np.array(s['uv1'], dtype=np.float32)
        v  = p1 - p0
        L  = float(np.linalg.norm(v))
        if not np.isfinite(L) or L < 1.0:
            continue

        v /= L
        n = np.array([-v[1], v[0]], dtype=np.float32)

        front_len = L * float(extend_ratio)
        rear_len  = 0.0                     # ★ 不向后扩张（手背方向）
        half_w0, half_w1 = width_px*0.5, width_px  # 尾部窄、前端宽（也可等宽）

        a0 = p0 + v*rear_len - n*half_w0
        a1 = p0 + v*rear_len + n*half_w0
        b0 = p1 + v*front_len - n*half_w1
        b1 = p1 + v*front_len + n*half_w1

        poly = np.stack([a0, a1, b1, b0], axis=0).astype(np.int32)
        cv2.fillConvexPoly(canvas, poly, 255)

        # 仅前端圆帽
        cap_center = (p1 + v*front_len).astype(np.int32)
        cv2.circle(canvas, tuple(cap_center), int(round(half_w1)), 255, -1)

    # 闭运算 + 面积过滤
    if close_ks and close_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, k)

    num, lab, stats, _ = cv2.connectedComponentsWithStats(canvas, connectivity=8)
    out = np.zeros_like(canvas)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
            out[lab == i] = 255

    return out
