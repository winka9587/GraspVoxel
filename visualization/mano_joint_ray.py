# import numpy as np
# import torch
# import cv2

# def _get_device_from_out(out):
#     # 用 betas 的设备作为基准
#     return out['pred_mano_params']['betas'].device

# def _as_torch(x, device, dtype=torch.float32):
#     if isinstance(x, np.ndarray):
#         return torch.from_numpy(x).to(device=device, dtype=dtype)
#     elif torch.is_tensor(x):
#         return x.to(device=device, dtype=dtype)
#     else:
#         return torch.as_tensor(x, device=device, dtype=dtype)

# def _get_faces_from_mano(mano):
#     faces = getattr(mano, 'faces', None)
#     if faces is None:
#         faces = getattr(mano, 'th_faces', None)
#     if torch.is_tensor(faces):
#         faces = faces.detach().cpu().numpy()
#     else:
#         faces = np.asarray(faces)
#     return faces

# def _get_joints_from_vertices_with_regressor(vertices_np, mano, device):
#     """
#     vertices_np: (778,3) numpy, 相机坐标
#     return: (16,3) numpy, MANO 16 关节（腕+15个手指关节）
#     """
#     verts_t = torch.from_numpy(vertices_np).to(device=device, dtype=torch.float32)  # (778,3)
#     J_reg = mano.J_regressor
#     # 统一成 torch.float32 同一 device
#     if hasattr(J_reg, "to"):  # torch 参数/张量
#         J_reg = J_reg.to(device=device, dtype=torch.float32)
#     elif hasattr(J_reg, "toarray"):  # scipy 稀疏
#         J_reg = torch.from_numpy(J_reg.toarray()).to(device=device, dtype=torch.float32)
#     else:  # numpy
#         J_reg = torch.from_numpy(np.asarray(J_reg)).to(device=device, dtype=torch.float32)

#     if getattr(J_reg, "is_sparse", False):
#         joints_t = torch.sparse.mm(J_reg, verts_t)    # (16,3)
#     else:
#         joints_t = J_reg @ verts_t                    # (16,3)
#     return joints_t.detach().cpu().numpy()

# def compute_mano_joint_rays(out, model, axis='y-', use_model_regressor=True):
#     """
#     计算 15 条关节射线（MANO 索引 1..15）。
#     参数：
#       - out: 推理输出，需包含 pred_vertices, pred_mano_params.{global_orient, hand_pose}, focal_length（投影时用）
#       - model: 需有 model.mano（SMPLX 的 MANO 实例，含 J_regressor）
#       - axis: 用哪个局部坐标轴可视化方向：
#               'x', 'y', 'z', 'x-', 'y-', 'z-' 或者直接传入长度为3的向量
#       - use_model_regressor: True=用 MANO 的 J_regressor 从顶点回归 16 关节；False=直接用 out['pred_keypoints_3d']
#     返回：
#       dict: {
#         'origins': (M,3) numpy, 关节起点（相机坐标），按有效关节顺序（通常 M=15）
#         'directions': (M,3) numpy, 归一化方向
#         'indices': (M,) list, 对应的 MANO 关节索引（1..15）
#         'all_joints': (16,3) numpy, 全部 16 个关节坐标（腕+15）
#       }
#     """
#     device = _get_device_from_out(out)
#     mano = model.mano

#     # 3D 顶点（相机坐标系）
#     verts = out['pred_vertices'][0].detach().cpu().numpy()  # (778,3)

#     # 关节坐标
#     if use_model_regressor or ('pred_keypoints_3d' not in out):
#         joints = _get_joints_from_vertices_with_regressor(verts, mano, device)  # (16,3)
#     else:
#         joints = out['pred_keypoints_3d'][0].detach().cpu().numpy()

#     # 旋转矩阵
#     g = out['pred_mano_params']['global_orient']
#     R_global = (g[0,0] if g.ndim == 4 else g[0]).detach().cpu().numpy()   # (3,3)
#     R_hand   = out['pred_mano_params']['hand_pose'][0].detach().cpu().numpy()  # (15,3,3)

#     # 选择可视化的局部轴
#     if isinstance(axis, str):
#         axis = axis.lower()
#         axis_map = {
#             'x':  np.array([1,0,0], dtype=np.float32),
#             'y':  np.array([0,1,0], dtype=np.float32),
#             'z':  np.array([0,0,1], dtype=np.float32),
#             'x-': np.array([-1,0,0], dtype=np.float32),
#             'y-': np.array([0,-1,0], dtype=np.float32),
#             'z-': np.array([0,0,-1], dtype=np.float32),
#             'y-': np.array([0,-1,0], dtype=np.float32),  # 兼容
#         }
#         axis_vec = axis_map.get(axis, np.array([0,-1,0], dtype=np.float32))  # 默认 y-
#     else:
#         axis_vec = np.asarray(axis, dtype=np.float32)
#         axis_vec = axis_vec / (np.linalg.norm(axis_vec) + 1e-12)

#     idx_map = list(range(1, 16))  # MANO 手指关节 1..15
#     origins = []
#     directions = []
#     valid_indices = []

#     for j, kp_idx in enumerate(idx_map):
#         if kp_idx >= len(joints):
#             continue
#         joint_pos = joints[kp_idx]       # (3,)
#         R_local   = R_hand[j]            # (3,3)
#         dir_vec   = R_global @ (R_local @ axis_vec)
#         if not np.isfinite(dir_vec).all():
#             continue
#         n = np.linalg.norm(dir_vec)
#         if n < 1e-8:
#             continue
#         directions.append(dir_vec / n)
#         origins.append(joint_pos)
#         valid_indices.append(kp_idx)

#     if len(origins) == 0:
#         return {'origins': np.zeros((0,3), np.float32),
#                 'directions': np.zeros((0,3), np.float32),
#                 'indices': [],
#                 'all_joints': joints}

#     return {'origins': np.stack(origins, axis=0).astype(np.float32),
#             'directions': np.stack(directions, axis=0).astype(np.float32),
#             'indices': valid_indices,
#             'all_joints': joints.astype(np.float32)}

# def _project_points_cam_to_px(P3, f, cx, cy):
#     """
#     P3: (N,3) 相机坐标
#     return: (N,2) 像素坐标; mask: Z>0 且有限
#     """
#     P3 = np.asarray(P3, dtype=np.float32)
#     Z = P3[:, 2]
#     mask = (Z > 1e-6) & np.isfinite(Z)
#     uv = np.zeros((P3.shape[0], 2), dtype=np.float32)
#     uv[mask, 0] = f * (P3[mask, 0] / Z[mask]) + cx
#     uv[mask, 1] = f * (P3[mask, 1] / Z[mask]) + cy
#     return uv, mask

# def draw_rays_on_image(img, rays, out, length3d=None, color=(0,0,255), thickness=2, tip_length=0.25):
#     """
#     在图像上绘制射线（箭头）。
#     参数：
#       - img: np.ndarray, HxWx3 (BGR, uint8)。若是 RGBA 会自动转为 BGR
#       - rays: compute_mano_joint_rays 的返回 dict
#       - out: 用于读取 focal_length
#       - length3d: 每条射线在 3D 中的长度（单位与 3D 一致）。若 None，则按关节包围盒自适应
#       - color: (B,G,R)
#       - thickness: 线宽
#       - tip_length: 箭头尖长度比例（cv2.arrowedLine 参数）
#     返回：
#       img_out: 已绘制结果
#     """
#     img_out = img.copy()
#     if img_out.dtype != np.uint8:
#         img_out = np.clip(img_out, 0, 255).astype(np.uint8)
#     if img_out.ndim == 3 and img_out.shape[2] == 4:  # RGBA -> BGR
#         img_out = cv2.cvtColor(img_out, cv2.COLOR_BGRA2BGR)

#     H, W = img_out.shape[:2]

#     # 焦距（像素），主点默认在图像中心
#     f = out['focal_length']
#     if torch.is_tensor(f):
#         # 可能是标量或者形如 [1] 的张量
#         f = float(f.detach().cpu().reshape(-1)[0])
#     else:
#         f = float(np.asarray(f).reshape(-1)[0])
#     cx, cy = W * 0.5, H * 0.5

#     O = rays['origins']     # (M,3)
#     D = rays['directions']  # (M,3)

#     if O.shape[0] == 0:
#         return img_out

#     # 自动长度
#     if length3d is None:
#         bbox = O.max(axis=0) - O.min(axis=0)
#         bbox_size = float(np.max(bbox))
#         length3d = 0.15 * bbox_size

#     # 计算两点并投影
#     P0 = O
#     P1 = O + D * length3d

#     uv0, m0 = _project_points_cam_to_px(P0, f, cx, cy)
#     uv1, m1 = _project_points_cam_to_px(P1, f, cx, cy)
#     mask = m0 & m1

#     # 画箭头与起点圆
#     for i in range(O.shape[0]):
#         if not mask[i]:
#             continue
#         p0 = tuple(np.round(uv0[i]).astype(int))
#         p1 = tuple(np.round(uv1[i]).astype(int))
#         print(f"ray: ({p0}, {p1}")
#         # 起点小圆
#         cv2.circle(img_out, p0, radius=max(1, thickness+1), color=color, thickness=-1, lineType=cv2.LINE_AA)
#         # 箭头
#         cv2.arrowedLine(img_out, p0, p1, color=color, thickness=thickness, tipLength=tip_length, line_type=cv2.LINE_AA)

#     return img_out


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

def _project_pinhole(P_cam, fx, fy, cx, cy):
    P = np.asarray(P_cam, dtype=np.float64)
    Z = P[:, 2]
    uv = np.zeros((len(P), 2), dtype=np.float64)
    mask = (Z > 1e-6) & np.isfinite(Z)
    uv[mask, 0] = fx * (P[mask, 0] / Z[mask]) + cx
    uv[mask, 1] = fy * (P[mask, 1] / Z[mask]) + cy
    return uv, mask

def _get_mano_joints_from_verts(verts_np, mano, device):
    """ verts_np: (778,3) numpy → 返回 (16,3) numpy（腕+15个手指关节，MANO顺序） """
    verts_t = torch.from_numpy(verts_np).to(device=device, dtype=torch.float32)  # (778,3)
    J_reg = mano.J_regressor
    if hasattr(J_reg, "to"):
        J_reg = J_reg.to(device=device, dtype=torch.float32)
    elif hasattr(J_reg, "toarray"):
        J_reg = torch.from_numpy(J_reg.toarray()).to(device=device, dtype=torch.float32)
    else:
        J_reg = torch.from_numpy(np.asarray(J_reg)).to(device=device, dtype=torch.float32)
    joints_t = (J_reg @ verts_t) if not getattr(J_reg, "is_sparse", False) else torch.sparse.mm(J_reg, verts_t)
    return joints_t.detach().cpu().numpy()  # (16,3)

# ---------- 函数 1：从 out 计算射线（起点与方向） ----------
def compute_mano_joint_rays(out, model, axis='y-'):
    """
    返回 15 条射线（MANO 关节 1..15）：
      origins: (M,3) 3D 起点（与 pred_vertices 同坐标系）
      directions: (M,3) 单位方向
      indices: 对应的关节索引（1..15）
      all_joints: (16,3) 全部关节坐标（腕+15）
    """
    device = out['pred_mano_params']['betas'].device
    mano = model.mano

    # 顶点（和你现有流程一致：直接用 out 的 verts）
    verts = out['pred_vertices'][0].detach().cpu().numpy()  # (778,3)

    # 关节：用 MANO 回归器，顺序与 hand_pose 一一对应
    joints = _get_mano_joints_from_verts(verts, mano, device)  # (16,3)

    # 旋转
    g = out['pred_mano_params']['global_orient']
    R_global = (g[0, 0] if g.ndim == 4 else g[0]).detach().cpu().numpy()  # (3,3)
    R_hand = out['pred_mano_params']['hand_pose'][0].detach().cpu().numpy()  # (15,3,3)

    # 局部轴选择（默认 -Y）
    if isinstance(axis, str):
        a = axis.lower()
        axis_map = {'x': [1,0,0], 'y': [0,1,0], 'z': [0,0,1],
                    'x-': [-1,0,0], 'y-': [0,-1,0], 'z-': [0,0,-1]}
        axis_vec = np.array(axis_map.get(a, [0,-1,0]), dtype=np.float64)
    else:
        axis_vec = np.asarray(axis, dtype=np.float64)
        n = np.linalg.norm(axis_vec); axis_vec = axis_vec/(n+1e-12)

    idx_map = list(range(1, 16))  # 1..15
    origins, directions, valid_idx = [], [], []

    for j, kp_idx in enumerate(idx_map):
        if kp_idx >= len(joints):
            continue
        R_local = R_hand[j]
        dir_vec = R_global @ (R_local @ axis_vec)
        if not np.isfinite(dir_vec).all():
            continue
        n = np.linalg.norm(dir_vec)
        if n < 1e-8:
            continue
        origins.append(joints[kp_idx])
        directions.append(dir_vec / n)
        valid_idx.append(kp_idx)

    if len(origins) == 0:
        return {'origins': np.zeros((0,3), np.float32),
                'directions': np.zeros((0,3), np.float32),
                'indices': [],
                'all_joints': joints.astype(np.float32)}

    return {'origins': np.asarray(origins, dtype=np.float32),
            'directions': np.asarray(directions, dtype=np.float32),
            'indices': valid_idx,
            'all_joints': joints.astype(np.float32)}

# ---------- 函数 2：按“silhouette”风格把射线画到图像坐标 ----------
# def draw_joint_rays_overlay(
#     rays, cam_t, img_w, img_h, focal_length,
#     line_color=(0, 0, 255), thickness=2, alpha_value=1.0,
#     apply_x_flip=False,      # 与 hand_silhouette_overlay 一致：只对 cam_t[0] 取负
#     apply_rotx_180=False,    # 若 True，对起点和方向都绕 X 轴旋转 180°
#     invert_x=False,          # 像平面绕中心镜像 X
#     invert_y=False,          # 像平面绕中心镜像 Y
#     ray_len=None             # 3D 中的长度；None 则自适应
# ):
#     """
#     返回 RGBA overlay（HxWx4, uint8）和绘制条数
#     """
#     O = np.asarray(rays['origins'], dtype=np.float64)   # (M,3)
#     D = np.asarray(rays['directions'], dtype=np.float64) # (M,3)

#     if O.size == 0:
#         overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
#         return overlay, 0

#     # 可选：绕 X 轴 180°
#     if apply_rotx_180:
#         R = _rotx_deg(180.0)
#         O = (R @ O.T).T
#         D = (R @ D.T).T

#     # 相机平移（与 hand_silhouette_overlay 一致）
#     cam_t = np.asarray(cam_t, dtype=np.float64).copy()
#     if apply_x_flip:
#         cam_t[0] *= -1.0

#     # 自适应射线长度
#     if ray_len is None:
#         bbox = O.max(axis=0) - O.min(axis=0)
#         ray_len = 0.15 * float(np.max(bbox) if np.isfinite(bbox).all() else 1.0)

#     P0_cam = O + cam_t
#     P1_cam = O + D * ray_len + cam_t

#     # 投影（与 hand_silhouette_overlay 同式）
#     fx = fy = float(focal_length)
#     cx, cy = img_w / 2.0, img_h / 2.0
#     uv0, m0 = _project_pinhole(P0_cam, fx, fy, cx, cy)
#     uv1, m1 = _project_pinhole(P1_cam, fx, fy, cx, cy)
#     mask = m0 & m1

#     # 像平面镜像
#     if invert_x:
#         uv0[:, 0] = 2 * cx - uv0[:, 0]
#         uv1[:, 0] = 2 * cx - uv1[:, 0]
#     if invert_y:
#         uv0[:, 1] = 2 * cy - uv0[:, 1]
#         uv1[:, 1] = 2 * cy - uv1[:, 1]

#     # 叠加图（与 silhouette 的写法一致）
#     overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
#     draw_view = overlay[:, :, :3]
#     needs_copyback = False
#     if not draw_view.flags['C_CONTIGUOUS']:
#         draw = np.ascontiguousarray(draw_view); needs_copyback = True
#     else:
#         draw = draw_view

#     c = (int(line_color[0]), int(line_color[1]), int(line_color[2]))
#     cnt = 0
#     for i in range(len(O)):
#         if not mask[i]:
#             continue
#         x0, y0 = uv0[i]; x1, y1 = uv1[i]
#         coords = np.array([x0, y0, x1, y1], dtype=np.float64)
#         if not np.isfinite(coords).all():
#             continue
#         p0 = (int(np.round(x0)), int(np.round(y0)))
#         p1 = (int(np.round(x1)), int(np.round(y1)))
#         # 起点小圆
#         cv2.circle(draw, p0, radius=max(1, thickness+1), color=c, thickness=-1, lineType=cv2.LINE_AA)
#         # 箭头
#         cv2.arrowedLine(draw, p0, p1, c, thickness=thickness, tipLength=0.25, line_type=cv2.LINE_AA)
#         cnt += 1

#     if needs_copyback:
#         overlay[:, :, :3] = draw
#     coverage = overlay[:, :, :3].max(axis=2, keepdims=True).astype(np.float32) / 255.0
#     overlay[:, :, :3] = np.where(coverage > 0, 255, overlay[:, :, :3])
#     overlay[:, :, 3:4] = (coverage * (alpha_value * 255)).astype(np.uint8)

#     return overlay, cnt

def draw_joint_rays_overlay(
    rays, cam_t, img_w, img_h, focal_length=None,
    line_color=(0, 0, 255), thickness=2, alpha_value=1.0,
    apply_x_flip=False,      # 与 hand_silhouette_overlay 一致：只对 cam_t[0] 取负
    apply_rotx_180=False,    # 若 True，对起点和方向都绕 X 轴旋转 180°
    invert_x=False,          # 像平面绕中心镜像 X
    invert_y=False,          # 像平面绕中心镜像 Y
    ray_len=None             # 3D 中的长度；None 则自适应
):
    """
    返回 RGBA overlay（HxWx4, uint8）和绘制条数
    说明：本实现不依赖相机内参，使用单位焦距投影 + 图像尺寸决定像素缩放。
    """
    O = np.asarray(rays.get('origins', []), dtype=np.float64)      # (M,3)
    D = np.asarray(rays.get('directions', []), dtype=np.float64)   # (M,3)

    if O.size == 0 or D.size == 0:
        overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        return overlay, 0

    # 可选：绕 X 轴 180°
    if apply_rotx_180:
        R = _rotx_deg(180.0)
        O = (R @ O.T).T
        D = (R @ D.T).T

    # 相机平移（与 hand_silhouette_overlay 一致）
    cam_t = np.asarray(cam_t, dtype=np.float64).copy()
    if apply_x_flip:
        cam_t[0] *= -1.0

    # 自适应射线长度
    if ray_len is None:
        bbox = O.max(axis=0) - O.min(axis=0)
        ray_len = 0.15 * float(np.max(bbox) if np.isfinite(bbox).all() else 1.0)

    # 3D 中两点：起点与终点（相机坐标系）
    P0_cam = O + cam_t
    P1_cam = O + D * ray_len + cam_t

    # -------- 无内参投影：单位焦距 + 图像尺寸缩放 ----------
    # 将 (x, y, z) 投到 z=1 的归一化平面： (u, v) = (x/z, y/z)
    # 然后用 s = 0.5 * min(W, H) 将归一化坐标映射到像素，并以 (cx, cy) 为中心
    cx, cy = img_w / 2.0, img_h / 2.0
    s = 0.5 * float(min(img_w, img_h))  # 全局尺度，不依赖真实内参

    def _project_unit_focal(P):
        # P: (N,3)
        z = P[:, 2]
        valid = z > 1e-6
        # 避免除零；对非法z做占位，后面用mask过滤
        z_safe = np.where(valid, z, 1.0)
        print("P[:, 0]:", P[:, 0])
        print("P[:, 1]:", P[:, 1])
        print("z_safe:", z_safe)
        print("s:", s)
        u = (P[:, 0] / z_safe) * s + cx
        v = (P[:, 1] / z_safe) * s + cy
        uv = np.stack([u, v], axis=-1)
        return uv, valid

    uv0, m0 = _project_unit_focal(P0_cam)
    uv1, m1 = _project_unit_focal(P1_cam)
    mask = m0 & m1

    # 像平面镜像（绕中心）
    if invert_x:
        uv0[:, 0] = 2 * cx - uv0[:, 0]
        uv1[:, 0] = 2 * cx - uv1[:, 0]
    if invert_y:
        uv0[:, 1] = 2 * cy - uv0[:, 1]
        uv1[:, 1] = 2 * cy - uv1[:, 1]

    # 叠加图（与 silhouette 的写法一致）
    overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    draw_view = overlay[:, :, :3]
    needs_copyback = False
    if not draw_view.flags['C_CONTIGUOUS']:
        draw = np.ascontiguousarray(draw_view); needs_copyback = True
    else:
        draw = draw_view

    c = (int(line_color[0]), int(line_color[1]), int(line_color[2]))
    cnt = 0
    for i in range(len(O)):
        if not mask[i]:
            continue
        x0, y0 = uv0[i]; x1, y1 = uv1[i]
        coords = np.array([x0, y0, x1, y1], dtype=np.float64)
        if not np.isfinite(coords).all():
            continue
        p0 = (int(np.round(x0)), int(np.round(y0)))
        p1 = (int(np.round(x1)), int(np.round(y1)))
        print("(p0, p1):", p0, p1)
        # 起点小圆
        cv2.circle(draw, p0, radius=max(1, thickness+1), color=c, thickness=-1, lineType=cv2.LINE_AA)
        # 箭头（注意：cv2 的关键字是 line_type）
        cv2.arrowedLine(draw, p0, p1, c, thickness=thickness, tipLength=0.25, line_type=cv2.LINE_AA)
        cnt += 1

    if needs_copyback:
        overlay[:, :, :3] = draw

    # 将有绘制处设为白色，alpha 由覆盖程度和 alpha_value 决定
    coverage = overlay[:, :, :3].max(axis=2, keepdims=True).astype(np.float32) / 255.0
    overlay[:, :, :3] = np.where(coverage > 0, 255, overlay[:, :, :3])
    overlay[:, :, 3:4] = (coverage * (alpha_value * 255)).astype(np.uint8)

    return overlay, cnt

import numpy as np
import cv2

# def compute_mano_joint_rays_mano_overlay(
#     out, model,
#     img_w, img_h,
#     focal_length=None,           # 建议传 out['scaled_focal'][0]
#     axis='y-',
#     line_color=(0, 0, 255), thickness=2, alpha_value=1.0,
#     apply_x_flip=False,          # 与 silhouette 一致：只对 cam_t[0] 取负
#     apply_rotx_180=False,        # 若 True，对起点与方向都绕 X 轴旋转 180°
#     invert_x=False, invert_y=False,
#     cam_t=None,                  # 若 None 则从 out['pred_cam_t'][0] 取
#     ray_len=None                 # None 自适应（基于 V_cam 的包围盒）
# ):
#     """
#     在 MANO 坐标系中计算 15 条关节射线，并按 hand_silhouette_overlay 的方式投影+绘制到图像平面。
#     返回: overlay(HxWx4, uint8), num_drawn
#     依赖: _rotx_deg, _get_mano_joints_from_verts（与你现有代码一致）
#     """
#     device = out['pred_mano_params']['betas'].device
#     mano = model.mano

#     # ---------- 取顶点与关节（均在 MANO/mesh 坐标系） ----------
#     verts = out['pred_vertices'][0].detach().cpu().numpy()  # (778,3)
#     joints = _get_mano_joints_from_verts(verts, mano, device)  # (16,3)

#     # ---------- 手的旋转（用于确定每指关节的局部朝向） ----------
#     g = out['pred_mano_params']['global_orient']
#     R_global = (g[0, 0] if g.ndim == 4 else g[0]).detach().cpu().numpy()  # (3,3)
#     R_hand = out['pred_mano_params']['hand_pose'][0].detach().cpu().numpy()  # (15,3,3)

#     # ---------- 方向轴 ----------
#     if isinstance(axis, str):
#         amap = {'x':[1,0,0], 'y':[0,1,0], 'z':[0,0,1],
#                 'x-':[-1,0,0], 'y-':[0,-1,0], 'z-':[0,0,-1]}
#         axis_vec = np.array(amap.get(axis.lower(), [0,-1,0]), dtype=np.float64)
#     else:
#         axis_vec = np.asarray(axis, dtype=np.float64)
#         axis_vec /= (np.linalg.norm(axis_vec) + 1e-12)

#     # ---------- 逐关节生成射线（MANO 坐标系） ----------
#     idx_map = list(range(1, 16))  # 1..15
#     origins = []
#     directions = []
#     for j, kp_idx in enumerate(idx_map):
#         if kp_idx >= len(joints): continue
#         R_local = R_hand[j]  # (3,3)
#         d = R_global @ (R_local @ axis_vec)  # MANO 系方向
#         if not np.isfinite(d).all(): continue
#         n = np.linalg.norm(d)
#         if n < 1e-8: continue
#         origins.append(joints[kp_idx])
#         directions.append(d / n)

#     # 没有有效射线则返回空 overlay
#     if len(origins) == 0:
#         overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
#         return overlay, 0

#     O = np.asarray(origins, dtype=np.float64)   # (M,3) MANO
#     D = np.asarray(directions, dtype=np.float64)

#     # ---------- 与 silhouette 保持一致的坐标处理 ----------
#     # 可选：绕 X 轴 180°
#     if apply_rotx_180:
#         R = _rotx_deg(180.0)
#         O = (R @ O.T).T
#         D = (R @ D.T).T

#     # 加法平移到相机坐标（与 silhouette 完全一致）
#     if cam_t is None:
#         cam_t = out['pred_cam_t'][0].detach().cpu().numpy()
#     cam_t = np.asarray(cam_t, dtype=np.float64).copy()
#     if apply_x_flip:
#         cam_t[0] *= -1.0

#     O_cam = O + cam_t

#     # 自适应射线长度（基于 V_cam 包围盒，保持与 silhouette 风格一致）
#     V = np.asarray(verts, dtype=np.float64).copy()
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

#     # ---------- 针孔投影（使用 scaled_focal/内参） ----------
#     if focal_length is None:
#         # 尝试从 out 获取 scaled_focal
#         if 'scaled_focal' in out:
#             focal_length = float(out['scaled_focal'][0].detach().cpu().numpy())
#         else:
#             # 兜底：用 0.5*min(W,H)
#             focal_length = 0.5 * float(min(img_w, img_h))

#     fx = fy = float(focal_length)
#     cx, cy = img_w / 2.0, img_h / 2.0

#     def _project_pinhole_points(P):
#         z = P[:, 2]
#         valid = z > 1e-6
#         z_safe = np.where(valid, z, 1.0)
#         u = (P[:, 0] / z_safe) * fx + cx
#         v = (P[:, 1] / z_safe) * fy + cy
#         return np.stack([u, v], axis=-1), valid

#     uv0, m0 = _project_pinhole_points(P0)
#     uv1, m1 = _project_pinhole_points(P1)
#     mask = m0 & m1

#     # 像平面镜像（绕中心）
#     if invert_x:
#         uv0[:, 0] = 2 * cx - uv0[:, 0]
#         uv1[:, 0] = 2 * cx - uv1[:, 0]
#     if invert_y:
#         uv0[:, 1] = 2 * cy - uv0[:, 1]
#         uv1[:, 1] = 2 * cy - uv1[:, 1]

#     # ---------- 绘制到 overlay（与 silhouette 的 alpha 逻辑一致） ----------
#     overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
#     draw_view = overlay[:, :, :3]
#     if not draw_view.flags['C_CONTIGUOUS']:
#         draw = np.ascontiguousarray(draw_view); needs_copyback = True
#     else:
#         draw = draw_view; needs_copyback = False

#     c = (int(line_color[0]), int(line_color[1]), int(line_color[2]))
#     cnt = 0
#     for i in range(len(P0)):
#         if not mask[i]:
#             continue
#         x0, y0 = uv0[i]; x1, y1 = uv1[i]
#         if not np.isfinite([x0, y0, x1, y1]).all():
#             continue
#         p0 = (int(np.round(x0)), int(np.round(y0)))
#         p1 = (int(np.round(x1)), int(np.round(y1)))
#         cv2.circle(draw, p0, radius=max(1, thickness+1), color=c, thickness=-1, lineType=cv2.LINE_AA)
#         cv2.arrowedLine(draw, p0, p1, c, thickness=thickness, tipLength=0.25, line_type=cv2.LINE_AA)
#         cnt += 1

#     if needs_copyback:
#         overlay[:, :, :3] = draw

#     coverage = overlay[:, :, :3].max(axis=2, keepdims=True).astype(np.float32) / 255.0
#     overlay[:, :, :3] = np.where(coverage > 0, 255, overlay[:, :, :3])
#     overlay[:, :, 3:4] = (coverage * (alpha_value * 255)).astype(np.uint8)

#     return overlay, cnt

# def _get_joints_from_verts_with_regressor(verts_np, J_regressor):
#     # verts_np: (778,3) np.float64
#     Jr = J_regressor
#     try:
#         Jr = Jr.coalesce().to_dense().cpu().numpy()  # torch.sparse -> dense
#     except Exception:
#         Jr = np.asarray(Jr.cpu().numpy())
#     return Jr @ verts_np  # (nJoints, 3)

# import numpy as np
# import cv2
# def compute_mano_joint_rays_mano_overlay(
#     out, model,
#     img_w, img_h,
#     cam_t,                       # ★ 必须：和 hand_silhouette_overlay 用同一份 cam_t（整图系）
#     focal_length,                # ★ 必须：和 hand_silhouette_overlay 用同一份 focal（整图系/scaled_focal）
#     axis='y-',
#     line_color=(0, 0, 255), thickness=2, alpha_value=1.0,
#     apply_x_flip=False,
#     apply_rotx_180=False,
#     invert_x=False, invert_y=False,
#     ray_len=None,
#     is_right=True
# ):
#     """在 MANO 坐标系计算 15 条射线，并用与 hand_silhouette_overlay 相同的参数/流程绘制。"""
#     device = out['pred_mano_params']['betas'].device
#     mano = model.mano

#     # MANO 坐标：顶点、关节
#     verts  = out['pred_vertices'][0].detach().cpu().numpy()     # (778,3)
#     joints = _get_mano_joints_from_verts(verts, mano, device)   # (16,3)

#     # 姿态旋转
#     g = out['pred_mano_params']['global_orient']
#     R_global = (g[0, 0] if g.ndim == 4 else g[0]).detach().cpu().numpy()   # (3,3)
#     R_hand = out['pred_mano_params']['hand_pose'][0].detach().cpu().numpy()# (15,3,3)

#     # 方向轴
#     if isinstance(axis, str):
#         amap = {'x':[1,0,0],'y':[0,1,0],'z':[0,0,1],'x-':[-1,0,0],'y-':[0,-1,0],'z-':[0,0,-1]}
#         axis_vec = np.array(amap.get(axis.lower(), [0,-1,0]), dtype=np.float64)
#     else:
#         axis_vec = np.asarray(axis, dtype=np.float64)
#         axis_vec /= (np.linalg.norm(axis_vec)+1e-12)

#     # 1..15 关节的射线（MANO 坐标系）
#     origins, directions = [], []
#     for j_idx, kp_idx in enumerate(range(1, 16)):
#         if kp_idx >= len(joints): continue
#         d = R_global @ (R_hand[j_idx] @ axis_vec)   # MANO 系方向
#         n = np.linalg.norm(d)
#         if not np.isfinite(d).all() or n < 1e-8: continue
#         origins.append(joints[kp_idx])
#         directions.append(d / n)

#     if not origins:
#         return np.zeros((img_h, img_w, 4), np.uint8), 0

#     O = np.asarray(origins, dtype=np.float64)
#     D = np.asarray(directions, dtype=np.float64)

#     # 与 silhouette 同顺序：可选 rotx180 → +cam_t 到相机系
#     if apply_rotx_180:
#         R = _rotx_deg(180.0)
#         O = (R @ O.T).T
#         D = (R @ D.T).T

#     cam_t = np.asarray(cam_t, dtype=np.float64).copy()
#     if apply_x_flip:
#         cam_t[0] *= -1.0
#     O_cam = O + cam_t

#     # 自适应长度基于 V_cam 包围盒（与 silhouette 同口径）
#     V = out['pred_vertices'][0].detach().cpu().numpy()
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

#     # 针孔投影（与 silhouette 同：fx=fy=focal_length，cx,cy=图像中心）
#     fx = fy = float(focal_length)
#     cx, cy = img_w / 2.0, img_h / 2.0

#     def _proj(P):
#         z = P[:, 2]
#         valid = z > 1e-6
#         z_safe = np.where(valid, z, 1.0)
#         u = (P[:, 0] / z_safe) * fx + cx
#         v = (P[:, 1] / z_safe) * fy + cy
#         return np.stack([u, v], axis=-1), valid

#     uv0, m0 = _proj(P0)
#     uv1, m1 = _proj(P1)
#     mask = m0 & m1

#     if invert_x:
#         uv0[:, 0] = 2 * cx - uv0[:, 0]
#         uv1[:, 0] = 2 * cx - uv1[:, 0]
#     if invert_y:
#         uv0[:, 1] = 2 * cy - uv0[:, 1]
#         uv1[:, 1] = 2 * cy - uv1[:, 1]

#     # 绘制 overlay（与 silhouette 的 alpha 逻辑一致）
#     overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
#     draw_view = overlay[:, :, :3]
#     if not draw_view.flags['C_CONTIGUOUS']:
#         draw = np.ascontiguousarray(draw_view); needs_copyback = True
#     else:
#         draw = draw_view; needs_copyback = False

#     c = (int(line_color[0]), int(line_color[1]), int(line_color[2]))
#     cnt = 0
#     for i in range(len(P0)):
#         if not mask[i]: continue
#         x0, y0 = uv0[i]; x1, y1 = uv1[i]
#         if not np.isfinite([x0, y0, x1, y1]).all(): continue
#         p0 = (int(np.round(x0)), int(np.round(y0)))
#         p1 = (int(np.round(x1)), int(np.round(y1)))
#         cv2.circle(draw, p0, radius=max(1, thickness+1), color=c, thickness=-1, lineType=cv2.LINE_AA)
#         cv2.arrowedLine(draw, p0, p1, c, thickness=thickness, tipLength=0.25, line_type=cv2.LINE_AA)
#         cnt += 1

#     if needs_copyback:
#         overlay[:, :, :3] = draw
#     cov = overlay[:, :, :3].max(axis=2, keepdims=True).astype(np.float32) / 255.0
#     overlay[:, :, :3] = np.where(cov > 0, 255, overlay[:, :, :3])
#     overlay[:, :, 3:4] = (cov * (alpha_value * 255)).astype(np.uint8)
#     return overlay, cnt


import numpy as np
import cv2

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
