import cv2
import numpy as np

def compute_interaction_region_from_overlay(overlay_ray,
                                            expand_px=24,   # 区域向四周扩张的像素宽度
                                            blur_sigma=11,  # 生成软热力的高斯核
                                            min_area=300    # 过滤太小的噪声区域
                                           ):
    """
    输入:
      overlay_ray: (H,W,4) uint8，上一段你算好的 RGBA 射线叠加图
    输出:
      region_mask: (H,W) uint8, {0,255}，可能交互/抓取的区域
      heatmap:     (H,W) float32, 0..1，软热度（可用于可视化或阈值化）
      contour:     该区域的外接最大轮廓（若存在），用于画多边形/包围盒
    """
    H, W = overlay_ray.shape[:2]
    alpha = overlay_ray[:, :, 3]

    # 1) 基于 alpha 得到“射线像素”二值图
    #    用 Otsu 或固定阈值都可；这里用固定小阈值更稳（你的 alpha==255 在线条上）
    ray_bin = (alpha > 8).astype(np.uint8)

    # 2) 膨胀：把细线变成“带宽”，近似手在 2D 上可能接触的带状区域
    k = int(max(5, round(expand_px)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
    band = cv2.dilate(ray_bin, kernel, iterations=1)

    # 3) 软化：做一个平滑热图，便于后续阈值/可视化
    heat = cv2.GaussianBlur(band.astype(np.float32), (0, 0), blur_sigma)
    if heat.max() > 1e-6:
        heat = heat / heat.max()  # 0..1

    # 4) 自适应阈值：把热图转换为最终区域（你也可直接用 band）
    #    这里选一个偏“宽松”的阈值，让区域更连贯
    thr = 0.2
    region = (heat >= thr).astype(np.uint8)

    # 5) 去噪 + 闭运算连接碎片
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    # 筛掉很小的连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    region_clean = np.zeros_like(region)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            region_clean[labels == i] = 1

    # 6) 取最大轮廓（可选）
    contours, _ = cv2.findContours(region_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_cnt = max(contours, key=cv2.contourArea) if contours else None

    region_mask = (region_clean * 255).astype(np.uint8)
    return region_mask, heat.astype(np.float32), main_cnt


import numpy as np, cv2

def _dense_regressor(J_regressor):
    try:
        Jr = J_regressor.coalesce().to_dense().cpu().numpy()
    except Exception:
        Jr = np.asarray(J_regressor.cpu().numpy())
    return Jr

def _norm(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def _rotx_deg(angle):
    a = np.deg2rad(angle); ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], np.float64)

def _make_dir_grid(W, H, fx, fy, cx, cy):
    xs = (np.arange(W, dtype=np.float32) - cx) / fx
    ys = (np.arange(H, dtype=np.float32) - cy) / fy
    X, Y = np.meshgrid(xs, ys)           # (H,W)
    D = np.stack([X, Y, np.ones_like(X)], axis=-1)  # (H,W,3)
    return _norm(D.astype(np.float32))   # 相机坐标下每个像素的视线方向

def _estimate_palm_normal_cam(joints_cam, tip_dirs_cam):
    """
    joints_cam: (>=16,3) wrist+15 in camera coords
    tip_dirs_cam: (K,3)  选取的指尖方向(相机系)
    """
    wrist = joints_cam[0]
    # 粗略用食指/小指 MCP 构平面；若索引不同可按你的关节映射调整
    idx_mcp, little_mcp = 4, 13
    v1 = joints_cam[idx_mcp]  - wrist
    v2 = joints_cam[little_mcp]- wrist
    n  = np.cross(v1, v2)
    n  = _norm(n)
    if tip_dirs_cam.size > 0:
        mean_tip = _norm(np.mean(tip_dirs_cam, axis=0, keepdims=True))[0]
        if np.dot(n, mean_tip) < 0:  # 方向与指尖整体现向不一致则取反
            n = -n
    return n

def refine_interaction_region_from_overlay_with_direction(
    overlay_ray,                      # (H,W,4) uint8，方案A得到的RGAA
    out, model,
    img_w, img_h,
    cam_t_list,                       # list[(3,)] 或 (N,3)
    focal_length,                     # 标量或 (N,)
    is_right_list,                    # 0/1，左/右手
    fids=None,                        # 每只手对应 out 的 batch 索引；默认 0..N-1
    verts_list_preflipped=None,       # 若你已和 silhouette 一样对左手做了 x 取反，传进来可避免重复
    axis='y-',                        # 生成指向时用的局部轴（与你画射线一致）
    apply_rotx_180=False, apply_x_flip=False,
    theta_front_deg=70,               # 掌心前半空间开角（越小越“紧”）
    phi_finger_deg=35,                # 指尖方向的锥角（越小越“紧”）
    expand_px=16,                     # 在 alpha 带基础上再做一次小膨胀
    min_area=200                      # 面积过滤
):
    """
    返回:
      region_mask: (H,W) uint8 {0,255}  —— 方向约束后的交互区域
      debug: dict  —— 可选诊断图
    """
    H, W = img_h, img_w
    # 0) 取出 alpha 带并适度扩张
    alpha = overlay_ray[:, :, 3]
    ray_bin = (alpha > 8).astype(np.uint8)
    if expand_px and expand_px > 0:
        k = int(max(3, round(expand_px)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
        band = cv2.dilate(ray_bin, kernel, iterations=1)
    else:
        band = ray_bin

    # 1) 相机内参 → 像素视线方向场
    if isinstance(focal_length, (list, tuple, np.ndarray)):
        fx = fy = float(np.asarray(focal_length).reshape(-1)[0])
    else:
        fx = fy = float(focal_length)
    cx, cy = W/2.0, H/2.0
    dir_grid = _make_dir_grid(W, H, fx, fy, cx, cy)   # (H,W,3), float32

    # 2) 累计所有“手的方向约束”掩码
    cam_t_list = np.asarray(cam_t_list, dtype=np.float64)
    if cam_t_list.ndim == 1: cam_t_list = cam_t_list.reshape(1,3)
    N = cam_t_list.shape[0]
    is_right_list = np.asarray(is_right_list).astype(int).reshape(-1)
    if is_right_list.size != N:
        is_right_list = (list(is_right_list)*N)[:N]
        is_right_list = np.asarray(is_right_list)

    if fids is None:
        fids = np.arange(N, dtype=int)
    else:
        fids = np.asarray(fids, dtype=int)

    # 方向阈值预计算
    cos_front = float(np.cos(np.deg2rad(theta_front_deg)))
    cos_tip   = float(np.cos(np.deg2rad(phi_finger_deg)))

    # 用于调试的可视化积累
    debug = {}

    acc_mask = np.zeros((H, W), np.uint8)

    # 稀疏回归器
    Jr = _dense_regressor(model.mano.J_regressor)

    # 指尖（末节）关节索引的一个常用集合（根据你的关节映射可调整）
    tip_kp_indices = [3, 6, 9, 12, 15]

    # 3) 逐手计算掌心法向/指尖方向（相机系），并生成掩码
    R180 = _rotx_deg(180.0) if apply_rotx_180 else None

    for i in range(N):
        fid = int(fids[i])
        is_right = bool(is_right_list[i])
        s = 1.0 if is_right else -1.0
        S = np.diag([s, 1.0, 1.0])  # 左手的 x 镜像

        # 顶点（MANO坐标，左手需 x 取反）
        if verts_list_preflipped is not None:
            verts = np.asarray(verts_list_preflipped[i], dtype=np.float64).copy()
        else:
            verts = out['pred_vertices'][fid].detach().cpu().numpy().astype(np.float64)
            verts[:, 0] *= s

        # 关节（MANO坐标）
        joints = Jr @ verts  # (nJ,3)

        # 姿态旋转
        g  = out['pred_mano_params']['global_orient']
        Rh = out['pred_mano_params']['hand_pose']
        Rg = (g[fid,0] if g.ndim == 4 else g[fid]).detach().cpu().numpy()      # (3,3)
        Rl = Rh[fid].detach().cpu().numpy()                                     # (15,3,3)

        # 指尖方向（相机系）
        tip_dirs_cam = []
        for j_local, kp_idx in enumerate(range(1, 16)):  # 1..15
            if kp_idx not in tip_kp_indices:  # 只要末节
                continue
            d = Rg @ (Rl[j_local] @ np.array({'x':[1,0,0],'y':[0,1,0],'z':[0,0,1],
                                             'x-':[-1,0,0],'y-':[0,-1,0],'z-':[0,0,-1]}[axis.lower()]
                                        if isinstance(axis, str) else np.asarray(axis, np.float64)))
            d = (S @ d)                                # 左手镜像
            if apply_rotx_180: d = (R180 @ d)
            d = _norm(d[None,:])[0]
            tip_dirs_cam.append(d)
        tip_dirs_cam = np.array(tip_dirs_cam, dtype=np.float64) if tip_dirs_cam else np.zeros((0,3), np.float64)

        # 关节转相机系（仅用于估计掌心法向；平移/翻转与 silhouette 保持一致）
        cam_t = cam_t_list[i].astype(np.float64).copy()
        if apply_x_flip: cam_t = cam_t.copy(); cam_t[0] *= -1.0
        J_cam = joints.copy()
        if apply_rotx_180: J_cam = (R180 @ J_cam.T).T
        J_cam = J_cam + cam_t

        # 掌心法向（相机系）
        n_palm = _estimate_palm_normal_cam(J_cam, tip_dirs_cam)

        # 生成“掌心前半空间”掩码：dot(dir_grid, n_palm) > cos_front
        dot_front = (dir_grid * n_palm.reshape(1,1,3)).sum(axis=-1)  # (H,W)
        mask_front = (dot_front > cos_front).astype(np.uint8)

        # 生成“指尖方向锥体”掩码：max_i dot(dir_grid, tip_d_i) > cos_tip
        if tip_dirs_cam.shape[0] > 0:
            max_dot = np.full((H, W), -1.0, np.float32)
            for d in tip_dirs_cam:
                max_dot = np.maximum(max_dot, (dir_grid * d.reshape(1,1,3)).sum(axis=-1).astype(np.float32))
            mask_tips = (max_dot > cos_tip).astype(np.uint8)
        else:
            mask_tips = np.ones((H, W), np.uint8)  # 没有指尖方向就只用掌心半空间

        # 与 alpha 带状区域相交
        hand_mask = band & mask_front & mask_tips

        acc_mask |= hand_mask

    # 4) 清理与连通域过滤
    acc_mask = cv2.morphologyEx(acc_mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11)))
    num, lab, stats, _ = cv2.connectedComponentsWithStats(acc_mask, connectivity=8)
    final = np.zeros_like(acc_mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            final[lab == i] = 255

    debug['band'] = band*255
    debug['mask_front'] = mask_front*255 if N>0 else None
    debug['mask_tips'] = mask_tips*255 if N>0 else None
    return final.astype(np.uint8), debug
