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


def _mano_parent(kp_idx: int) -> int:
    """
    MANO 关节父子（0=腕；1..3=拇指，4..6=食指，7..9=中指，10..12=无名指，13..15=小指）
    每根手指的首节点的父亲是 0，其余是各自上一个。
    """
    if kp_idx in (1,4,7,10,13):
        return 0
    return kp_idx - 1


def _dense_regressor(J_regressor):
    try:
        Jr = J_regressor.coalesce().to_dense().cpu().numpy()
    except Exception:
        Jr = np.asarray(J_regressor.cpu().numpy())
    return Jr


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

# new hand area

def _rodrigues_rotate(v, k, theta):
    """把向量 v 绕单位轴 k 旋转角 theta（弧度）"""
    k = k / (np.linalg.norm(k) + 1e-12)
    ct, st = np.cos(theta), np.sin(theta)
    return v*ct + np.cross(k, v)*st + k*(np.dot(k, v))*(1.0-ct)

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
    collect_segments=False,
    # ★ 改名为通用扭转目标，默认对“食指” (4,5,6)
    twist_kp_indices=(4, 5, 6),
    twist_deg=45.0               # 顺时针 45°
):
    import cv2

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

    #—— 为 rotx180 / cam_t 做准备（与 silhouette 一致）——
    R180 = _rotx_deg(180.0) if apply_rotx_180 else None
    cam_t = np.asarray(cam_t, dtype=np.float64).copy()
    if apply_x_flip:
        cam_t[0] *= -1.0

    # 生成射线（MANO 坐标）+ 记录关节索引
    origins, directions, kp_list = [], [], []
    twist_set = set(twist_kp_indices) if twist_kp_indices is not None else set()

    for j_local, kp_idx in enumerate(range(1, 16)):  # 1..15
        if kp_idx >= joints.shape[0]:
            continue

        # 基准方向（手局部轴 → 全局 → 左手镜像）
        d = R_global @ (R_hand[j_local] @ axis_vec)
        d = (S @ d)

        # ★ 对目标关节进行“绕骨轴顺时针旋转 twist_deg”
        if kp_idx in twist_set:
            parent = _mano_parent(kp_idx)
            b = joints[kp_idx] - joints[parent]  # 骨轴（MANO 坐标）
            if np.isfinite(b).all() and np.linalg.norm(b) > 1e-8:
                theta = -np.deg2rad(float(twist_deg))  # 约定：沿“父→子”看去为顺时针，取负角
                d = _rodrigues_rotate(d, b, theta)

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
        O = (R180 @ O.T).T
        D = (R180 @ D.T).T
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

    P0, P1 = O_cam, O_cam + D * L

    # 投影
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

    for idx in np.where(mask)[0]:
        x0, y0 = uv0[idx]; x1, y1 = uv1[idx]
        if not np.isfinite([x0, y0, x1, y1]).all():
            continue

        # if collect_segments:
        #     segments.append({
        #         'uv0': (float(x0), float(y0)),
        #         'uv1': (float(x1), float(y1)),
        #         'kp_idx': int(kp_list[idx]),
        #         'hand_id': int(fid)
        #     })
            # 有效 idx 循环里，收集段时一并保存 3D：
        if collect_segments:
            segments.append({
                'uv0': (float(x0), float(y0)),
                'uv1': (float(x1), float(y1)),
                'kp_idx': int(kp_list[idx]),
                'hand_id': int(fid),
                'O_cam': O_cam[idx].astype(np.float32),  # ★ 起点(相机系)
                'D_cam': D[idx].astype(np.float32),      # ★ 方向(单位)
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
    collect_segments=False,
    # ★ 透传：默认就是食指
    twist_kp_indices=(4, 5, 6),
    twist_deg=45.0
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
        common = dict(
            out=out, model=model,
            img_w=img_w, img_h=img_h,
            cam_t=cam_t_list[i],
            focal_length=focal_list[i],
            axis=axis,
            is_right_n=is_right_list[i],
            line_color=line_color, thickness=thickness, alpha_value=alpha_value,
            apply_x_flip=apply_x_flip, apply_rotx_180=apply_rotx_180,
            invert_x=invert_x, invert_y=invert_y,
            ray_len=ray_len, fid=fids[i],
            verts_preflipped=verts_i,
            twist_kp_indices=twist_kp_indices,  # ← 透传
            twist_deg=twist_deg                  # ← 透传
        )
        if collect_segments:
            ov_i, cnt_i, seg_i = compute_mano_joint_rays_mano_overlay_single(
                collect_segments=True, **common
            )
            all_segments.extend(seg_i)
        else:
            ov_i, cnt_i = compute_mano_joint_rays_mano_overlay_single(
                collect_segments=False, **common
            )
        acc += ov_i.astype(np.float32) / 255.0
        total_cnt += cnt_i

    overlay = (np.clip(acc, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return (overlay, total_cnt, all_segments) if collect_segments else (overlay, total_cnt)


import numpy as np, cv2

def build_forward_region_from_segments_filtered(
    segments, img_shape,
    kp_filter: set,                 # 这根手指的关节集合，例如 {4,5,6}
    hand_id_filter=None,            # 仅使用该 hand_id 的线段；None 表示不过滤
    width_px=18,
    extend_ratio=1.25,
    min_area=200,
    close_ks=9
):
    H, W = img_shape[:2]
    canvas = np.zeros((H, W), np.uint8)

    for s in segments:
        if hand_id_filter is not None and s.get('hand_id', None) != hand_id_filter:
            continue
        if s.get('kp_idx', None) not in kp_filter:
            continue

        p0 = np.array(s['uv0'], dtype=np.float32)
        p1 = np.array(s['uv1'], dtype=np.float32)
        v  = p1 - p0
        L  = float(np.linalg.norm(v))
        if not np.isfinite(L) or L < 1.0:
            continue

        v /= L
        n = np.array([-v[1], v[0]], dtype=np.float32)

        # 只向前扩张（手背方向不扩张）
        front_len = L * float(extend_ratio)
        rear_len  = 0.0
        half_w0, half_w1 = width_px*0.5, width_px

        a0 = p0 + v*rear_len - n*half_w0
        a1 = p0 + v*rear_len + n*half_w0
        b0 = p1 + v*front_len - n*half_w1
        b1 = p1 + v*front_len + n*half_w1

        poly = np.stack([a0, a1, b1, b0], axis=0).astype(np.int32)
        cv2.fillConvexPoly(canvas, poly, 255)

        cap_center = (p1 + v*front_len).astype(np.int32)  # 只画前端圆帽
        cv2.circle(canvas, tuple(cap_center), int(round(half_w1)), 255, -1)

    if close_ks and close_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, k)

    # 过滤小噪声
    num, lab, stats, _ = cv2.connectedComponentsWithStats(canvas, connectivity=8)
    out = np.zeros_like(canvas)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
            out[lab == i] = 255

    return out


def build_interaction_region_five_finger_intersection(
    segments, img_shape,
    groups_map=None,                 # 可自定义：{'thumb': {1,2,3}, ...}
    width_px=18,
    extend_ratio=1.25,
    min_area_per_finger=200,
    close_ks_per_finger=9,
    intersect_dilate_px=0            # 交集前，先对每根手指区域做微膨胀，避免“空交集”
):
    H, W = img_shape[:2]
    if groups_map is None:
        groups_map = {
            'thumb':  {1,2,3},
            'index':  {4,5,6},
            'middle': {7,8,9},
            'ring':   {10,11,12},
            'little': {13,14,15},
        }

    # 分手（hand_id）处理：同一只手内部做“五指交集”，最后各手做并集
    hand_ids = sorted({s.get('hand_id', 0) for s in segments})
    final_union = np.zeros((H, W), np.uint8)

    # 可选：交集前的轻度膨胀核
    dil_k = None
    if intersect_dilate_px and intersect_dilate_px > 0:
        ksz = int(intersect_dilate_px)*2 + 1
        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))

    for hid in hand_ids:
        per_finger_masks = []

        # 逐指生成候选
        for _, kp_set in groups_map.items():
            mask = build_forward_region_from_segments_filtered(
                segments, img_shape,
                kp_filter=set(kp_set),
                hand_id_filter=hid,
                width_px=width_px,
                extend_ratio=extend_ratio,
                min_area=min_area_per_finger,
                close_ks=close_ks_per_finger
            )
            if dil_k is not None and mask.any():
                mask = cv2.dilate(mask, dil_k, iterations=1)
            per_finger_masks.append(mask)

        # 若某些手指没有候选，交集会空；这里做一个“只和存在的做交集”的逻辑
        valid_masks = [m for m in per_finger_masks if m is not None and m.any()]
        if not valid_masks:
            continue

        inter = valid_masks[0].copy()
        for m in valid_masks[1:]:
            inter = cv2.bitwise_and(inter, m)

        # 若交集太空，可退一步：至少与（存在手指里）面积最大的两根手指做交集
        if inter.sum() == 0 and len(valid_masks) >= 2:
            areas = [int(v.sum()) for v in valid_masks]
            idx_sorted = np.argsort(areas)[::-1]  # 按面积降序
            inter = valid_masks[idx_sorted[0]].copy()
            inter = cv2.bitwise_and(inter, valid_masks[idx_sorted[1]])

        final_union = cv2.bitwise_or(final_union, inter)

    return final_union


def build_forward_region_from_segments_ray(segments, img_shape,
                                       width_px=18,        # 单侧半宽
                                       extend_ratio=1.25,  # 在前向的额外伸长倍数
                                       tips_only=True,     # 仅末节
                                       min_area=300,
                                       close_ks=11):
    """
    将每条线段的前向扩张方向从原来的“射线方向 v”改为
    “射线方向 v 与手指骨链方向 f 的角平分线 w = normalize(v_hat + f_hat)”，
    若两者近乎相反则退回 v。
    末节集合默认剔除了大拇指(13,14,15) —— 如需包含，重置 tip_kp 集合即可。
    """
    import numpy as np, cv2

    H, W = img_shape[:2]
    canvas = np.zeros((H, W), np.uint8)

    # --------- 工具：MANO 父/子关系（0=腕；1..3拇指，4..6食，7..9中，10..12无名，13..15小）---------
    def _mano_parent(kp):
        return 0 if kp in (1,4,7,10,13) else (kp-1)
    def _mano_child(kp):
        return None if kp in (3,6,9,12,15) else (kp+1)

    # --------- 先做每只手的 uv0 查表（避免跨手串联）---------
    # 若没有 hand_id 字段，也能工作（默认 hand_id=None）
    uv0_by_hand = {}
    for s in segments:
        hid = s.get('hand_id', None)
        kp  = s.get('kp_idx', None)
        if kp is None: 
            continue
        uv0_by_hand.setdefault(hid, {})[kp] = np.array(s['uv0'], dtype=np.float32)

    # --------- 末节集合（默认不含拇指末节 15；如需包含，改为 {3,6,9,12,15}）---------
    tip_kp = {3, 6, 9, 12} if tips_only else None

    for s in segments:
        hid = s.get('hand_id', None)
        kp  = s.get('kp_idx', 1)

        # tips_only 过滤
        if tips_only and kp not in tip_kp:
            continue

        p0 = np.array(s['uv0'], dtype=np.float32)   # 关节投影
        p1 = np.array(s['uv1'], dtype=np.float32)   # 射线端点投影
        v  = p1 - p0                                 # 射线 2D 方向
        Lv = float(np.linalg.norm(v))
        if not np.isfinite(Lv) or Lv < 1.0:
            continue
        v_hat = v / Lv

        # --------- 计算“指头方向” f_hat（沿骨链指向远端）---------
        # 优先用子关节（更接近“指向指尖”）；若无子，则用 parent->kp 方向
        uv0_map = uv0_by_hand.get(hid, {})
        child = _mano_child(kp)
        parent = _mano_parent(kp)

        f = None
        if child is not None and child in uv0_map:
            f = uv0_map[child] - p0            # kp -> child
        elif parent in uv0_map:
            f = p0 - uv0_map[parent]           # parent -> kp（指向远端）
        if f is None or not np.isfinite(f).all() or np.linalg.norm(f) < 1.0:
            f_hat = v_hat.copy()               # 兜底：退回射线方向
        else:
            f_hat = f / (np.linalg.norm(f) + 1e-12)

        # --------- 角平分线方向 ---------
        w = v_hat + f_hat
        Lw = float(np.linalg.norm(w))
        if not np.isfinite(Lw) or Lw < 1e-6:
            w_hat = v_hat                      # 近似对向时退回射线方向
        else:
            w_hat = w / Lw

        # 前向法线（与 w_hat 垂直），用于构造条带宽度
        n = np.array([-w_hat[1], w_hat[0]], dtype=np.float32)

        # 仅向前扩张
        front_len = Lv * float(extend_ratio)
        rear_len  = 0.0
        half_w0, half_w1 = width_px*0.5, width_px   # 可按需调节“前端更宽/等宽”

        a0 = p0 + w_hat*rear_len - n*half_w0
        a1 = p0 + w_hat*rear_len + n*half_w0
        b0 = p1 + w_hat*front_len - n*half_w1
        b1 = p1 + w_hat*front_len + n*half_w1

        poly = np.stack([a0, a1, b1, b0], axis=0).astype(np.int32)
        cv2.fillConvexPoly(canvas, poly, 255)

        # 前端圆帽（沿角平分线方向推进）
        cap_center = (p1 + w_hat*front_len).astype(np.int32)
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


from visualization.optim import fit_cylinder_gauss_newton, rasterize_cylinder_region
def build_interaction_region_cylinder_gn(
    segments, img_shape, focal_length,
    exclude_kp={13,14,15},
    max_iters=15, huber_delta=3.0, lambda_damp=1e-3,
    min_area=200, close_ks=9
):
    H, W = img_shape[:2]

    # 按 hand_id 分组 & 过滤
    hands = {}
    for s in segments:
        if s.get('kp_idx') in (exclude_kp or set()):
            continue
        # 要求 O_cam / D_cam
        if ('O_cam' not in s) or ('D_cam' not in s):
            continue
        hid = s.get('hand_id', 0)
        hands.setdefault(hid, []).append(s)

    region_all = np.zeros((H, W), np.uint8)

    for hid, rays in hands.items():
        if len(rays) < 4:
            continue  # 数据点太少，跳过
        c, a, r, t_stars, inlier = fit_cylinder_gauss_newton(
            rays, max_iters=max_iters, huber_delta=huber_delta, lambda_damp=lambda_damp
        )

        # 仅用内点生成区域（更稳）
        rays_in = [rays[i] for i in range(len(rays)) if inlier[i]]
        t_in    = [t_stars[i] for i in range(len(rays)) if inlier[i]]
        if len(rays_in) < 3:
            rays_in, t_in = rays, t_stars  # 兜底

        mask = rasterize_cylinder_region(
            rays_in, t_in, img_w=W, img_h=H, focal_length=focal_length,
            c=c, a=a, r=r, min_area=min_area, close_ks=close_ks
        )
        region_all = cv2.bitwise_or(region_all, mask)

    return region_all

from visualization.optim import fit_starconvex_polygon_from_segments, rasterize_polygon_mask
def build_interaction_region_starconvex(
    segments, img_shape,
    tips_only=True,
    exclude_kp={13,14,15},
    smooth_lambda=2.0,
    huber_delta=2.0,
    irls_iters=5,
    min_area=250,
    close_ks=9
):
    # 按 hand_id 分组
    by_hand = {}
    for s in segments:
        hid = s.get('hand_id', 0)
        by_hand.setdefault(hid, []).append(s)

    final_mask = np.zeros(img_shape[:2], np.uint8)
    polys = {}

    for hid, segs in by_hand.items():
        fit = fit_starconvex_polygon_from_segments(
            segs, tips_only=tips_only, exclude_kp=exclude_kp,
            smooth_lambda=smooth_lambda, huber_delta=huber_delta, irls_iters=irls_iters
        )
        if fit is None:
            continue
        mask = rasterize_polygon_mask(
            fit['polygon'], img_shape, min_area=min_area, close_ks=close_ks
        )
        final_mask = cv2.bitwise_or(final_mask, mask)
        polys[hid] = fit['polygon']

    return final_mask, polys


