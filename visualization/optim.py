import numpy as np

import cv2

def rasterize_cylinder_region(
    rays, t_stars, img_w, img_h, focal_length,
    c=None, a=None, r=None,
    min_area=200, close_ks=9
):
    """
    rays: 与 t_stars 一一对应，每项含 O_cam(3), D_cam(3)
    """
    fx = fy = float(focal_length)
    cx, cy = img_w / 2.0, img_h / 2.0

    canvas = np.zeros((img_h, img_w), np.uint8)
    for i, s in enumerate(rays):
        O = np.asarray(s['O_cam'], np.float64)
        D = np.asarray(s['D_cam'], np.float64)
        D = D / (np.linalg.norm(D) + 1e-12)
        t = float(t_stars[i])
        if not np.isfinite(t) or t <= 0:      # 忽略手后方/异常
            continue
        Q = O + t * D
        z = float(Q[2])
        if z <= 1e-6 or not np.isfinite(z):
            continue
        u = int(round(fx * (Q[0]/z) + cx))
        v = int(round(fy * (Q[1]/z) + cy))
        # 近似像素半径
        r_pix = int(round((r if r is not None else 0.03) * fx / z))
        r_pix = max(2, min(128, r_pix))
        cv2.circle(canvas, (u, v), r_pix, 255, -1)

    if close_ks and close_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, k)

    # 过滤小连通域
    num, lab, stats, _ = cv2.connectedComponentsWithStats(canvas, connectivity=8)
    out = np.zeros_like(canvas)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
            out[lab == i] = 255

    return out


def _proj_onto_plane(I, a):
    # 投影矩阵 B = I - a a^T
    return I - np.outer(a, a)

def _closest_t_to_axis(O, D, c, a):
    """
    沿射线 O + t D 到轴线 (c, a) 的最小径向距离对应的 t*（解析解）：
      令 B = I - a a^T,  u(t) = B (O - c + t D)
      最小化 ||u(t)||^2 => t* = - ( (B(O-c))·(BD) ) / ||BD||^2
    返回 t*, u(t*), ||u(t*)||
    """
    I = np.eye(3, dtype=np.float64)
    B = _proj_onto_plane(I, a)
    BOc = B @ (O - c)
    BD  = B @ D
    denom = float(BD @ BD) + 1e-12
    t_star = - float(BOc @ BD) / denom
    u = BOc + t_star * BD
    rho = float(np.linalg.norm(u))
    return t_star, u, rho

def _numeric_jacobian(fun, p, eps=1e-5):
    # 数值雅可比：列为各参数微扰
    p = p.astype(np.float64)
    r0 = fun(p)                  # (M,)
    J = np.zeros((r0.size, p.size), np.float64)
    for k in range(p.size):
        pk = p.copy();  pk[k] += eps
        rk = fun(pk)
        J[:, k] = (rk - r0) / eps
    return r0, J

def fit_cylinder_gauss_newton(
    rays,                 # list of dict with keys: 'O_cam' (3,), 'D_cam' (3,), and optional 'kp_idx'
    max_iters=15,
    huber_delta=3.0,      # Huber 阈值，单位 = 圆柱“半径误差”的 3D 距离标尺
    lambda_damp=1e-3,     # 阻尼，防发散
    init=None             # 可传 (c,a,r) 初值；否则自动估计
):
    """
    返回 (c, a, r, t_stars, inlier_mask)
    其中 a 为单位向量；t_stars 与 rays 一一对应。
    """
    # ---- 准备数据 ----
    O = np.stack([np.asarray(s['O_cam'], np.float64) for s in rays], axis=0)
    D = np.stack([np.asarray(s['D_cam'], np.float64) for s in rays], axis=0)
    # 单位化方向
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)

    # ---- 初值 ----
    if init is None:
        # 轴向：取 D 的主方向（PCA 第一主成分）
        U, S, Vt = np.linalg.svd(D - D.mean(0, keepdims=True), full_matrices=False)
        a = Vt[0];  a = a / (np.linalg.norm(a) + 1e-12)
        # 先假设 c ~ O 的中位数
        c = np.median(O, axis=0)
        # 估计半径：用各射线到轴的最小径向距离的中位数
        r_samples = []
        for i in range(len(O)):
            _, u, rho = _closest_t_to_axis(O[i], D[i], c, a)
            r_samples.append(rho)
        r = np.median(r_samples) if r_samples else 0.03  # 兜底 3cm（按你的单位调）
    else:
        c, a, r = [np.asarray(x, np.float64) for x in init]
        a = a / (np.linalg.norm(a) + 1e-12)
        r = float(r)

    # 参数向量 p = [c(3), a(3), r(1)]
    def pack(c, a, r):
        return np.concatenate([c, a, [r]]).astype(np.float64)
    def unpack(p):
        c = p[0:3]
        a = p[3:6]; a = a / (np.linalg.norm(a) + 1e-12)
        r = float(p[6])
        return c, a, r

    # 残差函数（Huber 权重在外面做）
    def residuals(p):
        c, a, r_cur = unpack(p)
        res = np.zeros(len(O), np.float64)
        for i in range(len(O)):
            _, u, rho = _closest_t_to_axis(O[i], D[i], c, a)
            res[i] = rho - r_cur
        return res

    p = pack(c, a, r)

    # ---- Gauss–Newton 主循环 ----
    for _ in range(max_iters):
        r_vec, J = _numeric_jacobian(residuals, p, eps=1e-5)  # J: (M,7)
        # Huber 权重
        abs_r = np.abs(r_vec)
        w = np.ones_like(abs_r)
        big = abs_r > huber_delta
        w[big] = huber_delta / (abs_r[big] + 1e-12)

        # 加权正规方程
        W = np.diag(w)
        A = J.T @ W @ J + lambda_damp * np.eye(J.shape[1])
        b = - J.T @ W @ r_vec

        try:
            dp = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break

        p_new = p + dp
        c_new, a_new, r_new = unpack(p_new)
        if r_new <= 1e-6 or not np.isfinite(p_new).all():
            break

        # 接受更新
        p = pack(c_new, a_new, r_new)

        # 简单收敛判据
        if np.linalg.norm(dp) < 1e-6:
            break

    # 输出
    c, a, r = unpack(p)

    # 计算每条射线的 t* 和内点掩码
    t_stars, inlier = [], []
    for i in range(len(O)):
        t_star, _, rho = _closest_t_to_axis(O[i], D[i], c, a)
        t_stars.append(t_star)
        inlier.append(abs(rho - r) <= 2.5 * huber_delta)
    inlier = np.array(inlier, np.bool_)
    return c, a, r, np.array(t_stars), inlier


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

import numpy as np
import cv2

def _mano_parent(kp): return 0 if kp in (1,4,7,10,13) else (kp-1)
def _mano_child(kp):  return None if kp in (3,6,9,12,15) else (kp+1)

import numpy as np
import cv2

import numpy as np
import cv2

def fit_starconvex_polygon_from_segments(
    segments,
    tips_only=True,
    exclude_kp={13,14,15},

    # —— 稳定性关键参数 ——
    smooth_lambda=0.5,        # 半径平滑强度（比原来小，避免收缩）
    huber_delta=3.0,          # IRLS Huber 阈值（像素）
    irls_iters=6,

    # —— 纵向锚定（尺度） ——
    anchor_mode='fraction',   # 'fraction' | 'p1' | 'none'
    anchor_frac=0.85,         # 让 q_i 靠近 p0→p1 的 85% 位置
    anchor_weight=1.2,        # 纵向锚定权重（相对正交约束）

    # —— 防塌缩（半径最小/全局先验） ——
    min_forward_px=10.0,      # 每条样本至少前进这么多像素（纵向目标的下限）
    min_radius_px=8.0,        # 每个 h_i 的软下限（像素）
    floor_weight=0.8,         # 下限软约束的权重；0 关闭
    global_scale_prior=True,  # 加一个“全局平均半径”先验，抗整体缩小
    scale_prior_weight=0.4    # 全局尺度先验权重；0 关闭
):
    """
    约束:
      1) 正交:  n_i^T (c + h_i u_i - p0_i) = 0,  n_i = [-v_y, v_x]  (只一行)
      2) 纵向:  v_i^T (c + h_i u_i - p0_i) ≈ target_i
         其中 target_i = max(anchor_frac * L_i, min_forward_px) 或 L_i (mode='p1')
      3) 平滑:  h_i - 0.5(h_{i-1}+h_{i+1}) ≈ 0
      4) 下限:  h_i ≳ min_radius_px  ->  以软等式 (h_i - min_radius_px) ≈ positive
      5) 全局:  (mean(h) - h0) ≈ 0, h0=median(target_i) 作为鲁棒尺度先验
    """
    # 1) 收集样本（按角排序）
    S = []
    tip_kp = {3,6,9,12} if tips_only else None
    for s in segments:
        kp = s.get('kp_idx', None)
        if kp is None: 
            continue
        if exclude_kp and kp in exclude_kp:
            continue
        if tips_only and kp not in tip_kp:
            continue

        p0 = np.asarray(s['uv0'], np.float64)
        p1 = np.asarray(s['uv1'], np.float64)
        v  = p1 - p0
        L  = np.linalg.norm(v)
        if not np.isfinite(L) or L < 1.0:
            continue
        v  = v / (L + 1e-12)
        th = np.arctan2(v[1], v[0])
        u  = np.array([np.cos(th), np.sin(th)], np.float64)
        n  = np.array([-v[1], v[0]], np.float64)  # 单行正交法向

        # 纵向目标
        if anchor_mode == 'fraction':
            tgt = max(float(anchor_frac) * L, float(min_forward_px))
        elif anchor_mode == 'p1':
            tgt = max(L, float(min_forward_px))
        else:  # 'none'
            tgt = None

        S.append((p0, p1, v, u, n, th, L, tgt))

    if len(S) < 3:
        return None

    S.sort(key=lambda t: t[5])  # 按角度
    M = len(S)

    # 2) 组装线性系统: 未知 x = [c_x, c_y, h_1..h_M]^T
    rows, rhs, row_w = [], [], []

    for i,(p0,_,v,u,n,_,_,tgt) in enumerate(S):
        # 正交约束（单行）: n^T c + (n^T u) h_i = n^T p0
        row = np.zeros(2+M, np.float64)
        row[0:2] = n
        row[2+i] = float(n @ u)
        rows.append(row)
        rhs.append(float(n @ p0))
        row_w.append(1.0)  # 正交权=1

        # 纵向锚定（可选）: v^T c + (v^T u) h_i ≈ v^T p0 + tgt
        if tgt is not None and anchor_weight > 0.0:
            row = np.zeros(2+M, np.float64)
            row[0:2] = v
            row[2+i] = float(v @ u)
            rows.append(row)
            rhs.append(float(v @ p0) + float(tgt))
            row_w.append(float(anchor_weight))

    A = np.vstack(rows)            # (#rows) x (2+M)
    b = np.asarray(rhs, np.float64)
    row_w = np.asarray(row_w, np.float64)

    # 3) 半径平滑: h_i - 0.5(h_{i-1}+h_{i+1}) ≈ 0
    R = []
    for i in range(M):
        r = np.zeros(2+M, np.float64)
        r[2+i] = 1.0
        r[2+(i-1)%M] -= 0.5
        r[2+(i+1)%M] -= 0.5
        R.append(r)
    R = np.vstack(R)  # M x (2+M)

    # 4) 全局尺度先验（可选）
    scale_R = None; scale_r = None
    if global_scale_prior:
        # 用纵向目标的中位数作为 h 的粗先验（鲁棒）
        tgts = [t for *_, t in S if t is not None]
        if len(tgts) >= 1:
            h0 = np.median(tgts)
            scale_R = np.zeros((1, 2+M), np.float64)
            scale_R[0, 2:] = 1.0 / max(1, M)  # mean(h)
            scale_r = np.array([h0], np.float64)

    # 5) IRLS（Huber） + 软下限
    W = np.ones(A.shape[0], np.float64)
    h_floor_rows = None  # 动态添加“低于下限”的软约束

    for _ in range(max(1, int(irls_iters))):
        # 行权 & Huber 权
        Aw = (A * row_w[:,None]) * W[:,None]
        bw = b * row_w * W

        AtA = Aw.T @ Aw + float(smooth_lambda) * (R.T @ R)
        Atb = Aw.T @ bw

        if global_scale_prior and (scale_R is not None):
            AtA += float(scale_prior_weight) * (scale_R.T @ scale_R)
            Atb += float(scale_prior_weight) * (scale_R.T @ scale_r)

        # 上一轮求解得到的 h，用于决定哪些点需要“下限软约束”
        if h_floor_rows is not None and len(h_floor_rows) > 0:
            F = np.vstack(h_floor_rows)  # k x (2+M)
            AtA += float(floor_weight) * (F.T @ F)
            # 目标为 min_radius_px：F x ≈ f  等价于 1*(h_i - min) ≈ 0
            f = np.zeros((F.shape[0],), np.float64)
            Atb += float(floor_weight) * (F.T @ f)

        # 解
        try:
            x = np.linalg.solve(AtA + 1e-6*np.eye(AtA.shape[0]), Atb)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(AtA, Atb, rcond=None)[0]

        # 更新 Huber 权
        res = (A @ x - b) * row_w
        absr = np.abs(res)
        W = np.ones_like(absr)
        big = absr > float(huber_delta)
        W[big] = float(huber_delta) / (absr[big] + 1e-12)

        # 基于当前解，构造“半径下限”的软等式行（只对 h_i < min_radius_px 的 i）
        h = x[2:]
        h_floor_rows = []
        for i in range(M):
            if h[i] < float(min_radius_px):
                r = np.zeros(2+M, np.float64)
                r[2+i] = 1.0  # 让 h_i 靠近 min_radius_px
                h_floor_rows.append(r)

    c = x[0:2]
    h = np.maximum(x[2:], float(min_radius_px))  # 最终再夹一下

    # 6) 生成多边形
    poly = []
    angles = []
    for i,(*_, th, _, _) in enumerate(S):
        u = np.array([np.cos(th), np.sin(th)], np.float64)
        q = c + h[i] * u
        poly.append(q)
        angles.append(th)
    poly = np.asarray(poly, np.float32)

    return dict(center=c.astype(np.float32),
                angles=np.asarray(angles, np.float32),
                radii=h.astype(np.float32),
                polygon=poly)


# def rasterize_polygon_mask(poly, img_shape, min_area=200, close_ks=7):
#     H, W = img_shape[:2]
#     mask = np.zeros((H,W), np.uint8)
#     if poly is None or len(poly) < 3:
#         return mask
#     pts = poly.reshape(-1,1,2).astype(np.int32)
#     cv2.fillPoly(mask, [pts], 255)
#     if close_ks and close_ks > 1:
#         k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
#     # 小区域过滤
#     num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
#     out = np.zeros_like(mask)
#     for i in range(1, num):
#         if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
#             out[lab==i] = 255
#     return out

import  math
import numpy as np, math

def compute_selected_sectors_mask(segments, fit_result, img_shape,
                                  tips_only=True, exclude_kp={13,14,15},
                                  augment_k=2, delta_deg=20.0,
                                  anchor_mode='fraction', anchor_frac=0.85,
                                  min_forward_px=None,
                                  tau_orth_px=3.0, tau_long_px=4.0):
    """
    返回: selected_mask (M,), residuals_orth (M,), residuals_long (M,)
    - tau_orth_px: 正交残差阈值（像素）
    - tau_long_px: 纵向残差阈值（像素）
    """
    H, W = img_shape[:2]
    if min_forward_px is None:
        min_forward_px = 0.03 * max(H, W)

    S = _collect_star_samples(segments, tips_only, exclude_kp, augment_k, delta_deg)
    if fit_result is None or len(S) == 0:
        return np.zeros((0,), bool), np.zeros((0,)), np.zeros((0,))

    c   = np.asarray(fit_result['center'], np.float64)     # (2,)
    ang = np.asarray(fit_result['angles'], np.float64)     # (M,)
    h   = np.asarray(fit_result['radii'],  np.float64)     # (M,)

    M = len(S)
    assert h.shape[0] == M, "angles/radii 与采样数不一致；请确保 augment_k/delta_deg 与拟合一致"

    sel = np.zeros(M, bool)
    r_o = np.zeros(M, np.float64)
    r_l = np.zeros(M, np.float64)

    for i,(p0, v, th, u, L) in enumerate(S):
        q  = c + h[i] * u
        n  = np.array([-v[1], v[0]], np.float64)     # 正交方向
        tgt = (max(anchor_frac * L, min_forward_px) if anchor_mode=='fraction'
               else max(L, min_forward_px))         # 'p1'

        # 正交残差: n^T (q - p0) ≈ 0
        r_orth = float(abs(n @ (q - p0)))
        # 纵向残差: v^T (q - p0) ≈ tgt
        r_long = float(abs(v @ (q - p0) - tgt))

        r_o[i] = r_orth
        r_l[i] = r_long
        sel[i] = (r_orth <= tau_orth_px) and (r_long <= tau_long_px)

    return sel, r_o, r_l


import numpy as np, cv2, math

# =========================
# A) 最小 2D 射线段长（可视化/拟合前调用）
# =========================
def enforce_min_segment_length(segments, min_len_px):
    """
    若 |uv1-uv0| < min_len_px，则沿方向把 uv1 推到该长度。
    仅影响 2D 拟合/可视化，不改 3D 数据。
    """
    out = []
    for s in segments:
        p0 = np.asarray(s['uv0'], np.float32)
        p1 = np.asarray(s['uv1'], np.float32)
        d  = p1 - p0
        L  = float(np.linalg.norm(d))
        if not np.isfinite(L) or L < 1e-6:
            continue
        if L < float(min_len_px):
            p1 = p0 + d * (float(min_len_px) / (L + 1e-12))
        t = dict(s)
        t['uv0'] = (float(p0[0]), float(p0[1]))
        t['uv1'] = (float(p1[0]), float(p1[1]))
        out.append(t)
    return out


# 角度增强采样（与拟合一致）
def _collect_star_samples(segments, tips_only=True, exclude_kp={13,14,15},
                          augment_k=2, delta_deg=20.0):
    """返回按角度排序的样本: [(p0, v_hat, theta, u, L)]"""
    samples = []
    tip_kp = {3,6,9,12} if tips_only else None
    for s in segments:
        kp = s.get('kp_idx', None)
        if kp is None: 
            continue
        if exclude_kp and kp in exclude_kp:
            continue
        if tips_only and kp not in tip_kp:
            continue
        p0 = np.asarray(s['uv0'], np.float64)
        p1 = np.asarray(s['uv1'], np.float64)
        v  = p1 - p0
        L  = np.linalg.norm(v)
        if not np.isfinite(L) or L < 1.0:
            continue
        v  = v / (L + 1e-12)
        th0 = math.atan2(v[1], v[0])

        samples.append((p0, v, th0, np.array([math.cos(th0), math.sin(th0)], np.float64), L))
        for k in range(1, int(augment_k)+1):
            d = math.radians(delta_deg * k)
            for sgn in (+1, -1):
                th = th0 + sgn * d
                u  = np.array([math.cos(th), math.sin(th)], np.float64)
                samples.append((p0, v, th, u, L))
    samples.sort(key=lambda t: t[2])
    return samples

# =========================
# C) 最小面积兜底（拟合后放大半径）
# =========================
def ensure_min_polygon_area(poly, center, min_area_frac, img_shape):
    """
    若多边形面积 < 画面面积 * min_area_frac，则以中心为原点等比例放大。
    返回放大后的 poly。
    """
    H, W = img_shape[:2]
    if poly is None or len(poly) < 3:
        return poly

    def _area(p):
        x = p[:,0]; y = p[:,1]
        return 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

    A = _area(poly)
    A_min = float(min_area_frac) * (H * W)
    if A < A_min and A > 1e-6:
        s = (A_min / A) ** 0.5
        poly = (center + (poly - center) * s).astype(np.float32)
    return poly


# ==============
# 区域生成封装
# ==============
def rasterize_polygon_mask(poly, img_shape, min_area=250, close_ks=9):
    H, W = img_shape[:2]
    mask = np.zeros((H,W), np.uint8)
    if poly is None or len(poly) < 3:
        return mask
    pts = poly.reshape(-1,1,2).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    if close_ks and close_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    # 小区域过滤
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
            out[lab==i] = 255
    return out

# =========================
# B) 鲁棒星形拟合（加入中心先验 + 更强尺度/下限）
# =========================
def fit_starconvex_polygon_from_segments_robust(
    segments, img_shape,
    tips_only=True, exclude_kp={13,14,15},

    # 角度增强
    augment_k=2, delta_deg=20.0,

    # 纵向锚定（尺度）
    anchor_mode='fraction',   # 'fraction' | 'p1'
    anchor_frac=0.85,
    anchor_weight=1.5,

    # 下限/先验
    min_forward_px=None,      # None -> 0.06*max(H,W)
    min_radius_px=None,       # None -> 0.6*min_forward_px
    floor_weight=1.0,
    center_prior_weight=0.6,  # ★ 新增：把中心拉向关节起点质心

    # 平滑/稳健
    smooth_lambda=0.6,
    huber_delta=3.0,
    irls_iters=6,

    # 全局尺度先验
    scale_prior_weight=0.5
):
    """
    返回 dict(center, angles, radii, polygon) 或 None
    约束：
      正交：n_i^T (c + h_i u_i - p0_i) = 0（单行）
      纵向：v_i^T (c + h_i u_i - p0_i) ≈ target_i
      平滑：h_i - 0.5(h_{i-1}+h_{i+1}) ≈ 0
      下限：h_i ≳ min_radius_px（软约束）
      中心先验：c ≈ mean({p0_i})
      全局尺度：mean(h) ≈ median(target_i)
    """
    H, W = img_shape[:2]
    if min_forward_px is None:
        min_forward_px = 0.06 * max(H, W)   # 比之前更大，抗塌缩
    if min_radius_px is None:
        min_radius_px  = 0.6 * float(min_forward_px)

    S = _collect_star_samples(segments, tips_only, exclude_kp, augment_k, delta_deg)
    if len(S) < 6:
        return None

    # 目标纵向距离
    def _tgt(L):
        return max(float(anchor_frac) * L, float(min_forward_px)) if anchor_mode=='fraction' \
               else max(L, float(min_forward_px))

    # 线性系统
    M = len(S)
    rows, rhs, row_w, thetas, tgts = [], [], [], [], []
    for i,(p0, v, th, u, L) in enumerate(S):
        n = np.array([-v[1], v[0]], np.float64)

        # 正交：n^T c + (n^T u) h_i = n^T p0
        row = np.zeros(2+M, np.float64)
        row[0:2] = n
        row[2+i] = float(n @ u)
        rows.append(row); rhs.append(float(n @ p0)); row_w.append(1.0)

        # 纵向：v^T c + (v^T u) h_i ≈ v^T p0 + tgt
        tgt = _tgt(L)
        row = np.zeros(2+M, np.float64)
        row[0:2] = v
        row[2+i] = float(v @ u)
        rows.append(row); rhs.append(float(v @ p0) + float(tgt)); row_w.append(float(anchor_weight))

        thetas.append(th); tgts.append(tgt)

    A = np.vstack(rows)
    b = np.asarray(rhs, np.float64)
    row_w = np.asarray(row_w, np.float64)
    thetas = np.asarray(thetas, np.float64)

    # 平滑项
    R = []
    for i in range(M):
        r = np.zeros(2+M, np.float64)
        r[2+i] = 1.0
        r[2+(i-1)%M] -= 0.5
        r[2+(i+1)%M] -= 0.5
        R.append(r)
    R = np.vstack(R)

    # 全局尺度先验：mean(h) ≈ median(tgt)
    h0 = np.median(tgts)
    Rg = np.zeros((1, 2+M), np.float64); Rg[0, 2:] = 1.0 / max(1, M)
    rg = np.array([h0], np.float64)

    # 中心先验：c ≈ mean(p0_i)
    p0s = [np.asarray(s['uv0'], np.float64) for s in segments]
    c0  = np.mean(np.stack(p0s, 0), axis=0)
    Rc = np.zeros((2, 2+M), np.float64); Rc[:, 0:2] = np.eye(2)
    rc = c0.reshape(2)

    # IRLS + 下限软约束
    Wv = np.ones(A.shape[0], np.float64)
    floor_rows = None

    for _ in range(int(max(1, irls_iters))):
        Aw = (A * row_w[:,None]) * Wv[:,None]
        bw = b * row_w * Wv

        AtA = Aw.T @ Aw \
              + float(smooth_lambda) * (R.T @ R) \
              + float(scale_prior_weight) * (Rg.T @ Rg) \
              + float(center_prior_weight) * (Rc.T @ Rc)

        Atb = Aw.T @ bw \
              + float(scale_prior_weight) * (Rg.T @ rg) \
              + float(center_prior_weight) * (Rc.T @ rc)

        if floor_rows:
            F = np.vstack(floor_rows)
            AtA += float(floor_weight) * (F.T @ F)

        try:
            x = np.linalg.solve(AtA + 1e-6*np.eye(AtA.shape[0]), Atb)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(AtA, Atb, rcond=None)[0]

        # Huber 权
        res = (A @ x - b) * row_w
        absr = np.abs(res)
        Wv = np.ones_like(absr)
        big = absr > float(huber_delta)
        Wv[big] = float(huber_delta) / (absr[big] + 1e-12)

        # 半径软下限
        h = x[2:]
        floor_rows = []
        for i in range(M):
            if h[i] < float(min_radius_px):
                r = np.zeros(2+M, np.float64)
                r[2+i] = 1.0
                floor_rows.append(r)

    c = x[0:2]
    h = np.maximum(x[2:], float(min_radius_px))

    # 输出多边形（按角度）
    order = np.argsort(thetas)
    thetas = thetas[order]; h = h[order]
    poly = np.stack([c[0] + h*np.cos(thetas), c[1] + h*np.sin(thetas)], axis=1).astype(np.float32)

    return dict(center=c.astype(np.float32),
                angles=thetas.astype(np.float32),
                radii=h.astype(np.float32),
                polygon=poly)



def build_interaction_region_starconvex_robust(
    segments, img_shape,
    tips_only=True, exclude_kp={13,14,15},
    # 以下参数原本是直接出现在签名里的；现在允许也从 **fit_kwargs 里传入
    augment_k=2, delta_deg=20.0,
    anchor_mode='fraction', anchor_frac=0.85, anchor_weight=1.5,
    smooth_lambda=0.6, huber_delta=3.0, irls_iters=6,
    scale_prior_weight=0.5, center_prior_weight=0.6,
    # —— 其余参数统一从 **fit_kwargs 里读取，避免误传到 fit_... —— 
    **fit_kwargs
):
    import numpy as np, cv2

    H, W = img_shape[:2]

    # 这两个是“封装专用”参数，只在本函数里用，不能传给 fit_...
    min_seg_len_frac = float(fit_kwargs.pop('min_seg_len_frac', 0.05))   # 默认 5%
    min_area_frac    = float(fit_kwargs.pop('min_area_frac', 0.0015))    # 默认 0.15%

    # 1) 拉长过短的 2D 射线段（仅影响拟合/可视化）
    segments2 = enforce_min_segment_length(
        segments, min_len_px=min_seg_len_frac * max(H, W)
    )

    # 2) 做鲁棒星形拟合（只传它能识别的参数）
    fit = fit_starconvex_polygon_from_segments_robust(
        segments2, img_shape,
        tips_only=tips_only, exclude_kp=exclude_kp,
        augment_k=augment_k, delta_deg=delta_deg,
        anchor_mode=anchor_mode, anchor_frac=anchor_frac, anchor_weight=anchor_weight,
        smooth_lambda=smooth_lambda, huber_delta=huber_delta, irls_iters=irls_iters,
        scale_prior_weight=scale_prior_weight, center_prior_weight=center_prior_weight,
        # 以及用户可能通过 **fit_kwargs 传入、且 fit_... 支持的键，比如：
        **{k: v for k, v in fit_kwargs.items()
           if k in {
               'min_forward_px','min_radius_px','floor_weight',
               'center_prior_weight','scale_prior_weight'
           }}
    )

    if fit is None:
        return np.zeros((H, W), np.uint8), None

    # 3) 拟合后做“最小面积兜底”放大
    poly = ensure_min_polygon_area(
        poly=fit['polygon'], center=fit['center'],
        min_area_frac=min_area_frac, img_shape=img_shape
    )
    if poly is not None:
        fit['polygon'] = poly
        fit['radii'] = np.linalg.norm(
            poly - fit['center'][None, :], axis=1
        ).astype(np.float32)

    mask = rasterize_polygon_mask(
        poly, img_shape,
        min_area=int(min_area_frac * H * W), close_ks=9
    )
    return mask, poly

import cv2
import numpy as np

def draw_star_sectors_overlay(img_bgr, fit_result, selected_mask,
                              all_color=(255,255,255), all_alpha=0.18,
                              sel_color=(0,165,255), sel_alpha=0.45,
                              edge_color=(200,200,200), edge_thickness=1):
    """
    输入:
      img_bgr: 原图 (H,W,3) BGR
      fit_result: {'center','angles','radii','polygon'}（来自拟合）
      selected_mask: (M,) 布尔数组，哪些扇区“被选中”
    输出:
      vis_bgr: 已叠加可视化的图像 (H,W,3)
    说明:
      - 每个扇区画成三角形 [c, poly[i], poly[i+1]]；所有先画白色，再把选中的覆盖成橙色
    """
    if fit_result is None or fit_result.get('polygon', None) is None:
        return img_bgr.copy()

    H, W = img_bgr.shape[:2]
    vis = img_bgr.copy()
    overlay = np.zeros_like(vis, np.uint8)

    poly = fit_result['polygon'].astype(np.int32)      # (M,2)
    c    = fit_result['center'].astype(np.int32)
    M    = poly.shape[0]
    assert selected_mask.shape[0] == M, "selected_mask 长度应与多边形点数一致"

    # 1) 画所有扇区（白色）
    for i in range(M):
        j = (i + 1) % M
        tri = np.array([c, poly[i], poly[j]], np.int32).reshape(-1,1,2)
        cv2.fillConvexPoly(overlay, tri, all_color)

    vis = cv2.addWeighted(overlay, float(all_alpha), vis, 1.0 - float(all_alpha), 0)

    # 2) 只画“选中”扇区（橙色）
    overlay_sel = np.zeros_like(vis, np.uint8)
    for i in range(M):
        if not bool(selected_mask[i]): 
            continue
        j = (i + 1) % M
        tri = np.array([c, poly[i], poly[j]], np.int32).reshape(-1,1,2)
        cv2.fillConvexPoly(overlay_sel, tri, sel_color)

    vis = cv2.addWeighted(overlay_sel, float(sel_alpha), vis, 1.0 - float(sel_alpha), 0)

    # 3) 可选：描边多边形
    if edge_thickness > 0:
        cv2.polylines(vis, [poly.reshape(-1,1,2)], True, edge_color, edge_thickness, cv2.LINE_AA)

    return vis



import numpy as np
import cv2

import numpy as np
import cv2

def _sigmoid(x): return 1.0/(1.0+np.exp(-x))
def _logit(p):  p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))

def _project_pinhole(X, fx, fy, cx, cy):
    z = X[:,2:3]
    z = np.where(z>1e-6, z, 1.0)
    u = fx*(X[:,0:1]/z)+cx
    v = fy*(X[:,1:2]/z)+cy
    return np.concatenate([u,v],axis=1)

class GraspVolumeHeatmap:
    """
    在相机坐标系维护 3D 抓握概率体素（log-odds 累积）。
    【修复】索引/维度完全统一：
      - origin_xyz: (x0,y0,z0)  世界坐标起点
      - dims_xyz:   (nx,ny,nz)  每轴体素数
      - L.shape  =  (nz,ny,nx)  存储顺序 Z,Y,X
      - xyz<->zyx 显式互转，越界检查基于 L.shape
    """
    def __init__(self,
                 voxel_size=0.01,
                 margin=0.06,
                 decay=0.98,
                 logit_init=_logit(0.05),
                 logit_min=-6.0, logit_max=6.0):
        self.voxel_size = float(voxel_size)
        self.margin     = float(margin)
        self.decay      = float(decay)
        self.L_init     = float(logit_init)
        self.L_min      = float(logit_min)
        self.L_max      = float(logit_max)

        self.origin_xyz = None   # (3,)
        self.dims_xyz   = None   # (nx,ny,nz)
        self.L          = None   # (nz,ny,nx) log-odds

    def probability(self):
        if self.L is None: return None
        return _sigmoid(self.L.astype(np.float64))

    def _get_splat_kernel(self, sigma_perp):
        """
        返回：
        offsets_xyz: (K,3) int 邻域体素偏移（x,y,z）
        kernel_w:    (K,)  对应的各向同性高斯权重（单位：世界长度）
        半径按 sigma_perp/voxel_size 自适应，并限制到 <=3 体素半径，避免过慢。
        """
        if not hasattr(self, "_splat_cache"):
            self._splat_cache = {}
        key = (round(float(sigma_perp)/max(self.voxel_size,1e-9), 3))
        if key in self._splat_cache:
            return self._splat_cache[key]

        r_vox = int(np.ceil(2.0 * float(sigma_perp) / float(self.voxel_size)))
        r_vox = int(max(1, min(3, r_vox)))  # 1..3
        xs = np.arange(-r_vox, r_vox+1, dtype=np.int32)
        grid = np.stack(np.meshgrid(xs, xs, xs, indexing='xy'), axis=-1).reshape(-1,3)  # (K,3)
        # 各向同性核（近似横向平滑；更快）
        d = np.linalg.norm(grid.astype(np.float64) * float(self.voxel_size), axis=1)    # 世界距离
        kernel = np.exp(-0.5 * (d / (float(sigma_perp)+1e-12))**2).astype(np.float32)
        # 过滤极小权重，减少写入量
        m = kernel > 1e-4
        offsets_xyz = grid[m]
        kernel_w    = kernel[m]
        self._splat_cache[key] = (offsets_xyz, kernel_w)
        return offsets_xyz, kernel_w

    # ---------- 内部：网格分配 ----------
    def _alloc_if_needed(self, V_cam):
        mins = V_cam.min(axis=0) - self.margin        # (x_min,y_min,z_min)
        maxs = V_cam.max(axis=0) + self.margin
        dims_xyz = np.maximum(1, np.ceil((maxs - mins)/self.voxel_size).astype(int))  # (nx,ny,nz)

        need_new = (
            self.L is None or
            self.dims_xyz is None or
            np.any(np.abs(dims_xyz - self.dims_xyz) > 4) or
            self.origin_xyz is None or
            np.linalg.norm((mins + self.margin) - (self.origin_xyz if self.origin_xyz is not None else mins)) > 2*self.margin
        )
        if need_new:
            self.origin_xyz = mins
            self.dims_xyz   = tuple(dims_xyz.tolist())     # (nx,ny,nz)
            nx, ny, nz = self.dims_xyz
            self.L = np.full((nz, ny, nx), self.L_init, dtype=np.float32)  # 注意顺序 (Z,Y,X)

    def decay_step(self):
        if self.L is not None:
            self.L *= self.decay

    # ---------- 坐标/索引辅助 ----------
    def _world_to_idx_xyz(self, P):
        """
        P: (N,3) 世界(相机)坐标 → 浮点索引 (x,y,z) 与就近整数 idx_xyz
        """
        rel = (P - self.origin_xyz[None,:]) / self.voxel_size  # (N,3) in xyz
        idx_xyz = np.round(rel).astype(int)
        return rel, idx_xyz

    @staticmethod
    def _xyz_to_zyx(idx_xyz):
        """(x,y,z) -> (z,y,x)"""
        return np.stack([idx_xyz[:,2], idx_xyz[:,1], idx_xyz[:,0]], axis=1)

    def _in_bounds_zyx(self, idx_zyx):
        nz, ny, nx = self.L.shape
        z_ok = (0 <= idx_zyx[:,0]) & (idx_zyx[:,0] < nz)
        y_ok = (0 <= idx_zyx[:,1]) & (idx_zyx[:,1] < ny)
        x_ok = (0 <= idx_zyx[:,2]) & (idx_zyx[:,2] < nx)
        return z_ok & y_ok & x_ok

    # ---------- 更新 ----------
    # 用这个函数替换你原有的 update_with_rays（整段覆盖）
    def update_with_rays(self, O_list, D_list,
                     V_cam_for_bbox,
                     palm_center=None, palm_normal=None,
                     # —— 这三个如果不传，将自动根据 bbox 自适应 ——
                     prefer_dist=0.08,      # None -> auto: 0.25 * (t1-t0)
                     sigma_para=0.03,       # None -> auto: 0.12 * (t1-t0)
                     sigma_perp=0.0,        # >0 启用邻域 splat；=0 只写最近体素（最快）
                     step=None,             # None -> auto: max(voxel_size*0.75, 0.01*diag)
                     alpha_hit=0.8,
                     alpha_gate=0.5):
        V_cam = np.asarray(V_cam_for_bbox, np.float64)
        self._alloc_if_needed(V_cam)
        if self.L is None:
            return

        bbox = V_cam.max(0) - V_cam.min(0)
        diag = float(np.linalg.norm(bbox))
        if not np.isfinite(diag) or diag <= 1e-6:
            return

        # 有效射线段
        t0 = 0.01 * max(diag, 1e-3)
        t1 = 0.30 * diag
        Lseg = max(1e-6, t1 - t0)

        # 自适应步长 / 至少采 8 个点
        if step is None:
            step = max(self.voxel_size * 0.75, 0.01 * diag)
        Nmin = 8
        N = max(int(np.ceil(Lseg / float(step))), Nmin)
        ts = np.linspace(t0, t1, N, dtype=np.float32)  # (T,)
        if ts.size == 0:
            return

        # 自适应纵向高斯（防止 prefer_dist 落到段外）
        if prefer_dist is None:
            prefer_dist = 0.25 * Lseg
        if sigma_para is None:
            sigma_para = max(0.5 * self.voxel_size, 0.12 * Lseg)

        O = np.asarray(O_list, np.float64).reshape(-1,3)
        D = np.asarray(D_list, np.float64).reshape(-1,3)
        D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        R = O.shape[0]; T = ts.size

        pts = O[:,None,:] + ts[None,:,None]*D[:,None,:]    # (R,T,3)
        pts = pts.reshape(-1,3)                             # (N,3), N=R*T

        pref = np.exp(-0.5*((ts - float(prefer_dist))/float(sigma_para))**2).astype(np.float32)  # (T,)
        pref = np.repeat(pref[None,:], R, axis=0).reshape(-1)                                     # (N,)

        if (palm_center is not None) and (palm_normal is not None):
            pc = np.asarray(palm_center, np.float64).reshape(1,3)
            pn = np.asarray(palm_normal, np.float64).reshape(1,3)
            pn = pn / (np.linalg.norm(pn)+1e-12)
            sgn = np.sum((pts - pc) * pn, axis=1)
            gate = (1.0/(1.0+np.exp(-sgn/0.02))).astype(np.float32)
            gate = (1.0 - float(alpha_gate)) + float(alpha_gate)*gate
        else:
            gate = 1.0

        base_w = float(alpha_hit) * pref * (gate if np.isscalar(gate) else gate)
        m_w = base_w > 1e-6
        if not np.any(m_w):
            return
        pts    = pts[m_w]
        base_w = base_w[m_w]

        # 最近体素索引
        _, idx_xyz = self._world_to_idx_xyz(pts)
        idx_zyx = self._xyz_to_zyx(idx_xyz)
        inb = self._in_bounds_zyx(idx_zyx)
        if not np.any(inb):
            return
        idx_zyx = idx_zyx[inb]
        base_w  = base_w[inb]

        nz, ny, nx = self.L.shape
        L_flat = self.L.ravel()
        stride_y = nx
        stride_z = nx * ny
        lin = (idx_zyx[:,0] * stride_z + idx_zyx[:,1] * stride_y + idx_zyx[:,2]).astype(np.int64)

        if not (sigma_perp is not None and float(sigma_perp) > 0.0):
            # 最近体素路径（最快）
            np.add.at(L_flat, lin, base_w)
            self.L[:] = np.clip(self.L, self.L_min, self.L_max)
            return

        # 邻域 splat（向量化）
        offsets_xyz, kernel_w = self._get_splat_kernel(float(sigma_perp))  # (K,3), (K,)
        base_xyz = idx_xyz[inb].astype(np.int32)
        nb_xyz   = (base_xyz[:,None,:] + offsets_xyz[None,:,:]).reshape(-1,3)  # (N*K,3)
        nb_zyx   = np.stack([nb_xyz[:,2], nb_xyz[:,1], nb_xyz[:,0]], axis=1)

        z_ok = (0 <= nb_zyx[:,0]) & (nb_zyx[:,0] < nz)
        y_ok = (0 <= nb_zyx[:,1]) & (nb_zyx[:,1] < ny)
        x_ok = (0 <= nb_zyx[:,2]) & (nb_zyx[:,2] < nx)
        m = z_ok & y_ok & x_ok
        if not np.any(m):
            return

        nb_zyx = nb_zyx[m]
        w_rep = (base_w[:,None] * kernel_w[None,:]).reshape(-1)
        w_rep = w_rep[m]

        lin_nb = (nb_zyx[:,0] * stride_z + nb_zyx[:,1] * stride_y + nb_zyx[:,2]).astype(np.int64)
        np.add.at(L_flat, lin_nb, w_rep)
        self.L[:] = np.clip(self.L, self.L_min, self.L_max)


    # ---------- 投影 ----------
    # def render_overlay(self, img_w, img_h, fx, fy, cx, cy,
    #                    alpha=0.55, colormap=cv2.COLORMAP_JET, thresh=0.08):
    #     """
    #     最大投影：对每个像素取体素最大概率。返回 RGBA uint8。
    #     【注意】thresh 默认放宽到 0.08，便于初始“点亮”。
    #     """
    #     if self.L is None:
    #         return np.zeros((img_h,img_w,4), np.uint8)

    #     p = self.probability()            # (nz,ny,nx)
    #     # print("p.max=", None if p is None else float(p.max()), " nonzero=", None if p is None else int((p>0.08).sum()))
    #     Z, Y, X = np.where(p > float(thresh))
    #     if len(Z) == 0:
    #         return np.zeros((img_h,img_w,4), np.uint8)

    #     # 体素中心坐标（世界/相机）：origin + idx_xyz * voxel
    #     pts_xyz = np.stack([X, Y, Z], axis=1).astype(np.float64)  # (N,3) idx_xyz
    #     centers = self.origin_xyz[None,:] + pts_xyz * self.voxel_size

    #     # 投影
    #     uv = _project_pinhole(centers, fx, fy, cx, cy)
    #     u = np.round(uv[:,0]).astype(int)
    #     v = np.round(uv[:,1]).astype(int)
    #     m = (u>=0)&(u<img_w)&(v>=0)&(v<img_h)&(centers[:,2]>1e-6)
    #     if not np.any(m):
    #         return np.zeros((img_h,img_w,4), np.uint8)

    #     u = u[m]; v = v[m]
    #     val = p[Z[m], Y[m], X[m]].astype(np.float32)

    #     heat = np.zeros((img_h,img_w), np.float32)
    #     np.maximum.at(heat, (v,u), val)   # 最大投影

    #     heat_u8 = (np.clip(heat,0,1)*255).astype(np.uint8)
    #     color = cv2.applyColorMap(heat_u8, colormap)   # BGR
    #     overlay = np.zeros((img_h,img_w,4), np.uint8)
    #     overlay[:,:,:3] = color
    #     overlay[:,:,3]  = (heat * 255 * float(alpha)).astype(np.uint8)
    #     return overlay

    def render_overlay(self, img_w, img_h, fx, fy, cx, cy,
                    alpha=0.65, colormap=cv2.COLORMAP_JET,
                    thresh=0.10, auto_thresh=True, q=80,
                    ema=0.15,         # ★ 阈值与pmax的EMA系数(0..1)，越小越稳
                    relax=0.20,       # ★ 在平滑阈值基础上再下调20% → 区域更密集
                    weak_rel=0.35,    # ★ 弱阈值比例：thr_weak = thr_strong*(1-weak_rel)
                    block_px=11, blur_px=3, gamma=0.85):
        import cv2, numpy as np

        if self.L is None:
            return np.zeros((img_h, img_w, 4), np.uint8)

        p = self.probability()  # (nz,ny,nx)
        if p is None:
            return np.zeros((img_h, img_w, 4), np.uint8)
        pmax_inst = float(p.max())
        if pmax_inst <= 1e-8:
            return np.zeros((img_h, img_w, 4), np.uint8)

        # --- 阈值（分位数） ---
        if auto_thresh:
            thr_inst = np.percentile(p.ravel(), float(q))  # 上 q 分位
            thr_inst = max(thr_inst, 0.03 * pmax_inst)     # 不低于 pmax 的3%
            thr_inst = min(thr_inst, float(thresh))        # 不高于用户给的阈值
        else:
            thr_inst = float(thresh)

        # --- 时序 EMA 平滑 ---
        if not hasattr(self, "_pmax_ema") or self._pmax_ema is None:
            self._pmax_ema = pmax_inst
        if not hasattr(self, "_thr_ema") or self._thr_ema is None:
            self._thr_ema = thr_inst
        self._pmax_ema = (1.0 - ema) * self._pmax_ema + ema * pmax_inst
        self._thr_ema  = (1.0 - ema) * self._thr_ema  + ema * thr_inst

        # 平滑后的强阈值 + 额外下调（relax）
        thr_strong = max(0.02 * self._pmax_ema, self._thr_ema)  # 不低于 pmax_ema 的2%
        thr_strong *= (1.0 - float(relax))                      # 再下调一些 → 更密集
        thr_strong = max(thr_strong, 1e-6)

        # 弱阈值（滞回）
        thr_weak = max(1e-6, thr_strong * (1.0 - float(weak_rel)))

        # --- 取体素中心投影 ---
        Z, Y, X = np.where(p > thr_weak)   # 用弱阈值先挑一批
        if len(Z) == 0:
            return np.zeros((img_h, img_w, 4), np.uint8)
        pts_xyz = np.stack([X, Y, Z], axis=1).astype(np.float64)
        centers = self.origin_xyz[None, :] + pts_xyz * self.voxel_size

        # 投影到像素
        z = centers[:, 2]
        u = (fx * (centers[:, 0] / np.where(z > 1e-6, z, 1.0)) + cx).round().astype(np.int32)
        v = (fy * (centers[:, 1] / np.where(z > 1e-6, z, 1.0)) + cy).round().astype(np.int32)
        m2 = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (z > 1e-6)
        if not np.any(m2):
            return np.zeros((img_h, img_w, 4), np.uint8)
        u = u[m2]; v = v[m2]
        val = p[Z[m2], Y[m2], X[m2]].astype(np.float32)

        # 最大投影到 2D
        heat = np.zeros((img_h, img_w), np.float32)
        np.maximum.at(heat, (v, u), val)

        # --- 滞回：强/弱双阈值 + 与上帧连通 ---
        strong2d = (heat >= thr_strong).astype(np.uint8)
        weak2d   = (heat >= thr_weak ).astype(np.uint8)

        # 上帧掩码
        if not hasattr(self, "_mask_prev") or self._mask_prev is None:
            self._mask_prev = np.zeros((img_h, img_w), np.uint8)

        # 只保留：强阈值区域 + （弱阈值且与上帧mask相连的一圈）
        dil_prev = cv2.dilate(self._mask_prev, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask2d = np.clip(strong2d + (weak2d & dil_prev), 0, 1).astype(np.uint8)

        # 平滑边界 & 打块
        if gamma is not None and gamma > 0 and gamma != 1.0:
            heat = np.power(np.clip(heat, 0, 1), float(gamma))
        h8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)

        # 仅在 mask2d 内保留热度，再膨胀成块
        h8 = h8 * mask2d
        if block_px and int(block_px) > 1:
            k = int(block_px) | 1
            h8 = cv2.dilate(h8, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), 1)
        if blur_px and int(blur_px) > 1:
            bp = int(blur_px) | 1
            h8 = cv2.GaussianBlur(h8, (bp, bp), 0)

        # 更新上帧掩码（对mask做一点闭运算更稳）
        m_close = cv2.morphologyEx(mask2d, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        self._mask_prev = m_close

        # 颜色映射 + 预乘alpha
        color_bgr = cv2.applyColorMap(h8, colormap)
        a = (h8.astype(np.float32) / 255.0) * float(alpha)
        a8 = np.clip(a * 255.0, 0, 255).astype(np.uint8)

        out = np.zeros((img_h, img_w, 4), np.uint8)
        out[:, :, :3] = color_bgr
        out[:, :, 3]  = a8
        return out


from visualization.mano_joint_ray import _dense_regressor, _rotx_deg

def compute_mano_rays_in_cam(out, model, fid, cam_t, is_right_n=1, axis='y-',
                             apply_rotx_180=False, exclude_kps={13,14,15}):
    """返回：O_cam (K,3), D_cam (K,3), V_cam, palm_center, palm_normal"""
    def _dense_regressor(J_regressor):
        try:
            return J_regressor.coalesce().to_dense().cpu().numpy()
        except Exception:
            return np.asarray(J_regressor.cpu().numpy())

    def _rotx_deg(angle):
        a = np.deg2rad(angle); ca, sa = np.cos(a), np.sin(a)
        return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], np.float64)

    sgn = 1.0 if int(is_right_n)==1 else -1.0

    # 顶点（MANO坐标，左手 x 取反）
    V = out['pred_vertices'][fid].detach().cpu().numpy().astype(np.float64).copy()
    V[:, 0] *= sgn
    if apply_rotx_180:
        V = (_rotx_deg(180.0) @ V.T).T

    cam_t = np.asarray(cam_t, np.float64).reshape(3)
    V_cam = V + cam_t  # 与 hand_silhouette_overlay 一致

    # 关节（MANO回归器）
    Jr = _dense_regressor(model.mano.J_regressor)
    joints = Jr @ V
    if apply_rotx_180:
        joints = (_rotx_deg(180.0) @ joints.T).T
    joints_cam = joints + cam_t

    # 方向：global + hand_pose + 左右镜像
    g  = out['pred_mano_params']['global_orient']
    Rg = (g[fid,0] if g.ndim==4 else g[fid]).detach().cpu().numpy()
    Rh = out['pred_mano_params']['hand_pose'][fid].detach().cpu().numpy()  # (15,3,3)

    amap = {'x':[1,0,0],'y':[0,1,0],'z':[0,0,1],'x-':[-1,0,0],'y-':[0,-1,0],'z-':[0,0,-1]}
    axis_vec = np.asarray(amap.get(axis.lower() if isinstance(axis,str) else 'y-', [0,-1,0]), np.float64)
    axis_vec /= (np.linalg.norm(axis_vec)+1e-12)

    S = np.diag([sgn, 1.0, 1.0])  # 左右镜像作用在方向上
    use_set = [i for i in range(1,16) if i not in set(exclude_kps)]

    O_cam, D_cam = [], []
    for j, kp_idx in enumerate(range(1,16)):
        if kp_idx not in use_set:
            continue
        d = Rg @ (Rh[j] @ axis_vec)
        d = S @ d
        d = d / (np.linalg.norm(d)+1e-12)
        O_cam.append(joints_cam[kp_idx])
        D_cam.append(d)

    O_cam = np.asarray(O_cam, np.float64)
    D_cam = np.asarray(D_cam, np.float64)

    # 手掌中心/法向（粗估）
    palm_idx = [0,1,4,7,10]
    P  = joints_cam[palm_idx]
    pc = P.mean(0)
    n1 = np.cross(P[1]-P[0], P[2]-P[0])
    n2 = np.cross(P[3]-P[0], P[4]-P[0])
    pn = n1 + n2
    if np.dot(pn, -pc) < 0: pn = -pn
    pn = pn / (np.linalg.norm(pn)+1e-12)

    return O_cam, D_cam, V_cam, pc, pn
