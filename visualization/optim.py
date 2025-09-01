import numpy as np
import cv2
import math

def _sigmoid(x): return 1.0/(1.0+np.exp(-x))
def _logit(p):  p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))

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

def render_heatmap_image_only(heat3d,
                              img_w, img_h, fx, fy, cx, cy,
                              gamma=0.9,            # 伽马：<1 提升暗部
                              block_px=12,          # 像素块膨胀（奇数）
                              blur_px=0,            # 可选轻微模糊（奇数，0=不用）
                              colormap=cv2.COLORMAP_PLASMA,  # 暗紫→黄→红
                              draw_grid=True, grid_step=16, grid_color=(48, 36, 64), grid_alpha=0.28):
    """
    返回 (H,W,3) uint8 的纯 heatmap 可视化（BGR）。
    - 使用体素最大投影；不做阈值，完整展示概率（更稳定）
    - block_px 控制“像素块”效果；grid 可开网格线
    """
    # 1) 拿 3D 概率
    p3 = heat3d.probability()
    if p3 is None or float(p3.max()) <= 1e-8:
        return np.zeros((img_h, img_w, 3), np.uint8)

    # 2) 体素中心投影（不阈值，直接最大投影）
    Z, Y, X = np.where(p3 > 0)
    if len(Z) == 0:
        return np.zeros((img_h, img_w, 3), np.uint8)

    pts_xyz = np.stack([X, Y, Z], axis=1).astype(np.float64)
    centers = heat3d.origin_xyz[None, :] + pts_xyz * heat3d.voxel_size

    z = centers[:, 2]
    u = np.round(fx * (centers[:, 0] / np.where(z > 1e-6, z, 1.0)) + cx).astype(np.int32)
    v = np.round(fy * (centers[:, 1] / np.where(z > 1e-6, z, 1.0)) + cy).astype(np.int32)
    m = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h) & (z > 1e-6)
    if not np.any(m):
        return np.zeros((img_h, img_w, 3), np.uint8)

    u = u[m]; v = v[m]
    val = p3[Z[m], Y[m], X[m]].astype(np.float32)

    heat = np.zeros((img_h, img_w), np.float32)
    np.maximum.at(heat, (v, u), val)  # 最大投影

    # 3) 归一化 + 伽马
    h = heat / (heat.max() + 1e-12)
    if gamma is not None and gamma != 1.0:
        h = np.power(np.clip(h, 0, 1), float(gamma))

    # 4) 像素块膨胀 + 可选模糊
    h8 = (np.clip(h, 0, 1) * 255).astype(np.uint8)
    if block_px and int(block_px) > 1:
        k = int(block_px) | 1
        h8 = cv2.dilate(h8, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), 1)
    if blur_px and int(blur_px) > 1:
        bp = int(blur_px) | 1
        h8 = cv2.GaussianBlur(h8, (bp, bp), 0)

    # 5) 颜色映射（PLASMA：暗紫→黄→红）
    color = cv2.applyColorMap(h8, colormap)  # BGR

    # 6) 叠网格线（可选）
    if draw_grid and grid_step >= 4:
        grid = color.astype(np.float32) / 255.0
        for x in range(0, img_w, int(grid_step)):
            cv2.line(grid, (x, 0), (x, img_h-1), tuple(c/255.0 for c in grid_color), 1, cv2.LINE_AA)
        for y in range(0, img_h, int(grid_step)):
            cv2.line(grid, (0, y), (img_w-1, y), tuple(c/255.0 for c in grid_color), 1, cv2.LINE_AA)
        color = np.clip(color * (1.0 - grid_alpha) + (grid * 255.0) * grid_alpha, 0, 255).astype(np.uint8)

    return color

def overlay_heat_on_image(bg_bgr_u8, heat_bgr_u8,
                          min_intensity=8,   # 忽略很暗的热图像素(0-255)
                          alpha_gain=0.85,   # 叠加强度
                          alpha_gamma=0.80,  # 置信度γ(<1 提升暗部可见度)
                          mode='screen'):    # 'screen' 更醒目；'normal' 普通混合
    # 尺寸对齐
    if bg_bgr_u8.shape[:2] != heat_bgr_u8.shape[:2]:
        heat_bgr_u8 = cv2.resize(heat_bgr_u8, (bg_bgr_u8.shape[1], bg_bgr_u8.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

    bg = bg_bgr_u8.astype(np.float32) / 255.0
    fg = heat_bgr_u8.astype(np.float32) / 255.0

    # 用热图亮度作为 alpha（置信度）
    gray = cv2.cvtColor(heat_bgr_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    a = np.power(np.clip(gray, 0, 1), float(alpha_gamma)) * float(alpha_gain)

    # 仅在亮度超过阈值时叠加
    mask = (gray >= (min_intensity/255.0)).astype(np.float32)[..., None]
    a = (a * mask[...,0])[..., None]  # (H,W,1)

    if mode == 'screen':
        screen = 1.0 - (1.0 - bg) * (1.0 - fg)  # 屏幕模式
        out = bg * (1.0 - a) + screen * a
    else:  # 'normal'
        out = bg * (1.0 - a) + fg * a

    return (np.clip(out, 0, 1) * 255.0 + 0.5).astype(np.uint8)
