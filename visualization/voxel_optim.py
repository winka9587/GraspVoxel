import numpy as np
import cv2

class GraspVolumeHeatmap:
    def __init__(self, voxel_size=0.01, margin=0.06):
        self.voxel_size = voxel_size
        self.margin = margin
        self.L = None  # 体素网格的 log-odds 或概率

    def _get_ray_samples(self, O, D, num_samples):
        """计算射线与体素网格交点的样本点"""
        t0 = 0.01  # 起始点
        t1 = 0.30  # 终点
        ts = np.linspace(t0, t1, num_samples)  # 在射线段上生成采样点
        pts = O + ts[:, None] * D  # 计算沿射线的样本点 (num_samples, 3)
        return pts, ts

    def update_with_rays(self, O, D, num_samples=100):
        """通过射线更新体素网格的概率"""
        pts, ts = self._get_ray_samples(O, D, num_samples)
        # 假设每个体素的概率是随机的（实际中通过某种方式计算）
        probabilities = np.random.rand(len(ts))  # 随机概率值（替代）
        cumulative_probability = 1.0

        # 计算射线经过每个体素时的累积概率
        for p in probabilities:
            cumulative_probability *= (1 - p * self.voxel_size)  # 更新累积概率
        return 1 - cumulative_probability  # 计算最终的交互概率

    def render_interaction_area(self, O, D, num_samples=100):
        """体渲染方式生成交互区域：根据射线和体素的交互概率"""
        P = self.update_with_rays(O, D, num_samples)
        # 将交互区域的概率渲染到2D平面（这里只是一个示例）
        print(f"Interaction Probability: {P}")
        return P

    def compute_loss(self, predicted_mask, true_mask):
        """计算与物体掩码的损失（交并比损失）"""
        intersection = np.sum(predicted_mask & true_mask)
        union = np.sum(predicted_mask | true_mask)
        iou = intersection / union if union != 0 else 0
        loss = 1 - iou  # IoU损失，目标是最大化IoU
        return loss


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

        # 颜色映射 + 预乘alpha
        color_bgr = cv2.applyColorMap(h8, colormap)
        a = (h8.astype(np.float32) / 255.0) * float(alpha)
        a8 = np.clip(a * 255.0, 0, 255).astype(np.uint8)

        out = np.zeros((img_h, img_w, 4), np.uint8)
        out[:, :, :3] = color_bgr
        out[:, :, 3]  = a8
        return out
