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

