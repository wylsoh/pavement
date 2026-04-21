import numpy as np
import h5py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from scipy import signal


# =====================================================================
# 🛠️ 工具 1: 原始物理高程保真拼接 (专为 水膜、积水、3D宏观物理仿真 设计)
# =====================================================================
def get_raw_stitched_matrix(h5_path, num_blocks=10, target_width_m=None, dx_mm=100.0, fill_holes=True):
    """
    极简原始数据拼接函数：
    不进行平面拟合、S型平滑、极值限幅等破坏原始高程尺度的操作。
    完全保留路面的真实宏观排水坡度、车辙深度和微观纹理。

    :param h5_path: H5 文件路径
    :param num_blocks: 读取的块数
    :param target_width_m: 目标裁剪宽度(米)，如果为None则保留全幅
    :param dx_mm: 传感器的物理采样步长 (横向)
    :param fill_holes: 是否用中位数填补传感器返回的 0 值死区
    :return: 保持原始 Z 轴单位的 2D numpy 矩阵
    """
    with h5py.File(h5_path, 'r') as h5f:
        group = h5f['road_segments']
        names = sorted(list(group.keys()))[:num_blocks]

        blocks = []
        for name in names:
            blocks.append(group[name][:])

        if not blocks:
            return None

        # 1. 极其暴力的物理直接拼接，不进行任何融合模糊
        raw_matrix = np.vstack(blocks)

        # 2. 填补传感器未扫到的黑洞 (0值)
        # 如果不填补，水膜算法中的雨水会全部流进这些深度为负无穷的黑洞中引发死循环
        if fill_holes:
            valid_mask = raw_matrix != 0
            if np.any(valid_mask):
                median_val = np.median(raw_matrix[valid_mask])
                raw_matrix = np.where(raw_matrix == 0, median_val, raw_matrix)

        # 3. 物理居中裁剪 (例如严格控制在 3.75m 单车道宽度内)
        if target_width_m is not None:
            total_pixels = raw_matrix.shape[1]
            target_pixels = int(target_width_m * 1000 / dx_mm)

            if target_pixels < total_pixels:
                trim_pixels = (total_pixels - target_pixels) // 2
                raw_matrix = raw_matrix[:, trim_pixels: trim_pixels + target_pixels]
            elif target_pixels > total_pixels:
                print(f"⚠️ 警告：原始数据宽度 ({total_pixels}列) 小于目标宽度 ({target_width_m}m)，跳过裁剪。")

        return raw_matrix


# =====================================================================
# 🛠️ 工具 2: 高精度平滑降噪拼接 (专为 功率谱PSD、粗糙度RMS、频域分析 设计)
# =====================================================================
def get_perfect_stitched_matrix(h5_path, num_blocks=10, correction_angle=0, overlap_rows=8, crop_edge_ratio=0,
                                max_std=15.0):
    """
    1. 鲁棒抗噪平面拟合 (去趋势，拉平路面，免疫飞点干扰)
    2. S 型曲线行重叠平滑融合
    3. 孤岛飞点软压制与严格限幅
    """
    with h5py.File(h5_path, 'r') as h5f:
        group = h5f['road_segments']
        names = sorted(list(group.keys()))[:num_blocks]

        zz = None

        # S 型平滑权重
        x = np.linspace(0, 1, overlap_rows)
        weights_1d = (1 + np.cos(np.pi * x)) / 2.0
        weights = weights_1d.reshape(-1, 1)

        for name in names:
            data = group[name][:]
            valid_mask = data != 0

            if np.sum(valid_mask) < 100:
                continue

            # ========================================================
            # 鲁棒二维平面拟合 (Robust 2D Detrending) - 拉平宏观坡度
            # ========================================================
            Y, X = np.indices(data.shape)
            x_val = X[valid_mask]
            y_val = Y[valid_mask]
            z_val = data[valid_mask]

            # 计算前先剔除最高和最低的 5% 噪点
            p5, p95 = np.percentile(z_val, [5, 95])
            safe_mask = (z_val > p5) & (z_val < p95)

            # 仅使用绝对安全的核心数据来拟合斜面
            x_safe = x_val[safe_mask]
            y_safe = y_val[safe_mask]
            z_safe = z_val[safe_mask]

            if len(z_safe) > 10:
                A = np.column_stack((x_safe, y_safe, np.ones_like(x_safe)))
                coeffs, _, _, _ = np.linalg.lstsq(A, z_safe, rcond=None)
                a, b, c = coeffs

                # 重构基准面并扣除
                plane = a * X + b * Y + c
                data[valid_mask] = data[valid_mask] - plane[valid_mask]

            # 异常路段免疫
            current_std = np.std(data[valid_mask])
            if current_std > max_std:
                print(f"⚠️ 触发保护机制: 跳过坏死路段 [{name}] (标准差={current_std:.2f})")
                continue

            # ---------------- 旋转与无缝拼接 ----------------
            if correction_angle != 0:
                data = rotate(data, angle=correction_angle, reshape=False, mode='nearest')
                trim_y = max(1, int(data.shape[1] * np.sin(np.deg2rad(abs(correction_angle))))) + 2
            else:
                trim_y = 1

            data = data[trim_y:-trim_y, 1:-1]

            if zz is None:
                zz = data
                continue

            width = zz.shape[1]
            mid_col, safe_w = width // 2, max(1, width // 4)
            start_c, end_c = max(0, mid_col - safe_w), min(width, mid_col + safe_w)

            edge_zz = zz[-overlap_rows:, start_c:end_c]
            edge_data = data[:overlap_rows, start_c:end_c]

            valid_zz = edge_zz[edge_zz != 0]
            valid_data = edge_data[edge_data != 0]

            if len(valid_zz) > 0 and len(valid_data) > 0:
                gaocha = np.median(valid_zz) - np.median(valid_data)
            else:
                gaocha = np.mean(zz[-overlap_rows:, :]) - np.mean(data[:overlap_rows, :])

            data += gaocha

            overlap_A = zz[-overlap_rows:, :]
            overlap_B = data[:overlap_rows, :]

            zz[-overlap_rows:, :] = overlap_A * weights + overlap_B * (1.0 - weights)
            zz = np.vstack((zz, data[overlap_rows:, :]))

    # ========================================================
    # 激进车流区提取与微观纹理保真限幅
    # ========================================================
    # 1. 物理裁切核心车流区
    if crop_edge_ratio > 0:
        trim_cols = int(zz.shape[1] * crop_edge_ratio)
        if trim_cols > 0:
            zz = zz[:, trim_cols:-trim_cols]

    # 2. 将数据整体基准强制居中到绝对 0 点
    valid_points = zz[zz != 0]
    if len(valid_points) > 0:
        zz[zz != 0] = zz[zz != 0] - np.median(valid_points)

        z_min, z_max = np.percentile(valid_points, [0.1, 99.9])
        zz = np.clip(zz, z_min, z_max)

    return zz


# =====================================================================
# 🛠️ 工具 3: 辅助轮迹带定位绘图器
# =====================================================================
def explore_transverse_wear_profile(master_matrix, dx_mm=100):
    """
    横向微观粗糙度扫描器：用于可视化定位轮迹带的像素位置。
    """
    print("⏳ 正在计算横向微观粗糙度分布，寻找轮迹带波谷...")

    # 1. 沿纵向 (axis=0) 去趋势，消除宏观起伏误差
    detrended_matrix = signal.detrend(master_matrix, axis=0)

    # 2. 计算每一列（纵向剖面）的标准差（即均方根粗糙度 RMS），并转为微米
    rms_profile = np.std(detrended_matrix, axis=0) * 1000

    # 3. 横向滑动平滑：消除单列飞点噪点的干扰 (例如平滑 20mm 的物理宽度)
    window_physical_mm = 20
    window_pixels = max(1, int(window_physical_mm / dx_mm))

    # 使用卷积进行移动平均平滑
    smoothed_rms = np.convolve(rms_profile, np.ones(window_pixels) / window_pixels, mode='valid')

    # 计算 X 轴对应的像素索引和物理坐标 (横向宽度)
    pixel_indices = np.arange(len(smoothed_rms)) + (window_pixels // 2)
    x_width_m = pixel_indices * (dx_mm / 1000.0)

    # 4. 绘制横向粗糙度分布大图
    fig, ax1 = plt.subplots(figsize=(12, 6), facecolor='white')

    # 绘制粗糙度曲线
    ax1.plot(pixel_indices, smoothed_rms, color='#E63946', linewidth=2.5, label='横向微观粗糙度 (μm)')

    # 绘制辅助均值线
    mean_rms = np.mean(smoothed_rms)
    ax1.axhline(mean_rms, color='#1D3557', linestyle='--', alpha=0.6, label='全局平均基准线')

    ax1.set_title('路面横向切片探查：通过粗糙度波谷定位轮迹带', fontsize=16, fontweight='bold')
    ax1.set_xlabel('横向像素索引 (列)', fontsize=13)
    ax1.set_ylabel('纵向微观粗糙度 RMS (μm)', fontsize=13, color='#E63946')
    ax1.tick_params(axis='y', labelcolor='#E63946')
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(ax1.get_xticks())
    ax2.set_xticklabels([f"{x * (dx_mm / 1000.0):.2f}" for x in ax1.get_xticks()])
    ax2.set_xlabel('横向物理宽度 (米)', fontsize=11, color='gray')

    ax1.text(0.5, 0.05,
             "▼ 寻找图中的【低谷区域】：记下对应的 X 轴像素索引，填入切片代码的 w_start 和 w_end\n"
             "▲ 寻找图中的【平稳高地】：记下对应的 X 轴像素索引，填入切片代码的 nw_start 和 nw_end",
             transform=ax1.transAxes, horizontalalignment='center',
             bbox=dict(facecolor='#F1FAEE', alpha=0.9, edgecolor='#A8DADC', boxstyle='round,pad=0.8'))

    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.88))
    plt.tight_layout()
    plt.show()

    return smoothed_rms


# ==========================================
# 🚀 测试运行模块
# ==========================================
if __name__ == "__main__":
    h5_file = 'data/PavementDatabase.h5'

    # 测试新加入的 Raw 函数
    print("⏳ 测试加载物理保真(RAW)矩阵...")
    raw_matrix = get_raw_stitched_matrix(h5_file, num_blocks=20, target_width_m=3.75, dx_mm=100.0)

    if raw_matrix is not None:
        print(f"✅ 获取 RAW 矩阵成功！形状: {raw_matrix.shape}，高度跨度: {np.max(raw_matrix) - np.min(raw_matrix):.2f}")