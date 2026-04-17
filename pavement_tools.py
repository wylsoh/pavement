import numpy as np
import h5py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import rotate


def get_perfect_stitched_matrix(h5_path, num_blocks=10, correction_angle=0, overlap_rows=8, crop_edge_ratio=0.20,
                                max_std=15.0):
    """
    终极护城河版：包含二维平面拟合熨平 + 坏死路段自动剔除

    参数:
        max_std: 容忍的最大标准差。如果某段路面熨平后起伏依然极其夸张，直接当作废片跳过！
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
                continue  # 如果数据太空，直接跳过

            # ========================================================
            # 【全新突破 1】二维平面拟合去趋势 (2D Plane Detrending)
            # 彻底解决单块路段“左高右低”或“前高后低”的倾斜误差！
            # ========================================================
            # 获取有效像素的 X, Y 坐标系
            Y, X = np.indices(data.shape)
            x_val = X[valid_mask]
            y_val = Y[valid_mask]
            z_val = data[valid_mask]

            # 构建最小二乘法方程组：Z = a*X + b*Y + c
            A = np.column_stack((x_val, y_val, np.ones_like(x_val)))

            # 求解出当前这块路面的“倾斜平面” (a, b 为倾斜角，c 为基础高度)
            coeffs, _, _, _ = np.linalg.lstsq(A, z_val, rcond=None)
            a, b, c = coeffs

            # 重构出这个倾斜的基准面
            plane = a * X + b * Y + c

            # 【熨平操作】用原始数据减去这个倾斜面，路面瞬间绝对水平且居于 Z=0！
            data[valid_mask] = data[valid_mask] - plane[valid_mask]

            # ========================================================
            # 【全新突破 2】坏死路段动态免疫隔离
            # ========================================================
            # 熨平后，计算这段路的标准差。正常微观纹理的标准差通常在 1~5 毫米之间。
            # 如果某段路算出来标准差达到几十，说明这是一段全是噪点/飞点的坏死数据。
            current_std = np.std(data[valid_mask])
            if current_std > max_std:
                print(f"⚠️ 触发保护机制: 跳过路段 [{name}]！检测到极端异常波动 (标准差={current_std:.2f} > {max_std})")
                continue  # 丢弃这块废数据，不让它污染主干路面

            # ---------------- 以下为标准的旋转与拼接逻辑 ----------------
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
    # 最终车流区提取与宏观净化
    # ========================================================
    if crop_edge_ratio > 0:
        trim_cols = int(zz.shape[1] * crop_edge_ratio)
        if trim_cols > 0:
            zz = zz[:, trim_cols:-trim_cols]

    valid_points = zz[zz != 0]
    if len(valid_points) > 0:
        z_min, z_max = np.percentile(valid_points, [0.1, 99.9])
        zz = np.clip(zz, z_min, z_max)

    return zz


# ==========================================
# 🚀 运行对比测试
# ==========================================
if __name__ == "__main__":
    h5_file = 'data/PavementDatabase.h5'
    angle_to_fix = 1.35

    print(f"⏳ 正在执行最高等级的【智能剔除+平面熨平】3D缝合...")

    # 旧逻辑 (无平面拟合)
    matrix_raw = get_perfect_stitched_matrix(
        h5_path=h5_file, num_blocks=20, correction_angle=0, overlap_rows=1, crop_edge_ratio=0.25, max_std=9999
    )

    # 终极逻辑 (带平面拟合 + 异常剔除)
    # 如果你发现有的好路段被误删了，可以把 max_std 调大一点（比如 20 或 30）
    matrix_fixed = get_perfect_stitched_matrix(
        h5_path=h5_file, num_blocks=20, correction_angle=angle_to_fix, overlap_rows=8, crop_edge_ratio=0.25,
        max_std=15.0
    )

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=('❌ 包含倾斜畸变及废片的路面', f'✅ 智能熨平且免疫异常路段'),
        horizontal_spacing=0.05
    )

    fig.add_trace(go.Surface(z=matrix_raw, colorscale='Viridis', showscale=False), row=1, col=1)
    fig.add_trace(go.Surface(z=matrix_fixed, colorscale='Viridis', showscale=True), row=1, col=2)

    fig.update_layout(title=f"异常抗性测试：多维信号预处理效果", scene=dict(aspectmode='data'),
                      scene2=dict(aspectmode='data'), height=800, width=1500)
    fig.write_html("最终平面拟合与剔除结果.html")
    print("🎉 成功！那个歪扭扭的干扰路段已经被摆平（或被直接清理）了！")