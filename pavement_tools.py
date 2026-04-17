import numpy as np
import h5py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import rotate


def get_perfect_stitched_matrix(h5_path, num_blocks=10, correction_angle=0, overlap_rows=5):
    """
    终极拼接函数：同时包含 X/Y轴(旋转纠偏) 和 Z轴(高度鲁棒对齐)
    """
    with h5py.File(h5_path, 'r') as h5f:
        group = h5f['road_segments']
        names = sorted(list(group.keys()))[:num_blocks]

        # ================= 第一块数据初始化 =================
        first_data = group[names[0]][:]

        # 1. X/Y轴处理：第一块旋转纠偏
        if correction_angle != 0:
            first_data = rotate(first_data, angle=correction_angle, reshape=False, mode='nearest')

        zz = first_data[1:-1, 1:-1]

        # ================= 循环拼接后续块 =================
        for name in names[1:]:
            data = group[name][:]

            # 1. X/Y轴处理：对当前块进行同样的旋转纠偏
            if correction_angle != 0:
                data = rotate(data, angle=correction_angle, reshape=False, mode='nearest')

            # ========================================================
            # 2. Z轴处理：中轴线安全采样对齐 (彻底修复空切片 Bug！)
            # ========================================================
            width = zz.shape[1]
            mid_col = width // 2

            # 动态计算安全宽度：只取路面宽度的正中间 50% 的区域
            safe_width = max(1, width // 4)

            # 边界保护：确保起始和结束索引绝对不会出现负数或越界
            start_col = max(0, mid_col - safe_width)
            end_col = min(width, mid_col + safe_width)

            # 提取上一个矩阵尾部中心，和当前矩阵头部中心的纯净数据
            edge_zz_safe = zz[-overlap_rows:, start_col:end_col]
            edge_data_safe = data[:overlap_rows, start_col:end_col]

            # 剔除无效的背景像素 0
            valid_zz = edge_zz_safe[edge_zz_safe != 0]
            valid_data = edge_data_safe[edge_data_safe != 0]

            # 安全计算高差（双重保护机制：确保数组不为空才求中位数）
            if len(valid_zz) > 0 and len(valid_data) > 0:
                gaocha = np.median(valid_zz) - np.median(valid_data)
            else:
                # 如果中间部分全是0(极端异常)，则退回使用整个截面的均值兜底
                gaocha = np.mean(zz[-overlap_rows:, :]) - np.mean(data[:overlap_rows, :])


            # 1. 加上高差补偿，获得修正后的当前块
            data_fixed = data[1:-1, 1:-1] + gaocha

            # ========================================================
            # 3. 边缘平滑渐变融合 (Alpha Blending 彻底消除接缝突变)
            # ========================================================

            # 提取上一块路面的“物理重叠区尾部”和当前路面的“物理重叠区头部”
            overlap_A = zz[-overlap_rows:, :]  # 上一块的最后几行
            overlap_B = data_fixed[:overlap_rows, :]  # 当前块的最前几行

            # 创建线性渐变权重 (一维数组：从 1.0 平滑过渡到 0.0)
            # 例如 overlap_rows=5 时，权重为 [1.0, 0.75, 0.5, 0.25, 0.0]
            weights_1d = np.linspace(1.0, 0.0, overlap_rows)

            # 将一维权重变成二维列向量，以便与矩阵相乘
            weights_matrix = weights_1d.reshape(-1, 1)

            # 核心融合公式：上一块的权重越来越小，下一块的权重越来越大
            blended_overlap = overlap_A * weights_matrix + overlap_B * (1.0 - weights_matrix)

            # 将完美融合后的区域“覆盖”回上一块矩阵的尾部
            zz[-overlap_rows:, :] = blended_overlap

            # 将当前块**剔除重叠区后**的剩余部分，拼接到大矩阵末尾
            # 注意：不再直接全量 vstack，否则会导致路面被拉长！
            zz = np.vstack((zz, data_fixed[overlap_rows:, :]))
    return zz


# ==========================================
# 🚀 运行对比测试
# ==========================================
if __name__ == "__main__":
    h5_file = 'data/PavementDatabase.h5'

    # 填入你观察到的单块偏移角度 (逆时针填负数，顺时针填正数)
    angle_to_fix = 1.35

    print("⏳ 正在进行高精度 3D 缝合...")

    # ❌ 未处理 (带累积偏差，且只用均值对齐高度)
    matrix_raw = get_perfect_stitched_matrix(h5_file, num_blocks=15, correction_angle=0)

    # ✅ 完美处理 (旋转纠偏 + 中位数高度对齐)
    matrix_fixed = get_perfect_stitched_matrix(h5_file, num_blocks=15, correction_angle=angle_to_fix)

    # 绘制双子图导出 HTML
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=('❌ 未处理直接拼接', f'✅ 完美对齐 (偏角 {angle_to_fix}° + 中位数高差补偿)'),
        horizontal_spacing=0.05
    )

    fig.add_trace(go.Surface(z=matrix_raw, colorscale='Viridis', showscale=False), row=1, col=1)
    fig.add_trace(go.Surface(z=matrix_fixed, colorscale='Viridis', showscale=True), row=1, col=2)

    fig.update_layout(
        title=f"路面 3D 拼接：旋转纠偏与高度对齐验证",
        scene=dict(aspectmode='data'),
        scene2=dict(aspectmode='data'),
        height=800, width=1500
    )

    output_html = "完美拼接对比结果.html"
    fig.write_html(output_html)
    print(f"🎉 成功！请打开查看：{output_html}")