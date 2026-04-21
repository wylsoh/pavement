# main_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import plotly.graph_objects as go
import os

from pavement_tools import get_perfect_stitched_matrix, explore_transverse_wear_profile

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

outputfile = "./output/"

# ==========================================
# 1. 3D矩阵转1D轮廓的“桥接适配器”
# ==========================================
def extract_1d_profile_from_matrix(matrix, dx_mm=0.1):
    if matrix is None or len(matrix) == 0:
        return None
    z_mm = np.mean(matrix, axis=1)
    x_m = np.arange(len(z_mm)) * (dx_mm / 1000.0)
    return np.column_stack((x_m, z_mm))


# ==========================================
# 2. 功率谱计算核心 (Welch 方法)
# ==========================================
def calc_power_spectrum_welch(data):
    x_m = data[:, 0]
    z_mm = data[:, 1]
    dx_mm = np.mean(np.diff(x_m)) * 1000
    fs = 1.0 / dx_mm

    z_detrend = signal.detrend(z_mm)
    win_len = max(256, len(z_detrend) // 8)
    f, psd = signal.welch(z_detrend, fs=fs, window='hann',
                          nperseg=win_len, noverlap=win_len // 2)

    valid_idx = f > 0
    q = f[valid_idx]
    Cq = psd[valid_idx]
    Cq = np.maximum(Cq, 1e-6)
    return q, Cq


# ==========================================
# 3. 粗糙度 HRMS 积分计算
# ==========================================
def calc_roughness(q, Cq, q1, q2):
    idx = (q >= q1) & (q <= q2)
    q_valid = q[idx]
    Cq_valid = Cq[idx]

    if len(q_valid) == 0: return 0.0
    integrand = q_valid * Cq_valid
    hrms_sq = np.trapezoid(integrand, q_valid)
    return np.sqrt(abs(hrms_sq))


# ==========================================
# 4. 主控分析与绘图
# ==========================================
def analyze_pavement(wheel_data, nonwheel_data):
    try:
        q_wheel, Cq_wheel = calc_power_spectrum_welch(wheel_data)
        q_nonwheel, Cq_nonwheel = calc_power_spectrum_welch(nonwheel_data)

        q1 = max(np.min(q_wheel), np.min(q_nonwheel))
        q2 = min(np.max(q_wheel), np.max(q_nonwheel))

        hrms_w_um = calc_roughness(q_wheel, Cq_wheel, q1, q2) * 1000
        hrms_nw_um = calc_roughness(q_nonwheel, Cq_nonwheel, q1, q2) * 1000
        delta_hrms_um = hrms_nw_um - hrms_w_um

        print("\n" + "=" * 40)
        print("🏆 智慧路面检测分析报告 (自适应 3.75m 重构)")
        print("=" * 40)
        print(f"▶ 轮迹带粗糙度 (HRMS):   {hrms_w_um:.2f} μm")
        print(f"▶ 非轮迹带粗糙度 (HRMS): {hrms_nw_um:.2f} μm")
        print(f"▶ 磨光损失量 (ΔHRMS):    {delta_hrms_um:.2f} μm")
        print("=" * 40 + "\n")

        plt.figure(figsize=(10, 6), facecolor='white')
        plt.loglog(q_wheel, Cq_wheel, color='#0080FF', linewidth=1.5, linestyle='-', label='轮迹带 (磨损区)')
        plt.loglog(q_nonwheel, Cq_nonwheel, color='#FF9900', linewidth=1.5, linestyle='--', label='非轮迹带 (原始区)')

        plt.xlim(q1, q2)
        plt.ylim(min(np.min(Cq_wheel), np.min(Cq_nonwheel)) * 0.8,
                 max(np.max(Cq_wheel), np.max(Cq_nonwheel)) * 2)

        plt.xlabel('空间频率 q (1/mm)', fontsize=12)
        plt.ylabel('功率谱密度 C(q) (mm$^2$)', fontsize=12)
        plt.title('路表微观纹理特征衰减分析 (PSD)', fontsize=15, fontweight='bold')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend(loc='upper right', fontsize=11)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"❌ 计算过程中发生错误：{e}")


# ==========================================
# 5. 3D 横截面与纹理局部展示
# ==========================================
def render_3d_segment(matrix, dx_mm, length_m=10.0):
    """
    渲染前 length_m 米的 3D 路面模型。
    自动进行智能降采样，防止数万列像素导致浏览器崩溃。
    """
    print(f"\n⏳ 正在生成前 {length_m}m 的 3D 局部纹理横截面...")
    rows_to_show = int(length_m * 1000 / dx_mm)
    rows_to_show = min(rows_to_show, matrix.shape[0])

    segment = matrix[:rows_to_show, :]

    # 动态降采样：保证在 X/Y 轴最多只渲染 800 个点，确保丝滑交互
    ds_step_x = max(1, segment.shape[1] // 800)
    ds_step_y = max(1, segment.shape[0] // 800)

    segment_ds = segment[::ds_step_y, ::ds_step_x]

    # 构建物理坐标 (米)
    x_m = np.arange(segment_ds.shape[1]) * (dx_mm * ds_step_x / 1000.0)
    y_m = np.arange(segment_ds.shape[0]) * (dx_mm * ds_step_y / 1000.0)

    fig = go.Figure(data=[go.Surface(
        z=segment_ds, x=x_m, y=y_m,
        colorscale='Viridis',
        colorbar=dict(title="高程起伏(mm)")
    )])

    fig.update_layout(
        title=f"标准车道 (3.75m) 局部 3D 宏观形变展示 (长度: {length_m}m)",
        scene=dict(
            xaxis_title='横向宽度 (m)',
            yaxis_title='纵向长度 (m)',
            zaxis_title='高程起伏 (mm)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=2.5, z=0.2)  # 适配宏观马路的视觉比例
        ),
        height=800, width=1200
    )

    output_html = outputfile + "局部3D路表宏观形变展示.html"
    fig.write_html(output_html)
    print(f"✅ 3D渲染完成！请在浏览器中打开: {output_html}\n")


# ==========================================
# 6. [新增] 2D 横截面剖视图 (直观展示车辙与切片位置)
# ==========================================
def plot_transverse_cross_section(matrix, dx_mm, w_start, w_end, nw_start, nw_end):
    print("⏳ 正在生成路面二维横截面剖视图...")
    # 沿纵向求均值，得到一条极具代表性的横向车辙剖面线
    mean_profile = np.mean(matrix, axis=0)
    # 将最高点归零，方便直观查看下凹深度
    mean_profile = mean_profile - np.max(mean_profile)

    x_m = np.arange(len(mean_profile)) * (dx_mm / 1000.0)

    plt.figure(figsize=(10, 5), facecolor='white')
    plt.plot(x_m, mean_profile, color='#2A9D8F', linewidth=2.5, label='平均横向高程')

    # 边界安全保护
    nw_end_idx = min(nw_end - 1, len(x_m) - 1)
    w_end_idx = min(w_end - 1, len(x_m) - 1)

    # 使用半透明色块标注出我们在代码中提取的两个特征区域
    plt.axvspan(x_m[nw_start], x_m[nw_end_idx], color='#FF9900', alpha=0.2, label='非轮迹带提取区 (中央隆起)')
    plt.axvspan(x_m[w_start], x_m[w_end_idx], color='#0080FF', alpha=0.2, label='轮迹带提取区 (车辙凹陷)')

    plt.title('标准车道 (3.75m) 宏观横截面与特征提取区对比', fontsize=15, fontweight='bold')
    plt.xlabel('横向物理宽度 (米)', fontsize=12)
    plt.ylabel('相对下凹深度 (mm)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower center', fontsize=11)
    plt.tight_layout()
    plt.show()


# ==========================================
# 🚀 程序执行入口 (物理裁切 + 3D渲染 + 2D横截面 + ROI提取)
# ==========================================
if __name__ == "__main__":
    h5_file = 'data/PavementDatabase.h5'

    # --- 核心物理参数 ---
    hardware_dx_mm = 100  # 采样步长 0.1m (即 100mm)
    original_width_m = 4.04  # 原始采集宽度
    target_width_m = 3.75  # 目标行车道标准宽度

    print(f"⏳ [1/6] 正在执行全幅宏观拼接 (关闭比例裁边以保留原始尺寸)...")
    master_matrix = get_perfect_stitched_matrix(h5_file, num_blocks=10, correction_angle=1.35, crop_edge_ratio=0.0)

    if master_matrix is not None:
        total_pixels = master_matrix.shape[1]
        print(f"✅ 拼接完成！原始矩阵总列数: {total_pixels}")

        # --- 物理裁剪核心逻辑 ---
        print(f"⏳ [2/6] 正在裁剪路缘部分 (从 {original_width_m}m 缩至 {target_width_m}m)...")
        target_pixels = int(target_width_m * 1000 / hardware_dx_mm)

        if target_pixels < total_pixels:
            trim_pixels = (total_pixels - target_pixels) // 2
            cropped_matrix = master_matrix[:, trim_pixels: trim_pixels + target_pixels]
            print(f"   - 切除左侧: {trim_pixels} 列 | 切除右侧: {trim_pixels} 列")
            print(f"   - 保留核心: {target_pixels} 列 (即标准 3.75 米宽度)")
        else:
            cropped_matrix = master_matrix
            print("   - 警告：原始数据不足 3.75m，跳过裁剪。")

        # --- 可视化横截面 3D 图 ---
        print("⏳ [3/6] 渲染宏观 3D 图像...")
        # 步长 0.1m，截取前 10米 (100行) 进行 3D 展现
        render_3d_segment(cropped_matrix, hardware_dx_mm, length_m=10.0)

        # --- ROI 精准切片计算 ---
        print("⏳ [4/6] 基于 3.75m 规范宽度，计算轮迹带与非轮迹带坐标...")
        current_width_pixels = cropped_matrix.shape[1]

        # 非轮迹带：取正中心 40% ~ 60%
        nw_start = int(current_width_pixels * 0.40)
        nw_end = int(current_width_pixels * 0.60)
        matrix_nonwheel = cropped_matrix[:, nw_start:nw_end]

        # 轮迹带：取右侧常压区域 70% ~ 90%
        w_start = int(current_width_pixels * 0.70)
        w_end = int(current_width_pixels * 0.90)
        matrix_wheel = cropped_matrix[:, w_start:w_end]

        real_wheel_data = extract_1d_profile_from_matrix(matrix_wheel, hardware_dx_mm)
        real_nonwheel_data = extract_1d_profile_from_matrix(matrix_nonwheel, hardware_dx_mm)

        print(f"   - 轮迹带坐标:   [{w_start}:{w_end}] 列")
        print(f"   - 非轮迹带坐标: [{nw_start}:{nw_end}] 列")

        # --- 可视化 2D 横截面图 ---
        print("⏳ [5/6] 绘制并展示 2D 横截面与特征区域对比图...")
        plot_transverse_cross_section(cropped_matrix, hardware_dx_mm, w_start, w_end, nw_start, nw_end)

        # --- 功率谱与等效荷载分析 ---
        print("⏳ [6/6] 开始执行功率谱积分与 ΔHRMS 磨光分析...")
        analyze_pavement(wheel_data=real_wheel_data, nonwheel_data=real_nonwheel_data)
    else:
        print("❌ 数据处理失败，请检查数据源。")