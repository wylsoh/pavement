import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure, binary_dilation
import plotly.graph_objects as go

# 导入上一轮我们为你重构的保真读取工具
from pavement_tools import get_raw_stitched_matrix

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 核心算法：全域水膜迭代计算
# ==========================================
def simulate_water_film_with_low_wall(data0, shuimo_h, wall_margin):
    print(f"⏳ [全域仿真] 正在进行全幅水膜物理模拟 (降雨量: {shuimo_h}, 边缘水墙裕量: {wall_margin})...")
    m, n = data0.shape

    # 动态计算水墙的绝对高程 (比路面最高点还高 wall_margin)
    wall_height = np.max(data0) + wall_margin
    qy = np.pad(data0, pad_width=1, mode='constant', constant_values=wall_height)

    V = shuimo_h * m * n
    structure = generate_binary_structure(2, 1)
    iteration = 0

    while V > 1e-6:
        iteration += 1

        # 4邻域探测
        top = qy[:-2, 1:-1]
        bottom = qy[2:, 1:-1]
        left = qy[1:-1, :-2]
        right = qy[1:-1, 2:]
        center = qy[1:-1, 1:-1]

        is_min_center = (center <= top) & (center <= bottom) & (center <= left) & (center <= right)
        is_min = np.pad(is_min_center, pad_width=1, mode='constant', constant_values=False)
        min_indices = np.argwhere(is_min)

        if len(min_indices) == 0:
            break

        # 均分当前剩余水量
        h_dist = V / len(min_indices)
        for r, c in min_indices:
            qy[r, c] += h_dist

        labeled_array, num_features = label(is_min, structure=structure)
        v_excess = np.zeros_like(qy)

        for region_idx in range(1, num_features + 1):
            region_mask = (labeled_array == region_idx)
            dilated = binary_dilation(region_mask, structure=structure)
            boundary_mask = dilated ^ region_mask

            if not np.any(boundary_mask):
                continue

            ljmin = np.min(qy[boundary_mask])
            region_pixels = np.argwhere(region_mask)
            r0, c0 = region_pixels[0]

            if qy[r0, c0] > ljmin:
                for r, c in region_pixels:
                    excess_water = qy[r, c] - ljmin

                    if ljmin < wall_height - 1e-5:
                        v_excess[r, c] = excess_water

                    # 水位被削平至临界点
                    qy[r, c] = ljmin

        V = np.sum(v_excess)

        if iteration > 2000:
            print("⚠️ 达到最大安全迭代次数，水可能已完全漫过水墙，强制退出。")
            break

    print(f"✅ 全域模拟完成，共分配 {iteration} 轮。")
    water_surface = qy[1:-1, 1:-1]

    water_depth = water_surface - data0
    water_depth[water_depth < 1e-4] = 0

    return water_surface, water_depth

# ==========================================
# 2. 3D 水膜可视化展示 (仅展示裁剪区域)
# ==========================================
def render_3d_water_film(original_matrix, water_surface, dx_mm, output_html):
    print(f"⏳ 正在生成中心车道 3D 渲染图...")
    z_original = original_matrix
    z_water = water_surface

    ds_step_x = max(1, z_original.shape[1] // 500)
    ds_step_y = max(1, z_original.shape[0] // 500)

    z_orig_ds = z_original[::ds_step_y, ::ds_step_x]
    z_wat_ds = z_water[::ds_step_y, ::ds_step_x]

    x_m = np.arange(z_orig_ds.shape[1]) * (dx_mm * ds_step_x / 1000.0)
    y_m = np.arange(z_orig_ds.shape[0]) * (dx_mm * ds_step_y / 1000.0)

    fig = go.Figure()

    fig.add_trace(go.Surface(z=z_orig_ds, x=x_m, y=y_m, colorscale='Greys', name='原始路表', showscale=False))

    water_only = np.where(z_wat_ds > z_orig_ds + 1e-4, z_wat_ds, np.nan)
    fig.add_trace(go.Surface(z=water_only, x=x_m, y=y_m, colorscale='Blues', opacity=0.7, name='积水分布',
                             colorbar=dict(title="水深")))

    fig.update_layout(
        title=f"路表降雨水膜三维分布 (基于全幅仿真，输出裁剪至单车道)",
        scene=dict(
            xaxis_title='横向宽度 (m)',
            yaxis_title='纵向长度 (m)',
            zaxis_title='高程起伏',
            aspectmode='manual',
            aspectratio=dict(x=1, y=2.5, z=0.5)
        ),
        height=800, width=1200
    )

    fig.write_html(output_html)
    print(f"✅ 3D渲染完成！已保存至: {output_html}")


# ==========================================
# 3. 横截面水膜积水深度剖视图
# ==========================================
def plot_water_cross_section(original_matrix, water_surface, dx_mm, output_png, row_idx=50, actual_width_m=3.75):
    profile_orig = original_matrix[row_idx, :]
    profile_water = water_surface[row_idx, :]

    x_m = np.arange(len(profile_orig)) * (dx_mm / 1000.0)

    plt.figure(figsize=(10, 5), facecolor='white')
    plt.fill_between(x_m, profile_orig, profile_water, color='#00a8ff', alpha=0.6, label='积水区域 (水膜)')
    plt.plot(x_m, profile_orig, color='#2f3640', linewidth=1.5, label=f'路面高程 ({actual_width_m}m)')
    plt.plot(x_m, profile_water, color='#0097e6', linewidth=1, linestyle='--', label='最终水面界线')

    plt.title(f'中心车道横截面水膜分布 (纵向位置: {row_idx * dx_mm / 1000.0}m)', fontsize=15, fontweight='bold')
    plt.xlabel('横向物理宽度 (米)', fontsize=12)
    plt.ylabel('原始高程 Z', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=11)
    plt.tight_layout()

    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 🚀 程序执行入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全域仿真+局部出图 水膜测试脚本")
    parser.add_argument('--input_file', type=str, default='data/PavementDatabase.h5')
    parser.add_argument('--out_dir', type=str, default='./output_full_to_crop')

    parser.add_argument('--rainfall', type=float, default=0.003, help='平均降雨量')
    parser.add_argument('--wall_margin', type=float, default=2, help='全域边缘水墙高度裕量')
    parser.add_argument('--dx_mm', type=float, default=100.0, help='横向步长')

    # 核心视角控制参数
    parser.add_argument('--target_width_m', type=float, default=3.75, help='结果最终裁剪的展示宽度(m)')
    parser.add_argument('--length_m', type=float, default=5.0, help='纵向测试长度(m)')
    parser.add_argument('--num_blocks', type=int, default=5, help='读取块数')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    param_tag = f"SimFull_Out{args.target_width_m}m_rain{args.rainfall}"

    # --- 第一步：读取全幅真实数据 (不裁剪) ---
    print("⏳ [1/4] 正在加载全幅物理高程矩阵...")
    matrix_full = get_raw_stitched_matrix(
        h5_path=args.input_file,
        num_blocks=args.num_blocks,
        target_width_m=None,  # 设定为 None 强制读取全幅
        dx_mm=args.dx_mm
    )

    if matrix_full is not None:
        # 纵向长度截断以节省计算时间
        test_rows = int(args.length_m * 1000 / args.dx_mm)
        matrix_full = matrix_full[:test_rows, :]

        # --- 第二步：在全幅矩阵上进行物理水膜仿真 ---
        print(f"⏳ [2/4] 在全幅宽度 ({matrix_full.shape[1]} 列) 上运行流体力学仿真...")
        water_surf_full, water_depth_full = simulate_water_film_with_low_wall(
            matrix_full,
            shuimo_h=args.rainfall,
            wall_margin=args.wall_margin
        )

        # --- 第三步：后处理 - 结果居中裁剪 ---
        print(f"⏳ [3/4] 仿真完毕。正在将结果裁剪至目标视野 {args.target_width_m}m ...")
        total_cols = matrix_full.shape[1]
        target_cols = int(args.target_width_m * 1000 / args.dx_mm)

        if target_cols < total_cols:
            trim_cols = (total_cols - target_cols) // 2
            # 同步裁剪底图、水面、水深三个核心矩阵
            matrix_crop = matrix_full[:, trim_cols: trim_cols + target_cols]
            water_surf_crop = water_surf_full[:, trim_cols: trim_cols + target_cols]
            water_depth_crop = water_depth_full[:, trim_cols: trim_cols + target_cols]
            print(f"   ✂️ 成功提取核心车道区域 ({target_cols} 列)。")
        else:
            print(f"   ⚠️ 原数据不足 {args.target_width_m}m，跳过裁剪。")
            matrix_crop, water_surf_crop, water_depth_crop = matrix_full, water_surf_full, water_depth_full

        # --- 第四步：基于裁剪后的核心车道生成报告与图表 ---
        print("\n" + "=" * 45)
        print("💧 核心车道 (3.75m) 水膜灾害评估报告")
        print("   (注：水流边界平衡基于全域物理引擎计算)")
        print("=" * 45)
        print(f"▶ 设定降雨量:     {args.rainfall}")
        print(f"▶ 核心车道高低差: {np.max(matrix_crop) - np.min(matrix_crop):.3f}")
        print(f"▶ 核心车道最大积水: {np.max(water_depth_crop):.3f}")
        print(f"▶ 核心积水覆盖率: {(np.count_nonzero(water_depth_crop) / water_depth_crop.size) * 100:.2f}%")
        print("=" * 45 + "\n")

        print("⏳ [4/4] 渲染局部视野图表...")
        html_output = os.path.join(args.out_dir, f"3D_{param_tag}.html")
        render_3d_water_film(matrix_crop, water_surf_crop, args.dx_mm, html_output)

        for row_idx in [10, int(test_rows / 2), test_rows - 2]:
            png_output = os.path.join(args.out_dir, f"2D_{param_tag}_row{row_idx}.png")
            plot_water_cross_section(
                matrix_crop, water_surf_crop, args.dx_mm,
                png_output, row_idx=row_idx, actual_width_m=args.target_width_m
            )

        print(f"✅ 所有流程执行完毕！结果保存在 {args.out_dir} 目录。")