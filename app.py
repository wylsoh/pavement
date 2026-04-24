import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd  # ⚠️ 新增引入 pandas 用于生成表格
import time
import os
import h5py
from datetime import datetime
from scipy.ndimage import label, generate_binary_structure, binary_dilation
from scipy.ndimage import zoom


plt.rcParams['font.sans-serif'] = ['simsun']
plt.rcParams['axes.unicode_minus'] = False
np.set_printoptions(suppress=True)

# ==========================================
# UI 页面与状态配置
# ==========================================
st.set_page_config(page_title="路面水膜灾害分析系统", layout="wide", page_icon="🌧️")

# 初始化全局状态变量
if 'road_loaded' not in st.session_state:
    st.session_state.road_loaded = False
if 'matrix_full' not in st.session_state:
    st.session_state.matrix_full = None
if 'matrix_crop' not in st.session_state:
    st.session_state.matrix_crop = None
if 'trim_cols' not in st.session_state:
    st.session_state.trim_cols = 0
if 'target_cols' not in st.session_state:
    st.session_state.target_cols = 0
if 'coverage_history' not in st.session_state:
    st.session_state.coverage_history = {}

st.title("🌧️ 路表降雨水膜动态物理推演系统")
st.markdown("支持点云高程去零裁剪、任意路段指定检视、以及分步动态降雨堆积演示。")

from scipy.ndimage import median_filter


# ==========================================
# 核心数据读取与预处理函数 (带异常路段免疫与无缝拼接)
# ==========================================
def load_and_preprocess_h5(h5_path, start_segment, num_blocks, max_std=15.0, overlap_rows=8):
    with h5py.File(h5_path, 'r') as h5f:
        group = h5f['road_segments']
        all_names = sorted(list(group.keys()))
        start_idx = all_names.index(start_segment)

        zz = None
        blocks_loaded = 0
        curr_idx = start_idx

        # 预先计算 S 型平滑权重 (用于接缝处，防止产生阶梯挡水大坝)
        x = np.linspace(0, 1, overlap_rows)
        weights_1d = (1 + np.cos(np.pi * x)) / 2.0
        weights = weights_1d.reshape(-1, 1)

        # 动态加载，如果遇到异常路段被跳过，则自动读取下一段补充
        while blocks_loaded < num_blocks and curr_idx < len(all_names):
            name = all_names[curr_idx]
            curr_idx += 1
            data = group[name][:]

            # 1. 裁剪 y=0 的无效扫描盲区
            data = data[1:, :]
            valid_mask = data != 0

            if np.sum(valid_mask) < 100:
                continue

            current_std = np.std(data[valid_mask])
            if current_std > max_std:
                # 在前端 UI 提示用户跳过了哪些坏数据
                st.toast(f"⚠️ 触发保护机制: 自动跳过坏死路段 [{name}] (标准差={current_std:.2f})")
                continue

            # 填补内部 0 值死区
            data_median = np.median(data[valid_mask])
            data = np.where(data == 0, data_median, data)

            # 3. SOR 统计滤波：去除局部的孤立飞点
            ref_matrix = median_filter(data, size=5)
            diff = np.abs(data - ref_matrix)
            mu, sigma = np.mean(diff), np.std(diff)
            outlier_mask = diff > (mu + 3 * sigma)
            data[outlier_mask] = ref_matrix[outlier_mask]

            # 4. 高程对齐与 S 型无缝拼接
            if zz is None:
                zz = data
                blocks_loaded += 1
                continue

            width = zz.shape[1]
            mid_col, safe_w = width // 2, max(1, width // 4)
            start_c, end_c = max(0, mid_col - safe_w), min(width, mid_col + safe_w)

            # 获取重叠区域的数据
            edge_zz = zz[-overlap_rows:, start_c:end_c]
            edge_data = data[:overlap_rows, start_c:end_c]

            # 计算两个路段之间的高程差 (漂移补偿)
            gaocha = np.mean(edge_zz) - np.mean(edge_data)
            data += gaocha

            # 仅对重叠的 8 行进行 S 型融合平滑，不破坏路面主体的宏观坡度
            overlap_A = zz[-overlap_rows:, :]
            overlap_B = data[:overlap_rows, :]
            zz[-overlap_rows:, :] = overlap_A * weights + overlap_B * (1.0 - weights)

            zz = np.vstack((zz, data[overlap_rows:, :]))
            blocks_loaded += 1

        if zz is None:
            return None

        # 5. 全局极端限幅兜底
        z_min, z_max = np.percentile(zz, [0.1, 99.9])
        zz = np.clip(zz, z_min, z_max)

        return zz


# ==========================================
# 辅助渲染函数
# ==========================================
def create_3d_figure(matrix, water_surf=None, water_depth=None, dx_mm=100.0):
    """生成 3D Plotly 图像 (严格还原长宽物理比例)"""

    x_dm = np.arange(matrix.shape[1]) * (dx_mm / 100.0)
    y_dm = np.arange(matrix.shape[0]) * (dx_mm / 100.0)

    # 🌟 修改 1: 取消 / 1000.0，直接保留毫米 (mm) 单位
    z_mm = matrix

    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=z_mm, x=x_dm, y=y_dm,
        colorscale='Portland', name='路表高程', showscale=False,
        contours=dict(
            x=dict(show=True, color='black', width=1, start=x_dm[0], end=x_dm[-1], size=dx_mm / 100.0),
            y=dict(show=True, color='black', width=1, start=y_dm[0], end=y_dm[-1], size=dx_mm / 100.0)
        )
    ))

    if water_surf is not None and water_depth is not None:
        # 🌟 修改 1: 水面高度也直接使用毫米单位
        water_surf_mm = water_surf
        water_only = np.where(water_depth > 1e-4, water_surf_mm, np.nan)

        if not np.all(np.isnan(water_only)):
            fig.add_trace(go.Surface(
                z=water_only, x=x_dm, y=y_dm,
                colorscale=[[0, 'aqua'], [1, 'aqua']], opacity=0.65,
                name='水膜', showscale=False, hoverinfo='skip'
            ))

    # 动态计算真实物理长宽比
    x_physical_length = x_dm[-1] - x_dm[0]
    y_physical_length = y_dm[-1] - y_dm[0]

    true_y_ratio = y_physical_length / x_physical_length if x_physical_length > 0 else 1

    fig.update_layout(
        scene=dict(
            xaxis_title='车道宽度(dm)',
            yaxis_title='路线长度(dm)',
            zaxis_title='路表高程(mm)',  # 🌟 修改 1: 更新轴标签
            aspectmode='manual',
            aspectratio=dict(x=1, y=true_y_ratio, z=0.5),

            # 🌟 修改 2: 将 eye 的绝对数值调小 (原为 x=1.8, y=-1.8, z=1.3)
            # x 和 y 控制水平方位的远近，z 控制俯视高度。你可以继续微调这三个值！
            camera=dict(eye=dict(x=1.0, y=-1.0, z=0.7))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600,
        plot_bgcolor='white'
    )

    return fig


# ==========================================
# 核心物理推演引擎 (升级版：8连通异形坑洼全捕捉)
# ==========================================
def simulate_water_film_with_low_wall(data0, shuimo_h, wall_margin, max_h_step=0.05):
    m, n = data0.shape
    wall_height = np.max(data0) + wall_margin
    qy = np.pad(data0, pad_width=1, mode='constant', constant_values=wall_height)

    V = shuimo_h * m * n

    # 这让连通域和边缘膨胀都能识别对角线水流，完美捕捉斜向车辙和异形坑
    structure = generate_binary_structure(2, 2)

    iteration = 0

    while V > 1e-6:
        iteration += 1

        top = qy[:-2, 1:-1]
        bottom = qy[2:, 1:-1]
        left = qy[1:-1, :-2]
        right = qy[1:-1, 2:]

        # 增加对角线探针
        top_left = qy[:-2, :-2]
        top_right = qy[:-2, 2:]
        bottom_left = qy[2:, :-2]
        bottom_right = qy[2:, 2:]

        center = qy[1:-1, 1:-1]

        # 只有比周围 8 个点都低(或相等)，才算真正的坑底
        is_min_center = (center <= top) & (center <= bottom) & \
                        (center <= left) & (center <= right) & \
                        (center <= top_left) & (center <= top_right) & \
                        (center <= bottom_left) & (center <= bottom_right)

        is_min = np.pad(is_min_center, pad_width=1, mode='constant', constant_values=False)

        num_minima = np.sum(is_min)
        if num_minima == 0:
            break

        # 限制步长，同步逐层填水
        theoretical_h_dist = V / num_minima
        h_dist = min(max_h_step, theoretical_h_dist)

        V_used = h_dist * num_minima
        V_remaining = V - V_used

        # 向量化加水
        qy[is_min] += h_dist

        # 这里的 label 已经因为 structure=2,2 变成了 8连通聚类
        labeled_array, num_features = label(is_min, structure=structure)
        v_excess = np.zeros_like(qy)

        for region_idx in range(1, num_features + 1):
            region_mask = (labeled_array == region_idx)
            # 膨胀寻找溢流口时，也会向对角线寻找
            boundary_mask = binary_dilation(region_mask, structure=structure) ^ region_mask

            if not np.any(boundary_mask):
                continue

            ljmin = np.min(qy[boundary_mask])
            region_val = qy[region_mask][0]

            if region_val > ljmin:
                excess_water = qy[region_mask] - ljmin
                if ljmin < wall_height - 1e-5:
                    v_excess[region_mask] = excess_water
                qy[region_mask] = ljmin

        V = V_remaining + np.sum(v_excess)

        if iteration > 5000:
            break

    water_surface = qy[1:-1, 1:-1]
    water_depth = water_surface - data0
    water_depth[water_depth < 1e-4] = 0
    return water_surface, water_depth


def plot_2d_cross_section(matrix, water_surf, dx_mm, row_idx=50):
    profile_orig = matrix[row_idx, :]
    profile_water = water_surf[row_idx, :] if water_surf is not None else profile_orig
    x_m = np.arange(len(profile_orig)) * (dx_mm / 1000.0)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    if water_surf is not None:
        ax.fill_between(x_m, profile_orig, profile_water, color='#00a8ff', alpha=0.6, label='积水区域')
        ax.plot(x_m, profile_water, color='#0097e6', linewidth=1, linestyle='--')

    ax.plot(x_m, profile_orig, color='#2f3640', linewidth=1.5, label=f'路面高程')
    ax.set_title(f'中心车道横截面状态 (纵向位置: {row_idx * dx_mm / 1000.0:.1f}m)', fontsize=12, fontweight='bold')
    ax.set_xlabel('横向物理宽度 (米)')
    ax.set_ylabel('高程 (mm)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig


# ==========================================
# 侧边栏：操作流配置区
# ==========================================
dx_mm = 100.0

with st.sidebar:
    st.header("📂 第一步：地形解析")
    uploaded_file = st.file_uploader("上传路面点云 (.h5)", type=['h5'])

    start_segment = None
    num_blocks = 1

    if uploaded_file is not None:
        with open("temp_data.h5", "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            with h5py.File("temp_data.h5", 'r') as h5f:
                segments = sorted(list(h5f['road_segments'].keys()))
            start_segment = st.selectbox("🎯 选择起始分析路段", segments)
            max_blocks = len(segments) - segments.index(start_segment)
            num_blocks = st.slider("连续读取路段数量", min_value=1, max_value=max_blocks, value=min(5, max_blocks))
        except Exception as e:
            st.error("无法读取 H5 文件结构。")

    target_width_m = st.number_input("核心车道宽度 (m)", value=3.75)
    length_m = st.number_input("纵向截取长度 (m)", value=3.0)

    btn_load_road = st.button("🗺️ 1. 解析并生成 3D 地形", type="primary", use_container_width=True)

    # 这就确保了只要点击过一次解析，下面的模拟按钮马上就会解锁！
    if btn_load_road:
        if uploaded_file is None or start_segment is None:
            st.error("请先上传 .h5 文件！")
        else:
            with st.spinner("⏳ 正在重建高精度 3D 物理底座..."):
                matrix_full = load_and_preprocess_h5("temp_data.h5", start_segment, num_blocks)
                if matrix_full is not None:
                    test_rows = int(length_m * 1000 / dx_mm)
                    matrix_full = matrix_full[:test_rows, :]

                    total_cols = matrix_full.shape[1]
                    target_cols = int(target_width_m * 1000 / dx_mm)
                    trim_cols = max(0, (total_cols - target_cols) // 2)

                    st.session_state.matrix_full = matrix_full
                    st.session_state.matrix_crop = matrix_full[:, trim_cols: trim_cols + target_cols]
                    st.session_state.trim_cols = trim_cols
                    st.session_state.target_cols = target_cols
                    st.session_state.road_loaded = True

    st.divider()
    st.header("⚙️ 第二步：动态水膜仿真")
    target_rainfall = st.slider("目标总降雨量 (mm)", 0.0, 50.0, 10.0, step=0.5)
    wall_margin = st.slider("边缘挡水墙裕量 (mm)", -10.0, 20.0, 2.0, step=0.5)
    anim_frames = st.slider("仿真动画帧数 (分解步数)", 1, 10, 5)
    max_h_step = st.slider("单次最大水位爬升步长(mm)", 0.0, 0.1, 0.02, step=0.01)
    st.divider()
    st.header("💾 3. 日志存储设置")
    log_dir = st.text_input("本地日志存储路径", value="./logs", help="建议使用绝对路径, 例如: D:/WaterFilmLogs")
    btn_run_sim = st.button("🌊 2. 开始动态降雨推演", type="primary", use_container_width=True,
                            disabled=not st.session_state.road_loaded)

# ==========================================
# 主界面逻辑处理
# ==========================================
st.divider()

col_left, col_right = st.columns([3, 2])
with col_left:
    st.subheader("🧊 三维空间水膜演化")
    plot3d_container = st.empty()
with col_right:
    st.subheader("📈 典型横截面 (二维波谷填充)")
    plot2d_container = st.empty()
    metrics_container = st.empty()

export_container = st.empty()

# 缺省骨架渲染
# 如果还没上传解析数据，展示提示性的空白面板
if not st.session_state.road_loaded:
    with metrics_container.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("当前降雨进度", "-- mm")
        c2.metric("最大积水深度", "-- mm")
        c3.metric("积水覆盖率", "-- %")
        c4.metric("地形最大高差", "-- mm")

    fig_empty = go.Figure().update_layout(title="请先在左侧配置并解析路面地形...",
                                          scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=2.5, z=0.5)),
                                          height=500)
    plot3d_container.plotly_chart(fig_empty, use_container_width=True)

    fig_empty_2d, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, "等待数据导入...", ha='center', va='center', color='gray')
    ax.set_xticks([]);
    ax.set_yticks([])
    plot2d_container.pyplot(fig_empty_2d)

# 已加载地形，渲染静态图
elif st.session_state.road_loaded and not btn_run_sim:
    matrix_crop = st.session_state.matrix_crop
    with metrics_container.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("当前降雨进度", "0.0 mm")
        c2.metric("最大积水深度", "0.00 mm")
        c3.metric("积水覆盖率", "0.0 %")
        c4.metric("地形最大高差", f"{np.max(matrix_crop) - np.min(matrix_crop):.1f} mm")

    plot3d_container.plotly_chart(create_3d_figure(matrix_crop, dx_mm=dx_mm), use_container_width=True)
    row_i = min(int(matrix_crop.shape[0] / 2), matrix_crop.shape[0] - 1)
    plot2d_container.pyplot(plot_2d_cross_section(matrix_crop, None, dx_mm, row_idx=row_i))

# ------------------------------------------
# 动作 2: 执行动态降雨动画
# ------------------------------------------
if btn_run_sim and st.session_state.road_loaded:
    matrix_crop = st.session_state.matrix_crop

    # 1. 将高耗时的插值计算移出循环，只做一次
    scale_factor = 2
    original_dx_mm = 100.0
    fine_dx_mm = original_dx_mm / scale_factor

    with st.spinner("⏳ 正在进行超分辨率地图插值..."):
        fine_matrix_crop = zoom(matrix_crop, scale_factor, order=3)

    progress_bar = st.progress(0)
    status_text = st.empty()
    final_depth_crop = None

    # ====== 新增：记录仿真数据的列表 ======
    simulation_history = []

    for step in range(1, anim_frames + 1):
        current_rain = target_rainfall * (step / anim_frames)
        status_text.text(f"正在进行流体力学演算... 阶段 {step}/{anim_frames} (雨量: {current_rain:.1f}mm)")

        # 2. 物理演算
        surf_crop, depth_crop = simulate_water_film_with_low_wall(
            fine_matrix_crop,
            current_rain / 1000.0,
            wall_margin,
            max_h_step
        )
        final_depth_crop = depth_crop

        # 记录每一阶段的数据
        coverage = (np.count_nonzero(depth_crop) / depth_crop.size) * 100

        st.session_state.coverage_history[round(current_rain, 2)] = coverage

        simulation_history.append({
            "降雨阶段": f"{step}/{anim_frames}",
            "当前降雨量 (mm)": round(current_rain, 2),
            "最大积水深度 (mm)": round(np.max(depth_crop), 2),
            "平均积水深度 (mm)": round(np.mean(depth_crop), 2),
            "积水覆盖率 (%)": f"{coverage:.2f}%"
        })

        with metrics_container.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("当前降雨进度", f"{current_rain:.1f} mm", delta=f"+{target_rainfall / anim_frames:.1f} mm")
            c2.metric("最大积水深度", f"{np.max(depth_crop):.2f} mm")
            c3.metric("积水覆盖率", f"{(np.count_nonzero(depth_crop) / depth_crop.size) * 100:.1f} %")
            c4.metric("地形最大高差", f"{np.max(matrix_crop) - np.min(matrix_crop):.1f} mm")

        # 3. 维度对齐与 Key 标识
        plot3d_container.plotly_chart(
            create_3d_figure(fine_matrix_crop, surf_crop, depth_crop, fine_dx_mm),
            use_container_width=True,
            key=f"sim_frame_{step}"  # 修复 Streamlit ID 冲突
        )

        # 2D 图也需要使用 fine 矩阵以保证横坐标点数匹配
        row_i = int(fine_matrix_crop.shape[0] / 2)
        plot2d_container.pyplot(plot_2d_cross_section(fine_matrix_crop, surf_crop, fine_dx_mm, row_idx=row_i))

        progress_bar.progress(step / anim_frames)
        time.sleep(0.05)

    status_text.success(f"✅ 物理推演完成！最终降雨量达到 {target_rainfall} mm。")

    final_coverage = (np.count_nonzero(final_depth_crop) / final_depth_crop.size) * 100
    st.session_state.coverage_history[target_rainfall] = final_coverage

    # ====== 后台自动存储日志逻辑 ======
    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file_path = os.path.join(log_dir, "simulation_logs.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 构建精简版日志结构
        log_lines = [
            f"--- 记录时间: {timestamp} ---\n",
            f"参数设置 -> 目标降雨量: {target_rainfall} mm | 单次爬升步长: {max_h_step} mm | 挡水墙裕量: {wall_margin} mm | 仿真步数: {anim_frames}\n",
            "执行结果:\n"
        ]

        for item in simulation_history:
            log_lines.append(
                f"  [{item['降雨阶段']}] 雨量: {item['目标降雨量 (mm)']:>5.2f} mm | "
                f"最大水深: {item['最大水深 (mm)']:>5.2f} mm | "
                f"平均水深: {item['平均水深 (mm)']:>5.2f} mm | "
                f"覆盖率: {item['覆盖率 (%)']}\n"
            )
        log_lines.append("\n")  # 添加空行分隔每次任务

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.writelines(log_lines)

        st.toast(f"📄 日志已自动保存至: {log_file_path}")

    except Exception as e:
        st.error(f"❌ 自动保存日志失败，请检查路径权限: {e}")

    # ====== 纯净版的前端表格展示 ======
    with export_container:
        st.markdown("### 📊 动态降雨过程数据详情")
        df_history = pd.DataFrame(simulation_history)
        st.dataframe(df_history, use_container_width=True)

# ==========================================
# 统计分析区
# ==========================================
if st.session_state.road_loaded:
    st.divider()
    st.subheader("📈 积水覆盖率随总降雨量变化趋势")

    col_chart, col_btn = st.columns([6, 1])

    with col_btn:
        st.write("")
        st.write("")
        if st.button("🗑️ 清空趋势图表", use_container_width=True):
            st.session_state.coverage_history = {}
            st.rerun()

    with col_chart:
        if st.session_state.coverage_history:
            sorted_rains = sorted(st.session_state.coverage_history.keys())
            sorted_coverages = [st.session_state.coverage_history[rain] for rain in sorted_rains]

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=sorted_rains,
                y=sorted_coverages,
                mode='lines+markers',
                line=dict(color='#ff4757', width=3),
                marker=dict(size=10, color='#ffffff', line=dict(width=2, color='#ff4757'))
            ))

            fig_trend.update_layout(
                xaxis_title="设定的总降雨量 (mm)",
                yaxis_title="路面积水覆盖率 (%)",
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#f1f2f6', zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='#f1f2f6', zeroline=False, rangemode='tozero')
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("💡 暂无历史趋势数据。请在左侧设定不同的降雨量并执行推演，即可在此处自动生成趋势图。")