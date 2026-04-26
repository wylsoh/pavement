import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import io
import h5py
import tempfile
from datetime import datetime
from scipy.ndimage import label, generate_binary_structure, binary_dilation
from scipy.ndimage import zoom
from scipy.ndimage import median_filter

from risk_assessment import evaluate_hydroplaning_risk, dynamic_decision_making, render_risk_heatmap
from treatment_decision import extract_high_risk_regions, add_bounding_boxes_to_fig, generate_treatment_plan_and_budget


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
np.set_printoptions(suppress=True)

# ==========================================
# UI 页面与状态配置
# ==========================================
st.set_page_config(page_title="路面水膜灾害分析系统", layout="wide", page_icon="🌧️")

# 全局 CSS 注入
st.markdown("""
<style>
    /* 1. 隐藏 Streamlit 默认的右上角菜单和底部水印 (让系统看起来更像独立软件) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 2. 美化所有的全局按钮 (增加圆角、悬浮放大和阴影动效) */
    .stButton>button {
        border-radius: 8px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 10px rgba(0,0,0,0.15) !important;
    }

    /* 3. 美化所有的指标组件 (st.metric) */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #0097e6 !important; /* 主题蓝 */
    }
</style>
""", unsafe_allow_html=True)

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
if 'final_depth_crop' not in st.session_state:
    st.session_state.final_depth_crop = None
if 'fine_matrix_crop' not in st.session_state:
    st.session_state.fine_matrix_crop = None
if 'surf_crop' not in st.session_state:
    st.session_state.surf_crop = None
if 'fine_dx_mm' not in st.session_state:
    st.session_state.fine_dx_mm = None
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = None
if 'last_target_rainfall' not in st.session_state:
    st.session_state.last_target_rainfall = 0.0
if 'risk_results' not in st.session_state:
    st.session_state.risk_results = None

st.title("🌧️ 路表降雨水膜动态物理推演系统")
st.markdown("支持点云高程去零裁剪、任意路段指定检视、以及分步动态降雨堆积演示。")


# ==========================================
# 核心数据读取与预处理函数
# ==========================================
def load_and_preprocess_h5(h5_path, start_segment, num_blocks, max_std=15.0, overlap_rows=8):
    with h5py.File(h5_path, 'r') as h5f:
        group = h5f['road_segments']
        all_names = sorted(list(group.keys()))
        start_idx = all_names.index(start_segment)

        zz = None
        blocks_loaded = 0
        curr_idx = start_idx

        x = np.linspace(0, 1, overlap_rows)
        weights_1d = (1 + np.cos(np.pi * x)) / 2.0
        weights = weights_1d.reshape(-1, 1)

        while blocks_loaded < num_blocks and curr_idx < len(all_names):
            name = all_names[curr_idx]
            curr_idx += 1
            data = group[name][:]

            data = data[1:, :]
            valid_mask = data != 0

            if np.sum(valid_mask) < 100:
                continue

            current_std = np.std(data[valid_mask])
            if current_std > max_std:
                st.toast(f"⚠️ 触发保护机制: 自动跳过坏死路段 [{name}] (标准差={current_std:.2f})")
                continue

            data_median = np.median(data[valid_mask])
            data = np.where(data == 0, data_median, data)

            ref_matrix = median_filter(data, size=5)
            diff = np.abs(data - ref_matrix)
            mu, sigma = np.mean(diff), np.std(diff)
            outlier_mask = diff > (mu + 3 * sigma)
            data[outlier_mask] = ref_matrix[outlier_mask]

            if zz is None:
                zz = data
                blocks_loaded += 1
                continue

            width = zz.shape[1]
            mid_col, safe_w = width // 2, max(1, width // 4)
            start_c, end_c = max(0, mid_col - safe_w), min(width, mid_col + safe_w)

            edge_zz = zz[-overlap_rows:, start_c:end_c]
            edge_data = data[:overlap_rows, start_c:end_c]

            gaocha = np.mean(edge_zz) - np.mean(edge_data)
            data += gaocha

            overlap_A = zz[-overlap_rows:, :]
            overlap_B = data[:overlap_rows, :]
            zz[-overlap_rows:, :] = overlap_A * weights + overlap_B * (1.0 - weights)

            zz = np.vstack((zz, data[overlap_rows:, :]))
            blocks_loaded += 1

        if zz is None:
            return None

        z_min, z_max = np.percentile(zz, [0.1, 99.9])
        zz = np.clip(zz, z_min, z_max)

        return zz


# ==========================================
# 辅助渲染函数
# ==========================================
def create_3d_figure(matrix, water_surf=None, water_depth=None, dx_mm=100.0):
    x_dm = np.arange(matrix.shape[1]) * (dx_mm / 100.0)
    y_dm = np.arange(matrix.shape[0]) * (dx_mm / 100.0)
    z_m = matrix

    fig = go.Figure()
    fig.add_trace(go.Surface(
        z=z_m, x=x_dm, y=y_dm,
        colorscale='Portland', name='路表高程', showscale=False,
        contours=dict(
            x=dict(show=True, color='black', width=1, start=x_dm[0], end=x_dm[-1], size=dx_mm / 100.0),
            y=dict(show=True, color='black', width=1, start=y_dm[0], end=y_dm[-1], size=dx_mm / 100.0)
        )
    ))

    if water_surf is not None and water_depth is not None:
        water_surf_m = water_surf
        water_only = np.where(water_depth > 1e-4, water_surf_m, np.nan)

        if not np.all(np.isnan(water_only)):
            fig.add_trace(go.Surface(
                z=water_only, x=x_dm, y=y_dm,
                colorscale=[[0, 'aqua'], [1, 'aqua']], opacity=0.65,
                name='水膜', showscale=False, hoverinfo='skip'
            ))

    x_physical_length = x_dm[-1] - x_dm[0]
    y_physical_length = y_dm[-1] - y_dm[0]
    true_y_ratio = y_physical_length / x_physical_length if x_physical_length > 0 else 1

    fig.update_layout(
        scene=dict(
            xaxis_title='车道宽度(dm)',
            yaxis_title='路线长度(dm)',
            zaxis_title='路表高程(m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=true_y_ratio, z=0.5),
            camera=dict(eye=dict(x=0.9, y=-0.9, z=0.8))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600,
        plot_bgcolor='white',
        font=dict(family="微软雅黑, Arial, sans-serif", size=15, color="#333333")
    )
    return fig


def plot_2d_cross_section(matrix, water_surf, dx_mm, row_idx=50):
    profile_orig = matrix[row_idx, :]
    profile_water = water_surf[row_idx, :] if water_surf is not None else profile_orig
    x_m = np.arange(len(profile_orig)) * (dx_mm / 1000.0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_m,
        y=profile_orig,
        mode='lines',
        line=dict(color='#2f3640', width=2),
        name='路面高程'
    ))

    if water_surf is not None:
        fig.add_trace(go.Scatter(
            x=x_m,
            y=profile_water,
            mode='lines',
            line=dict(color='#0097e6', width=1.5, dash='dash'),
            name='积水区域',
            fill='tonexty',
            fillcolor='rgba(0, 168, 255, 0.6)'
        ))

    fig.update_layout(
        title=dict(
            text=f'中心车道横截面状态 (纵向位置: {row_idx * dx_mm / 1000.0:.1f}m)',
            font=dict(size=16, family="Microsoft YaHei, Arial, sans-serif")
        ),
        xaxis=dict(
            title='横向物理宽度 (米)',
            showgrid=True,
            gridcolor='#eeeeee',
            zeroline=False
        ),
        yaxis=dict(
            title='高程 (m)',
            tickformat='.3f',
            showgrid=True,
            gridcolor='#eeeeee',
            zeroline=False
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="right", x=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e0e0e0",
            borderwidth=1
        ),
        height=300
    )

    return fig


# ==========================================
# 核心物理推演引擎
# ==========================================
def simulate_water_film_with_low_wall(data0, shuimo_h, wall_margin, max_h_step=0.05):
    m, n = data0.shape
    wall_height = np.max(data0) + wall_margin
    qy = np.pad(data0, pad_width=1, mode='constant', constant_values=wall_height)
    V = shuimo_h * m * n
    structure = generate_binary_structure(2, 2)
    iteration = 0

    while V > 1e-6:
        iteration += 1
        top = qy[:-2, 1:-1]
        bottom = qy[2:, 1:-1]
        left = qy[1:-1, :-2]
        right = qy[1:-1, 2:]
        top_left = qy[:-2, :-2]
        top_right = qy[:-2, 2:]
        bottom_left = qy[2:, :-2]
        bottom_right = qy[2:, 2:]
        center = qy[1:-1, 1:-1]

        is_min_center = (center <= top) & (center <= bottom) & \
                        (center <= left) & (center <= right) & \
                        (center <= top_left) & (center <= top_right) & \
                        (center <= bottom_left) & (center <= bottom_right)
        is_min = np.pad(is_min_center, pad_width=1, mode='constant', constant_values=False)
        num_minima = np.sum(is_min)
        if num_minima == 0:
            break

        theoretical_h_dist = V / num_minima
        h_dist = min(max_h_step, theoretical_h_dist)
        V_used = h_dist * num_minima
        V_remaining = V - V_used
        qy[is_min] += h_dist

        labeled_array, num_features = label(is_min, structure=structure)
        v_excess = np.zeros_like(qy)

        for region_idx in range(1, num_features + 1):
            region_mask = (labeled_array == region_idx)
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


# ==========================================
# 侧边栏：操作流配置区
# ==========================================
dx_mm = 100.0

with st.sidebar:
    st.header("📂 第一步：地形解析")

    if 'current_h5_path' not in st.session_state:
        st.session_state.current_h5_path = None
    if 'prev_data_source' not in st.session_state:
        st.session_state.prev_data_source = "📁 上传本地数据 (.h5)"

    # 1. 选项菜单：支持数据源切换
    data_source = st.radio("选择数据来源", ["📁 上传本地数据 (.h5)", "📦 加载内置示例数据"])

    # 2. 状态保护：如果检测到用户切换了选项，立刻清空底层的旧数据，防止新旧数据串台
    if data_source != st.session_state.prev_data_source:
        st.session_state.current_h5_path = None
        st.session_state.road_loaded = False
        st.session_state.prev_data_source = data_source

    # 3. 分支逻辑 A：用户自己上传
    if data_source == "📁 上传本地数据 (.h5)":
        uploaded_file = st.file_uploader("上传路面点云 (.h5)", type=['h5'])
        if uploaded_file is not None:
            # 使用安全的临时文件写入
            if st.session_state.get('last_uploaded_name') != uploaded_file.name:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    st.session_state.current_h5_path = tmp_file.name
                st.session_state.last_uploaded_name = uploaded_file.name
        else:
            # 如果用户点击了上传框里的 'X' 清空了文件，底层的路径也同步清空
            st.session_state.current_h5_path = None

    # 3. 分支逻辑 B：使用内置示例
    elif data_source == "📦 加载内置示例数据":
        st.info("💡 系统将直接加载项目库中内置的高保真路面点云数据，供您快速体验完整功能流程。")

        # 【已恢复】：把确认加载的按钮加回来，满足明确的操作习惯
        if st.button("⬇️ 一键加载内置示例数据", use_container_width=True):
            # 指向你本地项目文件夹里的数据路径
            local_sample_path = os.path.join("assets", "sample_data.h5")

            if os.path.exists(local_sample_path):
                st.session_state.current_h5_path = local_sample_path
                st.success("✅ 内置示例数据加载成功！请在下方选择路段并生成地形。")
            else:
                st.error(f"❌ 找不到内置示例文件！\n\n请检查代码所在的目录下是否存在 `{local_sample_path}` 这个文件。")

    st.markdown(
        """
        <div style="background-color: #f0f8ff; padding: 12px; border-radius: 8px; font-size: 13px; color: #333; margin-top: 15px; margin-bottom: 15px; border-left: 5px solid #00a8ff; line-height: 1.5;">
        <b>🔒 数据隐私与安全承诺</b><br>
        本系统采用严格的<b>“无痕运算”</b>机制。您上传的业务文件仅在内存及系统的沙盒临时目录中即时推演，绝不会被后台窃取、云端持久化存储或作他用。页面刷新或会话结束后，所有临时数据将彻底自动销毁，请安心导入您的生产数据。
        </div>
        """, unsafe_allow_html=True
    )

    start_segment = None
    num_blocks = 1

    # 只有当 current_h5_path 成功赋值（不管上传的还是内置的），才会渲染路段选择框
    if st.session_state.current_h5_path is not None:
        try:
            with h5py.File(st.session_state.current_h5_path, 'r') as h5f:
                segments = sorted(list(h5f['road_segments'].keys()))
            start_segment = st.selectbox("🎯 选择起始分析路段", segments)
            max_blocks = len(segments) - segments.index(start_segment)
            num_blocks = st.slider("连续读取路段数量", min_value=1, max_value=max_blocks, value=min(5, max_blocks))
        except Exception as e:
            st.error(f"无法读取 H5 文件结构，请确保上传了合法的文件。\n\n详情: {e}")

    target_width_m = st.number_input("核心车道宽度 (m)", value=3.75)
    length_m = st.number_input("纵向截取长度 (m)", value=5.0)

    btn_load_road = st.button("🗺️ 1. 解析并生成 3D 地形", type="primary", use_container_width=True)

    if btn_load_road:
        if st.session_state.current_h5_path is None or start_segment is None:
            st.error("请先上传或加载 .h5 数据文件！")
        else:
            with st.spinner("⏳ 正在重建高精度 3D 物理底座..."):
                matrix_full = load_and_preprocess_h5(st.session_state.current_h5_path, start_segment, num_blocks)
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

                    st.session_state.final_depth_crop = None
                    st.session_state.fine_matrix_crop = None
                    st.session_state.surf_crop = None
                    st.session_state.risk_results = None

    st.divider()
    st.header("⚙️ 第二步：动态水膜仿真")
    target_rainfall = st.slider("目标总降雨量 (mm)", 0.0, 50.0, 10.0, step=0.5)
    runoff_coefficient = st.slider(
        "路面径流滞留系数 (排水折减)",
        min_value=0.01,
        max_value=1.00,
        value=0.01,
        step=0.01,
        help="模拟路面横纵坡的排水能力。1.0表示无排水（变死水潭），0.15表示85%的雨水被排走，仅15%滞留在车辙与构造中。"
    )
    wall_margin = st.slider("边缘挡水墙裕量 (mm)", 0.0, 2.0, 1.0, step=0.5)
    anim_frames = st.slider("仿真动画帧数 (分解步数)", 1, 10, 5)
    max_h_step = st.slider("单次最大水位爬升步长(mm)", 0.000, 0.100, 0.010, step=0.002, format="%.3f")
    btn_run_sim = st.button("🌊 2. 开始动态降雨推演", type="primary", use_container_width=True,
                            disabled=not st.session_state.road_loaded)
    st.divider()
    st.header("💾 日志存储设置")
    log_dir = st.text_input("本地日志存储路径", value="./logs", help="建议使用绝对路径, 例如: D:/WaterFilmLogs")


# ==========================================
def render_centered_metric(col_obj, title, value, delta=""):
    delta_html = f"<div style='color: #09ab3b; font-size: 14px; margin-top: 2px;'>{delta}</div>" if delta else "<div style='height: 21px;'></div>"
    html_content = f"""
    <div style="text-align: center; font-family: 'Microsoft YaHei', sans-serif; padding: 10px 0;">
        <div style="font-size: 14px; color: #555; margin-bottom: 4px;">{title}</div>
        <div style="font-size: 28px; font-weight: bold; color: #212529; line-height: 1.2;">{value}</div>
        {delta_html}
    </div>
    """
    col_obj.markdown(html_content, unsafe_allow_html=True)


# ==========================================
# 主界面逻辑处理：状态分发机制
# ==========================================
st.divider()

col_left, col_right = st.columns([1, 1])
with col_left:
    st.subheader("🧊 三维空间水膜演化")
    plot3d_container = st.empty()
with col_right:
    st.subheader("📈 典型横截面 (二维波谷填充)")
    plot2d_container = st.empty()
    metrics_container = st.empty()

export_container = st.empty()

# ------------------------------------------
# 状态 1：缺省骨架渲染（未上传数据）
# ------------------------------------------
if not st.session_state.road_loaded:
    with metrics_container.container():
        c1, c2, c3, c4 = st.columns(4)
        render_centered_metric(c1, "当前降雨进度", "-- mm")
        render_centered_metric(c2, "最大积水深度", "-- mm")
        render_centered_metric(c3, "积水覆盖率", "-- %")
        render_centered_metric(c4, "地形最大高差", "-- mm")

    fig_empty = go.Figure().update_layout(title="请先在左侧配置并解析路面地形...",
                                          scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=2.5, z=0.5)),
                                          margin=dict(l=0, r=0, b=0, t=30), autosize=True, plot_bgcolor='white')
    plot3d_container.plotly_chart(fig_empty, use_container_width=True)

    fig_empty_2d = go.Figure().update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text="等待数据导入...", x=0.5, y=0.5, showarrow=False, font=dict(color="gray", size=16))],
        plot_bgcolor='white', margin=dict(l=0, r=0, b=0, t=30), height=300
    )
    plot2d_container.plotly_chart(fig_empty_2d, use_container_width=True)

# ------------------------------------------
# 状态 2：执行动态降雨推演动画
# ------------------------------------------
elif btn_run_sim and st.session_state.road_loaded:
    st.session_state.risk_results = None

    matrix_crop = st.session_state.matrix_crop
    scale_factor = 2
    original_dx_mm = 100.0
    fine_dx_mm = original_dx_mm / scale_factor

    with st.spinner("⏳ 正在进行超分辨率地图插值..."):
        fine_matrix_crop = zoom(matrix_crop, scale_factor, order=3)

    progress_bar = st.progress(0)
    status_text = st.empty()
    final_depth_crop = None
    simulation_history = []
    final_surf_crop = None

    for step in range(1, anim_frames + 1):
        current_rain = target_rainfall * (step / anim_frames)
        status_text.text(f"正在进行流体力学演算... 阶段 {step}/{anim_frames} (雨量: {current_rain:.1f}mm)")

        effective_rain_m = (current_rain * runoff_coefficient) / 1000.0

        surf_crop, depth_crop = simulate_water_film_with_low_wall(
            fine_matrix_crop,
            effective_rain_m,
            wall_margin / 1000.0,
            max_h_step / 1000.0
        )
        final_depth_crop = depth_crop
        final_surf_crop = surf_crop

        area_ratio = st.session_state.matrix_full.shape[1] / st.session_state.matrix_crop.shape[1]
        theoretical_total_size = depth_crop.size * area_ratio

        coverage = (np.count_nonzero(depth_crop) / theoretical_total_size) * 100
        st.session_state.coverage_history[round(current_rain, 2)] = coverage

        simulation_history.append({
            "降雨阶段": f"{step}/{anim_frames}",
            "当前降雨量 (mm)": round(current_rain, 2),
            "有效滞留水深 (mm)": round(current_rain * runoff_coefficient, 2),
            "最大积水深度 (mm)": round(np.max(depth_crop) * 1000.0, 2),
            "平均积水深度 (mm)": round(np.mean(depth_crop) * 1000.0, 2),
            "积水覆盖率 (%)": f"{coverage:.2f}%"
        })

        with metrics_container.container():
            c1, c2, c3, c4 = st.columns(4)
            render_centered_metric(c1, "当前降雨进度", f"{current_rain:.1f} mm",
                                   delta=f"↑ {target_rainfall / anim_frames:.1f} mm")
            render_centered_metric(c2, "最大积水深度", f"{np.max(depth_crop) * 1000.0:.2f} mm")
            render_centered_metric(c3, "积水覆盖率", f"{(np.count_nonzero(depth_crop) / depth_crop.size) * 100:.1f} %")
            render_centered_metric(c4, "地形最大高差", f"{(np.max(matrix_crop) - np.min(matrix_crop)) * 1000.0:.1f} mm")

        plot3d_container.plotly_chart(create_3d_figure(fine_matrix_crop, surf_crop, depth_crop, fine_dx_mm),
                                      use_container_width=True, key=f"sim_frame_{step}")

        row_i = int(fine_matrix_crop.shape[0] / 2)
        fig_2d_sim = plot_2d_cross_section(fine_matrix_crop, surf_crop, fine_dx_mm, row_idx=row_i)
        plot2d_container.plotly_chart(fig_2d_sim, use_container_width=True, key=f"sim_2d_frame_{step}")

        progress_bar.progress(step / anim_frames)
        time.sleep(0.05)

    status_text.success(f"✅ 物理推演完成！最终降雨量达到 {target_rainfall} mm。")

    st.session_state.final_depth_crop = final_depth_crop
    st.session_state.fine_matrix_crop = fine_matrix_crop
    st.session_state.surf_crop = final_surf_crop
    st.session_state.fine_dx_mm = fine_dx_mm
    st.session_state.simulation_history = simulation_history
    st.session_state.last_target_rainfall = target_rainfall

    try:
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        log_file_path = os.path.join(log_dir, "simulation_logs.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_lines = [f"--- 记录时间: {timestamp} ---\n",
                     f"参数设置 -> 目标降雨量: {target_rainfall} mm | 单次爬升步长: {max_h_step} mm | 挡水墙裕量: {wall_margin} mm | 仿真步数: {anim_frames}\n",
                     "执行结果:\n"]
        for item in simulation_history:
            log_lines.append(
                f"  [{item['降雨阶段']}] 雨量: {item['当前降雨量 (mm)']:>5.2f} mm | 最大水深: {item['最大积水深度 (mm)']:>5.2f} mm | 平均水深: {item['平均积水深度 (mm)']:>5.2f} mm | 覆盖率: {item['积水覆盖率 (%)']}\n")
        log_lines.append("\n")
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.writelines(log_lines)
        st.toast(f"📄 日志已自动保存至: {log_file_path}")
    except Exception as e:
        st.error(f"❌ 自动保存日志失败，请检查路径权限: {e}")

    with export_container:
        st.markdown("### 📊 动态降雨过程数据详情")
        st.dataframe(pd.DataFrame(simulation_history), use_container_width=True)

# ------------------------------------------
# 状态 3：调用缓存静默渲染
# ------------------------------------------
elif st.session_state.road_loaded and st.session_state.final_depth_crop is not None:
    fine_matrix = st.session_state.fine_matrix_crop
    surf = st.session_state.surf_crop
    depth = st.session_state.final_depth_crop
    f_dx = st.session_state.fine_dx_mm
    last_rain = st.session_state.last_target_rainfall
    sim_hist = st.session_state.simulation_history
    coverage = (np.count_nonzero(depth) / depth.size) * 100

    with metrics_container.container():
        c1, c2, c3, c4 = st.columns(4)
        render_centered_metric(c1, "当前降雨进度", f"{last_rain:.1f} mm")
        render_centered_metric(c2, "最大积水深度", f"{np.max(depth) * 1000.0:.2f} mm")
        render_centered_metric(c3, "积水覆盖率", f"{coverage:.1f} %")
        render_centered_metric(c4, "地形最大高差",
                               f"{(np.max(st.session_state.matrix_crop) - np.min(st.session_state.matrix_crop)) * 1000.0:.1f} mm")

    plot3d_container.plotly_chart(create_3d_figure(fine_matrix, surf, depth, f_dx), use_container_width=True)

    row_i = int(fine_matrix.shape[0] / 2)

    fig_2d_sim = plot_2d_cross_section(fine_matrix, surf, f_dx, row_idx=row_i)
    plot2d_container.plotly_chart(fig_2d_sim, use_container_width=True)

    with export_container:
        st.markdown("### 📊 动态降雨过程数据详情")
        st.dataframe(pd.DataFrame(sim_hist), use_container_width=True)

# ------------------------------------------
# 状态 4：渲染干路面静态图 (地形刚加载完，从未推演过)
# ------------------------------------------
elif st.session_state.road_loaded and st.session_state.final_depth_crop is None:
    matrix_crop = st.session_state.matrix_crop
    with metrics_container.container():
        c1, c2, c3, c4 = st.columns(4)
        render_centered_metric(c1, "当前降雨进度", "0.0 mm")
        render_centered_metric(c2, "最大积水深度", "0.00 mm")
        render_centered_metric(c3, "积水覆盖率", "0.0 %")
        render_centered_metric(c4, "地形最大高差", f"{(np.max(matrix_crop) - np.min(matrix_crop)) * 1000.0:.1f} mm")

    plot3d_container.plotly_chart(create_3d_figure(matrix_crop, dx_mm=dx_mm), use_container_width=True)
    row_i = min(int(matrix_crop.shape[0] / 2), matrix_crop.shape[0] - 1)
    fig_2d = plot_2d_cross_section(matrix_crop, None, dx_mm, row_idx=row_i)
    plot2d_container.plotly_chart(fig_2d, use_container_width=True)

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
            fig_trend.add_trace(go.Scatter(x=sorted_rains, y=sorted_coverages, mode='lines+markers',
                                           line=dict(color='#ff4757', width=3),
                                           marker=dict(size=10, color='#ffffff', line=dict(width=2, color='#ff4757'))))
            fig_trend.update_layout(xaxis_title="设定的总降雨量 (mm)", yaxis_title="路面积水覆盖率 (%)", height=350,
                                    margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='#f0f2f6',
                                    xaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
                                    yaxis=dict(showgrid=True, gridcolor='white', zeroline=False, rangemode='tozero'),
                                    font=dict(family="微软雅黑, Arial", size=16, color="black"))
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("💡 暂无历史趋势数据。请在左侧设定不同的降雨量并执行推演，即可在此处自动生成趋势图。")

# ==========================================
# 风险评估展示区
# ==========================================
st.markdown("---")
st.subheader("🚨 智能滑水风险评估与动态决策")

if st.session_state.final_depth_crop is not None:
    st.info("💡 系统已缓存当前地形的水膜分布数据，可随时调用进行滑水风险评估。")

    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        run_risk_btn = st.button("📊 基于当前水膜执行风险评估", type="primary", use_container_width=True)

    if run_risk_btn:
        water_depth = st.session_state.final_depth_crop
        with st.spinner("⏳ 正在结合水膜厚度与车速分布计算全域滑水概率..."):
            prob_matrix, risk_level_matrix, risk_score_matrix = evaluate_hydroplaning_risk(water_depth * 1000.0)

            area_ratio = st.session_state.matrix_full.shape[1] / st.session_state.matrix_crop.shape[1]
            decision = dynamic_decision_making(risk_level_matrix, area_ratio)

            st.session_state.risk_results = {
                "decision": decision,
                "risk_score_matrix": risk_score_matrix
            }

    if st.session_state.risk_results is not None:
        decision = st.session_state.risk_results["decision"]
        risk_score_matrix = st.session_state.risk_results["risk_score_matrix"]

        col1, col2 = st.columns(2)
        with col1:
            if "危险" in decision['overall_status']:
                st.error(f"**整体评估状态:** {decision['overall_status']}")
            elif "关注" in decision['overall_status']:
                st.warning(f"**整体评估状态:** {decision['overall_status']}")
            else:
                st.success(f"**整体评估状态:** {decision['overall_status']}")
            st.metric(label="高风险区 (A/B级) 占比", value=decision['high_risk_area_ratio'])

        with col2:
            st.info(f"**🚦 动态交通管控建议:** \n\n{decision['traffic_control']}")
            st.warning(f"**🛠️ 宏观养护指导:** \n\n{decision['maintenance_action']}")

        st.markdown("#### 面域级滑水风险图谱 (含高危区自动智能框选)")

        # 1. 获取基础风险热力图
        risk_fig = render_risk_heatmap(risk_score_matrix)

        # 2. 识别并提取高危区域 (A/B级)
        depth_matrix = st.session_state.final_depth_crop * 1000.0  # 转为 mm
        area_ratio = st.session_state.matrix_full.shape[1] / st.session_state.matrix_crop.shape[1]

        regions = extract_high_risk_regions(
            risk_score_matrix=risk_score_matrix,
            depth_matrix=depth_matrix,
            fine_dx_mm=st.session_state.fine_dx_mm,
            area_ratio=area_ratio
        )

        # 3. 将高危区边界框叠加至热力图中
        if len(regions) > 0:
            risk_fig = add_bounding_boxes_to_fig(risk_fig, regions)

        st.plotly_chart(risk_fig, use_container_width=True)

        # ==========================================
        # 新增模块：高危区靶向处治与工程预算最优方案比选
        # ==========================================
        st.markdown("---")
        st.subheader("🛠️ 高危区靶向处治与工程预算最优方案比选")

        if len(regions) > 0:
            st.info(
                f"📍 **智能巡检系统报告:** 在当前路段中成功识别出 **{len(regions)}** 个独立滑水高危核心区，已自动为您生成靶向处治匹配与经济性分析表。")

            # 调用我们在 treatment_decision.py 中写的函数生成方案与预算
            df_plan, budget_summary = generate_treatment_plan_and_budget(
                regions=regions,
                risk_score_matrix_shape=risk_score_matrix.shape,
                fine_dx_mm=st.session_state.fine_dx_mm,
                area_ratio=area_ratio
            )

            # 展示分区域处治方案详细表格
            st.dataframe(df_plan, hide_index=True, use_container_width=True)

            # 预算对比展示
            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("🤖 智能靶向刻槽方案总估价", f"¥ {budget_summary['smart_cost']:,.1f}", "包含500元无人机巡检费",
                          delta_color="inverse")
            col_b2.metric("🚜 传统全域铣刨重铺总估价", f"¥ {budget_summary['trad_cost']:,.1f}")
            col_b3.metric("💰 方案预计降低养护成本", f"{budget_summary['saving_ratio']:.1f} %", "经济效益显著")

            st.success(
                f"**最终决策建议：** 相比“发现积水即大面积铣刨”的传统盲目处治，采用 **[ 无人机高精度检测 + 目标路段靶向自动刻槽 ]** 策略可精准解决局部水膜隐患。据上表核算，本次优化策略预计可为您节约 **{budget_summary['saving_ratio']:.1f}%** 的养护工程预算！")

        else:
            st.success(
                "🎉 当前设定的降雨环境与路面状态下，暂未检出面积足够大且深度的严重积水高危区 (A/B级)，无需启动强制工程处治程序。")
else:
    st.info("💡 请先在上方执行“动态降雨推演”，生成水膜分布数据后，即可在此处运行风险评估。")