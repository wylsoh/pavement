import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import time
import os
import h5py
import tempfile
from scipy.ndimage import label, generate_binary_structure, binary_dilation
from scipy.ndimage import zoom
from scipy.ndimage import median_filter

from modules.risk_assessment import evaluate_hydroplaning_risk, dynamic_decision, risk_heatmap
from modules.treatment_decision import extract_high_risk_regions, add_bounding_boxes, generate_plan_and_budget
from modules.report_generator import render_report

np.set_printoptions(suppress=True)

# ==========================================
# UI 页面与状态配置
# ==========================================
st.set_page_config(page_title="路面水膜灾害分析系统", layout="wide", page_icon="🌧️")

# 初始化全局状态变量
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
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

# ==========================================
# 顶部标题与一键暗黑模式开关
# ==========================================
col_title, col_toggle = st.columns([7, 1])
with col_title:
    st.title("🌧️ 路表降雨水膜动态物理推演系统")
    st.markdown("支持点云高程去零裁剪、任意路段指定检视、以及分步动态降雨堆积演示。")
with col_toggle:
    st.write("")
    st.write("")
    st.toggle("🌙 暗黑模式", key="dark_mode")

# ==========================================
# CSS 主题注入
# ==========================================
base_css = """
    /* 隐藏菜单和底部 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}


    /* 核心指标字体颜色 */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #0097e6 !important; 
    }

    /* 所有的基础按钮过度动画配置 */
    .stButton > button {
        border-radius: 6px !important;
        font-weight: bold !important;
        transition: all 0.2s ease !important;
    }
"""

if st.session_state.dark_mode:
    theme_css = f"""
    <style>
    {base_css}
    /* 暗黑模式全局底色 */
    .stApp {{ background-color: #0d1117 !important; color: #c9d1d9 !important; }}

    /* 主内容卡片 (深空灰) */
    .main .block-container {{
        background-color: #161b22 !important;
        padding: 3rem; border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }}

    /* 侧边栏 */
    [data-testid="stSidebar"] {{ background-color: #010409 !important; }}

    /* 强制文本颜色 */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {{ color: #c9d1d9 !important; }}

    /* --- 暗黑模式按钮专属样式 --- */
    /* 普通按钮 (Secondary) */
    .stButton > button {{
        background-color: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
    }}
    .stButton > button:hover {{
        border-color: #0097e6 !important;
        color: #0097e6 !important;
    }}

    /* 主要按钮 (Primary) - 高亮操作 */
    .stButton > button[kind="primary"] {{
        background-color: #0078b4 !important;
        color: #ffffff !important;
        border: none !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: #0097e6 !important;
        box-shadow: 0 0 10px rgba(0, 151, 230, 0.4) !important;
    }}
    </style>
    """
else:
    theme_css = f"""
    <style>
    {base_css}
    /* 白天护眼模式全局底色 (莫兰迪蓝灰) */
    .stApp {{ background-color: #e9eef5 !important; color: #2c3e50 !important; }}

    /* 主内容卡片 (珍珠白) */
    .main .block-container {{
        background-color: #f4f7fa !important;
        padding: 3rem; border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
    }}

    /* 侧边栏 */
    [data-testid="stSidebar"] {{ background-color: #e0e6ed !important; }}

    /* 文本使用深藏青色代替纯黑 */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {{ color: #2c3e50 !important; }}

    /* --- 白天模式按钮专属样式 --- */
    /* 普通按钮 (Secondary) */
    .stButton > button {{
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        border: 1px solid #cbd5e1 !important;
    }}
    .stButton > button:hover {{
        border-color: #0097e6 !important;
        color: #0097e6 !important;
        background-color: #f8fafc !important;
    }}

    /* 主要按钮 (Primary) */
    .stButton > button[kind="primary"] {{
        background-color: #0097e6 !important;
        color: #ffffff !important;
        border: none !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: #00a8ff !important;
        box-shadow: 0 4px 12px rgba(0, 168, 255, 0.25) !important;
    }}
    </style>
    """
st.markdown(theme_css, unsafe_allow_html=True)


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

            if np.sum(valid_mask) < 100: continue

            current_std = np.std(data[valid_mask])
            if current_std > max_std:
                st.toast(f"⚠️ 触发保护机制: 自动跳过坏死路段 [{name}]")
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

        if zz is None: return None
        z_min, z_max = np.percentile(zz, [0.1, 99.9])
        zz = np.clip(zz, z_min, z_max)
        return zz


# ==========================================
# 辅助渲染函数 (动态主题支持)
# ==========================================
def create_3d_figure(matrix, water_surf=None, water_depth=None, dx_mm=100.0, show_grid=True, dark_mode=False):
    x_dm = np.arange(matrix.shape[1]) * (dx_mm / 100.0)
    y_dm = np.arange(matrix.shape[0]) * (dx_mm / 100.0)
    z_m = matrix

    template = "plotly_dark" if dark_mode else "plotly_white"
    font_color = "#c9d1d9" if dark_mode else "#2c3e50"
    grid_color = "rgba(255, 255, 255, 0.2)" if dark_mode else "rgba(0, 0, 0, 0.5)"

    fig = go.Figure()
    fig.add_trace(go.Surface(
        z=z_m, x=x_dm, y=y_dm,
        colorscale='Portland', name='路表高程', showscale=False,
        contours=dict(
            x=dict(show=show_grid, color=grid_color, width=1, start=x_dm[0], end=x_dm[-1], size=dx_mm / 100.0),
            y=dict(show=show_grid, color=grid_color, width=1, start=y_dm[0], end=y_dm[-1], size=dx_mm / 100.0)
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
        template=template,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",  # 透明背景墙
                gridcolor=grid_color,
                showbackground=True,
                title_text='车道宽度(dm)'
            ),
            yaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",  # 透明背景墙
                gridcolor=grid_color,
                showbackground=True,
                title_text='路线长度(dm)'
            ),
            zaxis=dict(
                backgroundcolor="rgba(0,0,0,0)",  # 透明背景墙
                gridcolor=grid_color,
                showbackground=True,
                title_text='高程(m)'
            ),
            xaxis_title='车道宽度(dm)', yaxis_title='路线长度(dm)', zaxis_title='路表高程(m)',
            aspectmode='manual', aspectratio=dict(x=1, y=true_y_ratio, z=0.4),
            camera=dict(eye=dict(x=0.9, y=-0.9, z=0.9))
        ),
        margin=dict(l=0, r=0, b=0, t=30), height=420,
        font=dict(family="微软雅黑, Arial, sans-serif", size=15, color=font_color)
    )
    return fig


def plot_2d_cross_section(matrix, water_surf, dx_mm, row_idx=50, dark_mode=False):
    profile_orig = matrix[row_idx, :]
    profile_water = water_surf[row_idx, :] if water_surf is not None else profile_orig
    x_m = np.arange(len(profile_orig)) * (dx_mm / 1000.0)

    template = "plotly_dark" if dark_mode else "plotly_white"
    font_color = "#c9d1d9" if dark_mode else "#2c3e50"
    line_color = "#c9d1d9" if dark_mode else "#34495e"
    grid_color = "#30363d" if dark_mode else "#e2e8f0"
    legend_bg = "rgba(22, 27, 34, 0.85)" if dark_mode else "rgba(244, 247, 250, 0.85)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x_m, y=profile_orig, mode='lines', line=dict(color=line_color, width=2), name='路面高程'))

    if water_surf is not None:
        fig.add_trace(
            go.Scatter(x=x_m, y=profile_water, mode='lines', line=dict(color='#0097e6', width=1.5, dash='dash'),
                       name='积水区域', fill='tonexty', fillcolor='rgba(0, 168, 255, 0.6)'))

    fig.update_layout(
        template=template,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title=dict(text=f'中心车道横截面状态 (纵向位置: {row_idx * dx_mm / 1000.0:.1f}m)',
                   font=dict(size=16, family="Microsoft YaHei", color=font_color)),
        xaxis=dict(title='横向物理宽度 (米)', showgrid=True, gridcolor=grid_color, zeroline=False),
        yaxis=dict(title='高程 (m)', tickformat='.3f', showgrid=True, gridcolor=grid_color, zeroline=False),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor=legend_bg, bordercolor=grid_color,
                    borderwidth=1),
        height=340, font=dict(color=font_color)
    )
    return fig


def render_centered_metric(col_obj, title, value, delta="", dark_mode=False):
    text_color = "#c9d1d9" if dark_mode else "#2c3e50"
    title_color = "#8b949e" if dark_mode else "#64748b"
    delta_html = f"<div style='color: #09ab3b; font-size: 14px; margin-top: 2px;'>{delta}</div>" if delta else "<div style='height: 21px;'></div>"
    html_content = f"""
    <div style="text-align: center; font-family: 'Microsoft YaHei', sans-serif; padding: 10px 0;">
        <div style="font-size: 14px; color: {title_color}; margin-bottom: 4px;">{title}</div>
        <div style="font-size: 28px; font-weight: bold; color: {text_color}; line-height: 1.2;">{value}</div>
        {delta_html}
    </div>
    """
    col_obj.markdown(html_content, unsafe_allow_html=True)


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
        top = qy[:-2, 1:-1];
        bottom = qy[2:, 1:-1];
        left = qy[1:-1, :-2];
        right = qy[1:-1, 2:]
        top_left = qy[:-2, :-2];
        top_right = qy[:-2, 2:];
        bottom_left = qy[2:, :-2];
        bottom_right = qy[2:, 2:]
        center = qy[1:-1, 1:-1]

        is_min_center = (center <= top) & (center <= bottom) & (center <= left) & (center <= right) & \
                        (center <= top_left) & (center <= top_right) & (center <= bottom_left) & (
                                    center <= bottom_right)
        is_min = np.pad(is_min_center, pad_width=1, mode='constant', constant_values=False)
        num_minima = np.sum(is_min)
        if num_minima == 0: break

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
            if not np.any(boundary_mask): continue
            ljmin = np.min(qy[boundary_mask])
            region_val = qy[region_mask][0]
            if region_val > ljmin:
                excess_water = qy[region_mask] - ljmin
                if ljmin < wall_height - 1e-5:
                    v_excess[region_mask] = excess_water
                qy[region_mask] = ljmin

        V = V_remaining + np.sum(v_excess)
        if iteration > 5000: break

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

    if 'current_h5_path' not in st.session_state: st.session_state.current_h5_path = None
    if 'prev_data_source' not in st.session_state: st.session_state.prev_data_source = "📁 上传本地数据 (.h5)"

    data_source = st.radio("选择数据来源", ["📁 上传本地数据 (.h5)", "📦 加载内置示例数据"])

    if data_source != st.session_state.prev_data_source:
        st.session_state.current_h5_path = None
        st.session_state.road_loaded = False
        st.session_state.prev_data_source = data_source

    if data_source == "📁 上传本地数据 (.h5)":
        uploaded_file = st.file_uploader("上传路面点云 (.h5)", type=['h5'])
        if uploaded_file is not None:
            if st.session_state.get('last_uploaded_name') != uploaded_file.name:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    st.session_state.current_h5_path = tmp_file.name
                st.session_state.last_uploaded_name = uploaded_file.name
        else:
            st.session_state.current_h5_path = None

    elif data_source == "📦 加载内置示例数据":
        st.info("💡 系统将直接加载项目库中内置的高保真路面点云数据，供您快速体验完整功能流程。")
        if st.button("⬇️ 一键加载内置示例数据", use_container_width=True):
            local_sample_path = os.path.join("assets", "sample_data.h5")
            if os.path.exists(local_sample_path):
                st.session_state.current_h5_path = local_sample_path
                st.success("✅ 内置示例数据加载成功！请在下方选择路段并生成地形。")
            else:
                st.error(f"❌ 找不到内置示例文件！\n\n请检查代码所在的目录下是否存在 `{local_sample_path}` 这个文件。")

    note_bg = "rgba(0, 151, 230, 0.1)" if st.session_state.dark_mode else "rgba(0, 151, 230, 0.05)"
    note_color = "#c9d1d9" if st.session_state.dark_mode else "#2c3e50"
    st.markdown(
        f"""
        <div style="background-color: {note_bg}; padding: 12px; border-radius: 8px; font-size: 13px; color: {note_color}; margin-top: 15px; margin-bottom: 15px; border-left: 4px solid #0097e6; line-height: 1.5;">
        <b>🔒 数据隐私与安全承诺</b><br>
        本系统采用严格的<b>“无痕运算”</b>机制。您上传的业务文件仅在内存及系统的沙盒临时目录中即时推演，页面刷新或会话结束后数据即自动销毁。
        </div>
        """, unsafe_allow_html=True
    )

    start_segment = None
    num_blocks = 1

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
    runoff_coefficient = st.slider("路面径流滞留系数", 0.01, 1.00, 0.01, step=0.01)
    wall_margin = st.slider("边缘挡水墙裕量 (mm)", 0.0, 2.0, 1.0, step=0.5)
    anim_frames = st.slider("仿真动画帧数", 1, 10, 5)
    max_h_step = st.slider("最大水位爬升步长(mm)", 0.000, 0.100, 0.010, step=0.002, format="%.3f")
    btn_run_sim = st.button("🌊 2. 开始动态降雨推演", type="primary", use_container_width=True,
                            disabled=not st.session_state.road_loaded)


# ==========================================
# 主界面逻辑处理：状态分发机制
# ==========================================
st.divider()

col_left, col_right = st.columns([1, 1])
with col_left:
    col_3d_title, col_3d_toggle = st.columns([3, 2])
    with col_3d_title:
        st.subheader("🧊 三维空间水膜演化")

    with col_3d_toggle:
        st.write("")  # 补一个空行，为了让开关在垂直方向上和左侧的标题下沉对齐
        show_3d_grid = st.toggle("🌐 开启物理网格线", value=True)
    plot3d_container = st.empty()
with col_right:
    st.subheader("📈 典型横截面 (二维波谷填充)")
    plot2d_container = st.empty()
    metrics_container = st.empty()

export_container = st.empty()

is_dark = st.session_state.dark_mode

# ------------------------------------------
# 状态 1：缺省骨架渲染（未上传数据）
# ------------------------------------------
if not st.session_state.road_loaded:
    with metrics_container.container():
        c1, c2, c3, c4 = st.columns(4)
        render_centered_metric(c1, "当前降雨进度", "-- mm", dark_mode=is_dark)
        render_centered_metric(c2, "最大积水深度", "-- mm", dark_mode=is_dark)
        render_centered_metric(c3, "积水覆盖率", "-- %", dark_mode=is_dark)
        render_centered_metric(c4, "地形最大高差", "-- mm", dark_mode=is_dark)

    fig_empty = go.Figure().update_layout(
        # title=dict(text="请先在左侧配置并解析路面地形...", font=dict(color="#c9d1d9" if is_dark else "#2c3e50")),
        scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=2.5, z=0.5)),
        margin=dict(l=0, r=0, b=0, t=30), autosize=True, height=420,
        template="plotly_dark" if is_dark else "plotly_white",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    plot3d_container.plotly_chart(fig_empty, use_container_width=True)

    fig_empty_2d = go.Figure().update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text="等待数据导入...", x=-0.5, y=-0.5, showarrow=False, font=dict(color="gray", size=16))],
        template="plotly_dark" if is_dark else "plotly_white",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, b=0, t=30), height=350
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
            fine_matrix_crop, effective_rain_m, wall_margin / 1000.0, max_h_step / 1000.0
        )
        final_depth_crop = depth_crop
        final_surf_crop = surf_crop

        area_ratio = st.session_state.matrix_full.shape[1] / st.session_state.matrix_crop.shape[1]
        theoretical_total_size = depth_crop.size * area_ratio

        coverage = (np.count_nonzero(depth_crop) / theoretical_total_size) * 100
        st.session_state.coverage_history[round(current_rain, 2)] = coverage

        simulation_history.append({
            "降雨阶段": f"{step}/{anim_frames}", "当前降雨量 (mm)": round(current_rain, 2),
            "有效滞留水深 (mm)": round(current_rain * runoff_coefficient, 2),
            "最大积水深度 (mm)": round(np.max(depth_crop) * 1000.0, 2),
            "平均积水深度 (mm)": round(np.mean(depth_crop) * 1000.0, 2), "积水覆盖率 (%)": f"{coverage:.2f}%"
        })

        with metrics_container.container():
            c1, c2, c3, c4 = st.columns(4)
            render_centered_metric(c1, "当前降雨进度", f"{current_rain:.1f} mm",
                                   delta=f"↑ {target_rainfall / anim_frames:.1f} mm", dark_mode=is_dark)
            render_centered_metric(c2, "最大积水深度", f"{np.max(depth_crop) * 1000.0:.2f} mm", dark_mode=is_dark)
            render_centered_metric(c3, "积水覆盖率", f"{(np.count_nonzero(depth_crop) / depth_crop.size) * 100:.1f} %",
                                   dark_mode=is_dark)
            render_centered_metric(c4, "地形最大高差", f"{(np.max(matrix_crop) - np.min(matrix_crop)) * 1000.0:.1f} mm",
                                   dark_mode=is_dark)

        plot3d_container.plotly_chart(
            create_3d_figure(fine_matrix_crop, surf_crop, depth_crop, fine_dx_mm, show_grid=show_3d_grid,
                             dark_mode=is_dark),
            use_container_width=True, key=f"sim_frame_{step}"
        )

        row_i = int(fine_matrix_crop.shape[0] / 2)
        fig_2d_sim = plot_2d_cross_section(fine_matrix_crop, surf_crop, fine_dx_mm, row_idx=row_i, dark_mode=is_dark)
        plot2d_container.plotly_chart(fig_2d_sim, use_container_width=True, key=f"sim_2d_frame_{step}")

        progress_bar.progress(step / anim_frames)
        time.sleep(0.05)

    status_text.success(f"✅ 物理推演完成！最终降雨量达到 {target_rainfall} mm。")

    st.session_state.final_depth_crop = final_depth_crop;
    st.session_state.fine_matrix_crop = fine_matrix_crop
    st.session_state.surf_crop = final_surf_crop;
    st.session_state.fine_dx_mm = fine_dx_mm
    st.session_state.simulation_history = simulation_history;
    st.session_state.last_target_rainfall = target_rainfall

    with export_container:
        st.markdown("### 📊 动态降雨过程数据详情")
        st.dataframe(pd.DataFrame(simulation_history), use_container_width=True)

# ------------------------------------------
# 状态 3：调用缓存静默渲染
# ------------------------------------------
elif st.session_state.road_loaded and st.session_state.final_depth_crop is not None:
    fine_matrix = st.session_state.fine_matrix_crop;
    surf = st.session_state.surf_crop
    depth = st.session_state.final_depth_crop;
    f_dx = st.session_state.fine_dx_mm
    last_rain = st.session_state.last_target_rainfall;
    sim_hist = st.session_state.simulation_history
    coverage = (np.count_nonzero(depth) / depth.size) * 100

    with metrics_container.container():
        c1, c2, c3, c4 = st.columns(4)
        render_centered_metric(c1, "当前降雨进度", f"{last_rain:.1f} mm", dark_mode=is_dark)
        render_centered_metric(c2, "最大积水深度", f"{np.max(depth) * 1000.0:.2f} mm", dark_mode=is_dark)
        render_centered_metric(c3, "积水覆盖率", f"{coverage:.1f} %", dark_mode=is_dark)
        render_centered_metric(c4, "地形最大高差",
                               f"{(np.max(st.session_state.matrix_crop) - np.min(st.session_state.matrix_crop)) * 1000.0:.1f} mm",
                               dark_mode=is_dark)

    plot3d_container.plotly_chart(
        create_3d_figure(fine_matrix, surf, depth, f_dx, show_grid=show_3d_grid, dark_mode=is_dark),
        use_container_width=True)
    row_i = int(fine_matrix.shape[0] / 2)
    fig_2d_sim = plot_2d_cross_section(fine_matrix, surf, f_dx, row_idx=row_i, dark_mode=is_dark)
    plot2d_container.plotly_chart(fig_2d_sim, use_container_width=True)

    with export_container:
        st.markdown("### 📊 动态降雨过程数据详情")
        st.dataframe(pd.DataFrame(sim_hist), use_container_width=True)

# ------------------------------------------
# 状态 4：渲染干路面静态图
# ------------------------------------------
elif st.session_state.road_loaded and st.session_state.final_depth_crop is None:
    matrix_crop = st.session_state.matrix_crop
    with metrics_container.container():
        c1, c2, c3, c4 = st.columns(4)
        render_centered_metric(c1, "当前降雨进度", "0.0 mm", dark_mode=is_dark)
        render_centered_metric(c2, "最大积水深度", "0.00 mm", dark_mode=is_dark)
        render_centered_metric(c3, "积水覆盖率", "0.0 %", dark_mode=is_dark)
        render_centered_metric(c4, "地形最大高差", f"{(np.max(matrix_crop) - np.min(matrix_crop)) * 1000.0:.1f} mm",
                               dark_mode=is_dark)

    plot3d_container.plotly_chart(create_3d_figure(matrix_crop, dx_mm=dx_mm, show_grid=show_3d_grid, dark_mode=is_dark),
                                  use_container_width=True)
    row_i = min(int(matrix_crop.shape[0] / 2), matrix_crop.shape[0] - 1)
    fig_2d = plot_2d_cross_section(matrix_crop, None, dx_mm, row_idx=row_i, dark_mode=is_dark)
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

            grid_c = "#30363d" if is_dark else "#e2e8f0"
            font_c = "#c9d1d9" if is_dark else "#2c3e50"
            fig_trend.update_layout(
                template="plotly_dark" if is_dark else "plotly_white",
                xaxis_title="设定的总降雨量 (mm)", yaxis_title="路面积水覆盖率 (%)", height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor=grid_c, zeroline=False),
                yaxis=dict(showgrid=True, gridcolor=grid_c, zeroline=False, rangemode='tozero'),
                font=dict(family="微软雅黑, Arial", size=16, color=font_c)
            )
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
            decision = dynamic_decision(risk_level_matrix, area_ratio)

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

        risk_fig = risk_heatmap(risk_score_matrix)

        # 热力图设置透明背景
        risk_fig.update_layout(
            template="plotly_dark" if is_dark else "plotly_white",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#c9d1d9" if is_dark else "#2c3e50")
        )

        depth_matrix = st.session_state.final_depth_crop * 1000.0
        area_ratio = st.session_state.matrix_full.shape[1] / st.session_state.matrix_crop.shape[1]

        regions = extract_high_risk_regions(
            risk_score_matrix=risk_score_matrix,
            depth_matrix=depth_matrix,
            fine_dx_mm=st.session_state.fine_dx_mm,
            area_ratio=area_ratio
        )

        if len(regions) > 0:
            risk_fig = add_bounding_boxes(risk_fig, regions)

        st.plotly_chart(risk_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("🛠️ 高危区靶向处治与工程预算最优方案比选")

        if len(regions) > 0:
            st.info(
                f"📍 **智能巡检系统报告:** 在当前路段中成功识别出 **{len(regions)}** 个独立滑水高危核心区，已自动为您生成靶向处治匹配与经济性分析表。")

            df_plan, budget_summary = generate_plan_and_budget(
                regions=regions,
                risk_score_matrix_shape=risk_score_matrix.shape,
                fine_dx_mm=st.session_state.fine_dx_mm,
                area_ratio=area_ratio
            )

            st.dataframe(df_plan, hide_index=True, use_container_width=True)

            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("🤖 智能靶向刻槽方案总估价", f"¥ {budget_summary['smart_cost']:,.1f}", "包含500元无人机巡检费",
                          delta_color="inverse")
            col_b2.metric("🚜 传统全域铣刨重铺总估价", f"¥ {budget_summary['trad_cost']:,.1f}")
            col_b3.metric("💰 方案预计降低养护成本", f"{budget_summary['saving_ratio']:.1f} %", "经济效益显著")

            st.success(
                f"**最终决策建议：** 相比“发现积水即大面积铣刨”的传统盲目处治，采用 **[ 无人机高精度检测 + 目标路段靶向自动刻槽 ]** 策略可精准解决局部水膜隐患。据上表核算，本次优化策略预计可为您节约 **{budget_summary['saving_ratio']:.1f}%** 的养护工程预算！")

            # 输出报告
            render_report(
                df_plan=df_plan,
                budget_summary=budget_summary,
                target_rainfall=st.session_state.last_target_rainfall,
                num_regions=len(regions)
            )
        else:
            st.success(
                "🎉 当前设定的降雨环境与路面状态下，暂未检出面积足够大且深度的严重积水高危区 (A/B级)，无需启动强制工程处治程序。")
else:
    st.info("💡 请先在上方执行“动态降雨推演”，生成水膜分布数据后，即可在此处运行风险评估。")