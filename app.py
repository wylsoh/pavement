import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from scipy import signal

from pavement_tools import stitch_road_from_h5

# ==========================================
# 页面基础设置
# ==========================================
st.set_page_config(
    page_title="路面纹理磨光分析软件",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 核心算法模块 1: 2D 图像纹理分析 (原有逻辑保留)
# ==========================================
def analyze_texture(image_file, elev_min, elev_max, q_min, q_max):
    """解析纹理图像，计算高程矩阵、HRMS及一维功率谱曲线"""
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    elev = (img.astype(float) / 255.0) * (elev_max - elev_min) + elev_min

    dx = 0.1
    Ny, Nx = elev.shape
    F = np.fft.fft2(elev)
    F_shift = np.fft.fftshift(F)
    Cq_2d = np.abs(F_shift) ** 2 / (Nx * Ny * dx ** 2)

    fx = np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(Ny, d=dx))
    FX, FY = np.meshgrid(fx, fy)
    q = np.sqrt(FX ** 2 + FY ** 2)

    idx = (q >= q_min) & (q <= q_max)
    dfx = 1.0 / (Nx * dx)
    dfy = 1.0 / (Ny * dx)
    integral = np.sum(Cq_2d[idx]) * dfx * dfy
    hrms_um = np.sqrt(np.abs(integral)) * 1000

    q_flat = q.flatten()
    Cq_flat = Cq_2d.flatten()
    q_plot_bins = np.logspace(np.log10(max(q_min, 0.01)), np.log10(q_max), 80)
    bin_indices = np.digitize(q_flat, q_plot_bins)

    q_plot, Cq_plot = [], []
    for i in range(1, len(q_plot_bins)):
        mask = bin_indices == i
        if np.any(mask):
            q_plot.append(np.sqrt(q_plot_bins[i - 1] * q_plot_bins[i]))
            Cq_plot.append(np.mean(Cq_flat[mask]))

    return elev, hrms_um, q_plot, Cq_plot


def plot_3d_surface(elev, title, colorscale="Cividis"):
    if max(elev.shape) > 300:
        step = max(elev.shape) // 300
        elev = elev[::step, ::step]

    fig = go.Figure(data=[go.Surface(z=elev, colorscale=colorscale)])
    fig.update_layout(
        title=dict(text=title, font=dict(family="Times New Roman", size=18, color="black")),
        font=dict(family="Times New Roman", size=12, color="black"),
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(aspectmode='data')
    )
    return fig


# ==========================================
# 核心算法模块 2: 1D 轮廓数据 PSD 分析 (新增优化)
# ==========================================
def analyze_profile_psd(df, fs):
    """
    接收 DataFrame 数据，使用 Welch 方法计算功率谱密度 (PSD)
    """
    # 提取第一列数据作为轮廓高程
    profile_data = df.iloc[:, 0].dropna().values

    if profile_data.ndim != 1 or len(profile_data) < 2:
        raise ValueError("数据提取失败：需确保Excel/CSV首列为有效的数值型数据！")

    # (1) 去趋势（消除基线倾斜）
    profile_detrend = signal.detrend(profile_data)

    # (2) 计算功率谱密度（Welch 方法）
    win_len = max(256, len(profile_data) // 8)
    f, psd = signal.welch(profile_detrend, fs=fs, window='hann',
                          nperseg=win_len, noverlap=win_len // 2)

    # 过滤掉 f=0 的点，避免在对数坐标下绘图报错
    valid = f > 0
    return profile_data, f[valid], psd[valid]


# ==========================================
# 界面布局与交互设计
# ==========================================
st.title("🛣️ 智慧路面纹理与磨光分析系统")
st.markdown("---")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 参数配置中心")

    st.subheader("通用物理参数")
    A = st.number_input("材料磨损系数 A", value=0.01, format="%.3f")
    K = st.number_input("磨损率常数 K", value=1e-15, format="%.2e")
    p = st.number_input("轮胎接地压力 p (Pa)", value=700000)
    v_kmh = st.number_input("行驶速度 v (km/h)", value=60)
    v_ms = v_kmh / 3.6

# 采用 Tabs 分页设计，融合两种输入模式
tab1, tab2 = st.tabs(["🖼️ 模式一：2D 图像区域分析 (FFT)", "📈 模式二：1D 轮廓高程分析 (Welch)"])

# ---------------------------------------------------------
# Tab 1: 原有的 2D 图像分析逻辑
# ---------------------------------------------------------
with tab1:
    st.header("📂 导入 2D 图像数据")
    with st.expander("配置图像频域参数", expanded=False):
        q_min = st.slider("最小空间频率 q_min (rad/mm)", 0.1, 1.0, 0.1, 0.05)
        q_max = st.slider("最大空间频率 q_max (rad/mm)", 1.0, 20.0, 10.0, 0.5)

    col1, col2 = st.columns(2)
    with col1:
        lt_img = st.file_uploader("上传【轮迹带】图像 (PNG/JPG)", type=['png', 'jpg'], key="img_lt")
    with col2:
        nt_img = st.file_uploader("上传【非轮迹带】图像 (PNG/JPG)", type=['png', 'jpg'], key="img_nt")

    if lt_img and nt_img:
        if st.button("🚀 执行 2D 图像纹理分析", use_container_width=True, type="primary", key="btn_2d"):
            with st.spinner("正在执行复杂的傅里叶变换与频域积分..."):
                lt_elev, hrms_lt, q_lt, Cq_lt = analyze_texture(lt_img, 0.18, 0.22, q_min, q_max)
                nt_elev, hrms_nt, q_nt, Cq_nt = analyze_texture(nt_img, 0.20, 0.35, q_min, q_max)

                delta_hrms = abs(hrms_lt - hrms_nt)
                N = (A * delta_hrms * 1e-6) / (K * p * v_ms) if K * p * v_ms != 0 else 0.0

                st.success("✅ 2D 图像分析计算完成！")

                # --- 量化卡片 ---
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("轮迹带粗糙度 (HRMS)", f"{hrms_lt:.2f} μm")
                m2.metric("非轮迹带粗糙度 (HRMS)", f"{hrms_nt:.2f} μm")
                m3.metric("磨光损失 (ΔHRMS)", f"{delta_hrms:.2f} μm")
                m4.metric("等效荷载次数 (N)", f"{N:,.0f} 次")

                # --- 3D 绘图 ---
                c1, c2 = st.columns(2)
                with c1: st.plotly_chart(plot_3d_surface(lt_elev, "轮迹带高程", "Plasma"), use_container_width=True)
                with c2: st.plotly_chart(plot_3d_surface(nt_elev, "非轮迹带高程", "Cividis"), use_container_width=True)

# ---------------------------------------------------------
# Tab 2: 新增的 1D Excel 轮廓分析逻辑 (整合你的 profilePSD 算法)
# ---------------------------------------------------------
with tab2:
    st.header("📂 导入 1D 轮廓数据 (Excel/CSV)")
    with st.expander("配置采样参数", expanded=True):
        fs = st.number_input("采样频率 Fs (Hz 或 点/mm)", min_value=100, value=3600, step=100,
                             help="对应你代码中的 Fs=3600，即每毫米/微米的采样点数")

    col3, col4 = st.columns(2)
    with col3:
        lt_excel = st.file_uploader("上传【轮迹带】数据 (Excel/CSV)", type=['xlsx', 'csv'], key="excel_lt")
    with col4:
        nt_excel = st.file_uploader("上传【非轮迹带】数据 (Excel/CSV)", type=['xlsx', 'csv'], key="excel_nt")

    if lt_excel and nt_excel:
        if st.button("🚀 执行 1D 轮廓功率谱分析", use_container_width=True, type="primary", key="btn_1d"):
            try:
                with st.spinner("正在解析数据并计算 Welch 功率谱..."):
                    # 根据文件后缀读取数据
                    read_func = pd.read_csv if lt_excel.name.endswith('.csv') else pd.read_excel
                    df_lt = read_func(lt_excel)
                    df_nt = read_func(nt_excel)

                    # 调用你提供算法的核心逻辑
                    prof_lt, f_lt, psd_lt = analyze_profile_psd(df_lt, fs)
                    prof_nt, f_nt, psd_nt = analyze_profile_psd(df_nt, fs)

                    st.success("✅ 1D 轮廓分析计算完成！")

                    # ==========================================
                    # 结果可视化：使用 Plotly 绘制学术风图表
                    # ==========================================
                    st.subheader("1. 表面纹理轮廓曲线")
                    fig_prof = go.Figure()
                    # 为了避免数据量过大导致浏览器卡顿，这里抽样展示
                    step = max(1, len(prof_lt) // 2000)
                    t_lt = np.arange(len(prof_lt)) / fs
                    t_nt = np.arange(len(prof_nt)) / fs

                    fig_prof.add_trace(go.Scatter(x=t_lt[::step], y=prof_lt[::step], mode='lines', name='轮迹带',
                                                  line=dict(color='blue', width=1)))
                    fig_prof.add_trace(go.Scatter(x=t_nt[::step], y=prof_nt[::step], mode='lines', name='非轮迹带',
                                                  line=dict(color='green', width=1)))

                    fig_prof.update_layout(
                        xaxis_title='位置 (mm)', yaxis_title='轮廓高度 (μm)',
                        template="simple_white", hovermode="x unified",
                        height=350, margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_prof, use_container_width=True)

                    st.subheader("2. 功率谱密度 (PSD) 对数曲线对比")
                    fig_psd_1d = go.Figure()
                    fig_psd_1d.add_trace(go.Scatter(x=f_lt, y=psd_lt, mode='lines', name='轮迹带 (Welch)',
                                                    line=dict(color='black', width=1.5)))
                    fig_psd_1d.add_trace(go.Scatter(x=f_nt, y=psd_nt, mode='lines', name='非轮迹带 (Welch)',
                                                    line=dict(color='#8B0000', dash='dashdot', width=1.5)))

                    fig_psd_1d.update_layout(
                        xaxis_title='频率 (1/mm)', yaxis_title='功率谱密度 (μm²)',
                        template="simple_white", hovermode="x unified",
                        height=450
                    )
                    # 激活双对数坐标！复刻原代码 loglog 的效果
                    fig_psd_1d.update_xaxes(type="log", showgrid=True, gridwidth=1, gridcolor='LightGray')
                    fig_psd_1d.update_yaxes(type="log", showgrid=True, gridwidth=1, gridcolor='LightGray')

                    st.plotly_chart(fig_psd_1d, use_container_width=True)

            except Exception as e:
                st.error(f"❌ 数据处理出错：{e}。请确保上传的Excel/CSV文件第一列为纯数字的高程数据！")

    else:
    # 如果两个 tab 都没有上传文件，不显示提示，因为各个 tab 内部处理了
        pass