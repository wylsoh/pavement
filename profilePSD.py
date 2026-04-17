# 功率谱计算

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# 设置中文显示（优先使用微软雅黑，备选黑体）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

## ===================== 1. 数据准备 =====================
# 采样频率 (Hz)，即每毫米/微米的采样点数
fs = 3600

try:
    # 读取 Excel 数据 (替换为你的实际路径)
    # 对应原代码：xlsread('...','Sheet1','c1:c3600')
    df = pd.read_excel(r'E:\code_python\PavementAnalyze\data\表面纹理及功率谱.xlsx', sheet_name='Sheet1', usecols="C", nrows=3600)
    profile_data = df.iloc[:, 0].values

    # csv_path = 'output/test_road_1d_profile.csv'
    # df = pd.read_csv(csv_path)
    #
    # # 提取 'Elevation' 这一列作为一维轮廓数据
    # profile_data = df['Elevation'].values
    # print(f"✅ 成功读取拼接轮廓数据！共包含 {len(profile_data)} 个采样点。")
except Exception as e:
    print(f"数据读取失败: {e}，将生成模拟数据进行演示。")
    t = np.linspace(0, 1, fs, endpoint=False)
    profile_data = 0.5 * np.sin(2 * np.pi * 5 * t) + 0.2 * np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn(len(t))

# 数据有效性检查
if profile_data.ndim != 1 or len(profile_data) < 2:
    raise ValueError("输入的轮廓数据必须是长度≥2的一维向量！")

## ===================== 2. 功率谱密度（PSD）计算 =====================
# (1) 去趋势（消除基线倾斜）
profile_detrend = signal.detrend(profile_data)

# (2) 计算功率谱密度（使用 Welch 方法）
# win_len 对应原代码：max(256, floor(L/8))
win_len = max(256, len(profile_data) // 8)

# nperseg: 窗长; noverlap: 重叠点数; window: 窗函数类型
f, pxx = signal.welch(profile_detrend, fs=fs, window='hann',
                      nperseg=win_len, noverlap=win_len // 2)

# 在 scipy.signal.welch 中，返回的 pxx 已经是功率谱密度 (V^2/Hz)
# 对应原代码：PSD = Pxx / Fs;
psd = pxx

## ===================== 3. 绘图 =====================
plt.figure(figsize=(10, 8), facecolor='white')

# 子图1：表面轮廓曲线
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(profile_data)), profile_data, color='blue', linewidth=1.2)
plt.xlabel('位置 / mm')
plt.ylabel('轮廓高度 / μm')
plt.title('表面纹理轮廓曲线', fontweight='bold')
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# 子图2：功率谱密度曲线（双对数坐标）
plt.subplot(2, 1, 2)
plt.loglog(f, psd, color='red', linewidth=1.2)
plt.xlabel('频率 / 1/mm')
plt.ylabel('功率谱密度 / μm$^2$')
plt.title('表面轮廓功率谱密度（PSD）曲线', fontweight='bold')
plt.grid(True, which='both', linestyle='--', alpha=0.5)

plt.suptitle('表面纹理轮廓及功率谱密度分析', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.savefig('表面纹理轮廓及功率谱密度分析.png')
plt.show()