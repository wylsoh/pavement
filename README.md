# 🌧️ 未雨绸缪——高速公路路面积水智能识别与处治系统

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Graphing-3F4F75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **项目网址体验:** [https://pavement-water-film.streamlit.app/](https://pavement-water-film.streamlit.app/)

## 📖 项目简介

高速公路路面雨天滞水存在极大的行车安全隐患（易引发“水漂”事故）。本项目提出“自动感知-精准识别-快速处治”的技术路线，基于无人机点云数据重构高精度 3D 路表形貌，结合流体力学（填洼模拟算法）进行动态水膜推演，并量化全域滑水风险。最后，系统通过多目标优化，自动生成兼顾排水与耐久性的“靶向自动刻槽”处治方案及工程预算。

## ✨ 核心功能特性

- 🗺️ **3D 高精度路表重构**：支持 `.h5` 点云高程数据极速读取，并进行局部缺陷修复与去零裁剪，实现沉浸式三维地形可视化。
- 🌊 **动态降雨流体推演**：引入多尺度“集水区水满则溢”算法，支持自定义降雨量、排水折减系数，分步推演路面积水过程，实时统计积水覆盖率。
- 🚨 **全域滑水风险评估**：内置车辆动力学模型（江守一郎公式），结合车速正态分布与水膜厚度，生成 A~E 级的多级滑水风险面域热力图。
- 🛠️ **智能排水处治与预算决策**：采用 CV 连通域算法精准框选高危病害区，自适应生成自动化刻槽几何参数（宽度、间距、角度），并动态对比传统微铣刨的成本经济效益（最高可节约 80% 成本）。


## 🛠️ 技术栈与依赖

本系统前端与后端均基于 Python 生态构建，核心依赖包括：

- **Web 框架**: `Streamlit`
- **数据处理**: `numpy`, `pandas`, `h5py`
- **算法计算**: `scipy` (ndimage 连通域、形态学处理、高斯滤波)
- **可视化**: `plotly`

## ⚙️ 本地部署与快速开始

1. **克隆项目仓库**
```bash
git clone https://github.com/wylsoh/pavement.git
cd pavement
```
2. **安装依赖包**

建议使用虚拟环境（如 conda/venv）。
```Bash
pip install -r requirements.txt
```
3. **运行 Streamlit 应用**

```Bash
streamlit run app.py
```
*运行后，浏览器将自动打开 [http://localhost:8501](http://localhost:8501)*。

4. **数据导入说明**

- 可在系统左侧边栏点击 “📦 一键加载内置示例数据” 快速体验。
- 可自行导入已经打包好的h5文件，或者使用仓库中的[转换代码](https://github.com/wylsoh/pavement/blob/master/assets/data_convert_to_h5.py)进行数据格式转换，支持多路段合并（目前仅支持mat->h5）

## 📂 项目结构
```Plaintext
Pavement/
├── .devcontainer/
│   └── devcontainer.json          # 容器化开发环境配置
├── assets/
│   ├── data_convert_to_h5.py      # 原始点云数据(.mat等)转.h5清洗脚本
│   └── sample_data.h5             # 系统内置的高保真路面点云体验数据
├── modules/
│   ├── risk_assessment.py         # 全域滑水风险评估与车辆动力学算法模块
│   └── treatment_decision.py      # 连通域高危病害识别与靶向处治造价测算模块
├── app.py                         # Streamlit Web 前端主程序入口
├── README.md                      # 项目说明文档
└── requirements.txt               # Python 运行环境依赖清单
```
