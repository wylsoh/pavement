import numpy as np
from scipy.stats import norm
import plotly.express as px


def evaluate_hydroplaning_risk(water_depth_mm):
    """
    结合车速分布与水膜厚度，计算全域滑水风险概率矩阵
    """
    # 设置车速分布参数 (参考限速80km/h时的实测正态分布)
    mu_v = 75.3  # 平均车速 km/h
    sigma_v = 5.7  # 车速标准差 km/h

    # 计算临界滑水速度 vc
    # 初始化一个极大的安全速度 (代表无积水区域不发生滑水)
    vc_matrix = np.full_like(water_depth_mm, 200.0, dtype=float)

    # 提取积水区域 (>0.1mm)
    wet_mask = water_depth_mm > 0.1

    # 江守一郎公式关系曲线 (水膜越厚，临界速度越低)
    # 在 2mm 时约为 110km/h，在 4mm 时约为 80km/h，在 10mm 时约为 57km/h
    vc_matrix[wet_mask] = 145.0 * (water_depth_mm[wet_mask] ** -0.4)

    # 计算滑水概率 P(v > vc) = 1 - CDF(vc)
    # 使用SciPy的生存函数计算车速超越临界速度的概率
    prob_matrix = norm.sf(vc_matrix, loc=mu_v, scale=sigma_v)

    # 根据概率划分风险等级
    risk_level_matrix = np.full_like(prob_matrix, 'E', dtype=object)
    risk_score_matrix = np.zeros_like(prob_matrix, dtype=int)

    # A(高风险): P >= 10^-1 (10%)
    mask_A = prob_matrix >= 1e-1
    # B(中高风险): 10^-3 <= P < 10^-1
    mask_B = (prob_matrix >= 1e-3) & (prob_matrix < 1e-1)
    # C(中风险): 10^-5 <= P < 10^-3
    mask_C = (prob_matrix >= 1e-5) & (prob_matrix < 1e-3)
    # D(中低风险): 10^-7 <= P < 10^-5
    mask_D = (prob_matrix >= 1e-7) & (prob_matrix < 1e-5)
    # E(低风险): P < 10^-7 默认已是 E 级

    risk_level_matrix[mask_A] = 'A'
    risk_score_matrix[mask_A] = 4
    risk_level_matrix[mask_B] = 'B'
    risk_score_matrix[mask_B] = 3
    risk_level_matrix[mask_C] = 'C'
    risk_score_matrix[mask_C] = 2
    risk_level_matrix[mask_D] = 'D'
    risk_score_matrix[mask_D] = 1

    return prob_matrix, risk_level_matrix, risk_score_matrix


def dynamic_decision(risk_level_matrix, area_ratio=1.0):
    """
    根据面域风险分布生成动态决策
    """
    total_pixels = risk_level_matrix.size * area_ratio
    # 统计高危区域 (A级 和 B级) 的面积占比
    high_risk_pixels = np.sum((risk_level_matrix == 'A') | (risk_level_matrix == 'B'))
    high_risk_ratio = high_risk_pixels / total_pixels

    if high_risk_ratio > 0.05:  # 超过 5% 的面积处于高危
        status = "危险 (全域或大面积滑水隐患)"
        traffic = "立即实施限速管控 (建议降至 60 km/h 以下)，或开启车道级封闭指引。"
        maint = "路面可能存在贯通性车辙或严重排水不畅，需重点排查病害区域，考虑铣刨重铺。"
    elif high_risk_ratio > 0.01:  # 1% ~ 5% 的高危面积
        status = "关注 (局部存在高危积水坑槽)"
        traffic = "建议开启雨天安全预警情报板，提示“前方局部积水，注意方向盘跑偏”。"
        maint = "关注积水最深的局部坑槽或微车辙，雨后建议安排日常局部修补。"
    else:
        status = "安全 (路面排水状况良好)"
        traffic = "路段整体处于低风险等级，维持正常限速 (80 km/h)。"
        maint = "暂无特殊养护需求，按常规周期巡检即可。"

    return {
        "overall_status": status,
        "high_risk_area_ratio": f"{high_risk_ratio * 100:.2f}%",
        "traffic_control": traffic,
        "maintenance_action": maint
    }


def risk_heatmap(risk_score_matrix):
    """
    绘制风险热力图
    """
    # 反转 Y 轴使其与路面图像的朝向一致
    transposed_matrix = risk_score_matrix.T
    fig = px.imshow(
        transposed_matrix,
        color_continuous_scale=[
            (0.0, "green"),  # E级: 绿
            (0.25, "lime"),  # D级: 浅绿
            (0.5, "yellow"),  # C级: 黄
            (0.75, "orange"),  # B级: 橙
            (1.0, "red")  # A级: 红
        ],
        zmin=0, zmax=4,
        labels={'color': '风险等级 (0=E, 4=A)'},
        # origin='lower'
    )

    fig.update_layout(
        xaxis_title='纵向行驶距离 (采样点)',
        yaxis_title='横向物理宽度 (采样点)',
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(
            tickvals=[0, 1, 2, 3, 4],
            ticktext=['E (低)', 'D (中低)', 'C (中)', 'B (中高)', 'A (高)']
        )
    )
    return fig