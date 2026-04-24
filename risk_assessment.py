import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go


# ==========================================
# 1. 临界滑水速度计算模块
# ==========================================
def calc_critical_hydroplaning_speed(water_depth_mm):
    """
    根据水膜厚度计算临界滑水速度 (km/h)
    参考: 采用Anderson经验公式/拟合曲线作为修正LuGre模型的极速平替
    """
    # 防止除以 0，限制最小水膜计算阈值
    wd = np.clip(water_depth_mm, 1e-3, None)

    # 采用 Anderson 模型形式 (v = 96.84 * d^(-0.259))
    # 在水膜 1~10mm 范围内与复杂流体力学模型趋势高度吻合
    vc = 96.84 * np.power(wd, -0.259)

    # 物理限幅：当水膜极其微小时（如 < 1mm），轮胎与路面有效接触，不会发生完全滑水
    vc[water_depth_mm < 1.0] = 150.0  # 设为绝对安全速度
    vc = np.clip(vc, 0, 150.0)

    return vc


# ==========================================
# 2. 全域滑水风险概率与等级评估模块
# ==========================================
def evaluate_hydroplaning_risk(water_depth_mm, speed_mean=75.3, speed_std=5.7):
    """
    基于正态分布的滑水风险量化评价 (对应赵鸿铎论文的蒙特卡洛/概率统计法)
    :param speed_mean: 该路段车辆平均车速 (km/h)
    :param speed_std: 该路段车辆车速标准差 (km/h)
    """
    print("⏳ 正在进行全域滑水风险概率计算与等级划分...")

    # 1. 获得临界滑水速度矩阵
    vc_matrix = calc_critical_hydroplaning_speed(water_depth_mm)

    # 2. 计算滑水概率：P(实际车速 >= 临界滑水速度)
    # 利用正态分布的累积分布函数(CDF)计算右尾概率
    prob_hydroplaning = 1.0 - norm.cdf(vc_matrix, loc=speed_mean, scale=speed_std)

    # 3. 风险等级划分 (A, B, C, D, E)
    risk_level = np.full(water_depth_mm.shape, 'E', dtype=object)
    risk_level[(prob_hydroplaning >= 1e-6) & (prob_hydroplaning < 1e-5)] = 'D'
    risk_level[(prob_hydroplaning >= 1e-5) & (prob_hydroplaning < 1e-4)] = 'C'
    risk_level[(prob_hydroplaning >= 1e-4) & (prob_hydroplaning < 1e-3)] = 'B'
    risk_level[prob_hydroplaning >= 1e-3] = 'A'

    # 为了方便后续画图，将 A-E 映射为 5-1 的数值矩阵
    risk_score = np.zeros_like(water_depth_mm, dtype=int)
    risk_score[risk_level == 'E'] = 1
    risk_score[risk_level == 'D'] = 2
    risk_score[risk_level == 'C'] = 3
    risk_score[risk_level == 'B'] = 4
    risk_score[risk_level == 'A'] = 5

    return prob_hydroplaning, risk_level, risk_score


# ==========================================
# 3. 动态管控与处治决策生成器
# ==========================================
def dynamic_decision_making(risk_level_matrix):
    """
    基于风险图谱进行动态决策，输出管控建议和刻槽处治方案
    """
    total_area = risk_level_matrix.size
    a_ratio = np.sum(risk_level_matrix == 'A') / total_area
    b_ratio = np.sum(risk_level_matrix == 'B') / total_area
    high_risk_ratio = a_ratio + b_ratio

    decision = {
        "overall_status": "安全 (Low Risk)",
        "traffic_control": "路面状态良好，保持常规监控",
        "maintenance_action": "暂无需特殊物理处治",
        "high_risk_area_ratio": f"{high_risk_ratio * 100:.2f}%"
    }

    if high_risk_ratio > 0.05:  # 中高风险区域超过 5%
        decision["overall_status"] = "危险！(High Risk A/B 级警告)"
        decision["traffic_control"] = "立即联动VMS可变情报板：发布【雨天积水，限速 60 km/h】预警，并建议封闭积水严重车道。"
        decision["maintenance_action"] = "触发【靶向微创处治】：提取A级风险坐标，引导自动化刻槽设备进行排水优化（推荐刻槽宽度 1.0 ~ 2.0cm，角度 30 ~ 60°）。"
    elif np.sum(risk_level_matrix == 'C') / total_area > 0.1:  # 中等风险
        decision["overall_status"] = "关注 (Medium Risk C 级)"
        decision["traffic_control"] = "雨天路滑，下发【保持车距，限速 80 km/h】提示。"
        decision["maintenance_action"] = "纳入重点养护观测路段，评估是否由于车辙加深导致滞水。"

    return decision


# ==========================================
# 4. 可视化：生成 2D 风险热力图
# ==========================================
def render_risk_heatmap(risk_score_matrix):
    """
    渲染面域级滑水风险图谱，直观展示 A-E 级危险区域
    """
    print("⏳ 正在生成全域滑水风险等级图谱...")
    # 自定义离散颜色映射 (绿->黄->橙->红->深红)
    colorscale = [
        [0.0, "rgb(0, 255, 0)"],  # 1: E级 低风险
        [0.25, "rgb(173, 255, 47)"],  # 2: D级 中低风险
        [0.5, "rgb(255, 255, 0)"],  # 3: C级 中风险
        [0.75, "rgb(255, 165, 0)"],  # 4: B级 中高风险
        [1.0, "rgb(255, 0, 0)"]  # 5: A级 高风险
    ]

    fig = go.Figure(data=go.Heatmap(
        z=risk_score_matrix,
        colorscale=colorscale,
        zmin=1, zmax=5,
        colorbar=dict(
            title="风险等级",
            tickvals=[14 - 18],
            ticktext=['E(极低)', 'D(较低)', 'C(中等)', 'B(较高)', 'A(极高)']
        )
    ))

    fig.update_layout(
        title="湿滑路面全域滑水风险量化分布图谱 (A-E级)",
        xaxis_title="横向坐标 (车道跨度)",
        yaxis_title="纵向坐标 (行驶里程)",
        yaxis=dict(autorange="reversed")
    )
    return fig