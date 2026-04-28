# treatment_decision.py
import numpy as np
import pandas as pd
from scipy.ndimage import label, find_objects
import plotly.graph_objects as go


def extract_high_risk_regions(risk_score_matrix, depth_matrix, fine_dx_mm, area_ratio=1.0):
    """
    通过连通域算法，识别图谱中的高危区域(A/B级)，提取包围框和区域特征。
    """
    # 风险评分 >= 4 视为高危区域 (A级/B级)
    high_risk_mask = risk_score_matrix >= 4

    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(high_risk_mask, structure=structure)

    regions = []
    if num_features == 0:
        return regions

    objects = find_objects(labeled_array)

    for i, obj in enumerate(objects):
        region_mask = (labeled_array == (i + 1))

        pixel_count = np.sum(region_mask)
        phys_area = pixel_count * ((fine_dx_mm / 1000.0) ** 2) * area_ratio

        # 过滤掉极小的噪点 (小于0.2平米不作为独立病害)
        if phys_area < 0.2:
            continue

        region_depths = depth_matrix[region_mask]
        max_d = np.max(region_depths)
        avg_d = np.mean(region_depths)

        # 提取网格坐标切片 (原矩阵的行列索引)
        y_slice, x_slice = obj
        ymin, ymax = y_slice.start, y_slice.stop
        xmin, xmax = x_slice.start, x_slice.stop

        regions.append({
            "id": i + 1,
            "ymin": ymin, "ymax": ymax,
            "xmin": xmin, "xmax": xmax,
            "area_m2": phys_area,
            "max_depth_mm": max_d,
            "avg_depth_mm": avg_d
        })

    return regions


def add_bounding_boxes_to_fig(fig, regions):
    """
    风险热力图上叠加高危区域的边界红框。
    """
    for reg in regions:
        # 同步转置逻辑
        x0, x1 = reg['ymin'], reg['ymax']
        y0, y1 = reg['xmin'], reg['xmax']

        # 在图上添加矩形虚线框
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="#FF0000", width=3, dash="dash"),
            fillcolor="rgba(255, 0, 0, 0.15)",
            layer="above"
        )

        # 添加区域标签
        fig.add_annotation(
            x=(x0 + x1) / 2, y=y1,
            text=f"⚠️ 核心高危区-{reg['id']}",
            showarrow=False,
            yshift=15,
            font=dict(color="#FFFFFF", size=12, weight="bold"),
            bgcolor="#cc0000",
            borderpad=3
        )
    return fig


def generate_treatment_plan_and_budget(regions, risk_score_matrix_shape, fine_dx_mm, area_ratio):
    """
    基于各个独立高危区域的面积和深度，进行分区域报价和方案比选。
    """
    INSPECTION_COST = 500.0  # 无人机巡检单次基准费
    GROOVE_UNIT_PRICE = 18.0  # 靶向自动刻槽单价 (元/m²)
    MILLING_UNIT_PRICE = 90.0  # 传统微铣刨重铺单价 (元/m²)
    TRENCH_UNIT_PRICE = 120.0  # 横向截水沟单价 (元/m²)

    plan_details = []
    total_smart_cost = INSPECTION_COST
    total_trad_cost = 0.0

    # 获取整条车道的真实物理宽度 (m)
    total_lane_width = risk_score_matrix_shape[1] * (fine_dx_mm / 1000.0) * area_ratio

    for reg in regions:
        area = reg["area_m2"]
        max_d = reg["max_depth_mm"]

        # 靶向智能处治面积：只针对真实水坑面积外扩30%作业面
        smart_treat_area = area * 1.3

        # 传统全域处治面积：受影响纵向长度 × 全车道宽度
        affected_length = (reg["ymax"] - reg["ymin"]) * (fine_dx_mm / 1000.0)
        # 传统工程中，单次铣刨重铺的纵向长度通常不低于 5 米
        if affected_length < 5.0:
            affected_length = 5.0
        trad_treat_area = affected_length * total_lane_width

        # 方案决策
        if max_d < 3.0:
            tech = "浅层靶向微创刻槽"
            params = "宽1.0cm, 距4.0m, 角30°"
            smart_unit_price = GROOVE_UNIT_PRICE
        elif max_d < 8.0:
            tech = "中深层自动刻槽"
            params = "宽1.5cm, 距2.5m, 角45°"
            smart_unit_price = GROOVE_UNIT_PRICE
        else:
            tech = "深水区复合处治"
            params = "宽2.0cm, 距1.2m, 增设横截沟"
            smart_unit_price = GROOVE_UNIT_PRICE + TRENCH_UNIT_PRICE * 0.3

        smart_cost = smart_treat_area * smart_unit_price
        trad_cost = trad_treat_area * MILLING_UNIT_PRICE

        total_smart_cost += smart_cost
        total_trad_cost += trad_cost

        plan_details.append({
            "高危区编号": f"高危区-{reg['id']}",
            "最大积水深度": f"{max_d:.2f} mm",
            "真实积水面积(m²)": f"{area:.2f}",
            "智能推荐靶向工艺": tech,
            "自动刻槽设计参数": params,
            "靶向智能造价": f"¥ {smart_cost:.1f}",
            "传统全面铣刨造价": f"¥ {trad_cost:.1f}"
        })

    df_plan = pd.DataFrame(plan_details)

    saving_ratio = (total_trad_cost - total_smart_cost) / total_trad_cost * 100 if total_trad_cost > 0 else 0

    summary = {
        "smart_cost": total_smart_cost,
        "trad_cost": total_trad_cost,
        "saving_ratio": saving_ratio
    }

    return df_plan, summary