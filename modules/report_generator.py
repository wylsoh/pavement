# report_generator.py
import streamlit as st
import pandas as pd
from datetime import datetime

def render_report_download_button(df_plan, budget_summary, target_rainfall, num_regions):
    """
    根据传入的处治方案、预算汇总和环境参数，生成专业的业务审计报告文本，
    并在前端渲染一个下载按钮。
    """
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = f"""==================================================
高速公路路面积水智能识别与处治评估报告
==================================================
报告生成时间：{report_time}

[ 1. 核心评估参数 ]
--------------------------------------------------
- 分析路段名称：智能检测路段 (示例)
- 设定的目标降雨量：{target_rainfall} mm
- 检出独立高危病害区(A/B级)：{num_regions} 处

[ 2. 靶向处治工程方案明细 ]
--------------------------------------------------
"""
    # 将 DataFrame 转换为文本格式追加到报告中
    report_content += df_plan.to_string(index=False)

    # 3. 追加预算审计结果
    report_content += f"""

[ 3. 养护工程预算审计 ]
--------------------------------------------------
- 传统全域大面积铣刨重铺预估造价：¥ {budget_summary['trad_cost']:,.2f}
- 智能靶向微创自动刻槽预估总造价：¥ {budget_summary['smart_cost']:,.2f} (包含单次无人机智能巡检费500元)

>> 核心经济效益结论：
>> 采用本次智能生成的靶向处治策略，预计可为本路段节约养护资金达 【 {budget_summary['saving_ratio']:.1f}% 】。

==================================================
* 本报告由 [高速公路路面积水智能识别与处治技术系统] 自动生成。
* 核心算法：流体力学水膜推演 & 车辆动力学滑水评估 & CV高危框选
==================================================
"""

    # 渲染下载按钮
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 一键导出《路面智能处治审计报告》(业务留痕)",
            data=report_content,
            file_name=f"水患处治与造价审计报告_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )