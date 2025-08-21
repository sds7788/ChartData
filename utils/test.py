import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
import os
from typing import Dict, List, Any

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. “地狱级”密度数据 ---
def create_dense_data():
    """动态生成高密度、值接近的柱状图数据"""
    categories = [f'{i}月' for i in range(1, 13)]
    num_categories = len(categories)
    
    # 设定一个基础波动值，模拟年度趋势
    base_values = 150 + 50 * np.sin(np.linspace(0, 2 * np.pi, num_categories))
    
    series_names = ['产品线A', '产品线B', '产品线C', '产品线D', '产品线E', '产品线F']
    series_data = []
    
    # 为每个系列在基础值上增加微小、不同的扰动，制造拥挤效果
    offsets = [2, -3, 5, -1, 4, -2]
    for i, name in enumerate(series_names):
        values = [round(base + offsets[i] + np.random.uniform(-2, 2), 2) for base in base_values]
        series_data.append({'name': name, 'values': values})
        
    return {'x_categories': categories, 'y_series': series_data}

chart_data = create_dense_data()


# --- 2. 绘图函数 ---
def create_bar_chart_test_plot(data: Dict[str, Any], settings: Dict[str, Any], filename="bar_chart_dense_test.png"):
    """
    根据给定的高密度数据和 adjustText 设置来创建并保存测试图表。
    """
    print("正在生成“地狱级”密度的柱状图...")
    
    x_labels = data.get('x_categories', [])
    series_data = data.get('y_series', [])

    if not x_labels or not series_data:
        print("错误：数据不完整。")
        return

    # 为高密度图表创建一个更宽的画布
    fig, ax = plt.subplots(figsize=(24, 14))
    
    x = np.arange(len(x_labels))
    num_series = len(series_data)
    total_width = 0.8
    bar_width = total_width / num_series
    
    texts = []
    
    # 从 settings 中获取字体大小
    label_fontsize = settings.pop('label_fontsize', 8)
    
    all_values = [val for series in series_data for val in series.get('values', [])]
    y_max = max(all_values) if all_values else 0
    label_offset = y_max * 0.01

    for i, series in enumerate(series_data):
        values = series.get('values', [])
        offset = (i - (num_series - 1) / 2) * bar_width
        rects = ax.bar(x + offset, values, bar_width, label=series.get('name'))
        
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                texts.append(ax.text(
                    x=rect.get_x() + rect.get_width() / 2.0,
                    y=height + label_offset,
                    s=f'{height}',
                    ha='center',
                    va='bottom',
                    fontsize=label_fontsize # 使用可配置的字体大小
                ))

    # --- 核心：调用 adjust_text ---
    adjust_text(texts, ax=ax, **settings)

    # --- 设置图表样式 ---
    ax.set_title("“地狱级”密度柱状图 adjustText 参数效果测试", fontsize=22, pad=25)
    ax.set_ylabel("销售额（万元）", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='产品线', fontsize=12)

    ylim_factor = settings.pop('ylim_factor', 1.4)
    ax.set_ylim(top=ax.get_ylim()[1] * ylim_factor)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    
    print(f"图表已保存为: {os.path.abspath(filename)}")


# --- 3. 主程序入口 ---
if __name__ == "__main__":
    
    # #######################################################################
    # ### 主要修改区域：在这里尽情调试 adjustText 的参数！  ###
    # #######################################################################
    
    adjust_settings = {
        # --- 力的调整 (Force adjustment) ---
        # 面对高密度，需要更强的垂直排斥力来将标签推开
        "force_text": (1.2, 5),

        # 保持文本与锚点的基本斥力
        "force_points": (0.01, 5),

        # --- 扩展区域 (Expand area) ---
        "expand_text": (1.1, 1.1),
        "expand_points": (1.1, 1.1),
        
        # --- 新增：标签字体大小 ---
        # 减小字体是应对高密度的有效方法
        "label_fontsize": 11,

        # --- 迭代次数 (Iterations) ---
        # 布局更复杂，可能需要更多次迭代来找到最优解
        "lim": 1000,
        
        # --- Y轴顶部空间因子 ---
        "ylim_factor": 1.4,

        # --- 连接线/箭头样式 (Arrow properties) ---
        "arrowprops": dict(
            arrowstyle="-",
            color='black',
            lw=0.5,
            alpha=0.8 # 使用半透明让线条不那么刺眼
        )
    }

    # 调用绘图函数生成结果
    create_bar_chart_test_plot(chart_data, adjust_settings)