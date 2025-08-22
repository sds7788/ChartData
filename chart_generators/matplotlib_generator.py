# chart_generators/matplotlib_generator.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import inspect
from typing import Dict, Any, List
from datetime import datetime

# --- 依赖库导入 ---
from matplotlib.sankey import Sankey
from wordcloud import WordCloud
import mplfinance as mpf
import calmap
from pandas.plotting import parallel_coordinates
from adjustText import adjust_text
from matplotlib.projections import register_projection
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import squarify
import networkx as nx
from matplotlib.patches import Wedge # 导入 Wedge

# 假设 base_generator 在同一路径下
from .base_generator import BaseChartGenerator

# --- 所有绘图辅助函数 ---

def radar_factory(num_vars: int, frame: str = 'circle'):
    """创建一个雷达图投影。"""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    class RadarAxes(plt.PolarAxes):
        name = 'radar'
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_offset(np.pi / 2)
            self.set_theta_direction(-1)
        def set_varlabels(self, labels: List[str]):
            self.set_thetagrids(np.degrees(theta), labels)
        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines: self._close_line(line)
        def _close_line(self, line):
            x, y = line.get_data()
            if x.size > 0 and (x[0] != x[-1] or y[0] != y[-1]):
                line.set_data(np.concatenate((x, [x[0]])), np.concatenate((y, [y[0]])))
        def fill(self, *args, **kwargs):
            closed_args = [np.concatenate((arg, [arg[0]])) if isinstance(arg, (list, np.ndarray)) and len(arg) > 0 and arg[0] != arg[-1] else arg for arg in args]
            return super().fill(*closed_args, **kwargs)
    register_projection(RadarAxes)
    return theta

# --- 绘图函数实现 ---

def _draw_bar_chart(ax: plt.Axes, data: Dict[str, Any]):
    """绘制条形图。"""
    chart_data = data.get('data', {})
    x_labels = chart_data.get('x_categories', [])
    series_data = chart_data.get('y_series', [])
    if not x_labels or not series_data: return
    x = np.arange(len(x_labels))
    num_series = len(series_data)
    bar_width = 0.8 / num_series if num_series > 0 else 0.8
    for i, series in enumerate(series_data):
        offset = (i - (num_series - 1) / 2) * bar_width
        ax.bar(x + offset, series.get('values', []), bar_width, label=series.get('name'))
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    if num_series > 1: ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

def _draw_line_chart(ax: plt.Axes, data: Dict[str, Any]):
    """绘制折线图。"""
    chart_data = data.get('data', {})
    x_labels = chart_data.get('x_categories', [])
    series_data = chart_data.get('y_series', [])
    if not x_labels or not series_data: return
    markers = ['o', 's', '^', 'v', 'D']
    for i, series in enumerate(series_data):
        ax.plot(x_labels, series.get('values', []), marker=markers[i % len(markers)], linestyle='-', label=series.get('name'))
    if len(series_data) > 1: ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

def _draw_pie_chart(ax: plt.Axes, data: Dict[str, Any]):
    """绘制饼图。"""
    pie_data = data.get('data', {}).get('pie_data', {})
    labels = pie_data.get('labels', [])
    values = pie_data.get('values', [])
    if not labels or not values: return
    explode = pie_data.get('explode')
    if not explode or len(explode) != len(labels):
        explode = [0.0] * len(labels)
    ax.pie(values, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')

def _draw_donut_chart(ax: plt.Axes, data: Dict[str, Any]):
    """绘制圆环图。"""
    pie_data = data.get('data', {}).get('pie_data', {})
    labels = pie_data.get('labels', [])
    values = pie_data.get('values', [])
    if not labels or not values: return
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, 
           wedgeprops=dict(width=0.4, edgecolor='w'))
    ax.axis('equal')

# --- 【重要】修改点 2：强化玫瑰图函数 ---
def _draw_rose_chart(fig: plt.Figure, data: Dict[str, Any]):
    """绘制南丁格尔玫瑰图 (已强化，可处理两种数据格式)。"""
    chart_data = data.get('data', {})
    pie_data = chart_data.get('pie_data')
    
    labels, values = None, None
    
    # 优先使用 pie_data 结构
    if pie_data and pie_data.get('labels') and pie_data.get('values'):
        labels = pie_data['labels']
        values = pie_data['values']
    # 如果 pie_data 不可用，则尝试从 y_series 中提取数据
    elif chart_data.get('x_categories') and chart_data.get('y_series'):
        print("警告: 玫瑰图缺少 'pie_data'，尝试从 'y_series' 回退。")
        labels = chart_data['x_categories']
        # 玫瑰图通常只展示一个数据系列，我们取第一个
        values = chart_data['y_series'][0].get('values')

    if not labels or not values:
        print("错误: 无法为玫瑰图提取有效的数据。")
        return None

    ax = fig.add_subplot(111, projection='polar')
    theta = np.linspace(0.0, 2 * np.pi, len(labels), endpoint=False)
    width = np.pi / len(labels) * 0.8
    bars = ax.bar(theta, values, width=width, bottom=0.0)
    colors = plt.cm.viridis(np.array(values) / (max(values) if max(values) > 0 else 1))
    for r, bar, color in zip(values, bars, colors):
        bar.set_facecolor(color)
        bar.set_alpha(0.7)
    ax.set_xticks(theta)
    ax.set_xticklabels(labels)
    return ax
# --- 修改结束 ---

def _draw_scatter_with_error_bars(ax: plt.Axes, data: Dict[str, Any]):
    """绘制带误差棒的散点图。"""
    chart_data = data.get('data', {})
    points = chart_data.get('scatter_points', [])
    if not points: return
    for point in points:
        ax.errorbar(x=point.get('x', 0), y=point.get('y', 0), xerr=point.get('x_error'), yerr=point.get('y_error'),
                    fmt='o', capsize=5, markersize=np.sqrt(point.get('size', 50)),
                    color=plt.cm.viridis(point.get('color_value', 0) / (len(points) or 1)))
    ax.grid(True, linestyle='--', alpha=0.7)

def _draw_area_chart(ax: plt.Axes, data: Dict[str, Any]):
    """绘制面积图。"""
    chart_data = data.get('data', {})
    x_labels = chart_data.get('x_categories', [])
    series_list = data.get('data', {}).get('y_series', [])
    if not x_labels or not series_list: return
    series = series_list[0]
    ax.fill_between(x_labels, series.get('values', []), alpha=0.4)
    ax.plot(x_labels, series.get('values', []))
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# --- 【重要】修改点 1：修复雷达图函数 ---
def _draw_radar_chart(fig: plt.Figure, data: Dict[str, Any]):
    """绘制雷达图 (已修复维度不匹配问题)。"""
    radar_data = data.get('data', {}).get('radar_data', {})
    labels = radar_data.get('labels', [])
    series_data = radar_data.get('series', [])
    if not labels or not series_data: return None
    
    num_vars = len(labels)
    # 创建角度，并手动闭合
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    theta_closed = np.concatenate((theta, [theta[0]]))

    ax = fig.add_subplot(111, projection='polar')
    ax.set_xticks(theta)
    ax.set_xticklabels(labels)

    for series in series_data:
        values = series.get('values', [])
        if len(values) != num_vars:
            print(f"警告: 雷达图系列 '{series.get('name')}' 的数据点数量与标签数量不匹配，已跳过。")
            continue
        
        # 手动闭合数值数组
        values_closed = np.concatenate((values, [values[0]]))
        
        ax.plot(theta_closed, values_closed, label=series.get('name'))
        ax.fill(theta_closed, values_closed, alpha=0.25)
        
    ax.legend()
    return ax
# --- 修改结束 ---

def _draw_heatmap(ax: plt.Axes, data: Dict[str, Any]):
    """绘制热力图。"""
    heatmap_data = data.get('data', {}).get('heatmap_data', {})
    x_labels = heatmap_data.get('x_labels', [])
    y_labels = heatmap_data.get('y_labels', [])
    values = np.array(heatmap_data.get('values', []))
    if not x_labels or not y_labels or values.shape != (len(y_labels), len(x_labels)): return
    im = ax.imshow(values, cmap="viridis")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", color="w")

def _draw_line_with_confidence_interval(ax: plt.Axes, data: Dict[str, Any]):
    """绘制带置信区间的折线图。"""
    chart_data = data.get('data', {})
    x_labels = chart_data.get('x_categories', [])
    series_data = chart_data.get('y_series', [])
    if not x_labels or not series_data: return
    for series in series_data:
        values = np.array(series.get('values', []), dtype=float)
        line, = ax.plot(x_labels, values, marker='o', linestyle='-', label=series.get('name'))
        ax.fill_between(x_labels, series.get('ci_lower', values), series.get('ci_upper', values), color=line.get_color(), alpha=0.2)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

def _draw_histogram(ax: plt.Axes, data: Dict[str, Any]):
    """绘制直方图。"""
    hist_data = data.get('data', {}).get('histogram_data', {})
    ax.hist(hist_data.get('values', []), bins=hist_data.get('bins', 'auto'), alpha=0.75, edgecolor='black')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

def _draw_3d_bar_chart(fig: plt.Figure, data: Dict[str, Any]):
    """绘制3D柱状图。"""
    d_data = data.get('data', {}).get('three_d_bar_data', {})
    x_cat, y_cat = d_data.get('x_categories', []), d_data.get('y_categories', [])
    z_values = np.array(d_data.get('z_values', []))
    if not x_cat or not y_cat: return None
    ax = fig.add_subplot(111, projection='3d')
    _xx, _yy = np.meshgrid(np.arange(len(x_cat)), np.arange(len(y_cat)))
    ax.bar3d(_xx.ravel(), _yy.ravel(), np.zeros_like(z_values.ravel()), 0.8, 0.8, z_values.ravel(), shade=True)
    ax.set_xticks(np.arange(len(x_cat)))
    ax.set_xticklabels(x_cat)
    ax.set_yticks(np.arange(len(y_cat)))
    ax.set_yticklabels(y_cat)
    return ax

def _draw_statistical_plot(ax: plt.Axes, data: Dict[str, Any], plot_type: str):
    """绘制统计图 (箱形图, 小提琴图, 带状图)。"""
    stat_data = data.get('data', {}).get('statistical_data', {})
    categories = stat_data.get('categories', [])
    data_series = stat_data.get('data_series', [])
    if not categories or not data_series: return
    if plot_type == 'boxplot': ax.boxplot(data_series, labels=categories, patch_artist=True)
    elif plot_type == 'violin': 
        parts = ax.violinplot(data_series, showmedians=True)
        ax.set_xticks(np.arange(1, len(categories) + 1))
        ax.set_xticklabels(categories)
    elif plot_type == 'strip':
        for i, series in enumerate(data_series):
            ax.plot(np.random.normal(i + 1, 0.04, size=len(series)), series, 'o', alpha=0.6)
        ax.set_xticks(np.arange(1, len(categories) + 1))
        ax.set_xticklabels(categories)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

def _draw_stacked_bar_chart(ax: plt.Axes, data: Dict[str, Any]):
    """绘制堆叠条形图 (已强化)。"""
    chart_data = data.get('data', {})
    x_labels = chart_data.get('x_categories', [])
    series_data = chart_data.get('y_series', [])
    if not x_labels or not series_data: return
    
    num_categories = len(x_labels)
    values_matrix, series_names = [], []

    # 筛选并对齐数据，确保数据健壮性
    for series in series_data:
        values = series.get('values', [])
        if len(values) == num_categories:
            values_matrix.append(values)
            series_names.append(series.get('name'))
    
    if not values_matrix:
        print("警告：堆叠条形图没有找到长度匹配的数据系列。")
        return

    values_matrix = np.array(values_matrix)
    bottom = np.zeros(num_categories)
    
    for i in range(len(values_matrix)):
        ax.bar(x_labels, values_matrix[i], label=series_names[i], bottom=bottom)
        bottom += values_matrix[i]
    ax.legend()

def _draw_percentage_stacked_bar_chart(ax: plt.Axes, data: Dict[str, Any]):
    """绘制百分比堆叠条形图 (已强化)。"""
    chart_data = data.get('data', {})
    x_labels = chart_data.get('x_categories', [])
    series_data = chart_data.get('y_series', [])
    if not x_labels or not series_data: return
    
    num_categories = len(x_labels)
    values_matrix, series_names = [], []

    for series in series_data:
        values = series.get('values', [])
        if len(values) == num_categories:
            values_matrix.append(values)
            series_names.append(series.get('name'))
    
    if not values_matrix:
        print("警告：百分比堆叠条形图没有找到长度匹配的数据系列。")
        return

    values_matrix = np.array(values_matrix)
    totals = values_matrix.sum(axis=0)
    totals[totals == 0] = 1 # 避免除以零
    percent_matrix = (values_matrix / totals) * 100
    bottom = np.zeros(num_categories)

    for i in range(len(values_matrix)):
        ax.bar(x_labels, percent_matrix[i], label=series_names[i], bottom=bottom)
        bottom += percent_matrix[i]
    ax.legend()
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)

def _draw_stacked_area_chart(ax: plt.Axes, data: Dict[str, Any]):
    """绘制堆叠面积图。"""
    chart_data = data.get('data', {})
    x_labels = chart_data.get('x_categories', [])
    series_data = chart_data.get('y_series', [])
    if not x_labels or not series_data: return
    values = np.array([s.get('values', []) for s in series_data])
    ax.stackplot(x_labels, values, labels=[s.get('name') for s in series_data], alpha=0.7)
    ax.legend(loc='upper left')

def _draw_sunburst_chart(ax: plt.Axes, data: Dict[str, Any]):
    """绘制旭日图 (已强化，带回退逻辑)。"""
    pie_data = data.get('data', {}).get('pie_data', {})
    labels = pie_data.get('labels', [])
    values = pie_data.get('values', [])
    parents = pie_data.get('parents')

    if not all([labels, values, parents]):
        print("警告: 旭日图缺少 'parents' 数据，回退到绘制饼图。")
        _draw_pie_chart(ax, data)
        return

    try:
        df = pd.DataFrame({'labels': labels, 'values': values, 'parents': parents})
        outer = df[df['parents'] != '']
        inner = df[df['parents'] == ''].groupby('labels')['values'].sum().reset_index()
        
        if not inner.empty:
            ax.pie(inner['values'], labels=inner['labels'], radius=0.7, wedgeprops=dict(width=0.4, edgecolor='w'))
        if not outer.empty:
            ax.pie(outer['values'], labels=outer['labels'], radius=1, wedgeprops=dict(width=0.3, edgecolor='w'))

        ax.axis('equal')
    except Exception as e:
        print(f"绘制旭日图时发生错误: {e}。回退到绘制饼图。")
        _draw_pie_chart(ax, data)

def _draw_treemap(ax: plt.Axes, data: Dict[str, Any]):
    """绘制树状图。"""
    pie_data = data.get('data', {}).get('pie_data', {})
    squarify.plot(sizes=pie_data.get('values', []), label=pie_data.get('labels', []), alpha=.8)
    plt.axis('off')

def _draw_waterfall_chart(ax: plt.Axes, data: Dict[str, Any]):
    """绘制瀑布图。"""
    wf_data = data.get('data', {}).get('waterfall_data', {})
    labels = wf_data.get('labels', [])
    values = np.array(wf_data.get('values', []))
    cumulative = np.cumsum(values)
    starts = np.append([0], cumulative[:-1])
    ax.bar(labels, values, bottom=starts, color=['g' if v > 0 else 'r' for v in values])
    for i in range(len(starts) - 1):
        ax.plot([i, i + 1], [cumulative[i], cumulative[i]], 'k:')
    ax.grid(axis='y', linestyle='--')

def _draw_contour_plot(fig: plt.Figure, data: Dict[str, Any]):
    """绘制等高线图。"""
    c_data = data.get('data', {}).get('contour_data', {})
    x_grid = c_data.get('x_grid', [])
    y_grid = c_data.get('y_grid', [])
    z_values = np.array(c_data.get('z_values', []))
    if not x_grid or not y_grid: return None
    X, Y = np.meshgrid(x_grid, y_grid)
    ax = fig.add_subplot(111)
    contour = ax.contourf(X, Y, z_values, cmap='viridis')
    fig.colorbar(contour, ax=ax)
    return ax

def _draw_network_diagram(ax: plt.Axes, data: Dict[str, Any]):
    """绘制网络图。"""
    net_data = data.get('data', {}).get('network_data', {})
    G = nx.Graph()
    for node in net_data.get('nodes', []): G.add_node(node['id'], label=node.get('label', node['id']))
    for edge in net_data.get('edges', []): G.add_edge(edge['source'], edge['target'])
    nx.draw(G, ax=ax, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='skyblue')

def _draw_forest_plot(ax: plt.Axes, data: Dict[str, Any]):
    """绘制森林图。"""
    f_data = data.get('data', {}).get('forest_plot_data', {})
    labels = f_data.get('labels', [])
    effects, lower, upper = map(np.array, [f_data.get(k, []) for k in ['effects', 'lower_ci', 'upper_ci']])
    if not labels: return
    y_pos = np.arange(len(labels))
    ax.errorbar(effects, y_pos, xerr=[effects - lower, upper - effects], fmt='o', color='black', capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(x=1, color='r', linestyle='--') # 森林图通常以1为基准线

def _draw_funnel_chart(ax: plt.Axes, data: Dict[str, Any]):
    """绘制漏斗图。"""
    funnel_data = data.get('data', {}).get('funnel_data', {})
    stages = funnel_data.get('stages', [])
    values = np.array(funnel_data.get('values', []))
    if not stages: return
    y_pos = np.arange(len(stages))
    ax.barh(y_pos, values, color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages)
    ax.invert_yaxis()
    for i, v in enumerate(values): ax.text(v, i, str(v), va='center')

def _draw_sankey_diagram(ax: plt.Axes, data: Dict[str, Any]):
    """绘制桑基图。"""
    sankey_data = data.get('data', {}).get('sankey_data', {})
    links = sankey_data.get('links', [])
    if not links: return
    sankey = Sankey(ax=ax, scale=0.01, offset=0.2, format='%.0f')
    for link in links:
        sankey.add(flows=[link['value'], -link['value']], labels=[link['source'], link['target']], orientations=[-1, 1])
    sankey.finish()
    ax.set_xticks([])
    ax.set_yticks([])

def _draw_candlestick_chart(ax: plt.Axes, data: Dict[str, Any]):
    """绘制K线图。"""
    candlestick_data = data.get('data', {}).get('candlestick_data', {})
    records = candlestick_data.get('records', [])
    if not records: return
    df = pd.DataFrame(records)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[['open', 'high', 'low', 'close']].rename(columns=lambda x: x.capitalize())
    mpf.plot(df, type='candle', ax=ax, style='yahoo')

def _draw_gauge_chart(ax: plt.Axes, data: Dict[str, Any]):
    """绘制仪表盘。"""
    gauge_data = data.get('data', {}).get('gauge_data', {})
    value = gauge_data.get('value')
    min_val = gauge_data.get('min_value', 0)
    max_val = gauge_data.get('max_value', 100)
    title = gauge_data.get('title', '')
    ranges = gauge_data.get('ranges') 
    
    if value is None: return
    
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1, 1.5); ax.axis('off')
    
    if ranges:
        for r in ranges:
            start, end, color = r['start'], r['end'], r['color']
            start_angle = 180 - (start - min_val) / (max_val - min_val) * 180
            end_angle = 180 - (end - min_val) / (max_val - min_val) * 180
            ax.add_patch(Wedge((0, 0), 1, end_angle, start_angle, facecolor=color, alpha=0.7))

    angle = 180 - (value - min_val) / (max_val - min_val) * 180
    ax.arrow(0, 0, 0.8 * np.cos(np.radians(angle)), 0.8 * np.sin(np.radians(angle)), width=0.03, head_width=0.08, fc='k', ec='k')
    
    ax.text(0, -0.2, f'{value:.2f}', ha='center', fontsize=20)
    ax.text(0, -0.4, title, ha='center', va='center', fontsize=12)

def _draw_word_cloud(ax: plt.Axes, data: Dict[str, Any]):
    """绘制词云图。"""
    wc_data = data.get('data', {}).get('word_cloud_data', {})
    entries = wc_data.get('entries', [])
    if not entries: return
    frequencies = {entry['word']: entry['weight'] for entry in entries}
    wc = WordCloud(background_color="white", width=800, height=400).generate_from_frequencies(frequencies)
    ax.imshow(wc, interpolation='bilinear'); ax.axis("off")

def _draw_calendar_heatmap(fig: plt.Figure, data: Dict[str, Any]):
    """绘制日历热力图。"""
    cal_data = data.get('data', {}).get('calendar_heatmap_data', {})
    entries = cal_data.get('entries', [])
    if not entries: return None
    
    date_values = {entry['date']: entry['value'] for entry in entries}
    events = pd.Series({pd.to_datetime(k): v for k, v in date_values.items()})
    
    ax = fig.add_subplot(111)
    calmap.yearplot(events, year=events.index.year.min(), ax=ax, cmap='viridis')
    fig.suptitle(data.get('title', ''), fontsize=16)
    return ax

def _draw_parallel_coordinates(ax: plt.Axes, data: Dict[str, Any]):
    """绘制平行坐标图。"""
    pc_data = data.get('data', {}).get('parallel_coordinates_data', {})
    dimensions = pc_data.get('dimensions', [])
    records = pc_data.get('data_records', [])
    if not dimensions or not records: return
    df = pd.DataFrame(records, columns=dimensions)
    if pc_data.get('group_by'):
        df['group'] = pc_data.get('group_by')
        parallel_coordinates(df, 'group', ax=ax, colormap='viridis')
    else:
        parallel_coordinates(df, None, ax=ax, color='skyblue')
    if ax.get_legend():
        ax.get_legend().remove()

class MatplotlibGenerator(BaseChartGenerator):
    """使用 Matplotlib 生成静态图表的生成器。"""

    @property
    def file_extension(self) -> str:
        return "png"

    def create_chart(self, output_path: str, figsize: tuple = (12, 8), dpi: int = 100) -> bool:
        # 尝试使用常见的无衬线中文字体，如果找不到则使用系统默认
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
        except:
            pass
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
        
        fig, ax = None, None
        try:
            is_special_fig = self.chart_type in ['radar', 'rose', '3d_bar', 'contour', 'calendar_heatmap']
            if is_special_fig:
                fig = plt.figure(figsize=figsize)
            else:
                fig, ax = plt.subplots(figsize=figsize)
            
            # 使用明确的字典映射，确保图表类型能被正确识别
            draw_funcs = {
                'bar': _draw_bar_chart,
                'line': _draw_line_chart,
                'pie': _draw_pie_chart,
                'donut': _draw_donut_chart,
                'rose': lambda f, d: _draw_rose_chart(f, d),
                'scatter': _draw_scatter_with_error_bars,
                'scatter_with_error_bars': _draw_scatter_with_error_bars,
                'area': _draw_area_chart,
                'radar': lambda f, d: _draw_radar_chart(f, d),
                'heatmap': _draw_heatmap,
                'line_with_confidence_interval': _draw_line_with_confidence_interval,
                'histogram': _draw_histogram,
                '3d_bar': lambda f, d: _draw_3d_bar_chart(f, d),
                'boxplot': lambda a, d: _draw_statistical_plot(a, d, 'boxplot'),
                'violin': lambda a, d: _draw_statistical_plot(a, d, 'violin'),
                'strip': lambda a, d: _draw_statistical_plot(a, d, 'strip'),
                'stacked_bar': _draw_stacked_bar_chart,
                'percentage_stacked_bar': _draw_percentage_stacked_bar_chart,
                'stacked_area': _draw_stacked_area_chart,
                'sunburst': _draw_sunburst_chart,
                'treemap': _draw_treemap,
                'waterfall': _draw_waterfall_chart,
                'contour': lambda f, d: _draw_contour_plot(f, d),
                'network': _draw_network_diagram,
                'forest_plot': _draw_forest_plot, # 修正 forest_plot 的键名
                'forest': _draw_forest_plot, # 添加别名以防万一
                'funnel': _draw_funnel_chart,
                'sankey': _draw_sankey_diagram,
                'candlestick': _draw_candlestick_chart,
                'gauge': _draw_gauge_chart,
                'word_cloud': _draw_word_cloud,
                'calendar_heatmap': lambda f, d: _draw_calendar_heatmap(f, d),
                'parallel_coordinates': _draw_parallel_coordinates,
            }

            draw_func = draw_funcs.get(self.chart_type)
            if not draw_func: raise ValueError(f"不支持的图表类型: {self.chart_type}")

            if is_special_fig:
                ax = draw_func(fig, self.chart_package)
            else:
                draw_func(ax, self.chart_package)

            title = self.chart_package.get('title', '')
            if self.chart_type != 'calendar_heatmap':
                if ax and not isinstance(ax, Axes3D): ax.set_title(title, fontsize=16)
                elif fig: fig.suptitle(title, fontsize=16)

            if ax and self.chart_type not in ['pie', 'donut', 'radar', 'rose', 'heatmap', 'treemap', 'network', '3d_bar', 'sankey', 'gauge', 'word_cloud', 'parallel_coordinates']:
                ax.set_xlabel(self.chart_package.get('x_label', '')); ax.set_ylabel(self.chart_package.get('y_label', ''))
            
            fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)
            fig.savefig(output_path, dpi=dpi)
            return True
        except Exception as e:
            print(f"生成图表 '{self.chart_type}' 时发生错误: {e}")
            return False
        finally:
            if fig: plt.close(fig)

    def generate_code(self, figsize: tuple = (12, 8), dpi: int = 100) -> str:
        helper_funcs = [radar_factory] + [func for name, func in globals().items() if name.startswith('_draw_')]
        sources = [inspect.getsource(func) for func in helper_funcs]
        helpers_code = "\n\n".join(sources)
        
        def json_serializer(obj):
            if isinstance(obj, (datetime, datetime.date)): return obj.isoformat()
            return str(obj)
        data_str = json.dumps(self.chart_package, indent=4, ensure_ascii=False, default=json_serializer)

        # 主逻辑的生成也改为使用静态字典，与create_chart保持一致
        main_logic_str = """
    try:
        # 添加字体设置
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
        except:
            pass
        plt.rcParams['axes.unicode_minus'] = False

        is_special_fig = chart_type in ['radar', 'rose', '3d_bar', 'contour', 'calendar_heatmap']
        if is_special_fig:
            fig = plt.figure(figsize=figsize)
        else:
            fig, ax = plt.subplots(figsize=figsize)
        
        draw_funcs = {
            'bar': _draw_bar_chart, 'line': _draw_line_chart, 'pie': _draw_pie_chart,
            'donut': _draw_donut_chart, 'rose': lambda f, d: _draw_rose_chart(f, d),
            'scatter': _draw_scatter_with_error_bars, 'scatter_with_error_bars': _draw_scatter_with_error_bars,
            'area': _draw_area_chart, 'radar': lambda f, d: _draw_radar_chart(f, d),
            'heatmap': _draw_heatmap, 'line_with_confidence_interval': _draw_line_with_confidence_interval,
            'histogram': _draw_histogram, '3d_bar': lambda f, d: _draw_3d_bar_chart(f, d),
            'boxplot': lambda a, d: _draw_statistical_plot(a, d, 'boxplot'),
            'violin': lambda a, d: _draw_statistical_plot(a, d, 'violin'),
            'strip': lambda a, d: _draw_statistical_plot(a, d, 'strip'),
            'stacked_bar': _draw_stacked_bar_chart, 'percentage_stacked_bar': _draw_percentage_stacked_bar_chart,
            'stacked_area': _draw_stacked_area_chart, 'sunburst': _draw_sunburst_chart,
            'treemap': _draw_treemap, 'waterfall': _draw_waterfall_chart,
            'contour': lambda f, d: _draw_contour_plot(f, d), 'network': _draw_network_diagram,
            'forest_plot': _draw_forest_plot, 'forest': _draw_forest_plot,
            'funnel': _draw_funnel_chart,
            'sankey': _draw_sankey_diagram, 'candlestick': _draw_candlestick_chart,
            'gauge': _draw_gauge_chart, 'word_cloud': _draw_word_cloud,
            'calendar_heatmap': lambda f, d: _draw_calendar_heatmap(f, d),
            'parallel_coordinates': _draw_parallel_coordinates,
        }

        draw_func = draw_funcs.get(chart_type)
        if not draw_func: raise ValueError(f"不支持的图表类型: {chart_type}")

        if is_special_fig:
            ax = draw_func(fig, chart_package)
        else:
            draw_func(ax, chart_package)

        if chart_type != 'calendar_heatmap':
            if ax and not isinstance(ax, Axes3D): ax.set_title(title, fontsize=16)
            elif fig: fig.suptitle(title, fontsize=16)

        if ax and chart_type not in ['pie', 'donut', 'radar', 'rose', 'heatmap', 'treemap', 'network', '3d_bar', 'sankey', 'gauge', 'word_cloud', 'parallel_coordinates']:
            ax.set_xlabel(chart_package.get('x_label', '')); ax.set_ylabel(chart_package.get('y_label', ''))
        
        fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)
        plt.show()
        
    except Exception as e:
        print(f"生成图表 '{chart_type}' 时发生错误: {e}")
    finally:
        if fig: plt.close(fig)
"""
        return f"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import inspect
from typing import Dict, Any, List
from datetime import datetime
from matplotlib.sankey import Sankey
from wordcloud import WordCloud
import mplfinance as mpf
import calmap
from pandas.plotting import parallel_coordinates
from adjustText import adjust_text
from matplotlib.projections import register_projection
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import squarify
import networkx as nx
from matplotlib.patches import Wedge

# --- 绘图辅助函数 ---
{helpers_code}

# --- 主执行逻辑 ---
def main():
    chart_package_str = '''{data_str}'''
    chart_package = json.loads(chart_package_str)

    chart_type = chart_package.get('chart_type')
    title = chart_package.get('title', '')
    figsize = {figsize}
    fig, ax = None, None
    {main_logic_str}

if __name__ == '__main__':
    main()
"""
