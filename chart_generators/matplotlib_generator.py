# chart_generators/matplotlib_generator.py
import matplotlib.pyplot as plt
import numpy as np
import json
import inspect
from typing import Dict, Any, List
from adjustText import adjust_text
from matplotlib.projections import register_projection
import matplotlib.colors as mcolors

from .base_generator import BaseChartGenerator

# --- 所有绘图辅助函数 ---

def radar_factory(num_vars: int, frame: str = 'circle'):
    """
    创建一个雷达图投影，并返回各个轴的角度。
    """
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
            for line in lines:
                self._close_line(line)
        def _close_line(self, line):
            x, y = line.get_data()
            if x.size > 0 and (x[0] != x[-1] or y[0] != y[-1]):
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
        def fill(self, *args, **kwargs):
            closed_args = []
            for arg in args:
                if isinstance(arg, (list, np.ndarray)) and len(arg) > 0:
                    if arg[0] != arg[-1]:
                        arg = np.concatenate((arg, [arg[0]]))
                closed_args.append(arg)
            return super().fill(*closed_args, **kwargs)

    register_projection(RadarAxes)
    return theta

def _draw_bar_chart(ax: plt.Axes, data: Dict[str, Any]):
    chart_data = data.get('data', {})
    x_labels = chart_data.get('x_categories', [])
    series_data = chart_data.get('y_series', [])
    if not x_labels or not series_data: return
    x = np.arange(len(x_labels))
    num_series = len(series_data)
    total_width = 0.8
    bar_width = total_width / num_series if num_series > 0 else total_width
    texts = []
    
    # --- Y轴修改点: 优先使用固定范围来计算标签偏移 ---
    y_max = 0
    if data.get('y_axis_range'):
         y_max = data['y_axis_range'][1]
    else:
        for series in series_data:
            values = series.get('values', [])
            if values: y_max = max(y_max, max((v for v in values if v is not None), default=0))
            
    label_offset = y_max * 0.015
    
    for i, series in enumerate(series_data):
        series_name = series.get('name')
        values = series.get('values')
        if not values or len(values) != len(x_labels): continue
        offset = (i - (num_series - 1) / 2) * bar_width
        rects = ax.bar(x + offset, values, bar_width, label=series_name)
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                texts.append(ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height + label_offset, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=11
                ))
    ax.set_xticks(x, x_labels)
    if num_series > 1: ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    if texts:
        adjust_text(texts, ax=ax, force_text=(0.5, 0.5), arrowprops=dict(arrowstyle="-", color='black', lw=0.5))
    
    # --- Y轴修改点: 应用固定范围 ---
    if 'y_axis_range' in data and data['y_axis_range']:
        ax.set_ylim(bottom=data['y_axis_range'][0], top=data['y_axis_range'][1])
        ax.set_ylim(top=ax.get_ylim()[1] * 1.1) # 为标签稍微增加顶部空间
    else:
        ax.set_ylim(top=ax.get_ylim()[1] * 1.2)

def _draw_line_chart(ax: plt.Axes, data: Dict[str, Any]):
    chart_data = data.get('data', {})
    x_labels = chart_data.get('x_categories', [])
    series_data = chart_data.get('y_series', [])
    if not x_labels or not series_data: return
    markers = ['o', 's', '^', 'v', 'D', '<', '>']
    for i, series in enumerate(series_data):
        series_name = series.get('name')
        values = series.get('values')
        if not values or len(values) != len(x_labels): continue
        ax.plot(x_labels, values, marker=markers[i % len(markers)], linestyle='-', label=series_name)
    if len(series_data) > 1: ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    # --- Y轴修改点 ---
    if 'y_axis_range' in data and data['y_axis_range']:
        ax.set_ylim(bottom=data['y_axis_range'][0], top=data['y_axis_range'][1])

def _draw_pie_chart(ax: plt.Axes, data: Dict[str, Any]):
    pie_data = data.get('data', {}).get('pie_data', {})
    labels = pie_data.get('labels', [])
    values = pie_data.get('values', [])
    if not labels or not values or len(labels) != len(values): return
    explode = pie_data.get('explode')
    if not explode or len(explode) != len(labels):
        explode = [0.0] * len(labels)
    ax.pie(values, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax.axis('equal')

def _draw_scatter_plot(ax: plt.Axes, data: Dict[str, Any]):
    chart_data = data.get('data', {})
    points = chart_data.get('scatter_points', [])
    if not points: return
    x_vals = [p.get('x', 0) for p in points]
    y_vals = [p.get('y', 0) for p in points]
    sizes = [p.get('size', 50) for p in points]
    colors = [p.get('color_value', 0) for p in points]
    scatter = ax.scatter(x_vals, y_vals, s=sizes, c=colors, alpha=0.7, cmap='viridis')
    if len(set(colors)) > 1:
        ax.figure.colorbar(scatter, ax=ax, label='Color Value')
    if len(set(sizes)) > 1:
        handles, legend_labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=5)
        ax.legend(handles, legend_labels, loc="upper right", title="Sizes")
    ax.grid(True, linestyle='--', alpha=0.7)
    # --- Y轴修改点 ---
    if 'y_axis_range' in data and data['y_axis_range']:
        ax.set_ylim(bottom=data['y_axis_range'][0], top=data['y_axis_range'][1])


def _draw_area_chart(ax: plt.Axes, data: Dict[str, Any]):
    chart_data = data.get('data', {})
    x_labels = chart_data.get('x_categories', [])
    series_data = chart_data.get('y_series', [])
    if not x_labels or not series_data: return
    series = series_data[0]
    values = series.get('values', [])
    if not values or len(values) != len(x_labels): return
    series_name = series.get('name')
    ax.plot(x_labels, values, color='skyblue', lw=2, label=series_name)
    ax.fill_between(x_labels, values, color="skyblue", alpha=0.4)
    if series_name: ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    # --- Y轴修改点 ---
    if 'y_axis_range' in data and data['y_axis_range']:
        ax.set_ylim(bottom=data['y_axis_range'][0], top=data['y_axis_range'][1])


def _draw_radar_chart(fig: plt.Figure, data: Dict[str, Any]):
    radar_data = data.get('data', {}).get('radar_data', {})
    labels = radar_data.get('labels', [])
    series_data = radar_data.get('series', [])
    if not labels or not series_data: return None
    num_vars = len(labels)
    theta = radar_factory(num_vars, frame='polygon')
    ax = fig.add_subplot(111, projection='radar')
    ax.set_varlabels(labels)
    # --- Y轴(R轴)修改点 ---
    if 'y_axis_range' in data and data['y_axis_range']:
        ax.set_rlim(data['y_axis_range'][0], data['y_axis_range'][1])
    else:
        all_values = np.concatenate([s.get('values', []) for s in series_data])
        if all_values.size > 0:
            ax.set_rlim(0, np.max(all_values) * 1.1)
            
    for series in series_data:
        series_name = series.get('name')
        values = series.get('values', [])
        if len(values) != num_vars: continue
        ax.plot(theta, values, label=series_name)
        ax.fill(theta, values, alpha=0.25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    return ax

def _draw_stacked_bar_chart(ax: plt.Axes, data: Dict[str, Any], percentage: bool = False):
    chart_data = data.get('data', {})
    x_labels = chart_data.get('x_categories', [])
    series_data = chart_data.get('y_series', [])
    if not x_labels or not series_data: return
    values_matrix = np.array([s.get('values', [0] * len(x_labels)) for s in series_data])
    if percentage:
        totals = values_matrix.sum(axis=0)
        totals[totals == 0] = 1
        values_matrix = (values_matrix / totals[np.newaxis, :]) * 100
        ax.set_ylabel("Percentage (%)")
    bottom = np.zeros(len(x_labels))
    for i, series in enumerate(series_data):
        values = values_matrix[i]
        ax.bar(x_labels, values, label=series.get('name'), bottom=bottom)
        bottom += values
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    # --- Y轴修改点 ---
    if 'y_axis_range' in data and data['y_axis_range']:
        ax.set_ylim(bottom=data['y_axis_range'][0], top=data['y_axis_range'][1])
    elif percentage:
        ax.set_ylim(0, 100)

def _draw_donut_chart(ax: plt.Axes, data: Dict[str, Any]):
    pie_data = data.get('data', {}).get('pie_data', {})
    labels = pie_data.get('labels', [])
    values = pie_data.get('values', [])
    if not labels or not values or len(labels) != len(values): return
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, 
           wedgeprops=dict(width=0.4, edgecolor='w'))
    ax.axis('equal')

def _draw_rose_chart(fig: plt.Figure, data: Dict[str, Any]):
    pie_data = data.get('data', {}).get('pie_data', {})
    labels = pie_data.get('labels', [])
    values = pie_data.get('values', [])
    if not labels or not values or len(labels) != len(values): return None
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

def _draw_boxplot(ax: plt.Axes, data: Dict[str, Any]):
    boxplot_data = data.get('data', {}).get('boxplot_data', {})
    categories = boxplot_data.get('categories', [])
    data_series = boxplot_data.get('data_series', [])
    if not categories or not data_series or len(categories) != len(data_series): return
    ax.boxplot(data_series, labels=categories, patch_artist=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    # --- Y轴修改点 ---
    if 'y_axis_range' in data and data['y_axis_range']:
        ax.set_ylim(bottom=data['y_axis_range'][0], top=data['y_axis_range'][1])

def _draw_heatmap(ax: plt.Axes, data: Dict[str, Any]):
    heatmap_data = data.get('data', {}).get('heatmap_data', {})
    x_labels = heatmap_data.get('x_labels', [])
    y_labels = heatmap_data.get('y_labels', [])
    values = np.array(heatmap_data.get('values', []))
    if not x_labels or not y_labels or values.shape != (len(y_labels), len(x_labels)): return
    im = ax.imshow(values, cmap="viridis")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            val = values[i, j]
            text_color = "w" if norm(val) < 0.5 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color)

def _draw_gantt_chart(ax: plt.Axes, data: Dict[str, Any]):
    gantt_data = data.get('data', {}).get('gantt_data', {})
    tasks = gantt_data.get('tasks', [])
    if not tasks: return
    task_names = [t.get('name') for t in tasks]
    starts = [t.get('start') for t in tasks]
    durations = [t.get('duration') for t in tasks]
    colors = [t.get('color') or 'skyblue' for t in tasks]
    y_pos = np.arange(len(task_names))
    ax.barh(y_pos, durations, left=starts, align='center', color=colors, alpha=0.8)
    ax.set_yticks(y_pos, labels=task_names)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.7)

def _draw_combo_bar_line_chart(ax: plt.Axes, data: Dict[str, Any]):
    chart_data = data.get('data', {})
    x_labels = chart_data.get('x_categories', [])
    series_data = chart_data.get('y_series', [])
    if not x_labels or not series_data: return
    ax2 = ax.twinx()
    bar_series = [s for s in series_data if s.get('type') == 'bar']
    line_series = [s for s in series_data if s.get('type') == 'line']
    x = np.arange(len(x_labels))
    num_bar_series = len(bar_series)
    total_width = 0.8
    bar_width = total_width / num_bar_series if num_bar_series > 0 else total_width
    for i, series in enumerate(bar_series):
        offset = (i - (num_bar_series - 1) / 2) * bar_width
        ax.bar(x + offset, series.get('values'), bar_width, label=f"{series.get('name')} (bar)", alpha=0.7)
    markers = ['o', 's', '^']
    for i, series in enumerate(line_series):
        ax2.plot(x, series.get('values'), marker=markers[i % len(markers)], linestyle='-', label=f"{series.get('name')} (line)")
    ax.set_xticks(x, x_labels)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylabel(data.get('y_label', 'Bar Y-axis'))
    ax2.set_ylabel("Line Y-axis")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    # --- Y轴修改点 ---
    if 'y_axis_range' in data and data['y_axis_range']:
        ax.set_ylim(bottom=data['y_axis_range'][0], top=data['y_axis_range'][1])


class MatplotlibGenerator(BaseChartGenerator):
    """使用 Matplotlib 生成静态图表的生成器。"""

    @property
    def file_extension(self) -> str:
        return "png"

    def create_chart(self, output_path: str, figsize: tuple = (12, 8), dpi: int = 100) -> bool:
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = None, None
        
        try:
            if self.chart_type in ['radar', 'rose']:
                fig = plt.figure(figsize=figsize)
                if self.chart_type == 'radar':
                    ax = _draw_radar_chart(fig, self.chart_package)
                elif self.chart_type == 'rose':
                    ax = _draw_rose_chart(fig, self.chart_package)
                if ax is None:
                    if fig: plt.close(fig)
                    return False
            else:
                fig, ax = plt.subplots(figsize=figsize)
                draw_funcs = {
                    'bar': _draw_bar_chart, 'line': _draw_line_chart, 'pie': _draw_pie_chart,
                    'scatter': _draw_scatter_plot, 'area': _draw_area_chart,
                    'stacked_bar': lambda a, d: _draw_stacked_bar_chart(a, d, percentage=False),
                    'percentage_stacked_bar': lambda a, d: _draw_stacked_bar_chart(a, d, percentage=True),
                    'donut': _draw_donut_chart, 'boxplot': _draw_boxplot, 'heatmap': _draw_heatmap,
                    'gantt': _draw_gantt_chart, 'combo_bar_line': _draw_combo_bar_line_chart,
                }
                draw_func = draw_funcs.get(self.chart_type)
                if draw_func:
                    draw_func(ax, self.chart_package)
                else:
                    raise ValueError(f"Matplotlib 不支持的图表类型: {self.chart_type}")

            if ax:
                ax.set_title(self.chart_package.get('title', ''))
                if self.chart_type not in ['pie', 'donut', 'radar', 'rose', 'heatmap', 'gantt']:
                    ax.set_xlabel(self.chart_package.get('x_label', ''))
                    ax.set_ylabel(self.chart_package.get('y_label', ''))
            
            fig.tight_layout()
            fig.savefig(output_path, dpi=dpi)
            return True
            
        except Exception as e:
            print(f"使用 Matplotlib 生成图表时发生错误: {e}")
            if fig: plt.close(fig)
            return False
        finally:
            if fig: plt.close(fig)

    def generate_code(self, figsize: tuple = (12, 8), dpi: int = 100) -> str:
        required_funcs = [
            radar_factory, _draw_bar_chart, _draw_line_chart, _draw_pie_chart,
            _draw_scatter_plot, _draw_area_chart, _draw_radar_chart,
            _draw_stacked_bar_chart, _draw_donut_chart, _draw_rose_chart,
            _draw_boxplot, _draw_heatmap, _draw_gantt_chart, _draw_combo_bar_line_chart
        ]
        sources = [inspect.getsource(func) for func in required_funcs]
        helpers_code = "\n\n".join(sources)
        data_str = json.dumps(self.chart_package, indent=4, ensure_ascii=False)

        script_template = f"""
import matplotlib.pyplot as plt
import numpy as np
import json
from typing import Dict, Any, List
from adjustText import adjust_text
from matplotlib.projections import register_projection
import matplotlib.colors as mcolors

# --- 辅助函数 ---
{helpers_code}

# --- 主执行逻辑 ---
def main():
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    chart_package = {data_str}
    chart_type = chart_package.get('chart_type')
    fig, ax = None, None
    figsize = {figsize}
    
    if chart_type in ['radar', 'rose']:
        fig = plt.figure(figsize=figsize)
        if chart_type == 'radar':
            ax = _draw_radar_chart(fig, chart_package)
        elif chart_type == 'rose':
            ax = _draw_rose_chart(fig, chart_package)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        draw_funcs = {{
            'bar': _draw_bar_chart, 'line': _draw_line_chart, 'pie': _draw_pie_chart,
            'scatter': _draw_scatter_plot, 'area': _draw_area_chart,
            'stacked_bar': lambda a, d: _draw_stacked_bar_chart(a, d, percentage=False),
            'percentage_stacked_bar': lambda a, d: _draw_stacked_bar_chart(a, d, percentage=True),
            'donut': _draw_donut_chart, 'boxplot': _draw_boxplot, 'heatmap': _draw_heatmap,
            'gantt': _draw_gantt_chart, 'combo_bar_line': _draw_combo_bar_line_chart,
        }}
        draw_func = draw_funcs.get(chart_type)
        if draw_func: draw_func(ax, chart_package)
        else: print(f"不支持的图表类型: {{chart_type}}"); return

    if ax:
        ax.set_title(chart_package.get('title', ''))
        if chart_type not in ['pie', 'donut', 'radar', 'rose', 'heatmap', 'gantt']:
            ax.set_xlabel(chart_package.get('x_label', '')); ax.set_ylabel(chart_package.get('y_label', ''))
    
    if fig:
        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
"""
        return script_template