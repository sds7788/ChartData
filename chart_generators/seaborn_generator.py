# chart_generators/seaborn_generator.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import inspect
from typing import Dict, Any

# 假设 base_generator 和 matplotlib_generator 在同一路径下
# from .base_generator import BaseChartGenerator
# from .matplotlib_generator import (
#     radar_factory, _draw_pie_chart, _draw_radar_chart, _draw_rose_chart,
#     _draw_gantt_chart, _draw_donut_chart
# )

# 为了让代码独立运行，我们在这里定义缺失的基类和函数
class BaseChartGenerator:
    def __init__(self, chart_package: Dict[str, Any]):
        self.chart_package = chart_package
        self.chart_type = chart_package.get("chart_type")

def radar_factory(num_vars, frame='circle'):
    pass # Placeholder
def _draw_pie_chart(ax, pkg):
    pass # Placeholder
def _draw_radar_chart(fig, pkg):
    pass # Placeholder
def _draw_rose_chart(fig, pkg):
    pass # Placeholder
def _draw_gantt_chart(ax, pkg):
    pass # Placeholder
def _draw_donut_chart(ax, pkg):
    pass # Placeholder

class SeabornGenerator(BaseChartGenerator):
    """使用 Seaborn 和 Matplotlib 兼容回退策略生成图表的生成器。"""

    @property
    def file_extension(self) -> str:
        return "png"

    def _json_to_dataframe(self, chart_type: str) -> pd.DataFrame:
        if chart_type in ['bar', 'line', 'area', 'stacked_bar', 'percentage_stacked_bar', 'combo_bar_line']:
            x_cat = self.chart_package.get('data', {}).get('x_categories', [])
            y_ser = self.chart_package.get('data', {}).get('y_series', [])
            records = []
            for series in y_ser:
                series_name = series.get('name')
                series_type = series.get('type', 'bar')
                for i, value in enumerate(series.get('values', [])):
                    records.append({
                        'x': x_cat[i],
                        'y': value,
                        'series': series_name,
                        'type': series_type
                    })
            return pd.DataFrame(records)
        if chart_type == 'scatter':
            points = self.chart_package.get('data', {}).get('scatter_points', [])
            return pd.DataFrame(points)
        if chart_type == 'boxplot':
            boxplot_data = self.chart_package.get('data', {}).get('boxplot_data', {})
            categories = boxplot_data.get('categories', [])
            data_series = boxplot_data.get('data_series', [])
            records = []
            for i, cat in enumerate(categories):
                for val in data_series[i]:
                    records.append({'category': cat, 'value': val})
            return pd.DataFrame(records)
        if chart_type == 'heatmap':
            heatmap_data = self.chart_package.get('data', {}).get('heatmap_data', {})
            y_labels = heatmap_data.get('y_labels', [])
            x_labels = heatmap_data.get('x_labels', [])
            values = heatmap_data.get('values', [])
            return pd.DataFrame(values, index=y_labels, columns=x_labels)
        return pd.DataFrame()

    def create_chart(self, output_path: str, figsize: tuple = (12, 8), dpi: int = 100) -> bool:
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
        sns.set_theme(style="whitegrid", rc={"axes.unicode_minus": False})
        
        fig, ax = None, None
        matplotlib_fallback_charts = ['radar', 'rose', 'pie', 'donut', 'gantt']

        try:
            if self.chart_type in matplotlib_fallback_charts:
                fig = plt.figure(figsize=figsize)
            else:
                fig, ax = plt.subplots(figsize=figsize)

            draw_func_map = {
                'bar': self._create_bar_chart, 'line': self._create_line_chart, 'scatter': self._create_scatter_chart,
                'area': self._create_area_chart, 'combo_bar_line': self._create_combo_bar_line_chart,
                'heatmap': self._create_heatmap_chart, 'stacked_bar': self._create_stacked_bar_chart,
                'boxplot': self._create_boxplot_chart, 'percentage_stacked_bar': self._create_percentage_stacked_bar_chart,
                'pie': lambda ax, fig: _draw_pie_chart(fig.add_subplot(111), self.chart_package),
                'donut': lambda ax, fig: _draw_donut_chart(fig.add_subplot(111), self.chart_package),
                'radar': lambda ax, fig: _draw_radar_chart(fig, self.chart_package),
                'rose': lambda ax, fig: _draw_rose_chart(fig, self.chart_package),
                'gantt': lambda ax, fig: _draw_gantt_chart(fig.add_subplot(111), self.chart_package),
            }
            
            draw_func = draw_func_map.get(self.chart_type)
            if draw_func:
                result_ax = draw_func(ax=ax, fig=fig)
                if self.chart_type in matplotlib_fallback_charts:
                    # 对于fallback图表，返回的可能是Figure或Axes对象
                    ax = result_ax if isinstance(result_ax, plt.Axes) else fig.gca()
                else:
                    ax = result_ax
            else: 
                raise ValueError(f"不支持的图表类型: {self.chart_type}")

            # --- 修改点: 统一设置坐标轴属性 ---
            if ax:
                self._apply_ax_settings(ax)
            
            fig.tight_layout()
            fig.savefig(output_path, dpi=dpi)
            return True
        except Exception as e:
            print(f"使用 Seaborn/Matplotlib 生成图表 '{self.chart_type}' 时发生严重错误: {e}")
            return False
        finally:
            if fig: plt.close(fig)

    # --- 新增点: 整合通用坐标轴设置 ---
    def _apply_ax_settings(self, ax: plt.Axes):
        """将标题、标签、刻度旋转等通用设置应用于给定的Axes对象。"""
        ax.set_title(self.chart_package.get('title', ''), fontsize=16)

        # 仅为非特殊图表设置x/y标签
        if self.chart_type not in ['pie', 'donut', 'radar', 'rose', 'heatmap', 'gantt']:
            ax.set_xlabel(self.chart_package.get('x_label', ''))
            ax.set_ylabel(self.chart_package.get('y_label', ''))

        # --- 修改点: 旋转X轴标签以防重叠 ---
        if self.chart_type not in ['pie', 'donut', 'radar', 'rose']:
            # 为热力图设置更陡峭的旋转角度
            rotation_angle = 45 if self.chart_type == 'heatmap' else 30
            ax.tick_params(axis='x', rotation=rotation_angle, ha='right')
            if self.chart_type == 'heatmap':
                 ax.tick_params(axis='y', rotation=0)

        # 应用Y轴范围
        self._apply_y_axis_range(ax)

    def _apply_y_axis_range(self, ax: plt.Axes):
        """辅助函数，用于在ax上应用固定的Y轴范围"""
        y_axis_range = self.chart_package.get('y_axis_range')
        # 仅在提供了有效范围时应用
        if y_axis_range and len(y_axis_range) == 2:
            ax.set_ylim(bottom=y_axis_range[0], top=y_axis_range[1])

    def _create_bar_chart(self, ax, fig):
        df = self._json_to_dataframe(self.chart_type)
        sns.barplot(data=df, x='x', y='y', hue='series', ax=ax)
        return ax

    def _create_line_chart(self, ax, fig):
        df = self._json_to_dataframe(self.chart_type)
        sns.lineplot(data=df, x='x', y='y', hue='series', marker='o', ax=ax)
        return ax

    def _create_scatter_chart(self, ax, fig):
        df = self._json_to_dataframe(self.chart_type)
        sns.scatterplot(data=df, x='x', y='y', size='size', hue='color_value', sizes=(50, 500), ax=ax, palette='viridis')
        return ax

    def _create_area_chart(self, ax, fig):
        df = self._json_to_dataframe(self.chart_type)
        # 确保绘图顺序与数据源一致
        series_order = [s['name'] for s in self.chart_package.get('data', {}).get('y_series', [])]
        for series_name in series_order:
            series_df = df[df['series'] == series_name]
            sns.lineplot(data=series_df, x='x', y='y', label=series_name, ax=ax, sort=False)
            ax.fill_between(series_df['x'], series_df['y'], alpha=0.3)
        return ax

    def _create_heatmap_chart(self, ax, fig):
        df = self._json_to_dataframe(self.chart_type)
        sns.heatmap(df, annot=True, fmt=".1f", cmap="viridis", ax=ax)
        return ax

    def _create_boxplot_chart(self, ax, fig):
        df = self._json_to_dataframe(self.chart_type)
        sns.boxplot(data=df, x='category', y='value', ax=ax)
        return ax

    # --- 修改点: 重构堆叠条形图，代码更简洁高效 ---
    def _create_stacked_bar_chart(self, ax, fig, percentage=False):
        """使用pandas.pivot重构，以生成（百分比）堆叠条形图。"""
        df = self._json_to_dataframe('stacked_bar') # 确保使用正确的类型进行数据转换
        
        # 使用pivot转换数据格式，x为索引，series为列，y为值
        pivot_df = df.pivot(index='x', columns='series', values='y').fillna(0)
        
        # 保证顺序与输入数据一致
        x_cat = self.chart_package.get('data', {}).get('x_categories', [])
        series_order = [s['name'] for s in self.chart_package.get('data', {}).get('y_series', [])]
        pivot_df = pivot_df.reindex(index=x_cat, columns=series_order)

        if percentage:
            # 计算每行的总和，并转换为百分比
            row_sum = pivot_df.sum(axis=1)
            pivot_df = pivot_df.div(row_sum, axis=0).multiply(100)
            ax.set_ylabel('Percentage (%)')
            # 百分比图的Y轴固定为0-100
            ax.set_ylim(0, 100)

        # 使用pandas内置的绘图功能
        pivot_df.plot(kind='bar', stacked=True, ax=ax, width=0.8)
        
        # 调整图例
        ax.legend(title='Series')
        
        # 移除自动生成的x轴标签"x"
        ax.set_xlabel('')
        
        return ax
    
    def _create_percentage_stacked_bar_chart(self, ax, fig):
        return self._create_stacked_bar_chart(ax, fig, percentage=True)

    def _create_combo_bar_line_chart(self, ax, fig):
        df = self._json_to_dataframe(self.chart_type)
        bar_df = df[df['type'] == 'bar']
        line_df = df[df['type'] == 'line']
        
        # 绘制条形图
        sns.barplot(data=bar_df, x='x', y='y', hue='series', ax=ax, alpha=0.7)
        
        # 创建共享x轴的次y轴
        ax2 = ax.twinx()
        sns.lineplot(data=line_df, x='x', y='y', hue='series', marker='o', ax=ax2, sort=False)
        
        # 合并图例
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, title='Series')
        ax2.get_legend().remove() # 移除次y轴的重复图例
        ax2.set_ylabel('') # 通常次坐标轴也需要标签

        return ax

    # --- 修改点: 完善代码生成功能，使其能够生成完整的、可运行的脚本 ---
    def generate_code(self, figsize: tuple = (12, 8), dpi: int = 100) -> str:
        """生成一个独立的、可运行的Python脚本来重现图表。"""
        data_str = json.dumps(self.chart_package, indent=4, ensure_ascii=False)
        
        # 获取所有需要的辅助函数的源代码
        source_code_map = {
            'radar_factory': radar_factory,
            '_draw_pie_chart': _draw_pie_chart,
            '_draw_donut_chart': _draw_donut_chart,
            '_draw_radar_chart': _draw_radar_chart,
            '_draw_rose_chart': _draw_rose_chart,
            '_draw_gantt_chart': _draw_gantt_chart,
            '_json_to_dataframe': self._json_to_dataframe,
            '_apply_ax_settings': self._apply_ax_settings,
            '_apply_y_axis_range': self._apply_y_axis_range,
            '_create_bar_chart': self._create_bar_chart,
            '_create_line_chart': self._create_line_chart,
            '_create_scatter_chart': self._create_scatter_chart,
            '_create_area_chart': self._create_area_chart,
            '_create_heatmap_chart': self._create_heatmap_chart,
            '_create_boxplot_chart': self._create_boxplot_chart,
            '_create_stacked_bar_chart': self._create_stacked_bar_chart,
            '_create_percentage_stacked_bar_chart': self._create_percentage_stacked_bar_chart,
            '_create_combo_bar_line_chart': self._create_combo_bar_line_chart
        }
        
        helpers_code_list = []
        for name, func in source_code_map.items():
            # 使用 inspect.getsource 获取函数源代码
            code = inspect.getsource(func)
            # 移除 self 参数以使其成为独立函数
            code = code.replace('self, ', '').replace('self.', '')
            helpers_code_list.append(code)
        
        helpers_code = "\n\n".join(helpers_code_list)

        main_logic = inspect.getsource(self.create_chart)
        # 移除类方法相关的部分，使其能在脚本中独立运行
        main_logic = main_logic.replace('self, output_path: str, figsize: tuple = (12, 8), dpi: int = 100', 'pkg, figsize, dpi')
        main_logic = main_logic.replace('self.chart_package', 'pkg')
        main_logic = main_logic.replace('self.chart_type', "pkg.get('chart_type')")
        main_logic = main_logic.replace('self.', '')
        main_logic = main_logic.replace('create_chart', 'create_chart_from_pkg') # 重命名以防混淆
        # 将保存文件替换为显示图表
        main_logic = main_logic.replace('fig.savefig(output_path, dpi=dpi)', 'plt.show()')

        return f"""
# -*- coding: utf-8 -*-
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import numpy as np

# --- 图表配置数据 ---
CHART_PACKAGE: Dict[str, Any] = {data_str}
FIGSIZE = {figsize}
DPI = {dpi}

# --- 辅助函数定义 ---
{helpers_code}

# --- 主绘图逻辑 ---
{main_logic}

def main():
    print(f"正在生成图表: {{CHART_PACKAGE.get('chart_type')}}...")
    create_chart_from_pkg(CHART_PACKAGE, FIGSIZE, DPI)
    print("图表生成完毕。")

if __name__ == '__main__':
    main()
"""