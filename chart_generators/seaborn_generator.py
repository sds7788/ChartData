# chart_generators/seaborn_generator.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import inspect
from typing import Dict, Any, List
from datetime import datetime

# 假设 base_generator 在同一路径下
from .base_generator import BaseChartGenerator
# 导入所有需要的 Matplotlib 绘图函数作为备用方案
from .matplotlib_generator import (
    _draw_pie_chart, _draw_donut_chart, _draw_radar_chart, _draw_rose_chart,
    _draw_3d_bar_chart, _draw_sunburst_chart, _draw_treemap, _draw_waterfall_chart,
    _draw_contour_plot, _draw_network_diagram, _draw_forest_plot, _draw_funnel_chart,
    _draw_stacked_area_chart, _draw_sankey_diagram, _draw_candlestick_chart,
    _draw_gauge_chart, _draw_word_cloud, _draw_calendar_heatmap,
    _draw_parallel_coordinates, radar_factory
)

class SeabornGenerator(BaseChartGenerator):
    """
    使用 Seaborn 和 Matplotlib 兼容回退策略生成图表的生成器。
    """

    @property
    def file_extension(self) -> str:
        return "png"

    def _json_to_dataframe(self, chart_type: str) -> pd.DataFrame:
        data = self.chart_package.get('data', {})
        if chart_type in ['bar', 'line', 'area', 'line_with_confidence_interval']:
            records = [{'x': x, 'y': val, 'series': s.get('name')}
                       for s in data.get('y_series', [])
                       for x, val in zip(data.get('x_categories', []), s.get('values', []))]
            return pd.DataFrame(records)
        if chart_type in ['scatter', 'scatter_with_error_bars', 'bubble']:
            return pd.DataFrame(data.get('scatter_points', []))
        if chart_type in ['boxplot', 'violin', 'strip']:
            stat_data = data.get('statistical_data', {})
            records = [{'category': cat, 'value': val}
                       for i, cat in enumerate(stat_data.get('categories', []))
                       for val in stat_data.get('data_series', [])[i]]
            return pd.DataFrame(records)
        if chart_type == 'histogram':
            return pd.DataFrame(data.get('histogram_data', {}).get('values', []), columns=['values'])
        if chart_type == 'heatmap':
            heatmap_data = data.get('heatmap_data', {})
            return pd.DataFrame(heatmap_data.get('values', []), index=heatmap_data.get('y_labels', []), columns=heatmap_data.get('x_labels', []))
        return pd.DataFrame()

    def create_chart(self, output_path: str, figsize: tuple = (12, 8), dpi: int = 100) -> bool:
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei', 'sans-serif']
        sns.set_theme(style="whitegrid", rc={"axes.unicode_minus": False})
        fig, ax = None, None
        
        matplotlib_fallback_charts = [
            'pie', 'donut', 'radar', 'rose', '3d_bar', 'sunburst', 'treemap', 'waterfall',
            'contour', 'network', 'forest', 'funnel', 'stacked_area', 'sankey',
            'candlestick', 'gauge', 'word_cloud', 'calendar_heatmap', 'parallel_coordinates'
        ]

        try:
            if self.chart_type in matplotlib_fallback_charts:
                fig = plt.figure(figsize=figsize)
            else:
                fig, ax = plt.subplots(figsize=figsize)

            draw_func_map = {
                'bar': self._create_bar_chart, 'line': self._create_line_chart,
                'line_with_confidence_interval': self._create_line_chart,
                'scatter': self._create_scatter_chart, 'histogram': self._create_histogram,
                'heatmap': self._create_heatmap_chart, 'boxplot': self._create_statistical_plot,
                'violin': self._create_statistical_plot, 'strip': self._create_statistical_plot,
                **{chart: globals()[f'_draw_{chart}'] for chart in matplotlib_fallback_charts}
            }
            
            draw_func = draw_func_map.get(self.chart_type)
            if not draw_func: raise ValueError(f"不支持的图表类型: {self.chart_type}")

            if self.chart_type in matplotlib_fallback_charts:
                if self.chart_type in ['radar', 'rose', '3d_bar', 'contour', 'calendar_heatmap']:
                     ax = draw_func(fig, self.chart_package)
                else:
                     # 对于需要ax的matplotlib函数，先创建subplot
                     ax = fig.add_subplot(111)
                     draw_func(ax, self.chart_package)
            else:
                ax = draw_func(ax)
            
            if ax is None and self.chart_type != 'calendar_heatmap': return False

            self._apply_ax_settings(ax)
            fig.tight_layout()
            fig.savefig(output_path, dpi=dpi)
            return True
        except Exception as e:
            print(f"使用 Seaborn/Matplotlib 生成图表 '{self.chart_type}' 时发生错误: {e}")
            return False
        finally:
            if fig: plt.close(fig)

    def _apply_ax_settings(self, ax: plt.Axes):
        if ax:
            ax.set_title(self.chart_package.get('title', ''), fontsize=16)
            if self.chart_type not in ['pie', 'donut', 'radar', 'rose', 'heatmap', 'treemap', 'network', 'gauge', 'word_cloud', 'calendar_heatmap']:
                ax.set_xlabel(self.chart_package.get('x_label', ''))
                ax.set_ylabel(self.chart_package.get('y_label', ''))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    def _create_bar_chart(self, ax):
        df = self._json_to_dataframe(self.chart_type)
        sns.barplot(data=df, x='x', y='y', hue='series', ax=ax)
        return ax

    def _create_line_chart(self, ax):
        df = self._json_to_dataframe(self.chart_type)
        sns.lineplot(data=df, x='x', y='y', hue='series', marker='o', ax=ax)
        return ax

    def _create_scatter_chart(self, ax):
        df = self._json_to_dataframe(self.chart_type)
        sns.scatterplot(data=df, x='x', y='y', size='size', hue='color_value', sizes=(50, 500), ax=ax, palette='viridis')
        return ax

    def _create_histogram(self, ax):
        df = self._json_to_dataframe(self.chart_type)
        sns.histplot(data=df, x='values', bins=self.chart_package.get('data', {}).get('histogram_data', {}).get('bins', 'auto'), ax=ax, kde=True)
        return ax

    def _create_heatmap_chart(self, ax):
        df = self._json_to_dataframe(self.chart_type)
        sns.heatmap(df, annot=True, fmt=".1f", cmap="viridis", ax=ax)
        return ax

    def _create_statistical_plot(self, ax):
        df = self._json_to_dataframe(self.chart_type)
        plot_func = getattr(sns, self.chart_type + 'plot')
        plot_func(data=df, x='category', y='value', ax=ax)
        return ax

    def generate_code(self, figsize: tuple = (12, 8), dpi: int = 100) -> str:
        def json_serializer(obj):
            if isinstance(obj, datetime): return obj.isoformat()
            return str(obj)
        data_str = json.dumps(self.chart_package, indent=4, ensure_ascii=False, default=json_serializer)
        
        helper_funcs_to_include = []
        is_fallback = self.chart_type in [
            'pie', 'donut', 'radar', 'rose', '3d_bar', 'sunburst', 'treemap', 'waterfall',
            'contour', 'network', 'forest', 'funnel', 'stacked_area', 'sankey',
            'candlestick', 'gauge', 'word_cloud', 'calendar_heatmap', 'parallel_coordinates'
        ]

        if is_fallback:
            helper_funcs_to_include.append(globals()[f'_draw_{self.chart_type}'])
            if self.chart_type == 'radar':
                helper_funcs_to_include.append(radar_factory)
        else:
            helper_funcs_to_include.append(self._json_to_dataframe)
            helper_funcs_to_include.append(self._apply_ax_settings)
            create_func_name = f'_create_{self.chart_type}_chart' if hasattr(self, f'_create_{self.chart_type}_chart') else f'_create_{self.chart_type}'
            if hasattr(self, create_func_name):
                 helper_funcs_to_include.append(getattr(self, create_func_name))
            else:
                 helper_funcs_to_include.append(self._create_statistical_plot)

        sources = [inspect.getsource(func).replace('self, ', '', 1).replace('self.', '') for func in helper_funcs_to_include]
        helpers_code = "\n\n".join(sources)

        main_logic = inspect.getsource(self.create_chart)
        main_logic = main_logic.replace('self, output_path: str, figsize: tuple = (12, 8), dpi: int = 100', 'chart_package, figsize, dpi')
        main_logic = main_logic.replace('self.chart_package', 'chart_package').replace('self.chart_type', "chart_package.get('chart_type')")
        main_logic = main_logic.replace('self.', '').replace('create_chart', 'generate_the_chart').replace('fig.savefig(output_path, dpi=dpi)', 'plt.show()')

        return f"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
from matplotlib.sankey import Sankey
from wordcloud import WordCloud
import mplfinance as mpf
import calmap
from pandas.plotting import parallel_coordinates
from matplotlib.projections import register_projection
from mpl_toolkits.mplot3d import Axes3D
import squarify
import networkx as nx

CHART_PACKAGE_STR = '''{data_str}'''
FIGSIZE = {figsize}

# --- 辅助函数 ---
{helpers_code}

# --- 主逻辑 ---
{main_logic}

if __name__ == '__main__':
    chart_package = json.loads(CHART_PACKAGE_STR)
    generate_the_chart(chart_package, FIGSIZE, {dpi})
"""
