# chart_generators/seaborn_generator.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import inspect
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
import traceback

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
    支持单图和多子图（分面）渲染。
    """

    @property
    def file_extension(self) -> str:
        return "png"

    def _json_to_dataframe(self, chart_package: Dict[str, Any], chart_type: str) -> pd.DataFrame:
        """根据图表类型将传入的JSON数据转换为Pandas DataFrame。"""
        data = chart_package.get('data', {})
        # [修改] 扩展了支持的图表类型列表
        if chart_type in ['bar', 'line', 'area', 'line_with_confidence_interval', 'stacked_bar', 'percentage_stacked_bar', 'stacked_area']:
            records = [{'x': x, 'y': val, 'series': s.get('name')}
                       for s in data.get('y_series', [])
                       for x, val in zip(data.get('x_categories', []), s.get('values', []))]
            return pd.DataFrame(records)
            
        if chart_type in ['scatter', 'scatter_with_error_bars', 'bubble']:
            return pd.DataFrame(data.get('scatter_points', []))
            
        if chart_type in ['boxplot', 'violin', 'strip']:
            stat_data = data.get('statistical_data', {})
            # 【修改】使用 zip 确保类别和数据系列安全配对，避免因长度不匹配导致 IndexError
            records = [{'category': cat, 'value': val}
                       for cat, values in zip(stat_data.get('categories', []), stat_data.get('data_series', []))
                       for val in values]
            return pd.DataFrame(records)
            
        if chart_type == 'histogram':
            return pd.DataFrame(data.get('histogram_data', {}).get('values', []), columns=['values'])
            
        if chart_type == 'heatmap':
            heatmap_data = data.get('heatmap_data', {})
            return pd.DataFrame(heatmap_data.get('values', []), index=heatmap_data.get('y_labels', []), columns=heatmap_data.get('x_labels', []))
            
        return pd.DataFrame()

    def create_chart(self, output_path: str, figsize: tuple = (12, 8), dpi: int = 100) -> bool:
        """
        主创建函数，根据数据结构决定渲染单图还是多子图。
        """
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei', 'sans-serif']
        sns.set_theme(style="whitegrid", rc={"axes.unicode_minus": False})
        
        facet_data = self.chart_package.get('facet_data')

        if facet_data and isinstance(facet_data, list) and len(facet_data) > 0:
            return self._create_facet_chart(output_path, figsize, dpi)
        else:
            return self._create_single_chart(output_path, figsize, dpi)

    def _create_facet_chart(self, output_path: str, figsize: tuple, dpi: int) -> bool:
        """
        创建多子图（分面）图表。
        """
        fig = None
        facet_data = self.chart_package.get('facet_data')
        
        unsupported_facet_types = [
            'pie', 'donut', 'radar', 'rose', '3d_bar', 'sunburst', 'treemap', 'waterfall',
            'contour', 'network', 'forest', 'funnel', 'sankey', 'candlestick', 
            'gauge', 'word_cloud', 'calendar_heatmap', 'parallel_coordinates'
        ]
        if self.chart_type in unsupported_facet_types:
            print(f"警告：图表类型 '{self.chart_type}' (Matplotlib回退) 不支持多子图渲染，将只渲染第一个分面的数据。")
            self.chart_package['data'] = facet_data[0]['data']
            return self._create_single_chart(output_path, figsize, dpi)

        try:
            num_facets = len(facet_data)
            cols = int(np.ceil(np.sqrt(num_facets)))
            rows = int(np.ceil(num_facets / cols))
            
            fig_width = figsize[0] * cols / 2
            fig_height = figsize[1] * rows / 2.5

            fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=dpi, squeeze=False)
            axes_flat = axes.flatten()

            for i, facet_item in enumerate(facet_data):
                ax = axes_flat[i]
                subplot_package = {'data': facet_item.get('data', {})}
                self._draw_subplot(ax, subplot_package)
                ax.set_title(facet_item.get('facet_value', ''))
            
            for j in range(num_facets, len(axes_flat)):
                axes_flat[j].axis('off')
            
            fig.suptitle(self.chart_package.get('title', ''), fontsize=20, y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(output_path, dpi=dpi)
            return True
        except Exception as e:
            print(f"使用 Seaborn 生成多子图 '{self.chart_type}' 时发生错误: {e}")
            traceback.print_exc()
            return False
        finally:
            if fig: plt.close(fig)

    def _draw_subplot(self, ax: plt.Axes, data_package: Dict[str, Any]):
        """
        在指定的ax上绘制一个子图。
        """
        df = self._json_to_dataframe(data_package, self.chart_type)
        if df.empty:
            ax.text(0.5, 0.5, 'No data for this facet', ha='center', va='center')
            return

        # 【增强】扩展了多子图支持的图表类型，并为不支持的类型提供警告
        if self.chart_type == 'bar':
            sns.barplot(data=df, x='x', y='y', hue='series', ax=ax)
        elif self.chart_type in ['line', 'line_with_confidence_interval', 'area', 'stacked_area']:
            sns.lineplot(data=df, x='x', y='y', hue='series', marker='o', ax=ax)
            if 'area' in self.chart_type:
                 # 简单实现面积图效果，注意：Seaborn没有原生堆叠面积图，这里是重叠的
                for series_name in df['series'].unique():
                    series_df = df[df['series'] == series_name]
                    ax.fill_between(series_df['x'], series_df['y'], alpha=0.3)
        elif self.chart_type == 'scatter':
            sns.scatterplot(data=df, x='x', y='y', size='size', hue='color_value', sizes=(50, 500), ax=ax, palette='viridis')
        elif self.chart_type == 'histogram':
            sns.histplot(data=df, x='values', bins=data_package.get('data', {}).get('histogram_data', {}).get('bins', 'auto'), ax=ax, kde=True)
        elif self.chart_type == 'heatmap':
             sns.heatmap(df, annot=True, fmt=".1f", cmap="viridis", ax=ax)
        elif self.chart_type in ['boxplot', 'violin', 'strip']:
            # [修复] 修正了 seaborn 函数名的拼接问题，防止 'boxplot' 变成 'boxplotplot'
            func_name = self.chart_type
            if func_name != 'boxplot':
                func_name += 'plot'
            plot_func = getattr(sns, func_name)
            plot_func(data=df, x='category', y='value', ax=ax)
        else:
            print(f"警告：图表类型 '{self.chart_type}' 当前不被多子图渲染函数 _draw_subplot 支持。")

        self._apply_ax_settings(ax, data_package)


    def _create_single_chart(self, output_path: str, figsize: tuple, dpi: int) -> bool:
        """
        创建单个图表的逻辑，包含对Matplotlib的回退处理。
        """
        fig, ax = None, None
        
        matplotlib_fallback_charts = [
            'pie', 'donut', 'radar', 'rose', '3d_bar', 'sunburst', 'treemap', 'waterfall',
            'contour', 'network', 'forest', 'funnel', 'stacked_area', 'sankey',
            'candlestick', 'gauge', 'word_cloud', 'calendar_heatmap', 'parallel_coordinates'
        ]

        try:
            if 'data' not in self.chart_package or not self.chart_package['data']:
                 self.chart_package['data'] = {}

            if self.chart_type in matplotlib_fallback_charts:
                fig = plt.figure(figsize=figsize)
            else:
                fig, ax = plt.subplots(figsize=figsize)

            # [修复] 扩展了 draw_func_map 以支持之前未处理的图表类型
            draw_func_map = {
                'bar': self._create_bar_chart, 'line': self._create_line_chart,
                'area': self._create_area_chart,
                'stacked_bar': self._create_stacked_bar_chart,
                'percentage_stacked_bar': self._create_percentage_stacked_bar_chart,
                'line_with_confidence_interval': self._create_line_chart,
                'scatter': self._create_scatter_chart,
                'scatter_with_error_bars': self._create_scatter_with_error_bars_chart,
                'histogram': self._create_histogram,
                'heatmap': self._create_heatmap_chart, 'boxplot': self._create_statistical_plot,
                'violin': self._create_statistical_plot, 'strip': self._create_statistical_plot,
                'pie': _draw_pie_chart, 'donut': _draw_donut_chart, 'radar': _draw_radar_chart,
                'rose': _draw_rose_chart, '3d_bar': _draw_3d_bar_chart, 'sunburst': _draw_sunburst_chart,
                'treemap': _draw_treemap, 'waterfall': _draw_waterfall_chart, 'contour': _draw_contour_plot,
                'network': _draw_network_diagram, 'forest': _draw_forest_plot,
                'forest_plot': _draw_forest_plot, # 兼容 'forest_plot' 写法
                'funnel': _draw_funnel_chart,
                'stacked_area': _draw_stacked_area_chart, 'sankey': _draw_sankey_diagram,
                'candlestick': _draw_candlestick_chart, 'gauge': _draw_gauge_chart,
                'word_cloud': _draw_word_cloud, 'calendar_heatmap': _draw_calendar_heatmap,
                'parallel_coordinates': _draw_parallel_coordinates
            }
            
            draw_func = draw_func_map.get(self.chart_type)
            if not draw_func: raise ValueError(f"不支持的图表类型: {self.chart_type}")

            if self.chart_type in matplotlib_fallback_charts:
                if self.chart_type in ['radar', 'rose', '3d_bar', 'contour', 'calendar_heatmap']:
                     ax = draw_func(fig, self.chart_package)
                else:
                     ax = fig.add_subplot(111)
                     draw_func(ax, self.chart_package)
            else:
                ax = draw_func(ax)
            
            if ax is None and self.chart_type != 'calendar_heatmap': return False

            if fig and self.chart_package.get('title'):
                fig.suptitle(self.chart_package.get('title', ''), fontsize=16)
            elif ax and self.chart_package.get('title'):
                ax.set_title(self.chart_package.get('title', ''), fontsize=16)

            self._apply_ax_settings(ax, self.chart_package)

            fig.tight_layout()
            fig.savefig(output_path, dpi=dpi)
            return True
        except Exception as e:
            print(f"使用 Seaborn/Matplotlib 生成单图 '{self.chart_type}' 时发生错误: {e}")
            traceback.print_exc()
            return False
        finally:
            if fig: plt.close(fig)

    def _apply_ax_settings(self, ax: plt.Axes, data_package: Dict[str, Any]):
        """应用通用的坐标轴设置。"""
        if ax:
            if self.chart_type not in ['pie', 'donut', 'radar', 'rose', 'heatmap', 'treemap', 'network', 'gauge', 'word_cloud', 'calendar_heatmap']:
                ax.set_xlabel(data_package.get('x_label', self.chart_package.get('x_label', '')))
                ax.set_ylabel(data_package.get('y_label', self.chart_package.get('y_label', '')))
            if ax.get_xticklabels():
                plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # --- 以下是用于单图渲染的辅助函数 ---
    def _create_bar_chart(self, ax):
        df = self._json_to_dataframe(self.chart_package, self.chart_type)
        sns.barplot(data=df, x='x', y='y', hue='series', ax=ax)
        return ax

    def _create_line_chart(self, ax):
        df = self._json_to_dataframe(self.chart_package, self.chart_type)
        sns.lineplot(data=df, x='x', y='y', hue='series', marker='o', ax=ax)
        return ax
        
    # [新增] 添加面积图的实现
    def _create_area_chart(self, ax):
        df = self._json_to_dataframe(self.chart_package, 'area')
        if df.empty: return ax
        
        sns.lineplot(data=df, x='x', y='y', hue='series', marker='o', ax=ax)
        # 填充每个序列下的面积
        for series_name in sorted(df['series'].unique()):
            series_df = df[df['series'] == series_name].sort_values('x')
            ax.fill_between(series_df['x'], series_df['y'], alpha=0.3)
        return ax

    def _create_scatter_chart(self, ax):
        df = self._json_to_dataframe(self.chart_package, self.chart_type)
        sns.scatterplot(data=df, x='x', y='y', size='size', hue='color_value', sizes=(50, 500), ax=ax, palette='viridis')
        return ax
        
    # [新增] 添加带误差线的散点图实现
    def _create_scatter_with_error_bars_chart(self, ax):
        df = self._json_to_dataframe(self.chart_package, 'scatter_with_error_bars')
        if df.empty: return ax
        # 使用 matplotlib 的 errorbar 函数，它简单直接
        # 假设数据中包含 x, y, xerr, yerr 列
        ax.errorbar(x=df['x'], y=df['y'],
                    yerr=df.get('yerr'), 
                    xerr=df.get('xerr'),
                    fmt='o',  # 'o' 表示绘制点
                    capsize=5, # 误差线端点的帽子长度
                    linestyle='None') # 不连接数据点
        return ax
        
    # [新增] 添加堆叠条形图和百分比堆叠条形图的实现
    def _create_stacked_bar_chart(self, ax, normalize=False):
        df = self._json_to_dataframe(self.chart_package, self.chart_type)
        if df.empty: return ax
        
        # 将长格式数据透视为宽格式，便于绘制堆叠图
        pivot_df = df.pivot(index='x', columns='series', values='y').fillna(0)
        
        # 如果是百分比堆叠，进行归一化处理
        if normalize:
            pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0).mul(100)
            ax.set_ylabel(self.chart_package.get('y_label', 'Percentage (%)'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

        # 使用 pandas 内置的绘图功能在指定的 matplotlib ax 上绘图
        pivot_df.plot(kind='bar', stacked=True, ax=ax, width=0.8)
        
        # 优化图例位置，防止遮挡
        ax.legend(title='Series', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        return ax

    def _create_percentage_stacked_bar_chart(self, ax):
        return self._create_stacked_bar_chart(ax, normalize=True)

    def _create_histogram(self, ax):
        df = self._json_to_dataframe(self.chart_package, self.chart_type)
        sns.histplot(data=df, x='values', bins=self.chart_package.get('data', {}).get('histogram_data', {}).get('bins', 'auto'), ax=ax, kde=True)
        return ax

    def _create_heatmap_chart(self, ax):
        df = self._json_to_dataframe(self.chart_package, self.chart_type)
        sns.heatmap(df, annot=True, fmt=".1f", cmap="viridis", ax=ax)
        return ax

    def _create_statistical_plot(self, ax):
        df = self._json_to_dataframe(self.chart_package, self.chart_type)
        # [修复] 修正了 seaborn 函数名的拼接问题，防止 'boxplot' 变成 'boxplotplot'
        func_name = self.chart_type
        if func_name != 'boxplot':
            func_name += 'plot'
        plot_func = getattr(sns, func_name)
        plot_func(data=df, x='category', y='value', ax=ax)
        return ax

    def generate_code(self, figsize: tuple = (12, 8), dpi: int = 100) -> str:
        """
        生成可复现图表的 Python 代码。
        注意：此功能当前仅支持生成单图的代码。
        """
        def json_serializer(obj):
            if isinstance(obj, datetime): return obj.isoformat()
            return str(obj)
        data_str = json.dumps(self.chart_package, indent=4, ensure_ascii=False, default=json_serializer)
        
        is_fallback = self.chart_type in [
            'pie', 'donut', 'radar', 'rose', '3d_bar', 'sunburst', 'treemap', 'waterfall',
            'contour', 'network', 'forest', 'funnel', 'stacked_area', 'sankey',
            'candlestick', 'gauge', 'word_cloud', 'calendar_heatmap', 'parallel_coordinates'
        ]

        helper_funcs_to_include = []
        if is_fallback:
            # 动态获取函数名，处理命名不一致的问题
            func_name_map = {
                'network': '_draw_network_diagram',
                'forest': '_draw_forest_plot',
                'forest_plot': '_draw_forest_plot',
                # ... 其他特殊命名可以在此添加
            }
            # 默认函数名规则
            default_func_name = f'_draw_{self.chart_type}_chart'
            # 获取正确的函数名
            func_name = func_name_map.get(self.chart_type, default_func_name)
            
            if func_name in globals():
                 helper_funcs_to_include.append(globals()[func_name])
            elif f'_draw_{self.chart_type}' in globals(): # 兼容旧的或简单的命名
                 helper_funcs_to_include.append(globals()[f'_draw_{self.chart_type}'])

            if self.chart_type == 'radar':
                helper_funcs_to_include.append(radar_factory)

        sources = [inspect.getsource(func) for func in helper_funcs_to_include]
        helpers_code = "\n\n".join(sources)

        main_logic = inspect.getsource(self._create_single_chart)
        main_logic = main_logic.replace('self, output_path: str, figsize: tuple = (12, 8), dpi: int = 100', 'chart_package, figsize, dpi')
        main_logic = main_logic.replace('self.chart_package', 'chart_package').replace('self.chart_type', "chart_package.get('chart_type')")
        main_logic = main_logic.replace('self.', '').replace('_create_single_chart', 'generate_the_chart').replace('fig.savefig(output_path, dpi=dpi)', 'plt.show()')

        return f"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
# 导入所有可能的依赖
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
# 警告：为确保独立运行，请将所有在 `generate_the_chart` 中调用的绘图函数 (_draw_..._chart) 粘贴于此
{helpers_code}

# --- 主逻辑 ---
{main_logic}

if __name__ == '__main__':
    chart_package = json.loads(CHART_PACKAGE_STR)
    
    # 【修改】同步修复了这里的健壮性问题
    def _json_to_dataframe(chart_package, chart_type):
        data = chart_package.get('data', {{}})
        if chart_type in ['bar', 'line', 'area', 'line_with_confidence_interval', 'stacked_bar', 'percentage_stacked_bar']:
            records = [{{'x': x, 'y': val, 'series': s.get('name')}}
                       for s in data.get('y_series', [])
                       for x, val in zip(data.get('x_categories', []), s.get('values', []))]
            return pd.DataFrame(records)
        if chart_type in ['scatter', 'scatter_with_error_bars', 'bubble']:
            return pd.DataFrame(data.get('scatter_points', []))
        if chart_type in ['boxplot', 'violin', 'strip']:
            stat_data = data.get('statistical_data', {{}})
            records = [{{'category': cat, 'value': val}}
                       for cat, values in zip(stat_data.get('categories', []), stat_data.get('data_series', []))
                       for val in values]
            return pd.DataFrame(records)
        if chart_type == 'histogram':
            return pd.DataFrame(data.get('histogram_data', {{}}).get('values', []), columns=['values'])
        if chart_type == 'heatmap':
            heatmap_data = data.get('heatmap_data', {{}})
            return pd.DataFrame(heatmap_data.get('values', []), index=heatmap_data.get('y_labels', []), columns=heatmap_data.get('x_labels', []))
        return pd.DataFrame()

    def _create_bar_chart(ax):
        df = _json_to_dataframe(chart_package, 'bar')
        sns.barplot(data=df, x='x', y='y', hue='series', ax=ax)
        return ax
    def _create_line_chart(ax):
        df = _json_to_dataframe(chart_package, 'line')
        sns.lineplot(data=df, x='x', y='y', hue='series', marker='o', ax=ax)
        return ax
    # [新增] 添加了缺失的绘图函数的本地实现
    def _create_area_chart(ax):
        df = _json_to_dataframe(chart_package, 'area')
        sns.lineplot(data=df, x='x', y='y', hue='series', marker='o', ax=ax)
        for series_name in sorted(df['series'].unique()):
            series_df = df[df['series'] == series_name].sort_values('x')
            ax.fill_between(series_df['x'], series_df['y'], alpha=0.3)
        return ax
    def _create_stacked_bar_chart(ax, normalize=False):
        df = _json_to_dataframe(chart_package, 'stacked_bar')
        pivot_df = df.pivot(index='x', columns='series', values='y').fillna(0)
        if normalize:
            pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0).mul(100)
            ax.set_ylabel('Percentage (%)')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{{y:.0f}}%'))
        pivot_df.plot(kind='bar', stacked=True, ax=ax, width=0.8)
        ax.legend(title='Series', bbox_to_anchor=(1.05, 1), loc='upper left')
        return ax
    def _create_percentage_stacked_bar_chart(ax):
        return _create_stacked_bar_chart(ax, normalize=True)
    def _create_scatter_with_error_bars_chart(ax):
        df = _json_to_dataframe(chart_package, 'scatter_with_error_bars')
        ax.errorbar(x=df['x'], y=df['y'], yerr=df.get('yerr'), xerr=df.get('xerr'), fmt='o', capsize=5, linestyle='None')
        return ax
    def _create_scatter_chart(ax):
        df = _json_to_dataframe(chart_package, 'scatter')
        sns.scatterplot(data=df, x='x', y='y', size='size', hue='color_value', sizes=(50, 500), ax=ax, palette='viridis')
        return ax
    def _create_histogram(ax):
        df = _json_to_dataframe(chart_package, 'histogram')
        sns.histplot(data=df, x='values', bins=chart_package.get('data', {{}}).get('histogram_data', {{}}).get('bins', 'auto'), ax=ax, kde=True)
        return ax
    def _create_heatmap_chart(ax):
        df = _json_to_dataframe(chart_package, 'heatmap')
        sns.heatmap(df, annot=True, fmt=".1f", cmap="viridis", ax=ax)
        return ax
    def _create_statistical_plot(ax):
        chart_type = chart_package.get('chart_type')
        df = _json_to_dataframe(chart_package, chart_type)
        func_name = chart_type
        if func_name != 'boxplot':
            func_name += 'plot'
        plot_func = getattr(sns, func_name)
        plot_func(data=df, x='category', y='value', ax=ax)
        return ax
    def _apply_ax_settings(ax, data_package):
        chart_type = data_package.get('chart_type')
        if ax and chart_type not in ['pie', 'donut', 'radar', 'rose', 'heatmap', 'treemap', 'network', 'gauge', 'word_cloud', 'calendar_heatmap']:
            ax.set_xlabel(data_package.get('x_label', ''))
            ax.set_ylabel(data_package.get('y_label', ''))
        if ax.get_xticklabels():
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    print("即将生成图表... 注意: 自动生成的代码可能需要手动引入Matplotlib备用函数才能运行。")
    # generate_the_chart(chart_package, FIGSIZE, {dpi})

"""