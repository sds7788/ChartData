import copy
import json
import random
from typing import Dict, Any, List
import numpy as np

def _get_y_axis_range(full_data: Dict[str, Any]) -> List[float] | None:
    """
    分析完整的图表数据，计算并返回Y轴的全局最小值和最大值。
    对于堆叠图，会计算堆叠后的总和作为最大值。
    """
    chart_type = full_data.get('chart_type')
    data = full_data.get('data', {})
    all_values = []

    if chart_type in ['line', 'bar', 'area', 'combo_bar_line', 'scatter', 'scatter_with_error_bars', 'line_with_confidence_interval']:
        # 适用于有 y_series 或 scatter_points 的图表
        if data.get('y_series'):
            for series in data.get('y_series', []):
                all_values.extend(v for v in series.get('values', []) if v is not None)
        if data.get('scatter_points'):
            all_values.extend(p.get('y') for p in data.get('scatter_points', []) if p.get('y') is not None)
        
        if not all_values: return None
        min_val, max_val = min(all_values), max(all_values)
        # 增加一些边距
        padding = (max_val - min_val) * 0.1
        return [min_val - padding, max_val + padding]

    elif chart_type in ['stacked_bar', 'stacked_area']:
        y_series = data.get('y_series', [])
        if not y_series or not y_series[0].get('values'): return None
        
        series_data = np.array([s.get('values', []) for s in y_series])
        stacked_sums = np.sum(series_data, axis=0)
        
        min_val = np.min([v for s in series_data for v in s if v is not None])
        max_val = np.max(stacked_sums)
        padding = (max_val - min_val) * 0.1
        return [min_val, max_val + padding]
        
    elif chart_type == 'boxplot' or chart_type == 'violin' or chart_type == 'strip':
        stat_data = data.get('statistical_data', {})
        if not stat_data or not stat_data.get('data_series'): return None
        all_values = [item for sublist in stat_data.get('data_series', []) for item in sublist]
        if not all_values: return None
        min_val, max_val = min(all_values), max(all_values)
        padding = (max_val - min_val) * 0.1
        return [min_val - padding, max_val + padding]

    return None


def create_chart_variations(full_chart_data: Dict[str, Any], total_versions: int) -> List[Dict[str, Any]]:
    """
    根据完整图表数据，生成多个不同复杂度（数据量）的版本。
    """
    if total_versions <= 1:
        return [full_chart_data]

    chart_type = full_chart_data.get('chart_type')

    # --- 【重要】修改点：识别并处理不适合生成多复杂度的图表类型 ---
    # 这些图表类型通常展示一个完整的整体或流程，删减数据会破坏其意义。
    simple_chart_types = {
        'pie', 'donut', 'funnel', 'waterfall', 'gauge', 'word_cloud', 
        'treemap', 'sunburst', 'forest', 'candlestick', 'sankey',
        'calendar_heatmap', 'network', 'contour', '3d_bar', 'percentage_stacked_bar',  'area', 'combo_bar_line'
    }
    if chart_type in simple_chart_types:
        print(f"  - 提示：图表类型 '{chart_type}' 不适合生成多复杂度版本，将只使用完整数据。")
        # 对于这些图表，只返回最原始、最完整的一个版本
        return [full_chart_data]
    # --- 修改结束 ---

    variations = []
    
    # 确定哪些数据是“核心”数据（回答问题所必需的）
    relevant_indices = []
    data_source = None
    flags = []

    if chart_type in ['line', 'bar', 'stacked_bar', 'stacked_area','line_with_confidence_interval']:
        data_source = full_chart_data.get('data', {}).get('y_series', [])
        flags = [s.get('is_relevant_for_answer', False) for s in data_source]
    elif chart_type in ['scatter', 'scatter_with_error_bars']:
        data_source = full_chart_data.get('data', {}).get('scatter_points', [])
        flags = [p.get('is_relevant_for_answer', False) for p in data_source]
    elif chart_type in ['boxplot', 'violin', 'strip']:
        data_source = full_chart_data.get('data', {}).get('statistical_data', {}).get('categories', [])
        flags = full_chart_data.get('data', {}).get('statistical_data', {}).get('is_relevant_for_answer', [])

    if data_source:
        relevant_indices = [idx for idx, rel in enumerate(flags) if rel]
        non_relevant_indices = [idx for idx, rel in enumerate(flags) if not rel]
        
        # 计算每个版本应该增加多少“非核心”数据
        num_steps = total_versions - 1
        items_to_add_per_step = len(non_relevant_indices) / num_steps if num_steps > 0 else 0

        for i in range(num_steps):
            current_data = copy.deepcopy(full_chart_data)
            num_to_add = int(items_to_add_per_step * (i + 1))
            
            # 随机选择要添加的非核心数据
            indices_to_add = random.sample(non_relevant_indices, min(num_to_add, len(non_relevant_indices)))
            indices_to_include = sorted(list(set(relevant_indices + indices_to_add)))

            # 根据图表类型，筛选数据
            if chart_type in ['line', 'bar', 'stacked_bar', 'stacked_area', 'line_with_confidence_interval']:
                current_data['data']['y_series'] = [data_source[j] for j in indices_to_include]
            elif chart_type in ['scatter', 'scatter_with_error_bars']:
                current_data['data']['scatter_points'] = [data_source[j] for j in indices_to_include]
            elif chart_type in ['boxplot', 'violin', 'strip']:
                full_stat_data = full_chart_data['data']['statistical_data']
                current_stat_data = current_data['data']['statistical_data']
                current_stat_data['categories'] = [full_stat_data['categories'][j] for j in indices_to_include]
                current_stat_data['data_series'] = [full_stat_data['data_series'][j] for j in indices_to_include]
            
            variations.append(current_data)

    variations.append(full_chart_data)

    # 去重，因为某些情况下可能生成重复的数据版本
    final_variations = []
    seen = set()
    for item in variations:
        item_str = json.dumps(item, sort_keys=True)
        if item_str not in seen:
            final_variations.append(item)
            seen.add(item_str)
    
    # 为所有版本注入统一的Y轴范围，以保证视觉上的可比性
    y_axis_range = _get_y_axis_range(full_chart_data)
    if y_axis_range:
        for v in final_variations:
            v['y_axis_range'] = y_axis_range

    return final_variations
