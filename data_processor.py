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

    if chart_type in ['line', 'bar', 'area', 'combo_bar_line']:
        y_series = data.get('y_series', [])
        if not y_series: return None
        # 收集所有非None的数据点
        for series in y_series:
            all_values.extend(v for v in series.get('data', []) if v is not None)
        if not all_values: return None
        return [min(all_values), max(all_values)]

    elif chart_type in ['stacked_bar', 'percentage_stacked_bar']:
        y_series = data.get('y_series', [])
        if not y_series or not y_series[0].get('data'): return None
        
        series_data = np.array([s.get('data', []) for s in y_series])
        
        if chart_type == 'percentage_stacked_bar':
            return [0, 100]
        else: # 普通堆叠图
            stacked_sums = np.sum(series_data, axis=0)
            max_val = np.max(stacked_sums)
            min_val = np.min(np.sum(series_data[series_data < 0], axis=0)) if np.any(series_data < 0) else 0
            return [float(min_val), float(max_val)]

    elif chart_type == 'boxplot':
        data_series = data.get('boxplot_data', {}).get('data_series', [])
        if not data_series: return None
        for box in data_series:
            all_values.extend(box)
        if not all_values: return None
        return [min(all_values), max(all_values)]
        
    return None

def _extract_relevant_data(full_data: Dict[str, Any]) -> Dict[str, Any]:
    # ... 此函数内容不变 ...
    simple_data = copy.deepcopy(full_data)
    chart_type = simple_data.get('chart_type')
    data = simple_data['data']

    if chart_type in ['bar', 'line', 'stacked_bar', 'percentage_stacked_bar', 'combo_bar_line']:
        if data.get('y_series'):
            data['y_series'] = [s for s in data['y_series'] if s.get('is_relevant_for_answer')]
    
    elif chart_type in ['donut', 'rose']:
        pie_data = data.get('pie_data')
        if pie_data and pie_data.get('is_relevant_for_answer'):
            flags = pie_data['is_relevant_for_answer']
            pie_data['labels'] = [lbl for lbl, rel in zip(pie_data['labels'], flags) if rel]
            pie_data['values'] = [val for val, rel in zip(pie_data['values'], flags) if rel]
            if pie_data.get('explode'):
                pie_data['explode'] = [exp for exp, rel in zip(pie_data['explode'], flags) if rel]
            pie_data.pop('is_relevant_for_answer', None)

    elif chart_type == 'radar':
        radar_data = data.get('radar_data')
        if radar_data and radar_data.get('series'):
            radar_data['series'] = [s for s in radar_data['series'] if s.get('is_relevant_for_answer')]

    elif chart_type == 'gantt':
        gantt_data = data.get('gantt_data')
        if gantt_data and gantt_data.get('tasks'):
            gantt_data['tasks'] = [t for t in gantt_data['tasks'] if t.get('is_relevant_for_answer')]

    elif chart_type == 'boxplot':
        boxplot_data = data.get('boxplot_data')
        if boxplot_data and boxplot_data.get('is_relevant_for_answer'):
            flags = boxplot_data['is_relevant_for_answer']
            boxplot_data['categories'] = [cat for cat, rel in zip(boxplot_data['categories'], flags) if rel]
            boxplot_data['data_series'] = [ds for ds, rel in zip(boxplot_data['data_series'], flags) if rel]
            boxplot_data.pop('is_relevant_for_answer', None)
        
    return simple_data


def create_chart_variations(full_chart_data: Dict[str, Any], total_versions: int = 5) -> List[Dict[str, Any]]:
    """
    接收带标注的完整图表数据，生成一个从简到繁、复杂度均衡递增的数据变体列表。
    修改版：
    1. 不会为了凑数而生成重复的数据。
    2. 对于 heatmap, area, pie, scatter 类型，直接返回单一完整版本。
    3. 为保持Y轴稳定，会计算并注入全局y_axis_range。
    """
    chart_type = full_chart_data.get('chart_type')
    if chart_type in ['heatmap', 'area', 'pie', 'scatter']:
        return [full_chart_data]

    # 在处理数据前，先从最完整的数据中计算Y轴范围
    y_axis_range = _get_y_axis_range(full_chart_data)

    if total_versions < 2:
        return [full_chart_data]

    variations = []
    
    simple_version = _extract_relevant_data(full_chart_data)
    variations.append(simple_version)

    data = full_chart_data.get('data', {})
    
    irrelevant_items = []
    irrelevant_indices = []

    if chart_type in ['bar', 'line', 'stacked_bar', 'percentage_stacked_bar', 'combo_bar_line']:
        irrelevant_items = [s for s in data.get('y_series', []) if not s.get('is_relevant_for_answer')]
    elif chart_type == 'radar':
        irrelevant_items = [s for s in data.get('radar_data', {}).get('series', []) if not s.get('is_relevant_for_answer')]
    elif chart_type == 'gantt':
        irrelevant_items = [t for t in data.get('gantt_data', {}).get('tasks', []) if not t.get('is_relevant_for_answer')]
    elif chart_type in ['donut', 'rose']:
        flags = data.get('pie_data', {}).get('is_relevant_for_answer', [])
        irrelevant_indices = [i for i, rel in enumerate(flags) if not rel]
    elif chart_type == 'boxplot':
        flags = data.get('boxplot_data', {}).get('is_relevant_for_answer', [])
        irrelevant_indices = [i for i, rel in enumerate(flags) if not rel]

    num_irrelevant = len(irrelevant_items) or len(irrelevant_indices)

    if num_irrelevant == 0:
        return [simple_version]

    random.shuffle(irrelevant_items)
    random.shuffle(irrelevant_indices)
    
    num_intermediate_steps = total_versions - 2
    num_intermediate_steps = min(num_intermediate_steps, num_irrelevant -1)

    for i in range(1, num_intermediate_steps + 1):
        num_to_add = round(num_irrelevant * (i / (num_intermediate_steps + 1)))
        num_to_add = max(1, int(num_to_add))
        current_data = copy.deepcopy(simple_version)

        if irrelevant_items:
            items_to_add = irrelevant_items[:num_to_add]
            if chart_type in ['bar', 'line', 'stacked_bar', 'percentage_stacked_bar', 'combo_bar_line']:
                current_data['data']['y_series'].extend(items_to_add)
            elif chart_type == 'radar':
                current_data['data']['radar_data']['series'].extend(items_to_add)
            elif chart_type == 'gantt':
                current_data['data']['gantt_data']['tasks'].extend(items_to_add)
            variations.append(current_data)

        elif irrelevant_indices:
            indices_to_add = irrelevant_indices[:num_to_add]
            if chart_type in ['donut', 'rose']:
                full_pie = full_chart_data['data']['pie_data']
                flags = full_pie.get('is_relevant_for_answer', [])
                relevant_indices = [idx for idx, rel in enumerate(flags) if rel]
                indices_to_include = sorted(relevant_indices + indices_to_add)
                
                current_data['data']['pie_data']['labels'] = [full_pie['labels'][j] for j in indices_to_include]
                current_data['data']['pie_data']['values'] = [full_pie['values'][j] for j in indices_to_include]
                if full_pie.get('explode'):
                    current_data['data']['pie_data']['explode'] = [full_pie['explode'][j] for j in indices_to_include]
            
            elif chart_type == 'boxplot':
                full_boxplot = full_chart_data['data']['boxplot_data']
                flags = full_boxplot.get('is_relevant_for_answer', [])
                relevant_indices = [idx for idx, rel in enumerate(flags) if rel]
                indices_to_include = sorted(relevant_indices + indices_to_add)
                
                current_data['data']['boxplot_data']['categories'] = [full_boxplot['categories'][j] for j in indices_to_include]
                current_data['data']['boxplot_data']['data_series'] = [full_boxplot['data_series'][j] for j in indices_to_include]
            variations.append(current_data)

    variations.append(full_chart_data)

    final_variations = []
    seen = set()
    for item in variations:
        item_str = json.dumps(item, sort_keys=True)
        if item_str not in seen:
            final_variations.append(item)
            seen.add(item_str)
    
    # 最后，为所有去重后的唯一变体注入Y轴范围
    final_variations_with_axis = []
    for variation in final_variations:
        if y_axis_range:
            variation['y_axis_range'] = copy.copy(y_axis_range)
        final_variations_with_axis.append(variation)

    return final_variations_with_axis