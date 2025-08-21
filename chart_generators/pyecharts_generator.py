# chart_generators/pyecharts_generator.py
import json
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie, Scatter, Radar, HeatMap, Boxplot
from typing import Dict, Any

from .base_generator import BaseChartGenerator

class PyechartsGenerator(BaseChartGenerator):
    """使用 Pyecharts 生成交互式 HTML 图表的生成器 (支持全类型和分辨率控制)。"""

    @property
    def file_extension(self) -> str:
        return "html"

    def create_chart(self, output_path: str, figsize: tuple = (12, 8), dpi: int = 100) -> bool:
        """根据数据包和分辨率参数创建图表并渲染到HTML文件。"""
        try:
            # 将 figsize 和 dpi 转换为 Pyecharts 使用的像素尺寸
            width_px = f"{int(figsize[0] * dpi)}px"
            height_px = f"{int(figsize[1] * dpi)}px"
            init_opts = opts.InitOpts(width=width_px, height=height_px)
            
            # 将初始化配置传递给基础图表创建函数
            chart = self._create_base_chart(init_opts=init_opts)
            
            if chart:
                chart.render(output_path)
                print(f"Pyecharts 图表已成功生成: {output_path} (尺寸: {width_px} x {height_px})")
                return True
            print(f"Pyecharts 当前不支持或未能处理的图表类型: {self.chart_type}")
            return False
        except Exception as e:
            print(f"使用 Pyecharts 生成图表 '{self.chart_type}' 时发生严重错误: {e}")
            return False

    def _create_base_chart(self, init_opts: opts.InitOpts):
        """图表创建的分发器，接收init_opts并传递给具体的创建函数。"""
        title = self.chart_package.get('title', '')
        
        draw_func_map = {
            'bar': self._create_bar_chart,
            'line': self._create_line_chart,
            'pie': self._create_pie_chart,
            'scatter': self._create_scatter_chart,
            'area': self._create_area_chart,
            'radar': self._create_radar_chart,
            'rose': self._create_rose_chart,
            'gantt': self._create_gantt_chart,
            'combo_bar_line': self._create_combo_bar_line_chart,
            'heatmap': self._create_heatmap_chart,
            'donut': self._create_donut_chart,
            'stacked_bar': self._create_stacked_bar_chart,
            'boxplot': self._create_boxplot_chart,
            'percentage_stacked_bar': self._create_percentage_stacked_bar_chart,
        }
        
        draw_func = draw_func_map.get(self.chart_type)
        if draw_func:
            return draw_func(title=title, init_opts=init_opts)
        return None

    def _get_common_data(self):
        """提取通用的 x_categories 和 y_series 数据。"""
        return (
            self.chart_package.get('data', {}).get('x_categories', []),
            self.chart_package.get('data', {}).get('y_series', [])
        )

    def _get_yaxis_opts(self) -> opts.AxisOpts | None:
        """辅助函数，生成Y轴配置"""
        if 'y_axis_range' in self.chart_package and self.chart_package['y_axis_range']:
            y_range = self.chart_package['y_axis_range']
            return opts.AxisOpts(min_=y_range[0], max_=y_range[1])
        return None

    def _create_bar_chart(self, title: str, init_opts: opts.InitOpts):
        x_categories, y_series = self._get_common_data()
        if not x_categories or not y_series: return None
        c = (
            Bar(init_opts=init_opts)
            .add_xaxis(x_categories)
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title, pos_left="center"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow"),
                legend_opts=opts.LegendOpts(pos_left="center", pos_top="bottom"),
                yaxis_opts=self._get_yaxis_opts()
            )
        )
        for series in y_series:
            c.add_yaxis(series.get('name'), series.get('values'))
        return c

    def _create_stacked_bar_chart(self, title: str, init_opts: opts.InitOpts, percentage=False):
        x_categories, y_series = self._get_common_data()
        if not x_categories or not y_series: return None
        
        values = np.array([s.get('values', [0]*len(x_categories)) for s in y_series])
        if percentage:
            totals = np.where(values.sum(axis=0) == 0, 1, values.sum(axis=0))
            values = (values / totals[np.newaxis, :]) * 100

        yaxis_options = self._get_yaxis_opts()
        if percentage:
            label_opts = opts.LabelOpts(formatter="{value} %")
            if yaxis_options:
                yaxis_options.opts['axislabel'] = label_opts
            else:
                yaxis_options = opts.AxisOpts(axislabel_opts=label_opts)
            
        c = (
            Bar(init_opts=init_opts)
            .add_xaxis(x_categories)
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title, pos_left="center"),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow"),
                legend_opts=opts.LegendOpts(pos_left="center", pos_top="bottom"),
                yaxis_opts=yaxis_options
            )
        )
        for i, series in enumerate(y_series):
            c.add_yaxis(series.get('name'), values[i].round(2).tolist(), stack="stack1")
        return c

    def _create_percentage_stacked_bar_chart(self, title: str, init_opts: opts.InitOpts):
        return self._create_stacked_bar_chart(title, init_opts, percentage=True)

    def _create_line_chart(self, title: str, init_opts: opts.InitOpts, with_area=False):
        x_categories, y_series = self._get_common_data()
        if not x_categories or not y_series: return None
        c = (
            Line(init_opts=init_opts)
            .add_xaxis(x_categories)
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title, pos_left="center"),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                legend_opts=opts.LegendOpts(pos_left="center", pos_top="bottom"),
                yaxis_opts=self._get_yaxis_opts()
            )
        )
        for series in y_series:
            c.add_yaxis(
                series_name=series.get('name'), 
                y_axis=series.get('values'),
                areastyle_opts=opts.AreaStyleOpts(opacity=0.5) if with_area else None
            )
        return c

    def _create_area_chart(self, title: str, init_opts: opts.InitOpts):
        return self._create_line_chart(title, init_opts, with_area=True)

    def _create_pie_chart(self, title: str, init_opts: opts.InitOpts, is_donut=False, is_rose=False):
        pie_data = self.chart_package.get('data', {}).get('pie_data', {})
        labels = pie_data.get('labels', [])
        values = pie_data.get('values', [])
        if not labels or not values: return None
        
        radius = ["40%", "70%"] if is_donut else ["0%", "75%"]
        rose_type = 'radius' if is_rose else None
        
        c = (
            Pie(init_opts=init_opts)
            .add(
                "", [list(z) for z in zip(labels, values)],
                radius=radius, center=["50%", "55%"], rosetype=rose_type
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title, pos_left="center"),
                legend_opts=opts.LegendOpts(orient="vertical", pos_top="15%", pos_left="2%")
            )
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)"))
        )
        return c

    def _create_donut_chart(self, title: str, init_opts: opts.InitOpts):
        return self._create_pie_chart(title, init_opts, is_donut=True)

    def _create_rose_chart(self, title: str, init_opts: opts.InitOpts):
        return self._create_pie_chart(title, init_opts, is_rose=True)

    def _create_scatter_chart(self, title: str, init_opts: opts.InitOpts):
        points = self.chart_package.get('data', {}).get('scatter_points', [])
        if not points: return None
        x_label, y_label = self.chart_package.get('x_label', 'X'), self.chart_package.get('y_label', 'Y')
        max_size = max([p.get('size', 1) for p in points])
        scatter_data = [[p.get('x'), p.get('y'), p.get('size', 1)] for p in points]
        
        yaxis_options = self._get_yaxis_opts()
        if yaxis_options:
            yaxis_options.opts['name'] = y_label
            yaxis_options.opts['type'] = "value"
            yaxis_options.opts['splitline_opts'] = opts.SplitLineOpts(is_show=True)
        else:
            yaxis_options = opts.AxisOpts(name=y_label, type_="value", splitline_opts=opts.SplitLineOpts(is_show=True))

        c = (
            Scatter(init_opts=init_opts)
            .add_xaxis([item[0] for item in scatter_data])
            .add_yaxis("数据点", [item[1:] for item in scatter_data], label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title, pos_left="center"),
                tooltip_opts=opts.TooltipOpts(formatter=f"{{a}}<br/>{x_label}: {{b}}<br/>{y_label}: {{c}}[0]<br/>大小: {{c}}[1]"),
                visualmap_opts=opts.VisualMapOpts(type_="size", max_=max_size, min_=0, pos_left="90%"),
                xaxis_opts=opts.AxisOpts(name=x_label, type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)),
                yaxis_opts=yaxis_options
            )
        )
        return c

    def _create_radar_chart(self, title: str, init_opts: opts.InitOpts):
        radar_data = self.chart_package.get('data', {}).get('radar_data', {})
        labels = radar_data.get('labels', [])
        series_data = radar_data.get('series', [])
        if not labels or not series_data: return None

        max_val = 0
        if 'y_axis_range' in self.chart_package and self.chart_package['y_axis_range']:
            max_val = self.chart_package['y_axis_range'][1]
        else:
            all_values = [v for s in series_data for v in s.get('values', [])]
            max_val = max(all_values) * 1.1 if all_values else 0
        
        indicator = [{"name": label, "max": max_val} for label in labels]
        c = (
            Radar(init_opts=init_opts)
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title, pos_left="center"),
                legend_opts=opts.LegendOpts(pos_left="center", pos_top="bottom")
            )
            .add_schema(schema=indicator)
        )
        for series in series_data:
            c.add(series.get('name'), [series.get('values')], areastyle_opts=opts.AreaStyleOpts(opacity=0.3))
        return c

    def _create_heatmap_chart(self, title: str, init_opts: opts.InitOpts):
        heatmap_data = self.chart_package.get('data', {}).get('heatmap_data', {})
        x_labels = heatmap_data.get('x_labels', [])
        y_labels = heatmap_data.get('y_labels', [])
        values = heatmap_data.get('values', []) 
        if not x_labels or not y_labels or not values: return None
        
        all_vals = [v[2] for v in values]
        min_v, max_v = min(all_vals), max(all_vals)
        c = (
            HeatMap(init_opts=init_opts)
            .add_xaxis(x_labels)
            .add_yaxis("热力值", y_labels, values)
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title, pos_left="center"),
                legend_opts=opts.LegendOpts(is_show=False),
                visualmap_opts=opts.VisualMapOpts(min_=min_v, max_=max_v, pos_left="90%")
            )
        )
        return c

    def _create_boxplot_chart(self, title: str, init_opts: opts.InitOpts):
        boxplot_data = self.chart_package.get('data', {}).get('boxplot_data', {})
        categories = boxplot_data.get('categories', [])
        data_series = boxplot_data.get('data_series', [])
        if not categories or not data_series: return None
        
        c = (
            Boxplot(init_opts=init_opts)
            .add_xaxis(categories)
            .add_yaxis("数值", Boxplot.prepare_data(data_series))
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title, pos_left="center"),
                yaxis_opts=self._get_yaxis_opts()
            )
        )
        return c

    def _create_gantt_chart(self, title: str, init_opts: opts.InitOpts):
        gantt_data = self.chart_package.get('data', {}).get('gantt_data', {})
        tasks = gantt_data.get('tasks', [])
        if not tasks: return None
        
        task_names = [t.get('name') for t in reversed(tasks)]
        starts = [t.get('start') for t in reversed(tasks)]
        durations = [t.get('duration') for t in reversed(tasks)]
        
        c = (
            Bar(init_opts=init_opts)
            .add_xaxis(task_names)
            .add_yaxis("开始时间", starts, stack="gantt", itemstyle_opts=opts.ItemStyleOpts(color="transparent"))
            .add_yaxis("持续时间", durations, stack="gantt", label_opts=opts.LabelOpts(is_show=True, position="right"))
            .reversal_axis()
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title, pos_left="center"),
                xaxis_opts=opts.AxisOpts(type_="value", name="时间/天"),
                yaxis_opts=opts.AxisOpts(type_="category", name="任务"),
                legend_opts=opts.LegendOpts(is_show=False)
            )
        )
        return c

    def _create_combo_bar_line_chart(self, title: str, init_opts: opts.InitOpts):
        x_categories, y_series = self._get_common_data()
        if not x_categories or not len(y_series) >= 2: return None
        
        bar_series = [s for s in y_series if s.get('type') == 'bar']
        line_series = [s for s in y_series if s.get('type') == 'line']

        bar = Bar(init_opts=init_opts).add_xaxis(x_categories)
        for s in bar_series:
            bar.add_yaxis(s.get('name'), s.get('values'))

        line = Line(init_opts=init_opts).add_xaxis(x_categories)
        for s in line_series:
            line.add_yaxis(s.get('name'), s.get('values'))

        bar.overlap(line)
        bar.set_global_opts(
            title_opts=opts.TitleOpts(title=title, pos_left="center"),
            legend_opts=opts.LegendOpts(pos_left="center", pos_top="bottom"),
            yaxis_opts=self._get_yaxis_opts()
        )
        return bar

    def generate_code(self, figsize: tuple = (12, 8), dpi: int = 100) -> str:
        """生成一个真正完整的、包含所有14种图表逻辑的复现脚本。"""
        data_str = json.dumps(self.chart_package, indent=4, ensure_ascii=False)
        
        # 将所有生成逻辑嵌入到脚本字符串中
        return f"""
# -*- coding: utf-8 -*-
# 可复现的 Pyecharts 图表生成脚本
# 请确保已安装: pip install pyecharts numpy
import json
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie, Scatter, Radar, HeatMap, Boxplot
from typing import Dict, Any

# --- 数据包和配置 ---
CHART_PACKAGE: Dict[str, Any] = {data_str}
FIGSIZE = {figsize}
DPI = {dpi}

# --- 以下是所有图表类型的生成函数 ---

def _gen_bar(pkg, init_opts):
    title, x_cat, y_ser = pkg.get('title',''), pkg.get('data',{{}}).get('x_categories',[]), pkg.get('data',{{}}).get('y_series',[])
    y_range = pkg.get('y_axis_range')
    yaxis_opts = opts.AxisOpts(min_=y_range[0], max_=y_range[1]) if y_range else None
    c = Bar(init_opts=init_opts).add_xaxis(x_cat).set_global_opts(title_opts=opts.TitleOpts(title=title, pos_left="center"), tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow"), legend_opts=opts.LegendOpts(pos_left="center", pos_top="bottom"), yaxis_opts=yaxis_opts)
    for s in y_ser: c.add_yaxis(s.get('name'), s.get('values'))
    return c

def _gen_stacked_bar(pkg, init_opts, percentage=False):
    title, x_cat, y_ser = pkg.get('title',''), pkg.get('data',{{}}).get('x_categories',[]), pkg.get('data',{{}}).get('y_series',[])
    y_range = pkg.get('y_axis_range')
    vals = np.array([s.get('values', [0]*len(x_cat)) for s in y_ser])
    if percentage:
        totals = np.where(vals.sum(axis=0) == 0, 1, vals.sum(axis=0))
        vals = (vals / totals[np.newaxis, :]) * 100
    
    yaxis_opts = opts.AxisOpts(min_=y_range[0], max_=y_range[1]) if y_range else None
    if percentage:
        label_opts = opts.LabelOpts(formatter="{{value}} %")
        if yaxis_opts: yaxis_opts.opts['axislabel'] = label_opts
        else: yaxis_opts = opts.AxisOpts(axislabel_opts=label_opts)

    c = Bar(init_opts=init_opts).add_xaxis(x_cat).set_global_opts(title_opts=opts.TitleOpts(title=title, pos_left="center"), tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow"), legend_opts=opts.LegendOpts(pos_left="center", pos_top="bottom"), yaxis_opts=yaxis_opts)
    for i, s in enumerate(y_ser): c.add_yaxis(s.get('name'), vals[i].round(2).tolist(), stack="stack1")
    return c

def _gen_line(pkg, init_opts, with_area=False):
    title, x_cat, y_ser = pkg.get('title',''), pkg.get('data',{{}}).get('x_categories',[]), pkg.get('data',{{}}).get('y_series',[])
    y_range = pkg.get('y_axis_range')
    yaxis_opts = opts.AxisOpts(min_=y_range[0], max_=y_range[1]) if y_range else None
    c = Line(init_opts=init_opts).add_xaxis(x_cat).set_global_opts(title_opts=opts.TitleOpts(title=title, pos_left="center"), tooltip_opts=opts.TooltipOpts(trigger="axis"), legend_opts=opts.LegendOpts(pos_left="center", pos_top="bottom"), yaxis_opts=yaxis_opts)
    for s in y_ser: c.add_yaxis(series_name=s.get('name'), y_axis=s.get('values'), areastyle_opts=opts.AreaStyleOpts(opacity=0.5) if with_area else None)
    return c

def _gen_pie(pkg, init_opts, is_donut=False, is_rose=False):
    title, pie_data = pkg.get('title',''), pkg.get('data',{{}}).get('pie_data',{{}})
    labels, values = pie_data.get('labels',[]), pie_data.get('values',[])
    radius = ["40%", "70%"] if is_donut else ["0%", "75%"]; rose_type = 'radius' if is_rose else None
    c = Pie(init_opts=init_opts).add("", [list(z) for z in zip(labels, values)], radius=radius, center=["50%", "55%"], rosetype=rose_type).set_global_opts(title_opts=opts.TitleOpts(title=title, pos_left="center"), legend_opts=opts.LegendOpts(orient="vertical", pos_top="15%", pos_left="2%")).set_series_opts(label_opts=opts.LabelOpts(formatter="{{b}}: {{c}} ({{d}}%)"))
    return c

def _gen_scatter(pkg, init_opts):
    title, points, y_range = pkg.get('title',''), pkg.get('data',{{}}).get('scatter_points',[]), pkg.get('y_axis_range')
    xl, yl = pkg.get('x_label','X'), pkg.get('y_label','Y')
    max_size = max([p.get('size', 1) for p in points]) if points else 1
    sd = [[p.get('x'), p.get('y'), p.get('size', 1)] for p in points]
    yaxis_opts = opts.AxisOpts(min_=y_range[0], max_=y_range[1]) if y_range else opts.AxisOpts()
    yaxis_opts.opts['name'], yaxis_opts.opts['type'], yaxis_opts.opts['splitline_opts'] = yl, "value", opts.SplitLineOpts(is_show=True)
    c = Scatter(init_opts=init_opts).add_xaxis([i[0] for i in sd]).add_yaxis("Data", [i[1:] for i in sd], label_opts=opts.LabelOpts(is_show=False)).set_global_opts(title_opts=opts.TitleOpts(title=title,pos_left="center"),tooltip_opts=opts.TooltipOpts(formatter=f"{{a}}<br/>{{xl}}: {{b}}<br/>{{yl}}: {{c}}[0]<br/>Size: {{c}}[1]"),visualmap_opts=opts.VisualMapOpts(type_="size",max_=max_size,min_=0,pos_left="90%"),xaxis_opts=opts.AxisOpts(name=xl,type_="value",splitline_opts=opts.SplitLineOpts(is_show=True)),yaxis_opts=yaxis_opts)
    return c

def _gen_radar(pkg, init_opts):
    title, radar_data, y_range = pkg.get('title',''), pkg.get('data',{{}}).get('radar_data',{{}}), pkg.get('y_axis_range')
    labels, series_data = radar_data.get('labels',[]), radar_data.get('series',[])
    max_val = y_range[1] if y_range else (max([v for s in series_data for v in s.get('values',[])])*1.1 if [v for s in series_data for v in s.get('values',[])] else 0)
    indicator = [{{"name":lbl,"max":max_val}} for lbl in labels]
    c = Radar(init_opts=init_opts).set_global_opts(title_opts=opts.TitleOpts(title=title,pos_left="center"),legend_opts=opts.LegendOpts(pos_left="center",pos_top="bottom")).add_schema(schema=indicator)
    for s in series_data: c.add(s.get('name'), [s.get('values')], areastyle_opts=opts.AreaStyleOpts(opacity=0.3))
    return c

def _gen_heatmap(pkg, init_opts):
    title, hmap_data = pkg.get('title',''), pkg.get('data',{{}}).get('heatmap_data',{{}})
    xl, yl, vals = hmap_data.get('x_labels',[]), hmap_data.get('y_labels',[]), hmap_data.get('values',[])
    all_v = [v[2] for v in vals]; min_v, max_v = min(all_v), max(all_v)
    c = HeatMap(init_opts=init_opts).add_xaxis(xl).add_yaxis("Value", yl, vals).set_global_opts(title_opts=opts.TitleOpts(title=title,pos_left="center"),legend_opts=opts.LegendOpts(is_show=False),visualmap_opts=opts.VisualMapOpts(min_=min_v,max_=max_v,pos_left="90%"))
    return c

def _gen_boxplot(pkg, init_opts):
    title, box_data, y_range = pkg.get('title',''), pkg.get('data',{{}}).get('boxplot_data',{{}}), pkg.get('y_axis_range')
    cats, series = box_data.get('categories',[]), box_data.get('data_series',[])
    yaxis_opts = opts.AxisOpts(min_=y_range[0], max_=y_range[1]) if y_range else None
    c = Boxplot(init_opts=init_opts).add_xaxis(cats).add_yaxis("Value", Boxplot.prepare_data(series)).set_global_opts(title_opts=opts.TitleOpts(title=title,pos_left="center"), yaxis_opts=yaxis_opts)
    return c

def _gen_gantt(pkg, init_opts):
    title, gantt_data = pkg.get('title',''), pkg.get('data',{{}}).get('gantt_data',{{}})
    tasks = gantt_data.get('tasks',[])
    tn, st, dr = [t.get('name') for t in reversed(tasks)], [t.get('start') for t in reversed(tasks)], [t.get('duration') for t in reversed(tasks)]
    c = Bar(init_opts=init_opts).add_xaxis(tn).add_yaxis("Start",st,stack="g",itemstyle_opts=opts.ItemStyleOpts(color="transparent")).add_yaxis("Duration",dr,stack="g",label_opts=opts.LabelOpts(is_show=True,position="right")).reversal_axis().set_global_opts(title_opts=opts.TitleOpts(title=title,pos_left="center"),xaxis_opts=opts.AxisOpts(type_="value",name="Time"),yaxis_opts=opts.AxisOpts(type_="category",name="Task"),legend_opts=opts.LegendOpts(is_show=False))
    return c

def _gen_combo(pkg, init_opts):
    title, x_cat, y_ser, y_range = pkg.get('title',''), pkg.get('data',{{}}).get('x_categories',[]), pkg.get('data',{{}}).get('y_series',[]), pkg.get('y_axis_range')
    bar_s, line_s = [s for s in y_ser if s.get('type')=='bar'], [s for s in y_ser if s.get('type')=='line']
    yaxis_opts = opts.AxisOpts(min_=y_range[0], max_=y_range[1]) if y_range else None
    bar = Bar(init_opts=init_opts).add_xaxis(x_cat)
    for s in bar_s: bar.add_yaxis(s.get('name'),s.get('values'))
    line = Line(init_opts=init_opts).add_xaxis(x_cat)
    for s in line_s: line.add_yaxis(s.get('name'),s.get('values'))
    bar.overlap(line).set_global_opts(title_opts=opts.TitleOpts(title=title,pos_left="center"),legend_opts=opts.LegendOpts(pos_left="center",pos_top="bottom"), yaxis_opts=yaxis_opts)
    return bar


# --- 主执行逻辑 ---
def main():
    chart_type = CHART_PACKAGE.get('chart_type')
    
    width_px = f"{{int(FIGSIZE[0] * DPI)}}px"
    height_px = f"{{int(FIGSIZE[1] * DPI)}}px"
    init_opts = opts.InitOpts(width=width_px, height=height_px)
    
    FUNC_MAP = {{
        "bar": _gen_bar, "line": lambda p, i: _gen_line(p, i, with_area=False),
        "pie": lambda p, i: _gen_pie(p, i, is_donut=False, is_rose=False),
        "scatter": _gen_scatter, "area": lambda p, i: _gen_line(p, i, with_area=True),
        "radar": _gen_radar, "rose": lambda p, i: _gen_pie(p, i, is_rose=True),
        "gantt": _gen_gantt, "combo_bar_line": _gen_combo,
        "heatmap": _gen_heatmap, "donut": lambda p, i: _gen_pie(p, i, is_donut=True),
        "stacked_bar": lambda p, i: _gen_stacked_bar(p, i, percentage=False),
        "boxplot": _gen_boxplot,
        "percentage_stacked_bar": lambda p, i: _gen_stacked_bar(p, i, percentage=True)
    }}
    
    gen_func = FUNC_MAP.get(chart_type)
    
    if not gen_func:
        print(f"错误：此脚本不支持图表类型 '{{chart_type}}'。")
        return
        
    chart_object = gen_func(CHART_PACKAGE, init_opts)
    
    if chart_object:
        output_filename = f"reproducible_{{chart_type}}.html"
        chart_object.render(output_filename)
        print(f"图表已成功生成: {{output_filename}} (尺寸: {{width_px}} x {{height_px}})")
    else:
        print("生成图表失败，可能是因为数据格式不正确。")

if __name__ == '__main__':
    main()
"""