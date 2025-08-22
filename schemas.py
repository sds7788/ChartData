# schemas.py
from typing import List, Optional, Union, Tuple
from pydantic import BaseModel, Field
import datetime

# --- 基础数据系列定义 ---
class YSeries(BaseModel):
    name: str = Field(..., description="数据系列的名称，例如 '公司A'。")
    values: List[Optional[float]] = Field(..., description="与x_categories对应的数值列表，允许空值。")
    type: Optional[str] = Field(None, description="系列类型，用于组合图，例如 'bar' 或 'line'。")
    ci_lower: Optional[List[Optional[float]]] = Field(None, description="置信区间的下界列表。")
    ci_upper: Optional[List[Optional[float]]] = Field(None, description="置信区间的上界列表。")
    is_relevant_for_answer: Optional[bool] = Field(False, description="此数据系列是否是回答问题的关键。")

class ScatterPoint(BaseModel):
    x: float = Field(..., description="散点图数据点的x坐标。")
    y: float = Field(..., description="散点图数据点的y坐标。")
    size: Optional[float] = Field(50, description="散点的大小（用于气泡图）。")
    color_value: Optional[float] = Field(0, description="用于颜色映射的数值。")
    x_error: Optional[float] = Field(None, description="X轴方向的误差值。")
    y_error: Optional[float] = Field(None, description="Y轴方向的误差值。")
    is_relevant_for_answer: Optional[bool] = Field(False, description="此散点是否是回答问题的关键。")

class PieData(BaseModel):
    labels: List[str] = Field(..., description="饼图、圆环图或旭日图每个扇区的标签。")
    values: List[float] = Field(..., description="每个扇区的数值。")
    parents: Optional[List[str]] = Field(None, description="用于旭日图和树状图，表示层级关系的父节点标签列表。空字符串''表示根节点。")
    explode: Optional[List[float]] = Field(None, description="一个可选的列表，用于指定哪些扇区需要从中心突出显示。")
    is_relevant_for_answer: Optional[List[bool]] = Field(None, description="一个布尔值列表，标记每个扇区是否是回答问题的关键。")

class RadarSeries(BaseModel):
    name: str = Field(..., description="雷达图数据系列的名称。")
    values: List[float] = Field(..., description="雷达图每个轴上的数值。")
    is_relevant_for_answer: Optional[bool] = Field(False, description="此数据系列是否是回答问题的关键。")

class RadarData(BaseModel):
    labels: List[str] = Field(..., description="雷达图各个轴的标签。")
    series: List[RadarSeries] = Field(..., description="雷达图的数据系列列表。")

class HistogramData(BaseModel):
    values: List[float] = Field(..., description="用于生成直方图的原始数据列表。")
    bins: Optional[int] = Field(None, description="直方图的箱数（区间数）。")
    is_relevant_for_answer: Optional[bool] = Field(False, description="此分布的整体或特定部分是否是回答问题的关键。")

class ThreeDBarData(BaseModel):
    x_categories: List[str] = Field(..., description="3D柱状图的X轴类别。")
    y_categories: List[str] = Field(..., description="3D柱状图的Y轴类别。")
    z_values: List[List[float]] = Field(..., description="一个二维列表（矩阵），表示每个(X, Y)点对应的Z轴高度。")
    relevant_cells: Optional[List[List[int]]] = Field(None, description="回答问题所需关键单元格的坐标列表，格式为 [[row_index, col_index], ...]。")

class StatisticalPlotData(BaseModel):
    categories: List[str] = Field(..., description="箱形图、小提琴图或带状图的分类标签。")
    data_series: List[List[float]] = Field(..., description="每个类别对应的数据点列表的列表。")
    is_relevant_for_answer: Optional[List[bool]] = Field(None, description="一个布尔值列表，标记每个类别是否是回答问题的关键。")

class WaterfallData(BaseModel):
    labels: List[str] = Field(..., description="瀑布图中每个条目的标签。")
    values: List[float] = Field(..., description="每个条目的数值变化（正数表示增加，负数表示减少）。")
    is_relevant_for_answer: Optional[List[bool]] = Field(None, description="标记每个条目是否是回答问题的关键。")

class ContourData(BaseModel):
    x_grid: List[float] = Field(..., description="X轴的网格点坐标。")
    y_grid: List[float] = Field(..., description="Y轴的网格点坐标。")
    z_values: List[List[float]] = Field(..., description="对应于(X, Y)网格的Z值矩阵。")
    is_relevant_for_answer: Optional[bool] = Field(False, description="此等高线图的特定区域或趋势是否是回答问题的关键。")

class NetworkNode(BaseModel):
    id: str = Field(..., description="节点的唯一标识符。")
    label: Optional[str] = Field(None, description="节点显示的标签，如果为None则使用id。")
    size: Optional[float] = Field(100, description="节点的大小。")
    group: Optional[Union[str, int]] = Field(None, description="节点所属的分组，可用于着色。")

class NetworkEdge(BaseModel):
    source: str = Field(..., description="边的起始节点的id。")
    target: str = Field(..., description="边的目标节点的id。")
    weight: Optional[float] = Field(1.0, description="边的权重。")

class NetworkData(BaseModel):
    nodes: List[NetworkNode] = Field(..., description="网络图的节点列表。")
    edges: List[NetworkEdge] = Field(..., description="网络图的边列表。")

class ForestPlotData(BaseModel):
    labels: List[str] = Field(..., description="每个研究或条目的标签。")
    effects: List[float] = Field(..., description="效应量（点估计值）。")
    lower_ci: List[float] = Field(..., description="置信区间的下界。")
    upper_ci: List[float] = Field(..., description="置信区间的上界。")
    is_relevant_for_answer: Optional[List[bool]] = Field(None, description="标记每个研究结果是否是回答问题的关键。")

class FunnelData(BaseModel):
    stages: List[str] = Field(..., description="漏斗图的各个阶段名称。")
    values: List[float] = Field(..., description="每个阶段对应的数值。")
    is_relevant_for_answer: Optional[List[bool]] = Field(None, description="标记每个阶段是否是回答问题的关键。")

class HeatmapData(BaseModel):
    x_labels: List[str] = Field(..., description="热力图X轴的标签。")
    y_labels: List[str] = Field(..., description="热力图Y轴的标签。")
    values: List[List[float]] = Field(..., description="热力图的二维数值矩阵。")
    relevant_cells: Optional[List[List[int]]] = Field(None, description="回答问题所需关键单元格的坐标列表。")

class SankeyLink(BaseModel):
    source: str = Field(..., description="流量来源节点的标签。")
    target: str = Field(..., description="流量目标节点的标签。")
    value: float = Field(..., description="流量的数值大小。")

class SankeyData(BaseModel):
    nodes: List[str] = Field(..., description="桑基图中所有节点的唯一标签列表。")
    links: List[SankeyLink] = Field(..., description="定义节点之间流量关系的链接列表。")
    is_relevant_for_answer: Optional[bool] = Field(False, description="此流动关系是否是回答问题的关键。")

class CandlestickRecord(BaseModel):
    time: Union[str, datetime.date] = Field(..., description="时间点或日期。")
    open: float = Field(..., description="开盘价。")
    high: float = Field(..., description="最高价。")
    low: float = Field(..., description="最低价。")
    close: float = Field(..., description="收盘价。")

class CandlestickData(BaseModel):
    records: List[CandlestickRecord] = Field(..., description="K线图的时间序列记录。")
    is_relevant_for_answer: Optional[bool] = Field(False, description="此价格趋势是否是回答问题的关键。")

# --- 【重要】修改点 1 ---
class GaugeRange(BaseModel):
    """定义仪表盘中的一个颜色范围。"""
    start: float = Field(..., description="范围的起始值。")
    end: float = Field(..., description="范围的结束值。")
    color: str = Field(..., description="范围的颜色。")

class GaugeData(BaseModel):
    value: float = Field(..., description="仪表盘指针指向的当前数值。")
    min_value: float = Field(0, description="仪表盘的最小值。")
    max_value: float = Field(100, description="仪表盘的最大值。")
    title: Optional[str] = Field(None, description="显示在仪表盘下方的标题。")
    # 使用对象列表替代元组列表
    ranges: Optional[List[GaugeRange]] = Field(None, description="定义不同区间的对象列表。")
# --- 修改结束 ---

class WordCloudEntry(BaseModel):
    word: str = Field(..., description="词语。")
    weight: float = Field(..., description="词语的权重或频率。")

class WordCloudData(BaseModel):
    entries: List[WordCloudEntry] = Field(..., description="词云图的数据条目列表。")

class DateValueEntry(BaseModel):
    """用于表示单个日期及其对应数值的条目。"""
    date: datetime.date = Field(..., description="日期")
    value: float = Field(..., description="该日期的数值")

class CalendarHeatmapData(BaseModel):
    """日历热力图的数据结构。"""
    entries: List[DateValueEntry] = Field(..., description="一个日期-数值对的列表，替代了原有的字典结构。")

class ParallelCoordinatesData(BaseModel):
    dimensions: List[str] = Field(..., description="平行坐标图的维度（坐标轴）列表。")
    data_records: List[List[float]] = Field(..., description="数据记录列表，每个子列表中的值与维度一一对应。")
    group_by: Optional[List[str]] = Field(None, description="用于给线条着色的类别标签列表，长度与data_records相同。")

class ChartSpecificData(BaseModel):
    x_categories: Optional[List[str]] = Field(None)
    y_series: Optional[List[YSeries]] = Field(None)
    scatter_points: Optional[List[ScatterPoint]] = Field(None)
    pie_data: Optional[PieData] = Field(None)
    radar_data: Optional[RadarData] = Field(None)
    heatmap_data: Optional[HeatmapData] = Field(None)
    statistical_data: Optional[StatisticalPlotData] = Field(None)
    histogram_data: Optional[HistogramData] = Field(None)
    three_d_bar_data: Optional[ThreeDBarData] = Field(None)
    contour_data: Optional[ContourData] = Field(None)
    forest_plot_data: Optional[ForestPlotData] = Field(None)
    waterfall_data: Optional[WaterfallData] = Field(None)
    funnel_data: Optional[FunnelData] = Field(None)
    network_data: Optional[NetworkData] = Field(None)
    sankey_data: Optional[SankeyData] = Field(None)
    candlestick_data: Optional[CandlestickData] = Field(None)
    gauge_data: Optional[GaugeData] = Field(None)
    word_cloud_data: Optional[WordCloudData] = Field(None)
    calendar_heatmap_data: Optional[CalendarHeatmapData] = Field(None)
    parallel_coordinates_data: Optional[ParallelCoordinatesData] = Field(None)

class Analysis(BaseModel):
    question: str = Field(..., description="一个基于生成数据的、有深度的问题。")
    answer: str = Field(..., description="对问题的简洁、确切的答案。")
    relevance_reasoning: str = Field(..., description="解释为什么被标记为'relevant'的数据是回答该问题的充分必要条件。")

class ChartData(BaseModel):
    chart_type: str = Field(..., description="图表的具体类型。")
    title: str = Field(..., description="图表的标题。")
    x_label: Optional[str] = Field(None)
    y_label: Optional[str] = Field(None)
    data: ChartSpecificData = Field(..., description="包含图表所有数据的容器。")
    analysis: Analysis = Field(..., description="基于图表数据的问答分析。")
