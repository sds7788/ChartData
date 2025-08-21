from typing import List, Optional, Union, Tuple
from pydantic import BaseModel, Field

# --- 基础数据系列定义 (已更新) ---
class YSeries(BaseModel):
    name: str = Field(..., description="数据系列的名称，例如 '公司A'。")
    values: List[float] = Field(..., description="与x_categories对应的数值列表。")
    type: Optional[str] = Field(None, description="系列类型，用于组合图，例如 'bar' 或 'line'。")
    is_relevant_for_answer: Optional[bool] = Field(False, description="此数据系列是否是回答问题的关键。")

class ScatterPoint(BaseModel):
    x: float = Field(..., description="散点图数据点的x坐标。")
    y: float = Field(..., description="散点图数据点的y坐标。")
    size: Optional[float] = Field(50, description="散点的大小。")
    color_value: Optional[float] = Field(0, description="用于颜色映射的数值。")
    is_relevant_for_answer: Optional[bool] = Field(False, description="此散点是否是回答问题的关键。")

class PieData(BaseModel):
    labels: List[str] = Field(..., description="饼图或圆环图每个扇区的标签。")
    values: List[float] = Field(..., description="饼图或圆环图每个扇区的数值。")
    explode: Optional[List[float]] = Field(None, description="一个可选的列表，用于指定哪些扇区需要从中心突出显示。")
    is_relevant_for_answer: Optional[List[bool]] = Field(None, description="一个布尔值列表，标记每个扇区是否是回答问题的关键。")

class RadarSeries(BaseModel):
    name: str = Field(..., description="雷达图数据系列的名称。")
    values: List[float] = Field(..., description="雷达图每个轴上的数值。")
    is_relevant_for_answer: Optional[bool] = Field(False, description="此数据系列是否是回答问题的关键。")

class RadarData(BaseModel):
    labels: List[str] = Field(..., description="雷达图各个轴的标签。")
    series: List[RadarSeries] = Field(..., description="雷达图的数据系列列表。")

# --- 新增图表类型的数据结构 (已更新) ---
class GanttTask(BaseModel):
    name: str = Field(..., description="任务的名称。")
    start: int = Field(..., description="任务的起始时间点。")
    duration: int = Field(..., description="任务的持续时长。")
    color: Optional[Union[str, List[float]]] = Field(None, description="表示任务的条形的颜色。")
    is_relevant_for_answer: Optional[bool] = Field(False, description="此任务是否是回答问题的关键。")

class GanttData(BaseModel):
    tasks: List[GanttTask] = Field(..., description="甘特图的任务列表。")

class BoxPlotData(BaseModel):
    categories: List[str] = Field(..., description="箱型图的分类标签。")
    data_series: List[List[float]] = Field(..., description="每个类别对应的数据点列表的列表。")
    is_relevant_for_answer: Optional[List[bool]] = Field(None, description="一个布尔值列表，标记每个类别是否是回答问题的关键。")

class HeatmapData(BaseModel):
    x_labels: List[str] = Field(..., description="热力图X轴的标签。")
    y_labels: List[str] = Field(..., description="热力图Y轴的标签。")
    values: List[List[float]] = Field(..., description="热力图的二维数值矩阵。")
    relevant_cells: Optional[List[List[int]]] = Field(None, description="回答问题所需关键单元格的坐标列表，格式为 [[row_index, col_index], ...]。")


# --- 统一数据结构 (已扩展) ---
class ChartSpecificData(BaseModel):
    x_categories: Optional[List[str]] = Field(None)
    y_series: Optional[List[YSeries]] = Field(None)
    scatter_points: Optional[List[ScatterPoint]] = Field(None)
    pie_data: Optional[PieData] = Field(None)
    radar_data: Optional[RadarData] = Field(None)
    gantt_data: Optional[GanttData] = Field(None)
    boxplot_data: Optional[BoxPlotData] = Field(None)
    heatmap_data: Optional[HeatmapData] = Field(None)

# --- 问答分析 (已更新) ---
class Analysis(BaseModel):
    question: str = Field(..., description="一个基于生成数据的、有深度的问题。")
    answer: str = Field(..., description="对问题的简洁、确切的答案。")
    relevance_reasoning: str = Field(..., description="解释为什么被标记为'relevant'的数据是回答该问题的充分必要条件。")

# --- 顶层Schema (已更新) ---
class ChartData(BaseModel):
    chart_type: str = Field(...)
    title: str = Field(...)
    x_label: Optional[str] = Field(None)
    y_label: Optional[str] = Field(None)
    data: ChartSpecificData = Field(...)
    analysis: Analysis = Field(...)
