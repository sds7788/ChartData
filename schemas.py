# schemas.py
from typing import List, Optional, Union, Tuple
from pydantic import BaseModel, Field
import datetime

# --- Basic Data Series Definitions (No changes here) ---
class YSeries(BaseModel):
    name: str = Field(..., description="Name of the data series, e.g., 'Company A'.")
    values: List[Optional[float]] = Field(..., description="List of numerical values corresponding to x_categories, nulls are allowed.")
    type: Optional[str] = Field(None, description="Type of the series for combo charts, e.g., 'bar' or 'line'.")
    ci_lower: Optional[List[Optional[float]]] = Field(None, description="List of lower bounds for the confidence interval.")
    ci_upper: Optional[List[Optional[float]]] = Field(None, description="List of upper bounds for the confidence interval.")
    is_relevant_for_answer: Optional[bool] = Field(False, description="Indicates if this data series is key to answering the question.")

class ScatterPoint(BaseModel):
    x: float = Field(..., description="The x-coordinate of the data point for a scatter plot.")
    y: float = Field(..., description="The y-coordinate of the data point for a scatter plot.")
    size: Optional[float] = Field(50, description="The size of the scatter point (for bubble charts).")
    color_value: Optional[float] = Field(0, description="A numerical value used for color mapping.")
    x_error: Optional[float] = Field(None, description="The error value in the X-axis direction.")
    y_error: Optional[float] = Field(None, description="The error value in the Y-axis direction.")
    is_relevant_for_answer: Optional[bool] = Field(False, description="Indicates if this scatter point is key to answering the question.")

class PieData(BaseModel):
    labels: List[str] = Field(..., description="Labels for each sector of a pie, donut, or sunburst chart.")
    values: List[float] = Field(..., description="Numerical values for each sector.")
    parents: Optional[List[str]] = Field(None, description="For sunburst and treemaps, a list of parent labels indicating hierarchy. An empty string '' denotes the root.")
    explode: Optional[List[float]] = Field(None, description="An optional list to specify which sectors should be pulled out from the center.")
    is_relevant_for_answer: Optional[List[bool]] = Field(None, description="A boolean list marking whether each sector is key to answering the question.")

class RadarSeries(BaseModel):
    name: str = Field(..., description="Name of the data series for a radar chart.")
    values: List[float] = Field(..., description="Numerical values for each axis of the radar chart.")
    is_relevant_for_answer: Optional[bool] = Field(False, description="Indicates if this data series is key to answering the question.")

class RadarData(BaseModel):
    labels: List[str] = Field(..., description="Labels for each axis of the radar chart.")
    series: List[RadarSeries] = Field(..., description="A list of data series for the radar chart.")

class HistogramData(BaseModel):
    values: List[float] = Field(..., description="A list of raw data to generate the histogram.")
    bins: Optional[int] = Field(None, description="The number of bins (intervals) for the histogram.")
    is_relevant_for_answer: Optional[bool] = Field(False, description="Indicates if the overall distribution or a specific part of it is key to answering the question.")

class ThreeDBarData(BaseModel):
    x_categories: List[str] = Field(..., description="Categories for the X-axis of the 3D bar chart.")
    y_categories: List[str] = Field(..., description="Categories for the Y-axis of the 3D bar chart.")
    z_values: List[List[float]] = Field(..., description="A 2D list (matrix) representing the Z-axis height for each (X, Y) point.")
    relevant_cells: Optional[List[List[int]]] = Field(None, description="A list of coordinates for key cells required to answer the question, format: [[row_index, col_index], ...].")

class StatisticalPlotData(BaseModel):
    categories: List[str] = Field(..., description="Category labels for a box plot, violin plot, or strip plot.")
    data_series: List[List[float]] = Field(..., description="A list of lists, where each inner list contains data points for the corresponding category.")
    is_relevant_for_answer: Optional[List[bool]] = Field(None, description="A boolean list marking whether each category is key to answering the question.")

class WaterfallData(BaseModel):
    labels: List[str] = Field(..., description="Labels for each item in the waterfall chart.")
    values: List[float] = Field(..., description="The value change for each item (positive for increase, negative for decrease).")
    is_relevant_for_answer: Optional[List[bool]] = Field(None, description="Marks whether each item is key to answering the question.")

class ContourData(BaseModel):
    x_grid: List[float] = Field(..., description="Grid point coordinates for the X-axis.")
    y_grid: List[float] = Field(..., description="Grid point coordinates for the Y-axis.")
    z_values: List[List[float]] = Field(..., description="A matrix of Z-values corresponding to the (X, Y) grid.")
    is_relevant_for_answer: Optional[bool] = Field(False, description="Indicates if a specific area or trend in this contour plot is key to answering the question.")

class NetworkNode(BaseModel):
    id: str = Field(..., description="Unique identifier for the node.")
    label: Optional[str] = Field(None, description="The display label for the node; if None, the id is used.")
    size: Optional[float] = Field(100, description="The size of the node.")
    group: Optional[Union[str, int]] = Field(None, description="The group to which the node belongs, can be used for coloring.")

class NetworkEdge(BaseModel):
    source: str = Field(..., description="The id of the starting node of the edge.")
    target: str = Field(..., description="The id of the target node of the edge.")
    weight: Optional[float] = Field(1.0, description="The weight of the edge.")

class NetworkData(BaseModel):
    nodes: List[NetworkNode] = Field(..., description="A list of nodes for the network diagram.")
    edges: List[NetworkEdge] = Field(..., description="A list of edges for the network diagram.")

class ForestPlotData(BaseModel):
    labels: List[str] = Field(..., description="Labels for each study or item.")
    effects: List[float] = Field(..., description="The effect size (point estimate).")
    lower_ci: List[float] = Field(..., description="The lower bound of the confidence interval.")
    upper_ci: List[float] = Field(..., description="The upper bound of the confidence interval.")
    is_relevant_for_answer: Optional[List[bool]] = Field(None, description="Marks whether each study result is key to answering the question.")

class FunnelData(BaseModel):
    stages: List[str] = Field(..., description="The names of the stages in the funnel chart.")
    values: List[float] = Field(..., description="The numerical value corresponding to each stage.")
    is_relevant_for_answer: Optional[List[bool]] = Field(None, description="Marks whether each stage is key to answering the question.")

class HeatmapData(BaseModel):
    x_labels: List[str] = Field(..., description="Labels for the X-axis of the heatmap.")
    y_labels: List[str] = Field(..., description="Labels for the Y-axis of the heatmap.")
    values: List[List[float]] = Field(..., description="The 2D numerical matrix for the heatmap.")
    relevant_cells: Optional[List[List[int]]] = Field(None, description="A list of coordinates for key cells required to answer the question.")

class SankeyLink(BaseModel):
    source: str = Field(..., description="The label of the source node for the flow.")
    target: str = Field(..., description="The label of the target node for the flow.")
    value: float = Field(..., description="The numerical size of the flow.")

class SankeyData(BaseModel):
    nodes: List[str] = Field(..., description="A list of unique labels for all nodes in the Sankey diagram.")
    links: List[SankeyLink] = Field(..., description="A list of links defining the flow relationships between nodes.")
    is_relevant_for_answer: Optional[bool] = Field(False, description="Indicates if this flow relationship is key to answering the question.")

class CandlestickRecord(BaseModel):
    time: Union[str, datetime.date] = Field(..., description="The time point or date.")
    open: float = Field(..., description="The opening price.")
    high: float = Field(..., description="The highest price.")
    low: float = Field(..., description="The lowest price.")
    close: float = Field(..., description="The closing price.")

class CandlestickData(BaseModel):
    records: List[CandlestickRecord] = Field(..., description="A time-series list of records for the candlestick chart.")
    is_relevant_for_answer: Optional[bool] = Field(False, description="Indicates if this price trend is key to answering the question.")

class GaugeRange(BaseModel):
    """Defines a color range in the gauge chart."""
    start: float = Field(..., description="The starting value of the range.")
    end: float = Field(..., description="The ending value of the range.")
    color: str = Field(..., description="The color of the range.")

class GaugeData(BaseModel):
    value: float = Field(..., description="The current value the gauge pointer indicates.")
    min_value: float = Field(0, description="The minimum value of the gauge.")
    max_value: float = Field(100, description="The maximum value of the gauge.")
    title: Optional[str] = Field(None, description="The title displayed below the gauge.")
    ranges: Optional[List[GaugeRange]] = Field(None, description="A list of objects defining different ranges.")

class WordCloudEntry(BaseModel):
    word: str = Field(..., description="The word.")
    weight: float = Field(..., description="The weight or frequency of the word.")

class WordCloudData(BaseModel):
    entries: List[WordCloudEntry] = Field(..., description="A list of data entries for the word cloud.")

class DateValueEntry(BaseModel):
    """An entry representing a single date and its corresponding value."""
    date: datetime.date = Field(..., description="The date.")
    value: float = Field(..., description="The value for that date.")

class CalendarHeatmapData(BaseModel):
    """Data structure for a calendar heatmap."""
    entries: List[DateValueEntry] = Field(..., description="A list of date-value pairs, replacing the previous dictionary structure.")

class ParallelCoordinatesData(BaseModel):
    dimensions: List[str] = Field(..., description="A list of dimensions (axes) for the parallel coordinates plot.")
    data_records: List[List[float]] = Field(..., description="A list of data records, where each sublist's values correspond to the dimensions.")
    group_by: Optional[List[str]] = Field(None, description="A list of category labels for coloring the lines, same length as data_records.")

# --- Common Building Blocks ---
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
    question: str = Field(..., description="An insightful question based on the generated data.")
    answer: str = Field(..., description="A concise and direct answer to the question.")
    relevance_reasoning: str = Field(..., description="An explanation of why the data marked as 'relevant' is necessary and sufficient to answer the question.")

# --- 【NEW】Schema for Single Plot Charts ---
class SingleChartData(BaseModel):
    chart_type: str = Field(..., description="The specific type of the chart.")
    title: str = Field(..., description="The title of the chart.")
    x_label: Optional[str] = Field(None, description="Label for the X-axis.")
    y_label: Optional[str] = Field(None, description="Label for the Y-axis.")
    data: ChartSpecificData = Field(..., description="A container for all the data for the chart.")
    analysis: Analysis = Field(..., description="A question-and-answer analysis based on the chart data.")

# --- 【NEW】Schema for Faceted (Multi-Plot) Charts ---
class FacetData(BaseModel):
    """Defines the data for a single subplot (facet)."""
    facet_value: str = Field(..., description="The value of the facet this subplot represents, e.g., 'North America' or 'Male'.")
    data: ChartSpecificData = Field(..., description="The specific data for the chart in this facet.")

class FacetChartData(BaseModel):
    chart_type: str = Field(..., description="The specific type of the chart.")
    title: str = Field(..., description="The overall title for the faceted chart display.")
    x_label: Optional[str] = Field(None, description="Common label for the X-axis across all subplots.")
    y_label: Optional[str] = Field(None, description="Common label for the Y-axis across all subplots.")
    facet_by: str = Field(..., description="The field name used for faceting, e.g., 'Region'.")
    facet_data: List[FacetData] = Field(..., description="A list of data for generating faceted subplots.")
    analysis: Analysis = Field(..., description="A question-and-answer analysis based on the overall chart data.")
