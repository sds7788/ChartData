# chart_generators/base_generator.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseChartGenerator(ABC):
    """
    所有图表生成器的抽象基类。
    """
    def __init__(self, chart_package: Dict[str, Any]):
        """
        初始化生成器。
        
        参数:
            chart_package (Dict[str, Any]): 从 Gemini API 获取并解析后的图表数据包。
        """
        self.chart_package = chart_package
        self.chart_type = chart_package.get('chart_type')

    @abstractmethod
    def create_chart(self, output_path: str, figsize: tuple = (12, 8), dpi: int = 100) -> bool:
        """
        根据数据包创建图表并保存到文件。

        参数:
            output_path (str): 图表文件的保存路径。
            figsize (tuple): 图像尺寸 (英寸), 对 pyecharts 会转换为像素。
            dpi (int): 图像分辨率 (每英寸点数), 对 pyecharts 会转换为像素。

        返回:
            bool: 如果成功生成则返回 True，否则返回 False。
        """
        pass

    @abstractmethod
    def generate_code(self, figsize: tuple = (12, 8), dpi: int = 100) -> str:
        """
        生成用于复现该图表的可执行 Python 代码字符串。
        
        参数:
            figsize (tuple): 图像尺寸。
            dpi (int): 图像分辨率。

        返回:
            str: 可复现的 Python 代码。
        """
        pass
        
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """
        返回此生成器创建的文件的扩展名 (例如 'png', 'html')。
        """
        pass