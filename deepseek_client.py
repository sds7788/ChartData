# deepseek_client.py
import json
from typing import Dict, Any, Type
from pydantic import BaseModel
from openai import OpenAI

# ==============================================================================
# 1. DeepSeek 客户端
# ==============================================================================

class DeepSeekClient:
    """
    一个封装了与 DeepSeek LLM API 交互的客户端类。
    """
    def __init__(self, api_key: str, model_name: str = "deepseek-chat"):
        """
        初始化 DeepSeek 客户端。

        Args:
            api_key (str): 您的 DeepSeek API 密钥。
            model_name (str): 要使用的 DeepSeek 模型名称，例如 "deepseek-chat"。
        """
        # 使用 OpenAI 的库来实例化客户端，但指向 DeepSeek 的 API 端点
        self.client = OpenAI(
            api_key=api_key, # 使用传入的 API Key
            base_url="https://api.deepseek.com/v1"
        )
        self.model_name: str = model_name

    def _build_prompt(self, topic: str, knowledge_domain: str, chart_type: str, schema: Type[BaseModel]) -> str:
        """
        根据输入动态构建一个复杂的提示，并附加 JSON Schema 以增强输出的可靠性。

        Args:
            topic (str): 数据集的核心主题。
            knowledge_domain (str): 主题所属的知识领域。
            chart_type (str): 期望生成的图表类型。
            schema (Type[BaseModel]): 用于约束输出格式的 Pydantic 模型类。

        Returns:
            str: 构建完成的、包含指令和 Schema 的完整提示字符串。
        """
        # 基础指令部分保持不变
        base_prompt = f"""
        You are a world-class expert in the {knowledge_domain} field, specializing in advanced data analysis and visualization.

        Your task is to follow these steps strictly:
        1.Generate a Complex Dataset: First, create a complex and extensive dataset for a "{chart_type}" chart centered around the topic of "{topic}". The dataset must be intricate but don't impact visualizationand meet the following requirements:
            a.Entity Count: Include at least 7-9 distinct entities (e.g., companies, countries, products) to ensure high complexity and a rich visualization.
            b.Data Dimensions: For each entity, generate multi-dimensional data across at least 4-6 distinct metrics.
            c.Subtle Correlations: The internal relationships and correlations within the data must be subtle, non-linear, and multi-dimensional. They should not be immediately obvious and should require deep exploration of the chart to be discovered.
        2.Ask a Challenging Question: Based on the dataset you generated, formulate a challenging analytical question that requires careful interpretation of complex relationships within a chart, rather than simple data retrieval. The following options are available: 
            a. The question should require 2-5 steps of reasoning to solve. 
            b. Ask about mathematical operations (differences, percentages, ratios) between data elements. 
            c. Focus on identifying patterns, extreme values, or anomalies in the data visualization. 
            d. Include questions about the relationships between different data points or series.
        3.Provide clear answers: Please provide clear and concise answers to your questions. The answer must be concise, such as a noun or a number or a year, etc.
        4.Identify and Annotate Core Evidence: This is a critical step. Review your question and answer, and precisely identify the "minimum necessary data subset" required to answer the question. Then, in the final JSON output, you must use the is_relevant_for_answer field (or relevant_cells for heatmaps) to mark only this minimum subset. All other irrelevant data points must have this field set to false.
        5.Explain the Reasoning for Annotation: In the analysis.relevance_reasoning field, clearly explain why the marked data is the sufficient and necessary condition to answer the question.

        Please note that the most important thing is to keep your answer concise!No additional explanation is needed!
        Be careful not to crowd the data points too closely together to avoid overlapping elements in the rendered chart.
        
        Your entire response must be a single, raw JSON object that strictly adheres to the JSON Schema I provide. Do not include any introductory or explanatory text, code block markers, or any other characters before or after the JSON object itself.
        """
        
        # 将 Pydantic Schema 转换为 JSON Schema 字符串
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        
        # 将 Schema 附加到提示的末尾，以指导模型输出
        full_prompt = (
            f"{base_prompt}\n\n"
            "Here is the JSON schema you MUST strictly adhere to:\n"
            f"```json\n{schema_json}\n```"
        )
        return full_prompt

    def generate_chart_data(self, topic: str, knowledge_domain: str, chart_type: str, schema: Type[BaseModel]) -> Dict[str, Any]:
        """
        调用 DeepSeek API 生成结构化的图表数据。

        Args:
            topic (str): 数据集的核心主题。
            knowledge_domain (str): 主题所属的知识领域。
            chart_type (str): 期望生成的图表类型。
            schema (Type[BaseModel]): 用于约束和验证输出格式的 Pydantic 模型类。

        Returns:
            Dict[str, Any]: 从 API 返回并解析、验证后的 JSON 数据（以 Python 字典形式）。
        
        Raises:
            Exception: 如果 API 调用失败或返回的数据无法通过 Pydantic 验证。
        """
        prompt = self._build_prompt(topic, knowledge_domain, chart_type, schema)
        
        try:
            print(f"向 DeepSeek API 发送请求：主题='{topic}', 图表类型='{chart_type}'...")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            print(f"成功从 DeepSeek API 接收到响应 (主题='{topic}')。")
            
            parsed_data = schema.model_validate_json(response_text)
            
            return parsed_data.model_dump()

        except Exception as e:
            print(f"调用 DeepSeek API 或解析数据时发生错误 (主题='{topic}'): {e}")
            raise