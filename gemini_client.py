import os
import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from google import genai

# ==============================================================================
# 1. Gemini 客户端 (已按照您的最新指示重构)
# ==============================================================================

class GeminiClient:
    """
    一个封装了与Google Gemini API交互的客户端类。
    此版本已更新，以使用最新的`client.generate_content`和`generation_config`方法。
    """
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        初始化客户端，配置API密钥和模型。

        Args:
            api_key (str): Google AI Studio提供的API密钥。
            model_name (str): 要使用的Gemini模型名称，例如 "gemini-1.5-pro-latest"。
        """
        # 严格按照您的要求，使用 genai.Client 进行初始化
        self.client = genai.Client(api_key=api_key)
        self.api_key = api_key  # 【修复】将api_key保存为实例的属性
        self.model_name: str = model_name

    def _build_prompt(self, topic: str, knowledge_domain: str, chart_type: str) -> str:
        """
        根据输入动态构建一个复杂的提示。(此方法无需更改)

        Args:
            topic (str): 数据集的核心主题。
            knowledge_domain (str): 主题所属的知识领域。
            chart_type (str): 期望生成的图表类型。

        Returns:
            str: 构建完成的完整提示字符串。
        """
        # prompt = f"""
        # You are a {knowledge_domain} expert, specializing in data analysis and visualization.
        # Your task is to generate a complex dataset with subtle internal relationships for a chart of type "{chart_type}", centered around the topic of "{topic}".
        # The correlations within the data should be multi-dimensional and not immediately obvious, requiring exploration of the chart to be discovered.
        # Based on the data you generate, pose a challenging analytical question that requires an in-depth interpretation of the chart to answer.
        # Provide a concise and definitive answer to your question (e.g., a specific name, an exact year, or a precise numerical value).
        # You must format your entire response as a single JSON object that strictly adheres to the JSON Schema I provide.
        # Do not include any explanatory text, code block markers, or any other characters before or after the JSON object.
        # """
        
        # prompt = f"""
        # You are a world-class {knowledge_domain} expert, specializing in advanced data analysis and visualization.
        
        # Your task is to generate a complex and extensive dataset suitable for a {chart_type} chart, centered around the topic of "{topic}".
        
        # The dataset must be highly intricate and designed to test analytical skills. Key requirements are:
        # 1.  **Entity Count**: The dataset must include at least 12-15 distinct entities (e.g., companies, countries, products, projects) to ensure high complexity and a rich visualization.
        # 2.  **Multi-Dimensional Data**: For each entity, you must generate data across multiple variables (at least 5-7 distinct metrics).
        # 3.  **Subtle Correlations**: The internal relationships and correlations within the data must be subtle, non-linear, and multi-dimensional. These patterns should not be immediately obvious and should require deep exploration of the chart to be discovered.

        # Based on the dataset you generate, you must then perform the following two steps:
        # 1.  **Formulate a Challenging Question**: Pose a highly challenging analytical question that demands a nuanced interpretation of the complex relationships within the chart. The question should go beyond simple data retrieval and require synthesizing information.
        # 2.  **Provide a Definitive Answer**: Provide a concise and definitive answer to your question. The answer must be a specific value (e.g., a specific entity's name, an exact year, or a precise numerical value) that is directly and unambiguously derivable from the dataset.

        # Your entire response must be a single, raw JSON object that strictly adheres to the JSON Schema provided. Do not include any introductory or explanatory text, code block markers (like ```json), or any other characters before or after the JSON object itself.
        # """
        
        prompt = f"""
        You are a world-class expert in the {knowledge_domain} field, specializing in advanced data analysis and visualization.

        Your task is to follow these steps strictly:
        1.Generate a Complex Dataset: First, create a complex and extensive dataset for a "{chart_type}" chart centered around the topic of "{topic}". The dataset must be intricate but don't impact visualizationand meet the following requirements:
            a.Entity Count: Include at least 7-9 distinct entities (e.g., companies, countries, products) to ensure high complexity and a rich visualization.
            b.Data Dimensions: For each entity, generate multi-dimensional data across at least 4-6 distinct metrics.
            c.Subtle Correlations: The internal relationships and correlations within the data must be subtle, non-linear, and multi-dimensional. They should not be immediately obvious and should require deep exploration of the chart to be discovered.
        2.Formulate a Challenging Question: Based on the dataset you generated, pose a highly challenging analytical question that demands a nuanced interpretation of the complex relationships within the chart, going beyond simple data retrieval.
        3.Provide clear answers: Please provide clear and concise answers to your questions. The answer must be concise, such as a noun, a number, a year, etc.
        4.Identify and Annotate Core Evidence: This is a critical step. Review your question and answer, and precisely identify the "minimum necessary data subset" required to answer the question. Then, in the final JSON output, you must use the is_relevant_for_answer field (or relevant_cells for heatmaps) to mark only this minimum subset. All other irrelevant data points must have this field set to false.
        5.Explain the Reasoning for Annotation: In the analysis.relevance_reasoning field, clearly explain why the marked data is the sufficient and necessary condition to answer the question.

        Please note that the most important thing is to keep your answer concise!
        
        Your entire response must be a single, raw JSON object that strictly adheres to the JSON Schema I provide. Do not include any introductory or explanatory text, code block markers, or any other characters before or after the JSON object itself.
        """
        
        # prompt = f"""
        # You are a {knowledge_domain} expert specializing in data visualization.

        # Your task is to generate a clear and representative dataset for a {chart_type} chart, centered around the topic of “{topic}”.

        # The dataset should be simple and intuitive for easy understanding. Key requirements are:
        # 1.Entity Count: Include 4-6 distinct entities (e.g., companies, countries, products).
        # 2.Clear Data: The data generated for these entities should be easy to visualize and compare directly.

        # Based on the dataset you generate, please complete the following two steps:
        # 1.Formulate a Simple Question: Design a straightforward question that can be answered by directly reading the chart data.
        # 2.Provide a Definitive Answer: Provide a concise and direct answer to your question. The answer must be a specific name, category, or numerical value from the dataset.

        # Your entire response must be a single, raw JSON object that strictly adheres to the provided JSON Schema. Do not include any introductory or explanatory text, code block markers (like ```json), or any other characters before or after the JSON object itself.
        # """
        
        return prompt

    def generate_chart_data(self, topic: str, knowledge_domain: str, chart_type: str, schema: BaseModel) -> Dict[str, Any]:
        """
        调用Gemini API生成结构化的图表数据。
        (已重构为使用 client.generate_content 和 generation_config)

        Args:
            topic (str): 数据集的核心主题。
            knowledge_domain (str): 主题所属的知识领域。
            chart_type (str): 期望生成的图表类型。
            schema (BaseModel): 用于约束输出格式的Pydantic模型。

        Returns:
            Dict[str, Any]: 从API返回并解析后的JSON数据（以Python字典形式）。
        
        Raises:
            Exception: 如果API调用失败或返回的数据无法解析。
        """
        prompt = self._build_prompt(topic, knowledge_domain, chart_type)
        
        # 按照您的示例，构建 generation_config 字典
        generation_config = {
            "response_mime_type": "application/json",
            "response_schema": schema, # 直接传递Pydantic模型
        }
        
        try:
            print(f"向Gemini发送请求：主题='{topic}', 图表类型='{chart_type}'...")
            
            # 使用最新的 client.generate_content 方法
            # 注意：模型名称可能需要 'models/' 前缀，这取决于SDK版本，但最新版通常可省略
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config = {
                "response_mime_type": "application/json",
                "response_schema": schema, # 直接传递Pydantic模型
                }
            )
            print(response.text)
            print("成功从Gemini接收到响应。")
            
            # 直接解析返回的JSON文本为Pydantic对象，然后转换为字典
            # response.text 包含了符合 schema 的 JSON 字符串
            parsed_data = schema.model_validate_json(response.text)
            return parsed_data.model_dump()

        except Exception as e:
            print(f"调用Gemini API时发生错误: {e}")
            # 在实际应用中，这里可以加入更复杂的重试逻辑
            raise
