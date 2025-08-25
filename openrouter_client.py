# import os
# import json
# from typing import Dict, Any, Type
# from pydantic import BaseModel
# from openai import OpenAI

# # ==============================================================================
# # 1. OpenRouter 客户端 (使用 openai SDK)
# # ==============================================================================

# class OpenRouterClient:
#     """
#     一个封装了通过 OpenRouter 与兼容OpenAI的API（如Gemini）交互的客户端。
#     使用 `openai` 库进行API调用，并支持结构化JSON输出。
#     """
#     def __init__(self, api_key: str, model_name: str = "google/gemini-2.5-flash-lite"):
#         """
#         初始化客户端。

#         Args:
#             api_key (str): 您的 OpenRouter API 密钥。
#             model_name (str): 要使用的模型名称，例如 "google/gemini-2.5-flash-lite"。
#         """
#         if not api_key:
#             raise ValueError("OpenRouter API key is required.")
        
#         # 按照您的要求，使用 OpenAI SDK 初始化客户端
#         self.client = OpenAI(
#             base_url="https://openrouter.ai/api/v1",
#             api_key=api_key,
#         )
#         self.api_key = api_key  # 保存api_key，用于main.py中的熔断机制
#         self.model_name = model_name
        
#         # 可以在这里设置您在OpenRouter上的可选标识
#         self.extra_headers = {
#             # "HTTP-Referer": "<YOUR_SITE_URL>", 
#             # "X-Title": "<YOUR_SITE_NAME>",
#         }

#     def _build_prompt(self, topic: str, knowledge_domain: str, chart_type: str, schema: Type[BaseModel]) -> str:
#         """
#         根据输入动态构建一个复杂的提示。
#         为了确保获得可靠的JSON输出，我们将JSON Schema直接包含在提示中。

#         Args:
#             topic (str): 数据集的核心主题。
#             knowledge_domain (str): 主题所属的知识领域。
#             chart_type (str): 期望生成的图表类型。
#             schema (Type[BaseModel]): 用于约束输出格式的Pydantic模型类。

#         Returns:
#             str: 构建完成的完整提示字符串。
#         """
#         # 将Pydantic模型转换为JSON Schema字典
#         json_schema = schema.model_json_schema()
        
#         # 将JSON Schema字典格式化为漂亮的字符串，以便放入提示
#         schema_str = json.dumps(json_schema, indent=2)

#         # 原始提示的核心逻辑保持不变
#         base_prompt = f"""
#         You are a world-class expert in the {knowledge_domain} field, specializing in advanced data analysis and visualization.

#         Your task is to follow these steps strictly:
#         1.Generate a Complex Dataset: First, create a complex and extensive dataset for a "{chart_type}" chart centered around the topic of "{topic}". The dataset must be intricate but don't impact visualizationand meet the following requirements:
#             a.Entity Count: Include at least 7-9 distinct entities (e.g., companies, countries, products) to ensure high complexity and a rich visualization.
#             b.Data Dimensions: For each entity, generate multi-dimensional data across at least 4-6 distinct metrics.
#             c.Subtle Correlations: The internal relationships and correlations within the data must be subtle, non-linear, and multi-dimensional. They should not be immediately obvious and should require deep exploration of the chart to be discovered.
#         2.Ask a Challenging Question: Based on the dataset you generated, formulate a challenging analytical question that requires careful interpretation of complex relationships within a chart, rather than simple data retrieval. The following options are available: 
#             a. The question should require 2-5 steps of reasoning to solve. 
#             b. Ask about mathematical operations (differences, percentages, ratios) between data elements. 
#             c. Focus on identifying patterns, extreme values, or anomalies in the data visualization. 
#             d. Include questions about the relationships between different data points or series.
#         3.Provide clear answers: Please provide clear and concise answers to your questions. The answer must be concise, such as a noun or a number or a year, etc.
#         4.Identify and Annotate Core Evidence: This is a critical step. Review your question and answer, and precisely identify the "minimum necessary data subset" required to answer the question. Then, in the final JSON output, you must use the is_relevant_for_answer field (or relevant_cells for heatmaps) to mark only this minimum subset. All other irrelevant data points must have this field set to false.
#         5.Explain the Reasoning for Annotation: In the analysis.relevance_reasoning field, clearly explain why the marked data is the sufficient and necessary condition to answer the question.

#         Please note that the most important thing is to keep your answer concise!No additional explanation is needed!
#         Be careful not to crowd the data points too closely together to avoid overlapping elements in the rendered chart.
#         """

#         # 将基础提示和Schema结合，形成最终的、要求严格JSON格式的提示
#         final_prompt = f"""
#         {base_prompt}

#         Your entire response must be a single, raw JSON object that strictly adheres to the following JSON Schema.
#         Do not include any introductory or explanatory text, code block markers (like ```json), or any other characters before or after the JSON object itself.

#         JSON Schema:
#         {schema_str}
#         """
#         return final_prompt

#     def generate_chart_data(self, topic: str, knowledge_domain: str, chart_type: str, schema: Type[BaseModel]) -> Dict[str, Any]:
#         """
#         调用 OpenRouter API 生成结构化的图表数据。

#         Args:
#             topic (str): 数据集的核心主题。
#             knowledge_domain (str): 主题所属的知识领域。
#             chart_type (str): 期望生成的图表类型。
#             schema (Type[BaseModel]): 用于约束输出格式的Pydantic模型类。

#         Returns:
#             Dict[str, Any]: 从API返回并解析后的JSON数据（以Python字典形式）。
        
#         Raises:
#             Exception: 如果API调用失败或返回的数据无法解析。
#         """
#         prompt = self._build_prompt(topic, knowledge_domain, chart_type, schema)
        
#         try:
#             print(f"向 OpenRouter (模型: {self.model_name}) 发送请求：主题='{topic}', 图表类型='{chart_type}'...")
            
#             completion = self.client.chat.completions.create(
#                 extra_headers=self.extra_headers,
#                 model=self.model_name,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": prompt,
#                     }
#                 ],
#                 # 开启JSON模式，以确保模型返回的是一个格式正确的JSON字符串
#                 response_format={"type": "json_object"},
#             )
            
#             print("成功从 OpenRouter 接收到响应。")
            
#             # 提取返回的JSON字符串内容
#             response_text = completion.choices[0].message.content
            
#             # 使用Pydantic模型进行验证和解析，如果验证失败会抛出异常
#             parsed_data = schema.model_validate_json(response_text)
            
#             # 返回验证通过的Python字典
#             return parsed_data.model_dump()

#         except Exception as e:
#             print(f"调用 OpenRouter API 时发生错误: {e}")
#             # 向上抛出异常，以便main.py中的熔断机制可以捕获
#             raise

# openrouter_client.py
import os
import json
from typing import Dict, Any, Type
from pydantic import BaseModel
from openai import OpenAI
# 【新增】导入 Analysis Schema 用于验证第二个 LLM 的返回结果
from schemas import Analysis 

# ==============================================================================
# 1. OpenRouter 客户端 (使用 openai SDK)
# ==============================================================================

class OpenRouterClient:
    """
    一个封装了通过 OpenRouter 与兼容OpenAI的API（如Gemini）交互的客户端。
    使用 `openai` 库进行API调用，并支持结构化JSON输出。
    """
    def __init__(self, api_key: str, model_name: str = "google/gemini-2.5-flash-lite"):
        # ... (构造函数保持不变) ...
        if not api_key:
            raise ValueError("OpenRouter API key is required.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.api_key = api_key
        self.model_name = model_name
        
        self.extra_headers = {}

    def _build_prompt(self, topic: str, knowledge_domain: str, chart_type: str, schema: Type[BaseModel]) -> str:
        # ... (此方法基本不变, 但可以简化, 因为不再需要生成 QA) ...
        # 【修改】简化了原始Prompt，现在它只专注于生成高质量数据
        json_schema = schema.model_json_schema()
        schema_str = json.dumps(json_schema, indent=2)

        base_prompt = f"""
        You are a world-class expert in the {knowledge_domain} field, specializing in data generation.
        Your task is to generate a complex and high-quality dataset for a "{chart_type}" chart on the topic of "{topic}".
        The dataset should be intricate, containing subtle patterns and non-obvious relationships.
        Ensure you include at least 7-9 distinct entities and 4-6 metrics to provide depth.

        Your entire response must be a single, raw JSON object that strictly adheres to the following JSON Schema.
        Do not include any text or formatting outside the JSON object.

        JSON Schema:
        {schema_str}
        """
        return base_prompt

    def generate_chart_data(self, topic: str, knowledge_domain: str, chart_type: str, schema: Type[BaseModel]) -> Dict[str, Any]:
        # ... (此方法的功能不变, 只是现在调用的 prompt 和 schema 简化了) ...
        prompt = self._build_prompt(topic, knowledge_domain, chart_type, schema)
        try:
            print(f"向 OpenRouter (模型: {self.model_name}) 发送请求 [1/2 - 获取数据]：主题='{topic}', 图表类型='{chart_type}'...")
            
            completion = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            
            print("成功从 OpenRouter 接收到数据响应。")
            response_text = completion.choices[0].message.content
            parsed_data = schema.model_validate_json(response_text)
            return parsed_data.model_dump()
        except Exception as e:
            print(f"调用 OpenRouter API [获取数据] 时发生错误: {e}")
            raise

    # --- 【新增】用于从代码生成 Q&A 的新方法 ---
    def generate_question_from_code(self, chart_code: str, chart_type: str, topic: str) -> Dict[str, Any]:
        """
        根据生成图表的Python代码，提出一个高质量的分析问题和答案。

        Args:
            chart_code (str): 用于生成图表的完整 Python 代码。
            chart_type (str): 图表类型。
            topic (str): 图表的主题。

        Returns:
            Dict[str, Any]: 包含 "question" 和 "answer" 的字典。
        """
        json_schema = Analysis.model_json_schema()
        schema_str = json.dumps(json_schema, indent=2)

        prompt = f"""
        You are a senior data analyst reviewing a Python script that generates a '{chart_type}' visualization about '{topic}'.
        Your task is to act as an expert and formulate a deep, insightful question that can only be answered by carefully examining the visual chart produced by this code.

        Here is the Python code that will be executed to generate the chart:
        ```python
        {chart_code}
        ```
        Imagine you’re a data analyst and your colleague only sends you this image of the final graph. You can’t see the original data at all. Now, ask an insightful question based on this image that your colleague can answer based on this image alone.
        
        Follow these instructions:
        1.  **Analyze the Code:** Understand what data is being plotted and how. Pay attention to the relationships, scales, and specific data points being visualized.
        2.  **Formulate a Question:** Ask a question that requires calculations involving multiple data points (e.g., sum, average, difference) or a comparison of aggregated information.Be sure to ensure that all the information needed to answer the question can be found in the chart generated by the code, and do not miss any corresponding label information!
        3.  **Provide a Concise Answer:** Based on your analysis of what the chart will look like, provide a direct and concise answer to your question. The answer should be a short phrase, a number, or a category name.
        
        Your entire response must be a single, raw JSON object that strictly adheres to the following JSON Schema.
        Do not include any text outside the JSON object.

        JSON Schema:
        {schema_str}
        """

        try:
            print(f"向 OpenRouter (模型: {self.model_name}) 发送请求 [2/2 - 生成分析]：主题='{topic}'...")
            
            completion = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            
            print("成功从 OpenRouter 接收到分析响应。")
            response_text = completion.choices[0].message.content
            parsed_analysis = Analysis.model_validate_json(response_text)
            return parsed_analysis.model_dump()

        except Exception as e:
            print(f"调用 OpenRouter API [生成分析] 时发生错误: {e}")
            raise