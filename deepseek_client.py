# # deepseek_client.py
# import json
# from typing import Dict, Any, Type
# from pydantic import BaseModel
# from openai import OpenAI

# # ==============================================================================
# # 1. DeepSeek 客户端
# # ==============================================================================

# class DeepSeekClient:
#     """
#     一个封装了与 DeepSeek LLM API 交互的客户端类。
#     """
#     def __init__(self, api_key: str, model_name: str = "deepseek-chat"):
#         """
#         初始化 DeepSeek 客户端。

#         Args:
#             api_key (str): 您的 DeepSeek API 密钥。
#             model_name (str): 要使用的 DeepSeek 模型名称，例如 "deepseek-chat"。
#         """
#         # 使用 OpenAI 的库来实例化客户端，但指向 DeepSeek 的 API 端点
#         self.client = OpenAI(
#             api_key=api_key, # 使用传入的 API Key
#             base_url="https://api.deepseek.com/v1"
#         )
#         self.model_name: str = model_name

#     def _build_prompt(self, topic: str, knowledge_domain: str, chart_type: str, schema: Type[BaseModel]) -> str:
#         """
#         根据输入动态构建一个复杂的提示，并附加 JSON Schema 以增强输出的可靠性。

#         Args:
#             topic (str): 数据集的核心主题。
#             knowledge_domain (str): 主题所属的知识领域。
#             chart_type (str): 期望生成的图表类型。
#             schema (Type[BaseModel]): 用于约束输出格式的 Pydantic 模型类。

#         Returns:
#             str: 构建完成的、包含指令和 Schema 的完整提示字符串。
#         """
#         # 基础指令部分保持不变
#         base_prompt = f"""
#         You are an expert in the field of {knowledge_domain}, specializing in data analysis.
#         Your task is to generate a JSON object for a {chart_type} chart on the topic of '{topic}'. This JSON object must include the data, a question, an answer, and evidence annotation.
#         Please strictly follow the steps and requirements below:
#         1.  **Generate Chart Data**:
#             * Create a dataset containing 4 to 6 main data objects.
#             * The data can include multiple dimensions or categories to add some complexity.
#         2.  **Generate a Question about the Chart**:
#             * Ask a question that requires calculations involving multiple data points (e.g., sum, average, difference) or a comparison of aggregated information. For example, "What is the total sum of categories A, B, and E?" or "What is the difference between the average values of category X and category Y?".
#         3.  **Provide a Clear and Concise Answer**:
#             * Clearly state the answer to the question based on the data you generated. The answer can only be a noun, number, or phrase.
#         4.  **Identify and Annotate the Core Evidence**:
#             * In your final JSON output, review your question and answer.
#             * Identify the "minimum necessary data subset" required to answer the question. For a medium-difficulty question, this typically includes all data points involved in the calculation or comparison.
#             * Set the `is_relevant_for_answer` field to `true` ONLY for this minimum subset of data points.
#             * All other data points not directly used to derive the answer must have this field set to `false`.

#         Please note that the most important thing is to keep your answer concise!No additional explanation is needed!
#         Be careful not to crowd the data points too closely together to avoid overlapping elements in the rendered chart.
        
#         Your entire response must be a single, raw JSON object that strictly adheres to the JSON Schema I provide. Do not include any introductory or explanatory text, code block markers, or any other characters before or after the JSON object itself.
#         """
        
#         # 将 Pydantic Schema 转换为 JSON Schema 字符串
#         schema_json = json.dumps(schema.model_json_schema(), indent=2)
        
#         # 将 Schema 附加到提示的末尾，以指导模型输出
#         full_prompt = (
#             f"{base_prompt}\n\n"
#             "Here is the JSON schema you MUST strictly adhere to:\n"
#             f"```json\n{schema_json}\n```"
#         )
#         return full_prompt

#     def generate_chart_data(self, topic: str, knowledge_domain: str, chart_type: str, schema: Type[BaseModel]) -> Dict[str, Any]:
#         """
#         调用 DeepSeek API 生成结构化的图表数据。

#         Args:
#             topic (str): 数据集的核心主题。
#             knowledge_domain (str): 主题所属的知识领域。
#             chart_type (str): 期望生成的图表类型。
#             schema (Type[BaseModel]): 用于约束和验证输出格式的 Pydantic 模型类。

#         Returns:
#             Dict[str, Any]: 从 API 返回并解析、验证后的 JSON 数据（以 Python 字典形式）。
        
#         Raises:
#             Exception: 如果 API 调用失败或返回的数据无法通过 Pydantic 验证。
#         """
#         prompt = self._build_prompt(topic, knowledge_domain, chart_type, schema)
        
#         try:
#             print(f"向 DeepSeek API 发送请求：主题='{topic}', 图表类型='{chart_type}'...")
            
#             response = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 response_format={"type": "json_object"}
#             )
            
#             response_text = response.choices[0].message.content
#             print(f"成功从 DeepSeek API 接收到响应 (主题='{topic}')。")
            
#             parsed_data = schema.model_validate_json(response_text)
            
#             return parsed_data.model_dump()

#         except Exception as e:
#             print(f"调用 DeepSeek API 或解析数据时发生错误 (主题='{topic}'): {e}")
#             raise

# deepseek_client.py
import json
from typing import Dict, Any, Type
from pydantic import BaseModel
from openai import OpenAI
# 【新增】导入 Analysis Schema 用于验证第二个 LLM 的返回结果
from schemas import Analysis

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
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.model_name: str = model_name
        self.api_key = api_key # 保存api_key

    def _build_data_prompt(self, topic: str, knowledge_domain: str, chart_type: str, schema: Type[BaseModel]) -> str:
        """
        【修改】构建一个【只用于生成数据】的提示。
        """
        json_schema = schema.model_json_schema()
        schema_str = json.dumps(json_schema, indent=2)

        base_prompt = f"""
        You are a world-class expert in the {knowledge_domain} field, specializing in data generation.
        Your task is to generate a complex and high-quality dataset for a "{chart_type}" chart on the topic of "{topic}".
        The dataset should be intricate, containing subtle patterns and non-obvious relationships.
        Ensure you include at least 4-6 distinct entities and multiple metrics to provide depth.

        Your entire response must be a single, raw JSON object that strictly adheres to the following JSON Schema.
        Do not include any text or formatting outside the JSON object.

        Here is the JSON schema you MUST strictly adhere to:
        ```json
        {schema_str}
        ```
        """
        return base_prompt

    def generate_chart_data(self, topic: str, knowledge_domain: str, chart_type: str, schema: Type[BaseModel]) -> Dict[str, Any]:
        """
        【修改】调用 DeepSeek API 【仅生成】结构化的图表数据。
        """
        prompt = self._build_data_prompt(topic, knowledge_domain, chart_type, schema)
        
        try:
            print(f"向 DeepSeek [1/2 - 获取数据]: 主题='{topic}', 图表='{chart_type}'...")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            print(f"成功从 DeepSeek 接收到数据 (主题='{topic}')。")
            
            parsed_data = schema.model_validate_json(response_text)
            return parsed_data.model_dump()

        except Exception as e:
            print(f"调用 DeepSeek API [获取数据] 时出错 (主题='{topic}'): {e}")
            raise

    # --- 【新增】用于从代码生成 Q&A 的新方法 ---
    def generate_question_from_code(self, chart_code: str, chart_type: str, topic: str) -> Dict[str, Any]:
        """
        根据生成图表的Python代码，提出一个高质量的分析问题和答案。
        """
        json_schema = Analysis.model_json_schema()
        schema_str = json.dumps(json_schema, indent=2)

        prompt = f"""
        You are a senior data analyst reviewing a Python script that generates a '{chart_type}' visualization about '{topic}'.
        Your task is to formulate a deep, insightful question that can only be answered by carefully examining the visual chart produced by this code.

        Here is the Python code that will be executed to generate the chart:
        ```python
        {chart_code}
        ```

        Follow these instructions:
        1.  **Analyze the Code:** Understand what data is being plotted and how.
        2.  **Formulate a Question:** Create a challenging question that requires interpreting the final chart. The question should focus on patterns, comparisons, or anomalies made apparent by the visualization logic.
        3.  **Provide a Concise Answer:** Based on your analysis of what the chart will look like, provide a direct and concise answer. The answer should be a short phrase, a number, or a category name.
        
        Your entire response must be a single, raw JSON object that strictly adheres to the following JSON Schema.

        JSON Schema:
        ```json
        {schema_str}
        ```
        """

        try:
            print(f"向 DeepSeek [2/2 - 生成分析]: 主题='{topic}'...")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            print(f"成功从 DeepSeek 接收到分析 (主题='{topic}')。")
            parsed_analysis = Analysis.model_validate_json(response_text)
            return parsed_analysis.model_dump()

        except Exception as e:
            print(f"调用 DeepSeek API [生成分析] 时出错 (主题='{topic}'): {e}")
            raise