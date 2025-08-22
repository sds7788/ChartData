# main.py
# import os
# import json
# import yaml
# import argparse
# import time
# from datetime import datetime
# from typing import Dict, Any, Type

# from config import GEMINI_API_KEYS
# from gemini_client import GeminiClient
# from schemas import ChartData

# # --- 导入所有生成器模块 ---
# from chart_generators.base_generator import BaseChartGenerator
# from chart_generators.matplotlib_generator import MatplotlibGenerator
# from chart_generators.pyecharts_generator import PyechartsGenerator
# from chart_generators.seaborn_generator import SeabornGenerator

# # --- 创建并填充生成器注册表 ---
# GENERATOR_MAP: Dict[str, Type[BaseChartGenerator]] = {
#     'matplotlib': MatplotlibGenerator,
#     'pyecharts': PyechartsGenerator,
#     'seaborn': SeabornGenerator,
# }

# def load_tasks_from_config(config_path: str) -> list:
#     """从YAML配置文件中加载任务列表。"""
#     try:
#         with open(config_path, 'r', encoding='utf-8') as f:
#             config = yaml.safe_load(f)
#         tasks = config.get('tasks', [])
#         if not isinstance(tasks, list):
#             print(f"错误：配置文件 '{config_path}' 中的 'tasks' 应该是一个列表。")
#             return []
#         return tasks
#     except FileNotFoundError:
#         print(f"错误：配置文件 '{config_path}' 未找到。")
#         return []
#     except yaml.YAMLError as e:
#         print(f"错误：解析YAML文件 '{config_path}' 时出错: {e}")
#         return []

# def main():
#     """项目主执行函数，由配置文件驱动，支持多种图表渲染后端。"""
#     parser = argparse.ArgumentParser(description="使用Gemini和多种渲染库自动化生成图表。")
#     parser.add_argument('--config', type=str, default='tasks.yaml', help='指定任务定义的YAML配置文件路径。 (默认: tasks.yaml)')
#     args = parser.parse_args()

#     output_dir = "output"
#     os.makedirs(output_dir, exist_ok=True)

#     if not GEMINI_API_KEYS or not isinstance(GEMINI_API_KEYS, list) or not GEMINI_API_KEYS[0].startswith("AIza"):
#         print("错误：请在 config.py 文件中正确设置您的 GEMINI_API_KEYS 列表。")
#         return

#     KEY_USAGE_LIMIT, REQUEST_DELAY_SECONDS = 98, 15
#     api_key_index, api_usage_count = 0, 0
#     client = GeminiClient(api_key=GEMINI_API_KEYS[api_key_index])
#     print(f"初始化客户端，使用 API Key 索引: {api_key_index}")
    
#     tasks = load_tasks_from_config(args.config)
#     if not tasks:
#         print("配置文件中没有找到任何有效任务，程序退出。")
#         return

#     print(f"从 '{args.config}' 加载了 {len(tasks)} 个任务。开始执行...\n")

#     for i, task in enumerate(tasks):
#         print(f"--- 开始执行任务 {i+1}/{len(tasks)} ---")
#         topic, domain, chart_type = task.get('topic'), task.get('domain'), task.get('chart_type')
        
#         renderer = task.get('renderer', 'matplotlib').lower()
#         GeneratorClass = GENERATOR_MAP.get(renderer)
        
#         if not GeneratorClass:
#             print(f"错误：未知的渲染器 '{renderer}'。有效选项为: {list(GENERATOR_MAP.keys())}。跳过此任务。")
#             continue
            
#         print(f"使用渲染器: {renderer}")
#         figsize, dpi = tuple(task.get('figsize', [12, 8])), task.get('dpi', 100)

#         if not all([topic, domain, chart_type]):
#             print(f"跳过一个格式不完整的任务: {task}\n"); continue

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         safe_domain = "".join(c if c.isalnum() else '_' for c in domain)
#         base_filename = f"{safe_domain.lower()}_{chart_type}_{renderer}_{timestamp}"
        
        # try:
        #     if api_usage_count >= KEY_USAGE_LIMIT:
        #         api_key_index = (api_key_index + 1) % len(GEMINI_API_KEYS)
        #         api_usage_count = 0
                
        #         # 获取新的API Key
        #         new_api_key = GEMINI_API_KEYS[api_key_index]
        #         print(f"\n[提示] API Key 达到使用上限，正在用新 Key (索引: {api_key_index}) 重新创建客户端...")
                
        #         # 用新的Key重新创建一个GeminiClient实例，并覆盖旧的client变量
        #         client = GeminiClient(api_key=new_api_key)

        #     api_usage_count += 1
#             chart_data_json = client.generate_chart_data(topic=topic, knowledge_domain=domain, chart_type=chart_type, schema=ChartData)
            
#             if not chart_data_json:
#                 print(f"未能从Gemini获取有效的数据，已跳过任务 '{topic}'。"); continue
            
#             print(f"当前 Key (索引: {api_key_index}) 已使用 {api_usage_count}/{KEY_USAGE_LIMIT} 次。")

#             generator = GeneratorClass(chart_data_json)
#             output_filename = f"{base_filename}.{generator.file_extension}"
#             output_path = os.path.join(output_dir, output_filename)
            
#             success = generator.create_chart(output_path, figsize=figsize, dpi=dpi)

#             reproducible_code = ""
#             if success:
#                 # --- 修改点: 将 dpi 也传递给 generate_code ---
#                 reproducible_code = generator.generate_code(figsize=figsize, dpi=dpi)
                
#                 if generator.file_extension == 'png':
#                     print(f"图表已保存至: {output_path} (尺寸: {int(figsize[0] * dpi)}x{int(figsize[1] * dpi)}px)")
#                 else: # pyecharts 的尺寸信息已在生成器内部打印
#                     pass # 保留此空分支以明确逻辑
#             else:
#                 print("生成图表失败。")

#             final_package = {
#                 "generation_info": {"topic": topic, "domain": domain, "chart_type": chart_type, "timestamp": timestamp, "renderer": renderer, "figsize": list(figsize), "dpi": dpi},
#                 "source_data": chart_data_json, "reproducible_code": reproducible_code,
#                 "question": chart_data_json.get('analysis', {}).get('question'), "answer": chart_data_json.get('analysis', {}).get('answer'),
#                 "output_filename": output_filename
#             }
            
#             output_json_path = os.path.join(output_dir, f"{base_filename}.json")
#             with open(output_json_path, 'w', encoding='utf-8') as f:
#                 json.dump(final_package, f, ensure_ascii=False, indent=4)
#             print(f"最终数据包已保存至: {output_json_path}")

#         except Exception as e:
#             print(f"处理任务 '{topic}' 时发生严重错误: {e}")
#         finally:
#             print(f"--- 任务 {i+1}/{len(tasks)} 执行完毕 ---")
#             if i < len(tasks) - 1:
#                  print(f"等待 {REQUEST_DELAY_SECONDS} 秒后继续下一个任务...\n"); time.sleep(REQUEST_DELAY_SECONDS)
#             else:
#                  print("\n所有任务执行完毕。")

# if __name__ == "__main__":
#     main()


# import os
# import json
# import yaml
# import argparse
# import time
# from datetime import datetime
# from typing import Dict, Type

# from config import GEMINI_API_KEYS
# from gemini_client import GeminiClient
# from schemas import ChartData
# # 新增导入
# from data_processor import create_chart_variations

# # 假设你的生成器都在这个包里
# from chart_generators.base_generator import BaseChartGenerator
# from chart_generators.matplotlib_generator import MatplotlibGenerator
# from chart_generators.seaborn_generator import SeabornGenerator

# GENERATOR_MAP: Dict[str, Type[BaseChartGenerator]] = {
#     'matplotlib': MatplotlibGenerator,
#     'seaborn': SeabornGenerator,
# }

# def load_tasks_from_config(config_path: str) -> list:
#     """从YAML配置文件中加载任务列表。"""
#     try:
#         with open(config_path, 'r', encoding='utf-8') as f:
#             config = yaml.safe_load(f)
#         return config.get('tasks', [])
#     except Exception as e:
#         print(f"加载或解析配置文件 '{config_path}' 出错: {e}")
#         return []

# def main():
#     parser = argparse.ArgumentParser(description="自动化生成多复杂度图表数据集。")
#     parser.add_argument('--config', type=str, default='tasks.yaml', help='任务定义的YAML配置文件路径。')
#     args = parser.parse_args()

#     output_dir = "output"
#     os.makedirs(output_dir, exist_ok=True)

#     if not GEMINI_API_KEYS or not GEMINI_API_KEYS[0].startswith("AIza"):
#         print("错误：请在 config.py 中正确设置 GEMINI_API_KEYS。")
#         return

#     KEY_USAGE_LIMIT, REQUEST_DELAY_SECONDS = 250, 5
#     api_key_index, api_usage_count = 0, 0
#     client = GeminiClient(api_key=GEMINI_API_KEYS[api_key_index])
    
#     tasks = load_tasks_from_config(args.config)
#     if not tasks:
#         print("未找到有效任务，程序退出。")
#         return

#     print(f"加载了 {len(tasks)} 个任务。开始执行...\n")

#     for i, task in enumerate(tasks):
#         print(f"--- 开始执行任务 {i+1}/{len(tasks)} ---")
#         topic = task.get('topic')
#         domain = task.get('domain')
#         chart_type = task.get('chart_type')
#         renderer = task.get('renderer', 'matplotlib').lower()
        
#         GeneratorClass = GENERATOR_MAP.get(renderer)
#         if not GeneratorClass:
#             print(f"错误：未知的渲染器 '{renderer}'。跳过此任务。")
#             continue
            
#         figsize, dpi = tuple(task.get('figsize', [12, 8])), task.get('dpi', 100)
#         num_versions = task.get('complexity_steps', 3)# 在YAML中定义复杂度层级，默认为3

#         if not all([topic, domain, chart_type]):
#             print(f"跳过格式不完整的任务: {task}\n"); continue

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         safe_domain = "".join(c if c.isalnum() else '_' for c in domain)
#         base_filename = f"{safe_domain.lower()}_{chart_type}_{renderer}_{timestamp}"
        
#         try:
#             # API Key 轮换逻辑
#             if api_usage_count >= KEY_USAGE_LIMIT:
#                 api_key_index = (api_key_index + 1) % len(GEMINI_API_KEYS)
#                 api_usage_count = 0
                
#                 # 获取新的API Key
#                 new_api_key = GEMINI_API_KEYS[api_key_index]
#                 print(f"\n[提示] API Key 达到使用上限，正在用新 Key (索引: {api_key_index}) 重新创建客户端...")
                
#                 # 用新的Key重新创建一个GeminiClient实例，并覆盖旧的client变量
#                 client = GeminiClient(api_key=new_api_key)

#             api_usage_count += 1
#             # 1. 从Gemini获取带标注的完整数据
#             full_chart_data = client.generate_chart_data(
#                 topic=topic, knowledge_domain=domain, chart_type=chart_type, schema=ChartData
#             )
            
#             if not full_chart_data:
#                 print(f"未能从Gemini获取有效数据，跳过任务 '{topic}'。"); continue
            
#             print(f"当前 Key (索引: {api_key_index}) 已使用 {api_usage_count}/{KEY_USAGE_LIMIT} 次。")

#             # 2. 生成数据变体
#             chart_variations = create_chart_variations(full_chart_data, total_versions=num_versions)
#             print(f"已生成 {len(chart_variations)} 个复杂度版本的数据。")

#             generated_files = []
#             # 3. 为每个变体生成图表
#             for idx, variant_data in enumerate(chart_variations):
#                 complexity_label = f"complexity_{idx}"
                
#                 generator = GeneratorClass(variant_data)
#                 variant_filename = f"{base_filename}_{complexity_label}.{generator.file_extension}"
#                 output_path = os.path.join(output_dir, variant_filename)
                
#                 success = generator.create_chart(output_path, figsize=figsize, dpi=dpi)

#                 if success:
#                     print(f"  - 已生成图表: {variant_filename}")
#                     generated_files.append({
#                         "complexity_level": idx,
#                         "filename": variant_filename,
#                         "data_used": variant_data # 保存用于生成此图的具体数据
#                     })
#                 else:
#                     print(f"  - 生成图表失败: {complexity_label}")

#             # 4. 打包最终的数据集文件
#             final_package = {
#                 "generation_info": {**task, "timestamp": timestamp, "base_filename": base_filename},
#                 "qa_pair": {
#                     "question": full_chart_data.get('analysis', {}).get('question'),
#                     "answer": full_chart_data.get('analysis', {}).get('answer'),
#                     "relevance_reasoning": full_chart_data.get('analysis', {}).get('relevance_reasoning'),
#                 },
#                 "source_data_annotated": full_chart_data,
#                 "generated_charts": generated_files
#             }
            
#             output_json_path = os.path.join(output_dir, f"{base_filename}_dataset.json")
#             with open(output_json_path, 'w', encoding='utf-8') as f:
#                 json.dump(final_package, f, ensure_ascii=False, indent=4)
#             print(f"最终数据集包已保存至: {output_json_path}")

#         except Exception as e:
#             print(f"处理任务 '{topic}' 时发生严重错误: {e}")
#         finally:
#             print(f"--- 任务 {i+1}/{len(tasks)} 执行完毕 ---")
#             if i < len(tasks) - 1:
#                  print(f"等待 {REQUEST_DELAY_SECONDS} 秒...\n"); time.sleep(REQUEST_DELAY_SECONDS)
#             else:
#                  print("\n所有任务执行完毕。")

# if __name__ == "__main__":
#     main()

# main.py
import os
import json
import yaml
import argparse
import time
from datetime import datetime
from typing import Dict, Type

from config import GEMINI_API_KEYS
from gemini_client import GeminiClient
# --- 【修改】导入新的分离式 Schema ---
from schemas import SingleChartData, FacetChartData
from data_processor import create_chart_variations

# 假设你的生成器都在这个包里
from chart_generators.base_generator import BaseChartGenerator
from chart_generators.matplotlib_generator import MatplotlibGenerator
from chart_generators.seaborn_generator import SeabornGenerator

GENERATOR_MAP: Dict[str, Type[BaseChartGenerator]] = {
    'matplotlib': MatplotlibGenerator,
    'seaborn': SeabornGenerator,
}

def load_tasks_from_config(config_path: str) -> list:
    """从YAML配置文件中加载任务列表。"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('tasks', [])
    except Exception as e:
        print(f"加载或解析配置文件 '{config_path}' 出错: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="自动化生成多复杂度图表数据集。")
    parser.add_argument('--config', type=str, default='tasks.yaml', help='任务定义的YAML配置文件路径。')
    args = parser.parse_args()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if not GEMINI_API_KEYS or not GEMINI_API_KEYS[0].startswith("AIza"):
        print("错误：请在 config.py 中正确设置 GEMINI_API_KEYS。")
        return

    KEY_USAGE_LIMIT, REQUEST_DELAY_SECONDS = 250, 5
    api_key_index, api_usage_count = 0, 0
    client = GeminiClient(api_key=GEMINI_API_KEYS[api_key_index])
    
    tasks = load_tasks_from_config(args.config)
    if not tasks:
        print("未找到有效任务，程序退出。")
        return

    print(f"加载了 {len(tasks)} 个任务。开始执行...\n")

    for i, task in enumerate(tasks):
        print(f"--- 开始执行任务 {i+1}/{len(tasks)} ---")
        topic = task.get('topic')
        domain = task.get('domain')
        chart_type = task.get('chart_type')
        renderer = task.get('renderer', 'matplotlib').lower()
        
        # --- 【修改】根据 task_type 选择 Schema ---
        task_type = task.get('task_type', 'single') # 默认为单图
        schema_to_use = FacetChartData if task_type == 'facet' else SingleChartData
        print(f"任务类型: '{task_type}', 使用 Schema: '{schema_to_use.__name__}'")
        # --- 修改结束 ---

        GeneratorClass = GENERATOR_MAP.get(renderer)
        if not GeneratorClass:
            print(f"错误：未知的渲染器 '{renderer}'。跳过此任务。")
            continue
            
        figsize, dpi = tuple(task.get('figsize', [12, 8])), task.get('dpi', 100)
        num_versions = task.get('complexity_steps', 3)

        if not all([topic, domain, chart_type]):
            print(f"跳过格式不完整的任务: {task}\n"); continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_domain = "".join(c if c.isalnum() else '_' for c in domain)
        base_filename = f"{safe_domain.lower()}_{chart_type}_{renderer}_{timestamp}"
        
        try:
            # API Key 轮换逻辑
            if api_usage_count >= KEY_USAGE_LIMIT:
                api_key_index = (api_key_index + 1) % len(GEMINI_API_KEYS)
                api_usage_count = 0
                new_api_key = GEMINI_API_KEYS[api_key_index]
                print(f"\n[提示] API Key 达到使用上限，正在用新 Key (索引: {api_key_index}) 重新创建客户端...")
                client = GeminiClient(api_key=new_api_key)

            api_usage_count += 1
            # --- 【修改】调用 Gemini 时传入选定的 Schema ---
            full_chart_data = client.generate_chart_data(
                topic=topic, knowledge_domain=domain, chart_type=chart_type, schema=schema_to_use
            )
            
            if not full_chart_data:
                print(f"未能从Gemini获取有效数据，跳过任务 '{topic}'。"); continue
            
            print(f"当前 Key (索引: {api_key_index}) 已使用 {api_usage_count}/{KEY_USAGE_LIMIT} 次。")

            # --- 【修改】复杂度生成逻辑现在只对单图任务有意义 ---
            # 对于多图任务，create_chart_variations 会直接返回原始数据，这符合我们的预期
            chart_variations = create_chart_variations(full_chart_data, total_versions=num_versions)
            print(f"已生成 {len(chart_variations)} 个复杂度版本的数据。")

            generated_files = []
            # 为每个变体生成图表
            for idx, variant_data in enumerate(chart_variations):
                complexity_label = f"complexity_{idx}"
                
                generator = GeneratorClass(variant_data)
                variant_filename = f"{base_filename}_{complexity_label}.{generator.file_extension}"
                output_path = os.path.join(output_dir, variant_filename)
                
                success = generator.create_chart(output_path, figsize=figsize, dpi=dpi)

                if success:
                    print(f"  - 已生成图表: {variant_filename}")
                    generated_files.append({
                        "complexity_level": idx,
                        "filename": variant_filename,
                        "data_used": variant_data
                    })
                else:
                    print(f"  - 生成图表失败: {complexity_label}")

            # 打包最终的数据集文件
            final_package = {
                "generation_info": {**task, "timestamp": timestamp, "base_filename": base_filename},
                "qa_pair": {
                    "question": full_chart_data.get('analysis', {}).get('question'),
                    "answer": full_chart_data.get('analysis', {}).get('answer'),
                    "relevance_reasoning": full_chart_data.get('analysis', {}).get('relevance_reasoning'),
                },
                "source_data_annotated": full_chart_data,
                "generated_charts": generated_files
            }
            
            output_json_path = os.path.join(output_dir, f"{base_filename}_dataset.json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_package, f, ensure_ascii=False, indent=4)
            print(f"最终数据集包已保存至: {output_json_path}")

        except Exception as e:
            print(f"处理任务 '{topic}' 时发生严重错误: {e}")
        finally:
            print(f"--- 任务 {i+1}/{len(tasks)} 执行完毕 ---")
            if i < len(tasks) - 1:
                 print(f"等待 {REQUEST_DELAY_SECONDS} 秒...\n"); time.sleep(REQUEST_DELAY_SECONDS)
            else:
                 print("\n所有任务执行完毕。")

if __name__ == "__main__":
    main()
