# # main.py
# import os
# import json
# import yaml
# # --- 【修改】不再需要 argparse ---
# import time
# from datetime import datetime
# from typing import Dict, Type, Any
# import threading
# from concurrent.futures import ThreadPoolExecutor

# from deepseek_config import DEEPSEEK_API_KEYS
# from deepseek_client import DeepSeekClient 
# from schemas import SingleChartData, FacetChartData
# from data_processor import create_chart_variations

# from chart_generators.base_generator import BaseChartGenerator
# from chart_generators.matplotlib_generator import MatplotlibGenerator
# from chart_generators.seaborn_generator import SeabornGenerator

# GENERATOR_MAP: Dict[str, Type[BaseChartGenerator]] = {
#     'matplotlib': MatplotlibGenerator,
#     'seaborn': SeabornGenerator,
# }

# OUTPUT_DIR = "deepseek_output"

# # --- 【修改】将配置硬编码为常量 ---
# CONFIG_FILE = 'tasks.yaml'
# LOOP_DELAY_SECONDS = 10 # 每轮循环结束后的等待时间（秒）

# def load_tasks_from_config(config_path: str) -> list:
#     """从YAML配置文件中加载任务列表。"""
#     try:
#         with open(config_path, 'r', encoding='utf-8') as f:
#             config = yaml.safe_load(f)
#         return config.get('tasks', [])
#     except FileNotFoundError:
#         print(f"错误：配置文件 '{config_path}' 未找到。请确保该文件存在。")
#         return []
#     except Exception as e:
#         print(f"加载或解析配置文件 '{config_path}' 出错: {e}")
#         return []

# def process_task(task: Dict[str, Any], api_key: str, task_index: int):
#     """
#     处理单个图表生成任务的完整流程（设计为在单个线程中运行）。
    
#     Args:
#         task (Dict[str, Any]): 从YAML加载的任务配置。
#         api_key (str): 用于此任务的 DeepSeek API 密钥。
#         task_index (int): 任务的序号，用于日志记录。
#     """
#     thread_name = threading.current_thread().name
#     print(f"--- [线程 {thread_name}, 任务 {task_index}] 开始执行 ---")

#     topic = task.get('topic')
#     domain = task.get('domain')
#     chart_type = task.get('chart_type')
#     renderer = task.get('renderer', 'matplotlib').lower()
    
#     task_type = task.get('task_type', 'single')
#     schema_to_use = FacetChartData if task_type == 'facet' else SingleChartData
    
#     GeneratorClass = GENERATOR_MAP.get(renderer)
#     if not GeneratorClass:
#         print(f"错误 [任务 {task_index}]：未知的渲染器 '{renderer}'。跳过此任务。")
#         return
        
#     figsize, dpi = tuple(task.get('figsize', [12, 8])), task.get('dpi', 100)
#     num_versions = task.get('complexity_steps', 3)

#     if not all([topic, domain, chart_type]):
#         print(f"跳过 [任务 {task_index}]：格式不完整的任务: {task}")
#         return

#     # 为文件名生成唯一时间戳，添加微秒以增强并发唯一性
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") 
#     safe_domain = "".join(c if c.isalnum() else '_' for c in domain)
#     base_filename = f"{safe_domain.lower()}_{chart_type}_{renderer}_{timestamp}_{task_index}"
    
#     try:
#         # 每个线程创建自己的客户端实例
#         client = DeepSeekClient(api_key=api_key)
        
#         print(f"[任务 {task_index}] 使用 Schema '{schema_to_use.__name__}' 调用 LLM...")
#         full_chart_data = client.generate_chart_data(
#             topic=topic, knowledge_domain=domain, chart_type=chart_type, schema=schema_to_use
#         )
        
#         if not full_chart_data:
#             print(f"[任务 {task_index}] 未能从 DeepSeek 获取有效数据，跳过任务 '{topic}'。")
#             return
        
#         chart_variations = create_chart_variations(full_chart_data, total_versions=num_versions)
#         print(f"[任务 {task_index}] 已生成 {len(chart_variations)} 个复杂度版本的数据。")

#         generated_files = []
#         for idx, variant_data in enumerate(chart_variations):
#             complexity_label = f"complexity_{idx}"
            
#             generator = GeneratorClass(variant_data)
#             variant_filename = f"{base_filename}_{complexity_label}.{generator.file_extension}"
#             output_path = os.path.join(OUTPUT_DIR, variant_filename)
            
#             success = generator.create_chart(output_path, figsize=figsize, dpi=dpi)

#             if success:
#                 print(f"   - [任务 {task_index}] 已生成图表: {variant_filename}")
#                 generated_files.append({
#                     "complexity_level": idx,
#                     "filename": variant_filename,
#                     "data_used": variant_data
#                 })
#             else:
#                 print(f"   - [任务 {task_index}] 生成图表失败: {complexity_label}")

#         final_package = {
#             "generation_info": {**task, "timestamp": timestamp, "base_filename": base_filename},
#             "qa_pair": {
#                 "question": full_chart_data.get('analysis', {}).get('question'),
#                 "answer": full_chart_data.get('analysis', {}).get('answer'),
#             },
#             "source_data_annotated": full_chart_data,
#             "generated_charts": generated_files
#         }
        
#         output_json_path = os.path.join(OUTPUT_DIR, f"{base_filename}_dataset.json")
#         with open(output_json_path, 'w', encoding='utf-8') as f:
#             json.dump(final_package, f, ensure_ascii=False, indent=4)
#         print(f"[任务 {task_index}] 最终数据集包已保存至: {output_json_path}")

#     except Exception as e:
#         print(f"处理任务 {task_index} ('{topic}') 时发生严重错误: {e}")
#     finally:
#         print(f"--- [线程 {thread_name}, 任务 {task_index}] 执行完毕 ---")

# def main():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     if not DEEPSEEK_API_KEYS or not isinstance(DEEPSEEK_API_KEYS, list) or not DEEPSEEK_API_KEYS[0]:
#         print("错误：请在 config.py 文件中正确设置您的 DEEPSEEK_API_KEYS 列表。")
#         return

#     tasks = load_tasks_from_config(CONFIG_FILE)
#     if not tasks:
#         print("未找到有效任务，程序退出。")
#         return

#     # --- 【修改】直接使用 API Key 数量作为线程数 ---
#     num_workers = len(DEEPSEEK_API_KEYS)

#     print(f"加载了 {len(tasks)} 个任务。将使用 {num_workers} 个线程并发执行。")
#     print(f"任务将无限循环执行。每轮结束后将等待 {LOOP_DELAY_SECONDS} 秒。")

#     run_count = 0
#     # --- 【修改】移除对 args.loop 的检查，使其成为无限循环 ---
#     while True:
#         run_count += 1
#         print(f"\n================== 开始第 {run_count} 轮任务 ==================")
        
#         with ThreadPoolExecutor(max_workers=num_workers) as executor:
#             # 使用 round-robin 方式为每个任务分配一个 API key
#             for i, task in enumerate(tasks):
#                 api_key_to_use = DEEPSEEK_API_KEYS[i % num_workers]
#                 executor.submit(process_task, task, api_key_to_use, i + 1)
        
#         print(f"\n================== 第 {run_count} 轮任务执行完毕 ==================")
#         print(f"等待 {LOOP_DELAY_SECONDS} 秒后开始下一轮...")
#         time.sleep(LOOP_DELAY_SECONDS)

# if __name__ == "__main__":
#     main()
# deepseek_main.py
import os
import json
import yaml
import time
from datetime import datetime
from typing import Dict, Type, Any
import threading
from concurrent.futures import ThreadPoolExecutor

from deepseek_config import DEEPSEEK_API_KEYS
from deepseek_client import DeepSeekClient
# --- 【修改】导入新的基础 Schema ---
from schemas import BaseChartData, BaseFacetChartData
from data_processor import create_chart_variations

from chart_generators.base_generator import BaseChartGenerator
from chart_generators.matplotlib_generator import MatplotlibGenerator
from chart_generators.seaborn_generator import SeabornGenerator

GENERATOR_MAP: Dict[str, Type[BaseChartGenerator]] = {
    'matplotlib': MatplotlibGenerator,
    'seaborn': SeabornGenerator,
}

OUTPUT_DIR = "deepseek_output"
CONFIG_FILE = 'tasks.yaml'
LOOP_DELAY_SECONDS = 10

def load_tasks_from_config(config_path: str) -> list:
    # ... (此函数保持不变) ...
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('tasks', [])
    except FileNotFoundError:
        print(f"错误：配置文件 '{config_path}' 未找到。")
        return []
    except Exception as e:
        print(f"加载或解析配置文件 '{config_path}' 出错: {e}")
        return []

def process_task(task: Dict[str, Any], api_key: str, task_index: int):
    """
    【修改】处理单个任务的完整流程，采用两阶段生成和精简输出。
    """
    thread_name = threading.current_thread().name
    print(f"--- [线程 {thread_name}, 任务 {task_index}] 开始执行: {task.get('topic')} ---")

    try:
        # 从任务配置中提取信息
        topic = task.get('topic')
        domain = task.get('domain')
        chart_type = task.get('chart_type')
        renderer = task.get('renderer', 'matplotlib').lower()
        task_type = task.get('task_type', 'single')
        
        GeneratorClass = GENERATOR_MAP.get(renderer)
        if not all([topic, domain, chart_type, GeneratorClass]):
            print(f"跳过 [任务 {task_index}]：配置不完整或渲染器无效。")
            return

        # 1. 初始化客户端
        client = DeepSeekClient(api_key=api_key)

        # 2. 选择【仅数据】的 Schema，并进行第一次 LLM 调用
        schema_for_data = BaseFacetChartData if task_type == 'facet' else BaseChartData
        chart_data_only = client.generate_chart_data(
            topic=topic, knowledge_domain=domain, chart_type=chart_type, schema=schema_for_data
        )
        if not chart_data_only:
            print(f"[任务 {task_index}] 未能获取有效数据，跳过。")
            return

        # 3. 生成可复现的 Python 代码
        # 为了让代码能分析最复杂的情况，我们用最完整的数据来生成它
        temp_full_package = chart_data_only.copy()
        temp_full_package.update(task) # 生成器需要顶层的 chart_type 等信息
        generator_for_code = GeneratorClass(temp_full_package)
        reproducible_code = generator_for_code.generate_code()
        if not reproducible_code:
            print(f"[任务 {task_index}] 生成代码失败，跳过。")
            return

        # 4. 进行第二次 LLM 调用，根据代码生成分析 (Q&A)
        analysis_data = client.generate_question_from_code(
            chart_code=reproducible_code, chart_type=chart_type, topic=topic
        )
        if not analysis_data:
            print(f"[任务 {task_index}] 未能从代码生成分析，跳过。")
            return

        # 5. 组合数据和分析，形成用于生成图表的完整数据包
        full_chart_data = {**chart_data_only, "analysis": analysis_data}

        # 6. 生成不同复杂度的图表图片
        num_versions = task.get('complexity_steps', 3)
        chart_variations = create_chart_variations(full_chart_data, total_versions=num_versions)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_domain = "".join(c if c.isalnum() else '_' for c in domain)
        base_filename = f"{safe_domain.lower()}_{chart_type}_{renderer}_{timestamp}_{task_index}"
        
        generated_files_info = []
        figsize, dpi = tuple(task.get('figsize', [12, 8])), task.get('dpi', 100)

        for idx, variant_data in enumerate(chart_variations):
            generator = GeneratorClass(variant_data)
            variant_filename = f"{base_filename}_complexity_{idx}.{generator.file_extension}"
            output_path = os.path.join(OUTPUT_DIR, variant_filename)
            
            if generator.create_chart(output_path, figsize=figsize, dpi=dpi):
                print(f"   - [任务 {task_index}] 已生成图表: {variant_filename}")
                generated_files_info.append({
                    "complexity": idx,
                    "filename": variant_filename
                })
        
        # 7. 【修改】构建精简版的 final_package
        final_package = {
            "topic": topic,
            "base_filename": base_filename,
            "question": analysis_data.get('question'),
            "answer": analysis_data.get('answer'),
            "source_data": chart_data_only, # 只包含纯数据
            "generated_code": reproducible_code,
            "generated_files": generated_files_info # 精简版的文件列表
        }
        
        output_json_path = os.path.join(OUTPUT_DIR, f"{base_filename}_dataset.json")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_package, f, ensure_ascii=False, indent=4)
        print(f"[任务 {task_index}] 精简版数据集包已保存: {output_json_path}")

    except Exception as e:
        import traceback
        print(f"处理任务 {task_index} ('{topic}') 时发生严重错误: {e}")
        traceback.print_exc()
    finally:
        print(f"--- [线程 {thread_name}, 任务 {task_index}] 执行完毕 ---")

def main():
    # ... (main 函数的逻辑保持不变) ...
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not DEEPSEEK_API_KEYS or not isinstance(DEEPSEEK_API_KEYS, list) or not DEEPSEEK_API_KEYS[0]:
        print("错误：请在 config.py 文件中正确设置您的 DEEPSEEK_API_KEYS 列表。")
        return

    tasks = load_tasks_from_config(CONFIG_FILE)
    if not tasks:
        print("未找到有效任务，程序退出。")
        return

    num_workers = len(DEEPSEEK_API_KEYS)
    print(f"加载了 {len(tasks)} 个任务。将使用 {num_workers} 个线程并发执行。")
    print(f"任务将无限循环执行。每轮结束后将等待 {LOOP_DELAY_SECONDS} 秒。")

    run_count = 0
    while True:
        run_count += 1
        print(f"\n================== 开始第 {run_count} 轮任务 ==================")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for i, task in enumerate(tasks):
                api_key_to_use = DEEPSEEK_API_KEYS[i % num_workers]
                executor.submit(process_task, task, api_key_to_use, i + 1)
        
        print(f"\n================== 第 {run_count} 轮任务执行完毕 ==================")
        print(f"等待 {LOOP_DELAY_SECONDS} 秒后开始下一轮...")
        time.sleep(LOOP_DELAY_SECONDS)

if __name__ == "__main__":
    main()