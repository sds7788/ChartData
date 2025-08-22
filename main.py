# # main.py
# import os
# import json
# import yaml
# import argparse
# import time
# from datetime import datetime
# from typing import Dict, Type

# from config import GEMINI_API_KEYS
# from gemini_client import GeminiClient
# # --- 【修改】导入新的分离式 Schema ---
# from schemas import SingleChartData, FacetChartData
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
        
#         # --- 【修改】根据 task_type 选择 Schema ---
#         task_type = task.get('task_type', 'single') # 默认为单图
#         schema_to_use = FacetChartData if task_type == 'facet' else SingleChartData
#         print(f"任务类型: '{task_type}', 使用 Schema: '{schema_to_use.__name__}'")
#         # --- 修改结束 ---

#         GeneratorClass = GENERATOR_MAP.get(renderer)
#         if not GeneratorClass:
#             print(f"错误：未知的渲染器 '{renderer}'。跳过此任务。")
#             continue
            
#         figsize, dpi = tuple(task.get('figsize', [12, 8])), task.get('dpi', 100)
#         num_versions = task.get('complexity_steps', 3)

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
#                 new_api_key = GEMINI_API_KEYS[api_key_index]
#                 print(f"\n[提示] API Key 达到使用上限，正在用新 Key (索引: {api_key_index}) 重新创建客户端...")
#                 client = GeminiClient(api_key=new_api_key)

#             api_usage_count += 1
#             # --- 【修改】调用 Gemini 时传入选定的 Schema ---
#             full_chart_data = client.generate_chart_data(
#                 topic=topic, knowledge_domain=domain, chart_type=chart_type, schema=schema_to_use
#             )
            
#             if not full_chart_data:
#                 print(f"未能从Gemini获取有效数据，跳过任务 '{topic}'。"); continue
            
#             print(f"当前 Key (索引: {api_key_index}) 已使用 {api_usage_count}/{KEY_USAGE_LIMIT} 次。")

#             # --- 【修改】复杂度生成逻辑现在只对单图任务有意义 ---
#             # 对于多图任务，create_chart_variations 会直接返回原始数据，这符合我们的预期
#             chart_variations = create_chart_variations(full_chart_data, total_versions=num_versions)
#             print(f"已生成 {len(chart_variations)} 个复杂度版本的数据。")

#             generated_files = []
#             # 为每个变体生成图表
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
#                         "data_used": variant_data
#                     })
#                 else:
#                     print(f"  - 生成图表失败: {complexity_label}")

#             # 打包最终的数据集文件
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

# main.py (多线程与循环执行重构版)
import os
import json
import yaml
import argparse
import time
import queue
import threading
import concurrent.futures
from datetime import datetime
from typing import Dict, Type, List, Any

from config import GEMINI_API_KEYS
from gemini_client import GeminiClient
from schemas import SingleChartData, FacetChartData
from data_processor import create_chart_variations

# 导入所有图表生成器
from chart_generators.base_generator import BaseChartGenerator
from chart_generators.matplotlib_generator import MatplotlibGenerator
from chart_generators.seaborn_generator import SeabornGenerator

# --- 全局配置 ---
# 生成器注册表，用于根据名称查找对应的生成器类
GENERATOR_MAP: Dict[str, Type[BaseChartGenerator]] = {
    'matplotlib': MatplotlibGenerator,
    'seaborn': SeabornGenerator,
}
# 一轮任务结束后，等待多少秒再开始下一轮
LOOP_DELAY_SECONDS = 60
# 输出目录
OUTPUT_DIR = "output"


def load_tasks_from_config(config_path: str) -> list:
    """
    从YAML配置文件中加载任务列表。
    增加了文件不存在和内容为空的检查。
    """
    if not os.path.exists(config_path):
        print(f"警告：配置文件 '{config_path}' 不存在。")
        return []
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not config:
            print(f"警告：配置文件 '{config_path}' 为空。")
            return []
        tasks = config.get('tasks', [])
        if not isinstance(tasks, list):
            print(f"错误：配置文件 '{config_path}' 中的 'tasks' 应该是一个列表。")
            return []
        return tasks
    except yaml.YAMLError as e:
        print(f"错误：解析YAML文件 '{config_path}' 时出错: {e}")
        return []


def process_task(task: Dict[str, Any], client_queue: queue.Queue):
    """
    处理单个任务的完整逻辑。
    此函数将在独立的线程中执行。

    参数:
        task (Dict[str, Any]): 从 tasks.yaml 中读取的单个任务配置。
        client_queue (queue.Queue): 存储 GeminiClient 实例的线程安全队列。
    """
    thread_id = threading.get_ident()
    topic = task.get('topic', '未知主题')
    print(f"[线程 {thread_id}] 开始处理任务: '{topic}'")

    client = None
    try:
        # 1. 从队列中获取一个可用的 Gemini 客户端
        client = client_queue.get()
        # 【修复】从我们自己定义的 client.api_key 属性安全地获取密钥用于日志记录
        print(f"[线程 {thread_id}] 获取到 API Key: ...{client.api_key[-4:]}")

        # 2. 解析任务参数
        domain = task.get('domain')
        chart_type = task.get('chart_type')
        renderer = task.get('renderer', 'matplotlib').lower()
        task_type = task.get('task_type', 'single')
        
        # 根据任务类型选择对应的 Pydantic Schema
        schema_to_use = FacetChartData if task_type == 'facet' else SingleChartData
        GeneratorClass = GENERATOR_MAP.get(renderer)

        if not all([topic, domain, chart_type, GeneratorClass]):
            print(f"[线程 {thread_id}] 任务 '{topic}' 配置不完整或渲染器无效，已跳过。")
            return

        # 3. 调用 Gemini API 生成数据
        full_chart_data = client.generate_chart_data(
            topic=topic, knowledge_domain=domain, chart_type=chart_type, schema=schema_to_use
        )
        if not full_chart_data:
            print(f"[线程 {thread_id}] 未能从Gemini获取任务 '{topic}' 的有效数据，已跳过。")
            return
        
        print(f"[线程 {thread_id}] 已成功获取任务 '{topic}' 的数据，开始生成图表...")

        # 4. 创建数据变体 (如果适用)
        num_versions = task.get('complexity_steps', 3)
        chart_variations = create_chart_variations(full_chart_data, total_versions=num_versions)

        # 5. 生成图表和数据包
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_domain = "".join(c if c.isalnum() else '_' for c in domain)
        base_filename = f"{safe_domain.lower()}_{chart_type}_{renderer}_{timestamp}_{thread_id}"
        
        generated_files = []
        figsize, dpi = tuple(task.get('figsize', [12, 8])), task.get('dpi', 100)

        for idx, variant_data in enumerate(chart_variations):
            complexity_label = f"complexity_{idx}"
            generator = GeneratorClass(variant_data)
            variant_filename = f"{base_filename}_{complexity_label}.{generator.file_extension}"
            output_path = os.path.join(OUTPUT_DIR, variant_filename)
            
            if generator.create_chart(output_path, figsize=figsize, dpi=dpi):
                generated_files.append({
                    "complexity_level": idx,
                    "filename": variant_filename,
                    "data_used": variant_data
                })
            else:
                print(f"[线程 {thread_id}]  - 生成图表失败: {complexity_label}")

        # 6. 保存最终的数据集 JSON 文件
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
        
        output_json_path = os.path.join(OUTPUT_DIR, f"{base_filename}_dataset.json")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_package, f, ensure_ascii=False, indent=4)
        
        print(f"[线程 {thread_id}] 任务 '{topic}' 处理完成！数据集包已保存至: {output_json_path}")

    except Exception as e:
        print(f"[线程 {thread_id}] 处理任务 '{topic}' 时发生严重错误: {e}")
    finally:
        # 无论成功还是失败，都必须将客户端实例放回队列，以供其他线程使用
        if client:
            client_queue.put(client)
            # 【修复】同样，在这里也使用安全的 client.api_key 属性
            print(f"[线程 {thread_id}] API Key: ...{client.api_key[-4:]} 已归还队列。")


def main():
    """
    项目主执行函数。
    - 使用多线程并行处理任务。
    - 无限循环执行，每次循环重新加载配置文件。
    """
    parser = argparse.ArgumentParser(description="使用Gemini和多线程自动化循环生成图表数据集。")
    parser.add_argument('--config', type=str, default='tasks.yaml', help='指定任务定义的YAML配置文件路径。')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not GEMINI_API_KEYS or not isinstance(GEMINI_API_KEYS, list) or not GEMINI_API_KEYS[0].startswith("AIza"):
        print("错误：请在 config.py 文件中正确设置您的 GEMINI_API_KEYS 列表。")
        return

    # 初始化线程安全的客户端队列
    client_queue = queue.Queue()
    for key in GEMINI_API_KEYS:
        client_queue.put(GeminiClient(api_key=key))

    # 设置最大并发线程数等于 API Key 的数量
    max_workers = len(GEMINI_API_KEYS)
    print(f"初始化完成。检测到 {max_workers} 个API Key，将启动 {max_workers} 个并发线程。")

    # 使用 ThreadPoolExecutor 管理线程池
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    try:
        while True:
            print("\n" + "="*50)
            print(f"开始新一轮任务... (时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            print("="*50)
            
            tasks = load_tasks_from_config(args.config)
            if not tasks:
                print(f"配置文件 '{args.config}' 中没有找到任务。将在 {LOOP_DELAY_SECONDS} 秒后重试。")
                time.sleep(LOOP_DELAY_SECONDS)
                continue

            print(f"从 '{args.config}' 加载了 {len(tasks)} 个任务。开始提交到线程池...")
            
            # 将所有任务提交到线程池执行
            future_to_task = {executor.submit(process_task, task, client_queue): task for task in tasks}
            
            # 等待所有提交的任务完成
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    # 获取任务结果（如果有）。如果任务执行中抛出异常，这里会重新抛出。
                    future.result()
                except Exception as exc:
                    print(f"任务 '{task.get('topic')}' 执行时产生了一个未处理的异常: {exc}")

            print("\n" + "="*50)
            print("当前轮次所有任务已处理完毕。")
            print(f"等待 {LOOP_DELAY_SECONDS} 秒后将开始下一轮...")
            print("="*50)
            time.sleep(LOOP_DELAY_SECONDS)

    except KeyboardInterrupt:
        print("\n检测到用户中断 (Ctrl+C)。正在等待当前任务完成并优雅地关闭线程池...")
        executor.shutdown(wait=True)
        print("所有线程已安全关闭。程序退出。")
    except Exception as e:
        print(f"\n主程序发生意外错误: {e}")
        executor.shutdown(wait=False) # 发生严重错误时尝试快速关闭
        print("线程池已关闭。")


if __name__ == "__main__":
    main()
