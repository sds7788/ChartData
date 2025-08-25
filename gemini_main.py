# main.py (多线程、循环执行并集成API Key熔断机制)
import os
import json
import yaml
import argparse
import time
import queue
import threading
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, Type, List, Any
from collections import defaultdict

from gemini_config import GEMINI_API_KEYS
from gemini_client import GeminiClient
from schemas import SingleChartData, FacetChartData
from data_processor import create_chart_variations

# 导入所有图表生成器
from chart_generators.base_generator import BaseChartGenerator
from chart_generators.matplotlib_generator import MatplotlibGenerator
from chart_generators.seaborn_generator import SeabornGenerator

# --- 全局配置 ---
# 生成器注册表
GENERATOR_MAP: Dict[str, Type[BaseChartGenerator]] = {
    'matplotlib': MatplotlibGenerator,
    'seaborn': SeabornGenerator,
}
# 一轮任务结束后，等待多少秒再开始下一轮
LOOP_DELAY_SECONDS = 60
# 输出目录
OUTPUT_DIR = "gemini_output"

# --- 新增：熔断机制配置 ---
FAILURE_THRESHOLD = 5  # 5次失败触发熔断
FAILURE_WINDOW_SECONDS = 60 # 时间窗口为1分钟
QUARANTINE_SECONDS = 300000 # 触发熔断后，禁用客户端5分钟

# --- 新增：用于熔断机制的全局状态变量 (需要线程安全) ---
# 记录每个API Key的失败时间戳
api_key_failures = defaultdict(list)
# 存放被隔离的客户端，格式: { "api_key": (client, re_enable_time) }
quarantined_clients = {}
# 用于保护上述两个全局变量的线程锁
state_lock = threading.Lock()


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

def handle_client_failure(client: GeminiClient) -> bool:
    """
    处理客户端失败的逻辑。记录失败并检查是否应触发熔断。
    
    Args:
        client (GeminiClient): 发生错误的客户端实例。

    Returns:
        bool: 如果客户端被隔离，则返回 True，否则返回 False。
    """
    with state_lock:
        now = time.time()
        key = client.api_key
        
        # 1. 记录本次失败时间戳
        failures = api_key_failures[key]
        failures.append(now)
        
        # 2. 移除时间窗口之外的旧失败记录
        recent_failures = [t for t in failures if now - t < FAILURE_WINDOW_SECONDS]
        api_key_failures[key] = recent_failures
        
        # 3. 检查失败次数是否达到阈值
        if len(recent_failures) >= FAILURE_THRESHOLD:
            re_enable_time = now + QUARANTINE_SECONDS
            quarantined_clients[key] = (client, re_enable_time)
            
            # 清空失败记录，以便恢复后重新计算
            del api_key_failures[key]
            
            re_enable_dt = datetime.fromtimestamp(re_enable_time).strftime('%H:%M:%S')
            print(f"!! 熔断警告 !! API Key ...{key[-4:]} 在过去 {FAILURE_WINDOW_SECONDS} 秒内失败 {len(recent_failures)} 次。")
            print(f"   -> 该Key已被禁用，将在 {QUARANTINE_SECONDS / 60:.1f} 分钟后 (约 {re_enable_dt}) 自动恢复。")
            return True # 表示客户端已被隔离
            
    return False # 表示客户端未被隔离

def check_and_revive_clients(client_queue: queue.Queue):
    """
    检查被隔离的客户端，如果隔离时间已到，则将其恢复到活动队列。
    """
    with state_lock:
        now = time.time()
        revived_keys = []
        for key, (client, re_enable_time) in quarantined_clients.items():
            if now >= re_enable_time:
                client_queue.put(client)
                revived_keys.append(key)
                print(f"** 客户端恢复 ** API Key ...{key[-4:]} 的隔离期已结束，已重新加入工作队列。")
        
        # 从隔离区移除已恢复的客户端
        for key in revived_keys:
            del quarantined_clients[key]

def process_task(task: Dict[str, Any], client_queue: queue.Queue):
    """
    处理单个任务的完整逻辑。
    此函数将在独立的线程中执行。
    """
    thread_id = threading.get_ident()
    topic = task.get('topic', '未知主题')
    print(f"[线程 {thread_id}] 开始处理任务: '{topic}'")

    client = None
    is_quarantined = False
    try:
        # 1. 从队列中获取一个可用的 Gemini 客户端
        client = client_queue.get()
        print(f"[线程 {thread_id}] 获取到 API Key: ...{client.api_key[-4:]}")

        # 2. 解析任务参数
        domain = task.get('domain')
        chart_type = task.get('chart_type')
        renderer = task.get('renderer', 'matplotlib').lower()
        task_type = task.get('task_type', 'single')
        
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

        # 4. 创建数据变体和生成图表 (省略部分未修改的代码)
        num_versions = task.get('complexity_steps', 3)
        chart_variations = create_chart_variations(full_chart_data, total_versions=num_versions)
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
                generated_files.append({"complexity_level": idx, "filename": variant_filename, "data_used": variant_data})
            else:
                print(f"[线程 {thread_id}]  - 生成图表失败: {complexity_label}")

        # 5. 保存最终的数据集 JSON 文件
        final_package = {
            "generation_info": {**task, "timestamp": timestamp, "base_filename": base_filename},
            "qa_pair": {
                "question": full_chart_data.get('analysis', {}).get('question'),
                "answer": full_chart_data.get('analysis', {}).get('answer'),
            },
            "source_data_annotated": full_chart_data,
            "generated_charts": generated_files
        }
        output_json_path = os.path.join(OUTPUT_DIR, f"{base_filename}_dataset.json")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_package, f, ensure_ascii=False, indent=4)
        print(f"[线程 {thread_id}] 任务 '{topic}' 处理完成！数据集包已保存至: {output_json_path}")

    except Exception as e:
        # 捕获包括API调用在内的所有异常
        print(f"[线程 {thread_id}] 处理任务 '{topic}' 时发生错误: {e}")
        if client:
            # 如果获取到了客户端，则调用失败处理逻辑
            is_quarantined = handle_client_failure(client)
    finally:
        # 6. 将客户端实例放回队列或保持在隔离区
        if client and not is_quarantined:
            # 只有在客户端未被隔离的情况下，才将其归还到队列
            client_queue.put(client)
            print(f"[线程 {thread_id}] API Key: ...{client.api_key[-4:]} 已归还队列。")


def main():
    """
    项目主执行函数。
    - 使用多线程并行处理任务。
    - 无限循环执行，每次循环重新加载配置文件并检查恢复客户端。
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

    max_workers = len(GEMINI_API_KEYS)
    print(f"初始化完成。检测到 {max_workers} 个API Key，将启动 {max_workers} 个并发线程。")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    try:
        while True:
            print("\n" + "="*50)
            print(f"开始新一轮任务... (时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            print("="*50)
            
            # 在每轮任务开始前，检查并恢复被隔离的客户端
            check_and_revive_clients(client_queue)
            
            tasks = load_tasks_from_config(args.config)
            if not tasks:
                print(f"配置文件 '{args.config}' 中没有找到任务。将在 {LOOP_DELAY_SECONDS} 秒后重试。")
                time.sleep(LOOP_DELAY_SECONDS)
                continue

            print(f"从 '{args.config}' 加载了 {len(tasks)} 个任务。开始提交到线程池...")
            
            future_to_task = {executor.submit(process_task, task, client_queue): task for task in tasks}
            
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
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
        executor.shutdown(wait=False)
        print("线程池已关闭。")


if __name__ == "__main__":
    main()
