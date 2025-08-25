# 导入所需的库
import json
from collections import Counter

def analyze_vqa_dataset(dataset_filename):
    """
    分析VQA数据集文件 (dataset.json)，并提供多维度的统计信息。
    此版本修复了因数据中存在 null 值而导致的排序 TypeError。

    Args:
        dataset_filename (str): 指向 dataset.json 文件的路径。
    """
    print(f"--- 正在分析数据集文件: {dataset_filename} ---")

    # --- 1. 加载数据并进行基础校验 ---
    try:
        with open(dataset_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"\n[错误] 文件未找到: '{dataset_filename}'")
        print("请检查文件路径是否正确。")
        return
    except json.JSONDecodeError:
        print(f"\n[错误] JSON格式错误: '{dataset_filename}' 不是一个有效的JSON文件。")
        return
    except Exception as e:
        print(f"\n[错误] 读取文件时发生未知错误: {e}")
        return

    images_data = data.get('images', [])
    annotations_data = data.get('annotations', [])

    num_images = len(images_data)
    num_annotations = len(annotations_data)

    print("\n--- 1. 数据完整性检查 ---")
    print(f"总图片数量: {num_images}")
    print(f"总问答对数量: {num_annotations}")

    if num_images != num_annotations:
        print("\n[警告] 图片数量与问答对数量不匹配！请重新检查数据提取过程。")
    else:
        print("\n[成功] 数据完整，图片与问答对数量 1:1 匹配。")

    if not images_data:
        print("\n数据集中没有图片信息，无法进行后续分析。")
        return

    # --- 2. 提取统计维度 (已修复) ---
    
    # 核心修复点：使用 `(img.get('key') or '未知')` 的方式
    # 这种写法可以同时处理 “键不存在” 和 “键存在但值为null” 这两种情况
    chart_types = [(img.get('chart_type') or '未知') for img in images_data]
    renderers = [(img.get('renderer') or '未知') for img in images_data]
    task_types = [(img.get('task_type') or '未知') for img in images_data]

    # --- 3. 计算各项统计数据 ---
    chart_type_counts = Counter(chart_types)
    renderer_counts = Counter(renderers)
    task_type_counts = Counter(task_types)

    # --- 4. 格式化并打印结果 ---
    
    def print_stats_table(title, data_counter):
        """一个漂亮地打印统计结果的辅助函数"""
        print(f"\n--- {title} ---")
        if not data_counter:
            print("无相关数据。")
            return
            
        max_key_len = max(len(str(key)) for key in data_counter.keys()) if data_counter else 10
        
        print(f"{'类别':<{max_key_len}} | {'数量':>8} | {'百分比':>8}")
        print(f"{'-' * max_key_len} | {'-' * 8} | {'-' * 8}")
        
        total_items = sum(data_counter.values())
        
        # 现在排序不会再出错了，因为所有键都是字符串
        for item, count in sorted(data_counter.items()):
            percentage = (count / total_items) * 100 if total_items > 0 else 0
            print(f"{str(item):<{max_key_len}} | {count:>8} | {percentage:7.2f}%")
        
        print(f"{'-' * max_key_len} | {'-' * 8} | {'-' * 8}")
        print(f"{'总计':<{max_key_len}} | {total_items:>8} | {100.00:7.2f}%")

    print_stats_table("2. 图表类型 (Chart Type) 统计", chart_type_counts)
    print_stats_table("3. 渲染器 (Renderer) 统计", renderer_counts)
    print_stats_table("4. 任务类型 (Task Type) 统计", task_type_counts)
    
    print("\n--- 分析完成 ---")


if __name__ == '__main__':
    # --- 请在这里修改为您的数据集文件路径 ---
    dataset_file_path = 'D:\\桌面\\数据存储\\dataset(2,20k)\\dataset.json'  # 示例路径

    analyze_vqa_dataset(dataset_file_path)