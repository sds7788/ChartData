import json
from collections import Counter

def count_chart_types_from_json(json_filepath='dataset.json'):
    """
    从指定的JSON数据集中读取并统计每种图表类型的数量。

    Args:
        json_filepath (str): 数据集JSON文件的路径。

    Returns:
        tuple: 包含图表类型计数器和总图片数的元组 (Counter, int)。
               如果文件未找到或格式错误，则返回 (None, 0)。
    """
    try:
        # 打开并加载JSON文件
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查'images'键是否存在且为列表
        if 'images' not in data or not isinstance(data['images'], list):
            print(f"错误: '{json_filepath}' 文件中没有找到有效的 'images' 列表。")
            return None, 0
            
        # 提取所有 'chart_type' 的值到一个列表中
        # 使用 .get('chart_type', 'Unknown') 来避免因缺少键而出错
        chart_types = [image.get('chart_type', 'Unknown') for image in data['images']]
        
        # 使用Counter来高效地统计每种类型的数量
        type_counts = Counter(chart_types)
        
        total_images = len(data['images'])
        
        return type_counts, total_images

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{json_filepath}'。请确保脚本和数据集文件在同一个目录下。")
        return None, 0
    except json.JSONDecodeError:
        print(f"错误: 文件 '{json_filepath}' 不是一个有效的JSON文件。")
        return None, 0
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None, 0

if __name__ == "__main__":
    # 调用函数来统计图表类型
    counts, total = count_chart_types_from_json()
    
    # 如果成功获取了统计数据，则打印结果
    if counts:
        print("--- 图表类型数量统计 ---")
        # 按照数量从多到少排序并打印
        for chart_type, count in counts.most_common():
            print(f"{chart_type:<25}: {count} 个")
        
        print("-" * 30)
        print(f"{'总计':<25}: {total} 张图片")
