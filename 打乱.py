# 导入所需的库
import yaml
import random

# 定义要操作的文件名
filename = 'tasks.yaml'

try:
    # --- 第1步: 读取YAML文件 ---
    # 使用 'r' 模式打开文件进行读取
    with open(filename, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    # 检查文件内容是否有效
    if data and 'tasks' in data and isinstance(data['tasks'], list):
        tasks_list = data['tasks']

        # --- 第2步: 随机打乱列表顺序 ---
        random.shuffle(tasks_list)
        print(f"成功读取并打乱了 {len(tasks_list)} 个任务。")

        # --- 第3步: 将打乱后的数据写回原文件 ---
        # 使用 'w' 模式重新打开同一个文件，这会清空文件内容并准备写入
        with open(filename, 'w', encoding='utf-8') as file:
            # 将修改后的数据（包含已打乱的列表）写回文件
            yaml.dump(data, file, allow_unicode=True, sort_keys=False, indent=2)
        
        print(f"任务已在原文件 '{filename}' 中成功随机分散并保存。")
    
    else:
        print(f"错误：在文件 '{filename}' 中没有找到有效的 'tasks' 列表。")

except FileNotFoundError:
    print(f"错误：找不到文件 '{filename}'。请确保该文件和脚本在同一个目录下。")
except Exception as e:
    print(f"处理过程中发生了一个错误: {e}")