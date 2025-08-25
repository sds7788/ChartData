import yaml

# --- 自定义格式化规则 ---
# 这个函数告诉PyYAML如何处理列表（list）
def flow_style_list_representer(dumper, data):
    """
    为短的、只包含数字的列表（比如 figsize）使用流格式 [item1, item2]。
    对于其他列表，使用默认的块格式。
    """
    # 条件：列表长度小于等于2，并且所有元素都是数字（整数或浮点数）
    if len(data) <= 2 and all(isinstance(item, (int, float)) for item in data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    # 否则，使用默认的多行块格式
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

# 将我们的自定义规则应用到PyYAML的Dumper中
# 这样，每次遇到列表时，它都会使用上面的函数来决定格式
yaml.add_representer(list, flow_style_list_representer, Dumper=yaml.SafeDumper)


def split_tasks_with_correct_style(input_filename='t.yaml'):
    """
    读取YAML文件，按'task_type'分类任务，并确保输出正确的figsize格式。
    """
    try:
        # 读取源文件
        with open(input_filename, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        if 'tasks' not in data or not isinstance(data['tasks'], list):
            print("错误: YAML文件格式不正确，未找到'tasks'列表。")
            return

        # 初始化任务列表
        single_tasks = []
        facet_tasks = []

        # 分类任务
        for task in data['tasks']:
            if 'task_type' in task:
                if task['task_type'] == 'single':
                    single_tasks.append(task)
                elif task['task_type'] == 'facet':
                    facet_tasks.append(task)
        
        # 定义输出文件名
        output_single_filename = 'single_tasks_correct_style.yaml'
        output_facet_filename = 'facet_tasks_correct_style.yaml'

        # 将'single'任务写入新文件
        with open(output_single_filename, 'w', encoding='utf-8') as file:
            # 使用配置了自定义规则的SafeDumper来写入文件
            yaml.dump(
                {'tasks': single_tasks}, 
                file, 
                Dumper=yaml.SafeDumper, 
                allow_unicode=True, 
                sort_keys=False,
                indent=2 # 设置缩进，让文件更美观
            )
        
        print(f"成功创建文件: {output_single_filename}，包含 {len(single_tasks)} 个任务。")

        # 将'facet'任务写入新文件
        with open(output_facet_filename, 'w', encoding='utf-8') as file:
            yaml.dump(
                {'tasks': facet_tasks}, 
                file, 
                Dumper=yaml.SafeDumper, 
                allow_unicode=True, 
                sort_keys=False,
                indent=2
            )

        print(f"成功创建文件: {output_facet_filename}，包含 {len(facet_tasks)} 个任务。")

    except FileNotFoundError:
        print(f"错误: 文件 '{input_filename}' 未找到。")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

# --- 运行脚本 ---
if __name__ == '__main__':
    split_tasks_with_correct_style()