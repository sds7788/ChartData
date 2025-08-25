# 导入所需的库
import json
import random
import os

# ==================== 在这里修改您的配置 ====================

# 1. 输入的原始 dataset.json 文件完整路径
# 提示: Windows路径建议在引号前加上 r，例如 r'C:\Users\YourName\Desktop\dataset.json'
INPUT_FILE_PATH = r'D:\\桌面\\数据存储\dataset(2,20k)\dataset.json'

# 2. 用来存放抽样结果文件的目录路径
OUTPUT_DIRECTORY = r'D:\\桌面\\数据存储\\dataset(2,20k)\samples'

# 3. 希望抽样的问答对数量
SAMPLING_SIZE = 2000

# ==========================================================


def sample_vqa_dataset(input_path, output_path, sampling_size):
    """
    从VQA数据集中抽样指定的问答对数量，并生成新的数据集文件。
    """
    try:
        # --- 1. 读取原始JSON文件 ---
        print(f"正在从 '{input_path}' 读取数据...")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # --- 2. 随机抽样 'annotations' ---
        annotations_total = len(data['annotations'])
        if sampling_size > annotations_total:
            print(f"警告: 抽样数量 ({sampling_size}) 大于总问答对数量 ({annotations_total})。")
            print("将使用所有可用的问答对。")
            sampled_annotations = data['annotations']
            # 更新实际的抽样数量
            sampling_size = annotations_total
        else:
            sampled_annotations = random.sample(data['annotations'], sampling_size)

        print(f"成功抽取 {len(sampled_annotations)} 个问答对。")

        # --- 3. 筛选相关的图像信息 ---
        # 提取抽样问答对对应的 image_id，使用集合(set)确保id唯一性
        sampled_image_ids = {anno['image_id'] for anno in sampled_annotations}

        # 根据抽样出的 image_id 筛选出相关的图像信息
        sampled_images = [image for image in data['images'] if image['id'] in sampled_image_ids]
        print(f"已关联到 {len(sampled_images)} 个唯一的图像。")

        # --- 4. 构建并写入新的JSON数据 ---
        new_data = {
            "info": data['info'],
            "images": sampled_images,
            "annotations": sampled_annotations
        }
        
        # 确保输出目录存在
        if OUTPUT_DIRECTORY:
            os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

        # 将新数据写入到输出JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)

        print("-" * 30)
        print("成功！")
        print(f"新的抽样数据集已保存到: '{output_path}'")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_path}' 未找到。请确保文件路径正确。")
    except json.JSONDecodeError:
        print(f"错误: 文件 '{input_path}' 不是有效的JSON格式。")
    except Exception as e:
        print(f"发生了一个未知错误: {e}")

# --- 主程序 ---
if __name__ == "__main__":
    # 根据顶部的配置构建完整的输出文件名
    # os.path.join 会自动使用正确的路径分隔符（\ 或 /）
    output_filename = os.path.join(OUTPUT_DIRECTORY, f'sampled_dataset_{SAMPLING_SIZE}.json')

    # 调用核心函数执行抽样任务
    sample_vqa_dataset(INPUT_FILE_PATH, output_filename, SAMPLING_SIZE)