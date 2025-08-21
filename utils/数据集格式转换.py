import json
import os

# --- 配置区 ---
# 输入的原始数据集文件名
input_file = 'D:\\桌面\\ChartData\\dataset\\train_dataset.json'
# 输出的目标文件名
output_file = 'D:\\桌面\\ChartData\\dataset\\train.json'
# 生成的图片路径前缀 (根据您的目录结构，'images/' 是最合适的相对路径)
# 如果您需要截图中的绝对路径样式，可以修改为 "data/chart/my/train/images/"
IMAGE_PATH_PREFIX = 'images/' 

def convert_dataset_format(input_path, output_path, image_prefix):
    """
    将数据集从 COCO-like 格式转换为目标格式，并附带原始标注数据。

    Args:
        input_path (str): 输入的 dataset.json 文件路径。
        output_path (str): 输出的 train.json 文件路径。
        image_prefix (str): 在最终文件中为图片指定的路径前缀。
    """
    print(f"正在读取原始数据集文件: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{input_path}'。请检查路径是否正确。")
        return
    except json.JSONDecodeError:
        print(f"错误: '{input_path}' 文件格式不正确，无法解析为 JSON。")
        return

    # 1. 为了快速查找，创建一个从 image_id 到 file_name 的映射字典
    # 例如: {1: '0001.png', 2: '0002.png', ...}
    image_id_to_filename = {img['id']: img['file_name'] for img in data.get('images', [])}
    
    # 2. 准备一个列表来存储所有转换后的数据
    converted_data = []
    
    print("开始转换标注数据...")
    
    # 3. 遍历数据集中的每一个标注 (annotation)
    annotations = data.get('annotations', [])
    for ann in annotations:
        question_id = ann.get('question_id')
        image_id = ann.get('image_id')
        question_text = ann.get('question')
        
        # 假设每个问题总是有且仅有一个答案
        # 从 'answers' 列表中获取第一个答案
        answer_text = ann.get('answers', [{}])[0].get('answer')
        
        # 4. 检查所有必要信息是否存在
        if not all([question_id, image_id, question_text, answer_text]):
            print(f"警告: 跳过 question_id={question_id}，因为信息不完整。")
            continue
            
        # 5. 从映射字典中找到对应的图片文件名
        image_filename = image_id_to_filename.get(image_id)
        
        if not image_filename:
            print(f"警告: 找不到 question_id={question_id} 对应的 image_id={image_id} 的图片信息，已跳过。")
            continue
            
        # 6. 构建新的数据记录
        new_record = {
            "id": f"Chart-{question_id}",  # 按照截图样式生成ID
            "images": [
                # 拼接路径并确保使用正斜杠，以适应不同操作系统环境
                os.path.join(image_prefix, image_filename).replace("\\", "/") 
            ],
            "problem": question_text,
            "answer": answer_text,
            # 【新增部分】在这里附上原始的、未修改的标注数据
            "source_data": ann
        }
        
        # 7. 将新记录添加到结果列表中
        converted_data.append(new_record)
        
    # 8. 将转换后的数据写入新的JSON文件
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # indent=4 使JSON文件格式化，更易读
            # ensure_ascii=False 确保中文字符能正确写入，而不是被转义
            json.dump(converted_data, f, indent=4, ensure_ascii=False)
        
        print("-" * 30)
        print("转换完成！")
        print(f"总共处理了 {len(converted_data)} 条数据。")
        print(f"结果已保存到文件: {output_path}")
        print("-" * 30)
        
    except Exception as e:
        print(f"错误: 写入文件 '{output_path}' 时发生错误: {e}")

# --- 脚本入口 ---
if __name__ == '__main__':
    # 检查输入文件路径是否存在
    if not os.path.exists(input_file):
         print(f"错误: 输入文件路径不存在: '{input_file}'")
    else:
        convert_dataset_format(input_file, output_file, IMAGE_PATH_PREFIX)