# 导入所需的库
import os
import json
import shutil
from PIL import Image
from datetime import datetime
from tqdm import tqdm

def create_vqa_dataset(output_dir, images_dir, dataset_filename):
    """
    从包含两种不同结构JSON文件的目录中，创建一个统一的视觉问答（VQA）数据集。
    此版本新增功能：
    1. 将每个条目对应的 `source_data`, `task_type`, 和 `renderer` 完整地打包到最终数据集中。
    2. 如果 `task_type` 字段缺失，则默认为 'single'。
    """
    # --- 1. 初始化和环境设置 ---
    dataset_root_dir = os.path.dirname(dataset_filename)
    if not os.path.exists(dataset_root_dir):
        os.makedirs(dataset_root_dir)
        
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        print(f"目录 '{images_dir}' 已创建。")

    current_date = datetime.now().strftime("%Y/%m/%d")
    current_year = datetime.now().year

    # 初始化最终数据集的结构
    dataset = {
        "info": {
            "description": "多复杂度与多来源图表VQA数据集 (VQA Dataset with Multi-complexity and Multi-source Charts)",
            "version": "3.4", # 版本更新，体现了 task_type 的默认值处理
            "year": current_year,
            "contributor": "Gemini Chart Automator & AI Annotator User",
            "date_created": current_date
        },
        "images": [],
        "annotations": []
    }

    image_id_counter = 0
    question_id_counter = 0 
    
    # --- 2. 遍历源文件并处理数据 ---
    print(f"正在从 '{output_dir}' 目录读取所有JSON文件...")
    
    files_to_process = [f for f in sorted(os.listdir(output_dir)) if f.endswith('.json')]
    
    if not files_to_process:
        print(f"警告：在 '{output_dir}' 中没有找到任何 '.json' 文件。")
        return

    for filename in tqdm(files_to_process, desc="打包数据集"):
        json_path = os.path.join(output_dir, filename)

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # --- 核心修改：兼容两种JSON结构的逻辑判断 ---

            # **分支1：处理“数据集包”类型的JSON**
            if 'generated_charts' in data and 'qa_pair' in data:
                qa_pair = data.get('qa_pair', {})
                question = qa_pair.get('question')
                answer = qa_pair.get('answer')
                
                # 从 generation_info 中提取元数据
                generation_info = data.get('generation_info', {})
                chart_type = generation_info.get('chart_type')
                
                # #################### 完善点 1 ####################
                # 如果 task_type 不存在，则默认为 'single'
                task_type = generation_info.get('task_type', 'single') 
                
                renderer = generation_info.get('renderer') # 提取 renderer

                generated_charts = data.get('generated_charts', [])
                
                # 从JSON的顶层提取 source_data
                source_data = data.get('source_data_annotated', {}) 
                
                if not all([question, answer, chart_type, generated_charts]):
                    print(f"\n警告: 数据集包 {filename} 中缺少关键信息，已跳过。")
                    continue

                sorted_charts = sorted(generated_charts, key=lambda x: x.get('complexity_level', 0))

                for chart_info in sorted_charts:
                    original_filename_in_json = chart_info.get('filename')
                    complexity_level = chart_info.get('complexity_level', -1)

                    if original_filename_in_json:
                        if original_filename_in_json.endswith('.html'):
                            target_filename = original_filename_in_json.replace('.html', '.png')
                        else:
                            target_filename = original_filename_in_json
                        
                        image_to_find_path = os.path.join(output_dir, target_filename)

                        if os.path.exists(image_to_find_path):
                            image_id_counter += 1
                            new_image_filename = f"{image_id_counter:05d}.png"
                            new_image_path = os.path.join(images_dir, new_image_filename)
                            shutil.copy2(image_to_find_path, new_image_path)

                            with Image.open(new_image_path) as img:
                                width, height = img.size

                            # 在 images 条目中添加 renderer 和 task_type 字段
                            dataset['images'].append({
                                "id": image_id_counter, "file_name": new_image_filename,
                                "height": height, "width": width, "chart_type": chart_type,
                                "complexity_level": complexity_level,
                                "task_type": task_type,
                                "renderer": renderer
                            })

                            question_id_counter += 1
                            
                            # 在 annotations 中添加 source_data 字段
                            dataset['annotations'].append({
                                "question_id": question_id_counter, 
                                "image_id": image_id_counter,
                                "question": question, 
                                "answers": [{"answer": str(answer)}],
                                "source_data": source_data
                            })
                        else:
                            print(f"\n警告: 找不到图片文件 {image_to_find_path}，已跳过。")
            
            # **分支2：处理“单一图表实体”类型的JSON**
            elif 'image_filename' in data and 'question' in data:
                question = data.get('question')
                answer = data.get('answer')
                original_image_filename = data.get('image_filename')
                
                # 同样尝试从 generation_info (如果存在) 或 source_data 中获取信息
                generation_info = data.get('generation_info', {})
                source_data = data.get('source_data', {})
                
                chart_type = generation_info.get('chart_type') or source_data.get('chart_type')
                
                # #################### 完善点 2 ####################
                # 如果 task_type 不存在，则默认为 'single'
                task_type = generation_info.get('task_type', 'single')
                
                renderer = generation_info.get('renderer') # 提取 renderer

                if not all([question, answer, chart_type, original_image_filename]):
                    print(f"\n警告: 单一实体 {filename} 中缺少关键信息，已跳过。")
                    continue
                
                complexity_level = 0 
                image_to_find_path = os.path.join(output_dir, original_image_filename)
                
                if os.path.exists(image_to_find_path):
                    image_id_counter += 1
                    new_image_filename = f"{image_id_counter:05d}.png"
                    new_image_path = os.path.join(images_dir, new_image_filename)
                    shutil.copy2(image_to_find_path, new_image_path)

                    with Image.open(new_image_path) as img:
                        width, height = img.size

                    # 在 images 条目中添加 renderer 和 task_type 字段
                    dataset['images'].append({
                        "id": image_id_counter, "file_name": new_image_filename,
                        "height": height, "width": width, "chart_type": chart_type,
                        "complexity_level": complexity_level,
                        "task_type": task_type,
                        "renderer": renderer
                    })

                    question_id_counter += 1
                    
                    # 在 annotations 中添加 source_data 字段
                    dataset['annotations'].append({
                        "question_id": question_id_counter, 
                        "image_id": image_id_counter,
                        "question": question, 
                        "answers": [{"answer": str(answer)}],
                        "source_data": source_data
                    })
                else:
                    print(f"\n警告: 找不到图片文件 {image_to_find_path}，已跳过。")

            # **分支3：处理未知结构的JSON文件**
            else:
                print(f"\n警告: 文件 {filename} 的结构无法识别，已跳过。")

        except json.JSONDecodeError:
            print(f"\n错误: 无法解析JSON文件 {filename}，已跳过。")
        except Exception as e:
            print(f"\n处理文件 {filename} 时发生未知错误: {e}")

    # --- 5. 保存最终的数据集文件 ---
    with open(dataset_filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print("-" * 30)
    print("数据集创建成功！")
    print(f"总共处理了 {image_id_counter} 张图片和 {len(dataset['annotations'])} 个问答对。")
    print(f"图片已保存到 '{images_dir}' 目录。")
    print(f"数据集信息已保存到 '{dataset_filename}' 文件。")


if __name__ == '__main__':
    # --- 请根据你的实际情况修改这里的路径 ---
    # 提示：在Windows路径中使用双反斜杠'\\'或正斜杠'/'可以避免转义字符问题
    create_vqa_dataset(
        output_dir='D:\\桌面\\数据存储\\dataset(1,20k)\\output', 
        images_dir='D:\\桌面\\数据存储\\dataset(1,20k)\\images', 
        dataset_filename='D:\\桌面\\数据存储\\dataset(1,20k)\\dataset.json'
    )