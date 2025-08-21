# import os
# import json
# import shutil
# from PIL import Image
# from datetime import datetime

# def create_vqa_dataset(output_dir='output', images_dir='images', dataset_filename='dataset.json'):
#     """
#     从包含成对的 JSON 和 PNG 文件的目录中创建一个视觉问答（VQA）数据集。

#     Args:
#         output_dir (str): 包含源 JSON 和 PNG 文件的目录。
#         images_dir (str): 用于存放重命名后的图片的目录。
#         dataset_filename (str): 输出的数据集 JSON 文件的名称。
#     """
#     # --- 1. 初始化和环境设置 ---
#     # 如果图片输出目录不存在，则创建它
#     if not os.path.exists(images_dir):
#         os.makedirs(images_dir)
#         print(f"目录 '{images_dir}' 已创建。")

#     # 获取当前日期，用于填充数据集信息
#     current_date = datetime.now().strftime("%Y/%m/%d")
#     current_year = datetime.now().year

#     # 初始化最终数据集的结构
#     dataset = {
#         "info": {
#             "description": "基准数据集 (Benchmark Dataset)",
#             "version": "1.0",
#             "year": current_year,
#             "contributor": "AI Annotator User",
#             "date_created": current_date
#         },
#         "images": [],
#         "annotations": []
#     }

#     # 初始化图片和问题的ID计数器
#     image_id_counter = 0
#     question_id_counter = 0  
    
#     # --- 2. 遍历源文件并处理数据 ---
#     print(f"正在从 '{output_dir}' 目录读取文件...")
    
#     # 获取所有文件名并进行排序，以确保处理顺序一致
#     files = sorted(os.listdir(output_dir))

#     for filename in files:
#         # 只处理 .json 文件
#         if filename.endswith('.json'):
#             json_path = os.path.join(output_dir, filename)

#             try:
#                 with open(json_path, 'r', encoding='utf-8') as f:
#                     data = json.load(f)

#                 # --- 提取问答对、图表类型和图片信息 ---
#                 chart_type = data.get('source_data', {}).get('chart_type') or data.get('generation_info', {}).get('chart_type')
                
#                 if not chart_type:
#                     print(f"警告: 在 {filename} 中未找到 'chart_type'，已跳过。")
#                     continue

#                 if 'source_data' in data and 'analysis' in data['source_data']:
#                     analysis_info = data['source_data']['analysis']
#                     question = analysis_info.get('question')
#                     answer = analysis_info.get('answer')
#                 elif 'question' in data and 'answer' in data:
#                     question = data.get('question')
#                     answer = data.get('answer')
#                 else:
#                     print(f"警告: 在 {filename} 中未找到有效的问答信息，已跳过。")
#                     continue
                
#                 if not question or not answer:
#                     print(f"警告: 在 {filename} 中问题或答案为空，已跳过。")
#                     continue

#                 original_image_filename = data.get('image_filename')
#                 if not original_image_filename:
#                     print(f"警告: 在 {filename} 中未找到 'image_filename'，已跳过。")
#                     continue

#                 original_image_path = os.path.join(output_dir, original_image_filename)

#                 # --- 3. 处理图片文件 ---
#                 if os.path.exists(original_image_path):
#                     image_id_counter += 1
                    
#                     new_image_filename = f"{image_id_counter:04d}.png"
#                     new_image_path = os.path.join(images_dir, new_image_filename)

#                     shutil.copy2(original_image_path, new_image_path)

#                     with Image.open(new_image_path) as img:
#                         width, height = img.size

#                     # --- 4. 构造数据集条目 ---
#                     # 添加图片信息，包含图表类型
#                     dataset['images'].append({
#                         "id": image_id_counter,
#                         "file_name": new_image_filename,
#                         "height": height,
#                         "width": width,
#                         "chart_type": chart_type  # 新增字段
#                     })

#                     question_id_counter += 1
                    
#                     # 添加标注（问答）信息，移除 answer_type
#                     dataset['annotations'].append({
#                         "question_id": question_id_counter,
#                         "image_id": image_id_counter,
#                         "question": question,
#                         "answers": [{"answer": answer}]
#                     })
#                 else:
#                     print(f"警告: 找不到图片文件 {original_image_path}，已跳过 {filename}。")

#             except json.JSONDecodeError:
#                 print(f"错误: 无法解析JSON文件 {filename}，已跳过。")
#             except Exception as e:
#                 print(f"处理文件 {filename} 时发生未知错误: {e}")

#     # --- 5. 保存最终的数据集文件 ---
#     with open(dataset_filename, 'w', encoding='utf-8') as f:
#         json.dump(dataset, f, indent=2, ensure_ascii=False)

#     print("-" * 30)
#     print("数据集创建成功！")
#     print(f"总共处理了 {image_id_counter} 张图片和 {len(dataset['annotations'])} 个问答对。")
#     print(f"图片已保存到 '{images_dir}' 目录。")
#     print(f"数据集信息已保存到 '{dataset_filename}' 文件。")


# if __name__ == '__main__':
#     # 假设 'output' 文件夹与此脚本位于同一目录下
#     create_vqa_dataset()

# 导入所需的库
# import os
# import json
# import shutil
# from PIL import Image
# from datetime import datetime
# from tqdm import tqdm

# def create_vqa_dataset(output_dir='output', images_dir='dataset/images', dataset_filename='dataset/dataset.json'):
#     """
#     从包含数据集包（一个JSON对应多张图）的目录中创建一个视觉问答（VQA）数据集。
#     新版逻辑：智能处理 Pyecharts 生成的、已被转换为 PNG 的图表。
#     """
#     # --- 1. 初始化和环境设置 ---
#     # 确保数据集的根目录存在
#     dataset_root_dir = os.path.dirname(dataset_filename)
#     if not os.path.exists(dataset_root_dir):
#         os.makedirs(dataset_root_dir)
        
#     # 确保存放图片的目录存在
#     if not os.path.exists(images_dir):
#         os.makedirs(images_dir)
#         print(f"目录 '{images_dir}' 已创建。")

#     # 获取当前日期和年份，用于填充数据集信息
#     current_date = datetime.now().strftime("%Y/%m/%d")
#     current_year = datetime.now().year

#     # 初始化最终数据集的结构
#     dataset = {
#         "info": {
#             "description": "多复杂度图表VQA数据集 (Multi-complexity Chart VQA Dataset)",
#             "version": "2.3", # 版本更新
#             "year": current_year,
#             "contributor": "Gemini Chart Automator",
#             "date_created": current_date
#         },
#         "images": [],
#         "annotations": []
#     }

#     # 初始化图片和问题的ID计数器
#     image_id_counter = 0
#     question_id_counter = 0 
    
#     # --- 2. 遍历源文件并处理数据 ---
#     print(f"正在从 '{output_dir}' 目录读取数据集文件...")
    
#     # --- 核心修改点 ---
#     # 修改筛选逻辑，从只读取 '_dataset.json' 文件，变为读取所有以 '.json' 结尾的文件
#     files_to_process = [f for f in sorted(os.listdir(output_dir)) if f.endswith('.json')]
    
#     # 检查是否找到了任何json文件
#     if not files_to_process:
#         print(f"警告：在 '{output_dir}' 中没有找到任何 '.json' 文件。请检查目录或文件名。")
#         return

#     # 使用tqdm创建进度条，遍历所有找到的json文件
#     for filename in tqdm(files_to_process, desc="打包数据集"):
#         json_path = os.path.join(output_dir, filename)

#         try:
#             # 打开并加载JSON文件内容
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)

#             # 提取问答对、图表类型和生成的图表列表
#             qa_pair = data.get('qa_pair', {})
#             question = qa_pair.get('question')
#             answer = qa_pair.get('answer')
#             chart_type = data.get('generation_info', {}).get('chart_type')
#             generated_charts = data.get('generated_charts', [])
            
#             # 检查关键信息是否存在，如果任一缺失则跳过此文件
#             if not all([question, answer, chart_type, generated_charts]):
#                 print(f"\n警告: 在 {filename} 中缺少关键信息（问答对、图表类型或图表列表），已跳过。")
#                 continue

#             # 按复杂度级别对图表进行排序
#             sorted_charts = sorted(generated_charts, key=lambda x: x.get('complexity_level', 0))

#             # 遍历排序后的每个图表信息
#             for chart_info in sorted_charts:
#                 original_filename_in_json = chart_info.get('filename')
#                 complexity_level = chart_info.get('complexity_level', -1)

#                 if not original_filename_in_json:
#                     print(f"\n警告: 在 {filename} 的图表列表中发现无文件名的条目，已跳过。")
#                     continue

#                 # --- 智能判断图片文件名 ---
#                 # 如果记录的文件名是 .html，则目标图片文件是同名的 .png 文件
#                 if original_filename_in_json.endswith('.html'):
#                     target_filename = original_filename_in_json.replace('.html', '.png')
#                 else:
#                     # 否则，直接使用记录中的文件名（可能本身就是 .png）
#                     target_filename = original_filename_in_json
                
#                 # 构建源图片文件的完整路径
#                 image_to_find_path = os.path.join(output_dir, target_filename)

#                 # 检查图片文件是否存在
#                 if os.path.exists(image_to_find_path):
#                     image_id_counter += 1
#                     # 创建新的、按顺序编号的图片文件名
#                     new_image_filename = f"{image_id_counter:05d}.png"
#                     new_image_path = os.path.join(images_dir, new_image_filename)

#                     # 复制并重命名图片到目标目录
#                     shutil.copy2(image_to_find_path, new_image_path)

#                     # 打开新图片以获取其尺寸
#                     with Image.open(new_image_path) as img:
#                         width, height = img.size

#                     # 将图片信息添加到数据集中
#                     dataset['images'].append({
#                         "id": image_id_counter, "file_name": new_image_filename,
#                         "height": height, "width": width, "chart_type": chart_type,
#                         "complexity_level": complexity_level
#                     })

#                     question_id_counter += 1
#                     # 将对应的问答标注信息添加到数据集中
#                     dataset['annotations'].append({
#                         "question_id": question_id_counter, "image_id": image_id_counter,
#                         "question": question, "answers": [{"answer": str(answer)}] # 确保答案是字符串
#                     })
#                 else:
#                     # 如果找不到对应的图片文件，则打印警告
#                     print(f"\n警告: 找不到图片文件 {image_to_find_path}，已跳过。")

#         except json.JSONDecodeError:
#             print(f"\n错误: 无法解析JSON文件 {filename}，文件可能已损坏，已跳过。")
#         except Exception as e:
#             print(f"\n处理文件 {filename} 时发生未知错误: {e}")

#     # --- 5. 保存最终的数据集文件 ---
#     # 将构建好的数据集以JSON格式写入文件
#     with open(dataset_filename, 'w', encoding='utf-8') as f:
#         json.dump(dataset, f, indent=2, ensure_ascii=False)

#     # 打印最终的总结信息
#     print("-" * 30)
#     print("数据集创建成功！")
#     print(f"总共处理了 {image_id_counter} 张图片和 {len(dataset['annotations'])} 个问答对。")
#     print(f"图片已保存到 '{images_dir}' 目录。")
#     print(f"数据集信息已保存到 '{dataset_filename}' 文件。")


# if __name__ == '__main__':
#     # 调用主函数，指定输入输出路径
#     create_vqa_dataset(
#         output_dir='output', 
#         images_dir='dataset/images', 
#         dataset_filename='dataset/dataset.json'
#     )

# 导入所需的库
# import os
# import json
# import shutil
# from PIL import Image
# from datetime import datetime
# from tqdm import tqdm

# def create_vqa_dataset(output_dir='output', images_dir='dataset/images', dataset_filename='dataset/dataset.json'):
#     """
#     从包含两种不同结构JSON文件的目录中，创建一个统一的视觉问答（VQA）数据集。
#     """
#     # --- 1. 初始化和环境设置 ---
#     dataset_root_dir = os.path.dirname(dataset_filename)
#     if not os.path.exists(dataset_root_dir):
#         os.makedirs(dataset_root_dir)
        
#     if not os.path.exists(images_dir):
#         os.makedirs(images_dir)
#         print(f"目录 '{images_dir}' 已创建。")

#     current_date = datetime.now().strftime("%Y/%m/%d")
#     current_year = datetime.now().year

#     # 初始化最终数据集的结构
#     dataset = {
#         "info": {
#             "description": "多复杂度与多来源图表VQA数据集 (VQA Dataset with Multi-complexity and Multi-source Charts)",
#             "version": "3.0", # 版本更新，体现兼容性
#             "year": current_year,
#             "contributor": "Gemini Chart Automator & AI Annotator User",
#             "date_created": current_date
#         },
#         "images": [],
#         "annotations": []
#     }

#     image_id_counter = 0
#     question_id_counter = 0 
    
#     # --- 2. 遍历源文件并处理数据 ---
#     print(f"正在从 '{output_dir}' 目录读取所有JSON文件...")
    
#     files_to_process = [f for f in sorted(os.listdir(output_dir)) if f.endswith('.json')]
    
#     if not files_to_process:
#         print(f"警告：在 '{output_dir}' 中没有找到任何 '.json' 文件。")
#         return

#     for filename in tqdm(files_to_process, desc="打包数据集"):
#         json_path = os.path.join(output_dir, filename)

#         try:
#             with open(json_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)

#             # --- 核心修改：兼容两种JSON结构的逻辑判断 ---

#             # **分支1：处理“数据集包”类型的JSON (如 agriculture_line_..._dataset.json)**
#             if 'generated_charts' in data and 'qa_pair' in data:
#                 qa_pair = data.get('qa_pair', {})
#                 question = qa_pair.get('question')
#                 answer = qa_pair.get('answer')
#                 chart_type = data.get('generation_info', {}).get('chart_type')
#                 generated_charts = data.get('generated_charts', [])
                
#                 if not all([question, answer, chart_type, generated_charts]):
#                     print(f"\n警告: 数据集包 {filename} 中缺少关键信息，已跳过。")
#                     continue

#                 sorted_charts = sorted(generated_charts, key=lambda x: x.get('complexity_level', 0))

#                 for chart_info in sorted_charts:
#                     original_filename_in_json = chart_info.get('filename')
#                     complexity_level = chart_info.get('complexity_level', -1)

#                     if original_filename_in_json:
#                         # (此部分逻辑与您之前的脚本相同)
#                         if original_filename_in_json.endswith('.html'):
#                             target_filename = original_filename_in_json.replace('.html', '.png')
#                         else:
#                             target_filename = original_filename_in_json
                        
#                         image_to_find_path = os.path.join(output_dir, target_filename)

#                         # (图片处理和数据添加的逻辑是共享的，下面会看到)
#                         if os.path.exists(image_to_find_path):
#                             image_id_counter += 1
#                             new_image_filename = f"{image_id_counter:05d}.png"
#                             new_image_path = os.path.join(images_dir, new_image_filename)
#                             shutil.copy2(image_to_find_path, new_image_path)

#                             with Image.open(new_image_path) as img:
#                                 width, height = img.size

#                             dataset['images'].append({
#                                 "id": image_id_counter, "file_name": new_image_filename,
#                                 "height": height, "width": width, "chart_type": chart_type,
#                                 "complexity_level": complexity_level
#                             })

#                             question_id_counter += 1
#                             dataset['annotations'].append({
#                                 "question_id": question_id_counter, "image_id": image_id_counter,
#                                 "question": question, "answers": [{"answer": str(answer)}]
#                             })
#                         else:
#                             print(f"\n警告: 找不到图片文件 {image_to_find_path}，已跳过。")
            
#             # **分支2：处理“单一图表实体”类型的JSON (如 advertising_..._stacked_bar_...json)**
#             elif 'image_filename' in data and 'question' in data:
#                 question = data.get('question')
#                 answer = data.get('answer')
#                 # 注意chart_type在不同的位置
#                 chart_type = data.get('source_data', {}).get('chart_type') 
#                 original_image_filename = data.get('image_filename')
                
#                 if not all([question, answer, chart_type, original_image_filename]):
#                     print(f"\n警告: 单一实体 {filename} 中缺少关键信息，已跳过。")
#                     continue
                
#                 # 由于是单一实体，没有复杂度级别，我们给一个默认值
#                 complexity_level = 0 
#                 image_to_find_path = os.path.join(output_dir, original_image_filename)
                
#                 # (这里的图片处理和数据添加逻辑与分支1完全一致)
#                 if os.path.exists(image_to_find_path):
#                     image_id_counter += 1
#                     new_image_filename = f"{image_id_counter:05d}.png"
#                     new_image_path = os.path.join(images_dir, new_image_filename)
#                     shutil.copy2(image_to_find_path, new_image_path)

#                     with Image.open(new_image_path) as img:
#                         width, height = img.size

#                     dataset['images'].append({
#                         "id": image_id_counter, "file_name": new_image_filename,
#                         "height": height, "width": width, "chart_type": chart_type,
#                         "complexity_level": complexity_level # 使用默认值
#                     })

#                     question_id_counter += 1
#                     dataset['annotations'].append({
#                         "question_id": question_id_counter, "image_id": image_id_counter,
#                         "question": question, "answers": [{"answer": str(answer)}]
#                     })
#                 else:
#                     print(f"\n警告: 找不到图片文件 {image_to_find_path}，已跳过。")

#             # **分支3：处理未知结构的JSON文件**
#             else:
#                 print(f"\n警告: 文件 {filename} 的结构无法识别，已跳过。")

#         except json.JSONDecodeError:
#             print(f"\n错误: 无法解析JSON文件 {filename}，已跳过。")
#         except Exception as e:
#             print(f"\n处理文件 {filename} 时发生未知错误: {e}")

#     # --- 5. 保存最终的数据集文件 ---
#     with open(dataset_filename, 'w', encoding='utf-8') as f:
#         json.dump(dataset, f, indent=2, ensure_ascii=False)

#     print("-" * 30)
#     print("数据集创建成功！")
#     print(f"总共处理了 {image_id_counter} 张图片和 {len(dataset['annotations'])} 个问答对。")
#     print(f"图片已保存到 '{images_dir}' 目录。")
#     print(f"数据集信息已保存到 '{dataset_filename}' 文件。")


# if __name__ == '__main__':
#     create_vqa_dataset(
#         output_dir='D:\桌面\ChartData\output', 
#         images_dir='D:\桌面\ChartData\dataset\images', 
#         dataset_filename='D:\桌面\ChartData\dataset\dataset.json'
#     )

# 导入所需的库
import os
import json
import shutil
from PIL import Image
from datetime import datetime
from tqdm import tqdm

def create_vqa_dataset(output_dir='D:\桌面\ChartData\dataset', images_dir='D:\桌面\ChartData\dataset\images', dataset_filename='D:\桌面\ChartData\dataset\dataset.json'):
    """
    从包含两种不同结构JSON文件的目录中，创建一个统一的视觉问答（VQA）数据集。
    此版本新增功能：将每个条目对应的 `source_data` 完整地打包到最终数据集中。
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
            "version": "3.1", # 版本更新，体现包含了源数据
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
                chart_type = data.get('generation_info', {}).get('chart_type')
                generated_charts = data.get('generated_charts', [])
                
                # #################### 修改点 1 ####################
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

                            dataset['images'].append({
                                "id": image_id_counter, "file_name": new_image_filename,
                                "height": height, "width": width, "chart_type": chart_type,
                                "complexity_level": complexity_level
                            })

                            question_id_counter += 1
                            
                            # #################### 修改点 2 ####################
                            # 在 annotations 中添加 source_data 字段
                            dataset['annotations'].append({
                                "question_id": question_id_counter, 
                                "image_id": image_id_counter,
                                "question": question, 
                                "answers": [{"answer": str(answer)}],
                                "source_data": source_data  # 在此处注入
                            })
                        else:
                            print(f"\n警告: 找不到图片文件 {image_to_find_path}，已跳过。")
            
            # **分支2：处理“单一图表实体”类型的JSON**
            elif 'image_filename' in data and 'question' in data:
                question = data.get('question')
                answer = data.get('answer')
                original_image_filename = data.get('image_filename')
                
                # #################### 修改点 3 ####################
                # 从JSON的顶层提取 source_data，并从中获取 chart_type
                source_data = data.get('source_data', {})
                chart_type = source_data.get('chart_type') 
                
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

                    dataset['images'].append({
                        "id": image_id_counter, "file_name": new_image_filename,
                        "height": height, "width": width, "chart_type": chart_type,
                        "complexity_level": complexity_level
                    })

                    question_id_counter += 1
                    
                    # #################### 修改点 4 ####################
                    # 在 annotations 中添加 source_data 字段
                    dataset['annotations'].append({
                        "question_id": question_id_counter, 
                        "image_id": image_id_counter,
                        "question": question, 
                        "answers": [{"answer": str(answer)}],
                        "source_data": source_data # 在此处注入
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
    # 请根据你的实际情况修改这里的路径
    # 示例路径：
    # output_dir='D:\\ChartData\\output', 
    # images_dir='D:\\ChartData\\dataset\\images', 
    # dataset_filename='D:\\ChartData\\dataset\\dataset.json'
    
    # 为了方便，这里使用相对路径，请确保 output 文件夹与此脚本在同一目录下
    create_vqa_dataset(
        output_dir='D:\桌面\ChartData\output', 
        images_dir='D:\桌面\ChartData\dataset\images', 
        dataset_filename='D:\桌面\ChartData\dataset\dataset.json'
    )