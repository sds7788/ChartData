# -*- coding: utf-8 -*-
import os
import json
import time
import tempfile
from PIL import Image
from tqdm import tqdm
import dashscope # 核心变化：使用 dashscope 库

# --- 1. 配置信息 (请根据您的实际情况修改) ---

# 在这里放入你所有的 Dashscope API Key
# 格式为 "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
api_keys = [
    "sk-01077499a51b4faabfd931dd2d6e0ccb",  # 您的第一个Key
    #"sk-yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",  # 您的第二个Key
    # ... 可以继续添加更多
]

# 要调用的模型名称 (已更换为Qwen-VL模型)
MODEL_NAME = "qwen2.5-vl-72b-instruct"

# 定义你的基准测试JSON文件路径
benchmark_json_file = "D:\桌面\ChartData\\dataset\\train_dataset.json"
# 定义存放图片的文件夹路径
image_directory = "D:\桌面\ChartData\dataset\images"
# 最终输出的JSON文件名
output_json_file = "D:\桌面\ChartData\dataset\eva.json" # 建议使用新文件名以区分不同模型的结果

# 每个Key允许的调用次数（为保险起见，可设置比API限制略低的次数）
CALLS_PER_KEY = 980 # Dashscope的QPS限制通常更灵活，但保留此机制用于多Key轮转

# --- 2. 核心功能函数 ---

def load_benchmark_data(json_path, img_dir):
    """
    智能加载基准测试数据，处理不同格式的答案，并返回查询列表。
    (此函数逻辑保留不变)
    """
    queries = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        
        for annotation in data['annotations']:
            image_id = annotation['image_id']
            if image_id in image_id_to_filename:
                ground_truth_answer = ''
                # 智能处理答案格式
                if 'answers' in annotation:
                    answers_list = annotation.get('answers', [])
                    if answers_list:
                        ground_truth_answer = answers_list[0].get('answer', '')
                elif 'choices' in annotation and 'correct_answer_idx' in annotation:
                    choices = annotation.get('choices', [])
                    correct_idx = annotation.get('correct_answer_idx')
                    if choices and correct_idx is not None and 0 <= correct_idx < len(choices):
                        ground_truth_answer = choices[correct_idx]

                query_item = {
                    'question_id': annotation.get('question_id', ''),
                    'question': annotation.get('question', ''),
                    'answer': ground_truth_answer,
                    'image_path': os.path.join(img_dir, image_id_to_filename[image_id])
                }
                queries.append(query_item)
        
        print(f"成功从 '{json_path}' 加载并处理了 {len(queries)} 条数据。")
        return queries

    except FileNotFoundError:
        print(f"错误: 未找到基准文件 '{json_path}'。")
        return None
    except Exception as e:
        print(f"处理JSON文件时发生错误: {e}")
        return None

def call_qwen_vision(api_key: str, image: Image.Image, question: str):
    """
    使用指定的API Key调用Qwen-VL模型。
    该函数逻辑参考自 InfoChartQA.ipynb。
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # 结合问题和指令 (保留原有的指令，以期望获得更一致的输出格式)
    instruction_prompt = "Based on the image, answer the following question. Provide only the final answer without any explanation or additional text."
    full_prompt = f"{instruction_prompt}\n\n{question}"
    
    # Qwen-VL API需要本地文件路径，因此创建临时文件
    temp_files = []
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            image.save(tmp_file, format="PNG")
            tmp_file_path = tmp_file.name
            temp_files.append(tmp_file_path)

        # 构建API需要的 messages 结构
        messages = [{
            'role': 'user',
            'content': [
                {'image': f'file://{tmp_file_path}'},
                {'text': full_prompt}
            ]
        }]
        
        # 调用API
        response = dashscope.MultiModalConversation.call(
            model=MODEL_NAME,
            messages=messages,
            api_key=api_key
        )

        # 解析返回结果
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            # content 是一个列表，提取其中的文本部分
            for part in content:
                if 'text' in part:
                    return part['text']
            return "API_RESPONSE_ERROR: No text found in response."
        else:
            return f"API_CALL_ERROR: Code: {response.code}, Message: {response.message}"

    except Exception as e:
        return f"SDK_ERROR: {str(e)}"
    
    finally:
        # 清理所有创建的临时文件
        for file_path in temp_files:
            if os.path.exists(file_path):
                os.remove(file_path)

def save_results_to_json(data, file_path):
    """将结果列表保存为格式化的JSON文件。(此函数逻辑保留不变)"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# --- 3. 主程序入口 ---
if __name__ == '__main__':
    # 加载数据集
    queries = load_benchmark_data(benchmark_json_file, image_directory)
    if not queries:
        exit() # 如果数据加载失败则退出

    # 断点续传逻辑 (保留不变)
    all_results = []
    processed_ids = set()
    if os.path.exists(output_json_file):
        try:
            with open(output_json_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
                if not isinstance(all_results, list):
                    all_results = []
                processed_ids = {item['question_id'] for item in all_results if 'question_id' in item}
            print(f"检测到已有进度。已加载 {len(all_results)} 条结果，将从断点处继续。")
        except (json.JSONDecodeError, TypeError):
            print(f"警告: '{output_json_file}' 文件存在但无法解析。将创建新文件。")
            all_results = []
            processed_ids = set()
    
    # 根据已处理的进度，计算当前应该使用的API Key索引 (保留不变)
    start_index_from_progress = len(all_results)
    current_api_key_index = (start_index_from_progress // CALLS_PER_KEY) % len(api_keys)
    
    print(f"Dashscope API Key 列表已加载，共 {len(api_keys)} 个 Key。")
    print(f"根据已有进度，当前使用 Key 索引: {current_api_key_index}")

    # 创建tqdm进度条
    pbar = tqdm(total=len(queries), desc="处理进度")
    pbar.update(len(all_results))
    
    # 主循环 (已修改为适配Dashscope)
    for i, item in enumerate(queries):
        
        # 跳过已处理的条目
        if item['question_id'] in processed_ids:
            continue
            
        # API Key 自动更换逻辑，基于绝对索引 i
        if i > 0 and i % CALLS_PER_KEY == 0:
            next_key_index = (i // CALLS_PER_KEY) % len(api_keys)
            if next_key_index != current_api_key_index:
                current_api_key_index = next_key_index
                pbar.write(f"\n已到达切换点 (索引 {i})，切换到下一个 API Key (索引 {current_api_key_index})")
                pbar.write("暂停10秒以确保Key切换生效...")
                time.sleep(10)

        # 准备数据
        question_id = item.get('question_id')
        question = item.get('question', '')
        image_path = item.get('image_path', '')
        model_output = ""

        if not image_path or not question:
            pbar.write(f"条目 (ID: {question_id}) 缺少图像路径或问题，已跳过。")
            model_output = "SKIPPED_MISSING_DATA"
        else:
            try:
                # 获取当前要使用的API Key
                current_key = api_keys[current_api_key_index]
                
                # 调用模型 (已更换为新的调用函数)
                image = Image.open(image_path)
                model_output = call_qwen_vision(current_key, image, question)

            except FileNotFoundError:
                error_message = f"ERROR: Image file not found at '{image_path}'"
                pbar.write(f"\n错误: 条目 (ID: {question_id}) 的图片文件未找到，路径: '{image_path}'")
                model_output = error_message
            except Exception as e:
                error_message = f"ERROR: {str(e)}"
                pbar.write(f"\n条目 (ID: {question_id}) 处理失败: {error_message}")
                model_output = error_message
        
        # 记录结果并实时保存
        result_record = {
            "question_id": question_id,
            "model_output": model_output
        }
        all_results.append(result_record)
        save_results_to_json(all_results, output_json_file)
        
        # 更新进度条
        pbar.update(1)
        # 短暂休眠以避免过快的API请求
        time.sleep(1.5)

    pbar.close()
    print(f"\n处理完成！所有结果已保存至 '{output_json_file}'。")