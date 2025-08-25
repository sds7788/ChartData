import json
import re

# --- 辅助函数：用于标准化处理 (遵照您的要求，保持不变) ---

def normalize_number(text):
    """
    从字符串或数字中提取和标准化数字。
    """
    if isinstance(text, (int, float)):
        return float(text)
    if not isinstance(text, str):
        return None
    text = text.strip().lower()
    if '%' in text:
        text = text.replace('%', '')
        try:
            return float(text) / 100.0
        except ValueError:
            return None
    text = text.replace(',', '')
    match = re.search(r'-?\d+\.?\d*', text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None

def normalize_text_list(text):
    """
    将逗号分隔的字符串转换为标准化的、排序的列表。
    """
    if not isinstance(text, str):
        return []
    items = text.lower().split(',')
    cleaned_items = [item.strip() for item in items if item.strip()]
    cleaned_items.sort()
    return cleaned_items

# --- 核心比较逻辑：统一处理函数 ---

def unified_comparison(gt_ans, model_ans):
    """
    不依赖answer_type，智能地比较两个答案。
    返回一个元组: (是否正确, 使用的比较方法, 标准化后的值)
    """
    if gt_ans is None or model_ans is None:
        return False, "N/A", "One of the answers was None"

    # --- 规则 1: 尝试进行数值比较 ---
    norm_gt_num = normalize_number(gt_ans)
    norm_model_num = normalize_number(model_ans)
    
    # 如果两者都能被解析为数字，则进行数值比较
    if norm_gt_num is not None and norm_model_num is not None:
        is_correct = abs(norm_gt_num - norm_model_num) < 1e-9
        method = "Numerical"
        details = f"GT: {norm_gt_num}, Model: {norm_model_num}"
        return is_correct, method, details

    # --- 规则 2: 尝试进行列表比较 ---
    gt_str = str(gt_ans)
    model_str = str(model_ans)
    # 启发式规则：如果任一答案包含逗号，则按列表处理
    if ',' in gt_str or ',' in model_str:
        norm_gt_list = normalize_text_list(gt_str)
        norm_model_list = normalize_text_list(model_str)
        # 确保列表不为空（例如 "a," 会产生空元素）
        if norm_gt_list and norm_model_list:
            is_correct = norm_gt_list == norm_model_list
            method = "List"
            details = f"GT: {norm_gt_list}, Model: {norm_model_list}"
            return is_correct, method, details

    # --- 规则 3: 默认进行字符串比较 ---
    norm_gt_str = gt_str.strip().lower()
    norm_model_str = model_str.strip().lower()
    is_correct = norm_gt_str == norm_model_str
    method = "String"
    details = f"GT: '{norm_gt_str}', Model: '{norm_model_str}'"
    return is_correct, method, details


# --- 主评估逻辑 ---

def evaluate_model(ground_truth_path, model_output_path, results_path):
    """
    根据统一比较逻辑，评估模型输出并计算得分。
    """
    # 1. 加载数据
    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
        with open(model_output_path, 'r', encoding='utf-8') as f:
            model_outputs_data = json.load(f)
    except Exception as e:
        print(f"错误: 加载或解析文件时出错 - {e}")
        return

    # 2. 预处理数据以便快速查找
    # 假设 ground_truth 结构为 {"annotations": [{"question_id": ..., "answers": [{"answer": ...}]}]}
    # 如果您的结构更简单 (如 [{"question_id": ..., "answer": ...}]), 请相应修改
    true_annotations = {int(ann['question_id']): ann for ann in ground_truth_data.get('annotations', [])}
    model_outputs = {int(out['question_id']): out['model_output'] for out in model_outputs_data}
    
    evaluation_results = []
    total_score = 0
    total_questions = len(true_annotations)

    if total_questions == 0:
        print("警告: 真实答案文件中没有找到任何标注。")
        return

    # 3. 逐一评估每个问题
    for qid, true_ann in true_annotations.items():
        # 假设答案在 'answers' 列表的第一个元素的 'answer' 键中
        true_answer = true_ann.get('answers', [{}])[0].get('answer')
        model_output = model_outputs.get(qid)
        
        score = 0
        method = "N/A"
        details = "Model did not provide an output"
        
        if model_output is not None and true_answer is not None:
            is_correct, method, details = unified_comparison(true_answer, model_output)
            if is_correct:
                score = 1
        
        total_score += score
        
        evaluation_results.append({
            "question_id": qid,
            "question": true_ann.get('question', 'N/A'),
            "true_answer": true_answer,
            "model_output": model_output if model_output is not None else "---NO OUTPUT---",
            "comparison_method": method,
            "comparison_details": details,
            "score": score
        })

    # 4. 计算并保存最终结果
    final_accuracy = (total_score / total_questions) * 100 if total_questions > 0 else 0
    
    final_report = {
        "summary": {
            "total_questions": total_questions,
            "correct_answers": total_score,
            "accuracy_percent": f"{final_accuracy:.2f}%"
        },
        "results": evaluation_results
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
        
    print("评估完成！")
    print(f"总问题数: {total_questions}")
    print(f"正确回答数: {total_score}")
    print(f"准确率: {final_accuracy:.2f}%")
    print(f"详细评估结果已保存至: {results_path}")

# --- 程序入口 ---
if __name__ == '__main__':
    # 定义你的文件名
    ground_truth_file = 'D:\桌面\数据存储\dataset(2,20k)\\sampled_dataset_2000.json'
    model_outputs_file = 'D:\桌面\数据存储\dataset(2,20k)\eva.json'
    evaluation_results_file = 'D:\桌面\数据存储\dataset(2,20k)\eva_result.json'

    # 运行评估
    evaluate_model(ground_truth_file, model_outputs_file, evaluation_results_file)