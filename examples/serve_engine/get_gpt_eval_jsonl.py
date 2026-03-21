import os
import json
import re
from pathlib import Path

# ===================== 配置项 =====================
# 根目录（包含{id}.wav、{id}_in.txt、{id}_out.txt的文件夹）
model_flag = "step_3_1205_1"
step_num = "26421"
dataset = "commoneval"
result_type = "a"
root_dir = f"/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/generations/{model_flag}/{step_num}_steps/{dataset}/{result_type}_as_input"
root_dir_baseline = f"/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/generations/{model_flag}/{step_num}_steps/{dataset}/{result_type}_as_input"

# result_type = "a"
root_dir_baseline = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/higgs/LLaMA-Omni2/examples/llama-omni2-3b/alpacaeval"
root_dir_baseline = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/VITA-Audio/ex_output/commoneval"
root_dir_baseline = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/higgs/GLM-4-Voice/glm-4-voice-finetune/ex_output/commoneval"
root_dir_baseline = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/higgs/Step-Audio2/ex_output_t/commoneval"
root_dir_baseline = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/higgs/Llama-3.1-8B-Omni/LLaMA-Omni/omni_speech/infer/examples/alpacaeval"
root_dir_baseline = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/higgs/LLaMA-Omni2/examples/llama-omni2-3b_a/commoneval"
root_dir_baseline = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/generations/step_3_1205_1/26421_steps/commoneval/t_as_input"

# 输出文件路径（保存到root_dir的上级目录，文件名a_as_input_gpt_eval.jsonl）
if "_as_input" in root_dir_baseline:
    output_path = os.path.join(os.path.dirname(root_dir_baseline), f"{result_type}_as_input_gpt_eval.jsonl")
else:
    output_path = os.path.join(os.path.dirname(root_dir_baseline), f"{dataset}_baseline_gpt_eval.jsonl")
# 匹配id的正则（提取{id}_in.txt/{id}_out.txt中的数字id）
ID_PATTERN = re.compile(r"^(\d+)_(in|out)\.txt$")

# ===================== 核心函数 =====================
def read_file_content(file_path):
    """读取文件内容，处理编码和空行，返回清洗后的字符串"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # 读取内容并去除首尾空白（换行/空格/制表符）
            content = f.read().strip()
            # 替换多余的换行和空格，避免格式问题
            content = re.sub(r"\s+", " ", content)
            return content
    except FileNotFoundError:
        print(f"⚠️  文件不存在：{file_path}")
        return ""
    except UnicodeDecodeError:
        # 兼容其他编码（如gbk）
        with open(file_path, "r", encoding="gbk") as f:
            content = f.read().strip()
            content = re.sub(r"\s+", " ", content)
            return content
    except Exception as e:
        print(f"❌  读取文件失败 {file_path}：{str(e)}")
        return ""

def collect_ids_and_generate_jsonl(root_dir, output_path):
    """收集所有id，读取对应文件内容，生成JSON Lines文件"""
    # 1. 遍历文件夹，收集所有唯一的id
    id_set = set()
    all_files = os.listdir(root_dir)
    
    for file_name in all_files:
        match = ID_PATTERN.match(file_name)
        if match:
            id_str = match.group(1)
            id_set.add(id_str)  # 收集id（字符串形式，避免数字/字符串混淆）
    
    # 按数字排序id，保证输出有序
    sorted_ids = sorted(id_set, key=lambda x: int(x))
    print(f"✅  共找到 {len(sorted_ids)} 个有效id：{sorted_ids[:5]}...（仅显示前5个）")

    # 2. 遍历每个id，读取in/out文件内容并构建JSON
    output_data = []
    for id_str in sorted_ids:
        # 拼接文件路径
        in_file = os.path.join(root_dir, f"{id_str}_in.txt")
        if "_as_input" in root_dir_baseline:
            out_file = os.path.join(root_dir, f"{id_str}_out.txt")
        else:
            out_file = os.path.join(root_dir_baseline, f"{id_str}.txt")
        
        # 读取内容
        prompt = read_file_content(in_file)
        response = read_file_content(out_file)
        
        # 跳过空内容（避免无效数据）
        if not prompt or not response:
            print(f"⚠️  id={id_str} 的prompt/response为空，跳过")
            continue
        
        # 构建JSON对象（不带reference版本）
        json_obj = {
            "prompt": prompt,
            "response": response
        }
        output_data.append(json_obj)
        print(f"✅  处理完成 id={id_str}：prompt={prompt[:50]}... | response={response[:50]}...")

    # 3. 保存为JSON Lines文件（每行一个JSON对象）
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for json_obj in output_data:
                # 每行写入一个JSON字符串（ensure_ascii=False支持特殊字符）
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        print(f"\n✅  所有数据已保存至：{output_path}")
        print(f"📊  最终生成 {len(output_data)} 条有效数据")
    except Exception as e:
        print(f"❌  保存文件失败：{str(e)}")

# ===================== 主流程 =====================
if __name__ == "__main__":
    # 检查根目录是否存在
    if not os.path.exists(root_dir):
        print(f"❌  根目录不存在：{root_dir}")
    else:
        print("========== 开始生成GPT评估用JSON Lines文件 ==========")
        collect_ids_and_generate_jsonl(root_dir, output_path)
        print("========== 脚本执行结束 ==========")
