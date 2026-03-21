import json

def calculate_score_average():
    model_flag = "step_3_1205_1"
    step_num = "26421"
    dataset = "alpacaeval"
    result_type = "t"
    # 目标文件路径
    file_path = f"/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/generations/{model_flag}/{step_num}_steps/{dataset}/result-{result_type}_as_input_gpt_eval.jsonl"

    file_path = f"/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/higgs/LLaMA-Omni2/examples/llama-omni2-3b/result-a_as_input_gpt_eval.jsonl"
    
    total_score = 0.0  # 总分数
    valid_count = 0    # 有效样本数
    total_lines = 0    # 总行数
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                total_lines += 1
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                try:
                    # 解析单行JSON
                    data = json.loads(line)
                    
                    # 提取score字段并转换为数值
                    score_list = data.get("score", [])
                    if not isinstance(score_list, list) or len(score_list) == 0:
                        print(f"⚠️  第{line_num+1}行：score字段为空或不是列表，跳过")
                        continue
                    
                    # 尝试将score第一个元素转为浮点数
                    score_str = score_list[0].strip()
                    score = float(score_str)
                    
                    total_score += score
                    valid_count += 1
                    
                except json.JSONDecodeError:
                    print(f"❌ 第{line_num+1}行：JSON解析失败，跳过")
                except (ValueError, TypeError):
                    print(f"❌ 第{line_num+1}行：score值'{score_list[0]}'无法转为数字，跳过")
                except Exception as e:
                    print(f"❌ 第{line_num+1}行：处理出错 - {str(e)}，跳过")
        
        # 计算平均值
        if valid_count == 0:
            print("\n🚫 无有效score数据")
            average_score = 0.0
        else:
            average_score = total_score / valid_count
        
        # 输出结果
        print("\n" + "="*50)
        print(f"文件总行数：{total_lines}")
        print(f"有效样本数：{valid_count}")
        print(f"总分数：{total_score:.3f}")
        print(f"Score平均值：{average_score:.3f}")
        print("="*50)
        
        return average_score
        
    except FileNotFoundError:
        print(f"❌ 文件不存在：{file_path}")
        return None
    except PermissionError:
        print(f"❌ 无权限访问文件：{file_path}")
        return None
    except Exception as e:
        print(f"❌ 程序执行出错：{str(e)}")
        return None

if __name__ == "__main__":
    calculate_score_average()
