import json
import os
import re

def simple_replace_audio_paths():
    # 配置（仅改这两个路径即可）
    json_file_path = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/LLaMA-Factory/data/emotion_sharegpt.json"
    new_audio_dir = "/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/generations/step_3_1206/50000_steps/GenEmotion-en/a_as_input"

    # 1. 读取原始JSON
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 遍历替换每个样本的音频路径 + 提取情绪词替换conversations
    for sample in data:
        # 提取音频ID并替换audios路径
        old_path = sample["audios"][0]
        audio_id = re.search(r"/(\d+)\.wav", old_path).group(1)  # 提取ID（适配任意位置的数字.wav）
        sample["audios"][0] = f"{new_audio_dir}/{audio_id}.wav"  # 替换音频路径

        # 读取对应txt文件并提取情绪词
        txt_path = f"{new_audio_dir}/{audio_id}_in.txt"
        if os.path.exists(txt_path):
            # 读取txt文件内容
            with open(txt_path, 'r', encoding='utf-8') as f:
                txt_content = f.read().strip()
            
            # 正则提取 a/an 后面、tone前面的情绪词（适配 "with a xxx tone" 或 "with an xxx tone"）
            emotion_match = re.search(r"with (a|an) (\w+) tone", txt_content, re.IGNORECASE)
            if emotion_match:
                emotion = emotion_match.group(2).capitalize()  # 提取情绪词并首字母大写（如angry→Angry）
                # 替换conversations[1].value为提取的情绪词（也可根据需求修改替换规则）
                sample["conversations"][1]["value"] = emotion  # 若需保留原格式可改为：f"\\boxed{{{emotion}}}"
                print(f"✅ 样本{audio_id}：提取情绪词 {emotion}，已替换conversations内容")
            else:
                print(f"⚠️  样本{audio_id}：txt文件未匹配到情绪词（a/an xxx tone），内容：{txt_content[:50]}...")
        else:
            print(f"⚠️  样本{audio_id}：txt文件不存在 → {txt_path}")

    # 3. 覆盖写回原文件
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("\n✅ 全部处理完成，已覆盖原文件")

if __name__ == "__main__":
    simple_replace_audio_paths()
