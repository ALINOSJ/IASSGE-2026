import os
import json
import re  
from funasr import AutoModel

def batch_transcribe(input_folder="dataset_16k_wav", output_folder="dataset_transcripts"):
    """
    使用 FunASR 批量转写音频，并仅输出带有精准毫秒级时间戳的“句级(Sentence)”数据
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("🚀 正在加载 FunASR 模型...")
    
    # 初始化模型（使用 CUDA 加速）
    model = AutoModel(
        model="paraformer-zh", 
        vad_model="fsmn-vad", 
        punc_model="ct-punc",
        disable_update=True,
        device="cuda:0" 
    )
    print("✅ 模型加载完成！")

    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    print(f"📁 找到 {len(audio_files)} 个标准音频文件，开始批量转写...")

    for file_name in audio_files:
        audio_path = os.path.join(input_folder, file_name)
        base_name = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_folder, f"{base_name}.json")

        print(f"⏳ 正在处理: {file_name} ...")
        
        try:
            # 运行模型推理
            res = model.generate(input=audio_path, batch_size_s=300)
            
            if res and len(res) > 0:
                result_data = res[0]
                raw_text = result_data.get("text", "")
                
                # 初始化极简 JSON 结构 (只保留文件名和句级分段)
                structured_output = {
                    "audio_file": file_name,
                    "sentence_segments": [] 
                }

                # 1. 内部计算：优先提取字级/词级时间戳 (仅作为中间变量，不写入JSON)
                word_timestamps_aligned = []
                if "timestamp" in result_data:
                    timestamps = result_data["timestamp"]
                    punctuations = set("，。！？、；：“”‘’（）【】《》,.!?;:\"'()[]<>")
                    
                    ts_index = 0
                    for char in raw_text:
                        if char in punctuations or char.isspace():
                            word_timestamps_aligned.append({"char": char, "start_time_ms": None, "end_time_ms": None})
                        else:
                            if ts_index < len(timestamps):
                                word_timestamps_aligned.append({
                                    "char": char,
                                    "start_time_ms": timestamps[ts_index][0],
                                    "end_time_ms": timestamps[ts_index][1]
                                })
                                ts_index += 1
                            else:
                                word_timestamps_aligned.append({"char": char, "start_time_ms": None, "end_time_ms": None})

                # 2. 逆向组装出完美的句级时间戳
                sentences = []
                current_sentence = ""
                current_start = None
                current_end = None

                for item in word_timestamps_aligned:
                    char = item["char"]
                    start = item["start_time_ms"]
                    end = item["end_time_ms"]

                    if char.isspace():
                        continue  # 跳过纯空格干扰

                    current_sentence += char
                    
                    # 更新当前句子的起止时间
                    if start is not None:
                        if current_start is None:
                            current_start = start  # 遇到句子的第一个有声字，记录开始时间
                        current_end = end          # 不断往后推，记录最新一个字的结束时间
                    
                    # 遇到断句标点，结算当前句子
                    if char in "。！？":
                        if current_sentence.strip():
                            sentences.append({
                                "text": current_sentence.strip(),
                                "start_time_ms": current_start,
                                "end_time_ms": current_end
                            })
                        # 重置变量，准备接收下一句
                        current_sentence = ""
                        current_start = None
                        current_end = None

                # 兜底：如果整个音频最后一句没有标点符号结尾，也要强制结算存下来
                if current_sentence.strip():
                    sentences.append({
                        "text": current_sentence.strip(),
                        "start_time_ms": current_start,
                        "end_time_ms": current_end
                    })

                # 仅将句级数据组装到最终结果里
                structured_output["sentence_segments"] = sentences

                # 写入 JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_output, f, ensure_ascii=False, indent=4)
                
                print(f"  └─ ✅ 成功提取精简版句级时间戳！已保存至 {output_path}")

        except Exception as e:
            print(f"  └─ ❌ 处理出错: {e}")

    print("\n🎉 所有音频初步转写与句级提取完成！")

if __name__ == "__main__":
    batch_transcribe()