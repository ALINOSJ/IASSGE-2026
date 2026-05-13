import os
import json
import re
from funasr import AutoModel

# 🌟 请替换为你自己的阿里云 DashScope API Key
DASHSCOPE_API_KEY = "your-dashscope-api-key"

class IASSGE_PC_Baseline:
    def __init__(self):
        print("🚀 [Track 1] 正在加载 FunASR 模型...")
        # 基于 FunASR 构建基础识别能力 (带时间戳、VAD、标点)
        self.asr_model = AutoModel(
            model="paraformer-zh", 
            vad_model="fsmn-vad", 
            punc_model="ct-punc",
            disable_update=True
        )
        print("✅ FunASR 模型加载完成！")

    def track2_enhance(self, raw_text):
        """
        [Track 2] 结合规则后处理与大模型的关键词增强机制
        """
        # 1. 规则后处理 (Rule-based post-processing)
        # 例如：将连续的数字强制转换为阿拉伯数字
        processed_text = re.sub(r'[零一二三四五六七八九十百千万亿]+', 
                                lambda m: self._cn_to_arabic(m.group()), 
                                raw_text)
        
        # 默认给定一个说话人标签
        processed_text = f"<speaker1>{processed_text}"

        # 2. 大模型增强 (LLM Enhancement) - 演示调用通义千问 API
        # 如果没有配置 API Key，则回退到基础正则规则
        if DASHSCOPE_API_KEY == "your-dashscope-api-key":
             # 简单的正则策略：把所有数字套上 <key> 标签
             return re.sub(r'(\d+)', r'<key>\1</key>', processed_text)
        
        try:
            import dashscope
            dashscope.api_key = DASHSCOPE_API_KEY
            prompt = f"""
            你是一个信息无障碍字幕处理助手。请提取以下句子中的核心关键词（如金额、时间、药名、地点等），并用 <key> 和 </key> 标签包裹起来。
            注意：标签内绝对不能包含标点符号。
            原句：{processed_text}
            处理后的句子：
            """
            response = dashscope.Generation.call(
                model=dashscope.Generation.Models.qwen_turbo,
                prompt=prompt
            )
            if response.status_code == 200:
                return response.output.text.strip()
            else:
                print(f"⚠️ LLM 调用失败: {response.code} - {response.message}")
                return processed_text # 失败时返回规则处理结果
        except Exception as e:
            print(f"⚠️ LLM 处理异常: {e}")
            return processed_text

    def _cn_to_arabic(self, cn_str):
        """简单的中文数字转阿拉伯数字辅助函数 (演示用)"""
        # 此处省略复杂的转换逻辑，实际比赛中需完善
        # 这里仅做简单替换演示
        cn_num_dict = {'零':'0','一':'1','二':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9'}
        res = ""
        for char in cn_str:
            if char in cn_num_dict:
                 res += cn_num_dict[char]
            else:
                 return cn_str # 遇到复杂单位直接返回原样
        return res if res else cn_str

    def process_audio(self, audio_path, output_json_path):
            print(f"⏳ 正在处理音频: {audio_path}")
            
            # [Track 1] 运行 FunASR 基础识别
            res = self.asr_model.generate(input=audio_path, batch_size_s=300)
            
            if not res or len(res) == 0:
                print("⚠️ 未识别到任何声音")
                return
                
            result_data = res[0]
            raw_text = result_data.get("text", "")
            
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
            
            # 逆向组装出完美的句级时间戳
            sentences = []
            current_sentence = ""
            current_start = None
            current_end = None

            for item in word_timestamps_aligned:
                char = item["char"]
                start = item["start_time_ms"]
                end = item["end_time_ms"]

                if char.isspace():
                    continue

                current_sentence += char
                
                if start is not None:
                    if current_start is None:
                        current_start = start
                    current_end = end
                
                if char in "。！？":
                    if current_sentence.strip():
                        sentences.append({
                            "text": current_sentence.strip(),
                            "start_time_ms": current_start,
                            "end_time_ms": current_end
                        })
                    current_sentence = ""
                    current_start = None
                    current_end = None

            if current_sentence.strip():
                sentences.append({
                    "text": current_sentence.strip(),
                    "start_time_ms": current_start,
                    "end_time_ms": current_end
                })

            # ========================================================
            # [Track 2] 将分好的句子传入结构化引擎进行规则与大模型增强
            # ========================================================
            structured_results = []
            for sentence in sentences:
                # 增强文本 (加说话人、加关键标签、数字规范化)
                final_text = self.track2_enhance(sentence["text"])
                
                structured_results.append({
                    "text": final_text,
                    "start_time_ms": sentence["start_time_ms"],
                    "end_time_ms": sentence["end_time_ms"]
                })
                
            # 输出 IASSGE-2026 标准 JSON 格式
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(structured_results, f, ensure_ascii=False, indent=4)
                
            print(f"🎉 处理完成！结果已保存至: {output_json_path}")

if __name__ == "__main__":
    baseline = IASSGE_PC_Baseline()
    test_audio = "test.wav" 
    if os.path.exists(test_audio):
        baseline.process_audio(test_audio, "predict_pc.json")
    else:
        print(f"⚠️ 找不到测试音频 {test_audio}，请放入音频后重试。")