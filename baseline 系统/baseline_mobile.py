import os
import json
import re

class IASSGE_Mobile_Baseline:
    def __init__(self):
        print("🚀 正在加载轻量化端侧方案...")
        # 在实际移动端，这里可能会加载量化后的 ONNX 模型或 MNN 模型
        # 例如：量化版的 Paraformer 或更轻量的流式 ASR 模型
        print("✅ 轻量化 ASR 模型加载完成 (模拟)")

        print("🚀 正在加载轻量化 N-gram 语言模型...")
        # 模拟加载一个轻量级的 N-gram 标点预测模型
        self.ngram_punc_model = {"然后": "，", "但是": "，", "呢": "？", "了": "。"}
        print("✅ N-gram 模型加载完成")

    def mock_asr_inference(self, audio_path):
        """模拟轻量化 ASR 模型的输出 (不带标点，只带简单词级时间戳)"""
        # 在真实场景中，这里调用端侧推理引擎
        return [
            {"word": "发工资", "start": 1000, "end": 1500},
            {"word": "这天", "start": 1500, "end": 2000},
            {"word": "女老师", "start": 2100, "end": 2800},
            {"word": "懵", "start": 2800, "end": 3000},
            {"word": "了", "start": 3000, "end": 3200}
        ]

    def track2_mobile_enhance(self, words_info):
        """
        [手机赛道 Track 2]
        使用 N-gram 模型进行低延迟的标点预测与结构化断句。
        由于端侧资源限制，这里不使用大语言模型。
        """
        structured_results = []
        current_sentence = ""
        current_start = None
        current_end = None
        
        # 1. 基于词元流进行低延迟标点预测与断句
        for i, item in enumerate(words_info):
            word = item["word"]
            if current_start is None:
                current_start = item["start"]
            
            current_sentence += word
            current_end = item["end"]
            
            # 查表预测标点 (基于简单的 N-gram 或规则字典)
            if word in self.ngram_punc_model:
                punc = self.ngram_punc_model[word]
                current_sentence += punc
                
                # 如果是句号/问号，进行断句
                if punc in ["。", "？", "！"]:
                    # 2. 端侧轻量级正则关键词提取
                    final_text = self._lightweight_keyword_extract(current_sentence)
                    final_text = f"<speaker1>{final_text}"
                    
                    structured_results.append({
                        "text": final_text.strip(),
                        "start_time_ms": current_start,
                        "end_time_ms": current_end
                    })
                    # 重置状态
                    current_sentence = ""
                    current_start = None
        
        # 处理最后未断句的部分
        if current_sentence:
             final_text = self._lightweight_keyword_extract(current_sentence)
             final_text = f"<speaker1>{final_text}。" # 强制加句号
             structured_results.append({
                 "text": final_text.strip(),
                 "start_time_ms": current_start,
                 "end_time_ms": current_end
             })

        return structured_results

    def _lightweight_keyword_extract(self, text):
        """端侧轻量化规则关键词提取"""
        # 仅使用高效率的正则表达式，不消耗算力
        text = re.sub(r'(\d+块)', r'<key>\1</key>', text) # 匹配金额
        return text

    def process_audio(self, audio_path, output_json_path):
        print(f"⏳ [端侧模拟] 正在处理音频: {audio_path}")
        
        # [Track 1] 运行轻量化基础识别
        words_info = self.mock_asr_inference(audio_path)
        
        # [Track 2] 运行低延迟断句与结构化
        structured_results = self.track2_mobile_enhance(words_info)
            
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(structured_results, f, ensure_ascii=False, indent=4)
            
        print(f"🎉 处理完成！结果已保存至: {output_json_path}")

if __name__ == "__main__":
    mobile_baseline = IASSGE_Mobile_Baseline()
    test_audio = "test.wav" 
    mobile_baseline.process_audio(test_audio, "predict_mobile.json")