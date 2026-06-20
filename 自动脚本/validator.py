import json
import re
import argparse

def validate_submission(file_path):
    errors = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"JSON解析失败: {e}"]

    # 兼容处理
    if isinstance(data, dict):
        if "sentence_segments" in data:
            data = data["sentence_segments"]
        else:
            return False, ["JSON是字典格式，但找不到包含句子的列表，请确认格式。"]

    if not isinstance(data, list):
        return False, ["JSON有效数据必须是列表 (List) 格式"]

    # 用于追踪上一句话的时间戳
    prev_start_ms = -1
    prev_end_ms = -1

    for i, item in enumerate(data):
        # 1. 字段完整性检查
        for key in ("text", "start_time_ms", "end_time_ms"):
            if key not in item:
                errors.append(f"第 {i+1} 条数据缺少必要字段: '{key}'")
        
        if errors: continue

        # 2. 句内时间戳逻辑检查
        start_ms, end_ms = item["start_time_ms"], item["end_time_ms"]
        if not isinstance(start_ms, (int, float)) or not isinstance(end_ms, (int, float)):
            errors.append(f"第 {i+1} 条数据的时间戳格式不正确，必须为数字")
        elif start_ms >= end_ms:
            errors.append(f"第 {i+1} 条数据的开始时间 ({start_ms}) 大于等于结束时间 ({end_ms})")

        # 跨句时间轴顺序与重叠检查
        if i > 0:
            if start_ms < prev_start_ms:
                errors.append(f"第 {i+1} 条数据的开始时间 ({start_ms}) 早于上一条的开始时间 ({prev_start_ms})，时间轴乱序！")
            elif prev_end_ms > start_ms:
                errors.append(f"第 {i} 条的结束时间 ({prev_end_ms}) 大于第 {i+1} 条的开始时间 ({start_ms})，存在时间重叠或输入错误！")

        # 更新记忆，为下一轮循环做准备
        prev_start_ms = start_ms
        prev_end_ms = end_ms

        # 3. 结构化标签合规性检查 (白名单机制)
        text = item.get("text", "")
        
        all_tags = re.findall(r'<[^>]+>', text)
        for tag in all_tags:
            if tag in ["<key>", "</key>"]:
                continue
            elif re.match(r'^<speaker\d+>$', tag):
                continue
            else:
                errors.append(f"第 {i+1} 条数据包含非法或拼写错误的标签: 发现 '{tag}'，仅允许 <speaker数字>, <key>, </key>")
        
        # 检查 <key> 标签闭合是否成对
        if text.count("<key>") != text.count("</key>"):
            errors.append(f"第 {i+1} 条数据的 <key> 标签数量不匹配（未正确闭合）")
            
        # 检查 <key> 内部是否违规包含标点符号
        if re.search(r'<key>[^<]*[，。！？、,\?!][^<]*</key>', text):
            errors.append(f"第 {i+1} 条数据的 <key> 标签内部违规包含了标点符号")

    if errors:
        return False, errors
    return True, ["校验通过！格式完全符合 CCL26-Eval 标准。"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCL26-Eval JSON 自动校验脚本")
    parser.add_argument("file", help="待校验的 JSON 文件路径")
    args = parser.parse_args()
    
    is_valid, messages = validate_submission(args.file)
    if is_valid:
        print("✅", messages[0])
    else:
        print("❌ 校验失败，发现以下问题：")
        for msg in messages:
            print("  -", msg)
