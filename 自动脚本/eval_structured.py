import json
import re
import argparse
from collections import Counter

def extract_features_and_speakers(text, last_speaker="UNKNOWN"):
    """从文本中分离纯文本、关键词和每个字符的说话人标签"""
    keys = re.findall(r'<key>(.*?)</key>', text)
    
    parts = re.split(r'(<speaker\d+>)', text)
    pure_text = ""
    speaker_labels = []
    
    current_speaker = last_speaker
    for part in parts:
        if part.startswith('<speaker') and part.endswith('>'):
            current_speaker = part[1:-1]
        else:
            clean_part = re.sub(r'<[^>]+>', '', part)
            # 去除常见的中英文标点符号，文本准确率不计标点
            clean_part = re.sub(r'[，。！？、；：“”‘’（）《》〈〉【】『』「」,.\!?;:\"\'\(\)\[\]\{\}]', '', clean_part)
            clean_part = clean_part.replace(" ", "")
            pure_text += clean_part
            speaker_labels.extend([current_speaker] * len(clean_part))
            
    return pure_text, keys, speaker_labels, current_speaker

def get_alignment_and_speaker_accuracy(pred_text, gt_text, pred_labels, gt_labels):
    """DP 计算编辑距离并回溯对齐字符，通过贪心策略解决说话人乱序问题计算准确率"""
    m, n = len(pred_text), len(gt_text)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_text[i - 1] == gt_text[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
                
    i, j = m, n
    confusion_matrix = Counter()
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and pred_text[i - 1] == gt_text[j - 1]:
            confusion_matrix[(gt_labels[j - 1], pred_labels[i - 1])] += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            confusion_matrix[(gt_labels[j - 1], pred_labels[i - 1])] += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
            
    mapping = {}
    sorted_pairs = sorted(confusion_matrix.items(), key=lambda x: x[1], reverse=True)
    matched_gt, matched_pred = set(), set()
    
    for (g, p), count in sorted_pairs:
        if g != "UNKNOWN" and p != "UNKNOWN" and g not in matched_gt and p not in matched_pred:
            mapping[p] = g
            matched_gt.add(g)
            matched_pred.add(p)
            
    correct_chars = 0
    for (g, p), count in confusion_matrix.items():
        if p in mapping and mapping[p] == g:
            correct_chars += count
            
    speaker_acc = correct_chars / len(gt_text) if len(gt_text) > 0 else 1.0
    return dp[m][n], speaker_acc, mapping

def calculate_time_iou(s1, e1, s2, e2):
    """计算两个时间段的交并比 (IoU)"""
    intersection = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return intersection / union if union > 0 else 0

def evaluate_time(preds, gts):
    """计算时间轴平均绝对误差(MAE)"""
    if not preds or not gts:
        return 0.0, 0.0
    
    start_diffs = []
    end_diffs = []
    
    # 针对每个 GT 寻找时间重叠度最大(或时间中心点最近)的 Pred 进行匹配
    for gt in gts:
        gt_s, gt_e = gt.get("start_time_ms", 0), gt.get("end_time_ms", 0)
        
        best_iou = -1
        best_pred = None
        for pred in preds:
            pred_s, pred_e = pred.get("start_time_ms", 0), pred.get("end_time_ms", 0)
            iou = calculate_time_iou(gt_s, gt_e, pred_s, pred_e)
            if iou > best_iou:
                best_iou = iou
                best_pred = pred
                
        if best_pred and best_iou > 0:
            pred_s, pred_e = best_pred.get("start_time_ms", 0), best_pred.get("end_time_ms", 0)
            start_diffs.append(abs(gt_s - pred_s))
            end_diffs.append(abs(gt_e - pred_e))
            
    avg_start_diff = sum(start_diffs) / len(start_diffs) if start_diffs else 0.0
    avg_end_diff = sum(end_diffs) / len(end_diffs) if end_diffs else 0.0
    
    return avg_start_diff, avg_end_diff

def evaluate(pred_path, gt_path):
    with open(pred_path, 'r', encoding='utf-8') as f:
        preds = json.load(f)
    with open(gt_path, 'r', encoding='utf-8') as f:
        gts = json.load(f)
        
    pred_full_text, gt_full_text = "", ""
    pred_keys, gt_keys = [], []
    pred_speaker_labels, gt_speaker_labels = [], []

    last_spk = "UNKNOWN"
    for item in preds:
        pt, pk, sl, last_spk = extract_features_and_speakers(item.get("text", ""), last_spk)
        pred_full_text += pt
        pred_keys.extend(pk)
        pred_speaker_labels.extend(sl)
        
    last_spk = "UNKNOWN"
    for item in gts:
        gt, gk, sl, last_spk = extract_features_and_speakers(item.get("text", ""), last_spk)
        gt_full_text += gt
        gt_keys.extend(gk)
        gt_speaker_labels.extend(sl)

    # 1. 计算 CER 与 说话人准确率
    distance, speaker_acc, speaker_mapping = get_alignment_and_speaker_accuracy(pred_full_text, gt_full_text, pred_speaker_labels, gt_speaker_labels)
    cer = distance / len(gt_full_text) if len(gt_full_text) > 0 else 1.0

    # 2. 时间轴偏差
    avg_start_diff, avg_end_diff = evaluate_time(preds, gts)
    time_penalty = min((avg_start_diff + avg_end_diff) / 2000, 1.0)

    # 3. 计算 Keyword 召回率
    gt_cnt = Counter(gt_keys)
    tp = 0
    for k, gt_count in gt_cnt.items():
        # 在选手的纯文本输出中统计关键词出现次数
        pred_count = pred_full_text.count(k)
        # 取最小值截断，防止复读机刷分
        tp += min(pred_count, gt_count)
        
    total_gt = sum(gt_cnt.values())
    recall = tp / total_gt if total_gt > 0 else 0.0
    
    # Track2 综合得分公式
    final_score = (1 - cer) * 0.45 + recall * 0.35 + speaker_acc * 0.1 + (1 - time_penalty) * 0.1
    final_score = max(final_score, 0.0)

    print("="*50)
    print("[CCL26-Eval 自动评分报告 - 结构化字幕赛道 (TRACK 2)]")
    print("="*50)
    print(f"字错误率 (CER): {cer:.2%}")
    print(f"时间轴平均偏差: 起始点 {avg_start_diff:.2f}ms, 结束点 {avg_end_diff:.2f}ms")
    print(f"说话人准确率 (Speaker Acc): {speaker_acc:.2%} ")
    print(f"关键词召回率 (Recall): {recall:.2%}")
    print(f"综合得分: {final_score:.4f}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCL26-Eval 结构化字幕评测脚本 (Track 2)")
    parser.add_argument("--pred", required=True, help="选手的预测 JSON 文件")
    parser.add_argument("--gt", required=True, help="官方 Ground Truth JSON 文件")
    args = parser.parse_args()
    evaluate(args.pred, args.gt)
