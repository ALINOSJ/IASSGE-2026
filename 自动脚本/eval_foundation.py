import json
import re
import argparse

def extract_features(text):
    """从文本中分离纯文本 (Track1 仅需要提取纯文本去标签，并去除标点符号)"""
    # 剥离所有 HTML/XML 标签，得到纯文本
    pure_text = re.sub(r'<[^>]+>', '', text)
    # 去除常见的中英文标点符号
    pure_text = re.sub(r'[，。！？、；：“”‘’（）《》〈〉【】『』「」,.\!?;:\"\'\(\)\[\]\{\}]', '', pure_text)
    # 去除多余空格
    pure_text = pure_text.replace(" ", "")
    return pure_text

def calculate_edit_distance(s1, s2):
    """DP 计算 Levenshtein 编辑距离"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
    return dp[m][n]

def calculate_time_iou(s1, e1, s2, e2):
    """计算两个时间段的交并比"""
    intersection = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return intersection / union if union > 0 else 0

def evaluate_time(preds, gts):
    """计算时间轴平均绝对误差"""
    if not preds or not gts:
        return 0.0, 0.0
    
    start_diffs = []
    end_diffs = []
    
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

    for item in preds:
        pred_full_text += extract_features(item.get("text", ""))
        
    for item in gts:
        gt_full_text += extract_features(item.get("text", ""))

    # 1. 计算 CER
    distance = calculate_edit_distance(pred_full_text, gt_full_text)
    cer = distance / len(gt_full_text) if len(gt_full_text) > 0 else 1.0

    # 2. 时间轴偏差
    avg_start_diff, avg_end_diff = evaluate_time(preds, gts)

    print("="*50)
    print("[CCL26-Eval 自动评分报告 - 基础字幕赛道 (TRACK 1)]")
    print("="*50)
    print(f"字错误率 (CER): {cer:.2%}")
    print(f"时间轴平均偏差: 起始点 {avg_start_diff:.2f}ms, 结束点 {avg_end_diff:.2f}ms")

    # Track 1 综合得分
    time_penalty = min((avg_start_diff + avg_end_diff) / 2000, 1.0)
    final_score = (1 - cer) * 0.8 + (1 - time_penalty) * 0.2
    final_score = max(final_score, 0.0)
    print(f"综合得分: {final_score:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCL26-Eval 基础字幕评测脚本 (Track 1)")
    parser.add_argument("--pred", required=True, help="选手的预测 JSON 文件")
    parser.add_argument("--gt", required=True, help="官方 Ground Truth JSON 文件")
    args = parser.parse_args()
    evaluate(args.pred, args.gt)
