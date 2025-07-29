# -*- coding: utf-8 -*-
"""
福彩3D推荐结果验证与奖金计算器
=================================

本脚本旨在自动评估 `fc3d_analyzer.py` 生成的推荐号码的实际表现。

工作流程:
1.  读取 `fc3d.csv` 文件，获取所有历史开奖数据。
2.  确定最新的一期为"评估期"，倒数第二期为"报告数据截止期"。
3.  根据"报告数据截止期"，在当前目录下查找对应的分析报告文件
    (fc3d_analysis_output_*.txt)。
4.  从找到的报告中解析出"直选推荐"的号码。
5.  使用"评估期"的实际开奖号码，核对所有推荐投注的中奖情况。
6.  计算总奖金，并将详细的中奖结果（包括中奖号码、奖级、金额）
    追加记录到主报告文件 `latest_fc3d_calculation.txt` 中。
7.  主报告文件会自动管理记录数量，只保留最新的N条评估记录和错误日志。
"""

import os
import re
import glob
import csv
from datetime import datetime
import traceback
from typing import Optional, Tuple, List, Dict
from collections import Counter

# ==============================================================================
# --- 配置区 ---
# ==============================================================================

# 脚本需要查找的分析报告文件名的模式
REPORT_PATTERN = "fc3d_analysis_output_*.txt"
# 开奖数据源CSV文件
CSV_FILE = "fc3d.csv"
# 最终生成的主评估报告文件名
MAIN_REPORT_FILE = "latest_fc3d_calculation.txt"

# 主报告文件中保留的最大记录数
MAX_NORMAL_RECORDS = 10  # 保留最近10次评估
MAX_ERROR_LOGS = 20      # 保留最近20条错误日志

# 奖金对照表 (元) - 更新为官方最新标准
PRIZE_TABLE = {
    '直选': 1040,   # 单选投注（包括豹子号）
    '组三': 346,    # 组选3中奖
    '组六': 173,    # 组选6中奖
}

# ==============================================================================
# --- 工具函数 ---
# ==============================================================================

def log_message(message: str, level: str = "INFO"):
    """一个简单的日志打印函数，用于在控制台显示脚本执行状态。"""
    print(f"[{level}] {datetime.now().strftime('%H:%M:%S')} - {message}")

def robust_file_read(file_path: str) -> Optional[str]:
    """一个健壮的文件读取函数，能自动尝试多种编码格式。"""
    if not os.path.exists(file_path):
        log_message(f"文件未找到: {file_path}", "ERROR")
        return None
    encodings = ['utf-8', 'gbk', 'latin-1']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, IOError):
            continue
    log_message(f"无法使用任何支持的编码打开文件: {file_path}", "ERROR")
    return None

# ==============================================================================
# --- 数据解析与查找模块 ---
# ==============================================================================

def get_period_data_from_csv(csv_content: str) -> Tuple[Optional[Dict], Optional[List]]:
    """从CSV文件内容中解析出所有期号的开奖数据。"""
    if not csv_content:
        log_message("输入的CSV内容为空。", "WARNING")
        return None, None
    period_map, periods_list = {}, []
    try:
        reader = csv.reader(csv_content.splitlines())
        next(reader)  # 跳过表头
        for i, row in enumerate(reader):
            if len(row) >= 3 and re.match(r'^\d{4,7}$', row[0]):
                try:
                    period, date, number = row[0], row[1], row[2]
                    # 验证3D号码格式
                    if len(number) != 3 or not number.isdigit():
                        continue
                    period_map[period] = {'date': date, 'number': number}
                    periods_list.append(period)
                except (ValueError, IndexError):
                    log_message(f"CSV文件第 {i+2} 行数据格式无效，已跳过: {row}", "WARNING")
    except Exception as e:
        log_message(f"解析CSV数据时发生严重错误: {e}", "ERROR")
        return None, None
    
    if not period_map:
        log_message("未能从CSV中解析到任何有效的开奖数据。", "WARNING")
        return None, None
        
    return period_map, sorted(periods_list, key=int)

def find_matching_report(target_period: str) -> Optional[str]:
    """在当前目录查找其数据截止期与 `target_period` 匹配的最新分析报告。"""
    log_message(f"正在查找数据截止期为 {target_period} 的分析报告...")
    candidates = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for file_path in glob.glob(os.path.join(script_dir, REPORT_PATTERN)):
        content = robust_file_read(file_path)
        if not content:
            continue
        
        match = re.search(r'分析基于数据:\s*截至\s*(\d+)\s*期', content)
        if match and match.group(1) == target_period:
            try:
                # 从文件名中提取时间戳以确定最新报告
                timestamp_str_match = re.search(r'_(\d{8}_\d{6})\.txt$', file_path)
                if timestamp_str_match:
                    timestamp_str = timestamp_str_match.group(1)
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    candidates.append((timestamp, file_path))
            except (AttributeError, ValueError):
                continue
    
    if not candidates:
        log_message(f"未找到数据截止期为 {target_period} 的分析报告。", "WARNING")
        return None
        
    candidates.sort(reverse=True)
    latest_report = candidates[0][1]
    log_message(f"找到匹配的最新报告: {os.path.basename(latest_report)}", "INFO")
    return latest_report

def parse_recommendations_from_report(content: str) -> List[str]:
    """从分析报告内容中解析出直选推荐号码。"""
    # 解析直选推荐 - 格式为 "注 1: 123 (综合分: 85.67)"
    rec_pattern = re.compile(r'注\s*\d+:\s*(\d{3})')
    recommendations = []
    for match in rec_pattern.finditer(content):
        try:
            number = match.group(1)
            if len(number) == 3:
                recommendations.append(number)
        except ValueError:
            continue
    
    log_message(f"解析到 {len(recommendations)} 注直选推荐。")
    return recommendations

def parse_hot_digits_from_report(report_content: str) -> Dict[str, List[str]]:
    """从分析报告中解析出各位置热门数字。
    
    Args:
        report_content: 报告文件内容
    
    Returns:
        包含各位置热门数字的字典，键为 'hundreds', 'tens', 'units'
    """
    hot_digits = {'hundreds': [], 'tens': [], 'units': []}
    
    if not report_content:
        return hot_digits
    
    try:
        # 使用正则表达式解析各位置热门数字
        hundreds_match = re.search(r'百位 \(Top \d+\): ([\d\s]+)', report_content)
        tens_match = re.search(r'十位 \(Top \d+\): ([\d\s]+)', report_content)
        units_match = re.search(r'个位 \(Top \d+\): ([\d\s]+)', report_content)
        
        if hundreds_match:
            hot_digits['hundreds'] = hundreds_match.group(1).strip().split()
        if tens_match:
            hot_digits['tens'] = tens_match.group(1).strip().split()
        if units_match:
            hot_digits['units'] = units_match.group(1).strip().split()
            
        log_message(f"解析热门数字: 百位{len(hot_digits['hundreds'])}个, 十位{len(hot_digits['tens'])}个, 个位{len(hot_digits['units'])}个")
        
    except Exception as e:
        log_message(f"解析热门数字时出现错误: {e}", "WARNING")
    
    return hot_digits

def generate_complex_combinations(hot_digits: Dict[str, List[str]]) -> List[str]:
    """根据各位置热门数字生成大复式投注组合。
    
    Args:
        hot_digits: 各位置热门数字字典
    
    Returns:
        生成的所有组合列表
    """
    combinations = []
    
    try:
        from itertools import product
        
        hundreds = hot_digits.get('hundreds', [])
        tens = hot_digits.get('tens', [])
        units = hot_digits.get('units', [])
        
        if hundreds and tens and units:
            # 生成所有可能的组合
            for h in hundreds:
                for t in tens:
                    for u in units:
                        combo = f"{h}{t}{u}"
                        if len(combo) == 3 and combo.isdigit():
                            combinations.append(combo)
        
        log_message(f"生成大复式组合 {len(combinations)} 注")
        
    except Exception as e:
        log_message(f"生成大复式组合时出现错误: {e}", "WARNING")
    
    return combinations

# ==============================================================================
# --- 奖金计算与报告生成模块 ---
# ==============================================================================

def get_prize_level(predicted: str, actual: str) -> Optional[str]:
    """根据预测号码和实际开奖号码，确定中奖等级。"""
    if len(predicted) != 3 or len(actual) != 3:
        return None
    
    if predicted == actual:
        return "直选"
    
    # 判断组选
    pred_sorted = ''.join(sorted(predicted))
    actual_sorted = ''.join(sorted(actual))
    
    if pred_sorted == actual_sorted:
        # 判断是组三还是组六
        pred_counter = Counter(predicted)
        if 2 in pred_counter.values():
            return "组三"
        else:
            return "组六"
    
    return None

def calculate_prize(recommendations: List[str], actual_number: str) -> Tuple[int, Dict, List]:
    """计算给定推荐列表的总奖金、奖级分布和中奖详情。"""
    breakdown = {level: 0 for level in PRIZE_TABLE}
    total_prize = 0
    winning_details = []

    for predicted in recommendations:
        prize_level = get_prize_level(predicted, actual_number)
        
        if prize_level and prize_level in PRIZE_TABLE:
            prize_amount = PRIZE_TABLE[prize_level]
            total_prize += prize_amount
            breakdown[prize_level] += 1
            winning_details.append({'predicted': predicted, 'level': prize_level})
            
    return total_prize, breakdown, winning_details

def format_winning_numbers_for_report(winning_list: List[Dict], actual_number: str) -> List[str]:
    """格式化中奖号码，高亮命中的数字，用于报告输出。"""
    formatted_lines = []
    for item in winning_list:
        predicted = item['predicted']
        level = item['level']
        
        # 高亮显示匹配的位置
        formatted_pred = ""
        for i, (p, a) in enumerate(zip(predicted, actual_number)):
            if p == a:
                formatted_pred += f"**{p}**"
            else:
                formatted_pred += p
        
        formatted_lines.append(f"  - 预测: {formatted_pred} 实际: {actual_number} -> {level}")
    return formatted_lines

def manage_report(new_entry: Optional[Dict] = None, new_error: Optional[str] = None):
    """维护主评估报告文件，自动追加新记录并清理旧记录。"""
    normal_marker, error_marker = "==== 评估记录 ====", "==== 错误日志 ===="
    content_str = robust_file_read(MAIN_REPORT_FILE) or ""
    
    # 分割文件内容为记录块和错误日志
    parts = content_str.split(error_marker)
    normal_part = parts[0]
    error_part = parts[1] if len(parts) > 1 else ""
    
    # 解析现有记录
    normal_entries = [entry.strip() for entry in normal_part.split('='*20) if entry.strip() and normal_marker not in entry]
    error_entries = [err.strip() for err in error_part.splitlines() if err.strip()]

    # 添加新记录
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if new_entry:
        # 计算总奖金（直选+大复式）
        total_prize = new_entry['rec_prize'] + new_entry.get('complex_prize', 0)
        
        entry_lines = [
            f"评估时间: {timestamp}",
            f"评估期号 (实际开奖): {new_entry['eval_period']}",
            f"分析报告数据截止期: {new_entry['report_cutoff_period']}",
            f"开奖号码: {new_entry['actual_number']}",
            f"直选奖金: {new_entry['rec_prize']:,} 元",
            f"复式奖金: {new_entry.get('complex_prize', 0):,} 元",
            f"总奖金: {total_prize:,} 元",
            "", "--- 直选推荐详情 ---"
        ]
        
        rec_prize, rec_bd, rec_winners = new_entry['rec_prize'], new_entry['rec_breakdown'], new_entry['rec_winners']
        if rec_prize > 0:
            entry_lines.append(f"奖金: {rec_prize:,}元 | 明细: " + ", ".join(f"{k}x{v}" for k,v in rec_bd.items() if v>0))
            entry_lines.extend(format_winning_numbers_for_report(rec_winners, new_entry['actual_number']))
        else:
            entry_lines.append("未中奖")
        
        # 添加大复式验证信息
        complex_count = new_entry.get('complex_count', 0)
        if complex_count > 0:
            entry_lines.append("")
            entry_lines.append("--- 大复式验证详情 ---")
            entry_lines.append(f"复式注数: {complex_count} 注")
            
            complex_prize = new_entry.get('complex_prize', 0)
            complex_bd = new_entry.get('complex_breakdown', {})
            complex_winners = new_entry.get('complex_winners', [])
            
            if complex_prize > 0:
                entry_lines.append(f"复式奖金: {complex_prize:,}元 | 明细: " + ", ".join(f"{k}x{v}" for k,v in complex_bd.items() if v>0))
                entry_lines.extend(format_winning_numbers_for_report(complex_winners, new_entry['actual_number']))
            else:
                entry_lines.append("复式未中奖")
        
        normal_entries.insert(0, "\n".join(entry_lines))

    if new_error:
        error_entries.insert(0, f"[{timestamp}] {new_error}")

    # 清理旧记录
    final_normal_entries = normal_entries[:MAX_NORMAL_RECORDS]
    final_error_entries = error_entries[:MAX_ERROR_LOGS]

    # 写回文件
    try:
        with open(MAIN_REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"{normal_marker}\n")
            if final_normal_entries:
                f.write(("\n" + "="*20 + "\n").join(final_normal_entries))
            f.write(f"\n\n{error_marker}\n")
            if final_error_entries:
                f.write("\n".join(final_error_entries))
        log_message(f"主报告已更新: {MAIN_REPORT_FILE}", "INFO")
    except IOError as e:
        log_message(f"写入主报告文件失败: {e}", "ERROR")

# ==============================================================================
# --- 主流程 ---
# ==============================================================================

def main_process():
    """主处理流程，串联所有功能。"""
    log_message("====== 主流程启动 ======", "INFO")
    
    csv_content = robust_file_read(CSV_FILE)
    if not csv_content:
        manage_report(new_error=f"无法读取或未找到CSV数据文件: {CSV_FILE}")
        return

    period_map, sorted_periods = get_period_data_from_csv(csv_content)
    if not period_map or not sorted_periods or len(sorted_periods) < 2:
        manage_report(new_error="CSV数据不足两期或解析失败，无法进行评估。")
        return

    eval_period = sorted_periods[-1]
    report_cutoff_period = sorted_periods[-2]
    log_message(f"评估期号: {eval_period}, 报告数据截止期: {report_cutoff_period}", "INFO")

    report_path = find_matching_report(report_cutoff_period)
    if not report_path:
        manage_report(new_error=f"未找到数据截止期为 {report_cutoff_period} 的分析报告。")
        return

    report_content = robust_file_read(report_path)
    if not report_content:
        manage_report(new_error=f"无法读取分析报告文件: {report_path}")
        return

    recommendations = parse_recommendations_from_report(report_content)
    
    if not recommendations:
         manage_report(new_error=f"未能从报告 {os.path.basename(report_path)} 中解析出任何有效推荐。")
         return
    
    # 解析热门数字并生成大复式组合
    hot_digits = parse_hot_digits_from_report(report_content)
    complex_combinations = generate_complex_combinations(hot_digits)
    
    prize_data = period_map[eval_period]
    actual_number = prize_data['number']
    log_message(f"获取到期号 {eval_period} 的开奖数据: {actual_number}", "INFO")
    
    # 计算单式推荐中奖情况
    rec_prize, rec_bd, rec_winners = calculate_prize(recommendations, actual_number)
    
    # 计算大复式中奖情况
    complex_prize, complex_bd, complex_winners = calculate_prize(complex_combinations, actual_number) if complex_combinations else (0, {}, [])
    
    report_entry = {
        'eval_period': eval_period,
        'report_cutoff_period': report_cutoff_period,
        'actual_number': actual_number,
        'total_prize': rec_prize,
        'rec_prize': rec_prize,
        'rec_breakdown': rec_bd,
        'rec_winners': rec_winners,
        'complex_prize': complex_prize,
        'complex_breakdown': complex_bd,
        'complex_winners': complex_winners,
        'complex_count': len(complex_combinations),
    }
    manage_report(new_entry=report_entry)
    
    # 发送微信推送
    try:
        from fc3d_wxpusher import send_verification_report
        log_message("正在发送验证报告微信推送...", "INFO")
        
        push_result = send_verification_report(report_entry)
        
        if push_result.get('success'):
            log_message("验证报告微信推送发送成功", "INFO")
        else:
            log_message(f"验证报告微信推送发送失败: {push_result.get('error', '未知错误')}", "WARNING")
            
    except ImportError:
        log_message("微信推送模块未找到，跳过推送功能", "WARNING")
    except Exception as e:
        log_message(f"验证报告微信推送发送异常: {e}", "ERROR")
    
    log_message(f"处理完成！总计奖金: {report_entry['total_prize']:,}元。", "INFO")
    log_message("====== 主流程结束 ======", "INFO")

if __name__ == "__main__":
    try:
        main_process()
    except Exception as e:
        tb_str = traceback.format_exc()
        error_message = f"主流程发生未捕获的严重异常: {type(e).__name__} - {e}\n{tb_str}"
        log_message(error_message, "CRITICAL")
        try:
            manage_report(new_error=error_message)
        except Exception as report_e:
            log_message(f"在记录严重错误时再次发生错误: {report_e}", "CRITICAL") 