# -*- coding: utf-8 -*-
"""
福彩3D微信推送模块
================

提供微信推送功能，用于推送福彩3D分析报告和验证报告
"""

import requests
import logging
import json
import os
from datetime import datetime
from typing import Optional, List, Dict
from collections import Counter

# 微信推送配置
# 支持从环境变量读取配置（用于GitHub Actions等CI环境）
APP_TOKEN = os.getenv("WXPUSHER_APP_TOKEN", "AT_FInZJJ0mUU8xvQjKRP7v6omvuHN3Fdqw")
USER_UIDS = os.getenv("WXPUSHER_USER_UIDS", "UID_yYObqdMVScIa66DGR2n2PCRFL10w").split(",")
TOPIC_IDS = [int(x) for x in os.getenv("WXPUSHER_TOPIC_IDS", "39909").split(",") if x.strip()]

def get_latest_verification_result() -> Optional[Dict]:
    """获取最新的验证结果
    
    Returns:
        最新验证结果字典，如果没有则返回None
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        calc_file = os.path.join(script_dir, 'latest_fc3d_calculation.txt')
        
        if not os.path.exists(calc_file):
            return None
            
        with open(calc_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析最新的验证记录
        lines = content.split('\n')
        
        # 查找最新的评估记录
        for i, line in enumerate(lines):
            if line.startswith('评估时间:'):
                # 解析评估信息
                result = {}
                
                # 解析期号和开奖号码
                for j in range(i, min(i+20, len(lines))):
                    if lines[j].startswith('评估期号'):
                        result['eval_period'] = lines[j].split(':')[1].strip().split()[0]
                    elif lines[j].startswith('开奖号码:'):
                        # 解析开奖号码: 123
                        draw_line = lines[j]
                        try:
                            number = draw_line.split(':')[1].strip()
                            if len(number) == 3 and number.isdigit():
                                result['actual_number'] = number
                        except:
                            pass
                    elif lines[j].startswith('总奖金:'):
                        try:
                            amount_str = lines[j].split(':')[1].strip().replace('元', '').replace(',', '')
                            result['total_prize'] = int(amount_str) if amount_str.isdigit() else 0
                        except:
                            result['total_prize'] = 0
                
                return result if result else None
                
        return None
        
    except Exception as e:
        logging.error(f"获取最新验证结果失败: {e}")
        return None

def send_wxpusher_message(content: str, title: str = None, topicIds: List[int] = None, uids: List[str] = None) -> Dict:
    """发送微信推送消息
    
    Args:
        content: 消息内容
        title: 消息标题
        topicIds: 主题ID列表，默认使用全局配置
        uids: 用户ID列表，默认使用全局配置
    
    Returns:
        API响应结果字典
    """
    url = "https://wxpusher.zjiecode.com/api/send/message"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "appToken": APP_TOKEN,
        "content": content,
        "uids": uids or USER_UIDS,
        "topicIds": topicIds or TOPIC_IDS,
        "summary": title or "福彩3D推荐更新",
        "contentType": 1,  # 1=文本，2=HTML
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if result.get("success", False):
            logging.info(f"微信推送成功: {title}")
            return {"success": True, "data": result}
        else:
            logging.error(f"微信推送失败: {result.get('msg', '未知错误')}")
            return {"success": False, "error": result.get('msg', '推送失败')}
            
    except requests.exceptions.RequestException as e:
        logging.error(f"微信推送网络错误: {e}")
        return {"success": False, "error": f"网络错误: {str(e)}"}
    except Exception as e:
        logging.error(f"微信推送异常: {e}")
        return {"success": False, "error": f"未知异常: {str(e)}"}

def send_analysis_report(report_content: str, period: int, recommendations: List[str], 
                         optuna_summary: Dict = None, backtest_stats: Dict = None) -> Dict:
    """发送福彩3D分析报告
    
    Args:
        report_content: 完整的分析报告内容
        period: 预测期号
        recommendations: 推荐号码列表
        optuna_summary: Optuna优化摘要
        backtest_stats: 回测统计数据
    
    Returns:
        推送结果字典
    """
    title = f"🎯 福彩3D第{period}期预测报告"
    
    # 提取关键信息制作详细版推送
    try:
        # 获取最新验证结果
        latest_verification = get_latest_verification_result()
        
        # 构建直选推荐内容 - 显示所有推荐号码，采用紧凑格式
        rec_summary = ""
        if recommendations:
            for i, rec in enumerate(recommendations):
                # 提取号码部分，简化格式
                import re
                number_match = re.search(r'(\d{3})', rec)
                
                if number_match:
                    number = number_match.group(1)
                    rec_summary += f"第{i+1:2d}注: {number}\n"
                else:
                    # 如果解析失败，使用原始格式但简化
                    rec_summary += f"第{i+1:2d}注: {rec}\n"
        
        # 构建优化信息
        optuna_info = ""
        if optuna_summary and optuna_summary.get('status') == '完成':
            optuna_info = f"🔬 Optuna优化得分: {optuna_summary.get('best_value', 0):.2f}\n"
        
        # 构建回测信息
        backtest_info = ""
        if backtest_stats:
            prize_counts = backtest_stats.get('prize_counts', {})
            if prize_counts:
                prize_info = []
                for prize, count in prize_counts.items():
                    if count > 0:
                        prize_info.append(f"{prize}x{count}")
                if prize_info:
                    backtest_info = f"📊 最近回测: {', '.join(prize_info)}\n"
        
        # 构建最新验证结果摘要
        verification_summary = ""
        if latest_verification:
            verification_summary = f"""
📅 最新验证（第{latest_verification.get('eval_period', '未知')}期）：
🎱 开奖: {latest_verification.get('actual_number', '未知')}
💰 中奖: {latest_verification.get('total_prize', 0)}元
"""
        
        # 提取各位置热门数字
        hot_digits_info = ""
        try:
            # 从报告内容中解析各位置热门数字
            import re
            hundreds_match = re.search(r'百位 \(Top \d+\): ([\d\s]+)', report_content)
            tens_match = re.search(r'十位 \(Top \d+\): ([\d\s]+)', report_content)  
            units_match = re.search(r'个位 \(Top \d+\): ([\d\s]+)', report_content)
            
            if hundreds_match and tens_match and units_match:
                hundreds = hundreds_match.group(1).strip().replace(' ', ' ')
                tens = tens_match.group(1).strip().replace(' ', ' ')
                units = units_match.group(1).strip().replace(' ', ' ')
                
                hot_digits_info = f"""
🔥 各位置热门数字：
百位: {hundreds}
十位: {tens}  
个位: {units}

"""
        except Exception as e:
            logging.debug(f"解析热门数字时出现错误: {e}")
        
        # 构建推送内容
        content = f"""🎯 福彩3D第{period}期AI智能预测

📊 直选推荐 (共{len(recommendations)}注)：
{rec_summary.strip()}
{hot_digits_info}{verification_summary}
📈 分析要点：
• 基于机器学习LightGBM算法
• 结合历史频率和遗漏分析  
• 运用关联规则挖掘技术
• 多因子加权评分优选
{optuna_info}{backtest_info}
⏰ 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

💡 仅供参考，理性投注！祝您好运！"""
        
        return send_wxpusher_message(content, title)
        
    except Exception as e:
        logging.error(f"构建分析报告推送内容失败: {e}")
        return {"success": False, "error": f"内容构建失败: {str(e)}"}

def send_verification_report(verification_data: Dict) -> Dict:
    """发送福彩3D验证报告
    
    Args:
        verification_data: 验证报告数据字典，包含中奖信息
    
    Returns:
        推送结果字典
    """
    try:
        period = verification_data.get('eval_period', '未知')
        title = f"✅ 福彩3D第{period}期验证报告"
        
        actual_number = verification_data.get('actual_number', '未知')
        rec_prize = verification_data.get('rec_prize', 0)
        total_prize = verification_data.get('total_prize', 0)
        
        # 构建单式中奖统计
        rec_breakdown = verification_data.get('rec_breakdown', {})
        
        rec_summary = "无中奖"
        if rec_prize > 0:
            rec_details = []
            for level, count in rec_breakdown.items():
                if count > 0:
                    rec_details.append(f"{level}x{count}")
            rec_summary = ", ".join(rec_details) if rec_details else "中奖但无详情"
        
        # 构建大复式中奖统计
        complex_prize = verification_data.get('complex_prize', 0)
        complex_breakdown = verification_data.get('complex_breakdown', {})
        complex_count = verification_data.get('complex_count', 0)
        
        complex_summary = "无中奖"
        if complex_prize > 0:
            complex_details = []
            for level, count in complex_breakdown.items():
                if count > 0:
                    complex_details.append(f"{level}x{count}")
            complex_summary = ", ".join(complex_details) if complex_details else "中奖但无详情"
        
        # 计算总投注数
        rec_count = len(verification_data.get('rec_winners', []))
        total_bets = rec_count + complex_count  # 单式 + 大复式
        
        # 构建验证报告内容
        complex_info = ""
        if complex_count > 0:
            complex_info = f"""
🎯 大复式验证（{complex_count}注）：
{complex_summary}
大复式奖金：{complex_prize:,}元
"""
        
        content = f"""✅ 福彩3D第{period}期开奖验证

🎱 开奖号码：{actual_number}

📊 验证结果：
直选推荐（{rec_count}注）：{rec_summary}
直选奖金：{rec_prize:,}元
{complex_info}
💰 总奖金：{total_prize + complex_prize:,}元

📈 投资回报：
估算成本：{total_bets * 2:,}元（按单注2元计算）
收益：{total_prize + complex_prize - total_bets * 2:,}元
回报率：{((total_prize + complex_prize - total_bets * 2) / (total_bets * 2) * 100):.2f}%

⏰ 验证时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}"""
        
        return send_wxpusher_message(content, title)
        
    except Exception as e:
        logging.error(f"构建验证报告推送内容失败: {e}")
        return {"success": False, "error": f"内容构建失败: {str(e)}"}

def send_error_notification(error_msg: str, script_name: str = "福彩3D系统") -> Dict:
    """发送错误通知
    
    Args:
        error_msg: 错误信息
        script_name: 脚本名称
    
    Returns:
        推送结果字典
    """
    title = f"⚠️ {script_name}运行异常"
    
    content = f"""⚠️ 系统运行异常通知

📍 异常位置：{script_name}
🕒 发生时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
❌ 错误信息：
{error_msg}

请及时检查系统状态！"""
    
    return send_wxpusher_message(content, title)

def send_daily_summary(analysis_success: bool, verification_success: bool, 
                      analysis_file: str = None, error_msg: str = None) -> Dict:
    """发送每日运行摘要
    
    Args:
        analysis_success: 分析是否成功
        verification_success: 验证是否成功
        analysis_file: 分析报告文件名
        error_msg: 错误信息（如有）
    
    Returns:
        推送结果字典
    """
    title = "📊 福彩3D系统日报"
    
    # 状态图标
    analysis_status = "✅" if analysis_success else "❌"
    verification_status = "✅" if verification_success else "❌"
    
    content = f"""📊 福彩3D AI预测系统日报

🕒 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

📈 任务执行状态：
{analysis_status} 数据分析与预测
{verification_status} 历史验证计算

📁 生成文件："""
    
    if analysis_file:
        content += f"\n• {analysis_file}"
    
    if error_msg:
        content += f"\n\n⚠️ 异常信息：\n{error_msg}"
    
    content += "\n\n🔔 系统已自动完成定时任务"
    
    return send_wxpusher_message(content, title)

def send_complete_recommendations(period: int, recommendations: List[str]) -> Dict:
    """发送完整的推荐号码列表
    
    Args:
        period: 预测期号
        recommendations: 推荐号码列表
    
    Returns:
        推送结果字典
    """
    try:
        # 获取最新验证结果
        latest_verification = get_latest_verification_result()
        
        # 构建验证结果摘要
        verification_summary = ""
        if latest_verification:
            verification_summary = f"""
📅 最新验证（第{latest_verification.get('eval_period', '未知')}期）：
🎱 开奖: {latest_verification.get('actual_number', '未知')}
💰 中奖: {latest_verification.get('total_prize', 0)}元
"""
        
        # 构建完整推荐内容
        content_parts = [f"🎯 福彩3D第{period}期完整推荐"]
        
        if verification_summary:
            content_parts.append(verification_summary.strip())
        
        content_parts.append(f"📊 全部{len(recommendations)}注直选推荐：")
        
        # 显示所有推荐号码
        rec_lines = []
        for i, rec in enumerate(recommendations):
            import re
            number_match = re.search(r'(\d{3})', rec)
            
            if number_match:
                number = number_match.group(1)
                rec_lines.append(f"{i+1:2d}. {number}")
            else:
                rec_lines.append(f"{i+1:2d}. {rec}")
        
        # 分两部分显示（前5注和后5注）
        mid_point = len(rec_lines) // 2
        content_parts.append("前半部分：")
        content_parts.extend(rec_lines[:mid_point])
        content_parts.append("\n后半部分：")
        content_parts.extend(rec_lines[mid_point:])
        
        content_parts.extend([
            "",
            f"⏰ 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "💡 仅供参考，理性投注！"
        ])
        
        # 合并所有内容
        full_content = '\n'.join(content_parts)
        
        title = f"🎯 福彩3D第{period}期完整推荐"
        
        return send_wxpusher_message(full_content, title)
        
    except Exception as e:
        logging.error(f"构建完整推荐推送内容失败: {e}")
        return {"success": False, "error": f"内容构建失败: {str(e)}"}

def test_wxpusher_connection() -> bool:
    """测试微信推送连接
    
    Returns:
        连接是否成功
    """
    test_content = f"🔧 福彩3D推送系统测试\n\n测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n如收到此消息，说明推送功能正常！"
    result = send_wxpusher_message(test_content, "🔧 推送测试")
    return result.get("success", False)

if __name__ == "__main__":
    # 测试推送功能
    print("正在测试福彩3D微信推送功能...")
    
    # 测试基本推送
    if test_wxpusher_connection():
        print("✅ 微信推送测试成功！")
        
        # 测试分析报告推送
        test_recommendations = [
            "注 1: 123 (综合分: 89.67)",
            "注 2: 456 (综合分: 85.34)",
            "注 3: 789 (综合分: 82.15)",
            "注 4: 012 (综合分: 78.92)",
            "注 5: 345 (综合分: 76.88)",
            "注 6: 678 (综合分: 74.26)",
            "注 7: 901 (综合分: 72.44)",
            "注 8: 234 (综合分: 70.17)",
            "注 9: 567 (综合分: 68.93)",
            "注 10: 890 (综合分: 67.25)"
        ]
        
        print("测试分析报告推送...")
        send_analysis_report(
            "测试报告内容", 
            2025001, 
            test_recommendations[:5]  # 摘要只显示前5注
        )
        
        print("测试完整推荐推送...")
        send_complete_recommendations(
            2025001, 
            test_recommendations  # 所有10注
        )
        
        print("测试验证报告推送...")
        test_verification = {
            'eval_period': 2024365,
            'actual_number': '123',
            'total_prize': 1000,
            'rec_prize': 1000,
            'rec_breakdown': {'直选': 1},
            'rec_winners': [{'predicted': '123', 'level': '直选'}]
        }
        send_verification_report(test_verification)
        
    else:
        print("❌ 微信推送测试失败！请检查配置。") 