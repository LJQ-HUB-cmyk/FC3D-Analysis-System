# -*- coding: utf-8 -*-
"""
福彩3D彩票数据分析与推荐系统
================================

本脚本整合了统计分析、机器学习和策略化组合生成，为福彩3D彩票提供数据驱动的
号码推荐。脚本支持两种运行模式，由全局变量 `ENABLE_OPTUNA_OPTIMIZATION` 控制：

1.  **分析模式 (默认 `False`)**:
    使用内置的 `DEFAULT_WEIGHTS` 权重，执行一次完整的历史数据分析、策略回测，
    并为下一期生成推荐号码。所有结果会输出到一个带时间戳的详细报告文件中。

2.  **优化模式 (`True`)**:
    在分析前，首先运行 Optuna 框架进行参数搜索，以找到在近期历史数据上
    表现最佳的一组权重。然后，自动使用这组优化后的权重来完成后续的分析、
    回测和推荐。优化过程和结果也会记录在报告中。

版本: 1.0 (福彩3D专版)
"""

# --- 标准库导入 ---
import os
import sys
import json
import time
import datetime
import logging
import io
import random
from collections import Counter
from contextlib import redirect_stdout
from typing import (Union, Optional, List, Dict, Tuple, Any)
from functools import partial
from itertools import product

# --- 第三方库导入 ---
import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import concurrent.futures

# ==============================================================================
# --- 全局常量与配置 ---
# ==============================================================================

# --------------------------
# --- 路径与模式配置 ---
# --------------------------
# 脚本文件所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 原始福彩3D数据CSV文件路径 (由 fc3d_data_processor.py 生成)
CSV_FILE_PATH = os.path.join(SCRIPT_DIR, 'fc3d.csv')
# 预处理后的数据缓存文件路径，避免每次都重新计算特征
PROCESSED_CSV_PATH = os.path.join(SCRIPT_DIR, 'fc3d_processed.csv')

# 运行模式配置:
# True  -> 运行参数优化，耗时较长，但可能找到更优策略。
# False -> 使用默认权重进行快速分析和推荐。
ENABLE_OPTUNA_OPTIMIZATION = False

# --------------------------
# --- 策略开关配置 ---
# --------------------------
# 是否启用最终推荐组合层面的"反向思维"策略 (移除得分最高的几注)
ENABLE_FINAL_COMBO_REVERSE = False
# 在启用反向思维并移除组合后，是否从候选池中补充新的组合以达到目标数量
ENABLE_REVERSE_REFILL = True

# --------------------------
# --- 福彩3D规则配置 ---
# --------------------------
# 每个位置的数字范围 (0到9)
DIGIT_RANGE = range(0, 10)
# 位置名称
POSITION_NAMES = ['百位', '十位', '个位']

# --------------------------
# --- 分析与执行参数配置 ---
# --------------------------
# 机器学习模型使用的滞后特征阶数 (e.g., 使用前1、3、5、10期的数据作为特征)
ML_LAG_FEATURES = [1, 3, 5, 10]
# 用于生成乘积交互特征的特征对
ML_INTERACTION_PAIRS = [('sum_value', 'span_value')]
# 用于生成自身平方交互特征的特征
ML_INTERACTION_SELF = ['span_value']
# 计算号码"近期"出现频率时所参考的期数窗口大小
RECENT_FREQ_WINDOW = 20
# 在分析模式下，进行策略回测时所评估的总期数
BACKTEST_PERIODS_COUNT = 100
# 在优化模式下，每次试验用于快速评估性能的回测期数 (数值越小优化越快)
OPTIMIZATION_BACKTEST_PERIODS = 20
# 在优化模式下，Optuna 进行参数搜索的总试验次数
OPTIMIZATION_TRIALS = 100
# 训练机器学习模型时，一个数字在历史数据中至少需要出现的次数
MIN_POSITIVE_SAMPLES_FOR_ML = 25

# ==============================================================================
# --- 默认权重配置 (这些参数可被Optuna优化) ---
# ==============================================================================
DEFAULT_WEIGHTS = {
    # --- 反向思维 ---
    'FINAL_COMBO_REVERSE_REMOVE_TOP_PERCENT': 0.3,

    # --- 组合生成 ---
    'NUM_COMBINATIONS_TO_GENERATE': 10,
    'TOP_N_DIGITS_FOR_CANDIDATE': 8,  # 每个位置选取前N个候选数字

    # --- 数字评分权重 ---
    'FREQ_SCORE_WEIGHT': 25.0,
    'OMISSION_SCORE_WEIGHT': 20.0,
    'MAX_OMISSION_RATIO_SCORE_WEIGHT': 15.0,
    'RECENT_FREQ_SCORE_WEIGHT': 15.0,
    'ML_PROB_SCORE_WEIGHT': 25.0,

    # --- 组合属性匹配奖励 ---
    'COMBINATION_SUM_MATCH_BONUS': 10.0,
    'COMBINATION_SPAN_MATCH_BONUS': 8.0,
    'COMBINATION_ODD_COUNT_MATCH_BONUS': 12.0,
    'COMBINATION_PRIME_COUNT_MATCH_BONUS': 8.0,
    'COMBINATION_FORM_MATCH_BONUS': 15.0,  # 豹子、组三、组六

    # --- 关联规则挖掘(ARM)参数与奖励 ---
    'ARM_MIN_SUPPORT': 0.01,
    'ARM_MIN_CONFIDENCE': 0.5,
    'ARM_MIN_LIFT': 1.2,
    'ARM_COMBINATION_BONUS_WEIGHT': 12.0,
    'ARM_BONUS_LIFT_FACTOR': 0.3,
    'ARM_BONUS_CONF_FACTOR': 0.2,

    # --- 组合多样性控制 ---
    'DIVERSITY_MIN_DIFFERENT_DIGITS': 1,  # 任意两组合至少有几个位置不同
}

# ==============================================================================
# --- 机器学习模型参数配置 ---
# ==============================================================================
LGBM_PARAMS = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'n_estimators': 80,
    'num_leaves': 12,
    'min_child_samples': 10,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'n_jobs': 1,
    'verbose': -1,
}

# ==============================================================================
# --- 日志系统配置 ---
# ==============================================================================
console_formatter = logging.Formatter('%(message)s')
detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

logger = logging.getLogger('fc3d_analyzer')
logger.setLevel(logging.DEBUG)
logger.propagate = False

progress_logger = logging.getLogger('progress_logger')
progress_logger.setLevel(logging.INFO)
progress_logger.propagate = False

global_console_handler = logging.StreamHandler(sys.stdout)
global_console_handler.setFormatter(console_formatter)

progress_console_handler = logging.StreamHandler(sys.stdout)
progress_console_handler.setFormatter(logging.Formatter('%(message)s'))

logger.addHandler(global_console_handler)
progress_logger.addHandler(progress_console_handler)

def set_console_verbosity(level=logging.INFO, use_simple_formatter=False):
    """动态设置主日志记录器在控制台的输出级别和格式。"""
    global_console_handler.setLevel(level)
    global_console_handler.setFormatter(console_formatter if use_simple_formatter else detailed_formatter)

# ==============================================================================
# --- 核心工具函数 ---
# ==============================================================================

class SuppressOutput:
    """一个上下文管理器，用于临时抑制标准输出和捕获标准错误。"""
    def __init__(self, suppress_stdout=True, capture_stderr=True):
        self.suppress_stdout, self.capture_stderr = suppress_stdout, capture_stderr
        self.old_stdout, self.old_stderr, self.stdout_io, self.stderr_io = None, None, None, None
    def __enter__(self):
        if self.suppress_stdout: self.old_stdout, self.stdout_io, sys.stdout = sys.stdout, io.StringIO(), self.stdout_io
        if self.capture_stderr: self.old_stderr, self.stderr_io, sys.stderr = sys.stderr, io.StringIO(), self.stderr_io
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.capture_stderr and self.old_stderr:
            sys.stderr = self.old_stderr; captured = self.stderr_io.getvalue(); self.stderr_io.close()
            if captured.strip(): logger.warning(f"在一个被抑制的输出块中捕获到标准错误:\n{captured.strip()}")
        if self.suppress_stdout and self.old_stdout:
            sys.stdout = self.old_stdout; self.stdout_io.close()
        return False

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

def format_time(seconds: float) -> str:
    """将秒数格式化为易于阅读的 HH:MM:SS 字符串。"""
    if seconds < 0: return "00:00:00"
    hours, remainder = divmod(seconds, 3600)
    minutes, sec = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(sec):02d}"

# ==============================================================================
# --- 数据处理模块 ---
# ==============================================================================

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """从CSV文件加载数据，并能自动尝试多种常用编码格式。"""
    if not os.path.exists(file_path):
        logger.error(f"数据文件未找到: {file_path}")
        return None
    for enc in ['utf-8', 'gbk', 'latin-1']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"使用编码 {enc} 加载 {file_path} 时出错: {e}")
            return None
    logger.error(f"无法使用任何支持的编码打开文件 {file_path}。")
    return None

def clean_and_structure(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """清洗和结构化原始DataFrame，确保数据类型正确，并转换为"一行一期"的格式。"""
    if df is None or df.empty:
        return None
    
    required_cols = ['期号', '号码']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"输入数据缺少必要列: {required_cols}")
        return None

    df.dropna(subset=required_cols, inplace=True)
    try:
        df['期号'] = pd.to_numeric(df['期号'], errors='coerce')
        df.dropna(subset=['期号'], inplace=True)
        df = df.astype({'期号': int})
    except (ValueError, TypeError) as e:
        logger.error(f"转换'期号'为整数时失败: {e}")
        return None

    df.sort_values(by='期号', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    parsed_rows = []
    for _, row in df.iterrows():
        try:
            # 解析3D号码
            number_str = str(row['号码']).strip()
            if len(number_str) != 3 or not number_str.isdigit():
                logger.warning(f"期号 {row['期号']} 的号码无效，已跳过: {number_str}")
                continue
            
            # 分解为百位、十位、个位
            hundreds = int(number_str[0])
            tens = int(number_str[1])
            units = int(number_str[2])
            
            # 构建结构化的记录
            record = {
                '期号': row['期号'],
                'hundreds': hundreds,
                'tens': tens,
                'units': units,
                'number': number_str
            }
            if '日期' in row and pd.notna(row['日期']):
                record['日期'] = row['日期']
            parsed_rows.append(record)
        except (ValueError, TypeError, IndexError):
            logger.warning(f"解析期号 {row['期号']} 的号码时失败，已跳过。")
            continue
            
    return pd.DataFrame(parsed_rows) if parsed_rows else None

def feature_engineer(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """为DataFrame计算各种衍生特征，如和值、跨度、奇偶分布等。"""
    if df is None or df.empty:
        return None
    
    df_fe = df.copy()
    
    # 基本统计特征
    df_fe['sum_value'] = df_fe['hundreds'] + df_fe['tens'] + df_fe['units']
    df_fe['span_value'] = df_fe[['hundreds', 'tens', 'units']].max(axis=1) - df_fe[['hundreds', 'tens', 'units']].min(axis=1)
    
    # 奇偶特征
    df_fe['odd_count'] = (df_fe['hundreds'] % 2) + (df_fe['tens'] % 2) + (df_fe['units'] % 2)
    
    # 质数特征 (2,3,5,7是质数)
    def is_prime(n):
        return n in [2, 3, 5, 7]
    
    df_fe['prime_count'] = df_fe['hundreds'].apply(is_prime).astype(int) + \
                          df_fe['tens'].apply(is_prime).astype(int) + \
                          df_fe['units'].apply(is_prime).astype(int)
    
    # 大小特征 (>=5为大数)
    df_fe['large_count'] = (df_fe['hundreds'] >= 5).astype(int) + \
                          (df_fe['tens'] >= 5).astype(int) + \
                          (df_fe['units'] >= 5).astype(int)
    
    # 形态特征
    def get_form_type(row):
        digits = [row['hundreds'], row['tens'], row['units']]
        digit_counts = Counter(digits)
        
        if len(digit_counts) == 1:
            return 'leopard'  # 豹子 (111, 222, ...)
        elif len(digit_counts) == 2:
            return 'group3'   # 组三 (112, 121, ...)
        else:
            return 'group6'   # 组六 (123, 456, ...)
    
    df_fe['form_type'] = df_fe.apply(get_form_type, axis=1)
    
    # 重号特征 (与上一期的重复个数)
    prev_numbers = df_fe[['hundreds', 'tens', 'units']].shift(1)
    curr_numbers = df_fe[['hundreds', 'tens', 'units']]
    
    def count_repeats(curr_row, prev_row):
        if pd.isna(prev_row).any():
            return 0
        curr_set = set([curr_row['hundreds'], curr_row['tens'], curr_row['units']])
        prev_set = set([prev_row['hundreds'], prev_row['tens'], prev_row['units']])
        return len(curr_set.intersection(prev_set))
    
    df_fe['repeat_count'] = [count_repeats(curr_numbers.iloc[i], prev_numbers.iloc[i]) 
                            for i in range(len(df_fe))]
    
    return df_fe

def create_lagged_features(df: pd.DataFrame, lags: List[int]) -> Optional[pd.DataFrame]:
    """为机器学习模型创建滞后特征（将历史期的特征作为当前期的输入）和交互特征。"""
    if df is None or df.empty or not lags:
        return None
    
    feature_cols = [col for col in df.columns if col in [
        'sum_value', 'span_value', 'odd_count', 'prime_count', 'large_count', 'repeat_count'
    ]]
    df_features = df[feature_cols].copy()
    
    # 创建交互特征
    for c1, c2 in ML_INTERACTION_PAIRS:
        if c1 in df_features and c2 in df_features:
            df_features[f'{c1}_x_{c2}'] = df_features[c1] * df_features[c2]
    for c in ML_INTERACTION_SELF:
        if c in df_features:
            df_features[f'{c}_sq'] = df_features[c]**2
            
    # 创建滞后特征
    all_feature_cols = df_features.columns.tolist()
    lagged_dfs = [df_features[all_feature_cols].shift(lag).add_suffix(f'_lag{lag}') for lag in lags]
    final_df = pd.concat(lagged_dfs, axis=1)
    final_df.dropna(inplace=True)
    
    return final_df if not final_df.empty else None

# ==============================================================================
# --- 分析与评分模块 ---
# ==============================================================================

def analyze_frequency_omission(df: pd.DataFrame) -> dict:
    """分析所有数字在各个位置的频率、当前遗漏、平均遗漏、最大遗漏和近期频率。"""
    if df is None or df.empty:
        return {}
    
    total_periods = len(df)
    most_recent_idx = total_periods - 1
    
    # 初始化统计字典
    freq_stats = {}
    
    # 分析每个位置的每个数字
    for pos_idx, pos_name in enumerate(['hundreds', 'tens', 'units']):
        pos_col = pos_name
        
        # 频率计算
        digit_freq = Counter(df[pos_col])
        
        # 遗漏和近期频率计算
        current_omission = {}
        max_hist_omission = {}
        recent_freq = Counter()
        
        # 计算近期频率
        if total_periods >= RECENT_FREQ_WINDOW:
            recent_freq.update(df.tail(RECENT_FREQ_WINDOW)[pos_col])
        
        for digit in DIGIT_RANGE:
            # 找到该数字在该位置出现的所有索引
            app_indices = df.index[df[pos_col] == digit].tolist()
            
            if app_indices:
                current_omission[digit] = most_recent_idx - app_indices[-1]
                gaps = np.diff([0] + app_indices) - 1
                max_hist_omission[digit] = max(gaps.max(), current_omission[digit])
            else:
                current_omission[digit] = max_hist_omission[digit] = total_periods
        
        # 平均间隔（理论遗漏）
        avg_interval = {digit: total_periods / (digit_freq.get(digit, 0) + 1e-9) for digit in DIGIT_RANGE}
        
        freq_stats[pos_name] = {
            'freq': digit_freq,
            'current_omission': current_omission,
            'average_interval': avg_interval,
            'max_historical_omission': max_hist_omission,
            'recent_freq': recent_freq
        }
    
    return freq_stats

def analyze_patterns(df: pd.DataFrame) -> dict:
    """分析历史数据中的常见模式，如最常见的和值、跨度、奇偶分布等。"""
    if df is None or df.empty:
        return {}
    
    res = {}
    def safe_mode(s):
        return s.mode().iloc[0] if not s.empty and not s.mode().empty else None
    
    # 分析各种特征的最常见值
    pattern_features = ['sum_value', 'span_value', 'odd_count', 'prime_count', 'large_count', 'form_type']
    for feature in pattern_features:
        if feature in df.columns:
            res[f'most_common_{feature}'] = safe_mode(df[feature])
    
    return res

def analyze_associations(df: pd.DataFrame, weights_config: Dict) -> pd.DataFrame:
    """使用Apriori算法挖掘数字之间的关联规则。"""
    min_s = weights_config.get('ARM_MIN_SUPPORT', 0.01)
    min_c = weights_config.get('ARM_MIN_CONFIDENCE', 0.5)
    min_l = weights_config.get('ARM_MIN_LIFT', 1.2)
    
    if df is None or df.empty:
        return pd.DataFrame()
    
    try:
        # 创建事务数据，每一期的三个数字作为一个事务
        transactions = []
        for _, row in df.iterrows():
            transaction = [str(row['hundreds']), str(row['tens']), str(row['units'])]
            transactions.append(transaction)
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_oh = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(df_oh, min_support=min_s, use_colnames=True)
        if frequent_itemsets.empty:
            return pd.DataFrame()
        
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_l)
        strong_rules = rules[rules['confidence'] >= min_c].sort_values(by='lift', ascending=False)
        return strong_rules
        
    except Exception as e:
        logger.error(f"关联规则分析失败: {e}")
        return pd.DataFrame()

def calculate_scores(freq_data: Dict, probabilities: Dict, weights: Dict) -> Dict[str, Dict[int, float]]:
    """根据所有分析结果计算每个位置每个数字的最终推荐分数。"""
    scores = {'hundreds': {}, 'tens': {}, 'units': {}}
    
    for pos_name in ['hundreds', 'tens', 'units']:
        pos_stats = freq_data.get(pos_name, {})
        pos_probs = probabilities.get(pos_name, {})
        
        freq = pos_stats.get('freq', {})
        omission = pos_stats.get('current_omission', {})
        avg_int = pos_stats.get('average_interval', {})
        max_hist_o = pos_stats.get('max_historical_omission', {})
        recent_freq = pos_stats.get('recent_freq', {})
        
        pos_scores = {}
        for digit in DIGIT_RANGE:
            # 频率分：出现次数越多，得分越高
            freq_s = freq.get(digit, 0) * weights['FREQ_SCORE_WEIGHT']
            
            # 遗漏分：当前遗漏接近平均遗漏时得分最高
            omit_s = np.exp(-0.01 * (omission.get(digit, 0) - avg_int.get(digit, 0))**2) * weights['OMISSION_SCORE_WEIGHT']
            
            # 最大遗漏比率分
            max_o_ratio = (omission.get(digit, 0) / max_hist_o.get(digit, 1)) if max_hist_o.get(digit, 0) > 0 else 0
            max_o_s = max_o_ratio * weights['MAX_OMISSION_RATIO_SCORE_WEIGHT']
            
            # 近期频率分
            recent_s = recent_freq.get(digit, 0) * weights['RECENT_FREQ_SCORE_WEIGHT']
            
            # ML预测分
            ml_s = pos_probs.get(digit, 0.0) * weights['ML_PROB_SCORE_WEIGHT']
            
            pos_scores[digit] = sum([freq_s, omit_s, max_o_s, recent_s, ml_s])
        
        scores[pos_name] = pos_scores
    
    # 归一化所有分数到0-100范围
    def normalize_scores(scores_dict):
        if not scores_dict:
            return {}
        vals = list(scores_dict.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {k: 50.0 for k in scores_dict}
        return {k: (v - min_v) / (max_v - min_v) * 100 for k, v in scores_dict.items()}
    
    for pos_name in scores:
        scores[pos_name] = normalize_scores(scores[pos_name])
    
    return scores

# ==============================================================================
# --- 机器学习模块 ---
# ==============================================================================

def train_single_lgbm_model(pos_name: str, digit: int, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Optional[LGBMClassifier], Optional[str]]:
    """为单个位置的单个数字训练一个LGBM二分类模型。"""
    if y_train.sum() < MIN_POSITIVE_SAMPLES_FOR_ML or y_train.nunique() < 2:
        return None, None
        
    model_key = f'lgbm_{pos_name}_{digit}'
    model_params = LGBM_PARAMS.copy()
    
    # 类别不平衡处理
    if (pos_count := y_train.sum()) > 0:
        model_params['scale_pos_weight'] = (len(y_train) - pos_count) / pos_count
        
    try:
        model = LGBMClassifier(**model_params)
        model.fit(X_train, y_train)
        return model, model_key
    except Exception as e:
        logger.debug(f"训练LGBM for {pos_name} {digit} 失败: {e}")
        return None, None

def train_prediction_models(df_train_raw: pd.DataFrame, ml_lags_list: List[int]) -> Optional[Dict[str, Any]]:
    """为所有位置的所有数字并行训练预测模型。"""
    if (X := create_lagged_features(df_train_raw.copy(), ml_lags_list)) is None or X.empty:
        logger.warning("创建滞后特征失败或结果为空，跳过模型训练。")
        return None
        
    if (target_df := df_train_raw.loc[X.index].copy()).empty:
        return None
    
    trained_models = {'hundreds': {}, 'tens': {}, 'units': {}, 'feature_cols': X.columns.tolist()}
    
    # 使用进程池并行训练，加快速度 (Windows下禁用以避免问题)
    import platform
    if platform.system() == 'Windows':
        # Windows下直接串行训练
        executor = None
    else:
        executor = concurrent.futures.ProcessPoolExecutor()
        
    if executor is None:
        # 串行训练模式
        for pos_name in ['hundreds', 'tens', 'units']:
            for digit in DIGIT_RANGE:
                y = (target_df[pos_name] == digit).astype(int)
                model, model_key = train_single_lgbm_model(pos_name, digit, X, y)
                if model and model_key:
                    trained_models[pos_name][model_key] = model
    else:
        # 并行训练模式
        with executor:
            futures = {}
        
                    # 为每个位置的每个数字提交训练任务
            for pos_name in ['hundreds', 'tens', 'units']:
                for digit in DIGIT_RANGE:
                    y = (target_df[pos_name] == digit).astype(int)
                    future = executor.submit(train_single_lgbm_model, pos_name, digit, X, y)
                    futures[future] = (pos_name, digit)
                
            for future in concurrent.futures.as_completed(futures):
                pos_name, digit = futures[future]
                try:
                    model, model_key = future.result()
                    if model and model_key:
                        trained_models[pos_name][model_key] = model
                except Exception as e:
                    logger.error(f"训练数字 {digit} ({pos_name}) 的模型时出现异常: {e}")

    return trained_models if any(trained_models[pos] for pos in ['hundreds', 'tens', 'units']) else None

def predict_next_draw_probabilities(df_historical: pd.DataFrame, trained_models: Optional[Dict], ml_lags_list: List[int]) -> Dict[str, Dict[int, float]]:
    """使用训练好的模型预测下一期每个位置每个数字的出现概率。"""
    probs = {'hundreds': {}, 'tens': {}, 'units': {}}
    if not trained_models or not (feat_cols := trained_models.get('feature_cols')):
        return probs
        
    max_lag = max(ml_lags_list) if ml_lags_list else 0
    if len(df_historical) < max_lag + 1:
        return probs
        
    if (predict_X := create_lagged_features(df_historical.tail(max_lag + 1), ml_lags_list)) is None:
        return probs
        
    predict_X = predict_X.reindex(columns=feat_cols, fill_value=0)
    
    for pos_name in ['hundreds', 'tens', 'units']:
        for digit in DIGIT_RANGE:
            if (model := trained_models.get(pos_name, {}).get(f'lgbm_{pos_name}_{digit}')):
                try:
                    probs[pos_name][digit] = model.predict_proba(predict_X)[0, 1]
                except Exception:
                    pass
    return probs

# ==============================================================================
# --- 组合生成与策略应用模块 ---
# ==============================================================================

def generate_combinations(scores_data: Dict, pattern_data: Dict, arm_rules: pd.DataFrame, weights_config: Dict) -> Tuple[List[Dict], List[str]]:
    """根据评分和策略生成最终的推荐组合。"""
    num_to_gen = weights_config['NUM_COMBINATIONS_TO_GENERATE']
    
    if not all(pos in scores_data for pos in ['hundreds', 'tens', 'units']):
        return [], ["无法生成推荐 (分数数据缺失)。"]

    # 1. 构建候选池
    top_n = int(weights_config['TOP_N_DIGITS_FOR_CANDIDATE'])
    candidate_pools = {}
    
    for pos_name in ['hundreds', 'tens', 'units']:
        pos_scores = scores_data[pos_name]
        candidates = [digit for digit, _ in sorted(pos_scores.items(), key=lambda i: i[1], reverse=True)[:top_n]]
        candidate_pools[pos_name] = candidates
    
    # 2. 生成组合
    all_combinations = list(product(candidate_pools['hundreds'], candidate_pools['tens'], candidate_pools['units']))
    
    # 3. 评分和筛选
    scored_combos = []
    for combo in all_combinations:
        h, t, u = combo
        # 基础分 = 各位置分数之和
        base_score = scores_data['hundreds'].get(h, 0) + scores_data['tens'].get(t, 0) + scores_data['units'].get(u, 0)
        
        # 模式匹配奖励
        bonus = 0
        combo_sum = h + t + u
        combo_span = max(combo) - min(combo)
        combo_odd_count = sum(x % 2 for x in combo)
        combo_prime_count = sum(1 for x in combo if x in [2, 3, 5, 7])
        
        # 形态判断
        combo_counter = Counter(combo)
        if len(combo_counter) == 1:
            combo_form = 'leopard'
        elif len(combo_counter) == 2:
            combo_form = 'group3'
        else:
            combo_form = 'group6'
        
        # 匹配奖励
        if pattern_data.get('most_common_sum_value') == combo_sum:
            bonus += weights_config['COMBINATION_SUM_MATCH_BONUS']
        if pattern_data.get('most_common_span_value') == combo_span:
            bonus += weights_config['COMBINATION_SPAN_MATCH_BONUS']
        if pattern_data.get('most_common_odd_count') == combo_odd_count:
            bonus += weights_config['COMBINATION_ODD_COUNT_MATCH_BONUS']
        if pattern_data.get('most_common_prime_count') == combo_prime_count:
            bonus += weights_config['COMBINATION_PRIME_COUNT_MATCH_BONUS']
        if pattern_data.get('most_common_form_type') == combo_form:
            bonus += weights_config['COMBINATION_FORM_MATCH_BONUS']
        
        total_score = base_score + bonus
        scored_combos.append({'combination': combo, 'score': total_score})
    
    # 4. 选择最优组合
    sorted_combos = sorted(scored_combos, key=lambda x: x['score'], reverse=True)
    final_recs = sorted_combos[:num_to_gen]
    
    # 5. 应用反向思维策略
    applied_msg = ""
    if ENABLE_FINAL_COMBO_REVERSE:
        num_to_remove = int(len(final_recs) * weights_config.get('FINAL_COMBO_REVERSE_REMOVE_TOP_PERCENT', 0))
        if 0 < num_to_remove < len(final_recs):
            removed, final_recs = final_recs[:num_to_remove], final_recs[num_to_remove:]
            applied_msg = f" (反向策略: 移除前{num_to_remove}注"
            if ENABLE_REVERSE_REFILL and len(sorted_combos) > len(final_recs) + num_to_remove:
                refill_candidates = sorted_combos[len(final_recs) + num_to_remove:len(final_recs) + num_to_remove + num_to_remove]
                final_recs.extend(refill_candidates)
                applied_msg += "并补充)"
            else:
                applied_msg += ")"

    final_recs = sorted(final_recs, key=lambda x: x['score'], reverse=True)[:num_to_gen]

    # 6. 生成输出字符串
    output_strs = [f"推荐组合 (Top {len(final_recs)}{applied_msg}):"]
    for i, c in enumerate(final_recs):
        combo = c['combination']
        number_str = ''.join(str(digit) for digit in combo)
        output_strs.append(f"  注 {i+1}: {number_str} (综合分: {c['score']:.2f})")
        
    return final_recs, output_strs

# ==============================================================================
# --- 核心分析与回测流程 ---
# ==============================================================================

def run_analysis_and_recommendation(df_hist: pd.DataFrame, ml_lags: List[int], weights_config: Dict, arm_rules: pd.DataFrame) -> Tuple:
    """执行一次完整的分析和推荐流程，用于特定一期。"""
    freq_data = analyze_frequency_omission(df_hist)
    patt_data = analyze_patterns(df_hist)
    ml_models = train_prediction_models(df_hist, ml_lags)
    probabilities = predict_next_draw_probabilities(df_hist, ml_models, ml_lags) if ml_models else {'hundreds': {}, 'tens': {}, 'units': {}}
    scores = calculate_scores(freq_data, probabilities, weights_config)
    recs, rec_strings = generate_combinations(scores, patt_data, arm_rules, weights_config)
    analysis_summary = {'frequency_omission': freq_data, 'patterns': patt_data}
    return recs, rec_strings, analysis_summary, ml_models, scores

def run_backtest(full_df: pd.DataFrame, ml_lags: List[int], weights_config: Dict, arm_rules: pd.DataFrame, num_periods: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """在历史数据上执行策略回测，以评估策略表现。"""
    min_data_needed = (max(ml_lags) if ml_lags else 0) + MIN_POSITIVE_SAMPLES_FOR_ML + num_periods
    if len(full_df) < min_data_needed:
        logger.error(f"数据不足以回测{num_periods}期。需要至少{min_data_needed}期，当前有{len(full_df)}期。")
        return pd.DataFrame(), {}

    start_idx = len(full_df) - num_periods
    results, prize_counts = [], Counter()
    
    logger.info("策略回测已启动...")
    start_time = time.time()
    
    for i in range(num_periods):
        current_iter = i + 1
        current_idx = start_idx + i
        
        with SuppressOutput(suppress_stdout=True, capture_stderr=True):
            hist_data = full_df.iloc[:current_idx]
            predicted_combos, _, _, _, _ = run_analysis_and_recommendation(hist_data, ml_lags, weights_config, arm_rules)
            
        actual_outcome = full_df.loc[current_idx]
        actual_number = str(actual_outcome['number'])
        
        if not predicted_combos:
            continue
        else:
            for combo_dict in predicted_combos:
                combo = combo_dict['combination']
                predicted_number = ''.join(str(digit) for digit in combo)
                prize = get_prize_level(predicted_number, actual_number)
                if prize:
                    prize_counts[prize] += 1
                results.append({'period': actual_outcome['期号'], 'predicted': predicted_number, 'actual': actual_number, 'prize': prize})

        # 打印进度
        if current_iter == 1 or current_iter % 10 == 0 or current_iter == num_periods:
            elapsed = time.time() - start_time
            avg_time = elapsed / current_iter
            remaining_time = avg_time * (num_periods - current_iter)
            progress_logger.info(f"回测进度: {current_iter}/{num_periods} | 平均耗时: {avg_time:.2f}s/期 | 预估剩余: {format_time(remaining_time)}")
            
    return pd.DataFrame(results), {'prize_counts': dict(prize_counts)}

# ==============================================================================
# --- Optuna 参数优化模块 ---
# ==============================================================================

def objective(trial: optuna.trial.Trial, df_for_opt: pd.DataFrame, ml_lags: List[int], arm_rules: pd.DataFrame) -> float:
    """Optuna 的目标函数，用于评估一组给定的权重参数的好坏。"""
    trial_weights = {}
    
    # 动态地从DEFAULT_WEIGHTS构建搜索空间
    for key, value in DEFAULT_WEIGHTS.items():
        if isinstance(value, int):
            if 'NUM_COMBINATIONS' in key:
                trial_weights[key] = trial.suggest_int(key, 5, 20)
            elif 'TOP_N' in key:
                trial_weights[key] = trial.suggest_int(key, 6, 10)
            else:
                trial_weights[key] = trial.suggest_int(key, max(0, value - 2), value + 2)
        elif isinstance(value, float):
            if any(k in key for k in ['PERCENT', 'FACTOR', 'SUPPORT', 'CONFIDENCE']):
                trial_weights[key] = trial.suggest_float(key, value * 0.5, value * 1.5)
            else:
                trial_weights[key] = trial.suggest_float(key, value * 0.5, value * 2.0)

    full_trial_weights = DEFAULT_WEIGHTS.copy()
    full_trial_weights.update(trial_weights)
    
    # 在快速回测中评估这组权重
    with SuppressOutput():
        _, backtest_stats = run_backtest(df_for_opt, ml_lags, full_trial_weights, arm_rules, OPTIMIZATION_BACKTEST_PERIODS)
        
    # 定义一个分数来衡量表现，直选权重最高
    prize_weights = {'直选': 1000, '组三': 100, '组六': 10}
    score = sum(prize_weights.get(p, 0) * c for p, c in backtest_stats.get('prize_counts', {}).items())
    return score

def optuna_progress_callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial, total_trials: int):
    """Optuna 的回调函数，用于在控制台报告优化进度。"""
    global OPTUNA_START_TIME
    current_iter = trial.number + 1
    if current_iter == 1 or current_iter % 10 == 0 or current_iter == total_trials:
        elapsed = time.time() - OPTUNA_START_TIME
        avg_time = elapsed / current_iter
        remaining_time = avg_time * (total_trials - current_iter)
        best_value = f"{study.best_value:.2f}" if study.best_trial else "N/A"
        progress_logger.info(f"Optuna进度: {current_iter}/{total_trials} | 当前最佳得分: {best_value} | 预估剩余: {format_time(remaining_time)}")

# ==============================================================================
# --- 主程序入口 ---
# ==============================================================================
if __name__ == "__main__":
    # 1. 初始化日志记录器，同时输出到控制台和文件
    log_filename = os.path.join(SCRIPT_DIR, f"fc3d_analysis_output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    file_handler = logging.FileHandler(log_filename, 'w', 'utf-8')
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    set_console_verbosity(logging.INFO, use_simple_formatter=True)

    logger.info("--- 福彩3D数据分析与推荐系统 ---")
    logger.info("启动数据加载和预处理...")

    # 2. 健壮的数据加载逻辑
    main_df = None
    if os.path.exists(PROCESSED_CSV_PATH):
        main_df = load_data(PROCESSED_CSV_PATH)
        if main_df is not None:
             logger.info("从缓存文件加载预处理数据成功。")

    if main_df is None or main_df.empty:
        logger.info("未找到或无法加载缓存数据，正在从原始文件生成...")
        raw_df = load_data(CSV_FILE_PATH)
        if raw_df is not None and not raw_df.empty:
            logger.info("原始数据加载成功，开始清洗...")
            cleaned_df = clean_and_structure(raw_df)
            if cleaned_df is not None and not cleaned_df.empty:
                logger.info("数据清洗成功，开始特征工程...")
                main_df = feature_engineer(cleaned_df)
                if main_df is not None and not main_df.empty:
                    logger.info("特征工程成功，保存预处理数据...")
                    try:
                        main_df.to_csv(PROCESSED_CSV_PATH, index=False)
                        logger.info(f"预处理数据已保存到: {PROCESSED_CSV_PATH}")
                    except IOError as e:
                        logger.error(f"保存预处理数据失败: {e}")
                else:
                    logger.error("特征工程失败，无法生成最终数据集。")
            else:
                logger.error("数据清洗失败。")
        else:
            logger.error("原始数据加载失败。")
    
    if main_df is None or main_df.empty:
        logger.critical("数据准备失败，无法继续。请检查 'fc3d_data_processor.py' 是否已成功运行并生成 'fc3d.csv'。程序终止。")
        sys.exit(1)
    
    logger.info(f"数据加载完成，共 {len(main_df)} 期有效数据。")
    last_period = main_df['期号'].iloc[-1]

    # 3. 根据模式执行：优化或直接分析
    active_weights = DEFAULT_WEIGHTS.copy()
    optuna_summary = None

    if ENABLE_OPTUNA_OPTIMIZATION:
        logger.info("\n" + "="*25 + " Optuna 参数优化模式 " + "="*25)
        set_console_verbosity(logging.INFO, use_simple_formatter=False)
        
        # 优化前先进行一次全局关联规则分析
        optuna_arm_rules = analyze_associations(main_df, DEFAULT_WEIGHTS)
        
        study = optuna.create_study(direction="maximize")
        global OPTUNA_START_TIME; OPTUNA_START_TIME = time.time()
        progress_callback_with_total = partial(optuna_progress_callback, total_trials=OPTIMIZATION_TRIALS)
        
        try:
            study.optimize(lambda t: objective(t, main_df, ML_LAG_FEATURES, optuna_arm_rules), n_trials=OPTIMIZATION_TRIALS, callbacks=[progress_callback_with_total])
            logger.info("Optuna 优化完成。")
            active_weights.update(study.best_params)
            optuna_summary = {"status": "完成", "best_value": study.best_value, "best_params": study.best_params}
        except Exception as e:
            logger.error(f"Optuna 优化过程中断: {e}", exc_info=True)
            optuna_summary = {"status": "中断", "error": str(e)}
            logger.warning("优化中断，将使用默认权重继续分析。")
    
    # 4. 切换到报告模式并打印报告头
    report_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(report_formatter)
    global_console_handler.setFormatter(report_formatter)
    
    logger.info("\n\n" + "="*60 + f"\n{' ' * 18}福彩3D策略分析报告\n" + "="*60)
    logger.info(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"分析基于数据: 截至 {last_period} 期 (共 {len(main_df)} 期)")
    logger.info(f"本次预测目标: 第 {last_period + 1} 期")
    logger.info(f"日志文件: {os.path.basename(log_filename)}")

    # 5. 打印优化摘要
    if ENABLE_OPTUNA_OPTIMIZATION and optuna_summary:
        logger.info("\n" + "="*25 + " Optuna 优化摘要 " + "="*25)
        logger.info(f"优化状态: {optuna_summary['status']}")
        if optuna_summary['status'] == '完成':
            logger.info(f"最佳性能得分: {optuna_summary['best_value']:.4f}")
            logger.info("--- 本次分析已采用以下优化参数 ---")
            best_params_str = json.dumps(optuna_summary['best_params'], indent=2, ensure_ascii=False)
            logger.info(best_params_str)
        else:
            logger.info(f"错误信息: {optuna_summary['error']}")
    else:
        logger.info("\n--- 本次分析使用脚本内置的默认权重 ---")

    # 6. 全局分析
    full_history_arm_rules = analyze_associations(main_df, active_weights)
    
    # 7. 回测并打印报告
    logger.info("\n" + "="*25 + " 策 略 回 测 摘 要 " + "="*25)
    backtest_results_df, backtest_stats = run_backtest(main_df, ML_LAG_FEATURES, active_weights, full_history_arm_rules, BACKTEST_PERIODS_COUNT)
    
    if not backtest_results_df.empty:
        num_periods_tested = len(backtest_results_df['period'].unique())
        num_combos_per_period = active_weights.get('NUM_COMBINATIONS_TO_GENERATE', 10)
        total_bets = len(backtest_results_df)
        logger.info(f"回测周期: 最近 {num_periods_tested} 期 | 每期注数: {num_combos_per_period} | 总投入注数: {total_bets}")
        logger.info("\n--- 1. 奖金与回报分析 ---")
        prize_dist = backtest_stats.get('prize_counts', {})
        prize_values = {'直选': 1000, '组三': 320, '组六': 160}
        total_revenue = sum(prize_values.get(p, 0) * c for p, c in prize_dist.items())
        total_cost = total_bets * 2
        roi = (total_revenue - total_cost) * 100 / total_cost if total_cost > 0 else 0
        logger.info(f"  - 估算总回报: {total_revenue:,.2f} 元 (总成本: {total_cost:,.2f} 元)")
        logger.info(f"  - 投资回报率 (ROI): {roi:.2f}%")
        logger.info("  - 中奖等级分布 (总计):")
        if prize_dist:
            for prize in prize_values.keys():
                if prize in prize_dist:
                    logger.info(f"    - {prize:<4s}: {prize_dist[prize]:>4d} 次")
        else:
            logger.info("    - 未命中任何奖级。")
    else:
        logger.warning("回测未产生有效结果，可能是数据量不足。")
    
    # 8. 最终推荐
    logger.info("\n" + "="*25 + f" 第 {last_period + 1} 期 号 码 推 荐 " + "="*25)
    final_recs, final_rec_strings, _, _, final_scores = run_analysis_and_recommendation(main_df, ML_LAG_FEATURES, active_weights, full_history_arm_rules)
    
    logger.info("\n--- 直选推荐 ---")
    for line in final_rec_strings:
        logger.info(line)
    
    logger.info("\n--- 各位置热门数字 ---")
    if final_scores:
        for pos_name in ['hundreds', 'tens', 'units']:
            if pos_name in final_scores:
                pos_scores = final_scores[pos_name]
                top_digits = sorted([digit for digit, _ in sorted(pos_scores.items(), key=lambda x: x[1], reverse=True)[:5]])
                pos_display = {'hundreds': '百位', 'tens': '十位', 'units': '个位'}[pos_name]
                logger.info(f"  {pos_display} (Top 5): {' '.join(str(d) for d in top_digits)}")
    
    logger.info("\n" + "="*60 + f"\n--- 报告结束 (详情请查阅: {os.path.basename(log_filename)}) ---\n")
    
    # 9. 微信推送
    try:
        from fc3d_wxpusher import send_analysis_report
        logger.info("正在发送微信推送...")
        
        # 读取完整的分析报告内容
        with open(log_filename, 'r', encoding='utf-8') as f:
            report_content = f.read()
        
        # 构建推荐号码列表 (从final_rec_strings中提取)
        recommendations = []
        for line in final_rec_strings:
            if line.startswith("  注"):
                # 提取号码，格式：注 1: 348 (综合分: 329.18)
                import re
                match = re.search(r'注\s+\d+:\s*(\d{3})', line)
                if match:
                    recommendations.append(match.group(1))
        
        # 发送完整分析报告推送
        push_result = send_analysis_report(
            report_content=report_content,
            period=last_period + 1,
            recommendations=recommendations,
            optuna_summary=optuna_summary,
            backtest_stats=backtest_stats
        )
        
        if push_result.get('success'):
            logger.info("微信推送发送成功（完整分析报告）")
        else:
            logger.warning(f"微信推送发送失败: {push_result.get('error', '未知错误')}")
            
    except ImportError:
        logger.warning("微信推送模块未找到，跳过推送功能")
    except Exception as e:
        logger.error(f"微信推送发送异常: {e}")
    
    # 10. 更新latest_fc3d_analysis.txt
    try:
        with open(os.path.join(SCRIPT_DIR, 'latest_fc3d_analysis.txt'), 'w', encoding='utf-8') as f:
            # 重新读取完整的日志文件内容
            with open(log_filename, 'r', encoding='utf-8') as log_f:
                f.write(log_f.read())
        logger.info("已更新 latest_fc3d_analysis.txt")
    except Exception as e:
        logger.error(f"更新 latest_fc3d_analysis.txt 失败: {e}") 