# shelf_optimizer_gui_final.py
# 版本号：vReX 1.0.1 beta

import customtkinter as ctk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import tkinter as tk
import threading
import queue
import pandas as pd
import numpy as np
import os
import math
from itertools import combinations, groupby
import warnings
import time
import json

# --- 可视化库导入 ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from matplotlib.figure import Figure
from collections import Counter

warnings.filterwarnings('ignore')
# 设置Matplotlib支持中文显示的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# --- 常量定义 ---
SIDE_SPACING = 50
INTER_SPACING = 100
NUM_BINS_FOR_AGGREGATION = 200

# 参数持久化配置
DEFAULT_CONFIG_FILE = "shelf_optimizer_config.json"
LAST_SESSION_FILE = "last_session_params.json"

# #############################################################################
# --- 核心计算逻辑 (后台线程运行) ---
# #############################################################################

def is_complementary(shelf1, shelf2, area_threshold, size_threshold):
    """
    检验两个货架是否互补 (使用动态阈值)。
    """
    area1 = shelf1['Lp'] * shelf1['Dp']; area2 = shelf2['Lp'] * shelf2['Dp']
    if max(area1, area2) == 0: return False
    area_ratio = min(area1, area2) / max(area1, area2)
    if area_ratio > area_threshold: return False
    if max(shelf1['Lp'], shelf2['Lp']) == 0 or max(shelf1['Dp'], shelf2['Dp']) == 0: return False
    length_ratio = min(shelf1['Lp'], shelf2['Lp']) / max(shelf1['Lp'], shelf2['Lp'])
    depth_ratio = min(shelf1['Dp'], shelf2['Dp']) / max(shelf1['Dp'], shelf2['Dp'])
    if length_ratio > size_threshold and depth_ratio > size_threshold: return False
    return True

def pre_filter_combinations(shelves, area_threshold, size_threshold):
    """
    前置检验：排除明显不互补的组合 (使用动态阈值)。
    """
    valid_combinations = []
    for i, shelf1 in enumerate(shelves):
        for j, shelf2 in enumerate(shelves[i+1:], i+1):
            if is_complementary(shelf1, shelf2, area_threshold, size_threshold):
                valid_combinations.append((shelf1, shelf2))
    return valid_combinations

def evaluate_shelf_combination(agg_data, two_shelves, coverage_target, allow_rotation):
    """
    评估两个货架规格组合的效果。
    """
    assignments = {0: [], 1: []}
    for _, sku_group in agg_data.iterrows():
        best_fit, best_efficiency = None, -1
        for i, shelf in enumerate(two_shelves):
            if sku_group['W'] > shelf['Wp']: continue
            orientations = [{'w': sku_group['L'], 'd': sku_group['D']}]
            if allow_rotation and sku_group['L'] != sku_group['D']:
                orientations.append({'w': sku_group['D'], 'd': sku_group['L']})
            for ori in orientations:
                w, d = ori['w'], ori['d']
                if w <= shelf['Lp'] and d <= shelf['Dp'] and shelf['Lp'] > 0 and shelf['Dp'] > 0:
                    space_utilization = (w * d) / (shelf['Lp'] * shelf['Dp'])
                    if space_utilization > best_efficiency:
                        best_efficiency = space_utilization
                        best_fit = {'shelf_idx': i, 'width': w, 'sku_group': sku_group}
        if best_fit:
            item = (best_fit['sku_group'].to_dict(), best_fit['width'], int(best_fit['sku_group']['count']))
            assignments[best_fit['shelf_idx']].append(item)
    total_count = agg_data['count'].sum(); total_volume = (agg_data['V'] * agg_data['count']).sum()
    placed_count = sum(item[2] for lst in assignments.values() for item in lst)
    placed_volume = sum(item[0]['V'] * item[2] for lst in assignments.values() for item in lst)
    coverage_count = placed_count / total_count if total_count > 0 else 0
    coverage_volume = placed_volume / total_volume if total_volume > 0 else 0
    if placed_count < total_count * coverage_target or placed_volume < total_volume * coverage_target:
        return {'status': 'failure', 'coverage_count': coverage_count, 'coverage_volume': coverage_volume}
    counts = [run_bulk_ffd_packing(assignments[i], two_shelves[i]['Lp']) for i in range(2)]
    return {'status': 'success', 'counts': counts, 'coverage_count': coverage_count, 'coverage_volume': coverage_volume}

def ld_calculator_complementary(agg_data, shelves, coverage_target, allow_rotation, q, params):
    """
    计算最优的互补L&D规格组合 (使用动态阈值)。
    """
    best_combo, min_total_shelves = None, float('inf')
    best_attempt_combo, max_achieved_coverage = None, 0.0
    q.put(("log", "--- 步骤 2a: 执行前置可行性检验 ---\n"))

    area_threshold = params.get('area_threshold', 0.7)
    size_threshold = params.get('size_threshold', 0.8)
    valid_combinations = pre_filter_combinations(shelves, area_threshold, size_threshold)

    if len(valid_combinations) == 0:
        q.put(("log", "警告: 前置检验未找到任何互补组合，将评估所有组合，可能耗时较长。\n"))
        valid_combinations = [(shelves[i], shelves[j]) for i in range(len(shelves)) for j in range(i+1, len(shelves))]
    total_combinations = len(valid_combinations)
    q.put(("log", f"前置检验完成，共需评估 {total_combinations} 种L&D组合。\n"))
    start_time = time.time()
    for i, (shelf1, shelf2) in enumerate(valid_combinations):
        try:
            two_shelves = [shelf1, shelf2]
            solution = evaluate_shelf_combination(agg_data, two_shelves, coverage_target, allow_rotation)
            if solution['coverage_count'] > max_achieved_coverage:
                max_achieved_coverage = solution['coverage_count']
                best_attempt_combo = (shelf1, shelf2)
            if solution['status'] == 'success':
                total_shelves = sum(solution['counts'])
                if total_shelves < min_total_shelves:
                    min_total_shelves = total_shelves
                    best_combo = (shelf1, shelf2)
        except Exception as e:
            q.put(("log", f"警告: 评估组合 {shelf1['Lp']:.0f}x{shelf1['Dp']:.0f} + {shelf2['Lp']:.0f}x{shelf2['Dp']:.0f} 时出错: {str(e)}\n"))
            continue
        q.put(("progress", (i + 1, total_combinations, start_time, f"评估L&D组合 {i+1}/{total_combinations}")))
    if best_combo: return ("success", best_combo)
    else:
        if best_attempt_combo:
            s1, s2 = best_attempt_combo
            error_msg = (f"L&D计算失败：在评估了 {total_combinations} 种组合后，未能找到满足覆盖率目标 ({coverage_target*100:.1f}%) 的货架组合。\n\n"
                         f"--- 诊断信息 ---\n能实现最高覆盖率的组合是:\n - 规格1: {s1['Lp']:.0f}(长) x {s1['Dp']:.0f}(深)\n - 规格2: {s2['Lp']:.0f}(长) x {s2['Dp']:.0f}(深)\n"
                         f" - 此组合达成的最高覆盖率: {max_achieved_coverage*100:.2f}%\n\n【建议】\n1. 尝试将覆盖率目标降低至 {max_achieved_coverage*100:.1f}% 或以下。\n2. 检查货架参数文件，确认货架尺寸是否能匹配大部分SKU。\n3. 尝试放宽'互补性'参数阈值。")
        else: error_msg = "L&D计算失败：未能找到任何有效的货架组合。请检查输入的SKU与货架尺寸数据是否匹配，或尝试放宽'互补性'参数阈值。"
        return ("failure", {"message": error_msg})

def read_excel_data(file_path, sku_sheet, cols, can_box_col, can_pallet_col, box_count_col, pallet_count_col, boxes_per_pallet_col, pallet_preset, box_preset):
    """从Excel的SKU Sheet中读取数据，包括装箱装托相关数据和两组独立的LDHWV数据"""
    df = pd.read_excel(file_path, sheet_name=sku_sheet); data = pd.DataFrame()
    data['sku_id'] = df[cols['sku_id']].astype(str)
    
    # 读取装箱装托限制（两列Y/N）
    if can_box_col in df.columns:
        data['can_box'] = df[can_box_col].astype(str).fillna('N').str.upper()
    else:
        raise ValueError(f"在SKU Sheet中未找到是否可装箱列：{can_box_col}")
    
    if can_pallet_col in df.columns:
        data['can_pallet'] = df[can_pallet_col].astype(str).fillna('N').str.upper()
    else:
        raise ValueError(f"在SKU Sheet中未找到是否可装托列：{can_pallet_col}")
    
    # 读取装箱数和装托数
    if box_count_col in df.columns:
        data['box_count'] = pd.to_numeric(df[box_count_col], errors='coerce')
    else:
        raise ValueError(f"在SKU Sheet中未找到装箱数列：{box_count_col}")
    
    if pallet_count_col in df.columns:
        data['pallet_count'] = pd.to_numeric(df[pallet_count_col], errors='coerce')
    else:
        raise ValueError(f"在SKU Sheet中未找到装托数列：{pallet_count_col}")
    
    if boxes_per_pallet_col in df.columns:
        data['boxes_per_pallet'] = pd.to_numeric(df[boxes_per_pallet_col], errors='coerce')
    else:
        raise ValueError(f"在SKU Sheet中未找到单托箱数列：{boxes_per_pallet_col}")
    
    # 读取装托LDHWV数据
    pallet_data_cols = ['pallet_l', 'pallet_d', 'pallet_h', 'pallet_w', 'pallet_v']
    for i, col_key in enumerate(['pallet_l', 'pallet_d', 'pallet_h', 'pallet_w', 'pallet_v']):
        col_name = cols[col_key]
        if col_name in df.columns:
            data[col_key] = pd.to_numeric(df[col_name], errors='coerce')
        else:
            data[col_key] = None  # 暂时设为None，后面会用预设值填充
    
    # 读取装箱LDHWV数据
    box_data_cols = ['box_l', 'box_d', 'box_h', 'box_w', 'box_v']
    for i, col_key in enumerate(['box_l', 'box_d', 'box_h', 'box_w', 'box_v']):
        col_name = cols[col_key]
        if col_name in df.columns:
            data[col_key] = pd.to_numeric(df[col_name], errors='coerce')
        else:
            data[col_key] = None  # 暂时设为None，后面会用预设值填充
    
    # 应用预设值处理缺失数据
    preset_mapping = {
        'pallet_l': pallet_preset['l'], 'pallet_d': pallet_preset['d'], 'pallet_h': pallet_preset['h'],
        'pallet_w': pallet_preset['w'], 'pallet_v': pallet_preset['v'],
        'box_l': box_preset['l'], 'box_d': box_preset['d'], 'box_h': box_preset['h'],
        'box_w': box_preset['w'], 'box_v': box_preset['v']
    }
    
    for col, preset_val in preset_mapping.items():
        data[col] = data[col].fillna(preset_val)
    
    data['original_index'] = df.index
    
    # 修改数据验证逻辑：只检查基本必需数据，LDHWV数据缺失用预设值处理
    required_cols = ['box_count', 'pallet_count', 'boxes_per_pallet']
    data = data.dropna(subset=required_cols)
    
    # 确保数值数据为正数
    numeric_cols = pallet_data_cols + box_data_cols + required_cols
    data = data[(data[numeric_cols] > 0).all(axis=1)]
    
    data.reset_index(drop=True, inplace=True)
    return data

def read_shelf_params(file_path, shelf_sheet, shelf_type_col, shelf_cols=None):
    """从Excel的货架Sheet中读取数据，包括货架类型标识"""
    df = pd.read_excel(file_path, sheet_name=shelf_sheet)
    shelves = []
    
    # 使用用户配置的列名，如果没有则使用默认值
    if shelf_cols is None:
        shelf_cols = {'lp': 'Lp', 'dp': 'Dp', 'wp': 'Wp'}
    
    # 检查必需的列是否存在
    required_cols = [shelf_cols['lp'], shelf_cols['dp'], shelf_cols['wp']]  # 长、深、承重
    if shelf_type_col not in df.columns:
        raise ValueError(f"在货架Sheet中未找到货架类型标识列：{shelf_type_col}")
    
    # 检查用户配置的列名是否存在
    col_mapping = {}
    for key, col_name in shelf_cols.items():
        if col_name not in df.columns:
            raise ValueError(f"在货架Sheet中未找到列：{col_name}。请检查货架LDW列名配置。")
        col_mapping[key] = col_name
    
    for i, row in df.iterrows():
        try:
            lp = float(row[col_mapping['lp']])
            dp = float(row[col_mapping['dp']])  
            wp = float(row[col_mapping['wp']])
            shelf_type = str(row[shelf_type_col]).strip()
            
            if lp > 0 and dp > 0 and wp > 0:  # 确保数据有效
                shelves.append({
                    'Lp': lp, 
                    'Dp': dp, 
                    'Wp': wp, 
                    'shelf_type': shelf_type,
                    'id': i
                })
        except (ValueError, TypeError):
            continue  # 跳过无效行
    
    if not shelves:
        raise ValueError("货架Sheet中没有找到有效的货架数据。")
    
    return shelves

def add_unified_ldh_columns(data, packing_decisions):
    """
    根据装箱装托判定结果，为每个SKU添加统一的L、D、H、W、V列
    
    Args:
        data: 包含装托和装箱数据的DataFrame
        packing_decisions: 装箱装托判定结果字典
    
    Returns:
        添加了L、D、H、W、V列的数据框
    """
    data = data.copy()
    
    # 初始化L、D、H、W、V列
    data['L'] = 0.0
    data['D'] = 0.0  
    data['H'] = 0.0
    data['W'] = 0.0
    data['V'] = 0.0
    
    for idx, row in data.iterrows():
        sku_id = row['sku_id']
        decision_info = packing_decisions.get(sku_id, {'decision': 'box_only'})  # 默认装箱
        decision = decision_info['decision']
        
        if decision in ['pallet_only', 'mixed']:
            # 使用装托数据（对于mixed类型，优先使用装托数据进行相关性分析）
            data.at[idx, 'L'] = row['pallet_l']
            data.at[idx, 'D'] = row['pallet_d']
            data.at[idx, 'H'] = row['pallet_h']
            data.at[idx, 'W'] = row['pallet_w']
            data.at[idx, 'V'] = row['pallet_v']
        else:
            # 使用装箱数据
            data.at[idx, 'L'] = row['box_l']
            data.at[idx, 'D'] = row['box_d']
            data.at[idx, 'H'] = row['box_h']
            data.at[idx, 'W'] = row['box_w']
            data.at[idx, 'V'] = row['box_v']
    
    return data

def correlation_analysis(data):
    # 检查是否有L、D、H列，如果没有则提示错误
    required_cols = ['L', 'D', 'H']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"数据缺少用于相关性分析的列：{missing_cols}。请确保在调用correlation_analysis之前先调用add_unified_ldh_columns函数。")
    
    data_for_corr = data[['L', 'D', 'H']].copy(); data_for_corr['Area'] = data_for_corr['L'] * data_for_corr['D']
    corrs = {'L vs H': abs(data_for_corr['L'].corr(data_for_corr['H'])), 'D vs H': abs(data_for_corr['D'].corr(data_for_corr['H'])), 'Area vs H': abs(data_for_corr['Area'].corr(data_for_corr['H']))}
    max_corr_label = max(corrs, key=corrs.get); max_corr_value = corrs[max_corr_label]
    if max_corr_value >= 0.7: grade = 'A级 (强相关)'
    elif max_corr_value >= 0.5: grade = 'B级 (中等相关)'
    elif max_corr_value >= 0.3: grade = 'C级 (弱相关)'
    else: grade = 'D级 (几乎不相关)'
    return max_corr_label, max_corr_value, grade

def analyze_packing_decision(data, pallet_decimal_threshold, q):
    """
    装箱/装托判定算法 - 基于两列Y/N控制
    
    步骤：
    1. 读取SKU的可装箱和可装托标识（Y/N）
    2. 根据标识组合进行判定：
       - 只能装箱：can_box=Y, can_pallet=N
       - 只能装托：can_box=N, can_pallet=Y
       - 都可以：can_box=Y, can_pallet=Y → 根据装托数小数部分判定
       - 都不可以：can_box=N, can_pallet=N → 错误，跳过
    3. 对于都可以的SKU，根据装托数小数部分与阈值比较进行判定
    
    返回：packing_decisions字典，key为sku_id，value为判定结果
    """
    q.put(("log", "--- 步骤 1: 装箱/装托判定分析 (基于两列Y/N控制) ---\n"))
    
    packing_decisions = {}
    box_only_count = 0
    pallet_only_count = 0
    mixed_packing_count = 0
    pure_pallet_count = 0
    error_count = 0
    error_skus = []
    
    for _, sku in data.iterrows():
        sku_id = sku['sku_id']
        can_box = sku['can_box'].strip().upper()
        can_pallet = sku['can_pallet'].strip().upper()
        
        # 第一步：根据两列Y/N的组合进行初步判定
        if can_box == 'Y' and can_pallet == 'N':
            # 只能装箱
            packing_decisions[sku_id] = {
                'decision': 'box_only',
                'final_box_count': sku['box_count'],
                'final_pallet_count': 0,
                'remaining_box_count': 0
            }
            box_only_count += 1
            continue
        elif can_box == 'N' and can_pallet == 'Y':
            # 只能装托
            packing_decisions[sku_id] = {
                'decision': 'pallet_only',
                'final_box_count': 0,
                'final_pallet_count': sku['pallet_count'],
                'remaining_box_count': 0
            }
            pallet_only_count += 1
            continue
        elif can_box == 'N' and can_pallet == 'N':
            # 都不可以 - 错误情况，跳过并记录
            error_skus.append(sku_id)
            error_count += 1
            continue
        elif can_box == 'Y' and can_pallet == 'Y':
            # 都可以 - 需要根据数据判定
            pass
        else:
            # 其他无效组合（非 Y/N 字符）
            error_skus.append(sku_id)
            error_count += 1
            continue
        
        # 第二步：对于都可以的SKU，读取装箱数、装托数、单托箱数
        box_count = sku['box_count']
        pallet_count = sku['pallet_count']
        boxes_per_pallet = sku['boxes_per_pallet']
        
        # 第三步：根据装托数小数部分判定
        pallet_decimal = pallet_count - int(pallet_count)  # 获取小数部分
        
        if pallet_decimal >= pallet_decimal_threshold:
            # 小数部分 >= 阈值，采用纯装托
            final_pallet_count = int(pallet_count) + 1  # 向上取整
            packing_decisions[sku_id] = {
                'decision': 'pallet_only',
                'final_box_count': 0,
                'final_pallet_count': final_pallet_count,
                'remaining_box_count': 0
            }
            pure_pallet_count += 1
        else:
            # 小数部分 < 阈值，采用混装
            final_pallet_count = int(pallet_count)  # 整数部分
            # 剩余装箱数 = 装箱数 - 装托数 × 单托箱数
            remaining_box_count = box_count - pallet_count * boxes_per_pallet
            remaining_box_count = max(0, remaining_box_count)  # 确保不为负数
            
            packing_decisions[sku_id] = {
                'decision': 'mixed',
                'final_box_count': remaining_box_count,
                'final_pallet_count': final_pallet_count,
                'remaining_box_count': remaining_box_count
            }
            mixed_packing_count += 1
    
    # 输出统计结果
    total_skus = len(data)
    valid_skus = total_skus - error_count
    q.put(("log", f"装箱/装托判定完成，共分析 {total_skus} 个SKU：\n"))
    q.put(("log", f"  - 只能装箱：{box_only_count} 个 ({box_only_count/valid_skus*100:.1f}% of valid)\n"))
    q.put(("log", f"  - 只能装托：{pallet_only_count} 个 ({pallet_only_count/valid_skus*100:.1f}% of valid)\n"))
    q.put(("log", f"  - 纯装托（小数部分≥{pallet_decimal_threshold}）：{pure_pallet_count} 个 ({pure_pallet_count/valid_skus*100:.1f}% of valid)\n"))
    q.put(("log", f"  - 托箱混装（小数部分<{pallet_decimal_threshold}）：{mixed_packing_count} 个 ({mixed_packing_count/valid_skus*100:.1f}% of valid)\n"))
    
    if error_count > 0:
        q.put(("log", f"  - 错误/跳过的SKU：{error_count} 个 ({error_count/total_skus*100:.1f}% of total)\n"))
        q.put(("log", f"    错误原因：装箱和装托都不允许或数据格式错误\n"))
    
    # 总结需要的货架类型
    needs_box_shelves = box_only_count + mixed_packing_count > 0
    needs_pallet_shelves = pallet_only_count + pure_pallet_count + mixed_packing_count > 0
    
    q.put(("log", f"需要的货架类型：\n"))
    if needs_box_shelves:
        q.put(("log", f"  - 需要装箱货架（B类型）\n"))
    if needs_pallet_shelves:
        q.put(("log", f"  - 需要装托货架（A类型）\n"))
    
    result_summary = {
        'decisions': packing_decisions,
        'needs_box_shelves': needs_box_shelves,
        'needs_pallet_shelves': needs_pallet_shelves,
        'error_skus': error_skus,
        'stats': {
            'box_only': box_only_count,
            'pallet_only': pallet_only_count,
            'pure_pallet': pure_pallet_count,
            'mixed': mixed_packing_count,
            'error': error_count
        }
    }
    
    return result_summary

def filter_shelves_by_packing_decision(shelves, packing_summary, pallet_char, box_char, q):
    """
    根据装箱/装托判定结果筛选合适的货架类型
    
    参数:
    - shelves: 所有货架列表
    - packing_summary: 装箱装托判定结果汇总
    - pallet_char: 装托标识字符（如'A'）
    - box_char: 装箱标识字符（如'B'）
    
    返回: 筛选后的货架列表
    """
    q.put(("log", "--- 步骤 2: 货架类型筛选 ---\n"))
    
    needs_box_shelves = packing_summary['needs_box_shelves']
    needs_pallet_shelves = packing_summary['needs_pallet_shelves']
    
    filtered_shelves = []
    
    for shelf in shelves:
        shelf_type = shelf['shelf_type'].strip()
        
        # 检查是否匹配装托货架
        if needs_pallet_shelves and shelf_type == pallet_char:
            filtered_shelves.append(shelf)
            continue
            
        # 检查是否匹配装箱货架  
        if needs_box_shelves and shelf_type == box_char:
            filtered_shelves.append(shelf)
            continue
    
    if not filtered_shelves:
        raise ValueError(f"未找到匹配的货架类型。需要装托货架('{pallet_char}'):{needs_pallet_shelves}, 需要装箱货架('{box_char}'):{needs_box_shelves}")
    
    # 统计筛选结果
    pallet_shelves = [s for s in filtered_shelves if s['shelf_type'] == pallet_char]
    box_shelves = [s for s in filtered_shelves if s['shelf_type'] == box_char]
    
    q.put(("log", f"货架筛选完成：\n"))
    q.put(("log", f"  - 原始货架总数：{len(shelves)} 种\n"))
    q.put(("log", f"  - 筛选后货架总数：{len(filtered_shelves)} 种\n"))
    
    if pallet_shelves:
        q.put(("log", f"  - 装托货架('{pallet_char}')：{len(pallet_shelves)} 种\n"))
    if box_shelves:
        q.put(("log", f"  - 装箱货架('{box_char}')：{len(box_shelves)} 种\n"))
    
    return filtered_shelves

def aggregate_skus_by_shelf_type(data, packing_decisions, shelf_type, num_bins):
    """
    根据货架类型和装箱装托决策进行数据聚合
    
    参数:
    - data: SKU数据
    - packing_decisions: 装箱装托决策结果
    - shelf_type: 货架类型 ('pallet' 或 'box')
    - num_bins: 聚合分箱数量
    """
    if data.empty:
        return pd.DataFrame()
    
    # 筛选出需要使用对应货架类型的SKU
    relevant_skus = []
    for sku_id, decision in packing_decisions.items():
        decision_type = decision['decision']
        if shelf_type == 'pallet' and decision_type in ['pallet_only', 'mixed']:
            relevant_skus.append(sku_id)
        elif shelf_type == 'box' and decision_type in ['box_only', 'mixed']:
            relevant_skus.append(sku_id)
    
    # 筛选数据
    filtered_data = data[data['sku_id'].isin(relevant_skus)].copy()
    if filtered_data.empty:
        return pd.DataFrame()
    
    # 根据货架类型选择对应的LDHWV数据
    if shelf_type == 'pallet':
        # 使用装托LDHWV数据
        filtered_data['L'] = filtered_data['pallet_l']
        filtered_data['D'] = filtered_data['pallet_d']
        filtered_data['H'] = filtered_data['pallet_h']
        filtered_data['W'] = filtered_data['pallet_w']
        filtered_data['V'] = filtered_data['pallet_v']
    else:
        # 使用装箱LDHWV数据
        filtered_data['L'] = filtered_data['box_l']
        filtered_data['D'] = filtered_data['box_d']
        filtered_data['H'] = filtered_data['box_h']
        filtered_data['W'] = filtered_data['box_w']
        filtered_data['V'] = filtered_data['box_v']
    
    # 进行聚合
    filtered_data['l_bin'] = pd.cut(filtered_data['L'], bins=num_bins, labels=False, duplicates='drop')
    filtered_data['d_bin'] = pd.cut(filtered_data['D'], bins=num_bins, labels=False, duplicates='drop')
    agg_data = filtered_data.groupby(['l_bin', 'd_bin']).agg(
        L=('L', 'median'), 
        D=('D', 'median'), 
        H=('H', 'median'), 
        W=('W', 'median'), 
        V=('V', 'median'), 
        count=('L', 'size'), 
        sku_ids=('sku_id', lambda x: list(x))
    ).reset_index()
    
    return agg_data

def aggregate_skus(data, num_bins):
    """保留原始聚合函数以保持向后兼容性"""
    if not data.empty:
        # 使用默认的LDHWV数据（装托数据）
        if 'pallet_l' in data.columns:
            data = data.copy()
            data['L'] = data['pallet_l']
            data['D'] = data['pallet_d']
            data['H'] = data['pallet_h']
            data['W'] = data['pallet_w']
            data['V'] = data['pallet_v']
        
        data['l_bin'] = pd.cut(data['L'], bins=num_bins, labels=False, duplicates='drop')
        data['d_bin'] = pd.cut(data['D'], bins=num_bins, labels=False, duplicates='drop')
        agg_data = data.groupby(['l_bin', 'd_bin']).agg(L=('L', 'median'), D=('D', 'median'), H=('H', 'median'), W=('W', 'median'), V=('V', 'median'), count=('L', 'size'), sku_ids=('sku_id', lambda x: list(x))).reset_index()
        return agg_data
    return pd.DataFrame()

def get_fittable_skus_for_shelf(agg_data, shelf, allow_rotation):
    fittable_skus = []
    for _, sku_group in agg_data.iterrows():
        l, d, w = sku_group['L'], sku_group['D'], sku_group['W']
        if w > shelf['Wp']: continue
        fit_width = -1
        if l <= shelf['Lp'] and d <= shelf['Dp']: fit_width = l
        if allow_rotation and d <= shelf['Lp'] and l <= shelf['Dp']: fit_width = min(l, d) if fit_width != -1 else d
        if fit_width != -1: fittable_skus.append((sku_group.to_dict(), fit_width, int(sku_group['count'])))
    return fittable_skus

def ld_calculator_single(agg_data, shelves, coverage_target, allow_rotation, q):
    best_shelf, min_shelf_count = None, float('inf')
    total_sku_count = agg_data['count'].sum(); total_sku_volume = (agg_data['V'] * agg_data['count']).sum()
    target_count = total_sku_count * coverage_target; target_volume = total_sku_volume * coverage_target
    best_attempt_shelf, max_achieved_count_coverage, max_achieved_volume_coverage = None, 0.0, 0.0
    total_shelves_to_eval = len(shelves); start_time = time.time()
    for i, shelf in enumerate(shelves):
        fittable_skus = get_fittable_skus_for_shelf(agg_data, shelf, allow_rotation)
        placed_count = sum(item[2] for item in fittable_skus)
        placed_volume = sum(item[0]['V'] * item[2] for item in fittable_skus)
        current_count_coverage = placed_count / total_sku_count if total_sku_count > 0 else 0
        if current_count_coverage > max_achieved_count_coverage:
            max_achieved_count_coverage = current_count_coverage
            max_achieved_volume_coverage = placed_volume / total_sku_volume if total_sku_volume > 0 else 0
            best_attempt_shelf = shelf
        if total_sku_count == 0 or placed_count < target_count or placed_volume < target_volume:
            q.put(("progress", (i + 1, total_shelves_to_eval, start_time, f"评估L&D规格 {i+1}/{total_shelves_to_eval} (跳过)")))
            continue
        shelf_count = run_bulk_ffd_packing(fittable_skus, shelf['Lp'])
        if shelf_count < min_shelf_count:
            min_shelf_count, best_shelf = shelf_count, shelf
        q.put(("progress", (i + 1, total_shelves_to_eval, start_time, f"评估L&D规格 {i+1}/{total_shelves_to_eval}")))
    if best_shelf: return ("success", (best_shelf, best_attempt_shelf, max_achieved_count_coverage))
    else: return ("failure", {"shelf": best_attempt_shelf, "count_coverage": max_achieved_count_coverage, "volume_coverage": max_achieved_volume_coverage})

def calculate_three_dimensional_score(solution, params):
    if solution['status'] != 'success': return 0.0, {'count_util': 0.0, 'volume_util': 0.0, 'height_util': 0.0}
    count_utilization = solution.get('coverage_count', 0); volume_utilization = solution.get('coverage_volume', 0)
    warehouse_h = params.get('warehouse_h', 6000); bottom_clearance = params.get('bottom_clearance', 150)
    layer_headroom = params.get('layer_headroom', 100); shelf_thickness = params.get('shelf_thickness', 50)
    usable_vertical_space = warehouse_h - bottom_clearance
    height_utilization = 0.0
    if solution.get('assignments'):
        final_shelves = solution.get('final_shelves', []); assignments = solution.get('assignments', {})
        shelf_counts = solution.get('counts', []); total_weighted_utilization = 0; total_shelf_units = sum(shelf_counts)
        if total_shelf_units > 0:
            for i, shelf in enumerate(final_shelves):
                shelf_h = shelf['H']; single_layer_total_height = shelf_h + shelf_thickness + layer_headroom; num_layers = 0
                if single_layer_total_height > 0 and usable_vertical_space > 0: num_layers = math.floor(usable_vertical_space / single_layer_total_height)
                items_on_this_shelf = assignments.get(i, [])
                total_sku_h_on_shelf = sum(item[0]['H'] * item[2] for item in items_on_this_shelf)
                total_sku_count_on_shelf = sum(item[2] for item in items_on_this_shelf)
                avg_sku_h = total_sku_h_on_shelf / total_sku_count_on_shelf if total_sku_count_on_shelf > 0 else 0
                utilization_for_this_spec = (num_layers * avg_sku_h) / warehouse_h if warehouse_h > 0 else 0
                total_weighted_utilization += utilization_for_this_spec * shelf_counts[i]
            height_utilization = total_weighted_utilization / total_shelf_units
    count_weight = params.get('count_weight', 0.25); volume_weight = params.get('volume_weight', 0.25); height_weight = params.get('height_weight', 0.5)
    total_score = (count_weight * count_utilization + volume_weight * volume_utilization + height_weight * height_utilization)
    metrics = {'count_util': count_utilization, 'volume_util': volume_utilization, 'height_util': height_utilization}
    return total_score, metrics

def h_calculator_three_dimensional(agg_data, operable_data, best_ld_shelf, coverage_target, allow_rotation, params, q):
    q.put(("log", "--- 步骤 3a: 正在生成智能候选高度组合 ---\n")); h_max = params['h_max']
    min_height = operable_data['H'].min(); max_height = operable_data['H'].max()
    percentiles = [10, 25, 40, 50, 60, 75, 85, 90, 95]
    candidate_heights = [round(h) for p in percentiles if min_height <= (h := np.percentile(operable_data['H'], p)) <= h_max]
    height_step = params.get('height_step', 50); boundary_heights = np.arange(min_height, h_max + height_step, height_step)
    all_candidates = sorted(list(set(candidate_heights + list(boundary_heights))))
    q.put(("log", f"基于SKU高度分布，生成了 {len(all_candidates)} 个候选高度点。\n")); min_height_diff = params.get('min_height_diff', 150)
    height_coverage = {h: len(operable_data[operable_data['H'] <= h]) for h in all_candidates}
    min_coverage_ratio = 0.3
    meaningful_heights = [h for h in all_candidates if (height_coverage[h] / len(operable_data) if len(operable_data) > 0 else 0) >= min_coverage_ratio]
    q.put(("log", f"预筛选后，保留了 {len(meaningful_heights)} 个有意义的高度点。\n")); height_combinations = list(combinations(meaningful_heights, 2))
    filtered_combinations = [(h1, h2) for h1, h2 in height_combinations if abs(h2 - h1) >= min_height_diff]; final_combinations = []
    if len(operable_data) > 0:
        for h1, h2 in filtered_combinations:
            coverage1 = height_coverage.get(h1, 0) / len(operable_data); coverage2 = height_coverage.get(h2, 0) / len(operable_data)
            if (coverage1 >= 0.5 or coverage2 >= 0.5) and abs(coverage2 - coverage1) >= 0.1: final_combinations.append(tuple(sorted((h1, h2))))
    final_combinations = sorted(list(set(final_combinations))); max_combinations = 75
    if len(final_combinations) > max_combinations:
        q.put(("log", f"候选组合过多({len(final_combinations)}), 将智能筛选评分最高的 {max_combinations} 种组合进行评估。\n")); final_combinations = final_combinations[:max_combinations]
    if not final_combinations:
        final_combinations = filtered_combinations[:max_combinations]
        if final_combinations: q.put(("log", "警告: 未找到满足智能筛选条件的组合，使用备选方案进行评估。\n"))
    if not final_combinations: raise ValueError("未能找到任何可评估的高度组合。\n\n【建议】\n1. 尝试放宽'最小高度差值'参数。\n2. 检查SKU数据的高度分布是否过于集中。")
    q.put(("log", f"最终筛选出 {len(final_combinations)} 种高度组合进入模拟评估阶段。\n--- 步骤 3b: 正在执行完整方案模拟评估 ---\n")); evaluation_details = []
    min_count_utilization = params.get('min_count_utilization', 0.8); total_combos, start_time = len(final_combinations), time.time()
    for i, (h1, h2) in enumerate(final_combinations):
        ld_specs = best_ld_shelf if isinstance(best_ld_shelf, (list, tuple)) else (best_ld_shelf, best_ld_shelf)

        if params['ld_method'] == 'complementary':
            ld1, ld2 = ld_specs[0], ld_specs[1]
            shelves_to_evaluate = [
                {'Lp': ld1['Lp'], 'Dp': ld1['Dp'], 'Wp': ld1['Wp'], 'H': h1},
                {'Lp': ld1['Lp'], 'Dp': ld1['Dp'], 'Wp': ld1['Wp'], 'H': h2},
                {'Lp': ld2['Lp'], 'Dp': ld2['Dp'], 'Wp': ld2['Wp'], 'H': h1},
                {'Lp': ld2['Lp'], 'Dp': ld2['Dp'], 'Wp': ld2['Wp'], 'H': h2},
            ]
        else:
            ld = ld_specs[0]
            shelves_to_evaluate = [
                {'Lp': ld['Lp'], 'Dp': ld['Dp'], 'Wp': ld['Wp'], 'H': h1},
                {'Lp': ld['Lp'], 'Dp': ld['Dp'], 'Wp': ld['Wp'], 'H': h2}
            ]

        solution = final_allocation_and_counting(agg_data, shelves_to_evaluate, coverage_target, allow_rotation)
        score, metrics = calculate_three_dimensional_score(solution, params)
        status_is_success = solution['status'] == 'success' and metrics['count_util'] >= min_count_utilization
        evaluation_details.append({"h1": h1, "h2": h2, "score": score, "metrics": metrics, "is_feasible": status_is_success, "solution": solution})
        q.put(("progress", (i + 1, total_combos, start_time, f"评估高度组合 {i+1}/{total_combos}")))
    if not evaluation_details: raise ValueError("在所有候选评估中均未能找到任何有效的高度组合方案。")
    feasible_solutions = [d for d in evaluation_details if d['is_feasible']]
    if feasible_solutions: best_solution_detail = max(feasible_solutions, key=lambda x: x['score'])
    else: q.put(("log", f"提示: 未找到满足最低数量利用率({min_count_utilization*100:.1f}%)的方案，将返回综合得分最高的方案。\n")); best_solution_detail = max(evaluation_details, key=lambda x: x['score'])
    best_combo = (best_solution_detail['h1'], best_solution_detail['h2'])
    q.put(("log", f"\n--- 步骤 3 完成：最优高度组合已确定 ---\n")); return sorted(list(best_combo)), evaluation_details

def h_calculator_coverage_driven(data, h_max, p1, p2):
    h1 = np.percentile(data['H'], p1); h2 = np.percentile(data['H'], p2); h1, h2 = min(h1, h_max), min(h2, h_max); return sorted(list(set([round(h1), round(h2)]))), []

def calculate_boundary_effects(data, h_max, step_size, volume_weight, q):
    heights = np.arange(data['H'].min(), h_max + step_size, step_size); results = []; last_count, last_volume = 0, 0
    total_steps, start_time = len(heights), time.time()
    for i, h in enumerate(heights):
        covered_data = data[data['H'] <= h]; current_count, current_volume = len(covered_data), covered_data['V'].sum()
        delta_count, delta_volume = current_count - last_count, current_volume - last_volume
        norm_delta_count = delta_count / len(data) if len(data) > 0 else 0
        norm_delta_volume = delta_volume / data['V'].sum() if data['V'].sum() > 0 else 0
        effect = (1 - volume_weight) * norm_delta_count + volume_weight * norm_delta_volume
        results.append({'h': h, 'effect': effect}); last_count, last_volume = current_count, current_volume
        if i % 10 == 0: q.put(("progress", (i + 1, total_steps, start_time, f"计算边界效应中 {i+1}/{total_steps}")))
    return pd.DataFrame(results)

def identify_candidate_heights(boundary_effects, min_diff, num_candidates=15):
    top_effects = boundary_effects.nlargest(num_candidates * 3, 'effect'); candidates = []
    for _, row in top_effects.iterrows():
        h = row['h']
        if all(abs(h - ch) >= min_diff for ch in candidates): candidates.append(h)
        if len(candidates) >= num_candidates: break
    return sorted(candidates)

def h_calculator_boundary_driven(agg_data, operable_data, best_ld_shelf, h_max, coverage_target, allow_rotation, params, q, best_theoretical_ld, best_theoretical_coverage):
    q.put(("log", "--- 步骤 3a: 正在计算高度边界效应 ---\n")); effects_df = calculate_boundary_effects(operable_data, h_max, params['height_step'], params['volume_weight'], q)
    q.put(("log", "--- 步骤 3b: 正在筛选候选高度点 ---\n")); candidate_heights = identify_candidate_heights(effects_df, params['min_height_diff'])
    if len(candidate_heights) < 2: raise ValueError(f"未能找到满足最小差值({params['min_height_diff']}mm)的候选高度。\n\n【建议】\n请尝试在左侧配置中减小'最小高度差值'参数的值再试。")
    q.put(("log", f"发现 {len(candidate_heights)} 个候选高度点: {[f'{h:.0f}' for h in candidate_heights]}\n")); height_combinations = list(combinations(candidate_heights, 2))
    q.put(("log", f"--- 步骤 3c: 正在评估 {len(height_combinations)} 种高度组合 ---\n"))
    best_combo, min_total_shelves = None, float('inf'); best_attempt_combo, max_coverage_in_h_step = None, 0.0; total_combos, start_time = len(height_combinations), time.time()
    for i, (h1, h2) in enumerate(height_combinations):
        ld_specs = best_ld_shelf if isinstance(best_ld_shelf, (list, tuple)) else (best_ld_shelf, best_ld_shelf)

        if params['ld_method'] == 'complementary':
            ld1, ld2 = ld_specs[0], ld_specs[1]
            shelves_to_evaluate = [
                {'Lp': ld1['Lp'], 'Dp': ld1['Dp'], 'Wp': ld1['Wp'], 'H': h1},
                {'Lp': ld1['Lp'], 'Dp': ld1['Dp'], 'Wp': ld1['Wp'], 'H': h2},
                {'Lp': ld2['Lp'], 'Dp': ld2['Dp'], 'Wp': ld2['Wp'], 'H': h1},
                {'Lp': ld2['Lp'], 'Dp': ld2['Dp'], 'Wp': ld2['Wp'], 'H': h2},
            ]
        else:
            ld = ld_specs[0]
            shelves_to_evaluate = [
                {'Lp': ld['Lp'], 'Dp': ld['Dp'], 'Wp': ld['Wp'], 'H': h1},
                {'Lp': ld['Lp'], 'Dp': ld['Dp'], 'Wp': ld['Wp'], 'H': h2}
            ]

        solution = final_allocation_and_counting(agg_data, shelves_to_evaluate, coverage_target, allow_rotation)
        if solution['coverage_count'] > max_coverage_in_h_step: max_coverage_in_h_step, best_attempt_combo = solution['coverage_count'], (h1, h2)
        if solution['status'] == 'success' and sum(solution['counts']) < min_total_shelves: min_total_shelves, best_combo = sum(solution['counts']), (h1, h2)
        q.put(("progress", (i + 1, total_combos, start_time, f"评估高度组合 {i+1}/{total_combos}")))
    if best_combo is None:
        if best_attempt_combo is None: raise ValueError("在评估高度组合时发生未知错误，未能找到任何有效的组合。")
        h1_best, h2_best = sorted(list(best_attempt_combo))
        raise ValueError(f"计算失败：未能找到满足覆盖率目标 ({coverage_target*100:.1f}%) 的高度组合。最高覆盖率 {max_coverage_in_h_step*100:.2f}% (在 {h1_best:.0f}mm 和 {h2_best:.0f}mm 时取得)")
    q.put(("log", "\n--- 步骤 3 完成：最优高度组合已确定 ---\n")); return sorted(list(best_combo)), []

def final_placement_with_individual_skus_mixed(operable_data, final_shelves, packing_decisions, allow_rotation, q):
    """
    混合模式最终精确装箱与统计函数
    - 根据货架类型和SKU的装箱装托决策使用相应的LDHWV数据
    - 分配SKU时遵循"三维体积利用率最高"原则
    - 精确计算货架需求数量
    """
    q.put(("log", "开始基于混合装箱装托模式进行精确分配与装箱...\n"))
    num_shelf_types = len(final_shelves)
    # assignments的结构：{shelf_idx: [(sku_dict, placed_width), ...]}
    assignments = {i: [] for i in range(num_shelf_types)}
    sku_shelf_assignments = {}

    total_skus = len(operable_data)
    start_time = time.time()
    
    # 1. 逐一分配真实SKU
    for idx, sku in operable_data.iterrows():
        sku_id = sku['sku_id']
        
        # 检查这个SKU是否有有效的装箱装托决策
        if sku_id not in packing_decisions:
            continue
            
        decision = packing_decisions[sku_id]['decision']
        possible_fits = []

        for i, shelf in enumerate(final_shelves):
            shelf_data_type = shelf.get('data_type', 'pallet')  # 默认使用装托数据
            
            # 根据装箱装托决策和货架类型选择适当的LDHWV数据
            use_this_shelf = False
            if decision == 'box_only' and shelf_data_type == 'box':
                use_this_shelf = True
            elif decision == 'pallet_only' and shelf_data_type == 'pallet':
                use_this_shelf = True
            elif decision == 'mixed':
                # 混装SKU可以使用任何类型的货架
                use_this_shelf = True
            
            if not use_this_shelf:
                continue
            
            # 根据货架数据类型选择对应的SKU尺寸数据
            if shelf_data_type == 'pallet':
                sku_l, sku_d, sku_h, sku_w, sku_v = sku['pallet_l'], sku['pallet_d'], sku['pallet_h'], sku['pallet_w'], sku['pallet_v']
            else:
                sku_l, sku_d, sku_h, sku_w, sku_v = sku['box_l'], sku['box_d'], sku['box_h'], sku['box_w'], sku['box_v']
            
            if sku_w > shelf['Wp'] or sku_h > shelf['H']:
                continue
            
            orientations = [{'w': sku_l, 'd': sku_d}]
            if allow_rotation and sku_l != sku_d:
                orientations.append({'w': sku_d, 'd': sku_l})
            
            for ori in orientations:
                w, d = ori['w'], ori['d']
                if w <= shelf['Lp'] and d <= shelf['Dp']:
                    vol_util = 0
                    if shelf['Lp'] > 0 and shelf['Dp'] > 0 and shelf['H'] > 0:
                        vol_util = (w * d * sku_h) / (shelf['Lp'] * shelf['Dp'] * shelf['H'])
                    possible_fits.append({'shelf_idx': i, 'width': w, 'vol_util': vol_util, 'data_type': shelf_data_type})
        
        if possible_fits:
            best_fit = max(possible_fits, key=lambda x: x['vol_util'])
            # 创建增强的SKU字典，包含使用的数据类型信息
            enhanced_sku_dict = sku.to_dict()
            enhanced_sku_dict['used_data_type'] = best_fit['data_type']
            assignments[best_fit['shelf_idx']].append((enhanced_sku_dict, best_fit['width']))
            sku_shelf_assignments[sku_id] = best_fit['shelf_idx']

        if idx > 0 and idx % 200 == 0: # 更新进度
            q.put(("progress", (idx + 1, total_skus, start_time, f"精确分配SKU {idx+1}/{total_skus}")))

    q.put(("log", "所有真实SKU分配完成，正在精确计算货架数...\n"))

    # 2. 精确计算货架数
    final_counts = []
    for i, shelf in enumerate(final_shelves):
        items_on_shelf = assignments.get(i, [])
        if not items_on_shelf:
            final_counts.append(0)
            continue
        
        packing_groups = []
        sorted_items = sorted(items_on_shelf, key=lambda x: x[1])
        for width, group in groupby(sorted_items, key=lambda x: x[1]):
            count = sum(1 for _ in group)
            packing_groups.append(({'placeholder': True}, width, count))
            
        shelf_count = run_bulk_ffd_packing(packing_groups, shelf['Lp'])
        final_counts.append(shelf_count)

    # 3. 产出最终结果
    placed_sku_ids = set(sku_shelf_assignments.keys())
    # 计算覆盖率时使用合适的体积数据
    total_volume = 0
    placed_volume = 0
    
    for _, sku in operable_data.iterrows():
        sku_id = sku['sku_id']
        if sku_id in packing_decisions:
            decision = packing_decisions[sku_id]['decision']
            # 根据决策类型选择体积计算方式
            if decision == 'box_only':
                vol = sku['box_v']
            elif decision == 'pallet_only':
                vol = sku['pallet_v']
            else:  # mixed
                vol = max(sku['pallet_v'], sku['box_v'])  # 使用较大的体积作为参考
            
            total_volume += vol
            if sku_id in placed_sku_ids:
                placed_volume += vol
    
    coverage_count = len(placed_sku_ids) / total_skus if total_skus > 0 else 0
    coverage_volume = placed_volume / total_volume if total_volume > 0 else 0
    
    return {
        'status': 'success',
        'counts': final_counts,
        'coverage_count': coverage_count,
        'coverage_volume': coverage_volume,
        'placed_sku_ids': placed_sku_ids,
        'sku_shelf_assignments': sku_shelf_assignments,
        'assignments': assignments, 
        'final_shelves': final_shelves
    }

def final_placement_with_individual_skus(operable_data, final_shelves, allow_rotation, q):
    """
    v3.6.2 (已修正): 最终精确装箱与统计函数
    - 使用未经聚合的`operable_data`进行计算，确保结果精确。
    - 分配SKU时遵循"三维体积利用率最高"原则。
    - 精确计算货架需求数量。
    """
    q.put(("log", "开始基于真实SKU数据进行精确分配与装箱...\n"))
    num_shelf_types = len(final_shelves)
    # assignments的结构：{shelf_idx: [(sku_dict, placed_width), ...]}
    assignments = {i: [] for i in range(num_shelf_types)}
    sku_shelf_assignments = {}

    total_skus = len(operable_data)
    start_time = time.time()
    
    # 1. 逐一分配真实SKU
    for idx, sku in operable_data.iterrows():
        possible_fits = []
        # 使用装托数据作为默认（向后兼容）
        sku_l = sku.get('pallet_l', sku.get('L', 0))
        sku_d = sku.get('pallet_d', sku.get('D', 0))
        sku_h = sku.get('pallet_h', sku.get('H', 0))
        sku_w = sku.get('pallet_w', sku.get('W', 0))
        
        orientations = [{'w': sku_l, 'd': sku_d}]
        if allow_rotation and sku_l != sku_d:
            orientations.append({'w': sku_d, 'd': sku_l})

        for i, shelf in enumerate(final_shelves):
            if sku_w > shelf['Wp'] or sku_h > shelf['H']:
                continue
            
            for ori in orientations:
                w, d = ori['w'], ori['d']
                if w <= shelf['Lp'] and d <= shelf['Dp']:
                    vol_util = 0
                    if shelf['Lp'] > 0 and shelf['Dp'] > 0 and shelf['H'] > 0:
                        vol_util = (w * d * sku_h) / (shelf['Lp'] * shelf['Dp'] * shelf['H'])
                    possible_fits.append({'shelf_idx': i, 'width': w, 'vol_util': vol_util})
        
        if possible_fits:
            best_fit = max(possible_fits, key=lambda x: x['vol_util'])
            assignments[best_fit['shelf_idx']].append((sku.to_dict(), best_fit['width']))
            sku_shelf_assignments[sku['sku_id']] = best_fit['shelf_idx']

        if idx > 0 and idx % 200 == 0: # 更新进度
            q.put(("progress", (idx + 1, total_skus, start_time, f"精确分配SKU {idx+1}/{total_skus}")))

    q.put(("log", "所有真实SKU分配完成，正在精确计算货架数...\n"))

    # 2. 精确计算货架数
    final_counts = []
    for i, shelf in enumerate(final_shelves):
        items_on_shelf = assignments.get(i, [])
        if not items_on_shelf:
            final_counts.append(0)
            continue
        
        packing_groups = []
        sorted_items = sorted(items_on_shelf, key=lambda x: x[1])
        for width, group in groupby(sorted_items, key=lambda x: x[1]):
            count = sum(1 for _ in group)
            packing_groups.append(({'placeholder': True}, width, count))
            
        shelf_count = run_bulk_ffd_packing(packing_groups, shelf['Lp'])
        final_counts.append(shelf_count)

    # 3. 产出最终结果
    placed_sku_ids = set(sku_shelf_assignments.keys())
    placed_skus_df = operable_data[operable_data['sku_id'].isin(placed_sku_ids)]
    
    coverage_count = len(placed_sku_ids) / total_skus if total_skus > 0 else 0
    
    # 计算体积覆盖率时使用适当的体积数据
    total_volume = operable_data.get('pallet_v', operable_data.get('V', pd.Series([0]))).sum()
    placed_volume = placed_skus_df.get('pallet_v', placed_skus_df.get('V', pd.Series([0]))).sum()
    coverage_volume = placed_volume / total_volume if total_volume > 0 else 0
    
    # 关键修正：不再执行二次聚合，直接返回最精确的assignments
    return {
        'status': 'success',
        'counts': final_counts,
        'coverage_count': coverage_count,
        'coverage_volume': coverage_volume,
        'placed_sku_ids': placed_sku_ids,
        'sku_shelf_assignments': sku_shelf_assignments,
        'assignments': assignments, 
        'final_shelves': final_shelves
    }

def final_allocation_and_counting(agg_data, final_shelves, coverage_target, allow_rotation):
    num_shelf_types = len(final_shelves)
    assignments = {i: [] for i in range(num_shelf_types)}
    sku_shelf_assignments = {}
    for _, sku_group in agg_data.iterrows():
        possible_fits = []
        orientations = [{'w': sku_group['L'], 'd': sku_group['D']}]
        if allow_rotation and sku_group['L'] != sku_group['D']: orientations.append({'w': sku_group['D'], 'd': sku_group['L']})
        for i, shelf in enumerate(final_shelves):
            if sku_group['W'] > shelf['Wp'] or sku_group['H'] > shelf['H']: continue
            for ori in orientations:
                w, d = ori['w'], ori['d']
                if w <= shelf['Lp'] and d <= shelf['Dp']:
                    rem_space = shelf['Lp'] * shelf['Dp'] - w * d
                    possible_fits.append({'shelf_idx': i, 'width': w, 'rem_space': rem_space})
        if not possible_fits: continue
        best_fit = min(possible_fits, key=lambda x: x['rem_space'])
        item_to_place = (sku_group.to_dict(), best_fit['width'], int(sku_group['count']))
        assignments[best_fit['shelf_idx']].append(item_to_place)
        for sku_id in sku_group['sku_ids']: sku_shelf_assignments[sku_id] = best_fit['shelf_idx']
    total_sku_count = agg_data['count'].sum(); total_sku_volume = (agg_data['V'] * agg_data['count']).sum()
    placed_count = sum(item[2] for lst in assignments.values() for item in lst)
    placed_volume = sum(item[0]['V'] * item[2] for lst in assignments.values() for item in lst)
    coverage_count = placed_count / total_sku_count if total_sku_count > 0 else 0
    coverage_volume = placed_volume / total_sku_volume if total_sku_volume > 0 else 0
    if total_sku_count == 0 or placed_count < total_sku_count * coverage_target or placed_volume < total_sku_volume * coverage_target:
        return {'status': 'failure', 'counts': [0] * num_shelf_types, 'coverage_count': coverage_count, 'coverage_volume': coverage_volume, 'placed_sku_ids': set(), 'sku_shelf_assignments': {}, 'assignments': assignments, 'final_shelves': final_shelves}
    final_counts = [run_bulk_ffd_packing(assignments[i], final_shelves[i]['Lp']) for i in range(num_shelf_types)]
    placed_sku_ids = {sid for lst in assignments.values() for item in lst for sid in item[0]['sku_ids']}
    return {'status': 'success', 'counts': final_counts, 'coverage_count': coverage_count, 'coverage_volume': coverage_volume, 'placed_sku_ids': placed_sku_ids, 'sku_shelf_assignments': sku_shelf_assignments, 'assignments': assignments, 'final_shelves': final_shelves}

def calculate_fit_capacity(shelf_length, bin_state, item_width):
    if item_width <= 0: return 0
    count_on_shelf = bin_state['count']
    if count_on_shelf == 0:
        first_item_cost = item_width + 2 * SIDE_SPACING
        if first_item_cost > shelf_length: return 0
        item_cost = item_width + INTER_SPACING
        return 1 + math.floor((shelf_length - first_item_cost) / item_cost) if item_cost > 0 else 1
    else:
        current_used_len = bin_state['sum_widths'] + (count_on_shelf - 1) * INTER_SPACING + 2 * SIDE_SPACING
        item_cost = item_width + INTER_SPACING
        if item_cost <= 0 or (shelf_length - current_used_len) < item_cost: return 0
        return math.floor((shelf_length - current_used_len) / item_cost)

def run_bulk_ffd_packing(sku_groups, shelf_length):
    if not sku_groups: return 0
    sorted_groups = sorted(sku_groups, key=lambda x: x[1], reverse=True); bins = []
    for _, width, count in sorted_groups:
        items_left_to_place = int(count)
        if items_left_to_place <= 0: continue
        for b in bins:
            capacity = calculate_fit_capacity(shelf_length, b, width)
            if capacity > 0:
                num_to_add = min(items_left_to_place, capacity)
                b['count'] += num_to_add; b['sum_widths'] += num_to_add * width
                items_left_to_place -= num_to_add
            if items_left_to_place == 0: break
        if items_left_to_place > 0:
            capacity_new_bin = calculate_fit_capacity(shelf_length, {'count': 0, 'sum_widths': 0}, width)
            if capacity_new_bin > 0:
                num_new_bins = math.ceil(items_left_to_place / capacity_new_bin)
                for _ in range(num_new_bins):
                    items_for_this_bin = min(items_left_to_place, capacity_new_bin)
                    bins.append({'count': items_for_this_bin, 'sum_widths': items_for_this_bin * width})
                    items_left_to_place -= items_for_this_bin
                    if items_left_to_place == 0: break
    return len(bins)

def get_detailed_unplaced_reasons_by_sku_id(unplaced_skus, final_shelves, h_max, allow_rotation):
    reasons = {}
    for _, sku in unplaced_skus.iterrows():
        sku_id = sku['sku_id']
        if sku['H'] > h_max:
            reasons[sku_id] = f"高度超限: SKU高({sku['H']:.0f}) > 最大允许高({h_max:.0f})"
            continue

        can_fit_any_shelf = False
        for shelf in final_shelves:
            if sku['W'] > shelf['Wp']: continue
            if sku['H'] > shelf['H']: continue
            can_fit_unrotated = sku['L'] <= shelf['Lp'] and sku['D'] <= shelf['Dp']
            can_fit_rotated = allow_rotation and (sku['D'] <= shelf['Lp'] and sku['L'] <= shelf['Dp'])
            if can_fit_unrotated or can_fit_rotated:
                can_fit_any_shelf = True
                break
        
        if not can_fit_any_shelf:
            # 权重最高的货架，用于提供代表性的尺寸信息
            representative_shelf = final_shelves[0]
            min_h_available = min(s['H'] for s in final_shelves) if final_shelves else 0
            if sku['W'] > representative_shelf['Wp']:
                reasons[sku_id] = f"超重: SKU重({sku['W']:.1f}kg) > 货架承重({representative_shelf['Wp']:.1f}kg)"
            elif sku['H'] > min_h_available:
                 reasons[sku_id] = f"过高: SKU高({sku['H']:.0f}) > 所有推荐货架的净高"
            else:
                reasons[sku_id] = f"尺寸不匹配: L/D({sku['L']:.0f}x{sku['D']:.0f})与所有推荐货架尺寸均不符"
        else:
              # SKU物理尺寸可以安放，但仍未被安放 - 可能是装箱装托类型不匹配
              reasons[sku_id] = "货架类型不匹配: SKU的装箱装托要求与推荐货架类型不符，或算法优化过程中被其他SKU占用"
    return reasons


def write_results_to_excel(original_file_path, placed_sku_ids, detailed_reasons, sku_id_col_name, sku_shelf_assignments=None, final_shelves=None):
    try:
        original_df = pd.read_excel(original_file_path); original_df[sku_id_col_name] = original_df[sku_id_col_name].astype(str)
        def get_status(sku_id): return "成功安放" if sku_id in placed_sku_ids else "未能安放"
        def get_reason(sku_id): return detailed_reasons.get(sku_id, "") if sku_id not in placed_sku_ids else ""
        def get_shelf_spec(sku_id):
            if sku_id in placed_sku_ids and sku_shelf_assignments and final_shelves:
                shelf_idx = sku_shelf_assignments.get(sku_id)
                if shelf_idx is not None and shelf_idx < len(final_shelves):
                    shelf = final_shelves[shelf_idx]; return f"{shelf['Lp']:.0f}×{shelf['Dp']:.0f}×{shelf['H']:.0f}mm"
            return ""
        original_df['[安放状态]'] = original_df[sku_id_col_name].apply(get_status)
        original_df['[未安放原因]'] = original_df[sku_id_col_name].apply(get_reason)
        original_df['[货架规格]'] = original_df[sku_id_col_name].apply(get_shelf_spec)
        base, ext = os.path.splitext(original_file_path); timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"{base}_results_{timestamp}{ext}"; original_df.to_excel(output_path, index=False)
        return True, output_path, None
    except Exception as e: return False, None, str(e)

def calculate_ldh_utilization(final_solution, params):
    log_lines = ["\n" + "-" * 70 + "\n" + " " * 24 + ">>> 三维空间利用率分析 <<<\n" + "-" * 70]
    assignments = final_solution.get('assignments', {}); final_shelves = final_solution.get('final_shelves', []); counts = final_solution.get('counts', [])
    for i, shelf_spec in enumerate(final_shelves):
        spec_line = f"\n对于规格 {i+1} ({shelf_spec['Lp']:.0f}L x {shelf_spec['Dp']:.0f}D x {shelf_spec['H']:.0f}H):"
        log_lines.append(spec_line); items_on_this_shelf = assignments.get(i, []); shelf_count = counts[i]
        if shelf_count == 0 or not items_on_this_shelf:
            log_lines.append("  - 未分配SKU，利用率均为 0%"); continue
        
        # vReX 1.0.1 修正: 直接处理精确的 (sku_dict, placed_width) 元组列表
        total_placed_width = sum(item[1] for item in items_on_this_shelf)
        total_item_count = len(items_on_this_shelf)
        
        # 修正深度利用率计算：根据实际放置情况计算使用的深度
        total_depth_sum = 0
        for item in items_on_this_shelf:
            sku_dict = item[0]
            placed_width = item[1]
            # 判断是否旋转：如果放置宽度等于SKU长度，则未旋转；否则已旋转
            if abs(placed_width - sku_dict['L']) < 1:  # 允许1mm误差
                # 未旋转，使用原始深度
                actual_depth = sku_dict['D']
            else:
                # 已旋转，长度变成了深度
                actual_depth = sku_dict['L']
            total_depth_sum += actual_depth
        
        total_sku_h_sum = sum(item[0]['H'] for item in items_on_this_shelf)

        total_available_length = (shelf_spec['Lp'] - 2 * SIDE_SPACING) * shelf_count
        l_util = total_placed_width / total_available_length if total_available_length > 0 else 0
        log_lines.append(f"  - L-利用率 (长度): {l_util:.2%}")
        
        weighted_avg_depth = total_depth_sum / total_item_count if total_item_count > 0 else 0
        d_util = weighted_avg_depth / shelf_spec['Dp'] if shelf_spec['Dp'] > 0 else 0
        log_lines.append(f"  - D-利用率 (深度): {d_util:.2%}")

        # 修正高度利用率计算：使用货架净高度而不是仓库总高度
        avg_sku_h = total_sku_h_sum / total_item_count if total_item_count > 0 else 0
        h_util = avg_sku_h / shelf_spec['H'] if shelf_spec['H'] > 0 else 0
        log_lines.append(f"  - H-利用率 (高度): {h_util:.2%}")
    log_lines.append("-" * 70); return "\n".join(log_lines)

def pre_calculation_worker(q, params):
    try:
        q.put(("log", "--- 步骤 0: 正在读取与预处理文件 ---\n")); q.put(("progress_update", ("正在读取SKU数据...", 0.1)))
        raw_data = read_excel_data(
            params['excel_file'], 
            params['sku_sheet'], 
            params['cols'], 
            params['can_box_col'],
            params['can_pallet_col'],
            params['box_count_col'],
            params['pallet_count_col'],
            params['boxes_per_pallet_col'],
            params['pallet_preset'],
            params['box_preset']
        )
        q.put(("log", f"从'{params['sku_sheet']}'Sheet读取了 {len(raw_data)} 条有效的SKU数据。\n"))
        
        q.put(("progress_update", ("正在读取货架数据...", 0.4)))
        shelves = read_shelf_params(params['excel_file'], params['shelf_sheet'], params['shelf_type_col'], params['shelf_cols'])
        q.put(("log", f"从'{params['shelf_sheet']}'Sheet读取了 {len(shelves)} 种货架规格。\n"))
        
        q.put(("progress_update", ("正在进行数据相关性检验...", 0.7)))
        
        # 在相关性分析前，需要先进行装箱装托判定并添加统一的L、D、H列
        packing_summary = analyze_packing_decision(raw_data, params['pallet_decimal_threshold'], q)
        raw_data_with_ldh = add_unified_ldh_columns(raw_data, packing_summary['decisions'])
        
        time.sleep(0.5); corr_label, corr_val, grade = correlation_analysis(raw_data_with_ldh)
        q.put(("pre_calculation_done", (raw_data, raw_data_with_ldh, shelves, corr_label, corr_val, grade, packing_summary)))
    except Exception as e: q.put(("error", f"在文件读取或预处理阶段发生错误：\n{str(e)}"))

def calculation_worker(q, params, raw_data, shelves, agg_data=None):
    current_step = "初始化"
    try:
        # --- 步骤 1: 装箱/装托判定分析 ---
        current_step = "步骤1: 装箱/装托判定分析"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
        
        # 首先进行装箱/装托判定（如果没有预计算的话）
        if 'packing_summary' in params and params['packing_summary']:
            packing_summary = params['packing_summary']
            q.put(("log", "使用预计算的装箱/装托判定结果...\n"))
        else:
            packing_summary = analyze_packing_decision(raw_data, params['pallet_decimal_threshold'], q)
        
        # 根据判定结果筛选货架
        filtered_shelves = filter_shelves_by_packing_decision(
            shelves, packing_summary, params['pallet_char'], params['box_char'], q
        )
        
        q.put(("log", f"--- {current_step} 完成 ---\n"))
        
        # --- 步骤 2: 数据聚合（基于筛选后的数据和货架类型） ---
        current_step = "步骤2: 数据聚合"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
        
        h_max = params['h_max']
        # 首先过滤高度并创建带LDH列的可操作数据
        temp_data = raw_data[(raw_data['pallet_h'] <= h_max) | (raw_data['box_h'] <= h_max)].copy()
        q.put(("log", f"已过滤掉 {len(raw_data) - len(temp_data)} 个装托和装箱高度都超过 {h_max}mm 的SKU。\n"))
        if len(temp_data) == 0: 
            raise ValueError("所有SKU的装托和装箱高度都超过最大允许高度，无法继续计算。")
        
        # 为过滤后的数据添加统一的LDH列，用于H计算函数
        operable_data = add_unified_ldh_columns(temp_data, packing_summary['decisions'])
        
        # 分别为装托货架和装箱货架聚合数据
        if agg_data is None:
            q.put(("log", "缓存未命中，正在为不同货架类型进行SKU聚合...\n"))
            
            # 为装托货架聚合数据
            pallet_agg_data = aggregate_skus_by_shelf_type(temp_data, packing_summary['decisions'], 'pallet', NUM_BINS_FOR_AGGREGATION)
            q.put(("log", f"装托货架数据聚合完成，规格组数量: {len(pallet_agg_data)}\n"))
            
            # 为装箱货架聚合数据
            box_agg_data = aggregate_skus_by_shelf_type(temp_data, packing_summary['decisions'], 'box', NUM_BINS_FOR_AGGREGATION)
            q.put(("log", f"装箱货架数据聚合完成，规格组数量: {len(box_agg_data)}\n"))
            
            agg_data = {'pallet': pallet_agg_data, 'box': box_agg_data}
            q.put(("agg_data_computed", (agg_data, operable_data)))
        else:
            q.put(("log", "聚合数据缓存命中，跳过聚合步骤。\n"))
            q.put(("agg_data_computed", (agg_data, operable_data)))
        
        q.put(("log", f"--- {current_step} 完成 ---\n"))

        # --- 步骤 3: 分别为装托和装箱货架计算最优L&D规格 ---
        current_step = f"步骤3: 计算最优L&D规格 (使用 {params['ld_method']} 算法)"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
        
        # 分别为装托和装箱货架计算最优规格
        best_ld_results = {}
        
        # 装托货架计算
        if packing_summary['needs_pallet_shelves'] and len(agg_data['pallet']) > 0:
            pallet_shelves = [s for s in filtered_shelves if s['shelf_type'] == params['pallet_char']]
            q.put(("log", f"正在为装托货架计算最优L&D规格...\n"))
            
            if params['ld_method'] == 'complementary':
                status, ld_return_value = ld_calculator_complementary(agg_data['pallet'], pallet_shelves, params['coverage_target'], params['allow_rotation'], q, params)
            else:
                status, ld_return_value = ld_calculator_single(agg_data['pallet'], pallet_shelves, params['coverage_target'], params['allow_rotation'], q)
            
            if status == "success":
                best_ld_results['pallet'] = ld_return_value
                q.put(("log", f"装托货架最优L&D规格计算完成\n"))
            else:
                q.put(("log", f"警告: 装托货架L&D计算失败\n"))
        
        # 装箱货架计算
        if packing_summary['needs_box_shelves'] and len(agg_data['box']) > 0:
            box_shelves = [s for s in filtered_shelves if s['shelf_type'] == params['box_char']]
            q.put(("log", f"正在为装箱货架计算最优L&D规格...\n"))
            
            if params['ld_method'] == 'complementary':
                status, ld_return_value = ld_calculator_complementary(agg_data['box'], box_shelves, params['coverage_target'], params['allow_rotation'], q, params)
            else:
                status, ld_return_value = ld_calculator_single(agg_data['box'], box_shelves, params['coverage_target'], params['allow_rotation'], q)
            
            if status == "success":
                best_ld_results['box'] = ld_return_value
                q.put(("log", f"装箱货架最优L&D规格计算完成\n"))
            else:
                q.put(("log", f"警告: 装箱货架L&D计算失败\n"))
        
        if not best_ld_results:
            q.put(("error", "L&D计算失败：无法为任何货架类型找到满足要求的规格。"))
            return
        
        best_ld_for_h_calc = best_ld_results

        q.put(("log", f"--- {current_step} 完成 ---\n")); current_step = f"步骤4: 计算最优H规格 (使用 {params['h_method']} 算法)"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
        h_calculators = {'manual': h_calculator_coverage_driven, 'three_dimensional': h_calculator_three_dimensional, 'boundary': h_calculator_boundary_driven}
        h_method = params['h_method']; h_cand, eval_details = [], []

        if h_method == 'manual':
            h_cand, eval_details = h_calculators[h_method](operable_data, h_max, params['p1'], params['p2'])
        else:
            # 使用装托数据进行高度计算（优先），如果没有则使用装箱数据
            primary_ld_result = best_ld_for_h_calc.get('pallet', best_ld_for_h_calc.get('box'))
            primary_agg_data = agg_data.get('pallet', agg_data.get('box'))
            h_cand, eval_details = h_calculators[h_method](primary_agg_data, operable_data, primary_ld_result, params['coverage_target'], params['allow_rotation'], params, q)

       # --- 步骤 5: 组合装托和装箱货架规格，执行精确装箱 ---
        current_step = "步骤5: 最终规格确定与精确装箱"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
        optimal_shelves = []

        # 组合装托和装箱货架的最优规格
        for shelf_type, ld_result in best_ld_results.items():
            if len(h_cand) >= 2:
                h1, h2 = h_cand[0], h_cand[1]
                
                if params['ld_method'] == 'complementary':
                    ld1, ld2 = ld_result[0], ld_result[1]
                    shelf_type_char = params['pallet_char'] if shelf_type == 'pallet' else params['box_char']
                    optimal_shelves.extend([
                        {'Lp': ld1['Lp'], 'Dp': ld1['Dp'], 'Wp': ld1['Wp'], 'H': h1, 'shelf_type': shelf_type_char, 'data_type': shelf_type},
                        {'Lp': ld1['Lp'], 'Dp': ld1['Dp'], 'Wp': ld1['Wp'], 'H': h2, 'shelf_type': shelf_type_char, 'data_type': shelf_type},
                        {'Lp': ld2['Lp'], 'Dp': ld2['Dp'], 'Wp': ld2['Wp'], 'H': h1, 'shelf_type': shelf_type_char, 'data_type': shelf_type},
                        {'Lp': ld2['Lp'], 'Dp': ld2['Dp'], 'Wp': ld2['Wp'], 'H': h2, 'shelf_type': shelf_type_char, 'data_type': shelf_type},
                    ])
                else:
                    ld = ld_result[0] if isinstance(ld_result, tuple) else ld_result
                    shelf_type_char = params['pallet_char'] if shelf_type == 'pallet' else params['box_char']
                    optimal_shelves.extend([
                        {'Lp': ld['Lp'], 'Dp': ld['Dp'], 'Wp': ld['Wp'], 'H': h1, 'shelf_type': shelf_type_char, 'data_type': shelf_type},
                        {'Lp': ld['Lp'], 'Dp': ld['Dp'], 'Wp': ld['Wp'], 'H': h2, 'shelf_type': shelf_type_char, 'data_type': shelf_type}
                    ])

        if not optimal_shelves:
            raise ValueError("未能确定最终的货架规格，计算中止。")
        
        q.put(("log", f"最优货架规格已确定（共{len(optimal_shelves)}种），开始执行精确装箱...\n"))
        
        # 添加覆盖率预估验证
        q.put(("log", "正在进行精确装箱前的覆盖率预估...\n"))
        pre_estimate = final_placement_with_individual_skus_mixed(operable_data, optimal_shelves, packing_summary['decisions'], params['allow_rotation'], q)
        actual_coverage_count = len(pre_estimate['placed_sku_ids']) / len(operable_data) if len(operable_data) > 0 else 0
        q.put(("log", f"预估结果：实际SKU覆盖率约为 {actual_coverage_count*100:.1f}%\n"))
        
        if actual_coverage_count < params['coverage_target'] * 0.8:  # 如果实际覆盖率低于目标的80%
            q.put(("log", f"警告：实际覆盖率({actual_coverage_count*100:.1f}%)显著低于目标({params['coverage_target']*100:.1f}%)，可能需要调整参数或检查数据匹配性。\n"))
        
        final_solution = pre_estimate  # 直接使用预估结果，避免重复计算
        q.put(("log", "精确装箱计算完成。\n"))

        if not final_solution['placed_sku_ids']:
              raise ValueError("计算失败：即使在最优货架标准下，也未能安放任何SKU。")

        # --- 步骤 6: 结果整理与输出 ---
        current_step = "步骤6: 结果整理"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
        q.put(("result", (optimal_shelves, final_solution, params['coverage_target'], packing_summary)))
        utilization_log = calculate_ldh_utilization(final_solution, params); q.put(("log", utilization_log))
        
        placed_sku_ids = final_solution['placed_sku_ids']; unplaced_skus_df = operable_data[~operable_data['sku_id'].isin(placed_sku_ids)]
        detailed_reasons = get_detailed_unplaced_reasons_by_sku_id(unplaced_skus_df, optimal_shelves, h_max, params['allow_rotation'])
        q.put(("diagnostics", (unplaced_skus_df, detailed_reasons)));
        q.put(("visualization_data", (operable_data, eval_details, final_solution)))

        if params['excel_file']:
            success, path, error_msg = write_results_to_excel(params['excel_file'], placed_sku_ids, detailed_reasons, params['cols']['sku_id'], final_solution.get('sku_shelf_assignments'), optimal_shelves)
            if success: q.put(("log", f"\n结果已成功写入到新文件:\n{path}\n"))
            else: q.put(("log", f"\n!!!!!! 写入Excel文件失败 !!!!!!\n错误: {error_msg}\n"))

        q.put(("done", "计算完成！"))
    except Exception as e:
        q.put(("error", f"在 [{current_step}] 阶段发生了一个意外错误。\n请检查您的输入数据和参数配置是否正确。\n\n技术细节: {str(e)}"))


# #############################################################################
# --- 图形用户界面 (GUI) ---
# #############################################################################

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("货架配置优化工具 ProVersion-4.0.0")
        self.geometry("1280x850")
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)
        
        # 绑定窗口关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.queue = queue.Queue(); self.current_params = None; self.cache = {}
        
        self.frame_left = ctk.CTkScrollableFrame(self, width=380, corner_radius=0, label_text="输入与配置", label_font=ctk.CTkFont(size=16, weight="bold"))
        self.frame_left.grid(row=0, column=0, rowspan=3, sticky="nsw")
        
        r = 0
        ctk.CTkLabel(self.frame_left, text="文件选择", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(10, 10)); r+=1
        self.excel_file_path_label = ctk.CTkLabel(self.frame_left, text="未选择Excel文件", text_color="gray", wraplength=300); self.excel_file_path_label.grid(row=r, column=0, columnspan=2, padx=20); r+=1
        ctk.CTkButton(self.frame_left, text="选择Excel文件", command=self.select_excel_file).grid(row=r, column=0, padx=(20,5), pady=10, sticky="ew")
        ctk.CTkButton(self.frame_left, text="粘贴", width=60, command=self.paste_excel_path).grid(row=r, column=1, padx=(5,20), pady=10, sticky="ew"); r+=1
        
        # Sheet名称配置
        ctk.CTkLabel(self.frame_left, text="Sheet名称").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_sku_sheet = ctk.CTkEntry(self.frame_left, placeholder_text="SKU数据的Sheet名"); self.entry_sku_sheet.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); r+=1
        ctk.CTkLabel(self.frame_left, text="货架Sheet名称").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_shelf_sheet = ctk.CTkEntry(self.frame_left, placeholder_text="货架数据的Sheet名"); self.entry_shelf_sheet.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); r+=1
        
        # 装箱装托配置
        ctk.CTkLabel(self.frame_left, text="装箱装托配置", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(20, 10)); r+=1
        ctk.CTkLabel(self.frame_left, text="可装箱列名(Y/N)").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_can_box_col = ctk.CTkEntry(self.frame_left, placeholder_text="如：可装箱"); self.entry_can_box_col.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); r+=1
        ctk.CTkLabel(self.frame_left, text="可装托列名(Y/N)").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_can_pallet_col = ctk.CTkEntry(self.frame_left, placeholder_text="如：可装托"); self.entry_can_pallet_col.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); r+=1
        ctk.CTkLabel(self.frame_left, text="装箱数列名").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_box_count_col = ctk.CTkEntry(self.frame_left, placeholder_text="如：装箱数"); self.entry_box_count_col.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); r+=1
        ctk.CTkLabel(self.frame_left, text="装托数列名").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_pallet_count_col = ctk.CTkEntry(self.frame_left, placeholder_text="如：装托数"); self.entry_pallet_count_col.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); r+=1
        ctk.CTkLabel(self.frame_left, text="单托箱数列名").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_boxes_per_pallet_col = ctk.CTkEntry(self.frame_left, placeholder_text="如：单托箱数"); self.entry_boxes_per_pallet_col.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); r+=1
        ctk.CTkLabel(self.frame_left, text="货架类型标识列名").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_shelf_type_col = ctk.CTkEntry(self.frame_left, placeholder_text="如：货架类型"); self.entry_shelf_type_col.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); r+=1
        ctk.CTkLabel(self.frame_left, text="装托标识字符").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_pallet_char = ctk.CTkEntry(self.frame_left, placeholder_text="如：A"); self.entry_pallet_char.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); r+=1
        ctk.CTkLabel(self.frame_left, text="装箱标识字符").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_box_char = ctk.CTkEntry(self.frame_left, placeholder_text="如：B"); self.entry_box_char.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); r+=1
        ctk.CTkLabel(self.frame_left, text="装托小数阈值").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_pallet_decimal_threshold = ctk.CTkEntry(self.frame_left); self.entry_pallet_decimal_threshold.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); self.entry_pallet_decimal_threshold.insert(0, "0.3"); r+=1

        ctk.CTkLabel(self.frame_left, text="核心参数配置", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(20, 10)); r+=1
        ctk.CTkLabel(self.frame_left, text="覆盖率目标 (%)").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_coverage = ctk.CTkEntry(self.frame_left); self.entry_coverage.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); self.entry_coverage.insert(0, "90"); r+=1
        ctk.CTkLabel(self.frame_left, text="最大允许高度 (mm)").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_hmax = ctk.CTkEntry(self.frame_left); self.entry_hmax.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); self.entry_hmax.insert(0, "1800"); r+=1
        self.check_allow_rotation = ctk.CTkCheckBox(self.frame_left, text="允许SKU旋转90°放置"); self.check_allow_rotation.grid(row=r, column=0, columnspan=2, padx=20, pady=10, sticky="w"); self.check_allow_rotation.select(); r+=1

        ctk.CTkLabel(self.frame_left, text="L&D计算方法", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(20, 10)); r+=1
        self.ld_method_var = tk.StringVar(value="complementary")
        ctk.CTkRadioButton(self.frame_left, text="单一货架算法 (2种最终规格)", variable=self.ld_method_var, value="single", command=self.toggle_ld_params).grid(row=r, column=0, columnspan=2, padx=20, pady=(5,0), sticky="w"); r+=1
        ctk.CTkRadioButton(self.frame_left, text="两种货架算法 (互补式, 4种最终规格)", variable=self.ld_method_var, value="complementary", command=self.toggle_ld_params).grid(row=r, column=0, columnspan=2, padx=20, pady=(5,0), sticky="w"); r+=1

        self.frame_complementary_params = ctk.CTkFrame(self.frame_left, fg_color="transparent")
        self.frame_complementary_params.grid(row=r, column=0, columnspan=2, padx=(40, 20), pady=0, sticky="ew")
        ctk.CTkLabel(self.frame_complementary_params, text="互补性-面积比例阈值").grid(row=0, column=0, sticky="w")
        self.entry_area_threshold = ctk.CTkEntry(self.frame_complementary_params); self.entry_area_threshold.grid(row=0, column=1, padx=(10,0), sticky="ew"); self.entry_area_threshold.insert(0, "0.7")
        ctk.CTkLabel(self.frame_complementary_params, text="互补性-尺寸比例阈值").grid(row=1, column=0, pady=5, sticky="w")
        self.entry_size_threshold = ctk.CTkEntry(self.frame_complementary_params); self.entry_size_threshold.grid(row=1, column=1, padx=(10,0), pady=5, sticky="ew"); self.entry_size_threshold.insert(0, "0.8")
        r+=1

        ctk.CTkLabel(self.frame_left, text="H高度计算方法", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(20, 10)); r+=1
        self.h_method_var = tk.StringVar(value="three_dimensional")
        ctk.CTkRadioButton(self.frame_left, text="三维评分体系算法 (推荐)", variable=self.h_method_var, value="three_dimensional", command=self.toggle_h_params).grid(row=r, column=0, columnspan=2, padx=20, pady=(5,0), sticky="w"); r+=1
        ctk.CTkRadioButton(self.frame_left, text="边界效应算法", variable=self.h_method_var, value="boundary", command=self.toggle_h_params).grid(row=r, column=0, columnspan=2, padx=20, pady=(5,0), sticky="w"); r+=1
        ctk.CTkRadioButton(self.frame_left, text="手动百分位算法", variable=self.h_method_var, value="manual", command=self.toggle_h_params).grid(row=r, column=0, columnspan=2, padx=20, pady=(5,10), sticky="w"); r+=1
        
        self.frame_three_dimensional_params = ctk.CTkFrame(self.frame_left, fg_color="transparent")
        self.frame_boundary_params = ctk.CTkFrame(self.frame_left, fg_color="transparent"); self.frame_manual_params = ctk.CTkFrame(self.frame_left, fg_color="transparent")
        
        ctk.CTkLabel(self.frame_three_dimensional_params, text="高度步长 (mm)").grid(row=0, column=0, sticky="w"); self.entry_height_step_3d = ctk.CTkEntry(self.frame_three_dimensional_params); self.entry_height_step_3d.grid(row=0, column=1, padx=(10,0), sticky="ew"); self.entry_height_step_3d.insert(0, "50")
        ctk.CTkLabel(self.frame_three_dimensional_params, text="最小高度差值 (mm)").grid(row=1, column=0, sticky="w"); self.entry_min_height_diff_3d = ctk.CTkEntry(self.frame_three_dimensional_params); self.entry_min_height_diff_3d.grid(row=1, column=1, padx=(10,0), pady=5, sticky="ew"); self.entry_min_height_diff_3d.insert(0, "150")
        ctk.CTkLabel(self.frame_three_dimensional_params, text="数量利用率权重").grid(row=2, column=0, sticky="w"); self.entry_count_weight = ctk.CTkEntry(self.frame_three_dimensional_params); self.entry_count_weight.grid(row=2, column=1, padx=(10,0), sticky="ew"); self.entry_count_weight.insert(0, "0.25")
        ctk.CTkLabel(self.frame_three_dimensional_params, text="体积利用率权重").grid(row=3, column=0, sticky="w"); self.entry_volume_weight_3d = ctk.CTkEntry(self.frame_three_dimensional_params); self.entry_volume_weight_3d.grid(row=3, column=1, padx=(10,0), pady=5, sticky="ew"); self.entry_volume_weight_3d.insert(0, "0.25")
        ctk.CTkLabel(self.frame_three_dimensional_params, text="高度空间利用率权重").grid(row=4, column=0, sticky="w"); self.entry_height_weight = ctk.CTkEntry(self.frame_three_dimensional_params); self.entry_height_weight.grid(row=4, column=1, padx=(10,0), sticky="ew"); self.entry_height_weight.insert(0, "0.5")
        ctk.CTkLabel(self.frame_three_dimensional_params, text="最低数量利用率阈值 (%)").grid(row=5, column=0, sticky="w"); self.entry_min_count_utilization = ctk.CTkEntry(self.frame_three_dimensional_params); self.entry_min_count_utilization.grid(row=5, column=1, padx=(10,0), pady=5, sticky="ew"); self.entry_min_count_utilization.insert(0, "80")
        
        ctk.CTkLabel(self.frame_boundary_params, text="高度步长 (mm)").grid(row=0, column=0, sticky="w"); self.entry_height_step = ctk.CTkEntry(self.frame_boundary_params); self.entry_height_step.grid(row=0, column=1, padx=(10,0), sticky="ew"); self.entry_height_step.insert(0, "10")
        ctk.CTkLabel(self.frame_boundary_params, text="最小高度差值 (mm)").grid(row=1, column=0, sticky="w"); self.entry_min_height_diff = ctk.CTkEntry(self.frame_boundary_params); self.entry_min_height_diff.grid(row=1, column=1, padx=(10,0), pady=5, sticky="ew"); self.entry_min_height_diff.insert(0, "150")
        ctk.CTkLabel(self.frame_boundary_params, text="体积权重系数 α").grid(row=2, column=0, sticky="w"); self.entry_volume_weight = ctk.CTkEntry(self.frame_boundary_params); self.entry_volume_weight.grid(row=2, column=1, padx=(10,0), sticky="ew"); self.entry_volume_weight.insert(0, "0.5")
        
        ctk.CTkLabel(self.frame_manual_params, text="H计算-第1百分位 (%)").grid(row=0, column=0, sticky="w"); self.entry_p1 = ctk.CTkEntry(self.frame_manual_params); self.entry_p1.grid(row=0, column=1, padx=(10,0), sticky="ew"); self.entry_p1.insert(0, "50")
        ctk.CTkLabel(self.frame_manual_params, text="H计算-第2百分位 (%)").grid(row=1, column=0, pady=5, sticky="w"); self.entry_p2 = ctk.CTkEntry(self.frame_manual_params); self.entry_p2.grid(row=1, column=1, padx=(10,0), pady=5, sticky="ew"); self.entry_p2.insert(0, "90")
        self.h_params_row = r;
        r+=1

        ctk.CTkLabel(self.frame_left, text="仓库环境参数", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(20, 10)); r+=1
        ctk.CTkLabel(self.frame_left, text="仓库总高度 (mm)").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_warehouse_h = ctk.CTkEntry(self.frame_left); self.entry_warehouse_h.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); self.entry_warehouse_h.insert(0, "6000"); r+=1
        ctk.CTkLabel(self.frame_left, text="底层离地高度 (mm)").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_bottom_clearance = ctk.CTkEntry(self.frame_left); self.entry_bottom_clearance.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); self.entry_bottom_clearance.insert(0, "150"); r+=1
        ctk.CTkLabel(self.frame_left, text="每层预留空间 (mm)").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_layer_headroom = ctk.CTkEntry(self.frame_left); self.entry_layer_headroom.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); self.entry_layer_headroom.insert(0, "100"); r+=1
        ctk.CTkLabel(self.frame_left, text="货架层板厚度 (mm)").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_shelf_thickness = ctk.CTkEntry(self.frame_left); self.entry_shelf_thickness.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); self.entry_shelf_thickness.insert(0, "50"); r+=1
        
        # 货架LDW列名配置
        ctk.CTkLabel(self.frame_left, text="货架LDW列名配置", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(20, 10)); r+=1
        self.shelf_col_entries = {}
        shelf_defaults = {'lp': 'Lp', 'dp': 'Dp', 'wp': 'Wp'}
        shelf_labels = {'lp': '货架长度列名 (Lp)', 'dp': '货架深度列名 (Dp)', 'wp': '货架承重列名 (Wp)'}
        for key, text in shelf_labels.items():
            ctk.CTkLabel(self.frame_left, text=text).grid(row=r, column=0, padx=20, pady=(5,0), sticky="w")
            entry = ctk.CTkEntry(self.frame_left); entry.grid(row=r, column=1, padx=20, pady=(5,0), sticky="ew"); entry.insert(0, shelf_defaults[key]); self.shelf_col_entries[key] = entry; r += 1
        
        # 装托LDHWV列名配置
        ctk.CTkLabel(self.frame_left, text="装托LDHWV列名配置", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(20, 10)); r+=1
        self.col_entries = {}
        pallet_defaults = {'sku_id': 'ArtNo', 'pallet_l': '装托长度(mm)', 'pallet_d': '装托深度(mm)', 'pallet_h': '装托高度(mm)', 'pallet_w': '装托重量(kg)', 'pallet_v': '装托体积(cbm)'}
        pallet_labels = {'sku_id': 'SKU编号', 'pallet_l': '装托长度 (L)', 'pallet_d': '装托深度 (D)', 'pallet_h': '装托高度 (H)', 'pallet_w': '装托重量 (W)', 'pallet_v': '装托体积 (V)'}
        for key, text in pallet_labels.items():
            ctk.CTkLabel(self.frame_left, text=text).grid(row=r, column=0, padx=20, pady=(5,0), sticky="w")
            entry = ctk.CTkEntry(self.frame_left); entry.grid(row=r, column=1, padx=20, pady=(5,0), sticky="ew"); entry.insert(0, pallet_defaults[key]); self.col_entries[key] = entry; r += 1
        
        # 装箱LDHWV列名配置
        ctk.CTkLabel(self.frame_left, text="装箱LDHWV列名配置", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(20, 10)); r+=1
        box_defaults = {'box_l': '装箱长度(mm)', 'box_d': '装箱深度(mm)', 'box_h': '装箱高度(mm)', 'box_w': '装箱重量(kg)', 'box_v': '装箱体积(cbm)'}
        box_labels = {'box_l': '装箱长度 (L)', 'box_d': '装箱深度 (D)', 'box_h': '装箱高度 (H)', 'box_w': '装箱重量 (W)', 'box_v': '装箱体积 (V)'}
        for key, text in box_labels.items():
            ctk.CTkLabel(self.frame_left, text=text).grid(row=r, column=0, padx=20, pady=(5,0), sticky="w")
            entry = ctk.CTkEntry(self.frame_left); entry.grid(row=r, column=1, padx=20, pady=(5,0), sticky="ew"); entry.insert(0, box_defaults[key]); self.col_entries[key] = entry; r += 1
        
        # 预设值配置
        ctk.CTkLabel(self.frame_left, text="装托预设值配置", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(20, 10)); r+=1
        self.pallet_preset_entries = {}
        pallet_preset_defaults = {'l': '1200', 'd': '800', 'h': '1800', 'w': '1000', 'v': '1.0'}
        pallet_preset_labels = {'l': '装托默认长度 (mm)', 'd': '装托默认深度 (mm)', 'h': '装托默认高度 (mm)', 'w': '装托默认重量 (kg)', 'v': '装托默认体积 (cbm)'}
        for key, text in pallet_preset_labels.items():
            ctk.CTkLabel(self.frame_left, text=text).grid(row=r, column=0, padx=20, pady=(5,0), sticky="w")
            entry = ctk.CTkEntry(self.frame_left); entry.grid(row=r, column=1, padx=20, pady=(5,0), sticky="ew"); entry.insert(0, pallet_preset_defaults[key]); self.pallet_preset_entries[key] = entry; r += 1
        
        ctk.CTkLabel(self.frame_left, text="装箱预设值配置", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(20, 10)); r+=1
        self.box_preset_entries = {}
        box_preset_defaults = {'l': '600', 'd': '400', 'h': '300', 'w': '20', 'v': '0.1'}
        box_preset_labels = {'l': '装箱默认长度 (mm)', 'd': '装箱默认深度 (mm)', 'h': '装箱默认高度 (mm)', 'w': '装箱默认重量 (kg)', 'v': '装箱默认体积 (cbm)'}
        for key, text in box_preset_labels.items():
            ctk.CTkLabel(self.frame_left, text=text).grid(row=r, column=0, padx=20, pady=(5,0), sticky="w")
            entry = ctk.CTkEntry(self.frame_left); entry.grid(row=r, column=1, padx=20, pady=(5,0), sticky="ew"); entry.insert(0, box_preset_defaults[key]); self.box_preset_entries[key] = entry; r += 1
        
        self.tabview = ctk.CTkTabview(self, width=250); self.tabview.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
        self.tabview.add("运行日志"); self.tabview.add("分析图表"); self.tabview.set("运行日志")
        self.output_textbox = ctk.CTkTextbox(self.tabview.tab("运行日志")); self.output_textbox.pack(expand=True, fill="both")
        self.charts_frame = ctk.CTkFrame(self.tabview.tab("分析图表"), fg_color="transparent"); self.charts_frame.pack(expand=True, fill="both")
        self.charts_label = ctk.CTkLabel(self.charts_frame, text="计算完成后将在此处显示图表", font=ctk.CTkFont(size=20)); self.charts_label.pack(expand=True)
        self.canvas = None
        # 创建按钮框架
        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.grid(row=1, column=1, padx=(10,20), pady=(0,10), sticky="ew")
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        
        self.button_run = ctk.CTkButton(self.button_frame, text="开始计算", height=40, font=ctk.CTkFont(size=18, weight="bold"), command=self.start_calculation)
        self.button_run.grid(row=0, column=0, padx=(0,5), sticky="ew")
        
        self.button_save_params = ctk.CTkButton(self.button_frame, text="保存参数", height=40, font=ctk.CTkFont(size=16, weight="bold"), command=self.save_parameters)
        self.button_save_params.grid(row=0, column=1, padx=(5,0), sticky="ew")
        
        # 添加加载参数按钮（可以通过右键菜单或快捷键访问）
        self.button_frame.bind("<Button-3>", self.show_context_menu)  # 右键菜单
        self.status_label = ctk.CTkLabel(self, text="状态: 空闲", width=400); self.status_label.grid(row=2, column=1, padx=(10,20), pady=5, sticky="w")
        self.progressbar = ctk.CTkProgressBar(self, width=300); self.progressbar.grid(row=2, column=1, padx=(10,20), pady=5, sticky="e"); self.progressbar.set(0)
        
        self.display_welcome_message(); self.toggle_ld_params(); self.toggle_h_params()
        
        # 尝试自动加载上次会话的参数
        self.auto_load_last_session()
        
        self.after(100, self.process_queue)

    def auto_save_session(self):
        """自动保存当前会话参数"""
        try:
            params = self.collect_params()
            # 移除不需要保存的运行时参数
            params_to_save = params.copy()
            if 'excel_file' in params_to_save:
                del params_to_save['excel_file']
            
            # 保存到会话文件
            with open(LAST_SESSION_FILE, 'w', encoding='utf-8') as f:
                json.dump(params_to_save, f, ensure_ascii=False, indent=2)
        except Exception:
            # 静默处理错误，不影响用户体验
            pass
    
    def auto_load_last_session(self):
        """自动加载上次会话的参数"""
        try:
            if os.path.exists(LAST_SESSION_FILE):
                with open(LAST_SESSION_FILE, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                self.set_gui_values_from_params(params)
                self.update_textbox("\n✅ 已自动恢复上次会话的参数设置。\n", False)
            elif os.path.exists(DEFAULT_CONFIG_FILE):
                with open(DEFAULT_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                self.set_gui_values_from_params(params)
                self.update_textbox("\n✅ 已加载默认参数配置。\n", False)
        except Exception:
            # 如果加载失败，使用程序内置默认值
            pass

    def save_parameters(self):
        """保存当前参数配置到JSON文件"""
        try:
            params = self.collect_params()
            # 移除不需要保存的运行时参数
            params_to_save = params.copy()
            if 'excel_file' in params_to_save:
                del params_to_save['excel_file']
            
            # 选择保存位置
            file_path = filedialog.asksaveasfilename(
                title="保存参数配置",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(params_to_save, f, ensure_ascii=False, indent=2)
                
                # 如果保存的文件名是默认配置文件，则也同时保存会话
                if os.path.basename(file_path) == DEFAULT_CONFIG_FILE:
                    self.auto_save_session()
                
                messagebox.showinfo("保存成功", f"参数配置已保存到：\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("保存失败", f"保存参数时发生错误：\n{str(e)}")
    
    def load_parameters(self):
        """从JSON文件加载参数配置"""
        try:
            file_path = filedialog.askopenfilename(
                title="加载参数配置",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                
                # 设置GUI控件的值
                self.set_gui_values_from_params(params)
                messagebox.showinfo("加载成功", f"参数配置已从以下文件加载：\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("加载失败", f"加载参数时发生错误：\n{str(e)}")
    
    def load_default_config(self):
        """加载默认配置"""
        try:
            if os.path.exists(DEFAULT_CONFIG_FILE):
                with open(DEFAULT_CONFIG_FILE, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                self.set_gui_values_from_params(params)
                messagebox.showinfo("恢复成功", "已恢复为默认参数配置")
            else:
                messagebox.showwarning("无默认配置", "未找到默认配置文件，请先保存一个默认配置")
        except Exception as e:
            messagebox.showerror("恢复失败", f"恢复默认参数时发生错误：\n{str(e)}")
    
    def save_as_default_config(self):
        """保存当前参数为默认配置"""
        try:
            params = self.collect_params()
            params_to_save = params.copy()
            if 'excel_file' in params_to_save:
                del params_to_save['excel_file']
            
            with open(DEFAULT_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(params_to_save, f, ensure_ascii=False, indent=2)
            
            # 同时保存为会话
            self.auto_save_session()
            
            messagebox.showinfo("保存成功", f"当前参数已保存为默认配置\n文件位置：{DEFAULT_CONFIG_FILE}")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存默认配置时发生错误：\n{str(e)}")
    
    def clear_session(self):
        """清除会话记录"""
        try:
            if os.path.exists(LAST_SESSION_FILE):
                os.remove(LAST_SESSION_FILE)
            messagebox.showinfo("清除成功", "会话记录已清除，下次启动将使用程序默认值")
        except Exception as e:
            messagebox.showerror("清除失败", f"清除会话记录时发生错误：\n{str(e)}")
    
    def set_gui_values_from_params(self, params):
        """根据参数字典设置GUI控件的值"""
        # Sheet名称
        if 'sku_sheet' in params:
            self.entry_sku_sheet.delete(0, tk.END)
            self.entry_sku_sheet.insert(0, params['sku_sheet'])
        if 'shelf_sheet' in params:
            self.entry_shelf_sheet.delete(0, tk.END)
            self.entry_shelf_sheet.insert(0, params['shelf_sheet'])
        
        # 装箱装托配置
        mapping = {
            'can_box_col': self.entry_can_box_col,
            'can_pallet_col': self.entry_can_pallet_col,
            'box_count_col': self.entry_box_count_col,
            'pallet_count_col': self.entry_pallet_count_col,
            'boxes_per_pallet_col': self.entry_boxes_per_pallet_col,
            'shelf_type_col': self.entry_shelf_type_col,
            'pallet_char': self.entry_pallet_char,
            'box_char': self.entry_box_char
        }
        
        for key, entry in mapping.items():
            if key in params:
                entry.delete(0, tk.END)
                entry.insert(0, params[key])
        
        # 列名配置
        if 'cols' in params:
            for key, value in params['cols'].items():
                if key in self.col_entries:
                    self.col_entries[key].delete(0, tk.END)
                    self.col_entries[key].insert(0, value)
        
        # 货架列名配置
        if 'shelf_cols' in params:
            for key, value in params['shelf_cols'].items():
                if key in self.shelf_col_entries:
                    self.shelf_col_entries[key].delete(0, tk.END)
                    self.shelf_col_entries[key].insert(0, value)
        
        # 预设值配置
        if 'pallet_preset' in params:
            for key, value in params['pallet_preset'].items():
                if key in self.pallet_preset_entries:
                    self.pallet_preset_entries[key].delete(0, tk.END)
                    self.pallet_preset_entries[key].insert(0, str(value))
        
        if 'box_preset' in params:
            for key, value in params['box_preset'].items():
                if key in self.box_preset_entries:
                    self.box_preset_entries[key].delete(0, tk.END)
                    self.box_preset_entries[key].insert(0, str(value))
        
        # 数值参数
        numeric_mapping = {
            'coverage_target': (self.entry_coverage, lambda x: x * 100),  # 转换为百分比
            'h_max': (self.entry_hmax, lambda x: x),
            'warehouse_h': (self.entry_warehouse_h, lambda x: x),
            'bottom_clearance': (self.entry_bottom_clearance, lambda x: x),
            'layer_headroom': (self.entry_layer_headroom, lambda x: x),
            'shelf_thickness': (self.entry_shelf_thickness, lambda x: x),
            'area_threshold': (self.entry_area_threshold, lambda x: x),
            'size_threshold': (self.entry_size_threshold, lambda x: x),
            'pallet_decimal_threshold': (self.entry_pallet_decimal_threshold, lambda x: x)
        }
        
        for key, (entry, transform) in numeric_mapping.items():
            if key in params:
                entry.delete(0, tk.END)
                entry.insert(0, str(transform(params[key])))
        
        # 布尔参数
        if 'allow_rotation' in params:
            if params['allow_rotation']:
                self.check_allow_rotation.select()
            else:
                self.check_allow_rotation.deselect()
        
        # 选择参数
        if 'ld_method' in params:
            self.ld_method_var.set(params['ld_method'])
        if 'h_method' in params:
            self.h_method_var.set(params['h_method'])
        
        # 方法特定参数
        h_method_mapping = {
            'p1': self.entry_p1,
            'p2': self.entry_p2,
            'height_step': self.entry_height_step,
            'min_height_diff': self.entry_min_height_diff,
            'volume_weight': self.entry_volume_weight
        }
        
        for key, entry in h_method_mapping.items():
            if key in params:
                entry.delete(0, tk.END)
                entry.insert(0, str(params[key]))
        
        # 三维算法参数
        three_d_mapping = {
            'height_step': self.entry_height_step_3d,
            'min_height_diff': self.entry_min_height_diff_3d,
            'count_weight': self.entry_count_weight,
            'volume_weight': self.entry_volume_weight_3d,
            'height_weight': self.entry_height_weight,
            'min_count_utilization': (self.entry_min_count_utilization, lambda x: x * 100)
        }
        
        for key, entry_info in three_d_mapping.items():
            if key in params:
                if isinstance(entry_info, tuple):
                    entry, transform = entry_info
                    entry.delete(0, tk.END)
                    entry.insert(0, str(transform(params[key])))
                else:
                    entry_info.delete(0, tk.END)
                    entry_info.insert(0, str(params[key]))
        
        # 刷新界面
        self.toggle_ld_params()
        self.toggle_h_params()
    
    def show_context_menu(self, event):
        """显示右键菜单"""
        try:
            context_menu = tk.Menu(self, tearoff=0)
            context_menu.add_command(label="📁 加载参数配置", command=self.load_parameters)
            context_menu.add_command(label="💾 保存参数配置", command=self.save_parameters)
            context_menu.add_separator()
            context_menu.add_command(label="🔄 恢复默认参数", command=self.load_default_config)
            context_menu.add_command(label="📋 保存为默认配置", command=self.save_as_default_config)
            context_menu.add_separator()
            context_menu.add_command(label="🗑️ 清除会话记录", command=self.clear_session)
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()

    def on_closing(self):
        """程序关闭时的处理"""
        try:
            # 自动保存当前会话参数
            self.auto_save_session()
        except Exception:
            pass  # 静默处理，不影响程序关闭
        finally:
            self.destroy()

    def display_welcome_message(self):
        self.update_textbox("""欢迎使用货架配置优化工具 ProVersion-4.0.0！

v4.0.0 重大更新:
- 装箱装托智能判定: 全新的算法可根据SKU特性和空托率自动判定装箱、装托或混装方案
- Excel统一管理: 支持从单一Excel文件的不同Sheet读取SKU和货架数据，简化文件管理
- 货架类型筛选: 根据装箱装托判定结果自动筛选合适的货架类型（A类装托、B类装箱）
- 智能决策流程: 装箱装托判定→货架筛选→L&D&H优化的全新计算流程
- 分离LDHWV数据: 支持装托和装箱两套独立的LDHWV参数，提供预设值处理缺失数据
- 智能参数管理: 自动保存和恢复参数，告别重复配置的烦恼

--- 参数管理功能 ---
🔄 自动会话恢复: 程序启动时自动加载上次使用的参数
💾 智能保存: 开始计算时自动保存当前参数设置
📁 配置管理: 右键按钮区域可访问完整的参数管理菜单
- 加载/保存参数配置文件
- 设置/恢复默认配置
- 清除会话记录

--- 参数说明 ---
[装箱装托配置]
- 可装箱/可装托列名: 从SKU数据中读取Y/N标识的列名
- 货架类型标识列名: 从货架数据中读取类型标识的列名
- 装托/装箱标识字符: 用于区分装托货架（如A）和装箱货架（如B）的标识
- 装托小数阈值: 当装托数小数部分超过此值时，采用纯装托策略（默认0.3）

[LDHWV数据配置]
- 装托/装箱LDHWV列名: 分别配置装托和装箱模式下的长深高重体参数列名
- 预设值: 当SKU缺少某些LDHWV数据时使用的默认值

[核心参数]
- 覆盖率目标: 算法在第一阶段寻找最优货架规格时所依据的核心业务指标。
- 最大允许高度: 过滤掉自身高度超过此值的SKU。

💡 小贴士: 右键点击按钮区域可访问参数管理菜单！
""", True)

    def select_excel_file(self):
        file_path = filedialog.askopenfilename(title="选择Excel文件", filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")])
        if file_path: self.update_path('excel', file_path)

    def paste_excel_path(self):
        try:
            path = self.clipboard_get().strip()
            if path and os.path.exists(path): self.update_path('excel', path)
            else: messagebox.showwarning("粘贴失败", "剪贴板中的路径无效或文件不存在")
        except tk.TclError: messagebox.showwarning("粘贴失败", "剪贴板为空或包含无效内容")
        
    def toggle_ld_params(self):
        if self.ld_method_var.get() == "complementary": self.frame_complementary_params.grid()
        else: self.frame_complementary_params.grid_remove()

    def collect_params(self):
        params = {'excel_file': self.selected_excel_file}
        
        # 收集Sheet名称和装箱装托配置
        params.update({
            'sku_sheet': self.entry_sku_sheet.get().strip(),
            'shelf_sheet': self.entry_shelf_sheet.get().strip(),
            'can_box_col': self.entry_can_box_col.get().strip(),
            'can_pallet_col': self.entry_can_pallet_col.get().strip(),
            'box_count_col': self.entry_box_count_col.get().strip(),
            'pallet_count_col': self.entry_pallet_count_col.get().strip(),
            'boxes_per_pallet_col': self.entry_boxes_per_pallet_col.get().strip(),
            'shelf_type_col': self.entry_shelf_type_col.get().strip(),
            'pallet_char': self.entry_pallet_char.get().strip(),
            'box_char': self.entry_box_char.get().strip()
        })
        
        # 验证必填项
        if not params['sku_sheet'] or not params['shelf_sheet']:
            raise ValueError("SKU Sheet名称和货架Sheet名称不能为空。")
        if not params['can_box_col'] or not params['can_pallet_col'] or not params['box_count_col'] or not params['pallet_count_col'] or not params['boxes_per_pallet_col']:
            raise ValueError("装箱装托相关的列名都不能为空。")
        if not params['shelf_type_col']:
            raise ValueError("货架类型标识列名不能为空。")
        if not params['pallet_char'] or not params['box_char']:
            raise ValueError("装托标识字符和装箱标识字符不能为空。")
        
        # 收集预设值
        try:
            params['pallet_preset'] = {k: float(e.get()) for k, e in self.pallet_preset_entries.items()}
            params['box_preset'] = {k: float(e.get()) for k, e in self.box_preset_entries.items()}
        except ValueError: 
            raise ValueError("预设值中包含无效的非数字输入。")
        
        try:
            params.update({k: float(e.get()) for k, e in {
                'coverage_target': self.entry_coverage, 'h_max': self.entry_hmax,
                'warehouse_h': self.entry_warehouse_h, 'bottom_clearance': self.entry_bottom_clearance,
                'layer_headroom': self.entry_layer_headroom, 'shelf_thickness': self.entry_shelf_thickness,
                'area_threshold': self.entry_area_threshold, 'size_threshold': self.entry_size_threshold,
                'pallet_decimal_threshold': self.entry_pallet_decimal_threshold
            }.items()})
        except ValueError: raise ValueError("通用参数或仓库环境参数中包含无效的非数字输入。")
        params['coverage_target'] /= 100.0
        
        params.update({'allow_rotation': self.check_allow_rotation.get() == 1, 'ld_method': self.ld_method_var.get(), 'h_method': self.h_method_var.get(), 'cols': {key: entry.get() for key, entry in self.col_entries.items()}})
        if not all(params['cols'].values()): raise ValueError("所有Excel列名都不能为空。")
        
        # 收集货架列名配置
        params['shelf_cols'] = {key: entry.get().strip() for key, entry in self.shelf_col_entries.items()}
        if not all(params['shelf_cols'].values()): raise ValueError("货架LDW列名都不能为空。")
        try:
            if params['h_method'] == 'manual': params.update({k: float(e.get()) for k, e in {'p1': self.entry_p1, 'p2': self.entry_p2}.items()})
            elif params['h_method'] == 'three_dimensional':
                params.update({k: float(e.get()) for k, e in {'height_step': self.entry_height_step_3d, 'min_height_diff': self.entry_min_height_diff_3d, 'count_weight': self.entry_count_weight, 'volume_weight': self.entry_volume_weight_3d, 'height_weight': self.entry_height_weight, 'min_count_utilization': self.entry_min_count_utilization}.items()})
                params['min_count_utilization'] /= 100.0
            else: params.update({k: float(e.get()) for k, e in {'height_step': self.entry_height_step, 'min_height_diff': self.entry_min_height_diff, 'volume_weight': self.entry_volume_weight}.items()})
        except ValueError: raise ValueError("H高度计算方法的参数中包含无效的非数字输入。")
        return params

    def process_queue(self):
        try:
            msg_type, msg_content = self.queue.get_nowait()
            if msg_type == "log": self.update_textbox(msg_content)
            elif msg_type == "progress": self.update_progress(*msg_content)
            elif msg_type == "progress_update": self.status_label.configure(text=f"状态: {msg_content[0]}"); self.progressbar.set(msg_content[1])
            elif msg_type == "pre_calculation_done":
                self.progressbar.set(1.0); raw_data, raw_data_with_ldh, shelves, corr_label, corr_val, grade, packing_summary = msg_content
                self.cache.update({
                    'excel_path': self.current_params['excel_file'], 
                    'sku_sheet': self.current_params['sku_sheet'],
                    'shelf_sheet': self.current_params['shelf_sheet'],
                    'raw_data': raw_data,
                    'raw_data_with_ldh': raw_data_with_ldh,
                    'shelves': shelves, 
                    'corr_result': (corr_label, corr_val, grade),
                    'packing_summary': packing_summary
                })
                self.proceed_with_correlation_check()
            elif msg_type == "agg_data_computed": self.cache.update({'agg_data': msg_content[0], 'operable_data': msg_content[1], 'h_max': self.current_params['h_max']})
            elif msg_type == "result": self.display_results(*msg_content)
            elif msg_type == "diagnostics": self.display_diagnostics(*msg_content)
            elif msg_type == "visualization_data": self.cache['viz_data'] = msg_content; self.update_charts(); self.tabview.set("分析图表")
            elif msg_type == "error": self.update_textbox(f"\n!!!!!! 计算出错 !!!!!!\n\n{msg_content}\n"); self.button_run.configure(state="normal"); self.status_label.configure(text="状态: 计算失败"); self.progressbar.set(0)
            elif msg_type == "done": self.status_label.configure(text=f"状态: {msg_content}"); self.button_run.configure(state="normal"); self.progressbar.set(1)
        except queue.Empty: pass
        self.after(100, self.process_queue)
    
    def start_calculation(self):
        if self.canvas: self.canvas.get_tk_widget().destroy(); self.canvas = None
        self.charts_label.pack(expand=True); self.tabview.set("运行日志")
        try:
            if not hasattr(self, 'selected_excel_file'): raise ValueError("请先选择Excel文件。")
            self.current_params = self.collect_params()
            
            # 自动保存当前会话参数
            self.auto_save_session()
            
            self.update_textbox("", True); self.button_run.configure(state="disabled"); self.progressbar.set(0)
            if self.cache.get('excel_path') == self.current_params['excel_file'] and self.cache.get('sku_sheet') == self.current_params['sku_sheet'] and self.cache.get('shelf_sheet') == self.current_params['shelf_sheet']:
                self.update_textbox("文件缓存命中，跳过文件读取和检验步骤。\n"); self.proceed_with_correlation_check()
            else: self.status_label.configure(text="状态: 正在初始化..."); threading.Thread(target=pre_calculation_worker, args=(self.queue, self.current_params)).start()
        except Exception as e: messagebox.showerror("输入或文件错误", f"发生错误: {e}"); self.button_run.configure(state="normal"); self.status_label.configure(text="状态: 空闲")
    
    def proceed_with_correlation_check(self):
        corr_label, corr_val, grade = self.cache['corr_result']
        raw_data, shelves = self.cache['raw_data'], self.cache['shelves']
        raw_data_with_ldh = self.cache['raw_data_with_ldh']
        self.update_textbox("--- 步骤 0 完成：文件预处理与检验完毕 ---\n"); self.update_textbox(f"最强相关性: '{corr_label}', r = {corr_val:.3f}, 评级: {grade}\n")
        self.update_textbox("\n--- 数据概览 ---\n"); self.update_textbox(f"有效SKU总数: {len(raw_data)}\n"); self.update_textbox(f"候选货架规格数量: {len(shelves)}\n")
        self.update_textbox(f"SKU尺寸(长*深*高)范围: {raw_data_with_ldh['L'].min():.0f}-{raw_data_with_ldh['L'].max():.0f} * {raw_data_with_ldh['D'].min():.0f}-{raw_data_with_ldh['D'].max():.0f} * {raw_data_with_ldh['H'].min():.0f}-{raw_data_with_ldh['H'].max():.0f} mm\n")
        self.update_textbox(f"货架尺寸(长*深)范围: {min(s['Lp'] for s in shelves):.0f}-{max(s['Lp'] for s in shelves):.0f} * {min(s['Dp'] for s in shelves):.0f}-{max(s['Dp'] for s in shelves):.0f} mm\n")
        if messagebox.askyesno("继续计算?", f"数据概览已显示，L/D与H的相关性为 {grade}。\n是否继续运行核心优化算法？"): self.start_core_calculation()
        else: self.update_textbox("用户选择取消操作。\n"); self.status_label.configure(text="状态: 已取消"); self.button_run.configure(state="normal")
    
    # --- vReX 1.0.1 全新可视化仪表盘 ---
    def update_charts(self):
        try:
            self.charts_label.pack_forget()
            if self.canvas: self.canvas.get_tk_widget().destroy()
            
            fig = Figure(figsize=(12, 10), dpi=100)
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1,1], hspace=0.4, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, 0]) # Panel 1: Overview
            ax2 = fig.add_subplot(gs[0, 1]) # Panel 2: Decision Basis
            ax3 = fig.add_subplot(gs[1, 1]) # Panel 3: Diagnostics
            ax4 = fig.add_subplot(gs[1, 0]) # Panel 4: Shelf Profile

            operable_data, _, final_solution = self.cache['viz_data']
            
            self.plot_solution_overview(ax1, final_solution)
            self.plot_decision_basis(ax2, operable_data, final_solution)
            self.plot_sku_diagnostics(ax3, operable_data, final_solution)
            self.plot_shelf_profile(ax4, final_solution)

            fig.tight_layout(pad=3.0)
            self.canvas = FigureCanvasTkAgg(fig, master=self.charts_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        except Exception as e:
            self.charts_label.pack(expand=True)
            self.charts_label.configure(text=f"图表绘制失败: {e}")
            self.update_textbox(f"\n!!! 图表绘制失败: {e}!!!\n")

    def plot_solution_overview(self, ax, final_solution):
        ax.set_title("图表一：方案总览与核心指标", fontweight="bold")
        
        specs = final_solution['final_shelves']
        counts = final_solution['counts']
        
        labels = [f"规格 {i+1}\n{s['Lp']:.0f}L×{s['Dp']:.0f}D×{s['H']:.0f}H" for i, s in enumerate(specs)]
        y_pos = np.arange(len(labels))
        
        bars = ax.barh(y_pos, counts, align='center', color=plt.cm.viridis(np.linspace(0.4, 0.9, len(labels))))
        ax.set_yticks(y_pos, labels=labels)
        ax.invert_yaxis()
        ax.set_xlabel("货架需求数量 (个)")
        ax.bar_label(bars, padding=3)
        ax.grid(axis='x', linestyle='--', alpha=0.6)

        total_shelves = sum(counts)
        cov_count = final_solution['coverage_count'] * 100
        cov_vol = final_solution['coverage_volume'] * 100
        
        summary_text = (f"货架总数: {total_shelves} 个\n"
                        f"SKU数量覆盖率: {cov_count:.2f}%\n"
                        f"SKU体积覆盖率: {cov_vol:.2f}%")
        
        ax.text(0.95, 0.05, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    def plot_decision_basis(self, ax, operable_data, final_solution):
        ax.set_title("图表二：决策依据 (SKU尺寸分布)", fontweight="bold")
        
        # 散点图
        sc = ax.scatter(operable_data['L'], operable_data['D'], c=operable_data['H'], 
                        cmap='viridis', alpha=0.6, s=15)
        
        # 叠加货架矩形
        unique_lds = { (s['Lp'], s['Dp']) for s in final_solution['final_shelves'] }
        colors = plt.cm.autumn(np.linspace(0, 1, len(unique_lds)))
        for i, (lp, dp) in enumerate(unique_lds):
            rect = patches.Rectangle((0, 0), lp, dp, linewidth=2, edgecolor=colors[i], 
                                     facecolor='none', label=f"推荐规格 {lp:.0f}×{dp:.0f}")
            ax.add_patch(rect)
        
        ax.set_xlabel("SKU 长度 (mm)")
        ax.set_ylabel("SKU 深度 (mm)")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 添加颜色条
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('SKU 高度 (mm)')

    def plot_sku_diagnostics(self, ax, operable_data, final_solution):
        ax.set_title("图表三：SKU安置情况诊断", fontweight="bold")

        total_skus = len(operable_data)
        placed_ids = final_solution['placed_sku_ids']
        unplaced_df = operable_data[~operable_data['sku_id'].isin(placed_ids)]
        
        placed_count = len(placed_ids)
        unplaced_count = len(unplaced_df)

        labels = ['成功安放']
        sizes = [placed_count]
        
        if unplaced_count > 0:
            reasons = get_detailed_unplaced_reasons_by_sku_id(unplaced_df, final_solution['final_shelves'], 
                                                             self.current_params['h_max'], self.current_params['allow_rotation'])
            reason_summary = Counter(reason.split(":")[0].strip() for reason in reasons.values())
            
            # 排序，让图例更整洁
            sorted_reasons = sorted(reason_summary.items(), key=lambda item: item[1], reverse=True)
            labels.extend([r[0] for r in sorted_reasons])
            sizes.extend([r[1] for r in sorted_reasons])

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
        
        # 移除饼图上的标签和百分比，以解决重叠问题
        wedges, _ = ax.pie(sizes, startangle=90, colors=colors, radius=1.2)
        ax.axis('equal')

        # 创建一个清晰的图例来展示详细信息
        legend_labels = [f'{label} - {size}个 ({size/total_skus:.1%})' for label, size in zip(labels, sizes)]
        ax.legend(wedges, legend_labels,
                  title="SKU 分类",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize='small')
        
    def plot_shelf_profile(self, ax, final_solution):
        ax.set_title("图表四：单货架装箱效果剖面图", fontweight="bold")

        if not final_solution['counts'] or sum(final_solution['counts']) == 0:
            ax.text(0.5, 0.5, "无货架分配", ha='center', va='center'); return

        # 1. 选择代表性货架 (数量最多的)
        counts = final_solution['counts']
        spec_idx = np.argmax(counts)
        spec = final_solution['final_shelves'][spec_idx]
        
        # 2. 计算层数
        params = self.current_params
        usable_vertical_space = params['warehouse_h'] - params['bottom_clearance']
        layer_total_h = spec['H'] + params['shelf_thickness'] + params['layer_headroom']
        num_layers = math.floor(usable_vertical_space / layer_total_h) if layer_total_h > 0 else 0
        if num_layers == 0:
            ax.text(0.5, 0.5, "仓库高度不足以放置一层", ha='center', va='center'); return

        # 3. 绘制货架结构
        ax.set_xlim(-0.1 * spec['Lp'], 1.1 * spec['Lp'])
        ax.set_ylim(0, params['warehouse_h'])
        ax.axhline(0, color='gray', linewidth=4)
        ax.add_patch(patches.Rectangle((0, 0), -50, usable_vertical_space, color='#ababab'))
        ax.add_patch(patches.Rectangle((spec['Lp'], 0), 50, usable_vertical_space, color='#ababab'))
        
        # 4. 模拟真实装箱密度并填满单层货架
        items_assigned = final_solution['assignments'].get(spec_idx, [])

        packed_items = []
        if items_assigned:
            shelf_len = spec['Lp']
            available_len = shelf_len - (2 * SIDE_SPACING)
            
            # 使用与真实装箱算法相同的FFD排序逻辑
            sorted_items = sorted(items_assigned, key=lambda x: x[1], reverse=True)  # 按宽度降序
            
            # 模拟真实的装箱过程，尽量填满一层
            current_pos = SIDE_SPACING
            remaining_length = available_len
            
            for item_dict, width in sorted_items:
                is_first_item = (len(packed_items) == 0)
                required_space = width if is_first_item else width + INTER_SPACING
                
                if remaining_length >= required_space:
                    if not is_first_item:
                        current_pos += INTER_SPACING
                        remaining_length -= INTER_SPACING
                    
                    packed_items.append({'pos': current_pos, 'width': width, 'height': item_dict['H']})
                    current_pos += width
                    remaining_length -= width
                
                # 如果剩余空间太小，停止装箱
                if remaining_length < 50:  # 预留最小空间，避免过度拥挤
                    break
            
            # 如果装箱物品太少，重复使用部分物品来模拟真实密度
            if len(packed_items) < 3 and items_assigned:
                # 计算理论容量
                avg_width = np.mean([item[1] for item in items_assigned])
                if avg_width > 0:
                    theoretical_items = int(available_len / (avg_width + INTER_SPACING))
                    
                    # 如果理论容量远大于当前装箱数，补充物品
                    if theoretical_items > len(packed_items) * 1.5:
                        additional_needed = min(theoretical_items - len(packed_items), len(items_assigned))
                        
                        # 重置装箱，使用循环采样来填满货架
                        packed_items = []
                        current_pos = SIDE_SPACING
                        remaining_length = available_len
                        item_cycle = 0
                        
                        while remaining_length > avg_width and len(packed_items) < theoretical_items:
                            item_dict, width = items_assigned[item_cycle % len(items_assigned)]
                            is_first_item = (len(packed_items) == 0)
                            required_space = width if is_first_item else width + INTER_SPACING
                            
                            if remaining_length >= required_space:
                                if not is_first_item:
                                    current_pos += INTER_SPACING
                                    remaining_length -= INTER_SPACING
                                
                                packed_items.append({'pos': current_pos, 'width': width, 'height': item_dict['H']})
                                current_pos += width
                                remaining_length -= width
                                item_cycle += 1
                            else:
                                break

        # 5. 绘制所有层和货物
        for i in range(num_layers):
            y_base = params['bottom_clearance'] + i * layer_total_h
            ax.add_patch(patches.Rectangle((0, y_base), spec['Lp'], params['shelf_thickness'], color='#d3d3d3'))
            
            if i == num_layers - 1 and packed_items: # 只在顶层画货物
                for item in packed_items:
                    ax.add_patch(patches.Rectangle((item['pos'], y_base + params['shelf_thickness']),
                                                 item['width'], item['height'],
                                                 facecolor=np.random.rand(3,), edgecolor='black'))
        
        # 6. 添加文本信息
        total_item_width = sum(it['width'] for it in packed_items)
        l_util = total_item_width / (spec['Lp'] - 2*SIDE_SPACING) if spec['Lp'] > 2*SIDE_SPACING else 0
        avg_h = np.mean([it['height'] for it in packed_items]) if packed_items else 0
        h_util = avg_h / spec['H'] if spec['H'] > 0 else 0

        info_text = (f"示意规格: {spec['Lp']:.0f}×{spec['Dp']:.0f}×{spec['H']:.0f}\n"
                     f"总需求: {counts[spec_idx]}个 | 仓库可堆叠: {num_layers}层\n"
                     f"顶层长度利用率: {l_util:.1%}\n"
                     f"顶层平均高度利用率: {h_util:.1%}")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', fc='aliceblue', alpha=0.8))
        ax.set_xticks([]); ax.set_yticks([])

    def toggle_h_params(self):
        self.frame_three_dimensional_params.grid_forget(); self.frame_boundary_params.grid_forget(); self.frame_manual_params.grid_forget()
        if self.h_method_var.get() == "three_dimensional": self.frame_three_dimensional_params.grid(row=self.h_params_row, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        elif self.h_method_var.get() == "boundary": self.frame_boundary_params.grid(row=self.h_params_row, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        else: self.frame_manual_params.grid(row=self.h_params_row, column=0, columnspan=2, padx=20, pady=5, sticky="ew")

    def update_path(self, file_type, path):
        if not path: return
        if path != getattr(self, f'selected_{file_type}_file', None):
            self.update_textbox("\n检测到输入文件已更改，缓存已清空。\n", clear=False); self.cache = {}
        label = self.excel_file_path_label
        setattr(self, f'selected_{file_type}_file', path); label.configure(text=os.path.basename(path), text_color="white")

    def update_textbox(self, text, clear=False):
        self.output_textbox.configure(state="normal")
        if clear: self.output_textbox.delete("1.0", "end")
        self.output_textbox.insert("end", text); self.output_textbox.see("end")
        self.output_textbox.configure(state="disabled"); self.update_idletasks()

    def update_progress(self, current, total, start_time, stage_text):
        progress = current / total if total > 0 else 0; self.progressbar.set(progress); elapsed_time = time.time() - start_time; remaining_text = ""
        if current > 5 and progress > 0.01:
            remaining_time = (elapsed_time / current) * (total - current); remaining_text = f" | 剩余: {remaining_time:.0f}s"
        short_stage_text = stage_text if len(stage_text) <= 30 else stage_text[:27] + "..."; self.status_label.configure(text=f"状态: {short_stage_text} | 已用: {elapsed_time:.0f}s{remaining_text}")

    def start_core_calculation(self):
        self.status_label.configure(text="状态: 正在初始化核心计算...")
        # 确保operable_data在缓存中
        agg_data_cache = self.cache.get('agg_data') if self.cache.get('h_max') == self.current_params['h_max'] else None
        
        # 如果有预计算的装箱装托判定结果，添加到参数中
        if 'packing_summary' in self.cache:
            self.current_params['packing_summary'] = self.cache['packing_summary']
        
        threading.Thread(target=calculation_worker, args=(self.queue, self.current_params, self.cache['raw_data'], self.cache['shelves'], agg_data_cache)).start()

    def display_results(self, final_shelves, solution, coverage_target, packing_summary=None):
        try: params = self.current_params; usable_vertical_space = params['warehouse_h'] - params['bottom_clearance']
        except (ValueError, KeyError): usable_vertical_space = -1; self.update_textbox("\n警告：仓库环境参数无效，无法计算货架层数。\n")
        
        counts = solution['counts']; header = "\n" + "="*70 + "\n" + " " * 22 + ">>> 最终优化方案推荐 <<<\n" + "="*70 + "\n"
        self.update_textbox(header)
        
        # 显示装箱/装托判定结果
        if packing_summary:
            self.update_textbox("装箱/装托判定结果:\n")
            stats = packing_summary['stats']
            total = sum(stats.values())
            for category, count in stats.items():
                category_name = {
                    'box_only': '只能装箱',
                    'pallet_only': '只能装托',
                    'pure_pallet': '纯装托',
                    'mixed': '混装'
                }.get(category, category)
                self.update_textbox(f"  - {category_name}: {count} 个 ({count/total*100:.1f}%)\n")
            self.update_textbox("-" * 70 + "\n")
        
        self.update_textbox(f"算法目标: 在满足~{coverage_target*100:.0f}%覆盖率下，确定最优货架规格\n" + "-" * 70 + "\n")
        self.update_textbox(f"最终方案所需货架总数: {sum(counts)} 个\n推荐的 {len(final_shelves)} 种货架规格及其所需数量:\n")
        for i, shelf in enumerate(final_shelves):
            shelf_type_info = f"(类型: {shelf.get('shelf_type', '未知')})" if 'shelf_type' in shelf else ""
            spec_line = f"  - 规格 {i+1}: {counts[i]} 个 | {shelf['Lp']:.0f}(长)×{shelf['Dp']:.0f}(深)×{shelf['H']:.0f}(高) | 承重: {shelf['Wp']:.0f}kg {shelf_type_info}"
            if usable_vertical_space > 0:
                layer_total_h = shelf['H'] + params['shelf_thickness'] + params['layer_headroom']
                num_layers = math.floor(usable_vertical_space / layer_total_h) if layer_total_h > 0 else 0
                spec_line += f" | 可摆 {num_layers} 层"
            self.update_textbox(spec_line + "\n")
        self.update_textbox("-" * 70 + "\n最终方案实际覆盖率:\n"); self.update_textbox(f"  - SKU数量覆盖率: {solution['coverage_count'] * 100:.2f}%\n  - SKU体积覆盖率: {solution['coverage_volume'] * 100:.2f}%\n")
    
    def display_diagnostics(self, unplaced_skus, detailed_reasons):
        header = "\n" + "="*70 + "\n" + " " * 22 + ">>> 未安放SKU诊断报告 <<<\n" + "="*70 + "\n"
        self.update_textbox(header)
        if unplaced_skus.empty: self.update_textbox("所有可操作的货物均成功安放！\n"); return
        reason_summary = {}
        for reason in detailed_reasons.values():
            simple_reason = reason.split(":")[0].strip()
            reason_summary[simple_reason] = reason_summary.get(simple_reason, 0) + 1
        self.update_textbox(f"总共有 {len(unplaced_skus)} 个可操作货物数因物理限制未能安放，原因汇总如下:\n")
        for reason, count in sorted(reason_summary.items(), key=lambda item: item[1], reverse=True): self.update_textbox(f"  - {reason}: {count} 个SKU\n")
        self.update_textbox("\n" + "-" * 70 + "\n详细原因已写入到输出的Excel文件中。\n");
        self.update_textbox("="*70 + "\n")

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()