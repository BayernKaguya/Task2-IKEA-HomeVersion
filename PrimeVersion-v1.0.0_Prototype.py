# shelf_optimizer_prime_version.py
# 版本号：v1.0.0 Prototype

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

# --- 可视化库导入 ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from matplotlib.figure import Figure
from collections import Counter

warnings.filterwarnings('ignore')
# 设置Matplotlib支持中文显示的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 常量定义 ---
SIDE_SPACING = 50
INTER_SPACING = 100
NUM_BINS_FOR_AGGREGATION = 200

# #############################################################################
# --- 核心计算逻辑 (后台线程运行) ---
# #############################################################################

# --- v1.0.0 新增: 数据读取与预处理 ---

def read_excel_data_with_quantity(file_path, cols):
    """
    v1.0.0: 读取包含SKU数量和存储模式的Excel文件。
    """
    df = pd.read_excel(file_path)
    data = pd.DataFrame()
    
    data['sku_id'] = df[cols['sku_id']].astype(str)
    data['qty'] = pd.to_numeric(df[cols['qty']], errors='coerce')
    data['allow_carton'] = df[cols['allow_carton']].astype(bool)
    data['allow_pallet'] = df[cols['allow_pallet']].astype(bool)

    data['L'] = pd.to_numeric(df[cols['l']], errors='coerce')
    data['D'] = pd.to_numeric(df[cols['d']], errors='coerce')
    data['H'] = pd.to_numeric(df[cols['h']], errors='coerce')
    data['W'] = pd.to_numeric(df[cols['w']], errors='coerce')
    data['V'] = pd.to_numeric(df[cols['v']], errors='coerce')
    
    data['original_index'] = df.index
    data = data.dropna()
    
    numeric_cols = ['L', 'D', 'H', 'W', 'V', 'qty']
    data = data[(data[numeric_cols] > 0).all(axis=1)]
    
    data.reset_index(drop=True, inplace=True)
    return data

def aggregate_skus_weighted(data, num_bins):
    """
    v1.0.0: 执行加权聚合，使用SKU数量作为权重。
    """
    if data.empty or len(data) < num_bins:
        agg_data = data.copy()
        agg_data.rename(columns={'qty': 'count'}, inplace=True)
        agg_data['sku_ids'] = agg_data['sku_id'].apply(lambda x: [x])
        return agg_data

    try:
        data['l_bin'] = pd.qcut(data['L'], q=min(num_bins, len(data)-1), labels=False, duplicates='drop')
        data['d_bin'] = pd.qcut(data['D'], q=min(num_bins, len(data)-1), labels=False, duplicates='drop')
    except Exception:
        data['l_bin'] = pd.cut(data['L'], bins=num_bins, labels=False, duplicates='drop')
        data['d_bin'] = pd.cut(data['D'], bins=num_bins, labels=False, duplicates='drop')
    
    agg_funcs = {'sku_ids': ('sku_id', lambda x: list(x)), 'total_qty': ('qty', 'sum')}
    grouped = data.groupby(['l_bin', 'd_bin'])
    agg_data = grouped.agg(**agg_funcs).reset_index()
    
    weighted_cols = ['L', 'D', 'H', 'W', 'V']
    def weighted_mean(g):
        return pd.Series({col: np.average(g[col], weights=g['qty']) for col in weighted_cols})

    weighted_data = grouped.apply(weighted_mean).reset_index()
    
    final_agg_data = pd.merge(agg_data, weighted_data, on=['l_bin', 'd_bin'])
    final_agg_data.rename(columns={'total_qty': 'count'}, inplace=True)
    return final_agg_data

def calculate_pallet_utilization(sku, pallet_params):
    """
    v1.0.0: 计算单个SKU在标准托盘上的满托空间利用率。
    """
    try:
        pallet_l, pallet_d, pallet_h = pallet_params['L'], pallet_params['D'], pallet_params['H']
        n1 = math.floor(pallet_l / sku['L']) * math.floor(pallet_d / sku['D'])
        n2 = math.floor(pallet_l / sku['D']) * math.floor(pallet_d / sku['L'])
        n_per_layer = max(n1, n2)
        if n_per_layer == 0: return 0
        num_layers = math.floor(pallet_h / sku['H'])
        if num_layers == 0: return 0
        total_sku_volume = n_per_layer * num_layers * sku['V']
        pallet_space_volume = pallet_l * pallet_d * pallet_h
        return total_sku_volume / pallet_space_volume if pallet_space_volume > 0 else 0
    except (ValueError, TypeError, ZeroDivisionError):
        return 0

def preprocess_and_decide_storage_mode(raw_data, pallet_params, q):
    """
    v1.0.0: 预处理模块，决策SKU的存储模式。
    """
    q.put(("log", "--- 步骤 1a: 开始进行存储模式决策 ---\n"))
    carton_only_mask = raw_data['allow_carton'] & ~raw_data['allow_pallet']
    pallet_only_mask = raw_data['allow_pallet'] & ~raw_data['allow_carton']
    choice_mask = raw_data['allow_carton'] & raw_data['allow_pallet']
    
    skus_for_pallet_indices = list(raw_data[pallet_only_mask].index)
    skus_for_carton_indices = list(raw_data[carton_only_mask].index)
    
    choice_df = raw_data[choice_mask]
    q.put(("log", f"发现 {len(choice_df)} 种SKU可选择存储模式，开始评估...\n"))
    
    for index, sku in choice_df.iterrows():
        utilization = calculate_pallet_utilization(sku, pallet_params)
        if utilization >= pallet_params['min_util_threshold']:
            skus_for_pallet_indices.append(index)
        else:
            skus_for_carton_indices.append(index)
            
    skus_for_pallets = raw_data.loc[skus_for_pallet_indices].copy()
    skus_for_shelves = raw_data.loc[skus_for_carton_indices].copy()
    
    q.put(("log", f"决策完成: {len(skus_for_pallets)} 种SKU分配至托盘, {len(skus_for_shelves)} 种SKU分配至货架。\n"))
    return skus_for_pallets, skus_for_shelves

def calculate_pallet_requirements(skus_for_pallets, pallet_params):
    """
    v1.0.0: 计算托盘路径下的总托盘需求。
    """
    if skus_for_pallets.empty:
        return 0, "无SKU被分配至托盘存储。"
    total_pallets = 0
    for _, sku in skus_for_pallets.iterrows():
        try:
            pallet_l, pallet_d, pallet_h = pallet_params['L'], pallet_params['D'], pallet_params['H']
            n1 = math.floor(pallet_l / sku['L']) * math.floor(pallet_d / sku['D'])
            n2 = math.floor(pallet_l / sku['D']) * math.floor(pallet_d / sku['L'])
            n_per_layer = max(n1, n2)
            if n_per_layer == 0: continue
            num_layers = math.floor(pallet_h / sku['H'])
            if num_layers == 0: continue
            skus_per_pallet = n_per_layer * num_layers
            pallets_needed = math.ceil(sku['qty'] / skus_per_pallet)
            total_pallets += pallets_needed
        except (ValueError, TypeError, ZeroDivisionError):
            continue
    report = f"总计需要 {total_pallets} 个标准托盘位来存储 {len(skus_for_pallets)} 种SKU。"
    return total_pallets, report

# --- v3.6 继承函数 ---

def is_complementary(shelf1, shelf2, area_threshold, size_threshold):
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
    valid_combinations = []
    for i, shelf1 in enumerate(shelves):
        for j, shelf2 in enumerate(shelves[i+1:], i+1):
            if is_complementary(shelf1, shelf2, area_threshold, size_threshold):
                valid_combinations.append((shelf1, shelf2))
    return valid_combinations

def evaluate_shelf_combination(agg_data, two_shelves, coverage_target, allow_rotation):
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

def read_shelf_params(file_path):
    shelves = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = [p.strip() for p in line.split(',')];
            if len(parts) != 3: continue
            lp, dp, wp = map(float, parts); shelves.append({'Lp': lp, 'Dp': dp, 'Wp': wp, 'id': i})
    return shelves

def correlation_analysis(data):
    data_for_corr = data[['L', 'D', 'H']].copy(); data_for_corr['Area'] = data_for_corr['L'] * data_for_corr['D']
    corrs = {'L vs H': abs(data_for_corr['L'].corr(data_for_corr['H'])), 'D vs H': abs(data_for_corr['D'].corr(data_for_corr['H'])), 'Area vs H': abs(data_for_corr['Area'].corr(data_for_corr['H']))}
    max_corr_label = max(corrs, key=corrs.get); max_corr_value = corrs[max_corr_label]
    if max_corr_value >= 0.7: grade = 'A级 (强相关)'
    elif max_corr_value >= 0.5: grade = 'B级 (中等相关)'
    elif max_corr_value >= 0.3: grade = 'C级 (弱相关)'
    else: grade = 'D级 (几乎不相关)'
    return max_corr_label, max_corr_value, grade

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
        covered_data = data[data['H'] <= h]; current_count, current_volume = len(covered_data), (covered_data['V'] * covered_data['qty']).sum()
        delta_count, delta_volume = current_count - last_count, current_volume - last_volume
        norm_delta_count = delta_count / len(data) if len(data) > 0 else 0
        norm_delta_volume = delta_volume / (data['V'] * data['qty']).sum() if (data['V'] * data['qty']).sum() > 0 else 0
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

def final_placement_with_individual_skus(operable_data, final_shelves, allow_rotation, q):
    q.put(("log", "开始基于真实SKU数据进行精确分配与装箱...\n"))
    num_shelf_types = len(final_shelves)
    assignments = {i: [] for i in range(num_shelf_types)}
    sku_shelf_assignments = {}
    total_skus_to_process = operable_data['qty'].sum()
    processed_skus_count = 0
    start_time = time.time()
    for _, sku_row in operable_data.iterrows():
        for _ in range(int(sku_row['qty'])):
            possible_fits = []
            orientations = [{'w': sku_row['L'], 'd': sku_row['D']}]
            if allow_rotation and sku_row['L'] != sku_row['D']:
                orientations.append({'w': sku_row['D'], 'd': sku_row['L']})
            for i, shelf in enumerate(final_shelves):
                if sku_row['W'] > shelf['Wp'] or sku_row['H'] > shelf['H']: continue
                for ori in orientations:
                    w, d = ori['w'], ori['d']
                    if w <= shelf['Lp'] and d <= shelf['Dp']:
                        vol_util = (w * d * sku_row['H']) / (shelf['Lp'] * shelf['Dp'] * shelf['H']) if shelf['Lp'] > 0 and shelf['Dp'] > 0 and shelf['H'] > 0 else 0
                        possible_fits.append({'shelf_idx': i, 'width': w, 'vol_util': vol_util})
            if possible_fits:
                best_fit = max(possible_fits, key=lambda x: x['vol_util'])
                assignments[best_fit['shelf_idx']].append((sku_row.to_dict(), best_fit['width']))
                sku_shelf_assignments[sku_row['sku_id']] = sku_shelf_assignments.get(sku_row['sku_id'], best_fit['shelf_idx'])
            processed_skus_count += 1
            if processed_skus_count > 0 and processed_skus_count % 200 == 0:
                q.put(("progress", (processed_skus_count, total_skus_to_process, start_time, f"精确分配SKU {processed_skus_count}/{total_skus_to_process}")))
    q.put(("log", "所有真实SKU分配完成，正在精确计算货架数...\n"))
    final_counts = []
    for i, shelf in enumerate(final_shelves):
        items_on_shelf = assignments.get(i, [])
        if not items_on_shelf:
            final_counts.append(0); continue
        packing_groups = []
        sorted_items = sorted(items_on_shelf, key=lambda x: x[1])
        for width, group in groupby(sorted_items, key=lambda x: x[1]):
            count = sum(1 for _ in group)
            packing_groups.append(({'placeholder': True}, width, count))
        shelf_count = run_bulk_ffd_packing(packing_groups, shelf['Lp'])
        final_counts.append(shelf_count)
    placed_sku_ids = set(sku_shelf_assignments.keys())
    placed_skus_df = operable_data[operable_data['sku_id'].isin(placed_sku_ids)]
    coverage_count = placed_skus_df['qty'].sum() / operable_data['qty'].sum() if operable_data['qty'].sum() > 0 else 0
    placed_volume = (placed_skus_df['V'] * placed_skus_df['qty']).sum()
    total_volume = (operable_data['V'] * operable_data['qty']).sum()
    coverage_volume = placed_volume / total_volume if total_volume > 0 else 0
    return {
        'status': 'success', 'counts': final_counts,
        'coverage_count': coverage_count, 'coverage_volume': coverage_volume,
        'placed_sku_ids': placed_sku_ids, 'sku_shelf_assignments': sku_shelf_assignments,
        'assignments': assignments, 'final_shelves': final_shelves
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

def get_detailed_unplaced_reasons_by_sku_id(unplaced_skus, final_shelves, h_max, allow_rotation, skus_on_pallets_ids):
    reasons = {}
    for _, sku in unplaced_skus.iterrows():
        sku_id = sku['sku_id']
        if sku_id in skus_on_pallets_ids:
            reasons[sku_id] = "决策分配: 托盘存储"
            continue
        if sku['H'] > h_max:
            reasons[sku_id] = f"高度超限: SKU高({sku['H']:.0f}) > 最大允许高({h_max:.0f})"
            continue
        can_fit_any_shelf = False
        if final_shelves:
            for shelf in final_shelves:
                if sku['W'] > shelf['Wp']: continue
                if sku['H'] > shelf['H']: continue
                can_fit_unrotated = sku['L'] <= shelf['Lp'] and sku['D'] <= shelf['Dp']
                can_fit_rotated = allow_rotation and (sku['D'] <= shelf['Lp'] and sku['L'] <= shelf['Dp'])
                if can_fit_unrotated or can_fit_rotated:
                    can_fit_any_shelf = True
                    break
        if not can_fit_any_shelf:
            if final_shelves:
                representative_shelf = final_shelves[0]
                min_h_available = min(s['H'] for s in final_shelves)
                if sku['W'] > representative_shelf['Wp']: reasons[sku_id] = f"超重: SKU重({sku['W']:.1f}kg) > 货架承重({representative_shelf['Wp']:.1f}kg)"
                elif sku['H'] > min_h_available: reasons[sku_id] = f"过高: SKU高({sku['H']:.0f}) > 所有推荐货架的净高"
                else: reasons[sku_id] = f"尺寸不匹配: L/D({sku['L']:.0f}x{sku['D']:.0f})与所有推荐货架尺寸均不符"
            else:
                reasons[sku_id] = "无适用货架方案"
        else:
              reasons[sku_id] = "未知原因 (逻辑异常)"
    return reasons

def write_results_to_excel(original_file_path, placed_sku_ids, detailed_reasons, sku_id_col_name, skus_on_pallets_ids, sku_shelf_assignments=None, final_shelves=None):
    try:
        original_df = pd.read_excel(original_file_path); original_df[sku_id_col_name] = original_df[sku_id_col_name].astype(str)
        
        def get_status(sku_id):
            if sku_id in placed_sku_ids: return "成功安放 (货架)"
            elif sku_id in skus_on_pallets_ids: return "成功安放 (托盘)"
            else: return "未能安放"
            
        def get_reason(sku_id): return detailed_reasons.get(sku_id, "")
        
        def get_shelf_spec(sku_id):
            if sku_id in placed_sku_ids and sku_shelf_assignments and final_shelves:
                shelf_idx = sku_shelf_assignments.get(sku_id)
                if shelf_idx is not None and shelf_idx < len(final_shelves):
                    shelf = final_shelves[shelf_idx]; return f"{shelf['Lp']:.0f}×{shelf['Dp']:.0f}×{shelf['H']:.0f}mm"
            return "托盘存储" if sku_id in skus_on_pallets_ids else ""
            
        original_df['[安放状态]'] = original_df[sku_id_col_name].apply(get_status)
        original_df['[未安放原因]'] = original_df[sku_id_col_name].apply(get_reason)
        original_df['[存储规格]'] = original_df[sku_id_col_name].apply(get_shelf_spec)
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
        
        total_placed_width = sum(item[1] for item in items_on_this_shelf)
        total_item_count = len(items_on_this_shelf)
        total_depth_sum = sum((item[0]['D'] if item[1] == item[0]['L'] else item[0]['L']) for item in items_on_this_shelf)
        total_sku_h_sum = sum(item[0]['H'] for item in items_on_this_shelf)
        total_available_length = (shelf_spec['Lp'] - 2 * SIDE_SPACING) * shelf_count
        l_util = total_placed_width / total_available_length if total_available_length > 0 else 0
        log_lines.append(f"  - L-利用率 (长度): {l_util:.2%}")
        weighted_avg_depth = total_depth_sum / total_item_count if total_item_count > 0 else 0
        d_util = weighted_avg_depth / shelf_spec['Dp'] if shelf_spec['Dp'] > 0 else 0
        log_lines.append(f"  - D-利用率 (深度): {d_util:.2%}")
        warehouse_h = params.get('warehouse_h', 6000); bottom_clearance = params.get('bottom_clearance', 150)
        layer_headroom = params.get('layer_headroom', 100); shelf_thickness = params.get('shelf_thickness', 50)
        usable_vertical_space = warehouse_h - bottom_clearance
        single_layer_total_height = shelf_spec['H'] + shelf_thickness + layer_headroom
        num_layers = math.floor(usable_vertical_space / single_layer_total_height) if single_layer_total_height > 0 and usable_vertical_space > 0 else 0
        avg_sku_h = total_sku_h_sum / total_item_count if total_item_count > 0 else 0
        h_util = (num_layers * avg_sku_h) / warehouse_h if warehouse_h > 0 else 0
        log_lines.append(f"  - H-利用率 (高度): {h_util:.2%}")
    log_lines.append("-" * 70); return "\n".join(log_lines)
    
def pre_calculation_worker(q, params):
    try:
        q.put(("log", "--- 步骤 0: 正在读取与预处理文件 ---\n")); q.put(("progress_update", ("正在读取SKU文件...", 0.1)))
        raw_data = read_excel_data_with_quantity(params['sku_file'], params['cols']); q.put(("log", f"读取了 {len(raw_data)} 条有效的SKU数据。\n"))
        q.put(("progress_update", ("正在读取货架文件...", 0.4))); shelves = read_shelf_params(params['shelf_file'])
        q.put(("log", f"读取了 {len(shelves)} 种货架规格。\n")); q.put(("progress_update", ("正在进行数据相关性检验...", 0.7)))
        time.sleep(0.5); corr_label, corr_val, grade = correlation_analysis(raw_data)
        q.put(("pre_calculation_done", (raw_data, shelves, corr_label, corr_val, grade)))
    except Exception as e: q.put(("error", f"在文件读取或预处理阶段发生错误：\n{str(e)}"))

def calculation_worker(q, params, raw_data, shelves, agg_data_cache=None):
    current_step = "初始化"
    try:
        # --- 步骤 1: 存储模式决策 ---
        current_step = "步骤1: 存储模式决策"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
        skus_for_pallets, skus_for_shelves = preprocess_and_decide_storage_mode(raw_data, params['pallet_params'], q)
        skus_on_pallets_ids = set(skus_for_pallets['sku_id'])

        # --- 步骤 2: 并行计算 ---
        # 2a. 托盘路径
        total_pallets_needed, pallet_report = calculate_pallet_requirements(skus_for_pallets, params['pallet_params'])
        q.put(("log", f"--- 托盘需求分析 ---\n{pallet_report}\n" + "-"*70 + "\n"))

        # 2b. 货架路径
        current_step = "步骤2: 货架路径 - 数据过滤与加权聚合"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
        h_max = params['h_max']
        operable_data = skus_for_shelves[skus_for_shelves['H'] <= h_max].copy()
        q.put(("log", f"已过滤掉 {len(skus_for_shelves) - len(operable_data)} 个高度超过 {h_max}mm 的货架SKU。\n"))
        
        final_solution_shelves = None
        if operable_data.empty:
            q.put(("log", "没有需要货架存储的SKU，货架优化部分跳过。\n"))
            final_solution_shelves = {'status': 'skipped', 'placed_sku_ids': set(), 'final_shelves': []}
        else:
            agg_data = aggregate_skus_weighted(operable_data, NUM_BINS_FOR_AGGREGATION)
            q.put(("agg_data_computed", (agg_data, operable_data)))
            q.put(("log", f"数据加权聚合完成，聚合后规格组数量: {len(agg_data)}\n"))
            
            current_step = f"步骤3: 计算最优L&D规格 (使用 {params['ld_method']} 算法)"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
            if params['ld_method'] == 'complementary':
                status, ld_return_value = ld_calculator_complementary(agg_data, shelves, params['coverage_target'], params['allow_rotation'], q, params)
            else: status, ld_return_value = ld_calculator_single(agg_data, shelves, params['coverage_target'], params['allow_rotation'], q)
            if status == "failure":
                error_msg = ld_return_value.get("message", "L&D计算失败") if params['ld_method'] == 'complementary' else (f"L&D计算失败：覆盖率目标 ({params['coverage_target']*100:.1f}%) 过高，无法实现。\n\n" f"--- 诊断信息 ---\n在所有候选货架中，能实现最高覆盖率的规格是:\n - 规格: {ld_return_value['shelf']['Lp']:.0f}(长) x {ld_return_value['shelf']['Dp']:.0f}(深)\n" f" - 最高可达数量覆盖率: {ld_return_value['count_coverage']*100:.2f}%\n\n【建议】\n请将覆盖率目标降低至 {ld_return_value['count_coverage']*100:.1f}% 或以下再尝试计算。")
                q.put(("error", error_msg)); return

            best_ld_for_h_calc = None
            if params['ld_method'] == 'complementary':
                best_ld_combo = ld_return_value
                q.put(("log", f"最优L&D组合确定: {best_ld_combo[0]['Lp']:.0f}x{best_ld_combo[0]['Dp']:.0f} + {best_ld_combo[1]['Lp']:.0f}x{best_ld_combo[1]['Dp']:.0f}\n"))
                best_ld_for_h_calc = best_ld_combo
            else:
                best_ld, _, _ = ld_return_value
                q.put(("log", f"最优L&D规格确定: {best_ld['Lp']:.0f}x{best_ld['Dp']:.0f}\n"))
                best_ld_for_h_calc = best_ld

            current_step = f"步骤4: 计算最优H规格 (使用 {params['h_method']} 算法)"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
            h_calculators = {'manual': h_calculator_coverage_driven, 'three_dimensional': h_calculator_three_dimensional, 'boundary': h_calculator_boundary_driven}
            h_method = params['h_method']; h_cand, eval_details = [], []

            if h_method == 'manual':
                h_cand, eval_details = h_calculators[h_method](operable_data, h_max, params['p1'], params['p2'])
            else:
                h_cand, eval_details = h_calculators[h_method](agg_data, operable_data, best_ld_for_h_calc, params['coverage_target'], params['allow_rotation'], params, q)

            current_step = "步骤5: 最终规格确定与精确装箱"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
            optimal_shelves = None
            if params['h_method'] == 'three_dimensional' and eval_details:
                 feasible_details = [d for d in eval_details if d['is_feasible']]
                 best_solution_detail = max(feasible_details, key=lambda x: x['score']) if feasible_details else max(eval_details, key=lambda x: x['score'])
                 optimal_shelves = best_solution_detail['solution']['final_shelves']
            else:
                if len(h_cand) >= 2:
                    h1, h2 = h_cand[0], h_cand[1]
                    if params['ld_method'] == 'complementary':
                        ld1, ld2 = best_ld_for_h_calc[0], best_ld_for_h_calc[1]
                        optimal_shelves = [
                            {'Lp': ld1['Lp'], 'Dp': ld1['Dp'], 'Wp': ld1['Wp'], 'H': h1}, {'Lp': ld1['Lp'], 'Dp': ld1['Dp'], 'Wp': ld1['Wp'], 'H': h2},
                            {'Lp': ld2['Lp'], 'Dp': ld2['Dp'], 'Wp': ld2['Wp'], 'H': h1}, {'Lp': ld2['Lp'], 'Dp': ld2['Dp'], 'Wp': ld2['Wp'], 'H': h2},
                        ]
                    else:
                        ld = best_ld_for_h_calc
                        optimal_shelves = [{'Lp': ld['Lp'], 'Dp': ld['Dp'], 'Wp': ld['Wp'], 'H': h1}, {'Lp': ld['Lp'], 'Dp': ld['Dp'], 'Wp': ld['Wp'], 'H': h2}]
            if optimal_shelves is None: raise ValueError("未能确定最终的货架规格，计算中止。")
            
            q.put(("log", "最优货架规格已确定，开始执行精确装箱...\n"))
            final_solution_shelves = final_placement_with_individual_skus(operable_data, optimal_shelves, params['allow_rotation'], q)
            q.put(("log", "精确装箱计算完成。\n"))
            if not final_solution_shelves['placed_sku_ids']: raise ValueError("计算失败：即使在最优货架标准下，也未能安放任何SKU。")

        # --- 步骤 6: 结果整理与输出 ---
        current_step = "步骤6: 结果整理"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
        
        q.put(("result", (final_solution_shelves, total_pallets_needed, params)))
        
        if final_solution_shelves.get('status') == 'success':
            utilization_log = calculate_ldh_utilization(final_solution_shelves, params); q.put(("log", utilization_log))

        all_placed_ids = final_solution_shelves.get('placed_sku_ids', set()).union(skus_on_pallets_ids)
        unplaced_skus_df = raw_data[~raw_data['sku_id'].isin(all_placed_ids)]
        detailed_reasons = get_detailed_unplaced_reasons_by_sku_id(raw_data, final_solution_shelves.get('final_shelves'), h_max, params['allow_rotation'], skus_on_pallets_ids)

        q.put(("diagnostics", (raw_data, unplaced_skus_df, detailed_reasons, skus_on_pallets_ids)))
        q.put(("visualization_data", (operable_data, eval_details if 'eval_details' in locals() else [], final_solution_shelves, skus_for_pallets)))

        if params['sku_file']:
            success, path, error_msg = write_results_to_excel(params['sku_file'], final_solution_shelves.get('placed_sku_ids', set()), detailed_reasons, params['cols']['sku_id'], skus_on_pallets_ids, final_solution_shelves.get('sku_shelf_assignments'), final_solution_shelves.get('final_shelves'))
            if success: q.put(("log", f"\n结果已成功写入到新文件:\n{path}\n"))
            else: q.put(("log", f"\n!!!!!! 写入Excel文件失败 !!!!!!\n错误: {error_msg}\n"))

        q.put(("done", "计算完成！"))
    except Exception as e:
        q.put(("error", f"在 [{current_step}] 阶段发生了一个意外错误。\n请检查您的输入数据和参数配置是否正确。\n\n技术细节: {str(e)}"))


if __name__ == "__main__":
    app = App()
    app.mainloop()