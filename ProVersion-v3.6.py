# shelf_optimizer_gui_final.py
# 版本号：v3.6.5 beta

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
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# --- 常量定义 ---
SIDE_SPACING = 50
INTER_SPACING = 100
NUM_BINS_FOR_AGGREGATION = 200

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

def read_excel_data(file_path, cols):
    df = pd.read_excel(file_path); data = pd.DataFrame()
    data['sku_id'] = df[cols['sku_id']].astype(str); data['L'] = pd.to_numeric(df[cols['l']], errors='coerce')
    data['D'] = pd.to_numeric(df[cols['d']], errors='coerce'); data['H'] = pd.to_numeric(df[cols['h']], errors='coerce')
    data['W'] = pd.to_numeric(df[cols['w']], errors='coerce'); data['V'] = pd.to_numeric(df[cols['v']], errors='coerce')
    data['original_index'] = df.index; data = data.dropna()
    numeric_cols = ['L', 'D', 'H', 'W', 'V']; data = data[(data[numeric_cols] > 0).all(axis=1)]
    data.reset_index(drop=True, inplace=True); return data

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

def aggregate_skus(data, num_bins):
    if not data.empty:
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

def final_placement_with_individual_skus(operable_data, final_shelves, allow_rotation, q):
    """
    v3.6.2 (已修正): 最终精确装箱与统计函数
    - 使用未经聚合的`operable_data`进行计算，确保结果精确。
    - 分配SKU时遵循“三维体积利用率最高”原则。
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
        orientations = [{'w': sku['L'], 'd': sku['D']}]
        if allow_rotation and sku['L'] != sku['D']:
            orientations.append({'w': sku['D'], 'd': sku['L']})

        for i, shelf in enumerate(final_shelves):
            if sku['W'] > shelf['Wp'] or sku['H'] > shelf['H']:
                continue
            
            for ori in orientations:
                w, d = ori['w'], ori['d']
                if w <= shelf['Lp'] and d <= shelf['Dp']:
                    vol_util = 0
                    if shelf['Lp'] > 0 and shelf['Dp'] > 0 and shelf['H'] > 0:
                        vol_util = (w * d * sku['H']) / (shelf['Lp'] * shelf['Dp'] * shelf['H'])
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
    placed_volume = placed_skus_df['V'].sum()
    total_volume = operable_data['V'].sum()
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
              # 由于现在最终分配基于真实SKU，此分支理论上不应再被触发
              reasons[sku_id] = "未知原因 (逻辑异常)"
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
        
        # v3.6.5 修正: 直接处理精确的 (sku_dict, placed_width) 元组列表
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
        raw_data = read_excel_data(params['sku_file'], params['cols']); q.put(("log", f"读取了 {len(raw_data)} 条有效的SKU数据。\n"))
        q.put(("progress_update", ("正在读取货架文件...", 0.4))); shelves = read_shelf_params(params['shelf_file'])
        q.put(("log", f"读取了 {len(shelves)} 种货架规格。\n")); q.put(("progress_update", ("正在进行数据相关性检验...", 0.7)))
        time.sleep(0.5); corr_label, corr_val, grade = correlation_analysis(raw_data)
        q.put(("pre_calculation_done", (raw_data, shelves, corr_label, corr_val, grade)))
    except Exception as e: q.put(("error", f"在文件读取或预处理阶段发生错误：\n{str(e)}"))

def calculation_worker(q, params, raw_data, shelves, agg_data=None):
    current_step = "初始化"
    try:
        # --- 步骤 1: 数据聚合 ---
        current_step = "步骤1: 数据聚合"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
        operable_data = None
        if agg_data is None:
            q.put(("log", "缓存未命中，正在进行SKU聚合...\n")); h_max = params['h_max']
            operable_data = raw_data[raw_data['H'] <= h_max].copy()
            q.put(("log", f"已过滤掉 {len(raw_data) - len(operable_data)} 个高度超过 {h_max}mm 的SKU。\n"))
            if len(operable_data) == 0: raise ValueError("所有SKU高度均超过最大允许高度，无法继续计算。")
            agg_data = aggregate_skus(operable_data, NUM_BINS_FOR_AGGREGATION)
            q.put(("agg_data_computed", (agg_data, operable_data)))
        else:
            q.put(("log", "聚合数据缓存命中，跳过聚合步骤。\n")); h_max = params['h_max']
            operable_data = raw_data[raw_data['H'] <= h_max].copy()
            q.put(("agg_data_computed", (agg_data, operable_data))) # 确保operable_data在缓存命中时也被传递
        q.put(("log", f"数据聚合完成，聚合后规格组数量: {len(agg_data)}\n--- {current_step} 完成 ---\n"))

        # --- 步骤 2 & 3: 使用聚合数据寻找最优货架规格 (标准确定) ---
        current_step = f"步骤2: 计算最优L&D规格 (使用 {params['ld_method']} 算法)"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
        if params['ld_method'] == 'complementary':
            status, ld_return_value = ld_calculator_complementary(agg_data, shelves, params['coverage_target'], params['allow_rotation'], q, params)
        else: status, ld_return_value = ld_calculator_single(agg_data, shelves, params['coverage_target'], params['allow_rotation'], q)
        if status == "failure":
            error_msg = ld_return_value.get("message", "L&D计算失败") if params['ld_method'] == 'complementary' else (
                f"L&D计算失败：覆盖率目标 ({params['coverage_target']*100:.1f}%) 过高，无法实现。\n\n"
                f"--- 诊断信息 ---\n在所有候选货架中，能实现最高覆盖率的规格是:\n - 规格: {ld_return_value['shelf']['Lp']:.0f}(长) x {ld_return_value['shelf']['Dp']:.0f}(深)\n"
                f" - 最高可达数量覆盖率: {ld_return_value['count_coverage']*100:.2f}%\n\n【建议】\n请将覆盖率目标降低至 {ld_return_value['count_coverage']*100:.1f}% 或以下再尝试计算。")
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

        q.put(("log", f"--- {current_step} 完成 ---\n")); current_step = f"步骤3: 计算最优H规格 (使用 {params['h_method']} 算法)"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
        h_calculators = {'manual': h_calculator_coverage_driven, 'three_dimensional': h_calculator_three_dimensional, 'boundary': h_calculator_boundary_driven}
        h_method = params['h_method']; h_cand, eval_details = [], []

        if h_method == 'manual':
            h_cand, eval_details = h_calculators[h_method](operable_data, h_max, params['p1'], params['p2'])
        else:
            h_cand, eval_details = h_calculators[h_method](agg_data, operable_data, best_ld_for_h_calc, params['coverage_target'], params['allow_rotation'], params, q)

       # --- 步骤 4: 确定最终规格并使用真实数据执行精确装箱 ---
        current_step = "步骤4: 最终规格确定与精确装箱"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
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
                        {'Lp': ld1['Lp'], 'Dp': ld1['Dp'], 'Wp': ld1['Wp'], 'H': h1},
                        {'Lp': ld1['Lp'], 'Dp': ld1['Dp'], 'Wp': ld1['Wp'], 'H': h2},
                        {'Lp': ld2['Lp'], 'Dp': ld2['Dp'], 'Wp': ld2['Wp'], 'H': h1},
                        {'Lp': ld2['Lp'], 'Dp': ld2['Dp'], 'Wp': ld2['Wp'], 'H': h2},
                    ]
                else:
                    ld = best_ld_for_h_calc
                    optimal_shelves = [{'Lp': ld['Lp'], 'Dp': ld['Dp'], 'Wp': ld['Wp'], 'H': h1}, {'Lp': ld['Lp'], 'Dp': ld['Dp'], 'Wp': ld['Wp'], 'H': h2}]

        if optimal_shelves is None:
            raise ValueError("未能确定最终的货架规格，计算中止。")
        
        q.put(("log", "最优货架规格已确定，开始执行精确装箱...\n"))
        final_solution = final_placement_with_individual_skus(operable_data, optimal_shelves, params['allow_rotation'], q)
        q.put(("log", "精确装箱计算完成。\n"))

        if not final_solution['placed_sku_ids']:
              raise ValueError("计算失败：即使在最优货架标准下，也未能安放任何SKU。")

        # --- 步骤 5: 结果整理与输出 ---
        current_step = "步骤5: 结果整理"; q.put(("log", f"\n--- {current_step} 开始 ---\n"))
        q.put(("result", (optimal_shelves, final_solution, params['coverage_target'])))
        utilization_log = calculate_ldh_utilization(final_solution, params); q.put(("log", utilization_log))
        
        placed_sku_ids = final_solution['placed_sku_ids']; unplaced_skus_df = operable_data[~operable_data['sku_id'].isin(placed_sku_ids)]
        detailed_reasons = get_detailed_unplaced_reasons_by_sku_id(unplaced_skus_df, optimal_shelves, h_max, params['allow_rotation'])
        q.put(("diagnostics", (unplaced_skus_df, detailed_reasons)));
        q.put(("visualization_data", (operable_data, eval_details, final_solution)))

        if params['sku_file']:
            success, path, error_msg = write_results_to_excel(params['sku_file'], placed_sku_ids, detailed_reasons, params['cols']['sku_id'], final_solution.get('sku_shelf_assignments'), optimal_shelves)
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
        self.title("货架配置优化工具 ProVersion-3.6.5 beta")
        self.geometry("1280x850")
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)
        
        self.queue = queue.Queue(); self.current_params = None; self.cache = {}
        
        self.frame_left = ctk.CTkScrollableFrame(self, width=380, corner_radius=0, label_text="输入与配置", label_font=ctk.CTkFont(size=16, weight="bold"))
        self.frame_left.grid(row=0, column=0, rowspan=3, sticky="nsw")
        
        r = 0
        ctk.CTkLabel(self.frame_left, text="文件选择", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(10, 10)); r+=1
        self.sku_file_path_label = ctk.CTkLabel(self.frame_left, text="未选择SKU文件", text_color="gray", wraplength=300); self.sku_file_path_label.grid(row=r, column=0, columnspan=2, padx=20); r+=1
        ctk.CTkButton(self.frame_left, text="选择SKU Excel", command=self.select_sku_file).grid(row=r, column=0, padx=(20,5), pady=10, sticky="ew")
        ctk.CTkButton(self.frame_left, text="粘贴", width=60, command=self.paste_sku_path).grid(row=r, column=1, padx=(5,20), pady=10, sticky="ew"); r+=1
        self.shelf_file_path_label = ctk.CTkLabel(self.frame_left, text="未选择货架参数文件", text_color="gray", wraplength=300); self.shelf_file_path_label.grid(row=r, column=0, columnspan=2, padx=20); r+=1
        ctk.CTkButton(self.frame_left, text="选择货架TXT", command=self.select_shelf_file).grid(row=r, column=0, padx=(20,5), pady=10, sticky="ew")
        ctk.CTkButton(self.frame_left, text="粘贴", width=60, command=self.paste_shelf_path).grid(row=r, column=1, padx=(5,20), pady=10, sticky="ew"); r+=1
        
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
        
        ctk.CTkLabel(self.frame_left, text="Excel列名配置", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(20, 10))
        self.col_entries = {}
        defaults = {'sku_id': 'ArtNo', 'l': 'UL gross length (mm)', 'd': 'UL gross width (mm)','h': 'UL gross height (mm)', 'w': 'UL gross weight (kg)','v': 'UL gross volume (cbm)'}
        labels = {'sku_id': 'SKU编号', 'l': '长度 (L)', 'd': '深度 (D)', 'h': '高度 (H)', 'w': '重量 (W)', 'v': '体积 (V)'}
        col_r = r + 1
        for key, text in labels.items():
            ctk.CTkLabel(self.frame_left, text=text).grid(row=col_r, column=0, padx=20, pady=(10,0), sticky="w")
            entry = ctk.CTkEntry(self.frame_left); entry.grid(row=col_r, column=1, padx=20, sticky="ew"); entry.insert(0, defaults[key]); self.col_entries[key] = entry; col_r += 1
        
        self.tabview = ctk.CTkTabview(self, width=250); self.tabview.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
        self.tabview.add("运行日志"); self.tabview.add("分析图表"); self.tabview.set("运行日志")
        self.output_textbox = ctk.CTkTextbox(self.tabview.tab("运行日志")); self.output_textbox.pack(expand=True, fill="both")
        self.charts_frame = ctk.CTkFrame(self.tabview.tab("分析图表"), fg_color="transparent"); self.charts_frame.pack(expand=True, fill="both")
        self.charts_label = ctk.CTkLabel(self.charts_frame, text="计算完成后将在此处显示图表", font=ctk.CTkFont(size=20)); self.charts_label.pack(expand=True)
        self.canvas = None
        self.button_run = ctk.CTkButton(self, text="开始计算", height=40, font=ctk.CTkFont(size=18, weight="bold"), command=self.start_calculation)
        self.button_run.grid(row=1, column=1, padx=(10,20), pady=(0,10), sticky="ew")
        self.status_label = ctk.CTkLabel(self, text="状态: 空闲", width=400); self.status_label.grid(row=2, column=1, padx=(10,20), pady=5, sticky="w")
        self.progressbar = ctk.CTkProgressBar(self, width=300); self.progressbar.grid(row=2, column=1, padx=(10,20), pady=5, sticky="e"); self.progressbar.set(0)
        
        self.display_welcome_message(); self.toggle_ld_params(); self.toggle_h_params(); self.after(100, self.process_queue)

    def display_welcome_message(self):
        self.update_textbox("""欢迎使用货架配置优化工具 ProVersion-3.6.5 beta！

v3.6.5 beta 更新:
- 可视化升级: 引入全新的四图表“分析仪表盘”，更直观地展示优化方案、决策依据、成果分析和真实装箱效果。
- 兼容性修复: 解决了新版精确装箱逻辑与旧版图表函数的数据结构冲突问题。

--- 参数说明 ---
[核心参数]
- 覆盖率目标: 算法在第一阶段寻找最优货架规格时所依据的核心业务指标。
- 最大允许高度: 过滤掉自身高度超过此值的SKU。

[L&D计算方法]
- 互补性-面积比例阈值: (0-1) 值越小，要求的两种货架底面积差异越大。
- 互补性-尺寸比例阈值: (0-1) 值越大，要求的两种货架长、深尺寸越相似。

[H高度计算方法]
- (三维评分) ...权重: 调整数量、体积、高度三个维度在综合评分中的重要性。
- (边界效应) 体积权重系数 α: (0-1) 靠近0优先覆盖更多SKU种类，靠近1优先覆盖大体积SKU。
- (通用) 高度步长/最小高度差值: 控制算法搜索的精细度和方案的实用性。

请在左侧配置好参数后，点击"开始计算"。
""", True)

    def select_sku_file(self):
        file_path = filedialog.askopenfilename(title="选择SKU Excel文件", filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")])
        if file_path: self.update_path('sku', file_path)

    def select_shelf_file(self):
        file_path = filedialog.askopenfilename(title="选择货架参数TXT文件", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path: self.update_path('shelf', file_path)

    def paste_sku_path(self):
        try:
            path = self.clipboard_get().strip()
            if path and os.path.exists(path): self.update_path('sku', path)
            else: messagebox.showwarning("粘贴失败", "剪贴板中的路径无效或文件不存在")
        except tk.TclError: messagebox.showwarning("粘贴失败", "剪贴板为空或包含无效内容")

    def paste_shelf_path(self):
        try:
            path = self.clipboard_get().strip()
            if path and os.path.exists(path): self.update_path('shelf', path)
            else: messagebox.showwarning("粘贴失败", "剪贴板中的路径无效或文件不存在")
        except tk.TclError: messagebox.showwarning("粘贴失败", "剪贴板为空或包含无效内容")
        
    def toggle_ld_params(self):
        if self.ld_method_var.get() == "complementary": self.frame_complementary_params.grid()
        else: self.frame_complementary_params.grid_remove()

    def collect_params(self):
        params = {'sku_file': self.selected_sku_file, 'shelf_file': self.selected_shelf_file}
        try:
            params.update({k: float(e.get()) for k, e in {
                'coverage_target': self.entry_coverage, 'h_max': self.entry_hmax,
                'warehouse_h': self.entry_warehouse_h, 'bottom_clearance': self.entry_bottom_clearance,
                'layer_headroom': self.entry_layer_headroom, 'shelf_thickness': self.entry_shelf_thickness,
                'area_threshold': self.entry_area_threshold, 'size_threshold': self.entry_size_threshold
            }.items()})
        except ValueError: raise ValueError("通用参数或仓库环境参数中包含无效的非数字输入。")
        params['coverage_target'] /= 100.0
        params.update({'allow_rotation': self.check_allow_rotation.get() == 1, 'ld_method': self.ld_method_var.get(), 'h_method': self.h_method_var.get(), 'cols': {key: entry.get() for key, entry in self.col_entries.items()}})
        if not all(params['cols'].values()): raise ValueError("所有Excel列名都不能为空。")
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
                self.progressbar.set(1.0); raw_data, shelves, corr_label, corr_val, grade = msg_content
                self.cache.update({'sku_path': self.current_params['sku_file'], 'shelf_path': self.current_params['shelf_file'], 'raw_data': raw_data, 'shelves': shelves, 'corr_result': (corr_label, corr_val, grade)})
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
            if not hasattr(self, 'selected_sku_file') or not hasattr(self, 'selected_shelf_file'): raise ValueError("请先选择SKU和货架文件。")
            self.current_params = self.collect_params()
            self.update_textbox("", True); self.button_run.configure(state="disabled"); self.progressbar.set(0)
            if self.cache.get('sku_path') == self.current_params['sku_file'] and self.cache.get('shelf_path') == self.current_params['shelf_file']:
                self.update_textbox("文件缓存命中，跳过文件读取和检验步骤。\n"); self.proceed_with_correlation_check()
            else: self.status_label.configure(text="状态: 正在初始化..."); threading.Thread(target=pre_calculation_worker, args=(self.queue, self.current_params)).start()
        except Exception as e: messagebox.showerror("输入或文件错误", f"发生错误: {e}"); self.button_run.configure(state="normal"); self.status_label.configure(text="状态: 空闲")
    
    def proceed_with_correlation_check(self):
        corr_label, corr_val, grade = self.cache['corr_result']
        raw_data, shelves = self.cache['raw_data'], self.cache['shelves']
        self.update_textbox("--- 步骤 0 完成：文件预处理与检验完毕 ---\n"); self.update_textbox(f"最强相关性: '{corr_label}', r = {corr_val:.3f}, 评级: {grade}\n")
        self.update_textbox("\n--- 数据概览 ---\n"); self.update_textbox(f"有效SKU总数: {len(raw_data)}\n"); self.update_textbox(f"候选货架规格数量: {len(shelves)}\n")
        self.update_textbox(f"SKU尺寸(长*深*高)范围: {raw_data['L'].min():.0f}-{raw_data['L'].max():.0f} * {raw_data['D'].min():.0f}-{raw_data['D'].max():.0f} * {raw_data['H'].min():.0f}-{raw_data['H'].max():.0f} mm\n")
        self.update_textbox(f"货架尺寸(长*深)范围: {min(s['Lp'] for s in shelves):.0f}-{max(s['Lp'] for s in shelves):.0f} * {min(s['Dp'] for s in shelves):.0f}-{max(s['Dp'] for s in shelves):.0f} mm\n")
        if messagebox.askyesno("继续计算?", f"数据概览已显示，L/D与H的相关性为 {grade}。\n是否继续运行核心优化算法？"): self.start_core_calculation()
        else: self.update_textbox("用户选择取消操作。\n"); self.status_label.configure(text="状态: 已取消"); self.button_run.configure(state="normal")
    
    # --- v3.6.5 全新可视化仪表盘 ---
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
        label = self.sku_file_path_label if file_type == 'sku' else self.shelf_file_path_label
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
        
        threading.Thread(target=calculation_worker, args=(self.queue, self.current_params, self.cache['raw_data'], self.cache['shelves'], agg_data_cache)).start()

    def display_results(self, final_shelves, solution, coverage_target):
        try: params = self.current_params; usable_vertical_space = params['warehouse_h'] - params['bottom_clearance']
        except (ValueError, KeyError): usable_vertical_space = -1; self.update_textbox("\n警告：仓库环境参数无效，无法计算货架层数。\n")
        counts = solution['counts']; header = "\n" + "="*70 + "\n" + " " * 22 + ">>> 最终优化方案推荐 <<<\n" + "="*70 + "\n"
        self.update_textbox(header); self.update_textbox(f"算法目标: 在满足~{coverage_target*100:.0f}%覆盖率下，确定最优货架规格\n" + "-" * 70 + "\n")
        self.update_textbox(f"最终方案所需货架总数: {sum(counts)} 个\n推荐的 {len(final_shelves)} 种货架规格及其所需数量:\n")
        for i, shelf in enumerate(final_shelves):
            spec_line = f"  - 规格 {i+1}: {counts[i]} 个 | {shelf['Lp']:.0f}(长)×{shelf['Dp']:.0f}(深)×{shelf['H']:.0f}(高) | 承重: {shelf['Wp']:.0f}kg"
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