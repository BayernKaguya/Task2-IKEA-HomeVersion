# shelf_optimizer_gui_final.py
# 版本号：v1.3
# 功能描述：
# 1. 新增功能：新增了一个功能，用户可以在GUI中直接输入货架参数，而不需要通过文件读取。
# 2. 优化了GUI布局，新增了一个标签，用于显示当前版本号。
# 3. 新增了一个按钮，用于打开关于对话框。
# 4. 新增了一个按钮，用于打开帮助文档。
# 5. 新增了一个按钮，用于打开项目介绍文档。


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
from itertools import combinations
import warnings
import time

warnings.filterwarnings('ignore')

# --- 常量定义 ---
SIDE_SPACING = 50
INTER_SPACING = 100
NUM_BINS_FOR_AGGREGATION = 200

# #############################################################################
# --- 核心计算逻辑 (后台线程运行) ---
# #############################################################################

def read_excel_data(file_path, cols):
    df = pd.read_excel(file_path)
    data = pd.DataFrame()
    # 确保SKU编号列被当作字符串处理
    data['sku_id'] = df[cols['sku_id']].astype(str)
    data['L'] = pd.to_numeric(df[cols['l']], errors='coerce')
    data['D'] = pd.to_numeric(df[cols['d']], errors='coerce')
    data['H'] = pd.to_numeric(df[cols['h']], errors='coerce')
    data['W'] = pd.to_numeric(df[cols['w']], errors='coerce')
    data['V'] = pd.to_numeric(df[cols['v']], errors='coerce')
    
    # 在数据清洗之前，保存原始行号
    data['original_index'] = df.index
    
    # 清洗掉任何字段为空的行
    data = data.dropna()
    
    # --- 修正点 ---
    # 定义需要检查>0的数值列
    numeric_cols = ['L', 'D', 'H', 'W', 'V']
    # 只对这些数值列进行 > 0 的检查，然后用这个结果来筛选整个数据框
    # 这样就避免了拿 'sku_id' 列和数字 0 进行比较
    data = data[(data[numeric_cols] > 0).all(axis=1)]
    
    data.reset_index(drop=True, inplace=True)
    return data

def read_shelf_params(file_path):
    shelves = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 3: continue
            lp, dp, wp = map(float, parts)
            shelves.append({'Lp': lp, 'Dp': dp, 'Wp': wp, 'id': i})
    return shelves

def correlation_analysis(data):
    data_for_corr = data[['L', 'D', 'H']].copy()
    data_for_corr['Area'] = data_for_corr['L'] * data_for_corr['D']
    corrs = {
        'L vs H': abs(data_for_corr['L'].corr(data_for_corr['H'])),
        'D vs H': abs(data_for_corr['D'].corr(data_for_corr['H'])),
        'Area vs H': abs(data_for_corr['Area'].corr(data_for_corr['H']))
    }
    max_corr_label = max(corrs, key=corrs.get)
    max_corr_value = corrs[max_corr_label]
    
    if max_corr_value >= 0.7: grade = 'A级 (强相关)'
    elif max_corr_value >= 0.5: grade = 'B级 (中等相关)'
    elif max_corr_value >= 0.3: grade = 'C级 (弱相关)'
    else: grade = 'D级 (几乎不相关)'
    
    return max_corr_label, max_corr_value, grade

def aggregate_skus(data, num_bins):
    # 确保l_bin和d_bin只在有数据时创建
    if not data.empty:
        data['l_bin'] = pd.cut(data['L'], bins=num_bins, labels=False, duplicates='drop')
        data['d_bin'] = pd.cut(data['D'], bins=num_bins, labels=False, duplicates='drop')
        agg_data = data.groupby(['l_bin', 'd_bin']).agg(
            L=('L', 'median'), D=('D', 'median'), H=('H', 'median'), 
            W=('W', 'median'), V=('V', 'median'), count=('L', 'size'),
            # 收集组内所有的SKU ID
            sku_ids=('sku_id', lambda x: list(x))
        ).reset_index()
        return agg_data
    return pd.DataFrame()


def get_fittable_skus_for_shelf(agg_data, shelf, allow_rotation):
    fittable_skus = []
    for _, sku_group in agg_data.iterrows():
        l, d, w = sku_group['L'], sku_group['D'], sku_group['W']
        
        if w > shelf['Wp']:
            continue
            
        fit_width = -1
        
        if l <= shelf['Lp'] and d <= shelf['Dp']:
            fit_width = l
        
        if allow_rotation and d <= shelf['Lp'] and l <= shelf['Dp']:
            fit_width = min(l, d) if fit_width != -1 else d
        
        if fit_width != -1:
            fittable_skus.append((sku_group.to_dict(), fit_width, int(sku_group['count'])))
            
    return fittable_skus

def ld_calculator_single(agg_data, shelves, coverage_target, allow_rotation, q):
    best_shelf, min_shelf_count = None, float('inf')
    total_sku_count = agg_data['count'].sum()
    total_sku_volume = (agg_data['V'] * agg_data['count']).sum()
    target_count = total_sku_count * coverage_target
    target_volume = total_sku_volume * coverage_target
    
    total_shelves_to_eval = len(shelves)
    start_time = time.time()

    for i, shelf in enumerate(shelves):
        fittable_skus = get_fittable_skus_for_shelf(agg_data, shelf, allow_rotation)
        
        placed_count = sum(item[2] for item in fittable_skus)
        placed_volume = sum(item[0]['V'] * item[2] for item in fittable_skus)

        if total_sku_count == 0 or placed_count < target_count or placed_volume < target_volume:
            q.put(("progress", (i + 1, total_shelves_to_eval, start_time, f"评估L&D规格 {i+1}/{total_shelves_to_eval} (跳过)")))
            continue

        shelf_count = run_bulk_ffd_packing(fittable_skus, shelf['Lp'])
        if shelf_count < min_shelf_count:
            min_shelf_count = shelf_count
            best_shelf = shelf
        
        q.put(("progress", (i + 1, total_shelves_to_eval, start_time, f"评估L&D规格 {i+1}/{total_shelves_to_eval}")))
    
    return best_shelf

def h_calculator_coverage_driven(data, h_max, p1, p2):
    h1 = np.percentile(data['H'], p1)
    h2 = np.percentile(data['H'], p2)
    h1, h2 = min(h1, h_max), min(h2, h_max)
    return sorted(list(set([round(h1), round(h2)])))

def calculate_boundary_effects(data, h_max, step_size, volume_weight, q):
    heights = np.arange(data['H'].min(), h_max + step_size, step_size)
    results = []
    last_count, last_volume = 0, 0
    total_steps, start_time = len(heights), time.time()

    for i, h in enumerate(heights):
        covered_data = data[data['H'] <= h]
        current_count, current_volume = len(covered_data), covered_data['V'].sum()
        delta_count, delta_volume = current_count - last_count, current_volume - last_volume
        
        norm_delta_count = delta_count / len(data) if len(data) > 0 else 0
        norm_delta_volume = delta_volume / data['V'].sum() if data['V'].sum() > 0 else 0
        
        effect = (1 - volume_weight) * norm_delta_count + volume_weight * norm_delta_volume
        results.append({'h': h, 'effect': effect})
        last_count, last_volume = current_count, current_volume

        if i % 10 == 0:
            q.put(("progress", (i + 1, total_steps, start_time, f"计算边界效应中 {i+1}/{total_steps}")))
    return pd.DataFrame(results)

def identify_candidate_heights(boundary_effects, min_diff, num_candidates=15):
    top_effects = boundary_effects.nlargest(num_candidates * 3, 'effect')
    candidates = []
    for _, row in top_effects.iterrows():
        h = row['h']
        if all(abs(h - ch) >= min_diff for ch in candidates):
            candidates.append(h)
        if len(candidates) >= num_candidates: break
    return sorted(candidates)

def h_calculator_boundary_driven(agg_data, operable_data, best_ld_shelf, h_max, coverage_target, allow_rotation, params, q):
    q.put(("log", "--- 步骤 3a: 正在计算高度边界效应 ---\n"))
    effects_df = calculate_boundary_effects(operable_data, h_max, params['height_step'], params['volume_weight'], q)
    
    q.put(("log", "--- 步骤 3b: 正在筛选候选高度点 ---\n"))
    candidate_heights = identify_candidate_heights(effects_df, params['min_height_diff'])
    if len(candidate_heights) < 2:
        raise ValueError(f"未能找到满足最小差值({params['min_height_diff']}mm)的候选高度。请尝试减小该值。")
    q.put(("log", f"发现 {len(candidate_heights)} 个候选高度点: {[f'{h:.0f}' for h in candidate_heights]}\n"))
    
    height_combinations = list(combinations(candidate_heights, 2))
    q.put(("log", f"--- 步骤 3c: 正在评估 {len(height_combinations)} 种高度组合 ---\n"))

    best_combo, min_total_shelves = None, float('inf')
    total_combos, start_time = len(height_combinations), time.time()

    for i, (h1, h2) in enumerate(height_combinations):
        final_shelves = [{'Lp': best_ld_shelf['Lp'], 'Dp': best_ld_shelf['Dp'], 'Wp': best_ld_shelf['Wp'], 'H': h1},
                         {'Lp': best_ld_shelf['Lp'], 'Dp': best_ld_shelf['Dp'], 'Wp': best_ld_shelf['Wp'], 'H': h2}]
        solution = final_allocation_and_counting(agg_data, final_shelves, coverage_target, allow_rotation)
        if solution and sum(solution['counts']) < min_total_shelves:
            min_total_shelves = sum(solution['counts'])
            best_combo = (h1, h2)
        q.put(("progress", (i + 1, total_combos, start_time, f"评估高度组合 {i+1}/{total_combos}")))

    if best_combo is None:
        raise ValueError("在所有候选高度组合中，均未能找到满足覆盖率目标的方案。请尝试降低覆盖率目标。")
    return sorted(list(best_combo))

def final_allocation_and_counting(agg_data, two_final_shelves, coverage_target, allow_rotation):
    assignments = {0: [], 1: []}
    for _, sku_group in agg_data.iterrows():
        possible_fits = []
        orientations = [{'w': sku_group['L'], 'd': sku_group['D']}]
        if allow_rotation and sku_group['L'] != sku_group['D']:
            orientations.append({'w': sku_group['D'], 'd': sku_group['L']})
            
        for i, shelf in enumerate(two_final_shelves):
            if sku_group['W'] > shelf['Wp'] or sku_group['H'] > shelf['H']: continue
            for ori in orientations:
                w, d = ori['w'], ori['d']
                if w <= shelf['Lp'] and d <= shelf['Dp']:
                    rem_space = shelf['Lp'] * shelf['Dp'] - w * d
                    possible_fits.append({'shelf_idx': i, 'width': w, 'sku_group': sku_group, 'rem_space': rem_space})
        
        if not possible_fits: continue
        best_fit = min(possible_fits, key=lambda x: x['rem_space'])
        item_to_place = (best_fit['sku_group'].to_dict(), best_fit['width'], int(best_fit['sku_group']['count']))
        assignments[best_fit['shelf_idx']].append(item_to_place)

    total_sku_count = agg_data['count'].sum()
    total_sku_volume = (agg_data['V'] * agg_data['count']).sum()
    placed_count = sum(item[2] for lst in assignments.values() for item in lst)
    placed_volume = sum(item[0]['V'] * item[2] for lst in assignments.values() for item in lst)
    
    if total_sku_count == 0 or placed_count < total_sku_count * coverage_target or placed_volume < total_sku_volume * coverage_target:
        return None 

    final_counts = [run_bulk_ffd_packing(assignments[i], two_final_shelves[i]['Lp']) for i in range(2)]
    placed_sku_ids = {sid for lst in assignments.values() for item in lst for sid in item[0]['sku_ids']}
    
    return {'counts': final_counts, 
            'coverage_count': placed_count / total_sku_count if total_sku_count > 0 else 0,
            'coverage_volume': placed_volume / total_sku_volume if total_sku_volume > 0 else 0,
            'placed_sku_ids': placed_sku_ids}

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
    sorted_groups = sorted(sku_groups, key=lambda x: x[1], reverse=True)
    bins = []
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
    best_footprint = final_shelves[0]
    for _, sku in unplaced_skus.iterrows():
        sku_id = sku['sku_id']
        if sku['H'] > h_max:
            reasons[sku_id] = f"高度({sku['H']:.0f}mm) > 最大允许高度({h_max:.0f}mm)"
            continue
        if sku['W'] > best_footprint['Wp']:
            reasons[sku_id] = f"重量({sku['W']:.1f}kg) > 货架承重({best_footprint['Wp']:.1f}kg)"
            continue
            
        can_fit_unrotated = sku['L'] <= best_footprint['Lp'] and sku['D'] <= best_footprint['Dp']
        can_fit_rotated = allow_rotation and (sku['D'] <= best_footprint['Lp'] and sku['L'] <= best_footprint['Dp'])
        
        if not can_fit_unrotated and not can_fit_rotated:
            reason_parts = []
            if sku['L'] > best_footprint['Lp'] and sku['D'] > best_footprint['Lp']:
                reason_parts.append(f"长和深均 > 货架长({best_footprint['Lp']:.0f}mm)")
            if sku['L'] > best_footprint['Dp'] and sku['D'] > best_footprint['Dp']:
                reason_parts.append(f"长和深均 > 货架深({best_footprint['Dp']:.0f}mm)")
            if not reason_parts: # Generic fallback
                 reason_parts.append(f"尺寸不匹配 ({sku['L']:.0f}x{sku['D']:.0f} vs {best_footprint['Lp']:.0f}x{best_footprint['Dp']:.0f})")
            reasons[sku_id] = " | ".join(reason_parts)
            continue
        reasons[sku_id] = "因满足覆盖率目标未被选中"
    return reasons

def write_results_to_excel(original_file_path, placed_sku_ids, detailed_reasons, sku_id_col_name):
    try:
        original_df = pd.read_excel(original_file_path)
        
        # 确保SKU ID列是字符串类型以便匹配
        original_df[sku_id_col_name] = original_df[sku_id_col_name].astype(str)
        
        def get_status(sku_id):
            return "成功安放" if sku_id in placed_sku_ids else "未能安放"
            
        def get_reason(sku_id):
            return detailed_reasons.get(sku_id, "") if sku_id not in placed_sku_ids else ""

        original_df['[安放状态]'] = original_df[sku_id_col_name].apply(get_status)
        original_df['[未安放原因]'] = original_df[sku_id_col_name].apply(get_reason)
        
        base, ext = os.path.splitext(original_file_path)
        output_path = f"{base}_results{ext}"
        original_df.to_excel(output_path, index=False)
        return True, output_path, None
    except Exception as e:
        return False, None, str(e)

def calculation_worker(q, params, raw_data, shelves):
    try:
        h_max = params['h_max']
        operable_data = raw_data[raw_data['H'] <= h_max].copy()
        q.put(("log", f"\n已过滤掉 {len(raw_data) - len(operable_data)} 个高度超过 {h_max}mm 的SKU。\n"))
        if len(operable_data) == 0: raise ValueError("所有SKU高度均超过最大允许高度。")
        agg_data = aggregate_skus(operable_data, NUM_BINS_FOR_AGGREGATION)
        q.put(("log", f"数据聚合完成，聚合后规格组数量: {len(agg_data)}\n"))

        q.put(("log", "--- 步骤 2: 正在计算最优L&D规格 ---\n"))
        best_ld = ld_calculator_single(agg_data, shelves, params['coverage_target'], params['allow_rotation'], q)
        if best_ld is None: raise ValueError("L&D计算器未能找到有效的底面积规格。请检查货架参数或SKU数据。")
        q.put(("log", f"\n最优L&D规格确定: {best_ld['Lp']:.0f}x{best_ld['Dp']:.0f}\n\n"))

        q.put(("log", "--- 步骤 3: 正在计算最优H规格 ---\n"))
        if params['h_method'] == 'manual':
            h_cand = h_calculator_coverage_driven(operable_data, h_max, params['p1'], params['p2'])
            q.put(("log", f"使用手动百分位法，最优H规格确定: {h_cand[0]:.0f}mm 和 {h_cand[1]:.0f}mm\n\n"))
        else:
            h_cand = h_calculator_boundary_driven(agg_data, operable_data, best_ld, h_max, params['coverage_target'], params['allow_rotation'], params, q)
            q.put(("log", f"\n使用边界效应算法，最优H规格确定: {h_cand[0]:.0f}mm 和 {h_cand[1]:.0f}mm\n\n"))

        h1, h2 = h_cand[0], h_cand[1]
        final_shelves = [{'Lp': best_ld['Lp'], 'Dp': best_ld['Dp'], 'Wp': best_ld['Wp'], 'H': h1},
                         {'Lp': best_ld['Lp'], 'Dp': best_ld['Dp'], 'Wp': best_ld['Wp'], 'H': h2}]
        
        q.put(("log", "--- 步骤 4: 正在进行最终优化分配与数量核算 ---\n"))
        final_solution = final_allocation_and_counting(agg_data, final_shelves, params['coverage_target'], params['allow_rotation'])
        q.put(("log", "最终优化计算完成。\n"))

        if final_solution is None:
            raise ValueError("最终优化步骤未能找到满足目标覆盖率的方案。")
        else:
            q.put(("result", (final_shelves, final_solution, params['coverage_target'])))
            
            operable_sku_ids = set(operable_data['sku_id'])
            placed_sku_ids = final_solution['placed_sku_ids']
            unplaced_sku_ids = operable_sku_ids - placed_sku_ids
            unplaced_skus_df = operable_data[operable_data['sku_id'].isin(unplaced_sku_ids)]

            detailed_reasons = get_detailed_unplaced_reasons_by_sku_id(unplaced_skus_df, final_shelves, h_max, params['allow_rotation'])
            q.put(("diagnostics", (unplaced_skus_df, detailed_reasons)))
            
            if params['sku_file']:
                success, path, error_msg = write_results_to_excel(params['sku_file'], placed_sku_ids, detailed_reasons, params['cols']['sku_id'])
                if success:
                    q.put(("log", f"\n结果已成功写入到新文件:\n{path}\n"))
                else:
                    q.put(("log", f"\n!!!!!! 写入Excel文件失败 !!!!!!\n错误: {error_msg}\n"))

        q.put(("done", "计算完成！"))
    except Exception as e:
        q.put(("error", str(e)))

# #############################################################################
# --- 图形用户界面 (GUI) ---
# #############################################################################

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("货架配置优化工具 ProVersion-1.0")
        self.geometry("1150x800")
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)
        self.queue = queue.Queue(); self.calc_thread = None
        
        self.frame_left = ctk.CTkScrollableFrame(self, width=350, corner_radius=0, label_text="输入与配置", label_font=ctk.CTkFont(size=16, weight="bold"))
        self.frame_left.grid(row=0, column=0, rowspan=3, sticky="nsw")
        
        r = 0
        ctk.CTkLabel(self.frame_left, text="文件选择", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(10, 10)); r+=1
        self.sku_file_path_label = ctk.CTkLabel(self.frame_left, text="未选择SKU文件", text_color="gray", wraplength=300); self.sku_file_path_label.grid(row=r, column=0, columnspan=2, padx=20); r+=1
        ctk.CTkButton(self.frame_left, text="选择SKU Excel", command=self.select_sku_file).grid(row=r, column=0, padx=(20,5), pady=10, sticky="ew")
        ctk.CTkButton(self.frame_left, text="粘贴", width=60, command=self.paste_sku_path).grid(row=r, column=1, padx=(5,20), pady=10, sticky="ew"); r+=1
        self.shelf_file_path_label = ctk.CTkLabel(self.frame_left, text="未选择货架参数文件", text_color="gray", wraplength=300); self.shelf_file_path_label.grid(row=r, column=0, columnspan=2, padx=20); r+=1
        ctk.CTkButton(self.frame_left, text="选择货架TXT", command=self.select_shelf_file).grid(row=r, column=0, padx=(20,5), pady=10, sticky="ew")
        ctk.CTkButton(self.frame_left, text="粘贴", width=60, command=self.paste_shelf_path).grid(row=r, column=1, padx=(5,20), pady=10, sticky="ew"); r+=1

        ctk.CTkLabel(self.frame_left, text="通用参数配置", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(20, 10)); r+=1
        ctk.CTkLabel(self.frame_left, text="覆盖率目标 (%)").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_coverage = ctk.CTkEntry(self.frame_left); self.entry_coverage.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); self.entry_coverage.insert(0, "90"); r+=1
        ctk.CTkLabel(self.frame_left, text="最大允许高度 (mm)").grid(row=r, column=0, padx=20, pady=(10,0), sticky="w"); self.entry_hmax = ctk.CTkEntry(self.frame_left); self.entry_hmax.grid(row=r, column=1, padx=20, pady=(10,0), sticky="ew"); self.entry_hmax.insert(0, "1800"); r+=1
        self.check_allow_rotation = ctk.CTkCheckBox(self.frame_left, text="允许SKU旋转90°放置"); self.check_allow_rotation.grid(row=r, column=0, columnspan=2, padx=20, pady=10, sticky="w"); self.check_allow_rotation.select(); r+=1
        
        ctk.CTkLabel(self.frame_left, text="H高度计算方法", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r, column=0, columnspan=2, padx=20, pady=(20, 10)); r+=1
        self.h_method_var = tk.StringVar(value="boundary")
        ctk.CTkRadioButton(self.frame_left, text="边界效应算法 (推荐)", variable=self.h_method_var, value="boundary", command=self.toggle_h_params).grid(row=r, column=0, columnspan=2, padx=20, pady=(5,0), sticky="w"); r+=1
        ctk.CTkRadioButton(self.frame_left, text="手动百分位算法", variable=self.h_method_var, value="manual", command=self.toggle_h_params).grid(row=r, column=0, columnspan=2, padx=20, pady=(5,10), sticky="w"); r+=1

        self.frame_boundary_params = ctk.CTkFrame(self.frame_left, fg_color="transparent"); self.frame_manual_params = ctk.CTkFrame(self.frame_left, fg_color="transparent")
        ctk.CTkLabel(self.frame_boundary_params, text="高度步长 (mm)").grid(row=0, column=0, sticky="w"); self.entry_height_step = ctk.CTkEntry(self.frame_boundary_params); self.entry_height_step.grid(row=0, column=1, padx=(10,0), sticky="ew"); self.entry_height_step.insert(0, "10")
        ctk.CTkLabel(self.frame_boundary_params, text="最小高度差值 (mm)").grid(row=1, column=0, sticky="w"); self.entry_min_height_diff = ctk.CTkEntry(self.frame_boundary_params); self.entry_min_height_diff.grid(row=1, column=1, padx=(10,0), pady=5, sticky="ew"); self.entry_min_height_diff.insert(0, "150")
        ctk.CTkLabel(self.frame_boundary_params, text="体积权重系数 α").grid(row=2, column=0, sticky="w"); self.entry_volume_weight = ctk.CTkEntry(self.frame_boundary_params); self.entry_volume_weight.grid(row=2, column=1, padx=(10,0), sticky="ew"); self.entry_volume_weight.insert(0, "0.5")
        
        ctk.CTkLabel(self.frame_manual_params, text="H计算-第1百分位 (%)").grid(row=0, column=0, sticky="w"); self.entry_p1 = ctk.CTkEntry(self.frame_manual_params); self.entry_p1.grid(row=0, column=1, padx=(10,0), sticky="ew"); self.entry_p1.insert(0, "50")
        ctk.CTkLabel(self.frame_manual_params, text="H计算-第2百分位 (%)").grid(row=1, column=0, pady=5, sticky="w"); self.entry_p2 = ctk.CTkEntry(self.frame_manual_params); self.entry_p2.grid(row=1, column=1, padx=(10,0), pady=5, sticky="ew"); self.entry_p2.insert(0, "90")
        self.h_params_row = r

        ctk.CTkLabel(self.frame_left, text="Excel列名配置", font=ctk.CTkFont(size=14, weight="bold")).grid(row=r+1, column=0, columnspan=2, padx=20, pady=(20, 10))
        self.col_entries = {}
        defaults = {'sku_id': 'ArtNo', 'l': 'UL gross length (mm)', 'd': 'UL gross width (mm)','h': 'UL gross height (mm)', 'w': 'UL gross weight (kg)','v': 'UL gross volume (cbm)'}
        labels = {'sku_id': 'SKU编号', 'l': '长度 (L)', 'd': '深度 (D)', 'h': '高度 (H)', 'w': '重量 (W)', 'v': '体积 (V)'}
        col_r = r + 2
        for key, text in labels.items():
            ctk.CTkLabel(self.frame_left, text=text).grid(row=col_r, column=0, padx=20, pady=(10,0), sticky="w")
            entry = ctk.CTkEntry(self.frame_left); entry.grid(row=col_r, column=1, padx=20, sticky="ew"); entry.insert(0, defaults[key]); self.col_entries[key] = entry; col_r += 1

        self.output_textbox = ctk.CTkTextbox(self); self.output_textbox.grid(row=0, column=1, padx=(10,20), pady=20, sticky="nsew")
        # 修正后的代码
        self.button_run = ctk.CTkButton(self, text="开始计算", height=40, font=ctk.CTkFont(size=18, weight="bold"), command=self.start_calculation)
        self.button_run.grid(row=1, column=1, padx=(10,20), pady=(0,10), sticky="ew")
        self.status_label = ctk.CTkLabel(self, text="状态: 空闲"); self.status_label.grid(row=2, column=1, padx=(10,20), pady=5, sticky="w")
        self.progressbar = ctk.CTkProgressBar(self, width=300); self.progressbar.grid(row=2, column=1, padx=(10,20), pady=5, sticky="e"); self.progressbar.set(0)
        
        self.display_welcome_message(); self.toggle_h_params()

    def toggle_h_params(self):
        self.frame_boundary_params.grid_forget(); self.frame_manual_params.grid_forget()
        if self.h_method_var.get() == "boundary": self.frame_boundary_params.grid(row=self.h_params_row, column=0, columnspan=2, padx=20, pady=5, sticky="ew")
        else: self.frame_manual_params.grid(row=self.h_params_row, column=0, columnspan=2, padx=20, pady=5, sticky="ew")

    def display_welcome_message(self):
        self.update_textbox("""欢迎使用货架配置优化工具 ProVersion-1.0！
本工具旨在通过数据驱动的方式，为您推荐最优的两种货架规格，以最小化货架总数。

新增“边界效应算法”参数说明:
- 高度步长: 算法的‘搜索精度’。步长越小(如10mm)，搜索得越仔细，但耗时稍长。
- 最小高度差值: 确保推荐的两种货架规格有实际应用价值，避免尺寸过于接近。
- 体积权重系数 α: 0-1之间的‘决策天平’。靠近0优先覆盖更多种类，靠近1优先覆盖大体积。0.5代表两者同等重要。

请在左侧配置好参数后，点击“开始计算”。
""", True)

    def select_sku_file(self): self.update_path('sku', filedialog.askopenfilename(title="选择SKU数据文件", filetypes=[("Excel files", "*.xlsx *.xls")]))
    def select_shelf_file(self): self.update_path('shelf', filedialog.askopenfilename(title="选择货架参数文件", filetypes=[("Text files", "*.txt")]))
    def paste_sku_path(self): self.paste_path('sku')
    def paste_shelf_path(self): self.paste_path('shelf')

    def paste_path(self, file_type):
        try:
            path = self.clipboard_get().strip().replace('"', '')
            if os.path.isfile(path): self.update_path(file_type, path)
            else: self.status_label.configure(text="状态: 剪贴板内容不是有效的文件路径。")
        except: pass

    def update_path(self, file_type, path):
        if not path: return
        label = self.sku_file_path_label if file_type == 'sku' else self.shelf_file_path_label
        setattr(self, f'selected_{file_type}_file', path)
        label.configure(text=os.path.basename(path), text_color="white")

    def update_textbox(self, text, clear=False):
        self.output_textbox.configure(state="normal")
        if clear: self.output_textbox.delete("1.0", "end")
        self.output_textbox.insert("end", text); self.output_textbox.see("end")
        self.output_textbox.configure(state="disabled"); self.update_idletasks()

    def process_queue(self):
        try:
            msg_type, msg_content = self.queue.get_nowait()
            if msg_type == "log": self.update_textbox(msg_content)
            elif msg_type == "progress": self.update_progress(*msg_content)
            elif msg_type == "result": self.display_results(*msg_content)
            elif msg_type == "diagnostics": self.display_diagnostics(*msg_content)
            elif msg_type == "error":
                self.update_textbox(f"\n!!!!!! 计算出错 !!!!!!\n错误信息: {msg_content}\n"); self.button_run.configure(state="normal")
            elif msg_type == "done":
                self.status_label.configure(text=f"状态: {msg_content}"); self.button_run.configure(state="normal"); self.progressbar.set(1); return
        except queue.Empty: pass
        self.after(100, self.process_queue)
    
    def update_progress(self, current, total, start_time, stage_text):
        progress = current / total if total > 0 else 0; self.progressbar.set(progress)
        elapsed_time = time.time() - start_time; remaining_text = ""
        if current > 5 and progress > 0.01:
            remaining_time = (elapsed_time / current) * (total - current)
            remaining_text = f" | 预计剩余: {remaining_time:.0f}s"
        self.status_label.configure(text=f"状态: {stage_text} | 已用: {elapsed_time:.0f}s{remaining_text}")

    def start_calculation(self):
        try:
            if not hasattr(self, 'selected_sku_file') or not hasattr(self, 'selected_shelf_file'): raise ValueError("请先选择SKU和货架文件。")
            params = {'sku_file': self.selected_sku_file, 'shelf_file': self.selected_shelf_file}
            params.update({k: float(e.get()) for k, e in {'coverage_target': self.entry_coverage, 'h_max': self.entry_hmax}.items()})
            params['coverage_target'] /= 100.0
            params['allow_rotation'] = self.check_allow_rotation.get() == 1
            params['h_method'] = self.h_method_var.get()
            params['cols'] = {key: entry.get() for key, entry in self.col_entries.items()}

            if not all(params['cols'].values()): raise ValueError("所有Excel列名都不能为空。")
            if params['h_method'] == 'manual':
                params.update({k: float(e.get()) for k, e in {'p1': self.entry_p1, 'p2': self.entry_p2}.items()})
            else:
                params.update({k: float(e.get()) for k, e in {'height_step': self.entry_height_step, 'min_height_diff': self.entry_min_height_diff, 'volume_weight': self.entry_volume_weight}.items()})
            
            # --- Start Calculation ---
            self.update_textbox("", True); self.button_run.configure(state="disabled")
            self.status_label.configure(text="状态: 正在读取文件...")
            raw_data = read_excel_data(params['sku_file'], params['cols'])
            shelves = read_shelf_params(params['shelf_file'])
            
            self.update_textbox("文件读取成功。\n--- 步骤 1: 正在进行L/D与H相关性检查 ---\n")
            corr_label, corr_val, grade = correlation_analysis(raw_data)
            self.update_textbox(f"最强相关性: '{corr_label}', r = {corr_val:.3f}, 评级: {grade}\n")
            if messagebox.askyesno("相关性检查", f"检测到L/D与H的相关性为 {grade}。\n是否继续运行？"):
                self.status_label.configure(text="状态: 正在初始化计算...")
                self.calc_thread = threading.Thread(target=calculation_worker, args=(self.queue, params, raw_data, shelves))
                self.calc_thread.start(); self.after(100, self.process_queue)
            else:
                self.update_textbox("用户选择取消操作。\n"); self.status_label.configure(text="状态: 已取消"); self.button_run.configure(state="normal")
        except Exception as e:
            messagebox.showerror("输入或文件错误", f"发生错误: {e}"); self.button_run.configure(state="normal"); self.status_label.configure(text="状态: 空闲")

    def display_results(self, two_shelves, solution, coverage_target):
        counts = solution['counts']; header = "\n" + "="*70 + "\n" + " " * 22 + ">>> 最终优化方案推荐 <<<\n" + "="*70 + "\n"
        self.update_textbox(header); self.update_textbox(f"优化目标: 在满足~{coverage_target*100:.0f}%覆盖率下，最小化货架总数\n" + "-" * 70 + "\n")
        self.update_textbox(f"最优方案所需货架总数: {sum(counts)} 个\n推荐的两种货架规格及其所需数量:\n")
        for i, shelf in enumerate(two_shelves):
            self.update_textbox(f"  - 规格 {i+1}: {counts[i]} 个 | {shelf['Lp']:.0f}(长)×{shelf['Dp']:.0f}(深)×{shelf['H']:.0f}(高) | 承重: {shelf['Wp']:.0f}kg\n")
        self.update_textbox("-" * 70 + "\n方案覆盖率指标:\n")
        self.update_textbox(f"  - SKU数量覆盖率: {solution['coverage_count'] * 100:.2f}%\n  - SKU体积覆盖率: {solution['coverage_volume'] * 100:.2f}%\n")
    
    def display_diagnostics(self, unplaced_skus, detailed_reasons):
        header = "\n" + "="*70 + "\n" + " " * 22 + ">>> 未安放SKU诊断报告 <<<\n" + "="*70 + "\n"
        self.update_textbox(header)
        if unplaced_skus.empty: self.update_textbox("所有可操作的SKU均成功安放！\n"); return
        reason_summary = {}
        for reason in detailed_reasons.values():
            simple_reason = reason.split("(")[0].strip()
            reason_summary[simple_reason] = reason_summary.get(simple_reason, 0) + 1
        self.update_textbox(f"总共有 {len(unplaced_skus)} 个可操作SKU未能安放，原因汇总如下:\n")
        for reason, count in reason_summary.items(): self.update_textbox(f"  - {reason}: {count} 个SKU\n")
        self.update_textbox("-" * 70 + "\n详细原因已写入到输出的Excel文件中。\n" + "="*70 + "\n")

if __name__ == "__main__":
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    app = App()
    app.mainloop()