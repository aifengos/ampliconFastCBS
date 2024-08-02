# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import time
from scipy.stats import ttest_ind, ttest_1samp, levene

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)



class FastCBS:
    def __init__(self, input_data, seg_p_thr=0.05, combine_p_thr=0.01, sig_diff_stop=True, jump_thr=0.3, diff_fold=3,
                 extreme_point_num=3, pos_cnv_num=2, sample_gene=''):
        self.input_data = input_data
        self.seg_p_thr = seg_p_thr
        self.combine_p_thr = combine_p_thr
        self.sig_diff_stop = sig_diff_stop
        self.jump_thr = jump_thr
        self.diff_fold = diff_fold
        self.extreme_point_num = extreme_point_num
        self.pos_cnv_num = pos_cnv_num
        self.min_index = min(self.input_data.index)
        self.max_index = max(self.input_data.index)
        self.sample_gene = sample_gene

    def run(self):
        depth_segments = list()
        # print(self.input_data, self.min_index, self.max_index)
        cnv_segments = self.rec_segment(self.input_data, self.min_index, self.max_index, depth_segments)
        # print('Filter Segments: ', '\n', cnv_segments)
        # return cnv_segments
        return self.combine_segment(self.input_data, cnv_segments)

    def rec_segment(self, cbs_data, start, end, depth_segments=None, break_point=False):
        if depth_segments is None:
            depth_segments = list()
        if (end - start < 4) or (break_point and self.sig_diff_stop):
            depth_segments.append((start, end))
        else:
            # print(start, end)
            prev_break_point = break_point
            # print(start, end, cbs_data.loc[start:end])
            break_point, p, s, e = self.cbs(cbs_data, start, end)
            # if s - start < 2:
            #     s = start
            # if end - e < 2:
            #     e = end
            # print('Proposed partition of {} to {} from {} to {} with p value {}, significant difference is {}'.format(
            #     start, end, s, e, p, break_point))
            if (not break_point and not prev_break_point) or (e - s == end - start):
                depth_segments.append((start, end))
            else:
                if s - 1 >= start:
                    self.rec_segment(cbs_data, start, s - 1, depth_segments)
                if e > s:
                    self.rec_segment(cbs_data, s, e, depth_segments, break_point)
                if e + 1 <= end:
                    self.rec_segment(cbs_data, e + 1, end, depth_segments)
        return depth_segments

    def cbs(self, cbs_data, start, end):
        select_data = cbs_data.loc[start:end]
        trans_data = select_data.to_frame(name='Values')
        trans_data['Values_Offset'] = trans_data['Values'] - trans_data['Values'].mean()
        trans_data['Offset_Sum'] = trans_data['Values_Offset'].cumsum().shift(1)
        trans_data['Offset_Diff1'] = trans_data['Values_Offset'].diff()
        trans_data['Offset_Diff2'] = trans_data['Values_Offset'].diff(periods=2).shift(-1)
        trans_data[['Offset_Diff_Min', 'Offset_Diff_Max']] = trans_data[['Offset_Diff1', 'Offset_Diff2']].agg(
            lambda x: self.diff_extremal(x), axis=1, result_type='expand')
        trans_data['Offset_Sum_Fix'] = trans_data['Offset_Sum'] - self.diff_fold * trans_data['Offset_Diff_Min']
        # print(diffs, sums_fix)

        jump_thr_candidates = dict()

        up_candidate_min_pos = trans_data[trans_data['Offset_Diff_Min'] > self.jump_thr]
        if not up_candidate_min_pos.empty:
            jump_thr_candidates['Up'] = up_candidate_min_pos['Offset_Sum_Fix'].nsmallest(
                self.extreme_point_num).index.tolist()
        else:
            up_candidate_max_pos = trans_data[trans_data['Offset_Diff_Max'] > self.jump_thr]
            if not up_candidate_max_pos.empty:
                jump_thr_candidates['Up'] = up_candidate_max_pos['Offset_Sum_Fix'].nsmallest(
                    self.extreme_point_num).index.tolist()
            else:
                jump_thr_candidates['Up'] = list()

        down_candidate_min_pos = trans_data[trans_data['Offset_Diff_Min'] < -1 * self.jump_thr]
        if not down_candidate_min_pos.empty:
            jump_thr_candidates['Down'] = down_candidate_min_pos['Offset_Sum_Fix'].nlargest(
                self.extreme_point_num).index.tolist()
        else:
            down_candidate_max_pos = trans_data[trans_data['Offset_Diff_Max'] < -1 * self.jump_thr]
            if not down_candidate_max_pos.empty:
                jump_thr_candidates['Down'] = down_candidate_max_pos['Offset_Sum_Fix'].nlargest(
                    self.extreme_point_num).index.tolist()
            else:
                jump_thr_candidates['Down'] = list()

        pair_type = {'Up': 'Down', 'Down': 'Up'}
        breakpoint_candidates = list()
        for break_type in jump_thr_candidates:
            if jump_thr_candidates[break_type]:
                for break_end in jump_thr_candidates[break_type]:
                    if jump_thr_candidates[pair_type[break_type]]:
                        for pair_end in jump_thr_candidates[pair_type[break_type]]:
                            jump_bp_s, jump_bp_end = min(break_end, pair_end), max(break_end, pair_end)
                            if (jump_bp_s, jump_bp_end) not in breakpoint_candidates:
                                breakpoint_candidates.append((jump_bp_s, jump_bp_end))
                            if (jump_bp_s - start < 2) and (
                                    (start, jump_bp_end) not in breakpoint_candidates):
                                breakpoint_candidates.append((start, jump_bp_end))
                            if (end - jump_bp_end) < 2 and ((jump_bp_s, end) not in breakpoint_candidates):
                                breakpoint_candidates.append((jump_bp_s, end))
                    else:
                        if abs(select_data.loc[start:break_end].mean() - 1) > abs(
                                select_data.loc[break_end:end].mean() - 1):
                            breakpoint_candidates.append((start, break_end))
                        else:
                            breakpoint_candidates.append((break_end, end))
        if breakpoint_candidates:
            breakpoint_p = dict()
            for breakpoint_candidate in breakpoint_candidates:
                candidate_bp_s, candidate_bp_e = breakpoint_candidate

                if candidate_bp_e != end:
                    candidate_bp_e -= 1

                xt = trans_data['Values_Offset'].loc[candidate_bp_s:candidate_bp_e]
                xn = trans_data['Values_Offset'].drop(xt.index)
                if (len(xt) > 1) and (len(xn) > 1):
                    test_stat, test_p = self.t_test(xt, xn)
                    breakpoint_p[(candidate_bp_s, candidate_bp_e)] = test_p

            if breakpoint_p:
                if self.sample_gene == ['24021011B001D01_24021011B001D01L01_S250087264_1', 'BRCA2']:
                    print(breakpoint_p)
                # print(breakpoint_p)
                best_p = min(breakpoint_p.values())
                best_bp_s, best_bp_e = min(breakpoint_p, key=breakpoint_p.get)
                if best_p < self.seg_p_thr:
                    return True, best_p, best_bp_s, best_bp_e
                else:
                    return False, best_p, best_bp_s, best_bp_e
            else:
                return False, 1, start, end
        else:
            return False, 1, start, end

    @staticmethod
    def t_test(a_data, b_data, sd_thr=0.05):
        if len(a_data) > 1 and len(b_data) > 1:
            if len(a_data) > 2 and len(b_data) > 2:
                equal_val_stat, equal_val_p = levene(a_data, b_data)
                if equal_val_p < sd_thr:
                    equal_val_bool = False
                else:
                    equal_val_bool = True
            else:
                equal_val_bool = True
            ttest_stat, ttest_p = ttest_ind(a_data, b_data, equal_var=equal_val_bool)
        elif len(a_data) == 1 and len(b_data) > 1:
            ttest_stat, ttest_p = ttest_1samp(b_data, a_data.values[0])
        elif len(a_data) > 1 and len(a_data) == 1:
            ttest_stat, ttest_p = ttest_1samp(a_data, b_data.values[0])
        else:
            ttest_stat, ttest_p = np.nan, np.nan
        return ttest_stat, ttest_p

    def combine_segment(self, depth_data, depth_segments):
        if len(depth_segments) > 3:
            depth_segments = sorted(depth_segments, key=lambda x: x[0])
            depth_segment_p = dict()
            for depth_segment in depth_segments:
                xt = depth_data.loc[depth_segment[0]:depth_segment[1]]
                xn = depth_data.drop(xt.index)
                test_stat, test_p = self.t_test(xt, xn)
                depth_segment_p[depth_segment] = test_p
            pos_to_call = self.pos_cnv_num
            cnv_segments = list()
            while pos_to_call:
                min_p_segment = min(depth_segment_p, key=depth_segment_p.get)
                min_p_segment_p = depth_segment_p[min_p_segment]
                del depth_segment_p[min_p_segment]
                depth_segments.remove(min_p_segment)
                if min_p_segment_p < self.seg_p_thr:
                    flank_segments_available = {'Left': True, 'Right': False}
                    while any(flank_segments_available.values()):
                        flank_segments_p = dict()
                        if flank_segments_available['Left']:
                            flank_segments_available['Left'] = False
                            left_flank_end = min_p_segment[0] - 1
                            for depth_segment in depth_segments:
                                if depth_segment[1] == left_flank_end:
                                    flank_segments_available['Left'] = True
                                    flank_segments_p[depth_segment] = depth_segment_p[depth_segment]
                        if flank_segments_available['Right']:
                            flank_segments_available['Right'] = False
                            right_flank_start = min_p_segment[1] + 1
                            for depth_segment in depth_segments:
                                if depth_segment[0] == right_flank_start:
                                    flank_segments_available['Right'] = True
                                    flank_segments_p[depth_segment] = depth_segment_p[depth_segment]
                        if any(flank_segments_available.values()):
                            flank_segments_p = sorted(flank_segments_p, key=lambda x: x[0])
                            for flank_segment in flank_segments_p:
                                if flank_segment[0] < min_p_segment[0]:
                                    merged_start = flank_segment[0]
                                    merged_end = min_p_segment[1]
                                    flank_type = 'Left'
                                else:
                                    merged_start = min_p_segment[0]
                                    merged_end = flank_segment[1]
                                    flank_type = 'Right'
                                xt = depth_data.loc[min_p_segment[0]:min_p_segment[1]]
                                xf = depth_data.loc[flank_segment[0]:flank_segment[1]]
                                tf_stat, tf_p = self.t_test(xt, xf)
                                if tf_p > self.seg_p_thr:
                                    xm = depth_data.loc[merged_start:merged_end]
                                    xn = depth_data.drop(xt.index)
                                    test_stat, test_p = self.t_test(xm, xn)
                                    if test_p < self.seg_p_thr:
                                        # print(min_p_segment, flank_segment)
                                        depth_segments.remove(flank_segment)
                                        del depth_segment_p[flank_segment]
                                        min_p_segment = (merged_start, merged_end)
                                        flank_segments_available[flank_type] = True
                                    else:
                                        flank_segments_available[flank_type] = False
                                else:
                                    flank_segments_available[flank_type] = False
                    cnv_segments.append(min_p_segment)
                    pos_to_call -= 1
                else:
                    depth_segments.append(min_p_segment)
                    pos_to_call = 0
            # if self.sample_gene[1] == 'MET':
            #     print(self.sample_gene, depth_segment_p)
            if depth_segments:
                prev_segment = depth_segments[0]
                for depth_segment in depth_segments[1:]:
                    if prev_segment[1] + 1 == depth_segment[0]:
                        xp = depth_data.loc[prev_segment[0]:prev_segment[1]]
                        xc = depth_data.loc[depth_segment[0]:depth_segment[1]]
                        test_stat, test_p = self.t_test(xp, xc)
                        if test_p < self.combine_p_thr:
                            cnv_segments.append(prev_segment)
                            prev_segment = depth_segment
                        else:
                            prev_segment = (prev_segment[0], depth_segment[1])
                    else:
                        cnv_segments.append(prev_segment)
                        prev_segment = depth_segment
                cnv_segments.append(prev_segment)
            cnv_segments = sorted(cnv_segments, key=lambda x: x[0])
            # print(self.sample_gene, depth_segments, cnv_segments)
            return cnv_segments
        return depth_segments

    @staticmethod
    def diff_extremal(row):
        if row.isnull().values.any():
            return [np.nan, np.nan]
        else:
            min_row_idx = row.abs().idxmin()
            max_row_idx = row.abs().idxmax()
            # print(row, min_row_idx)
            return [row[min_row_idx], row[max_row_idx]]


def cbs_call(sample_data, call_depth_col, amp_thr, region_thr, cnv_amps_thr, call_mode='Germline',
             unstable_amp_ratio=0.33, gene_multiple_cnv=2):
    # Amplicon Call
    sample_data[['Amp_States', 'Region_States', 'Region_Gene_Str', 'Stable_Region_States',
                 'Unstable_Region_States', 'Region_Ratio', 'Region_Ratio_Std', 'Region_States_Amps', 'Region_Amps',
                 'Region_States_Amp_ratio', 'Region_Loc', 'Region_Length', 'Region_Exon_Count',
                 'Adjacent_Ratio_Gap', 'Region_Type', 'Region_Index', 'Region_Index_Raw', 'Gene_Ratio',
                 'Gene_Ratio_Std']] = np.nan

    for amp_type in ['Stable', 'Unstable']:
        sample_data.loc[(sample_data['Amp_Type'] == amp_type) &
                        (sample_data[call_depth_col] > amp_thr[amp_type]['Gain']), [
            'Amp_States']] = 'Gain'
        if call_mode == 'Germline':
            sample_data.loc[(sample_data['Amp_Type'] == amp_type) &
                            (sample_data[call_depth_col] < amp_thr[amp_type]['Loss']), [
                'Amp_States']] = 'Loss'
    # print(amp_thr, self.call_mode, sample_data[['Amplicon', call_depth_col, 'Amp_Type', 'Amp_States']])
    sample_data['Amp_States'].fillna('Negative', inplace=True)

    for sample_gene in sample_data[['Sample_Name', 'Gene']].drop_duplicates().values.tolist():
        sample_gene_data = sample_data[(sample_data['Sample_Name'] == sample_gene[0]) &
                                       (sample_data['Gene'] == sample_gene[1])]
        gene_ratio, gene_ratio_std = calc_mean_std(sample_gene_data[call_depth_col])
        sample_data.loc[sample_gene_data.index, 'Gene_Ratio'] = gene_ratio
        sample_data.loc[sample_gene_data.index, 'Gene_Ratio_Std'] = gene_ratio_std
        # cbs_depth_segments = FastCBS(sample_gene_data[call_depth_col], 0.05, 0.01, False, 0.3, 3, 3,
        #                              gene_multiple_cnv, sample_gene).run()
        cbs_depth_segments = FastCBS(sample_gene_data[call_depth_col], 0.05, 0.01, False, 0.3, 3, 3,
                                     gene_multiple_cnv, sample_gene).run()
        # if sample_gene[1] == 'ERBB2':
        #     print(sample_gene, cbs_depth_segments)
        # Fix miss split amps at boundary
        split_segments = list()
        for segment_index, cbs_depth_segment in enumerate(cbs_depth_segments):
            segment_start, segment_end = cbs_depth_segment
            segment_data = sample_gene_data.loc[segment_start:segment_end]
            segment_cnv_type, segment_stability, segment_type_amp_ratio, segment_ratio, segment_ratio_std = segment_cnv_check(
                segment_data, call_depth_col, unstable_amp_ratio, region_thr)
            if call_mode == 'Germline':
                segment_states = segment_cnv_type[segment_stability]
                if (segment_states == 'Negative') and (segment_type_amp_ratio < 0.5):
                    segment_type_miss_amps = segment_data[segment_data['Amp_States'].isin(['Negative'])]
                    if not segment_type_miss_amps.empty:
                        # print(segment_type_miss_amps[['Amplicon', call_depth_col]])
                        amp_thr_failed = segment_type_miss_amps.index.values.tolist()
                        if len(amp_thr_failed) < len(segment_data) / 2:
                            amp_thr_failed_groups = int_continuous_split(amp_thr_failed, 1)
                            fix_board = False
                            for amp_thr_failed_group in amp_thr_failed_groups:
                                # print(amp_thr_failed_group)
                                if segment_start in amp_thr_failed_group:
                                    split_segments.append((min(amp_thr_failed_group), max(amp_thr_failed_group)))
                                    segment_start = max(amp_thr_failed_group) + 1
                                    fix_board = True
                                if segment_end in amp_thr_failed_group:
                                    split_segments.append((min(amp_thr_failed_group), max(amp_thr_failed_group)))
                                    segment_end = min(amp_thr_failed_group) - 1
                                    fix_board = True
                            if fix_board:
                                # print(segment_start, segment_end)
                                segment_data = sample_gene_data.loc[segment_start:segment_end]
                                segment_cnv_type, segment_stability, segment_type_amp_ratio, segment_ratio, segment_ratio_std = segment_cnv_check(
                                    segment_data, call_depth_col, unstable_amp_ratio, region_thr)
            sample_data.loc[segment_data.index, 'Region_Index_Raw'] = segment_index
            sample_data.loc[segment_data.index, 'Region_Type'] = segment_stability
            sample_data.loc[segment_data.index, 'Region_Ratio'] = segment_ratio
            sample_data.loc[segment_data.index, 'Region_Ratio_Std'] = segment_ratio_std
            sample_data.loc[segment_data.index, 'Stable_Region_States'] = segment_cnv_type['Stable']
            sample_data.loc[segment_data.index, 'Unstable_Region_States'] = segment_cnv_type['Unstable']
            sample_data.loc[segment_data.index, 'Region_States'] = segment_cnv_type[segment_stability]
        if split_segments:
            for segment_index, split_segment in enumerate(split_segments):
                segment_start, segment_end = split_segment
                segment_data = sample_gene_data.loc[segment_start:segment_end]
                segment_cnv_type, segment_stability, segment_type_amp_ratio, segment_ratio, segment_ratio_std = segment_cnv_check(
                    segment_data, call_depth_col, unstable_amp_ratio, region_thr)
                sample_data.loc[segment_data.index, 'Region_Index_Raw'] = 1000 + segment_index
                sample_data.loc[segment_data.index, 'Region_Type'] = segment_stability
                sample_data.loc[segment_data.index, 'Region_Ratio'] = segment_ratio
                sample_data.loc[segment_data.index, 'Region_Ratio_Std'] = segment_ratio_std
                sample_data.loc[segment_data.index, 'Stable_Region_States'] = segment_cnv_type['Stable']
                sample_data.loc[segment_data.index, 'Unstable_Region_States'] = segment_cnv_type['Unstable']
                sample_data.loc[segment_data.index, 'Region_States'] = segment_cnv_type[segment_stability]
            keep_segments = sorted(
                sample_data.loc[sample_gene_data.index]['Region_Index_Raw'].drop_duplicates().values.tolist())
            if keep_segments:
                keep_segments = {item: index for index, item in enumerate(keep_segments)}
                sample_data.loc[sample_gene_data.index, 'Region_Index_Raw'] = sample_data.loc[
                    sample_gene_data.index, 'Region_Index_Raw'].replace(keep_segments)
        # Merge consecutive intervals of the same CNV state
        sample_gene_data = sample_data[(sample_data['Sample_Name'] == sample_gene[0]) &
                                       (sample_data['Gene'] == sample_gene[1])]
        for cnv_type in sample_gene_data['Region_States'].drop_duplicates().values.tolist():
            cnv_type_segments = sample_gene_data[sample_gene_data['Region_States'] == cnv_type][
                'Region_Index_Raw'].drop_duplicates().values.tolist()
            if cnv_type_segments:
                type_segments_groups = int_continuous_split(cnv_type_segments, 1)
                for segments_group in type_segments_groups:
                    if len(segments_group) > 1:
                        segment_group_data = sample_gene_data[sample_gene_data['Region_Index_Raw'].isin(segments_group)]
                        group_cnv, group_stability, group_type_amp_ratio, group_ratio, group_ratio_std = segment_cnv_check(
                            segment_group_data, call_depth_col, unstable_amp_ratio, region_thr)
                        group_cnv_type = group_cnv[group_stability]
                        if group_cnv_type == cnv_type:
                            sample_data.loc[segment_group_data.index, 'Region_Index'] = min(segments_group)
                            sample_data.loc[segment_group_data.index, 'Region_Type'] = group_stability
                            sample_data.loc[segment_group_data.index, 'Region_Ratio'] = group_ratio
                            sample_data.loc[segment_group_data.index, 'Region_Ratio_Std'] = group_ratio_std
                            sample_data.loc[segment_group_data.index, 'Stable_Region_States'] = group_cnv[
                                'Stable']
                            sample_data.loc[segment_group_data.index, 'Unstable_Region_States'] = group_cnv[
                                'Unstable']
                            sample_data.loc[segment_group_data.index, 'Region_States'] = group_cnv_type
        sample_data.loc[sample_gene_data.index, 'Region_Index'] = sample_data.loc[sample_gene_data.index][
            'Region_Index'].fillna(sample_data.loc[sample_gene_data.index]['Region_Index_Raw'])
        keep_segments = sorted(
            sample_data.loc[sample_gene_data.index]['Region_Index'].drop_duplicates().values.tolist())
        if keep_segments:
            keep_segments = {item: index for index, item in enumerate(keep_segments)}
            sample_data.loc[sample_gene_data.index, 'Region_Index'] = sample_data.loc[
                sample_gene_data.index, 'Region_Index'].replace(keep_segments)

            # Merge consecutive intervals with the same positive CNV state
            # if sample_gene[1] == 'MET':
            #     print(keep_segments, '\n', sample_data.loc[sample_gene_data.index][['Region_Index', 'Region_Index_Raw']])
            sample_gene_data = sample_data[(sample_data['Sample_Name'] == sample_gene[0]) &
                                           (sample_data['Gene'] == sample_gene[1])]
            for cnv_type in ['Gain', 'Loss']:
                cnv_type_segments = sample_gene_data[sample_gene_data['Region_States'] == cnv_type][
                    'Region_Index'].drop_duplicates().values.tolist()
                if cnv_type_segments:
                    type_segments_groups = int_continuous_split(cnv_type_segments, 2)
                    for segments_group in type_segments_groups:
                        seg_index_min, segment_index_max = min(segments_group), max(segments_group)
                        if (segment_index_max - seg_index_min + 1) > len(segments_group):
                            segment_group_data = sample_gene_data[sample_gene_data['Region_Index'].between(
                                seg_index_min, segment_index_max, inclusive='both')]
                            group_cnv, group_stability, group_type_amp_ratio, group_ratio, group_ratio_std = segment_cnv_check(
                                segment_group_data, call_depth_col, unstable_amp_ratio, region_thr)
                            other_type_mean = sample_gene_data[sample_gene_data['Region_States'] != cnv_type][
                                call_depth_col].mean()
                            if (other_type_mean - 1) > 0.8 * (region_thr[group_stability][cnv_type] - 1):
                                group_cnv_type = group_cnv[group_stability]
                                if group_cnv_type == cnv_type:
                                    sample_data.loc[segment_group_data.index, 'Region_Index'] = min(segments_group)
                                    sample_data.loc[segment_group_data.index, 'Region_Type'] = group_stability
                                    sample_data.loc[segment_group_data.index, 'Region_Ratio'] = group_ratio
                                    sample_data.loc[segment_group_data.index, 'Region_Ratio_Std'] = group_ratio_std
                                    sample_data.loc[segment_group_data.index, 'Stable_Region_States'] = group_cnv[
                                        'Stable']
                                    sample_data.loc[segment_group_data.index, 'Unstable_Region_States'] = group_cnv[
                                        'Unstable']
                                    sample_data.loc[segment_group_data.index, 'Region_States'] = group_cnv_type
        keep_segments = sorted(
            sample_data.loc[sample_gene_data.index]['Region_Index'].drop_duplicates().values.tolist())
        if keep_segments:
            keep_segments = {item: index for index, item in enumerate(keep_segments)}
            # if sample_gene[1] == 'MET':
            #     print(keep_segments)
            sample_data.loc[sample_gene_data.index, 'Region_Index'] = sample_data.loc[
                sample_gene_data.index, 'Region_Index'].replace(keep_segments)
        sample_gene_data = sample_data[(sample_data['Sample_Name'] == sample_gene[0]) &
                                       (sample_data['Gene'] == sample_gene[1])]
        for index_cnv_type in sample_gene_data[['Region_Index', 'Region_States']].drop_duplicates().values:
            segment_index, cnv_type = index_cnv_type
            segment_data = sample_gene_data[sample_gene_data['Region_Index'] == segment_index]
            if cnv_type in ['Loss', 'Gain']:
                region_amps = segment_data.shape[0]
                if region_amps >= cnv_amps_thr[cnv_type]:
                    region_cnv_amps = segment_data[segment_data['Amp_States'] == cnv_type].shape[0]
                    sample_data.loc[segment_data.index, 'Region_States_Amps'] = region_cnv_amps
                    sample_data.loc[segment_data.index, 'Region_Amps'] = region_amps
                    type_amp_ratio = region_cnv_amps / region_amps
                    sample_data.loc[segment_data.index, 'Region_States_Amp_ratio'] = type_amp_ratio

                    max_exon_num = segment_data['Exon_Str_End'].max()
                    min_exon_num = segment_data['Exon_Str_Start'].min()
                    if max_exon_num == min_exon_num:
                        exon_region_str = str(min_exon_num)
                    else:
                        exon_region_str = str(min_exon_num) + '-' + str(max_exon_num)
                    cnv_exon_count = max_exon_num - min_exon_num + 1
                    if call_mode == 'Germline':
                        cnv_region_str = sample_gene[1] + '_E' + exon_region_str
                        next_exon_order = segment_data['Exon_Order'].max() + 1
                        prev_exon_order = segment_data['Exon_Order'].min() - 1
                        # region_adjacent_data = sample_gene_data.loc[sample_gene_data['Exon_Order'].isin(
                        #     [prev_exon_order, next_exon_order])]
                        region_adjacent_data = sample_gene_data.loc[
                            sample_gene_data['Exon_Order'].between(prev_exon_order, next_exon_order,
                                                                   inclusive='both') & ~sample_gene_data.index.isin(
                                segment_data.index)]
                        sample_data.loc[segment_data.index, 'Region_Exon_Count'] = cnv_exon_count
                    else:
                        cnv_region_str = sample_gene[1]
                        region_adjacent_data = pd.DataFrame()
                    sample_data.loc[segment_data.index, 'Region_Gene_Str'] = cnv_region_str
                    region_start_loc = segment_data['Scan_Start'].min()
                    region_end_loc = segment_data['Scan_End'].max()
                    region_length = region_end_loc - region_start_loc
                    amp_region = str(region_start_loc) + '-' + str(region_end_loc)
                    chr_str = '-'.join(segment_data['Chr'].drop_duplicates().values.tolist())
                    sample_data.loc[segment_data.index, 'Region_Loc'] = chr_str + ':' + amp_region
                    sample_data.loc[segment_data.index, 'Region_Length'] = region_length
                    segment_region_ratio = segment_data[call_depth_col].mean()
                    if not region_adjacent_data.empty:
                        adjacent_ratio_gap = abs(
                            region_adjacent_data[call_depth_col].mean() - segment_region_ratio)
                    else:
                        adjacent_ratio_gap = 0.5
                    sample_data.loc[segment_data.index, 'Adjacent_Ratio_Gap'] = adjacent_ratio_gap
                else:
                    sample_data.loc[segment_data.index, 'Region_States'] = 'Negative'

        sample_gene_data = sample_data[(sample_data['Sample_Name'] == sample_gene[0]) &
                                       (sample_data['Gene'] == sample_gene[1])]
        negative_segments = sample_gene_data[sample_gene_data['Region_States'] == 'Negative'][
            'Region_Index'].drop_duplicates().values.tolist()
        if negative_segments:
            neg_segments_groups = int_continuous_split(negative_segments, 1)
            for neg_segments_group in neg_segments_groups:
                if len(neg_segments_group) > 1:
                    seg_index_min, segment_index_max = min(neg_segments_group), max(neg_segments_group)
                    segment_group_data = sample_gene_data[sample_gene_data['Region_Index'].between(
                        seg_index_min, segment_index_max, inclusive='both')]
                    group_cnv, group_stability, group_type_amp_ratio, group_ratio, group_ratio_std = segment_cnv_check(
                        segment_group_data, call_depth_col, unstable_amp_ratio, region_thr)
                    sample_data.loc[segment_group_data.index, 'Region_Index'] = min(neg_segments_group)
                    sample_data.loc[segment_group_data.index, 'Region_Type'] = group_stability
                    sample_data.loc[segment_group_data.index, 'Region_Ratio'] = group_ratio
                    sample_data.loc[segment_group_data.index, 'Region_Ratio_Std'] = group_ratio_std
                    sample_data.loc[segment_group_data.index, 'Stable_Region_States'] = 'Negative'
                    sample_data.loc[segment_group_data.index, 'Unstable_Region_States'] = 'Negative'
        keep_segments = sorted(
            sample_data.loc[sample_gene_data.index]['Region_Index'].drop_duplicates().values.tolist())
        if keep_segments:
            keep_segments = {item: index for index, item in enumerate(keep_segments)}
            # if sample_gene[1] == 'MET':
            #     print(keep_segments)
            sample_data.loc[sample_gene_data.index, 'Region_Index'] = sample_data.loc[
                sample_gene_data.index, 'Region_Index'].replace(keep_segments)

    return sample_data


def segment_cnv_check(segment_data, call_depth_col, unstable_amp_ratio, region_thr):
    region_amps = segment_data.shape[0]
    segment_region_ratio, segment_region_ratio_std = calc_mean_std(segment_data[call_depth_col])
    region_unstable_amps = segment_data[segment_data['Amp_Type'] == 'Unstable'].shape[0]
    if region_unstable_amps / region_amps > unstable_amp_ratio:
        region_stability = 'Unstable'
    else:
        region_stability = 'Stable'
        # print(segment_data, gene)
    stability_cnv_type = dict()
    for stability_type in ['Stable', 'Unstable']:
        if segment_region_ratio < region_thr[stability_type]['Loss']:
            cnv_type = 'Loss'
            type_amp_ratio = len(segment_data[segment_data[call_depth_col] <
                                              region_thr[stability_type]['Loss']]) / region_amps
        elif segment_region_ratio > region_thr[stability_type]['Gain']:
            cnv_type = 'Gain'
            type_amp_ratio = len(segment_data[segment_data[call_depth_col] >
                                              region_thr[stability_type]['Gain']]) / region_amps
        else:
            cnv_type = 'Negative'
            type_amp_ratio = len(segment_data[segment_data[call_depth_col].between(
                region_thr[stability_type]['Loss'], region_thr[stability_type]['Gain'],
                inclusive='neither')]) / region_amps
        stability_cnv_type[stability_type] = cnv_type
    return stability_cnv_type, region_stability, type_amp_ratio, segment_region_ratio, segment_region_ratio_std


def calc_mean_std(calc_data):
    data_len = len(calc_data)
    if data_len > 1:
        return calc_data.mean(), calc_data.std(ddof=0)
    else:
        return calc_data.values[0], 0.001


def int_continuous_split(int_list, max_gap_value):
    """ Check number order Continuity
    """

    num_groups = list()
    pre_num = int_list[0]
    num_sequence_list = [pre_num]
    for num_item in int_list[1:]:
        if abs(num_item - pre_num) > max_gap_value:
            num_groups.append(num_sequence_list)
            num_sequence_list = [num_item]
        else:
            num_sequence_list.append(num_item)
        pre_num = num_item
    num_groups.append(num_sequence_list)
    return num_groups


if __name__ == '__main__':
    region_thr = {"Loss": 0.64, "Gain": 1.34}
    amp_thr = {"Loss": 0.64, "Gain": 1.34}
    cnv_amps_thr = {"Loss": 2, "Gain": 2}
    pos_data = pd.read_csv('./amplicon_depth_read_data.tsv', sep='\t', low_memory=False)
    t1 = time.time()
    call_col = 'Ref_Norm_Depth'
    cbs_data = cbs_call(pos_data, call_col, amp_thr, region_thr, cnv_amps_thr)
    t2 = time.time()
    cost_time = t2 - t1
    print('Run time: {}s'.format(cost_time))
