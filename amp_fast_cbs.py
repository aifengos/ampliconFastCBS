# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import time
import itertools
from scipy.stats import mannwhitneyu

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# Fast Circular Binary Segmentation
class FastCBS:
    def __init__(self, input_data, cbs_depth_col, amp_thr, region_thr, unstable_amp_ratio=0.33, gene_multiple_cnv=2,
                 call_mode='Germline', seg_p_thr=0.05, combine_p_thr=0.01, sig_diff_stop=True, jump_thr=0.3,
                 diff_fold=3, extreme_point_num=5, break_point_num=5):
        self.input_data = input_data
        self.cbs_depth_col = cbs_depth_col
        self.amp_thr = amp_thr
        self.region_thr = region_thr
        self.unstable_amp_ratio = unstable_amp_ratio
        self.gene_multiple_cnv = gene_multiple_cnv
        self.call_mode = call_mode
        self.seg_p_thr = seg_p_thr
        self.combine_p_thr = combine_p_thr
        self.sig_diff_stop = sig_diff_stop
        self.jump_thr = jump_thr
        self.diff_fold = diff_fold
        self.extreme_point_num = extreme_point_num
        self.break_point_num = break_point_num
        self.cnv_amps_thr = {'Loss': 2, 'Gain': 2}
        self.pcf_thr = 0.005
        self.cpf_weight = 0.005

    def run(self):
        samples_data = self.input_data.sort_values(by=['Path', 'Amp_Order']).reset_index(
            drop=True)

        samples_data[['Amp_States', 'Region_States', 'Region_Gene_Str', 'Stable_Region_States',
                      'Unstable_Region_States', 'Region_Ratio', 'Region_Ratio_Std', 'Region_States_Amps', 'Region_Amps',
                      'Region_States_Amp_ratio', 'Region_Loc', 'Region_Length', 'Region_Exon_Count',
                      'Region_Exon_Amp_Count', 'Adjacent_Ratio_Gap', 'Region_Type', 'Region_Index',
                      'Region_Index_Raw', 'Gene_Ratio', 'Gene_Ratio_Std']] = np.nan
        samples_data.rename(columns={self.cbs_depth_col: self.cbs_depth_col + '_Raw'}, inplace=True)
        samples_data[self.cbs_depth_col] = samples_data.groupby(['Sample_Name', 'Gene'])[
            self.cbs_depth_col + '_Raw'].apply(lambda x: self.outlier_smooth(x))
        call_depth_col = self.cbs_depth_col
        for amp_type in ['Stable', 'Unstable']:
            samples_data.loc[(samples_data['Amp_Type'] == amp_type) & (
                    samples_data[self.cbs_depth_col] > self.amp_thr[amp_type]['Gain']), ['Amp_States']] = 'Gain'
            if self.call_mode == 'Germline':
                samples_data.loc[(samples_data['Amp_Type'] == amp_type) & (
                        samples_data[self.cbs_depth_col] < self.amp_thr[amp_type]['Loss']), ['Amp_States']] = 'Loss'
        samples_data['Amp_States'].fillna('Negative', inplace=True)

        for sample_gene in samples_data[['Sample_Name', 'Gene']].drop_duplicates().values.tolist():
            sample_gene_data = samples_data[(samples_data['Sample_Name'] == sample_gene[0]) &
                                            (samples_data['Gene'] == sample_gene[1])]
            gene_ratio, gene_ratio_std = self.calc_mean_std(sample_gene_data[call_depth_col])
            samples_data.loc[sample_gene_data.index, 'Gene_Ratio'] = gene_ratio
            samples_data.loc[sample_gene_data.index, 'Gene_Ratio_Std'] = gene_ratio_std
            sample_gene_cbs_data = sample_gene_data.set_index('Amp_Order')[call_depth_col].copy()
            start_index = sample_gene_data.first_valid_index() - sample_gene_cbs_data.first_valid_index()
            sample_gene_depth_segments = list()
            min_index = sample_gene_cbs_data.first_valid_index()
            max_index = sample_gene_cbs_data.last_valid_index()
            cbs_depth_segments_values = self.rec_cbs(sample_gene_cbs_data, min_index, max_index,
                                                         sample_gene_depth_segments, sample_gene)
            merged_cbs_depth_segments = self.segment_merge(sample_gene_cbs_data, cbs_depth_segments_values, sample_gene)
            # if debug_list and (sample_gene in debug_list):
            #     print(sample_gene, cbs_depth_segments_values, merged_cbs_depth_segments)
            cbs_depth_segments = [(start_index + segment[0], start_index + segment[1]) for segment in
                                  merged_cbs_depth_segments]
            split_segments = list()
            for segment_index, cbs_depth_segment in enumerate(cbs_depth_segments):
                segment_start, segment_end = cbs_depth_segment
                segment_data = sample_gene_data.loc[segment_start:segment_end]
                segment_cnv_type, segment_stability, segment_type_amp_ratio, segment_ratio, segment_ratio_std = self.segment_cnv_check(
                    segment_data, call_depth_col)
                if self.call_mode == 'Germline':
                    segment_states = segment_cnv_type[segment_stability]
                    # Correct the erroneous boundary division.
                    if (segment_states in ['Gain', 'Loss']) or (
                            (segment_states == 'Negative') and (segment_type_amp_ratio < 0.5)):
                        segment_type_miss_amps = segment_data[segment_data['Amp_States'].isin(['Negative'])]
                        if not segment_type_miss_amps.empty:
                            amp_thr_failed = segment_type_miss_amps.index.values.tolist()
                            if len(amp_thr_failed) < len(segment_data) / 2:
                                amp_thr_failed_groups = int_continuous_split(amp_thr_failed, 1)
                                fix_board = False
                                for amp_thr_failed_group in amp_thr_failed_groups:
                                    if segment_start in amp_thr_failed_group:
                                        split_segments.append(
                                            (min(amp_thr_failed_group), max(amp_thr_failed_group), segment_index - 0.5))
                                        segment_start = max(amp_thr_failed_group) + 1
                                        fix_board = True
                                    if segment_end in amp_thr_failed_group:
                                        split_segments.append(
                                            (min(amp_thr_failed_group), max(amp_thr_failed_group), segment_index + 0.5))
                                        segment_end = min(amp_thr_failed_group) - 1
                                        fix_board = True
                                if fix_board:
                                    segment_data = sample_gene_data.loc[segment_start:segment_end]
                                    segment_cnv_type, segment_stability, segment_type_amp_ratio, segment_ratio, segment_ratio_std = self.segment_cnv_check(
                                        segment_data, call_depth_col)
                samples_data.loc[segment_data.index, 'Region_Index_Raw'] = segment_index
                samples_data.loc[segment_data.index, 'Region_Type'] = segment_stability
                samples_data.loc[segment_data.index, 'Region_Ratio'] = segment_ratio
                samples_data.loc[segment_data.index, 'Region_Ratio_Std'] = segment_ratio_std
                samples_data.loc[segment_data.index, 'Stable_Region_States'] = segment_cnv_type['Stable']
                samples_data.loc[segment_data.index, 'Unstable_Region_States'] = segment_cnv_type['Unstable']
                samples_data.loc[segment_data.index, 'Region_States'] = segment_cnv_type[segment_stability]

            # Renumber the incorrectly divided areas
            if split_segments:
                for split_segment in split_segments:
                    segment_start, segment_end, split_segment_index = split_segment
                    segment_data = sample_gene_data.loc[segment_start:segment_end]
                    segment_cnv_type, segment_stability, segment_type_amp_ratio, segment_ratio, segment_ratio_std = self.segment_cnv_check(
                        segment_data, call_depth_col)
                    samples_data.loc[segment_data.index, 'Region_Index_Raw'] = split_segment_index
                    samples_data.loc[segment_data.index, 'Region_Type'] = segment_stability
                    samples_data.loc[segment_data.index, 'Region_Ratio'] = segment_ratio
                    samples_data.loc[segment_data.index, 'Region_Ratio_Std'] = segment_ratio_std
                    samples_data.loc[segment_data.index, 'Stable_Region_States'] = segment_cnv_type['Stable']
                    samples_data.loc[segment_data.index, 'Unstable_Region_States'] = segment_cnv_type['Unstable']
                    samples_data.loc[segment_data.index, 'Region_States'] = segment_cnv_type[segment_stability]

            samples_data.loc[sample_gene_data.index, 'Region_Index'] = samples_data.loc[
                sample_gene_data.index, 'Region_Index_Raw']
            keep_segments = sorted(
                samples_data.loc[sample_gene_data.index]['Region_Index'].drop_duplicates().values.tolist())
            if keep_segments:
                keep_segments = {item: index for index, item in enumerate(keep_segments)}
                samples_data.loc[sample_gene_data.index, 'Region_Index'] = samples_data.loc[
                    sample_gene_data.index, 'Region_Index'].replace(keep_segments)

            # Merge consecutive intervals with the same positive CNV state
            sample_gene_data = samples_data[(samples_data['Sample_Name'] == sample_gene[0]) &
                                           (samples_data['Gene'] == sample_gene[1])]

            for cnv_type in ['Gain', 'Loss']:
                cnv_type_segments = sample_gene_data[sample_gene_data['Region_States'] == cnv_type][
                    'Region_Index'].drop_duplicates().values.tolist()
                if cnv_type_segments:
                    type_segments_groups = int_continuous_split(cnv_type_segments, 1)
                    for segments_group in type_segments_groups:
                        seg_index_min, segment_index_max = min(segments_group), max(segments_group)
                        segment_group_data = sample_gene_data[sample_gene_data['Region_Index'].between(
                            seg_index_min, segment_index_max, inclusive='both')]
                        group_cnv, group_stability, group_type_amp_ratio, group_ratio, group_ratio_std = self.segment_cnv_check(
                            segment_group_data, call_depth_col)
                        group_cnv_type = group_cnv[group_stability]
                        if group_cnv_type == cnv_type:
                            samples_data.loc[segment_group_data.index, 'Region_Index'] = min(segments_group)
                            samples_data.loc[segment_group_data.index, 'Region_Type'] = group_stability
                            samples_data.loc[segment_group_data.index, 'Region_Ratio'] = group_ratio
                            samples_data.loc[segment_group_data.index, 'Region_Ratio_Std'] = group_ratio_std
                            samples_data.loc[segment_group_data.index, 'Stable_Region_States'] = group_cnv[
                                'Stable']
                            samples_data.loc[segment_group_data.index, 'Unstable_Region_States'] = group_cnv[
                                'Unstable']
                            samples_data.loc[segment_group_data.index, 'Region_States'] = group_cnv_type

            keep_segments = sorted(
                samples_data.loc[sample_gene_data.index]['Region_Index'].drop_duplicates().values.tolist())
            if keep_segments:
                keep_segments = {item: index for index, item in enumerate(keep_segments)}
                samples_data.loc[sample_gene_data.index, 'Region_Index'] = samples_data.loc[
                    sample_gene_data.index, 'Region_Index'].replace(keep_segments)

            # Merge consecutive intervals with the same positive CNV state with gaps
            sample_gene_data = samples_data[(samples_data['Sample_Name'] == sample_gene[0]) &
                                           (samples_data['Gene'] == sample_gene[1])]

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
                            group_cnv, group_stability, group_type_amp_ratio, group_ratio, group_ratio_std = self.segment_cnv_check(
                                segment_group_data, call_depth_col)
                            other_type_data = segment_group_data[segment_group_data['Region_States'] != cnv_type]
                            other_type_ratio = other_type_data.shape[0] / segment_group_data.shape[0]
                            other_type_mean = other_type_data[call_depth_col].mean()
                            if (abs(other_type_mean - 1) > 0.8 * abs(
                                    self.region_thr[group_stability][cnv_type] - 1)) or (other_type_ratio < 0.33):
                                group_cnv_type = group_cnv[group_stability]
                                if group_cnv_type == cnv_type:
                                    samples_data.loc[segment_group_data.index, 'Region_Index'] = min(segments_group)
                                    samples_data.loc[segment_group_data.index, 'Region_Type'] = group_stability
                                    samples_data.loc[segment_group_data.index, 'Region_Ratio'] = group_ratio
                                    samples_data.loc[segment_group_data.index, 'Region_Ratio_Std'] = group_ratio_std
                                    samples_data.loc[segment_group_data.index, 'Stable_Region_States'] = group_cnv[
                                        'Stable']
                                    samples_data.loc[segment_group_data.index, 'Unstable_Region_States'] = group_cnv[
                                        'Unstable']
                                    samples_data.loc[segment_group_data.index, 'Region_States'] = group_cnv_type

            keep_segments = sorted(
                samples_data.loc[sample_gene_data.index]['Region_Index'].drop_duplicates().values.tolist())
            if keep_segments:
                keep_segments = {item: index for index, item in enumerate(keep_segments)}
                samples_data.loc[sample_gene_data.index, 'Region_Index'] = samples_data.loc[
                    sample_gene_data.index, 'Region_Index'].replace(keep_segments)

            # Merge the adjacent segments with weak positive CNV State
            sample_gene_data = samples_data[(samples_data['Sample_Name'] == sample_gene[0]) &
                                           (samples_data['Gene'] == sample_gene[1])]

            for cnv_type in ['Gain', 'Loss']:
                cnv_type_segments = sample_gene_data[sample_gene_data['Region_States'] == cnv_type][
                    'Region_Index'].drop_duplicates().values.tolist()
                if cnv_type_segments:
                    for cnv_type_segment in cnv_type_segments:
                        flank_merge = [cnv_type_segment]
                        for flank_index in [cnv_type_segment - 1, cnv_type_segment + 1]:
                            flank_negative_data = sample_gene_data[
                                (sample_gene_data['Region_Index'] == flank_index) &
                                (sample_gene_data['Region_States'] == 'Negative')]
                            flank_negative_amps = flank_negative_data.shape[0]
                            if flank_negative_amps > 1:
                                flank_ratio = flank_negative_data['Region_Ratio'].values[0]
                                flank_stability = flank_negative_data['Region_Type'].values[0]
                                flank_cnv_amps = flank_negative_data[flank_negative_data['Amp_States'] == cnv_type].shape[0]
                                flank_amp_ratio = flank_cnv_amps / flank_negative_amps
                                if (abs(flank_ratio - 1) > 0.8 * abs(
                                        self.region_thr[flank_stability][cnv_type] - 1)) and (flank_amp_ratio > 0.5):
                                    flank_merge.append(flank_index)
                        if len(flank_merge) > 1:
                            merge_data = sample_gene_data[sample_gene_data['Region_Index'].isin(flank_merge)]
                            group_cnv, group_stability, group_type_amp_ratio, group_ratio, group_ratio_std = self.segment_cnv_check(
                                merge_data, call_depth_col)
                            group_cnv_type = group_cnv[group_stability]
                            if group_cnv_type == cnv_type:
                                samples_data.loc[merge_data.index, 'Region_Index'] = cnv_type_segment
                                samples_data.loc[merge_data.index, 'Region_Type'] = group_stability
                                samples_data.loc[merge_data.index, 'Region_Ratio'] = group_ratio
                                samples_data.loc[merge_data.index, 'Region_Ratio_Std'] = group_ratio_std
                                samples_data.loc[merge_data.index, 'Stable_Region_States'] = group_cnv[
                                    'Stable']
                                samples_data.loc[merge_data.index, 'Unstable_Region_States'] = group_cnv[
                                    'Unstable']
                                samples_data.loc[merge_data.index, 'Region_States'] = group_cnv_type
                                sample_gene_data = samples_data[(samples_data['Sample_Name'] == sample_gene[0]) &
                                                               (samples_data['Gene'] == sample_gene[1])]

            keep_segments = sorted(
                samples_data.loc[sample_gene_data.index]['Region_Index'].drop_duplicates().values.tolist())
            if keep_segments:
                keep_segments = {item: index for index, item in enumerate(keep_segments)}
                samples_data.loc[sample_gene_data.index, 'Region_Index'] = samples_data.loc[
                    sample_gene_data.index, 'Region_Index'].replace(keep_segments)

            # Merging two neighboring CNV-positive segments with a shorter negative segment in the middle
            sample_gene_data = samples_data[(samples_data['Sample_Name'] == sample_gene[0]) &
                                           (samples_data['Gene'] == sample_gene[1])]

            for cnv_type in ['Gain', 'Loss']:
                cnv_type_segments = sample_gene_data[sample_gene_data['Region_States'] == cnv_type][
                    'Region_Index'].drop_duplicates().values.tolist()
                if cnv_type_segments:
                    for cnv_type_segment in cnv_type_segments:
                        segment_data = sample_gene_data[sample_gene_data['Region_Index'] == cnv_type_segment]
                        region_amps = segment_data.shape[0]
                        if region_amps >= self.cnv_amps_thr[cnv_type]:
                            region_cnv_amps = segment_data[segment_data['Amp_States'] == cnv_type].shape[0]
                            samples_data.loc[segment_data.index, 'Region_States_Amps'] = region_cnv_amps
                            samples_data.loc[segment_data.index, 'Region_Amps'] = region_amps
                            type_amp_ratio = region_cnv_amps / region_amps
                            samples_data.loc[segment_data.index, 'Region_States_Amp_ratio'] = type_amp_ratio

                            max_exon_num = segment_data['Exon_Str_End'].max()
                            min_exon_num = segment_data['Exon_Str_Start'].min()
                            if max_exon_num == min_exon_num:
                                exon_region_str = str(min_exon_num)
                            else:
                                exon_region_str = str(min_exon_num) + '-' + str(max_exon_num)
                            cnv_exon_count = max_exon_num - min_exon_num + 1

                            if self.call_mode == 'Germline':
                                cnv_region_str = sample_gene[1] + '_E' + exon_region_str
                                next_exon_order = segment_data['Exon_Order'].max() + 1
                                prev_exon_order = segment_data['Exon_Order'].min() - 1
                                region_adjacent_data = sample_gene_data.loc[sample_gene_data['Exon_Order'].between(
                                    prev_exon_order, next_exon_order, inclusive='both') & ~sample_gene_data.index.isin(
                                    segment_data.index)]
                                samples_data.loc[segment_data.index, 'Region_Exon_Count'] = cnv_exon_count
                                cnv_exon_amp_count = sample_gene_data[sample_gene_data['Exon_Order'].between(
                                    prev_exon_order + 1, next_exon_order - 1, inclusive='both')].shape[0]
                                samples_data.loc[segment_data.index, 'Region_Exon_Amp_Count'] = cnv_exon_amp_count
                            else:
                                cnv_region_str = sample_gene[1]
                                region_adjacent_data = pd.DataFrame()
                            samples_data.loc[segment_data.index, 'Region_Gene_Str'] = cnv_region_str
                            region_start_loc = segment_data['Scan_Start'].min()
                            region_end_loc = segment_data['Scan_End'].max()
                            region_length = region_end_loc - region_start_loc
                            amp_region = str(region_start_loc) + '-' + str(region_end_loc)
                            chr_str = '-'.join(segment_data['Chr'].drop_duplicates().values.tolist())
                            samples_data.loc[segment_data.index, 'Region_Loc'] = chr_str + ':' + amp_region
                            samples_data.loc[segment_data.index, 'Region_Length'] = region_length
                            segment_region_ratio = segment_data[call_depth_col].mean()
                            if not region_adjacent_data.empty:
                                adjacent_ratio_gap = abs(
                                    region_adjacent_data[call_depth_col].mean() - segment_region_ratio)
                            else:
                                adjacent_ratio_gap = 0.5
                            samples_data.loc[segment_data.index, 'Adjacent_Ratio_Gap'] = adjacent_ratio_gap
                        else:
                            samples_data.loc[segment_data.index, 'Region_States'] = 'Negative'

            keep_segments = sorted(
                samples_data.loc[sample_gene_data.index]['Region_Index'].drop_duplicates().values.tolist())
            if keep_segments:
                keep_segments = {item: index for index, item in enumerate(keep_segments)}
                samples_data.loc[sample_gene_data.index, 'Region_Index'] = samples_data.loc[
                    sample_gene_data.index, 'Region_Index'].replace(keep_segments)

            # Merge adjacent CNV-negative segments
            sample_gene_data = samples_data[(samples_data['Sample_Name'] == sample_gene[0]) &
                                           (samples_data['Gene'] == sample_gene[1])]

            negative_segments = sample_gene_data[sample_gene_data['Region_States'] == 'Negative'][
                'Region_Index'].drop_duplicates().values.tolist()
            if negative_segments:
                neg_segments_groups = int_continuous_split(negative_segments, 1)
                for neg_segments_group in neg_segments_groups:
                    if len(neg_segments_group) > 1:
                        seg_index_min, segment_index_max = min(neg_segments_group), max(neg_segments_group)
                        segment_group_data = sample_gene_data[sample_gene_data['Region_Index'].between(
                            seg_index_min, segment_index_max, inclusive='both')]
                        group_cnv, group_stability, group_type_amp_ratio, group_ratio, group_ratio_std = self.segment_cnv_check(
                            segment_group_data, call_depth_col)
                        samples_data.loc[segment_group_data.index, 'Region_Index'] = min(neg_segments_group)
                        samples_data.loc[segment_group_data.index, 'Region_Type'] = group_stability
                        samples_data.loc[segment_group_data.index, 'Region_Ratio'] = group_ratio
                        samples_data.loc[segment_group_data.index, 'Region_Ratio_Std'] = group_ratio_std
                        samples_data.loc[segment_group_data.index, 'Stable_Region_States'] = 'Negative'
                        samples_data.loc[segment_group_data.index, 'Unstable_Region_States'] = 'Negative'
            keep_segments = sorted(
                samples_data.loc[sample_gene_data.index]['Region_Index'].drop_duplicates().values.tolist())
            if keep_segments:
                keep_segments = {item: index for index, item in enumerate(keep_segments)}
                samples_data.loc[sample_gene_data.index, 'Region_Index'] = samples_data.loc[
                    sample_gene_data.index, 'Region_Index'].replace(keep_segments)
        return samples_data

    @staticmethod
    def outlier_smooth(input_data, window_size=3):

        smooth_data = input_data.copy()
        data_std = input_data.std()
        replace_value = dict()
        for i in smooth_data.index:
            obs_value = smooth_data.loc[i]
            obs_data = smooth_data.loc[i-window_size:i+window_size]
            obs_max = obs_data.max()
            obs_min = obs_data.min()
            if obs_value == obs_max:
                second_max_value = obs_data.nlargest(2).iloc[1]
                closest_value = obs_value - second_max_value
                if closest_value > 3 * data_std:
                    replace_value[i] = obs_data.median() + 2 * data_std
            elif obs_value == obs_min:
                second_min_value = obs_data.nsmallest(2).iloc[1]
                closest_value = second_min_value - obs_value
                if closest_value > 3 * data_std:
                    replace_value[i] = obs_data.median() - 2 * data_std
        if replace_value:
            for index, value in replace_value.items():
                smooth_data.loc[index] = value
        return smooth_data

    @staticmethod
    def calc_mean_std(calc_data):
        data_len = len(calc_data)
        if data_len > 1:
            return calc_data.mean(), calc_data.std(ddof=0)
        else:
            return calc_data.values[0], 0.001

    def rec_cbs(self, cbs_data, start, end, depth_segments=None, sample_gene=''):
        if depth_segments is None:
            depth_segments = list()
        if end - start < 4:
            depth_segments.append((start, end))
        else:
            break_points = self.cbs_segment(cbs_data, start, end, sample_gene)
            if not break_points:
                depth_segments.append((start, end))
            else:
                self.rec_cbs(cbs_data, start, break_points[0] - 1, depth_segments, sample_gene)
                if len(break_points) == 2:
                    self.rec_cbs(cbs_data, break_points[0], break_points[1] - 1, depth_segments, sample_gene)
                self.rec_cbs(cbs_data, break_points[-1], end, depth_segments, sample_gene)
        return depth_segments

    def cbs_segment(self, cbs_data, start, end, sample_gene=''):
        select_data = cbs_data.loc[start:end]
        trans_data = select_data.to_frame(name='Values')
        trans_data['Values_Offset'] = trans_data['Values'] - trans_data['Values'].mean()
        trans_data['Offset_Sum'] = trans_data['Values_Offset'].cumsum().shift(1)
        trans_data['Offset_Diff1'] = trans_data['Values_Offset'].diff()
        trans_data['Offset_Diff2'] = trans_data['Values_Offset'].diff(periods=2).shift(-1)
        trans_data[['Offset_Diff_Min', 'Offset_Diff_Max']] = trans_data[['Offset_Diff1', 'Offset_Diff2']].agg(
            lambda x: diff_extremal(x), axis=1, result_type='expand')
        trans_data['Offset_Diff_Mean'] = trans_data[['Offset_Diff1', 'Offset_Diff2']].agg(
            lambda x: x.mean(), axis=1)
        trans_data['Offset_Sum_Fix'] = trans_data['Offset_Sum'] - self.diff_fold * trans_data['Offset_Diff_Mean']

        up_candidate_pos = trans_data[(trans_data['Offset_Diff_Min'] > self.jump_thr) |
                                      (trans_data['Offset_Diff_Max'] > 1.45 * self.jump_thr) |
                                      (trans_data['Offset_Diff_Mean'] > 1.1 * self.jump_thr)].copy()

        if up_candidate_pos.empty:
            up_candidate_pos = trans_data[(trans_data['Offset_Diff_Max'] > self.jump_thr) |
                                          (trans_data['Offset_Diff_Mean'] > 0.8 * self.jump_thr)].copy()
            up_candidate_pos['Up_Score'] = 1
        else:
            up_candidate_pos['Up_Score'] = 2
        if not up_candidate_pos.empty:
            up_candidate_pos_index = up_candidate_pos['Offset_Sum_Fix'].nsmallest(
                self.extreme_point_num).index.tolist()
            up_expanded_index = np.unique(np.clip(np.concatenate([
                    np.arange(i - 2, i + 2 + 1) for i in up_candidate_pos_index]), start, end))
            expand_up_candidate_pos_index = up_candidate_pos[
                up_candidate_pos.index.isin(up_expanded_index)].index.tolist()

            jump_thr_candidates = up_candidate_pos.loc[expand_up_candidate_pos_index]['Up_Score'].to_dict()
        else:
            jump_thr_candidates = dict()

        # Find jump downwards breakpoints candidates
        down_candidate_pos = trans_data[(trans_data['Offset_Diff_Min'] < -1 * self.jump_thr) |
                                        (trans_data['Offset_Diff_Max'] < -1.45 * self.jump_thr) |
                                        (trans_data['Offset_Diff_Mean'] < -1.1 * self.jump_thr)].copy()
        if down_candidate_pos.empty:
            down_candidate_pos = trans_data[(trans_data['Offset_Diff_Max'] < -1 * self.jump_thr) |
                                            (trans_data['Offset_Diff_Mean'] < -0.8 * self.jump_thr)].copy()
            down_candidate_pos['Down_Score'] = 1
        else:
            down_candidate_pos['Down_Score'] = 2
        if not down_candidate_pos.empty:
            down_candidate_pos_index = down_candidate_pos['Offset_Sum_Fix'].nlargest(
                self.extreme_point_num).index.tolist()
            down_expanded_index = np.unique(np.clip(np.concatenate([
                    np.arange(i - 2, i + 2 + 1) for i in down_candidate_pos_index]), start, end))
            expand_down_candidate_pos_index = down_candidate_pos[
                down_candidate_pos.index.isin(down_expanded_index)].index.tolist()

            jump_thr_candidates.update(down_candidate_pos.loc[expand_down_candidate_pos_index]['Down_Score'].to_dict())

        if jump_thr_candidates:
            breakpoint_p = dict()
            # breakpoint_var = dict()
            candidate_bps = sorted(jump_thr_candidates.keys())
            candidate_groups = list(itertools.combinations(candidate_bps, 1))
            candidate_groups.extend(list(itertools.combinations(candidate_bps, 2)))
            for bp_group in candidate_groups:
                group_score = 0
                for bp in bp_group:
                    group_score += jump_thr_candidates[bp]
                if (group_score > len(bp_group)) and (bp_group[0] - start >= 2) and (end - bp_group[-1] >= 2):
                    if len(bp_group) == 2:
                        xt = trans_data['Values'].loc[bp_group[0]:bp_group[1] - 1]
                        xn = trans_data['Values'].drop(xt.index)
                    else:
                        xt = trans_data['Values'].loc[start:bp_group[0] - 1]
                        xn = trans_data['Values'].loc[bp_group[0]:end]
                    if (len(xt) > 1) and (len(xn) > 1):
                        test_stat, test_p = mannwhitneyu(xt, xn)
                        breakpoint_p[bp_group] = test_p

            if breakpoint_p:
                best_bp_group = min(breakpoint_p, key=breakpoint_p.get)
                best_p = breakpoint_p[best_bp_group]
                if best_p < self.seg_p_thr:
                    return best_bp_group
                else:
                    return ()
            else:
                return ()
        else:
            return ()

    # Determine the CNV state of the segment based on the average probe depth and threshold
    def segment_cnv_check(self, segment_data, call_depth_col):
        region_amps = segment_data.shape[0]
        segment_region_ratio, segment_region_ratio_std = self.calc_mean_std(segment_data[call_depth_col])
        region_unstable_amps = segment_data[segment_data['Amp_Type'] == 'Unstable'].shape[0]
        if region_unstable_amps / region_amps > self.unstable_amp_ratio:
            region_stability = 'Unstable'
        else:
            region_stability = 'Stable'
        stability_cnv_type = dict()
        for stability_type in ['Stable', 'Unstable']:
            if segment_region_ratio < self.region_thr[stability_type]['Loss']:
                cnv_type = 'Loss'
                type_amp_ratio = len(segment_data[segment_data[call_depth_col] <
                                                  self.region_thr[stability_type]['Loss']]) / region_amps
            elif segment_region_ratio > self.region_thr[stability_type]['Gain']:
                cnv_type = 'Gain'
                type_amp_ratio = len(segment_data[segment_data[call_depth_col] >
                                                  self.region_thr[stability_type]['Gain']]) / region_amps
            else:
                cnv_type = 'Negative'
                type_amp_ratio = len(segment_data[segment_data[call_depth_col].between(
                    self.region_thr[stability_type]['Loss'], self.region_thr[stability_type]['Gain'],
                    inclusive='neither')]) / region_amps
            stability_cnv_type[stability_type] = cnv_type
        return stability_cnv_type, region_stability, type_amp_ratio, segment_region_ratio, segment_region_ratio_std

    def segment_merge(self, depth_data, depth_segments, sample_gene):
        if len(depth_segments) > 3:
            start_index = depth_data.first_valid_index()
            end_index = depth_data.last_valid_index()
            data_len = len(depth_data)
            bp_ends = [bp[1] for bp in depth_segments[:-1]]
            bp_ends = sorted(set(filter(lambda x: start_index < x < end_index, bp_ends)))
            bp_pair_var = {}
            cycle_pass = 2
            select_bps = (start_index - 1, end_index)
            while cycle_pass:
                group_var = dict()
                group_test_p = dict()
                candidate_bps = [item for item in bp_ends if item not in select_bps]
                candidate_groups = list(itertools.combinations(candidate_bps, 1))
                two_candidate_groups = list(itertools.combinations(candidate_bps, 2))
                two_candidate_groups = [item for item in two_candidate_groups if not list(filter(
                    lambda x: item[0]< x < item[1], select_bps))]
                candidate_groups.extend(two_candidate_groups)
                for bp_group in candidate_groups:
                    right_end_bps = list(filter(lambda x: x > max(bp_group), select_bps))
                    left_end_bps = list(filter(lambda x: x < min(bp_group), select_bps))
                    if left_end_bps and right_end_bps:
                        right_end = min(right_end_bps)
                        left_end = max(left_end_bps)
                        if right_end - left_end >= 4:
                            if len(bp_group) == 1:
                                bp = bp_group[0]
                                xt = depth_data.loc[left_end + 1:bp]
                                xn = depth_data.loc[bp + 1:right_end]
                            else:
                                xt = depth_data.loc[bp_group[0] + 1:bp_group[1]]
                                xf = depth_data.loc[left_end + 1:right_end]
                                xn = xf.drop(xt.index)
                            if (len(xt) > 1) and (len(xn) > 1):
                                test_stat, test_p = mannwhitneyu(xt, xn)
                                bp_group += select_bps
                                bp_group = tuple(sorted(bp_group))
                                # print(bp_group)
                                bp_group_var = 0
                                for index, bp in enumerate(bp_group[: -1]):
                                    if (bp + 1, bp_group[index + 1]) not in bp_pair_var:
                                        bp_pair_var[(bp + 1, bp_group[index + 1])] = self.cpf_var(
                                            depth_data.loc[bp + 1:bp_group[index + 1]])
                                    bp_group_var += bp_pair_var[(bp + 1, bp_group[index + 1])]
                                group_var[bp_group] = bp_group_var / data_len
                                group_test_p[bp_group] = test_p
                cycle_pass = 0
                if group_var:
                    best_group = min(group_var, key=group_var.get)
                    best_group_p = group_test_p[best_group]
                    if best_group_p < self.seg_p_thr:
                        select_bps = best_group
                        cycle_pass = 2
            if cycle_pass:
                merge_depth_segments = [(select_bps[i] + 1, select_bps[i + 1]) for i in
                                        range(len(select_bps) - 1)]
            else:
                merge_depth_segments = depth_segments
        else:
            merge_depth_segments = depth_segments
        return merge_depth_segments

    @staticmethod
    def cpf_var(input_data):
        mean = input_data.mean()
        return input_data.sub(mean).pow(2).sum()


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


def diff_extremal(row):
    if row.isnull().values.any():
        return [np.nan, np.nan]
    else:
        min_row_idx = row.abs().idxmin()
        max_row_idx = row.abs().idxmax()
        # print(row, min_row_idx)
        return [row[min_row_idx], row[max_row_idx]]


if __name__ == '__main__':
    region_thr = {"Loss": 0.64, "Gain": 1.34}
    amp_thr = {"Loss": 0.64, "Gain": 1.34}
    cnv_amps_thr = {"Loss": 2, "Gain": 2}
    pos_data = pd.read_csv('./amplicon_depth_read_data.tsv', sep='\t', low_memory=False)
    t1 = time.time()
    call_col = 'Ref_Norm_Depth'
    cbs_data = FastCBS(pos_data, call_col, amp_thr, region_thr).run()
    t2 = time.time()
    cost_time = t2 - t1
    print('Run time: {}s'.format(cost_time))
