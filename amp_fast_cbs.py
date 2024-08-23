# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import time
import itertools
from scipy.stats import ttest_ind, ttest_1samp, levene

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
        samples_data = self.input_data.copy()
        samples_data[['Amp_States', 'Region_States', 'Region_Gene_Str', 'Stable_Region_States',
                      'Unstable_Region_States', 'Region_Ratio', 'Region_Ratio_Std', 'Region_States_Amps', 'Region_Amps',
                      'Region_States_Amp_ratio', 'Region_Loc', 'Region_Length', 'Region_Exon_Count',
                      'Adjacent_Ratio_Gap', 'Region_Type', 'Region_Index', 'Region_Index_Raw', 'Gene_Ratio',
                      'Gene_Ratio_Std']] = np.nan
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
            # print(self.input_data, self.min_index, self.max_index)
            cbs_depth_segments_values = self.rec_segment(sample_gene_cbs_data, min_index, max_index,
                                                         sample_gene_depth_segments, sample_gene)
            merged_cbs_depth_segments = self.cpf_merge(sample_gene_cbs_data, cbs_depth_segments_values, sample_gene)
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
                    if (segment_states == 'Negative') and (segment_type_amp_ratio < 0.5):
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
                        if (segment_index_max - seg_index_min + 1) > len(segments_group):
                            group_cnv, group_stability, group_type_amp_ratio, group_ratio, group_ratio_std = self.segment_cnv_check(
                                segment_group_data, call_depth_col)
                            other_type_mean = sample_gene_data[sample_gene_data['Region_States'] != cnv_type][
                                call_depth_col].mean()
                            if (other_type_mean - 1) > 0.8 * (self.region_thr[group_stability][cnv_type] - 1):
                                group_cnv_type = group_cnv[group_stability]
                            else:
                                group_cnv_type = 'Negative'
                        else:
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
                            other_type_mean = sample_gene_data[sample_gene_data['Region_States'] != cnv_type][
                                call_depth_col].mean()
                            if (other_type_mean - 1) > 0.8 * (self.region_thr[group_stability][cnv_type] - 1):
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
                        for flank_index in [cnv_type_segment - 1, cnv_type_segment + 1]:
                            flank_negative_data = sample_gene_data[
                                (sample_gene_data['Region_Index'] == flank_index) &
                                (sample_gene_data['Region_States'] == 'Negative')]
                            if not flank_negative_data.empty:
                                flank_ratio = flank_negative_data['Region_Ratio'].values[0]
                                flank_stability = flank_negative_data['Region_Type'].values[0]
                                if (flank_ratio - 1) > 0.8 * (self.region_thr[flank_stability][cnv_type] - 1):
                                    merge_data = sample_gene_data[sample_gene_data['Region_Index'].isin(
                                        [cnv_type_segment, flank_index])]
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
                                region_adjacent_data = sample_gene_data.loc[
                                    sample_gene_data['Exon_Order'].between(prev_exon_order, next_exon_order,
                                                                           inclusive='both') & ~sample_gene_data.index.isin(
                                        segment_data.index)]
                                samples_data.loc[segment_data.index, 'Region_Exon_Count'] = cnv_exon_count
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

    def rec_segment(self, cbs_data, start, end, depth_segments=None, sample_gene=''):
        if depth_segments is None:
            depth_segments = list()
        if end - start < 4:
            depth_segments.append((start, end))
        else:
            s, e, break_point = self.cbs(cbs_data, start, end, sample_gene)
            if (not break_point) or (e - s == end - start):
                depth_segments.append((start, end))
            else:
                if s - 1 >= start:
                    self.rec_segment(cbs_data, start, s - 1, depth_segments, sample_gene)
                if e > s:
                    self.rec_segment(cbs_data, s, e, depth_segments, sample_gene)
                if e + 1 <= end:
                    self.rec_segment(cbs_data, e + 1, end, depth_segments, sample_gene)
        return depth_segments

    # call segment by jump point and t test
    def cbs(self, cbs_data, start, end, sample_gene=''):
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

        jump_thr_candidates = dict()

        # Find jump upwards breakpoints candidates
        up_candidate_pos = trans_data[(trans_data['Offset_Diff_Min'] > self.jump_thr) |
                                      (trans_data['Offset_Diff_Max'] > 1.5 * self.jump_thr)].copy()
        if up_candidate_pos.empty:
            up_candidate_pos = trans_data[(trans_data['Offset_Diff_Max'] > self.jump_thr)].copy()
            up_candidate_pos['Up_Score'] = 1
        else:
            up_candidate_pos['Up_Score'] = 2
        if up_candidate_pos.empty:
            jump_thr_candidates['Up'] = dict()
        else:
            up_candidate_pos_index = up_candidate_pos['Offset_Sum_Fix'].nsmallest(
                self.extreme_point_num).index.tolist()
            jump_thr_candidates['Up'] = up_candidate_pos.loc[up_candidate_pos_index]['Up_Score'].to_dict()

        # Find jump downwards breakpoints candidates
        down_candidate_pos = trans_data[(trans_data['Offset_Diff_Min'] < -1 * self.jump_thr) |
                                        (trans_data['Offset_Diff_Max'] < -1.5 * self.jump_thr)].copy()
        if down_candidate_pos.empty:
            down_candidate_pos = trans_data[(trans_data['Offset_Diff_Max'] < -1 * self.jump_thr)].copy()
            down_candidate_pos['Down_Score'] = 1
        else:
            down_candidate_pos['Down_Score'] = 2
        if down_candidate_pos.empty:
            jump_thr_candidates['Down'] = dict()
        else:
            down_candidate_po_index = down_candidate_pos['Offset_Sum_Fix'].nlargest(
                self.extreme_point_num).index.tolist()
            jump_thr_candidates['Down'] = down_candidate_pos.loc[down_candidate_po_index]['Down_Score'].to_dict()

        pair_type = {'Up': 'Down', 'Down': 'Up'}
        breakpoint_candidates = list()
        for break_type in jump_thr_candidates:
            if jump_thr_candidates[break_type]:
                for break_end in jump_thr_candidates[break_type]:
                    break_score = jump_thr_candidates[break_type][break_end]
                    if jump_thr_candidates[pair_type[break_type]]:
                        for pair_end in jump_thr_candidates[pair_type[break_type]]:
                            pair_score = jump_thr_candidates[pair_type[break_type]][pair_end]
                            break_pair_score = break_score + pair_score
                            if break_pair_score > 2:
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
                best_p = min(breakpoint_p.values())
                best_bp_s, best_bp_e = min(breakpoint_p, key=breakpoint_p.get)
                if best_p < self.seg_p_thr:
                    statistics_check = True
                else:
                    statistics_check = False
                return best_bp_s, best_bp_e, statistics_check
            else:
                return start, end, False
        else:
            return start, end, False

    # Calculate the variance of the segment
    @staticmethod
    def t_test(a_data, b_data, sd_thr=0.05):
        if len(a_data) > 1 and len(b_data) > 1:
            equal_val_stat, equal_val_p = levene(a_data, b_data)
            if equal_val_p < sd_thr:
                equal_val_bool = False
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

    # Merge the adjacent segments with the same CNV state
    def cpf_merge(self, depth_data, depth_segments, sample_gene):
        if len(depth_segments) > 3:
            start_index = depth_data.first_valid_index()
            end_index = depth_data.last_valid_index()
            data_len = len(depth_data)
            bp_ends = [bp[1] for bp in depth_segments[:-1]]
            bp_pair_var = {}
            best_var = self.cpf_var(depth_data) / data_len + self.cpf_weight
            best_var_group = []
            best_p = 1
            best_p_var = 100
            best_p_group = []
            cycle_pass = 2
            max_bp = min(self.break_point_num + 2, len(bp_ends) + 1)
            for r in range(1, max_bp):
                if cycle_pass:
                    cycle_pass -= 1
                    cycle_groups = list(itertools.combinations(bp_ends, r))
                    for bp_group in cycle_groups:
                        bp_group = list(bp_group)
                        bp_group.insert(0, start_index - 1)
                        bp_group.append(end_index)
                        cpf_var = 0
                        for index, bp in enumerate(bp_group[: -1]):
                            if (bp + 1, bp_group[index + 1]) not in bp_pair_var:
                                bp_pair_var[(bp + 1, bp_group[index + 1])] = self.cpf_var(
                                    depth_data.loc[bp + 1:bp_group[index + 1]])
                            cpf_var += bp_pair_var[(bp + 1, bp_group[index + 1])]
                        cpf_var = cpf_var / data_len + (r + 1) * self.cpf_weight

                        if len(bp_group) in [3, 4] and (bp_group[2] - bp_group[1] < 10):
                            xt = depth_data.loc[bp_group[1] + 1:bp_group[2]]
                            xn = depth_data.drop(xt.index)
                            if (len(xt) > 1) and (len(xn) > 1):
                                test_stat, test_p = self.t_test(xt, xn)
                                if test_p < best_p and (test_p < self.seg_p_thr) and (cpf_var < best_p_var):
                                    best_p = test_p
                                    best_p_var = cpf_var
                                    best_p_group = bp_group
                        if cpf_var < best_var:
                            cycle_pass = 2
                            best_var = cpf_var
                            best_var_group = bp_group
            if best_var_group:
                merge_depth_segments = [(best_var_group[i] + 1, best_var_group[i + 1]) for i in
                                        range(len(best_var_group) - 1)]
            else:
                if best_p_group:
                    merge_depth_segments = [(best_p_group[i] + 1, best_p_group[i + 1]) for i in
                                            range(len(best_p_group) - 1)]
                else:
                    merge_depth_segments = [(start_index, end_index)]
        else:
            merge_depth_segments = depth_segments
        return merge_depth_segments

    # Calculate the sum of the square of the deviation of the data
    @staticmethod
    def cpf_var(input_data):
        mean = input_data.mean()
        return input_data.sub(mean).pow(2).sum()
        # return input_data.pow(2).sum() / len(input_data)


# Split the list of integers into continuous groups
def int_continuous_split(int_list, max_gap_value):
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


# Find the minimum and maximum values in the list
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
    cbs_data = cbs_call(pos_data, call_col, amp_thr, region_thr, cnv_amps_thr)
    t2 = time.time()
    cost_time = t2 - t1
    print('Run time: {}s'.format(cost_time))
