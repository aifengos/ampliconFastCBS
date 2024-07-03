# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import time
from scipy.stats import ttest_ind, ttest_1samp, levene

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


class FastCBS:
    def __init__(self, input_data, seg_p_thr=0.05, combine_p_thr=0.01, sig_diff_stop=True, jump_thr=0.3, diff_fold=3,
                 extreme_number=3):
        self.input_data = input_data
        self.seg_p_thr = seg_p_thr
        self.combine_p_thr = combine_p_thr
        self.sig_diff_stop = sig_diff_stop
        self.jump_thr = jump_thr
        self.diff_fold = diff_fold
        self.extreme_number = extreme_number
        self.min_index = min(self.input_data.index)
        self.max_index = max(self.input_data.index)

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
                self.extreme_number).index.tolist()
        else:
            up_candidate_max_pos = trans_data[trans_data['Offset_Diff_Max'] > self.jump_thr]
            if not up_candidate_max_pos.empty:
                jump_thr_candidates['Up'] = up_candidate_max_pos['Offset_Sum_Fix'].nsmallest(
                    self.extreme_number).index.tolist()
            else:
                jump_thr_candidates['Up'] = list()

        down_candidate_min_pos = trans_data[trans_data['Offset_Diff_Min'] < -1 * self.jump_thr]
        if not down_candidate_min_pos.empty:
            jump_thr_candidates['Down'] = down_candidate_min_pos['Offset_Sum_Fix'].nlargest(
                self.extreme_number).index.tolist()
        else:
            down_candidate_max_pos = trans_data[trans_data['Offset_Diff_Max'] < -1 * self.jump_thr]
            if not down_candidate_max_pos.empty:
                jump_thr_candidates['Down'] = down_candidate_max_pos['Offset_Sum_Fix'].nlargest(
                    self.extreme_number).index.tolist()
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
                # if candidate_bp_s - start < 2:
                #     candidate_bp_s = start
                # if end - se_candidate < 2:
                #     se_candidate = end
                # print(candidate_bp_s, candidate_bp_e, candidate_bp_type)

                xt = trans_data['Values_Offset'].loc[candidate_bp_s:candidate_bp_e]
                xn = trans_data['Values_Offset'].drop(xt.index)
                if (len(xt) > 1) and (len(xn) > 1):
                    test_stat, test_p = self.t_test(xt, xn)
                    breakpoint_p[(candidate_bp_s, candidate_bp_e)] = test_p

            if breakpoint_p:
                print(breakpoint_p)
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

    def combine_segment(self, depth_data, depth_segments):
        if len(depth_segments) > 3:
            depth_segments = sorted(depth_segments, key=lambda x: x[0])
            depth_segment_p = dict()
            for depth_segment in depth_segments:
                xt = depth_data.loc[depth_segment[0]:depth_segment[1]]
                xn = depth_data.drop(xt.index)
                test_stat, test_p = self.t_test(xt, xn)
                depth_segment_p[depth_segment] = test_p
            merged_count = 2
            cnv_segments = list()
            while merged_count:
                min_p_segment = min(depth_segment_p, key=depth_segment_p.get)
                min_p_segment_p = depth_segment_p[min_p_segment]
                del depth_segment_p[min_p_segment]
                depth_segments.remove(min_p_segment)
                print(merged_count, min_p_segment, depth_segments)
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
                    merged_count -= 1
                else:
                    merged_count = 0
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


if __name__ == '__main__':
    region_thr = {"Loss": 0.64, "Gain": 1.34}
    pos_data = pd.read_csv('./amplicon_depth_read_data.tsv', sep='\t', low_memory=False)
    t1 = time.time()
    for sample_gene in pos_data[['Sample_Name', 'Gene']].drop_duplicates().values.tolist():
        sample_gene_data = pos_data[(pos_data['Sample_Name'] == sample_gene[0]) &
                                    (pos_data['Gene'] == sample_gene[1])]
        for depth_type in ['Ref_Norm_Depth']:
            depth_segments = FastCBS(sample_gene_data[depth_type]).run()
            if len(depth_segments) > 1:
                for segment_index, depth_segment in enumerate(depth_segments):
                    pos_data.loc[depth_segment[0]:depth_segment[1], depth_type + '_Segment'] = segment_index
                    segment_depth_mean = pos_data.loc[depth_segment[0]:depth_segment[1], depth_type].mean()
                    pos_data.loc[depth_segment[0]:depth_segment[1], depth_type + '_Segment_Mean'] = segment_depth_mean
                    if segment_depth_mean < region_thr['Loss']:
                        pos_data.loc[depth_segment[0]:depth_segment[1], depth_type + '_Segment_Type'] = 'Loss'
                    if segment_depth_mean > region_thr['Gain']:
                        pos_data.loc[depth_segment[0]:depth_segment[1], depth_type + '_Segment_Type'] = 'Gain'
            else:
                pos_data.loc[sample_gene_data.index, depth_type + '_Segment'] = 0

    t2 = time.time()
    cost_time = t2 - t1
    print('Run time: {}s'.format(cost_time))
