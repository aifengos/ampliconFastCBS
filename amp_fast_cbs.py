# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import time
from scipy.stats import ttest_ind, levene

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


class FastCBS:
    def __init__(self, input_data, p_thr=0.01, sig_diff_stop=True, jump_thr=0.3, diff_fold=3, fix_width=2):
        self.input_data = input_data
        self.p_thr = p_thr
        self.sig_diff_stop = sig_diff_stop
        self.fix_width = fix_width
        self.jump_thr = jump_thr
        self.diff_fold = diff_fold
        self.min_index = min(self.input_data.index)
        self.max_index = max(self.input_data.index)

    def run(self):
        depth_segments = list()
        # print(self.input_data, self.min_index, self.max_index)
        return self.rec_segment(self.input_data, self.min_index, self.max_index, depth_segments)

    def rec_segment(self, cbs_data, start, end, depth_segments=None, break_type='Sum_Fix', break_point=False):
        if depth_segments is None:
            depth_segments = list()
        if (end - start < 4) or (break_point and self.sig_diff_stop):
            depth_segments.append((start, end, break_type))
        else:
            print(start, end)
            prev_break_point, prev_break_type = break_point, break_type
            # print(start, end, cbs_data.loc[start:end])
            break_point, p, s, e, break_type = self.cbs(cbs_data, start, end)
            # if s - start < 2:
            #     s = start
            # if end - e < 2:
            #     e = end
            print('Proposed partition of {} to {} from {} to {} with p value {}, significant difference is {}'.format(
                start, end, s, e, p, break_point))
            if (not break_point) or (prev_break_point and (break_type != 'Sum_Fix')) or (e - s == end - start):
                depth_segments.append((start, end, break_type))
            else:
                if s - 1 > start:
                    self.rec_segment(cbs_data, start, s - 1, depth_segments, break_type)
                if e > s:
                    self.rec_segment(cbs_data, s, e, depth_segments, break_type, break_point)
                if e + 1 < end:
                    self.rec_segment(cbs_data, e + 1, end, depth_segments, break_type)
        return depth_segments

    def cbs(self, cbs_data, start, end):
        select_data = cbs_data.loc[start:end]
        trans_data = select_data.to_frame(name='Values')
        trans_data['Values_Offset'] = trans_data['Values'] - trans_data['Values'].mean()
        trans_data['Offset_Sum'] = trans_data['Values_Offset'].cumsum().shift(1)
        trans_data['Offset_Diff1'] = trans_data['Values_Offset'].diff()
        trans_data['Offset_Diff2'] = trans_data['Values_Offset'].diff(periods=2).shift(-1)
        trans_data[['Offset_Diff_Min', 'Offset_Diff_Max']] = trans_data[['Offset_Diff1', 'Offset_Diff2']].agg(
            lambda x: diff_extremal(x), axis=1, result_type='expand')
        trans_data['Offset_Sum_Fix'] = trans_data['Offset_Sum'] - self.diff_fold * trans_data['Offset_Diff_Min']
        # print(diffs, sums_fix)

        bp0, bp1 = trans_data['Offset_Sum_Fix'].idxmin(), trans_data['Offset_Sum_Fix'].idxmax()

        bp_start, bp_end = min(bp0, bp1), max(bp0, bp1)
        print(bp0, bp1)

        candidate_pair = {bp0: bp1, bp1: bp0}

        jump_candidates = dict()

        for bp_pos in [bp0, bp1]:
            jump_candidates[bp_pos] = dict()
            if bp_pos - self.fix_width > start:
                jump_candidates[bp_pos]['Start'] = bp_pos - self.fix_width
            else:
                jump_candidates[bp_pos]['Start'] = start
            if bp_pos + self.fix_width < end:
                jump_candidates[bp_pos]['End'] = bp_pos + self.fix_width
            else:
                jump_candidates[bp_pos]['End'] = end

        min_diffs_max = trans_data['Offset_Diff_Max'].loc[jump_candidates[bp0]['Start']:jump_candidates[bp0]['End']]
        max_diffs_max = trans_data['Offset_Diff_Max'].loc[jump_candidates[bp1]['Start']:jump_candidates[bp1]['End']]
        jump_valid_max = {bp0: min_diffs_max[min_diffs_max > self.jump_thr],
                          bp1: max_diffs_max[max_diffs_max < -1 * self.jump_thr]}
        candidate_diffs_min = trans_data['Offset_Diff_Min']
        jump_valid_min = {bp0: candidate_diffs_min[candidate_diffs_min > self.jump_thr],
                          bp1: candidate_diffs_min[candidate_diffs_min < -1 * self.jump_thr]}

        break_checks = {bp0: True, bp1: True}
        for break_end in break_checks:
            # print(break_end, jump_valid_max[break_end])
            if jump_valid_max[break_end].empty and (break_end != start) and (break_end != end):
                break_checks[break_end] = False

        break_scores = sum(break_checks.values())
        breakpoint_candidates = list()
        if break_scores == 2:
            jump_bp_s, jump_bp_end = min(bp0, bp1), max(bp0, bp1)
            breakpoint_candidates.append((jump_bp_s, jump_bp_end, 'Sum_Fix'))
            if (jump_bp_s - start < 2) and ((start, jump_bp_end, 'Sum_Fix') not in breakpoint_candidates):
                breakpoint_candidates.append((start, jump_bp_end, 'Sum_Fix'))
            if (end - jump_bp_end) < 2 and ((jump_bp_s, end, 'Sum_Fix') not in breakpoint_candidates):
                breakpoint_candidates.append((jump_bp_s, end, 'Sum_Fix'))
        elif break_scores == 1:
            # print(break_checks)
            false_bp = [key for key in break_checks if not break_checks[key]][0]
            pair_bp = candidate_pair[false_bp]
            jump_bps = {'Jump_Max': jump_valid_max[false_bp], 'Jump_Min': jump_valid_min[false_bp]}
            for jump_type in jump_bps:
                if not jump_bps[jump_type].empty:
                    for jump_bp_index in jump_bps[jump_type].index:
                        jump_bp_s, jump_bp_end = min(pair_bp, jump_bp_index), max(pair_bp, jump_bp_index)
                        if (jump_bp_s, jump_bp_end, jump_type) not in breakpoint_candidates:
                            breakpoint_candidates.append((jump_bp_s, jump_bp_end, jump_type))
                        if (jump_bp_s - start < 2) and ((start, jump_bp_end, jump_type) not in breakpoint_candidates):
                            breakpoint_candidates.append((start, jump_bp_end, jump_type))
                        if (end - jump_bp_end) < 2 and ((jump_bp_s, end, jump_type) not in breakpoint_candidates):
                            breakpoint_candidates.append((jump_bp_s, end, jump_type))
            if abs(select_data.loc[start:pair_bp - 1].mean() - 1) > abs(
                    select_data.loc[pair_bp + 1:end].mean() - 1):
                breakpoint_candidates.append((start, pair_bp, 'Endpoint'))
            else:
                breakpoint_candidates.append((pair_bp, end, 'Endpoint'))
        print(break_checks, breakpoint_candidates)
        if breakpoint_candidates:
            breakpoint_p = dict()
            for breakpoint_candidate in breakpoint_candidates:
                candidate_bp_s, candidate_bp_e, candidate_bp_type = breakpoint_candidate
                if candidate_bp_e != end:
                    candidate_bp_e -= 1
                # if candidate_bp_s - start < 2:
                #     candidate_bp_s = start
                # if end - se_candidate < 2:
                #     se_candidate = end
                print(candidate_bp_s, candidate_bp_e, candidate_bp_type)

                xt = trans_data['Values_Offset'].loc[candidate_bp_s:candidate_bp_e]
                xn = trans_data['Values_Offset'].drop(xt.index)
                if (len(xt) > 1) and (len(xn) > 1):
                    test_stat, test_p = self.t_test(xt, xn)
                    breakpoint_p[(candidate_bp_s, candidate_bp_e, candidate_bp_type)] = test_p
            if breakpoint_p:
                print(breakpoint_p)
                best_p = min(breakpoint_p.values())
                best_bp_s, best_bp_e, best_bp_type = min(breakpoint_p, key=breakpoint_p.get)
                if best_p < self.p_thr:
                    return True, best_p, best_bp_s, best_bp_e, best_bp_type
                else:
                    return False, best_p, best_bp_s, best_bp_e, best_bp_type
            else:
                return False, 1, bp_start, bp_end, 'Sum_Fix'
        else:
            if bp_end != end:
                bp_end -= 1
            return False, 1, bp_start, bp_end, 'Sum_Fix'

    @staticmethod
    def t_test(a_data, b_data, sd_thr=0.05):
        equal_val_stat, equal_val_p = levene(a_data, b_data)
        if equal_val_p < sd_thr:
            equal_val_bool = False
        else:
            equal_val_bool = True
        ttest_stat, ttest_p = ttest_ind(a_data, b_data, equal_var=equal_val_bool)
        return ttest_stat, ttest_p


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
    for sample_gene in pos_data[['Sample_Name', 'Gene']].drop_duplicates().values.tolist():

        sample_gene_data = pos_data[(pos_data['Sample_Name'] == sample_gene[0]) &
                                    (pos_data['Gene'] == sample_gene[1])]
        for depth_type in ['Ref_Norm_Depth']:
            t1 = time.time()
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
            print(sample_gene, depth_type, 'Run time: {}s'.format(cost_time))
