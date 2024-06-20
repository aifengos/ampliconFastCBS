# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import time
from scipy.stats import ttest_ind, levene

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


class FastCBS:
    def __init__(self, input_data, p_thr=0.01):
        self.input_data = input_data
        self.p_thr = p_thr
        self.min_index = min(self.input_data.index)
        self.max_index = max(self.input_data.index)

    def run(self):
        depth_segments = list()
        # print(self.input_data, self.min_index, self.max_index)
        return self.rsegment(self.input_data, self.min_index, self.max_index, depth_segments)

    def rsegment(self, cbs_data, start, end, depth_segments=None, break_point='Low'):
        if depth_segments is None:
            depth_segments = list()
        if (end - start < 4) or (break_point == 'High'):
            depth_segments.append((start, end, break_point))
        else:
            print(start, end)
            # print(start, end, cbs_data.loc[start:end])
            break_point, p, s, e = self.cbs(cbs_data, start, end)
            if s - start < 2:
                s = start
            if end - e < 2:
                e = end
            print('Proposed partition of {} to {} from {} to {} with p value {} is {}'.format(
                start, end, s, e, p, break_point))
            if (break_point == 'Low') or (e - s == end - start):
                depth_segments.append((start, end, break_point))
            else:
                if s - 1 > start:
                    self.rsegment(cbs_data, start, s - 1, depth_segments)
                if e > s:
                    self.rsegment(cbs_data, s, e, depth_segments, break_point)
                if e + 1 < end:
                    self.rsegment(cbs_data, e + 1, end, depth_segments)
        return depth_segments

    def cbs(self, cbs_data, start, end, fix_width=3, diff_thr=0.3, diff_fold=2):
        # check_index = list(range(start, end))
        select_data = cbs_data.loc[start:end]
        offset_data = select_data - select_data.mean()
        sums = offset_data.cumsum().shift(1)
        diffs1 = offset_data.diff()
        diffs2 = offset_data.diff(periods=2).shift(-1)
        trans_data = pd.concat([diffs1, diffs2], axis=1, sort=False)
        trans_data.columns = ['Xbar_Diff1', 'Xbar_Diff2']
        trans_data['Xbar_Diff'] = trans_data[['Xbar_Diff1', 'Xbar_Diff2']].agg(lambda x: self.min_diff(x), axis=1)
        diffs = trans_data['Xbar_Diff']
        sums_fix = sums - diff_fold * diffs
        e0, e1 = sums_fix.idxmin(), sums_fix.idxmax()
        print(e0, e1)
        ss, se = min(e0, e1), max(e0, e1)

        if ss - fix_width > start:
            candidate_s = ss - fix_width
        else:
            candidate_s = start
        if se + fix_width < end:
            candidate_e = se + fix_width
        else:
            candidate_e = end

        candidate_diffs1 = diffs1.loc[candidate_s:candidate_e]
        candidate_valid1 = {e0: candidate_diffs1[candidate_diffs1 > diff_thr],
                           e1: candidate_diffs1[candidate_diffs1 < -1 * diff_thr]}
        candidate_diffs2 = diffs2.loc[candidate_s:candidate_e]
        candidate_valid2 = {e0: candidate_diffs2[candidate_diffs2 > diff_thr],
                            e1: candidate_diffs2[candidate_diffs2 < -1 * diff_thr]}
        candidate_pair = {e0: e1, e1: e0}
        print(ss, se)
        break_checks = {e0: True, e1: True}
        for break_end in break_checks:
            if candidate_valid1[break_end].empty and (break_end != start) and (break_end != end):
                break_checks[break_end] = False

        break_scores = sum(break_checks.values())
        if break_scores == 2:
            breakpoint_candidates = [(e0, e1)]
        elif break_scores == 1:
            breakpoint_candidates = list()
            false_point = [key for key in break_checks if not break_checks[key]][0]
            pair_point = candidate_pair[false_point]
            if not candidate_valid2[false_point].empty:
                breakpoint_candidates.append((pair_point, candidate_valid2[false_point].index.min()))
            if abs(1 - select_data.loc[start:pair_point - 1].mean()) > abs(1 - select_data.loc[pair_point + 1:end].mean()):
                breakpoint_candidates.append((pair_point, start))
            else:
                breakpoint_candidates.append((pair_point, end))
        else:
            breakpoint_candidates = list()
        if breakpoint_candidates:
            breakpoint_p = dict()
            for breakpoint_candidate in breakpoint_candidates:
                ss_candidate, se_candidate = min(breakpoint_candidate), max(breakpoint_candidate)
                if ss_candidate - start < 2:
                    ss_candidate = start
                if end - se_candidate < 2:
                    se_candidate = end
                if se_candidate != end:
                    se_candidate -= 1
                xt = offset_data.loc[ss_candidate:se_candidate]
                xn = offset_data.drop(xt.index)

                if (len(xt) > 1) and (len(xn) > 1):
                    test_stat, test_p = self.t_test(xt, xn)
                    breakpoint_p[(ss_candidate, se_candidate)] = test_p
            if breakpoint_p:
                best_p = min(breakpoint_p.values())
                best_ss, best_se = min(breakpoint_p, key=breakpoint_p.get)
                print(breakpoint_p)
                if best_p < self.p_thr:
                    return 'High', best_p, best_ss, best_se
                elif 0.05 > best_p > self.p_thr:
                    return 'Mid', best_p, best_ss, best_se
                else:
                    return 'Low', best_p, best_ss, best_se
            else:
                return 'Low', 1, ss, se
        else:
            return 'Low', 1, ss, se

    @staticmethod
    def t_test(a_data, b_data, sd_thr=0.05):
        equal_val_stat, equal_val_p = levene(a_data, b_data)
        if equal_val_p < sd_thr:
            equal_val_bool = False
        else:
            equal_val_bool = True
        ttest_stat, ttest_p = ttest_ind(a_data, b_data, equal_var=equal_val_bool)
        return ttest_stat, ttest_p

    @staticmethod
    def min_diff(row):
        if row.isnull().values.any():
            return np.nan
        else:
            min_row_idx = row.abs().idxmin()
            # print(row, min_row_idx)
            return row[min_row_idx]


if __name__ == '__main__':
    pos_data = pd.read_csv('./amplicon_depth_read_data.tsv', sep='\t', low_memory=False)
    for sample_gene in pos_data[['Sample_Name', 'Gene']].drop_duplicates().values.tolist():

        sample_gene_data = pos_data[(pos_data['Sample_Name'] == sample_gene[0]) &
                                    (pos_data['Gene'] == sample_gene[1])]
        for depth_type in ['Ref_Norm_Depth']:
            t1 = time.time()
            depth_segments = FastCBS(sample_gene_data[depth_type]).run()
            t2 = time.time()
            cost_time = t2 - t1
            print(sample_gene, depth_type, 'Run time: {}s'.format(cost_time))
