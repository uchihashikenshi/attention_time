#coding:utf-8
import numpy
import sklearn
from sklearn.linear_model import LogisticRegression
import pandas as pd
import random
from scipy import stats

class Function():

    def __init__(self):
        pass

    def get_cv_prediction(self, clf, train_features, train_labels, K):
        pred = sklearn.cross_validation.cross_val_predict(clf, train_features, train_labels, cv=K)
        return pred


    def matching(self, data, ps_score, treat):
        """
        ps_scoreを用いてマッチングを取る.
        input
            data: pandasオブジェクト
            ps_score: proba[:, 1]
        output
            matched_tr_ind_list: マッチングを取ったtreat側のインデックス
            matched_co_ind_list: マッチングを取ったcontrol側のインデックス
        """
        columns = numpy.array(data.columns).tolist()
        columns.append('ps_score') ## 傾向スコアは'ps_score'という名前で追加

        ps_score = pd.Series(ps_score, index=data.index, name='ps_score')
        data_ps = data.T.append(ps_score).T

        tr = data_ps.ix[data_ps[treat] == 1]
        co = data_ps.ix[data_ps[treat] == 0]

        if len(tr) > len(co):
            searcher, being_searched = co, tr
        else:
            searcher, being_searched = tr, co

        bs_copy = being_searched.copy()
        matched_tr_ind_list, matched_co_ind_list = [], []

        for s_ind, s_ps_score in zip(searcher.index, searcher['ps_score']):
            res_min = 1000

            for bs_ind, bs_ps_score in zip(bs_copy.index, bs_copy['ps_score']):
                if abs(s_ps_score - bs_ps_score) < res_min:
                    ## tr - coで処置効果を見る
                    if len(tr) > len(co):
                        res_min = abs(bs_ps_score - s_ps_score)
                    else:
                        res_min = abs(s_ps_score - bs_ps_score)
                    matched_bs_ind = bs_ind

            if len(tr) > len(co):
                matched_tr_ind_list.append(matched_bs_ind)
                matched_co_ind_list.append(s_ind)
            else:
                matched_tr_ind_list.append(s_ind)
                matched_co_ind_list.append(matched_bs_ind)

            bs_copy = bs_copy.drop([matched_bs_ind])

        return matched_tr_ind_list, matched_co_ind_list


    def ps_matched_res(self, data, treat, matched_tr_ind_list, matched_co_ind_list):
        """
        ps_matchした場合の共変量の差を計算.
        """

        tr = data.ix[data[treat] == 1]
        co = data.ix[data[treat] == 0]

        res_array = []
        for column_name in data.columns:
            res_list = []
            for tr_ind, co_ind in zip(matched_tr_ind_list, matched_co_ind_list):
                res_list.append(tr[column_name][tr_ind] - co[column_name][co_ind])

            if res_array == []:
                res_array = numpy.array(res_list)
            else:
                res_array = numpy.vstack((res_array, res_list))

        res_array = res_array.T
        res_pd = pd.DataFrame(res_array, columns=data.columns)

        return res_pd


    def random_matched_res(self, data, treat):
        """
        random_matchした場合の共変量の差を計算.
        """

        tr = data.ix[data[treat] == 1]
        co = data.ix[data[treat] == 0]

        tr_ind_list, co_ind_list = range(len(tr)), range(len(co))
        random.shuffle(tr_ind_list)
        random.shuffle(co_ind_list)

        res_array = []
        for tr_rand_ind, co_rand_ind in zip(tr_ind_list, co_ind_list):
            res_list = numpy.array(tr)[tr_rand_ind] - numpy.array(co)[co_rand_ind]

            if res_array == []:
                res_array = numpy.array(res_list)
            else:
                res_array = numpy.vstack((res_array, res_list))

        res_pd = pd.DataFrame(res_array, columns=data.columns)

        return res_pd


    def eval_mean(self, ps_matched_res, random_matched_res):
        """
        ps_matchとrandom_matchそれぞれの共変量の差の平均を算出.
        """
        ps_mean, random_mean = {}, {}

        for column in ps_matched_res.columns:
            ps_mean = numpy.mean(numpy.array(ps_matched_res[column]))
            for i in xrange(len(ps_matched_res.index)):
                ps_sum += ps_matched_res[column][i]
                random_sum += random_matched_res[column][i]
            ps_mean.update({column:ps_sum / len(ps_matched_res.index)})
            random_mean.update({column:random_sum / len(ps_matched_res.index)})

        return ps_mean, random_mean


    def t_value(self, data, matched_tr_ind_list, matched_co_ind_list, treat_col_name):
        """
        t値を計算する
        """
        ps_tr, ps_co = data.ix[matched_tr_ind_list], data.ix[matched_co_ind_list]
        rand_tr, rand_co = data.ix[data[treat_col_name]==1], data.ix[data[treat_col_name]==0]
        del ps_tr[treat_col_name]
        del ps_co[treat_col_name]
        del rand_tr[treat_col_name]
        del rand_co[treat_col_name]
        ps_t_val_dict, rand_t_val_dict = {}, {}

        for column in ps_tr.columns:
            ## ps scoreでマッチングを取った場合のT値
            ps_tr_array, ps_co_array = numpy.array(ps_tr[column]), numpy.array(ps_co[column])
            # ps_tr_n, ps_co_n = ps_tr.shape[0], ps_co.shape[0]
            # ps_tr_mean, ps_co_mean = numpy.mean(ps_tr_array), numpy.mean(ps_co_array)
            # ps_tr_var, ps_co_var = numpy.var(ps_tr_array), numpy.var(ps_co_array)

            ps_t_val = stats.ttest_ind(ps_tr_array, ps_co_array)[0]

            ps_t_val_dict.update({column:ps_t_val})

            ## マッチングしていない場合のT値
            rand_tr_array, rand_co_array = numpy.array(rand_tr[column]), numpy.array(rand_co[column])
            # rand_tr_n, rand_co_n = rand_tr.shape[0], rand_co.shape[0]
            # rand_tr_mean, rand_co_mean = numpy.mean(rand_tr_array), numpy.mean(rand_co_array)
            # rand_tr_var, rand_co_var = numpy.var(rand_tr_array), numpy.var(rand_co_array)

            rand_t_val = stats.ttest_ind(rand_tr_array, rand_co_array)[0]

            rand_t_val_dict.update({column:rand_t_val})

        return ps_t_val_dict, rand_t_val_dict


    def t_variation(self, ps_t_val_dict, rand_t_val_dict):
        """
        T値の変分を正規化する
        """
        t_vari = {}

        for key in ps_t_val_dict.keys():
            ps_t = ps_t_val_dict[key]
            rand_t= rand_t_val_dict[key]
            t_vari_val = (rand_t - ps_t) / rand_t
            t_vari.update({key:t_vari_val})

        return t_vari


    def dict_sum_update(self, dict, sum_dict, k):
        """
        辞書型に対して合計を更新する.
        """
        for key in dict.keys():
            val = dict[key]
            if k == 0:
                sum_dict.update({key:val})
            else:
                val_past = sum_dict[key]
                sum_dict.update({key:val + val_past})

        return sum_dict


    def dict_mean_sum_update(self, sum_dict, mean_sum_dict, k_folds, test_index):
        test_dict = {}
        for key in sum_dict.keys():
            sum = sum_dict[key]
            test_dict.update({key:sum / k_folds})
            if test_index == 0:
                mean_sum_dict.update({key:sum / k_folds})
            else:
                mean_val_past = mean_sum_dict[key]
                mean_sum_dict.update({key:sum / k_folds + mean_val_past})

        return mean_sum_dict, test_dict


    def dict_mean(self, dict, n):
        for key in dict.keys():
            sum = dict[key]
            dict.update({key:sum / n})

        return dict


    def kl_div(self, p, q):
        """
        input:
            p, q
        output:
            KL div
        """
        p = numpy.array(p) + 0.000001
        q = numpy.array(q) + 0.000001

        if len(p) > len(q):
            p = p[-len(q):]
        if len(p) < len(q):
            q = q[-len(p):]

        return stats.entropy(p, q, 2)


    def js_div(self, p, q):
        """
        input:
            p, q
        output:
            Jensen-Shannon div
        """
        p = numpy.array(p)
        q = numpy.array(q)

        if len(p) > len(q):
            p = p[-len(q):]
        if len(p) < len(q):
            q = q[-len(p):]

        m = 0.5 * (p + q)

        return 0.5 * self.kl_div(p, m) + 0.5 * self.kl_div(q, m)