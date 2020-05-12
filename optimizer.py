###### optimizer.py #####
#                                           Last Update:  2020/4/13
#
# 差分進化アルゴリズムの詳細アルゴリズムファイル
# インスタンスはoptとして生成

# 他ファイル,モジュールのインポート
import function as fc
import numpy as np
from scipy.stats import cauchy

# 差分進化アルゴリズム（実数）クラス
class DifferentialEvolution:

    """ コンストラクタ """
    # 初期化メソッド
    def __init__(self, cnf, fnc):
        self.cnf            = cnf       # 設定
        self.fnc            = fnc       # 関数
        self.pop            = []        # 個体群
        self.archive        = []        # 劣解アーカイブ
        self.s_history      = []        # 成功時のパラメータ平均値格納メモリ
        self.history_idx    = 0         # 履歴メモリのインデックス
        self.sum_mutNum     = 0         # 淘汰の成功回数
        self.sum_scaling    = 0.        # 淘汰成功時のスケーリングファクタの総和
        self.sum_scaling2   = 0.        # 淘汰成功時のスケーリングファクタの二乗和
        self.sum_CR         = 0.        # 淘汰成功時の交叉率の総和

    """ インスタンスメソッド """
    # 初期化
    def initializeSolutions(self):
        for i in range(self.cnf.history_size):
            self.s_history.append(History(self.cnf.init_scaling, self.cnf.init_CR))
        for i in range(self.cnf.max_pop):
            self.pop.append(Solution(self.cnf, self.fnc, self.s_history))
            self.getFitness(self.pop[i])

    # 次世代個体群生成
    def getNextPopulation(self):
        self.sort_Population()
        self.generateOffspring()
        for i in range(self.cnf.max_pop):
            self.getFitness(self.pop[i + self.cnf.max_pop])
        self.selection()
        self.resize_archive()
        self.update_parameter()
        self.reset_parameter()

    # 集団Pのソート(昇順)
    def sort_Population(self):
        self.pop.sort(key=lambda func: func.f)

    # 変異ベクトルの生成(current-to-pbest/1)
    def mutation(self):
        mut = []
        for i in range(self.cnf.max_pop):
            num = list(range(self.cnf.max_pop))
            choice_R = self.cnf.rd.randint(self.cnf.min_choice_R, self.cnf.max_choice_R) / 100
            best_num = list(range(int(self.cnf.max_pop * choice_R)))
            num.remove(i)
            if i in best_num:
                best_num.remove(i)
            if len(self.archive) == 0:
                idx = list(self.cnf.rd.choice(num, 2, replace=False))
            else:
                idx = list(self.cnf.rd.choice(num, 1))
                num.remove(idx[0])
                num.extend(range(self.cnf.max_pop, (self.cnf.max_pop + len(self.archive))))
                idx.append(self.cnf.rd.choice(num))
            b_idx = list(self.cnf.rd.choice(best_num, 1))
            if idx[1] < self.cnf.max_pop:
                v = self.pop[i].x + self.pop[i].scaling * (self.pop[b_idx[0]].x - self.pop[i].x) + self.pop[i].scaling * (self.pop[idx[0]].x - self.pop[idx[1]].x)
            else:
                v = self.pop[i].x + self.pop[i].scaling * (self.pop[b_idx[0]].x - self.pop[i].x) + self.pop[i].scaling * (self.pop[idx[0]].x - self.archive[idx[1] - self.cnf.max_pop].x)
            mut.append(v)
        return mut

    # 交叉(binomial交叉)
    def apply_binomial_Xover(self, p_v, p_x):
        x_next = Solution(self.cnf, self.fnc, self.s_history)
        j_rand = self.cnf.rd.randint(0, self.cnf.prob_dim)
        for i in range(self.cnf.prob_dim):
            if self.cnf.rd.rand() <= self.pop[i].CR or i == j_rand:
                x_next.x[i] = p_v[i]
            else:
                x_next.x[i] = p_x[i]
            # 定義域外の探索防止
            x_next.x[i] = np.clip(x_next.x[i], self.fnc.axis_range[0], self.fnc.axis_range[1])
        return x_next

    # 子個体の生成
    def generateOffspring(self):
        mut = self.mutation()
        for i in range(self.cnf.max_pop):
            self.pop.append(self.apply_binomial_Xover(mut[i], self.pop[i].x))

    # 評価値fの計算
    def getFitness(self, solution):
        solution.f = self.fnc.doEvaluate(solution.x)

    # 淘汰
    def selection(self):
        for i in range(self.cnf.max_pop):
            if self.pop[i].f >= self.pop[i + self.cnf.max_pop].f:
                self.archive.append(self.pop[i])
                self.pop[i] = self.pop[i + self.cnf.max_pop]
                self.sum_mutNum += 1
                self.sum_scaling += self.pop[i + self.cnf.max_pop].scaling
                self.sum_scaling2 += self.pop[i + self.cnf.max_pop].scaling ** 2
                self.sum_CR += self.pop[i + self.cnf.max_pop].CR
            else:
                pass
        del self.pop[self.cnf.max_pop : 2 * self.cnf.max_pop]

    # アーカイブのサイズ調整
    def resize_archive(self):
        if len(self.archive) > self.cnf.archive_size:
            while len(self.archive) > self.cnf.archive_size:
                del self.archive[self.cnf.rd.randint(0, len(self.archive))]
        else:
            pass

    # 平均値の更新
    def update_parameter(self):
        if self.sum_scaling != 0 and self.sum_CR != 0:
            self.s_history[self.history_idx].h_scaling = (1 - self.cnf.learning_R) * self.s_history[self.history_idx].h_scaling + self.cnf.learning_R * self.sum_scaling2 / self.sum_scaling
            self.s_history[self.history_idx].h_CR = (1 - self.cnf.learning_R) * self.s_history[self.history_idx].h_CR + self.cnf.learning_R * self.sum_CR / self.sum_mutNum
            self.history_idx += 1
            if self.history_idx >= self.cnf.history_size:
                self.history_idx = 0
        else:
            pass

    # パラメータのリセット
    def reset_parameter(self):
        self.sum_mutNum     = 0
        self.sum_scaling    = 0.
        self.sum_scaling2   = 0.
        self.sum_CR         = 0.

#個体のクラス
class Solution:
    """ コンストラクタ """
    # 初期化メソッド
    def __init__(self, cnf, fnc, history):
        self.cnf, self.fnc, self.x, self.f, self.scaling, self.CR = cnf, fnc, [], 0., -1., -1.
        idx = self.cnf.rd.randint(0, self.cnf.history_size)
        # 個体の初期化
        self.x = [self.cnf.rd.uniform(self.fnc.axis_range[0], self.fnc.axis_range[1]) for i in range(self.cnf.prob_dim)]
        while self.scaling < 0.:
            self.scaling = cauchy.rvs(loc=history[idx].h_scaling, scale=self.cnf.param_scaling)
        if self.scaling > 1.:
            self.scaling = 1.
        self.CR = self.cnf.rd.normal(loc=history[idx].h_CR, scale=self.cnf.param_CR)
        self.CR = np.clip(self.CR, 0., 1.)
        # リスト -> ndarray
        self.x = np.array(self.x)

#履歴メモリのクラス
class History:
    """ コンストラクタ """
    # 初期化メソッド
    def __init__(self, scaling, CR):
        self.h_scaling = scaling
        self.h_CR = CR
