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
        self.cnf = cnf      # 設定
        self.fnc = fnc      # 関数
        self.pop = []       # 個体群
        self.scaling_means  = 0.5                   #スケーリングファクタの平均値
        self.CR_means       = 0.5                   # 交叉率の平均値
        self.sum_mutNum     = 0                     # 淘汰の成功回数
        self.sum_scaling    = 0.                    # 淘汰成功時のスケーリングファクタの総和
        self.sum_scaling2   = 0.                    # 淘汰成功時のスケーリングファクタの二乗和
        self.sum_CR         = 0.                    # 淘汰成功時の交叉率の総和

    """ インスタンスメソッド """
    # 初期化
    def initializeSolutions(self):
        for i in range(self.cnf.max_pop):
            self.pop.append(Solution(self.cnf, self.fnc, self.scaling_means, self.CR_means))
            self.getFitness(self.pop[i])

    # 次世代個体群生成
    def getNextPopulation(self):
        self.sort_Population()
        self.generateOffspring()
        for i in range(self.cnf.max_pop):
            self.getFitness(self.pop[i + self.cnf.max_pop])
        self.selection()
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
            best_num = list(range(int(self.cnf.max_pop * self.cnf.choice_R)))
            num.remove(i)
            if i in best_num:
                best_num.remove(i)
            idx = self.cnf.rd.choice(num, 2, replace=False)
            b_idx = self.cnf.rd.choice(best_num, 1)
            v = self.pop[i].x + self.pop[i].scaling * (self.pop[b_idx[0]].x - self.pop[i].x) + self.pop[i].scaling * (self.pop[idx[0]].x - self.pop[idx[1]].x)
            mut.append(v)
        return mut

    # 交叉(binomial交叉)
    def apply_binomial_Xover(self, p_v, p_x):
        x_next = Solution(self.cnf, self.fnc, self.scaling_means, self.CR_means)
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
                self.pop[i] = self.pop[i + self.cnf.max_pop]
                self.sum_mutNum += 1
                self.sum_scaling += self.pop[i + self.cnf.max_pop].scaling
                self.sum_scaling2 += self.pop[i + self.cnf.max_pop].scaling ** 2
                self.sum_CR += self.pop[i + self.cnf.max_pop].CR
            else:
                pass
        del self.pop[self.cnf.max_pop : 2 * self.cnf.max_pop]

    # 平均値の更新
    def update_parameter(self):
        if self.sum_scaling == 0:
            self.scaling_means = (1 - self.cnf.learning_R) * self.scaling_means
        else:
            self.scaling_means = (1 - self.cnf.learning_R) * self.scaling_means + self.cnf.learning_R * self.sum_scaling2 / self.sum_scaling
        if self.sum_mutNum == 0:
            self.CR_means = (1 - self.cnf.learning_R) * self.CR_means
        else:
            self.CR_means = (1 - self.cnf.learning_R) * self.CR_means + self.cnf.learning_R * self.sum_CR / self.sum_mutNum

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
    def __init__(self, cnf, fnc, scaling_ave, CR_ave):
        self.cnf, self.fnc, self.x, self.f, self.scaling, self.CR = cnf, fnc, [], 0., -1., -1.
        # 個体の初期化
        self.x = [self.cnf.rd.uniform(self.fnc.axis_range[0], self.fnc.axis_range[1]) for i in range(self.cnf.prob_dim)]
        while self.scaling < 0.:
            self.scaling = cauchy.rvs(loc=scaling_ave, scale=self.cnf.param_scaling)
        if self.scaling > 1.:
            self.scaling = 1.
        self.CR = np.random.normal(loc=CR_ave, scale=self.cnf.param_CR)
        self.CR = np.clip(self.CR, 0., 1.)
        # リスト -> ndarray
        self.x = np.array(self.x)
