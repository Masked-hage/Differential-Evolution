###### optimizer.py #####
#                                           Last Update:  2020/4/13
#
# 差分進化アルゴリズムの詳細アルゴリズムファイル
# インスタンスはoptとして生成

# 他ファイル,モジュールのインポート
import function as fc
import numpy as np

# 差分進化アルゴリズム（実数）クラス
class DifferentialEvolution:

    """ コンストラクタ """
    # 初期化メソッド
    def __init__(self, cnf, fnc):
        self.cnf = cnf      # 設定
        self.fnc = fnc      # 関数
        self.pop = []       # 個体群

    """ インスタンスメソッド """
    # 初期化
    def initializeSolutions(self):
        for i in range(self.cnf.max_pop):
            self.pop.append(Solution(self.cnf, self.fnc))
            self.getFitness(self.pop[i])

    # 次世代個体群生成
    def getNextPopulation(self):
        self.mutation()
        self.generateOffspring()
        for i in range(self.cnf.max_pop):
            self.getFitness(self.pop[i + self.cnf.max_pop])
        self.selection()

    # 変異ベクトルの生成(rand/1)
    def mutation(self):
        mut = []
        for i in range(self.cnf.max_pop):
            num = list(range(self.cnf.max_pop))
            num.remove(i)
            idx = self.cnf.rd.choice(num, 3, replace=False)
            v = self.pop[idx[0]].x + self.cnf.scaling * (self.pop[idx[1]].x - self.pop[idx[2]].x)
            mut.append(v)
        return mut

    # 交叉(binomial交叉)
    def apply_binomial_Xover(self, p_v, p_x):
        x_next = Solution(self.cnf, self.fnc, parent=None)
        for i in range(self.cnf.prob_dim):
            if self.cnf.rd.rand() <= self.cnf.CR or i == self.cnf.rd.randint(0, self.cnf.prob_dim):
                x_next.x[i] = p_v[i]
            else:
                x_next.x[i] = p_x[i]
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
            else:
                pass
        del self.pop[self.cnf.max_pop : 2 * self.cnf.max_pop]

#個体のクラス
class Solution:
    """ コンストラクタ """
    # 初期化メソッド
    def __init__(self, cnf, fnc, parent=None):
        self.cnf, self.fnc, self.x, self.f = cnf, fnc, [], 0.
        # 個体の初期化
        if parent == None:
            self.x = [self.cnf.rd.uniform(self.fnc.axis_range[0], self.fnc.axis_range[1]) for i in range(self.cnf.prob_dim)]
        # 親個体のコピー
        else:
            self.x = [parent.x[i] for i in range(self.cnf.prob_dim)]
        # リスト -> ndarray
        self.x = np.array(self.x)
