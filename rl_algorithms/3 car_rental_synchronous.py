#######################################################################
# 版权声明                                                            #
# 2016 Shangtong Zhang (zhangshangtong.cpp@gmail.com)                 #
# 2016 Kenta Shimada (hyperkentakun@gmail.com)                        #
# 2017 Aja Rangaswamy (aja004@gmail.com)                              #
# 允许修改本代码，但必须保留顶部这段声明                               #
#######################################################################

# 本文件由 Tahsincan Köse 贡献：
# 这里实现的是“同步式策略评估（synchronous policy evaluation）”，
# 而 car_rental.py 实现的是“异步式策略评估（asynchronous policy evaluation）”。
# 此外，本文件还使用了多进程来加速计算，并包含了习题 4.5 的解答。

import numpy as np
import matplotlib.pyplot as plt
import math
import multiprocessing as mp#多进程
from functools import partial#函数柯里化
import time#计时工具
import itertools#迭代工具

############# 问题相关常量 #######################
MAX_CARS = 20                  # 每个地点最多可停放的车辆数
MAX_MOVE = 5                   # 每晚最多可调度的车辆数
MOVE_COST = -2                 # 每调动一辆车的成本
ADDITIONAL_PARK_COST = -4      # 超额停车的额外成本（习题 4.5 扩展）

RENT_REWARD = 10               # 每租出一辆车获得的收益

# 地点1每天租车请求数的泊松分布期望
RENTAL_REQUEST_FIRST_LOC = 3

# 地点2每天租车请求数的泊松分布期望
RENTAL_REQUEST_SECOND_LOC = 4

# 地点1每天还车数量的泊松分布期望
RETURNS_FIRST_LOC = 3

# 地点2每天还车数量的泊松分布期望
RETURNS_SECOND_LOC = 2
#################################################

# 泊松概率缓存，避免重复计算
poisson_cache = dict()


def poisson(n, lam):
    """
    计算泊松分布 P(X=n)，其中参数 lambda = lam
    并使用缓存提高效率
    """
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache.keys():
        poisson_cache[key] = math.exp(-lam) * math.pow(lam, n) / math.factorial(n)
    return poisson_cache[key]


class PolicyIteration:
    def __init__(self, truncate, parallel_processes, delta=1e-2, gamma=0.9, solve_4_5=False):
        self.TRUNCATE = truncate                          # 泊松分布枚举截断上限
        self.NR_PARALLEL_PROCESSES = parallel_processes  # 并行进程数
        self.actions = np.arange(-MAX_MOVE, MAX_MOVE + 1)  # 所有可能动作：从 -5 到 5
        self.inverse_actions = {el: ind[0] for ind, el in np.ndenumerate(self.actions)}
        self.values = np.zeros((MAX_CARS + 1, MAX_CARS + 1))  # 状态价值函数 V(s)
        self.policy = np.zeros(self.values.shape, dtype=np.int)  # 当前策略
        self.delta = delta                                # 策略评估收敛阈值
        self.gamma = gamma                                # 折扣因子
        self.solve_extension = solve_4_5                  # 是否启用习题 4.5 的扩展条件

    def solve(self):
        """
        执行完整的策略迭代：
        1. 策略评估
        2. 策略改进
        直到策略不再变化
        """
        iterations = 0
        total_start_time = time.time()
        while True:
            start_time = time.time()
            self.values = self.policy_evaluation(self.values, self.policy)
            elapsed_time = time.time() - start_time
            print(f'PE => Elapsed time {elapsed_time} seconds')

            start_time = time.time()
            policy_change, self.policy = self.policy_improvement(self.actions, self.values, self.policy)
            elapsed_time = time.time() - start_time
            print(f'PI => Elapsed time {elapsed_time} seconds')

            if policy_change == 0:
                break
            iterations += 1

        total_elapsed_time = time.time() - total_start_time
        print(f'Optimal policy is reached after {iterations} iterations in {total_elapsed_time} seconds')

    # 同步更新（out-place）：每轮使用旧 values 计算新的 values
    def policy_evaluation(self, values, policy):

        global MAX_CARS
        while True:
            new_values = np.copy(values)
            k = np.arange(MAX_CARS + 1)

            # 构造所有状态的笛卡尔积：(i, j)
            all_states = ((i, j) for i, j in itertools.product(k, k))

            with mp.Pool(processes=self.NR_PARALLEL_PROCESSES) as p:
                cook = partial(self.expected_return_pe, policy, values)
                results = p.map(cook, all_states)

            # 将每个状态计算得到的新价值填回 new_values
            for v, i, j in results:
                new_values[i, j] = v

            # 使用所有状态价值变化的绝对值之和作为收敛判据
            difference = np.abs(new_values - values).sum()
            print(f'Difference: {difference}')

            values = new_values
            if difference < self.delta:
                print(f'Values are converged!')
                return values

    def policy_improvement(self, actions, values, policy):
        """
        对每个状态，枚举所有可能动作，
        选择使期望回报最大的动作，得到新策略
        """
        new_policy = np.copy(policy)

        # expected_action_returns[i, j, a] 表示在状态 (i,j) 下采取动作 a 的期望回报
        expected_action_returns = np.zeros((MAX_CARS + 1, MAX_CARS + 1, np.size(actions)))
        cooks = dict()

        with mp.Pool(processes=8) as p:
            for action in actions:
                k = np.arange(MAX_CARS + 1)
                all_states = ((i, j) for i, j in itertools.product(k, k))
                cooks[action] = partial(self.expected_return_pi, values, action)
                results = p.map(cooks[action], all_states)

                for v, i, j, a in results:
                    expected_action_returns[i, j, self.inverse_actions[a]] = v

        # 对每个状态选择最优动作
        for i in range(expected_action_returns.shape[0]):
            for j in range(expected_action_returns.shape[1]):
                new_policy[i, j] = actions[np.argmax(expected_action_returns[i, j])]

        # 统计策略发生变化的状态数
        policy_change = (new_policy != policy).sum()
        print(f'Policy changed in {policy_change} states')
        return policy_change, new_policy

    # O(n^4) 复杂度：枚举所有可能的租车请求与还车情况
    def bellman(self, values, action, state):
        """
        根据 Bellman 方程计算：
        在给定状态 state 下执行动作 action 的期望回报
        """
        expected_return = 0

        # 计算调车成本
        if self.solve_extension:
            if action > 0:
                # 如果从地点1往地点2调车，第一辆免费（习题 4.5 扩展）
                expected_return += MOVE_COST * (action - 1)
            else:
                expected_return += MOVE_COST * abs(action)
        else:
            expected_return += MOVE_COST * abs(action)

        # 枚举两个地点的租车请求数量
        for req1 in range(0, self.TRUNCATE):
            for req2 in range(0, self.TRUNCATE):

                # 调车后两个地点的车辆数
                num_of_cars_first_loc = int(min(state[0] - action, MAX_CARS))
                num_of_cars_second_loc = int(min(state[1] + action, MAX_CARS))

                # 实际租出去的车不能超过现有车辆数
                real_rental_first_loc = min(num_of_cars_first_loc, req1)
                real_rental_second_loc = min(num_of_cars_second_loc, req2)

                # 租车收益
                reward = (real_rental_first_loc + real_rental_second_loc) * RENT_REWARD

                # 习题 4.5 扩展：停车超过 10 辆要额外收费
                if self.solve_extension:
                    if num_of_cars_first_loc >= 10:
                        reward += ADDITIONAL_PARK_COST
                    if num_of_cars_second_loc >= 10:
                        reward += ADDITIONAL_PARK_COST

                # 更新租车后的剩余车辆数
                num_of_cars_first_loc -= real_rental_first_loc
                num_of_cars_second_loc -= real_rental_second_loc

                # 当前租车请求组合出现的概率
                prob = poisson(req1, RENTAL_REQUEST_FIRST_LOC) * \
                       poisson(req2, RENTAL_REQUEST_SECOND_LOC)

                # 枚举两个地点的还车数量
                for ret1 in range(0, self.TRUNCATE):
                    for ret2 in range(0, self.TRUNCATE):

                        # 还车后新状态的车辆数，不能超过最大容量
                        num_of_cars_first_loc_ = min(num_of_cars_first_loc + ret1, MAX_CARS)
                        num_of_cars_second_loc_ = min(num_of_cars_second_loc + ret2, MAX_CARS)

                        # 当前完整转移（请求+还车）的联合概率
                        prob_ = poisson(ret1, RETURNS_FIRST_LOC) * \
                                poisson(ret2, RETURNS_SECOND_LOC) * prob

                        # Bellman 状态价值方程：
                        # prob_ 对应 p(s'|s,a)
                        # s' 即 (num_of_cars_first_loc_, num_of_cars_second_loc_)
                        expected_return += prob_ * (
                                reward + self.gamma * values[num_of_cars_first_loc_, num_of_cars_second_loc_]
                        )
        return expected_return

    # 为了并行化，单独拆出的辅助函数
    # 用于“策略评估”的期望回报计算
    def expected_return_pe(self, policy, values, state):

        action = policy[state[0], state[1]]
        expected_return = self.bellman(values, action, state)
        return expected_return, state[0], state[1]

    # 为了并行化，单独拆出的辅助函数
    # 用于“策略改进”的期望回报计算
    def expected_return_pi(self, values, action, state):

        # 非法动作直接返回负无穷，表示该动作不可选
        # 例如：地点1车不够却还要往地点2调车
        if ((action >= 0 and state[0] >= action) or (action < 0 and state[1] >= abs(action))) == False:
            return -float('inf'), state[0], state[1], action

        expected_return = self.bellman(values, action, state)
        return expected_return, state[0], state[1], action

    def plot(self):
        """
        将最终策略以表格形式画出来
        """
        print(self.policy)
        plt.figure()
        plt.xlim(0, MAX_CARS + 1)
        plt.ylim(0, MAX_CARS + 1)
        plt.table(cellText=np.flipud(self.policy), loc=(0, 0), cellLoc='center')
        plt.show()


if __name__ == '__main__':
    TRUNCATE = 9
    solver = PolicyIteration(TRUNCATE, parallel_processes=4, delta=1e-1, gamma=0.9, solve_4_5=True)
    solver.solve()
    solver.plot()