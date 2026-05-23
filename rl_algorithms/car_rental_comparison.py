"""
租车问题 (Jack's Car Rental) —— 策略迭代 vs 价值迭代 对比实验
参考: Sutton & Barto, Reinforcement Learning, Example 4.2
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import time

# ====================== 问题参数 ======================
MAX_CARS = 20          # 每个地点最多停放20辆车
MAX_MOVE = 5           # 每晚最多移动5辆车
MOVE_COST = -2         # 每移动一辆车的费用（美元）
RENT_REWARD = 10       # 每租出一辆车的收益（美元）
RENTAL_REQUEST_FIRST_LOC = 3   # 地点1租车请求泊松期望
RENTAL_REQUEST_SECOND_LOC = 4  # 地点2租车请求泊松期望
RETURNS_FIRST_LOC = 3          # 地点1还车泊松期望
RETURNS_SECOND_LOC = 2         # 地点2还车泊松期望
TRUNCATE = 9           # 泊松分布截断（概率 < 1e-5 后忽略）
GAMMA = 0.9            # 折扣因子
DELTA = 1e-1           # 收敛阈值（相对宽松，加快演示速度）

# ====================== 预计算泊松概率表 ======================
def build_poisson_table(lam, max_n):
    table = np.zeros(max_n)
    for n in range(max_n):
        table[n] = math.exp(-lam) * (lam ** n) / math.factorial(n)
    return table

# 预计算4个泊松分布
p_req1 = build_poisson_table(RENTAL_REQUEST_FIRST_LOC, TRUNCATE)
p_req2 = build_poisson_table(RENTAL_REQUEST_SECOND_LOC, TRUNCATE)
p_ret1 = build_poisson_table(RETURNS_FIRST_LOC, TRUNCATE)
p_ret2 = build_poisson_table(RETURNS_SECOND_LOC, TRUNCATE)

# ====================== 预计算状态转移矩阵（核心加速） ======================
print("预计算状态转移矩阵...")
precompute_start = time.time()

N = MAX_CARS + 1  # 0..20, 共21个取值
ACTIONS = np.arange(-MAX_MOVE, MAX_MOVE + 1)  # -5..5, 共11个动作
n_actions = len(ACTIONS)
n_states = N * N

reward_matrix = np.full((n_actions, N, N), -np.inf)
trans_matrix = np.zeros((n_actions, n_states, n_states))

for a_idx, action in enumerate(ACTIONS):
    for i in range(N):
        for j in range(N):
            if action > 0 and i < action:
                continue
            if action < 0 and j < abs(action):
                continue

            move_cost = MOVE_COST * abs(action)
            cars1 = min(i - action, MAX_CARS)
            cars2 = min(j + action, MAX_CARS)

            exp_reward = 0.0
            state_idx = i * N + j

            for req1 in range(TRUNCATE):
                for req2 in range(TRUNCATE):
                    real_rent1 = min(cars1, req1)
                    real_rent2 = min(cars2, req2)
                    rent_reward = (real_rent1 + real_rent2) * RENT_REWARD
                    c1_after_rent = cars1 - real_rent1
                    c2_after_rent = cars2 - real_rent2
                    p_rq = p_req1[req1] * p_req2[req2]

                    for ret1 in range(TRUNCATE):
                        for ret2 in range(TRUNCATE):
                            c1_new = min(c1_after_rent + ret1, MAX_CARS)
                            c2_new = min(c2_after_rent + ret2, MAX_CARS)
                            p_all = p_rq * p_ret1[ret1] * p_ret2[ret2]
                            exp_reward += p_all * rent_reward
                            next_idx = c1_new * N + c2_new
                            trans_matrix[a_idx, state_idx, next_idx] += p_all

            reward_matrix[a_idx, i, j] = move_cost + exp_reward

precompute_time = time.time() - precompute_start
print(f"预计算完成，耗时 {precompute_time:.1f}s\n")


def bellman_vec(values_flat, a_idx, i, j):
    state_idx = i * N + j
    r = reward_matrix[a_idx, i, j]
    if r == -np.inf:
        return -np.inf
    v_next = trans_matrix[a_idx, state_idx] @ values_flat
    return r + GAMMA * v_next


def compute_all_action_values(values_flat):
    Q = np.full((n_actions, N, N), -np.inf)
    for a_idx in range(n_actions):
        v_next_all = trans_matrix[a_idx] @ values_flat
        v_next_mat = v_next_all.reshape(N, N)
        valid_mask = reward_matrix[a_idx] > -np.inf
        Q[a_idx][valid_mask] = reward_matrix[a_idx][valid_mask] + GAMMA * v_next_mat[valid_mask]
    return Q


# ====================== 策略迭代 ======================
def policy_iteration():
    values = np.zeros((N, N))
    policy = np.zeros((N, N), dtype=np.int32)
    policy_action_idx = np.full((N, N), MAX_MOVE, dtype=np.int32)

    iteration_count = 0
    pe_times = []
    pi_times = []
    policies = [np.copy(policy)]

    total_start = time.time()

    while True:
        pe_start = time.time()
        eval_iters = 0
        while True:
            values_flat = values.flatten()
            new_values = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    a_idx = policy_action_idx[i, j]
                    new_values[i, j] = bellman_vec(values_flat, a_idx, i, j)
            diff = np.abs(new_values - values).sum()
            values = new_values
            eval_iters += 1
            if diff < DELTA:
                break
        pe_time = time.time() - pe_start
        pe_times.append(pe_time)
        print(f'  [PI] round {iteration_count+1} eval: {eval_iters} sweeps, {pe_time:.2f}s')

        pi_start = time.time()
        values_flat = values.flatten()
        Q = compute_all_action_values(values_flat)
        new_action_idx = np.argmax(Q, axis=0)
        new_policy = ACTIONS[new_action_idx]

        policy_change = (new_policy != policy).sum()
        policy_action_idx = new_action_idx
        policy = new_policy
        pi_time = time.time() - pi_start
        pi_times.append(pi_time)
        iteration_count += 1
        policies.append(np.copy(policy))
        print(f'  [PI] round {iteration_count} improve: {pi_time:.2f}s, changed={policy_change}')

        if policy_change == 0:
            break

    total_time = time.time() - total_start
    print(f'  [PI] total: {iteration_count} outer loops, {total_time:.2f}s\n')
    return values, policy, total_time, policies, pe_times, pi_times


# ====================== 价值迭代 ======================
def value_iteration():
    values = np.zeros((N, N))

    iteration_count = 0
    iter_times = []
    diffs = []

    total_start = time.time()

    while True:
        iter_start = time.time()
        values_flat = values.flatten()
        Q = compute_all_action_values(values_flat)
        new_values = np.max(Q, axis=0)
        diff = np.abs(new_values - values).sum()
        values = new_values
        iteration_count += 1
        iter_time = time.time() - iter_start
        iter_times.append(iter_time)
        diffs.append(diff)
        print(f'  [VI] round {iteration_count}: diff={diff:.4f}, {iter_time:.2f}s')
        if diff < DELTA:
            break

    values_flat = values.flatten()
    Q = compute_all_action_values(values_flat)
    best_action_idx = np.argmax(Q, axis=0)
    policy = ACTIONS[best_action_idx]

    total_time = time.time() - total_start
    print(f'  [VI] total: {iteration_count} sweeps, {total_time:.2f}s\n')
    return values, policy, total_time, iter_times, diffs


# ====================== 可视化（全中文） ======================
def plot_results(pi_values, pi_policy, pi_time, pi_policies, pe_times, pi_impr_times,
                 vi_values, vi_policy, vi_time, vi_iter_times, vi_diffs):

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # ====== 图1: 策略迭代 — 策略演变过程 ======
    n = len(pi_policies)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4.5))
    if n == 1:
        axes = [axes]
    titles = ['初始策略'] + [f'策略 #{i}' for i in range(1, n-1)] + ['最优策略']
    for idx, (p, ax) in enumerate(zip(pi_policies, axes)):
        im = ax.imshow(np.flipud(p), cmap='RdBu', vmin=-MAX_MOVE, vmax=MAX_MOVE,
                       aspect='auto', extent=[-0.5, MAX_CARS+0.5, -0.5, MAX_CARS+0.5])
        ax.set_title(titles[idx] if idx < len(titles) else f'策略 #{idx}', fontsize=10)
        ax.set_xlabel('网点2车辆数')
        if idx == 0:
            ax.set_ylabel('网点1车辆数')
    fig.colorbar(im, ax=axes, label='移动车辆数 (正:1->2, 负:2->1)')
    fig.suptitle("策略迭代: 策略演变过程", fontsize=13)
    plt.tight_layout()
    plt.savefig('fig1_policy_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig1_policy_evolution.png")

    # ====== 图2: 两种算法最终最优策略对比 ======
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, pol, title in [(ax1, pi_policy, '策略迭代 - 最优策略'),
                            (ax2, vi_policy, '价值迭代 - 最优策略')]:
        im = ax.imshow(np.flipud(pol), cmap='RdBu', vmin=-MAX_MOVE, vmax=MAX_MOVE,
                       aspect='auto', extent=[-0.5, MAX_CARS+0.5, -0.5, MAX_CARS+0.5])
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('网点2车辆数')
        ax.set_ylabel('网点1车辆数')
        for i in range(0, N, 4):
            for j in range(0, N, 4):
                ax.text(j, MAX_CARS - i, str(pol[i, j]),
                        ha='center', va='center', fontsize=6, color='k')
    fig.colorbar(im, ax=[ax1, ax2], label='移动车辆数 (正:1->2, 负:2->1)')
    fig.suptitle('最优策略对比', fontsize=14)
    plt.tight_layout()
    plt.savefig('fig2_policy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig2_policy_comparison.png")

    # ====== 图3: 最优价值函数 3D 曲面 ======
    fig = plt.figure(figsize=(14, 5))
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    for subplot_idx, (vals, title) in enumerate([(pi_values, '策略迭代 V*(s)'),
                                                   (vi_values, '价值迭代 V*(s)')]):
        ax = fig.add_subplot(1, 2, subplot_idx+1, projection='3d')
        ax.plot_surface(X, Y, vals, cmap=cm.coolwarm, alpha=0.85, linewidth=0)
        ax.set_xlabel('网点2车辆数')
        ax.set_ylabel('网点1车辆数')
        ax.set_zlabel('V*(s)')
        ax.set_title(title, fontsize=11)
    fig.suptitle('最优价值函数 V*(s)', fontsize=14)
    plt.tight_layout()
    plt.savefig('fig3_value_function.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig3_value_function.png")

    # ====== 图4: 时间对比 ======
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    labels = ['策略迭代', '价值迭代']
    times = [pi_time, vi_time]
    colors = ['#1976D2', '#F57C00']
    bars = ax.bar(labels, times, color=colors, width=0.45, zorder=3)
    ax.grid(axis='y', alpha=0.4, zorder=0)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(times)*0.01,
                f'{t:.2f}s', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.set_ylabel('总耗时 (秒)', fontsize=11)
    ax.set_title('总求解时间', fontsize=12)
    ratio = vi_time / pi_time
    ax.annotate(f'价值迭代/策略迭代 = {ratio:.2f}x', xy=(0.5, 0.85),
                xycoords='axes fraction', ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax = axes[1]
    x = np.arange(len(pe_times))
    w = 0.35
    ax.bar(x - w/2, pe_times, w, label='策略评估', color='#1976D2', zorder=3)
    ax.bar(x + w/2, pi_impr_times, w, label='策略改进', color='#64B5F6', zorder=3)
    ax.set_xlabel('外循环轮次')
    ax.set_ylabel('耗时 (秒)')
    ax.set_title('策略迭代: 每轮耗时分解')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i+1) for i in x])
    ax.legend()
    ax.grid(axis='y', alpha=0.4, zorder=0)

    ax = axes[2]
    color1 = '#F57C00'
    color2 = '#4CAF50'
    ax2b = ax.twinx()
    ax.bar(range(1, len(vi_iter_times)+1), vi_iter_times, color=color1, alpha=0.7, label='每轮耗时', zorder=3)
    ax2b.plot(range(1, len(vi_diffs)+1), vi_diffs, 'o-', color=color2, linewidth=2, markersize=5, label='||DV||')
    ax2b.axhline(y=DELTA, color='red', linestyle='--', linewidth=1, label='delta={}'.format(DELTA))
    ax.set_xlabel('迭代轮次')
    ax.set_ylabel('每轮耗时 (秒)', color=color1)
    ax2b.set_ylabel('||DV||', color=color2)
    ax.set_title('价值迭代: 收敛过程')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    fig.suptitle('求解时间定量对比: 策略迭代 vs 价值迭代', fontsize=13)
    plt.tight_layout()
    plt.savefig('fig4_time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig4_time_comparison.png")

    # ====== 图5: 差异热图 ======
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    diff_policy = np.abs(pi_policy - vi_policy)
    diff_value = np.abs(pi_values - vi_values)
    im1 = axes[0].imshow(np.flipud(diff_policy), cmap='hot',
                          extent=[-0.5, MAX_CARS+0.5, -0.5, MAX_CARS+0.5])
    axes[0].set_title('策略差异 (最大={})'.format(diff_policy.max()), fontsize=11)
    axes[0].set_xlabel('网点2车辆数')
    axes[0].set_ylabel('网点1车辆数')
    fig.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(np.flipud(diff_value), cmap='hot',
                          extent=[-0.5, MAX_CARS+0.5, -0.5, MAX_CARS+0.5])
    axes[1].set_title('价值差异 (最大={:.4f})'.format(diff_value.max()), fontsize=11)
    axes[1].set_xlabel('网点2车辆数')
    axes[1].set_ylabel('网点1车辆数')
    fig.colorbar(im2, ax=axes[1])
    fig.suptitle('收敛质量对比: 策略迭代 vs 价值迭代', fontsize=13)
    plt.tight_layout()
    plt.savefig('fig5_difference.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig5_difference.png")


# ====================== 主程序 ======================
if __name__ == '__main__':
    print('=' * 60)
    print('Jack\'s Car Rental: Policy Iteration vs Value Iteration')
    print('=' * 60)

    print('\n>>> Running Policy Iteration...')
    pi_values, pi_policy, pi_time, pi_policies, pe_times, pi_impr_times = policy_iteration()

    print('\n>>> Running Value Iteration...')
    vi_values, vi_policy, vi_time, vi_iter_times, vi_diffs = value_iteration()

    print('\n' + '=' * 60)
    print('RESULTS SUMMARY')
    print('=' * 60)
    print(f'  Precompute time:              {precompute_time:.2f}s')
    print(f'  Policy Iteration total time:  {pi_time:.2f}s  ({len(pe_times)} outer iterations)')
    print(f'    - Policy evaluation rounds: {sum(pe_times):.2f}s')
    print(f'    - Policy improvement rounds:{sum(pi_impr_times):.2f}s')
    print(f'  Value Iteration total time:   {vi_time:.2f}s  ({len(vi_iter_times)} sweeps)')
    print(f'  Speed ratio (PI/VI):          {pi_time/vi_time:.2f}x')
    policy_diff = (pi_policy != vi_policy).sum()
    value_max_diff = np.abs(pi_values - vi_values).max()
    print(f'  States where policies differ: {policy_diff} / {N*N}')
    print(f'  Max |V*_PI - V*_VI|:          {value_max_diff:.6f}')

    print('\n>>> Plotting...')
    plot_results(pi_values, pi_policy, pi_time, pi_policies, pe_times, pi_impr_times,
                 vi_values, vi_policy, vi_time, vi_iter_times, vi_diffs)
    print('Done! All figures saved.')
